//! Property tests of the OpenQASM round trip (contract C-2, canonical form):
//! for circuits generated over the whole vocabulary, with arbitrary finite
//! angles, the export is a fixed point of export → import → export, in
//! OpenQASM 2.0 and in the OpenQASM 3 profile (C-10), and the OpenQASM 2.0
//! import gives back the same instructions, angles bit for bit, in canonical
//! form (a barrier on every qubit in order is the whole-register barrier, and
//! the measurements `q[k] -> c[k]` of every qubit in order are `MeasureAll`).
//!
//! `tests/contracts.rs` pins the vocabulary gate by gate and in one fixed
//! circuit; this file checks the same guarantees on arbitrary sequences, so
//! interactions it cannot list (operand orders, angle spellings, declarations
//! next to built-ins, measurements after barriers) are exercised too. The
//! cases and the seed are fixed, so every run checks the same circuits.

use polypus_circuit::{GateInstruction, GateParam, ParameterizedCircuit};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

/// Built-in unitary kinds the generator draws from, with their qubit count.
/// Their order is the generator's own; `generator_covers_the_whole_vocabulary`
/// checks the list against [`GateInstruction::KINDS`].
const UNITARIES: [(&str, usize); 42] = [
    ("h", 1),
    ("x", 1),
    ("y", 1),
    ("z", 1),
    ("s", 1),
    ("t", 1),
    ("sdg", 1),
    ("tdg", 1),
    ("id", 1),
    ("rx", 1),
    ("ry", 1),
    ("rz", 1),
    ("cx", 2),
    ("cz", 2),
    ("swap", 2),
    ("rzz", 2),
    ("rxx", 2),
    ("cp", 2),
    ("u3", 1),
    ("sx", 1),
    ("sxdg", 1),
    ("cy", 2),
    ("ch", 2),
    ("csx", 2),
    ("ccx", 3),
    ("cswap", 3),
    ("crx", 2),
    ("cry", 2),
    ("crz", 2),
    ("cu1", 2),
    ("cu3", 2),
    ("cu", 2),
    ("u0", 1),
    ("p", 1),
    ("u1", 1),
    ("u2", 1),
    ("u", 1),
    ("rccx", 3),
    ("rc3x", 4),
    ("c3x", 4),
    ("c3sqrtx", 4),
    ("c4x", 5),
];

/// The widest register generated: wide enough for `c4x`.
const MAX_QUBITS: usize = 6;

/// One drawn instruction, before it is fitted to the register.
#[derive(Debug, Clone)]
enum Op {
    /// `UNITARIES[kind]` on the first operands of `qubits`, with `angles`.
    Unitary {
        kind: usize,
        qubits: Vec<usize>,
        angles: [f64; 4],
    },
    /// A call of the declared gate `g(t) a,b`.
    Declared { qubits: Vec<usize>, angle: f64 },
    /// `barrier q;` (`None`) or a barrier on the first `k` of `qubits`.
    Barrier {
        qubits: Vec<usize>,
        k: Option<usize>,
    },
}

/// A finite angle: mostly an ordinary rotation, sometimes any finite binary64
/// (subnormal, huge, negative zero), whose shortest spelling the exporter must
/// read back to the same bits.
fn angle() -> impl Strategy<Value = f64> {
    use proptest::num::f64 as f;
    prop_oneof![
        3 => -10.0..10.0f64,
        1 => f::POSITIVE | f::NEGATIVE | f::NORMAL | f::SUBNORMAL | f::ZERO,
        1 => Just(-0.0),
        1 => prop::sample::select(vec![
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_4,
            1e-300,
            1.7976931348623157e308,
        ]),
    ]
}

fn op(num_qubits: usize) -> impl Strategy<Value = Op> {
    let qubits = Just((0..num_qubits).collect::<Vec<_>>()).prop_shuffle();
    let unitary = (
        0..UNITARIES.len(),
        qubits.clone(),
        [angle(), angle(), angle(), angle()],
    )
        .prop_map(|(kind, qubits, angles)| Op::Unitary {
            kind,
            qubits,
            angles,
        });
    let declared =
        (qubits.clone(), angle()).prop_map(|(qubits, angle)| Op::Declared { qubits, angle });
    let barrier = (qubits, prop::option::of(1..=num_qubits))
        .prop_map(|(qubits, k)| Op::Barrier { qubits, k });
    prop_oneof![8 => unitary, 1 => declared, 1 => barrier]
}

/// How a circuit ends.
#[derive(Debug, Clone)]
enum Ending {
    Nothing,
    MeasureAll,
    /// `measure q[order[i]] -> c[cbits[i]]` for the first `k` qubits of `order`.
    Some {
        order: Vec<usize>,
        cbits: Vec<usize>,
        k: usize,
    },
}

fn ending(num_qubits: usize) -> impl Strategy<Value = Ending> {
    let order = Just((0..num_qubits).collect::<Vec<_>>()).prop_shuffle();
    let cbits = Just((0..num_qubits + 2).collect::<Vec<_>>()).prop_shuffle();
    prop_oneof![
        Just(Ending::Nothing),
        Just(Ending::MeasureAll),
        (order, cbits, 1..=num_qubits).prop_map(|(order, cbits, k)| Ending::Some {
            order,
            cbits,
            k
        }),
    ]
}

fn circuit_parts() -> impl Strategy<Value = (usize, Vec<Op>, Ending)> {
    (1..=MAX_QUBITS).prop_flat_map(|n| (Just(n), prop::collection::vec(op(n), 0..24), ending(n)))
}

/// The declared gate the generated calls use: its body divides its argument,
/// so the call is checked through the expression, as an imported one is.
fn declared_gate() -> polypus_circuit::CustomGate {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate g(t) a,b { rz(t/2) a; cx a,b; }\n\
               qreg q[2];\ng(0) q[0],q[1];\n";
    match &ParameterizedCircuit::from_qasm2(src).unwrap().gates()[0] {
        GateInstruction::Custom(call) => call.clone(),
        other => panic!("expected a declared gate call, got {other:?}"),
    }
}

/// The instruction `UNITARIES[kind]` on `q` with angles `a`.
fn unitary(kind: usize, q: &[usize], a: [f64; 4]) -> GateInstruction {
    use GateInstruction as G;
    let f = GateParam::Fixed;
    match UNITARIES[kind].0 {
        "h" => G::H(q[0]),
        "x" => G::X(q[0]),
        "y" => G::Y(q[0]),
        "z" => G::Z(q[0]),
        "s" => G::S(q[0]),
        "t" => G::T(q[0]),
        "sdg" => G::Sdg(q[0]),
        "tdg" => G::Tdg(q[0]),
        "id" => G::Id(q[0]),
        "rx" => G::Rx {
            qubit: q[0],
            theta: f(a[0]),
        },
        "ry" => G::Ry {
            qubit: q[0],
            theta: f(a[0]),
        },
        "rz" => G::Rz {
            qubit: q[0],
            theta: f(a[0]),
        },
        "cx" => G::Cx(q[0], q[1]),
        "cz" => G::Cz(q[0], q[1]),
        "swap" => G::Swap(q[0], q[1]),
        "rzz" => G::Rzz {
            q0: q[0],
            q1: q[1],
            theta: f(a[0]),
        },
        "rxx" => G::Rxx {
            q0: q[0],
            q1: q[1],
            theta: f(a[0]),
        },
        "cp" => G::Cp {
            q0: q[0],
            q1: q[1],
            theta: f(a[0]),
        },
        "u3" => G::U {
            qubit: q[0],
            theta: f(a[0]),
            phi: f(a[1]),
            lam: f(a[2]),
        },
        "sx" => G::Sx(q[0]),
        "sxdg" => G::Sxdg(q[0]),
        "cy" => G::Cy(q[0], q[1]),
        "ch" => G::Ch(q[0], q[1]),
        "csx" => G::Csx(q[0], q[1]),
        "ccx" => G::Ccx(q[0], q[1], q[2]),
        "cswap" => G::Cswap(q[0], q[1], q[2]),
        "crx" => G::Crx {
            control: q[0],
            target: q[1],
            theta: f(a[0]),
        },
        "cry" => G::Cry {
            control: q[0],
            target: q[1],
            theta: f(a[0]),
        },
        "crz" => G::Crz {
            control: q[0],
            target: q[1],
            theta: f(a[0]),
        },
        "cu1" => G::Cu1 {
            q0: q[0],
            q1: q[1],
            theta: f(a[0]),
        },
        "cu3" => G::Cu3 {
            control: q[0],
            target: q[1],
            theta: f(a[0]),
            phi: f(a[1]),
            lam: f(a[2]),
        },
        "cu" => G::Cu {
            control: q[0],
            target: q[1],
            theta: f(a[0]),
            phi: f(a[1]),
            lam: f(a[2]),
            gamma: f(a[3]),
        },
        "u0" => G::U0 {
            qubit: q[0],
            gamma: f(a[0]),
        },
        "p" => G::P {
            qubit: q[0],
            lam: f(a[0]),
        },
        "u1" => G::U1 {
            qubit: q[0],
            lam: f(a[0]),
        },
        "u2" => G::U2 {
            qubit: q[0],
            phi: f(a[0]),
            lam: f(a[1]),
        },
        "u" => G::UGate {
            qubit: q[0],
            theta: f(a[0]),
            phi: f(a[1]),
            lam: f(a[2]),
        },
        "rccx" => G::Rccx(q[0], q[1], q[2]),
        "rc3x" => G::Rc3x(q[0], q[1], q[2], q[3]),
        "c3x" => G::C3x(q[0], q[1], q[2], q[3]),
        "c3sqrtx" => G::C3sqrtx(q[0], q[1], q[2], q[3]),
        "c4x" => G::C4x(q[0], q[1], q[2], q[3], q[4]),
        name => unreachable!("{name} is not in UNITARIES"),
    }
}

/// The circuit the parts describe. An instruction wider than the register is
/// skipped, so every drawn sequence builds.
fn build(num_qubits: usize, ops: &[Op], ending: &Ending) -> ParameterizedCircuit {
    let declared = declared_gate();
    let mut c = ParameterizedCircuit::new(num_qubits);
    for op in ops {
        let gate = match op {
            Op::Unitary {
                kind,
                qubits,
                angles,
            } => {
                if UNITARIES[*kind].1 > num_qubits {
                    continue;
                }
                unitary(*kind, qubits, *angles)
            }
            Op::Declared { qubits, angle } => {
                if num_qubits < 2 {
                    continue;
                }
                let call = declared
                    .with_arguments(vec![GateParam::Fixed(*angle)], qubits[..2].to_vec())
                    .expect("a finite angle is a valid argument");
                GateInstruction::Custom(call)
            }
            Op::Barrier { qubits, k } => match k {
                Some(k) => GateInstruction::Barrier(qubits[..*k].to_vec()),
                None => GateInstruction::Barrier(Vec::new()),
            },
        };
        c.try_push(gate).expect("a generated instruction is valid");
    }
    match ending {
        Ending::Some { order, cbits, k } => order[..*k]
            .iter()
            .zip(cbits)
            .fold(c, |c, (&q, &b)| c.measure(q, b)),
        Ending::Nothing => c,
        Ending::MeasureAll => c.measure_all(),
    }
}

/// The program of `gates` in the canonical form of C-2 (the one the export
/// writes): a run `measure q[k] -> c[k]` of every qubit, in order, is
/// `MeasureAll`, and a barrier that lists every qubit, in order, is the
/// whole-register barrier. Written here on purpose, without the crate's
/// `normalized`, so that the test does not agree with the exporter by
/// construction.
fn canonical(gates: &[GateInstruction], n: usize) -> Vec<GateInstruction> {
    let mut out = Vec::with_capacity(gates.len());
    let mut i = 0;
    while i < gates.len() {
        let full_run = n > 0
            && i + n <= gates.len()
            && (0..n).all(|k| {
                matches!(
                    gates[i + k],
                    GateInstruction::Measure { qubit, cbit } if qubit == k && cbit == k
                )
            });
        if full_run {
            out.push(GateInstruction::MeasureAll);
            i += n;
            continue;
        }
        match &gates[i] {
            GateInstruction::Barrier(v)
                if !v.is_empty() && v.len() == n && v.iter().enumerate().all(|(k, &q)| q == k) =>
            {
                out.push(GateInstruction::Barrier(Vec::new()));
            }
            other => out.push(other.clone()),
        }
        i += 1;
    }
    out
}

/// Fixed cases and a fixed seed: every run, and CI, checks the same circuits.
/// No failure persistence: a failure prints its minimal circuit, and the fixed
/// seed reproduces it.
fn config() -> Config {
    Config {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0xC2C2),
        failure_persistence: None,
        ..Config::default()
    }
}

/// The instructions with their angles spelled bit for bit: `Debug` prints an
/// `f64` as the shortest decimal that reads back as the same binary64, sign of
/// zero included, where `==` would take `-0.0` for `0.0`.
fn exactly(gates: &[GateInstruction]) -> String {
    format!("{gates:?}")
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn qasm2_export_is_a_fixed_point_and_reads_back((n, ops, end) in circuit_parts()) {
        let c = build(n, &ops, &end);
        let qasm = c.to_qasm2_with_params(&[]).unwrap();
        let imported = ParameterizedCircuit::from_qasm2(&qasm)
            .unwrap_or_else(|e| panic!("{e}\n{qasm}"));
        prop_assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), qasm.clone());
        prop_assert_eq!(
            exactly(imported.gates()),
            exactly(&canonical(c.gates(), n)),
            "{}",
            qasm
        );
    }

    #[test]
    fn qasm3_export_is_a_fixed_point((n, ops, end) in circuit_parts()) {
        let c = build(n, &ops, &end);
        let qasm = c.to_qasm3().unwrap();
        let imported = ParameterizedCircuit::from_qasm3(&qasm)
            .unwrap_or_else(|e| panic!("{e}\n{qasm}"));
        prop_assert_eq!(imported.to_qasm3().unwrap(), qasm);
    }
}

#[test]
fn generator_covers_the_whole_vocabulary() {
    // Every unitary kind, a declared-gate call, both barriers and both
    // measurements: with `MeasureAll`, that is every kind `GateInstruction`
    // numbers, so a new instruction fails here until the generator draws it.
    let ops: Vec<Op> = (0..UNITARIES.len())
        .map(|kind| Op::Unitary {
            kind,
            qubits: vec![4, 0, 3, 1, 2],
            angles: [0.1, 0.2, 0.3, 0.4],
        })
        .chain([
            Op::Declared {
                qubits: vec![1, 0],
                angle: 0.5,
            },
            Op::Barrier {
                qubits: vec![2, 0],
                k: Some(2),
            },
        ])
        .collect();
    let partial = Ending::Some {
        order: vec![0, 1],
        cbits: vec![1, 0],
        k: 2,
    };
    let mut seen = [false; GateInstruction::KINDS];
    for gate in build(5, &ops, &partial)
        .gates()
        .iter()
        .chain(build(5, &[], &Ending::MeasureAll).gates())
    {
        seen[gate.kind_index()] = true;
    }
    let missing: Vec<usize> = (0..GateInstruction::KINDS).filter(|&k| !seen[k]).collect();
    assert!(
        missing.is_empty(),
        "kinds the generator never draws: {missing:?}"
    );
}
