//! Simulator-side enforcement of the `polypus-circuit` contracts
//! (see `docs/CONTRACTS.md`):
//!
//! - **C-2 · Gate vocabulary symmetry (QIR half).** Each non-trivial QIR
//!   decomposition must realise the *same unitary* as the native gate up to a
//!   global phase, with the native simulator as the reference. The older
//!   decompositions (`swap`, `rzz`, `rxx`, `cp`, `u3`) are mirrored by hand
//!   below; `c2_every_gate_emitted_qir_matches_the_native_gate` covers the
//!   whole vocabulary by parsing the QIR `to_qir` actually emits.
//! - **C-4 · Terminal measurement placement (simulator half).** The simulator
//!   rejects a circuit that operates on an already-measured qubit rather than
//!   silently treating the measurement as a no-op.

use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
use polypus_sim::{SimError, Simulator, Statevector, StatevectorSimulator, C64};
use std::f64::consts::{FRAC_PI_3, PI};

// ─────────────────── C-2 · QIR-vs-simulator equivalence ───────────────────

/// Column `k` of the unitary realised by `gates` on `n` qubits: prepare the
/// basis state |k> (via `x` gates), evolve, read the amplitudes.
fn column(n: usize, k: usize, gates: &[G]) -> Vec<C64> {
    let mut sv = Statevector::new(n).unwrap();
    for q in 0..n {
        if (k >> q) & 1 == 1 {
            sv.apply(&G::X(q)).unwrap();
        }
    }
    for g in gates {
        sv.apply(g).unwrap();
    }
    sv.amplitudes().to_vec()
}

/// Assert two gate sequences realise the same `n`-qubit unitary up to one
/// global phase (tolerance 1e-12). The native gate is the reference (C-2).
fn assert_equiv_up_to_global_phase(n: usize, native: &[G], decomposed: &[G]) {
    let dim = 1usize << n;
    let mut u_native = Vec::with_capacity(dim * dim);
    let mut u_decomp = Vec::with_capacity(dim * dim);
    for k in 0..dim {
        u_native.extend(column(n, k, native));
        u_decomp.extend(column(n, k, decomposed));
    }
    // Pin the global phase from the first sizeable native amplitude.
    let i = u_native
        .iter()
        .position(|z| z.norm() > 1e-9)
        .expect("native unitary is entirely zero");
    let phase = u_decomp[i] / u_native[i];
    assert!(
        (phase.norm() - 1.0).abs() < 1e-9,
        "phase factor is not unit-modulus ({})",
        phase.norm()
    );
    for (a, b) in u_native.iter().zip(&u_decomp) {
        assert!(
            (b - a * phase).norm() < 1e-12,
            "decomposition differs from the native gate by more than a global phase"
        );
    }
}

const ANGLES: [f64; 5] = [0.3, 0.7, 1.25, -2.0, FRAC_PI_3];

#[test]
fn c2_qir_swap_decomposition_matches_native() {
    // swap a,b = cnot a,b; cnot b,a; cnot a,b (no swap intrinsic in QIR base).
    let native = [G::Swap(0, 1)];
    let decomp = [G::Cx(0, 1), G::Cx(1, 0), G::Cx(0, 1)];
    assert_equiv_up_to_global_phase(2, &native, &decomp);
}

#[test]
fn c2_qir_rzz_decomposition_matches_native() {
    // rzz(θ) = cnot · rz(θ) · cnot
    for &t in &ANGLES {
        let native = [G::Rzz {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t),
            },
            G::Cx(0, 1),
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_rxx_decomposition_matches_native() {
    // rxx(θ) = (h⊗h) · cnot · rz(θ) · cnot · (h⊗h)
    for &t in &ANGLES {
        let native = [G::Rxx {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::H(0),
            G::H(1),
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t),
            },
            G::Cx(0, 1),
            G::H(0),
            G::H(1),
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_cp_decomposition_matches_native() {
    // cp(θ) = rz(θ/2) q0; cnot; rz(−θ/2) q1; cnot; rz(θ/2) q1
    // (audit item C3: the old `cz; rz; cz` collapsed to rz(θ) and was wrong).
    for &t in &ANGLES {
        let native = [G::Cp {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::Rz {
                qubit: 0,
                theta: Fixed(t / 2.0),
            },
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(-t / 2.0),
            },
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t / 2.0),
            },
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_u3_decomposition_matches_native() {
    // u3(θ,φ,λ) applied left-to-right is rz(λ), ry(θ), rz(φ).
    for &(th, ph, la) in &[(0.1, 0.2, 0.3), (1.0, -0.5, 2.0), (PI / 2.0, 0.0, PI)] {
        let native = [G::U {
            qubit: 0,
            theta: Fixed(th),
            phi: Fixed(ph),
            lam: Fixed(la),
        }];
        let decomp = [
            G::Rz {
                qubit: 0,
                theta: Fixed(la),
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(th),
            },
            G::Rz {
                qubit: 0,
                theta: Fixed(ph),
            },
        ];
        assert_equiv_up_to_global_phase(1, &native, &decomp);
    }
}

// ─────────────────────── C-4 · terminal measurement ───────────────────────

#[test]
fn c4_simulator_rejects_gate_after_measure() {
    let cc = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![G::Measure { qubit: 0, cbit: 0 }, G::X(0)],
    };
    let err = StatevectorSimulator::new().run(&cc).unwrap_err();
    assert_eq!(err, SimError::GateAfterMeasure { qubit: 0 });
}

#[test]
fn c4_simulator_accepts_terminal_measurement() {
    let cc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![G::H(0), G::Cx(0, 1), G::MeasureAll],
    };
    let sv = StatevectorSimulator::new().run(&cc).unwrap();
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn c4_simulator_rejects_three_qubit_gate_on_measured_operand() {
    let cc = ConcreteCircuit {
        num_qubits: 3,
        gates: vec![G::Measure { qubit: 1, cbit: 0 }, G::Ccx(0, 1, 2)],
    };
    let err = StatevectorSimulator::new().run(&cc).unwrap_err();
    assert_eq!(err, SimError::GateAfterMeasure { qubit: 1 });
}

// ──────────── C-2 · every gate's emitted QIR vs. the native gate ─────────────
//
// The hand-mirrored decompositions above pin the older gates. This check needs
// no mirroring: it parses the QIR that `to_qir` *actually emits* back into the
// base gate set, and compares that sequence's unitary with the native gate's,
// for every instruction of the vocabulary. A wrong decomposition — or a
// control/target swap in one — fails here.

/// Qubit index of a QIR `%Qubit*` operand (`null` or `inttoptr (i64 k …)`).
fn qir_qubit(operand: &str) -> usize {
    let operand = operand.trim();
    if operand == "%Qubit* null" {
        return 0;
    }
    let digits = operand
        .strip_prefix("%Qubit* inttoptr (i64 ")
        .and_then(|rest| rest.split(' ').next())
        .unwrap_or_else(|| panic!("unexpected qubit operand {operand:?}"));
    digits.parse().unwrap()
}

/// The `double` angle of a QIR rotation (hexadecimal IEEE-754 literal).
fn qir_angle(operand: &str) -> f64 {
    let hex = operand
        .trim()
        .strip_prefix("double 0x")
        .unwrap_or_else(|| panic!("unexpected angle operand {operand:?}"));
    f64::from_bits(u64::from_str_radix(hex, 16).unwrap())
}

/// Parse the quantum-instruction calls of a QIR module back into base gates.
fn qir_to_gates(ir: &str) -> Vec<G> {
    let mut gates = Vec::new();
    for line in ir.lines().map(str::trim) {
        let Some(call) = line.strip_prefix("call void @__quantum__qis__") else {
            continue;
        };
        let (name, args) = call.split_once('(').unwrap();
        let args: Vec<&str> = args.trim_end_matches(')').split(", ").collect();
        let gate = match name {
            "h__body" => G::H(qir_qubit(args[0])),
            "x__body" => G::X(qir_qubit(args[0])),
            "y__body" => G::Y(qir_qubit(args[0])),
            "z__body" => G::Z(qir_qubit(args[0])),
            "s__body" => G::S(qir_qubit(args[0])),
            "s__adj" => G::Sdg(qir_qubit(args[0])),
            "t__body" => G::T(qir_qubit(args[0])),
            "t__adj" => G::Tdg(qir_qubit(args[0])),
            "rx__body" => G::Rx {
                qubit: qir_qubit(args[1]),
                theta: Fixed(qir_angle(args[0])),
            },
            "ry__body" => G::Ry {
                qubit: qir_qubit(args[1]),
                theta: Fixed(qir_angle(args[0])),
            },
            "rz__body" => G::Rz {
                qubit: qir_qubit(args[1]),
                theta: Fixed(qir_angle(args[0])),
            },
            "cnot__body" => G::Cx(qir_qubit(args[0]), qir_qubit(args[1])),
            "cz__body" => G::Cz(qir_qubit(args[0]), qir_qubit(args[1])),
            "mz__body" => continue,
            other => panic!("unexpected QIR intrinsic {other:?}"),
        };
        gates.push(gate);
    }
    gates
}

#[test]
fn c2_every_gate_emitted_qir_matches_the_native_gate() {
    let (a, b, c) = (0.37, -1.21, 2.03);
    // Every unitary instruction, on 3 qubits, operands in non-ascending order.
    let vocabulary = [
        G::H(1),
        G::X(2),
        G::Y(0),
        G::Z(1),
        G::S(2),
        G::T(0),
        G::Sdg(1),
        G::Tdg(2),
        G::Id(0),
        G::Sx(1),
        G::Sxdg(2),
        G::Rx {
            qubit: 0,
            theta: Fixed(a),
        },
        G::Ry {
            qubit: 1,
            theta: Fixed(b),
        },
        G::Rz {
            qubit: 2,
            theta: Fixed(c),
        },
        G::U {
            qubit: 1,
            theta: Fixed(a),
            phi: Fixed(b),
            lam: Fixed(c),
        },
        G::Cx(2, 0),
        G::Cz(1, 2),
        G::Swap(2, 1),
        G::Cy(2, 0),
        G::Ch(1, 0),
        G::Csx(0, 2),
        G::Ccx(2, 0, 1),
        G::Cswap(1, 2, 0),
        G::Rzz {
            q0: 2,
            q1: 0,
            theta: Fixed(a),
        },
        G::Rxx {
            q0: 1,
            q1: 2,
            theta: Fixed(b),
        },
        G::Cp {
            q0: 2,
            q1: 1,
            theta: Fixed(c),
        },
        G::Cu1 {
            q0: 0,
            q1: 2,
            theta: Fixed(a),
        },
        G::Crx {
            control: 2,
            target: 1,
            theta: Fixed(a),
        },
        G::Cry {
            control: 1,
            target: 0,
            theta: Fixed(b),
        },
        G::Crz {
            control: 0,
            target: 2,
            theta: Fixed(c),
        },
        G::Cu3 {
            control: 2,
            target: 0,
            theta: Fixed(a),
            phi: Fixed(b),
            lam: Fixed(c),
        },
        G::Cu {
            control: 1,
            target: 2,
            theta: Fixed(a),
            phi: Fixed(b),
            lam: Fixed(c),
            gamma: Fixed(0.61),
        },
    ];
    for gate in vocabulary {
        let ir = ConcreteCircuit {
            num_qubits: 3,
            gates: vec![gate.clone()],
        }
        .to_qir();
        let lowered = qir_to_gates(&ir);
        assert_equiv_up_to_global_phase(3, std::slice::from_ref(&gate), &lowered);
    }
}
