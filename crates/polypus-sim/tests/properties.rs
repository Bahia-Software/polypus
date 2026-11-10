//! Property test of parallel == sequential (ENGINEERING.md §4): on random
//! circuits over the whole gate vocabulary, the statevector the parallel
//! kernels produce (`parallel_threshold: 0`, every gate on the rayon path) is
//! bit-identical to the sequential one (`parallel_threshold: usize::MAX`), with
//! gate fusion off and on.
//!
//! The kernel unit tests compare the two paths gate by gate on fixed states;
//! this compares whole circuits, so the order of gates, fusion's grouping and
//! the lowering of the gates without a dedicated kernel take part too. The
//! cases and the seed are fixed, so every run checks the same circuits.
#![cfg(feature = "parallel")]

use polypus_circuit::{ConcreteCircuit, ParameterizedCircuit};
use polypus_sim::{Simulator, StatevectorSimulator};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

/// Every unitary of the vocabulary as `(name, qubits, angles)`, plus `g`, a
/// gate the program declares (simulated through its body).
const GATES: [(&str, usize, usize); 43] = [
    ("h", 1, 0),
    ("x", 1, 0),
    ("y", 1, 0),
    ("z", 1, 0),
    ("s", 1, 0),
    ("t", 1, 0),
    ("sdg", 1, 0),
    ("tdg", 1, 0),
    ("id", 1, 0),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("cx", 2, 0),
    ("cz", 2, 0),
    ("swap", 2, 0),
    ("rzz", 2, 1),
    ("rxx", 2, 1),
    ("cp", 2, 1),
    ("u3", 1, 3),
    ("sx", 1, 0),
    ("sxdg", 1, 0),
    ("cy", 2, 0),
    ("ch", 2, 0),
    ("csx", 2, 0),
    ("ccx", 3, 0),
    ("cswap", 3, 0),
    ("crx", 2, 1),
    ("cry", 2, 1),
    ("crz", 2, 1),
    ("cu1", 2, 1),
    ("cu3", 2, 3),
    ("cu", 2, 4),
    ("u0", 1, 1),
    ("p", 1, 1),
    ("u1", 1, 1),
    ("u2", 1, 2),
    ("u", 1, 3),
    ("rccx", 3, 0),
    ("rc3x", 4, 0),
    ("c3x", 4, 0),
    ("c3sqrtx", 4, 0),
    ("c4x", 5, 0),
    ("g", 2, 1),
];

/// The widest register generated: wide enough for `c4x`, small enough for
/// hundreds of cases in well under a second each.
const MAX_QUBITS: usize = 7;

/// One gate statement: `GATES[kind]` on the first operands of `qubits`.
fn statement(num_qubits: usize) -> impl Strategy<Value = (usize, Vec<usize>, [f64; 4])> {
    let angle = || -7.0..7.0f64;
    (
        0..GATES.len(),
        Just((0..num_qubits).collect::<Vec<_>>()).prop_shuffle(),
        [angle(), angle(), angle(), angle()],
    )
}

fn program() -> impl Strategy<Value = String> {
    (1..=MAX_QUBITS)
        .prop_flat_map(|n| (Just(n), prop::collection::vec(statement(n), 1..40)))
        .prop_map(|(n, statements)| {
            let mut src = String::from(
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
                 gate g(t) a,b { ry(t) a; cx a,b; rz(t/3) b; }\n",
            );
            src.push_str(&format!("qreg q[{n}];\n"));
            for (kind, qubits, angles) in statements {
                let (name, arity, params) = GATES[kind];
                if arity > n {
                    continue;
                }
                let args: Vec<String> = angles[..params].iter().map(|a| format!("{a:?}")).collect();
                let operands: Vec<String> =
                    qubits[..arity].iter().map(|q| format!("q[{q}]")).collect();
                if params == 0 {
                    src.push_str(&format!("{name} {};\n", operands.join(",")));
                } else {
                    src.push_str(&format!(
                        "{name}({}) {};\n",
                        args.join(","),
                        operands.join(",")
                    ));
                }
            }
            src
        })
}

fn config() -> Config {
    Config {
        cases: 256,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5E9A_11E1),
        failure_persistence: None,
        ..Config::default()
    }
}

/// The amplitudes' bits: `==` on `f64` would take `-0.0` for `0.0`.
fn bits(sim: &StatevectorSimulator, circuit: &ConcreteCircuit) -> Vec<(u64, u64)> {
    sim.run(circuit)
        .expect("a generated circuit simulates")
        .amplitudes()
        .iter()
        .map(|a| (a.re.to_bits(), a.im.to_bits()))
        .collect()
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn parallel_kernels_match_the_sequential_ones_bit_for_bit(src in program()) {
        let circuit = ParameterizedCircuit::from_qasm2(&src)
            .and_then(|c| c.assign_parameters(&[]))
            .unwrap_or_else(|e| panic!("{e}\n{src}"));
        for fusion in [false, true] {
            let sequential = StatevectorSimulator {
                parallel_threshold: usize::MAX,
                fusion,
                ..StatevectorSimulator::new()
            };
            let parallel = StatevectorSimulator {
                parallel_threshold: 0,
                ..sequential.clone()
            };
            prop_assert_eq!(
                bits(&parallel, &circuit),
                bits(&sequential, &circuit),
                "fusion {}:\n{}", fusion, src
            );
        }
    }
}
