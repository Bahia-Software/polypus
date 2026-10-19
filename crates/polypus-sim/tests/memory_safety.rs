//! Regression guard for the P0 memory-safety hole: a gate that references a
//! qubit outside the circuit's register must be rejected with a typed
//! [`SimError`] *before any kernel runs* — in release as well as debug.
//!
//! The kernels' only bounds guard is a `debug_assert!(q < n)` that vanishes in a
//! release build. There, the *parallel* kernel writes through a raw pointer at an
//! index derived from the out-of-range qubit (`1 << q`) with no bounds check —
//! an out-of-bounds write, i.e. undefined behaviour. (The sequential kernel
//! indexes a `Vec`, which still bounds-check-panics in release — a crash, not
//! UB, but still not a clean failure.) The way in is a hand-assembled or
//! field-mutated [`ConcreteCircuit`] whose public `gates` field bypasses the
//! builder's per-push `QubitOutOfRange` validation.
//!
//! The same holds for a unitary that names one qubit twice (`Cx(0, 0)`, #342):
//! the two-qubit kernels compute `1 << (n - 2)` and assume distinct operands.
//!
//! These assertions hold identically under `cargo test` and `cargo test
//! --release`, including a release build without overflow-checks (the default
//! for downstream crates; CI runs this file that way with `parallel` on). Before the fix, the debug run panics in the kernel assertion and
//! the release run either is UB (parallel path) or bounds-check-panics
//! (sequential) — never the clean typed error asserted here.

use polypus_circuit::{ConcreteCircuit, GateInstruction, ParameterizedCircuit};
use polypus_sim::{SimError, Simulator, Statevector, StatevectorSimulator};

#[test]
fn simulator_rejects_out_of_range_single_qubit_gate() {
    // A valid 2-qubit circuit, then an out-of-range gate appended straight onto
    // the public `gates` vector (the builder would have rejected `h(5)`).
    let mut circuit = ParameterizedCircuit::new(2)
        .h(0)
        .assign_parameters(&[])
        .expect("a bound 2-qubit circuit");
    circuit.gates.push(GateInstruction::H(5));

    let err = StatevectorSimulator::new().run(&circuit).unwrap_err();
    assert_eq!(
        err,
        SimError::QubitIndexOutOfRange {
            qubit: 5,
            num_qubits: 2
        }
    );
}

#[test]
fn simulator_rejects_out_of_range_control_qubit() {
    // Two-qubit gate with an out-of-range operand, on a field-built circuit.
    let circuit = ConcreteCircuit::new(2, vec![GateInstruction::Cx(0, 9)]);
    let err = StatevectorSimulator::new().run(&circuit).unwrap_err();
    assert_eq!(
        err,
        SimError::QubitIndexOutOfRange {
            qubit: 9,
            num_qubits: 2
        }
    );
}

#[test]
fn statevector_apply_rejects_out_of_range_qubit() {
    // `Statevector::apply` is public, so it must be safe on its own — not only
    // behind the whole-circuit pre-check the simulator does before allocating.
    let mut sv = Statevector::new(2).expect("2 is well below MAX_QUBITS");
    let err = sv.apply(&GateInstruction::H(5)).unwrap_err();
    assert_eq!(
        err,
        SimError::QubitIndexOutOfRange {
            qubit: 5,
            num_qubits: 2
        }
    );
}

#[test]
fn fusion_does_not_bypass_the_out_of_range_guard() {
    // The whole-circuit guard must run *before* the gate-fusion machinery, so an
    // out-of-range operand is rejected even when it sits inside a sequence the
    // fuser would otherwise fold — a diagonal run collapsed into one buffer pass,
    // or a dense connected component composed into one matrix. Fusion is on by
    // default (`StatevectorSimulator::new`); were the guard ever moved after the
    // fusion loop, the out-of-range `1 << q` would reach the kernels: a silently
    // wrong result on the (safe) diagonal path, and an out-of-bounds raw-pointer
    // write on the dense path's parallel kernel in release — the very UB this
    // whole file guards against.

    // A run of consecutive diagonal gates (the #131 fusion path) with an
    // out-of-range member in the middle of the run.
    let diagonal_run = ConcreteCircuit::new(
        3,
        vec![
            GateInstruction::Z(0),
            GateInstruction::Z(7),
            GateInstruction::Z(1),
        ],
    );
    assert_eq!(
        StatevectorSimulator::new().run(&diagonal_run).unwrap_err(),
        SimError::QubitIndexOutOfRange {
            qubit: 7,
            num_qubits: 3
        }
    );

    // A chain of dense gates sharing a qubit (the #132 connected-component path:
    // `H`·`X` on qubit 0 would compose into one 2×2) followed by an out-of-range
    // dense gate that would open its own component and reach `apply_1q`.
    let dense_component = ConcreteCircuit::new(
        3,
        vec![
            GateInstruction::H(0),
            GateInstruction::X(0),
            GateInstruction::Y(8),
        ],
    );
    assert_eq!(
        StatevectorSimulator::new()
            .run(&dense_component)
            .unwrap_err(),
        SimError::QubitIndexOutOfRange {
            qubit: 8,
            num_qubits: 3
        }
    );
}

/// Every simulator configuration a repeated operand must be rejected under:
/// the parallel kernels at every size (`parallel_threshold: 0`, the #342
/// trigger) and the sequential ones, each fused and unfused.
fn every_configuration() -> Vec<StatevectorSimulator> {
    let mut sims = Vec::new();
    for parallel_threshold in [0, usize::MAX] {
        for fusion in [false, true] {
            sims.push(StatevectorSimulator {
                parallel_threshold,
                fusion,
                ..StatevectorSimulator::new()
            });
        }
    }
    sims
}

#[test]
fn simulator_rejects_a_repeated_operand_on_one_qubit() {
    // #342: `Cx(0, 0)` on a 1-qubit register made the controlled kernel compute
    // `1 << (n - 2)` with `n = 1` and, on the parallel path in a release build
    // without overflow-checks, write out of bounds (SIGSEGV).
    let circuit = ConcreteCircuit::new(1, vec![GateInstruction::Cx(0, 0)]);
    for sim in every_configuration() {
        assert_eq!(
            sim.run(&circuit).unwrap_err(),
            SimError::RepeatedQubit { qubit: 0 },
            "{sim:?}"
        );
    }
}

/// One gate of every multi-qubit shape with a repeated operand.
fn repeated_operand_gates() -> Vec<GateInstruction> {
    use polypus_circuit::GateParam::Fixed;
    vec![
        GateInstruction::Cx(1, 1),
        GateInstruction::Cz(1, 1),
        GateInstruction::Swap(1, 1),
        GateInstruction::Rzz {
            q0: 1,
            q1: 1,
            theta: Fixed(0.3),
        },
        GateInstruction::Rxx {
            q0: 1,
            q1: 1,
            theta: Fixed(0.3),
        },
        GateInstruction::Cp {
            q0: 1,
            q1: 1,
            theta: Fixed(0.3),
        },
        GateInstruction::Ch(1, 1),
        GateInstruction::Ccx(0, 1, 1),
        GateInstruction::Cswap(1, 0, 1),
    ]
}

#[test]
fn simulator_rejects_repeated_operands_on_every_multi_qubit_gate() {
    for gate in repeated_operand_gates() {
        // Preceded by valid gates so fusion has something to fold it into.
        let circuit = ConcreteCircuit::new(
            3,
            vec![GateInstruction::H(1), GateInstruction::Z(1), gate.clone()],
        );
        for sim in every_configuration() {
            assert_eq!(
                sim.run(&circuit).unwrap_err(),
                SimError::RepeatedQubit { qubit: 1 },
                "{gate:?} under {sim:?}"
            );
        }
    }
}

#[test]
fn statevector_apply_rejects_a_repeated_operand() {
    let mut sv = Statevector::new(1).expect("1 is well below MAX_QUBITS");
    assert_eq!(
        sv.apply(&GateInstruction::Cx(0, 0)).unwrap_err(),
        SimError::RepeatedQubit { qubit: 0 }
    );
    for gate in repeated_operand_gates() {
        let mut sv = Statevector::new(3).expect("3 is well below MAX_QUBITS");
        assert_eq!(
            sv.apply(&gate).unwrap_err(),
            SimError::RepeatedQubit { qubit: 1 },
            "{gate:?}"
        );
        // Rejected before any kernel ran: the state is untouched.
        assert_eq!(sv.amplitudes()[0], polypus_sim::C64::new(1.0, 0.0));
    }
}

#[test]
fn valid_circuits_are_unaffected() {
    // The guard must not perturb a well-formed run.
    let circuit = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .assign_parameters(&[])
        .expect("a bound 2-qubit circuit");
    let sv = StatevectorSimulator::new()
        .run(&circuit)
        .expect("a valid circuit still runs");
    assert_eq!(sv.num_qubits(), 2);
}
