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
//! These assertions hold identically under `cargo test` and `cargo test
//! --release`. Before the fix, the debug run panics in the kernel assertion and
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
    let circuit = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![GateInstruction::Cx(0, 9)],
    };
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
    let diagonal_run = ConcreteCircuit {
        num_qubits: 3,
        gates: vec![
            GateInstruction::Z(0),
            GateInstruction::Z(7),
            GateInstruction::Z(1),
        ],
    };
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
    let dense_component = ConcreteCircuit {
        num_qubits: 3,
        gates: vec![
            GateInstruction::H(0),
            GateInstruction::X(0),
            GateInstruction::Y(8),
        ],
    };
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
