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
