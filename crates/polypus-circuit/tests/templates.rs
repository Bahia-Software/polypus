//! Integration tests for polypus-circuit circuit templates.

use polypus_circuit::templates::{qft, qft_with_options};
use polypus_circuit::{GateInstruction, GateParam};
use std::f64::consts::PI;

/// Controlled-phase angle used by the QFT between target `j` and control `k`.
fn cp(k: usize, j: usize, sign: f64) -> GateInstruction {
    GateInstruction::Cp {
        q0: k,
        q1: j,
        theta: GateParam::Fixed(sign * PI / 2f64.powi((j - k) as i32)),
    }
}

#[test]
fn qft_three_qubits_matches_qiskit_convention() {
    let qc = qft(3);

    // No free parameters, no measurements: directly exportable / composable.
    assert_eq!(qc.num_qubits, 3);
    assert_eq!(qc.num_params, 0);
    assert_eq!(qc.num_clbits(), 0);

    assert_eq!(
        qc.gates,
        vec![
            GateInstruction::H(2),
            cp(1, 2, 1.0), // π/2
            cp(0, 2, 1.0), // π/4
            GateInstruction::H(1),
            cp(0, 1, 1.0), // π/2
            GateInstruction::H(0),
            GateInstruction::Swap(0, 2),
        ]
    );
}

#[test]
fn qft_inverse_is_the_adjoint_of_the_forward() {
    let inv = qft_with_options(3, true, true);

    // Reverse of the forward circuit, phases negated, swaps first.
    assert_eq!(
        inv.gates,
        vec![
            GateInstruction::Swap(0, 2),
            GateInstruction::H(0),
            cp(0, 1, -1.0), // −π/2
            GateInstruction::H(1),
            cp(0, 2, -1.0), // −π/4
            cp(1, 2, -1.0), // −π/2
            GateInstruction::H(2),
        ]
    );
}

#[test]
fn qft_without_swaps_omits_the_swap_network() {
    let qc = qft_with_options(3, false, false);
    assert!(!qc
        .gates
        .iter()
        .any(|g| matches!(g, GateInstruction::Swap(..))));
    // Same rotation block as the default, just without the trailing swap.
    assert_eq!(qc.gates.len(), 6);
}

#[test]
fn qft_single_qubit_is_a_lone_hadamard() {
    let qc = qft(1);
    assert_eq!(qc.gates, vec![GateInstruction::H(0)]);
}

#[test]
fn qft_zero_qubits_is_empty() {
    let qc = qft(0);
    assert_eq!(qc.num_qubits, 0);
    assert!(qc.gates.is_empty());
}

#[test]
fn qft_exports_to_qasm() {
    let qasm = qft(3).measure_all().to_qasm2_with_params(&[]).unwrap();
    assert!(qasm.contains("h q[2];"));
    assert!(qasm.contains("cp(1.570796326795) q[1],q[2];"));
    assert!(qasm.contains("swap q[0],q[2];"));
    assert!(qasm.contains("measure q -> c;"));
}
