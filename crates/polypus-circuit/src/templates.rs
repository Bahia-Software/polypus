//! Ready-made circuit templates.
//!
//! Each template returns a [`ParameterizedCircuit`] so it stays composable: the
//! caller can keep chaining gates, append `measure_all`, or export it directly.
//! Templates add no measurements of their own.

use crate::ParameterizedCircuit;
use std::f64::consts::PI;

/// Quantum Fourier Transform on `num_qubits` qubits.
///
/// Follows the Qiskit `QFT` convention: big-endian, with the trailing
/// qubit-reversal swaps included. Equivalent to
/// [`qft_with_options(num_qubits, false, true)`](qft_with_options).
///
/// The circuit has no free parameters and no measurements.
///
/// ```
/// use polypus_circuit::templates::qft;
///
/// let qasm = qft(3).measure_all().to_qasm2_with_params(&[]).unwrap();
/// assert!(qasm.contains("swap q[0],q[2];"));
/// ```
pub fn qft(num_qubits: usize) -> ParameterizedCircuit {
    qft_with_options(num_qubits, false, true)
}

/// QFT with explicit control over inversion and the final swap network.
///
/// * `inverse` — build the inverse transform (QFT†) instead of the forward one.
///   It is the exact adjoint of the forward circuit with the same arguments.
/// * `swaps` — include the qubit-reversal swaps that reconcile the
///   big-endian convention with standard output order (appended for the forward
///   transform, prepended for the inverse). Set `false` when the surrounding
///   circuit already accounts for bit order.
///
/// `num_qubits == 0` yields an empty circuit; `num_qubits == 1` is a single
/// Hadamard (no controlled phases, no swaps).
pub fn qft_with_options(num_qubits: usize, inverse: bool, swaps: bool) -> ParameterizedCircuit {
    let mut qc = ParameterizedCircuit::new(num_qubits);
    if num_qubits == 0 {
        return qc;
    }

    // cp(θ) is symmetric in its two qubits; the controlled-phase angle between
    // qubits j and k is π / 2^(j-k), negated for the inverse.
    let sign = if inverse { -1.0 } else { 1.0 };
    let angle = |j: usize, k: usize| sign * PI / 2f64.powi((j - k) as i32);

    let add_swaps = |mut qc: ParameterizedCircuit| {
        for i in 0..num_qubits / 2 {
            qc = qc.swap(i, num_qubits - i - 1);
        }
        qc
    };

    if inverse {
        // Adjoint of the forward circuit: swaps first, then the rotation block
        // in reverse (each Hadamard last on its qubit, phases with reversed
        // sign and ascending target index).
        if swaps {
            qc = add_swaps(qc);
        }
        for j in 0..num_qubits {
            for k in 0..j {
                qc = qc.cp(k, j, angle(j, k));
            }
            qc = qc.h(j);
        }
    } else {
        // Forward: Hadamard on the most-significant qubit, then the descending
        // ladder of controlled phases; the swaps close the circuit.
        for j in (0..num_qubits).rev() {
            qc = qc.h(j);
            for k in (0..j).rev() {
                qc = qc.cp(k, j, angle(j, k));
            }
        }
        if swaps {
            qc = add_swaps(qc);
        }
    }
    qc
}
