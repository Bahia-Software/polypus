//! The operations templates emit before they become gates.
//!
//! A block writes its [`Op`]s, with its qubits already resolved, onto the end
//! of a buffer it is handed instead of building a circuit of its own. Nested
//! blocks then copy no gate, and a template can reserve its exact size before
//! it emits anything. `Op` has only the gates the templates use, which keeps
//! [`Op::inverse`] and [`invert`] total.

use crate::{CircuitError, GateInstruction, GateParam, ParameterizedCircuit};
use std::f64::consts::PI;

/// One gate of a template: its qubits resolved and its angle fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Op {
    /// Hadamard.
    H(usize),
    /// Pauli-X.
    X(usize),
    /// Phase gate: qubit, angle.
    P(usize, f64),
    /// Controlled phase: control, target, angle.
    Cp(usize, usize, f64),
    /// CNOT: control, target.
    Cx(usize, usize),
    /// SWAP.
    Swap(usize, usize),
    /// Fredkin: control, then the two qubits it swaps.
    Cswap(usize, usize, usize),
}

impl Op {
    /// The inverse operation: `P` and `Cp` with the angle negated; every other
    /// operation is its own inverse.
    pub(crate) fn inverse(self) -> Op {
        match self {
            Op::P(qubit, lam) => Op::P(qubit, -lam),
            Op::Cp(control, target, theta) => Op::Cp(control, target, -theta),
            Op::H(_) | Op::X(_) | Op::Cx(..) | Op::Swap(..) | Op::Cswap(..) => self,
        }
    }

    /// The gate instruction, with `Fixed` angles.
    pub(crate) fn to_gate(self) -> GateInstruction {
        match self {
            Op::H(qubit) => GateInstruction::H(qubit),
            Op::X(qubit) => GateInstruction::X(qubit),
            Op::P(qubit, lam) => GateInstruction::P {
                qubit,
                lam: GateParam::Fixed(lam),
            },
            Op::Cp(q0, q1, theta) => GateInstruction::Cp {
                q0,
                q1,
                theta: GateParam::Fixed(theta),
            },
            Op::Cx(control, target) => GateInstruction::Cx(control, target),
            Op::Swap(a, b) => GateInstruction::Swap(a, b),
            Op::Cswap(control, a, b) => GateInstruction::Cswap(control, a, b),
        }
    }
}

/// Replace `ops` by their inverse: the reverse order, each operation inverted.
/// On the tail of a buffer (`invert(&mut out[start..])`) it inverts the block
/// just emitted, with no second buffer.
pub(crate) fn invert(ops: &mut [Op]) {
    ops.reverse();
    for op in ops.iter_mut() {
        *op = op.inverse();
    }
}

/// Append `ops` to `qc`, in order, each through
/// [`try_push`](ParameterizedCircuit::try_push).
pub(crate) fn push_ops(qc: &mut ParameterizedCircuit, ops: &[Op]) -> Result<(), CircuitError> {
    for op in ops {
        qc.try_push(op.to_gate())?;
    }
    Ok(())
}

/// A circuit on `num_qubits` qubits made of `ops`, with its gate list reserved
/// at its exact size.
pub(crate) fn circuit_from_ops(
    num_qubits: usize,
    ops: &[Op],
) -> Result<ParameterizedCircuit, CircuitError> {
    let mut qc = ParameterizedCircuit::new(num_qubits);
    qc.gates.reserve_exact(ops.len());
    push_ops(&mut qc, ops)?;
    Ok(qc)
}

/// Emit the QFT of [`qft_with_options`](super::qft_with_options) with its
/// qubit `i` on `register[i]`.
pub(crate) fn emit_qft(out: &mut Vec<Op>, register: &[usize], inverse: bool, swaps: bool) {
    let n = register.len();
    // cp(θ) is symmetric in its two qubits; the controlled-phase angle between
    // qubits j and k is π / 2^(j-k), negated for the inverse.
    let sign = if inverse { -1.0 } else { 1.0 };
    let angle = |j: usize, k: usize| sign * PI / 2f64.powi((j - k) as i32);
    let reversal = (0..n / 2).map(|i| Op::Swap(register[i], register[n - i - 1]));

    if inverse {
        // Adjoint of the forward circuit: swaps first, then the rotation block
        // in reverse (each Hadamard last on its qubit, phases with reversed
        // sign and ascending target index).
        if swaps {
            out.extend(reversal);
        }
        for j in 0..n {
            for k in 0..j {
                out.push(Op::Cp(register[k], register[j], angle(j, k)));
            }
            out.push(Op::H(register[j]));
        }
    } else {
        // Forward: Hadamard on the most-significant qubit, then the descending
        // ladder of controlled phases; the swaps close the circuit.
        for j in (0..n).rev() {
            out.push(Op::H(register[j]));
            for k in (0..j).rev() {
                out.push(Op::Cp(register[k], register[j], angle(j, k)));
            }
        }
        if swaps {
            out.extend(reversal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_of_the_forward_qft_is_the_inverse_qft() {
        let mut a = Vec::new();
        emit_qft(&mut a, &[0, 1, 2], false, false);
        invert(&mut a);
        let mut b = Vec::new();
        emit_qft(&mut b, &[0, 1, 2], true, false);
        assert_eq!(a, b);
    }
}
