//! Arithmetic on quantum registers, the building blocks of Shor's algorithm:
//! Draper's adder in the Fourier basis and, on top of it, Beauregard's modular
//! circuits.
//!
//! A register holds its value least-significant bit first: qubit `k` of the
//! register is bit `k` (contract C-3). The *Fourier basis* of an `m`-qubit
//! register is its state after
//! [`qft_with_options(m, false, false)`](super::qft_with_options), the QFT
//! without swaps. There qubit `j` of a register holding `b` carries the phase
//! `2π · (b mod 2^(j+1)) / 2^(j+1)`, so adding a constant is one phase per
//! qubit and needs no carry.
//!
//! Each block is a function that validates its arguments and returns a circuit
//! with its qubits laid out as its documentation says. The template of Shor's
//! algorithm builds on the same blocks without their checks or intermediate
//! circuits.

use super::emit::{circuit_from_ops, Op};
use super::MAX_QFT_QUBITS;
use crate::{CircuitError, ParameterizedCircuit};
use std::f64::consts::PI;

/// The qubits an emitted block is controlled by: none, one or two.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Controls {
    Zero,
    One(usize),
    Two(usize, usize),
}

/// Draper's adder: add the constant `addend` modulo `2^num_qubits` to a
/// register in the Fourier basis, when every control is `1`.
///
/// Qubits `0..num_controls` are the controls (at most two) and qubits
/// `num_controls..num_controls + num_qubits` the register, which must already
/// be in the Fourier basis: wrap the adder between
/// [`qft_with_options(num_qubits, false, false)`](super::qft_with_options) and
/// its inverse to add to a register holding a plain value. Uncontrolled, it is
/// one phase gate per qubit; with one control, one controlled phase per
/// qubit; with two, `3 · num_qubits + 2` gates.
///
/// The circuit has no free parameters and no measurements.
///
/// ```
/// use polypus_circuit::templates::arithmetic::phase_adder;
///
/// // Adds 5 to a 3-qubit register when qubit 0 is 1.
/// let qc = phase_adder(3, 5, 1).unwrap();
/// assert_eq!(qc.num_qubits(), 4);
/// assert_eq!(qc.gates().len(), 3);
/// ```
///
/// # Errors
///
/// - [`CircuitError::InvalidTemplateArgument`] for more than two controls or
///   an empty register.
/// - [`CircuitError::TemplateTooLarge`] above
///   [`MAX_QFT_QUBITS`](super::MAX_QFT_QUBITS) register qubits, the limit of
///   the QFT the adder works with.
pub fn phase_adder(
    num_qubits: usize,
    addend: u64,
    num_controls: usize,
) -> Result<ParameterizedCircuit, CircuitError> {
    let controls = match num_controls {
        0 => Controls::Zero,
        1 => Controls::One(0),
        2 => Controls::Two(0, 1),
        _ => {
            return Err(CircuitError::invalid_template_argument(
                "phase_adder",
                "num_controls",
                format!("at most 2 controls, got {num_controls}"),
            ))
        }
    };
    if num_qubits == 0 {
        return Err(CircuitError::invalid_template_argument(
            "phase_adder",
            "num_qubits",
            "the register needs at least one qubit",
        ));
    }
    if num_qubits > MAX_QFT_QUBITS {
        return Err(CircuitError::TemplateTooLarge {
            template: "phase_adder",
            num_qubits,
            max_qubits: MAX_QFT_QUBITS,
        });
    }
    let register: Vec<usize> = (num_controls..num_controls + num_qubits).collect();
    let len = match controls {
        Controls::Two(..) => 3 * num_qubits + 2,
        Controls::Zero | Controls::One(_) => num_qubits,
    };
    let mut ops = Vec::with_capacity(len);
    emit_phase_adder(&mut ops, controls, &register, addend, false);
    circuit_from_ops(num_controls + num_qubits, &ops)
}

/// Emit [`phase_adder`] for `addend` on `register`, controlled by `controls`;
/// with `subtract`, the same gates in the same order with every angle negated,
/// which subtract `addend` instead.
///
/// Preconditions, not checked: `register` is not empty and shares no qubit
/// with `controls`.
pub(crate) fn emit_phase_adder(
    out: &mut Vec<Op>,
    controls: Controls,
    register: &[usize],
    addend: u64,
    subtract: bool,
) {
    let sign = if subtract { -1.0 } else { 1.0 };
    let angles = || {
        register
            .iter()
            .enumerate()
            .map(move |(j, &qubit)| (qubit, sign * adder_angle(addend, j)))
    };
    match controls {
        Controls::Zero => out.extend(angles().map(|(qubit, theta)| Op::P(qubit, theta))),
        Controls::One(c) => out.extend(angles().map(|(qubit, theta)| Op::Cp(c, qubit, theta))),
        Controls::Two(c1, c2) => {
            // The doubly-controlled phase, decomposed and grouped over the
            // register: θ/2 on c2, -θ/2 on c1 ⊕ c2 and θ/2 on c1 add up to θ
            // exactly when both controls are 1.
            out.extend(angles().map(|(qubit, theta)| Op::Cp(c2, qubit, theta / 2.0)));
            out.push(Op::Cx(c1, c2));
            out.extend(angles().map(|(qubit, theta)| Op::Cp(c2, qubit, -theta / 2.0)));
            out.push(Op::Cx(c1, c2));
            out.extend(angles().map(|(qubit, theta)| Op::Cp(c1, qubit, theta / 2.0)));
        }
    }
}

/// The phase Draper's adder gives qubit `j` of the register to add `addend`:
/// `2π · (addend mod 2^(j+1)) / 2^(j+1)`.
///
/// Total and always finite. The reduction never shifts by 64 bits or more: from
/// `j + 1 = 64` on, `addend` is already below `2^(j+1)`. From `j + 1 = 1024` on
/// the divisor is infinite and the angle is `0`.
pub(crate) fn adder_angle(addend: u64, j: usize) -> f64 {
    let bits = j + 1;
    let reduced = if bits < 64 {
        addend & ((1u64 << bits) - 1)
    } else {
        addend
    };
    let divisor = i32::try_from(bits).map_or(f64::INFINITY, |bits| 2f64.powi(bits));
    2.0 * PI * reduced as f64 / divisor
}
