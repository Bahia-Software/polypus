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

use super::emit::{circuit_from_ops, emit_qft, invert, Op};
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

/// Beauregard's modular adder (figure 5 of "Circuit for Shor's algorithm
/// using 2n+3 qubits", 2003): add `addend` modulo `modulus` to a register in
/// the Fourier basis, when both controls are `1`.
///
/// With `n` the bit length of `modulus`, the qubits are `[c1, c2, b_0..b_n,
/// anc]`: the two controls, an `(n + 1)`-qubit register (one bit above the
/// modulus, for the overflow) and an ancilla, `n + 4` qubits in all. `addend`
/// is reduced modulo `modulus`. For a register holding `b < modulus`, the
/// circuit leaves `(b + addend) mod modulus` when both controls are `1` and
/// `b` otherwise, and returns the ancilla to `|0⟩`; for `b ≥ modulus` both the
/// register and the ancilla are undefined. It has `2m² + 13m + 10` gates for
/// `m = n + 1`, no free parameters and no measurements.
///
/// ```
/// use polypus_circuit::templates::arithmetic::modular_phase_adder;
///
/// let qc = modular_phase_adder(7, 15).unwrap();
/// assert_eq!(qc.num_qubits(), 8);
/// assert_eq!(qc.gates().len(), 125);
/// ```
///
/// # Errors
///
/// [`CircuitError::InvalidTemplateArgument`] for a `modulus` below 2.
pub fn modular_phase_adder(
    addend: u64,
    modulus: u64,
) -> Result<ParameterizedCircuit, CircuitError> {
    if modulus < 2 {
        return Err(CircuitError::invalid_template_argument(
            "modular_phase_adder",
            "modulus",
            format!("must be at least 2, got {modulus}"),
        ));
    }
    let m = bit_length(modulus) + 1;
    let register: Vec<usize> = (2..2 + m).collect();
    let anc = 2 + m;
    let mut ops = Vec::with_capacity(2 * m * m + 13 * m + 10);
    emit_modular_phase_adder(&mut ops, (0, 1), &register, anc, addend % modulus, modulus);
    circuit_from_ops(m + 3, &ops)
}

/// Emit [`modular_phase_adder`] on `register`, controlled by `controls` and
/// with the ancilla `anc`.
///
/// Preconditions, not checked: `modulus ≥ 2`, `addend < modulus`, `register`
/// has `bit_length(modulus) + 1` qubits, and the controls, the register and
/// `anc` are distinct qubits.
pub(crate) fn emit_modular_phase_adder(
    out: &mut Vec<Op>,
    controls: (usize, usize),
    register: &[usize],
    anc: usize,
    addend: u64,
    modulus: u64,
) {
    let (c1, c2) = controls;
    let top = register[register.len() - 1];
    let controlled = Controls::Two(c1, c2);
    // b + a, minus N: the top bit flags b + a < N, which the ancilla records
    // and undoes by adding N back.
    emit_phase_adder(out, controlled, register, addend, false);
    emit_phase_adder(out, Controls::Zero, register, modulus, true);
    emit_qft(out, register, true, false);
    out.push(Op::Cx(top, anc));
    emit_qft(out, register, false, false);
    emit_phase_adder(out, Controls::One(anc), register, modulus, false);
    // Subtracting a again leaves the top bit clear exactly when the ancilla
    // is set, which resets the ancilla; adding a back restores the sum.
    emit_phase_adder(out, controlled, register, addend, true);
    emit_qft(out, register, true, false);
    out.extend([Op::X(top), Op::Cx(top, anc), Op::X(top)]);
    emit_qft(out, register, false, false);
    emit_phase_adder(out, controlled, register, addend, false);
}

/// Beauregard's controlled modular multiplier (figure 6): add `multiplier · x`
/// modulo `modulus` to the register `b`, when the control is `1`.
///
/// With `n` the bit length of `modulus`, the qubits are `[c, x_0..x_{n-1},
/// b_0..b_n, anc]`, `2n + 3` in all: the control, the `n`-qubit factor `x`, an
/// `(n + 1)`-qubit accumulator `b` in plain (not Fourier) values, and the
/// ancilla of the modular adder. `multiplier` is reduced modulo `modulus`. For
/// `x` and `b` below `modulus`, the circuit maps `|x⟩|b⟩` to
/// `|x⟩|(b + multiplier · x) mod modulus⟩` when the control is `1`, leaves
/// both registers as they are otherwise, and returns the ancilla to `|0⟩`. It
/// is the QFT of `b`, one [`modular_phase_adder`] of `(multiplier · 2^i) mod
/// modulus` controlled by `c` and `x_i` for each bit of `x`, and the inverse
/// QFT: `m(m + 1) + n(2m² + 13m + 10)` gates for `m = n + 1`.
///
/// ```
/// use polypus_circuit::templates::arithmetic::controlled_modular_multiplier;
///
/// let qc = controlled_modular_multiplier(7, 15).unwrap();
/// assert_eq!(qc.num_qubits(), 11);
/// assert_eq!(qc.gates().len(), 530);
/// ```
///
/// # Errors
///
/// [`CircuitError::InvalidTemplateArgument`] for a `modulus` below 2.
pub fn controlled_modular_multiplier(
    multiplier: u64,
    modulus: u64,
) -> Result<ParameterizedCircuit, CircuitError> {
    let layout = MultiplierLayout::new("controlled_modular_multiplier", modulus)?;
    let mut ops = Vec::with_capacity(layout.multiplier_len());
    emit_controlled_multiplier(
        &mut ops,
        layout.control,
        &layout.x,
        &layout.b,
        layout.anc,
        multiplier % modulus,
        modulus,
    );
    circuit_from_ops(layout.num_qubits(), &ops)
}

/// Beauregard's controlled modular multiplication in place (figure 7):
/// multiply the register `x` by `multiplier` modulo `modulus`, when the
/// control is `1`.
///
/// The qubits are those of [`controlled_modular_multiplier`], `[c,
/// x_0..x_{n-1}, b_0..b_n, anc]`. For `x` below `modulus` and `b` and the
/// ancilla at `0`, the circuit maps `|x⟩` to `|multiplier · x mod modulus⟩`
/// when the control is `1` and leaves it otherwise; `b` and the ancilla end
/// at `0`. It is the multiplier with `multiplier`, a controlled swap of each
/// `x_i` with `b_i`, and the inverse of the multiplier with the inverse of
/// `multiplier` modulo `modulus`: `2 · multiplier + n` gates, with `multiplier`
/// the count of [`controlled_modular_multiplier`].
///
/// ```
/// use polypus_circuit::templates::arithmetic::controlled_modular_multiplication;
///
/// let qc = controlled_modular_multiplication(7, 15).unwrap();
/// assert_eq!(qc.num_qubits(), 11);
/// assert_eq!(qc.gates().len(), 1064);
/// ```
///
/// # Errors
///
/// [`CircuitError::InvalidTemplateArgument`] for a `modulus` below 2, or a
/// `multiplier` that shares a factor with `modulus`: it has no inverse modulo
/// `modulus`, and the multiplication could not be undone.
pub fn controlled_modular_multiplication(
    multiplier: u64,
    modulus: u64,
) -> Result<ParameterizedCircuit, CircuitError> {
    const TEMPLATE: &str = "controlled_modular_multiplication";
    let layout = MultiplierLayout::new(TEMPLATE, modulus)?;
    let reduced = multiplier % modulus;
    let Some(inverse) = mod_inverse(reduced, modulus) else {
        let factor = gcd(multiplier, modulus);
        return Err(CircuitError::invalid_template_argument(
            TEMPLATE,
            "multiplier",
            format!(
                "{multiplier} shares the factor {factor} with the modulus {modulus}, \
                 so it has no inverse modulo {modulus}"
            ),
        ));
    };
    let mut ops = Vec::with_capacity(2 * layout.multiplier_len() + layout.x.len());
    emit_controlled_multiplication(
        &mut ops,
        layout.control,
        &layout.x,
        &layout.b,
        layout.anc,
        reduced,
        inverse,
        modulus,
    );
    circuit_from_ops(layout.num_qubits(), &ops)
}

/// The qubits of the multiplication blocks: `[c, x_0..x_{n-1}, b_0..b_n, anc]`.
struct MultiplierLayout {
    control: usize,
    x: Vec<usize>,
    b: Vec<usize>,
    anc: usize,
}

impl MultiplierLayout {
    /// The layout for `modulus`, which `template` requires to be at least 2.
    fn new(template: &'static str, modulus: u64) -> Result<Self, CircuitError> {
        if modulus < 2 {
            return Err(CircuitError::invalid_template_argument(
                template,
                "modulus",
                format!("must be at least 2, got {modulus}"),
            ));
        }
        let n = bit_length(modulus);
        Ok(MultiplierLayout {
            control: 0,
            x: (1..=n).collect(),
            b: (n + 1..2 * n + 2).collect(),
            anc: 2 * n + 2,
        })
    }

    fn num_qubits(&self) -> usize {
        self.anc + 1
    }

    /// Gates of [`controlled_modular_multiplier`] on this layout.
    fn multiplier_len(&self) -> usize {
        let (n, m) = (self.x.len(), self.b.len());
        m * (m + 1) + n * (2 * m * m + 13 * m + 10)
    }
}

/// Emit [`controlled_modular_multiplier`] on `x` and `b`, controlled by
/// `control` and with the ancilla `anc`.
///
/// Preconditions, not checked: `modulus ≥ 2`, `multiplier < modulus`, `x` has
/// `bit_length(modulus)` qubits and `b` one more, and every qubit is distinct.
pub(crate) fn emit_controlled_multiplier(
    out: &mut Vec<Op>,
    control: usize,
    x: &[usize],
    b: &[usize],
    anc: usize,
    multiplier: u64,
    modulus: u64,
) {
    emit_qft(out, b, false, false);
    // multiplier · 2^i mod N, doubled in u128 so that 2 · (N - 1) cannot wrap.
    let mut addend = multiplier;
    for &x_i in x {
        emit_modular_phase_adder(out, (control, x_i), b, anc, addend, modulus);
        addend = (u128::from(addend) * 2 % u128::from(modulus)) as u64;
    }
    emit_qft(out, b, true, false);
}

/// Emit [`controlled_modular_multiplication`] on `x` and `b`, controlled by
/// `control` and with the ancilla `anc`; `inverse` is the inverse of
/// `multiplier` modulo `modulus`.
///
/// The inverse multiplier is emitted onto the end of `out` and inverted in
/// place, so the block needs no buffer of its own. Preconditions, not checked:
/// those of [`emit_controlled_multiplier`], and `multiplier · inverse ≡ 1`
/// modulo `modulus`.
pub(crate) fn emit_controlled_multiplication(
    out: &mut Vec<Op>,
    control: usize,
    x: &[usize],
    b: &[usize],
    anc: usize,
    multiplier: u64,
    inverse: u64,
    modulus: u64,
) {
    emit_controlled_multiplier(out, control, x, b, anc, multiplier, modulus);
    out.extend(
        x.iter()
            .zip(b)
            .map(|(&x_i, &b_i)| Op::Cswap(control, x_i, b_i)),
    );
    let start = out.len();
    emit_controlled_multiplier(out, control, x, b, anc, inverse, modulus);
    invert(&mut out[start..]);
}

/// The greatest common divisor of `a` and `b` (`gcd(0, 0) = 0`).
pub(crate) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The inverse of `a` modulo `modulus`, or `None` when they share a factor
/// (or `modulus` is 0). The extended Euclidean algorithm runs in `i128`, so its
/// negative coefficients cannot overflow.
pub(crate) fn mod_inverse(a: u64, modulus: u64) -> Option<u64> {
    let (mut old_r, mut r) = (i128::from(a.checked_rem(modulus)?), i128::from(modulus));
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    if old_r != 1 {
        return None;
    }
    u64::try_from(old_s.rem_euclid(i128::from(modulus))).ok()
}

/// The number of bits of `n`: `0` for `0`, else one more than the index of its
/// highest set bit.
pub(crate) fn bit_length(n: u64) -> usize {
    (u64::BITS - n.leading_zeros()) as usize
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
