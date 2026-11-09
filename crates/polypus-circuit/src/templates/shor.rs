//! The order-finding circuit of Shor's algorithm, after Beauregard's
//! "Circuit for Shor's algorithm using 2n+3 qubits" (2003), with the counting
//! register kept whole so the circuit runs on any backend without mid-circuit
//! measurement.

use super::arithmetic::{
    bit_length, emit_controlled_multiplication, gcd, mod_inverse, multiplier_gate_count,
};
use super::emit::{emit_qft, push_ops, Op};
use crate::{CircuitError, ParameterizedCircuit};

/// Most bits of the modulus [`shor_order_finding`] takes. With the default
/// `2n` counting qubits a 24-bit modulus builds 3,716,641 gates, within the
/// 4,000,000 instructions the OpenQASM importers read back; a 25-bit one
/// would build 4,322,801.
pub const MAX_SHOR_MODULUS_BITS: usize = 24;

const TEMPLATE: &str = "shor_order_finding";

/// The quantum part of Shor's algorithm: estimate the order `r` of `a` modulo
/// `modulus`, the smallest `r > 0` with `a^r ≡ 1 (mod modulus)`.
///
/// With `n` the bit length of `modulus` and `t` counting qubits (`2n` by
/// default), the qubits are the counting register `0..t`, the work register
/// `x` on `t..t + n`, the accumulator `b` on `t + n..t + 2n + 1` and an
/// ancilla on `t + 2n + 1`: `t + 2n + 2` qubits in all. The circuit sets `x`
/// to `1`, puts the counting register in uniform superposition, multiplies
/// `x` by `a^(2^k) mod modulus` controlled by counting qubit `k` (Beauregard's
/// in-place multiplication), and ends with the inverse QFT, swaps included, of
/// the counting register. Read as an integer, least-significant bit on qubit
/// 0, the counting register then concentrates on the values closest to
/// `s · 2^t / r` for `s = 0..r`; `b` and the ancilla end at `0`.
///
/// The circuit uses only `h`, `x`, `p`, `cp`, `cx`, `swap` and `cswap`, with
/// fixed angles, and has no free parameters and no measurements: measure the
/// counting register before running it on a backend. It has
/// `1 + t + t · c + t(t + 1)/2 + ⌊t/2⌋` gates, where
/// `c = 2(m(m + 1) + n(2m² + 13m + 10)) + n` and `m = n + 1`; at the largest
/// modulus, [`MAX_SHOR_MODULUS_BITS`] bits, that is 3,716,641 gates on 98
/// qubits. With the default counting register,
/// [`to_qasm3`](ParameterizedCircuit::to_qasm3) exports moduli of up to 16
/// bits; from 17 bits on it returns [`CircuitError::ExportLimit`], as the
/// program text would pass the 32 MiB the importer reads.
///
/// ```
/// use polypus_circuit::templates::shor_order_finding;
///
/// let qc = shor_order_finding(7, 15, None).unwrap();
/// assert_eq!(qc.num_qubits(), 18); // 8 counting qubits, x, b and the ancilla
/// let qc = qc.measure(0, 0).measure(1, 1).measure(2, 2);
/// ```
///
/// # Errors
///
/// [`CircuitError::InvalidTemplateArgument`], checked in this order: a
/// `modulus` below 3 or of more than [`MAX_SHOR_MODULUS_BITS`] bits; an `a`
/// outside `1..modulus`, or one that shares a factor with `modulus` (that
/// factor already divides it); `counting_qubits` outside `1..=2n`.
pub fn shor_order_finding(
    a: u64,
    modulus: u64,
    counting_qubits: Option<usize>,
) -> Result<ParameterizedCircuit, CircuitError> {
    let invalid = |argument, reason: String| {
        Err(CircuitError::invalid_template_argument(
            TEMPLATE, argument, reason,
        ))
    };
    if modulus < 3 {
        return invalid("modulus", format!("must be at least 3, got {modulus}"));
    }
    let n = bit_length(modulus);
    if n > MAX_SHOR_MODULUS_BITS {
        return invalid(
            "modulus",
            format!(
                "{modulus} has {n} bits, more than MAX_SHOR_MODULUS_BITS = \
                 {MAX_SHOR_MODULUS_BITS}: its circuit would pass the 4,000,000 \
                 instructions the OpenQASM importers read back"
            ),
        );
    }
    if !(1..modulus).contains(&a) {
        return invalid(
            "a",
            format!("must be between 1 and {}, got {a}", modulus - 1),
        );
    }
    let Some(a_inverse) = mod_inverse(a, modulus) else {
        let factor = gcd(a, modulus);
        return invalid(
            "a",
            format!(
                "must be coprime with the modulus, but {a} and {modulus} share \
                 the factor {factor}, which already divides {modulus}"
            ),
        );
    };
    let t = counting_qubits.unwrap_or(2 * n);
    if !(1..=2 * n).contains(&t) {
        return invalid(
            "counting_qubits",
            format!(
                "must be between 1 and {} (twice the bits of the modulus), got {t}",
                2 * n
            ),
        );
    }

    let x: Vec<usize> = (t..t + n).collect();
    let b: Vec<usize> = (t + n..t + 2 * n + 1).collect();
    let anc = t + 2 * n + 1;
    let counting: Vec<usize> = (0..t).collect();
    let mut qc = ParameterizedCircuit::new(t + 2 * n + 2);
    qc.gates.reserve_exact(shor_gate_count(n, t));

    // One buffer, emptied and refilled for each block: every gate is written
    // once, and memory peaks at the circuit plus one multiplication.
    let multiplication_len = 2 * multiplier_gate_count(n) + n;
    let mut ops = Vec::with_capacity(multiplication_len.max(t * (t + 1) / 2 + t / 2));
    ops.push(Op::X(x[0]));
    ops.extend(counting.iter().map(|&k| Op::H(k)));
    push_ops(&mut qc, &ops)?;
    let (mut power, mut inverse_power) = (a, a_inverse);
    for k in 0..t {
        ops.clear();
        emit_controlled_multiplication(&mut ops, k, &x, &b, anc, power, inverse_power, modulus);
        push_ops(&mut qc, &ops)?;
        power = square_mod(power, modulus);
        inverse_power = square_mod(inverse_power, modulus);
    }
    ops.clear();
    emit_qft(&mut ops, &counting, true, true);
    push_ops(&mut qc, &ops)?;
    Ok(qc)
}

/// Gates of [`shor_order_finding`] for an `n`-bit modulus and `t` counting
/// qubits: the `X`, `t` Hadamards, `t` controlled multiplications and the
/// inverse QFT with its swaps.
pub(crate) fn shor_gate_count(modulus_bits: usize, counting_qubits: usize) -> usize {
    let (n, t) = (modulus_bits, counting_qubits);
    let multiplication = 2 * multiplier_gate_count(n) + n;
    1 + t + t * multiplication + t * (t + 1) / 2 + t / 2
}

/// `x² mod modulus`, squared in `u128` so it cannot overflow.
fn square_mod(x: u64, modulus: u64) -> u64 {
    (u128::from(x) * u128::from(x) % u128::from(modulus)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::MAX_INSTRUCTIONS;

    #[test]
    fn max_shor_modulus_bits_is_the_largest_within_max_instructions() {
        assert_eq!(shor_gate_count(24, 48), 3_716_641);
        assert!(
            shor_gate_count(MAX_SHOR_MODULUS_BITS, 2 * MAX_SHOR_MODULUS_BITS) <= MAX_INSTRUCTIONS
        );
        assert_eq!(shor_gate_count(25, 50), 4_322_801);
        assert!(
            shor_gate_count(MAX_SHOR_MODULUS_BITS + 1, 2 * MAX_SHOR_MODULUS_BITS + 2)
                > MAX_INSTRUCTIONS
        );
    }
}
