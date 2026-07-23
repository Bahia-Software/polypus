//! [`AmplitudeEncoder`]: prepare `|ψ(x)⟩ = Σⱼ x̃ⱼ|j⟩` on the active qubits,
//! with `x̃ = x/‖x‖₂` zero-padded to `2^k` (design doc §6.2).
//!
//! v1 is restricted to **real, non-negative amplitudes after normalization**
//! (the classical-data case). The state is prepared with the optimized
//! Möttönen scheme: one *multiplexed* `Ry` rotation per level of a binary
//! tree over the `k` active qubits, each decomposed into plain `ry` + `cx`
//! gates so it stays inside the C-2 vocabulary — no native multi-controlled
//! gate is required.
//!
//! ## Cost
//!
//! This is **the expensive layer of the catalogue**: `O(2^k)` gates per sample
//! (`2^k - 1` `Ry` and `2^k - 2` `Cx`), and every template pays it once. On
//! top of that, the *classical* angle precomputation
//! ([`walsh_hadamard_transform`]) is `O(n²)` per level (`n = 2^l`), which for
//! large `k` can dominate the `O(2^k)` gate count itself. v1 keeps the direct
//! `O(n²)` transform for simplicity rather than the fast Walsh–Hadamard
//! transform (FWHT); this is a deliberate, documented choice (design doc §12).

use polypus_circuit::{Fixed, GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// Amplitude encoding via multiplexed `Ry` state preparation (Möttönen).
///
/// Takes no configuration in v1. Must be the **first** layer of a model — it
/// prepares a state from `|0…0⟩`, so it cannot compose on top of earlier gates
/// (see [`plan`](LayerOps::plan)). Consumes **no** trainable parameters: every
/// angle is a [`Fixed`] value derived from the sample.
#[derive(Debug, Clone, PartialEq)]
pub struct AmplitudeEncoder;

impl LayerOps for AmplitudeEncoder {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        // Must be first: it prepares |ψ⟩ from |0…0⟩, not a composable unitary.
        if ctx.layers_planned != 0 {
            return Err(ValidationError::AmplitudeEncoderNotFirst);
        }
        let k = ctx.active.len();
        let max = 1usize << k;
        if ctx.num_features > max {
            return Err(ValidationError::TooManyFeatures {
                max,
                got: ctx.num_features,
            });
        }
        // Like every encoder, reserves no θ: the range stays empty.
        Ok(LayerAllocation {
            params: ctx.param_cursor..ctx.param_cursor,
            active: ctx.active.clone(),
        })
    }

    fn emit(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        x: &[f64],
    ) -> Result<(), QmlError> {
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(QmlError::ZeroNormSample);
        }
        let k = alloc.active.len();
        let dim = 1usize << k;
        // The normalized amplitudes, zero-padded to 2^k. Index `j` addresses
        // basis state |j⟩ under the C-3 little-endian convention (`plan`
        // guarantees `x.len() <= dim`, so the write never overflows).
        let mut amps = vec![0.0; dim];
        for (j, &xi) in x.iter().enumerate() {
            amps[j] = xi / norm;
        }
        // Fix qubits MSB-first: `active[k-1]` (the most significant bit of the
        // amplitude index) is set at level 0 with the coarsest split, down to
        // `active[0]` at the last level.
        let qubits: Vec<usize> = alloc.active.iter().rev().copied().collect();
        for l in 0..k {
            let thetas = level_angles(&amps, l, k);
            emit_multiplexed_ry(qc, &qubits[..l], qubits[l], &thetas)?;
        }
        Ok(())
    }
}

/// The `2^l` conditional `Ry` angles for level `l` of the preparation tree.
///
/// Each prefix `p` (the `l` already-fixed most-significant bits) owns a
/// contiguous block of `2^{k-l}` amplitudes. Its angle rotates the mass
/// balance between the block's lower half (next bit `0`) and its full mass:
/// `θ_p = 2·arccos(√(M_{p‖0} / M_p))`, where `M` is a sum of squared
/// amplitudes. A prefix carrying no mass (`M_p == 0`) contributes an
/// irrelevant `θ_p = 0` (nothing to steer).
fn level_angles(amps: &[f64], l: usize, k: usize) -> Vec<f64> {
    let block = 1usize << (k - l);
    let half = block / 2;
    (0..(1usize << l))
        .map(|p| {
            let seg = &amps[p * block..(p + 1) * block];
            let m_p: f64 = seg.iter().map(|v| v * v).sum();
            if m_p <= 0.0 {
                return 0.0;
            }
            let m_p0: f64 = seg[..half].iter().map(|v| v * v).sum();
            2.0 * (m_p0 / m_p).sqrt().acos()
        })
        .collect()
}

/// The angle transform of a multiplexed `Ry` — the rotation angles the
/// decomposition emits, given the `2^l` target angles (Möttönen).
///
/// This is a Walsh–Hadamard-style transform, but with the **Gray code** of the
/// output index in the sign: `φ_t = (1/n)·Σⱼ (-1)^popcount(gray(t) & j)·θⱼ`,
/// `gray(t) = t ^ (t >> 1)`. The Gray-code permutation is not cosmetic — the
/// `Cx` gates are placed at Gray-code positions (see [`emit_multiplexed_ry`]),
/// so the matrix inverting that placement is Möttönen's `Mₜⱼ =
/// (-1)^popcount(gray(t) & j)`, not the plain Sylvester–Hadamard `(-1)^popcount(t
/// & j)`. The two coincide only for `l ≤ 1` (Gray code is the identity on a
/// single bit); they diverge from `l = 2` onward (`gray(2)=3`, `gray(3)=2`),
/// where the plain transform prepares the wrong state.
///
/// Direct `O(n²)` evaluation (`n = thetas.len() == 2^l`), kept direct for
/// simplicity — see the module docs for why the FWHT is deliberately *not*
/// used in v1.
fn walsh_hadamard_transform(thetas: &[f64]) -> Vec<f64> {
    let n = thetas.len();
    (0..n)
        .map(|t| {
            let gray_t = t ^ (t >> 1);
            let sum: f64 = (0..n)
                .map(|j| {
                    let sign = if (gray_t & j).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * thetas[j]
                })
                .sum();
            sum / n as f64
        })
        .collect()
}

/// Emit a multiplexed `Ry`: apply `Ry(thetas[s])` to `target` conditioned on
/// the classical control pattern `s` of `controls`, decomposed into plain `ry`
/// + `cx` (no native multi-controlled gate).
///
/// `controls[0]` is the most significant bit of the pattern index,
/// `controls[l-1]` the least. With `l` controls the circuit is `2^l` `Ry`
/// rotations (the [`walsh_hadamard_transform`] of `thetas`) interleaved with
/// `2^l` `Cx` in Gray-code order, the last `Cx` wrapping around to close the
/// pattern. The `l == 0` case is a single unconditional `Ry(thetas[0])`.
fn emit_multiplexed_ry(
    qc: &mut ParameterizedCircuit,
    controls: &[usize],
    target: usize,
    thetas: &[f64],
) -> Result<(), QmlError> {
    let l = controls.len();
    if l == 0 {
        qc.try_push(GateInstruction::Ry {
            qubit: target,
            theta: Fixed(thetas[0]),
        })?;
        return Ok(());
    }
    let n = thetas.len(); // == 2^l
    let phis = walsh_hadamard_transform(thetas);
    for (t, &phi) in phis.iter().enumerate() {
        qc.try_push(GateInstruction::Ry {
            qubit: target,
            theta: Fixed(phi),
        })?;
        let next = (t + 1) % n; // wrap around after the last rotation
        let (gray_t, gray_next) = (t ^ (t >> 1), next ^ (next >> 1));
        let flipped_bit = (gray_t ^ gray_next).trailing_zeros() as usize;
        // Bit 0 (LSB of the Gray-code integer) is the *least* significant
        // control, i.e. controls[l-1]; bit (l-1) is controls[0].
        let control_index = l - 1 - flipped_bit;
        qc.try_push(GateInstruction::Cx(controls[control_index], target))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(num_qubits: usize, num_features: usize) -> LayerContext {
        LayerContext {
            active: (0..num_qubits).collect(),
            num_features,
            param_cursor: 0,
            layers_planned: 0,
        }
    }

    /// Assert the circuit is exactly one `Ry(theta)` on `qubit` (the k=1 case).
    fn assert_single_ry(gates: &[GateInstruction], qubit: usize, theta: f64) {
        assert_eq!(gates.len(), 1, "expected a single gate, got {gates:?}");
        match &gates[0] {
            GateInstruction::Ry {
                qubit: q,
                theta: Fixed(t),
            } => {
                assert_eq!(*q, qubit);
                assert!((t - theta).abs() < 1e-12, "theta {t} != expected {theta}");
            }
            other => panic!("expected Ry, got {other:?}"),
        }
    }

    fn emit_gates(enc: &AmplitudeEncoder, num_qubits: usize, x: &[f64]) -> Vec<GateInstruction> {
        let mut c = ctx(num_qubits, x.len());
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(num_qubits);
        enc.emit(&mut qc, &alloc, x).unwrap();
        qc.gates
    }

    // --- k = 1: hand-verifiable base cases (the sanity anchor) -------------

    #[test]
    fn k1_all_mass_in_branch_zero_is_identity() {
        // x = [1, 0]: all mass on |0⟩ → Ry(0).
        assert_single_ry(&emit_gates(&AmplitudeEncoder, 1, &[1.0, 0.0]), 0, 0.0);
    }

    #[test]
    fn k1_all_mass_in_branch_one_is_pi() {
        // x = [0, 1]: all mass on |1⟩ → Ry(π) (Ry(π)|0⟩ = |1⟩).
        assert_single_ry(
            &emit_gates(&AmplitudeEncoder, 1, &[0.0, 1.0]),
            0,
            std::f64::consts::PI,
        );
    }

    #[test]
    fn k1_equal_superposition_is_half_pi() {
        // x = [1, 1]: equal split → Ry(π/2).
        assert_single_ry(
            &emit_gates(&AmplitudeEncoder, 1, &[1.0, 1.0]),
            0,
            std::f64::consts::FRAC_PI_2,
        );
    }

    // --- gate counts: 2^k - 1 Ry and 2^k - 2 Cx (counted, not assumed) -----

    fn count_ry_cx(gates: &[GateInstruction]) -> (usize, usize) {
        let ry = gates
            .iter()
            .filter(|g| matches!(g, GateInstruction::Ry { .. }))
            .count();
        let cx = gates
            .iter()
            .filter(|g| matches!(g, GateInstruction::Cx(..)))
            .count();
        // Nothing else should be emitted.
        assert_eq!(ry + cx, gates.len(), "unexpected gate kind in {gates:?}");
        (ry, cx)
    }

    #[test]
    fn k2_gate_counts() {
        let gates = emit_gates(&AmplitudeEncoder, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(count_ry_cx(&gates), (3, 2)); // 2^2 - 1, 2^2 - 2
    }

    #[test]
    fn k3_gate_counts() {
        let gates = emit_gates(
            &AmplitudeEncoder,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        assert_eq!(count_ry_cx(&gates), (7, 6)); // 2^3 - 1, 2^3 - 2
    }

    // --- validation and runtime errors -------------------------------------

    #[test]
    fn plan_rejects_encoder_not_first() {
        let mut c = ctx(2, 2);
        c.layers_planned = 1; // a layer was already planned before this one
        assert!(matches!(
            AmplitudeEncoder.plan(&mut c),
            Err(ValidationError::AmplitudeEncoderNotFirst)
        ));
    }

    #[test]
    fn plan_rejects_too_many_features() {
        // 5 features cannot fit in 2^2 = 4 amplitudes.
        let mut c = ctx(2, 5);
        assert!(matches!(
            AmplitudeEncoder.plan(&mut c),
            Err(ValidationError::TooManyFeatures { max: 4, got: 5 })
        ));
    }

    #[test]
    fn plan_reserves_no_params() {
        let mut c = ctx(3, 4);
        let alloc = AmplitudeEncoder.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..0);
        assert_eq!(c.param_cursor, 0);
        assert_eq!(alloc.active, vec![0, 1, 2]);
    }

    #[test]
    fn emit_rejects_zero_norm_sample() {
        let enc = AmplitudeEncoder;
        let mut c = ctx(2, 4);
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        assert_eq!(
            enc.emit(&mut qc, &alloc, &[0.0, 0.0, 0.0, 0.0]),
            Err(QmlError::ZeroNormSample)
        );
    }

    // --- the classical angle transform -------------------------------------

    #[test]
    fn walsh_hadamard_transform_matches_definition() {
        // n = 2: [[1,1],[1,-1]]/2 · thetas.
        let out = walsh_hadamard_transform(&[3.0, 1.0]);
        assert!((out[0] - 2.0).abs() < 1e-12); // (3 + 1)/2
        assert!((out[1] - 1.0).abs() < 1e-12); // (3 - 1)/2
    }
}
