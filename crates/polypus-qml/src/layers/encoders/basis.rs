//! [`BasisEncoder`]: binary feature `x_j` → `X` on the active qubit at logical
//! position `j`, applied exactly when `x_j == 1.0` (design doc §6.7).

use polypus_circuit::{GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// Computational-basis encoding: a binary sample `x ∈ {0, 1}^n` becomes the
/// basis state `|x⟩`, one feature per qubit. Feature `x_j` flips the active
/// qubit at logical position `j` with a plain `X` when it is `1.0`, and leaves
/// it alone when it is `0.0` — the cheapest encoder of the catalogue: at most
/// one gate per feature, and no angle arithmetic at all.
///
/// PennyLane calls the equivalent operator `BasisEmbedding`. This crate names
/// its layers by **mechanism** rather than by mirroring an external library's
/// spelling — the same reason [`IqpEncoder`] is not called "ZZFeatureMap"
/// (design doc §6.6) — so the mechanism (encode into a *basis* state) names it
/// here.
///
/// Takes no configuration, like [`AmplitudeEncoder`]: the only thing to
/// configure would be which qubits to use, and that is already the model's
/// active set. Consumes **no** trainable parameters.
///
/// ## Features must be exactly `0.0` or `1.0`
///
/// Any other value — `0.5`, `2.0`, `-1.0` alike — is
/// [`QmlError::NonBinaryFeature`], reporting the offending position. There is no
/// rounding, no thresholding and no silent coercion: a non-binary feature handed
/// to a *basis* encoder is a data error the caller must resolve (scale or
/// binarize the dataset), exactly as a zero-norm sample is for
/// [`AmplitudeEncoder`]. Values outside `[0, 1]` are not special-cased — `0.5`
/// and `2.0` fail the same way, because neither names a basis state.
///
/// ## Position in the model
///
/// Unlike [`AmplitudeEncoder`], this encoder is **not** restricted to being the
/// first layer: an `X` conditioned on a *classical* feature is an ordinary
/// unitary, well defined on whatever state precedes it, so it composes anywhere
/// in the stack (the same reasoning as [`IqpEncoder`]'s). It therefore never
/// consults [`LayerContext::layers_planned`].
///
/// Surplus qubits (when `num_features < active.len()`) receive no gate at all,
/// the same convention as [`AngleEncoder`].
///
/// [`AngleEncoder`]: crate::AngleEncoder
/// [`AmplitudeEncoder`]: crate::AmplitudeEncoder
/// [`IqpEncoder`]: crate::IqpEncoder
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BasisEncoder;

impl LayerOps for BasisEncoder {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        // One qubit per feature, exactly like `AngleEncoder`/`IqpEncoder`.
        // There is no upper bound to check: the capacity is linear, not the
        // `2^k` amplitude budget `AmplitudeEncoder` has to guard.
        if ctx.num_features > ctx.active.len() {
            return Err(ValidationError::NotEnoughQubits {
                needed: ctx.num_features,
                active: ctx.active.len(),
            });
        }
        // Encoders reserve no θ: the range stays empty and the cursor unmoved.
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
        // One conditional `X` per feature, on the active qubit at that logical
        // position. If `x` is shorter than the active set, the surplus qubits
        // are left untouched (documented in the type docs).
        for (j, &qubit) in alloc.active.iter().take(x.len()).enumerate() {
            let bit = x[j];
            if bit == 1.0 {
                qc.try_push(GateInstruction::X(qubit))?;
            } else if bit != 0.0 {
                return Err(QmlError::NonBinaryFeature {
                    feature: j,
                    got: bit,
                });
            }
        }
        Ok(())
    }
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

    /// Plan `BasisEncoder` over `num_qubits` and emit it on `x`.
    fn emitted(num_qubits: usize, x: &[f64]) -> Vec<GateInstruction> {
        let mut c = ctx(num_qubits, x.len());
        let alloc = BasisEncoder.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(num_qubits);
        BasisEncoder.emit(&mut qc, &alloc, x).unwrap();
        assert_eq!(qc.num_params, 0, "an encoder must bind no trainable θ");
        qc.gates
    }

    #[test]
    fn plan_reserves_no_params_and_snapshots_active() {
        let mut c = ctx(3, 3);
        let alloc = BasisEncoder.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..0);
        assert_eq!(alloc.active, vec![0, 1, 2]);
        assert_eq!(c.param_cursor, 0);
    }

    #[test]
    fn plan_reserves_no_params_from_a_nonzero_cursor() {
        // A basis encoder is *not* restricted to being the first layer, so its
        // empty range must sit wherever the cursor already is, unmoved.
        let mut c = LayerContext {
            param_cursor: 5,
            layers_planned: 3,
            ..ctx(2, 2)
        };
        let alloc = BasisEncoder.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 5..5);
        assert_eq!(c.param_cursor, 5);
    }

    #[test]
    fn plan_rejects_more_features_than_active_qubits() {
        let mut c = ctx(3, 4);
        assert!(matches!(
            BasisEncoder.plan(&mut c),
            Err(ValidationError::NotEnoughQubits {
                needed: 4,
                active: 3,
            })
        ));
    }

    #[test]
    fn emit_flips_exactly_the_qubits_whose_feature_is_one() {
        // x = [1, 0, 1] over 3 qubits: qubit 1 must receive no gate at all —
        // pinned by comparing the whole instruction list, not by counting Xs.
        assert_eq!(
            emitted(3, &[1.0, 0.0, 1.0]),
            vec![GateInstruction::X(0), GateInstruction::X(2)]
        );
    }

    #[test]
    fn an_all_zero_sample_emits_nothing() {
        // |0…0⟩ is already the encoded state, so the layer is the identity.
        assert_eq!(emitted(3, &[0.0, 0.0, 0.0]), vec![]);
    }

    #[test]
    fn fewer_features_than_qubits_leaves_surplus_untouched() {
        // num_features = 1 over 3 active qubits: only the first is addressed.
        let mut c = ctx(3, 1);
        let alloc = BasisEncoder.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        BasisEncoder.emit(&mut qc, &alloc, &[1.0]).unwrap();
        assert_eq!(qc.gates, vec![GateInstruction::X(0)]);
    }

    #[test]
    fn emit_addresses_physical_qubits_of_the_active_set() {
        // After a pooling layer the active set is not `0..n`: logical position 0
        // may be physical qubit 1. The `X` must go through `active`.
        let alloc = LayerAllocation {
            params: 0..0,
            active: vec![1, 3],
        };
        let mut qc = ParameterizedCircuit::new(4);
        BasisEncoder.emit(&mut qc, &alloc, &[0.0, 1.0]).unwrap();
        assert_eq!(qc.gates, vec![GateInstruction::X(3)]);
    }

    /// Emit `x` over 2 qubits and return the error, for the rejection cases.
    fn emit_err(x: &[f64]) -> QmlError {
        let mut c = ctx(2, x.len());
        let alloc = BasisEncoder.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        BasisEncoder.emit(&mut qc, &alloc, x).unwrap_err()
    }

    #[test]
    fn emit_rejects_a_non_binary_feature_inside_the_unit_interval() {
        // 0.5 is neither basis state: no rounding, no thresholding. The reported
        // index is the *feature*'s, which only `emit` (iterating one at a time)
        // can know.
        assert_eq!(
            emit_err(&[0.0, 0.5]),
            QmlError::NonBinaryFeature {
                feature: 1,
                got: 0.5,
            }
        );
    }

    #[test]
    fn emit_rejects_a_non_binary_feature_outside_the_unit_interval() {
        // Values beyond [0, 1] are not a special case — they fail exactly like
        // the 0.5 above, because neither names a basis state.
        assert_eq!(
            emit_err(&[2.0, 0.0]),
            QmlError::NonBinaryFeature {
                feature: 0,
                got: 2.0,
            }
        );
        assert_eq!(
            emit_err(&[1.0, -1.0]),
            QmlError::NonBinaryFeature {
                feature: 1,
                got: -1.0,
            }
        );
    }
}
