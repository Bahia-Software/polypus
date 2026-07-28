//! [`IqpEncoder`]: the IQP / `ZZFeatureMap` construction — `H⊗n`, then
//! `rz(x_i)`, then `rzz(x_i·x_j)` over the pairs of an [`Entanglement`]
//! pattern (design doc §6.6).

use polypus_circuit::{Fixed, GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::{entanglement_pairs, Entanglement};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// IQP feature encoding: a layer of Hadamards, one `Rz(x_i)` per feature, and
/// one `Rzz(x_i·x_j)` per pair — the `ZZFeatureMap` pattern, whose non-linear
/// dependence on the data (the `x_i·x_j` products) is what makes its kernel
/// non-linear. Consumes **no** trainable parameters: every angle is a
/// [`Fixed`](polypus_circuit::GateParam::Fixed) value derived from `x`.
///
/// The three phases are emitted in order, over the first `num_features` active
/// qubits in logical-position order:
///
/// 1. `H` on each — unconditional, unlike [`AngleEncoder`]'s optional
///    `prepend_hadamard`: the `H⊗n` layer *is* the IQP structure, since the
///    diagonal `Rz`/`Rzz` that follow would otherwise only add a global phase.
/// 2. `Rz(x_i)` on each.
/// 3. `Rzz(x_i·x_j)` on every pair the [`entanglement`](Self::entanglement)
///    pattern selects — the plain product of the two features, as specified in
///    the design doc, not Qiskit's `(π − x_i)(π − x_j)` variant.
///
/// Surplus qubits (when `num_features < active.len()`) are left **completely**
/// untouched: no `H`, no `Rz`, and no participation in any pair. Putting a
/// featureless qubit in `|+⟩` would change its state to no purpose.
///
/// There is no `reps` field: re-uploading the whole `H`+`Rz`+`Rzz` block is
/// done by stacking a second `IqpEncoder` later in the model's layer list, the
/// same pattern [`AngleEncoder`] already establishes (design doc §6.1). And
/// unlike [`AmplitudeEncoder`], this encoder is not restricted to being the
/// first layer: its gates are ordinary unitaries that compose on whatever state
/// precedes them.
///
/// Scaling features into a sensible angle range is the caller's responsibility,
/// via [`Dataset::scale_features_to`](crate::Dataset::scale_features_to) —
/// note that the pair angles are *products*, so they scale quadratically.
///
/// [`AngleEncoder`]: crate::AngleEncoder
/// [`AmplitudeEncoder`]: crate::AmplitudeEncoder
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IqpEncoder {
    /// The connectivity of the `Rzz` interaction, over the logical positions of
    /// the encoded qubits.
    pub entanglement: Entanglement,
}

impl IqpEncoder {
    /// Build an IQP encoder with the original `ZZFeatureMap` (Havlíček et al.)
    /// connectivity: [`Entanglement::Full`].
    pub fn new() -> Self {
        IqpEncoder {
            entanglement: Entanglement::Full,
        }
    }
}

impl Default for IqpEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerOps for IqpEncoder {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
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
        // Only the qubits that carry a feature take part; the surplus ones are
        // left entirely alone (see the type docs).
        let encoded: &[usize] = &alloc.active[..x.len().min(alloc.active.len())];

        // Phase 1: H on every encoded qubit.
        for &qubit in encoded {
            qc.try_push(GateInstruction::H(qubit))?;
        }
        // Phase 2: Rz(x_i), the feature straight through as the angle.
        for (i, &qubit) in encoded.iter().enumerate() {
            qc.try_push(GateInstruction::Rz {
                qubit,
                theta: Fixed(x[i]),
            })?;
        }
        // Phase 3: Rzz(x_i · x_j) over the pattern's pairs — the non-linear
        // part of the map. Pairs are logical positions within `encoded`.
        for (i, j) in entanglement_pairs(encoded.len(), self.entanglement) {
            qc.try_push(GateInstruction::Rzz {
                q0: encoded[i],
                q1: encoded[j],
                theta: Fixed(x[i] * x[j]),
            })?;
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

    fn rz(qubit: usize, theta: f64) -> GateInstruction {
        GateInstruction::Rz {
            qubit,
            theta: Fixed(theta),
        }
    }

    fn rzz(q0: usize, q1: usize, theta: f64) -> GateInstruction {
        GateInstruction::Rzz {
            q0,
            q1,
            theta: Fixed(theta),
        }
    }

    /// Plan `enc` over `num_qubits`/`num_features` and emit it on `x`.
    fn emitted(enc: &IqpEncoder, num_qubits: usize, x: &[f64]) -> Vec<GateInstruction> {
        let mut c = ctx(num_qubits, x.len());
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(num_qubits);
        enc.emit(&mut qc, &alloc, x).unwrap();
        assert_eq!(qc.num_params, 0, "an encoder must bind no trainable θ");
        qc.gates
    }

    #[test]
    fn new_defaults_to_full_connectivity() {
        assert_eq!(IqpEncoder::new().entanglement, Entanglement::Full);
        assert_eq!(IqpEncoder::default(), IqpEncoder::new());
    }

    #[test]
    fn plan_reserves_no_params_and_snapshots_active() {
        let enc = IqpEncoder::new();
        let mut c = ctx(3, 3);
        let alloc = enc.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..0);
        assert_eq!(alloc.active, vec![0, 1, 2]);
        assert_eq!(c.param_cursor, 0);
    }

    #[test]
    fn plan_reserves_no_params_from_a_nonzero_cursor() {
        // An IQP encoder is *not* restricted to being the first layer, so its
        // empty range must sit wherever the cursor already is, unmoved.
        let enc = IqpEncoder::new();
        let mut c = LayerContext {
            param_cursor: 7,
            layers_planned: 2,
            ..ctx(2, 2)
        };
        let alloc = enc.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 7..7);
        assert_eq!(c.param_cursor, 7);
    }

    #[test]
    fn two_features_full_emits_the_three_phases_in_order() {
        // 2 features, Full → the single pair (0,1). The exact gate list pins
        // the phase order (all H, then all Rz, then all Rzz) and the angles:
        // the feature verbatim for Rz, its plain product for Rzz.
        let gates = emitted(&IqpEncoder::new(), 2, &[0.3, 0.5]);
        assert_eq!(
            gates,
            vec![
                GateInstruction::H(0),
                GateInstruction::H(1),
                rz(0, 0.3),
                rz(1, 0.5),
                rzz(0, 1, 0.3 * 0.5),
            ]
        );
    }

    #[test]
    fn three_features_full_pairs_every_combination() {
        let gates = emitted(&IqpEncoder::new(), 3, &[0.2, 0.4, 0.8]);
        assert_eq!(
            gates,
            vec![
                GateInstruction::H(0),
                GateInstruction::H(1),
                GateInstruction::H(2),
                rz(0, 0.2),
                rz(1, 0.4),
                rz(2, 0.8),
                rzz(0, 1, 0.2 * 0.4),
                rzz(0, 2, 0.2 * 0.8),
                rzz(1, 2, 0.4 * 0.8),
            ]
        );
    }

    #[test]
    fn linear_and_circular_follow_their_pattern() {
        let linear = IqpEncoder {
            entanglement: Entanglement::Linear,
        };
        let gates = emitted(&linear, 3, &[0.2, 0.4, 0.8]);
        assert_eq!(
            gates[6..],
            [rzz(0, 1, 0.2 * 0.4), rzz(1, 2, 0.4 * 0.8)],
            "Linear pairs the chain only"
        );

        let circular = IqpEncoder {
            entanglement: Entanglement::Circular,
        };
        let gates = emitted(&circular, 3, &[0.2, 0.4, 0.8]);
        assert_eq!(
            gates[6..],
            [
                rzz(0, 1, 0.2 * 0.4),
                rzz(1, 2, 0.4 * 0.8),
                // The wrap-around pair (2,0): the angle is still x_2 · x_0.
                rzz(2, 0, 0.8 * 0.2),
            ],
            "Circular adds the wrap-around pair"
        );
    }

    #[test]
    fn single_feature_emits_no_interaction() {
        // One encoded qubit: no pair exists under any pattern.
        let gates = emitted(&IqpEncoder::new(), 1, &[0.9]);
        assert_eq!(gates, vec![GateInstruction::H(0), rz(0, 0.9)]);
    }

    #[test]
    fn plan_rejects_more_features_than_active_qubits() {
        let enc = IqpEncoder::new();
        let mut c = ctx(3, 4);
        assert!(matches!(
            enc.plan(&mut c),
            Err(ValidationError::NotEnoughQubits {
                needed: 4,
                active: 3,
            })
        ));
    }

    #[test]
    fn surplus_qubits_receive_no_gate_at_all() {
        // 2 features over 4 active qubits: qubits 2 and 3 must not appear in a
        // single instruction — not even a Hadamard, and not in any pair.
        let gates = emitted(&IqpEncoder::new(), 4, &[0.3, 0.5]);
        assert_eq!(
            gates,
            vec![
                GateInstruction::H(0),
                GateInstruction::H(1),
                rz(0, 0.3),
                rz(1, 0.5),
                rzz(0, 1, 0.3 * 0.5),
            ]
        );
    }

    #[test]
    fn emit_addresses_physical_qubits_of_the_active_set() {
        // After a pooling layer the active set is not `0..n`: logical position
        // 0 may be physical qubit 1. Every phase must go through `active`.
        let enc = IqpEncoder::new();
        let alloc = LayerAllocation {
            params: 0..0,
            active: vec![1, 3],
        };
        let mut qc = ParameterizedCircuit::new(4);
        enc.emit(&mut qc, &alloc, &[0.3, 0.5]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                GateInstruction::H(1),
                GateInstruction::H(3),
                rz(1, 0.3),
                rz(3, 0.5),
                rzz(1, 3, 0.3 * 0.5),
            ]
        );
    }
}
