//! [`AngleEncoder`]: feature `x_j` → rotation `axis(x_j)` on the active qubit
//! at logical position `j`.

use polypus_circuit::{Fixed, GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::RotationAxis;
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// Angle encoding: each feature becomes a single-qubit rotation about a fixed
/// axis. Consumes **no** trainable parameters — every angle is a
/// [`Fixed`](polypus_circuit::GateParam::Fixed) value taken straight from `x`.
///
/// `Rz` on `|0⟩` is only a global phase, so when `axis == Rz` a Hadamard is
/// prepended by default (the `ZFeatureMap` pattern: `H` then `Rz(x)`);
/// `Rx`/`Ry` need no such preparation.
///
/// Scaling features into a sensible angle range (the usual convention is
/// `[0, π]`) is the caller's responsibility, via
/// [`Dataset::scale_features_to`](crate::Dataset::scale_features_to).
#[derive(Debug, Clone, PartialEq)]
pub struct AngleEncoder {
    /// The rotation axis applied to every encoded feature.
    pub axis: RotationAxis,
    /// Whether to prepend a Hadamard before each rotation. Defaults to `true`
    /// for `Rz` (otherwise the rotation is a no-op on `|0⟩`), `false` otherwise.
    pub prepend_hadamard: bool,
}

impl AngleEncoder {
    /// Build an angle encoder for `axis`, with `prepend_hadamard` defaulted to
    /// `true` exactly when `axis == RotationAxis::Rz`.
    pub fn new(axis: RotationAxis) -> Self {
        AngleEncoder {
            axis,
            prepend_hadamard: axis == RotationAxis::Rz,
        }
    }
}

impl LayerOps for AngleEncoder {
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
        // One rotation per feature, on the active qubit at that logical
        // position. If `x` is shorter than the active set, the surplus qubits
        // are left untouched (documented in `plan`'s validation).
        for (j, &qubit) in alloc.active.iter().take(x.len()).enumerate() {
            if self.prepend_hadamard {
                qc.try_push(GateInstruction::H(qubit))?;
            }
            let theta = Fixed(x[j]);
            let gate = match self.axis {
                RotationAxis::Rx => GateInstruction::Rx { qubit, theta },
                RotationAxis::Ry => GateInstruction::Ry { qubit, theta },
                RotationAxis::Rz => GateInstruction::Rz { qubit, theta },
            };
            qc.try_push(gate)?;
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

    #[test]
    fn new_prepends_hadamard_only_for_rz() {
        assert!(!AngleEncoder::new(RotationAxis::Rx).prepend_hadamard);
        assert!(!AngleEncoder::new(RotationAxis::Ry).prepend_hadamard);
        assert!(AngleEncoder::new(RotationAxis::Rz).prepend_hadamard);
    }

    #[test]
    fn plan_reserves_no_params_and_snapshots_active() {
        let enc = AngleEncoder::new(RotationAxis::Ry);
        let mut c = ctx(3, 3);
        let alloc = enc.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..0);
        assert_eq!(alloc.active, vec![0, 1, 2]);
        assert_eq!(c.param_cursor, 0);
    }

    #[test]
    fn ry_emits_exact_rotation_sequence_without_hadamard() {
        let enc = AngleEncoder::new(RotationAxis::Ry);
        let mut c = ctx(3, 3);
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        enc.emit(&mut qc, &alloc, &[0.1, 0.2, 0.3]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                GateInstruction::Ry {
                    qubit: 0,
                    theta: Fixed(0.1)
                },
                GateInstruction::Ry {
                    qubit: 1,
                    theta: Fixed(0.2)
                },
                GateInstruction::Ry {
                    qubit: 2,
                    theta: Fixed(0.3)
                },
            ]
        );
        assert_eq!(qc.num_params, 0);
    }

    #[test]
    fn rz_prepends_hadamard_before_each_rotation() {
        let enc = AngleEncoder::new(RotationAxis::Rz);
        let mut c = ctx(2, 2);
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        enc.emit(&mut qc, &alloc, &[0.5, 0.6]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                GateInstruction::H(0),
                GateInstruction::Rz {
                    qubit: 0,
                    theta: Fixed(0.5)
                },
                GateInstruction::H(1),
                GateInstruction::Rz {
                    qubit: 1,
                    theta: Fixed(0.6)
                },
            ]
        );
    }

    #[test]
    fn plan_rejects_more_features_than_active_qubits() {
        let enc = AngleEncoder::new(RotationAxis::Rx);
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
    fn fewer_features_than_qubits_leaves_surplus_untouched() {
        // num_features = 1 over 3 active qubits: only the first is rotated.
        let enc = AngleEncoder::new(RotationAxis::Ry);
        let mut c = ctx(3, 1);
        let alloc = enc.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        enc.emit(&mut qc, &alloc, &[0.7]).unwrap();
        assert_eq!(
            qc.gates,
            vec![GateInstruction::Ry {
                qubit: 0,
                theta: Fixed(0.7)
            }]
        );
    }
}
