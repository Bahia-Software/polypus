//! [`HardwareEfficientAnsatz`]: alternating blocks of fresh single-qubit
//! rotations and a fixed entangling pattern — the `TwoLocal` family.

use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::RotationAxis;
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// The two-qubit entangling gate used between rotation blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entangler {
    /// Controlled-NOT (`cx`).
    Cx,
    /// Controlled-Z (`cz`).
    Cz,
}

/// The connectivity pattern of the entangling block, over logical positions of
/// the active qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entanglement {
    /// Nearest-neighbour chain: `(0,1),(1,2),…,(n-2,n-1)`.
    Linear,
    /// `Linear` plus the wrap-around pair `(n-1,0)`.
    Circular,
    /// Every ordered pair `(i,j)` with `i < j`.
    Full,
}

/// A hardware-efficient ansatz: `reps` repetitions of `(rotation block,
/// entangling block)`, optionally followed by one final rotation block.
///
/// Each rotation block emits, for every axis in [`rotations`](Self::rotations)
/// (list order) and then every active qubit (logical-position order), one fresh
/// rotation bound to a new `θ` index — axis-major, qubit-minor, like Qiskit's
/// `TwoLocal`. There is no parameter sharing here.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareEfficientAnsatz {
    /// Number of `(rotation, entangling)` repetitions.
    pub reps: usize,
    /// Rotation axes emitted per rotation block, in order.
    pub rotations: Vec<RotationAxis>,
    /// The entangling gate placed between rotation blocks.
    pub entangler: Entangler,
    /// The connectivity of the entangling block.
    pub entanglement: Entanglement,
    /// Whether to append a final rotation block after the last entangling
    /// block (no entangling follows it). `TwoLocal` default: `true`.
    pub final_rotation_layer: bool,
}

impl HardwareEfficientAnsatz {
    /// The `TwoLocal` default: `[Ry, Rz]` rotations, linear `Cx` entanglement,
    /// and a final rotation layer.
    pub fn new(reps: usize) -> Self {
        HardwareEfficientAnsatz {
            reps,
            rotations: vec![RotationAxis::Ry, RotationAxis::Rz],
            entangler: Entangler::Cx,
            entanglement: Entanglement::Linear,
            final_rotation_layer: true,
        }
    }

    /// The `RealAmplitudes` preset: a single `Ry` rotation per block, linear
    /// `Cx` entanglement, final rotation layer (the recommended default).
    pub fn real_amplitudes(reps: usize) -> Self {
        HardwareEfficientAnsatz {
            rotations: vec![RotationAxis::Ry],
            ..Self::new(reps)
        }
    }
}

impl LayerOps for HardwareEfficientAnsatz {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        // One fresh θ per (rotation-block, axis, active qubit). Blocks =
        // `reps` + one more if a final rotation layer is requested. The
        // num_params == 0 case is caught model-wide by NoTrainableParams, so
        // there is nothing to validate here.
        let blocks = self.reps + self.final_rotation_layer as usize;
        let count = ctx.active.len() * self.rotations.len() * blocks;
        let params = ctx.param_cursor..ctx.param_cursor + count;
        ctx.param_cursor += count;
        Ok(LayerAllocation {
            params,
            active: ctx.active.clone(),
        })
    }

    fn emit(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        _x: &[f64],
    ) -> Result<(), QmlError> {
        // Fresh θ indices are handed out from the start of this layer's range.
        let mut next_param = alloc.params.start;
        for _ in 0..self.reps {
            emit_rotation_block(qc, &self.rotations, &alloc.active, &mut next_param)?;
            emit_entangling_block(qc, &alloc.active, self.entangler, self.entanglement)?;
        }
        if self.final_rotation_layer {
            emit_rotation_block(qc, &self.rotations, &alloc.active, &mut next_param)?;
        }
        Ok(())
    }
}

/// Emit one rotation block: axis-major, qubit-minor, one fresh `Param` each.
fn emit_rotation_block(
    qc: &mut ParameterizedCircuit,
    rotations: &[RotationAxis],
    active: &[usize],
    next_param: &mut usize,
) -> Result<(), QmlError> {
    for axis in rotations {
        for &qubit in active {
            let theta = Param(*next_param);
            let gate = match axis {
                RotationAxis::Rx => GateInstruction::Rx { qubit, theta },
                RotationAxis::Ry => GateInstruction::Ry { qubit, theta },
                RotationAxis::Rz => GateInstruction::Rz { qubit, theta },
            };
            qc.try_push(gate)?;
            *next_param += 1;
        }
    }
    Ok(())
}

/// Emit one entangling block over the logical positions of `active`. A no-op
/// when fewer than two qubits are active.
fn emit_entangling_block(
    qc: &mut ParameterizedCircuit,
    active: &[usize],
    entangler: Entangler,
    entanglement: Entanglement,
) -> Result<(), QmlError> {
    if active.len() < 2 {
        return Ok(());
    }
    let n = active.len();
    let pairs: Vec<(usize, usize)> = match entanglement {
        Entanglement::Linear => (0..n - 1).map(|i| (i, i + 1)).collect(),
        Entanglement::Circular => {
            let mut pairs: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            pairs.push((n - 1, 0));
            pairs
        }
        Entanglement::Full => {
            let mut pairs = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    pairs.push((i, j));
                }
            }
            pairs
        }
    };
    for (a, b) in pairs {
        // First position is control, second is target (both logical → physical
        // via `active`).
        let (control, target) = (active[a], active[b]);
        let gate = match entangler {
            Entangler::Cx => GateInstruction::Cx(control, target),
            Entangler::Cz => GateInstruction::Cz(control, target),
        };
        qc.try_push(gate)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(num_qubits: usize) -> LayerContext {
        LayerContext {
            active: (0..num_qubits).collect(),
            num_features: 0,
            param_cursor: 0,
        }
    }

    fn ry(qubit: usize, p: usize) -> GateInstruction {
        GateInstruction::Ry {
            qubit,
            theta: Param(p),
        }
    }

    /// A single-Ry, linear-Cx, no-final-layer ansatz on 3 qubits: exact order.
    fn linear_ansatz() -> HardwareEfficientAnsatz {
        HardwareEfficientAnsatz {
            reps: 1,
            rotations: vec![RotationAxis::Ry],
            entangler: Entangler::Cx,
            entanglement: Entanglement::Linear,
            final_rotation_layer: false,
        }
    }

    #[test]
    fn linear_emits_rotations_then_chain() {
        let a = linear_ansatz();
        let mut c = ctx(3);
        let alloc = a.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        a.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                ry(0, 0),
                ry(1, 1),
                ry(2, 2),
                GateInstruction::Cx(0, 1),
                GateInstruction::Cx(1, 2),
            ]
        );
        assert_eq!(qc.num_params, 3);
    }

    #[test]
    fn circular_adds_wraparound_pair() {
        let a = HardwareEfficientAnsatz {
            entanglement: Entanglement::Circular,
            ..linear_ansatz()
        };
        let mut c = ctx(3);
        let alloc = a.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        a.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                ry(0, 0),
                ry(1, 1),
                ry(2, 2),
                GateInstruction::Cx(0, 1),
                GateInstruction::Cx(1, 2),
                GateInstruction::Cx(2, 0),
            ]
        );
    }

    #[test]
    fn full_uses_every_ordered_pair() {
        let a = HardwareEfficientAnsatz {
            entanglement: Entanglement::Full,
            ..linear_ansatz()
        };
        let mut c = ctx(3);
        let alloc = a.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        a.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates,
            vec![
                ry(0, 0),
                ry(1, 1),
                ry(2, 2),
                GateInstruction::Cx(0, 1),
                GateInstruction::Cx(0, 2),
                GateInstruction::Cx(1, 2),
            ]
        );
    }

    #[test]
    fn entangling_block_is_a_noop_on_a_single_qubit() {
        let a = HardwareEfficientAnsatz {
            reps: 1,
            rotations: vec![RotationAxis::Ry, RotationAxis::Rz],
            entangler: Entangler::Cx,
            entanglement: Entanglement::Linear,
            final_rotation_layer: true,
        };
        let mut c = ctx(1);
        let alloc = a.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(1);
        a.emit(&mut qc, &alloc, &[]).unwrap();
        // 1 qubit × 2 axes × (1 rep + final) = 4 rotations, no entangling gate.
        assert_eq!(
            qc.gates,
            vec![
                GateInstruction::Ry {
                    qubit: 0,
                    theta: Param(0)
                },
                GateInstruction::Rz {
                    qubit: 0,
                    theta: Param(1)
                },
                GateInstruction::Ry {
                    qubit: 0,
                    theta: Param(2)
                },
                GateInstruction::Rz {
                    qubit: 0,
                    theta: Param(3)
                },
            ]
        );
    }

    #[test]
    fn param_count_formula() {
        // count = active × rotations × (reps + final_rotation_layer)
        let cases = [
            // (num_qubits, rotations, reps, final, expected)
            (3, 1usize, 1usize, false, 3),
            (3, 1, 1, true, 6),
            (4, 2, 2, true, 24),
            (2, 2, 3, false, 12),
        ];
        for (nq, nrot, reps, final_layer, expected) in cases {
            let a = HardwareEfficientAnsatz {
                reps,
                rotations: vec![RotationAxis::Ry; nrot],
                entangler: Entangler::Cx,
                entanglement: Entanglement::Linear,
                final_rotation_layer: final_layer,
            };
            let mut c = ctx(nq);
            let alloc = a.plan(&mut c).unwrap();
            assert_eq!(alloc.params.end - alloc.params.start, expected);
            assert_eq!(c.param_cursor, expected);
        }
    }

    #[test]
    fn presets_have_expected_defaults() {
        let new = HardwareEfficientAnsatz::new(2);
        assert_eq!(new.reps, 2);
        assert_eq!(new.rotations, vec![RotationAxis::Ry, RotationAxis::Rz]);
        assert_eq!(new.entangler, Entangler::Cx);
        assert_eq!(new.entanglement, Entanglement::Linear);
        assert!(new.final_rotation_layer);

        let ra = HardwareEfficientAnsatz::real_amplitudes(3);
        assert_eq!(ra.reps, 3);
        assert_eq!(ra.rotations, vec![RotationAxis::Ry]);
        assert_eq!(ra.entangler, Entangler::Cx);
        assert_eq!(ra.entanglement, Entanglement::Linear);
        assert!(ra.final_rotation_layer);
    }
}
