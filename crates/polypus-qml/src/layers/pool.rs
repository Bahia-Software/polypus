//! [`PoolLayer`]: unitary pooling for a QCNN (design doc §6.5).
//!
//! Canonical QCNN pooling measures the discarded qubit mid-circuit and
//! conditions gates on the outcome, which would violate contract C-4. The
//! chosen alternative is **unitary pooling**: a controlled block flows the
//! discarded qubit's information into the retained one, after which the
//! discarded qubit simply *leaves the active set* and receives no further gate
//! (C-4-clean by construction). This is the first layer in the crate that
//! shrinks [`LayerContext::active`](crate::model::LayerContext).
//!
//! Pairs are always adjacent ([`even_pairs`](crate::layers::even_pairs)); the
//! [`KeepRule`] decides which position of each pair survives. Parameters are
//! shared across pairs, exactly as in [`conv`](crate::layers::conv).

use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::even_pairs;
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// The two-qubit block emitted on every pooled pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolBlock {
    /// Three shared rotations around a `cx` from the discarded to the retained
    /// qubit (`rz(θ0) desc · ry(θ1) ret · cx(desc→ret) · ry(θ2) ret`).
    Basic,
}

/// Which position of each adjacent pair survives pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepRule {
    /// Retain the **first** (lower) position of each pair; discard the second
    /// (the default).
    EvenPositions,
    /// Retain the **second** position of each pair; discard the first.
    OddPositions,
}

/// A pooling layer: adjacent pairs of active qubits are each reduced to one
/// retained qubit, the discarded qubit's information flowing into it through a
/// shared [`PoolBlock`]. Discarded qubits are removed from the active set. With
/// an odd number of active qubits the last one is left unpaired and stays
/// active untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolLayer {
    /// The two-qubit block emitted on every pooled pair.
    pub block: PoolBlock,
    /// Which position of each pair survives.
    pub keep: KeepRule,
}

impl PoolLayer {
    /// Build a pooling layer with the default [`KeepRule::EvenPositions`].
    pub fn new(block: PoolBlock) -> Self {
        PoolLayer {
            block,
            keep: KeepRule::EvenPositions,
        }
    }

    /// The number of shared `θ` this block reserves for the whole layer.
    fn param_count(&self) -> usize {
        match self.block {
            PoolBlock::Basic => 3,
        }
    }

    /// For a logical pair `(a, b)`, the `(discarded, retained)` logical
    /// positions under this layer's [`KeepRule`].
    fn discard_keep(&self, a: usize, b: usize) -> (usize, usize) {
        match self.keep {
            KeepRule::EvenPositions => (b, a),
            KeepRule::OddPositions => (a, b),
        }
    }
}

impl LayerOps for PoolLayer {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        let active = ctx.active.len();
        if active < 2 {
            return Err(ValidationError::PoolNeedsTwoQubits { active });
        }
        // Parameter sharing: reserve the block's θ once for the whole layer.
        let count = self.param_count();
        let params = ctx.param_cursor..ctx.param_cursor + count;
        ctx.param_cursor += count;

        // The allocation snapshots the active set *before* pooling — `emit`
        // needs it to map every logical position to its physical qubit.
        let snapshot = ctx.active.clone();

        // Every pair discards one logical position; remove those, preserving
        // the relative order of the survivors (and any odd leftover).
        let discarded: Vec<usize> = even_pairs(active)
            .into_iter()
            .map(|(a, b)| self.discard_keep(a, b).0)
            .collect();
        ctx.active = snapshot
            .iter()
            .enumerate()
            .filter(|(position, _)| !discarded.contains(position))
            .map(|(_, &qubit)| qubit)
            .collect();

        Ok(LayerAllocation {
            params,
            active: snapshot,
        })
    }

    fn emit(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        _x: &[f64],
    ) -> Result<(), QmlError> {
        let start = alloc.params.start;
        // Recompute the same pairs and keep rule `plan` used — a pure function
        // of the pre-pooling active length, so the two passes cannot diverge.
        for (a, b) in even_pairs(alloc.active.len()) {
            let (discarded, retained) = self.discard_keep(a, b);
            // Logical positions → physical qubit indices.
            let (desc, ret) = (alloc.active[discarded], alloc.active[retained]);
            match self.block {
                PoolBlock::Basic => emit_basic(qc, desc, ret, start)?,
            }
        }
        Ok(())
    }
}

/// `rz(θ0) desc · ry(θ1) ret · cx(desc→ret) · ry(θ2) ret`, reusing the shared
/// range `start..start+3` on every pair. After it the discarded qubit receives
/// no further gate.
fn emit_basic(
    qc: &mut ParameterizedCircuit,
    desc: usize,
    ret: usize,
    start: usize,
) -> Result<(), QmlError> {
    qc.try_push(GateInstruction::Rz {
        qubit: desc,
        theta: Param(start),
    })?;
    qc.try_push(GateInstruction::Ry {
        qubit: ret,
        theta: Param(start + 1),
    })?;
    qc.try_push(GateInstruction::Cx(desc, ret))?;
    qc.try_push(GateInstruction::Ry {
        qubit: ret,
        theta: Param(start + 2),
    })?;
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

    #[test]
    fn basic_emits_exact_sequence_on_one_pair() {
        let layer = PoolLayer::new(PoolBlock::Basic);
        let mut c = ctx(2);
        let alloc = layer.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        // EvenPositions: keep position 0 (physical 0), discard 1 (physical 1).
        assert_eq!(
            qc.gates,
            vec![
                GateInstruction::Rz {
                    qubit: 1,
                    theta: Param(0),
                },
                GateInstruction::Ry {
                    qubit: 0,
                    theta: Param(1),
                },
                GateInstruction::Cx(1, 0),
                GateInstruction::Ry {
                    qubit: 0,
                    theta: Param(2),
                },
            ]
        );
    }

    #[test]
    fn even_positions_keep_the_lower_qubit_of_each_pair() {
        let layer = PoolLayer::new(PoolBlock::Basic);
        let mut c = ctx(4);
        let alloc = layer.plan(&mut c).unwrap();
        // Pairs (0,1),(2,3): keep 0 and 2, discard 1 and 3.
        assert_eq!(c.active, vec![0, 2]);
        // The allocation kept the pre-pooling snapshot.
        assert_eq!(alloc.active, vec![0, 1, 2, 3]);
    }

    #[test]
    fn odd_positions_keep_the_higher_qubit_of_each_pair() {
        let layer = PoolLayer {
            block: PoolBlock::Basic,
            keep: KeepRule::OddPositions,
        };
        let mut c = ctx(4);
        layer.plan(&mut c).unwrap();
        // Pairs (0,1),(2,3): keep 1 and 3, discard 0 and 2.
        assert_eq!(c.active, vec![1, 3]);
    }

    #[test]
    fn odd_active_leaves_the_leftover_untouched_and_active() {
        let layer = PoolLayer::new(PoolBlock::Basic);
        let mut c = ctx(3);
        let alloc = layer.plan(&mut c).unwrap();
        // Pair (0,1) keeps 0; position 2 is unpaired and stays active.
        assert_eq!(c.active, vec![0, 2]);

        let mut qc = ParameterizedCircuit::new(3);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        // Only one block (4 gates); qubit 2 receives no gate.
        assert_eq!(qc.gates.len(), 4);
        assert!(qc.gates.iter().all(|g| !mentions(g, 2)));
    }

    /// Whether a gate touches `qubit` (enough for the leftover-untouched check).
    fn mentions(gate: &GateInstruction, qubit: usize) -> bool {
        match gate {
            GateInstruction::Rz { qubit: q, .. } | GateInstruction::Ry { qubit: q, .. } => {
                *q == qubit
            }
            GateInstruction::Cx(a, b) => *a == qubit || *b == qubit,
            _ => false,
        }
    }

    #[test]
    fn basic_shares_parameters_across_pairs() {
        // 4 qubits → 2 pairs, but still only 3 shared θ.
        let layer = PoolLayer::new(PoolBlock::Basic);
        let mut c = ctx(4);
        let alloc = layer.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..3);
        assert_eq!(c.param_cursor, 3);

        let mut qc = ParameterizedCircuit::new(4);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        // 2 pairs × 4 gates.
        assert_eq!(qc.gates.len(), 8);
        assert_eq!(qc.num_params, 3);
    }

    #[test]
    fn plan_rejects_fewer_than_two_active_qubits() {
        let layer = PoolLayer::new(PoolBlock::Basic);
        let mut c = ctx(1);
        assert!(matches!(
            layer.plan(&mut c),
            Err(ValidationError::PoolNeedsTwoQubits { active: 1 })
        ));
    }
}
