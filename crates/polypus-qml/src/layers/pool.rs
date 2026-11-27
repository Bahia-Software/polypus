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
use crate::layers::{even_pair_count, even_pairs};
use crate::model::{InputTrace, LayerAllocation, LayerContext, LayerOps};

/// The two-qubit block emitted on every pooled pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoolBlock {
    /// Three shared rotations around a `cx` from the discarded to the retained
    /// qubit (`rz(θ0) desc · ry(θ1) ret · cx(desc→ret) · ry(θ2) ret`).
    Basic,
}

/// Which position of each adjacent pair survives pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        // Parameter sharing: reserve the block's θ once for the whole layer,
        // while every pair gets the block's four gates ([`emit_basic`]).
        let gates = even_pair_count(active).checked_mul(4);
        let params = ctx.reserve(gates, Some(self.param_count()))?;

        // The allocation snapshots the active set *before* pooling — `emit`
        // needs it to map every logical position to its physical qubit.
        let snapshot = ctx.active.clone();

        // Every pair discards one logical position; remove those, preserving
        // the relative order of the survivors (and any odd leftover). The pairs
        // are those of `even_pairs`, so position `p` belongs to the pair that
        // starts at `p - p % 2`, and an odd count's last position to none: one
        // pass, linear in the active qubits.
        ctx.active = snapshot
            .iter()
            .enumerate()
            .filter(|&(position, _)| {
                let first = position - position % 2;
                first + 1 >= active || self.discard_keep(first, first + 1).1 == position
            })
            .map(|(_, &qubit)| qubit)
            .collect();

        Ok(LayerAllocation {
            params,
            active: snapshot,
        })
    }

    fn emit_traced(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        _x: &[f64],
        _trace: &mut InputTrace,
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
    use crate::error::SizeLimit;

    fn ctx(num_qubits: usize) -> LayerContext {
        LayerContext {
            active: (0..num_qubits).collect(),
            num_features: 0,
            param_cursor: 0,
            layers_planned: 0,
            gate_count: 0,
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
            qc.gates(),
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
        assert_eq!(qc.gates().len(), 4);
        assert!(qc.gates().iter().all(|g| !mentions(g, 2)));
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
        assert_eq!(qc.gates().len(), 8);
        assert_eq!(qc.num_params(), 3);
    }

    #[test]
    fn plan_rejects_counts_that_do_not_fit_after_the_earlier_ones() {
        let overflow = ValidationError::ModelTooLarge {
            layer: 4,
            reason: SizeLimit::Overflow,
        };
        // The 8 gates of two pairs after the earlier ones'.
        let mut c = LayerContext {
            gate_count: usize::MAX - 7,
            layers_planned: 4,
            ..ctx(4)
        };
        assert_eq!(
            PoolLayer::new(PoolBlock::Basic).plan(&mut c).unwrap_err(),
            overflow
        );
        // Neither the counts nor the active qubits change.
        assert_eq!((c.param_cursor, c.gate_count), (0, usize::MAX - 7));
        assert_eq!(c.active, vec![0, 1, 2, 3]);
        // The shared θ after the earlier ones' (a state `compile` cannot
        // reach, see `LayerContext::reserve`): refused the same way.
        let mut c = LayerContext {
            param_cursor: usize::MAX - 1,
            layers_planned: 4,
            ..ctx(4)
        };
        assert_eq!(
            PoolLayer::new(PoolBlock::Basic).plan(&mut c).unwrap_err(),
            overflow
        );
        assert_eq!((c.param_cursor, c.gate_count), (usize::MAX - 1, 0));
        assert_eq!(c.active, vec![0, 1, 2, 3]);
        // Within the budget, the pool counts its gates.
        let mut c = ctx(4);
        PoolLayer::new(PoolBlock::Basic).plan(&mut c).unwrap();
        assert_eq!(c.gate_count, 8);
    }

    /// The active set pooling leaves, computed the slow, obvious way: list the
    /// discarded positions, then drop every position found in that list.
    fn reference_survivors(layer: &PoolLayer, active: &[usize]) -> Vec<usize> {
        let discarded: Vec<usize> = even_pairs(active.len())
            .into_iter()
            .map(|(a, b)| layer.discard_keep(a, b).0)
            .collect();
        active
            .iter()
            .enumerate()
            .filter(|(position, _)| !discarded.contains(position))
            .map(|(_, &qubit)| qubit)
            .collect()
    }

    #[test]
    fn plan_leaves_the_same_active_set_as_the_reference() {
        for keep in [KeepRule::EvenPositions, KeepRule::OddPositions] {
            let layer = PoolLayer {
                block: PoolBlock::Basic,
                keep,
            };
            for n in 2..=9 {
                // Non-contiguous physical qubits, as after an earlier pooling.
                let active: Vec<usize> = (0..n).map(|q| 3 * q + 1).collect();
                let mut c = LayerContext {
                    active: active.clone(),
                    ..ctx(0)
                };
                let alloc = layer.plan(&mut c).unwrap();
                assert_eq!(alloc.active, active, "{keep:?}, n = {n}");
                assert_eq!(
                    c.active,
                    reference_survivors(&layer, &active),
                    "{keep:?}, n = {n}"
                );
            }
        }
    }

    #[test]
    fn plan_is_linear_in_the_active_qubits() {
        // A quadratic plan would make ~2.5·10^11 comparisons here; a linear one
        // takes milliseconds.
        let n = 1_000_000;
        for (keep, first_kept) in [(KeepRule::EvenPositions, 0), (KeepRule::OddPositions, 1)] {
            let layer = PoolLayer {
                block: PoolBlock::Basic,
                keep,
            };
            let mut c = ctx(n);
            layer.plan(&mut c).unwrap();
            assert_eq!(c.active.len(), n / 2);
            assert!(c
                .active
                .iter()
                .enumerate()
                .all(|(i, &q)| q == 2 * i + first_kept));
        }
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
