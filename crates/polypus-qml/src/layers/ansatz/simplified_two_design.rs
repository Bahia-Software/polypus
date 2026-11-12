//! [`SimplifiedTwoDesignAnsatz`]: PennyLane's `SimplifiedTwoDesign` — an
//! initial `Ry` layer, then blocks of `CZ` + `Ry` on the even pairs and then
//! the odd pairs of active qubits (Cerezo et al., "Cost function dependent
//! barren plateaus in shallow parametrized quantum circuits").

use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::{even_pairs, odd_pairs};
use crate::model::{InputTrace, LayerAllocation, LayerContext, LayerOps};

/// A simplified 2-design ansatz: one `Ry` on every active qubit, then `reps`
/// blocks. A block visits the even pairs `(0,1),(2,3),…` and then the odd
/// pairs `(1,2),(3,4),…` of logical positions and, pair by pair, applies
/// `CZ(a, b)` followed by `Ry` on `a` and `Ry` on `b`.
///
/// Every `Ry` is bound to its own fresh `θ` index, so the exact
/// parameter-shift gradient applies. The `n` initial weights come first
/// (position `q` owns `start + q`); then block `r`, pair `k` of the even-then-odd
/// list owns `start + n + 2·(n−1)·r + 2k` for its first qubit and the next index
/// for its second — the C-order flattening of PennyLane's `(n,)` initial
/// weights followed by its `(reps, n−1, 2)` block weights. That is
/// `n + 2·(n−1)·reps` parameters for `n` active qubits. There is no final
/// rotation layer.
///
/// With fewer than two active qubits there are no pairs, so only the initial
/// layer remains, whatever `reps`. PennyLane's one-wire shape still lists
/// `(reps,)` unused block weights; none are reserved here.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimplifiedTwoDesignAnsatz {
    /// Number of `CZ` + `Ry` blocks after the initial `Ry` layer.
    pub reps: usize,
}

impl SimplifiedTwoDesignAnsatz {
    /// A simplified 2-design with `reps` blocks.
    pub fn new(reps: usize) -> Self {
        SimplifiedTwoDesignAnsatz { reps }
    }
}

impl LayerOps for SimplifiedTwoDesignAnsatz {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        let n = ctx.active.len();
        // n initial weights, then two per pair (n − 1 pairs) per block.
        let count = n
            .saturating_sub(1)
            .checked_mul(2)
            .and_then(|per_block| per_block.checked_mul(self.reps))
            .and_then(|blocks| blocks.checked_add(n))
            .ok_or(ValidationError::TooManyParams {
                layer: ctx.layers_planned,
            })?;
        let params = ctx.reserve(count)?;
        Ok(LayerAllocation {
            params,
            active: ctx.active.clone(),
        })
    }

    fn emit_traced(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        _x: &[f64],
        _trace: &mut InputTrace,
    ) -> Result<(), QmlError> {
        let active = &alloc.active;
        let n = active.len();
        let mut next_param = alloc.params.start;
        for &qubit in active {
            qc.try_push(GateInstruction::Ry {
                qubit,
                theta: Param(next_param),
            })?;
            next_param += 1;
        }
        // No pairs: every block would be empty. Return now rather than loop
        // `reps` times for nothing (`plan` accepts any `reps` here, since the
        // blocks reserve no parameters).
        if n < 2 {
            return Ok(());
        }
        let mut pairs = even_pairs(n);
        pairs.extend(odd_pairs(n));
        for _ in 0..self.reps {
            for &(a, b) in &pairs {
                let (a, b) = (active[a], active[b]);
                qc.try_push(GateInstruction::Cz(a, b))?;
                qc.try_push(GateInstruction::Ry {
                    qubit: a,
                    theta: Param(next_param),
                })?;
                qc.try_push(GateInstruction::Ry {
                    qubit: b,
                    theta: Param(next_param + 1),
                })?;
                next_param += 2;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(num_qubits: usize) -> LayerContext {
        LayerContext {
            active: (0..num_qubits).collect(),
            num_features: 0,
            param_cursor: 0,
            layers_planned: 0,
        }
    }

    fn ry(qubit: usize, param: usize) -> GateInstruction {
        GateInstruction::Ry {
            qubit,
            theta: Param(param),
        }
    }

    /// `CZ(a, b)`, `Ry(a)` bound to `first`, `Ry(b)` bound to `first + 1`.
    fn pair(a: usize, b: usize, first: usize) -> Vec<GateInstruction> {
        vec![GateInstruction::Cz(a, b), ry(a, first), ry(b, first + 1)]
    }

    /// The gates of `ansatz` over `num_qubits` contiguous qubits.
    fn gates(ansatz: &SimplifiedTwoDesignAnsatz, num_qubits: usize) -> Vec<GateInstruction> {
        let mut c = ctx(num_qubits);
        let alloc = ansatz.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(num_qubits);
        ansatz.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(qc.num_params(), alloc.params.len());
        qc.gates().to_vec()
    }

    #[test]
    fn four_qubits_visit_the_even_pairs_then_the_odd_pair() {
        let mut expected = vec![ry(0, 0), ry(1, 1), ry(2, 2), ry(3, 3)];
        expected.extend(pair(0, 1, 4));
        expected.extend(pair(2, 3, 6));
        expected.extend(pair(1, 2, 8));
        assert_eq!(gates(&SimplifiedTwoDesignAnsatz::new(1), 4), expected);
    }

    #[test]
    fn three_qubits_have_one_even_and_one_odd_pair() {
        let mut expected = vec![ry(0, 0), ry(1, 1), ry(2, 2)];
        expected.extend(pair(0, 1, 3));
        expected.extend(pair(1, 2, 5));
        assert_eq!(gates(&SimplifiedTwoDesignAnsatz::new(1), 3), expected);
    }

    #[test]
    fn five_qubits_over_two_blocks_number_the_blocks_in_turn() {
        // 4 pairs per block, 8 parameters each, after the 5 initial weights.
        let mut expected: Vec<_> = (0..5).map(|q| ry(q, q)).collect();
        for first in [5, 13] {
            expected.extend(pair(0, 1, first));
            expected.extend(pair(2, 3, first + 2));
            expected.extend(pair(1, 2, first + 4));
            expected.extend(pair(3, 4, first + 6));
        }
        assert_eq!(gates(&SimplifiedTwoDesignAnsatz::new(2), 5), expected);
    }

    #[test]
    fn two_qubits_repeat_their_only_pair() {
        let mut expected = vec![ry(0, 0), ry(1, 1)];
        expected.extend(pair(0, 1, 2));
        expected.extend(pair(0, 1, 4));
        assert_eq!(gates(&SimplifiedTwoDesignAnsatz::new(2), 2), expected);
    }

    #[test]
    fn zero_reps_is_the_initial_layer_only() {
        assert_eq!(
            gates(&SimplifiedTwoDesignAnsatz::new(0), 3),
            vec![ry(0, 0), ry(1, 1), ry(2, 2)]
        );
    }

    #[test]
    fn a_single_qubit_gets_the_initial_rotation_whatever_the_reps() {
        // No pairs, so no block emits anything.
        for reps in [0, 3] {
            assert_eq!(
                gates(&SimplifiedTwoDesignAnsatz::new(reps), 1),
                vec![ry(0, 0)]
            );
        }
        // A huge `reps` must neither overflow the count nor loop. Emitted on
        // another thread so that iterating the blocks fails here with a
        // message instead of hanging the test binary (the stuck thread is
        // abandoned when the binary exits).
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        let emitter = std::thread::spawn(move || {
            let _ = tx.send(gates(&SimplifiedTwoDesignAnsatz::new(usize::MAX), 1));
        });
        let emitted = match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(emitted) => emitted,
            Err(RecvTimeoutError::Timeout) => {
                panic!("a one-qubit layer with reps = usize::MAX must not iterate the blocks")
            }
            // The emitting thread panicked before sending: surface its own panic.
            Err(RecvTimeoutError::Disconnected) => match emitter.join() {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(()) => unreachable!("the emitter drops its sender only after sending"),
            },
        };
        assert_eq!(emitted, vec![ry(0, 0)]);
    }

    #[test]
    fn the_layer_addresses_logical_positions() {
        // After pooling the active set is a subset of the physical qubits; the
        // pairs run over logical positions and map them through `active`.
        let ansatz = SimplifiedTwoDesignAnsatz::new(1);
        let mut c = LayerContext {
            active: vec![0, 2, 4],
            param_cursor: 3,
            ..ctx(0)
        };
        let alloc = ansatz.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 3..10);
        let mut qc = ParameterizedCircuit::new(5);
        ansatz.emit(&mut qc, &alloc, &[]).unwrap();
        let mut expected = vec![ry(0, 3), ry(2, 4), ry(4, 5)];
        expected.extend(pair(0, 2, 6));
        expected.extend(pair(2, 4, 8));
        assert_eq!(qc.gates(), expected);
    }

    #[test]
    fn param_count_is_n_plus_two_per_pair_per_block() {
        // (num_qubits, reps, expected)
        let cases = [
            (3, 2, 11),
            (4, 2, 16),
            (5, 1, 13),
            (2, 1, 4),
            (4, 1, 10),
            (3, 0, 3),
            (1, 3, 1),
            (1, usize::MAX, 1),
        ];
        for (nq, reps, expected) in cases {
            // Start past an earlier layer's parameters: the range is offset.
            let mut c = LayerContext {
                param_cursor: 5,
                ..ctx(nq)
            };
            let alloc = SimplifiedTwoDesignAnsatz::new(reps).plan(&mut c).unwrap();
            assert_eq!(alloc.params, 5..5 + expected, "n = {nq}, reps = {reps}");
            assert_eq!(c.param_cursor, 5 + expected);
        }
    }

    #[test]
    fn plan_rejects_a_param_count_that_does_not_fit() {
        // 2 × 1 pair × reps overflows on 2 qubits…
        let mut c = LayerContext {
            param_cursor: 7,
            layers_planned: 3,
            ..ctx(2)
        };
        assert_eq!(
            SimplifiedTwoDesignAnsatz::new(usize::MAX)
                .plan(&mut c)
                .unwrap_err(),
            ValidationError::TooManyParams { layer: 3 }
        );
        assert_eq!(c.param_cursor, 7);
        // …and so does adding the n initial weights to a block total that
        // just fits: 2 × (2^63 − 1) + 2 = 2^64.
        let mut c = ctx(2);
        assert_eq!(
            SimplifiedTwoDesignAnsatz::new(usize::MAX / 2)
                .plan(&mut c)
                .unwrap_err(),
            ValidationError::TooManyParams { layer: 0 }
        );
        assert_eq!(c.param_cursor, 0);
        // A count that fits on its own but not after the earlier layers'.
        let mut c = LayerContext {
            param_cursor: usize::MAX - 3,
            layers_planned: 1,
            ..ctx(2)
        };
        assert_eq!(
            SimplifiedTwoDesignAnsatz::new(1).plan(&mut c).unwrap_err(),
            ValidationError::TooManyParams { layer: 1 }
        );
        assert_eq!(c.param_cursor, usize::MAX - 3);
    }

    #[test]
    fn new_sets_the_reps() {
        assert_eq!(SimplifiedTwoDesignAnsatz::new(3).reps, 3);
    }
}
