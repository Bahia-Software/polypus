//! [`StronglyEntanglingAnsatz`]: PennyLane's `StronglyEntanglingLayers` — a
//! general single-qubit rotation on every qubit, then a ring of entanglers
//! whose reach (the *range*) changes from layer to layer.

use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::Entangler;
use crate::model::{InputTrace, LayerAllocation, LayerContext, LayerOps};

/// A strongly entangling ansatz: `reps` layers, each a general rotation
/// `Rot(φ, θ, ω)` on every active qubit followed by an entangler from every
/// position `i` to position `(i + r) mod n`, where `r` is the layer's range and
/// `n` the number of active qubits.
///
/// `Rot(φ, θ, ω) = Rz(ω)·Ry(θ)·Rz(φ)` is emitted as the three Pauli rotations
/// `Rz(φ)`, `Ry(θ)`, `Rz(ω)`, in that order of application, each bound to its
/// own fresh `θ` index: the exact parameter-shift gradient needs every
/// parameter to enter once, as the angle of a Pauli rotation. Layer `l`,
/// logical position `q` owns the indices `start + 3·n·l + 3·q + {0, 1, 2}` for
/// `(φ, θ, ω)` — the C-order flattening of PennyLane's `(reps, n, 3)` weights.
/// There is no final rotation block, unlike [`HardwareEfficientAnsatz`].
///
/// With fewer than two active qubits no entangler is emitted (there is no
/// valid range), so only the rotations remain.
///
/// [`HardwareEfficientAnsatz`]: crate::HardwareEfficientAnsatz
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StronglyEntanglingAnsatz {
    /// Number of `(rotation, entangling)` layers.
    pub reps: usize,
    /// The range of each layer's entanglers, one per layer, each in
    /// `1..=n-1` for `n` active qubits. `None` is PennyLane's default,
    /// `r_l = (l mod (n-1)) + 1`.
    pub ranges: Option<Vec<usize>>,
    /// The two-qubit entangling gate.
    pub entangler: Entangler,
}

impl StronglyEntanglingAnsatz {
    /// PennyLane's default: the per-layer default ranges and `Cx` entanglers.
    pub fn new(reps: usize) -> Self {
        StronglyEntanglingAnsatz {
            reps,
            ranges: None,
            entangler: Entangler::Cx,
        }
    }

    /// The range of layer `layer` over `n ≥ 2` active qubits: the explicit one
    /// if given, else the default `(layer mod (n-1)) + 1`. Callers have
    /// validated the explicit ranges in `plan`.
    fn range(&self, layer: usize, n: usize) -> usize {
        match &self.ranges {
            Some(ranges) => ranges[layer],
            None => layer % (n - 1) + 1,
        }
    }
}

impl LayerOps for StronglyEntanglingAnsatz {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        let n = ctx.active.len();
        if let Some(ranges) = &self.ranges {
            if ranges.len() != self.reps {
                return Err(ValidationError::StronglyEntanglingRangesLength {
                    reps: self.reps,
                    got: ranges.len(),
                });
            }
            // With a single active qubit no entangler is emitted, so there is
            // no valid range to check the values against.
            if n >= 2 {
                if let Some((layer, &range)) =
                    ranges.iter().enumerate().find(|(_, &r)| r == 0 || r >= n)
                {
                    return Err(ValidationError::StronglyEntanglingRangeOutOfBounds {
                        layer,
                        range,
                        active: n,
                    });
                }
            }
        }
        // Three fresh θ per (layer, active qubit). The num_params == 0 case is
        // caught model-wide by NoTrainableParams.
        let count = n
            .checked_mul(3)
            .and_then(|per_layer| per_layer.checked_mul(self.reps))
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
        for layer in 0..self.reps {
            for &qubit in active {
                let (phi, theta, omega) = (
                    Param(next_param),
                    Param(next_param + 1),
                    Param(next_param + 2),
                );
                qc.try_push(GateInstruction::Rz { qubit, theta: phi })?;
                qc.try_push(GateInstruction::Ry { qubit, theta })?;
                qc.try_push(GateInstruction::Rz {
                    qubit,
                    theta: omega,
                })?;
                next_param += 3;
            }
            if n < 2 {
                continue;
            }
            let range = self.range(layer, n);
            for i in 0..n {
                let (control, target) = (active[i], active[(i + range) % n]);
                let gate = match self.entangler {
                    Entangler::Cx => GateInstruction::Cx(control, target),
                    Entangler::Cz => GateInstruction::Cz(control, target),
                };
                qc.try_push(gate)?;
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

    /// `Rz(φ)`, `Ry(θ)`, `Rz(ω)` on `qubit`, bound to `first`, `first + 1` and
    /// `first + 2`: one `Rot`.
    fn rot(qubit: usize, first: usize) -> Vec<GateInstruction> {
        vec![
            GateInstruction::Rz {
                qubit,
                theta: Param(first),
            },
            GateInstruction::Ry {
                qubit,
                theta: Param(first + 1),
            },
            GateInstruction::Rz {
                qubit,
                theta: Param(first + 2),
            },
        ]
    }

    /// One layer's rotations over `0..n`, starting at `θ` index `first`.
    fn rotations(n: usize, first: usize) -> Vec<GateInstruction> {
        (0..n).flat_map(|q| rot(q, first + 3 * q)).collect()
    }

    fn cx_ring(pairs: &[(usize, usize)]) -> Vec<GateInstruction> {
        pairs
            .iter()
            .map(|&(c, t)| GateInstruction::Cx(c, t))
            .collect()
    }

    fn gates(ansatz: &StronglyEntanglingAnsatz, num_qubits: usize) -> Vec<GateInstruction> {
        let mut c = ctx(num_qubits);
        let alloc = ansatz.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(num_qubits);
        ansatz.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(qc.num_params(), 3 * num_qubits * ansatz.reps);
        qc.gates().to_vec()
    }

    fn with_ranges(reps: usize, ranges: Vec<usize>) -> StronglyEntanglingAnsatz {
        StronglyEntanglingAnsatz {
            ranges: Some(ranges),
            ..StronglyEntanglingAnsatz::new(reps)
        }
    }

    #[test]
    fn default_ranges_on_three_qubits_are_one_then_two() {
        let mut expected = rotations(3, 0);
        expected.extend(cx_ring(&[(0, 1), (1, 2), (2, 0)]));
        expected.extend(rotations(3, 9));
        expected.extend(cx_ring(&[(0, 2), (1, 0), (2, 1)]));
        assert_eq!(gates(&StronglyEntanglingAnsatz::new(2), 3), expected);
    }

    #[test]
    fn default_ranges_on_four_qubits_cycle_one_two_three_one() {
        let mut expected = Vec::new();
        for (layer, range) in [1, 2, 3, 1].into_iter().enumerate() {
            expected.extend(rotations(4, 12 * layer));
            let ring: Vec<_> = (0..4).map(|i| (i, (i + range) % 4)).collect();
            expected.extend(cx_ring(&ring));
        }
        assert_eq!(gates(&StronglyEntanglingAnsatz::new(4), 4), expected);
    }

    #[test]
    fn two_qubits_entangle_both_ways() {
        let mut expected = rotations(2, 0);
        expected.extend(cx_ring(&[(0, 1), (1, 0)]));
        assert_eq!(gates(&StronglyEntanglingAnsatz::new(1), 2), expected);
    }

    #[test]
    fn a_single_qubit_gets_only_rotations() {
        // n == 1: no valid range, so no entangler, and explicit range values
        // are not checked (only their count).
        let mut expected = rotations(1, 0);
        expected.extend(rotations(1, 3));
        assert_eq!(gates(&StronglyEntanglingAnsatz::new(2), 1), expected);
        assert_eq!(gates(&with_ranges(2, vec![0, 7]), 1), expected);
    }

    #[test]
    fn cz_replaces_every_entangler() {
        let ansatz = StronglyEntanglingAnsatz {
            entangler: Entangler::Cz,
            ..StronglyEntanglingAnsatz::new(1)
        };
        let mut expected = rotations(3, 0);
        expected.extend([
            GateInstruction::Cz(0, 1),
            GateInstruction::Cz(1, 2),
            GateInstruction::Cz(2, 0),
        ]);
        assert_eq!(gates(&ansatz, 3), expected);
    }

    #[test]
    fn explicit_ranges_override_the_default() {
        let mut expected = rotations(3, 0);
        expected.extend(cx_ring(&[(0, 2), (1, 0), (2, 1)]));
        expected.extend(rotations(3, 9));
        expected.extend(cx_ring(&[(0, 1), (1, 2), (2, 0)]));
        assert_eq!(gates(&with_ranges(2, vec![2, 1]), 3), expected);
    }

    #[test]
    fn the_layer_addresses_logical_positions() {
        // After pooling the active set is a subset of the physical qubits; the
        // ring runs over logical positions and maps them through `active`.
        let ansatz = StronglyEntanglingAnsatz::new(1);
        let mut c = LayerContext {
            active: vec![1, 3],
            ..ctx(0)
        };
        let alloc = ansatz.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(4);
        ansatz.emit(&mut qc, &alloc, &[]).unwrap();
        let mut expected = rot(1, 0);
        expected.extend(rot(3, 3));
        expected.extend(cx_ring(&[(1, 3), (3, 1)]));
        assert_eq!(qc.gates(), expected);
    }

    #[test]
    fn param_count_is_three_per_qubit_per_layer() {
        // (num_qubits, reps, ranges, expected)
        let cases = [
            (3, 2, None, 18),
            (4, 4, None, 48),
            (1, 3, None, 9),
            (4, 0, None, 0),
            (3, 2, Some(vec![2, 2]), 18),
        ];
        for (nq, reps, ranges, expected) in cases {
            let ansatz = StronglyEntanglingAnsatz {
                ranges,
                ..StronglyEntanglingAnsatz::new(reps)
            };
            // Start past an earlier layer's parameters: the range is offset.
            let mut c = LayerContext {
                param_cursor: 5,
                ..ctx(nq)
            };
            let alloc = ansatz.plan(&mut c).unwrap();
            assert_eq!(alloc.params, 5..5 + expected);
            assert_eq!(c.param_cursor, 5 + expected);
        }
    }

    #[test]
    fn ranges_must_have_one_value_per_layer() {
        for ranges in [vec![1], vec![1, 1, 1], vec![]] {
            let got = ranges.len();
            assert_eq!(
                with_ranges(2, ranges).plan(&mut ctx(3)).unwrap_err(),
                ValidationError::StronglyEntanglingRangesLength { reps: 2, got }
            );
        }
        // Also on a single qubit, where the values themselves are not checked.
        assert!(with_ranges(2, vec![1]).plan(&mut ctx(1)).is_err());
    }

    #[test]
    fn ranges_must_lie_between_one_and_n_minus_one() {
        // (num_qubits, ranges, offending layer, offending range)
        let cases = [
            (3, vec![1, 0], 1, 0),
            (3, vec![3, 1], 0, 3),
            (2, vec![2], 0, 2),
            (4, vec![1, 2, 3, 9], 3, 9),
        ];
        for (nq, ranges, layer, range) in cases {
            let reps = ranges.len();
            let mut c = ctx(nq);
            assert_eq!(
                with_ranges(reps, ranges).plan(&mut c).unwrap_err(),
                ValidationError::StronglyEntanglingRangeOutOfBounds {
                    layer,
                    range,
                    active: nq,
                }
            );
            // A rejected layer reserves nothing.
            assert_eq!(c.param_cursor, 0);
        }
        // The bounds themselves are valid.
        assert!(with_ranges(2, vec![1, 3]).plan(&mut ctx(4)).is_ok());
    }

    #[test]
    fn plan_rejects_a_param_count_that_does_not_fit() {
        // 3 × 2 qubits × reps overflows.
        let mut c = LayerContext {
            param_cursor: 7,
            layers_planned: 3,
            ..ctx(2)
        };
        assert_eq!(
            StronglyEntanglingAnsatz::new(usize::MAX)
                .plan(&mut c)
                .unwrap_err(),
            ValidationError::TooManyParams { layer: 3 }
        );
        assert_eq!(c.param_cursor, 7);
        // A count that fits on its own but not after the earlier layers'.
        let mut c = LayerContext {
            param_cursor: usize::MAX - 5,
            layers_planned: 1,
            ..ctx(2)
        };
        assert_eq!(
            StronglyEntanglingAnsatz::new(1).plan(&mut c).unwrap_err(),
            ValidationError::TooManyParams { layer: 1 }
        );
        assert_eq!(c.param_cursor, usize::MAX - 5);
    }

    #[test]
    fn new_is_pennylane_s_default() {
        let a = StronglyEntanglingAnsatz::new(3);
        assert_eq!(a.reps, 3);
        assert_eq!(a.ranges, None);
        assert_eq!(a.entangler, Entangler::Cx);
    }
}
