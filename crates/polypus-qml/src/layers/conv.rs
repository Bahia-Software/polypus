//! [`ConvLayer`]: a convolution block applied to pairs of active qubits with
//! **parameter sharing** — the essence of a QCNN convolution (design doc §6.4).
//!
//! The block reserves its `θ` slice *once per layer*, and every pair reuses the
//! same [`Param`] indices (translational invariance, few parameters). Two block
//! shapes are offered: a cheap [`ConvBlock::Basic`] and the full Cartan
//! interaction core [`ConvBlock::Cartan`], whose `ryy` factor is synthesized
//! from the C-2 vocabulary (no native `ryy`) as `(sdg⊗sdg)·rxx·(s⊗s)`.

use polypus_circuit::{GateInstruction, Param, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::{even_pair_count, even_pairs, odd_pair_count, odd_pairs};
use crate::model::{InputTrace, LayerAllocation, LayerContext, LayerOps};

/// The two-qubit block emitted on every pair of a [`ConvLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConvBlock {
    /// A cheap block of four shared rotations around a single `cx`
    /// (`ry(θ0) a · ry(θ1) b · cx(a,b) · ry(θ2) a · ry(θ3) b`).
    Basic,
    /// The full interaction core `exp(-i(α XX + β YY + γ ZZ)/2)` as
    /// `rxx(θ0) · ryy(θ1) · rzz(θ2)`, with `ryy` synthesized from the C-2
    /// vocabulary as `sdg,sdg,rxx,s,s` (since `Y = S·X·S†`).
    Cartan,
}

/// How a [`ConvLayer`] pairs up its active qubits (over logical positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pairing {
    /// `(0,1),(2,3),…` — `even_pairs`. Needs at least two active qubits.
    EvenPairs,
    /// `(1,2),(3,4),…` — `odd_pairs`. Needs at least three active qubits: on
    /// two there is no odd pair.
    OddPairs,
    /// All [`EvenPairs`](Self::EvenPairs) followed by all
    /// [`OddPairs`](Self::OddPairs) — the Cong–Choi–Lukin pattern (the default).
    /// Needs at least two active qubits.
    Alternating,
}

impl Pairing {
    /// The fewest active qubits on which this pairing selects a pair: 3 for
    /// [`OddPairs`](Self::OddPairs) (its first pair is `(1,2)`), 2 for the
    /// others (their first pair is `(0,1)`).
    pub(crate) fn min_active_qubits(self) -> usize {
        match self {
            Pairing::OddPairs => 3,
            Pairing::EvenPairs | Pairing::Alternating => 2,
        }
    }

    /// The spelling of this pairing in the Python API (`conv(pairing=…)`),
    /// which is what an error message names.
    pub(crate) fn python_name(self) -> &'static str {
        match self {
            Pairing::EvenPairs => "even_pairs",
            Pairing::OddPairs => "odd_pairs",
            Pairing::Alternating => "alternating",
        }
    }
}

/// A convolution layer: one shared block ([`ConvBlock`]) applied to every pair
/// of active qubits chosen by a [`Pairing`]. Every pair reuses the same `θ`
/// indices (parameter sharing), so the layer's parameter count is fixed by the
/// block alone, independent of the number of qubits.
///
/// Its `plan` refuses a position where the pairing selects no pair (fewer
/// active qubits than [`Pairing`] needs) with
/// [`ValidationError::ConvNeedsQubits`], so every `θ` it reserves is read by a
/// gate.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvLayer {
    /// The two-qubit block emitted on every pair.
    pub block: ConvBlock,
    /// The pairing pattern over the active qubits.
    pub pairing: Pairing,
}

impl ConvLayer {
    /// Build a convolution layer with the default [`Pairing::Alternating`].
    pub fn new(block: ConvBlock) -> Self {
        ConvLayer {
            block,
            pairing: Pairing::Alternating,
        }
    }

    /// The number of shared `θ` this block reserves for the whole layer.
    fn param_count(&self) -> usize {
        match self.block {
            ConvBlock::Basic => 4,
            ConvBlock::Cartan => 3,
        }
    }

    /// The gates the block emits on one pair ([`emit_basic`] and
    /// [`emit_cartan`]).
    fn block_gates(&self) -> usize {
        match self.block {
            ConvBlock::Basic => 5,
            ConvBlock::Cartan => 7,
        }
    }

    /// How many pairs [`pairs`](Self::pairs) gives over `n` active qubits.
    fn pair_count(&self, n: usize) -> usize {
        match self.pairing {
            Pairing::EvenPairs => even_pair_count(n),
            Pairing::OddPairs => odd_pair_count(n),
            Pairing::Alternating => even_pair_count(n) + odd_pair_count(n),
        }
    }

    /// The pairs, over logical positions of `n` active qubits, this layer's
    /// [`Pairing`] selects.
    fn pairs(&self, n: usize) -> Vec<(usize, usize)> {
        match self.pairing {
            Pairing::EvenPairs => even_pairs(n),
            Pairing::OddPairs => odd_pairs(n),
            Pairing::Alternating => {
                let mut pairs = even_pairs(n);
                pairs.extend(odd_pairs(n));
                pairs
            }
        }
    }
}

impl LayerOps for ConvLayer {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        let active = ctx.active.len();
        // Below this the pairing selects no pair: the θ reserved next would be
        // read by no gate.
        let needed = self.pairing.min_active_qubits();
        if active < needed {
            return Err(ValidationError::ConvNeedsQubits {
                pairing: self.pairing,
                needed,
                active,
            });
        }
        // Parameter sharing: the block reserves its θ once for the whole layer,
        // *not* per pair. The cursor advances by the block width only, while
        // the gates are the block's on every pair.
        let gates = self.pair_count(active).checked_mul(self.block_gates());
        let params = ctx.reserve(gates, Some(self.param_count()))?;
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
        let start = alloc.params.start;
        for (a, b) in self.pairs(alloc.active.len()) {
            // Logical positions → physical qubit indices.
            let (pa, pb) = (alloc.active[a], alloc.active[b]);
            match self.block {
                ConvBlock::Basic => emit_basic(qc, pa, pb, start)?,
                ConvBlock::Cartan => emit_cartan(qc, pa, pb, start)?,
            }
        }
        Ok(())
    }
}

/// `ry(θ0) a · ry(θ1) b · cx(a,b) · ry(θ2) a · ry(θ3) b`, reusing the shared
/// range `start..start+4` on every pair.
fn emit_basic(
    qc: &mut ParameterizedCircuit,
    a: usize,
    b: usize,
    start: usize,
) -> Result<(), QmlError> {
    qc.try_push(GateInstruction::Ry {
        qubit: a,
        theta: Param(start),
    })?;
    qc.try_push(GateInstruction::Ry {
        qubit: b,
        theta: Param(start + 1),
    })?;
    qc.try_push(GateInstruction::Cx(a, b))?;
    qc.try_push(GateInstruction::Ry {
        qubit: a,
        theta: Param(start + 2),
    })?;
    qc.try_push(GateInstruction::Ry {
        qubit: b,
        theta: Param(start + 3),
    })?;
    Ok(())
}

/// `rxx(θ0) · ryy(θ1) · rzz(θ2)`, with `ryy(θ1)` synthesized as the exact
/// sequence `sdg,sdg,rxx(θ1),s,s`. Reuses the shared range `start..start+3` on
/// every pair.
fn emit_cartan(
    qc: &mut ParameterizedCircuit,
    a: usize,
    b: usize,
    start: usize,
) -> Result<(), QmlError> {
    qc.try_push(GateInstruction::Rxx {
        q0: a,
        q1: b,
        theta: Param(start),
    })?;
    // ryy(θ1) = (sdg⊗sdg)·rxx(θ1)·(s⊗s), in circuit order.
    qc.try_push(GateInstruction::Sdg(a))?;
    qc.try_push(GateInstruction::Sdg(b))?;
    qc.try_push(GateInstruction::Rxx {
        q0: a,
        q1: b,
        theta: Param(start + 1),
    })?;
    qc.try_push(GateInstruction::S(a))?;
    qc.try_push(GateInstruction::S(b))?;
    qc.try_push(GateInstruction::Rzz {
        q0: a,
        q1: b,
        theta: Param(start + 2),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

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

    fn ry(qubit: usize, p: usize) -> GateInstruction {
        GateInstruction::Ry {
            qubit,
            theta: Param(p),
        }
    }

    fn rxx(q0: usize, q1: usize, p: usize) -> GateInstruction {
        GateInstruction::Rxx {
            q0,
            q1,
            theta: Param(p),
        }
    }

    #[test]
    fn even_odd_pairs_over_various_n() {
        assert_eq!(even_pairs(4), vec![(0, 1), (2, 3)]);
        assert_eq!(even_pairs(5), vec![(0, 1), (2, 3)]); // odd n: last unpaired
        assert_eq!(even_pairs(3), vec![(0, 1)]);
        assert_eq!(even_pairs(1), vec![]);
        assert_eq!(even_pairs(0), vec![]);

        assert_eq!(odd_pairs(4), vec![(1, 2)]);
        assert_eq!(odd_pairs(5), vec![(1, 2), (3, 4)]);
        assert_eq!(odd_pairs(3), vec![(1, 2)]);
        assert_eq!(odd_pairs(2), vec![]);
        assert_eq!(odd_pairs(1), vec![]);
        assert_eq!(odd_pairs(0), vec![]);
    }

    #[test]
    fn alternating_is_even_then_odd() {
        let layer = ConvLayer::new(ConvBlock::Basic);
        // n = 5: even (0,1),(2,3) then odd (1,2),(3,4).
        assert_eq!(layer.pairs(5), vec![(0, 1), (2, 3), (1, 2), (3, 4)]);
    }

    #[test]
    fn basic_emits_exact_five_gate_sequence_on_one_pair() {
        let layer = ConvLayer::new(ConvBlock::Basic);
        let mut c = ctx(2);
        let alloc = layer.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates(),
            vec![
                ry(0, 0),
                ry(1, 1),
                GateInstruction::Cx(0, 1),
                ry(0, 2),
                ry(1, 3),
            ]
        );
    }

    #[test]
    fn cartan_emits_exact_seven_gate_sequence_with_ryy_synthesis() {
        let layer = ConvLayer::new(ConvBlock::Cartan);
        let mut c = ctx(2);
        let alloc = layer.plan(&mut c).unwrap();
        let mut qc = ParameterizedCircuit::new(2);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates(),
            vec![
                rxx(0, 1, 0),
                GateInstruction::Sdg(0),
                GateInstruction::Sdg(1),
                rxx(0, 1, 1),
                GateInstruction::S(0),
                GateInstruction::S(1),
                GateInstruction::Rzz {
                    q0: 0,
                    q1: 1,
                    theta: Param(2),
                },
            ]
        );
    }

    #[test]
    fn basic_shares_parameters_across_pairs() {
        // 4 qubits, Alternating → 3 pairs, but still only 4 shared θ.
        let layer = ConvLayer::new(ConvBlock::Basic);
        let mut c = ctx(4);
        let alloc = layer.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..4);
        assert_eq!(c.param_cursor, 4);

        let mut qc = ParameterizedCircuit::new(4);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        // 3 pairs × 5 gates.
        assert_eq!(qc.gates().len(), 15);
        // No Param index outside 0..4 is ever used — the pairs reuse the range.
        assert_eq!(qc.num_params(), 4);
    }

    #[test]
    fn cartan_shares_parameters_across_pairs() {
        // 4 qubits, Alternating → 3 pairs, but still only 3 shared θ.
        let layer = ConvLayer::new(ConvBlock::Cartan);
        let mut c = ctx(4);
        let alloc = layer.plan(&mut c).unwrap();
        assert_eq!(alloc.params, 0..3);
        assert_eq!(c.param_cursor, 3);

        let mut qc = ParameterizedCircuit::new(4);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        // 3 pairs × 7 gates.
        assert_eq!(qc.gates().len(), 21);
        assert_eq!(qc.num_params(), 3);
    }

    #[test]
    fn plan_rejects_counts_that_do_not_fit_after_the_earlier_ones() {
        let overflow = ValidationError::ModelTooLarge {
            layer: 4,
            reason: SizeLimit::Overflow,
        };
        for block in [ConvBlock::Basic, ConvBlock::Cartan] {
            // The gates of the one pair after the earlier ones'.
            let mut c = LayerContext {
                gate_count: usize::MAX - 2,
                layers_planned: 4,
                ..ctx(2)
            };
            assert_eq!(ConvLayer::new(block).plan(&mut c).unwrap_err(), overflow);
            assert_eq!((c.param_cursor, c.gate_count), (0, usize::MAX - 2));
            // The shared θ after the earlier ones' (a state `compile` cannot
            // reach, see `LayerContext::reserve`): refused the same way.
            let mut c = LayerContext {
                param_cursor: usize::MAX - 2,
                layers_planned: 4,
                ..ctx(2)
            };
            assert_eq!(ConvLayer::new(block).plan(&mut c).unwrap_err(), overflow);
            assert_eq!((c.param_cursor, c.gate_count), (usize::MAX - 2, 0));
        }
    }

    /// Every parameter index some gate of `qc` reads. `num_params` is only the
    /// highest index plus one, so it cannot tell a skipped `θ` apart. Panics on
    /// a gate neither block emits, so a new angle-carrying gate cannot slip
    /// past unread.
    fn read_params(qc: &ParameterizedCircuit) -> BTreeSet<usize> {
        let mut read = BTreeSet::new();
        for gate in qc.gates() {
            match gate {
                GateInstruction::Ry { theta, .. }
                | GateInstruction::Rxx { theta, .. }
                | GateInstruction::Rzz { theta, .. } => match theta {
                    Param(i) => {
                        read.insert(*i);
                    }
                    other => panic!("unexpected angle {other:?}"),
                },
                GateInstruction::Cx(..) | GateInstruction::S(_) | GateInstruction::Sdg(_) => {}
                other => panic!("unexpected gate {other:?}"),
            }
        }
        read
    }

    const BLOCKS: [ConvBlock; 2] = [ConvBlock::Basic, ConvBlock::Cartan];
    const PAIRINGS: [Pairing; 3] = [Pairing::EvenPairs, Pairing::OddPairs, Pairing::Alternating];

    #[test]
    fn plan_rejects_fewer_than_two_active_qubits() {
        for pairing in [Pairing::EvenPairs, Pairing::Alternating] {
            for active in [0, 1] {
                let layer = ConvLayer {
                    block: ConvBlock::Basic,
                    pairing,
                };
                let mut c = ctx(active);
                assert_eq!(
                    layer.plan(&mut c).unwrap_err(),
                    ValidationError::ConvNeedsQubits {
                        pairing,
                        needed: 2,
                        active,
                    }
                );
                assert_eq!(c.param_cursor, 0);
            }
        }
    }

    #[test]
    fn plan_rejects_odd_pairs_without_an_odd_pair() {
        for block in BLOCKS {
            for active in [1, 2] {
                let layer = ConvLayer {
                    block,
                    pairing: Pairing::OddPairs,
                };
                let mut c = ctx(active);
                assert_eq!(
                    layer.plan(&mut c).unwrap_err(),
                    ValidationError::ConvNeedsQubits {
                        pairing: Pairing::OddPairs,
                        needed: 3,
                        active,
                    }
                );
                // A rejected layer reserves nothing.
                assert_eq!(c.param_cursor, 0);
            }
        }
    }

    #[test]
    fn plan_accepts_each_pairing_from_its_first_pair() {
        // OddPairs from three qubits: its one pair (1,2) gets the whole block.
        for (block, width) in [(ConvBlock::Basic, 4), (ConvBlock::Cartan, 3)] {
            let layer = ConvLayer {
                block,
                pairing: Pairing::OddPairs,
            };
            let mut c = ctx(3);
            assert_eq!(layer.plan(&mut c).unwrap().params, 0..width);
            assert_eq!(c.param_cursor, width);
        }
        let layer = ConvLayer {
            block: ConvBlock::Basic,
            pairing: Pairing::OddPairs,
        };
        let alloc = layer.plan(&mut ctx(3)).unwrap();
        let mut qc = ParameterizedCircuit::new(3);
        layer.emit(&mut qc, &alloc, &[]).unwrap();
        assert_eq!(
            qc.gates(),
            vec![
                ry(1, 0),
                ry(2, 1),
                GateInstruction::Cx(1, 2),
                ry(1, 2),
                ry(2, 3),
            ]
        );

        // EvenPairs and Alternating from two qubits: the pair (0,1).
        for block in BLOCKS {
            for pairing in [Pairing::EvenPairs, Pairing::Alternating] {
                assert!(ConvLayer { block, pairing }.plan(&mut ctx(2)).is_ok());
            }
        }
    }

    #[test]
    fn every_reserved_parameter_is_read() {
        // Whatever the block, the pairing and the width, a layer `plan`
        // accepts emits at least one gate and reads every θ it reserved; a
        // layer it refuses had no pair to act on.
        for block in BLOCKS {
            for pairing in PAIRINGS {
                for n in 0..=6 {
                    let layer = ConvLayer { block, pairing };
                    match layer.plan(&mut ctx(n)) {
                        Ok(alloc) => {
                            let mut qc = ParameterizedCircuit::new(n);
                            layer.emit(&mut qc, &alloc, &[]).unwrap();
                            assert!(!qc.gates().is_empty(), "{block:?} {pairing:?} n={n}");
                            assert_eq!(
                                read_params(&qc),
                                alloc.params.clone().collect::<BTreeSet<_>>(),
                                "{block:?} {pairing:?} n={n}"
                            );
                        }
                        Err(_) => assert!(
                            layer.pairs(n).is_empty(),
                            "{block:?} {pairing:?} n={n} refused with pairs to act on"
                        ),
                    }
                }
            }
        }
    }
}
