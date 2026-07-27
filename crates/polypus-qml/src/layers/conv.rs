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
use crate::layers::{even_pairs, odd_pairs};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

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
    /// `(0,1),(2,3),…` — [`even_pairs`](crate::layers::even_pairs).
    EvenPairs,
    /// `(1,2),(3,4),…` — [`odd_pairs`](crate::layers::odd_pairs).
    OddPairs,
    /// All [`EvenPairs`](Self::EvenPairs) followed by all
    /// [`OddPairs`](Self::OddPairs) — the Cong–Choi–Lukin pattern (the default).
    Alternating,
}

/// A convolution layer: one shared block ([`ConvBlock`]) applied to every pair
/// of active qubits chosen by a [`Pairing`]. Every pair reuses the same `θ`
/// indices (parameter sharing), so the layer's parameter count is fixed by the
/// block alone, independent of the number of qubits.
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
        if active < 2 {
            return Err(ValidationError::NotEnoughQubits { needed: 2, active });
        }
        // Parameter sharing: the block reserves its θ once for the whole layer,
        // *not* per pair. The cursor advances by the block width only.
        let count = self.param_count();
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
    use super::*;

    fn ctx(num_qubits: usize) -> LayerContext {
        LayerContext {
            active: (0..num_qubits).collect(),
            num_features: 0,
            param_cursor: 0,
            layers_planned: 0,
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
            qc.gates,
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
            qc.gates,
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
        assert_eq!(qc.gates.len(), 15);
        // No Param index outside 0..4 is ever used — the pairs reuse the range.
        assert_eq!(qc.num_params, 4);
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
        assert_eq!(qc.gates.len(), 21);
        assert_eq!(qc.num_params, 3);
    }

    #[test]
    fn plan_rejects_fewer_than_two_active_qubits() {
        let layer = ConvLayer::new(ConvBlock::Basic);
        let mut c = ctx(1);
        assert!(matches!(
            layer.plan(&mut c),
            Err(ValidationError::NotEnoughQubits {
                needed: 2,
                active: 1,
            })
        ));
    }
}
