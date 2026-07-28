//! The layer catalogue: encoders (features → angles) and ansätze (trainable
//! `θ`), plus the shared [`RotationAxis`] and (added alongside the model) the
//! closed [`Layer`] enum that ties them together.
//!
//! Each concrete layer implements the internal
//! [`LayerOps`](crate::model::LayerOps) trait; the public API is the closed
//! `Layer` enum rather than a trait object, so the catalogue stays
//! enumerable, serializable and FFI-friendly (design doc D1).

mod ansatz;
mod conv;
mod encoders;
mod pool;

pub use ansatz::{Entanglement, Entangler, HardwareEfficientAnsatz};
pub use conv::{ConvBlock, ConvLayer, Pairing};
pub use encoders::{AmplitudeEncoder, AngleEncoder, IqpEncoder};
pub use pool::{KeepRule, PoolBlock, PoolLayer};

use polypus_circuit::ParameterizedCircuit;

use crate::error::{QmlError, ValidationError};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// Adjacent, non-overlapping pairs over `n` logical positions:
/// `(0,1),(2,3),…` — pair `k` is `(2k, 2k+1)` for `k` in `0..n/2`. With `n`
/// odd the last position is left unpaired (design doc §6.4). Shared by
/// [`conv`] and [`pool`].
pub(crate) fn even_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n / 2).map(|k| (2 * k, 2 * k + 1)).collect()
}

/// Adjacent, non-overlapping pairs shifted by one over `n` logical positions:
/// `(1,2),(3,4),…` — pair `k` is `(2k+1, 2k+2)` for `k` in `0..(n-1)/2`. The
/// `n == 0` case is guarded explicitly so the `n - 1` never underflows; for
/// `n == 1` the integer division already yields the empty range (design doc
/// §6.4).
pub(crate) fn odd_pairs(n: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    (0..(n - 1) / 2).map(|k| (2 * k + 1, 2 * k + 2)).collect()
}

/// The pairs of logical positions an [`Entanglement`] pattern selects over `n`
/// active qubits: `Linear` → `(0,1),…,(n-2,n-1)`; `Circular` → `Linear` plus
/// the wrap-around `(n-1,0)`; `Full` → every `(i,j)` with `i < j`, in
/// `i`-major order. Fewer than two positions yields an empty list — the sole
/// guard for the degenerate cases, so callers need none of their own (`n == 0`
/// would underflow `n - 1`, and `Circular` on `n == 1` would emit the
/// self-pair `(0,0)`). Shared by [`ansatz`] and [`encoders`].
pub(crate) fn entanglement_pairs(n: usize, entanglement: Entanglement) -> Vec<(usize, usize)> {
    if n < 2 {
        return Vec::new();
    }
    match entanglement {
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
    }
}

/// The rotation axis of a single-qubit rotation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RotationAxis {
    /// Rotation about the X axis (`rx`).
    Rx,
    /// Rotation about the Y axis (`ry`).
    Ry,
    /// Rotation about the Z axis (`rz`).
    Rz,
}

/// A model layer: the closed set of building blocks a [`QuantumModel`] stacks.
///
/// Closed (not a trait object) so the catalogue is enumerable, serializable and
/// FFI-friendly (design doc D1). Adding a layer is a new variant plus its
/// struct. Measurement is **not** a layer — it is fixed by the model
/// (design doc D2).
///
/// [`QuantumModel`]: crate::QuantumModel
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Layer {
    /// Feature encoding via single-qubit rotations (consumes no `θ`).
    AngleEncoder(AngleEncoder),
    /// State-preparation feature encoding via multiplexed `Ry` rotations
    /// (consumes no `θ`; must be the first layer).
    AmplitudeEncoder(AmplitudeEncoder),
    /// IQP / `ZZFeatureMap` feature encoding: `H⊗n`, `Rz(x_i)` and
    /// `Rzz(x_i·x_j)` over the pairs of an [`Entanglement`] pattern (consumes
    /// no `θ`).
    Iqp(IqpEncoder),
    /// A hardware-efficient variational block (consumes `θ`).
    HardwareEfficient(HardwareEfficientAnsatz),
    /// A convolution block with parameter sharing across pairs (consumes `θ`).
    Conv(ConvLayer),
    /// A unitary pooling block that shrinks the active set (consumes `θ`).
    Pool(PoolLayer),
}

impl LayerOps for Layer {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        match self {
            Layer::AngleEncoder(l) => l.plan(ctx),
            Layer::AmplitudeEncoder(l) => l.plan(ctx),
            Layer::Iqp(l) => l.plan(ctx),
            Layer::HardwareEfficient(l) => l.plan(ctx),
            Layer::Conv(l) => l.plan(ctx),
            Layer::Pool(l) => l.plan(ctx),
        }
    }

    fn emit(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        x: &[f64],
    ) -> Result<(), QmlError> {
        match self {
            Layer::AngleEncoder(l) => l.emit(qc, alloc, x),
            Layer::AmplitudeEncoder(l) => l.emit(qc, alloc, x),
            Layer::Iqp(l) => l.emit(qc, alloc, x),
            Layer::HardwareEfficient(l) => l.emit(qc, alloc, x),
            Layer::Conv(l) => l.emit(qc, alloc, x),
            Layer::Pool(l) => l.emit(qc, alloc, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_pairs_form_a_chain() {
        assert_eq!(
            entanglement_pairs(4, Entanglement::Linear),
            vec![(0, 1), (1, 2), (2, 3)]
        );
        assert_eq!(
            entanglement_pairs(3, Entanglement::Linear),
            vec![(0, 1), (1, 2)]
        );
        assert_eq!(entanglement_pairs(2, Entanglement::Linear), vec![(0, 1)]);
    }

    #[test]
    fn circular_pairs_add_the_wraparound() {
        assert_eq!(
            entanglement_pairs(4, Entanglement::Circular),
            vec![(0, 1), (1, 2), (2, 3), (3, 0)]
        );
        assert_eq!(
            entanglement_pairs(3, Entanglement::Circular),
            vec![(0, 1), (1, 2), (2, 0)]
        );
        // n == 2: the wrap-around (1,0) is a second, reversed pair on the same
        // two positions — kept, since the gate is not always symmetric.
        assert_eq!(
            entanglement_pairs(2, Entanglement::Circular),
            vec![(0, 1), (1, 0)]
        );
    }

    #[test]
    fn full_pairs_are_every_i_lt_j_in_i_major_order() {
        assert_eq!(
            entanglement_pairs(4, Entanglement::Full),
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
        assert_eq!(
            entanglement_pairs(3, Entanglement::Full),
            vec![(0, 1), (0, 2), (1, 2)]
        );
        assert_eq!(entanglement_pairs(2, Entanglement::Full), vec![(0, 1)]);
    }

    #[test]
    fn fewer_than_two_positions_yields_no_pairs() {
        // The guard every caller relies on: `n - 1` must not underflow at
        // n == 0, and `Circular` must not emit the self-pair (0,0) at n == 1.
        for entanglement in [
            Entanglement::Linear,
            Entanglement::Circular,
            Entanglement::Full,
        ] {
            assert_eq!(entanglement_pairs(0, entanglement), vec![]);
            assert_eq!(entanglement_pairs(1, entanglement), vec![]);
        }
    }
}
