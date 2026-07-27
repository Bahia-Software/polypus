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
pub use encoders::{AmplitudeEncoder, AngleEncoder};
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
            Layer::HardwareEfficient(l) => l.emit(qc, alloc, x),
            Layer::Conv(l) => l.emit(qc, alloc, x),
            Layer::Pool(l) => l.emit(qc, alloc, x),
        }
    }
}
