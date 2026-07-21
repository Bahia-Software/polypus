//! The layer catalogue: encoders (features → angles) and ansätze (trainable
//! `θ`), plus the shared [`RotationAxis`] and (added alongside the model) the
//! closed [`Layer`] enum that ties them together.
//!
//! Each concrete layer implements the internal
//! [`LayerOps`](crate::model::LayerOps) trait; the public API is the closed
//! `Layer` enum rather than a trait object, so the catalogue stays
//! enumerable, serializable and FFI-friendly (design doc D1).

mod ansatz;
mod encoders;

pub use ansatz::{Entanglement, Entangler, HardwareEfficientAnsatz};
pub use encoders::AngleEncoder;

use polypus_circuit::ParameterizedCircuit;

use crate::error::{QmlError, ValidationError};
use crate::model::{LayerAllocation, LayerContext, LayerOps};

/// The rotation axis of a single-qubit rotation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub enum Layer {
    /// Feature encoding via single-qubit rotations (consumes no `θ`).
    AngleEncoder(AngleEncoder),
    /// A hardware-efficient variational block (consumes `θ`).
    HardwareEfficient(HardwareEfficientAnsatz),
}

impl LayerOps for Layer {
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError> {
        match self {
            Layer::AngleEncoder(l) => l.plan(ctx),
            Layer::HardwareEfficient(l) => l.plan(ctx),
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
            Layer::HardwareEfficient(l) => l.emit(qc, alloc, x),
        }
    }
}
