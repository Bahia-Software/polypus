//! Ansätze: trainable layers that consume `θ` parameters (design doc §5.1).

mod hardware_efficient;
mod strongly_entangling;

pub use hardware_efficient::{Entanglement, Entangler, HardwareEfficientAnsatz};
pub use strongly_entangling::StronglyEntanglingAnsatz;
