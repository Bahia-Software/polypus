//! Ansätze: trainable layers that consume `θ` parameters (design doc §5.1).

mod hardware_efficient;
mod simplified_two_design;
mod strongly_entangling;

pub use hardware_efficient::{Entanglement, Entangler, HardwareEfficientAnsatz};
pub use simplified_two_design::SimplifiedTwoDesignAnsatz;
pub use strongly_entangling::StronglyEntanglingAnsatz;
