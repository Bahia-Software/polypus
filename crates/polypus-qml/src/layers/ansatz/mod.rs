//! Ansätze: trainable layers that consume `θ` parameters (design doc §5.1).

mod hardware_efficient;

pub use hardware_efficient::{Entanglement, Entangler, HardwareEfficientAnsatz};
