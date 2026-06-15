//! Error types for the physics layer.

use std::fmt;

/// Errors raised by physics calculations, validation, and Monte Carlo
/// transport.
///
/// This crate has no dependency on `polypus-circuit`, so there is
/// deliberately **no** `From<CircuitError>` conversion here.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsError {
    /// Energy is zero or negative where a positive value is required.
    NonPositiveEnergy {
        /// The offending energy value (MeV).
        energy_mev: f64,
    },
    /// A cross-section is undefined at this energy / medium combination.
    CrossSectionUndefined {
        /// Human-readable description of the problem.
        message: String,
    },
    /// The medium has unphysical parameters (e.g. negative density).
    InvalidMedium {
        /// Human-readable description of the problem.
        message: String,
    },
    /// A Monte Carlo run was aborted (e.g. `max_steps` exceeded with no
    /// absorption, or a state failed validation mid-transport).
    SimulationError {
        /// Human-readable description of the problem.
        message: String,
    },
    /// A particle energy spectrum was configured with invalid parameters
    /// (e.g. non-positive energies, `min ≥ max`, or empty / all-zero bins).
    InvalidSpectrum {
        /// Human-readable description of the problem.
        message: String,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::NonPositiveEnergy { energy_mev } => {
                write!(f, "non-positive energy: {energy_mev} MeV")
            }
            PhysicsError::CrossSectionUndefined { message } => {
                write!(f, "cross-section undefined: {message}")
            }
            PhysicsError::InvalidMedium { message } => {
                write!(f, "invalid medium: {message}")
            }
            PhysicsError::SimulationError { message } => {
                write!(f, "simulation error: {message}")
            }
            PhysicsError::InvalidSpectrum { message } => {
                write!(f, "invalid spectrum: {message}")
            }
        }
    }
}

impl std::error::Error for PhysicsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_energy_value() {
        let err = PhysicsError::NonPositiveEnergy { energy_mev: -1.0 };
        assert!(err.to_string().contains("-1"));
    }

    #[test]
    fn error_trait_is_implemented() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&PhysicsError::SimulationError {
            message: "x".into(),
        });
    }
}
