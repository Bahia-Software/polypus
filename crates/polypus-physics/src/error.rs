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
    /// No embedded ENDF-6 data is available for a given element symbol.
    UnknownElement {
        symbol: String,
    },
    /// The ENDF-6 file does not have the expected format.
    MalformedEndfData {
        message: String,
    },
    /// A CSV export (or other file I/O) operation failed.
    IoError {
        message: String,
    },
    /// A chemical formula string is not well-formed (e.g. does not start
    /// with an uppercase letter, or has an invalid atom count).
    InvalidChemicalFormula {
        message: String,
    },
    /// The constituent elements of a compound/mixture have no overlapping
    /// energy range in their ENDF-6 evaluations, so no common energy grid
    /// can be built.
    NoEnergyOverlap,
    UntabulatedMedium {
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
            PhysicsError::UnknownElement { symbol } => {
                write!(f, "unknown element symbol: {symbol}")
            }
            PhysicsError::MalformedEndfData { message } => {
                write!(f, "malformed ENDF-6 data: {message}")
            }
            PhysicsError::IoError { message } => {
                write!(f, "I/O error: {message}")
            }
            PhysicsError::InvalidChemicalFormula { message } => {
                write!(f, "invalid chemical formula: {message}")
            }
            PhysicsError::NoEnergyOverlap => {
                write!(f, "constituent elements have no overlapping energy range")
            }
            PhysicsError::UntabulatedMedium { message } => {
                write!(f, "untabulated medium: {message}")
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
