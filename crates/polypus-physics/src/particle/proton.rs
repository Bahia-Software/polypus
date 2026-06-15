//! The proton: a spin-½, charge +1 hadron (PDG ID 2212).
//!
//! This module is the canonical template for future charged hadrons: copy the
//! pattern, change the species constants, and implement a matching
//! [`InteractionModel`](crate::interactions::InteractionModel).

use super::{Charge, Particle, ParticleState, Spin};
use crate::error::PhysicsError;

/// The proton — spin-½, charge +1, rest mass 938.272 MeV/c² (PDG ID 2212).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Proton;

impl Particle for Proton {
    fn pdg_id(&self) -> i32 {
        2212
    }

    fn rest_mass_mev(&self) -> f64 {
        crate::constants::PROTON_MASS_MEV
    }

    fn spin(&self) -> Spin {
        Spin(0.5)
    }

    fn charge(&self) -> Charge {
        Charge(1.0)
    }

    fn name(&self) -> &'static str {
        "proton"
    }

    /// No single-qubit encoding is defined at this abstraction level.
    fn qubit_encoding(&self, _state: &ParticleState) -> Option<[f64; 2]> {
        None
    }

    /// Validate that the proton's total energy is at least its rest mass and
    /// that its direction is a unit vector.
    ///
    /// # Errors
    ///
    /// - [`PhysicsError::NonPositiveEnergy`] if `energy_mev < m_p·c²`.
    /// - [`PhysicsError::SimulationError`] if `|direction| ≠ 1`
    ///   (tolerance `1e-6`).
    fn validate_state(&self, state: &ParticleState) -> Result<(), PhysicsError> {
        if state.momentum.energy_mev < crate::constants::PROTON_MASS_MEV {
            return Err(PhysicsError::NonPositiveEnergy {
                energy_mev: state.momentum.energy_mev,
            });
        }
        let [dx, dy, dz] = state.momentum.direction;
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if (mag - 1.0).abs() > 1e-6 {
            return Err(PhysicsError::SimulationError {
                message: format!("proton direction magnitude {mag:.8} ≠ 1"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{FourMomentum, Position};

    fn proton_state(energy_mev: f64) -> ParticleState {
        ParticleState {
            position: Position([0.0, 0.0, 0.0]),
            momentum: FourMomentum {
                energy_mev,
                direction: [0.0, 0.0, 1.0],
            },
            alive: true,
        }
    }

    #[test]
    fn rest_mass_is_proton_mass() {
        assert!((Proton.rest_mass_mev() - 938.272).abs() < 1e-2);
    }

    #[test]
    fn energy_below_rest_mass_rejected() {
        let state = proton_state(900.0);
        assert!(matches!(
            Proton.validate_state(&state),
            Err(PhysicsError::NonPositiveEnergy { .. })
        ));
    }

    #[test]
    fn energy_above_rest_mass_accepted() {
        let state = proton_state(1000.0);
        assert!(Proton.validate_state(&state).is_ok());
    }
}
