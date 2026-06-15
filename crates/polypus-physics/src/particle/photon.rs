//! The photon: a massless, spin-1, electrically neutral boson (PDG ID 22).

use super::{Charge, FourMomentum, Particle, ParticleState, Position, Spin};
use crate::error::PhysicsError;

/// The photon — a massless, spin-1, electrically neutral boson (PDG ID 22).
///
/// A unit struct because photons have no tunable species parameters; all
/// physics is encoded in the energy and direction of each [`ParticleState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Photon;

impl Particle for Photon {
    fn pdg_id(&self) -> i32 {
        22
    }

    fn rest_mass_mev(&self) -> f64 {
        0.0
    }

    fn spin(&self) -> Spin {
        Spin(1.0)
    }

    fn charge(&self) -> Charge {
        Charge(0.0)
    }

    fn name(&self) -> &'static str {
        "photon"
    }

    /// Encode photon polarization as a qubit.
    ///
    /// The transverse plane (x-y) defines the qubit basis:
    /// `|H⟩ ≡ |0⟩` (horizontal), `|V⟩ ≡ |1⟩` (vertical). The returned pair
    /// `[α, β]` is the normalized transverse projection of the propagation
    /// direction.
    ///
    /// Returns `None` for a photon travelling exactly along ±z
    /// (direction cosines `dx = dy = 0` — no transverse component to encode).
    fn qubit_encoding(&self, state: &ParticleState) -> Option<[f64; 2]> {
        let [dx, dy, _] = state.momentum.direction;
        let norm = (dx * dx + dy * dy).sqrt();
        if norm < 1e-12 {
            return None;
        }
        Some([dx / norm, dy / norm])
    }

    /// Validate that the photon has positive energy and a unit direction
    /// vector.
    ///
    /// # Errors
    ///
    /// - [`PhysicsError::NonPositiveEnergy`] if `energy_mev ≤ 0`.
    /// - [`PhysicsError::SimulationError`] if `|direction| ≠ 1`
    ///   (tolerance `1e-6`).
    fn validate_state(&self, state: &ParticleState) -> Result<(), PhysicsError> {
        if state.momentum.energy_mev <= 0.0 {
            return Err(PhysicsError::NonPositiveEnergy {
                energy_mev: state.momentum.energy_mev,
            });
        }
        let [dx, dy, dz] = state.momentum.direction;
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if (mag - 1.0).abs() > 1e-6 {
            return Err(PhysicsError::SimulationError {
                message: format!("photon direction magnitude {mag:.8} ≠ 1"),
            });
        }
        Ok(())
    }
}

impl Photon {
    /// Photon wavelength `λ = h·c / E` (metres).
    ///
    /// `energy_mev` is the photon energy in MeV. Uses the conversion
    /// `h·c = 1.23984193e-12 MeV·m`.
    pub fn wavelength_m(energy_mev: f64) -> f64 {
        // hc = 1239.84193 eV·nm = 1.23984193e-12 MeV·m
        1.239_841_93e-12 / energy_mev
    }

    /// Angular frequency `ω = E / ħ` (rad/s).
    ///
    /// `energy_mev` is the photon energy in MeV; it is converted to eV before
    /// dividing by ħ (in eV·s).
    pub fn angular_frequency_rad_s(energy_mev: f64) -> f64 {
        (energy_mev * 1e6) / crate::constants::HBAR_EV_S
    }

    /// Construct a photon [`ParticleState`] travelling along +z with the given
    /// energy (MeV). Useful for benchmarks and unit tests.
    pub fn state_along_z(energy_mev: f64) -> ParticleState {
        ParticleState {
            position: Position([0.0, 0.0, 0.0]),
            momentum: FourMomentum {
                energy_mev,
                direction: [0.0, 0.0, 1.0],
            },
            alive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavelength_of_one_mev_photon() {
        let lambda = Photon::wavelength_m(1.0);
        assert!((lambda - 1.240e-12).abs() < 1e-15);
    }

    #[test]
    fn non_positive_energy_rejected() {
        let mut state = Photon::state_along_z(1.0);
        state.momentum.energy_mev = 0.0;
        assert!(matches!(
            Photon.validate_state(&state),
            Err(PhysicsError::NonPositiveEnergy { .. })
        ));
    }

    #[test]
    fn non_unit_direction_rejected() {
        let mut state = Photon::state_along_z(1.0);
        state.momentum.direction = [1.0, 1.0, 1.0];
        assert!(matches!(
            Photon.validate_state(&state),
            Err(PhysicsError::SimulationError { .. })
        ));
    }

    #[test]
    fn z_directed_photon_has_no_qubit_encoding() {
        let state = Photon::state_along_z(1.0);
        assert_eq!(Photon.qubit_encoding(&state), None);
    }

    #[test]
    fn x_directed_photon_encodes_horizontal() {
        let mut state = Photon::state_along_z(1.0);
        state.momentum.direction = [1.0, 0.0, 0.0];
        assert_eq!(Photon.qubit_encoding(&state), Some([1.0, 0.0]));
    }

    #[test]
    fn valid_state_accepted() {
        assert!(Photon.validate_state(&Photon::state_along_z(1.0)).is_ok());
    }
}
