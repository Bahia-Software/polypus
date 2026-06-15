//! Photon interaction model: combines photoelectric, Compton, and pair
//! production into a single [`InteractionModel`].

pub mod compton;
pub mod pair_production;
pub mod photoelectric;

use super::{InteractionEvent, InteractionModel};
use crate::error::PhysicsError;
use crate::medium::Medium;
use crate::particle::photon::Photon;
use crate::particle::ParticleState;
use rand::Rng;

/// Photon interaction model over an arbitrary [`Medium`].
///
/// Couples the three classical photon processes relevant at keV–MeV energies:
/// photoelectric absorption, Compton scattering, and electron–positron pair
/// production. Process selection is by cumulative roulette on the partial
/// cross-sections.
pub struct PhotonInteractionModel;

impl InteractionModel for PhotonInteractionModel {
    type P = Photon;
    type M = dyn Medium;

    /// Total macroscopic cross-section `Σ_tot` (m⁻¹), the sum of the three
    /// atomic process cross-sections multiplied by the atom number density.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::NonPositiveEnergy`] if the photon energy is `≤ 0`.
    fn total_cross_section_per_m(
        &self,
        _particle: &Self::P,
        state: &ParticleState,
        medium: &Self::M,
    ) -> Result<f64, PhysicsError> {
        let e = state.momentum.energy_mev;
        if e <= 0.0 {
            return Err(PhysicsError::NonPositiveEnergy { energy_mev: e });
        }
        let z = medium.effective_z();
        let n = medium.number_density();
        let sigma_atom = photoelectric::cross_section(e, z)
            + compton::cross_section(e, z)
            + pair_production::cross_section(e, z);
        Ok(sigma_atom * n) // m⁻¹
    }

    /// Sample one photon interaction.
    ///
    /// 1. Compute the three partial atomic cross-sections.
    /// 2. Draw `u ~ Uniform(0, σ_tot)` and select the process by cumulative
    ///    roulette: photoelectric, then Compton, then pair production.
    /// 3. Dispatch to the corresponding sub-module sampler.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::NonPositiveEnergy`] if the photon energy is `≤ 0`, or
    /// [`PhysicsError::CrossSectionUndefined`] if all partial cross-sections
    /// vanish (no interaction is possible).
    fn sample_interaction(
        &self,
        _particle: &Self::P,
        state: &ParticleState,
        medium: &Self::M,
        rng: &mut impl Rng,
    ) -> Result<InteractionEvent, PhysicsError> {
        let e = state.momentum.energy_mev;
        if e <= 0.0 {
            return Err(PhysicsError::NonPositiveEnergy { energy_mev: e });
        }
        let z = medium.effective_z();

        let tau = photoelectric::cross_section(e, z);
        let sigma = compton::cross_section(e, z);
        let kappa = pair_production::cross_section(e, z);
        let total = tau + sigma + kappa;

        if total <= 0.0 {
            return Err(PhysicsError::CrossSectionUndefined {
                message: format!("all photon cross-sections vanish at E = {e} MeV"),
            });
        }

        let u: f64 = rng.gen_range(0.0..total);
        if u < tau {
            let (deposit, secondaries) = photoelectric::sample(state);
            Ok(InteractionEvent::Absorbed {
                energy_deposit_mev: deposit,
                secondaries,
            })
        } else if u < tau + sigma {
            let (new_state, deposit, secondaries) = compton::sample(state, rng);
            Ok(InteractionEvent::Scattered {
                new_state,
                energy_deposit_mev: deposit,
                secondaries,
            })
        } else {
            let (deposit, secondaries) = pair_production::sample(state);
            Ok(InteractionEvent::Absorbed {
                energy_deposit_mev: deposit,
                secondaries,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    use crate::particle::photon::Photon;

    #[test]
    fn compton_dominates_in_water_at_100kev() {
        let z = HomogeneousMedium::water().effective_z;
        assert!(compton::cross_section(0.1, z) > photoelectric::cross_section(0.1, z));
    }

    #[test]
    fn photoelectric_dominates_in_lead_at_10kev() {
        let z = HomogeneousMedium::lead().effective_z;
        assert!(photoelectric::cross_section(0.01, z) > compton::cross_section(0.01, z));
    }

    #[test]
    fn total_cross_section_one_mev_water_matches_nist() {
        let model = PhotonInteractionModel;
        let medium = HomogeneousMedium::water();
        let state = Photon::state_along_z(1.0);
        let sigma_tot = model
            .total_cross_section_per_m(&Photon, &state, &medium)
            .unwrap();
        // NIST XCOM linear attenuation for 1 MeV in water is μ ≈ 7.07 m⁻¹
        // (mean free path ≈ 0.14 m). At 1 MeV water attenuation is ~99 %
        // Compton; with the electron-density-consistent A_eff the free-electron
        // Klein–Nishina total reproduces NIST to ~1 % (≈ 7.06 m⁻¹).
        assert!(
            (6.8..=7.4).contains(&sigma_tot),
            "Σ_tot = {sigma_tot} m⁻¹ outside [6.8, 7.4]"
        );
    }
}
