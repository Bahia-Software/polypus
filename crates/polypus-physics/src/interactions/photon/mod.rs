//! Photon interaction model: combines photoelectric, Compton, and pair
//! production into a single [`InteractionModel`].

pub mod coherent;
pub mod compton;
pub mod cross_section_plots;
pub mod mass_attenuation_coefficients;
pub mod mass_attenuation_coefficients_plots;
pub mod pair_production;
pub mod photoelectric;

use super::{InteractionEvent, InteractionModel};
use crate::error::PhysicsError;
use crate::medium::compound::PhotonChannel;
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

impl PhotonInteractionModel {
    /// Macroscopic cross-section (m⁻¹) contribution from one channel,
    /// straight from the medium's embedded ENDF-6 data.
    fn macroscopic_cross_section(
        &self,
        medium: &dyn Medium,
        channel: PhotonChannel,
        e: f64,
    ) -> Result<f64, PhysicsError> {
        let mu_m = medium.tabulated_mu_m_cm2_g(channel, e).ok_or_else(|| {
            PhysicsError::UntabulatedMedium {
                message: format!("no tabulated data for channel {channel:?} at E = {e} MeV"),
            }
        })?;
        Ok(mu_m * medium.density_kg_m3() * 0.1)
    }
}

impl InteractionModel for PhotonInteractionModel {
    type P = Photon;
    type M = dyn Medium;

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
        Ok(
            self.macroscopic_cross_section(medium, PhotonChannel::Photoelectric, e)?
                + self.macroscopic_cross_section(medium, PhotonChannel::Coherent, e)?
                + self.macroscopic_cross_section(medium, PhotonChannel::Incoherent, e)?
                + self.macroscopic_cross_section(medium, PhotonChannel::PairProductionTotal, e)?,
        )
    }

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

        let tau = self.macroscopic_cross_section(medium, PhotonChannel::Photoelectric, e)?;
        let rayleigh = self.macroscopic_cross_section(medium, PhotonChannel::Coherent, e)?;
        let sigma = self.macroscopic_cross_section(medium, PhotonChannel::Incoherent, e)?;
        let kappa =
            self.macroscopic_cross_section(medium, PhotonChannel::PairProductionTotal, e)?;
        let total = tau + rayleigh + sigma + kappa;

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
        } else if u < tau + rayleigh {
            let (new_state, deposit, secondaries) = coherent::sample(state, rng);
            Ok(InteractionEvent::Scattered {
                new_state,
                energy_deposit_mev: deposit,
                secondaries,
            })
        } else if u < tau + rayleigh + sigma {
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
    use crate::medium::CompoundMedium;
    use crate::particle::photon::Photon;

    fn water() -> CompoundMedium {
        CompoundMedium::new("H2O", 1000.0, 5000).unwrap()
    }

    fn lead() -> CompoundMedium {
        CompoundMedium::new("Pb", 11_350.0, 5000).unwrap()
    }

    #[test]
    fn compton_dominates_in_water_at_100kev() {
        let medium = water();
        let compton_mu = medium.mu_m_cm2_g(PhotonChannel::Incoherent, 0.1);
        let photoelectric_mu = medium.mu_m_cm2_g(PhotonChannel::Photoelectric, 0.1);
        assert!(compton_mu > photoelectric_mu);
    }

    #[test]
    fn photoelectric_dominates_in_lead_at_10kev() {
        let medium = lead();
        let compton_mu = medium.mu_m_cm2_g(PhotonChannel::Incoherent, 0.01);
        let photoelectric_mu = medium.mu_m_cm2_g(PhotonChannel::Photoelectric, 0.01);
        assert!(photoelectric_mu > compton_mu);
    }

    #[test]
    fn total_cross_section_one_mev_water_matches_nist() {
        let model = PhotonInteractionModel;
        let medium = water();
        let state = Photon::state_along_z(1.0);
        let sigma_tot = model
            .total_cross_section_per_m(&Photon, &state, &medium)
            .unwrap();
        assert!(
            (6.8..=7.4).contains(&sigma_tot),
            "Σ_tot = {sigma_tot} m⁻¹ outside [6.8, 7.4]"
        );
    }
}
