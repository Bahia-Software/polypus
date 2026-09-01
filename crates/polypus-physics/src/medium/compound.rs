//! A material defined by a real chemical formula, backed by embedded
//! ENDF-6 photon cross-sections rather than an analytic approximation.

use std::collections::HashMap;

use super::Medium;
use crate::error::PhysicsError;
use crate::interactions::photon::mass_attenuation_coefficients::{
    interpolate_loglog_precomputed, mu_m_for_compound, precompute_loglog,
};

/// The ENDF-6 photon reaction channels this medium keeps tabulated data for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhotonChannel {
    /// MT=501: total cross-section.
    Total,
    /// MT=502: coherent (Rayleigh) scattering.
    Coherent,
    /// MT=504: incoherent (Compton) scattering.
    Incoherent,
    /// MT=522: photoionization.
    Photoelectric,
    /// MT=516: pair production (total).
    PairProductionTotal,
}

impl PhotonChannel {
    fn mt(self) -> u32 {
        match self {
            PhotonChannel::Total => 501,
            PhotonChannel::Coherent => 502,
            PhotonChannel::Incoherent => 504,
            PhotonChannel::Photoelectric => 522,
            PhotonChannel::PairProductionTotal => 516,
        }
    }
}

/// A material defined by a real chemical formula (e.g. `"H2O"`,
/// `"Fe2O3"`), whose photon cross-sections come from embedded ENDF-6
/// evaluations rather than an analytic approximation — see
/// [`crate::interactions::photon::mass_attenuation_coefficients`].
#[derive(Debug, Clone)]
pub struct CompoundMedium {
    pub formula: String,
    pub density_kg_m3: f64,
    channels: HashMap<PhotonChannel, (Vec<f64>, Vec<f64>)>, // (log_energias, log_mu), ya calculados
}

impl CompoundMedium {
    pub fn new(formula: &str, density_kg_m3: f64, n_points: usize) -> Result<Self, PhysicsError> {
        let mut channels = HashMap::new();
        for channel in [
            PhotonChannel::Total,
            PhotonChannel::Coherent,
            PhotonChannel::Incoherent,
            PhotonChannel::Photoelectric,
            PhotonChannel::PairProductionTotal,
        ] {
            let result = mu_m_for_compound(formula, channel.mt(), n_points)?;
            channels.insert(channel, precompute_loglog(&result.points));
        }
        Ok(CompoundMedium {
            formula: formula.to_string(),
            density_kg_m3,
            channels,
        })
    }

    pub fn mu_m_cm2_g(&self, channel: PhotonChannel, energy_mev: f64) -> f64 {
        let (log_energies, log_mu) = &self.channels[&channel];
        interpolate_loglog_precomputed(log_energies, log_mu, energy_mev * 1e6)
    }

    pub fn macroscopic_cross_section_per_m(&self, channel: PhotonChannel, energy_mev: f64) -> f64 {
        let density_g_cm3 = self.density_kg_m3 * 1e-3;
        self.mu_m_cm2_g(channel, energy_mev) * density_g_cm3 * 100.0
    }
}

impl Medium for CompoundMedium {
    fn density_kg_m3(&self) -> f64 {
        self.density_kg_m3
    }

    fn effective_z(&self) -> f64 {
        f64::NAN // not used by this medium's real photon physics
    }

    fn effective_a(&self) -> f64 {
        f64::NAN // not used by this medium's real photon physics
    }

    fn tabulated_mu_m_cm2_g(&self, channel: PhotonChannel, energy_mev: f64) -> Option<f64> {
        Some(self.mu_m_cm2_g(channel, energy_mev))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_total_cross_section_matches_nist_at_1_mev() {
        let medium = CompoundMedium::new("H2O", 1000.0, 5000).unwrap();
        let sigma = medium.macroscopic_cross_section_per_m(PhotonChannel::Total, 1.0);
        // Same NIST reference already used by PhotonInteractionModel's own test.
        assert!((6.8..=7.4).contains(&sigma), "sigma = {sigma} m^-1");
    }

    #[test]
    fn channels_sum_close_to_total() {
        let medium = CompoundMedium::new("H2O", 1000.0, 5000).unwrap();
        let total = medium.mu_m_cm2_g(PhotonChannel::Total, 1.0);
        let sum = medium.mu_m_cm2_g(PhotonChannel::Coherent, 1.0)
            + medium.mu_m_cm2_g(PhotonChannel::Incoherent, 1.0)
            + medium.mu_m_cm2_g(PhotonChannel::Photoelectric, 1.0)
            + medium.mu_m_cm2_g(PhotonChannel::PairProductionTotal, 1.0);
        assert!(
            (total - sum).abs() / total < 1e-2,
            "total={total} sum={sum}"
        );
    }
}
