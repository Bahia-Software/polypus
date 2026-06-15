//! Proton interactions: Bethe–Bloch stopping power (compilable stub).
//!
//! Continuous slowing-down of protons is governed by the Bethe–Bloch formula.
//! The full implementation (with shell corrections and the Barkas term) is
//! deferred; this stub compiles and returns [`f64::NAN`] as a sentinel so that
//! callers can detect the not-yet-implemented state.

use crate::medium::Medium;

/// Mean stopping power `−dE/dx` for protons via the Bethe–Bloch formula
/// (MeV·m²/kg).
///
/// `kinetic_energy_mev` is the proton kinetic energy in MeV; `medium` is the
/// material being traversed. The full implementation with shell corrections
/// and the Barkas term is deferred — this stub returns [`f64::NAN`].
pub fn bethe_bloch_stopping_power_mev_m2_kg(kinetic_energy_mev: f64, medium: &impl Medium) -> f64 {
    let _ = (kinetic_energy_mev, medium);
    // TODO: implement full Bethe-Bloch.
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn stub_returns_nan() {
        let s = bethe_bloch_stopping_power_mev_m2_kg(100.0, &HomogeneousMedium::water());
        assert!(s.is_nan());
    }
}
