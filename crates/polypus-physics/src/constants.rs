//! Physical constants used throughout `polypus-physics`.
//!
//! Every constant is a `pub const f64`. Each entry documents its physical
//! meaning and the unit convention in which it is expressed. Values are CODATA
//! 2018 recommended values where applicable.
//!
//! Two unit systems appear in this crate:
//! - **SI**: metres, seconds, kilograms, coulombs.
//! - **Natural / particle-physics units**: energies and masses in MeV (mass as
//!   MeV/c²), often with `ħ = c = 1` in the quantum-Hamiltonian layer.

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT_M_S: f64 = 2.997_924_58e8;

/// Reduced Planck constant ħ (eV·s).
pub const HBAR_EV_S: f64 = 6.582_119_569e-16;

/// Planck constant h (eV·s).
pub const PLANCK_EV_S: f64 = 4.135_667_696e-15;

/// Electron rest mass (MeV/c²).
pub const ELECTRON_MASS_MEV: f64 = 0.510_998_950;

/// Proton rest mass (MeV/c²).
pub const PROTON_MASS_MEV: f64 = 938.272_088_16;

/// Elementary charge (C).
pub const ELECTRON_CHARGE_C: f64 = 1.602_176_634e-19;

/// Energy conversion factor from megaelectronvolts to joules (J/MeV).
///
/// One electronvolt is `e` joules, so `1 MeV = 1e6 · e = 1.602_176_634e-13 J`.
/// Used to convert a voxel's deposited energy (MeV) into an absorbed dose in
/// gray (J/kg).
pub const MEV_TO_JOULE: f64 = 1.602_176_634e-13;

/// Classical electron radius r_e (m).
pub const CLASSICAL_ELECTRON_RADIUS_M: f64 = 2.817_940_3e-15;

/// Avogadro constant N_A (mol⁻¹).
pub const AVOGADRO: f64 = 6.022_140_76e23;

/// Fine-structure constant α (dimensionless).
pub const FINE_STRUCTURE: f64 = 1.0 / 137.035_999_084;

/// Pair-production energy threshold (MeV): 2 · m_e · c².
pub const PAIR_PRODUCTION_THRESHOLD_MEV: f64 = 2.0 * ELECTRON_MASS_MEV;

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative difference between `a` and `b`.
    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn electron_mass_matches_codata() {
        assert!(rel_diff(ELECTRON_MASS_MEV, 0.510_999) < 1e-5);
    }

    #[test]
    fn pair_production_threshold_is_two_electron_masses() {
        assert!(rel_diff(PAIR_PRODUCTION_THRESHOLD_MEV, 1.022) < 1e-3);
    }

    #[test]
    fn fine_structure_matches_reference() {
        assert!(rel_diff(FINE_STRUCTURE, 1.0 / 137.036) < 1e-5);
    }

    #[test]
    fn mev_to_joule_matches_elementary_charge() {
        // 1 MeV = 1e6 eV = 1e6 · e joules.
        assert!(rel_diff(MEV_TO_JOULE, 1e6 * ELECTRON_CHARGE_C) < 1e-9);
    }
}
