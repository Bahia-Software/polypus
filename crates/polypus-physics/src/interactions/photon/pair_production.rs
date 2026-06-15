//! Electron–positron pair production cross-section and sampling.
//!
//! Above the `2·m_e·c² ≈ 1.022 MeV` threshold a photon may convert into an
//! electron–positron pair in the field of a nucleus. The atomic cross-section
//! uses the **Maximon (1968) low-energy Born-approximation formula**
//! (L. C. Maximon, *J. Res. NBS* **72B**, 79) — a closed form, no data tables:
//!
//! ```text
//! σ_pp(E,Z) = α · r_e² · Z(Z+1) · (2π/3) · ((k−2)/k)³ · Q(ρ),   k = E / m_e c²
//!
//! Q(ρ) = 1 + ρ/2 + (23/40)ρ² + (11/60)ρ³ + (29/960)ρ⁴
//! ρ    = (2k − 4) / (2 + k + 2·√(2k))
//! ```
//!
//! - The `((k−2)/k)³` factor makes the cross-section vanish exactly at the
//!   threshold (`k = 2`, `E = 1.022 MeV`) and rise progressively from there —
//!   unlike the high-energy asymptote `28/9·ln(2k) − 218/27`, which is
//!   unphysically negative below ~3.4 MeV.
//! - `Q(ρ)` is the Maximon polynomial in the kinematic variable `ρ` (which runs
//!   from `0` at threshold upward); it shapes the rise above threshold so the
//!   cross-section matches the Bethe–Heitler theory.
//! - The `Z(Z+1)` factor includes the nuclear field (`Z²`) plus an approximate
//!   electron-field (triplet, `∝ Z`) contribution.
//!
//! This low-energy form is accurate up to `k ≈ 40` (`E ≈ 20 MeV`), covering the
//! keV–MeV range of this crate; the high-energy Maximon branch (with Coulomb
//! and screening corrections) is a future enhancement. Returns `0` below
//! threshold.

use crate::constants::{
    CLASSICAL_ELECTRON_RADIUS_M as R_E, ELECTRON_MASS_MEV, FINE_STRUCTURE,
    PAIR_PRODUCTION_THRESHOLD_MEV,
};
use crate::particle::{FourMomentum, ParticleState, Position};
use std::f64::consts::PI;

/// Maximon kinematic variable `ρ = (2k − 4) / (2 + k + 2·√(2k))`.
///
/// `k = E / m_e c²` is the reduced photon energy. `ρ` is `0` at the pair
/// threshold (`k = 2`) and increases with energy.
fn rho(k: f64) -> f64 {
    (2.0 * k - 4.0) / (2.0 + k + 2.0 * (2.0 * k).sqrt())
}

/// Maximon shape polynomial `Q(ρ) = 1 + ρ/2 + (23/40)ρ² + (11/60)ρ³ + (29/960)ρ⁴`.
fn shape_polynomial(r: f64) -> f64 {
    1.0 + 0.5 * r
        + (23.0 / 40.0) * r * r
        + (11.0 / 60.0) * r * r * r
        + (29.0 / 960.0) * r * r * r * r
}

/// Atomic pair-production cross-section `σ_pp(E, Z)` (m² per atom).
///
/// `energy_mev` is the photon energy in MeV; `z` is the effective atomic
/// number. Returns exactly `0.0` for `E ≤ 1.022 MeV` and rises continuously
/// from the threshold (it is `0` at threshold because the `((k−2)/k)³` factor
/// vanishes there).
///
/// Formula: Maximon (1968) low-energy form — see the module documentation.
pub fn cross_section(energy_mev: f64, z: f64) -> f64 {
    if energy_mev <= PAIR_PRODUCTION_THRESHOLD_MEV {
        return 0.0;
    }
    let k = energy_mev / ELECTRON_MASS_MEV; // reduced energy E / m_e c²
    let threshold_factor = ((k - 2.0) / k).powi(3); // 0 at k = 2, → 1 at high k
    FINE_STRUCTURE
        * R_E
        * R_E
        * z
        * (z + 1.0)
        * (2.0 * PI / 3.0)
        * threshold_factor
        * shape_polynomial(rho(k))
}

/// Sample the pair-production outcome.
///
/// The incident photon is absorbed. Two secondary stubs (an electron and a
/// positron), each at `0.511 MeV`, are produced along the photon direction for
/// future tracking. The full kinetic-energy partition is a future enhancement.
///
/// Returns `(energy_deposit_mev, secondaries)`. Energy deposited locally is the
/// photon energy minus the rest-mass energy carried away by the pair
/// (`E − 2·m_e·c²`); the two stubs carry the rest-mass energy.
pub fn sample(state: &ParticleState) -> (f64, Vec<ParticleState>) {
    let e = state.momentum.energy_mev;
    let make = || ParticleState {
        position: Position(state.position.0),
        momentum: FourMomentum {
            energy_mev: ELECTRON_MASS_MEV,
            direction: state.momentum.direction,
        },
        alive: true,
    };
    let secondaries = vec![make(), make()];
    // Local deposit is the kinetic energy shared by the pair.
    let deposit = (e - PAIR_PRODUCTION_THRESHOLD_MEV).max(0.0);
    (deposit, secondaries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_below_threshold() {
        assert_eq!(cross_section(1.0, 82.0), 0.0);
    }

    #[test]
    fn zero_at_threshold() {
        // Exactly at 1.022 MeV the phase-space factor vanishes.
        assert_eq!(cross_section(PAIR_PRODUCTION_THRESHOLD_MEV, 82.0), 0.0);
    }

    #[test]
    fn positive_just_above_threshold() {
        // The key fix: the cross-section must be > 0 just above 1.022 MeV, not
        // zero until ~3.4 MeV as the high-energy asymptote wrongly gave.
        assert!(cross_section(1.1, 82.0) > 0.0);
        assert!(cross_section(1.5, 82.0) > 0.0);
        assert!(cross_section(2.0, 82.0) > 0.0);
    }

    #[test]
    fn increases_with_energy_above_threshold() {
        // Progressive growth from threshold upward.
        let s11 = cross_section(1.1, 82.0);
        let s15 = cross_section(1.5, 82.0);
        let s50 = cross_section(5.0, 82.0);
        assert!(s11 < s15, "{s11:.3e} !< {s15:.3e}");
        assert!(s15 < s50, "{s15:.3e} !< {s50:.3e}");
    }

    #[test]
    fn positive_at_high_energy_in_lead() {
        assert!(cross_section(10.0, 82.0) > 0.0);
    }

    #[test]
    fn comparable_to_compton_at_high_energy_in_water() {
        use super::super::compton;
        // In water at 10 MeV pair production should be within an order of
        // magnitude of Compton (they cross over near 20–30 MeV for low Z).
        let z = 7.42;
        let ratio = cross_section(10.0, z) / compton::cross_section(10.0, z);
        assert!((0.1..10.0).contains(&ratio), "pair/Compton = {ratio:.3}");
    }

    #[test]
    fn crossover_with_compton_is_physical() {
        use super::super::compton;
        // Maximon shape puts the pair/Compton crossover near ~5 MeV in lead
        // (high Z) and well above 15 MeV in water (low Z).
        let find_crossover = |z: f64| {
            let mut e = 1.1;
            while e < 100.0 {
                if cross_section(e, z) > compton::cross_section(e, z) {
                    return e;
                }
                e *= 1.02;
            }
            f64::INFINITY
        };
        let lead = find_crossover(82.0);
        let water = find_crossover(7.42);
        assert!((3.0..8.0).contains(&lead), "lead crossover {lead:.1} MeV");
        assert!(water > 15.0, "water crossover {water:.1} MeV");
    }
}
