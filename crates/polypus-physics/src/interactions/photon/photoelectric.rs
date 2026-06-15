//! Photoelectric absorption cross-section and sampling.
//!
//! The atomic photoelectric cross-section is modelled as a sum of two
//! power-law terms expressed in the reduced photon energy
//! `ε = E / (m_e c²)`:
//!
//! ```text
//! τ(E, Z) = K · Z^p · ( 1/ε^3.5 + 1/ε )     [m² per atom]
//! ```
//!
//! Why two terms instead of a single `E⁻³·⁵` power law? A single fixed
//! exponent is a straight line on a log-log plot, which is *not* how the real
//! photoelectric cross-section behaves: its local slope is steep
//! (≈ −3.5) at low energy where the ejected photoelectron is non-relativistic
//! (the classical `E⁻³·⁵` Born result), and flattens towards ≈ −1 at high
//! energy where it becomes relativistic (Pratt/Sauter–Gavrila behaviour). The
//! `1/ε^3.5` term reproduces the non-relativistic regime and the `1/ε` term
//! the relativistic one; their sum is concave on a log-log plot and crosses
//! over near `ε = 1` (`E = m_e c² = 0.511 MeV`), exactly where the
//! photoelectron turns relativistic. No external data tables are used — only
//! this closed-form parametrisation calibrated against the well-known process
//! ranking.
//!
//! `E` is in MeV and `Z` is the effective atomic number.
//!
//! ## Absorption edges
//!
//! On top of the smooth shape the model adds the **K absorption edge**: a
//! discontinuous jump where the photon energy crosses the K-shell binding
//! energy. Below the edge the K-shell electrons can no longer be ejected, so
//! the cross-section drops abruptly by the *jump ratio*. The edge energy is
//! obtained from **Moseley's law** (1913) — a closed-form formula, no data
//! tables — so this stays within the data-free design. Only the K edge is
//! modelled; the lower-energy L/M edges remain a future enhancement.

use crate::constants::ELECTRON_MASS_MEV;
use crate::particle::{FourMomentum, ParticleState, Position};

/// Atomic-number exponent `p` of the photoelectric cross-section.
///
/// The single-shell Born result scales as `Z⁵`; summed over all shells the
/// effective atomic exponent is closer to `4.0`–`4.6`. The standard
/// medical-physics compromise value `4.5` is used.
const Z_EXPONENT: f64 = 4.5;

/// Overall normalisation `K` (m² per atom).
///
/// Calibrated so that in water (`Z_eff = 7.42`) the photoelectric cross-section
/// crosses the Compton cross-section near `E ≈ 27 keV` — matching the
/// well-known photoelectric→Compton transition energy for low-`Z` media (the
/// crossover lands near `~410 keV` in lead, consistent with high-`Z` data). No
/// data tables are involved.
const K: f64 = 1.838e-36;

/// Rydberg energy (MeV). Constant of Moseley's law for shell-edge energies.
const RYDBERG_MEV: f64 = 13.605_693e-6;

/// K-edge jump ratio `r_K = τ(just above) / τ(just below)`.
///
/// When the photon energy drops below the K edge the two K-shell electrons stop
/// contributing, so the cross-section falls by this factor. `5.0` is
/// representative of the medium/high-`Z` elements whose K edge lands in the
/// keV–MeV window (iodine, tungsten, lead all have `r_K ≈ 4`–`6`).
const K_JUMP_RATIO: f64 = 5.0;

/// K-shell absorption-edge energy `E_K` (MeV) from **Moseley's law**.
///
/// `E_K ≈ Ry · (Z − σ_K)²` with K-shell screening constant `σ_K = 1`. For lead
/// (`Z = 82`) this gives `89.3 keV` against the measured `88.0 keV` (~1.5 %).
/// Pure formula — no data tables. `z` is the (effective) atomic number.
pub fn k_edge_energy_mev(z: f64) -> f64 {
    let z_eff = z - 1.0;
    RYDBERG_MEV * z_eff * z_eff
}

/// Multiplicative absorption-edge correction applied to the smooth shape.
///
/// Returns `1.0` at or above the K edge and `1/r_K` just below it, producing
/// the characteristic discontinuous K-edge jump. `energy_mev` is the photon
/// energy (MeV); `z` is the (effective) atomic number.
fn edge_factor(energy_mev: f64, z: f64) -> f64 {
    if energy_mev >= k_edge_energy_mev(z) {
        1.0
    } else {
        1.0 / K_JUMP_RATIO
    }
}

/// Atomic photoelectric cross-section `τ(E, Z)` (m² per atom).
///
/// `energy_mev` is the photon energy in MeV; `z` is the effective atomic
/// number. Returns `0.0` for non-positive energy.
///
/// Formula: `τ = K · Z^p · (1/ε^3.5 + 1/ε) · edge(E, Z)` with `ε = E / (m_e c²)`.
/// The smooth factor has a log-log slope running from ≈ −3.5 below ~100 keV to
/// ≈ −1 above a few MeV (passing through ≈ −2.25 at `ε = 1`, `E = m_e c² ≈
/// 0.511 MeV`, where the two terms are equal); the `edge` factor superimposes
/// the discontinuous K jump.
pub fn cross_section(energy_mev: f64, z: f64) -> f64 {
    if energy_mev <= 0.0 {
        return 0.0;
    }
    let epsilon = energy_mev / ELECTRON_MASS_MEV;
    let energy_shape = epsilon.powf(-3.5) + 1.0 / epsilon;
    K * z.powf(Z_EXPONENT) * energy_shape * edge_factor(energy_mev, z)
}

/// Sample the photoelectric absorption outcome.
///
/// The incident photon is fully absorbed. A single secondary photoelectron is
/// produced at kinetic energy `E_e = E − E_binding`. Shell-binding tables are a
/// future enhancement, so `E_binding ≈ 0` is used for the stub: the
/// photoelectron carries the full photon energy along the photon's direction.
///
/// Returns `(energy_deposit_mev, secondaries)`: all the photon energy is
/// deposited locally (the photoelectron is the carrier), and the secondary is
/// the photoelectron stub.
pub fn sample(state: &ParticleState) -> (f64, Vec<ParticleState>) {
    let e = state.momentum.energy_mev;
    // Photoelectron stub: full photon energy, same direction as the photon.
    let electron = ParticleState {
        position: Position(state.position.0),
        momentum: FourMomentum {
            energy_mev: e,
            direction: state.momentum.direction,
        },
        alive: true,
    };
    (e, vec![electron])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local log-log slope of the cross-section between two energies.
    fn loglog_slope(e1: f64, e2: f64, z: f64) -> f64 {
        (cross_section(e2, z).ln() - cross_section(e1, z).ln()) / (e2.ln() - e1.ln())
    }

    #[test]
    fn higher_z_has_larger_cross_section() {
        // 100 keV: lead (Z=82) vs aluminium (Z=13).
        assert!(cross_section(0.1, 82.0) > cross_section(0.1, 13.0));
    }

    #[test]
    fn cross_section_decreases_with_energy() {
        // Aluminium: 1 MeV vs 10 keV.
        assert!(cross_section(1.0, 13.0) < cross_section(0.01, 13.0));
    }

    #[test]
    fn non_positive_energy_yields_zero() {
        assert_eq!(cross_section(0.0, 13.0), 0.0);
    }

    #[test]
    fn slope_is_not_constant() {
        // A single power law would give the same slope everywhere (a straight
        // line on log-log). The two-term model must be steeper at low energy
        // than at high energy, i.e. the curve is concave.
        let low = loglog_slope(0.01, 0.02, 7.42); // ≈ −3
        let high = loglog_slope(2.0, 4.0, 7.42); // ≈ −1
        assert!(low < -2.5, "low-energy slope {low:.3} should be near −3");
        assert!(high > -1.5, "high-energy slope {high:.3} should be near −1");
        assert!(low < high - 1.0, "curve must flatten with energy (concave)");
    }

    #[test]
    fn slope_steepens_to_non_relativistic_limit() {
        // At ε = 1 (E = m_e c²) the two terms 1/ε^3.5 and 1/ε are equal, so the
        // local slope is the average of −3.5 and −1, i.e. −2.25.
        let slope = loglog_slope(0.40, 0.65, 7.42);
        assert!(
            (slope + 2.25).abs() < 0.3,
            "slope at ~0.5 MeV was {slope:.3}"
        );
    }

    #[test]
    fn crossover_with_compton_in_water_is_in_diagnostic_range() {
        use super::super::compton;
        // The photoelectric/Compton crossover in water should sit in the
        // 20–35 keV band (low-Z diagnostic range), not at ~60 keV as a fixed
        // E⁻³·⁵ law gave.
        let z = 7.42;
        assert!(cross_section(0.020, z) > compton::cross_section(0.020, z));
        assert!(cross_section(0.035, z) < compton::cross_section(0.035, z));
    }

    #[test]
    fn lead_k_edge_is_near_measured_value() {
        // Moseley estimate vs the measured lead K-edge of 88.0 keV.
        let e_k = k_edge_energy_mev(82.0);
        assert!((e_k - 0.088).abs() < 0.005, "lead K-edge {e_k:.4} MeV");
    }

    #[test]
    fn lead_cross_section_jumps_at_k_edge() {
        // Crossing the K edge from below produces a discontinuous jump upward
        // by roughly the jump ratio.
        let e_k = k_edge_energy_mev(82.0);
        let below = cross_section(e_k * 0.999, 82.0);
        let above = cross_section(e_k * 1.001, 82.0);
        assert!(above > below, "expected an upward jump at the K edge");
        let ratio = above / below;
        assert!(
            (ratio - K_JUMP_RATIO).abs() < 0.5,
            "K jump ratio was {ratio:.2}, expected ≈ {K_JUMP_RATIO}"
        );
    }

    #[test]
    fn water_has_no_edge_in_diagnostic_range() {
        // Water's K edge (oxygen, ~0.5 keV) lies far below the 10 keV–10 MeV
        // range, so the cross-section is smooth (factor 1.0) throughout.
        let z = 7.42;
        for &e in &[0.01, 0.03, 0.05, 0.1, 0.5, 1.0] {
            assert_eq!(edge_factor(e, z), 1.0);
        }
    }
}
