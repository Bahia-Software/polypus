//! Compton (incoherent) scattering: Klein–Nishina cross-section and sampling.
//!
//! The differential cross-section per electron is
//!
//! ```text
//! dσ/dΩ = (r_e²/2) · (E'/E)² · (E/E' + E'/E − sin²θ)
//! ```
//!
//! with the scattered-photon energy
//!
//! ```text
//! E' = E / [1 + α(1 − cosθ)],     α = E / (m_e c²).
//! ```
//!
//! The total cross-section per electron is the analytic integral
//!
//! ```text
//! σ_C = 2π·r_e² · {
//!     (1+α)/α³ · [2α(1+α)/(1+2α) − ln(1+2α)]
//!     + ln(1+2α)/(2α)
//!     − (1+3α)/(1+2α)²
//! }.
//! ```

use crate::constants::{CLASSICAL_ELECTRON_RADIUS_M as R_E, ELECTRON_MASS_MEV};
use crate::particle::{FourMomentum, ParticleState, Position};
use rand::Rng;
use std::f64::consts::PI;

/// Scattered-photon energy `E'` (MeV) for incident energy `energy_mev` and
/// scattering angle cosine `cos_theta`.
///
/// `E' = E / [1 + α(1 − cosθ)]` with `α = E / (m_e c²)`.
pub fn scattered_energy(energy_mev: f64, cos_theta: f64) -> f64 {
    let alpha = energy_mev / ELECTRON_MASS_MEV;
    energy_mev / (1.0 + alpha * (1.0 - cos_theta))
}

/// Klein–Nishina differential cross-section `dσ/dΩ` per electron
/// (m²/steradian) at scattering-angle cosine `cos_theta`.
///
/// `energy_mev` is the incident photon energy in MeV.
pub fn differential_cross_section(energy_mev: f64, cos_theta: f64) -> f64 {
    let e_prime = scattered_energy(energy_mev, cos_theta);
    let ratio = e_prime / energy_mev;
    let sin2 = 1.0 - cos_theta * cos_theta;
    0.5 * R_E * R_E * ratio * ratio * (1.0 / ratio + ratio - sin2)
}

/// Total Klein–Nishina cross-section per electron `σ_C` (m²).
///
/// `energy_mev` is the incident photon energy in MeV. As `α → 0` this tends to
/// the Thomson cross-section `(8π/3)·r_e²`. Returns `0.0` for non-positive
/// energy.
pub fn cross_section_per_electron(energy_mev: f64) -> f64 {
    if energy_mev <= 0.0 {
        return 0.0;
    }
    let alpha = energy_mev / ELECTRON_MASS_MEV;
    let one_plus_2a = 1.0 + 2.0 * alpha;
    let term1 = (1.0 + alpha) / (alpha * alpha * alpha)
        * (2.0 * alpha * (1.0 + alpha) / one_plus_2a - one_plus_2a.ln());
    let term2 = one_plus_2a.ln() / (2.0 * alpha);
    let term3 = (1.0 + 3.0 * alpha) / (one_plus_2a * one_plus_2a);
    2.0 * PI * R_E * R_E * (term1 + term2 - term3)
}

/// Atomic Compton cross-section `σ_C(E, Z) = Z · σ_C^{electron}(E)` (m² per
/// atom).
///
/// Independent-electron approximation: the atomic cross-section is `Z` times
/// the per-electron Klein–Nishina value. `z` is the effective atomic number.
pub fn cross_section(energy_mev: f64, z: f64) -> f64 {
    z * cross_section_per_electron(energy_mev)
}

/// Sample a scattering-angle cosine via the Kahn rejection method.
///
/// Candidate `cosθ` values are drawn uniformly on `[−1, 1]` and accepted with
/// probability `(dσ/dΩ) / (dσ/dΩ)_max`. The maximum of `dσ/dΩ` over
/// `cosθ ∈ [−1, 1]` is at the forward direction (`cosθ = 1`).
fn sample_cos_theta(energy_mev: f64, rng: &mut impl Rng) -> f64 {
    // dσ/dΩ is maximal in the forward direction (no energy loss).
    let d_max = differential_cross_section(energy_mev, 1.0);
    loop {
        let cos_theta: f64 = rng.gen_range(-1.0..=1.0);
        let p = differential_cross_section(energy_mev, cos_theta) / d_max;
        let u: f64 = rng.gen_range(0.0..1.0);
        if u <= p {
            return cos_theta;
        }
    }
}

/// Build an orthonormal basis `(u, v)` perpendicular to the unit vector `w`.
///
/// Used to rotate a direction by a polar/azimuthal `(θ, φ)` pair while handling
/// the polar-axis singularity (when `w` is close to `±z`, a different reference
/// axis is chosen).
fn perpendicular_basis(w: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [wx, wy, wz] = w;
    // Choose a reference axis not parallel to w.
    let reference = if wz.abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    // u = normalize(w × reference)
    let ux = wy * reference[2] - wz * reference[1];
    let uy = wz * reference[0] - wx * reference[2];
    let uz = wx * reference[1] - wy * reference[0];
    let u_norm = (ux * ux + uy * uy + uz * uz).sqrt();
    let u = [ux / u_norm, uy / u_norm, uz / u_norm];
    // v = w × u (already unit length since w and u are orthonormal).
    let v = [
        wy * u[2] - wz * u[1],
        wz * u[0] - wx * u[2],
        wx * u[1] - wy * u[0],
    ];
    (u, v)
}

/// Rotate unit direction `dir` by polar angle `theta` (about a transverse axis)
/// and azimuth `phi`, returning a new unit vector.
///
/// The new direction is `cosθ·w + sinθ(cosφ·u + sinφ·v)`, where `(u, v, w)` is
/// an orthonormal frame with `w = dir`.
pub fn rotate_direction(dir: [f64; 3], cos_theta: f64, phi: f64) -> [f64; 3] {
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let (u, v) = perpendicular_basis(dir);
    let (sin_phi, cos_phi) = phi.sin_cos();
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = cos_theta * dir[i] + sin_theta * (cos_phi * u[i] + sin_phi * v[i]);
    }
    // Renormalize defensively against accumulated floating-point error.
    let norm = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
    [out[0] / norm, out[1] / norm, out[2] / norm]
}

/// Sample one Compton scattering event.
///
/// Returns `(new_state, energy_deposit_mev, secondaries)`:
/// - `new_state`: the scattered photon at energy `E'` and rotated direction.
/// - `energy_deposit_mev = E − E'`.
/// - `secondaries`: one recoil-electron stub at kinetic energy `E − E'`,
///   travelling along the original photon direction (full kinematics are a
///   future enhancement).
pub fn sample(
    state: &ParticleState,
    rng: &mut impl Rng,
) -> (ParticleState, f64, Vec<ParticleState>) {
    let e = state.momentum.energy_mev;
    let cos_theta = sample_cos_theta(e, rng);
    let phi: f64 = rng.gen_range(0.0..(2.0 * PI));
    let e_prime = scattered_energy(e, cos_theta);
    let new_dir = rotate_direction(state.momentum.direction, cos_theta, phi);

    let new_state = ParticleState {
        position: Position(state.position.0),
        momentum: FourMomentum {
            energy_mev: e_prime,
            direction: new_dir,
        },
        alive: true,
    };

    let deposit = e - e_prime;
    let recoil_electron = ParticleState {
        position: Position(state.position.0),
        momentum: FourMomentum {
            energy_mev: deposit,
            direction: state.momentum.direction,
        },
        alive: true,
    };

    (new_state, deposit, vec![recoil_electron])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn thomson_limit_at_low_energy() {
        // As α → 0, σ_C → σ_Thomson = (8π/3)·r_e² ≈ 6.65e-29 m².
        let thomson = 8.0 * PI / 3.0 * R_E * R_E;
        let sigma = cross_section_per_electron(1e-4); // 100 eV → α ≈ 2e-4
        assert!((sigma - thomson).abs() / thomson < 1e-3);
    }

    #[test]
    fn cross_section_at_alpha_one() {
        // E = 0.511 MeV (α = 1), hydrogen Z = 1. The analytic Klein–Nishina
        // total at α = 1 is ≈ 2.87e-29 m² per electron (0.287 barn).
        let sigma = cross_section(ELECTRON_MASS_MEV, 1.0);
        assert!((sigma - 2.87e-29).abs() / 2.87e-29 < 0.05);
    }

    #[test]
    fn scattered_energy_never_exceeds_incident() {
        let mut rng = StdRng::seed_from_u64(7);
        let e = 1.0;
        for _ in 0..10_000 {
            let cos_theta = sample_cos_theta(e, &mut rng);
            let e_prime = scattered_energy(e, cos_theta);
            assert!(e_prime <= e + 1e-12);
        }
    }

    #[test]
    fn forward_scatter_has_no_energy_loss() {
        let e = 1.0;
        assert!((scattered_energy(e, 1.0) - e).abs() < 1e-12);
    }

    #[test]
    fn rotated_direction_is_unit_vector() {
        let mut rng = StdRng::seed_from_u64(11);
        let dir = [0.0, 0.0, 1.0];
        for _ in 0..1000 {
            let cos_theta = sample_cos_theta(1.0, &mut rng);
            let phi: f64 = rng.gen_range(0.0..(2.0 * PI));
            let r = rotate_direction(dir, cos_theta, phi);
            let mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!((mag - 1.0).abs() < 1e-9);
        }
    }
}
