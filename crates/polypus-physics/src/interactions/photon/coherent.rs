//! Coherent (Rayleigh) scattering: cross-section and angular sampling.
//!
//! Rayleigh scattering is elastic — the photon changes direction but not
//! energy — so there is no `scattered_energy` analogue here, unlike Compton.
//!
//! The differential cross-section is
//!
//! ```text
//! dσ/dΩ = (r_e²/2) · (1 + cos²θ) · F(x, Z)²
//! ```
//!
//! where `F(x, Z)` is the atomic form factor. This module uses the crude
//! approximation `F(x, Z) = 1` (see Podgorsak, p. 331, for the proper
//! form-factor treatment) — a known, deliberate simplification, not a full
//! atomic Rayleigh model. Because of it, the cross-section below does not
//! depend on photon energy at all.

use super::compton::rotate_direction;
use crate::constants::CLASSICAL_ELECTRON_RADIUS_M as R_E;
use crate::particle::{FourMomentum, ParticleState, Position};
use rand::Rng;
use std::f64::consts::PI;

/// Differential cross-section `dσ/dΩ` (m²/steradian) at scattering-angle
/// cosine `cos_theta`, under the crude form-factor approximation `F = 1`.
pub fn differential_cross_section(cos_theta: f64) -> f64 {
    0.5 * R_E * R_E * (1.0 + cos_theta * cos_theta)
}

/// Sample a scattering-angle cosine via rejection sampling.
///
/// The maximum of `dσ/dΩ` over `cosθ ∈ [−1, 1]` is at either endpoint
/// (`cosθ = ±1`), where `1 + cos²θ` is maximal.
fn sample_cos_theta(rng: &mut impl Rng) -> f64 {
    let d_max = differential_cross_section(1.0);
    loop {
        let cos_theta: f64 = rng.gen_range(-1.0..=1.0); // Random number in [-1, 1]
        let p = differential_cross_section(cos_theta) / d_max;
        let u: f64 = rng.gen_range(0.0..1.0);
        if u <= p {
            return cos_theta;
        }
    }
}

/// Sample one Rayleigh (coherent) scattering event.
///
/// Returns `(new_state, energy_deposit_mev, secondaries)`, matching
/// [`super::compton::sample`]'s shape. Since Rayleigh scattering is
/// elastic, `energy_deposit_mev` is always `0.0` and `secondaries` is
/// always empty — there is no recoil electron, unlike Compton.
pub fn sample(
    state: &ParticleState,
    rng: &mut impl Rng,
) -> (ParticleState, f64, Vec<ParticleState>) {
    let cos_theta = sample_cos_theta(rng);
    let phi: f64 = rng.gen_range(0.0..(2.0 * PI));
    let new_dir = rotate_direction(state.momentum.direction, cos_theta, phi);

    let new_state = ParticleState {
        position: Position(state.position.0),
        momentum: FourMomentum {
            energy_mev: state.momentum.energy_mev,
            direction: new_dir,
        },
        alive: true,
    };

    (new_state, 0.0, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn cross_section_is_maximal_at_forward_and_backward() {
        let forward = differential_cross_section(1.0);
        let backward = differential_cross_section(-1.0);
        let sideways = differential_cross_section(0.0);
        assert_eq!(forward, backward);
        assert!(forward > sideways);
    }

    #[test]
    fn scattering_preserves_energy() {
        let mut rng = StdRng::seed_from_u64(3);
        let state = ParticleState {
            position: Position([0.0, 0.0, 0.0]),
            momentum: FourMomentum {
                energy_mev: 0.1,
                direction: [0.0, 0.0, 1.0],
            },
            alive: true,
        };
        for _ in 0..1000 {
            let (scattered, deposit, secondaries) = sample(&state, &mut rng);
            assert_eq!(scattered.momentum.energy_mev, 0.1);
            assert_eq!(deposit, 0.0);
            assert!(secondaries.is_empty());
        }
    }

    #[test]
    fn scattered_direction_is_unit_vector() {
        let mut rng = StdRng::seed_from_u64(5);
        let state = ParticleState {
            position: Position([0.0, 0.0, 0.0]),
            momentum: FourMomentum {
                energy_mev: 0.1,
                direction: [0.0, 0.0, 1.0],
            },
            alive: true,
        };
        for _ in 0..1000 {
            let (scattered, _deposit, _secondaries) = sample(&state, &mut rng);
            let [x, y, z] = scattered.momentum.direction;
            let mag = (x * x + y * y + z * z).sqrt();
            assert!((mag - 1.0).abs() < 1e-9);
        }
    }
}
