//! Step-length sampling and position advancement helpers.

use rand::Rng;

/// Sample a step length from the exponential distribution
/// `d = −λ · ln(ξ)`, with `ξ ~ Uniform(0, 1)` and `λ = 1/Σ_tot` the mean free
/// path (m).
///
/// `mean_free_path_m` is `λ`. `ξ` is drawn from `[ε, 1)` (excluding 0) so that
/// `ln(ξ)` is finite.
pub fn sample_step_length(mean_free_path_m: f64, rng: &mut impl Rng) -> f64 {
    let xi: f64 = rng.gen_range(f64::EPSILON..1.0);
    -mean_free_path_m * xi.ln()
}

/// Advance `state.position` by `step_m` metres along
/// `state.momentum.direction`.
pub fn advance(state: &mut crate::particle::ParticleState, step_m: f64) {
    let [dx, dy, dz] = state.momentum.direction;
    let [x, y, z] = state.position.0;
    state.position = crate::particle::Position([x + dx * step_m, y + dy * step_m, z + dz * step_m]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::photon::Photon;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn mean_step_length_matches_lambda() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_step_length(1.0, &mut rng);
        }
        let mean = sum / n as f64;
        assert!((mean - 1.0).abs() < 0.05);
    }

    #[test]
    fn advance_moves_along_direction() {
        let mut state = Photon::state_along_z(1.0);
        advance(&mut state, 2.5);
        assert_eq!(state.position.0, [0.0, 0.0, 2.5]);
    }
}
