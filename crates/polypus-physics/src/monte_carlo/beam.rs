//! Divergent point-source beam, collimated to a square field at the
//! phantom's entrance surface (z = 0).
//!
//! All lengths are in **metres**

use rand::Rng;

use crate::particle::{FourMomentum, ParticleState, Position};

use super::spectrum::EnergySpectrum;

#[derive(Debug)]
pub struct DivergentBeam {
    /// Distance from the source to the surface (m).
    pub source_to_surface_distance_m: f64,
    /// Side length of the square field at the surface (m).
    pub field_side_m: f64,
    /// Photon energy.
    pub energy_source: Box<dyn EnergySpectrum>,
}

use rand::RngCore;

/// A source of a primary particle's full initial state (position,
/// direction, and energy), sampled once per Monte Carlo history.
///
/// Object-safe: `Box<dyn BeamSource>` and `&dyn BeamSource` are usable — the
/// same contract already used by [`super::spectrum::EnergySpectrum`].
pub trait BeamSource: Send + Sync + std::fmt::Debug {
    /// Sample one primary's full initial state.
    fn sample_state(&self, rng: &mut dyn RngCore) -> ParticleState;
}

impl DivergentBeam {
    /// Sample one photon's initial state: a position uniformly distributed
    /// within the field on the surface, and the direction it would have if
    /// it had actually travelled there from the point source at
    /// `(0, 0, -source_to_surface_distance_m)`.
    pub fn sample(&self, rng: &mut dyn RngCore) -> ParticleState {
        let half_field = self.field_side_m / 2.0;
        let x0 = rng.gen_range(-half_field..=half_field);
        let y0 = rng.gen_range(-half_field..=half_field);
        let z0 = 0.0;

        let dx = x0;
        let dy = y0;
        let dz = self.source_to_surface_distance_m;
        let norm = (dx * dx + dy * dy + dz * dz).sqrt();

        ParticleState {
            position: Position([x0, y0, z0]),
            momentum: FourMomentum {
                energy_mev: self.energy_source.sample_energy_mev(rng),
                direction: [dx / norm, dy / norm, dz / norm],
            },
            alive: true,
        }
    }
}

impl BeamSource for DivergentBeam {
    fn sample_state(&self, rng: &mut dyn RngCore) -> ParticleState {
        self.sample(rng)
    }
}

/// A broad, parallel beam: primaries start at a fixed depth `z0`, with
/// position uniformly distributed over a rectangle in the transverse
/// plane, and a fixed direction (0, 0, 1) — no divergence.
///
/// Mirrors Geant4 example B1's default `PrimaryGeneratorAction`.
#[derive(Debug)]
pub struct ParallelBeam {
    pub half_width_x_m: f64,
    pub half_width_y_m: f64,
    pub z0_m: f64,
    pub energy_source: Box<dyn EnergySpectrum>,
}

impl ParallelBeam {
    pub fn sample(&self, rng: &mut dyn RngCore) -> ParticleState {
        let x0 = rng.gen_range(-self.half_width_x_m..=self.half_width_x_m);
        let y0 = rng.gen_range(-self.half_width_y_m..=self.half_width_y_m);
        ParticleState {
            position: Position([x0, y0, self.z0_m]),
            momentum: FourMomentum {
                energy_mev: self.energy_source.sample_energy_mev(rng),
                direction: [0.0, 0.0, 1.0],
            },
            alive: true,
        }
    }
}

impl BeamSource for ParallelBeam {
    fn sample_state(&self, rng: &mut dyn RngCore) -> ParticleState {
        self.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::spectrum::Monoenergetic;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn beam() -> DivergentBeam {
        DivergentBeam {
            source_to_surface_distance_m: 0.10,
            field_side_m: 0.10,
            energy_source: Box::new(Monoenergetic::new(0.1).unwrap()),
        }
    }

    #[test]
    fn position_lands_on_surface_within_field() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let state = beam().sample(&mut rng);
            let [x, y, z] = state.position.0;
            assert_eq!(z, 0.0);
            assert!(x.abs() <= 0.05);
            assert!(y.abs() <= 0.05);
        }
    }

    #[test]
    fn direction_is_unit_vector() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..1000 {
            let state = beam().sample(&mut rng);
            let [dx, dy, dz] = state.momentum.direction;
            let mag = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((mag - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn direction_points_forward_into_the_phantom() {
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..1000 {
            let state = beam().sample(&mut rng);
            assert!(state.momentum.direction[2] > 0.0);
        }
    }
}
