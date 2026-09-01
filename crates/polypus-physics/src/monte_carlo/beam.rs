//! Divergent point-source beam, collimated to a square field at the
//! phantom's entrance surface (z = 0).
//!
//! All lengths are in **metres**

use rand::Rng;

use crate::particle::{FourMomentum, ParticleState, Position};

pub struct DivergentBeam {
    /// Distance from the source to the surface (m).
    pub source_to_surface_distance_m: f64,
    /// Side length of the square field at the surface (m).
    pub field_side_m: f64,
    /// Photon energy (MeV).
    pub energy_mev: f64,
}

impl DivergentBeam {
    /// Sample one photon's initial state: a position uniformly distributed
    /// within the field on the surface, and the direction it would have if
    /// it had actually travelled there from the point source at
    /// `(0, 0, -source_to_surface_distance_m)`.
    pub fn sample(&self, rng: &mut impl Rng) -> ParticleState {
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
                energy_mev: self.energy_mev,
                direction: [dx / norm, dy / norm, dz / norm],
            },
            alive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn beam() -> DivergentBeam {
        DivergentBeam {
            source_to_surface_distance_m: 0.10,
            field_side_m: 0.10,
            energy_mev: 0.1,
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