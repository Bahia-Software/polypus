//! Bounded transport geometry for the Monte Carlo engine.
//!
//! The engine transports particles through a homogeneous [`Medium`] whose
//! *material* properties are uniform everywhere. Geometry adds an optional
//! *spatial boundary*: a finite region outside of which a particle is
//! considered to have **escaped**. This is what turns the historical
//! "infinite medium" model (every photon eventually absorbed) into a finite
//! phantom such as the 10 × 10 × 10 cm water cube used for depth-dose
//! validation.
//!
//! Dispatch is a plain `enum` (static dispatch) rather than a trait object,
//! matching the crate's preference for `enum`-based geometry: there is a small,
//! closed set of shapes and no need for `Box<dyn …>`.
//!
//! [`Medium`]: crate::medium::Medium

use crate::particle::Position;

/// Spatial boundary of the simulation volume.
///
/// All coordinates are in **metres**, matching the rest of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Geometry {
    /// Infinite medium: no boundary. A particle is never considered to have
    /// escaped, reproducing the engine's original behaviour. This is the
    /// [`Default`], so pre-existing runs (and the `run` / `run_with_spectrum`
    /// entry points) are unaffected.
    #[default]
    Unbounded,
    /// Axis-aligned bounding box `[min, max]` (metres), inclusive of its faces.
    ///
    /// For the depth-dose experiment use `min = [-0.05, -0.05, 0.0]`,
    /// `max = [0.05, 0.05, 0.10]` — a 10 cm water cube whose entrance face lies
    /// at `z = 0`, centred on `x = y = 0`.
    Box {
        /// Minimum corner `[x, y, z]` (m).
        min: [f64; 3],
        /// Maximum corner `[x, y, z]` (m).
        max: [f64; 3],
    },
}

impl Geometry {
    /// Return `true` if `p` lies inside (or on the boundary of) the volume.
    ///
    /// [`Geometry::Unbounded`] always returns `true`. [`Geometry::Box`] tests
    /// each axis against the inclusive range `[min, max]`, so a particle that
    /// starts exactly on the entrance face (`z = min_z`) counts as *inside* —
    /// the beam in the depth-dose experiment is launched from that face.
    pub fn contains(&self, p: Position) -> bool {
        match self {
            Geometry::Unbounded => true,
            Geometry::Box { min, max } => {
                let [x, y, z] = p.0;
                (min[0]..=max[0]).contains(&x)
                    && (min[1]..=max[1]).contains(&y)
                    && (min[2]..=max[2]).contains(&z)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water_cube() -> Geometry {
        Geometry::Box {
            min: [-0.05, -0.05, 0.0],
            max: [0.05, 0.05, 0.10],
        }
    }

    #[test]
    fn unbounded_contains_everything() {
        let g = Geometry::Unbounded;
        assert!(g.contains(Position([0.0, 0.0, 0.0])));
        assert!(g.contains(Position([1e9, -1e9, 1e9])));
    }

    #[test]
    fn default_is_unbounded() {
        assert_eq!(Geometry::default(), Geometry::Unbounded);
    }

    #[test]
    fn box_contains_interior_point() {
        assert!(water_cube().contains(Position([0.0, 0.0, 0.05])));
    }

    #[test]
    fn box_excludes_points_outside_each_face() {
        let g = water_cube();
        assert!(!g.contains(Position([0.0, 0.0, -0.001]))); // before entrance
        assert!(!g.contains(Position([0.0, 0.0, 0.1001]))); // past exit
        assert!(!g.contains(Position([0.0501, 0.0, 0.05]))); // +x face
        assert!(!g.contains(Position([0.0, -0.0501, 0.05]))); // -y face
    }

    #[test]
    fn box_includes_boundary_faces_and_corners() {
        let g = water_cube();
        // Entrance face (where the beam starts).
        assert!(g.contains(Position([0.0, 0.0, 0.0])));
        // Exit face.
        assert!(g.contains(Position([0.0, 0.0, 0.10])));
        // Min and max corners.
        assert!(g.contains(Position([-0.05, -0.05, 0.0])));
        assert!(g.contains(Position([0.05, 0.05, 0.10])));
    }
}
