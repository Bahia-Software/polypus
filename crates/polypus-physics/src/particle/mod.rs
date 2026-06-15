//! Particle species definitions and kinematic state.
//!
//! The core abstraction is the [`Particle`] trait, which encodes the
//! invariant (species-level) properties of a particle. The mutable kinematic
//! state — position, four-momentum, alive flag — is carried externally in
//! [`ParticleState`], so a single `Particle` value can describe an arbitrary
//! number of in-flight particles of that species.

pub mod photon;
pub mod proton;

use crate::error::PhysicsError;

/// Four-momentum of a particle: total energy plus propagation direction.
///
/// The direction is a unit vector; the magnitude of the spatial momentum is
/// recovered from the energy and rest mass via [`momentum_mag_mev_c`].
///
/// [`momentum_mag_mev_c`]: FourMomentum::momentum_mag_mev_c
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourMomentum {
    /// Total energy (MeV). For massless particles this equals `|p|·c`.
    pub energy_mev: f64,
    /// Unit direction vector `[dx, dy, dz]`. Must satisfy `|d| = 1`.
    pub direction: [f64; 3],
}

impl FourMomentum {
    /// Relativistic momentum magnitude `|p| = sqrt(E² − m²c⁴) / c` (MeV/c).
    ///
    /// `rest_mass_mev` is the particle rest mass in MeV/c². For `E < m·c²`
    /// (energetically forbidden) the radicand is clamped to zero, yielding `0`.
    pub fn momentum_mag_mev_c(&self, rest_mass_mev: f64) -> f64 {
        let e2 = self.energy_mev * self.energy_mev;
        let m2 = rest_mass_mev * rest_mass_mev;
        (e2 - m2).max(0.0).sqrt()
    }

    /// Lorentz factor `γ = E / (m·c²)` (dimensionless).
    ///
    /// Returns [`f64::INFINITY`] for massless particles (`rest_mass_mev == 0`).
    pub fn lorentz_factor(&self, rest_mass_mev: f64) -> f64 {
        if rest_mass_mev == 0.0 {
            return f64::INFINITY;
        }
        self.energy_mev / rest_mass_mev
    }

    /// Velocity `β = |p|·c / E` (dimensionless, `≤ 1`).
    ///
    /// Returns `1.0` for massless particles. Returns `0.0` if the energy is
    /// non-positive (no physically meaningful velocity).
    pub fn beta(&self, rest_mass_mev: f64) -> f64 {
        if self.energy_mev <= 0.0 {
            return 0.0;
        }
        self.momentum_mag_mev_c(rest_mass_mev) / self.energy_mev
    }
}

/// 3-D position in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position(pub [f64; 3]);

/// Intrinsic spin quantum number `s` (e.g. `0.5` for electron/proton,
/// `1.0` for the photon).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spin(pub f64);

/// Electric charge in units of the elementary charge `e`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charge(pub f64);

/// Complete kinematic state of one particle during transport.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleState {
    /// Current position (m).
    pub position: Position,
    /// Current four-momentum (energy + direction).
    pub momentum: FourMomentum,
    /// `false` once the particle is absorbed or falls below the energy cutoff.
    pub alive: bool,
}

/// Core trait for every physical particle species.
///
/// Implementors encode invariant (species-level) properties. The mutable
/// kinematic state is carried externally in [`ParticleState`].
///
/// The trait is object-safe, so `Box<dyn Particle>` is usable for future
/// heterogeneous multi-particle transport.
///
/// # Adding a new particle species
///
/// 1. Create `src/particle/<name>.rs` and implement this trait.
/// 2. Create `src/interactions/<name>/mod.rs` and implement
///    [`InteractionModel`](crate::interactions::InteractionModel).
/// 3. Optionally create `src/hamiltonians/<name>_hamiltonian.rs`.
///
/// No changes to `monte_carlo`, `hamiltonians/mod.rs`, `medium`, or `lib.rs`
/// are required.
pub trait Particle: Send + Sync + std::fmt::Debug {
    /// PDG Monte Carlo particle ID (<https://pdg.lbl.gov/mcdata>).
    fn pdg_id(&self) -> i32;

    /// Rest mass in MeV/c². Zero for massless particles (photon).
    fn rest_mass_mev(&self) -> f64;

    /// Intrinsic spin quantum number.
    fn spin(&self) -> Spin;

    /// Electric charge in units of `e`.
    fn charge(&self) -> Charge;

    /// Human-readable species name, e.g. `"photon"`, `"proton"`.
    fn name(&self) -> &'static str;

    /// Map the particle's primary quantum degree of freedom (e.g. photon
    /// polarization, spin-½ for fermions) to a qubit amplitude pair `[α, β]`
    /// such that `|ψ⟩ = α|0⟩ + β|1⟩`.
    ///
    /// Returns `None` when no natural single-qubit encoding exists for this
    /// particle / state combination at the current abstraction level.
    fn qubit_encoding(&self, state: &ParticleState) -> Option<[f64; 2]>;

    /// Assert that `state` is physically self-consistent for this species
    /// (e.g. massless particles must have positive energy and a unit
    /// direction vector).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::NonPositiveEnergy`] or [`PhysicsError::SimulationError`]
    /// when the state violates a species invariant.
    fn validate_state(&self, state: &ParticleState) -> Result<(), PhysicsError>;
}

pub use photon::Photon;
pub use proton::Proton;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::ELECTRON_MASS_MEV;

    #[test]
    fn massless_lorentz_factor_is_infinite() {
        let p = FourMomentum {
            energy_mev: 1.0,
            direction: [0.0, 0.0, 1.0],
        };
        assert!(p.lorentz_factor(0.0).is_infinite());
        assert_eq!(p.beta(0.0), 1.0);
    }

    #[test]
    fn massive_beta_below_one() {
        // 1 MeV total energy electron.
        let p = FourMomentum {
            energy_mev: 1.0,
            direction: [0.0, 0.0, 1.0],
        };
        let beta = p.beta(ELECTRON_MASS_MEV);
        assert!(beta > 0.0 && beta < 1.0);
    }

    #[test]
    fn momentum_clamped_below_rest_mass() {
        let p = FourMomentum {
            energy_mev: 0.1,
            direction: [0.0, 0.0, 1.0],
        };
        assert_eq!(p.momentum_mag_mev_c(ELECTRON_MASS_MEV), 0.0);
    }

    #[test]
    fn particle_trait_is_object_safe() {
        let particles: Vec<Box<dyn Particle>> = vec![Box::new(Photon), Box::new(Proton)];
        assert_eq!(particles[0].pdg_id(), 22);
        assert_eq!(particles[1].pdg_id(), 2212);
    }
}
