//! Cross-section calculations and interaction sampling.
//!
//! The [`InteractionModel`] trait is the contract the Monte Carlo engine uses
//! to step a particle: it provides the total macroscopic cross-section (for
//! mean-free-path sampling) and samples the outcome of an individual
//! interaction.

pub mod photon;
pub mod proton;

use crate::medium::Medium;
use crate::particle::{Particle, ParticleState};
use rand::Rng;

/// Physical outcome of one interaction event.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionEvent {
    /// Particle fully absorbed (photoelectric effect, pair production, …).
    Absorbed {
        /// Energy deposited locally in the medium (MeV).
        energy_deposit_mev: f64,
        /// Secondary particles produced by the absorption (e.g. the
        /// photoelectron, or the e⁺e⁻ pair). The engine queues them for
        /// transport in a subsequent step.
        secondaries: Vec<ParticleState>,
    },
    /// Particle scattered into a new kinematic state.
    Scattered {
        /// Post-scatter particle state.
        new_state: ParticleState,
        /// Energy deposited locally in the medium (MeV).
        energy_deposit_mev: f64,
        /// Secondary particles produced (e.g. recoil electron). Each carries
        /// its own [`ParticleState`]; the engine queues them for transport in
        /// a subsequent step.
        secondaries: Vec<ParticleState>,
    },
}

/// Physics model that computes cross-sections and samples interactions for a
/// specific particle species in a medium.
///
/// # Adding a new interaction channel
///
/// Implement this trait for the `(Particle, Medium)` pair. The Monte Carlo
/// engine calls only these two methods — zero changes elsewhere.
pub trait InteractionModel: Send + Sync {
    /// The particle species this model applies to.
    type P: Particle;
    /// The medium type. May be unsized (`dyn Medium`) for polymorphic media.
    type M: Medium + ?Sized;

    /// Total macroscopic cross-section `Σ_tot` (m⁻¹) `= Σ_i σ_i(E,Z) · n`.
    /// The mean free path is `λ = 1 / Σ_tot`.
    ///
    /// # Errors
    ///
    /// [`PhysicsError`](crate::error::PhysicsError) when the cross-section is
    /// undefined at this energy / medium combination.
    fn total_cross_section_per_m(
        &self,
        particle: &Self::P,
        state: &ParticleState,
        medium: &Self::M,
    ) -> Result<f64, crate::error::PhysicsError>;

    /// Sample one interaction event.
    ///
    /// Steps the implementor performs:
    ///   1. Compute partial cross-sections for each process.
    ///   2. Select a process by roulette sampling.
    ///   3. Sample post-interaction kinematics for that process.
    ///   4. Return the resulting [`InteractionEvent`].
    ///
    /// # Errors
    ///
    /// [`PhysicsError`](crate::error::PhysicsError) when sampling fails (e.g.
    /// total cross-section is non-positive).
    fn sample_interaction(
        &self,
        particle: &Self::P,
        state: &ParticleState,
        medium: &Self::M,
        rng: &mut impl Rng,
    ) -> Result<InteractionEvent, crate::error::PhysicsError>;
}
