//! Classical Monte Carlo transport engine.
//!
//! The engine is generic over the particle species ([`Particle`]) and the
//! concrete medium ([`Medium`]). Interaction physics is supplied by an
//! [`InteractionModel`] whose associated medium type is `dyn Medium`, so the
//! same model works for any material. Adding a new particle species requires
//! zero changes to this file.

pub mod history;
pub mod sampler;
pub mod spectrum;

pub use history::{ParticleHistory, TrackPoint};
pub use spectrum::{
    EnergySpectrum, KramersSpectrum, Monoenergetic, TabulatedSpectrum, UniformSpectrum,
};

use crate::error::PhysicsError;
use crate::interactions::{InteractionEvent, InteractionModel};
use crate::medium::Medium;
use crate::particle::{FourMomentum, Particle, ParticleState, Position};
use rand::Rng;

/// Configuration for a Monte Carlo simulation run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Number of primary particle histories to simulate.
    pub n_histories: usize,
    /// Safety cutoff: maximum steps per history before forced termination.
    pub max_steps: usize,
    /// Particle is killed when its energy falls below this value (MeV).
    pub energy_cutoff_mev: f64,
    /// RNG seed for reproducible runs.
    pub seed: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n_histories: 1_000,
            max_steps: 10_000,
            energy_cutoff_mev: 0.001, // 1 keV
            seed: 0,
        }
    }
}

/// Aggregated output of a completed simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// One [`ParticleHistory`] per primary particle.
    pub histories: Vec<ParticleHistory>,
    /// Mean energy deposited per primary history (MeV).
    pub mean_deposit_mev: f64,
    /// Variance of energy deposit across histories (MeV²).
    pub variance_deposit_mev2: f64,
}

/// Generic Monte Carlo transport engine.
///
/// Fully generic over particle, medium, and interaction model. Adding a new
/// particle species (e.g. electron, neutron) requires zero changes here, as
/// long as its [`InteractionModel`] uses `type M = dyn Medium`.
pub struct MonteCarloEngine<P, M, I>
where
    P: Particle,
    M: Medium + 'static,
    I: InteractionModel<P = P, M = dyn Medium>,
{
    /// The particle species being transported.
    pub particle: P,
    /// The (homogeneous) medium filling the simulation volume.
    pub medium: M,
    /// The interaction physics model.
    pub model: I,
    /// Run configuration.
    pub config: RunConfig,
}

impl<P, M, I> MonteCarloEngine<P, M, I>
where
    P: Particle,
    M: Medium + 'static,
    I: InteractionModel<P = P, M = dyn Medium>,
{
    /// Construct a new engine from its components.
    pub fn new(particle: P, medium: M, model: I, config: RunConfig) -> Self {
        MonteCarloEngine {
            particle,
            medium,
            model,
            config,
        }
    }

    /// Transport a single particle to absorption, energy cutoff, or
    /// `max_steps`, recording its track and total energy deposit.
    ///
    /// Secondaries produced along the way are returned in `pending` for the
    /// caller to enqueue.
    ///
    /// Transport a single primary particle to absorption, energy cutoff, or
    /// `max_steps`, recording its track and total energy deposit.
    ///
    /// Secondary particles produced along the way (recoil electrons,
    /// photoelectrons, e⁺e⁻ pairs) are a **different species** than the
    /// transported `particle`, so this single-species engine does not
    /// re-transport them. Instead each is attached as an informational *leaf*
    /// [`ParticleHistory`] (a single track point at its creation site). Their
    /// kinetic energy is already counted in the parent interaction's
    /// `energy_deposit_mev` (continuous-slowing-down approximation), so the
    /// leaves carry zero additional deposit to avoid double counting. Full
    /// multi-species transport is a future enhancement.
    ///
    /// # Errors
    ///
    /// Propagates any [`PhysicsError`] from cross-section evaluation or
    /// interaction sampling.
    fn transport_one(
        &self,
        mut state: ParticleState,
        rng: &mut impl Rng,
    ) -> Result<ParticleHistory, PhysicsError> {
        let medium: &dyn Medium = &self.medium;
        let mut history = ParticleHistory::new(self.particle.pdg_id());

        // Record the starting point.
        history.track.push(TrackPoint {
            position: state.position,
            energy_mev: state.momentum.energy_mev,
            step_length_m: 0.0,
        });

        let mut steps = 0;
        while state.alive && steps < self.config.max_steps {
            steps += 1;

            // a. Total macroscopic cross-section → mean free path.
            let sigma_tot = self
                .model
                .total_cross_section_per_m(&self.particle, &state, medium)?;
            if sigma_tot <= 0.0 {
                // No interaction possible: particle escapes.
                state.alive = false;
                break;
            }
            let mean_free_path = 1.0 / sigma_tot;

            // b–c. Sample step length and advance position.
            let step = sampler::sample_step_length(mean_free_path, rng);
            sampler::advance(&mut state, step);

            // d. Record the track point.
            history.track.push(TrackPoint {
                position: state.position,
                energy_mev: state.momentum.energy_mev,
                step_length_m: step,
            });

            // e. Sample the interaction.
            let event = self
                .model
                .sample_interaction(&self.particle, &state, medium, rng)?;

            // f. Apply the outcome.
            match event {
                InteractionEvent::Absorbed {
                    energy_deposit_mev,
                    secondaries,
                } => {
                    history.total_deposit_mev += energy_deposit_mev;
                    record_secondaries(&mut history, secondaries);
                    state.alive = false;
                }
                InteractionEvent::Scattered {
                    new_state,
                    energy_deposit_mev,
                    secondaries,
                } => {
                    history.total_deposit_mev += energy_deposit_mev;
                    record_secondaries(&mut history, secondaries);
                    state = new_state;
                }
            }

            // g. Kill below the energy cutoff.
            if state.momentum.energy_mev < self.config.energy_cutoff_mev {
                state.alive = false;
            }
        }

        Ok(history)
    }

    /// Run all primary histories from a fixed `initial_state` and return
    /// aggregated statistics.
    ///
    /// Per-history algorithm:
    ///   1. Clone `initial_state` for each primary.
    ///   2. Transport the primary to absorption / cutoff / `max_steps`,
    ///      attaching any secondaries as informational leaf histories.
    ///   3. Compute the mean and variance of `total_deposit_mev` across all
    ///      primary histories.
    ///
    /// For a polyenergetic source, use
    /// [`run_with_spectrum`](Self::run_with_spectrum) instead.
    ///
    /// # Errors
    ///
    /// Propagates any [`PhysicsError`] raised during transport.
    pub fn run(
        &self,
        initial_state: ParticleState,
        rng: &mut impl Rng,
    ) -> Result<SimulationResult, PhysicsError> {
        let mut histories = Vec::with_capacity(self.config.n_histories);
        for _ in 0..self.config.n_histories {
            let primary = self.transport_one(initial_state.clone(), rng)?;
            histories.push(primary);
        }
        Ok(finalize(histories))
    }

    /// Run all primary histories, drawing each primary's initial energy from an
    /// [`EnergySpectrum`], and return aggregated statistics.
    ///
    /// Every primary starts at `origin` travelling along `direction` (which is
    /// normalized internally); only the energy varies, sampled per history from
    /// `spectrum`. This is the entry point for polyenergetic sources such as an
    /// X-ray tube spectrum.
    ///
    /// # Errors
    ///
    /// - [`PhysicsError::SimulationError`] if `direction` is a (near-)zero
    ///   vector.
    /// - Any error raised by the particle's
    ///   [`validate_state`](crate::particle::Particle::validate_state) on a
    ///   sampled state, or by transport.
    pub fn run_with_spectrum<S>(
        &self,
        spectrum: &S,
        origin: Position,
        direction: [f64; 3],
        rng: &mut impl Rng,
    ) -> Result<SimulationResult, PhysicsError>
    where
        S: EnergySpectrum + ?Sized,
    {
        let unit_dir = normalize_direction(direction)?;
        let mut histories = Vec::with_capacity(self.config.n_histories);
        for _ in 0..self.config.n_histories {
            let energy_mev = spectrum.sample_energy_mev(rng);
            let state = ParticleState {
                position: origin,
                momentum: FourMomentum {
                    energy_mev,
                    direction: unit_dir,
                },
                alive: true,
            };
            self.particle.validate_state(&state)?;
            let primary = self.transport_one(state, rng)?;
            histories.push(primary);
        }
        Ok(finalize(histories))
    }
}

/// PDG ID used for charged secondary stubs (electron). Secondaries produced by
/// photon interactions are electrons or positrons; both are recorded with this
/// ID at the current abstraction level.
const SECONDARY_ELECTRON_PDG: i32 = 11;

/// Aggregate completed `histories` into a [`SimulationResult`], computing the
/// mean and (population) variance of the per-history total energy deposit.
fn finalize(histories: Vec<ParticleHistory>) -> SimulationResult {
    let n = histories.len() as f64;
    let (mean, variance) = if n > 0.0 {
        let mean = histories.iter().map(|h| h.total_deposit_mev).sum::<f64>() / n;
        let variance = histories
            .iter()
            .map(|h| (h.total_deposit_mev - mean) * (h.total_deposit_mev - mean))
            .sum::<f64>()
            / n;
        (mean, variance)
    } else {
        (0.0, 0.0)
    };
    SimulationResult {
        histories,
        mean_deposit_mev: mean,
        variance_deposit_mev2: variance,
    }
}

/// Normalize `direction` to a unit vector.
///
/// # Errors
///
/// [`PhysicsError::SimulationError`] if the vector magnitude is below `1e-12`
/// (no well-defined propagation direction).
fn normalize_direction(direction: [f64; 3]) -> Result<[f64; 3], PhysicsError> {
    let [dx, dy, dz] = direction;
    let mag = (dx * dx + dy * dy + dz * dz).sqrt();
    if mag < 1e-12 {
        return Err(PhysicsError::SimulationError {
            message: "direction vector has (near-)zero magnitude".to_string(),
        });
    }
    Ok([dx / mag, dy / mag, dz / mag])
}

/// Attach each secondary [`ParticleState`] to `parent` as a leaf history with a
/// single track point at its creation site and zero deposit (its energy is
/// already counted in the parent interaction's local deposit).
fn record_secondaries(parent: &mut ParticleHistory, secondaries: Vec<ParticleState>) {
    for s in secondaries {
        let mut leaf = ParticleHistory::new(SECONDARY_ELECTRON_PDG);
        leaf.track.push(TrackPoint {
            position: s.position,
            energy_mev: s.momentum.energy_mev,
            step_length_m: 0.0,
        });
        parent.secondaries.push(leaf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interactions::photon::PhotonInteractionModel;
    use crate::medium::HomogeneousMedium;
    use crate::particle::photon::Photon;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn engine(
        n_histories: usize,
        seed: u64,
    ) -> MonteCarloEngine<Photon, HomogeneousMedium, PhotonInteractionModel> {
        MonteCarloEngine::new(
            Photon,
            HomogeneousMedium::water(),
            PhotonInteractionModel,
            RunConfig {
                n_histories,
                seed,
                ..Default::default()
            },
        )
    }

    #[test]
    fn deposit_never_exceeds_initial_energy() {
        let eng = engine(50, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let result = eng.run(Photon::state_along_z(10.0), &mut rng).unwrap();
        for h in &result.histories {
            assert!(h.total_deposit_mev <= 10.0 + 1e-6);
        }
    }

    #[test]
    fn same_seed_is_reproducible() {
        let eng = engine(100, 7);
        let mut rng1 = StdRng::seed_from_u64(7);
        let mut rng2 = StdRng::seed_from_u64(7);
        let r1 = eng.run(Photon::state_along_z(1.25), &mut rng1).unwrap();
        let r2 = eng.run(Photon::state_along_z(1.25), &mut rng2).unwrap();
        assert_eq!(r1.mean_deposit_mev, r2.mean_deposit_mev);
    }

    #[test]
    fn standard_error_is_small() {
        let n = 500;
        let eng = engine(n, 99);
        let mut rng = StdRng::seed_from_u64(99);
        let result = eng.run(Photon::state_along_z(1.25), &mut rng).unwrap();
        let std_dev = result.variance_deposit_mev2.sqrt();
        let std_err = std_dev / (n as f64).sqrt();
        assert!(std_err < 0.1 * result.mean_deposit_mev);
    }

    #[test]
    fn run_with_spectrum_is_reproducible_and_bounded() {
        // 100 kVp tube spectrum, 10 keV cutoff → energies in [0.01, 0.1] MeV.
        let spec = KramersSpectrum::from_kvp(100.0, 10.0).unwrap();
        let eng = engine(200, 5);
        let origin = Position([0.0, 0.0, 0.0]);
        let dir = [0.0, 0.0, 1.0];

        let mut rng1 = StdRng::seed_from_u64(5);
        let mut rng2 = StdRng::seed_from_u64(5);
        let r1 = eng
            .run_with_spectrum(&spec, origin, dir, &mut rng1)
            .unwrap();
        let r2 = eng
            .run_with_spectrum(&spec, origin, dir, &mut rng2)
            .unwrap();

        // Same seed → identical result.
        assert_eq!(r1.mean_deposit_mev, r2.mean_deposit_mev);
        // No primary can deposit more than the spectrum's endpoint energy.
        for h in &r1.histories {
            assert!(h.total_deposit_mev <= spec.max_energy_mev() + 1e-6);
        }
    }

    #[test]
    fn run_with_spectrum_rejects_zero_direction() {
        let spec = Monoenergetic::new(0.1).unwrap();
        let eng = engine(1, 0);
        let mut rng = StdRng::seed_from_u64(0);
        let err = eng.run_with_spectrum(&spec, Position([0.0; 3]), [0.0; 3], &mut rng);
        assert!(matches!(err, Err(PhysicsError::SimulationError { .. })));
    }

    #[test]
    fn run_with_spectrum_accepts_boxed_trait_object() {
        let spec: Box<dyn EnergySpectrum> = Box::new(Monoenergetic::new(0.1).unwrap());
        let eng = engine(10, 1);
        let mut rng = StdRng::seed_from_u64(1);
        let result = eng
            .run_with_spectrum(spec.as_ref(), Position([0.0; 3]), [0.0, 0.0, 1.0], &mut rng)
            .unwrap();
        assert_eq!(result.histories.len(), 10);
    }
}
