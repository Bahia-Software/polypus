//! Differential Evolution (DE) optimizer.

use crate::error::OptimizerError;
use crate::objective::EvaluationOracle;
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{argmax, check_oracle_len, population_converged, rows_to_candidates};
use ndarray::{Array1, Array2};
use rand::{seq::SliceRandom, Rng};
use std::f64::consts::PI;

/// Differential Evolution optimizer.
pub struct AlgorithmDifferentialEvolution;

/// Arguments for the Differential Evolution optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// all evaluation is delegated to [`oracle`](AlgorithmDifferentialEvolutionArgs::oracle).
///
/// # Preconditions
///
/// `population_size >= 4` — each trial mutation samples 3 *distinct* other
/// members of the population.
pub struct AlgorithmDifferentialEvolutionArgs {
    /// Oracle that maps parameter vectors to fitness values.
    pub oracle: Box<dyn EvaluationOracle>,
    pub population_size: u32,
    pub generations: u32,
    pub dimensions: u32,
    pub tolerance: f64,
    /// Optional RNG seed. `None` (the default) uses [`rand::thread_rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
}

impl AlgorithmDifferentialEvolution {
    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("Differential Evolution")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from("Trains a variational quantum circuit using Differential Evolution.")
    }

    /// The smallest valid `population_size`: each trial mutation samples 3
    /// *distinct* other members, so at least 4 members are required.
    const MIN_POPULATION: usize = 4;

    fn run_with_rng<R: Rng>(
        args: AlgorithmDifferentialEvolutionArgs,
        rng: &mut R,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmDifferentialEvolutionArgs {
            oracle,
            population_size,
            generations,
            dimensions,
            tolerance,
            seed: _,
        } = args;

        let popsize = population_size as usize;
        let max_gen = generations as usize;
        let dims = dimensions as usize;

        // Precondition (documented on the struct): sampling 3 distinct other
        // members needs `popsize >= 4`. Reject *before* any RNG draw or oracle
        // call so the caller gets a typed error instead of the out-of-bounds
        // `sel[2]` panic that fired one line into the trial loop.
        if popsize < Self::MIN_POPULATION {
            return Err(OptimizerError::PopulationTooSmall {
                got: popsize,
                min: Self::MIN_POPULATION,
            });
        }

        // Initialise population with angles in [0, 2pi)
        let mut pop = Array2::<f64>::zeros((popsize, dims));
        for mut row in pop.outer_iter_mut() {
            for e in row.iter_mut() {
                *e = rng.gen_range(0.0..2.0 * PI);
            }
        }

        // Evaluate initial population (fixes the bug of starting fitness at 0.0)
        let init_candidates = rows_to_candidates(&pop);
        let mut fitness = oracle.evaluate_batch(&init_candidates);
        check_oracle_len(init_candidates.len(), fitness.len())?;
        let mut best_idx = argmax(&fitness);
        let mut best = pop.row(best_idx).to_vec();

        let mut iterations_run = 0usize;
        let mut converged = false;

        for generation in 0..max_gen {
            iterations_run = generation + 1;

            // Build all trial vectors for this generation.
            let mut trials: Vec<Array1<f64>> = Vec::with_capacity(popsize);
            for i in 0..popsize {
                let ids: Vec<usize> = (0..popsize).filter(|&j| j != i).collect();
                let sel: Vec<usize> = ids.choose_multiple(rng, 3).cloned().collect();
                let c1 = pop.row(sel[0]).to_owned();
                let c2 = pop.row(sel[1]).to_owned();
                let c3 = pop.row(sel[2]).to_owned();
                let mut mutant = &c1 + 0.8 * (&c2 - &c3);
                mutant.mapv_inplace(|x| x.rem_euclid(2.0 * PI));
                let cross: Array1<bool> = Array1::from_shape_fn(dims, |_| rng.gen_bool(0.7));
                let trial = cross
                    .iter()
                    .zip(mutant.iter())
                    .zip(pop.row(i).iter())
                    .map(|((cp, m), p)| if *cp { *m } else { *p })
                    .collect::<Array1<f64>>();
                trials.push(trial);
            }

            // Evaluate all trials in a single oracle call.
            let trial_candidates: Vec<Vec<f64>> = trials.iter().map(|t| t.to_vec()).collect();
            let trial_fitness = oracle.evaluate_batch(&trial_candidates);
            check_oracle_len(trial_candidates.len(), trial_fitness.len())?;

            // Selection: greedily accept every trial that beats its parent.
            for i in 0..popsize {
                if trial_fitness[i] > fitness[i] {
                    fitness[i] = trial_fitness[i];
                    pop.row_mut(i).assign(&trials[i]);
                }
            }

            // Recompute the champion only after the whole selection pass, the
            // same way PSO recomputes its global best (src/pso.rs:136-138).
            // Tracking best_idx/best *inside* the per-i branch hit an ordering
            // trap: when i == best_idx, fitness[best_idx] was mutated on the
            // line above before the `trial_fitness[i] > fitness[best_idx]`
            // check, so the comparison became `x > x` (always false) and the
            // champion was left pointing at a stale vector even though its own
            // slot had improved — breaking the C-5 invariant
            // fitness[best_idx] == f(best). argmax reads the just-updated
            // fitness, so best_params and best_fitness always agree.
            best_idx = argmax(&fitness);
            best = pop.row(best_idx).to_vec();

            if population_converged(&pop, tolerance, generation) {
                converged = true;
                break;
            }
        }

        Ok(OptimizationOutcome {
            best_fitness: fitness[best_idx],
            best_params: best,
            iterations_run,
            converged,
        })
    }
}

impl Optimizer for AlgorithmDifferentialEvolution {
    type Args = AlgorithmDifferentialEvolutionArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
