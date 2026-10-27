//! Differential Evolution (DE) optimizer.

use crate::error::OptimizerError;
use crate::objective::EvaluationOracle;
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::shade;
use crate::util::{
    argmax, check_initial_point, check_oracle_len, fitness_stagnated, rows_to_candidates,
    wrap_angle, wrap_difference,
};
use ndarray::{Array1, Array2};
use rand::{seq::IndexedRandom, Rng, RngExt};
use std::f64::consts::PI;
use std::fmt;
use std::str::FromStr;

/// Differential Evolution optimizer.
///
/// [`Optimizer::optimize`] runs the classic DE/rand/1/bin
/// ([`DeStrategy::Rand1Bin`]);
/// [`optimize_with_strategy`](Self::optimize_with_strategy) also runs the
/// adaptive SHADE and L-SHADE strategies.
pub struct AlgorithmDifferentialEvolution;

/// How Differential Evolution builds its trial vectors.
///
/// Every strategy takes the same [`AlgorithmDifferentialEvolutionArgs`] and
/// shares the rest of the run: angles initialised in `[0, 2π)`, difference
/// vectors taken as the shortest signed arc in `(−π, π]`, trial vectors
/// wrapped back into `[0, 2π)`, one oracle batch per
/// generation, the evaluation budget `population_size × (generations + 1)`
/// (the initial population plus `generations` full generations), the
/// fitness-stagnation early stop, and the postconditions of contract C-5. The
/// strategies differ in mutation and parameter control:
///
/// - [`Rand1Bin`](Self::Rand1Bin), the default: DE/rand/1/bin with a fixed
///   mutation factor `F` and crossover rate `CR` (0.8 and 0.7 unless set)
///   and one coordinate always taken from the mutant.
/// - [`Shade`](Self::Shade): SHADE (Tanabe & Fukunaga, CEC 2013):
///   current-to-pbest/1 mutation with an archive of replaced parents, and `F`
///   and `CR` drawn per individual from a memory of the values that produced
///   improvements.
/// - [`LShade`](Self::LShade): L-SHADE (Tanabe & Fukunaga, CEC 2014): SHADE
///   plus a population that shrinks after every generation, linearly in the
///   evaluations spent, from `population_size` to 4 as the budget runs out.
///   For L-SHADE `generations` therefore sets the evaluation budget rather
///   than the number of generations: it spends the budget in more, smaller
///   batches, stopping when the next would not fit.
///   [`max_generations`](Self::max_generations) says how many.
///
/// `Rand1Bin` and `Shade` keep their population, so they run exactly
/// `generations` generations unless they stop early.
///
/// The names `polypus.DE(strategy=...)` accepts are [`as_str`](Self::as_str)
/// and parse back with [`FromStr`]:
///
/// ```
/// use polypus_optimizers::DeStrategy;
///
/// assert_eq!("lshade".parse::<DeStrategy>(), Ok(DeStrategy::LShade));
/// assert_eq!(DeStrategy::default().as_str(), "rand1bin");
/// assert!("LSHADE".parse::<DeStrategy>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeStrategy {
    /// DE/rand/1/bin with a fixed `F` and `CR` (0.8 and 0.7 by default).
    #[default]
    Rand1Bin,
    /// SHADE: success-history adaptive DE with a fixed population.
    Shade,
    /// L-SHADE: SHADE with linear population size reduction.
    LShade,
}

impl DeStrategy {
    /// The strategy's name, as `polypus.DE(strategy=...)` takes it:
    /// `"rand1bin"`, `"shade"` or `"lshade"`.
    pub fn as_str(self) -> &'static str {
        match self {
            DeStrategy::Rand1Bin => "rand1bin",
            DeStrategy::Shade => "shade",
            DeStrategy::LShade => "lshade",
        }
    }

    /// How many generations, one oracle batch each after the initial
    /// population's, a run with these arguments makes unless the
    /// fitness-stagnation stop ends it sooner.
    ///
    /// For [`Rand1Bin`](Self::Rand1Bin) and [`Shade`](Self::Shade) that is
    /// `generations`. [`LShade`](Self::LShade) spends the same budget,
    /// `population_size × (generations + 1)` evaluations, with a shrinking
    /// population, so it runs more generations. Its schedule depends on these
    /// two numbers alone, never on fitness, and this counts it by stepping the
    /// same schedule the optimizer follows, so the two cannot disagree. A
    /// `population_size` below 4, which every strategy rejects, gives a
    /// meaningless count.
    ///
    /// ```
    /// use polypus_optimizers::DeStrategy;
    ///
    /// assert_eq!(DeStrategy::Shade.max_generations(50, 100), 100);
    /// assert!(DeStrategy::LShade.max_generations(50, 100) > 100);
    /// // Four members cannot shrink, so they run the plain 100 generations.
    /// assert_eq!(DeStrategy::LShade.max_generations(4, 100), 100);
    /// ```
    pub fn max_generations(self, population_size: u32, generations: u32) -> u64 {
        match self {
            DeStrategy::Rand1Bin | DeStrategy::Shade => u64::from(generations),
            DeStrategy::LShade => shade::Schedule::new(
                shade::Variant::LShade,
                population_size as usize,
                generations,
            )
            .generations(),
        }
    }
}

impl fmt::Display for DeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DeStrategy {
    type Err = OptimizerError;

    /// Parse one of the names [`as_str`](Self::as_str) returns. Matching is
    /// exact; any other string is [`OptimizerError::InvalidConfig`] for
    /// `strategy`, which the Python binding raises as `ValueError`.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "rand1bin" => Ok(DeStrategy::Rand1Bin),
            "shade" => Ok(DeStrategy::Shade),
            "lshade" => Ok(DeStrategy::LShade),
            other => Err(OptimizerError::InvalidConfig {
                parameter: "strategy",
                reason: format!("must be \"rand1bin\", \"shade\" or \"lshade\", got {other:?}"),
            }),
        }
    }
}

/// Arguments for the Differential Evolution optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// all evaluation is delegated to [`oracle`](AlgorithmDifferentialEvolutionArgs::oracle).
///
/// # Preconditions
///
/// `population_size >= 4` — each trial mutation samples 3 *distinct* other
/// members of the population. `mutation_factor` is finite and in `(0, 2]`,
/// `crossover_rate` finite and in `[0, 1]`, and both are `None` for SHADE and
/// L-SHADE. `initial_point`, when given, holds `dimensions` finite values.
pub struct AlgorithmDifferentialEvolutionArgs {
    /// Oracle that maps parameter vectors to fitness values.
    pub oracle: Box<dyn EvaluationOracle>,
    pub population_size: u32,
    /// Generations to run, one oracle batch each after the initial
    /// population's: the evaluation budget is `population_size ×
    /// (generations + 1)`. [`DeStrategy::LShade`] spends that budget in more,
    /// smaller generations (see [`DeStrategy::max_generations`]).
    pub generations: u32,
    pub dimensions: u32,
    /// Minimum cumulative improvement of the **best fitness** over the last
    /// [`patience`](Self::patience) generations below which the run stops early.
    ///
    /// DE maximises (higher fitness is better), so this is a threshold on
    /// *quality progress* in the oracle's own fitness units — **not** a
    /// population-spread threshold in parameter units. See
    /// `crate::util::fitness_stagnated` for the exact test.
    pub tolerance: f64,
    /// Number of generations over which the best-fitness improvement is measured
    /// for the stagnation early stop. A stop can never fire before generation
    /// `patience`, so a larger value makes the optimizer more patient (runs
    /// longer before giving up). The Python `DE` binding defaults it to 20.
    pub patience: u32,
    /// Optional RNG seed. `None` (the default) uses [`rand::rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
    /// Optional starting point, one angle per dimension, all finite. Every
    /// strategy draws its initial population as usual and then replaces the
    /// first member with this point wrapped into `[0, 2π)`, so it is the first
    /// candidate of the initial batch while the other members, and the RNG
    /// stream, are the same as without it. `None` draws the whole population.
    pub initial_point: Option<Vec<f64>>,
    /// DE/rand/1/bin's mutation factor `F`, which scales the difference
    /// vector; `None` is
    /// [`DEFAULT_MUTATION_FACTOR`](AlgorithmDifferentialEvolution::DEFAULT_MUTATION_FACTOR).
    /// SHADE and L-SHADE adapt `F` themselves and reject a value.
    pub mutation_factor: Option<f64>,
    /// DE/rand/1/bin's crossover rate `CR`, the probability that a coordinate
    /// comes from the mutant (one coordinate always does); `None` is
    /// [`DEFAULT_CROSSOVER_RATE`](AlgorithmDifferentialEvolution::DEFAULT_CROSSOVER_RATE).
    /// SHADE and L-SHADE adapt `CR` themselves and reject a value.
    pub crossover_rate: Option<f64>,
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
    /// *distinct* other members, so at least 4 members are required. Every
    /// strategy shares it; 4 is also the size L-SHADE shrinks towards.
    pub(crate) const MIN_POPULATION: usize = 4;

    /// DE/rand/1/bin's mutation factor `F` when none is given.
    pub const DEFAULT_MUTATION_FACTOR: f64 = 0.8;

    /// DE/rand/1/bin's crossover rate `CR` when none is given.
    pub const DEFAULT_CROSSOVER_RATE: f64 = 0.7;

    /// Check the preconditions of [`AlgorithmDifferentialEvolutionArgs`] that
    /// every strategy checks before any RNG draw or oracle call, in the same
    /// order, without running anything: `population_size` at least 4, then
    /// `mutation_factor` and `crossover_rate` (in range for
    /// [`DeStrategy::Rand1Bin`], `None` for the adaptive strategies, as
    /// [`OptimizerError::InvalidConfig`]), then `initial_point` (`dimensions`
    /// finite values). `train` and `qml.train` call it before any backend
    /// exists.
    pub fn check_preconditions(
        strategy: DeStrategy,
        population_size: u32,
        dimensions: u32,
        mutation_factor: Option<f64>,
        crossover_rate: Option<f64>,
        initial_point: Option<&[f64]>,
    ) -> Result<(), OptimizerError> {
        // Sampling 3 distinct other members needs `popsize >= 4`: rejected
        // with a typed error instead of the out-of-bounds `sel[2]` panic that
        // fired one line into the trial loop.
        let popsize = population_size as usize;
        if popsize < Self::MIN_POPULATION {
            return Err(OptimizerError::PopulationTooSmall {
                got: popsize,
                min: Self::MIN_POPULATION,
            });
        }
        Self::check_rates(strategy, mutation_factor, crossover_rate)?;
        check_initial_point(initial_point, dimensions as usize, None)
    }

    /// Check `mutation_factor` and `crossover_rate` for `strategy`: for
    /// [`DeStrategy::Rand1Bin`] each, when set, is finite and in `(0, 2]`
    /// (`F`) or `[0, 1]` (`CR`); SHADE and L-SHADE adapt both themselves, so
    /// neither may be set. A violation is [`OptimizerError::InvalidConfig`].
    /// Part of [`check_preconditions`](Self::check_preconditions); `polypus.DE`
    /// also calls it when it is built or changed.
    pub fn check_rates(
        strategy: DeStrategy,
        mutation_factor: Option<f64>,
        crossover_rate: Option<f64>,
    ) -> Result<(), OptimizerError> {
        let check = |parameter, value: Option<f64>, domain, in_range: bool| match value {
            Some(value) if strategy != DeStrategy::Rand1Bin => Err(OptimizerError::InvalidConfig {
                parameter,
                reason: format!(
                    "the {strategy} strategy adapts it itself, so it must be None (got {value}); \
                     set it to None before switching strategy"
                ),
            }),
            Some(value) if !in_range => Err(OptimizerError::InvalidConfig {
                parameter,
                reason: format!("must be finite and {domain}, got {value}"),
            }),
            _ => Ok(()),
        };
        // NaN and the infinities fall outside both ranges.
        let f = mutation_factor.unwrap_or(Self::DEFAULT_MUTATION_FACTOR);
        check(
            "mutation_factor",
            mutation_factor,
            "in (0, 2]",
            f > 0.0 && f <= 2.0,
        )?;
        let cr = crossover_rate.unwrap_or(Self::DEFAULT_CROSSOVER_RATE);
        check(
            "crossover_rate",
            crossover_rate,
            "in [0, 1]",
            (0.0..=1.0).contains(&cr),
        )?;
        Ok(())
    }

    /// Run Differential Evolution with the given `strategy`.
    ///
    /// [`DeStrategy::Rand1Bin`] is exactly [`Optimizer::optimize`]: the same
    /// code, RNG draws and outcome for a given seed. [`DeStrategy::Shade`] and
    /// [`DeStrategy::LShade`] take the same arguments and return the same
    /// errors (those of [`check_preconditions`](Self::check_preconditions),
    /// an oracle that returns the wrong number of values); see [`DeStrategy`] for what they change, including
    /// L-SHADE's use of `generations` as an evaluation budget.
    ///
    /// ```
    /// use polypus_optimizers::{
    ///     AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, DeStrategy,
    ///     EvaluationOracle,
    /// };
    ///
    /// // Fitness = -Σ(xᵢ - 1)² is maximised (value 0) at xᵢ = 1.
    /// struct Sphere;
    /// impl EvaluationOracle for Sphere {
    ///     fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
    ///         candidates
    ///             .iter()
    ///             .map(|c| -c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>())
    ///             .collect()
    ///     }
    /// }
    ///
    /// let outcome = AlgorithmDifferentialEvolution
    ///     .optimize_with_strategy(
    ///         DeStrategy::LShade,
    ///         AlgorithmDifferentialEvolutionArgs {
    ///             oracle: Box::new(Sphere),
    ///             population_size: 40,
    ///             generations: 200,
    ///             dimensions: 2,
    ///             tolerance: 1e-6,
    ///             patience: 20,
    ///             seed: Some(7),
    ///             initial_point: None,
    ///             mutation_factor: None,
    ///             crossover_rate: None,
    ///         },
    ///     )
    ///     .expect("valid arguments optimize successfully");
    ///
    /// assert!(outcome.best_fitness > -1e-2, "fitness = {}", outcome.best_fitness);
    /// ```
    pub fn optimize_with_strategy(
        &self,
        strategy: DeStrategy,
        args: AlgorithmDifferentialEvolutionArgs,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        match strategy {
            DeStrategy::Rand1Bin => self.optimize(args),
            DeStrategy::Shade => with_seeded_rng(args.seed, |rng| {
                shade::run_with_rng(args, shade::Variant::Shade, rng)
            }),
            DeStrategy::LShade => with_seeded_rng(args.seed, |rng| {
                shade::run_with_rng(args, shade::Variant::LShade, rng)
            }),
        }
    }

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
            patience,
            seed: _,
            initial_point,
            mutation_factor,
            crossover_rate,
        } = args;

        let popsize = population_size as usize;
        let max_gen = generations as usize;
        let dims = dimensions as usize;
        let patience = patience as usize;

        // Preconditions (documented on the struct), before any RNG draw or
        // oracle call.
        Self::check_preconditions(
            DeStrategy::Rand1Bin,
            population_size,
            dimensions,
            mutation_factor,
            crossover_rate,
            initial_point.as_deref(),
        )?;
        let f = mutation_factor.unwrap_or(Self::DEFAULT_MUTATION_FACTOR);
        let cr = crossover_rate.unwrap_or(Self::DEFAULT_CROSSOVER_RATE);

        // Initialise population with angles in [0, 2pi)
        let mut pop = Array2::<f64>::zeros((popsize, dims));
        for mut row in pop.outer_iter_mut() {
            for e in row.iter_mut() {
                *e = rng.random_range(0.0..2.0 * PI);
            }
        }
        // The caller's point replaces the first member once every member is
        // drawn, so the rest of the population and the RNG stream are unchanged.
        if let (Some(point), Some(mut first)) = (&initial_point, pop.outer_iter_mut().next()) {
            for (e, &x) in first.iter_mut().zip(point) {
                *e = wrap_angle(x);
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
        // Best fitness at the end of each generation; drives the fitness-
        // stagnation early stop and is reported in the outcome.
        let mut fitness_history: Vec<f64> = Vec::with_capacity(max_gen);

        for generation in 0..max_gen {
            iterations_run = generation + 1;

            // Build all trial vectors for this generation.
            let mut trials: Vec<Array1<f64>> = Vec::with_capacity(popsize);
            for i in 0..popsize {
                let ids: Vec<usize> = (0..popsize).filter(|&j| j != i).collect();
                let sel: Vec<usize> = ids.sample(rng, 3).cloned().collect();
                let c1 = pop.row(sel[0]).to_owned();
                let c2 = pop.row(sel[1]).to_owned();
                let c3 = pop.row(sel[2]).to_owned();
                // Angles live on a circle: the difference is the shortest arc.
                let difference = (&c2 - &c3).mapv(wrap_difference);
                let mut mutant = &c1 + f * &difference;
                mutant.mapv_inplace(wrap_angle);
                // One coordinate always comes from the mutant (`j_rand`), so a
                // trial is never a plain copy of its parent.
                let j_rand = if dims > 0 {
                    rng.random_range(0..dims)
                } else {
                    0
                };
                let cross: Array1<bool> =
                    Array1::from_shape_fn(dims, |j| rng.random_bool(cr) || j == j_rand);
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

            // Record this generation's best and stop once the best fitness has
            // stagnated (improved by < tolerance over the last `patience`
            // generations). This replaces the former population-std collapse
            // test, which on QAOA-like landscapes fired within a handful of
            // generations — long before the fitness had actually plateaued —
            // and contradicted the C-5 contract's fitness-stagnation prose. The
            // search dynamics (DE/rand/1/bin, [0, 2π) init) are
            // unchanged; only the stopping rule differs.
            fitness_history.push(fitness[best_idx]);
            if fitness_stagnated(&fitness_history, tolerance, patience, generation) {
                converged = true;
                break;
            }
        }

        Ok(OptimizationOutcome {
            best_fitness: fitness[best_idx],
            best_params: best,
            iterations_run,
            converged,
            fitness_history,
        })
    }
}

impl Optimizer for AlgorithmDifferentialEvolution {
    type Args = AlgorithmDifferentialEvolutionArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
