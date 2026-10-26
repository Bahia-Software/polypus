//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with IPOP restarts.

use crate::error::OptimizerError;
use crate::linalg::symmetric_eigen;
use crate::objective::EvaluationOracle;
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{
    check_initial_point, check_oracle_len, recent_improvement, stagnated, wrap_angle,
};
use ndarray::{Array1, Array2};
use rand::{Rng, RngExt};
use std::cmp::Ordering;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::fmt;

/// Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
///
/// The standard (μ/μ_w, λ)-CMA-ES of N. Hansen, *The CMA Evolution Strategy:
/// A Tutorial* (arXiv:1604.00772, Table 1 and Fig. 6): λ samples per
/// generation from `N(m, σ²C)`, the μ = ⌊λ/2⌋ best recombined with positive
/// log-decreasing weights, rank-one and rank-μ covariance updates, cumulative
/// step-size adaptation (CSA) and the `h_σ` stall of the rank-one path, all
/// with the tutorial's default learning rates. The eigendecomposition of `C`
/// (this crate's own cyclic Jacobi solver; no new dependency) is recomputed
/// every `max(1, ⌊1 / (10·d·(c₁ + c_μ))⌋)` generations, Hansen's lazy-update
/// gap, which for the few parameters of a variational circuit is every
/// generation.
///
/// It maximises: candidates are ranked by their negated fitness, best first,
/// with `NaN` ranked worst, and nothing but that ranking reaches the search
/// distribution. A strictly increasing transformation of the fitness
/// therefore samples exactly the same candidates; only the stagnation stop
/// below compares fitness values.
///
/// # Angles
///
/// The parameters are angles of period 2π. The mean and the samples live
/// unwrapped in ℝ^d, so the covariance adapts across the `0`/`2π` seam, and
/// each candidate is wrapped into `[0, 2π)` only when it is handed to the
/// oracle (`f(x) = f(x mod 2π)`). The initial mean is drawn uniformly from
/// `[0, 2π)`, or is [`initial_point`](AlgorithmCmaEsArgs::initial_point)
/// wrapped into it, and the returned `best_params` is the wrapped candidate the
/// oracle scored.
///
/// The largest coordinate standard deviation, `σ·max_i √C_ii`, is capped at
/// `max(σ₀, π/2)` by shrinking σ after each update, like pycma's `maxstd`.
/// A wider normal wraps to an almost uniform distribution on the circle, where
/// the ranking no longer tells the search which way to move and the step-size
/// adaptation drifts without bound. Without the cap, a run whose mean starts
/// near a fitness minimum (where longer steps rank better) can grow σ past the
/// period and wander uniformly until the stagnation stop ends it.
///
/// # Stopping and IPOP restarts
///
/// A run (the first one, or a restart) ends when one of its stop criteria
/// fires:
///
/// - **fitness stagnation**: the run's own best fitness improved by less than
///   `tolerance` over its last `patience` generations — DE's rule (contract
///   C-5) applied to the run, so without restarts it is exactly DE's stop;
/// - **degenerate distribution** (Hansen's termination criteria): a non-finite
///   mean, step size, evolution path or covariance; a covariance whose
///   condition number exceeds `1e14` (or that is no longer positive definite);
///   a coordinate spread `σ·max_i √C_ii` below `1e-12·σ₀` (TolX: the
///   distribution has collapsed onto a point) or above `1e4·σ₀` (TolXUp: it
///   diverges).
///
/// While restarts remain (`restarts`), the next run starts IPOP-style (Auger
/// and Hansen, CEC 2005): a fresh uniform mean, `σ₀`, `C = I`, and twice the
/// previous population. Otherwise the whole optimisation stops with
/// `converged = true`. Which criterion fired is logged at debug level. The
/// generation budget is shared by all runs; when it runs out first,
/// `converged` is `false`.
///
/// # Outcome (contract C-5)
///
/// `best_params` is the best candidate the oracle scored over all runs (never
/// the unevaluated mean) and `best_fitness` its value. `fitness_history`
/// holds the best fitness so far after each generation, across restarts, so
/// it never decreases (with `NaN` as the worst value) and ends at
/// `best_fitness`; `iterations_run` is its length, the generations executed.
///
/// # Batches
///
/// Each generation is one `evaluate_batch` call of λ candidates, its length
/// checked; there is no separate initial batch. λ doubles at each restart, so
/// later batches can be larger than the first: [`CmaEsBatchSchedule`] bounds
/// them before the run.
///
/// # Example
///
/// ```
/// use polypus_optimizers::{AlgorithmCmaEs, AlgorithmCmaEsArgs, EvaluationOracle, Optimizer};
/// use std::f64::consts::TAU;
///
/// // Fitness Σ cos(xᵢ - 1) is maximised (value 3) at xᵢ = 1 (mod 2π).
/// struct Cosines;
/// impl EvaluationOracle for Cosines {
///     fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
///         candidates
///             .iter()
///             .map(|c| c.iter().map(|x| (x - 1.0).cos()).sum())
///             .collect()
///     }
/// }
///
/// let outcome = AlgorithmCmaEs
///     .optimize(AlgorithmCmaEsArgs {
///         oracle: Box::new(Cosines),
///         population_size: None, // 4 + ⌊3 ln 3⌋ = 7
///         generations: 300,
///         dimensions: 3,
///         sigma0: AlgorithmCmaEs::DEFAULT_SIGMA0,
///         tolerance: 1e-12,
///         patience: 30,
///         restarts: 0,
///         seed: Some(7),
///         initial_point: None,
///     })
///     .expect("valid arguments optimize successfully");
///
/// assert!(outcome.best_fitness > 3.0 - 1e-9, "fitness = {}", outcome.best_fitness);
/// for x in &outcome.best_params {
///     assert!((0.0..TAU).contains(x));
///     assert!((x - 1.0).abs() < 1e-4, "param off target: {x}");
/// }
/// ```
pub struct AlgorithmCmaEs;

/// Arguments for [`AlgorithmCmaEs`].
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// all evaluation is delegated to [`oracle`](AlgorithmCmaEsArgs::oracle).
///
/// # Preconditions
///
/// Checked before any RNG draw or oracle call, each violation an
/// [`OptimizerError::InvalidConfig`] naming the parameter:
///
/// - `dimensions >= 1`: CMA-ES's learning rates are undefined in zero
///   dimensions (they divide by `d`). DE and PSO accept zero dimensions; this
///   is the `dimensions >= 1` precondition contract C-5 lists.
/// - `generations >= 1`: CMA-ES has no initial batch, so zero generations
///   would evaluate nothing and leave no candidate to return.
/// - `sigma0` finite and in `(0, 2π]`: a larger step samples the circle no
///   more uniformly (a wrapped normal with σ = 2π is uniform to within 1e-8)
///   and could overflow the samples.
/// - `population_size >= 2` when given: μ = ⌊λ/2⌋ must be at least 1.
/// - `restarts` small enough that the largest population, `λ₀·2^restarts`,
///   fits in a `u32` like `population_size`.
/// - `initial_point`, when given, holds `dimensions` finite values.
pub struct AlgorithmCmaEsArgs {
    /// Oracle that maps parameter vectors to fitness values (higher is better).
    pub oracle: Box<dyn EvaluationOracle>,
    /// Population size λ of the first run; `None` uses Hansen's default
    /// `4 + ⌊3 ln d⌋` ([`AlgorithmCmaEs::default_population_size`]). Each
    /// IPOP restart doubles it. With several QPUs, a multiple of their count
    /// keeps them all busy.
    pub population_size: Option<u32>,
    /// Total generations, over the first run and all restarts: the run makes
    /// at most this many `evaluate_batch` calls, one per generation.
    pub generations: u32,
    /// Number of parameters `d`.
    pub dimensions: u32,
    /// Initial step size σ₀, in radians, in `(0, 2π]`; every restart starts
    /// from it again. [`AlgorithmCmaEs::DEFAULT_SIGMA0`] (π/2) suits angles
    /// spread over the whole circle. Much larger values spread the samples
    /// over more than one turn, where the periodic fitness no longer tells
    /// the search which way to move, and runs stall more often before they
    /// converge. The coordinate standard deviations never exceed
    /// `max(sigma0, π/2)` during the run.
    pub sigma0: f64,
    /// Minimum improvement of a run's best fitness over its last
    /// [`patience`](Self::patience) generations below which the run ends
    /// (contract C-5's rule, in the oracle's fitness units).
    pub tolerance: f64,
    /// Generations over which a run's best-fitness improvement is measured;
    /// no run ends on stagnation before it has run `patience + 1` generations.
    pub patience: u32,
    /// IPOP restarts allowed after the first run; each doubles the population.
    /// The generation budget is shared, so restarts trade generations of one
    /// run for runs with larger batches: with `generations` fixed, keep this
    /// small.
    pub restarts: u32,
    /// Optional RNG seed. `None` (the default) uses [`rand::rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
    /// Optional mean of the first run, one value per dimension, all finite,
    /// wrapped into `[0, 2π)` so a far angle keeps its precision. The
    /// uniform mean is still drawn and then replaced, so the rest of the run
    /// draws what it would without a point, and every IPOP restart draws a
    /// fresh mean as usual. `sigma0` is not adjusted: to refine around a good
    /// point rather than explore from it, pass a smaller `sigma0` too.
    /// `None` draws the first mean uniformly from `[0, 2π)`.
    pub initial_point: Option<Vec<f64>>,
}

impl AlgorithmCmaEs {
    /// The default initial step size σ₀: π/2 radians, a quarter turn, so the
    /// initial `m ± 2σ₀` covers the whole circle around a uniform start.
    pub const DEFAULT_SIGMA0: f64 = FRAC_PI_2;

    /// The smallest valid `population_size`: μ = ⌊λ/2⌋ must be at least 1.
    pub const MIN_POPULATION: u32 = 2;

    /// The largest valid `sigma0`: one full turn, 2π.
    pub const MAX_SIGMA0: f64 = TAU;

    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("CMA-ES")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit using the Covariance Matrix Adaptation \
             Evolution Strategy (CMA-ES), with optional IPOP restarts.",
        )
    }

    /// Hansen's default population size for `dimensions` parameters,
    /// `4 + ⌊3 ln d⌋`: 4 for one parameter, 9 for six, 12 for twenty.
    /// `dimensions == 0`, which [`optimize`](Optimizer::optimize) rejects,
    /// counts as one.
    pub fn default_population_size(dimensions: u32) -> u32 {
        let d = f64::from(dimensions.max(1));
        // 3 ln d < 67 for any u32, so the cast is exact.
        4 + (3.0 * d.ln()).floor() as u32
    }

    fn run_with_rng<R: Rng>(
        args: AlgorithmCmaEsArgs,
        rng: &mut R,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmCmaEsArgs {
            oracle,
            population_size,
            generations,
            dimensions,
            sigma0,
            tolerance,
            patience,
            restarts,
            seed: _,
            mut initial_point,
        } = args;

        let schedule = validate(dimensions, population_size, generations, sigma0, restarts)?;
        let n = dimensions as usize;
        check_initial_point(initial_point.as_deref(), n)?;
        let budget = generations as usize;
        let patience = patience as usize;

        let mut best = Incumbent::new();
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut converged = false;
        // Index of the current run: 0 is the first, k the k-th restart.
        let mut run = 0u32;

        'runs: while fitness_history.len() < budget {
            // `validate` keeps every population this loop can reach within u32.
            let lambda = schedule.population(run) as usize;
            let strategy = Strategy::new(n, lambda);
            let drawn: Array1<f64> = (0..n).map(|_| rng.random_range(0.0..TAU)).collect();
            // The first run starts at the caller's point, if any, wrapped like
            // every candidate so a huge angle keeps its precision; the mean is
            // drawn anyway so the RNG stream does not depend on it, and every
            // restart draws a fresh one.
            let mean = initial_point
                .take()
                .map_or(drawn, |point| point.into_iter().map(wrap_angle).collect());
            let mut dist = Distribution::new(mean, sigma0);
            let mut run_best = Incumbent::new();
            let mut run_history: Vec<f64> = Vec::new();

            while fitness_history.len() < budget {
                let generation = fitness_history.len();
                let steps = dist.sample(&strategy, rng);
                let candidates: Vec<Vec<f64>> = steps.iter().map(|y| dist.candidate(y)).collect();
                let fitness = oracle.evaluate_batch(&candidates);
                check_oracle_len(candidates.len(), fitness.len())?;

                for (candidate, &value) in candidates.iter().zip(&fitness) {
                    best.offer(value, candidate);
                    run_best.offer(value, candidate);
                }
                fitness_history.push(best.fitness);
                run_history.push(run_best.fitness);
                dist.update(&strategy, &steps, &ranking(&fitness));

                log::debug!(
                    "Generation {generation}: run {}, population {lambda}, sigma {:.3e}, \
                     best fitness {:.6}, improvement of this run over last {patience} \
                     generations {:?}",
                    run + 1,
                    dist.sigma,
                    best.fitness,
                    recent_improvement(&run_history, patience),
                );

                let stop = dist.degeneration(sigma0).or_else(|| {
                    stagnated(&run_history, tolerance, patience).then_some(StopReason::Stagnation)
                });
                let Some(reason) = stop else {
                    continue;
                };
                if run < restarts && fitness_history.len() < budget {
                    run += 1;
                    log::debug!(
                        "Restarting at generation {generation} due to {reason}: IPOP restart \
                         {run} of {restarts}, population {}",
                        schedule.population(run)
                    );
                    continue 'runs;
                }
                log::debug!("Stopping early at generation {generation} due to {reason}");
                converged = true;
                break 'runs;
            }
        }

        Ok(OptimizationOutcome {
            best_fitness: best.fitness,
            best_params: best.params,
            iterations_run: fitness_history.len(),
            converged,
            fitness_history,
        })
    }
}

impl Optimizer for AlgorithmCmaEs {
    type Args = AlgorithmCmaEsArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}

/// Upper bounds on the batches an [`AlgorithmCmaEs`] run submits, known before
/// it starts: what a caller needs to budget the run's oracle calls (e.g. a
/// walltime projection).
///
/// A run makes one `evaluate_batch` call per generation, at most
/// [`generations`](Self::generations) of them, with no separate initial batch.
/// Its first run submits [`initial_population`](Self::initial_population)
/// candidates per batch and IPOP restart `k` submits
/// [`population(k)`](Self::population) `= initial_population · 2^k`, the rule
/// the optimizer itself follows. When a restart happens depends on the search,
/// so the actual sizes are not known in advance; the bounds assume every
/// restart comes as early as it can, one generation after the previous run
/// started, which is the worst case but rarely the real one (a run usually
/// lasts at least `patience + 1` generations). Without restarts they are
/// exact: every batch has `initial_population` candidates.
///
/// Projecting from the first batch: if a batch of `k·λ` candidates takes at
/// most `k` times as long as one of `λ` (true for a backend that spreads a
/// batch over its QPUs), the whole optimisation takes at most
/// `first_batch_time · max_evaluations() / initial_population()`.
///
/// ```
/// use polypus_optimizers::CmaEsBatchSchedule;
///
/// // 6 parameters: λ₀ = 4 + ⌊3 ln 6⌋ = 9; 100 generations, up to 2 restarts.
/// let schedule = CmaEsBatchSchedule::new(6, None, 100, 2);
/// assert_eq!(schedule.initial_population(), 9);
/// assert_eq!(schedule.population(2), 36);
/// assert_eq!(schedule.max_batch_size(0), 9);
/// assert_eq!(schedule.max_batch_size(1), 18);
/// assert_eq!(schedule.max_batch_size(99), 36);
/// assert_eq!(schedule.max_batch_size(100), 0); // past the budget
/// assert_eq!(schedule.max_evaluations(), 9 + 18 + 98 * 36);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CmaEsBatchSchedule {
    initial_population: u32,
    generations: u32,
    restarts: u32,
}

impl CmaEsBatchSchedule {
    /// The schedule of a run with these [`AlgorithmCmaEsArgs`] settings. It
    /// does not validate them (that is [`Optimizer::optimize`]'s job), so it
    /// never fails; for a configuration `optimize` rejects the numbers are
    /// meaningless but finite, saturating at `u64::MAX`.
    pub fn new(
        dimensions: u32,
        population_size: Option<u32>,
        generations: u32,
        restarts: u32,
    ) -> Self {
        Self {
            initial_population: population_size
                .unwrap_or_else(|| AlgorithmCmaEs::default_population_size(dimensions)),
            generations,
            restarts,
        }
    }

    /// λ₀, the batch size of the first run and the size of the first batch.
    pub fn initial_population(&self) -> u32 {
        self.initial_population
    }

    /// The most batches the run submits: one per generation.
    pub fn generations(&self) -> u32 {
        self.generations
    }

    /// The batch size of run `run` (0 is the first run, `k` the `k`-th
    /// restart): `initial_population · 2^run`, saturating at `u64::MAX`.
    pub fn population(&self, run: u32) -> u64 {
        self.checked_population(run).unwrap_or(u64::MAX)
    }

    /// `initial_population · 2^run`, or `None` past `u64::MAX`.
    fn checked_population(&self, run: u32) -> Option<u64> {
        1u64.checked_shl(run)
            .and_then(|factor| u64::from(self.initial_population).checked_mul(factor))
    }

    /// The largest batch generation `generation` (0-based) can submit:
    /// `initial_population · 2^min(generation, restarts)`, since at most one
    /// restart happens per generation; 0 from `generations` on, where the run
    /// has ended.
    pub fn max_batch_size(&self, generation: u32) -> u64 {
        if generation >= self.generations {
            0
        } else {
            self.population(generation.min(self.restarts))
        }
    }

    /// The largest batch of the whole run, `max_batch_size(generations - 1)`.
    pub fn max_population(&self) -> u64 {
        self.generations
            .checked_sub(1)
            .map_or(0, |last| self.max_batch_size(last))
    }

    /// The sum of [`max_batch_size`](Self::max_batch_size) over all
    /// generations: an upper bound on the candidates the run evaluates,
    /// `initial_population · generations` without restarts.
    pub fn max_evaluations(&self) -> u64 {
        // The bound doubles for the first `k = min(restarts, generations)`
        // generations, Σ_{g<k} λ₀·2^g = λ₀·(2^k − 1), and stays at λ₀·2^k after.
        let k = self.restarts.min(self.generations);
        self.checked_population(k)
            .and_then(|peak| {
                let growing = peak - u64::from(self.initial_population);
                let steady = u64::from(self.generations - k).checked_mul(peak)?;
                growing.checked_add(steady)
            })
            .unwrap_or(u64::MAX)
    }
}

/// α_cov in Hansen's c₁ and c_μ (tutorial Table 1).
const ALPHA_COV: f64 = 2.0;
/// TolX: a run ends once every coordinate's standard deviation, `σ·√C_ii`, is
/// below this times σ₀ (Hansen's default).
const TOL_X: f64 = 1e-12;
/// TolXUp: a run ends once some coordinate's standard deviation exceeds this
/// times σ₀, a sign of divergence (Hansen's default).
const TOL_X_UP: f64 = 1e4;
/// ConditionCov: a run ends once the condition number of `C` exceeds this
/// (Hansen's default; beyond it the decomposition loses all precision).
const MAX_CONDITION: f64 = 1e14;
/// The cap on the largest coordinate standard deviation, `σ·max_i √C_ii`,
/// when `σ₀` is smaller: a quarter turn, so `m ± 2σ` already spans the circle.
const MAX_SPREAD: f64 = FRAC_PI_2;

/// Check every precondition of [`AlgorithmCmaEsArgs`] and return the run's
/// batch schedule.
fn validate(
    dimensions: u32,
    population_size: Option<u32>,
    generations: u32,
    sigma0: f64,
    restarts: u32,
) -> Result<CmaEsBatchSchedule, OptimizerError> {
    let invalid = |parameter: &'static str, reason: String| {
        Err(OptimizerError::InvalidConfig { parameter, reason })
    };
    if dimensions == 0 {
        return invalid(
            "dimensions",
            "must be at least 1 (CMA-ES's learning rates are undefined in zero dimensions)"
                .to_string(),
        );
    }
    if generations == 0 {
        return invalid(
            "generations",
            "must be at least 1 (CMA-ES has no initial batch, so 0 generations never \
             evaluates the objective)"
                .to_string(),
        );
    }
    if !sigma0.is_finite() || sigma0 <= 0.0 || sigma0 > AlgorithmCmaEs::MAX_SIGMA0 {
        return invalid(
            "sigma0",
            format!("must be finite and in (0, 2π] (radians), got {sigma0}"),
        );
    }
    if let Some(size) = population_size {
        if size < AlgorithmCmaEs::MIN_POPULATION {
            return invalid(
                "population_size",
                format!(
                    "must be at least {} (CMA-ES recombines the best ⌊λ/2⌋ candidates), got {size}",
                    AlgorithmCmaEs::MIN_POPULATION
                ),
            );
        }
    }
    let schedule = CmaEsBatchSchedule::new(dimensions, population_size, generations, restarts);
    if schedule.population(restarts) > u64::from(u32::MAX) {
        return invalid(
            "restarts",
            format!(
                "{restarts} IPOP restarts would double the population from {} past {}",
                schedule.initial_population(),
                u32::MAX
            ),
        );
    }
    Ok(schedule)
}

/// Hansen's default strategy parameters for population size `lambda` in `n`
/// dimensions (tutorial Table 1, positive recombination weights only).
#[derive(Debug, Clone)]
struct Strategy {
    lambda: usize,
    /// Recombination weights of the μ = ⌊λ/2⌋ best candidates: positive,
    /// decreasing, summing to 1.
    weights: Vec<f64>,
    /// Variance-effective selection mass, `1 / Σ wᵢ²`.
    mueff: f64,
    /// Decay rate of the step-size path p_σ.
    cs: f64,
    /// Damping of the step-size update.
    ds: f64,
    /// Decay rate of the rank-one path p_c.
    cc: f64,
    /// Learning rate of the rank-one covariance update.
    c1: f64,
    /// Learning rate of the rank-μ covariance update.
    cmu: f64,
    /// `E‖N(0, I)‖`, approximated as in the tutorial.
    chi_n: f64,
    /// Generations between two eigendecompositions of `C`.
    eigen_gap: usize,
}

impl Strategy {
    fn new(n: usize, lambda: usize) -> Self {
        let nf = n as f64;
        let mu = lambda / 2;
        // w'ᵢ = ln((λ + 1)/2) − ln i, positive for i ≤ μ, normalised to sum 1.
        let half = (lambda as f64 + 1.0) / 2.0;
        let raw: Vec<f64> = (1..=mu).map(|i| half.ln() - (i as f64).ln()).collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let c1 = ALPHA_COV / ((nf + 1.3).powi(2) + mueff);
        // The 1/4 offset keeps a small rank-μ update when μ_eff = 1 (λ ≤ 3), as
        // in pycma (`rankmu_offset`).
        let cmu = (1.0 - c1).min(
            ALPHA_COV * (0.25 + mueff + 1.0 / mueff - 2.0)
                / ((nf + 2.0).powi(2) + ALPHA_COV * mueff / 2.0),
        );
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        // Hansen's lazy update: the eigensystem changes little over
        // 1/(10·n·(c₁ + c_μ)) generations. (As a count of evaluations this is
        // λ/(10·n·(c₁ + c_μ)).)
        let eigen_gap = ((1.0 / (10.0 * nf * (c1 + cmu))).floor() as usize).max(1);

        Self {
            lambda,
            weights,
            mueff,
            cs,
            ds,
            cc,
            c1,
            cmu,
            chi_n,
            eigen_gap,
        }
    }
}

/// Why a run ended.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StopReason {
    /// The run's best fitness improved by less than `tolerance` over its last
    /// `patience` generations.
    Stagnation,
    /// The mean, the step size, a path or the covariance is not finite.
    NonFinite,
    /// The condition number of `C` exceeds `MAX_CONDITION` (infinite when `C`
    /// is no longer positive definite).
    IllConditioned(f64),
    /// The largest coordinate standard deviation is below `TOL_X·σ₀`.
    Collapsed(f64),
    /// The largest coordinate standard deviation is above `TOL_X_UP·σ₀`.
    Diverged(f64),
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopReason::Stagnation => write!(f, "fitness stagnation"),
            StopReason::NonFinite => write!(f, "a non-finite search distribution"),
            StopReason::IllConditioned(condition) => write!(
                f,
                "an ill-conditioned covariance (condition number {condition:.3e} > {MAX_CONDITION:e})"
            ),
            StopReason::Collapsed(spread) => write!(
                f,
                "a collapsed search distribution (TolX: largest coordinate std {spread:.3e} < \
                 {TOL_X:e}·sigma0)"
            ),
            StopReason::Diverged(spread) => write!(
                f,
                "a diverging search distribution (TolXUp: largest coordinate std {spread:.3e} > \
                 {TOL_X_UP:e}·sigma0)"
            ),
        }
    }
}

/// The search distribution `N(m, σ²C)` of one run and its evolution paths.
struct Distribution {
    /// The mean `m`, unwrapped.
    mean: Array1<f64>,
    /// The step size σ.
    sigma: f64,
    /// The covariance matrix `C`, kept exactly symmetric.
    cov: Array2<f64>,
    /// `B`: the eigenvectors of `C` at the last decomposition, as columns.
    basis: Array2<f64>,
    /// `D`: the square roots of the matching eigenvalues.
    scales: Array1<f64>,
    /// `C^(-1/2) = B·D⁻¹·Bᵀ`, from the same decomposition.
    inv_sqrt: Array2<f64>,
    /// The step-size evolution path p_σ.
    path_sigma: Array1<f64>,
    /// The rank-one evolution path p_c.
    path_c: Array1<f64>,
    /// Generations completed in this run (the tutorial's `g`).
    generation: usize,
    /// `generation` at the last eigendecomposition.
    decomposed_at: usize,
    /// The condition number of `C` at the last decomposition, when it was out
    /// of bounds.
    ill_conditioned: Option<f64>,
    /// The cap on `σ·max_i √C_ii`: `max(σ₀, MAX_SPREAD)`.
    max_spread: f64,
}

impl Distribution {
    /// A run's initial distribution: `N(mean, σ₀²·I)`, with zero paths.
    /// `sigma` is σ₀.
    fn new(mean: Array1<f64>, sigma: f64) -> Self {
        let n = mean.len();
        Self {
            mean,
            sigma,
            cov: Array2::eye(n),
            basis: Array2::eye(n),
            scales: Array1::ones(n),
            inv_sqrt: Array2::eye(n),
            path_sigma: Array1::zeros(n),
            path_c: Array1::zeros(n),
            generation: 0,
            decomposed_at: 0,
            ill_conditioned: None,
            max_spread: sigma.max(MAX_SPREAD),
        }
    }

    /// λ steps `y_k = B·D·z_k ~ N(0, C)`, `z_k ~ N(0, I)`; candidate `k` is
    /// `m + σ·y_k`.
    fn sample<R: Rng>(&self, strategy: &Strategy, rng: &mut R) -> Vec<Array1<f64>> {
        let n = self.mean.len();
        (0..strategy.lambda)
            .map(|_| {
                let z: Array1<f64> = (0..n).map(|_| standard_normal(rng)).collect();
                self.basis.dot(&(&self.scales * &z))
            })
            .collect()
    }

    /// The candidate `m + σ·y`, wrapped into `[0, 2π)` for the oracle.
    fn candidate(&self, step: &Array1<f64>) -> Vec<f64> {
        self.mean
            .iter()
            .zip(step)
            .map(|(m, y)| wrap_angle(m + self.sigma * y))
            .collect()
    }

    /// One generation's update (tutorial Fig. 6, eqs. 41–47) from the sampled
    /// `steps` and their `order`, best first.
    fn update(&mut self, s: &Strategy, steps: &[Array1<f64>], order: &[usize]) {
        let n = self.mean.len();
        let nf = n as f64;
        // Selected steps, best first, with their weights (μ of them).
        let selected: Vec<(f64, &Array1<f64>)> = s
            .weights
            .iter()
            .zip(order)
            .map(|(&w, &k)| (w, &steps[k]))
            .collect();

        // ⟨y⟩_w and the mean (c_m = 1).
        let mut step = Array1::<f64>::zeros(n);
        for &(w, y) in &selected {
            step.scaled_add(w, y);
        }
        self.mean.scaled_add(self.sigma, &step);

        // Cumulative step-size adaptation: p_σ ← (1 − c_σ)·p_σ +
        // √(c_σ(2 − c_σ)μ_eff)·C^(-1/2)·⟨y⟩_w.
        self.path_sigma *= 1.0 - s.cs;
        self.path_sigma.scaled_add(
            (s.cs * (2.0 - s.cs) * s.mueff).sqrt(),
            &self.inv_sqrt.dot(&step),
        );
        let ps_norm = self.path_sigma.dot(&self.path_sigma).sqrt();

        // h_σ stalls the rank-one path while ‖p_σ‖ is large, i.e. while σ grows
        // fast; the denominator corrects the path's start-up bias.
        let start_up = 1.0 - (1.0 - s.cs).powf(2.0 * (self.generation as f64 + 1.0));
        let h_sigma = ps_norm / start_up.sqrt() < (1.4 + 2.0 / (nf + 1.0)) * s.chi_n;
        self.path_c *= 1.0 - s.cc;
        if h_sigma {
            self.path_c
                .scaled_add((s.cc * (2.0 - s.cc) * s.mueff).sqrt(), &step);
        }

        // C ← (1 + c₁δ(h_σ) − c₁ − c_μ)·C + c₁·p_c·p_cᵀ + c_μ·Σ wᵢ·yᵢ·yᵢᵀ, the
        // weights summing to 1. `decay` is ≥ 0 up to rounding (c_μ ≤ 1 − c₁).
        let delta = if h_sigma { 0.0 } else { s.cc * (2.0 - s.cc) };
        let decay = (1.0 + s.c1 * delta - s.c1 - s.cmu).max(0.0);
        for i in 0..n {
            for j in i..n {
                let rank_mu: f64 = selected.iter().map(|&(w, y)| w * y[i] * y[j]).sum();
                let c = decay * self.cov[[i, j]]
                    + s.c1 * self.path_c[i] * self.path_c[j]
                    + s.cmu * rank_mu;
                self.cov[[i, j]] = c;
                self.cov[[j, i]] = c;
            }
        }

        // σ ← σ·exp((c_σ/d_σ)·(‖p_σ‖/E‖N(0, I)‖ − 1)).
        self.sigma *= ((s.cs / s.ds) * (ps_norm / s.chi_n - 1.0)).exp();
        // Cap the widest coordinate (`AlgorithmCmaEs`, `# Angles`). A non-finite
        // spread is left to `degeneration`.
        let spread = self.sigma * self.max_coordinate_std();
        if spread > self.max_spread {
            self.sigma *= self.max_spread / spread;
        }

        self.generation += 1;
        if self.generation - self.decomposed_at >= s.eigen_gap {
            self.decompose();
        }
    }

    /// Recompute `B`, `D` and `C^(-1/2)` from `C`. A covariance that is not
    /// finite keeps the previous ones ([`degeneration`](Self::degeneration)
    /// reports it); one whose condition number is out of bounds (infinite when
    /// an eigenvalue is not positive, or the decomposition not finite) keeps
    /// them too and is recorded in `ill_conditioned`.
    fn decompose(&mut self) {
        self.decomposed_at = self.generation;
        let Some(eig) = symmetric_eigen(&self.cov) else {
            return;
        };
        let (low, high) = eig
            .values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let finite = eig.values.iter().chain(&eig.vectors).all(|v| v.is_finite());
        let condition = if finite && low > 0.0 {
            high / low
        } else {
            f64::INFINITY
        };
        if condition > MAX_CONDITION {
            self.ill_conditioned = Some(condition);
            return;
        }
        self.scales = eig.values.mapv(f64::sqrt);
        // B·D⁻¹ divides column j of B by D_j; times Bᵀ gives C^(-1/2).
        self.inv_sqrt = (&eig.vectors / &self.scales).dot(&eig.vectors.t());
        self.basis = eig.vectors;
    }

    /// `max_i √C_ii`, the largest coordinate standard deviation per unit σ.
    fn max_coordinate_std(&self) -> f64 {
        self.cov
            .diag()
            .iter()
            .fold(0.0_f64, |m, &c| m.max(c))
            .sqrt()
    }

    /// The first of Hansen's termination criteria that holds, if any.
    fn degeneration(&self, sigma0: f64) -> Option<StopReason> {
        let finite = self.sigma.is_finite()
            && self
                .mean
                .iter()
                .chain(&self.path_sigma)
                .chain(&self.path_c)
                .chain(&self.cov)
                .all(|x| x.is_finite());
        if !finite {
            return Some(StopReason::NonFinite);
        }
        if let Some(condition) = self.ill_conditioned {
            return Some(StopReason::IllConditioned(condition));
        }
        // σ·max_i √C_ii: the largest coordinate standard deviation.
        let spread = self.sigma * self.max_coordinate_std();
        if spread < TOL_X * sigma0 {
            return Some(StopReason::Collapsed(spread));
        }
        if spread > TOL_X_UP * sigma0 {
            return Some(StopReason::Diverged(spread));
        }
        None
    }
}

/// The best candidate seen so far and its fitness. The first offer is always
/// taken (so a run that only ever sees `NaN` still returns a scored
/// candidate); later ones only when strictly better in [`rank_order`], so ties
/// keep the earlier candidate.
struct Incumbent {
    fitness: f64,
    params: Vec<f64>,
    evaluated: bool,
}

impl Incumbent {
    fn new() -> Self {
        Self {
            fitness: f64::NEG_INFINITY,
            params: Vec::new(),
            evaluated: false,
        }
    }

    fn offer(&mut self, fitness: f64, params: &[f64]) {
        if !self.evaluated || rank_order(fitness, self.fitness) == Ordering::Less {
            self.fitness = fitness;
            self.params = params.to_vec();
            self.evaluated = true;
        }
    }
}

/// The order CMA-ES ranks fitness values in, best first: by negated fitness
/// ascending (CMA-ES minimises `-fitness`), with `NaN` after every number,
/// `-∞` included. A total order: `Less` means `a` is the better value, and
/// `NaN`s, like `0.0` and `-0.0`, tie.
fn rank_order(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // Neither is NaN, so the comparison is defined.
        (false, false) => (-a).partial_cmp(&-b).unwrap_or(Ordering::Equal),
    }
}

/// Candidate indices from best (highest fitness) to worst in [`rank_order`].
/// Ties keep their sampling order (a stable sort), so the ranking is a
/// deterministic function of the fitness order alone and never panics.
fn ranking(fitness: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..fitness.len()).collect();
    order.sort_by(|&a, &b| rank_order(fitness[a], fitness[b]));
    order
}

/// A standard normal draw by the Box–Muller transform (the crate has no
/// `rand_distr`). `1 − u` lies in `(0, 1]`, so the logarithm is finite.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u: f64 = rng.random();
    let v: f64 = rng.random();
    (-2.0 * (1.0 - u).ln()).sqrt() * (TAU * v).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::with_seeded_rng;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn default_population_size_matches_hansen() {
        // 4 + ⌊3 ln d⌋, as in pycma's `4 + int(3 * log(N))`.
        let expected = [
            (1, 4),
            (2, 6),
            (3, 7),
            (6, 9),
            (10, 10),
            (20, 12),
            (100, 17),
        ];
        for (d, lambda) in expected {
            assert_eq!(
                AlgorithmCmaEs::default_population_size(d),
                lambda,
                "d = {d}"
            );
        }
        assert_eq!(AlgorithmCmaEs::default_population_size(0), 4);
        assert_eq!(AlgorithmCmaEs::default_population_size(u32::MAX), 70);
    }

    #[test]
    fn strategy_parameters_match_the_tutorial_defaults() {
        // Reference values for d = 10, λ = 10, computed in Python from the
        // tutorial's Table 1 formulas. The weights, μ_eff, c_c, c₁ and c_μ are
        // also exactly pycma 4.5.0's (`CMA_active=False`); its c_σ and d_σ use
        // a variant of the tutorial's.
        let s = Strategy::new(10, 10);
        assert_eq!(s.weights.len(), 5);
        let close = |got: f64, want: f64| (got - want).abs() < 1e-14 * want.abs().max(1.0);
        let want_weights = [
            0.456_272_646_903_405_85,
            0.270_753_097_001_785_1,
            0.162_231_117_158_669_75,
            0.085_233_547_100_164_45,
            0.025_509_591_835_974_77,
        ];
        for (w, want) in s.weights.iter().zip(want_weights) {
            assert!(close(*w, want), "weight {w} vs {want}");
        }
        assert!(close(s.weights.iter().sum::<f64>(), 1.0));
        assert!(close(s.mueff, 3.167_299_281_410_703_5), "mueff {}", s.mueff);
        assert!(close(s.cs, 0.284_428_587_946_367_5), "cs {}", s.cs);
        assert!(close(s.ds, 1.284_428_587_946_367_5), "ds {}", s.ds);
        assert!(close(s.cc, 0.294_990_383_035_622_25), "cc {}", s.cc);
        assert!(close(s.c1, 0.015_283_824_524_751_714), "c1 {}", s.c1);
        assert!(close(s.cmu, 0.023_551_776_650_417_49), "cmu {}", s.cmu);
        assert!(close(s.chi_n, 3.084_726_565_169_012_3), "chi_n {}", s.chi_n);
        // 1/(10·10·(c₁ + c_μ)) ≈ 0.26 → every generation.
        assert_eq!(s.eigen_gap, 1);
    }

    #[test]
    fn strategy_parameters_stay_in_range() {
        for n in [1, 2, 3, 6, 20, 100, 1000] {
            for lambda in [2, 3, 4, 9, 30, 300, 3000] {
                let s = Strategy::new(n, lambda);
                assert_eq!(s.weights.len(), lambda / 2);
                assert!(s.weights.windows(2).all(|w| w[0] > w[1]), "{:?}", s.weights);
                assert!(s.weights.iter().all(|&w| w > 0.0));
                assert!((s.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!((1.0..=(lambda / 2) as f64 + 1e-9).contains(&s.mueff));
                assert!(s.cs > 0.0 && s.cs < 1.0);
                assert!(s.ds >= 1.0);
                assert!(s.cc > 0.0 && s.cc <= 1.0);
                assert!(s.c1 > 0.0 && s.cmu > 0.0);
                assert!(s.c1 + s.cmu <= 1.0 + 1e-15, "n {n} λ {lambda}");
                assert!(s.eigen_gap >= 1);
            }
        }
    }

    #[test]
    fn eigen_gap_grows_with_dimension() {
        // Hansen's max(1, ⌊1/(10·n·(c₁ + c_μ))⌋): every generation for the few
        // parameters of a circuit, sparser for hundreds.
        let gap = |n: usize| {
            Strategy::new(
                n,
                AlgorithmCmaEs::default_population_size(n as u32) as usize,
            )
            .eigen_gap
        };
        assert_eq!(gap(6), 1);
        assert_eq!(gap(50), 1);
        assert!(gap(200) >= 2, "{}", gap(200));
        assert!(gap(1000) > gap(200));
        let s = Strategy::new(200, 19);
        assert_eq!(
            s.eigen_gap,
            (1.0 / (10.0 * 200.0 * (s.c1 + s.cmu))).floor() as usize
        );
    }

    #[test]
    fn ranking_puts_nan_last_and_keeps_ties_in_order() {
        let fitness = [
            1.0,
            f64::NAN,
            3.0,
            f64::NEG_INFINITY,
            3.0,
            f64::INFINITY,
            f64::NAN,
        ];
        assert_eq!(ranking(&fitness), vec![5, 2, 4, 0, 3, 1, 6]);
        // Signed zeros tie.
        assert_eq!(ranking(&[-0.0, 0.0]), vec![0, 1]);
        assert_eq!(ranking(&[0.0, -0.0]), vec![0, 1]);
        assert!(ranking(&[]).is_empty());
    }

    #[test]
    fn ranking_is_invariant_to_increasing_transformations() {
        let fitness: [f64; 6] = [0.3, -1.2, 2.5, 0.31, -7.0, 1.0];
        let transformed: Vec<f64> = fitness.iter().map(|f| f.exp() * 3.0 - 1.0).collect();
        assert_eq!(ranking(&fitness), ranking(&transformed));
    }

    #[test]
    fn incumbent_takes_the_first_offer_then_only_strict_improvements() {
        let mut best = Incumbent::new();
        best.offer(f64::NAN, &[1.0]);
        assert!(best.fitness.is_nan());
        assert_eq!(best.params, vec![1.0]);
        best.offer(f64::NEG_INFINITY, &[2.0]);
        assert_eq!(best.fitness, f64::NEG_INFINITY);
        best.offer(f64::NAN, &[3.0]);
        assert_eq!(best.params, vec![2.0]);
        best.offer(0.5, &[4.0]);
        best.offer(0.5, &[5.0]);
        assert_eq!((best.fitness, best.params.clone()), (0.5, vec![4.0]));
        best.offer(0.25, &[6.0]);
        assert_eq!(best.params, vec![4.0]);
    }

    #[test]
    fn standard_normal_has_unit_moments() {
        let draws: Vec<f64> = with_seeded_rng(Some(3), |rng| {
            (0..200_000).map(|_| standard_normal(rng)).collect()
        });
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let kurt = draws.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n / (var * var);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "var {var}");
        assert!((kurt - 3.0).abs() < 0.05, "kurtosis {kurt}");
        assert!(draws.iter().all(|x| x.is_finite()));
    }

    /// Run one CMA-ES run (no restarts, no stop criteria) on `fitness` for
    /// `generations` generations with `strategy`, returning the best fitness
    /// and the final distribution.
    fn drive(
        fitness: impl Fn(&[f64]) -> f64,
        strategy: &Strategy,
        mean: Array1<f64>,
        sigma0: f64,
        generations: usize,
        seed: u64,
    ) -> (f64, Distribution) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dist = Distribution::new(mean, sigma0);
        let mut best = Incumbent::new();
        for _ in 0..generations {
            let steps = dist.sample(strategy, &mut rng);
            let values: Vec<f64> = steps.iter().map(|y| fitness(&dist.candidate(y))).collect();
            for (y, &v) in steps.iter().zip(&values) {
                best.offer(v, &dist.candidate(y));
            }
            dist.update(strategy, &steps, &ranking(&values));
        }
        (best.fitness, dist)
    }

    /// The ellipsoid `−Σ 10^(6i/(d−1))·(xᵢ − π)²`, Hessian condition 1e6.
    fn ellipsoid(x: &[f64]) -> f64 {
        let d = x.len();
        -x.iter()
            .enumerate()
            .map(|(i, xi)| {
                10f64.powf(6.0 * i as f64 / (d - 1) as f64) * (xi - std::f64::consts::PI).powi(2)
            })
            .sum::<f64>()
    }

    #[test]
    fn covariance_adaptation_learns_the_ellipsoid() {
        // On an ellipsoid with Hessian condition 1e6, CMA-ES learns C ∝ H⁻¹:
        // after 400 generations cond(C) is about 1e6 and the run is at the
        // optimum to 1e-15 (it reaches 1e-20). The same run with the covariance
        // frozen (c₁ = c_μ = 0, a plain CSA evolution strategy) is still at
        // about -9 then: the covariance adaptation is what solves the
        // ill-conditioning.
        let d = 6;
        let mean = Array1::from_elem(d, std::f64::consts::PI + 0.5);
        let strategy = Strategy::new(
            d,
            AlgorithmCmaEs::default_population_size(d as u32) as usize,
        );
        let (adapted, dist) = drive(ellipsoid, &strategy, mean.clone(), 0.3, 400, 11);
        assert!(adapted > -1e-15, "CMA-ES best {adapted:e}");
        let eig = symmetric_eigen(&dist.cov).expect("finite covariance");
        let (lo, hi) = eig
            .values
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(l, h), &v| (l.min(v), h.max(v)));
        let condition = hi / lo;
        assert!((3e5..3e6).contains(&condition), "cond(C) = {condition:e}");

        let frozen = Strategy {
            c1: 0.0,
            cmu: 0.0,
            ..strategy.clone()
        };
        let (isotropic, dist) = drive(ellipsoid, &frozen, mean, 0.3, 400, 11);
        assert_eq!(dist.cov, Array2::<f64>::eye(d));
        assert!(isotropic < -1.0, "isotropic best {isotropic:e}");
    }

    #[test]
    fn step_size_grows_on_a_linear_slope_and_shrinks_at_an_optimum() {
        // CSA: on a (locally) linear function selection keeps pointing the
        // same way, so ‖p_σ‖ exceeds its expectation and σ grows; at the
        // optimum of a sphere σ shrinks.
        let strategy = Strategy::new(4, 8);
        let (_, dist) = drive(
            |x| x[0] * 1e-3,
            &strategy,
            Array1::from_elem(4, 3.0),
            1e-3,
            30,
            5,
        );
        assert!(dist.sigma > 1e-2, "sigma {}", dist.sigma);
        let sphere = |x: &[f64]| -x.iter().map(|v| (v - 3.0).powi(2)).sum::<f64>();
        let (_, dist) = drive(sphere, &strategy, Array1::from_elem(4, 3.0), 1.0, 150, 5);
        assert!(dist.sigma < 1e-4, "sigma {}", dist.sigma);
    }

    #[test]
    fn step_size_stays_within_a_quarter_turn_at_a_fitness_minimum() {
        // Started at the minimum of the periodic sphere (the 0/2π seam), the
        // longest steps rank best, so CSA grows σ every generation. The cap
        // keeps σ·max√C_ii at π/2, and once the mean leaves the minimum the
        // run converges to the maximum at π.
        let strategy = Strategy::new(2, 6);
        let sphere = |x: &[f64]| {
            -x.iter()
                .map(|v| (v - std::f64::consts::PI).powi(2))
                .sum::<f64>()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let mut dist = Distribution::new(Array1::zeros(2), 0.3);
        let mut capped = false;
        for _ in 0..200 {
            let steps = dist.sample(&strategy, &mut rng);
            let values: Vec<f64> = steps.iter().map(|y| sphere(&dist.candidate(y))).collect();
            dist.update(&strategy, &steps, &ranking(&values));
            let spread = dist.sigma * dist.max_coordinate_std();
            assert!(spread <= FRAC_PI_2 * (1.0 + 1e-12), "spread {spread}");
            capped |= spread > FRAC_PI_2 * (1.0 - 1e-12);
        }
        assert!(capped, "the run never reached the cap");
        let x: Vec<f64> = dist.mean.iter().map(|&m| wrap_angle(m)).collect();
        assert!(sphere(&x) > -1e-12, "mean {x:?}");

        // A wider σ₀ is its own cap.
        assert_eq!(Distribution::new(Array1::zeros(2), 3.0).max_spread, 3.0);
        assert_eq!(
            Distribution::new(Array1::zeros(2), 0.1).max_spread,
            FRAC_PI_2
        );
    }

    #[test]
    fn decomposition_flags_an_ill_conditioned_covariance() {
        let mut dist = Distribution::new(Array1::zeros(3), 1.0);
        dist.cov = Array2::from_diag(&Array1::from(vec![1.0, 1e-3, 1e-15]));
        dist.decompose();
        let Some(StopReason::IllConditioned(condition)) = dist.degeneration(1.0) else {
            panic!("expected an ill-conditioned covariance");
        };
        assert!((condition - 1e15).abs() < 1e3, "{condition:e}");
        // The sampling basis is left as it was.
        assert_eq!(dist.basis, Array2::<f64>::eye(3));
    }

    #[test]
    fn degeneration_reports_each_criterion() {
        let healthy = || Distribution::new(Array1::from(vec![1.0, 2.0]), 0.5);
        assert_eq!(healthy().degeneration(0.5), None);

        let mut d = healthy();
        d.mean[1] = f64::NAN;
        assert_eq!(d.degeneration(0.5), Some(StopReason::NonFinite));
        let mut d = healthy();
        d.cov[[0, 1]] = f64::INFINITY;
        assert_eq!(d.degeneration(0.5), Some(StopReason::NonFinite));
        let mut d = healthy();
        d.sigma = f64::INFINITY;
        assert_eq!(d.degeneration(0.5), Some(StopReason::NonFinite));

        let mut d = healthy();
        d.sigma = 1e-13;
        assert!(matches!(
            d.degeneration(0.5),
            Some(StopReason::Collapsed(_))
        ));
        let mut d = healthy();
        d.sigma = 1e4;
        assert!(matches!(d.degeneration(0.5), Some(StopReason::Diverged(_))));
        // The spread is the largest coordinate's: one wide coordinate keeps a
        // distribution that is collapsed in the other from TolX.
        let mut d = healthy();
        d.sigma = 1e-13;
        d.cov[[1, 1]] = 1e10;
        assert_eq!(d.degeneration(0.5), None);
    }

    #[test]
    fn batch_schedule_bounds() {
        let s = CmaEsBatchSchedule::new(6, None, 5, 0);
        assert_eq!(s.initial_population(), 9);
        assert_eq!(s.max_evaluations(), 45);
        assert_eq!(s.max_population(), 9);
        assert!((0..5).all(|g| s.max_batch_size(g) == 9));

        // More restarts than generations: the bound doubles every generation.
        let s = CmaEsBatchSchedule::new(1, Some(3), 4, 10);
        let sizes: Vec<u64> = (0..5).map(|g| s.max_batch_size(g)).collect();
        assert_eq!(sizes, vec![3, 6, 12, 24, 0]);
        assert_eq!(s.max_evaluations(), 45);
        assert_eq!(s.max_population(), 24);

        // The closed form agrees with the sum of the per-generation bounds.
        for (pop, gens, restarts) in [(9, 100, 2), (4, 1, 0), (2, 7, 7), (17, 30, 3), (5, 3, 9)] {
            let s = CmaEsBatchSchedule::new(1, Some(pop), gens, restarts);
            let sum: u64 = (0..gens).map(|g| s.max_batch_size(g)).sum();
            assert_eq!(s.max_evaluations(), sum, "{pop} {gens} {restarts}");
        }

        // No generations, no batches.
        let s = CmaEsBatchSchedule::new(3, None, 0, 4);
        assert_eq!((s.max_evaluations(), s.max_population()), (0, 0));

        // Unvalidated extremes saturate instead of overflowing.
        let s = CmaEsBatchSchedule::new(1, Some(u32::MAX), u32::MAX, u32::MAX);
        assert_eq!(s.population(64), u64::MAX);
        assert_eq!(s.max_evaluations(), u64::MAX);
        assert_eq!(s.max_population(), u64::MAX);
    }

    #[test]
    fn stop_reasons_read_well_in_the_log() {
        assert_eq!(StopReason::Stagnation.to_string(), "fitness stagnation");
        assert!(StopReason::IllConditioned(2e14)
            .to_string()
            .contains("condition number 2.000e14 > 1e14"));
        assert!(StopReason::Collapsed(1e-13).to_string().contains("TolX"));
        assert!(StopReason::Diverged(1e5).to_string().contains("TolXUp"));
    }
}
