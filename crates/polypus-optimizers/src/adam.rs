//! Adam optimizer (adaptive moment estimation).

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{check_initial_point, check_oracle_len, fitness_stagnated, BestMeasured};
use rand::{Rng, RngExt};

/// Adam optimizer (Kingma and Ba, 2015).
pub struct AlgorithmAdam;

/// Arguments for the Adam optimizer.
///
/// Like [`AlgorithmQNG`](crate::AlgorithmQNG), Adam is decoupled from circuits
/// and infrastructure: the gradient comes from
/// [`gradient_oracle`](AlgorithmAdamArgs::gradient_oracle) and the fitness used
/// to track the best solution from [`oracle`](AlgorithmAdamArgs::oracle). It
/// needs no [`VarianceOracle`](crate::VarianceOracle): its per-parameter step
/// size comes from running averages of the gradient and of its square.
///
/// Each iteration measures the current `θ` and the gradient there, then steps.
/// When the gradient oracle scores `θ` in its own batch
/// ([`GradientOracle::value_and_gradient`]), that is one call: with a
/// [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient), one batch of
/// `2·dimensions + 1` candidates, `θ` first. Otherwise `oracle` scores `θ`
/// and then the gradient oracle is asked for the gradient. `best_params` is
/// the measured `θ` with the best fitness (a `NaN` never wins), and
/// `fitness_history` holds the best measured up to each iteration; the step
/// after the last measurement is not measured.
///
/// # Preconditions
///
/// Checked before any RNG draw or oracle call
/// ([`OptimizerError::InvalidConfig`] or [`OptimizerError::InvalidBounds`]):
/// `max_iters >= 1` (each iteration measures `θ`, so zero iterations would
/// measure nothing and could not report a real `best_fitness`), `dimensions >= 1`,
/// `learning_rate` finite and `> 0`, `beta1` and `beta2` in `[0, 1)` (at `1`
/// the bias correction divides by zero), `epsilon` finite and `> 0`,
/// `bounds` finite with `lb < ub` (the initial `θ` is drawn uniformly from it),
/// and `initial_point`, when given, `dimensions` finite values.
pub struct AlgorithmAdamArgs {
    /// Maps parameter vectors to fitness values (higher is better). Scores `θ`
    /// once per iteration to track the best solution, unless the gradient
    /// oracle scores it.
    pub oracle: Box<dyn EvaluationOracle>,
    /// `∇fitness(θ)`, uphill (see [`GradientOracle`]); the update adds it. One
    /// that also returns `θ`'s fitness
    /// ([`value_and_gradient`](GradientOracle::value_and_gradient)) must take
    /// it from [`oracle`](AlgorithmAdamArgs::oracle), since Adam reports it as
    /// measured (C-5): a [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient)
    /// over an [`Arc`](std::sync::Arc) shared with it.
    pub gradient_oracle: Box<dyn GradientOracle>,
    pub max_iters: u32,
    /// Step size applied to the bias-corrected, moment-normalised update.
    pub learning_rate: f64,
    /// Decay rate of the first moment (running mean of the gradient).
    pub beta1: f64,
    /// Decay rate of the second moment (running mean of the squared gradient).
    pub beta2: f64,
    /// Added to the denominator of the update so it never divides by zero.
    pub epsilon: f64,
    pub bounds: (f64, f64),
    pub dimensions: u32,
    /// Minimum improvement of the best fitness over
    /// [`patience`](AlgorithmAdamArgs::patience) iterations below which the run
    /// ends, in the oracle's fitness units: the stagnation rule of contract C-5,
    /// the same as DE's and CMA-ES's.
    pub tolerance: f64,
    /// Iterations over which the best-fitness improvement is measured; no run
    /// ends on stagnation before it has run `patience + 1` iterations.
    pub patience: u32,
    /// Optional RNG seed. `None` (the default) uses [`rand::rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
    /// Optional initial `θ`, one value per dimension, all finite; it may lie
    /// outside `bounds`, which only shape the random draw (θ is never clamped).
    /// It is the first `θ` measured, and the first gradient is taken at it.
    /// `None` draws `θ` uniformly from `bounds`.
    pub initial_point: Option<Vec<f64>>,
}

impl AlgorithmAdam {
    /// The default `beta1`: the first-moment decay rate of the original paper.
    pub const DEFAULT_BETA1: f64 = 0.9;
    /// The default `beta2`: the second-moment decay rate of the original paper.
    pub const DEFAULT_BETA2: f64 = 0.999;
    /// The default `epsilon` of the original paper.
    pub const DEFAULT_EPSILON: f64 = 1e-8;

    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("Adam")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit with Adam: gradient ascent with a \
             per-parameter step size adapted from running averages of the gradient and its \
             square, with bias correction. Stops early when the best fitness stagnates.",
        )
    }

    fn run_with_rng<R: Rng>(
        args: AlgorithmAdamArgs,
        rng: &mut R,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmAdamArgs {
            oracle,
            gradient_oracle,
            max_iters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bounds,
            dimensions,
            tolerance,
            patience,
            seed: _,
            initial_point,
        } = args;

        validate(max_iters, dimensions, learning_rate, beta1, beta2, epsilon)?;
        let (lb, ub) = bounds;
        // The same check as QNG's: `lb >= ub` is empty, and a non-finite bound
        // (`NaN` fails every comparison, an infinity would make the sampler panic)
        // cannot be drawn from.
        if !lb.is_finite() || !ub.is_finite() || lb >= ub {
            return Err(OptimizerError::InvalidBounds { lb, ub });
        }

        let dims = dimensions as usize;
        let patience = patience as usize;
        check_initial_point(initial_point.as_deref(), dims)?;
        // Draw θ uniformly in [lb, ub), or start at the caller's point: θ is
        // drawn either way, so the RNG stream does not depend on it.
        let drawn: Vec<f64> = (0..dims).map(|_| rng.random_range(lb..ub)).collect();
        let mut theta = initial_point.unwrap_or(drawn);
        let mut m = vec![0.0_f64; dims];
        let mut v = vec![0.0_f64; dims];
        // β₁ᵗ and β₂ᵗ, one product each per iteration rather than `powi`, whose
        // rounding Rust leaves unspecified: with only +, −, ×, ÷ and `sqrt`,
        // all correctly rounded in IEEE 754, a seeded run reproduces bit for
        // bit on every platform (C-7).
        let mut beta1_t = 1.0_f64;
        let mut beta2_t = 1.0_f64;
        let mut best = BestMeasured::new();
        let mut iterations_run = 0usize;
        let mut converged = false;
        // Best fitness after each iteration (see
        // `OptimizationOutcome::fitness_history`).
        let mut fitness_history: Vec<f64> = Vec::with_capacity(max_iters as usize);

        for iteration in 0..max_iters as usize {
            iterations_run = iteration + 1;

            // ── 1. θ's fitness and the gradient there, uphill ─────────────────
            // One call when the gradient oracle scores θ in its own batch (a
            // finite-difference stencil); otherwise θ is scored here first.
            let (fitness, grad) = match gradient_oracle.value_and_gradient(&theta, dims) {
                Some(measured) => measured,
                None => {
                    let fitness = oracle.evaluate_batch(std::slice::from_ref(&theta));
                    check_oracle_len(1, fitness.len())?;
                    (fitness[0], gradient_oracle.gradient_batch(&theta, dims))
                }
            };
            check_oracle_len(dims, grad.len())?;
            log::debug!("Iteration {iteration}: fitness {fitness:.4}");

            // ── 2. Track the best measured θ ──────────────────────────────────
            best.offer(&theta, fitness);
            // The running best, not this iteration's fitness: a gradient step can
            // overshoot, and the history must be non-decreasing with the reported
            // best as its last entry (C-5).
            fitness_history.push(best.value());

            // ── 3. Bias-corrected moment update (t is 1-based) ────────────────
            beta1_t *= beta1;
            beta2_t *= beta2;
            let bias1 = 1.0 - beta1_t;
            let bias2 = 1.0 - beta2_t;
            for i in 0..dims {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                theta[i] += learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }

            // ── 4. Stop when the best fitness stagnates (C-5) ─────────────────
            if fitness_stagnated(&fitness_history, tolerance, patience, iteration) {
                converged = true;
                break;
            }
        }

        // `max_iters >= 1` and every iteration measures θ before it can end, so
        // `best` holds a measured θ.
        let (best_params, best_fitness) = best.into_parts();
        Ok(OptimizationOutcome {
            best_params,
            best_fitness,
            iterations_run,
            converged,
            fitness_history,
        })
    }
}

/// Reject a configuration Adam cannot run, before any RNG draw or oracle call.
fn validate(
    max_iters: u32,
    dimensions: u32,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
) -> Result<(), OptimizerError> {
    let invalid = |parameter: &'static str, reason: String| {
        Err(OptimizerError::InvalidConfig { parameter, reason })
    };
    if max_iters == 0 {
        return invalid(
            "max_iters",
            "must be at least 1 (0 iterations never evaluates the objective)".to_string(),
        );
    }
    if dimensions == 0 {
        return invalid(
            "dimensions",
            "must be at least 1 (there is nothing to optimise)".to_string(),
        );
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return invalid(
            "learning_rate",
            format!("must be finite and > 0, got {learning_rate}"),
        );
    }
    for (parameter, beta) in [("beta1", beta1), ("beta2", beta2)] {
        if !(0.0..1.0).contains(&beta) {
            return invalid(parameter, format!("must be in [0, 1), got {beta}"));
        }
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return invalid("epsilon", format!("must be finite and > 0, got {epsilon}"));
    }
    Ok(())
}

impl Optimizer for AlgorithmAdam {
    type Args = AlgorithmAdamArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
