//! Adam optimizer (adaptive moment estimation).

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{check_initial_point, check_oracle_len, fitness_stagnated};
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
/// # Preconditions
///
/// Checked before any RNG draw or oracle call
/// ([`OptimizerError::InvalidConfig`] or [`OptimizerError::InvalidBounds`]):
/// `max_iters >= 1` (Adam evaluates nothing before its first step, so zero
/// iterations could not report a real `best_fitness`), `dimensions >= 1`,
/// `learning_rate` finite and `> 0`, `beta1` and `beta2` in `[0, 1)` (at `1`
/// the bias correction divides by zero), `epsilon` finite and `> 0`,
/// `bounds` finite with `lb < ub` (the initial `θ` is drawn uniformly from it),
/// and `initial_point`, when given, `dimensions` finite values.
pub struct AlgorithmAdamArgs {
    /// Maps parameter vectors to fitness values (higher is better). Called once
    /// per iteration, on the updated `θ`, to track the best solution.
    pub oracle: Box<dyn EvaluationOracle>,
    /// `∇fitness(θ)`, uphill (see [`GradientOracle`]); the update adds it.
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
    /// The first gradient is taken at it. `None` draws `θ` uniformly from
    /// `bounds`.
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

    /// Check the preconditions of [`AlgorithmAdamArgs`] that
    /// [`optimize`](Optimizer::optimize) checks before any RNG draw or oracle
    /// call, in the same order, without running anything: the configuration
    /// listed on [`AlgorithmAdamArgs`], `bounds`, then `initial_point`
    /// (`dimensions` finite values, inside `bounds` or not). `train` and
    /// `qml.train` call it before any backend exists.
    pub fn check_preconditions(
        max_iters: u32,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        bounds: (f64, f64),
        dimensions: u32,
        initial_point: Option<&[f64]>,
    ) -> Result<(), OptimizerError> {
        validate(max_iters, dimensions, learning_rate, beta1, beta2, epsilon)?;
        let (lb, ub) = bounds;
        // The same check as QNG's: `lb >= ub` is empty, and a non-finite bound
        // (`NaN` fails every comparison, an infinity would make the sampler panic)
        // cannot be drawn from.
        if !lb.is_finite() || !ub.is_finite() || lb >= ub {
            return Err(OptimizerError::InvalidBounds { lb, ub });
        }
        check_initial_point(initial_point, dimensions as usize, None)
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

        Self::check_preconditions(
            max_iters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bounds,
            dimensions,
            initial_point.as_deref(),
        )?;
        let (lb, ub) = bounds;

        let dims = dimensions as usize;
        let patience = patience as usize;
        // Draw θ uniformly in [lb, ub), or start at the caller's point: θ is
        // drawn either way, so the RNG stream does not depend on it.
        let drawn: Vec<f64> = (0..dims).map(|_| rng.random_range(lb..ub)).collect();
        let mut theta = initial_point.unwrap_or(drawn);
        let mut m = vec![0.0_f64; dims];
        let mut v = vec![0.0_f64; dims];
        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_theta = theta.clone();
        let mut iterations_run = 0usize;
        let mut converged = false;
        // Best fitness after each iteration (see
        // `OptimizationOutcome::fitness_history`).
        let mut fitness_history: Vec<f64> = Vec::with_capacity(max_iters as usize);

        for iteration in 0..max_iters as usize {
            iterations_run = iteration + 1;

            // ── 1. Gradient, uphill, one component per parameter ──────────────
            let grad = gradient_oracle.gradient_batch(&theta, dims);
            check_oracle_len(dims, grad.len())?;
            // A non-finite component would leave θ non-finite from here on (C-5).
            if let Some((index, &value)) = grad.iter().enumerate().find(|(_, g)| !g.is_finite()) {
                return Err(OptimizerError::NonFiniteGradient { index, value });
            }

            // ── 2. Bias-corrected moment update (t is 1-based) ────────────────
            let t = iteration as i32 + 1;
            let bias1 = 1.0 - beta1.powi(t);
            let bias2 = 1.0 - beta2.powi(t);
            for i in 0..dims {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                theta[i] += learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }

            // ── 3. Score the new θ and track the best solution ────────────────
            let fitness = oracle.evaluate_batch(std::slice::from_ref(&theta));
            check_oracle_len(1, fitness.len())?;
            let fitness = fitness[0];
            log::debug!("Iteration {iteration}: fitness {fitness:.4}");
            if fitness > best_fitness {
                best_fitness = fitness;
                best_theta = theta.clone();
            }
            // The running best, not this iteration's fitness: a gradient step can
            // overshoot, and the history must be non-decreasing with the reported
            // best as its last entry (C-5).
            fitness_history.push(best_fitness);

            // ── 4. Stop when the best fitness stagnates (C-5) ─────────────────
            if fitness_stagnated(&fitness_history, tolerance, patience, iteration) {
                converged = true;
                break;
            }
        }

        Ok(OptimizationOutcome {
            best_params: best_theta,
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
