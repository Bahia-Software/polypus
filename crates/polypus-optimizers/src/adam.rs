//! Adam optimizer (adaptive moment estimation), with Polyak's heavy-ball
//! momentum as an alternative update rule ([`AdamRule`]).
//!
//! References:
//!
//! - D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization",
//!   ICLR 2015, arXiv:1412.6980: the [`AdamRule::Adam`] update and its bias
//!   corrections.
//! - B. T. Polyak, "Some methods of speeding up the convergence of iteration
//!   methods", USSR Computational Mathematics and Mathematical Physics 4(5),
//!   1964: the heavy ball of [`AdamRule::Momentum`].
//! - B. D. M. Jones, L. Mineh and A. Montanaro, "Benchmarking a wide range of
//!   optimisers for solving the Fermi-Hubbard model using the variational
//!   quantum eigensolver", Quantum Sci. Technol. 10(4), 045032 (2025),
//!   arXiv:2411.13742: with finite-difference gradients under shot noise,
//!   momentum and Adam reach the best final precision, and the difference
//!   step matters (`0.4` works, `0.01` ruins the gradient). `polypus.Adam`
//!   defaults to a step of `0.2` and a `learning_rate` of `0.1`.

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{check_initial_point, check_oracle_len, fitness_stagnated, BestMeasured};
use rand::{Rng, RngExt};
use std::fmt;
use std::str::FromStr;

/// Adam optimizer (Kingma and Ba, 2015).
pub struct AlgorithmAdam;

/// How [`AlgorithmAdam`] turns the gradient `g` into a step (`polypus.Adam(rule=...)`).
///
/// - [`Adam`](Self::Adam), the default: Kingma and Ba's Adam. Bias-corrected
///   running averages of `g` and of `g²` scale each parameter's step to about
///   `learning_rate`, whatever the scale of the cost, so one configuration
///   suits costs of different scales.
/// - [`Momentum`](Self::Momentum): Polyak's heavy ball, with `beta1` as the
///   momentum μ: a velocity `v ← μ·v + learning_rate·g` is added to `θ`. The
///   step grows with `g`, so the learning rate that works depends on the
///   scale of the cost. `beta2` and `epsilon` are not used (they are still
///   validated).
///
/// The names `polypus.Adam(rule=...)` accepts are [`as_str`](Self::as_str)
/// and parse back with [`FromStr`]:
///
/// ```
/// use polypus_optimizers::AdamRule;
///
/// assert_eq!("momentum".parse::<AdamRule>(), Ok(AdamRule::Momentum));
/// assert_eq!(AdamRule::default().as_str(), "adam");
/// assert!("Adam".parse::<AdamRule>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AdamRule {
    /// Adam: bias-corrected, moment-normalised steps.
    #[default]
    Adam,
    /// Polyak's heavy-ball momentum.
    Momentum,
}

impl AdamRule {
    /// The rule's name, as `polypus.Adam(rule=...)` takes it: `"adam"` or
    /// `"momentum"`.
    pub fn as_str(self) -> &'static str {
        match self {
            AdamRule::Adam => "adam",
            AdamRule::Momentum => "momentum",
        }
    }
}

impl fmt::Display for AdamRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AdamRule {
    type Err = OptimizerError;

    /// Parse one of the names [`as_str`](Self::as_str) returns. Matching is
    /// exact; any other string is [`OptimizerError::InvalidConfig`] for
    /// `rule`, which the Python binding raises as `ValueError`.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "adam" => Ok(AdamRule::Adam),
            "momentum" => Ok(AdamRule::Momentum),
            other => Err(OptimizerError::InvalidConfig {
                parameter: "rule",
                reason: format!("must be \"adam\" or \"momentum\", got {other:?}"),
            }),
        }
    }
}

/// Arguments for the Adam optimizer.
///
/// Like [`AlgorithmQNG`](crate::AlgorithmQNG), Adam is decoupled from circuits
/// and infrastructure: the gradient comes from
/// [`gradient_oracle`](AlgorithmAdamArgs::gradient_oracle) and the fitness used
/// to track the best solution from [`oracle`](AlgorithmAdamArgs::oracle). It
/// needs no [`VarianceOracle`](crate::VarianceOracle): its per-parameter step
/// size comes from running averages of the gradient and of its square.
///
/// Each iteration measures the current `θ` and the gradient there, then steps
/// by [`rule`](AlgorithmAdamArgs::rule).
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
/// the bias correction divides by zero), `epsilon` finite and `> 0` (with
/// either rule, although momentum uses neither `beta2` nor `epsilon`),
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
    /// Step size applied to the bias-corrected, moment-normalised update; with
    /// [`AdamRule::Momentum`], the factor of the gradient in the velocity.
    pub learning_rate: f64,
    /// Decay rate of the first moment (running mean of the gradient); with
    /// [`AdamRule::Momentum`], the momentum μ.
    pub beta1: f64,
    /// Decay rate of the second moment (running mean of the squared gradient).
    /// Unused by [`AdamRule::Momentum`].
    pub beta2: f64,
    /// Added to the denominator of the update so it never divides by zero.
    /// Unused by [`AdamRule::Momentum`].
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
    /// How the gradient becomes a step: Adam (the default) or Polyak's
    /// momentum.
    pub rule: AdamRule,
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
             square, with bias correction, or with Polyak's heavy-ball momentum. Stops \
             early when the best fitness stagnates.",
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
            rule,
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
            // A non-finite component would leave θ non-finite from here on (C-5).
            if let Some((index, &value)) = grad.iter().enumerate().find(|(_, g)| !g.is_finite()) {
                return Err(OptimizerError::NonFiniteGradient { index, value });
            }
            log::debug!("Iteration {iteration}: fitness {fitness:.4}");

            // ── 2. Track the best measured θ ──────────────────────────────────
            best.offer(&theta, fitness);
            // The running best, not this iteration's fitness: a gradient step can
            // overshoot, and the history must be non-decreasing with the reported
            // best as its last entry (C-5).
            fitness_history.push(best.value());

            // ── 3. Step uphill by the rule ────────────────────────────────────
            match rule {
                // Bias-corrected moment update (t is 1-based).
                AdamRule::Adam => {
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
                }
                // Heavy ball: `m` is the velocity and β₁ its momentum.
                AdamRule::Momentum => {
                    for i in 0..dims {
                        m[i] = beta1 * m[i] + learning_rate * grad[i];
                        theta[i] += m[i];
                    }
                }
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
