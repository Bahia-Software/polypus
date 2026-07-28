//! Quantum Natural Gradient (QNG) optimizer.

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle, VarianceOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::check_oracle_len;
use rand::Rng;

/// Quantum Natural Gradient optimizer.
pub struct AlgorithmQNG;

/// Arguments for the Quantum Natural Gradient optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure: the
/// fitness gradient is delegated to
/// [`gradient_oracle`](AlgorithmQNGArgs::gradient_oracle), the fitness itself
/// (for best-solution tracking) to [`oracle`](AlgorithmQNGArgs::oracle), and the
/// QFIM diagonal to [`variance_oracle`](AlgorithmQNGArgs::variance_oracle). Each
/// stays a separate contract because the gradient, the raw fitness and
/// `Var[H_a | θ]` are distinct mathematical quantities, not one circuit-execution
/// concern.
///
/// # Preconditions
///
/// `bounds` must be a non-empty interval (`lb < ub`); the initial `θ` is drawn
/// uniformly from it.
pub struct AlgorithmQNGArgs {
    /// Oracle that maps parameter vectors → fitness values (used for tracking
    /// the best solution across iterations).
    pub oracle: Box<dyn EvaluationOracle>,
    /// Exact fitness gradient `∇fitness(θ)` (parameter-shift). Its ascent-sign
    /// convention matches [`oracle`](AlgorithmQNGArgs::oracle): higher fitness is
    /// better, so this points uphill and the QNG update *adds* it.
    pub gradient_oracle: Box<dyn GradientOracle>,
    pub max_iters: u32,
    pub learning_rate: f64,
    pub bounds: (f64, f64),
    pub dimensions: u32,
    /// Early-stopping tolerance on the gradient norm. After each full iteration
    /// the L2 norm `‖∇fitness(θ)‖` of that iteration's gradient is compared
    /// against this value. Mirrors the `tolerance` field DE/PSO use for their
    /// population-collapse test, but the stop needs
    /// [`patience`](AlgorithmQNGArgs::patience) *consecutive* iterations below it,
    /// not just one.
    pub tolerance: f64,
    /// Number of **consecutive** iterations whose gradient norm must stay below
    /// [`tolerance`](AlgorithmQNGArgs::tolerance) before the run stops early with
    /// `converged = true`. The counter increments on each sub-tolerance iteration
    /// and resets to `0` on any iteration that is not, so three iterations below
    /// the tolerance scattered among larger ones never trigger a stop with
    /// `patience = 3`. `patience = 1` is exactly the single-iteration rule; `0`
    /// behaves like `1`, since the streak is only tested after an iteration that
    /// was itself below the tolerance.
    ///
    /// Why it is not `1`: the norm handed to the optimizer may be a *minibatch*
    /// gradient (the [`GradientOracle`] contract deliberately hides whether it
    /// is), and a minibatch can cancel to exactly zero at a `θ` whose
    /// full-dataset gradient is far from zero — see the minibatch note beside
    /// C-5/C-7 in `docs/CONTRACTS.md`. Requiring several consecutive
    /// sub-tolerance iterations makes that coincidence far less likely; it does
    /// not make it impossible.
    pub patience: usize,
    /// Returns `Var[H_a | θ]`, the diagonal QFIM element for parameter index `a`.
    pub variance_oracle: Box<dyn VarianceOracle>,
    /// Tikhonov regularisation added to each QFIM element to avoid near-zero division.
    pub tikhonov_reg: f64,
    /// Optional RNG seed. `None` (the default) uses [`rand::thread_rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the diagonal of the Fubini–Study metric (QFIM) via `variance_oracle`,
/// adding the Tikhonov regularisation term to every element.
///
/// The whole diagonal is requested in a single
/// [`VarianceOracle::variance_diagonal`] call so that runtime-backed oracles
/// (e.g. a Python callback) can amortise their setup cost across all
/// dimensions — preserving the "acquire the runtime once, loop over `0..dims`"
/// semantics of the original implementation.
fn compute_qfim_diagonal(
    variance_oracle: &dyn VarianceOracle,
    theta: &[f64],
    dims: usize,
    tikhonov_reg: f64,
) -> Vec<f64> {
    let mut diag = variance_oracle.variance_diagonal(theta, dims);
    for v in diag.iter_mut() {
        *v += tikhonov_reg;
    }
    diag
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer implementation
// ─────────────────────────────────────────────────────────────────────────────

impl AlgorithmQNG {
    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("Quantum Natural Gradient")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit using the Quantum Natural Gradient (QNG) \
             optimizer. The exact fitness gradient (parameter-shift) is preconditioned by the \
             diagonal Fubini-Study metric (QFIM). Stops early once the gradient norm has \
             stayed below the configured tolerance for `patience` consecutive iterations.",
        )
    }

    fn run_with_rng<R: Rng>(
        args: AlgorithmQNGArgs,
        rng: &mut R,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmQNGArgs {
            oracle,
            gradient_oracle,
            max_iters,
            learning_rate,
            bounds,
            dimensions,
            tolerance,
            patience,
            variance_oracle,
            tikhonov_reg,
            seed: _,
        } = args;

        let dims = dimensions as usize;
        let (lb, ub) = bounds;
        // θ is drawn from the half-open interval [lb, ub), which is empty when
        // `lb >= ub` and panics inside the sampler. Reject before any RNG draw
        // or oracle call, as PSO does; requiring `partial_cmp` to be
        // `Some(Less)` also rejects a non-finite (`NaN`) bound.
        if !matches!(lb.partial_cmp(&ub), Some(std::cmp::Ordering::Less)) {
            return Err(OptimizerError::InvalidBounds { lb, ub });
        }

        // Initialise θ uniformly in [lb, ub)
        let mut theta: Vec<f64> = (0..dims).map(|_| rng.gen_range(lb..ub)).collect();
        let mut best_energy = f64::NEG_INFINITY;
        let mut best_theta = theta.clone();
        let mut iterations_run = 0usize;
        let mut converged = false;
        // Consecutive iterations whose gradient norm stayed below `tolerance`.
        let mut below_tolerance_streak = 0usize;

        for iteration in 0..max_iters as usize {
            iterations_run = iteration + 1;

            // ── 1. Exact fitness gradient via parameter-shift ────────────────
            //    `gradient_oracle` returns ∇fitness directly (ascent sign), one
            //    value per parameter; check the length as any oracle output is.
            let grad = gradient_oracle.gradient_batch(&theta, dims);
            check_oracle_len(dims, grad.len())?;

            // ── 2. Diagonal QFIM with Tikhonov regularisation ────────────────
            let qfim_diag =
                compute_qfim_diagonal(variance_oracle.as_ref(), &theta, dims, tikhonov_reg);

            // ── 3. QNG update: θ ← θ + η · G⁻¹ · ∇fitness ────────────────────
            //    `grad` is ∇fitness (uphill), so ascend it to maximise fitness.
            for i in 0..dims {
                theta[i] += learning_rate * grad[i] / qfim_diag[i];
            }

            // ── 4. Evaluate energy and track best solution ────────────────────
            let energy_batch = oracle.evaluate_batch(&[theta.clone()]);
            check_oracle_len(1, energy_batch.len())?;
            let energy = energy_batch[0];
            log::debug!("Iteration {iteration}: Energy: {energy:.4}");

            if energy > best_energy {
                best_energy = energy;
                best_theta = theta.clone();
            }

            // ── 5. Early stopping on the gradient norm ────────────────────────
            //    Decide after the full iteration's work (the same placement as
            //    DE/PSO's population_converged test). A *single* sub-tolerance
            //    iteration is not enough: `patience` consecutive ones are, and
            //    any iteration above the tolerance clears the streak. See the
            //    `patience` doc-comment for why one iteration cannot be trusted.
            let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < tolerance {
                below_tolerance_streak += 1;
                if below_tolerance_streak >= patience {
                    converged = true;
                    break;
                }
            } else {
                below_tolerance_streak = 0;
            }
        }

        Ok(OptimizationOutcome {
            best_params: best_theta,
            best_fitness: best_energy,
            iterations_run,
            converged,
        })
    }
}

impl Optimizer for AlgorithmQNG {
    type Args = AlgorithmQNGArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
