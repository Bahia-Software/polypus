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
             diagonal Fubini-Study metric (QFIM).",
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
        }

        Ok(OptimizationOutcome {
            best_params: best_theta,
            best_fitness: best_energy,
            iterations_run,
            // QNG runs a fixed iteration budget; it has no early-stopping test.
            converged: false,
        })
    }
}

impl Optimizer for AlgorithmQNG {
    type Args = AlgorithmQNGArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
