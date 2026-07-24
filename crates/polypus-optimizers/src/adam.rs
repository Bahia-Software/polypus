//! Adam optimizer (adaptive moment estimation).

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::check_oracle_len;
use rand::Rng;

/// Adam optimizer.
pub struct AlgorithmAdam;

/// Arguments for the Adam optimizer.
///
/// Like [`AlgorithmQNG`](crate::AlgorithmQNG), Adam is completely decoupled from
/// circuits and infrastructure: the exact fitness gradient is delegated to
/// [`gradient_oracle`](AlgorithmAdamArgs::gradient_oracle) and the fitness itself
/// (for best-solution tracking) to [`oracle`](AlgorithmAdamArgs::oracle). Unlike
/// QNG it needs no [`VarianceOracle`](crate::VarianceOracle): the adaptive
/// per-parameter step size comes from the running first/second moments of the
/// gradient, not from the Fubini–Study metric.
///
/// # Preconditions
///
/// `bounds` must be a non-empty interval (`lb < ub`); the initial `θ` is drawn
/// uniformly from it.
pub struct AlgorithmAdamArgs {
    /// Oracle that maps parameter vectors → fitness values (used for tracking
    /// the best solution across iterations).
    pub oracle: Box<dyn EvaluationOracle>,
    /// Exact fitness gradient `∇fitness(θ)` (parameter-shift). Its ascent-sign
    /// convention matches [`oracle`](AlgorithmAdamArgs::oracle): higher fitness
    /// is better, so this points uphill and the Adam update *adds* it.
    pub gradient_oracle: Box<dyn GradientOracle>,
    pub max_iters: u32,
    /// Step-size scale applied to the bias-corrected, moment-normalised update.
    pub learning_rate: f64,
    /// Exponential decay rate for the first moment (running mean of the gradient).
    pub beta1: f64,
    /// Exponential decay rate for the second moment (running mean of the squared
    /// gradient).
    pub beta2: f64,
    /// Small constant added to the denominator to avoid division by zero.
    pub epsilon: f64,
    pub bounds: (f64, f64),
    pub dimensions: u32,
    /// Early-stopping tolerance on the gradient norm. After each full iteration
    /// the L2 norm `‖∇fitness(θ)‖` of that iteration's gradient is compared
    /// against this value; when it falls below `tolerance` the run stops early
    /// with `converged = true`. Mirrors the `tolerance` field DE/PSO use for
    /// their population-collapse test.
    pub tolerance: f64,
    /// Optional RNG seed. `None` (the default) uses [`rand::thread_rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
}

impl AlgorithmAdam {
    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("Adam")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit using the Adam optimizer. The exact fitness \
             gradient (parameter-shift) drives per-parameter adaptive steps from the running \
             first and second moments of the gradient, with standard bias correction. Stops \
             early once the gradient norm falls below the configured tolerance.",
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
            seed: _,
        } = args;

        let dims = dimensions as usize;
        let (lb, ub) = bounds;
        // θ is drawn from the half-open interval [lb, ub), which is empty when
        // `lb >= ub` and panics inside the sampler. Reject before any RNG draw
        // or oracle call, exactly as QNG/PSO do; requiring `partial_cmp` to be
        // `Some(Less)` also rejects a non-finite (`NaN`) bound.
        if !matches!(lb.partial_cmp(&ub), Some(std::cmp::Ordering::Less)) {
            return Err(OptimizerError::InvalidBounds { lb, ub });
        }

        // Initialise θ uniformly in [lb, ub) and the moment accumulators at 0.
        let mut theta: Vec<f64> = (0..dims).map(|_| rng.gen_range(lb..ub)).collect();
        let mut m = vec![0.0_f64; dims];
        let mut v = vec![0.0_f64; dims];
        let mut best_energy = f64::NEG_INFINITY;
        let mut best_theta = theta.clone();
        let mut iterations_run = 0usize;
        let mut converged = false;

        for iteration in 0..max_iters as usize {
            iterations_run = iteration + 1;
            // Adam's bias correction is 1-indexed by the iteration count `t`.
            let t = iteration as i32 + 1;

            // ── 1. Exact fitness gradient via parameter-shift ────────────────
            //    `gradient_oracle` returns ∇fitness directly (ascent sign), one
            //    value per parameter; check the length as any oracle output is.
            let grad = gradient_oracle.gradient_batch(&theta, dims);
            check_oracle_len(dims, grad.len())?;

            // ── 2. Adam update: bias-corrected adaptive moment ascent ────────
            //    `grad` is ∇fitness (uphill), so ascend it to maximise fitness.
            let bias1 = 1.0 - beta1.powi(t);
            let bias2 = 1.0 - beta2.powi(t);
            for i in 0..dims {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                theta[i] += learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }

            // ── 3. Evaluate energy and track best solution ───────────────────
            let energy_batch = oracle.evaluate_batch(&[theta.clone()]);
            check_oracle_len(1, energy_batch.len())?;
            let energy = energy_batch[0];
            log::debug!("Iteration {iteration}: Energy: {energy:.4}");

            if energy > best_energy {
                best_energy = energy;
                best_theta = theta.clone();
            }

            // ── 4. Early stopping on the gradient norm ────────────────────────
            //    Decide after the full iteration's work (the same placement as
            //    DE/PSO's population_converged test): if ‖∇fitness(θ)‖ from this
            //    iteration's `grad` has fallen below `tolerance`, no further
            //    iteration is needed.
            let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < tolerance {
                converged = true;
                break;
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

impl Optimizer for AlgorithmAdam {
    type Args = AlgorithmAdamArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
