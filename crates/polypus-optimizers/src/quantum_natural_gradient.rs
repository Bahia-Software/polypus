//! Quantum Natural Gradient (QNG) optimizer.

use crate::objective::{EvaluationOracle, VarianceOracle};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use rand::Rng;

/// Quantum Natural Gradient optimizer.
pub struct AlgorithmQNG;

/// Arguments for the Quantum Natural Gradient optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// gradient circuit evaluation is delegated to
/// [`oracle`](AlgorithmQNGArgs::oracle) and the QFIM diagonal to
/// [`variance_oracle`](AlgorithmQNGArgs::variance_oracle). The latter stays a
/// separate contract because computing `Var[H_a | θ]` is an algorithm-specific
/// mathematical concept, not a circuit-execution concern.
pub struct AlgorithmQNGArgs {
    /// Oracle that maps parameter vectors → fitness values (used for gradient + energy eval).
    pub oracle: Box<dyn EvaluationOracle>,
    pub max_iters: u32,
    pub learning_rate: f64,
    pub finite_difference_step: f64,
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

/// Estimate the gradient of **−E(θ)** using the parameter-shift rule.
///
/// Builds a flat list of 2·dims shifted candidates and submits them to the
/// oracle in one batch call, then assembles the gradient from the results.
fn compute_gradient(oracle: &dyn EvaluationOracle, theta: &[f64], h: f64) -> Vec<f64> {
    let dims = theta.len();
    // Layout: [θ+h·e₀, θ−h·e₀, θ+h·e₁, θ−h·e₁, …]
    let candidates: Vec<Vec<f64>> = (0..dims)
        .flat_map(|i| {
            let mut tp = theta.to_vec();
            let mut tm = theta.to_vec();
            tp[i] += h;
            tm[i] -= h;
            [tp, tm]
        })
        .collect();

    let expectations = oracle.evaluate_batch(&candidates);

    // grad[i] = [E(θ−h·eᵢ) − E(θ+h·eᵢ)] / (2h)  →  minimise −E, maximise E
    let mut grad = vec![0.0f64; dims];
    for i in 0..dims {
        let e_plus = expectations[2 * i];
        let e_minus = expectations[2 * i + 1];
        grad[i] = (e_minus - e_plus) / (2.0 * h);
    }
    grad
}

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
             optimizer. Gradients are estimated via the parameter-shift rule and \
             preconditioned by the diagonal Fubini-Study metric (QFIM).",
        )
    }

    fn run_with_rng<R: Rng>(args: AlgorithmQNGArgs, rng: &mut R) -> OptimizationOutcome {
        let AlgorithmQNGArgs {
            oracle,
            max_iters,
            learning_rate,
            finite_difference_step,
            bounds,
            dimensions,
            variance_oracle,
            tikhonov_reg,
            seed: _,
        } = args;

        let dims = dimensions as usize;
        let (lb, ub) = bounds;

        // Initialise θ uniformly in [lb, ub)
        let mut theta: Vec<f64> = (0..dims).map(|_| rng.gen_range(lb..ub)).collect();
        let mut best_energy = f64::NEG_INFINITY;
        let mut best_theta = theta.clone();
        let mut iterations_run = 0usize;

        for iteration in 0..max_iters as usize {
            iterations_run = iteration + 1;

            // ── 1. Gradient of −E via parameter-shift rule ───────────────────
            let grad = compute_gradient(oracle.as_ref(), &theta, finite_difference_step);

            // ── 2. Diagonal QFIM with Tikhonov regularisation ────────────────
            let qfim_diag =
                compute_qfim_diagonal(variance_oracle.as_ref(), &theta, dims, tikhonov_reg);

            // ── 3. QNG update: θ ← θ − η · G⁻¹ · ∇(−E) ─────────────────────
            //    Equivalent to θ ← θ + η · G⁻¹ · ∇E  (maximise expectation)
            for i in 0..dims {
                theta[i] -= learning_rate * grad[i] / qfim_diag[i];
            }

            // ── 4. Evaluate energy and track best solution ────────────────────
            let energy = oracle.evaluate_batch(&[theta.clone()])[0];
            log::debug!("Iteration {iteration}: Energy: {energy:.4}");

            if energy > best_energy {
                best_energy = energy;
                best_theta = theta.clone();
            }
        }

        OptimizationOutcome {
            best_params: best_theta,
            best_fitness: best_energy,
            iterations_run,
            // QNG runs a fixed iteration budget; it has no early-stopping test.
            converged: false,
        }
    }
}

impl Optimizer for AlgorithmQNG {
    type Args = AlgorithmQNGArgs;

    fn optimize(&self, args: Self::Args) -> OptimizationOutcome {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}
