use rand::{thread_rng, Rng};
use pyo3::prelude::*;
use crate::algorithms::AlgorithmTrait;
use crate::evaluation::EvaluationOracle;

pub struct AlgorithmQNG;

/// Arguments for the Quantum Natural Gradient optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// gradient circuit evaluation is delegated to `oracle`.
/// The QFIM diagonal is still computed via `variance_function` because it is
/// an algorithm-specific mathematical concept, not a circuit execution concern.
pub struct AlgorithmQNGArgs {
    /// Oracle that maps parameter vectors → fitness values (used for gradient + energy eval).
    pub oracle: Box<dyn EvaluationOracle>,
    pub max_iters: u32,
    pub learning_rate: f64,
    pub finite_difference_step: f64,
    pub bounds: (f64, f64),
    pub dimensions: u32,
    /// Python callable `fn(theta: list[float], a: int) -> float` that returns
    /// `Var[H_a | theta]`, the diagonal QFIM element for parameter index `a`.
    pub variance_function: Py<PyAny>,
    /// Tikhonov regularisation added to each QFIM element to avoid near-zero division.
    pub tikhonov_reg: f64,
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

/// Compute the diagonal of the Fubini–Study metric (QFIM) by delegating to
/// the Python `variance_function(theta, a) -> float` for each dimension `a`.
///
/// `variance_function` is pure Python and must run under the GIL, so the calls
/// are serialised regardless of threading. We therefore acquire the GIL once
/// and evaluate every dimension in a tight loop — this avoids the per-call
/// thread-spawn and GIL release/reacquire overhead that a Tokio-based version
/// would incur for no parallelism gain.
fn compute_qfim_diagonal(
    variance_function: &Py<PyAny>,
    theta: &[f64],
    dims: usize,
    tikhonov_reg: f64,
) -> Vec<f64> {
    Python::with_gil(|py| {
        let theta_vec = theta.to_vec();
        (0..dims)
            .map(|a| {
                let variance: f64 = variance_function
                    .bind(py)
                    .call1((theta_vec.clone(), a as u32))
                    .expect("Error calling variance_function")
                    .extract()
                    .expect("Failed to extract float from variance_function");
                variance + tikhonov_reg
            })
            .collect()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// AlgorithmTrait implementation
// ─────────────────────────────────────────────────────────────────────────────

impl AlgorithmTrait for AlgorithmQNG {
    type Args = AlgorithmQNGArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmQNGArgs) -> PyObject {
        let AlgorithmQNGArgs {
            oracle,
            max_iters,
            learning_rate,
            finite_difference_step,
            bounds,
            dimensions,
            variance_function,
            tikhonov_reg,
        } = args;

        let dims = dimensions as usize;
        let (lb, ub) = bounds;
        let mut rng = thread_rng();

        // Initialise θ uniformly in [lb, ub)
        let mut theta: Vec<f64> = (0..dims).map(|_| rng.gen_range(lb..ub)).collect();
        let mut best_energy = f64::NEG_INFINITY;
        let mut best_theta = theta.clone();

        for iteration in 0..max_iters as usize {
            // ── 1. Gradient of −E via parameter-shift rule ───────────────────
            let grad = compute_gradient(oracle.as_ref(), &theta, finite_difference_step);

            // ── 2. Diagonal QFIM with Tikhonov regularisation ────────────────
            let qfim_diag =
                compute_qfim_diagonal(&variance_function, &theta, dims, tikhonov_reg);

            // ── 3. QNG update: θ ← θ − η · G⁻¹ · ∇(−E) ─────────────────────
            //    Equivalent to θ ← θ + η · G⁻¹ · ∇E  (maximise expectation)
            for i in 0..dims {
                theta[i] -= learning_rate * grad[i] / qfim_diag[i];
            }

            // ── 4. Evaluate energy and track best solution ────────────────────
            let energy = oracle.evaluate_batch(&[theta.clone()])[0];
            println!("Iteration {iteration}: Energy: {energy:.4}");

            if energy > best_energy {
                best_energy = energy;
                best_theta = theta.clone();
            }
        }

        Python::with_gil(|py| {
            best_theta
                .into_pyobject(py)
                .expect("Error converting best_theta to PyObject")
                .unbind()
        })
    }

    fn name(&self) -> String {
        String::from("Quantum Natural Gradient")
    }

    fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit using the Quantum Natural Gradient (QNG) \
             optimizer. Gradients are estimated via the parameter-shift rule and \
             preconditioned by the diagonal Fubini-Study metric (QFIM).",
        )
    }
}

