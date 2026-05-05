use rand::{thread_rng, Rng};
use pyo3::prelude::*;
use pyo3::types::{PyModule, IntoPyDict};
use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

pub struct AlgorithmQNG;

/// Arguments required to run the Quantum Natural Gradient (QNG) algorithm.
///
/// - base: Base algorithm arguments (circuit, shots, infrastructure, etc.).
/// - max_iters: Maximum number of QNG iterations.
/// - learning_rate: Step size η for the QNG update rule.
/// - finite_difference_step: Step h used in the parameter-shift gradient estimator.
/// - bounds: (lower, upper) initial parameter bounds.
/// - dimensions: Number of variational parameters (2 * p_layers for QAOA).
/// - expectation_function: Python callable `fn(bitstring: str) -> float` used to
///   compute the circuit's expectation value from measurement counts.
/// - variance_function: Python callable `fn(theta: list[float], a: int) -> float`
///   that computes the QFIM diagonal element for parameter index `a`.
///   This function is responsible for building and running the appropriate
///   intermediate circuit (up to parameter a) and returning Var[H_a | theta].
/// - tikhonov_reg: Regularisation constant added to each diagonal QFIM element
///   to avoid division by near-zero values (equivalent to epsilon in the Python
///   implementation where 0.05 is used as default).
pub struct AlgorithmQNGArgs {
    pub base: AlgorithmArgs,
    pub max_iters: u32,
    pub learning_rate: f64,
    pub finite_difference_step: f64,
    pub bounds: (f64, f64),
    pub dimensions: u32,
    pub expectation_function: Py<PyAny>,
    pub variance_function: Py<PyAny>,
    pub tikhonov_reg: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Clone the template circuit from `base.qcs[0]` and bind `params` to it.
fn assign_parameters(base: &AlgorithmArgs, params: &[f64]) -> Py<PyAny> {
    Python::with_gil(|py| {
        let qc_any = base.qcs[0]
            .clone_ref(py)
            .into_pyobject(py)
            .expect("Failed to get qc as PyObject");
        let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
        qc_any
            .call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .expect("Error assigning parameters to circuit")
            .unbind()
    })
}

/// Execute a flat list of bound circuits through `runner` in batches of
/// `base.n_qpus`, then collect and return one expectation value per circuit.
fn run_circuits_batched(
    base: &AlgorithmArgs,
    runner: &dyn QuantumRunner,
    expectation_function: &Py<PyAny>,
    qcs: Vec<Py<PyAny>>,
) -> Vec<f64> {
    let total = qcs.len();
    let batch_size = base.n_qpus as usize;
    let mut all_expectations = vec![0.0f64; total];

    for i in (0..total).step_by(batch_size) {
        let end = (i + batch_size).min(total);
        let batch_qcs: Vec<Py<PyAny>> = qcs[i..end]
            .iter()
            .map(|qc| Python::with_gil(|py| qc.clone_ref(py)))
            .collect();

        let batch_args = AlgorithmArgs {
            id: base.id.clone(),
            qcs: batch_qcs,
            shots: base.shots,
            n_qpus: base.n_qpus,
            infrastructure: base.infrastructure.clone(),
            backend: base.backend.clone(),
            nodes: base.nodes,
            cores_per_qpu: base.cores_per_qpu,
        };

        let running_result = runner.run(&batch_args);

        let batch_expectations: Vec<f64> = Python::with_gil(|py| {
            let polypus_python = PyModule::import(py, "polypus_python")
                .expect("Failed to import polypus_python");
            polypus_python
                .call_method(
                    "expectation_values",
                    (running_result, expectation_function),
                    None,
                )
                .expect("Error calling polypus_python.expectation_values")
                .extract::<Vec<f64>>()
                .expect("Failed to extract expectation values as Vec<f64>")
        });

        for (local_idx, global_idx) in (i..end).enumerate() {
            all_expectations[global_idx] = batch_expectations[local_idx];
        }
    }

    all_expectations
}

/// Estimate the gradient of **-E(θ)** (the objective we minimise) using the
/// parameter-shift rule:
///
///   ∂(-E)/∂θ_i ≈ [-E(θ + h·eᵢ) + E(θ − h·eᵢ)] / (2h)
///              = [E(θ − h·eᵢ) − E(θ + h·eᵢ)] / (2h)
///
/// All 2·dims shifted circuits are submitted to the runner in batches of
/// `base.n_qpus` for efficient parallel execution.
fn compute_gradient(
    base: &AlgorithmArgs,
    runner: &dyn QuantumRunner,
    expectation_function: &Py<PyAny>,
    theta: &[f64],
    h: f64,
) -> Vec<f64> {
    let dims = theta.len();
    // Layout: [theta+h for dim 0, theta-h for dim 0, theta+h for dim 1, ...]
    let mut qcs: Vec<Py<PyAny>> = Vec::with_capacity(2 * dims);
    for i in 0..dims {
        let mut theta_plus = theta.to_vec();
        let mut theta_minus = theta.to_vec();
        theta_plus[i] += h;
        theta_minus[i] -= h;
        qcs.push(assign_parameters(base, &theta_plus));
        qcs.push(assign_parameters(base, &theta_minus));
    }

    let expectations = run_circuits_batched(base, runner, expectation_function, qcs);

    // grad[i] = ∂(-E)/∂θ_i = [-E(θ+h) - (-E(θ-h))] / (2h)
    //         = [E(θ-h) - E(θ+h)] / (2h)
    let mut grad = vec![0.0f64; dims];
    for i in 0..dims {
        let e_plus  = expectations[2 * i];      // E(θ + h·eᵢ)
        let e_minus = expectations[2 * i + 1];  // E(θ − h·eᵢ)
        grad[i] = (e_minus - e_plus) / (2.0 * h);
    }
    grad
}

/// Compute the diagonal of the Fubini–Study metric (QFIM) by delegating to
/// the Python `variance_function(theta, a) -> float` for each dimension `a`.
/// Tikhonov regularisation `tikhonov_reg` is added to each element to prevent
/// division by very small values during the QNG update.
fn compute_qfim_diagonal(
    variance_function: &Py<PyAny>,
    theta: &[f64],
    dims: usize,
    tikhonov_reg: f64,
) -> Vec<f64> {
    Python::with_gil(|py| {
        let vf = variance_function.bind(py);
        (0..dims)
            .map(|a| {
                let variance: f64 = vf
                    .call1((theta.to_vec(), a as u32))
                    .expect("Error calling variance_function")
                    .extract()
                    .expect("Failed to extract float from variance_function");
                variance + tikhonov_reg
            })
            .collect()
    })
}

/// Evaluate the expectation value at a single parameter point `theta`.
fn evaluate_single(
    base: &AlgorithmArgs,
    runner: &dyn QuantumRunner,
    expectation_function: &Py<PyAny>,
    theta: &[f64],
) -> f64 {
    let qc = assign_parameters(base, theta);
    let batch_args = AlgorithmArgs {
        id: base.id.clone(),
        qcs: vec![qc],
        shots: base.shots,
        n_qpus: 1,
        infrastructure: base.infrastructure.clone(),
        backend: base.backend.clone(),
        nodes: base.nodes,
        cores_per_qpu: base.cores_per_qpu,
    };
    let result = runner.run(&batch_args);
    Python::with_gil(|py| {
        let polypus_python = PyModule::import(py, "polypus_python")
            .expect("Failed to import polypus_python");
        polypus_python
            .call_method("expectation_values", (result, expectation_function), None)
            .expect("Error computing single expectation value")
            .extract::<Vec<f64>>()
            .expect("Failed to extract expectation value")[0]
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// AlgorithmTrait implementation
// ─────────────────────────────────────────────────────────────────────────────

impl AlgorithmTrait for AlgorithmQNG {
    type Args = AlgorithmQNGArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmQNGArgs) -> Self::AlgorithmReturnType {
        let AlgorithmQNGArgs {
            base,
            max_iters,
            learning_rate,
            finite_difference_step,
            bounds,
            dimensions,
            expectation_function,
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

        println!("Running QNG with infrastructure: {}", &base.infrastructure);
        let infra = Infrastructure::from_str(&base.infrastructure);
        let runner: Box<dyn QuantumRunner> = match infra {
            Infrastructure::Local => Box::new(LocalRunner),
            Infrastructure::Cunqa => Box::new(CunqaRunner::new(base.n_qpus, base.nodes, &base.id, base.cores_per_qpu)),
        };

        for iteration in 0..max_iters as usize {
            // ── 1. Gradient of -E via parameter-shift rule ───────────────────
            let grad = compute_gradient(
                &base,
                runner.as_ref(),
                &expectation_function,
                &theta,
                finite_difference_step,
            );

            // ── 2. Diagonal QFIM with Tikhonov regularisation ────────────────
            let qfim_diag = compute_qfim_diagonal(
                &variance_function,
                &theta,
                dims,
                tikhonov_reg,
            );

            // ── 3. QNG update: θ ← θ − η · G⁻¹ · ∇(-E) ─────────────────────
            //    Equivalent to θ ← θ + η · G⁻¹ · ∇E  (maximise expectation)
            for i in 0..dims {
                theta[i] -= learning_rate * grad[i] / qfim_diag[i];
            }

            // ── 4. Evaluate energy and track best solution ────────────────────
            let energy = evaluate_single(
                &base,
                runner.as_ref(),
                &expectation_function,
                &theta,
            );
            println!("Iteration {}: Energy: {:.4}", iteration, energy);

            if energy > best_energy {
                best_energy = energy;
                best_theta = theta.clone();
            }
        }

        runner.close();

        Python::with_gil(|py| {
            match best_theta.into_pyobject(py) {
                Ok(obj) => obj.unbind(),
                Err(e) => panic!("Error converting best_theta to PyObject: {}", e),
            }
        })
    }

    fn name(&self) -> String {
        String::from("Quantum Natural Gradient")
    }

    fn description(&self) -> String {
        String::from(
            "Trains a QAOA circuit using the Quantum Natural Gradient (QNG) optimizer. \
             Gradients are estimated via the parameter-shift rule and preconditioned by \
             the diagonal Fubini-Study metric (QFIM), which is supplied as a Python callback.",
        )
    }
}
