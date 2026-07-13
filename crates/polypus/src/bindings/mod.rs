use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyResult;

pub mod circuit;
pub mod de;
pub mod logging;
pub mod pso;
pub mod qng;

use circuit::{statevector, Circuit, Param};
use de::DE;
use logging::init_logger;
use pso::PSO;
use qng::{PyVarianceOracle, QNG};

use crate::algorithms::{AlgorithmArgs, AlgorithmSingleRun, AlgorithmTrait, DistributeByShotsRun};
use crate::evaluation::{CircuitSource, EvaluationOracle, OracleErrorSlot, QmlOracle, VqcOracle};
use crate::infrastructure::execution_config::random_seed;
#[cfg(feature = "qmio")]
use crate::infrastructure::execution_config::QmioProgramFormat;
use crate::infrastructure::{
    BackendConfig, BoundCircuit, ExecutionConfig, Infrastructure, OptLevel,
};
use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, AlgorithmPSO,
    AlgorithmPSOArgs, AlgorithmQNG, AlgorithmQNGArgs, OptimizationOutcome, Optimizer,
    OptimizerError,
};
use std::sync::Arc;

/// Result of [`run_quantum_circuit`]: the measurement counts plus the run
/// manifest that lets a run be logged and replayed (contract C-7).
///
/// `counts` is the exact payload the runner produced before this wrapper
/// existed — a `list[dict[str, int]]` for a single-QPU run
/// ([`AlgorithmSingleRun`](crate::algorithms::AlgorithmSingleRun)), or a single
/// merged `dict[str, int]` for a distributed (`n_qpus > 1`) run
/// ([`DistributeByShotsRun`](crate::algorithms::DistributeByShotsRun)); the
/// per-dict format is contract C-3. The manifest fields make a simulated run
/// reproducible: feeding the reported [`seed`](Self::seed) back into
/// `run_quantum_circuit(..., seed=...)` reproduces the counts byte-for-byte on
/// any of the native, Aer, or CUNQA (simulated-QPU) backends. `seed` is `None`
/// only for the `qmio` infrastructure, which runs on real hardware that
/// Polypus cannot seed.
#[pyclass(frozen)]
pub struct RunResult {
    /// Measurement counts. Shape depends on `n_qpus` (see the type docs).
    #[pyo3(get)]
    pub counts: PyObject,
    /// Run identifier used for logging, temp files and SLURM job names.
    #[pyo3(get)]
    pub id: String,
    /// Effective RNG seed used by the backend's shot sampling (user-supplied or OS-entropy).
    /// Now supported for `native`, `aer`, and `cunqa`. `None` for `qmio` hardware.
    #[pyo3(get)]
    pub seed: Option<u64>,
    /// Device selected within the infrastructure (`"aer"`, `"polypus"`, …).
    #[pyo3(get)]
    pub backend: String,
    /// Execution infrastructure label (`"local"`, `"cunqa"`, `"qmio"`).
    #[pyo3(get)]
    pub infrastructure: String,
}

#[pymethods]
impl RunResult {
    fn __repr__(&self) -> String {
        format!(
            "RunResult(id={:?}, seed={:?}, backend={:?}, infrastructure={:?})",
            self.id, self.seed, self.backend, self.infrastructure
        )
    }
}

/// Result of [`train`] / [`qml_train`]: the full [`OptimizationOutcome`] plus
/// the effective RNG seed used, so a training run can be reported and reproduced
/// (contract C-7).
///
/// Replaces the former bare `list[float]` return, which discarded the fitness,
/// iteration count and convergence flag. `best_params` remains available as a
/// field; the previously dropped [`OptimizationOutcome`] fields are now exposed
/// alongside it, and `seed` records the value that drove the optimizer.
#[pyclass(frozen)]
pub struct TrainResult {
    /// Best parameter vector found.
    #[pyo3(get)]
    pub best_params: Vec<f64>,
    /// Fitness of [`best_params`](Self::best_params) (higher is better).
    #[pyo3(get)]
    pub best_fitness: f64,
    /// Generations/iterations actually executed (below the budget when an
    /// early-stopping criterion fired).
    #[pyo3(get)]
    pub iterations_run: usize,
    /// Whether the optimizer's convergence criterion was satisfied.
    #[pyo3(get)]
    pub converged: bool,
    /// Effective RNG seed that drove the optimizer (and, on the native backend,
    /// shot sampling): the explicit `seed` kwarg, else the optimizer object's
    /// `seed`, else a fresh OS-entropy value (contract C-7).
    #[pyo3(get)]
    pub seed: u64,
}

#[pymethods]
impl TrainResult {
    fn __repr__(&self) -> String {
        format!(
            "TrainResult(best_fitness={}, iterations_run={}, converged={}, seed={}, best_params={:?})",
            self.best_fitness, self.iterations_run, self.converged, self.seed, self.best_params
        )
    }
}

/// Wrap an [`OptimizationOutcome`] and the effective `seed` into the
/// [`TrainResult`] that `train` / `qml_train` now return.
///
/// This is the current public Python contract for those entry points (contract
/// C-7): the whole outcome — `best_params`, `best_fitness`, `iterations_run`,
/// `converged` — plus the `seed`, rather than the bare best-parameter list the
/// former `outcome_to_pyobject` produced.
fn outcome_to_train_result(
    py: Python<'_>,
    outcome: OptimizationOutcome,
    seed: u64,
) -> PyResult<PyObject> {
    Py::new(
        py,
        TrainResult {
            best_params: outcome.best_params,
            best_fitness: outcome.best_fitness,
            iterations_run: outcome.iterations_run,
            converged: outcome.converged,
            seed,
        },
    )
    .map(|result| result.into_any())
}

/// Resolve the optimizer seed for `train` / `qml_train` (contract C-7).
///
/// Precedence: the explicit `seed` kwarg wins when provided; otherwise the
/// `seed` field pinned on the `DE`/`PSO`/`QNG` instance; otherwise a fresh
/// OS-entropy value. The chosen value both drives the optimizer and is reported
/// back in the [`TrainResult`], so the run can be replayed.
fn resolve_optimizer_seed(kwarg_seed: Option<u64>, method_seed: Option<u64>) -> u64 {
    kwarg_seed.or(method_seed).unwrap_or_else(random_seed)
}

/// Read the `seed` field pinned on the optimizer object passed as `method`,
/// whichever of `DE`/`PSO`/`QNG` it is (`None` if it is none of them — the type
/// error is surfaced later by the dispatch that actually runs the optimizer).
fn method_seed(method: &Bound<'_, PyAny>) -> Option<u64> {
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        return de.seed;
    }
    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        return pso.seed;
    }
    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        return qng.seed;
    }
    None
}

/// Turn an optimizer result into the value the Python entry point returns.
///
/// An oracle failure recorded in `errors` takes precedence over the optimizer's
/// own result: because [`EvaluationOracle`]/`VarianceOracle` cannot return a
/// `Result`, an oracle records its first failure in the shared
/// [`OracleErrorSlot`] and yields finite sentinels, so `optimize` may return
/// `Ok` with a meaningless outcome. Surfacing the recorded error here is what
/// makes the FFI boundary report the real cause instead of that garbage.
fn finish_optimization(
    py: Python<'_>,
    result: Result<OptimizationOutcome, OptimizerError>,
    errors: &OracleErrorSlot,
    seed: u64,
) -> PyResult<PyObject> {
    if let Some(eval_err) = errors.take() {
        return Err(eval_err.into());
    }
    let outcome = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    outcome_to_train_result(py, outcome, seed)
}

/// Map the public `infrastructure` + `backend` strings and provider parameters
/// into a typed [`BackendConfig`]. Centralising this keeps the string→variant
/// mapping in one place and guarantees the config matches the selected backend.
///
/// `backend` selects the *device* within an infrastructure. For `local`:
/// `"aer"` (default) runs Qiskit Aer; `"polypus"` runs the pure-Rust native
/// statevector simulator. The choice is ignored for CUNQA, which manages its
/// own simulated QPUs.
fn build_backend_config(
    infrastructure: &str,
    backend: &str,
    sim_method: &str,
    noise_model: Option<Py<PyAny>>,
    nodes: u32,
    cores_per_qpu: u32,
) -> PyResult<BackendConfig> {
    match Infrastructure::from_str(infrastructure)? {
        Infrastructure::Local => match backend {
            "aer" | "AerSimulator" => Ok(BackendConfig::Local {
                backend: "AerSimulator".to_string(),
                sim_method: sim_method.to_string(),
                noise_model,
            }),
            "polypus" | "statevector" | "polypus_statevector" => {
                if noise_model.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "the native 'polypus' backend is a noiseless statevector simulator \
						 and does not accept a noise_model; use backend=\"aer\"",
                    ));
                }
                Ok(BackendConfig::LocalNative)
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown local backend '{other}'; expected \"aer\" or \"polypus\""
            ))),
        },
        Infrastructure::Cunqa => Ok(BackendConfig::Cunqa {
            backend: "AerSimulator".to_string(),
            sim_method: sim_method.to_string(),
            nodes,
            cores_per_qpu,
        }),
        Infrastructure::Qmio => build_qmio_backend_config(backend),
    }
}

/// Build the [`BackendConfig::Qmio`] for the CESGA QMIO QPU.
///
/// The endpoint is read from the `ZMQ_SERVER` environment variable (the same
/// variable the reference `qmio` Python client uses), falling back to the
/// documented CESGA address. The public `backend` argument selects the program
/// representation submitted to the QPU. Only available with `--features qmio`;
/// otherwise it returns an actionable error instead of silently degrading.
#[cfg(feature = "qmio")]
fn build_qmio_backend_config(backend: &str) -> PyResult<BackendConfig> {
    // Default endpoint documented by CESGA; overridden by ZMQ_SERVER when set.
    const DEFAULT_QMIO_ENDPOINT: &str = "tcp://10.133.29.226:5556";
    let endpoint =
        std::env::var("ZMQ_SERVER").unwrap_or_else(|_| DEFAULT_QMIO_ENDPOINT.to_string());
    let program_format = match backend {
        // `"aer"` is the entry-point default, so treat it (and the explicit
        // aliases) as OpenQASM for the QMIO path.
        "aer" | "qmio" | "openqasm" | "qasm" => QmioProgramFormat::OpenQasm,
        "qir" | "qir_text" => QmioProgramFormat::QirText,
        "qir_bitcode" | "qir_compiled" => QmioProgramFormat::QirBitcode,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"unknown qmio program format '{other}'; expected \"openqasm\", \"qir\", or \"qir_bitcode\""
			)))
        }
    };
    Ok(BackendConfig::Qmio {
        endpoint,
        program_format,
        // Sensible defaults; the optimisation level / results format are not yet
        // exposed as Python kwargs (kept extensible in BackendConfig::Qmio).
        optimization: 0,
        repetition_period: None,
        res_format: "binary_count".to_string(),
    })
}

/// Without the `qmio` feature, selecting the QMIO infrastructure fails with a
/// clear, actionable message instead of pulling a ZeroMQ stack into every build.
#[cfg(not(feature = "qmio"))]
fn build_qmio_backend_config(_backend: &str) -> PyResult<BackendConfig> {
    Err(pyo3::exceptions::PyValueError::new_err(
        "the 'qmio' infrastructure requires compiling polypus with --features qmio",
    ))
}

/// Whether `backend` selects the pure-Rust native statevector simulator, which
/// (unlike Aer) cannot consume a Qiskit `QuantumCircuit`.
fn is_native_backend(backend: &str) -> bool {
    matches!(backend, "polypus" | "statevector" | "polypus_statevector")
}

/// Validate the shot/QPU parameters at the Python-facing boundary, so the
/// orchestration layer can assume `shots >= 1` and `n_qpus >= 1` (contract
/// C-3). Rejecting here also avoids the division-by-zero panic that `n_qpus = 0`
/// would otherwise trigger in `DistributeByShotsRun`. Shared by every entry
/// point rather than duplicated per function.
fn validate_shots_and_qpus(shots: u32, n_qpus: u32) -> PyResult<()> {
    if shots < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shots must be >= 1, got {shots}"
        )));
    }
    if n_qpus < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_qpus must be >= 1, got {n_qpus}"
        )));
    }
    Ok(())
}

/// Interpret the `qc` argument of an entry point as a parameterised circuit
/// template. A `polypus.Circuit` becomes [`CircuitSource::Native`] (binding
/// will run GIL-free); any other object is assumed to be a Qiskit
/// `QuantumCircuit`, preserving the original behaviour.
fn extract_circuit_source(qc: &Bound<'_, PyAny>) -> CircuitSource {
    if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
        CircuitSource::Native(native.native().clone())
    } else {
        CircuitSource::Qiskit(qc.clone().unbind())
    }
}

/// Interpret the `qc` argument of `run_quantum_circuit` as a fully bound,
/// executable circuit:
/// - `polypus.Circuit` → native circuit (must have no free parameters),
/// - `str` → raw OpenQASM 2.0 program,
/// - anything else → Qiskit `QuantumCircuit` (original behaviour).
fn extract_bound_circuit(qc: &Bound<'_, PyAny>) -> PyResult<BoundCircuit> {
    if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
        let concrete = native.native().assign_parameters(&[]).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "circuit has unbound parameters and cannot be executed directly: {e}. \
				 Bind values first via to_qasm2(params) or use polypus.train"
            ))
        })?;
        return Ok(BoundCircuit::Native(concrete));
    }
    if let Ok(qasm) = qc.extract::<String>() {
        return Ok(BoundCircuit::Qasm2(qasm));
    }
    Ok(BoundCircuit::Qiskit(qc.clone().unbind()))
}

/// Function to run a quantum circuit called from Python.
///
/// `qc` may be a Qiskit `QuantumCircuit`, a `polypus.Circuit` (fully bound),
/// or an OpenQASM 2.0 string. `backend` selects the local device: `"aer"`
/// (default) or the pure-Rust `"polypus"` statevector simulator.
///
/// `seed` controls shot sampling (contract C-7) on every simulated backend —
/// native, Aer (`infrastructure="local"`) and CUNQA's simulated QPUs
/// (`infrastructure="cunqa"`): an explicit `seed` reproduces the counts
/// byte-for-byte, and omitting it draws a fresh OS-entropy seed so repeated
/// runs give genuinely independent noise. Passing a `seed` with
/// `infrastructure="qmio"` raises `ValueError` rather than silently ignoring
/// it, since that infrastructure is real hardware and cannot be seeded.
/// Returns a [`RunResult`] carrying the counts plus a manifest (`id`,
/// effective `seed`, `backend`, `infrastructure`) for logging and replay.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1, sim_method="automatic", noise_model=None, backend="aer", seed=None))]
// Rich FFI entry point mirroring a many-kwarg Python API; same rationale and
// convention as `train`/`qml_train` below.
#[allow(clippy::too_many_arguments)]
pub fn run_quantum_circuit<'py>(
    qc: Bound<'py, PyAny>,
    shots: u32,
    infrastructure: String,
    n_qpus: u32,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<pyo3::PyObject> {
    // Entry-point trace carrying the full circuit `Debug` repr on every call:
    // large and high-volume, so it stays at `debug` rather than the default log.
    log::debug!(
        "run_quantum_circuit called with qc: {qc:?}, shots: {shots}, \
         infrastructure: {infrastructure}, n_qpus: {n_qpus}, backend: {backend}, seed: {seed:?}"
    );
    validate_shots_and_qpus(shots, n_qpus)?;
    let bound_qc = extract_bound_circuit(&qc)?;
    if is_native_backend(backend) {
        if let BoundCircuit::Qiskit(_) = &bound_qc {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
				 pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"",
            ));
        }
    }
    // The QMIO path serialises circuits to QASM/QIR in Rust (GIL-free) and cannot
    // read a Qiskit QuantumCircuit, whose gates are only accessible via Python.
    if infrastructure == "qmio" {
        if let BoundCircuit::Qiskit(_) = &bound_qc {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
				 serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
				 2.0 string",
            ));
        }
    }
    // Resolve the shot-sampling seed. Every simulated backend (native, Aer,
    // CUNQA's simulated QPUs) is seeded by Polypus; `qmio` is real hardware, so
    // an explicit seed there would be silently ineffective — reject it rather
    // than give false confidence in reproducibility. When simulated, `None`
    // means "draw a fresh OS-entropy seed", resolved here so the effective
    // value can be reported in the manifest.
    let effective_seed: Option<u64> = if infrastructure == "qmio" {
        if seed.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
            ));
        }
        None
    } else {
        Some(seed.unwrap_or_else(random_seed))
    };

    let id = format!("run_{}_{}", n_qpus, infrastructure);
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        0,
        2,
    )?;
    let config = ExecutionConfig {
        id: id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        seed: effective_seed,
    };
    let args = AlgorithmArgs {
        qcs: vec![bound_qc],
        config,
    };

    let algorithm: Box<
        dyn AlgorithmTrait<Args = AlgorithmArgs, AlgorithmReturnType = PyResult<PyObject>>,
    > = if n_qpus == 1 {
        Box::new(AlgorithmSingleRun)
    } else {
        Box::new(DistributeByShotsRun)
    };

    // Keep the original counts payload intact and wrap it with the run manifest.
    let counts = algorithm.run(args)?;
    Python::with_gil(|py| {
        Py::new(
            py,
            RunResult {
                counts,
                id,
                seed: effective_seed,
                backend: backend.to_string(),
                infrastructure,
            },
        )
        .map(|result| result.into_any())
    })
}

/// Unified entry point: train a variational quantum circuit with a chosen optimizer.
///
/// `method` must be an instance of `DE`, `PSO`, or `QNG`.
///
/// `seed` makes the optimizer reproducible (contract C-7): precedence is the
/// explicit `seed` kwarg, then the optimizer object's `seed` field, then a fresh
/// OS-entropy value. On the native backend the same seed also drives shot
/// sampling, so a native-backend run reproduces exactly. Returns a
/// [`TrainResult`] exposing `best_params`, `best_fitness`, `iterations_run`,
/// `converged` and the effective `seed`.
///
/// Example:
///
/// ```ignore
///     result = polypus.train(
///         qc, polypus.DE(generations=200, population_size=50),
///         shots=1024, n_qpus=1, dimensions=4,
///         expectation_function=my_cost,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
///     )
/// ```
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn train<'py>(
    qc: Bound<'py, PyAny>,
    method: Bound<'py, PyAny>,
    shots: u32,
    n_qpus: u32,
    dimensions: u32,
    expectation_function: Bound<'py, PyAny>,
    infrastructure: String,
    nodes: u32,
    cores_per_qpu: u32,
    id: String,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    validate_shots_and_qpus(shots, n_qpus)?;
    // Resolve the seed that drives the optimizer's RNG (contract C-7): explicit
    // kwarg > optimizer object's `seed` field > fresh OS entropy. Unlike
    // `run_quantum_circuit`, a seed is always meaningful here — it seeds the
    // optimizer regardless of backend — and it is also threaded into
    // `ExecutionConfig::seed` so the native backend's shot sampling becomes
    // deterministic too, making a native-backend training run fully reproducible.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method));
    // Native circuits know their parameter count — catch a mismatch with the
    // requested optimisation dimensions before any QPU work starts.
    let circuit_source = extract_circuit_source(&qc);
    if let Some(num_params) = circuit_source.num_params() {
        if num_params != dimensions as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"dimensions ({dimensions}) does not match the circuit's free parameters ({num_params})"
			)));
        }
    }
    // The native statevector backend runs pure-Rust circuits only; a Qiskit
    // template can't be simulated without the interpreter.
    if is_native_backend(backend) {
        if let CircuitSource::Qiskit(_) = &circuit_source {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the native 'polypus' backend requires a native polypus.Circuit; \
				 got a Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"",
            ));
        }
    }
    // QMIO serialises GIL-free, so it likewise needs a native circuit, not a
    // Qiskit template that can only be read through the interpreter.
    if infrastructure == "qmio" {
        if let CircuitSource::Qiskit(_) = &circuit_source {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
				 serialization); got a Qiskit circuit. Build it with polypus.Circuit",
            ));
        }
    }
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
    )?;
    let config = Arc::new(ExecutionConfig {
        id: id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // Consumed only by the native backend; ignored by Aer/CUNQA/QMIO. Set
        // unconditionally so a native-backend training run reproduces exactly.
        seed: Some(effective_seed),
    });
    let backend = Infrastructure::create_backend(&config)?;
    // Shared error slot: the oracles record the first evaluation failure here
    // (the optimizer traits cannot return a `Result`) and it is surfaced by
    // `finish_optimization` after `optimize` returns.
    let errors = OracleErrorSlot::new();
    let oracle: Box<dyn EvaluationOracle> = Box::new(VqcOracle {
        circuit: circuit_source,
        config: Arc::clone(&config),
        backend,
        expectation_fn: expectation_function.unbind(),
        errors: errors.clone(),
    });

    if let Ok(de) = method.extract::<PyRef<DE>>() {
        let args = AlgorithmDifferentialEvolutionArgs {
            oracle,
            population_size: de.population_size,
            generations: de.generations,
            dimensions,
            tolerance: de.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            method.py(),
            AlgorithmDifferentialEvolution.optimize(args),
            &errors,
            effective_seed,
        );
    }

    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        let args = AlgorithmPSOArgs {
            oracle,
            population_size: pso.population_size,
            generations: pso.generations,
            dimensions,
            bounds: pso.bounds,
            inertia_weight: pso.inertia_weight,
            cognitive_weight: pso.cognitive_weight,
            social_weight: pso.social_weight,
            tolerance: pso.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            method.py(),
            AlgorithmPSO.optimize(args),
            &errors,
            effective_seed,
        );
    }

    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        let args = AlgorithmQNGArgs {
            oracle,
            max_iters: qng.max_iters,
            learning_rate: qng.learning_rate,
            finite_difference_step: qng.finite_difference_step,
            bounds: qng.bounds,
            dimensions,
            variance_oracle: Box::new(PyVarianceOracle {
                variance_function: qng.variance_function.clone_ref(method.py()),
                errors: errors.clone(),
            }),
            tikhonov_reg: qng.tikhonov_reg,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            method.py(),
            AlgorithmQNG.optimize(args),
            &errors,
            effective_seed,
        );
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, or polypus.QNG",
    ))
}

/// QML entry point: train a data-encoding VQC where `feature_map` encodes each
/// training sample and `ansatz` holds the trainable weights.
///
/// Internally, this function:
/// 1. Composes `feature_map` and `ansatz` into a single circuit.
/// 2. Pre-binds each row of `x_train` to the feature-map parameters, producing
///    one partially-bound circuit per training sample.
/// 3. Delegates to the chosen optimizer with `TrainMode::Qml`, so that for
///    every candidate parameter vector θ the optimizer binds θ to all training
///    circuits, runs them, and averages the expectation values into a single
///    fitness value.
///
/// `seed` follows the same precedence as [`train`] and makes the optimizer's
/// search reproducible; it returns a [`TrainResult`]. `qml.train` runs on the
/// Qiskit/Aer path (the native backend is rejected), and Aer's shot sampling
/// is now seeded too (contract C-7), so a `qml.train` run is fully
/// reproducible end-to-end given the same seed.
///
/// Example:
///
/// ```ignore
///     result = polypus.qml.train(
///         feature_map, ansatz, X_train,
///         polypus.PSO(generations=50, population_size=20, bounds=(0, np.pi)),
///         shots=1024, n_qpus=4, dimensions=12,
///         expectation_function=my_loss,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="qml_run",
///     )
/// ```
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn qml_train<'py>(
    feature_map: Bound<'py, PyAny>,
    ansatz: Bound<'py, PyAny>,
    x_train: Bound<'py, PyAny>,
    method: Bound<'py, PyAny>,
    shots: u32,
    n_qpus: u32,
    dimensions: u32,
    expectation_function: Bound<'py, PyAny>,
    infrastructure: String,
    nodes: u32,
    cores_per_qpu: u32,
    id: String,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    validate_shots_and_qpus(shots, n_qpus)?;
    // Same seed precedence as `train` (contract C-7): kwarg > optimizer field >
    // OS entropy. qml.train always runs on a Qiskit/Aer path (native rejected
    // below); this seed governs the optimizer's RNG and, since it's threaded
    // into ExecutionConfig::seed below, Aer's shot sampling too.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method));
    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path; the native statevector backend cannot consume a Qiskit
    // `QuantumCircuit`. Accept `backend` for API symmetry but reject native.
    if is_native_backend(backend) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the native 'polypus' backend is not supported for qml.train (feature maps \
			 and ansätze are Qiskit circuits); use backend=\"aer\"",
        ));
    }
    let py = feature_map.py();

    // 1. Compose feature_map + ansatz
    let composed = feature_map.call_method1("compose", (&ansatz,))?;

    // 2. Add measurements if the composed circuit has no classical bits.
    //    Qiskit's AerSimulator requires classical bits to return counts.
    let num_clbits: usize = composed.getattr("num_clbits")?.extract()?;
    if num_clbits == 0 {
        composed.call_method0("measure_all")?;
    }

    // 3. Collect feature-map parameters in their canonical (sorted-by-name) order
    let fm_params = feature_map.getattr("parameters")?;
    let builtins = PyModule::import(py, "builtins")?;
    let fm_params_list = builtins.call_method1("list", (&fm_params,))?;

    // 4. Pre-bind each training sample to the feature-map parameters.
    //    We pass a dict so Qiskit performs *partial* binding, leaving the ansatz
    //    parameters unbound for the optimizer to fill in later.
    let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
    let mut qcs: Vec<Py<PyAny>> = Vec::new();
    for row_result in x_train.try_iter()? {
        let row = row_result?;
        let param_dict = PyDict::new(py);
        for (param, val) in fm_params_list.try_iter()?.zip(row.try_iter()?) {
            param_dict.set_item(param?, val?)?;
        }
        let bound_qc = composed
            .call_method("assign_parameters", (&param_dict,), Some(&kwargs_assign))?
            .unbind();
        qcs.push(bound_qc);
    }

    if qcs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x_train must contain at least one training sample",
        ));
    }

    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path (native backend already rejected above): `backend` can only be
    // an Aer variant here.
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
    )?;
    let config = Arc::new(ExecutionConfig {
        id: id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // Consumed only by the native backend; ignored by Aer/CUNQA/QMIO. Set
        // unconditionally so a native-backend training run reproduces exactly.
        seed: Some(effective_seed),
    });
    let backend = Infrastructure::create_backend(&config)?;
    // Shared error slot (see `train`): oracles record the first evaluation
    // failure here and `finish_optimization` surfaces it after `optimize`.
    let errors = OracleErrorSlot::new();
    let oracle: Box<dyn EvaluationOracle> = Box::new(QmlOracle {
        training_circuits: qcs,
        config: Arc::clone(&config),
        backend,
        expectation_fn: expectation_function.unbind(),
        errors: errors.clone(),
    });

    if let Ok(de) = method.extract::<PyRef<DE>>() {
        let args = AlgorithmDifferentialEvolutionArgs {
            oracle,
            population_size: de.population_size,
            generations: de.generations,
            dimensions,
            tolerance: de.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            py,
            AlgorithmDifferentialEvolution.optimize(args),
            &errors,
            effective_seed,
        );
    }

    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        let args = AlgorithmPSOArgs {
            oracle,
            population_size: pso.population_size,
            generations: pso.generations,
            dimensions,
            bounds: pso.bounds,
            inertia_weight: pso.inertia_weight,
            cognitive_weight: pso.cognitive_weight,
            social_weight: pso.social_weight,
            tolerance: pso.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(py, AlgorithmPSO.optimize(args), &errors, effective_seed);
    }

    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        let args = AlgorithmQNGArgs {
            oracle,
            max_iters: qng.max_iters,
            learning_rate: qng.learning_rate,
            finite_difference_step: qng.finite_difference_step,
            bounds: qng.bounds,
            dimensions,
            variance_oracle: Box::new(PyVarianceOracle {
                variance_function: qng.variance_function.clone_ref(py),
                errors: errors.clone(),
            }),
            tikhonov_reg: qng.tikhonov_reg,
            seed: Some(effective_seed),
        };
        return finish_optimization(py, AlgorithmQNG.optimize(args), &errors, effective_seed);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, or polypus.QNG",
    ))
}

/// Number of backend resource-cleanup (`close`/`Drop`) failures recorded this
/// process. A `Drop` must never panic, so a failed teardown (e.g. releasing a
/// CUNQA SLURM allocation) is logged and counted rather than raised; this
/// exposes the consultable count to Python for diagnostics/monitoring.
#[pyfunction]
fn backend_cleanup_failures() -> u64 {
    crate::infrastructure::cleanup_failure_count()
}

#[pymodule]
pub fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::exceptions::register(m)?;
    m.add_class::<DE>()?;
    m.add_class::<PSO>()?;
    m.add_class::<QNG>()?;
    m.add_class::<Circuit>()?;
    m.add_class::<Param>()?;
    m.add_class::<RunResult>()?;
    m.add_class::<TrainResult>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(statevector, m)?)?;
    m.add_function(wrap_pyfunction!(init_logger, m)?)?;
    m.add_function(wrap_pyfunction!(backend_cleanup_failures, m)?)?;

    // qml submodule — exposes polypus.qml.train()
    let py = m.py();
    let qml = PyModule::new(py, "qml")?;
    qml.add_function(wrap_pyfunction!(qml_train, &qml)?)?;
    m.add_submodule(&qml)?;
    // Register in sys.modules so `import polypus.qml` also works
    let sys = PyModule::import(py, "sys")?;
    sys.getattr("modules")?.set_item("polypus.qml", &qml)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyString;
    use std::collections::HashMap;

    /// 3-qubit uniform superposition as OpenQASM 2.0. Its eight equally likely
    /// outcomes make "counts differ across unseeded runs" non-flaky (a 2-outcome
    /// Bell state collides far too often to assert inequality reliably).
    fn uniform3_qasm() -> String {
        polypus_circuit::ParameterizedCircuit::new(3)
            .h(0)
            .h(1)
            .h(2)
            .measure_all()
            .assign_parameters(&[])
            .expect("no free parameters")
            .to_qasm2()
    }

    /// Drive `run_quantum_circuit` on the native backend and return the effective
    /// seed reported in the manifest plus the wrapped counts payload.
    fn native_run(
        py: Python<'_>,
        qasm: &str,
        seed: Option<u64>,
    ) -> (Option<u64>, Vec<HashMap<String, u64>>) {
        let qc = PyString::new(py, qasm).into_any();
        let result = run_quantum_circuit(
            qc,
            500,
            "local".to_string(),
            1,
            "automatic",
            None,
            "polypus",
            seed,
        )
        .expect("native run_quantum_circuit succeeds");
        let bound = result.bind(py);
        let reported = bound
            .getattr("seed")
            .unwrap()
            .extract::<Option<u64>>()
            .unwrap();
        let counts = bound
            .getattr("counts")
            .unwrap()
            .extract::<Vec<HashMap<String, u64>>>()
            .unwrap();
        (reported, counts)
    }

    /// Acceptance criterion (defect #1) through the public entry point: an
    /// explicit seed round-trips into the manifest and reproduces the counts.
    #[test]
    fn native_seed_round_trips_and_reproduces_counts() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qasm = uniform3_qasm();
            let (s1, c1) = native_run(py, &qasm, Some(42));
            let (s2, c2) = native_run(py, &qasm, Some(42));
            assert_eq!(s1, Some(42));
            assert_eq!(s2, Some(42));
            assert_eq!(c1, c2, "same seed must reproduce counts");
        });
    }

    /// Acceptance criterion (defect #1), negative half: with no seed, each run
    /// draws a fresh entropy seed (reported in the manifest) and the counts
    /// differ across calls.
    #[test]
    fn native_omitted_seed_is_entropy_and_differs() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qasm = uniform3_qasm();
            let (s1, c1) = native_run(py, &qasm, None);
            let (s2, c2) = native_run(py, &qasm, None);
            assert!(
                s1.is_some() && s2.is_some(),
                "an entropy seed must be reported"
            );
            assert_ne!(s1, s2, "each unseeded run must draw a fresh seed");
            assert_ne!(c1, c2, "unseeded runs must produce independent noise");
        });
    }

    /// The manifest carries the full run metadata for logging/replay.
    #[test]
    fn native_manifest_reports_run_metadata() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qc = PyString::new(py, &uniform3_qasm()).into_any();
            let result = run_quantum_circuit(
                qc,
                100,
                "local".to_string(),
                1,
                "automatic",
                None,
                "polypus",
                Some(7),
            )
            .expect("native run succeeds");
            let bound = result.bind(py);
            let get = |name: &str| bound.getattr(name).unwrap().extract::<String>().unwrap();
            assert_eq!(get("backend"), "polypus");
            assert_eq!(get("infrastructure"), "local");
            assert_eq!(get("id"), "run_1_local");
            assert_eq!(
                bound
                    .getattr("seed")
                    .unwrap()
                    .extract::<Option<u64>>()
                    .unwrap(),
                Some(7)
            );
        });
    }

    /// A seed passed with `infrastructure="qmio"` is rejected, never silently
    /// dropped — that infrastructure is real hardware and Polypus cannot seed
    /// it, unlike the native, Aer, and CUNQA simulated backends.
    #[test]
    fn seed_rejected_for_qmio_hardware() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qasm = uniform3_qasm();
            let qc = pyo3::types::PyString::new(py, &qasm).into_any();
            let result = run_quantum_circuit(
                qc,
                100,
                "qmio".to_string(),
                1,
                "automatic",
                None,
                "aer",
                Some(3),
            );
            assert!(
                result.is_err(),
                "an explicit seed must be rejected for infrastructure=qmio"
            );
        });
    }

    #[test]
    fn resolve_optimizer_seed_follows_precedence() {
        // Explicit kwarg wins over the optimizer field...
        assert_eq!(resolve_optimizer_seed(Some(7), Some(9)), 7);
        // ...the optimizer field is the fallback...
        assert_eq!(resolve_optimizer_seed(None, Some(9)), 9);
        // ...and with neither set, a fresh entropy seed is drawn each time.
        assert_ne!(
            resolve_optimizer_seed(None, None),
            resolve_optimizer_seed(None, None)
        );
    }

    /// The seed the binding resolves must actually make the optimizer
    /// reproducible (same seed ⇒ identical outcome) and its entropy fallback must
    /// vary (no seed ⇒ different outcomes) — the `train`/`qml.train` acceptance
    /// criterion, exercised at the binding's seed-resolution boundary. `train()`
    /// itself needs a Python oracle + `polypus_python`, which the
    /// Python-runtime-free Rust suite forbids, so the optimizer is driven
    /// directly with a pure-Rust oracle (mirroring
    /// `optimizers.rs::de_is_deterministic_for_a_fixed_seed`, one layer up).
    #[test]
    fn resolved_seed_drives_the_optimizer_deterministically() {
        use polypus_optimizers::{
            AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, EvaluationOracle,
            Optimizer,
        };

        struct Quadratic;
        impl EvaluationOracle for Quadratic {
            fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
                candidates
                    .iter()
                    .map(|c| -c.iter().map(|x| (x - 0.7).powi(2)).sum::<f64>())
                    .collect()
            }
        }

        let run = |kwarg: Option<u64>, field: Option<u64>| {
            let seed = resolve_optimizer_seed(kwarg, field);
            AlgorithmDifferentialEvolution
                .optimize(AlgorithmDifferentialEvolutionArgs {
                    oracle: Box::new(Quadratic),
                    population_size: 20,
                    generations: 40,
                    dimensions: 3,
                    tolerance: 1e-9,
                    seed: Some(seed),
                })
                .expect("valid DE args optimize successfully")
        };

        // Same explicit seed ⇒ identical outcome (params, fitness, iters, converged).
        assert_eq!(run(Some(123), None), run(Some(123), None));
        // A method-field seed reproduces just as well.
        assert_eq!(run(None, Some(55)), run(None, Some(55)));
        // Omitted seed ⇒ (almost surely) different outcomes.
        assert_ne!(run(None, None), run(None, None));
    }
}
