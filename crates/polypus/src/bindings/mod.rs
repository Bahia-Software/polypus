use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyResult;

pub mod calibration;
pub mod circuit;
pub mod de;
pub mod logging;
pub mod observable;
pub mod pso;
pub mod qng;

use calibration::calibrate_parallel_threshold;
use circuit::{qft, statevector, Circuit, Param};
use de::DE;
use logging::init_logger;
use observable::{CachedCost, Ising, Qubo, SampleCost};
use pso::PSO;
use qng::QNG;

use crate::evaluation::{
    CircuitSource, CostObservable, Label, OracleErrorSlot, PyCallbackObservable, PyLabelledCost,
    PySampleCost, PyVarianceOracle, QmlObjective, QmlOracleFactory, SupervisedObjective,
    VqcOracleFactory,
};
use crate::infrastructure::execution_config::random_seed;
#[cfg(feature = "qmio")]
use crate::infrastructure::execution_config::QmioProgramFormat;
use crate::infrastructure::{
    BackendConfig, BoundCircuit, Counts, ExecutionConfig, Infrastructure, InfrastructureError,
    OptLevel, Planner, ShotDistributingPlanner,
};
use crate::orchestration::{
    DeConfig, Method, OracleError, PsoConfig, QngConfig, Resources, RunCircuitFlow, Scheduler,
    TrainFlow,
};
use polypus_optimizers::{OptimizationOutcome, VarianceOracle};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// Between-wave interrupt guard backed by CPython's pending-signal check, so a
/// Ctrl+C (`SIGINT`) aborts a run at the next wave boundary.
///
/// The [`Planner`] is now pyo3-free (in `polypus-backend`): instead of calling
/// `py.check_signals()` itself, it invokes the [`Interrupt`](crate::infrastructure::Interrupt)
/// guard optionally attached to the run's `CancelToken`. This is the guard the edge
/// attaches. A pending signal is boxed into `BackendError::External` and re-raised
/// verbatim at the FFI edge (contract C-1), preserving its `KeyboardInterrupt` class.
struct SignalInterrupt;

impl crate::infrastructure::Interrupt for SignalInterrupt {
    fn poll(&self) -> Result<(), crate::infrastructure::BackendError> {
        Python::with_gil(|py| py.check_signals()).map_err(crate::infrastructure::seam_error)
    }
}

/// A run token whose wave-boundary check honours a pending Ctrl+C, restoring the
/// interruptibility the planner's former inline `check_signals` provided.
fn interruptible_token() -> crate::infrastructure::CancelToken {
    crate::infrastructure::CancelToken::with_interrupt(Arc::new(SignalInterrupt))
}

/// Result of [`run_quantum_circuit`]: the measurement counts plus the run
/// manifest that lets a run be logged and replayed (contract C-7).
///
/// `counts` is the exact payload the runner produced before this wrapper
/// existed — a `list[dict[str, int]]` for a single-QPU run (the backend's default
/// atomic-wave planner), or a single merged `dict[str, int]` for a distributed
/// (`n_qpus > 1`) run (the shot-distributing planner); the
/// per-dict format is contract C-3. The manifest fields make a simulated run
/// reproducible: feeding the reported [`seed`](Self::seed) back into
/// `run_quantum_circuit(..., seed=...)` reproduces the counts byte-for-byte on
/// any of the native, Aer, or CUNQA (simulated-QPU) backends. `seed` is `None`
/// only for the `qmio` infrastructure, which runs on real hardware that
/// Polypus cannot seed.
#[pyclass(module = "polypus", frozen)]
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
#[pyclass(module = "polypus", frozen)]
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
    /// Best fitness recorded at the end of each generation/iteration, in order
    /// (`len() == iterations_run`). Exposes the optimizer's quality trajectory —
    /// for DE this is exactly the series its fitness-stagnation early stop is
    /// computed from (contract C-5).
    #[pyo3(get)]
    pub fitness_history: Vec<f64>,
    /// Effective RNG seed that drove the optimizer (and, on the native backend,
    /// shot sampling): the explicit `seed` kwarg, else the optimizer object's
    /// `seed`, else a fresh OS-entropy value (contract C-7).
    #[pyo3(get)]
    pub seed: u64,
    /// Run identifier used for logging, temp files and SLURM job names. This is
    /// the caller-supplied `id` prefix suffixed with a UUID v4 for uniqueness —
    /// see #75 — so it differs from the `id` kwarg that was passed in and must
    /// never be used to correlate runs by content, only to identify a run's
    /// allocation/log stream. Mirrors [`RunResult::id`].
    #[pyo3(get)]
    pub id: String,
}

#[pymethods]
impl TrainResult {
    fn __repr__(&self) -> String {
        format!(
            "TrainResult(id={:?}, best_fitness={}, iterations_run={}, converged={}, seed={}, best_params={:?})",
            self.id, self.best_fitness, self.iterations_run, self.converged, self.seed, self.best_params
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
    id: String,
) -> PyResult<PyObject> {
    Py::new(
        py,
        TrainResult {
            best_params: outcome.best_params,
            best_fitness: outcome.best_fitness,
            iterations_run: outcome.iterations_run,
            converged: outcome.converged,
            fitness_history: outcome.fitness_history,
            seed,
            id,
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

/// Suffix `base` with a UUID v4 so concurrent calls never share an
/// `ExecutionConfig::id` — which names SLURM families/allocations, temp files
/// and log streams. Used by `run_quantum_circuit`'s auto-generated id,
/// `train`'s and `qml_train`'s caller-supplied id (#45 / PR #70, #75): in
/// every case `base` is kept as a human-readable prefix on the effective id.
fn unique_id(base: &str) -> String {
    format!("{}_{}", base, Uuid::new_v4())
}

/// Announce the start of a training run at the default log level.
///
/// The fields are exactly the C-7 manifest data — id, infrastructure, backend,
/// n_qpus, shots, effective seed — so an operator watching a multi-hour run can
/// tell from the log alone what was executed where, and with which seed to
/// replay it. Shared by `train` and `qml_train`, whose start records are
/// identical; nothing high-volume (no circuit dumps) belongs at this level.
fn log_training_start(
    id: &str,
    infrastructure: &str,
    backend: &str,
    n_qpus: u32,
    shots: u32,
    seed: u64,
) {
    log::info!(
        "training run {id} starting: infrastructure={infrastructure}, backend={backend}, \
         n_qpus={n_qpus}, shots={shots}, seed={seed}"
    );
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
///
/// Both entry points funnel every DE/PSO/QNG branch through here, so this is
/// also where the run's completion is logged — once, on the success path only
/// (an oracle failure was already logged at `error!` where it was recorded).
/// `start` is the [`Instant`] captured on entry to the entry point, so the
/// reported duration covers the whole call.
///
/// [`OracleError::Config`] is a rejected optimizer configuration
/// (`population_size` too small for DE, empty PSO/QNG `bounds`, …), caught before
/// any oracle call — unlike an oracle failure it has nowhere else to be logged,
/// so it is logged here too; [`OracleError::Evaluation`] re-raises the oracle's
/// recorded failure with its original class preserved.
fn finish_optimization(
    py: Python<'_>,
    result: Result<OptimizationOutcome, OracleError>,
    seed: u64,
    id: String,
    start: Instant,
) -> PyResult<PyObject> {
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(OracleError::Evaluation(boxed)) => {
            // The oracle boxed an `EvaluationError` into the type-erased slot
            // (the scheduler crate is pyo3-free); recover it to re-raise with its
            // original Python class preserved. The slot only ever holds an
            // `EvaluationError`, so the downcast fails only in an impossible case,
            // where we still surface a typed evaluation error rather than panic.
            return Err(
                match boxed.downcast::<crate::evaluation::EvaluationError>() {
                    Ok(eval_err) => crate::exceptions::evaluation_error_to_pyerr(*eval_err),
                    Err(other) => crate::exceptions::EvaluationError::new_err(other.to_string()),
                },
            );
        }
        Err(OracleError::Config(config_err)) => {
            log::error!("run {id}: optimizer rejected the configuration: {config_err}");
            return Err(pyo3::exceptions::PyValueError::new_err(
                config_err.to_string(),
            ));
        }
    };
    log::info!(
        "training run {id} completed: iterations_run={}, converged={}, duration={:?}",
        outcome.iterations_run,
        outcome.converged,
        start.elapsed()
    );
    outcome_to_train_result(py, outcome, seed, id)
}

/// Parse a `polypus.DE` / `PSO` / `QNG` object into a pyo3-free [`Method`].
///
/// The QNG `variance_function` is adapted into a [`PyVarianceOracle`] here — the
/// only Python touch — so [`dispatch_optimizer`] itself stays Python-free. A
/// non-method object is a `TypeError`, exactly as the previous inline dispatch
/// (shared now by both `train` and `qml_train`).
fn method_from_pyclass(
    method: &Bound<'_, PyAny>,
    errors: &OracleErrorSlot,
    run_id: &str,
) -> PyResult<Method> {
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        return Ok(Method::De(DeConfig {
            generations: de.generations,
            population_size: de.population_size,
            tolerance: de.tolerance,
            patience: de.patience,
        }));
    }
    if let Ok(pso) = method.extract::<PyRef<PSO>>() {
        return Ok(Method::Pso(PsoConfig {
            generations: pso.generations,
            population_size: pso.population_size,
            bounds: pso.bounds,
            inertia_weight: pso.inertia_weight,
            cognitive_weight: pso.cognitive_weight,
            social_weight: pso.social_weight,
            tolerance: pso.tolerance,
        }));
    }
    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        let variance_oracle: Box<dyn VarianceOracle> = Box::new(PyVarianceOracle {
            variance_function: qng.variance_function.clone_ref(method.py()),
            errors: errors.clone(),
            run_id: run_id.to_string(),
        });
        return Ok(Method::Qng(
            QngConfig {
                max_iters: qng.max_iters,
                learning_rate: qng.learning_rate,
                finite_difference_step: qng.finite_difference_step,
                bounds: qng.bounds,
                tikhonov_reg: qng.tikhonov_reg,
            },
            variance_oracle,
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, or polypus.QNG",
    ))
}

/// Map the public `infrastructure` + `backend` strings and provider parameters
/// into a typed [`BackendConfig`]. Centralising this keeps the string→variant
/// mapping in one place and guarantees the config matches the selected backend.
///
/// `backend` selects the *device* within an infrastructure. For `local`:
/// `"aer"` (default) runs Qiskit Aer; `"polypus"` runs the pure-Rust native
/// statevector simulator. The choice is ignored for CUNQA, which manages its
/// own simulated QPUs.
///
/// `fusion` only applies to the native `"polypus"` backend (see
/// [`BackendConfig::LocalNative`]), the only one that fuses gates. It is an
/// `Option` so the three cases stay distinct:
///
/// * omitted (`None`): no opinion — the native backend uses its default
///   (`true`); every other backend ignores it, so sweeping the same kwargs
///   across backends needs no per-call special-casing (as with
///   `nodes`/`cores_per_qpu`, which `local`/`qmio` also ignore).
/// * `Some(false)`: "do not fuse". Every backend can honour this — the
///   non-native ones never fuse anyway — so it is always accepted; on the
///   native backend it forces a strictly gate-by-gate run.
/// * `Some(true)`: "fuse". Only the native backend can. Asking for it on a
///   backend that cannot fuse is a request that cannot be met, so it is
///   rejected here rather than silently ignored (which would mislead the
///   caller into believing fusion was in effect).
fn build_backend_config(
    infrastructure: &str,
    backend: &str,
    sim_method: &str,
    noise_model: Option<Py<PyAny>>,
    nodes: u32,
    cores_per_qpu: u32,
    fusion: Option<bool>,
) -> PyResult<BackendConfig> {
    let infrastructure_kind = Infrastructure::from_str(infrastructure)
        .map_err(crate::exceptions::backend_error_to_pyerr)?;
    // Only the native statevector backend fuses gates. An explicit `Some(true)`
    // anywhere else is an unmeetable request (see the doc above) — reject it
    // before building anything, so it never looks like it took effect. `None`
    // and `Some(false)` pass through: both are honourable everywhere.
    let is_native = matches!(infrastructure_kind, Infrastructure::Local)
        && matches!(backend, "polypus" | "statevector" | "polypus_statevector");
    if fusion == Some(true) && !is_native {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fusion=True applies only to the native backend=\"polypus\"; backend \"{backend}\" \
             on infrastructure \"{infrastructure}\" cannot fuse gates. Omit fusion, or pass \
             fusion=False for a gate-by-gate run."
        )));
    }
    match infrastructure_kind {
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
                Ok(BackendConfig::LocalNative {
                    fusion: fusion.unwrap_or(true),
                })
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

/// Validate the CUNQA allocation parameters at the Python-facing boundary. Only
/// the `"cunqa"` infrastructure consumes `nodes`/`cores_per_qpu` (they flow to
/// SLURM's `qraise` as the C-1 kwargs `n_nodes`/`cores_per_qpu` via
/// `CunqaBackend`); the `local` and `qmio` match arms of `build_backend_config`
/// ignore them, so validating unconditionally would reject harmless non-cunqa
/// calls. A zero for either value has no sane meaning once it reaches SLURM, so
/// reject it here before any seam call rather than forward it. Shared by every
/// entry point rather than duplicated per function.
fn validate_cunqa_allocation(infrastructure: &str, nodes: u32, cores_per_qpu: u32) -> PyResult<()> {
    if infrastructure != "cunqa" {
        return Ok(());
    }
    if nodes < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "nodes must be >= 1 for the 'cunqa' infrastructure, got {nodes}"
        )));
    }
    if cores_per_qpu < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cores_per_qpu must be >= 1 for the 'cunqa' infrastructure, got {cores_per_qpu}"
        )));
    }
    Ok(())
}

/// Maximum length accepted for the caller-supplied `id` prefix
/// (`train`/`qml.train`). Bounded so the effective id — prefix plus the 36-char
/// UUID v4 suffix [`unique_id`] appends — stays comfortably inside the limits
/// SLURM job names and filesystem path components impose downstream.
const MAX_ID_LEN: usize = 64;

/// Validate the caller-supplied `id` at the Python-facing boundary
/// (defense-in-depth, `docs/ENGINEERING.md` §8; contract C-9).
///
/// `id` becomes the CUNQA SLURM `family_name`/`family_id` (contract C-1,
/// `crate::infrastructure::cunqa`) and is documented as naming temp files and
/// log streams ([`ExecutionConfig::id`]), so an unrestricted string could carry
/// whitespace, path separators (`../`) or shell metacharacters into whatever
/// `qraise`/SLURM does with it downstream — which this crate cannot see. The
/// accepted charset is `[A-Za-z0-9._-]`, non-empty and at most
/// [`MAX_ID_LEN`] characters.
///
/// Checked before [`unique_id`] appends the UUID suffix, so a rejected id never
/// produces a partially-valid effective id. The charset is checked before the
/// length so a non-ASCII id gets the specific "invalid character" message
/// (and, past that check, every character is one byte, making the reported
/// length exact). Shared by every entry point rather than duplicated per
/// function, mirroring [`validate_shots_and_qpus`] and
/// [`validate_cunqa_allocation`].
fn validate_id(id: &str) -> PyResult<()> {
    if id.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not be empty",
        ));
    }
    if let Some(c) = id
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "id contains invalid character {c:?}; only ASCII letters, digits, \
			 '.', '_' and '-' are allowed (got {id:?})"
        )));
    }
    if id.len() > MAX_ID_LEN {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "id must be at most {MAX_ID_LEN} characters, got {} ({id:?})",
            id.len()
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
    // Anything else is a Qiskit `QuantumCircuit`: carry it opaquely through the
    // pyo3-free `BoundCircuit` enum via its `Foreign` escape hatch, so the Aer/CUNQA
    // backends still receive the native Qiskit object unchanged.
    Ok(crate::infrastructure::QiskitCircuit::into_bound(
        qc.clone().unbind(),
    ))
}

/// Interpret the `expectation_function` argument as a cost observable.
///
/// A `polypus.Qubo` / `polypus.Ising` becomes the corresponding native,
/// GIL-free evaluator; any other **callable** becomes a [`PyCallbackObservable`]
/// (the cost function is invoked once per unique bitstring per batch, then
/// aggregation runs in Rust). This is the dispatch mirror of
/// [`extract_circuit_source`], so the existing `expectation_function=<callable>`
/// API keeps working unchanged while native observables opt into the fast path.
fn extract_cost_observable(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn CostObservable>> {
    if let Ok(q) = obj.extract::<PyRef<'_, Qubo>>() {
        // Clone into the concrete Arc first, then coerce to the trait object
        // (`Arc::clone`'s generic is pinned by its `&Arc<T>` argument).
        let concrete = Arc::clone(&q.inner);
        let obs: Arc<dyn CostObservable> = concrete;
        return Ok(obs);
    }
    if let Ok(i) = obj.extract::<PyRef<'_, Ising>>() {
        let concrete = Arc::clone(&i.inner);
        let obs: Arc<dyn CostObservable> = concrete;
        return Ok(obs);
    }
    // A callable wrapped in polypus.CachedCost keeps a cross-generation memo;
    // a bare callable deduplicates only within each batch.
    if let Ok(cached) = obj.extract::<PyRef<'_, CachedCost>>() {
        let obs: Arc<dyn CostObservable> = Arc::new(PyCallbackObservable::new(
            cached.cost_fn.clone_ref(obj.py()),
            true,
        ));
        return Ok(obs);
    }
    if obj.is_callable() {
        let obs: Arc<dyn CostObservable> =
            Arc::new(PyCallbackObservable::new(obj.clone().unbind(), false));
        return Ok(obs);
    }
    // Not callable either; this only gives a better message than the one below.
    if obj.extract::<PyRef<'_, SampleCost>>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "polypus.SampleCost scores each training sample against its label, so it needs \
             labels: use it with polypus.qml.train(..., y_train=...). Without labels, pass a \
             callable (bitstring -> float)",
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expectation_function must be a callable (bitstring -> float), a \
         polypus.CachedCost(callable), or a polypus.Qubo / polypus.Ising observable",
    ))
}

/// Interpret `qml.train`'s `expectation_function` when `y_train` is given: a
/// callable is a per-shot `f(bitstring, label)` ([`PyLabelledCost`]),
/// `polypus.CachedCost(f)` the same with a memo, and `polypus.SampleCost(g)` a
/// per-sample `g(counts, label)` ([`PySampleCost`]). A `Qubo`/`Ising` cannot read
/// labels, so it is a `TypeError` rather than silently training without them.
fn extract_supervised_objective(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn SupervisedObjective>> {
    if let Ok(sample) = obj.extract::<PyRef<'_, SampleCost>>() {
        let objective: Arc<dyn SupervisedObjective> =
            Arc::new(PySampleCost::new(sample.cost_fn.clone_ref(obj.py())));
        return Ok(objective);
    }
    if let Ok(cached) = obj.extract::<PyRef<'_, CachedCost>>() {
        let objective: Arc<dyn SupervisedObjective> = Arc::new(PyLabelledCost::new(
            cached.cost_fn.clone_ref(obj.py()),
            true,
        ));
        return Ok(objective);
    }
    if obj.extract::<PyRef<'_, Qubo>>().is_ok() || obj.extract::<PyRef<'_, Ising>>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "a polypus.Qubo / polypus.Ising observable cannot read labels; with y_train, \
             expectation_function must be a callable (bitstring, label) -> float, a \
             polypus.CachedCost(callable) or a polypus.SampleCost(callable)",
        ));
    }
    if obj.is_callable() {
        let objective: Arc<dyn SupervisedObjective> =
            Arc::new(PyLabelledCost::new(obj.clone().unbind(), false));
        return Ok(objective);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "with y_train, expectation_function must be a callable (bitstring, label) -> float, \
         a polypus.CachedCost(callable) or a polypus.SampleCost((counts, label) -> float)",
    ))
}

/// Normalise `y_train` into one [`Label`] per sample (contract C-8). If every
/// element is an integer (Python or NumPy ints and bools) they are `Class` labels,
/// otherwise all are `Real`. A non-number or a nested row is a `TypeError`, and
/// `NaN`/`inf` a `ValueError`, naming the index. The caller checks the count.
fn extract_labels(y_train: &Bound<'_, PyAny>) -> PyResult<Vec<Label>> {
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::types::PyString;

    enum Raw {
        Int(i64),
        Float(f64),
    }
    // For error messages only, so an unreadable name is not itself an error.
    let type_name = |item: &Bound<'_, PyAny>| {
        item.get_type()
            .name()
            .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string())
    };
    let mut raw = Vec::new();
    for (idx, item) in y_train.try_iter()?.enumerate() {
        let item = item?;
        if item.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(format!(
                "y_train[{idx}] is a str; labels must be numbers (int class labels or float \
                 targets) — encode class names as integers first"
            )));
        }
        // Checked before the numeric reads: NumPy converts a length-1 array to a
        // float, which would silently accept a column vector.
        if let Ok(len) = item.len() {
            return Err(PyTypeError::new_err(format!(
                "y_train[{idx}] is a sequence ({} of length {len}), not a single label; pass \
                 one number per x_train row (class indices rather than one-hot rows; \
                 y.ravel() for a column vector)",
                type_name(&item)
            )));
        }
        // Through `__index__`: accepts Python and NumPy integers, never truncates
        // a float.
        if let Ok(value) = item.extract::<i64>() {
            raw.push(Raw::Int(value));
            continue;
        }
        // A NumPy bool has no `__index__`, but a boolean mask is a natural binary
        // `y`. Detected through `dtype.kind`, so NumPy is never imported.
        let is_numpy_bool = item
            .getattr("dtype")
            .and_then(|dtype| dtype.getattr("kind"))
            .and_then(|kind| kind.extract::<String>())
            .is_ok_and(|kind| kind == "b");
        if is_numpy_bool {
            raw.push(Raw::Int(i64::from(item.is_truthy()?)));
            continue;
        }
        match item.extract::<f64>() {
            Ok(value) if value.is_finite() => raw.push(Raw::Float(value)),
            Ok(value) => {
                return Err(PyValueError::new_err(format!(
                    "y_train[{idx}] is {value}; labels must be finite numbers"
                )));
            }
            Err(_) => {
                return Err(PyTypeError::new_err(format!(
                    "y_train[{idx}] has type {}; labels must be numbers (int class labels or \
                     float targets)",
                    type_name(&item)
                )));
            }
        }
    }
    let all_integers = raw.iter().all(|r| matches!(r, Raw::Int(_)));
    Ok(raw
        .into_iter()
        .map(|r| match r {
            Raw::Int(value) if all_integers => Label::Class(value),
            Raw::Int(value) => Label::Real(value as f64),
            Raw::Float(value) => Label::Real(value),
        })
        .collect())
}

/// Function to run a quantum circuit called from Python.
///
/// `qc` may be a Qiskit `QuantumCircuit`, a `polypus.Circuit` (fully bound),
/// or an OpenQASM 2.0 string. The meaning of `backend` depends on
/// `infrastructure`: for `"local"` it selects the device — `"aer"` (default)
/// or the pure-Rust `"polypus"` statevector simulator; for `"qmio"` it
/// instead selects the program format submitted to the QPU (`"openqasm"`
/// (default), `"qir"`, or `"qir_bitcode"`); CUNQA ignores it.
///
/// `seed` controls shot sampling (contract C-7) on every simulated backend —
/// native, Aer (`infrastructure="local"`) and CUNQA's simulated QPUs
/// (`infrastructure="cunqa"`): an explicit `seed` reproduces the counts
/// byte-for-byte, and omitting it draws a fresh OS-entropy seed so repeated
/// runs give genuinely independent noise. Passing a `seed` with
/// `infrastructure="qmio"` raises `ValueError` rather than silently ignoring
/// it, since that infrastructure is real hardware and cannot be seeded.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are meaningful
/// only for `infrastructure="cunqa"` (they become CUNQA's `qraise` parameters);
/// the `local` and `qmio` infrastructures ignore them. For `cunqa` both must be
/// `>= 1` — a zero has no sane meaning once it reaches SLURM and is rejected.
/// They default to `nodes=1, cores_per_qpu=2`.
///
/// Returns a [`RunResult`] carrying the counts plus a manifest (`id`,
/// effective `seed`, `backend`, `infrastructure`) for logging and replay.
///
/// `fusion` applies only to `backend="polypus"`, the one backend that fuses
/// gates. Omit it (the default) and each backend does its own thing — the
/// native one fuses, the rest never did. `fusion=False` forces a strictly
/// gate-by-gate run and is accepted everywhere (a non-fusing backend already
/// meets it). `fusion=True` on any backend that cannot fuse is rejected with a
/// `ValueError` rather than silently ignored, so it never looks like it took
/// effect.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1, nodes=1, cores_per_qpu=2, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None))]
pub fn run_quantum_circuit<'py>(
    qc: Bound<'py, PyAny>,
    shots: u32,
    infrastructure: String,
    n_qpus: u32,
    nodes: u32,
    cores_per_qpu: u32,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    fusion: Option<bool>,
) -> PyResult<pyo3::PyObject> {
    let start = Instant::now();
    // Entry-point trace carrying the full circuit `Debug` repr on every call:
    // large and high-volume, so it stays at `debug` rather than the default log.
    log::debug!(
        "run_quantum_circuit called with qc: {qc:?}, shots: {shots}, \
         infrastructure: {infrastructure}, n_qpus: {n_qpus}, backend: {backend}, seed: {seed:?}"
    );
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu)?;
    let bound_qc = extract_bound_circuit(&qc)?;
    if is_native_backend(backend) && bound_qc.is_foreign() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
             pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"",
        ));
    }
    // The QMIO path serialises circuits to QASM/QIR in Rust (GIL-free) and cannot
    // read a Qiskit QuantumCircuit, whose gates are only accessible via Python.
    if infrastructure == "qmio" && bound_qc.is_foreign() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
             serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
             2.0 string",
        ));
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

    // Append a UUID v4 so two concurrent calls with identical `n_qpus`/
    // `infrastructure` never collide: the id names SLURM families/allocations,
    // temp files and log streams (see ExecutionConfig::id), so a byte-identical
    // id across runs is a real hazard, not a cosmetic one. The `run_{n}_{infra}`
    // prefix is kept for human-readable debuggability.
    let id = unique_id(&format!("run_{}_{}", n_qpus, infrastructure));
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
        fusion,
    )?;
    // Only the native statevector backend consults the gate-parallel threshold,
    // so surface the one-time default-visible warning only when this run
    // actually resolves to it — never for Aer/CUNQA/QMIO. Gated on the resolved
    // `BackendConfig`, not the raw `backend` string, so it tracks the real
    // dispatch (e.g. `backend="polypus"` under `infrastructure="cunqa"`, which
    // Aer-simulates, correctly does not warn).
    if matches!(backend_config, BackendConfig::LocalNative { .. }) {
        calibration::warn_if_using_default_threshold(qc.py())?;
    }
    let config = ExecutionConfig {
        id: id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        seed: effective_seed,
    };
    // Lifecycle record at the default level, carrying the C-7 manifest data only
    // (the circuit itself stays in the `debug!` above): enough to see what ran
    // where, and with which seed to replay it.
    log::info!(
        "run {id} starting: infrastructure={infrastructure}, backend={backend}, \
         n_qpus={n_qpus}, shots={shots}, seed={effective_seed:?}"
    );

    // Release the GIL for the whole run, mirroring `train`: circuit execution is
    // GIL-free on the native backend (and internally reacquires the GIL where
    // Aer/CUNQA need it), so holding it here would stall every other Python thread
    // and (with the Planner's between-wave check_signals) keep Ctrl+C from taking
    // effect until the run finishes. See docs/ENGINEERING.md §3. `n_qpus > 1`
    // selects the shot-distributing planner (which apportions this one circuit's
    // shots across replicas and merges, conserving the total per C-3); otherwise
    // the backend's default atomic-wave planner runs the circuit as-is.
    let counts_result =
        qc.py()
            .allow_threads(move || -> Result<Vec<Counts>, InfrastructureError> {
                let backend = Infrastructure::create_backend(&config)
                    .map_err(InfrastructureError::Backend)?;
                let planner: Option<Arc<dyn Planner>> = if n_qpus == 1 {
                    None
                } else {
                    Some(Arc::new(ShotDistributingPlanner::new(n_qpus)))
                };
                let resources = Resources::new(backend, planner, Arc::new(config.run_params()))?;
                let scheduler = Scheduler::ephemeral(resources);
                // A Ctrl+C during the run aborts it at the next wave boundary: the
                // token carries the `check_signals`-backed interrupt guard the
                // pyo3-free planner calls between waves.
                let out = scheduler.run_cancellable(
                    RunCircuitFlow {
                        circuits: vec![bound_qc],
                        shots,
                    },
                    &interruptible_token(),
                );
                scheduler.close();
                out
            });
    let counts_vec = counts_result.map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    // Completion counterpart of the start record above. There is no
    // iterations/convergence notion on this path (those are `TrainResult`
    // fields), so this reports only the run and how long it took.
    log::info!("run {id} completed: duration={:?}", start.elapsed());
    // Convert at the FFI boundary, preserving the historical output shapes:
    // `n_qpus == 1` yields one `list[dict]` (one map per circuit); `n_qpus > 1`
    // yields the single merged `dict`.
    let counts: PyObject = Python::with_gil(|py| -> PyResult<PyObject> {
        if n_qpus == 1 {
            Ok(counts_vec.into_pyobject(py)?.into_any().unbind())
        } else {
            let total = counts_vec.into_iter().next().unwrap_or_default();
            let py_dict = PyDict::new(py);
            for (k, v) in total {
                py_dict.set_item(k, v)?;
            }
            Ok(py_dict.into_any().unbind())
        }
    })?;
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
/// `converged`, the effective `seed` and the effective `id`.
///
/// The `id` kwarg is a human-readable *prefix*, not the literal effective id:
/// a UUID v4 is appended for uniqueness (mirroring [`run_quantum_circuit`]),
/// so `TrainResult.id` differs from what was passed in and names the run's
/// SLURM allocation / temp files / log stream (see #75). Because it travels
/// into those downstream names, the prefix is restricted to `[A-Za-z0-9._-]`,
/// non-empty and at most 64 characters (contract C-9); anything else — a space,
/// a path separator, a shell metacharacter — raises `ValueError` naming the
/// offending character, before the UUID suffix is generated.
///
/// Interruption: pressing Ctrl+C (or otherwise sending `SIGINT`) stops the
/// optimization promptly and raises `KeyboardInterrupt` in Python, instead of
/// waiting for the run to finish. An exception raised by
/// `expectation_function` propagates the same way, as itself.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are consumed only
/// by `infrastructure="cunqa"`; `local`/`qmio` accept but ignore them. For
/// `cunqa` both must be `>= 1` (a zero is meaningless to SLURM and rejected).
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
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None))]
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
    fusion: Option<bool>,
) -> PyResult<PyObject> {
    let start = Instant::now();
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu)?;
    validate_id(&id)?;
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
        fusion,
    )?;
    // Suffix the caller-supplied `id` with a UUID v4 so two concurrent training
    // runs sharing the same `id` never collide on the SLURM family/allocation,
    // temp files or log streams named by ExecutionConfig::id (#75, mirroring
    // run_quantum_circuit). The prefix is kept for debuggability and the
    // effective id is reported back in the TrainResult.
    let effective_id = unique_id(&id);
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // Consumed only by the native backend; ignored by Aer/CUNQA/QMIO. Set
        // unconditionally so a native-backend training run reproduces exactly.
        seed: Some(effective_seed),
    });
    // Log before `backend` is shadowed below: the `&str` device selector is what
    // belongs in the manifest record, not the constructed backend object.
    log_training_start(
        &effective_id,
        &infrastructure,
        backend,
        n_qpus,
        shots,
        effective_seed,
    );
    let backend = Infrastructure::create_backend(&config)
        .map_err(crate::exceptions::backend_error_to_pyerr)?;
    // Shared error slot: the oracle records the first evaluation failure here
    // (the optimizer traits cannot return a `Result`) and it is surfaced by
    // `finish_optimization` after `optimize` returns.
    let errors = OracleErrorSlot::new();
    // A callable stays a Python-callback observable (optimized fallback); a
    // polypus.Qubo/Ising opts into the native, GIL-free evaluation path.
    let observable = extract_cost_observable(&expectation_function)?;
    let method_enum = method_from_pyclass(&method, &errors, &effective_id)?;
    // Pair the backend with its default planner (the atomic-wave SequentialPlanner)
    // and validate the pairing up front.
    let resources = Resources::new(backend, None, Arc::new(config.run_params()))
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    let scheduler = Scheduler::ephemeral(resources);
    let flow = TrainFlow {
        factory: VqcOracleFactory {
            circuit: circuit_source,
            observable,
        },
        method: method_enum,
        dimensions,
        seed: effective_seed,
        errors: errors.clone(),
    };
    // Release the GIL for the whole optimization: parameter binding and native
    // simulation are GIL-free, so holding it would stall every other Python
    // thread and (with the Planner's between-wave check_signals) keep Ctrl+C from
    // taking effect until the run ends. See docs/ENGINEERING.md §3. The run token
    // carries the `check_signals`-backed interrupt guard the pyo3-free planner
    // calls between waves, so Ctrl+C still aborts at the next wave boundary.
    let token = interruptible_token();
    let result = method
        .py()
        .allow_threads(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    finish_optimization(method.py(), result, effective_seed, effective_id, start)
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
///    circuits, runs them, and averages the per-sample values into a single
///    fitness value (maximised).
///
/// # Labels (`y_train`)
///
/// Without `y_train`, every sample is scored by the same
/// `expectation_function(bitstring) -> float`. With `y_train` (keyword-only, one
/// label per `x_train` row), each sample is scored against its own label and
/// `expectation_function` is one of:
///
/// - a callable `f(bitstring, label) -> float`, averaged over each sample's shots
///   (the expected accuracy, when `f` returns `1.0` for a correct read-out);
/// - `polypus.CachedCost(f)`, the same memoised by `(label, bitstring)`;
/// - `polypus.SampleCost(g)`, where `g(counts, label) -> float` sees the sample's
///   whole distribution, for non-linear losses such as a log-likelihood.
///
/// Integer labels reach the objective as `int`; if any label is not an integer,
/// all are passed as `float`. `y_train` is validated before anything runs
/// (contract C-8).
///
/// `seed` follows the same precedence as [`train`] and makes the optimizer's
/// search reproducible; it returns a [`TrainResult`]. `qml.train` runs on the
/// Qiskit/Aer path (the native backend is rejected), and Aer's shot sampling
/// is now seeded too (contract C-7), so a `qml.train` run is fully
/// reproducible end-to-end given the same seed.
///
/// As in [`train`], the `id` kwarg is a human-readable *prefix*: a UUID v4 is
/// appended for uniqueness (mirroring [`run_quantum_circuit`]), so the returned
/// `TrainResult.id` differs from what was passed in and names the run's SLURM
/// allocation / temp files / log stream (see #75). It carries the same charset
/// restriction as [`train`] — `[A-Za-z0-9._-]`, non-empty, at most 64
/// characters (contract C-9) — and an invalid prefix raises `ValueError`
/// naming the offending character.
///
/// Interruption: same behaviour as [`train`] — Ctrl+C stops the optimization
/// promptly and raises `KeyboardInterrupt` rather than waiting for the run to
/// finish, and an exception raised by `expectation_function` propagates as
/// itself.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are consumed only
/// by `infrastructure="cunqa"`; `local`/`qmio` accept but ignore them. For
/// `cunqa` both must be `>= 1` (a zero is meaningless to SLURM and rejected).
///
/// Example (supervised binary classifier, parity read-out):
///
/// ```ignore
///     def correct(bitstring, label):
///         return float(bitstring.count("1") % 2 == label)
///
///     result = polypus.qml.train(
///         feature_map, ansatz, X_train,
///         polypus.PSO(generations=50, population_size=20, bounds=(0, np.pi)),
///         shots=1024, n_qpus=4, dimensions=12,
///         expectation_function=correct,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="qml_run",
///         y_train=y_train,
///     )
/// ```
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None, *, y_train=None))]
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
    y_train: Option<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    let start = Instant::now();
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu)?;
    validate_id(&id)?;
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
    // The optimizer searches a `dimensions`-wide vector and binds it to the
    // ansatz's free parameters; a mismatch surfaces later as a cryptic Qiskit
    // binding error inside the oracle. Catch it upfront, mirroring how `train`
    // validates `dimensions` against the circuit's parameter count (contract
    // C-8). `len()` calls `__len__`, working on Qiskit's ParameterView the same
    // as on a list.
    let num_ansatz_params = ansatz.getattr("parameters")?.len()?;
    if num_ansatz_params != dimensions as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dimensions ({dimensions}) does not match the ansatz's free parameters ({num_ansatz_params})"
        )));
    }
    // Type-checked before any circuit is composed; the count is checked once
    // `x_train` has been read (contract C-8).
    let labels = y_train.as_ref().map(extract_labels).transpose()?;
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
    // Each row must supply exactly one value per feature-map parameter. Zipping
    // the two iterators would stop at the shorter one — a longer row silently
    // drops features, a shorter row leaves feature-map parameters unbound and
    // fails later as a cryptic Qiskit error inside the oracle. Materialize both
    // lengths and reject a mismatch upfront with the row index and both lengths
    // (contract C-8).
    let fm_len = fm_params_list.len()?;
    for (row_idx, row_result) in x_train.try_iter()?.enumerate() {
        let row = row_result?;
        let row_len = row.len()?;
        if row_len != fm_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x_train row {row_idx} has {row_len} features, but feature_map expects {fm_len} \
                 (len(feature_map.parameters))"
            )));
        }
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

    // One label per row (contract C-8), and an objective that can read labels:
    // both checked before any backend, and so any CUNQA allocation, exists.
    let supervised = match labels {
        Some(labels) => {
            if labels.len() != qcs.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "y_train has {} labels, but x_train has {} rows; qml.train needs exactly \
                     one label per training sample",
                    labels.len(),
                    qcs.len()
                )));
            }
            Some((extract_supervised_objective(&expectation_function)?, labels))
        }
        None => None,
    };

    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path (native backend already rejected above): `backend` can only be
    // an Aer variant here, which never fuses. Pass `None` (no fusion opinion)
    // rather than `Some(true)` — the latter would trip build_backend_config's
    // "fusion=True on a non-fusing backend" rejection for a value the caller
    // never chose.
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
        None,
    )?;
    // Suffix the caller-supplied `id` with a UUID v4 (see `train` and #75) so
    // concurrent qml.train runs sharing the same `id` never collide on the
    // SLURM family/allocation, temp files or log streams named by
    // ExecutionConfig::id. The effective id is reported back in the TrainResult.
    let effective_id = unique_id(&id);
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // Consumed only by the native backend; ignored by Aer/CUNQA/QMIO. Set
        // unconditionally so a native-backend training run reproduces exactly.
        seed: Some(effective_seed),
    });
    // Log before `backend` is shadowed below (see `train`): the device selector
    // string is the manifest value, not the constructed backend object.
    log_training_start(
        &effective_id,
        &infrastructure,
        backend,
        n_qpus,
        shots,
        effective_seed,
    );
    let backend = Infrastructure::create_backend(&config)
        .map_err(crate::exceptions::backend_error_to_pyerr)?;
    // Shared error slot (see `train`): the oracle records the first evaluation
    // failure here and `finish_optimization` surfaces it after `optimize`.
    let errors = OracleErrorSlot::new();
    let objective = match supervised {
        Some((objective, labels)) => QmlObjective::Supervised { objective, labels },
        None => QmlObjective::Unsupervised(extract_cost_observable(&expectation_function)?),
    };
    let method_enum = method_from_pyclass(&method, &errors, &effective_id)?;
    let resources = Resources::new(backend, None, Arc::new(config.run_params()))
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    let scheduler = Scheduler::ephemeral(resources);
    let flow = TrainFlow {
        factory: QmlOracleFactory {
            training_circuits: qcs,
            objective,
        },
        method: method_enum,
        dimensions,
        seed: effective_seed,
        errors: errors.clone(),
    };
    // Release the GIL for the optimization (see `train` and docs/ENGINEERING.md
    // §3): the Qiskit binding + Aer calls re-acquire it, and the run token's
    // `check_signals`-backed interrupt guard — which the pyo3-free planner calls
    // between waves — keeps Ctrl+C prompt.
    let token = interruptible_token();
    let result = py.allow_threads(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    finish_optimization(py, result, effective_seed, effective_id, start)
}

/// Number of backend resource-cleanup (`close`/`Drop`) failures recorded this
/// process. A `Drop` must never panic, so a failed teardown (e.g. releasing a
/// CUNQA SLURM allocation) is logged and counted rather than raised; this
/// exposes the consultable count to Python for diagnostics/monitoring.
#[pyfunction]
fn backend_cleanup_failures() -> u64 {
    crate::infrastructure::cleanup_failure_count()
}

/// Polypus — distributed quantum computing (Rust core, Python bindings).
///
/// Same project, three names: install `polypus-quantum` from PyPI
/// (`pip install polypus-quantum`), but import it as `polypus`; the Rust crate
/// is `polypus` (`cargo add polypus`). The PyPI name `polypus` is taken by an
/// unrelated project, so the distribution is published as `polypus-quantum`
/// while the import name and the Rust crate both stay `polypus`.
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
    m.add_class::<Qubo>()?;
    m.add_class::<Ising>()?;
    m.add_class::<CachedCost>()?;
    m.add_class::<SampleCost>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(statevector, m)?)?;
    m.add_function(wrap_pyfunction!(init_logger, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(backend_cleanup_failures, m)?)?;

    // qml submodule — exposes polypus.qml.train()
    let py = m.py();
    let qml = PyModule::new(py, "polypus.qml")?;
    qml.add_function(wrap_pyfunction!(qml_train, &qml)?)?;
    // Attach under the short key: `add_submodule` would use the dotted `__name__`
    // verbatim as the attribute name, breaking `polypus.qml.train` access.
    m.add("qml", &qml)?;
    // Register in sys.modules so `import polypus.qml` also works
    let sys = PyModule::import(py, "sys")?;
    sys.getattr("modules")?.set_item("polypus.qml", &qml)?;

    // circuits.templates subpackage — exposes polypus.circuits.templates.qft()
    let circuits = PyModule::new(py, "polypus.circuits")?;
    let templates = PyModule::new(py, "polypus.circuits.templates")?;
    templates.add_function(wrap_pyfunction!(qft, &templates)?)?;
    // Attach under short keys (see the qml note above) and mirror both levels
    // into sys.modules so `import polypus.circuits.templates` also resolves.
    circuits.add("templates", &templates)?;
    m.add("circuits", &circuits)?;
    let modules = sys.getattr("modules")?;
    modules.set_item("polypus.circuits", &circuits)?;
    modules.set_item("polypus.circuits.templates", &templates)?;

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
            1,
            2,
            "automatic",
            None,
            "polypus",
            seed,
            None,
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
                1,
                2,
                "automatic",
                None,
                "polypus",
                Some(7),
                None,
            )
            .expect("native run succeeds");
            let bound = result.bind(py);
            let get = |name: &str| bound.getattr(name).unwrap().extract::<String>().unwrap();
            assert_eq!(get("backend"), "polypus");
            assert_eq!(get("infrastructure"), "local");
            // The id keeps the human-readable `run_{n}_{infra}_` prefix but is
            // suffixed with a per-call UUID (see the uniqueness test below), so
            // only the stable prefix is asserted here.
            assert!(
                get("id").starts_with("run_1_local_"),
                "id must keep the run_{{n}}_{{infra}}_ prefix, got {}",
                get("id")
            );
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

    /// Regression for #45: two runs with identical arguments must not share an
    /// id. The id names SLURM families/allocations, temp files and log streams,
    /// so a collision (as with the old `run_{n}_{infra}` format) could make two
    /// concurrent runs clobber one another. The per-call UUID v4 guarantees
    /// uniqueness while the `run_1_local_` prefix stays stable for debugging.
    #[test]
    fn auto_generated_id_is_unique_per_call() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qasm = uniform3_qasm();
            let id_of = || {
                let qc = PyString::new(py, &qasm).into_any();
                let result = run_quantum_circuit(
                    qc,
                    100,
                    "local".to_string(),
                    1,
                    1,
                    2,
                    "automatic",
                    None,
                    "polypus",
                    Some(7),
                    None,
                )
                .expect("native run succeeds");
                result
                    .bind(py)
                    .getattr("id")
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            };
            let id1 = id_of();
            let id2 = id_of();
            assert!(id1.starts_with("run_1_local_") && id2.starts_with("run_1_local_"));
            assert_ne!(
                id1, id2,
                "ids from identical calls must differ (unique per run)"
            );
        });
    }

    /// Regression for #75, at the shared-helper boundary. `train`/`qml_train`
    /// take a caller-supplied `id` and used to pass it verbatim into
    /// `ExecutionConfig::id` — which names SLURM families/allocations, temp
    /// files and log streams — so two concurrent training runs sharing an `id`
    /// (the doc examples all use `id="run1"`/`id="qml_run"`) could clobber one
    /// another. All three id-bearing entry points now funnel through
    /// [`unique_id`], which keeps the caller's string as a human-readable prefix
    /// and appends a UUID v4 for uniqueness.
    ///
    /// This asserts that guarantee directly on `unique_id` rather than by
    /// calling `train`/`qml_train`: those need a live Python oracle
    /// (`polypus_python` → Qiskit/Aer) and `qml_train` needs Qiskit for its
    /// feature-map/ansatz composition, but the Rust suite is Python-runtime-free
    /// by design (see `.github/workflows/ci.yml` and
    /// `resolved_seed_drives_the_optimizer_deterministically`). The end-to-end
    /// `train`/`qml.train` id-uniqueness checks live in the Python suite
    /// (`tests/python/test_seed_reproducibility.py`), mirroring
    /// `run_quantum_circuit`'s `test_id_is_unique_across_identical_calls`.
    #[test]
    fn unique_id_keeps_prefix_and_is_unique_per_call() {
        // Same base (as with `train(..., id="run1")` twice) ⇒ unique ids that
        // both keep the supplied string as a prefix.
        for base in ["run1", "qml_run", "run_1_local"] {
            let id1 = unique_id(base);
            let id2 = unique_id(base);
            let prefix = format!("{base}_");
            assert!(
                id1.starts_with(&prefix) && id2.starts_with(&prefix),
                "the caller-supplied id must survive as a prefix, got {id1} / {id2}"
            );
            assert_ne!(
                id1, id2,
                "ids built from an identical base must differ (unique per call)"
            );
        }
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
                1,
                2,
                "automatic",
                None,
                "aer",
                Some(3),
                None,
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
                    patience: 20,
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

    // ─────────────────────────────────────────────────────────────────────────
    // Direct unit coverage of the private entry-point helpers.
    //
    // `tests/python/test_backend_selection.py` already exercises these through
    // the full `#[pyfunction]`s end-to-end; these tests pin the helpers
    // themselves, which the audit flagged as having no unit-level coverage. They
    // are deliberately lean — the goal is closing that gap, not re-proving the
    // end-to-end behaviour. Nothing here needs an installed package: the
    // `Py<PyAny>` arguments are `py.None()` against a bare interpreter.
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn validate_shots_and_qpus_accepts_the_minimum() {
        assert!(validate_shots_and_qpus(1, 1).is_ok());
        assert!(validate_shots_and_qpus(1024, 8).is_ok());
    }

    #[test]
    fn validate_shots_and_qpus_rejects_zero() {
        // `n_qpus = 0` used to reach `DistributeByShotsRun` and divide by zero;
        // `shots = 0` silently ran an empty execution (contract C-3).
        assert!(validate_shots_and_qpus(0, 1).is_err());
        assert!(validate_shots_and_qpus(1, 0).is_err());
        assert!(validate_shots_and_qpus(0, 0).is_err());
    }

    #[test]
    fn build_backend_config_selects_the_local_variants() {
        let aer = build_backend_config("local", "aer", "automatic", None, 1, 2, None)
            .expect("aer is a valid local backend");
        assert!(matches!(
            aer,
            BackendConfig::Local {
                ref backend,
                ref sim_method,
                noise_model: None,
            } if backend == "AerSimulator" && sim_method == "automatic"
        ));

        for name in ["polypus", "statevector", "polypus_statevector"] {
            let native = build_backend_config("local", name, "automatic", None, 1, 2, Some(true))
                .unwrap_or_else(|_| panic!("'{name}' selects the native backend"));
            assert!(matches!(
                native,
                BackendConfig::LocalNative { fusion: true }
            ));
        }
    }

    /// The `fusion` kwarg reaches `BackendConfig::LocalNative` unchanged, in
    /// both directions — it is not silently forced to `true`.
    #[test]
    fn build_backend_config_forwards_fusion_for_the_native_backend() {
        let with_fusion =
            build_backend_config("local", "polypus", "automatic", None, 1, 2, Some(true))
                .expect("polypus is a valid local backend");
        assert!(matches!(
            with_fusion,
            BackendConfig::LocalNative { fusion: true }
        ));

        let without_fusion =
            build_backend_config("local", "polypus", "automatic", None, 1, 2, Some(false))
                .expect("polypus is a valid local backend");
        assert!(matches!(
            without_fusion,
            BackendConfig::LocalNative { fusion: false }
        ));
    }

    /// Omitting `fusion` (`None`) leaves the native backend on its default
    /// (`true`) — the crate-wide `StatevectorSimulator::default().fusion`.
    #[test]
    fn build_backend_config_defaults_fusion_to_enabled_when_omitted() {
        let native = build_backend_config("local", "polypus", "automatic", None, 1, 2, None)
            .expect("polypus is a valid local backend");
        assert!(matches!(
            native,
            BackendConfig::LocalNative { fusion: true }
        ));
    }

    /// `fusion=True` on a backend that cannot fuse is rejected: a request that
    /// cannot be met must not look like it took effect. `fusion=False` and an
    /// omitted `fusion` — both honourable everywhere, since a non-fusing backend
    /// already runs gate-by-gate — are accepted unchanged.
    #[test]
    fn build_backend_config_rejects_fusion_true_on_non_fusing_backends() {
        pyo3::prepare_freethreaded_python();
        for (infra, backend) in [("local", "aer"), ("cunqa", "aer")] {
            let err = build_backend_config(infra, backend, "automatic", None, 1, 2, Some(true))
                .expect_err("fusion=True on a non-fusing backend must be rejected");
            Python::with_gil(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("fusion=True applies only to"),
                    "unexpected error: {err}"
                );
            });

            // fusion=False and an omitted fusion must build a config, not error.
            build_backend_config(infra, backend, "automatic", None, 1, 2, Some(false))
                .expect("fusion=False is accepted everywhere");
            build_backend_config(infra, backend, "automatic", None, 1, 2, None)
                .expect("omitted fusion is accepted everywhere");
        }
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_local_backend() {
        let err = build_backend_config("local", "does-not-exist", "automatic", None, 1, 2, None)
            .expect_err("an unknown local backend must be rejected");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown local backend"));
        });
    }

    #[test]
    fn build_backend_config_rejects_a_noise_model_on_the_native_backend() {
        // The native simulator is noiseless by construction, so a noise_model
        // must be an error rather than silently ignored. Any Python object will
        // do — the check is `Option::is_some`, not a Qiskit type check.
        pyo3::prepare_freethreaded_python();
        let noise_model = Python::with_gil(|py| py.None());
        let err = build_backend_config(
            "local",
            "polypus",
            "automatic",
            Some(noise_model),
            1,
            2,
            None,
        )
        .expect_err("a noise model on the native backend must be rejected");
        Python::with_gil(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("noise_model"),
                "the message must name the offending kwarg: {err}"
            );
        });
    }

    #[test]
    fn build_backend_config_keeps_a_noise_model_for_aer() {
        pyo3::prepare_freethreaded_python();
        let noise_model = Python::with_gil(|py| py.None());
        let config = build_backend_config(
            "local",
            "aer",
            "density_matrix",
            Some(noise_model),
            1,
            2,
            None,
        )
        .expect("aer accepts a noise model");
        assert!(matches!(
            config,
            BackendConfig::Local {
                noise_model: Some(_),
                ref sim_method,
                ..
            } if sim_method == "density_matrix"
        ));
    }

    #[test]
    fn build_backend_config_forwards_the_cunqa_allocation() {
        let config = build_backend_config("cunqa", "aer", "statevector", None, 3, 4, None)
            .expect("cunqa is a valid infrastructure");
        assert!(matches!(
            config,
            BackendConfig::Cunqa {
                nodes: 3,
                cores_per_qpu: 4,
                ref sim_method,
                ..
            } if sim_method == "statevector"
        ));
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_infrastructure() {
        let err = build_backend_config("quantum-cloud", "aer", "automatic", None, 1, 2, None)
            .expect_err("an unknown infrastructure must be rejected");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown infrastructure"));
        });
    }

    #[test]
    fn extract_bound_circuit_reads_a_qasm_string() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qasm = uniform3_qasm();
            let bound = extract_bound_circuit(&PyString::new(py, &qasm).into_any())
                .expect("a str is an OpenQASM 2.0 program");
            match bound {
                BoundCircuit::Qasm2(text) => assert_eq!(text, qasm),
                _ => panic!("a str must become BoundCircuit::Qasm2"),
            }
        });
    }

    #[test]
    fn extract_bound_circuit_treats_anything_else_as_a_qiskit_circuit() {
        // The classification is "not a polypus.Circuit and not a str", so any
        // other object lands on the Qiskit arm — which is exactly why the
        // native/qmio guards downstream can reject it without importing Qiskit.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let bound = extract_bound_circuit(py.None().bind(py))
                .expect("a non-Circuit, non-str object is assumed to be a Qiskit circuit");
            // A Qiskit circuit rides through the pyo3-free enum's `Foreign` hatch.
            assert!(bound.is_foreign());
        });
    }

    #[test]
    fn extract_bound_circuit_reads_a_fully_bound_native_circuit() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let circuit = Py::new(
                py,
                Circuit {
                    inner: polypus_circuit::ParameterizedCircuit::new(1)
                        .x(0)
                        .measure_all(),
                },
            )
            .expect("the pyclass instantiates");
            let bound = extract_bound_circuit(circuit.bind(py).as_any())
                .expect("a parameter-free polypus.Circuit is executable as-is");
            assert!(matches!(bound, BoundCircuit::Native(_)));
        });
    }

    #[test]
    fn extract_bound_circuit_rejects_a_native_circuit_with_free_parameters() {
        // An unbound `polypus.Circuit` cannot be executed directly; it must be a
        // clear ValueError pointing at `train`, not a panic inside binding.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let circuit = Py::new(
                py,
                Circuit {
                    inner: polypus_circuit::ParameterizedCircuit::new(1)
                        .ry(0, polypus_circuit::GateParam::Param(0))
                        .measure_all(),
                },
            )
            .expect("the pyclass instantiates");
            // `BoundCircuit` is not `Debug`, so unwrap the `Result` by hand
            // rather than widening a production type just for a test.
            let err = match extract_bound_circuit(circuit.bind(py).as_any()) {
                Err(err) => err,
                Ok(_) => panic!("an unbound template cannot be executed directly"),
            };
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("unbound parameters"),
                "the message must explain what to do instead: {err}"
            );
        });
    }

    /// `extract_labels` over a Python expression (builtins only: no NumPy here).
    fn labels_of(py: Python<'_>, expr: &std::ffi::CStr) -> PyResult<Vec<Label>> {
        let y_train = py
            .eval(expr, None, None)
            .expect("the test expression evaluates");
        extract_labels(&y_train)
    }

    /// The `(class, message)` of a rejected `y_train`.
    fn label_error(py: Python<'_>, expr: &std::ffi::CStr) -> (String, String) {
        let err = labels_of(py, expr).expect_err("these labels must be rejected");
        let class = err
            .get_type(py)
            .name()
            .expect("an exception type has a name")
            .to_string();
        (class, err.to_string())
    }

    #[test]
    fn extract_labels_keeps_all_integer_labels_as_classes() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert_eq!(
                labels_of(py, c"[0, 2, -1, True]").expect("integers are labels"),
                [0, 2, -1, 1].map(Label::Class),
                "ints and bools stay class labels"
            );
            // y_train is only iterated, so a generator works too.
            assert_eq!(
                labels_of(py, c"(k % 2 for k in range(3))").expect("a generator is iterable"),
                [0, 1, 0].map(Label::Class)
            );
        });
    }

    #[test]
    fn extract_labels_makes_every_label_a_float_when_any_is_not_an_integer() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert_eq!(
                labels_of(py, c"[0, 1.5, 2]").expect("mixed numbers are labels"),
                [0.0, 1.5, 2.0].map(Label::Real),
                "one float makes every label a float"
            );
            // The type decides, not the value.
            assert_eq!(
                labels_of(py, c"[1.0, 0.0]").expect("floats are labels"),
                [1.0, 0.0].map(Label::Real)
            );
        });
    }

    #[test]
    fn extract_labels_rejects_non_numbers_and_nested_rows_naming_the_index() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let (class, msg) = label_error(py, c"[0, 'cat']");
            assert_eq!(class, "TypeError");
            assert!(msg.contains("y_train[1] is a str"), "{msg}");

            let (class, msg) = label_error(py, c"[0, [1, 0]]");
            assert_eq!(class, "TypeError");
            assert!(
                msg.contains("y_train[1] is a sequence (list of length 2)"),
                "a one-hot row is not a label: {msg}"
            );

            let (class, msg) = label_error(py, c"[None]");
            assert_eq!(class, "TypeError");
            assert!(msg.contains("y_train[0] has type NoneType"), "{msg}");
        });
    }

    #[test]
    fn extract_labels_rejects_non_finite_values_naming_the_index() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let (class, msg) = label_error(py, c"[0.5, float('nan')]");
            assert_eq!(class, "ValueError");
            assert!(msg.contains("y_train[1] is NaN"), "{msg}");

            let (class, msg) = label_error(py, c"[float('-inf')]");
            assert_eq!(class, "ValueError");
            assert!(msg.contains("y_train[0] is -inf"), "{msg}");
        });
    }

    #[test]
    fn extract_supervised_objective_accepts_callables_and_the_two_wrappers() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let callable = py
                .eval(c"lambda bitstring, label: 0.0", None, None)
                .expect("a lambda evaluates");
            let cached = Py::new(
                py,
                CachedCost {
                    cost_fn: callable.clone().unbind(),
                },
            )
            .expect("the pyclass instantiates");
            let sample = Py::new(
                py,
                SampleCost {
                    cost_fn: callable.clone().unbind(),
                },
            )
            .expect("the pyclass instantiates");
            for accepted in [
                callable,
                cached.into_bound(py).into_any(),
                sample.into_bound(py).into_any(),
            ] {
                assert!(
                    extract_supervised_objective(&accepted).is_ok(),
                    "{accepted:?} is a valid supervised objective"
                );
            }
        });
    }

    #[test]
    fn extract_supervised_objective_rejects_observables_that_cannot_read_labels() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let qubo = Py::new(
                py,
                Qubo {
                    inner: Arc::new(
                        polypus_observable::QuboObservable::new(
                            1,
                            vec![(0, 1.0)],
                            vec![],
                            0.0,
                            1.0,
                        )
                        .expect("a one-variable QUBO is valid"),
                    ),
                },
            )
            .expect("the pyclass instantiates");
            let err = extract_supervised_objective(qubo.bind(py).as_any())
                .err()
                .expect("a Qubo cannot read labels");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("cannot read labels"), "{err}");

            let not_callable = 42i64.into_pyobject(py).expect("an int converts").into_any();
            let err = extract_supervised_objective(&not_callable)
                .err()
                .expect("a non-callable is not an objective");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("with y_train"), "{err}");
        });
    }

    #[test]
    fn extract_cost_observable_rejects_a_sample_cost_without_labels() {
        // Without labels (polypus.train, or qml.train without y_train) a SampleCost
        // gets a TypeError that says why, not the generic message.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let sample = Py::new(
                py,
                SampleCost {
                    cost_fn: py
                        .eval(c"lambda counts, label: 0.0", None, None)
                        .expect("a lambda evaluates")
                        .unbind(),
                },
            )
            .expect("the pyclass instantiates");
            let err = extract_cost_observable(sample.bind(py).as_any())
                .err()
                .expect("a SampleCost needs labels");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("y_train"), "{err}");
        });
    }
}
