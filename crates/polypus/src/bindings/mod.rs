use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBool, PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use pyo3::PyResult;

pub mod adam;
pub mod calibration;
pub mod circuit;
pub mod cmaes;
mod compatibility;
pub mod de;
pub mod expr;
pub mod external;
pub mod logging;
pub mod observable;
pub mod pso;
pub mod qml;
pub mod qng;
pub mod rotosolve;
pub mod spsa;

use adam::Adam;
use calibration::calibrate_parallel_threshold;
use circuit::{qft, statevector, Circuit, Param};
use cmaes::CMAES;
use compatibility::backend_compatibility;
use de::DE;
use external::External;
use logging::init_logger;
use observable::{CachedCost, Ising, Qubo, SampleCost};
use pso::PSO;
use qng::QNG;
use rotosolve::Rotosolve;
use spsa::Spsa;

use crate::evaluation::{
    CircuitSource, CostObservable, Label, OracleErrorSlot, PyCallbackObservable, PyLabelledCost,
    PySampleCost, PyVarianceOracle, QmlObjective, QmlOracleFactory, QmlSource, SupervisedObjective,
    VqcOracleFactory,
};
use crate::infrastructure::execution_config::{random_id_bytes, random_seed};
use crate::infrastructure::{
    merge_counts, validate_walltime, BackendConfig, BoundCircuit, Counts, EntropyError,
    ExecutionConfig, Infrastructure, InfrastructureError, OptLevel, OptionValue, Planner,
    ShotDistributingPlanner, DEFAULT_WALLTIME,
};
use crate::orchestration::{
    AdamConfig, CmaesConfig, DeConfig, ExternalConfig, FinalEvaluation, Method, OracleError,
    PsoConfig, QngConfig, Resources, RotosolveConfig, RunCircuitFlow, Scheduler, SpsaConfig,
    TrainFlow, TrainOutcome,
};
use polypus_optimizers::{OptimizationOutcome, VarianceOracle};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Between-wave interrupt guard backed by CPython's pending-signal check, so a
/// Ctrl+C (`SIGINT`) aborts a run at the next wave boundary.
///
/// The [`Planner`] is now pyo3-free (in `polypus-backend`): instead of calling
/// `py.check_signals()` itself, it invokes the [`Interrupt`](crate::infrastructure::Interrupt)
/// guard optionally attached to the run's `CancelToken`. This is the guard the edge
/// attaches. A pending signal is boxed into `BackendError::External` and re-raised
/// verbatim at the FFI edge (contract C-1), preserving its `KeyboardInterrupt` class.
///
/// `poll` runs while the thread is detached (the run is inside `py.detach`), so
/// attaching here is a fresh attach, not a reuse of the caller's: `Python::attach`
/// could panic, hang, or attach to a finalizing interpreter, if the interpreter
/// were shutting down (on any Python version). It attaches with
/// [`attach_or`](crate::infrastructure::attach_or) instead, and an unreachable
/// interpreter aborts the run with
/// [`InterpreterUnavailable`](crate::infrastructure::InterpreterUnavailable), the
/// error every other site reached while detached returns. The edge raises it as
/// `KeyboardInterrupt`, the same class a Ctrl+C gets, since the run's result
/// could never be handed back to Python (docs/ENGINEERING.md §3 and §9).
struct SignalInterrupt;

impl crate::infrastructure::Interrupt for SignalInterrupt {
    fn poll(&self) -> Result<(), crate::infrastructure::BackendError> {
        crate::infrastructure::attach_or(
            || Err(crate::infrastructure::InterpreterUnavailable.into()),
            |py| {
                py.check_signals()
                    .map_err(crate::infrastructure::seam_error)
            },
        )
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
/// `counts` is always a `list[dict[str, int]]`, whatever `n_qpus` is (contract
/// C-3). For [`run_quantum_circuit`] it holds one dict per QPU replica, in the
/// order the shots were apportioned (length `n_qpus`, so length 1 for a
/// single-QPU run); a replica apportioned zero shots (`shots < n_qpus`) is an
/// empty dict. `merged_counts` is the key-by-key sum of those dicts as one
/// `dict[str, int]` — the circuit's total over all `shots`. For [`qml_predict`]
/// `counts` holds one dict per input row, and since the rows are different
/// circuits, not replicas of one, `merged_counts` is `None` there. Every dict
/// follows the C-3 format (keys in ascending bitstring order).
///
/// The manifest fields make a simulated run
/// reproducible: feeding the reported [`seed`](Self::seed) back into
/// `run_quantum_circuit(..., seed=...)` reproduces the counts byte-for-byte on
/// any of the native, Aer, or CUNQA (simulated-QPU) backends. `seed` is `None`
/// only for the `qmio` infrastructure, which runs on real hardware that
/// Polypus cannot seed.
#[pyclass(module = "polypus", frozen)]
pub struct RunResult {
    /// Measurement counts: always a `list[dict]`, one per QPU replica (or per
    /// `qml.predict` row). See the type docs.
    #[pyo3(get)]
    pub counts: Py<PyAny>,
    /// The key-by-key sum of every dict in `counts` (a `dict`), or `None` for
    /// `qml.predict`, whose entries are distinct circuits.
    #[pyo3(get)]
    pub merged_counts: Py<PyAny>,
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
    /// Every field, in declaration order. `counts` / `merged_counts` use their
    /// Python `repr` (sorted keys, C-3), and `seed` reads `None` rather than
    /// Rust's `Some(..)`/`None`, so the output looks like a Python value.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let seed = self
            .seed
            .map_or_else(|| "None".to_string(), |s| s.to_string());
        Ok(format!(
            "RunResult(counts={}, merged_counts={}, id={:?}, seed={seed}, backend={:?}, \
             infrastructure={:?})",
            self.counts.bind(py).repr()?,
            self.merged_counts.bind(py).repr()?,
            self.id,
            self.backend,
            self.infrastructure
        ))
    }
}

/// A counts map as a Python `dict` with its keys in ascending bitstring order
/// (contract C-3). A `HashMap` iterates in an order that differs per map and per
/// process, and a `dict` keeps insertion order, so a float reduction over the
/// returned `.items()` could otherwise differ in the last bit between runs.
fn counts_to_pydict<'py>(py: Python<'py>, counts: &Counts) -> PyResult<Bound<'py, PyDict>> {
    let mut outcomes: Vec<(&String, &u64)> = counts.iter().collect();
    outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let dict = PyDict::new(py);
    for (bitstring, n) in outcomes {
        dict.set_item(bitstring, n)?;
    }
    Ok(dict)
}

/// One [`counts_to_pydict`] dict per map, in order, as a Python `list`.
fn counts_to_pylist<'py>(py: Python<'py>, counts: &[Counts]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for map in counts {
        list.append(counts_to_pydict(py, map)?)?;
    }
    Ok(list)
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
    /// early-stopping criterion fired); for `External`, the batches evaluated
    /// and told.
    #[pyo3(get)]
    pub iterations_run: usize,
    /// Whether the optimizer's convergence criterion was satisfied; for
    /// `External`, whether the external optimizer ended the run with an empty
    /// `ask()` (`False` when `max_evaluations` stopped it).
    #[pyo3(get)]
    pub converged: bool,
    /// Best fitness recorded at the end of each generation/iteration, in order
    /// (`len() == iterations_run`). Exposes the optimizer's quality trajectory —
    /// for DE this is exactly the series its fitness-stagnation early stop is
    /// computed from (contract C-5).
    #[pyo3(get)]
    pub fitness_history: Vec<f64>,
    /// With `final_shots`, the candidate the final batch chose: the best
    /// candidates (`best_params` and, for DE, PSO, CMA-ES and `External`, up
    /// to four more), re-evaluated once with `final_shots` shots after the optimizer
    /// stopped; the highest value wins, `best_params` on a tie. `None`
    /// without `final_shots` (contract C-5).
    #[pyo3(get)]
    pub final_params: Option<Vec<f64>>,
    /// The value of [`final_params`](Self::final_params) in the final batch;
    /// `None` without `final_shots`.
    #[pyo3(get)]
    pub final_fitness: Option<f64>,
    /// Effective RNG seed that drove the optimizer (and, on the native backend,
    /// shot sampling): the explicit `seed` kwarg, else the optimizer object's
    /// `seed`, else a fresh OS-entropy value (contract C-7). With `External`,
    /// whose library owns its seed, it drives the shot sampling only.
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
        // Unchanged without `final_shots`; with it, the final value after the
        // best and the final point after the best one.
        let final_fitness = self
            .final_fitness
            .map_or(String::new(), |value| format!(", final_fitness={value}"));
        let final_params = self
            .final_params
            .as_ref()
            .map_or(String::new(), |params| format!(", final_params={params:?}"));
        format!(
            "TrainResult(id={:?}, best_fitness={}{final_fitness}, iterations_run={}, \
             converged={}, seed={}, best_params={:?}{final_params})",
            self.id,
            self.best_fitness,
            self.iterations_run,
            self.converged,
            self.seed,
            self.best_params
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
    final_evaluation: Option<FinalEvaluation>,
    seed: u64,
    id: String,
) -> PyResult<Py<PyAny>> {
    let (final_params, final_fitness) = match final_evaluation {
        Some(evaluation) => (Some(evaluation.params), Some(evaluation.fitness)),
        None => (None, None),
    };
    Py::new(
        py,
        TrainResult {
            best_params: outcome.best_params,
            best_fitness: outcome.best_fitness,
            iterations_run: outcome.iterations_run,
            converged: outcome.converged,
            fitness_history: outcome.fitness_history,
            final_params,
            final_fitness,
            seed,
            id,
        },
    )
    .map(|result| result.into_any())
}

/// Resolve the optimizer seed for `train` / `qml_train` (contract C-7).
///
/// Precedence: the explicit `seed` kwarg wins when provided; otherwise the
/// `seed` field pinned on the `DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA`/`Rotosolve` instance;
/// otherwise a fresh OS-entropy value, drawn with `draw` (`random_seed` in production). The chosen
/// value both drives the optimizer and is reported back in the [`TrainResult`],
/// so the run can be replayed.
fn resolve_optimizer_seed(
    kwarg_seed: Option<u64>,
    method_seed: Option<u64>,
    draw: impl FnOnce() -> Result<u64, EntropyError>,
) -> PyResult<u64> {
    seed_or_entropy(kwarg_seed.or(method_seed), draw)
}

/// `seed` if given, else a fresh OS-entropy value from `draw` (`random_seed` in
/// production) — the default-seed rule every entry point shares (contract C-7).
/// `draw` is only called when `seed` is `None`, so an explicit seed never touches
/// the OS entropy source. A failed draw raises `polypus.BackendError`, never a
/// panic (C-1).
fn seed_or_entropy(
    seed: Option<u64>,
    draw: impl FnOnce() -> Result<u64, EntropyError>,
) -> PyResult<u64> {
    match seed {
        Some(seed) => Ok(seed),
        None => draw().map_err(|e| {
            crate::exceptions::backend_error_to_pyerr(crate::infrastructure::BackendError::from(e))
        }),
    }
}

/// Suffix `base` with a UUID v4 so concurrent calls never share an
/// `ExecutionConfig::id` — which names SLURM families/allocations, temp files
/// and log streams. Used by `run_quantum_circuit`'s auto-generated id,
/// `train`'s and `qml_train`'s caller-supplied id (#45 / PR #70, #75): in
/// every case `base` is kept as a human-readable prefix on the effective id.
///
/// A failed OS-entropy read raises `polypus.BackendError`, never a panic (C-1,
/// C-7) — which is why the UUID is built from [`random_id_bytes`] rather than
/// drawn by the `uuid` crate itself, which panics on that failure (#264).
fn unique_id(base: &str) -> PyResult<String> {
    unique_id_from(base, random_id_bytes)
}

/// `base` suffixed with a UUID v4 built from the bytes `draw` returns
/// (`random_id_bytes` in production): the body of [`unique_id`], with the
/// entropy source injectable so its failure path can be tested.
fn unique_id_from(
    base: &str,
    draw: impl FnOnce() -> Result<[u8; 16], EntropyError>,
) -> PyResult<String> {
    let bytes = draw().map_err(|e| {
        crate::exceptions::backend_error_to_pyerr(crate::infrastructure::BackendError::from(e))
    })?;
    Ok(format!(
        "{}_{}",
        base,
        uuid::Builder::from_random_bytes(bytes).into_uuid()
    ))
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
/// whichever of `DE`/`PSO`/`QNG`/`CMAES`/`Adam`/`SPSA`/`Rotosolve` it is (`None` if it is none
/// of them — the type error is surfaced later by the dispatch that actually runs the
/// optimizer). `External` has no seed field: its library owns its seed (contract
/// C-7), so it is `None` too, and the run's seed then only fixes the shot
/// sampling.
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
    if let Ok(cmaes) = method.extract::<PyRef<CMAES>>() {
        return cmaes.seed;
    }
    if let Ok(adam) = method.extract::<PyRef<Adam>>() {
        return adam.seed;
    }
    if let Ok(spsa) = method.extract::<PyRef<Spsa>>() {
        return spsa.seed;
    }
    if let Ok(rotosolve) = method.extract::<PyRef<Rotosolve>>() {
        return rotosolve.seed;
    }
    None
}

/// `ValueError` when `method` is a `polypus.External` and an `initial_point`
/// is given: the external library chooses its own start. Checked after
/// External's configuration and instead of the point's length and finiteness,
/// so the message says why any point is refused.
fn check_external_initial_point(
    method: &Bound<'_, PyAny>,
    initial_point: Option<&[f64]>,
) -> PyResult<()> {
    if initial_point.is_none() || method.extract::<PyRef<External>>().is_err() {
        return Ok(());
    }
    Err(crate::exceptions::configuration_error(
        "polypus.External takes no initial_point: the external optimizer chooses its own \
         start. Give the starting point to the library, or to the initial_point of the \
         polypus.ask_tell adapter (from_optuna, from_nevergrad); for pycma it is the x0 of \
         cma.CMAEvolutionStrategy",
    ))
}

/// Whether `method` is a `polypus.Rotosolve`, which only some circuits and
/// objectives qualify for.
fn is_rotosolve(method: &Bound<'_, PyAny>) -> bool {
    method.extract::<PyRef<Rotosolve>>().is_ok()
}

/// The `ValueError` for a circuit Rotosolve cannot train, with the first
/// reason the check found.
fn rotosolve_circuit_error(reason: &str) -> PyErr {
    crate::exceptions::configuration_error(format!(
        "polypus.Rotosolve cannot train this circuit: {reason}. Rotosolve needs every \
         trainable parameter to be exactly one angle of an rx, ry, rz, rxx, rzz, p, u1, u, \
         u2 or u3 gate (also ryy, rzx, or the θ of an r, in a Qiskit circuit), as ±θ + c, \
         so that the fitness is a sinusoid of each parameter; train this circuit with \
         another optimizer"
    ))
}

/// Check, before anything runs, that `polypus.Rotosolve` can train `circuit`
/// (issue #398): every trainable parameter the angle of exactly one
/// single-Pauli rotation, as `±θ + c`. A `polypus.Circuit` is checked in
/// Rust ([`ParameterizedCircuit::check_single_frequency`]); a Qiskit circuit
/// by `polypus_python.rotosolve`, with `trainable` its trainable parameters
/// (`None`: all of them; `qml.train` passes the ansatz's).
///
/// [`ParameterizedCircuit::check_single_frequency`]: crate::circuit::ParameterizedCircuit::check_single_frequency
fn check_rotosolve_circuit(
    circuit: &Bound<'_, PyAny>,
    source: Option<&CircuitSource>,
    trainable: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(CircuitSource::Native(native)) = source {
        return native
            .check_single_frequency()
            .map_err(|problem| rotosolve_circuit_error(&problem.to_string()));
    }
    let py = circuit.py();
    let problem = PyModule::import(py, "polypus_python.rotosolve")?
        .getattr("single_frequency_problem")?
        .call1((circuit, trainable))?;
    if problem.is_none() {
        return Ok(());
    }
    Err(rotosolve_circuit_error(&problem.extract::<String>()?))
}

/// Check the optimizer's preconditions, `initial_point` included, before
/// anything runs in `train`/`qml.train` (contract C-5): every check the
/// optimizer makes before its first RNG draw, in its order, so its
/// configuration errors come before the point's (`dimensions` finite values,
/// within `bounds` for PSO), through [`Method::check_preconditions`]. A failure
/// raises `ValueError` with the optimizer's own message before any QPU is
/// requested; a `method` that is no optimizer raises the `TypeError` of
/// [`method_from_pyclass`]. `External` refuses any point after its
/// configuration, with [`check_external_initial_point`]'s message.
fn validate_method(
    method: &Bound<'_, PyAny>,
    dimensions: u32,
    initial_point: Option<&[f64]>,
) -> PyResult<()> {
    // A throwaway `Method`: the checks read only its configuration, never the
    // QNG variance oracle or the External optimizer, so its error slot and run
    // id are placeholders.
    let parsed = method_from_pyclass(method, dimensions, &OracleErrorSlot::new(), "")?;
    // `External`'s point is refused below, with the message that names the
    // adapters, rather than with dispatch's.
    let point = match parsed {
        Method::External(..) => None,
        _ => initial_point,
    };
    parsed
        .check_preconditions(dimensions, point)
        .map_err(|e| crate::exceptions::configuration_error(e.to_string()))?;
    check_external_initial_point(method, initial_point)
}

/// Check the `final_shots` of `train`/`qml.train` before anything runs
/// (contract C-5), as `shots` is checked: `None`, or an integer from 1 to
/// 4294967295, the shots a circuit takes, before any backend or CUNQA
/// allocation exists. What is and is not an integer, and which error each
/// raises, is [`positive_index`]'s.
fn validate_final_shots(final_shots: Option<&Bound<'_, PyAny>>) -> PyResult<Option<u32>> {
    let Some(value) = final_shots.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let index = positive_index(value, "final_shots")?;
    match index.extract::<u32>() {
        Ok(shots) => Ok(Some(shots)),
        Err(_) => Err(too_large(value, "final_shots", u32::MAX)),
    }
}

/// `value` as a Python `int` `>= 1`, for the setting `name`: a Python `int` or
/// anything with `__index__` (a NumPy integer) is an integer, and one below 1
/// raises `ValueError` (a `polypus.ConfigurationError`). A `bool`, a `float`
/// (even `4.0`), a `str`, `None` or anything else is not an integer and raises
/// `TypeError`, as an integer argument such as `shots` does. Only the
/// `TypeError` of `operator.index` means "not an integer"; any other
/// exception, raised inside a custom `__index__` or a `KeyboardInterrupt`,
/// propagates unchanged. The caller converts the result to its own type and
/// reports a value too large for it with [`too_large`].
fn positive_index<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let message = || format!("{name} must be an integer >= 1, got {}", shown_repr(value));
    let not_an_integer = || pyo3::exceptions::PyTypeError::new_err(message());
    if value.is_instance_of::<PyBool>() {
        return Err(not_an_integer());
    }
    let py = value.py();
    let index = match PyModule::import(py, "operator")?.call_method1("index", (value,)) {
        Ok(index) => index,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
            return Err(not_an_integer());
        }
        Err(err) => return Err(err),
    };
    if index.lt(1)? {
        return Err(crate::exceptions::configuration_error(message()));
    }
    Ok(index)
}

/// The `ValueError` for a setting `name` whose integer `value` is above `max`.
fn too_large(value: &Bound<'_, PyAny>, name: &str, max: impl std::fmt::Display) -> PyErr {
    crate::exceptions::configuration_error(format!(
        "{name} must be at most {max}, got {}",
        shown_repr(value)
    ))
}

/// `value`'s `repr` for an error message, or a fixed text when it fails.
fn shown_repr(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .map_or_else(|_| "an unprintable value".to_string(), |r| r.to_string())
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
/// Both entry points funnel every DE/PSO/QNG/CMAES/Adam/SPSA/Rotosolve/External branch
/// through here, so this is also where the run's completion is logged — once,
/// on the success path only (an oracle failure was already logged at `error!`
/// where it was recorded).
/// `start` is the [`Instant`] captured on entry to the entry point, so the
/// reported duration covers the whole call.
///
/// [`OracleError::Config`] is a rejected optimizer configuration
/// (`population_size` too small for DE, empty PSO/QNG `bounds`, …), caught before
/// any oracle call — unlike an oracle failure it has nowhere else to be logged,
/// so it is logged here too — except an oracle that returned the wrong number
/// of values or a non-finite gradient, a failure of the run's evaluation that
/// is raised as `polypus.EvaluationError`; [`OracleError::Evaluation`]
/// re-raises the oracle's recorded failure with its original class preserved.
fn finish_optimization(
    py: Python<'_>,
    result: Result<TrainOutcome, OracleError>,
    seed: u64,
    id: String,
    start: Instant,
) -> PyResult<Py<PyAny>> {
    let (outcome, final_evaluation) = match result {
        Ok(outcome) => (outcome.optimization, outcome.final_evaluation),
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
        Err(OracleError::Config(
            oracle_err @ (polypus_optimizers::OptimizerError::OracleLengthMismatch { .. }
            | polypus_optimizers::OptimizerError::NonFiniteGradient { .. }),
        )) => {
            log::error!("run {id}: evaluation failed: {oracle_err}");
            return Err(crate::exceptions::EvaluationError::new_err(
                oracle_err.to_string(),
            ));
        }
        Err(OracleError::Config(config_err)) => {
            log::error!("run {id}: optimizer rejected the configuration: {config_err}");
            return Err(crate::exceptions::configuration_error(
                config_err.to_string(),
            ));
        }
    };
    let final_fitness = final_evaluation
        .as_ref()
        .map_or(String::new(), |evaluation| {
            format!(", final_fitness={}", evaluation.fitness)
        });
    log::info!(
        "training run {id} completed: iterations_run={}, converged={}{final_fitness}, duration={:?}",
        outcome.iterations_run,
        outcome.converged,
        start.elapsed()
    );
    outcome_to_train_result(py, outcome, final_evaluation, seed, id)
}

/// Parse a `polypus.DE` / `PSO` / `QNG` / `CMAES` / `Adam` / `SPSA` / `Rotosolve` /
/// `External` object into a pyo3-free [`Method`].
///
/// The QNG `variance_function` is adapted into a [`PyVarianceOracle`] here, and
/// the `External` optimizer into a [`PyAskTell`](external::PyAskTell), which
/// reads each row up to one value past `dimensions` — the only Python touches —
/// so [`dispatch_optimizer`] itself stays Python-free. A non-method object is a
/// `TypeError`, exactly as the previous inline dispatch (shared now by both
/// `train` and `qml_train`).
fn method_from_pyclass(
    method: &Bound<'_, PyAny>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    run_id: &str,
) -> PyResult<Method> {
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        return Ok(Method::De(DeConfig {
            generations: de.generations,
            population_size: de.population_size,
            tolerance: de.tolerance,
            patience: de.patience,
            strategy: de.strategy,
            mutation_factor: de.mutation_factor,
            crossover_rate: de.crossover_rate,
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
    if let Ok(cmaes) = method.extract::<PyRef<CMAES>>() {
        return Ok(Method::Cmaes(CmaesConfig {
            generations: cmaes.generations,
            population_size: cmaes.population_size,
            sigma0: cmaes.sigma0,
            tolerance: cmaes.tolerance,
            patience: cmaes.patience,
            restarts: cmaes.restarts,
        }));
    }
    if let Ok(adam) = method.extract::<PyRef<Adam>>() {
        return Ok(Method::Adam(AdamConfig {
            max_iters: adam.max_iters,
            learning_rate: adam.learning_rate,
            beta1: adam.beta1,
            beta2: adam.beta2,
            epsilon: adam.epsilon,
            bounds: adam.bounds,
            tolerance: adam.tolerance,
            patience: adam.patience,
            finite_difference_step: adam.finite_difference_step,
            rule: adam.rule,
        }));
    }
    if let Ok(spsa) = method.extract::<PyRef<Spsa>>() {
        return Ok(Method::Spsa(SpsaConfig {
            max_iters: spsa.max_iters,
            resamplings: spsa.resamplings,
            perturbation: spsa.perturbation,
            target_step: spsa.target_step,
            alpha: spsa.alpha,
            gamma: spsa.gamma,
            stability_constant: spsa.stability_constant,
            eval_interval: spsa.eval_interval,
        }));
    }
    if let Ok(rotosolve) = method.extract::<PyRef<Rotosolve>>() {
        return Ok(Method::Rotosolve(RotosolveConfig {
            sweeps: rotosolve.sweeps,
            reevaluate_every: rotosolve.reevaluate_every,
            tolerance: rotosolve.tolerance,
            patience: rotosolve.patience,
        }));
    }
    if let Ok(external) = method.extract::<PyRef<External>>() {
        let (max_evaluations, optimizer) =
            external.for_run(method.py(), dimensions, errors, run_id);
        return Ok(Method::External(
            ExternalConfig { max_evaluations },
            Box::new(optimizer),
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, polypus.QNG, polypus.CMAES, \
         polypus.Adam, polypus.SPSA, polypus.Rotosolve, or polypus.External",
    ))
}

/// The `options` kwarg of the entry points, converted at the FFI edge: a `dict` of
/// `str` keys, each mapped to a `str` or a `list`/`tuple` of `str`
/// ([`OptionValue`]). Anything else — an `int`, `bool`, `float`, `None`, a `dict`, a
/// list holding a non-`str` — is a `TypeError` naming the key (and the position, for
/// a list element), so a mistyped value never reaches a backend factory.
///
/// A local newtype because `FromPyObject` and `OptionValue` are both foreign to this
/// crate. An empty list is accepted here; whether it is meaningful is the backend's
/// call (the subprocess bridge rejects an empty `command`).
pub(crate) struct OptionsArg(HashMap<String, OptionValue>);

impl OptionsArg {
    /// The converted options, as the backend config carries them.
    fn into_inner(self) -> HashMap<String, OptionValue> {
        self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for OptionsArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        use pyo3::exceptions::PyTypeError;
        use pyo3::types::{PyString, PyTuple};

        // For error messages only, so an unreadable name is not itself an error.
        let type_name = |item: &Bound<'_, PyAny>| {
            item.get_type()
                .name()
                .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string())
        };
        let dict = ob.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "options must be a dict of str keys to str or list[str] values, got {}",
                type_name(&ob)
            ))
        })?;
        let mut options = HashMap::with_capacity(dict.len());
        // Walk a snapshot of the items, and read each list/tuple by index rather than
        // through `__iter__`: a subclass's `__iter__` is user code that could mutate
        // the dict mid-walk, and PyO3's dict iterator panics when that happens.
        for item in dict.items().iter() {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            let key = match key.cast::<PyString>() {
                Ok(key) => key.to_str()?.to_owned(),
                Err(_) => {
                    return Err(PyTypeError::new_err(format!(
                        "options keys must be str, got {}",
                        type_name(&key)
                    )))
                }
            };
            let value = if let Ok(text) = value.cast::<PyString>() {
                OptionValue::Str(text.to_str()?.to_owned())
            } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
                let elements: Vec<Bound<'py, PyAny>> = match value.cast::<PyList>() {
                    Ok(list) => list.iter().collect(),
                    Err(_) => value.cast::<PyTuple>()?.iter().collect(),
                };
                let mut items = Vec::with_capacity(elements.len());
                for (idx, item) in elements.into_iter().enumerate() {
                    match item.cast::<PyString>() {
                        Ok(text) => items.push(text.to_str()?.to_owned()),
                        Err(_) => {
                            return Err(PyTypeError::new_err(format!(
                                "options['{key}'][{idx}] must be a str, got {}",
                                type_name(&item)
                            )))
                        }
                    }
                }
                OptionValue::List(items)
            } else {
                return Err(PyTypeError::new_err(format!(
                    "options['{key}'] must be a str or a list of str, got {}",
                    type_name(&value)
                )));
            };
            options.insert(key, value);
        }
        Ok(OptionsArg(options))
    }
}

/// Map the public `infrastructure` + `backend` strings and provider parameters
/// into a typed [`BackendConfig`]. Centralising this keeps the string→variant
/// mapping in one place and guarantees the config matches the selected backend.
///
/// `backend` selects the *device* within an infrastructure. For `local`:
/// `"aer"` (default) runs Qiskit Aer; `"polypus"` runs the pure-Rust native
/// statevector simulator. CUNQA's QPUs always run Aer, with the simulator fixed
/// when they are raised (`qraise`), so `cunqa` accepts only `"aer"` (default) or
/// `"AerSimulator"` and rejects a `noise_model`: both would otherwise be dropped
/// silently. For the same reason the native backend, `qmio` and
/// registry-dispatched backends reject a `noise_model` and a non-default
/// `sim_method` ([`reject_aer_only_args`] for the last two), and registry
/// backends a non-default `backend`.
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
    walltime: Option<&str>,
    fusion: Option<bool>,
    options: Option<HashMap<String, OptionValue>>,
) -> PyResult<BackendConfig> {
    // Only the native statevector backend fuses gates. An explicit `Some(true)`
    // anywhere else is an unmeetable request (see the doc above) — reject it
    // before building anything, so it never looks like it took effect. `None`
    // and `Some(false)` pass through: both are honourable everywhere.
    let is_native = infrastructure == "local"
        && matches!(backend, "polypus" | "statevector" | "polypus_statevector");
    if fusion == Some(true) && !is_native {
        return Err(crate::exceptions::configuration_error(format!(
            "fusion=True applies only to the native backend=\"polypus\"; backend \"{backend}\" \
             on infrastructure \"{infrastructure}\" cannot fuse gates. Omit fusion, or pass \
             fusion=False for a gate-by-gate run."
        )));
    }
    // `options` is consumed only by registry-dispatched backends (qmio, subprocess,
    // any third-party). The typed built-ins would silently ignore it, so a non-empty
    // `options` there is rejected rather than left looking like it took effect — the
    // same spirit as the `fusion=True` rejection above.
    if matches!(infrastructure, "local" | "cunqa") {
        if let Some(opts) = &options {
            if !opts.is_empty() {
                return Err(crate::exceptions::configuration_error(format!(
                    "the '{infrastructure}' infrastructure does not accept 'options' ({} given); \
                     the 'options' dict applies only to registry-backed backends (\"qmio\", \
                     \"subprocess\", or a registered third-party backend)",
                    opts.len()
                )));
            }
        }
    }
    // CUNQA's QPUs run Aer with a simulator fixed at `qraise`, and its run takes
    // no noise model, so `BackendConfig::Cunqa` has nowhere to carry either: a
    // non-Aer `backend` or a `noise_model` would be dropped while the manifest
    // echoed the caller's `backend`. Reject both here, before any QPU is raised.
    if infrastructure == "cunqa" {
        if !matches!(backend, "aer" | "AerSimulator") {
            return Err(crate::exceptions::configuration_error(format!(
                "unknown cunqa backend '{backend}'; CUNQA's QPUs run Aer, expected \"aer\""
            )));
        }
        if noise_model.is_some() {
            return Err(crate::exceptions::configuration_error(
                "the 'cunqa' infrastructure does not accept a noise_model: its simulated QPUs \
                 are configured when they are raised, not per run; omit it, or use \
                 infrastructure=\"local\" with backend=\"aer\"",
            ));
        }
    }
    match infrastructure {
        "local" => match backend {
            "aer" | "AerSimulator" => Ok(BackendConfig::Local {
                sim_method: sim_method.to_string(),
                noise_model,
            }),
            "polypus" | "statevector" | "polypus_statevector" => {
                if noise_model.is_some() {
                    return Err(crate::exceptions::configuration_error(
                        "the native 'polypus' backend is a noiseless statevector simulator \
                         and does not accept a noise_model; use backend=\"aer\"",
                    ));
                }
                // `LocalNative` has no simulation method to choose, so a
                // non-default `sim_method` would be dropped silently (#334).
                if sim_method != "automatic" {
                    return Err(crate::exceptions::configuration_error(format!(
                        "the native 'polypus' backend is a statevector simulator and does \
                         not accept sim_method=\"{sim_method}\"; omit it, or use \
                         backend=\"aer\""
                    )));
                }
                Ok(BackendConfig::LocalNative {
                    fusion: fusion.unwrap_or(true),
                })
            }
            other => Err(crate::exceptions::configuration_error(format!(
                "unknown local backend '{other}'; expected \"aer\" or \"polypus\""
            ))),
        },
        "cunqa" => Ok(BackendConfig::Cunqa {
            sim_method: sim_method.to_string(),
            nodes,
            cores_per_qpu,
            walltime: walltime.unwrap_or(DEFAULT_WALLTIME).to_string(),
        }),
        // QMIO is now dispatched through the runtime registry (it registers a
        // factory), so its config is a `Registered { name: "qmio", options }`.
        // `backend` picks its program format, so only the Aer arguments are
        // checked.
        "qmio" => {
            let config = build_qmio_registered(backend, options)?;
            reject_aer_only_args(infrastructure, sim_method, noise_model.as_ref())?;
            Ok(config)
        }
        // Any other name is dispatched through the runtime backend registry: the
        // built-in `"subprocess"` bridge, or a backend a Rust embedder registered
        // before this call. The caller-supplied `options` reach the factory via the
        // `BackendBuildContext`. Registration of the built-ins is ensured here.
        other => {
            crate::infrastructure::register_builtin_backends();
            if crate::infrastructure::is_registered(other) {
                reject_aer_only_args(other, sim_method, noise_model.as_ref())?;
                // `Registered` carries only `options`, so `backend` would be
                // dropped while `RunResult.backend` echoed it.
                if backend != "aer" {
                    return Err(crate::exceptions::configuration_error(format!(
                        "the registered backend '{other}' does not accept backend=\"{backend}\"; \
                         omit it and pass backend-specific settings through 'options'"
                    )));
                }
                Ok(BackendConfig::Registered {
                    name: other.to_string(),
                    options: options.unwrap_or_default(),
                })
            } else {
                Err(crate::exceptions::configuration_error(format!(
                    "unknown infrastructure '{other}'; expected \"local\", \"cunqa\", \"qmio\", \
                     or a registered backend name (see docs/backends.md)"
                )))
            }
        }
    }
}

/// Reject the Aer-only arguments on an infrastructure that cannot honour them:
/// QMIO (real hardware) and every registry-dispatched backend, whose
/// `BackendConfig::Registered` has no field for either, so they would be
/// dropped silently. `sim_method` is compared with the entry points' default,
/// `"automatic"`, so a call that leaves it alone passes.
fn reject_aer_only_args(
    infrastructure: &str,
    sim_method: &str,
    noise_model: Option<&Py<PyAny>>,
) -> PyResult<()> {
    if noise_model.is_some() {
        return Err(crate::exceptions::configuration_error(format!(
            "the '{infrastructure}' infrastructure does not accept a noise_model; it applies \
             only to Aer (infrastructure=\"local\" with backend=\"aer\")"
        )));
    }
    if sim_method != "automatic" {
        return Err(crate::exceptions::configuration_error(format!(
            "the '{infrastructure}' infrastructure does not accept sim_method=\"{sim_method}\"; \
             it applies only to Aer (infrastructure=\"local\" or \"cunqa\")"
        )));
    }
    Ok(())
}

/// Build the QMIO backend's registry config for the CESGA QMIO QPU.
///
/// The endpoint is read from the `ZMQ_SERVER` environment variable (the same
/// variable the reference `qmio` Python client uses); there is no default, so with
/// neither `ZMQ_SERVER` nor an `endpoint` option the backend fails to build with
/// `polypus.QmioError`. The public `backend` argument selects the program
/// representation submitted to the QPU. Caller-supplied `options` override the
/// computed defaults (e.g. `optimization`, `repetition_period`, `res_format`, a
/// custom `endpoint`). Only available with `--features qmio`; otherwise it returns
/// an actionable error instead of silently degrading.
#[cfg(feature = "qmio")]
fn build_qmio_registered(
    backend: &str,
    options: Option<HashMap<String, OptionValue>>,
) -> PyResult<BackendConfig> {
    qmio_registered_config(backend, std::env::var("ZMQ_SERVER").ok(), options)
}

/// [`build_qmio_registered`] with the `ZMQ_SERVER` value passed in, so tests do not
/// depend on the process environment.
#[cfg(feature = "qmio")]
fn qmio_registered_config(
    backend: &str,
    zmq_server: Option<String>,
    options: Option<HashMap<String, OptionValue>>,
) -> PyResult<BackendConfig> {
    let program_format = match backend {
        // `"aer"` is the entry-point default, so treat it (and the explicit
        // aliases) as OpenQASM for the QMIO path.
        "aer" | "qmio" | "openqasm" | "qasm" => "openqasm",
        "qir" | "qir_text" => "qir_text",
        "qir_bitcode" | "qir_compiled" => "qir_bitcode",
        other => {
            return Err(crate::exceptions::configuration_error(format!(
                "unknown qmio program format '{other}'; expected \"openqasm\", \"qir\", or \"qir_bitcode\""
            )))
        }
    };
    let mut opts: HashMap<String, OptionValue> = HashMap::new();
    if let Some(endpoint) = zmq_server {
        opts.insert("endpoint".to_string(), OptionValue::from(endpoint));
    }
    opts.insert(
        "program_format".to_string(),
        OptionValue::from(program_format),
    );
    opts.insert("optimization".to_string(), OptionValue::from("0"));
    opts.insert("res_format".to_string(), OptionValue::from("binary_count"));
    if let Some(user) = options {
        opts.extend(user);
    }
    Ok(BackendConfig::Registered {
        name: "qmio".to_string(),
        options: opts,
    })
}

/// Without the `qmio` feature, selecting the QMIO infrastructure fails with a
/// clear, actionable message instead of pulling a ZeroMQ stack into every build.
#[cfg(not(feature = "qmio"))]
fn build_qmio_registered(
    _backend: &str,
    _options: Option<HashMap<String, OptionValue>>,
) -> PyResult<BackendConfig> {
    Err(crate::exceptions::configuration_error(QMIO_NOT_COMPILED))
}

/// Why `infrastructure="qmio"` fails in a build without the `qmio` feature;
/// `backend_compatibility` reports the same text.
#[cfg(not(feature = "qmio"))]
const QMIO_NOT_COMPILED: &str =
    "the 'qmio' infrastructure requires compiling polypus with --features qmio";

/// Why `run_quantum_circuit(..., infrastructure="qmio")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
const QMIO_REJECTS_QISKIT: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
     2.0 string";

/// Why `qml.train` and `qml.predict` reject `infrastructure="qmio"`.
const QMIO_REJECTS_QML: &str =
    "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize the Qiskit circuits of a QML model";

/// Why `run_quantum_circuit(..., backend="polypus")` rejects a Qiskit
/// `QuantumCircuit`; `backend_compatibility` reports the same text.
const NATIVE_REJECTS_QISKIT: &str =
    "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
     pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"";

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
        return Err(crate::exceptions::configuration_error(format!(
            "shots must be >= 1, got {shots}"
        )));
    }
    if n_qpus < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "n_qpus must be >= 1, got {n_qpus}"
        )));
    }
    Ok(())
}

/// Validate the CUNQA allocation parameters at the Python-facing boundary. Only
/// the `"cunqa"` infrastructure consumes `nodes`/`cores_per_qpu`/`walltime` (they
/// flow to SLURM's `qraise` as the C-1 kwargs `n_nodes`/`cores_per_qpu`/`t` via
/// `CunqaBackend`); the `local` and `qmio` match arms of `build_backend_config`
/// ignore them, so validating unconditionally would reject harmless non-cunqa
/// calls. A zero for either count has no sane meaning once it reaches SLURM, and
/// a `walltime` that is not `HH:MM:SS` would only fail inside `qraise`, so
/// reject both here before any seam call rather than forward them. Shared by
/// every entry point rather than duplicated per function.
fn validate_cunqa_allocation(
    infrastructure: &str,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<&str>,
) -> PyResult<()> {
    if infrastructure != "cunqa" {
        return Ok(());
    }
    if nodes < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "nodes must be >= 1 for the 'cunqa' infrastructure, got {nodes}"
        )));
    }
    if cores_per_qpu < 1 {
        return Err(crate::exceptions::configuration_error(format!(
            "cores_per_qpu must be >= 1 for the 'cunqa' infrastructure, got {cores_per_qpu}"
        )));
    }
    if let Some(walltime) = walltime {
        validate_walltime(walltime)
            .map_err(|e| crate::exceptions::configuration_error(e.to_string()))?;
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
        return Err(crate::exceptions::configuration_error(
            "id must not be empty",
        ));
    }
    if let Some(c) = id
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
    {
        return Err(crate::exceptions::configuration_error(format!(
            "id contains invalid character {c:?}; only ASCII letters, digits, \
			 '.', '_' and '-' are allowed (got {id:?})"
        )));
    }
    if id.len() > MAX_ID_LEN {
        return Err(crate::exceptions::configuration_error(format!(
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

/// Start of the `ValueError` message for a circuit with free parameters, the
/// same for a `polypus.Circuit` and a Qiskit `QuantumCircuit`.
const UNBOUND_PARAMETERS: &str = "circuit has unbound parameters and cannot be executed directly";

/// Interpret the `qc` argument of `run_quantum_circuit` as a fully bound,
/// executable circuit:
/// - `polypus.Circuit` → native circuit (must have no free parameters),
/// - `str` → raw OpenQASM 2.0 program,
/// - anything else → Qiskit `QuantumCircuit` (must have no free parameters:
///   no backend binds them, and Aer would fail with its own error, issue #308).
fn extract_bound_circuit(qc: &Bound<'_, PyAny>) -> PyResult<BoundCircuit> {
    if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
        let concrete = native.native().assign_parameters(&[]).map_err(|e| {
            crate::exceptions::configuration_error(format!(
                "{UNBOUND_PARAMETERS}: {e}. \
                 Bind values first via to_qasm2(params) or use polypus.train"
            ))
        })?;
        return Ok(BoundCircuit::Native(concrete));
    }
    if let Ok(qasm) = qc.extract::<String>() {
        return Ok(BoundCircuit::Qasm2(qasm));
    }
    let free = free_parameter_names(qc)?;
    if !free.is_empty() {
        return Err(crate::exceptions::configuration_error(format!(
            "{UNBOUND_PARAMETERS}: {}. \
             Bind values first via QuantumCircuit.assign_parameters or use polypus.train",
            free.join(", ")
        )));
    }
    // Anything else is a Qiskit `QuantumCircuit`: carry it opaquely through the
    // pyo3-free `BoundCircuit` enum via its `Foreign` escape hatch, so the Aer/CUNQA
    // backends still receive the native Qiskit object unchanged.
    Ok(crate::infrastructure::QiskitCircuit::into_bound(
        qc.clone().unbind(),
    ))
}

/// The names of the free parameters of a Qiskit `QuantumCircuit` (its
/// `parameters`), in Qiskit's order. An object without that attribute has none.
fn free_parameter_names(qc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let Ok(parameters) = qc.getattr(pyo3::intern!(qc.py(), "parameters")) else {
        return Ok(Vec::new());
    };
    parameters
        .try_iter()?
        .map(|parameter| Ok(parameter?.str()?.to_string()))
        .collect()
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
    use pyo3::exceptions::PyTypeError;
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
                return Err(crate::exceptions::configuration_error(format!(
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
/// (default), `"qir"`, or `"qir_bitcode"`); for `"cunqa"`, whose QPUs run Aer,
/// it must be `"aer"` (default) or `"AerSimulator"`. Any other value, or a
/// `noise_model` with `"cunqa"`, raises `ValueError` before any QPU is requested.
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
/// `walltime` is the SLURM walltime of the CUNQA QPUs (sent to `qraise` as `t`),
/// again meaningful only for `infrastructure="cunqa"`. It must be `"HH:MM:SS"`:
/// hours of one or more digits (`"48:00:00"` is valid), minutes and seconds of
/// two digits below 60, total above zero; SLURM's `D-HH:MM:SS` is rejected. A
/// malformed value raises `ValueError` before any QPU is requested. Omitted, it
/// is `"10:00:00"`.
///
/// Returns a [`RunResult`] carrying the counts — `counts`, always a
/// `list[dict]` with one dict per QPU replica, and `merged_counts`, their total as
/// one `dict` — plus a manifest (`id`, effective `seed`, `backend`,
/// `infrastructure`) for logging and replay.
///
/// `fusion` applies only to `backend="polypus"`, the one backend that fuses
/// gates. Omit it (the default) and each backend does its own thing — the
/// native one fuses, the rest never did. `fusion=False` forces a strictly
/// gate-by-gate run and is accepted everywhere (a non-fusing backend already
/// meets it). `fusion=True` on any backend that cannot fuse is rejected with a
/// `ValueError` rather than silently ignored, so it never looks like it took
/// effect.
///
/// `options` configures a registry-backed backend (`"qmio"`, `"subprocess"`, or a
/// registered third-party name): a `dict` whose values are each a `str` or a
/// `list`/`tuple` of `str` (e.g. `{"command": ["python3", "/my dir/worker.py"]}`).
/// Any other value type is a `TypeError` naming the key; a non-empty `options` on
/// `local`/`cunqa`, which would ignore it, is a `ValueError`.
///
/// With `infrastructure="qmio"` the QPU endpoint (`ZMQ_SERVER`, or the
/// `endpoint` option) must be trusted: the ZeroMQ link is neither encrypted nor
/// authenticated, and the 64 MiB reply cap is checked only after a reply has
/// been received, so a hostile endpoint can make the process allocate as much
/// memory as it announces. Off CESGA's private network, use a tunnel (see "QMIO
/// transport threat model" in `docs/backends.md`).
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1, nodes=1, cores_per_qpu=2, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None, options=None, walltime=None))]
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
    options: Option<OptionsArg>,
    walltime: Option<String>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    // Entry-point trace carrying the full circuit `Debug` repr on every call:
    // large and high-volume, so it stays at `debug` rather than the default log.
    log::debug!(
        "run_quantum_circuit called with qc: {qc:?}, shots: {shots}, \
         infrastructure: {infrastructure}, n_qpus: {n_qpus}, backend: {backend}, seed: {seed:?}"
    );
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
    let bound_qc = extract_bound_circuit(&qc)?;
    if is_native_backend(backend) && bound_qc.is_foreign() {
        return Err(crate::exceptions::configuration_error(
            NATIVE_REJECTS_QISKIT,
        ));
    }
    // The QMIO path serialises circuits to QASM/QIR in Rust (GIL-free) and cannot
    // read a Qiskit QuantumCircuit, whose gates are only accessible via Python.
    if infrastructure == "qmio" && bound_qc.is_foreign() {
        return Err(crate::exceptions::configuration_error(QMIO_REJECTS_QISKIT));
    }
    // Resolve the shot-sampling seed. Every simulated backend (native, Aer,
    // CUNQA's simulated QPUs) is seeded by Polypus; `qmio` is real hardware, so
    // an explicit seed there would be silently ineffective — reject it rather
    // than give false confidence in reproducibility. When simulated, `None`
    // means "draw a fresh OS-entropy seed", resolved here so the effective
    // value can be reported in the manifest.
    let effective_seed: Option<u64> = if infrastructure == "qmio" {
        if seed.is_some() {
            return Err(crate::exceptions::configuration_error(
                "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
            ));
        }
        None
    } else {
        Some(seed_or_entropy(seed, random_seed)?)
    };

    // Append a UUID v4 so two concurrent calls with identical `n_qpus`/
    // `infrastructure` never collide: the id names SLURM families/allocations,
    // temp files and log streams (see ExecutionConfig::id), so a byte-identical
    // id across runs is a real hazard, not a cosmetic one. The `run_{n}_{infra}`
    // prefix is kept for human-readable debuggability.
    let id = unique_id(&format!("run_{}_{}", n_qpus, infrastructure))?;
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
        walltime.as_deref(),
        fusion,
        options.map(OptionsArg::into_inner),
    )?;
    // Only the native statevector backend consults the gate-parallel threshold,
    // so calibrate it lazily (#351) and surface the one-time default-visible
    // warning only when this run actually resolves to it — never for
    // Aer/CUNQA/QMIO. Gated on the resolved `BackendConfig`, not the raw
    // `backend` string, so it tracks the real dispatch rather than re-deriving
    // it from the kwargs.
    if matches!(backend_config, BackendConfig::LocalNative { .. }) {
        calibration::prepare_native_threshold(qc.py())?;
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
    // shots across replicas, conserving the total per C-3, and returns one counts
    // map per replica); otherwise the backend's default atomic-wave planner runs
    // the circuit as-is, as its single replica.
    let counts_result = qc
        .py()
        .detach(move || -> Result<Vec<Counts>, InfrastructureError> {
            let backend =
                Infrastructure::create_backend(&config).map_err(InfrastructureError::Backend)?;
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
    // Convert at the FFI boundary with one shape for every `n_qpus` (C-3):
    // `counts` is a `list[dict]` with one map per replica, and `merged_counts` is
    // their key-by-key sum as a single `dict`. Keys are sorted in both.
    let merged = merge_counts(&counts_vec);
    Python::attach(|py| {
        let counts = counts_to_pylist(py, &counts_vec)?.into_any().unbind();
        let merged_counts = counts_to_pydict(py, &merged)?.into_any().unbind();
        Py::new(
            py,
            RunResult {
                counts,
                merged_counts,
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
/// `method` must be an instance of `DE`, `PSO`, `QNG`, `CMAES`, `Adam`, `SPSA`,
/// `Rotosolve`, or `External`. Rotosolve trains only a circuit whose every
/// parameter is the angle of exactly one single-Pauli rotation, as `±θ + c`:
/// the circuit is checked before anything runs, and a circuit that does not
/// qualify raises `ValueError` naming the first gate or parameter that does
/// not. `External` runs an external ask/tell optimizer (Optuna, Nevergrad,
/// pycma or your own object), one batch per `ask()`; an exception its `ask()`
/// or `tell()` raises propagates as itself.
///
/// `seed` makes the optimizer reproducible (contract C-7): precedence is the
/// explicit `seed` kwarg, then the optimizer object's `seed` field, then a fresh
/// OS-entropy value. `External` has no `seed` field and its library owns its
/// seed, so there the seed only fixes the shot sampling. On the native backend
/// the same seed also drives shot sampling, so a native-backend run reproduces
/// exactly. Returns a
/// [`TrainResult`] exposing `best_params`, `best_fitness`, `iterations_run`,
/// `converged`, `fitness_history`, `final_params`, `final_fitness`, the
/// effective `seed` and the effective `id`.
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
/// `initial_point` (default `None`) starts the search from a given point, such
/// as the `best_params` of an earlier run: a list, tuple or NumPy array of
/// `dimensions` finite numbers, otherwise `ValueError` before anything runs.
/// DE (every `strategy`) evaluates it, wrapped into `[0, 2π)`, as the first
/// member of its initial population; PSO as its first particle, which must lie
/// within the PSO `bounds`; QNG and Adam start from it as θ₀; CMA-ES centres
/// its first run on it (restarts start elsewhere), with its `sigma0`
/// unchanged; SPSA starts from it as θ₀ and centres its calibration batch on
/// it; Rotosolve starts from it as θ₀, the first point it measures. Without it
/// every optimizer starts as before (contract C-5). `External` refuses it with
/// `ValueError`, whatever the point: the external library chooses its start
/// (give it to the library, or to the `initial_point` of a `polypus.ask_tell`
/// adapter).
///
/// `final_shots` (default `None`) re-evaluates the best candidates once the
/// optimizer has stopped, in one more batch with that many shots per circuit:
/// `best_params` and, for DE, PSO, CMA-ES and `External`, the four best other
/// distinct candidates they scored. `TrainResult.final_params` is the one with the
/// highest re-evaluated value (`best_params` on a tie) and `final_fitness`
/// that value; both are `None` without it, and `best_params`, `best_fitness`
/// and `fitness_history` are the same with and without it, seeded runs
/// included (contract C-5). It must be an integer `>= 1`, checked before
/// anything runs: a smaller one raises `ValueError`, and a value that is not
/// an integer (a `bool` or a `float` too) `TypeError`, as `shots` does. The batch is sent
/// whatever stopped the optimizer, but not after an error or an interrupt; it
/// is not part of the CUNQA walltime projection, and if it fails, `train`
/// raises as for any batch.
///
/// Interruption: pressing Ctrl+C (or otherwise sending `SIGINT`) stops the
/// optimization promptly and raises `KeyboardInterrupt` in Python, instead of
/// waiting for the run to finish. An exception raised by
/// `expectation_function` propagates the same way, as itself.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are consumed only
/// by `infrastructure="cunqa"`; `local`/`qmio` accept but ignore them. For
/// `cunqa` both must be `>= 1` (a zero is meaningless to SLURM and rejected).
/// `walltime` (default `"10:00:00"`) is the SLURM walltime of the CUNQA QPUs,
/// also `cunqa` only: `"HH:MM:SS"` (hours may exceed 24, e.g. `"48:00:00"`;
/// `D-HH:MM:SS` is rejected), otherwise `ValueError` before any QPU is requested.
/// With `infrastructure="cunqa"`, whose QPUs run Aer, `backend` must be `"aer"`
/// (default) or `"AerSimulator"` and `noise_model` is not accepted; either one
/// raises `ValueError` before any QPU is requested.
///
/// `options` configures a registry-backed backend (`"qmio"`, `"subprocess"`, or a
/// registered third-party name): a `dict` whose values are each a `str` or a
/// `list`/`tuple` of `str` (e.g. `{"command": ["python3", "/my dir/worker.py"]}`).
/// Any other value type is a `TypeError` naming the key; a non-empty `options` on
/// `local`/`cunqa`, which would ignore it, is a `ValueError`.
///
/// With `infrastructure="qmio"` the QPU endpoint (`ZMQ_SERVER`, or the
/// `endpoint` option) must be trusted: the ZeroMQ link is neither encrypted nor
/// authenticated, and the 64 MiB reply cap is checked only after a reply has
/// been received, so a hostile endpoint can make the process allocate as much
/// memory as it announces. Off CESGA's private network, use a tunnel (see "QMIO
/// transport threat model" in `docs/backends.md`).
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
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None, fusion=None, options=None, walltime=None, initial_point=None, final_shots=None))]
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
    options: Option<OptionsArg>,
    walltime: Option<String>,
    initial_point: Option<Vec<f64>>,
    final_shots: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
    validate_id(&id)?;
    // Resolve the seed that drives the optimizer's RNG (contract C-7): explicit
    // kwarg > optimizer object's `seed` field > fresh OS entropy. Unlike
    // `run_quantum_circuit`, a seed is always meaningful here — it seeds the
    // optimizer regardless of backend — and it is also threaded into
    // `ExecutionConfig::seed` so the native backend's shot sampling becomes
    // deterministic too, making a native-backend training run fully reproducible.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method), random_seed)?;
    // Native circuits know their parameter count — catch a mismatch with the
    // requested optimisation dimensions before any QPU work starts.
    let circuit_source = extract_circuit_source(&qc);
    if let Some(num_params) = circuit_source.num_params() {
        if num_params != dimensions as usize {
            return Err(crate::exceptions::configuration_error(format!(
                "dimensions ({dimensions}) does not match the circuit's free parameters ({num_params})"
            )));
        }
    }
    validate_method(&method, dimensions, initial_point.as_deref())?;
    let final_shots = validate_final_shots(final_shots.as_ref())?;
    // The native statevector backend runs pure-Rust circuits only; a Qiskit
    // template can't be simulated without the interpreter.
    if is_native_backend(backend) {
        if let CircuitSource::Qiskit(_) = &circuit_source {
            return Err(crate::exceptions::configuration_error(
                "the native 'polypus' backend requires a native polypus.Circuit; \
                 got a Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"",
            ));
        }
    }
    // QMIO serialises GIL-free, so it likewise needs a native circuit, not a
    // Qiskit template that can only be read through the interpreter.
    if infrastructure == "qmio" {
        if let CircuitSource::Qiskit(_) = &circuit_source {
            return Err(crate::exceptions::configuration_error(
                "the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
                 serialization); got a Qiskit circuit. Build it with polypus.Circuit",
            ));
        }
    }
    // Rotosolve's step is exact only on single-frequency circuits: checked
    // before any backend, and so any CUNQA allocation, exists.
    if is_rotosolve(&method) {
        check_rotosolve_circuit(&qc, Some(&circuit_source), None)?;
    }
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
        walltime.as_deref(),
        fusion,
        options.map(OptionsArg::into_inner),
    )?;
    // As in `run_quantum_circuit`: the native backend's simulator resolves the
    // gate-parallel threshold, so calibrate it lazily before building it (#351).
    if matches!(backend_config, BackendConfig::LocalNative { .. }) {
        calibration::prepare_native_threshold(qc.py())?;
    }
    // Suffix the caller-supplied `id` with a UUID v4 so two concurrent training
    // runs sharing the same `id` never collide on the SLURM family/allocation,
    // temp files or log streams named by ExecutionConfig::id (#75, mirroring
    // run_quantum_circuit). The prefix is kept for debuggability and the
    // effective id is reported back in the TrainResult.
    let effective_id = unique_id(&id)?;
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // The run's base seed: the native backend seeds each circuit from it,
        // Aer, CUNQA and the subprocess bridge each call (C-7); QMIO ignores
        // it. Set unconditionally so a seeded training run reproduces exactly.
        seed: Some(effective_seed),
    });
    // Shared error slot: the oracle records the first evaluation failure here
    // (the optimizer traits cannot return a `Result`) and it is surfaced by
    // `finish_optimization` after `optimize` returns.
    let errors = OracleErrorSlot::new();
    // A callable stays a Python-callback observable (optimized fallback); a
    // polypus.Qubo/Ising opts into the native, GIL-free evaluation path.
    // Converted, like the method, before the backend exists: on CUNQA creating
    // it requests a SLURM allocation (issue #418).
    let observable = extract_cost_observable(&expectation_function)?;
    let method_enum = method_from_pyclass(&method, dimensions, &errors, &effective_id)?;
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
        initial_point,
        final_shots,
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
        .detach(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    finish_optimization(method.py(), result, effective_seed, effective_id, start)
}

/// Compose `feature_map` with `ansatz` into the QML circuit template, adding a
/// terminal `measure_all` when there are no classical bits (Aer needs them to
/// return counts).
///
/// A Qiskit exception from these calls (an ansatz wider than the feature map,
/// a parameter-name clash) is raised as `polypus.EvaluationError`, chaining the
/// original (issue #218); nothing has run on a backend yet.
fn compose_qml_template<'py>(
    feature_map: &Bound<'py, PyAny>,
    ansatz: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let qiskit = crate::exceptions::qiskit_error_to_evaluation_error;
    let composed = feature_map
        .call_method1("compose", (ansatz,))
        .map_err(qiskit)?;
    let num_clbits: usize = composed.getattr("num_clbits")?.extract()?;
    if num_clbits == 0 {
        composed.call_method0("measure_all").map_err(qiskit)?;
    }
    Ok(composed)
}

/// Bind each row of `rows` by name to the feature-map parameters of `template`,
/// one circuit per row. The binding is partial: the ansatz parameters stay free.
///
/// A row whose length differs from `len(feature_map.parameters)` is a
/// `ValueError` naming `rows_name` and the 0-based row (contract C-8); zipping
/// would otherwise drop extra features or leave some unbound. A Qiskit exception
/// from binding a row (a value Qiskit rejects, such as a complex number) is
/// raised as `polypus.EvaluationError`, chaining the original (issue #218).
fn bind_feature_rows<'py>(
    template: &Bound<'py, PyAny>,
    feature_map: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    rows_name: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    let py = template.py();
    let fm_params = feature_map.getattr("parameters")?;
    let builtins = PyModule::import(py, "builtins")?;
    let fm_params_list = builtins.call_method1("list", (&fm_params,))?;
    let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
    let fm_len = fm_params_list.len()?;
    let mut circuits: Vec<Py<PyAny>> = Vec::new();
    for (row_idx, row_result) in rows.try_iter()?.enumerate() {
        let row = row_result?;
        let row_len = row.len()?;
        if row_len != fm_len {
            return Err(crate::exceptions::configuration_error(format!(
                "{rows_name} row {row_idx} has {row_len} features, but feature_map expects \
                 {fm_len} (len(feature_map.parameters))"
            )));
        }
        let param_dict = PyDict::new(py);
        for (param, val) in fm_params_list.try_iter()?.zip(row.try_iter()?) {
            param_dict.set_item(param?, val?)?;
        }
        circuits.push(
            template
                .call_method("assign_parameters", (&param_dict,), Some(&kwargs_assign))
                .map_err(crate::exceptions::qiskit_error_to_evaluation_error)?
                .unbind(),
        );
    }
    Ok(circuits)
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
/// Qiskit/Aer path (the native backend and `infrastructure="qmio"` are
/// rejected before anything runs), and Aer's shot sampling
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
/// `initial_point` (keyword-only, default `None`) starts the search from a
/// given point of the ansatz weights, as in [`train`]: `dimensions` finite
/// numbers, otherwise `ValueError` before anything runs. `polypus.External`
/// refuses it, as in [`train`].
///
/// With `method=polypus.Rotosolve(...)`, every ansatz parameter must be the
/// angle of exactly one single-Pauli rotation, as `±θ + c`; the feature map's
/// parameters are bound to each row, so they may appear anywhere. A
/// `polypus.SampleCost`, which is not an average of a cost per bitstring, is
/// rejected too. Both raise `ValueError` before anything runs.
///
/// `final_shots` (keyword-only, default `None`) re-evaluates the best
/// candidates with that many shots once the optimizer has stopped, as in
/// [`train`]: every training circuit of each candidate runs with them, and
/// `TrainResult.final_params` and `final_fitness` hold the result.
///
/// Interruption: same behaviour as [`train`] — Ctrl+C stops the optimization
/// promptly and raises `KeyboardInterrupt` rather than waiting for the run to
/// finish, and an exception raised by `expectation_function` propagates as
/// itself.
///
/// `nodes` and `cores_per_qpu` size the SLURM allocation and are consumed only
/// by `infrastructure="cunqa"`; `local`/`qmio` accept but ignore them. For
/// `cunqa` both must be `>= 1` (a zero is meaningless to SLURM and rejected).
/// `walltime` (default `"10:00:00"`) is the SLURM walltime of the CUNQA QPUs,
/// also `cunqa` only: `"HH:MM:SS"` (hours may exceed 24, e.g. `"48:00:00"`;
/// `D-HH:MM:SS` is rejected), otherwise `ValueError` before any QPU is requested.
/// With `infrastructure="cunqa"`, whose QPUs run Aer, `backend` must be `"aer"`
/// (default) or `"AerSimulator"` and `noise_model` is not accepted; either one
/// raises `ValueError` before any QPU is requested.
///
/// `options` configures a registry-backed backend (`"qmio"`, `"subprocess"`, or a
/// registered third-party name): a `dict` whose values are each a `str` or a
/// `list`/`tuple` of `str` (e.g. `{"command": ["python3", "/my dir/worker.py"]}`).
/// Any other value type is a `TypeError` naming the key; a non-empty `options` on
/// `local`/`cunqa`, which would ignore it, is a `ValueError`.
///
/// With `infrastructure="qmio"` the QPU endpoint (`ZMQ_SERVER`, or the
/// `endpoint` option) must be trusted: the ZeroMQ link is neither encrypted nor
/// authenticated, and the 64 MiB reply cap is checked only after a reply has
/// been received, so a hostile endpoint can make the process allocate as much
/// memory as it announces. Off CESGA's private network, use a tunnel (see "QMIO
/// transport threat model" in `docs/backends.md`).
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
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer", seed=None, options=None, walltime=None, *, y_train=None, initial_point=None, final_shots=None))]
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
    options: Option<OptionsArg>,
    walltime: Option<String>,
    y_train: Option<Bound<'py, PyAny>>,
    initial_point: Option<Vec<f64>>,
    final_shots: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
    validate_id(&id)?;
    // Native models have their own entry point (plan D4-R).
    qml::reject_native_pieces("train", &feature_map, &ansatz)?;
    // Same seed precedence as `train` (contract C-7): kwarg > optimizer field >
    // OS entropy. qml.train always runs on a Qiskit/Aer path (native rejected
    // below); this seed governs the optimizer's RNG and, since it's threaded
    // into ExecutionConfig::seed below, Aer's shot sampling too.
    let effective_seed = resolve_optimizer_seed(seed, method_seed(&method), random_seed)?;
    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path; the native statevector backend cannot consume a Qiskit
    // `QuantumCircuit`. Accept `backend` for API symmetry but reject native.
    if is_native_backend(backend) {
        return Err(crate::exceptions::configuration_error(
            "the native 'polypus' backend is not supported for qml.train (feature maps \
             and ansätze are Qiskit circuits); use backend=\"aer\"",
        ));
    }
    if infrastructure == "qmio" {
        return Err(crate::exceptions::configuration_error(QMIO_REJECTS_QML));
    }
    // The optimizer searches a `dimensions`-wide vector and binds it to the
    // ansatz's free parameters; a mismatch surfaces later as a cryptic Qiskit
    // binding error inside the oracle. Catch it upfront, mirroring how `train`
    // validates `dimensions` against the circuit's parameter count (contract
    // C-8). `len()` calls `__len__`, working on Qiskit's ParameterView the same
    // as on a list.
    let num_ansatz_params = ansatz.getattr("parameters")?.len()?;
    if num_ansatz_params != dimensions as usize {
        return Err(crate::exceptions::configuration_error(format!(
            "dimensions ({dimensions}) does not match the ansatz's free parameters ({num_ansatz_params})"
        )));
    }
    validate_method(&method, dimensions, initial_point.as_deref())?;
    let final_shots = validate_final_shots(final_shots.as_ref())?;
    // Type-checked before any circuit is composed; the count is checked once
    // `x_train` has been read (contract C-8).
    let labels = y_train.as_ref().map(extract_labels).transpose()?;
    let py = feature_map.py();

    // Rotosolve needs a fitness that averages a cost per bitstring, which a
    // `SampleCost` (a function of each sample's whole distribution) is not.
    let rotosolve = is_rotosolve(&method);
    if rotosolve
        && expectation_function
            .extract::<PyRef<'_, SampleCost>>()
            .is_ok()
    {
        return Err(crate::exceptions::configuration_error(
            "polypus.Rotosolve cannot train with a polypus.SampleCost: Rotosolve needs a \
             fitness that averages a cost per bitstring over the shots, so that it is a \
             sinusoid of each parameter, and a SampleCost scores each sample's whole \
             distribution. Use a callable (bitstring, label) -> float or a \
             polypus.CachedCost, or another optimizer",
        ));
    }

    let composed = compose_qml_template(&feature_map, &ansatz)?;
    // Only the ansatz is trained: the feature map's parameters are bound to
    // each row below, so they are constants wherever they appear.
    if rotosolve {
        check_rotosolve_circuit(&composed, None, Some(&ansatz.getattr("parameters")?))?;
    }
    // Pre-bind each sample, leaving the ansatz parameters for the optimizer.
    let qcs = bind_feature_rows(&composed, &feature_map, &x_train, "x_train")?;

    if qcs.is_empty() {
        return Err(crate::exceptions::configuration_error(
            "x_train must contain at least one training sample",
        ));
    }

    // One label per row (contract C-8), and an objective that can read labels:
    // both checked before any backend, and so any CUNQA allocation, exists.
    let supervised = match labels {
        Some(labels) => {
            if labels.len() != qcs.len() {
                return Err(crate::exceptions::configuration_error(format!(
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
        walltime.as_deref(),
        None,
        options.map(OptionsArg::into_inner),
    )?;
    // Suffix the caller-supplied `id` with a UUID v4 (see `train` and #75) so
    // concurrent qml.train runs sharing the same `id` never collide on the
    // SLURM family/allocation, temp files or log streams named by
    // ExecutionConfig::id. The effective id is reported back in the TrainResult.
    let effective_id = unique_id(&id)?;
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        // The run's base seed: the native backend seeds each circuit from it,
        // Aer, CUNQA and the subprocess bridge each call (C-7); QMIO ignores
        // it. Set unconditionally so a seeded training run reproduces exactly.
        seed: Some(effective_seed),
    });
    // Shared error slot (see `train`): the oracle records the first evaluation
    // failure here and `finish_optimization` surfaces it after `optimize`.
    let errors = OracleErrorSlot::new();
    // Converted before the backend exists, as in `train` (issue #418).
    let objective = match supervised {
        Some((objective, labels)) => QmlObjective::Supervised { objective, labels },
        None => QmlObjective::Unsupervised(extract_cost_observable(&expectation_function)?),
    };
    let method_enum = method_from_pyclass(&method, dimensions, &errors, &effective_id)?;
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
    let resources = Resources::new(backend, None, Arc::new(config.run_params()))
        .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    let scheduler = Scheduler::ephemeral(resources);
    let flow = TrainFlow {
        factory: QmlOracleFactory {
            source: QmlSource::Qiskit {
                training_circuits: qcs,
                objective,
            },
            minibatch: None,
        },
        method: method_enum,
        dimensions,
        seed: effective_seed,
        initial_point,
        final_shots,
        errors: errors.clone(),
    };
    // Release the GIL for the optimization (see `train` and docs/ENGINEERING.md
    // §3): the Qiskit binding + Aer calls re-acquire it, and the run token's
    // `check_signals`-backed interrupt guard — which the pyo3-free planner calls
    // between waves — keeps Ctrl+C prompt.
    let token = interruptible_token();
    let result = py.detach(|| scheduler.run_cancellable(flow, &token));
    scheduler.close();
    finish_optimization(py, result, effective_seed, effective_id, start)
}

/// Batched QML inference: run a trained `qml.train` model on every row of `x` in
/// one scheduled run, returning a [`RunResult`] whose `counts` holds one dict per
/// row, in row order, and whose `merged_counts` is `None` (the rows are distinct
/// circuits, so summing them has no meaning).
///
/// The circuits are built as in training: each row bound to the feature map, then
/// `params` (typically `TrainResult.best_params`) bound positionally to the
/// ansatz, as `QmlOracle` binds a candidate. With `n_qpus > 1` the rows are spread
/// over the QPUs; one row's shots are never split. `seed` behaves as in
/// `run_quantum_circuit` (contract C-7). The native backend and `qmio` are
/// rejected, since the model is a Qiskit circuit. Row width, the number and
/// finiteness of `params`, and a non-empty `x` are checked before anything runs
/// (contract C-8).
///
/// `nodes`, `cores_per_qpu` and `walltime` size the SLURM allocation and are
/// consumed only by `infrastructure="cunqa"`, with the same rules as in
/// [`run_quantum_circuit`]; `local` ignores them.
/// With `infrastructure="cunqa"`, whose QPUs run Aer, `backend` must be `"aer"`
/// (default) or `"AerSimulator"` and `noise_model` is not accepted; either one
/// raises `ValueError` before any QPU is requested.
///
/// Example:
///
/// ```ignore
///     run = polypus.qml.predict(
///         feature_map, ansatz, X_test, result.best_params,
///         shots=1024, infrastructure="local", seed=7,
///     )
///     predictions = [readout(counts) for counts in run.counts]
/// ```
#[pyfunction(name = "predict", signature = (feature_map, ansatz, x, params, shots, infrastructure, n_qpus=1, nodes=1, cores_per_qpu=2, sim_method="automatic", noise_model=None, backend="aer", seed=None, walltime=None))]
pub fn qml_predict<'py>(
    feature_map: Bound<'py, PyAny>,
    ansatz: Bound<'py, PyAny>,
    x: Bound<'py, PyAny>,
    params: Vec<f64>,
    shots: u32,
    infrastructure: String,
    n_qpus: u32,
    nodes: u32,
    cores_per_qpu: u32,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    walltime: Option<String>,
) -> PyResult<Py<PyAny>> {
    let start = Instant::now();
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
    qml::reject_native_pieces("predict", &feature_map, &ansatz)?;
    // The model is a Qiskit circuit: neither the native backend nor QMIO runs it.
    if is_native_backend(backend) {
        return Err(crate::exceptions::configuration_error(
            "the native 'polypus' backend is not supported for qml.predict (feature maps \
             and ansätze are Qiskit circuits); use backend=\"aer\"",
        ));
    }
    if infrastructure == "qmio" {
        return Err(crate::exceptions::configuration_error(QMIO_REJECTS_QML));
    }
    // The weights must fill the ansatz exactly (contract C-8).
    let num_ansatz_params = ansatz.getattr("parameters")?.len()?;
    if params.len() != num_ansatz_params {
        return Err(crate::exceptions::configuration_error(format!(
            "params has {} values, but the ansatz has {num_ansatz_params} free parameters",
            params.len()
        )));
    }
    if let Some((index, value)) = params.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(crate::exceptions::configuration_error(format!(
            "params[{index}] is {value}; every weight must be a finite number"
        )));
    }
    let py = feature_map.py();

    // Built as in training: rows by name, then the weights positionally.
    let composed = compose_qml_template(&feature_map, &ansatz)?;
    let rows = bind_feature_rows(&composed, &feature_map, &x, "x")?;
    if rows.is_empty() {
        return Err(crate::exceptions::configuration_error(
            "x must contain at least one sample",
        ));
    }
    let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
    let circuits: Vec<BoundCircuit> = rows
        .iter()
        .map(|row| {
            let circuit = row
                .bind(py)
                .call_method("assign_parameters", (params.clone(),), Some(&kwargs_assign))
                // Qiskit rejecting the weights (issue #218): `polypus.EvaluationError`.
                .map_err(crate::exceptions::qiskit_error_to_evaluation_error)?
                .unbind();
            Ok(crate::infrastructure::QiskitCircuit::into_bound(circuit))
        })
        .collect::<PyResult<_>>()?;

    // Always simulated (qmio is rejected), so a seed always applies (contract C-7).
    let effective_seed = seed_or_entropy(seed, random_seed)?;
    let id = unique_id(&format!("predict_{}_{}", n_qpus, infrastructure))?;
    let backend_config = build_backend_config(
        &infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
        walltime.as_deref(),
        None,
        // predict runs on aer/local only (qmio rejected above), so there is no
        // registered/provider backend to pass options to.
        None,
    )?;
    let config = ExecutionConfig {
        id: id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.clone(),
        backend_config,
        opt_level: OptLevel::default(),
        seed: Some(effective_seed),
    };
    let n_rows = circuits.len();
    log::info!(
        "qml.predict {id} starting: rows={n_rows}, infrastructure={infrastructure}, \
         backend={backend}, n_qpus={n_qpus}, shots={shots}, seed={effective_seed}"
    );

    // One run, GIL released. The default planner (never the shot-distributing one)
    // sends the rows in waves of the backend's concurrency; Ctrl+C is honoured
    // between waves.
    let counts_result = py.detach(move || -> Result<Vec<Counts>, InfrastructureError> {
        let backend =
            Infrastructure::create_backend(&config).map_err(InfrastructureError::Backend)?;
        let resources = Resources::new(backend, None, Arc::new(config.run_params()))?;
        let scheduler = Scheduler::ephemeral(resources);
        let out =
            scheduler.run_cancellable(RunCircuitFlow { circuits, shots }, &interruptible_token());
        scheduler.close();
        out
    });
    let counts_vec = counts_result.map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
    log::info!("qml.predict {id} completed: duration={:?}", start.elapsed());
    let counts = counts_to_pylist(py, &counts_vec)?.into_any().unbind();
    Py::new(
        py,
        RunResult {
            counts,
            // Each entry is a different row's circuit, not a replica of one, so
            // there is no meaningful merged total (C-3).
            merged_counts: py.None(),
            id,
            seed: Some(effective_seed),
            backend: backend.to_string(),
            infrastructure,
        },
    )
    .map(|result| result.into_any())
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
    // Baked in at compile time, so it names the binary actually loaded (C-6).
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    crate::exceptions::register(m)?;
    m.add_class::<DE>()?;
    m.add_class::<PSO>()?;
    m.add_class::<QNG>()?;
    m.add_class::<CMAES>()?;
    m.add_class::<Adam>()?;
    m.add_class::<Spsa>()?;
    m.add_class::<Rotosolve>()?;
    m.add_class::<External>()?;
    m.add_class::<Circuit>()?;
    m.add_class::<Param>()?;
    expr::register(m)?;
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
    m.add_function(wrap_pyfunction!(backend_compatibility, m)?)?;

    // qml submodule — exposes polypus.qml.train()
    let py = m.py();
    let qml = PyModule::new(py, "polypus.qml")?;
    qml.add_function(wrap_pyfunction!(qml_train, &qml)?)?;
    qml.add_function(wrap_pyfunction!(qml_predict, &qml)?)?;
    qml::register(&qml)?;
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

    /// Set in the child process started by [`run_in_fresh_process`].
    const FRESH_PROCESS_ENV: &str = "POLYPUS_FRESH_PROCESS_CHILD";

    /// Run the test `name` alone in a fresh copy of this test binary, one that
    /// never initializes the interpreter (other tests here call
    /// `Python::initialize()`, which cannot be undone). Mirrors the helper in
    /// `polypus_infrastructure::attach`, which is test-only and so not visible
    /// from this crate.
    fn run_in_fresh_process(name: &str) {
        let exe = std::env::current_exe().expect("the test binary path is available");
        let output = std::process::Command::new(exe)
            .args(["--exact", name, "--test-threads=1", "--nocapture"])
            .env(FRESH_PROCESS_ENV, "1")
            .output()
            .expect("the test binary can be re-executed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child run failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child must actually run the test, not filter it out:\n{stdout}"
        );
    }

    /// Without an interpreter, both signal-polling hooks that run while
    /// detached cancel instead of panicking: the planner's guard aborts the run
    /// with `InterpreterUnavailable` boxed in `BackendError::External`, the
    /// error every other site reached while detached returns (the edge's
    /// `KeyboardInterrupt`), and the `statevector` hook asks the gate loop to
    /// stop, with no pending exception. "Not initialized" is also how a
    /// shutting-down interpreter looks to PyO3 (CPython clears its
    /// `initialized` flag right after marking itself finalizing), but a real
    /// shutdown cannot be reproduced from a Rust test.
    #[test]
    fn signal_polling_cancels_when_the_interpreter_is_unavailable() {
        if std::env::var_os(FRESH_PROCESS_ENV).is_none() {
            run_in_fresh_process(
                "bindings::tests::signal_polling_cancels_when_the_interpreter_is_unavailable",
            );
            return;
        }
        use crate::infrastructure::{BackendError, InterpreterUnavailable, Interrupt};
        match SignalInterrupt.poll() {
            Err(BackendError::External(boxed)) => assert!(
                boxed.is::<InterpreterUnavailable>(),
                "an unreachable interpreter must abort the run with InterpreterUnavailable: {boxed}"
            ),
            other => panic!("an unreachable interpreter must abort the run, got {other:?}"),
        }

        let mut pending = None;
        assert!(
            circuit::poll_for_cancellation(&mut pending),
            "an unreachable interpreter must cancel the simulation"
        );
        assert!(
            pending.is_none(),
            "no exception can be recorded without an interpreter"
        );
    }

    /// `statevector`'s detached simulation really goes through
    /// `poll_for_cancellation`: with no interpreter the first checkpoint
    /// cancels the run, with no pending exception, instead of panicking.
    ///
    /// Timing-independent in the same way as `polypus-sim`'s
    /// `tests/cancellation.rs`: the circuit ends in an `Rx(NaN)`, so
    /// `Cancelled` proves the loop stopped early and `NonFiniteAmplitude`
    /// would prove it never hit a checkpoint. The size (14 qubits, 60 000
    /// barrier-separated rotations) outlives the 25ms checkpoint interval by a
    /// wide margin in both profiles; the run is never completed.
    #[test]
    fn statevector_simulation_cancels_when_the_interpreter_is_unavailable() {
        if std::env::var_os(FRESH_PROCESS_ENV).is_none() {
            run_in_fresh_process(
                "bindings::tests::statevector_simulation_cancels_when_the_interpreter_is_unavailable",
            );
            return;
        }
        use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
        const N: usize = 14;
        let mut gates = Vec::new();
        for i in 0..60_000 {
            gates.push(G::Rx {
                qubit: i % N,
                theta: Fixed(0.1),
            });
            gates.push(G::Barrier(Vec::new()));
        }
        gates.push(G::Rx {
            qubit: 0,
            theta: Fixed(f64::NAN),
        });
        let circuit = ConcreteCircuit::new_unchecked(N, gates);

        let mut pending = None;
        let outcome = circuit::simulate_cancellable(&circuit, true, &mut pending);
        assert!(
            matches!(outcome, Err(polypus_sim::SimError::Cancelled)),
            "an unreachable interpreter must cancel the simulation mid-run, got {:?}",
            outcome.map(|_| "a completed run")
        );
        assert!(
            pending.is_none(),
            "no exception can be recorded without an interpreter"
        );
    }

    /// A cancelled `statevector` re-raises the pending signal's own exception,
    /// and, when the interpreter became unreachable, the same
    /// `KeyboardInterrupt` every other site reached while detached ends in:
    /// same class, same text.
    #[test]
    fn statevector_cancellation_raises_keyboard_interrupt() {
        Python::initialize();
        Python::attach(|py| {
            let unreachable = circuit::cancellation_error(None);
            assert!(unreachable.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py));
            assert_eq!(
                unreachable.value(py).to_string(),
                format!(
                    "the run was aborted: {}",
                    crate::infrastructure::InterpreterUnavailable
                )
            );

            let signal = pyo3::exceptions::PyKeyboardInterrupt::new_err("Ctrl+C");
            let signal_value = signal.value(py).clone();
            let raised = circuit::cancellation_error(Some(signal));
            assert!(
                raised.value(py).is(&signal_value),
                "the pending exception must be re-raised verbatim"
            );
        });
    }
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
            .try_to_qasm2()
            .unwrap()
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
            None,
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
                None,
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
                    None,
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
            let id1 = unique_id(base).unwrap();
            let id2 = unique_id(base).unwrap();
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
                None,
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
        assert_eq!(
            resolve_optimizer_seed(Some(7), Some(9), random_seed).unwrap(),
            7
        );
        // ...the optimizer field is the fallback...
        assert_eq!(
            resolve_optimizer_seed(None, Some(9), random_seed).unwrap(),
            9
        );
        // ...and with neither set, a fresh entropy seed is drawn each time.
        assert_ne!(
            resolve_optimizer_seed(None, None, random_seed).unwrap(),
            resolve_optimizer_seed(None, None, random_seed).unwrap()
        );
    }

    /// An entropy source that cannot be read, as `random_seed` reports it.
    fn failing_draw() -> Result<u64, EntropyError> {
        Err(EntropyError::new(std::io::Error::other(
            "getrandom: device not available",
        )))
    }

    /// A draw that must never happen: an explicit seed must not touch OS entropy.
    fn forbidden_draw() -> Result<u64, EntropyError> {
        panic!("an explicit seed must not draw from the OS entropy source")
    }

    /// Assert `err` is the `polypus.BackendError` a failed entropy draw raises
    /// (contract C-1/C-7): catchable as `Exception`, never a `PanicException`.
    fn assert_entropy_failure(err: &PyErr) {
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(
                err.is_instance_of::<crate::exceptions::BackendError>(py),
                "wrong exception class: {err}"
            );
            assert!(
                !err.is_instance_of::<pyo3::panic::PanicException>(py),
                "an entropy failure must not cross as a PanicException: {err}"
            );
            assert!(
                err.to_string().contains("OS entropy source unavailable"),
                "message lost: {err}"
            );
        });
    }

    #[test]
    fn seed_or_entropy_uses_an_explicit_seed_without_drawing() {
        assert_eq!(seed_or_entropy(Some(11), forbidden_draw).unwrap(), 11);
    }

    #[test]
    fn seed_or_entropy_returns_the_drawn_value() {
        assert_eq!(seed_or_entropy(None, || Ok(31)).unwrap(), 31);
    }

    #[test]
    fn seed_or_entropy_raises_backend_error_when_the_draw_fails() {
        let err = seed_or_entropy(None, failing_draw).expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
    }

    #[test]
    fn resolve_optimizer_seed_propagates_a_failed_draw() {
        // Either explicit seed short-circuits the draw...
        assert_eq!(
            resolve_optimizer_seed(Some(7), None, forbidden_draw).unwrap(),
            7
        );
        assert_eq!(
            resolve_optimizer_seed(None, Some(9), forbidden_draw).unwrap(),
            9
        );
        // ...and with neither, the entropy failure surfaces as an error, not a panic.
        let err = resolve_optimizer_seed(None, None, failing_draw)
            .expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
    }

    /// The run id's UUID is built from exactly the bytes drawn, with the v4
    /// version and RFC 4122 variant bits set — the id format is unchanged.
    #[test]
    fn unique_id_from_builds_a_v4_uuid_from_the_drawn_bytes() {
        let id = unique_id_from("run1", || Ok(std::array::from_fn(|i| i as u8))).unwrap();
        assert_eq!(id, "run1_00010203-0405-4607-8809-0a0b0c0d0e0f");
        let suffix = uuid::Uuid::parse_str(id.strip_prefix("run1_").unwrap()).unwrap();
        assert_eq!(suffix.get_version(), Some(uuid::Version::Random));
        assert_eq!(suffix.get_variant(), uuid::Variant::RFC4122);
    }

    /// Every entry point draws a run id, even with an explicit seed or on
    /// `qmio`: a failed draw must raise `polypus.BackendError` (catchable as
    /// `Exception`), never a `PanicException` (#264, contract C-1/C-7).
    #[test]
    fn unique_id_from_raises_backend_error_when_the_draw_fails() {
        let err = unique_id_from("run1", || {
            Err(EntropyError::for_run_id(std::io::Error::other(
                "getrandom: device not available",
            )))
        })
        .expect_err("a failed draw must be an Err");
        assert_entropy_failure(&err);
        Python::attach(|py| {
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyException>(py),
                "must be catchable as Exception: {err}"
            );
            assert!(
                err.to_string().contains("run id"),
                "message must say the run id draw failed: {err}"
            );
        });
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
            let seed = resolve_optimizer_seed(kwarg, field, random_seed).unwrap();
            AlgorithmDifferentialEvolution
                .optimize(AlgorithmDifferentialEvolutionArgs {
                    oracle: Box::new(Quadratic),
                    population_size: 20,
                    generations: 40,
                    dimensions: 3,
                    tolerance: 1e-9,
                    patience: 20,
                    seed: Some(seed),
                    initial_point: None,
                    mutation_factor: None,
                    crossover_rate: None,
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

    /// `polypus.DE(strategy=...)` rejects an unknown name with `ValueError`
    /// when it is passed or set, before any run, and `method_from_pyclass`
    /// hands the parsed strategy to dispatch.
    #[test]
    fn de_strategy_is_validated_and_reaches_the_method() {
        use polypus_optimizers::DeStrategy;
        Python::initialize();
        Python::attach(|py| {
            let err = DE::new(100, 50, 0.01, 20, None, "LSHADE", None, None)
                .err()
                .expect("an unknown strategy is rejected");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("\"lshade\""), "{err}");

            let de = Py::new(
                py,
                DE::new(100, 50, 0.01, 20, None, "shade", None, None).unwrap(),
            )
            .unwrap();
            let bound = de.bind(py);
            assert!(bound.setattr("strategy", "jade").is_err());
            assert_eq!(
                bound
                    .getattr("strategy")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "shade",
                "a rejected name leaves the strategy unchanged"
            );
            bound.setattr("strategy", "lshade").unwrap();
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::De(cfg)) => assert_eq!(cfg.strategy, DeStrategy::LShade),
                _ => panic!("a polypus.DE parses into Method::De"),
            }
        });
    }

    /// `polypus.Adam(rule=...)` rejects an unknown name with `ValueError`
    /// when it is passed or set, before any run, naming the valid ones.
    #[test]
    fn adam_rule_is_validated_when_passed_or_set() {
        Python::initialize();
        Python::attach(|py| {
            let new = |rule| {
                Adam::new(
                    100,
                    0.05,
                    0.9,
                    0.999,
                    1e-8,
                    (-1.0, 1.0),
                    0.01,
                    20,
                    0.1,
                    rule,
                    None,
                )
            };
            let err = new("Momentum").err().expect("an unknown rule is rejected");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("\"adam\" or \"momentum\""),
                "{err}"
            );

            let adam = Py::new(py, new("adam").unwrap()).unwrap();
            let bound = adam.bind(py);
            let err = bound
                .setattr("rule", "nesterov")
                .expect_err("an unknown rule is rejected when set");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("\"adam\" or \"momentum\""),
                "{err}"
            );
            assert_eq!(
                bound.getattr("rule").unwrap().extract::<String>().unwrap(),
                "adam",
                "a rejected name leaves the rule unchanged"
            );
            bound.setattr("rule", "momentum").unwrap();
            assert_eq!(
                bound.getattr("rule").unwrap().extract::<String>().unwrap(),
                "momentum"
            );
        });
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
        let aer = build_backend_config("local", "aer", "automatic", None, 1, 2, None, None, None)
            .expect("aer is a valid local backend");
        assert!(matches!(
            aer,
            BackendConfig::Local {
                ref sim_method,
                noise_model: None,
            } if sim_method == "automatic"
        ));

        for name in ["polypus", "statevector", "polypus_statevector"] {
            let native = build_backend_config(
                "local",
                name,
                "automatic",
                None,
                1,
                2,
                None,
                Some(true),
                None,
            )
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
        let with_fusion = build_backend_config(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            Some(true),
            None,
        )
        .expect("polypus is a valid local backend");
        assert!(matches!(
            with_fusion,
            BackendConfig::LocalNative { fusion: true }
        ));

        let without_fusion = build_backend_config(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            Some(false),
            None,
        )
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
        let native = build_backend_config(
            "local",
            "polypus",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
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
        pyo3::Python::initialize();
        for (infra, backend) in [("local", "aer"), ("cunqa", "aer")] {
            let err = build_backend_config(
                infra,
                backend,
                "automatic",
                None,
                1,
                2,
                None,
                Some(true),
                None,
            )
            .expect_err("fusion=True on a non-fusing backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("fusion=True applies only to"),
                    "unexpected error: {err}"
                );
            });

            // fusion=False and an omitted fusion must build a config, not error.
            build_backend_config(
                infra,
                backend,
                "automatic",
                None,
                1,
                2,
                None,
                Some(false),
                None,
            )
            .expect("fusion=False is accepted everywhere");
            build_backend_config(infra, backend, "automatic", None, 1, 2, None, None, None)
                .expect("omitted fusion is accepted everywhere");
        }
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_local_backend() {
        let err = build_backend_config(
            "local",
            "does-not-exist",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("an unknown local backend must be rejected");
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown local backend"));
        });
    }

    /// CUNQA's QPUs run Aer, so any other `backend` (the native aliases
    /// included) is rejected instead of being dropped while the manifest echoes it.
    #[test]
    fn build_backend_config_rejects_an_unknown_cunqa_backend() {
        pyo3::Python::initialize();
        for backend in ["polypus", "statevector", "polypus_statevector", "foo"] {
            let err =
                build_backend_config("cunqa", backend, "automatic", None, 1, 2, None, None, None)
                    .expect_err("a non-Aer cunqa backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("unknown cunqa backend"),
                    "unexpected error for {backend}: {err}"
                );
            });
        }
        for backend in ["aer", "AerSimulator"] {
            let config =
                build_backend_config("cunqa", backend, "automatic", None, 1, 2, None, None, None)
                    .expect("cunqa accepts the Aer backend names");
            assert!(matches!(config, BackendConfig::Cunqa { .. }));
        }
    }

    #[test]
    fn build_backend_config_rejects_a_noise_model_on_cunqa() {
        // `BackendConfig::Cunqa` has no noise model to carry, so one must be an
        // error rather than a silently noiseless run. As on the native backend,
        // any Python object will do — the check is `Option::is_some`.
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let err = build_backend_config(
            "cunqa",
            "aer",
            "automatic",
            Some(noise_model),
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("a noise model on cunqa must be rejected");
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("noise_model"),
                "the message must name the offending kwarg: {err}"
            );
        });
        assert!(matches!(
            build_backend_config("cunqa", "aer", "automatic", None, 1, 2, None, None, None),
            Ok(BackendConfig::Cunqa { .. })
        ));
    }

    #[test]
    fn build_backend_config_rejects_aer_only_args_on_a_registered_backend() {
        // `subprocess` is a built-in registry backend; `Registered` has no field
        // for these, so each must be an error naming the kwarg.
        pyo3::Python::initialize();
        let build = |backend: &str, sim_method: &str, noise: bool| {
            let noise_model = noise.then(|| Python::attach(|py| py.None()));
            build_backend_config(
                "subprocess",
                backend,
                sim_method,
                noise_model,
                1,
                2,
                None,
                None,
                None,
            )
        };
        assert!(build("aer", "automatic", false).is_ok());
        for (backend, sim_method, noise, kwarg) in [
            ("aer", "automatic", true, "noise_model"),
            ("aer", "statevector", false, "sim_method"),
            ("polypus", "automatic", false, "backend"),
        ] {
            let err = build(backend, sim_method, noise)
                .expect_err("an argument the backend cannot honour must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains(kwarg), "{kwarg}: {err}");
            });
        }
    }

    #[cfg(feature = "qmio")]
    #[test]
    fn build_backend_config_rejects_a_noise_model_and_sim_method_on_qmio() {
        // Real hardware: a noise model or an Aer method would be dropped.
        // `backend` stays the program format, so "qir" is accepted.
        pyo3::Python::initialize();
        let build = |sim_method: &str, noise: bool| {
            let noise_model = noise.then(|| Python::attach(|py| py.None()));
            build_backend_config(
                "qmio",
                "qir",
                sim_method,
                noise_model,
                1,
                2,
                None,
                None,
                None,
            )
        };
        assert!(build("automatic", false).is_ok());
        for (sim_method, noise, kwarg) in [
            ("automatic", true, "noise_model"),
            ("density_matrix", false, "sim_method"),
        ] {
            let err = build(sim_method, noise).expect_err("must be rejected on qmio");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains(kwarg), "{kwarg}: {err}");
            });
        }
    }

    #[test]
    fn build_backend_config_rejects_a_noise_model_on_the_native_backend() {
        // The native simulator is noiseless by construction, so a noise_model
        // must be an error rather than silently ignored. Any Python object will
        // do — the check is `Option::is_some`, not a Qiskit type check.
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let err = build_backend_config(
            "local",
            "polypus",
            "automatic",
            Some(noise_model),
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("a noise model on the native backend must be rejected");
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("noise_model"),
                "the message must name the offending kwarg: {err}"
            );
        });
    }

    #[test]
    fn build_backend_config_rejects_a_sim_method_on_the_native_backend() {
        // `LocalNative` has no field for `sim_method`, so anything but the
        // default must be an error rather than silently ignored (#334), on
        // every name of the native backend.
        pyo3::Python::initialize();
        for backend in ["polypus", "statevector", "polypus_statevector"] {
            let err = build_backend_config(
                "local",
                backend,
                "statevector",
                None,
                1,
                2,
                None,
                None,
                None,
            )
            .expect_err("a sim_method on the native backend must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("sim_method"),
                    "the message must name the offending kwarg: {err}"
                );
            });
        }
    }

    #[test]
    fn build_backend_config_keeps_a_noise_model_for_aer() {
        pyo3::Python::initialize();
        let noise_model = Python::attach(|py| py.None());
        let config = build_backend_config(
            "local",
            "aer",
            "density_matrix",
            Some(noise_model),
            1,
            2,
            None,
            None,
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
        let config =
            build_backend_config("cunqa", "aer", "statevector", None, 3, 4, None, None, None)
                .expect("cunqa is a valid infrastructure");
        assert!(matches!(
            config,
            BackendConfig::Cunqa {
                nodes: 3,
                cores_per_qpu: 4,
                ref sim_method,
                ref walltime,
                ..
            } if sim_method == "statevector" && walltime == DEFAULT_WALLTIME
        ));
    }

    #[test]
    fn build_backend_config_forwards_an_explicit_cunqa_walltime() {
        let config = build_backend_config(
            "cunqa",
            "aer",
            "statevector",
            None,
            1,
            2,
            Some("48:00:00"),
            None,
            None,
        )
        .expect("cunqa is a valid infrastructure");
        assert!(matches!(
            config,
            BackendConfig::Cunqa { ref walltime, .. } if walltime == "48:00:00"
        ));
    }

    #[test]
    fn validate_cunqa_allocation_checks_the_walltime_only_for_cunqa() {
        let msg = |r: PyResult<()>| r.expect_err("must be rejected").to_string();
        assert!(
            msg(validate_cunqa_allocation("cunqa", 1, 2, Some("1-00:00:00"))).contains("HH:MM:SS")
        );
        assert!(validate_cunqa_allocation("cunqa", 1, 2, Some("48:00:00")).is_ok());
        assert!(validate_cunqa_allocation("cunqa", 1, 2, None).is_ok());
        // local/qmio ignore the kwarg, as they do nodes/cores_per_qpu.
        for infra in ["local", "qmio"] {
            assert!(validate_cunqa_allocation(infra, 1, 2, Some("garbage")).is_ok());
        }
    }

    #[test]
    fn build_backend_config_rejects_an_unknown_infrastructure() {
        let err = build_backend_config(
            "quantum-cloud",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            None,
        )
        .expect_err("an unknown infrastructure must be rejected");
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(err.to_string().contains("unknown infrastructure"));
        });
    }

    /// A name registered at runtime resolves to `BackendConfig::Registered`, carrying
    /// the caller's `options` through to the factory — the success path of the
    /// register-by-name dispatch.
    #[test]
    fn build_backend_config_routes_a_registered_name_to_registered() {
        // Register a throwaway backend (build_backend_config only builds the config,
        // it never invokes the factory, so the body is irrelevant here).
        crate::infrastructure::register_backend(
            "edge-registered-test",
            |_ctx: &crate::infrastructure::BackendBuildContext| {
                Err(crate::infrastructure::BackendError::Conversion(
                    "unused".to_string(),
                ))
            },
        );
        let opts = HashMap::from([
            ("endpoint".to_string(), OptionValue::from("tcp://x:1")),
            (
                "command".to_string(),
                OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()]),
            ),
        ]);
        let config = build_backend_config(
            "edge-registered-test",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            Some(opts),
        )
        .expect("a registered name builds a Registered config");
        match config {
            BackendConfig::Registered { name, options } => {
                assert_eq!(name, "edge-registered-test");
                assert_eq!(
                    options.get("endpoint"),
                    Some(&OptionValue::from("tcp://x:1"))
                );
                // A list value reaches `Registered` intact (argument with a space
                // included).
                assert_eq!(
                    options.get("command"),
                    Some(&OptionValue::from(vec![
                        "python3".to_string(),
                        "/a b/w.py".to_string()
                    ]))
                );
            }
            other => panic!("expected BackendConfig::Registered, got {other:?}"),
        }
    }

    /// `options` on a typed built-in (`local`/`cunqa`) that would ignore it is
    /// rejected rather than silently dropped.
    #[test]
    fn build_backend_config_rejects_options_on_a_typed_builtin() {
        pyo3::Python::initialize();
        let str_opts = HashMap::from([("command".to_string(), OptionValue::from("x"))]);
        let list_opts = HashMap::from([(
            "command".to_string(),
            OptionValue::from(vec!["x".to_string()]),
        )]);
        for (infra, opts) in [
            ("local", &str_opts),
            ("cunqa", &str_opts),
            ("local", &list_opts),
            ("cunqa", &list_opts),
        ] {
            let err = build_backend_config(
                infra,
                "aer",
                "automatic",
                None,
                1,
                2,
                None,
                None,
                Some(opts.clone()),
            )
            .expect_err("options on a typed built-in must be rejected");
            Python::attach(|py| {
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(
                    err.to_string().contains("does not accept 'options'"),
                    "unexpected message: {err}"
                );
            });
        }
        // An empty options dict is fine (nothing to ignore).
        assert!(build_backend_config(
            "local",
            "aer",
            "automatic",
            None,
            1,
            2,
            None,
            None,
            Some(HashMap::new()),
        )
        .is_ok());
    }

    /// The user's options are merged over the qmio defaults: a user value replaces the
    /// default for its key, the other defaults stay, and a list value is carried
    /// through untouched (the factory rejects it later if the key is a string one).
    #[cfg(feature = "qmio")]
    #[test]
    fn build_qmio_registered_merges_user_options_over_the_defaults() {
        let user = HashMap::from([
            ("optimization".to_string(), OptionValue::from("2")),
            (
                "res_format".to_string(),
                OptionValue::from(vec!["raw".to_string()]),
            ),
            (
                "custom".to_string(),
                OptionValue::from(vec!["a b".to_string()]),
            ),
        ]);
        let config = qmio_registered_config(
            "qir",
            Some("tcp://qmio.example:5556".to_string()),
            Some(user),
        )
        .expect("qir is a qmio format");
        let BackendConfig::Registered { name, options } = config else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(name, "qmio");
        // Defaults the user did not override stay.
        assert_eq!(
            options.get("endpoint"),
            Some(&OptionValue::from("tcp://qmio.example:5556"))
        );
        assert_eq!(
            options.get("program_format"),
            Some(&OptionValue::from("qir_text"))
        );
        // User values win, whatever their shape.
        assert_eq!(options.get("optimization"), Some(&OptionValue::from("2")));
        assert_eq!(
            options.get("res_format"),
            Some(&OptionValue::from(vec!["raw".to_string()]))
        );
        assert_eq!(
            options.get("custom"),
            Some(&OptionValue::from(vec!["a b".to_string()]))
        );
        assert_eq!(options.len(), 5);
    }

    /// Without `ZMQ_SERVER` no endpoint is invented: the key is absent (the factory
    /// then rejects the config) unless the caller passes one.
    #[cfg(feature = "qmio")]
    #[test]
    fn qmio_registered_config_has_no_default_endpoint() {
        let BackendConfig::Registered { options, .. } =
            qmio_registered_config("qmio", None, None).expect("qmio is a qmio format")
        else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(options.get("endpoint"), None);

        let user = HashMap::from([(
            "endpoint".to_string(),
            OptionValue::from("tcp://qmio.example:5556"),
        )]);
        let BackendConfig::Registered { options, .. } =
            qmio_registered_config("qmio", None, Some(user)).expect("qmio is a qmio format")
        else {
            panic!("qmio must build a Registered config");
        };
        assert_eq!(
            options.get("endpoint"),
            Some(&OptionValue::from("tcp://qmio.example:5556"))
        );
    }

    /// Convert a Python expression through the `options` kwarg's extractor.
    fn extract_options(expr: &str) -> PyResult<HashMap<String, OptionValue>> {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let code = std::ffi::CString::new(expr).expect("no NUL in the test expression");
            py.eval(&code, None, None)?
                .extract::<OptionsArg>()
                .map(OptionsArg::into_inner)
        })
    }

    #[test]
    fn options_arg_accepts_str_list_and_tuple_values() {
        let options = extract_options(
            "{'endpoint': 'tcp://x:1', 'command': ['python3', '/a b/w.py'], \
              'argv': ('a', 'b c'), 'empty': []}",
        )
        .expect("str, list[str] and tuple[str] values convert");
        assert_eq!(options.len(), 4);
        assert_eq!(options["endpoint"], OptionValue::from("tcp://x:1"));
        assert_eq!(
            options["command"],
            OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()])
        );
        assert_eq!(
            options["argv"],
            OptionValue::from(vec!["a".to_string(), "b c".to_string()])
        );
        // An empty list is accepted at the edge; the backend decides on it.
        assert_eq!(options["empty"], OptionValue::from(Vec::new()));
    }

    /// A `list`/`tuple` subclass whose `__iter__` mutates the `options` dict must not
    /// reach Python code while the dict is walked: PyO3's dict iterator panics when
    /// the dict changes size under it, and a panic must never cross the FFI edge.
    /// The converter reads the dict through a snapshot of its items and the sequences
    /// by index, so `__iter__` never runs and the dict is left untouched.
    #[test]
    fn options_arg_does_not_run_user_iterators_while_walking_the_dict() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let code = std::ffi::CString::new(
                "opts = {}\n\
                 class L(list):\n\
                 \x20   def __iter__(self):\n\
                 \x20       opts['injected-by-list'] = 'x'\n\
                 \x20       return super().__iter__()\n\
                 class T(tuple):\n\
                 \x20   def __iter__(self):\n\
                 \x20       opts['injected-by-tuple'] = 'x'\n\
                 \x20       return super().__iter__()\n\
                 opts['a'] = L(['python3', '/a b/w.py'])\n\
                 opts['b'] = 'y'\n\
                 opts['c'] = T(('t',))\n",
            )
            .expect("no NUL in the test program");
            py.run(&code, Some(&globals), None)
                .expect("the test program runs");
            let opts = globals
                .get_item("opts")
                .expect("globals lookup")
                .expect("the test program defines opts");
            let options = opts
                .extract::<OptionsArg>()
                .map(OptionsArg::into_inner)
                .expect("list/tuple subclasses of str convert");
            assert_eq!(
                options["a"],
                OptionValue::from(vec!["python3".to_string(), "/a b/w.py".to_string()])
            );
            assert_eq!(options["b"], OptionValue::from("y"));
            assert_eq!(options["c"], OptionValue::from(vec!["t".to_string()]));
            assert_eq!(options.len(), 3);
            // The overridden `__iter__` never ran.
            assert_eq!(opts.len().expect("opts is a dict"), 3, "{opts}");
        });
    }

    #[test]
    fn options_arg_rejects_unsupported_values_with_a_type_error_naming_the_key() {
        for (expr, needle) in [
            (
                "{'k': 1}",
                "options['k'] must be a str or a list of str, got int",
            ),
            (
                "{'k': True}",
                "options['k'] must be a str or a list of str, got bool",
            ),
            (
                "{'k': 1.5}",
                "options['k'] must be a str or a list of str, got float",
            ),
            (
                "{'k': None}",
                "options['k'] must be a str or a list of str, got NoneType",
            ),
            (
                "{'k': {'a': 'b'}}",
                "options['k'] must be a str or a list of str, got dict",
            ),
            ("{'k': ['a', 1]}", "options['k'][1] must be a str, got int"),
            ("{'k': [['a']]}", "options['k'][0] must be a str, got list"),
            (
                "{'k': ('a', None)}",
                "options['k'][1] must be a str, got NoneType",
            ),
            ("{1: 'a'}", "options keys must be str, got int"),
            ("['a']", "options must be a dict"),
        ] {
            let err = extract_options(expr)
                .map(|_| ())
                .expect_err(&format!("{expr} must be rejected"));
            Python::attach(|py| {
                assert!(
                    err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{expr}: expected TypeError, got {err}"
                );
            });
            assert!(
                err.to_string().contains(needle),
                "{expr}: message {err} should contain {needle:?}"
            );
        }
    }

    #[test]
    fn extract_bound_circuit_reads_a_qasm_string() {
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
            let bound = extract_bound_circuit(py.None().bind(py))
                .expect("a non-Circuit, non-str object is assumed to be a Qiskit circuit");
            // A Qiskit circuit rides through the pyo3-free enum's `Foreign` hatch.
            assert!(bound.is_foreign());
        });
    }

    #[test]
    fn extract_bound_circuit_rejects_a_qiskit_circuit_with_free_parameters() {
        // Stands in for a `QuantumCircuit` (only `parameters` is read), so the
        // test needs no Qiskit.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qc = py
                .eval(
                    c"type('QC', (), {'parameters': ['t', 'theta[0]']})()",
                    None,
                    None,
                )
                .expect("the stand-in builds");
            let err = match extract_bound_circuit(&qc) {
                Err(err) => err,
                Ok(_) => panic!("a circuit with free parameters cannot be executed directly"),
            };
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = err.to_string();
            assert!(message.contains("unbound parameters"), "{message}");
            assert!(message.contains("directly: t, theta[0]."), "{message}");
            let unbound = py
                .eval(c"type('QC', (), {'parameters': []})()", None, None)
                .expect("the stand-in builds");
            assert!(extract_bound_circuit(&unbound).is_ok_and(|b| b.is_foreign()));
        });
    }

    #[test]
    fn extract_bound_circuit_reads_a_fully_bound_native_circuit() {
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
            let (class, msg) = label_error(py, c"[0.5, float('nan')]");
            assert_eq!(class, "ConfigurationError");
            assert!(msg.contains("y_train[1] is NaN"), "{msg}");

            let (class, msg) = label_error(py, c"[float('-inf')]");
            assert_eq!(class, "ConfigurationError");
            assert!(msg.contains("y_train[0] is -inf"), "{msg}");
        });
    }

    #[test]
    fn extract_supervised_objective_accepts_callables_and_the_two_wrappers() {
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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

    /// `polypus.CMAES` parses into `Method::Cmaes` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn cmaes_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let cmaes = Py::new(py, CMAES::new(40, Some(12), 0.75, 0.5, 7, 3, Some(99)))
                .expect("the pyclass instantiates");
            let bound = cmaes.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Cmaes(cfg)) => {
                    assert_eq!(cfg.generations, 40);
                    assert_eq!(cfg.population_size, Some(12));
                    assert_eq!(cfg.sigma0, 0.75);
                    assert_eq!(cfg.tolerance, 0.5);
                    assert_eq!(cfg.patience, 7);
                    assert_eq!(cfg.restarts, 3);
                }
                _ => panic!("a polypus.CMAES parses into Method::Cmaes"),
            }
            bound.setattr("population_size", py.None()).unwrap();
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Cmaes(cfg)) => assert_eq!(cfg.population_size, None),
                _ => panic!("a polypus.CMAES parses into Method::Cmaes"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            assert_eq!(method_seed(not_a_method.as_any()), None);
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.CMAES"), "{err}");
        });
    }

    /// `polypus.Adam` parses into `Method::Adam` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn adam_reaches_the_method_and_its_seed_is_read() {
        use polypus_optimizers::AdamRule;
        Python::initialize();
        Python::attach(|py| {
            let adam = Py::new(
                py,
                Adam::new(
                    40,
                    0.2,
                    0.8,
                    0.99,
                    1e-6,
                    (-1.0, 2.0),
                    0.5,
                    7,
                    0.05,
                    "momentum",
                    Some(99),
                )
                .expect("a valid rule"),
            )
            .expect("the pyclass instantiates");
            let bound = adam.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Adam(cfg)) => {
                    assert_eq!(cfg.max_iters, 40);
                    assert_eq!(cfg.learning_rate, 0.2);
                    assert_eq!(cfg.beta1, 0.8);
                    assert_eq!(cfg.beta2, 0.99);
                    assert_eq!(cfg.epsilon, 1e-6);
                    assert_eq!(cfg.bounds, (-1.0, 2.0));
                    assert_eq!(cfg.tolerance, 0.5);
                    assert_eq!(cfg.patience, 7);
                    assert_eq!(cfg.finite_difference_step, 0.05);
                    assert_eq!(cfg.rule, AdamRule::Momentum);
                }
                _ => panic!("a polypus.Adam parses into Method::Adam"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.Adam"), "{err}");
        });
    }

    /// `polypus.SPSA` parses into `Method::Spsa` with every field, its pinned
    /// seed is the C-7 fallback, and a non-method object's `TypeError` names it.
    #[test]
    fn spsa_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let spsa = Py::new(py, Spsa::new(40, 8, 0.2, 1.2, 1.0, 0.25, 5, Some(99), 4.0))
                .expect("the pyclass instantiates");
            let bound = spsa.bind(py);
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Spsa(cfg)) => {
                    assert_eq!(cfg.max_iters, 40);
                    assert_eq!(cfg.resamplings, 8);
                    assert_eq!(cfg.perturbation, 0.2);
                    assert_eq!(cfg.target_step, 1.2);
                    assert_eq!(cfg.alpha, 1.0);
                    assert_eq!(cfg.gamma, 0.25);
                    assert_eq!(cfg.stability_constant, 4.0);
                    assert_eq!(cfg.eval_interval, 5);
                }
                _ => panic!("a polypus.SPSA parses into Method::Spsa"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.SPSA"), "{err}");
        });
    }

    /// `polypus.Rotosolve` parses into `Method::Rotosolve` with every field,
    /// its pinned seed is the C-7 fallback, and a non-method object's
    /// `TypeError` names it.
    #[test]
    fn rotosolve_reaches_the_method_and_its_seed_is_read() {
        Python::initialize();
        Python::attach(|py| {
            let rotosolve = Py::new(py, Rotosolve::new(7, 3, 0.25, 4, Some(99)))
                .expect("the pyclass instantiates");
            let bound = rotosolve.bind(py);
            assert!(is_rotosolve(bound.as_any()));
            assert_eq!(method_seed(bound.as_any()), Some(99));
            match method_from_pyclass(bound.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::Rotosolve(cfg)) => {
                    assert_eq!(cfg.sweeps, 7);
                    assert_eq!(cfg.reevaluate_every, 3);
                    assert_eq!(cfg.tolerance, 0.25);
                    assert_eq!(cfg.patience, 4);
                }
                _ => panic!("a polypus.Rotosolve parses into Method::Rotosolve"),
            }

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            assert!(!is_rotosolve(not_a_method.as_any()));
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("polypus.Rotosolve"), "{err}");
        });
    }

    /// `polypus.External` parses into `Method::External` with its budget, has
    /// no seed of its own (C-7), refuses any `initial_point` before the point
    /// is checked, and is named by a non-method object's `TypeError`.
    #[test]
    fn external_reaches_the_method_without_a_seed() {
        Python::initialize();
        Python::attach(|py| {
            let optimizer = PyModule::from_code(
                py,
                c"class Done:\n    def ask(self):\n        return None\n    def tell(self, c, v):\n        pass\n",
                c"external_done.py",
                c"external_done",
            )
            .and_then(|module| module.getattr("Done")?.call0())
            .expect("the optimizer instantiates");
            let external = Bound::new(
                py,
                External::new(optimizer, 12_i32.into_pyobject(py).unwrap().into_any())
                    .expect("a valid External"),
            )
            .expect("the pyclass instantiates");
            assert_eq!(method_seed(external.as_any()), None);
            match method_from_pyclass(external.as_any(), 1, &OracleErrorSlot::new(), "run") {
                Ok(Method::External(cfg, _)) => assert_eq!(cfg.max_evaluations, 12),
                _ => panic!("a polypus.External parses into Method::External"),
            }

            assert!(check_external_initial_point(external.as_any(), None).is_ok());
            // Even a point of the wrong length gets External's reason.
            let err = check_external_initial_point(external.as_any(), Some(&[0.5]))
                .expect_err("External refuses a point");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.value(py)
                    .to_string()
                    .starts_with("polypus.External takes no initial_point"),
                "{err}"
            );
            let rotosolve = Py::new(py, Rotosolve::new(7, 3, 0.25, 4, None)).unwrap();
            assert!(
                check_external_initial_point(rotosolve.bind(py).as_any(), Some(&[0.5])).is_ok()
            );

            let not_a_method = 3_i32.into_pyobject(py).unwrap();
            let err = method_from_pyclass(not_a_method.as_any(), 1, &OracleErrorSlot::new(), "run")
                .err()
                .expect("an int is not a method");
            assert!(err.to_string().contains("or polypus.External"), "{err}");
        });
    }

    /// A native circuit is checked in Rust: one that qualifies passes, and the
    /// first problem of one that does not is the `ValueError`'s reason.
    #[test]
    fn a_native_circuit_is_checked_for_rotosolve_in_rust() {
        use crate::circuit::{Param, ParameterizedCircuit};
        Python::initialize();
        Python::attach(|py| {
            let unused = 0_i32.into_pyobject(py).unwrap();
            let good = CircuitSource::Native(
                ParameterizedCircuit::new(2)
                    .ry(0, Param(0))
                    .cx(0, 1)
                    .rz(1, Param(1)),
            );
            assert!(check_rotosolve_circuit(unused.as_any(), Some(&good), None).is_ok());
            let bad = CircuitSource::Native(ParameterizedCircuit::new(2).ry(0, Param(0)).crx(
                0,
                1,
                Param(1),
            ));
            let err = check_rotosolve_circuit(unused.as_any(), Some(&bad), None)
                .expect_err("a controlled rotation does not qualify");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = err.value(py).to_string();
            assert!(
                message.starts_with(
                    "polypus.Rotosolve cannot train this circuit: parameter 1 ('theta_1') is \
                     an angle of 'crx' (instruction 1), which is not a rotation generated by a \
                     single Pauli operator. Rotosolve needs"
                ),
                "{message}"
            );
        });
    }

    /// A configuration the optimizer rejects once the run has started
    /// (`OracleError::Config`) is a `polypus.ConfigurationError`, like every
    /// configuration check made before it (issue #418).
    #[test]
    fn a_rejected_optimizer_configuration_is_a_configuration_error() {
        Python::initialize();
        let err = Python::attach(|py| {
            finish_optimization(
                py,
                Err(OracleError::Config(
                    polypus_optimizers::OptimizerError::InvalidConfig {
                        parameter: "population_size",
                        reason: "must be at least 4, got 3".to_string(),
                    },
                )),
                0,
                "run".to_string(),
                Instant::now(),
            )
            .expect_err("a rejected configuration is an error")
        });
        crate::exceptions::tests::assert_configuration_error(&err, "must be at least 4, got 3");
    }

    /// An oracle that returned the wrong number of values or a non-finite
    /// gradient failed during the run: `polypus.EvaluationError`, not a
    /// rejected configuration (issue #467).
    #[test]
    fn an_oracle_failure_reported_by_the_optimizer_is_an_evaluation_error() {
        Python::initialize();
        for (failure, message) in [
            (
                polypus_optimizers::OptimizerError::OracleLengthMismatch {
                    expected: 3,
                    got: 2,
                },
                "returned 2 fitness value(s) for 3 candidate(s)",
            ),
            (
                polypus_optimizers::OptimizerError::NonFiniteGradient {
                    index: 1,
                    value: f64::NAN,
                },
                "gradient oracle returned NaN for parameter 1",
            ),
        ] {
            Python::attach(|py| {
                let err = finish_optimization(
                    py,
                    Err(OracleError::Config(failure)),
                    0,
                    "run".to_string(),
                    Instant::now(),
                )
                .expect_err("an oracle failure is an error");
                assert!(
                    err.is_instance_of::<crate::exceptions::EvaluationError>(py),
                    "{err}"
                );
                assert!(
                    !err.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                    "{err}"
                );
                assert!(err.value(py).to_string().contains(message), "{err}");
            });
        }
    }

    /// Register `name` as a backend whose factory counts its calls in `builds`
    /// and fails, so a test can tell whether an entry point created it.
    fn register_sentinel_backend(name: &str, builds: &'static std::sync::atomic::AtomicUsize) {
        crate::infrastructure::register_backend(
            name,
            move |_ctx: &crate::infrastructure::BackendBuildContext| {
                builds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err(crate::infrastructure::BackendError::Conversion(
                    "the sentinel backend was created".to_string(),
                ))
            },
        );
    }

    /// A `polypus.DE` built through its Python signature, so these tests do not
    /// depend on the order of `DE::new`'s arguments.
    fn de<'py>(py: Python<'py>, population_size: u32) -> Bound<'py, PyAny> {
        let kwargs = [("generations", 1), ("population_size", population_size)]
            .into_py_dict(py)
            .expect("the kwargs build");
        py.get_type::<DE>()
            .call((), Some(&kwargs))
            .expect("the pyclass instantiates")
    }

    /// `train` with every argument but these fixed: one dimension, a Qiskit-like
    /// placeholder circuit (nothing reads it before the backend exists).
    fn train_on<'py>(
        py: Python<'py>,
        infrastructure: &str,
        method: Bound<'py, PyAny>,
        expectation_function: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        train(
            py.None().into_bound(py),
            method,
            16,
            1,
            1,
            expectation_function,
            infrastructure.to_string(),
            1,
            1,
            "edge".to_string(),
            "automatic",
            None,
            "aer",
            Some(3),
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Stand-ins for a Qiskit feature map and ansatz: what `qml.train` reads of
    /// them before the backend exists (`parameters`, `compose`, `num_clbits`,
    /// `assign_parameters`).
    fn fake_qml_circuits(py: Python<'_>) -> (Bound<'_, PyAny>, Bound<'_, PyAny>) {
        let module = PyModule::from_code(
            py,
            c"class Circuit:
    num_clbits = 1
    def __init__(self, *parameters):
        self.parameters = list(parameters)
    def compose(self, other):
        return Circuit(*self.parameters, *other.parameters)
    def assign_parameters(self, values, inplace=False):
        return self
",
            c"fake_qml_circuits.py",
            c"fake_qml_circuits",
        )
        .expect("the module compiles");
        let circuit = module.getattr("Circuit").expect("the class exists");
        (
            circuit.call1(("x",)).expect("a feature map"),
            circuit.call1(("w",)).expect("an ansatz"),
        )
    }

    /// `qml.train` with every argument but these fixed: one feature, one weight,
    /// two training rows.
    fn qml_train_on<'py>(
        py: Python<'py>,
        infrastructure: &str,
        method: Bound<'py, PyAny>,
        expectation_function: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let (feature_map, ansatz) = fake_qml_circuits(py);
        let rows = py.eval(c"[[0.1], [0.2]]", None, None).expect("the rows");
        qml_train(
            feature_map,
            ansatz,
            rows,
            method,
            16,
            1,
            1,
            expectation_function,
            infrastructure.to_string(),
            1,
            1,
            "edge".to_string(),
            "automatic",
            None,
            "aer",
            Some(3),
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// `train` and `qml.train` convert `expectation_function` and check the
    /// optimizer before they create the backend, which on CUNQA requests a
    /// SLURM allocation (issue #418).
    #[test]
    fn training_checks_the_configuration_before_creating_the_backend() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TRAIN_BUILDS: AtomicUsize = AtomicUsize::new(0);
        static QML_BUILDS: AtomicUsize = AtomicUsize::new(0);
        register_sentinel_backend("sentinel-train-418", &TRAIN_BUILDS);
        register_sentinel_backend("sentinel-qml-train-418", &QML_BUILDS);
        Python::initialize();
        Python::attach(|py| {
            let not_an_observable = 42_i32.into_pyobject(py).unwrap().into_any();
            let cost = py.eval(c"lambda bits: 0.0", None, None).unwrap();
            type Run = for<'py> fn(
                Python<'py>,
                &str,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
            ) -> PyResult<Py<PyAny>>;
            let entry_points: [(&str, &str, &AtomicUsize, Run); 2] = [
                ("train", "sentinel-train-418", &TRAIN_BUILDS, train_on),
                (
                    "qml.train",
                    "sentinel-qml-train-418",
                    &QML_BUILDS,
                    qml_train_on,
                ),
            ];
            for (name, infrastructure, builds, run) in entry_points {
                let err = run(py, infrastructure, de(py, 4), not_an_observable.clone())
                    .expect_err("an invalid observable is rejected");
                assert!(
                    err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{name}: {err}"
                );
                assert!(
                    err.to_string().contains("expectation_function"),
                    "{name}: {err}"
                );

                let err = run(py, infrastructure, de(py, 3), cost.clone())
                    .expect_err("a population too small is rejected");
                crate::exceptions::tests::assert_configuration_error(&err, "population_size");
                assert_eq!(
                    builds.load(Ordering::SeqCst),
                    0,
                    "{name} created the backend"
                );

                // Control: a valid configuration does create it.
                let err = run(py, infrastructure, de(py, 4), cost.clone())
                    .expect_err("the sentinel backend fails");
                assert!(
                    err.to_string().contains("sentinel backend was created"),
                    "{name}: {err}"
                );
                assert_eq!(builds.load(Ordering::SeqCst), 1, "{name}");
            }
        });
    }

    /// `qml.train` and `qml.predict` refuse `infrastructure="qmio"` before they
    /// read the model, which QMIO could not serialize.
    #[test]
    fn qml_rejects_qmio_before_reading_the_model() {
        Python::initialize();
        Python::attach(|py| {
            let none = || py.None().into_bound(py);
            let cost = py.eval(c"lambda bits: 0.0", None, None).unwrap();
            let train_err = qml_train(
                none(),
                none(),
                none(),
                de(py, 4),
                16,
                1,
                1,
                cost,
                "qmio".to_string(),
                1,
                1,
                "edge".to_string(),
                "automatic",
                None,
                "aer",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect_err("qmio is rejected");
            let predict_err = qml_predict(
                none(),
                none(),
                none(),
                vec![0.0],
                16,
                "qmio".to_string(),
                1,
                1,
                1,
                "automatic",
                None,
                "aer",
                None,
                None,
            )
            .expect_err("qmio is rejected");
            for err in [train_err, predict_err] {
                crate::exceptions::tests::assert_configuration_error(
                    &err,
                    "cannot serialize the Qiskit circuits of a QML model",
                );
            }
        });
    }

    /// The argument checks shared by the entry points raise
    /// `polypus.ConfigurationError`.
    #[test]
    fn argument_checks_raise_configuration_error() {
        // Before the cases: without an interpreter the class cannot be looked up.
        Python::initialize();
        let cases: [(PyResult<()>, &str); 6] = [
            (validate_id(""), "id must not be empty"),
            (validate_id("a b"), "' '"),
            (validate_id(&"a".repeat(65)), "64"),
            (validate_cunqa_allocation("cunqa", 0, 1, None), "nodes"),
            (
                validate_cunqa_allocation("cunqa", 1, 0, None),
                "cores_per_qpu",
            ),
            (
                validate_cunqa_allocation("cunqa", 1, 1, Some("1-00:00:00")),
                "walltime",
            ),
        ];
        for (result, message) in cases {
            let err = result.expect_err("the argument is rejected");
            crate::exceptions::tests::assert_configuration_error(&err, message);
        }
    }
}
