//! What the entry points return: [`RunResult`] and [`TrainResult`] (contract
//! C-7), the counts conversion of contract C-3, and how an optimizer's outcome,
//! or its recorded failure, becomes the value or exception Python sees.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::infrastructure::Counts;
use crate::orchestration::{FinalEvaluation, OracleError, TrainOutcome};
use polypus_optimizers::OptimizationOutcome;

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
pub(super) fn counts_to_pydict<'py>(
    py: Python<'py>,
    counts: &Counts,
) -> PyResult<Bound<'py, PyDict>> {
    let mut outcomes: Vec<(&String, &u64)> = counts.iter().collect();
    outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let dict = PyDict::new(py);
    for (bitstring, n) in outcomes {
        dict.set_item(bitstring, n)?;
    }
    Ok(dict)
}

/// One [`counts_to_pydict`] dict per map, in order, as a Python `list`.
pub(super) fn counts_to_pylist<'py>(
    py: Python<'py>,
    counts: &[Counts],
) -> PyResult<Bound<'py, PyList>> {
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
    /// What ended the optimizer's loop, the snake_case name of a
    /// [`StopReason`](polypus_optimizers::StopReason): `"iteration_budget"`,
    /// `"fitness_tolerance"`, `"callback"`, … L-SHADE's budget is one of
    /// evaluations, so it reports `"evaluation_budget"`.
    #[pyo3(get)]
    pub stop_reason: &'static str,
    /// Candidates scored as a fitness, the final batch not included (see
    /// [`TrainOutcome::n_evaluations`]). The shifted circuits of an exact
    /// gradient (`Adam` in `qml.Model.fit`) are not counted; the point it
    /// measures in their batch is.
    #[pyo3(get)]
    pub n_evaluations: u64,
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
             converged={}, stop_reason={:?}, n_evaluations={}, seed={}, best_params={:?}\
             {final_params})",
            self.id,
            self.best_fitness,
            self.iterations_run,
            self.converged,
            self.stop_reason,
            self.n_evaluations,
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
pub(super) fn outcome_to_train_result(
    py: Python<'_>,
    outcome: OptimizationOutcome,
    final_evaluation: Option<FinalEvaluation>,
    n_evaluations: u64,
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
            stop_reason: outcome.stop_reason.as_str(),
            n_evaluations,
            final_params,
            final_fitness,
            seed,
            id,
        },
    )
    .map(|result| result.into_any())
}

/// Announce the start of a training run at the default log level.
///
/// The fields are exactly the C-7 manifest data — id, infrastructure, backend,
/// n_qpus, shots, effective seed — so an operator watching a multi-hour run can
/// tell from the log alone what was executed where, and with which seed to
/// replay it. Shared by `train` and `qml_train`, whose start records are
/// identical; nothing high-volume (no circuit dumps) belongs at this level.
pub(super) fn log_training_start(
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
pub(super) fn finish_optimization(
    py: Python<'_>,
    result: Result<TrainOutcome, OracleError>,
    seed: u64,
    id: String,
    start: Instant,
) -> PyResult<Py<PyAny>> {
    let (outcome, final_evaluation, n_evaluations) = match result {
        Ok(outcome) => (
            outcome.optimization,
            outcome.final_evaluation,
            outcome.n_evaluations,
        ),
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
        "training run {id} completed: iterations_run={}, converged={}, stop_reason={}, \
         n_evaluations={n_evaluations}{final_fitness}, duration={:?}",
        outcome.iterations_run,
        outcome.converged,
        outcome.stop_reason,
        start.elapsed()
    );
    outcome_to_train_result(py, outcome, final_evaluation, n_evaluations, seed, id)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
