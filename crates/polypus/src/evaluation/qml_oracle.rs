use crate::evaluation::{
    assign_parameters_qiskit, run_and_evaluate, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use pyo3::prelude::*;
use std::sync::Arc;

/// Oracle for QML training with feature-map encoding.
///
/// Holds N pre-bound training circuits (one per training sample, with
/// feature-map parameters already fixed). For each candidate `θ`, it binds `θ`
/// to every training circuit, runs them (batched by `config.n_qpus`), and
/// returns the **mean** expectation value as the fitness.
///
/// Each candidate is evaluated concurrently via Tokio `spawn_blocking`. Note
/// that the GIL serialises the actual simulation calls; once a native Rust
/// backend is available the parallelism will be genuine.
pub struct QmlOracle {
    /// Pre-bound training circuits (feature-map parameters already fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    pub expectation_fn: Py<PyAny>,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates) {
            Ok(values) => values,
            Err(e) => {
                self.errors.record(e);
                vec![0.0; candidates.len()]
            }
        }
    }
}

impl QmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for QML evaluation: {e}"
            )))
        })?;

        let handles: Vec<_> = candidates
            .iter()
            .map(|theta| {
                // Clone per-task data (GIL required for Py<T> clone).
                let training_circuits: Vec<Py<PyAny>> = Python::with_gil(|py| {
                    self.training_circuits
                        .iter()
                        .map(|qc| qc.clone_ref(py))
                        .collect()
                });
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let ef = Python::with_gil(|py| self.expectation_fn.clone_ref(py));
                let theta = theta.clone();

                rt.spawn_blocking(move || {
                    evaluate_qml_single(&training_circuits, &config, backend.as_ref(), &ef, &theta)
                })
            })
            .collect();

        // The calling thread entered Rust from a PyO3 `#[pyfunction]` and still
        // holds the GIL. Each `spawn_blocking` worker needs to acquire the GIL
        // (binding circuits, running them, computing expectations), so we MUST
        // release it here while blocking on them — otherwise this thread holds
        // the GIL inside `block_on` while the workers wait for it: a deadlock.
        Python::with_gil(|py| {
            let out = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        // A `JoinError` means the worker task itself panicked;
                        // turn it into a typed error rather than re-panicking.
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("QML evaluation task failed: {e}"),
                            ))
                        })?;
                        out.push(single?);
                    }
                    Ok::<_, EvaluationError>(out)
                })
            })?;
            // The workers ran off the main thread, where `PyErr_CheckSignals` is
            // a no-op, so a pending SIGINT (Ctrl+C) was not seen there. Check it
            // here on the calling (main) thread so `qml.train` is interruptible
            // too, at the same per-batch granularity as the native path; the
            // KeyboardInterrupt is carried verbatim via `EvaluationError::Python`.
            py.check_signals().map_err(EvaluationError::Python)?;
            Ok(out)
        })
    }
}

/// Evaluate one candidate `theta` against all training circuits.
///
/// Binds `theta` to each training circuit, runs them in batches of
/// `config.n_qpus`, and returns the mean expectation value. Any failure is
/// returned as an [`EvaluationError`] instead of panicking.
fn evaluate_qml_single(
    training_circuits: &[Py<PyAny>],
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    expectation_fn: &Py<PyAny>,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
    // Training circuits are Qiskit objects (feature-map pre-binding is
    // Qiskit-specific); native QML circuits arrive with a later phase.
    let bound: Vec<BoundCircuit> = training_circuits
        .iter()
        .map(|qc_xi| {
            Ok(BoundCircuit::Qiskit(assign_parameters_qiskit(
                qc_xi, theta,
            )?))
        })
        .collect::<Result<_, EvaluationError>>()?;

    let batch_size = backend.max_batch_size(bound.len()).max(1);
    let mut all_ev: Vec<f64> = Vec::with_capacity(bound.len());
    for chunk in bound.chunks(batch_size) {
        let ev = run_and_evaluate(backend, chunk, config, expectation_fn)?;
        all_ev.extend(ev);
    }

    // Defense-in-depth (contract C-5): `run_and_evaluate` already guarantees
    // exactly `chunk.len()` values per chunk, and the chunks partition `bound`
    // exactly, so this can only ever hold. Unlike `VqcOracle` this function
    // returns one scalar per *candidate* (the mean over the training circuits),
    // so the invariant here is `all_ev.len() == bound.len()` (one expectation
    // per training circuit), not `candidates.len()`. Kept as an explicit,
    // self-documenting invariant guarding the mean below — do not "simplify" it
    // away. Reported as a `Result`, never a panic (rule 4; runs under
    // `OracleErrorSlot`).
    if all_ev.len() != bound.len() {
        return Err(EvaluationError::WrongLength {
            expected: bound.len(),
            got: all_ev.len(),
        });
    }

    Ok(all_ev.iter().sum::<f64>() / all_ev.len() as f64)
}
