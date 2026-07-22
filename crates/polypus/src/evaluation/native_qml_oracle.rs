use crate::evaluation::{EvaluationError, EvaluationOracle, OracleErrorSlot};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_qml::QmlProblem;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Oracle for native (pure-Rust) QML training (contract C-8).
///
/// This is the native counterpart to [`QmlOracle`](crate::evaluation::QmlOracle):
/// where `QmlOracle` holds Qiskit `QuantumCircuit`s and computes expectations
/// through Python, this holds a [`QmlProblem`] — a compiled model, a training
/// set and a loss produced entirely by `polypus-qml` — and lets it bind
/// parameters and score counts in pure Rust. It works on **any** simulated
/// backend: the native statevector simulator (GIL-free end to end) or, via the
/// C-1 seam, Aer / CUNQA's simulated QPUs.
///
/// For each candidate `θ` it binds `θ` to every training circuit
/// ([`QmlProblem::bind_batch`]), runs them (batched by `max_batch_size`), and
/// turns the counts into a single fitness ([`QmlProblem::fitness_from_counts`],
/// `= −mean_loss`, since the optimizers maximise).
///
/// Each candidate is evaluated concurrently via Tokio `spawn_blocking`, mirroring
/// `QmlOracle`. On the native backend that parallelism is genuine (no GIL); on a
/// Python backend the GIL serialises the actual simulation calls, exactly as it
/// does for `QmlOracle`.
pub struct NativeQmlOracle {
    /// The trainable problem: bind parameters in, get fitness out (C-8).
    pub problem: QmlProblem,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for NativeQmlOracle {
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

impl NativeQmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to start the Tokio runtime for native QML evaluation: {e}"
            )))
        })?;

        // Share the problem across tasks by reference-counting it once: `QmlProblem`
        // is pure Rust (no `Py<T>`), so unlike `QmlOracle` there is nothing to
        // clone under the GIL — a single `Arc` bump per candidate suffices.
        let problem = Arc::new(self.problem.clone());
        let handles: Vec<_> = candidates
            .iter()
            .map(|theta| {
                let problem = Arc::clone(&problem);
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                let theta = theta.clone();

                rt.spawn_blocking(move || {
                    evaluate_native_qml_single(&problem, &config, backend.as_ref(), &theta)
                })
            })
            .collect();

        // The calling thread entered Rust from a PyO3 `#[pyfunction]` and still
        // holds the GIL. A `spawn_blocking` worker on a Python backend (Aer/CUNQA)
        // needs to acquire the GIL to run circuits, so we MUST release it here
        // while blocking on the workers — otherwise this thread holds the GIL
        // inside `block_on` while a worker waits for it: a deadlock. (On the
        // native backend the workers never touch the GIL, but releasing it is
        // harmless and keeps the two oracle paths identical.)
        Python::with_gil(|py| {
            let out = py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        // A `JoinError` means the worker task itself panicked;
                        // turn it into a typed error rather than re-panicking.
                        let single = h.await.map_err(|e| {
                            EvaluationError::Python(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("native QML evaluation task failed: {e}"),
                            ))
                        })?;
                        out.push(single?);
                    }
                    Ok::<_, EvaluationError>(out)
                })
            })?;
            // The workers ran off the main thread, where `PyErr_CheckSignals` is
            // a no-op, so a pending SIGINT (Ctrl+C) was not seen there. Check it
            // here on the calling (main) thread so `qml.train` stays interruptible
            // at the same per-batch granularity as `QmlOracle`; the
            // KeyboardInterrupt is carried verbatim via `EvaluationError::Python`.
            py.check_signals().map_err(EvaluationError::Python)?;
            Ok(out)
        })
    }
}

/// Evaluate one candidate `theta` against the whole training set.
///
/// Binds `theta` into one circuit per training sample ([`QmlProblem::bind_batch`]),
/// runs them in batches of `max_batch_size`, reassembles the counts in the same
/// (stable, sample-major) order, and returns the fitness computed by
/// [`QmlProblem::fitness_from_counts`]. Any failure is returned as an
/// [`EvaluationError`] instead of panicking.
fn evaluate_native_qml_single(
    problem: &QmlProblem,
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
    // Bind the candidate into one native circuit per training sample (C-8 (a)).
    let bound: Vec<BoundCircuit> = problem
        .bind_batch(theta)?
        .into_iter()
        .map(BoundCircuit::Native)
        .collect();

    let batch_size = backend.max_batch_size(bound.len()).max(1);
    let mut all_counts: Vec<HashMap<String, u64>> = Vec::with_capacity(bound.len());
    for chunk in bound.chunks(batch_size) {
        // Counts come back one dict per circuit, in submission order (C-3), so
        // extending preserves the sample-major order `fitness_from_counts` needs.
        all_counts.extend(backend.run_circuits(chunk, config)?);
    }

    Ok(problem.fitness_from_counts(&all_counts)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{BackendConfig, NativeStatevectorBackend, OptLevel};
    use polypus_circuit::CircuitError;
    use polypus_qml::{
        Dataset, Decision, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout,
        RotationAxis,
    };

    /// A tiny, fully-Rust `QmlProblem`: a 2-qubit angle-encoder + hardware-efficient
    /// ansatz reading `⟨Z₀⟩` with a `Sign` decision, trained with `Hinge` over two
    /// well-separated samples. Its compiled model reserves 8 trainable parameters.
    fn small_problem() -> QmlProblem {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout);
        let ds = Dataset::from_rows(&[vec![0.4, 0.5], vec![2.6, 2.7]], &[-1.0, 1.0]).unwrap();
        let compiled = model.compile(ds.num_features()).unwrap();
        QmlProblem::new(compiled, ds, Loss::Hinge).unwrap()
    }

    /// A `LocalNative` execution config seeded deterministically.
    fn native_config() -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "native_qml_oracle_test".to_string(),
            shots: 256,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    fn oracle(errors: OracleErrorSlot) -> NativeQmlOracle {
        NativeQmlOracle {
            problem: small_problem(),
            config: native_config(),
            backend: Arc::new(NativeStatevectorBackend::new(7)),
            errors,
        }
    }

    /// A well-formed batch returns exactly `candidates.len()` finite fitnesses and
    /// records no error (contract C-5 length + C-8 (b) finiteness, via the oracle).
    #[test]
    fn evaluate_batch_returns_one_finite_fitness_per_candidate() {
        // `evaluate_batch` acquires the GIL internally (like `QmlOracle`); the
        // interpreter must exist, but no Qiskit/Aer/`polypus_python` is involved —
        // the whole path is native (ENGINEERING §3).
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(errors.clone());
        // The compiled model reserves 8 parameters.
        let candidates = vec![vec![0.1_f64; 8], vec![0.2_f64; 8], vec![0.3_f64; 8]];
        let out = oracle.evaluate_batch(&candidates);
        assert_eq!(out.len(), candidates.len());
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(errors.take().is_none(), "a valid batch records no error");
    }

    /// A real failure (a `θ` of the wrong length) yields the finite `0.0` sentinel
    /// in every position **and** leaves the true error recoverable via the shared
    /// slot — the sentinel + `OracleErrorSlot` contract the entry point relies on.
    #[test]
    fn evaluate_batch_sentinels_on_failure_and_records_error() {
        pyo3::prepare_freethreaded_python();
        let errors = OracleErrorSlot::new();
        let oracle = oracle(errors.clone());
        // 3 parameters where the model needs 8: binding fails per candidate.
        let candidates = vec![vec![0.1_f64; 3], vec![0.2_f64; 3]];
        let out = oracle.evaluate_batch(&candidates);
        assert_eq!(out, vec![0.0, 0.0]);
        let err = errors
            .take()
            .expect("the real error is recoverable via the slot");
        assert!(
            matches!(
                err,
                EvaluationError::Qml(polypus_qml::QmlError::Circuit(
                    CircuitError::WrongNumberOfParams { .. }
                ))
            ),
            "unexpected error variant: {err:?}"
        );
    }
}
