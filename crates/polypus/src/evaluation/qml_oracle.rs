use crate::evaluation::{assign_parameters_qiskit, run_and_expect, EvaluationOracle};
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
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let rt = crate::utils::tokio_runtime();

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
            py.allow_threads(|| {
                rt.block_on(async {
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        out.push(h.await.expect("QML eval task panicked"));
                    }
                    out
                })
            })
        })
    }
}

/// Evaluate one candidate `theta` against all training circuits.
///
/// Binds `theta` to each training circuit, runs them in batches of
/// `config.n_qpus`, and returns the mean expectation value.
fn evaluate_qml_single(
    training_circuits: &[Py<PyAny>],
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    expectation_fn: &Py<PyAny>,
    theta: &[f64],
) -> f64 {
    // Training circuits are Qiskit objects (feature-map pre-binding is
    // Qiskit-specific); native QML circuits arrive with a later phase.
    let bound: Vec<BoundCircuit> = training_circuits
        .iter()
        .map(|qc_xi| BoundCircuit::Qiskit(assign_parameters_qiskit(qc_xi, theta)))
        .collect();

    let batch_size = backend.max_batch_size(bound.len()).max(1);
    let mut all_ev: Vec<f64> = Vec::with_capacity(bound.len());
    for chunk in bound.chunks(batch_size) {
        let ev = run_and_expect(backend, chunk, config, expectation_fn);
        all_ev.extend(ev);
    }

    all_ev.iter().sum::<f64>() / all_ev.len() as f64
}
