pub mod vqc_oracle;
pub mod qml_oracle;

pub use vqc_oracle::VqcOracle;
pub use qml_oracle::QmlOracle;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use crate::infrastructure::{QuantumBackend, ExecutionConfig};

/// Contract between optimization algorithms and quantum circuit evaluation.
///
/// An oracle encapsulates everything needed to translate a parameter vector
/// into a scalar fitness value: the circuit template (or training circuits),
/// the backend, and the expectation function.
///
/// Algorithms only call [`EvaluationOracle::evaluate_batch`] and have no
/// knowledge of circuits, QPUs, infrastructure, or training modes.
///
/// To add a new evaluation strategy (e.g. noisy readout mitigation, hardware
/// native gates, …) implement this trait without touching any algorithm.
pub trait EvaluationOracle: Send + Sync {
    /// Evaluate a batch of candidate parameter vectors.
    ///
    /// Returns one fitness value per candidate. Higher is better
    /// (algorithms maximise the expectation value).
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64>;
}

/// Execute a batch of bound circuits through `backend` and extract expectation
/// values using the Python `expectation_fn`.
///
/// This is the **single place** in the codebase that calls
/// `polypus_python.expectation_values`, eliminating the duplication that
/// previously existed across DE, PSO, QNG, and the orchestration layer.
pub(crate) fn run_and_expect(
    backend: &dyn QuantumBackend,
    qcs: &[Py<PyAny>],
    config: &ExecutionConfig,
    expectation_fn: &Py<PyAny>,
) -> Vec<f64> {
    let counts = backend.run_circuits(qcs, config);
    Python::with_gil(|py| {
        // Convert the native counts back into a Python `list[dict]` for the
        // Python `expectation_values` function. Once expectation computation is
        // also native this round-trip disappears entirely.
        let py_counts = counts
            .into_pyobject(py)
            .expect("Failed to convert counts to a Python object");
        PyModule::import(py, "polypus_python")
            .expect("Failed to import polypus_python")
            .call_method("expectation_values", (py_counts, expectation_fn), None)
            .expect("Error computing expectation values")
            .extract::<Vec<f64>>()
            .expect("Failed to extract expectation values as Vec<f64>")
    })
}
