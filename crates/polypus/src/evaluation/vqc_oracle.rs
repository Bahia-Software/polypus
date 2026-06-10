use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use crate::infrastructure::{QuantumBackend, ExecutionConfig};
use crate::evaluation::{EvaluationOracle, run_and_expect};

/// Oracle for standard VQC training.
///
/// Holds a single parameterised circuit template. For each candidate parameter
/// vector `θ`, it binds `θ` to the template, runs the resulting circuit through
/// the backend, and returns the expectation value computed by `expectation_fn`.
///
/// Circuits are submitted to the backend in chunks of `config.n_qpus` so that
/// each chunk maps to one backend call (one QPU batch for CUNQA, one Aer call
/// for local).
pub struct VqcOracle {
    /// Parameterised circuit template (ansatz parameters unbound).
    pub circuit: Py<PyAny>,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    pub expectation_fn: Py<PyAny>,
}

impl EvaluationOracle for VqcOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Bind each candidate to the circuit template.
        let bound: Vec<Py<PyAny>> = candidates
            .iter()
            .map(|params| assign_parameters(&self.circuit, params))
            .collect();

        // Submit circuits in backend-sized batches and collect expectations.
        // Local runs the whole batch in one Aer call (parallel experiments);
        // CUNQA caps each call at n_qpus (one circuit per QPU).
        let batch_size = self.backend.max_batch_size(bound.len()).max(1);
        let mut results = Vec::with_capacity(candidates.len());
        for chunk in bound.chunks(batch_size) {
            let ev = run_and_expect(self.backend.as_ref(), chunk, &self.config, &self.expectation_fn);
            results.extend(ev);
        }
        results
    }
}

/// Bind `params` to a copy of `circuit` and return the bound circuit.
pub(crate) fn assign_parameters(circuit: &Py<PyAny>, params: &[f64]) -> Py<PyAny> {
    Python::with_gil(|py| {
        let qc = circuit
            .clone_ref(py)
            .into_pyobject(py)
            .expect("Failed to get circuit as PyObject");
        let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
        qc.call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .expect("Error assigning parameters to circuit")
            .unbind()
    })
}
