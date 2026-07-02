use crate::evaluation::{run_and_expect, CircuitSource, EvaluationOracle};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use pyo3::prelude::*;
use std::sync::Arc;

/// Oracle for standard VQC training.
///
/// Holds a single parameterised circuit template ([`CircuitSource`]). For each
/// candidate parameter vector `θ`, it binds `θ` to the template, runs the
/// resulting circuit through the backend, and returns the expectation value
/// computed by `expectation_fn`.
///
/// With a [`CircuitSource::Native`] template the per-candidate binding is pure
/// Rust (no GIL); with [`CircuitSource::Qiskit`] it calls Python's
/// `assign_parameters` as before.
///
/// Circuits are submitted to the backend in chunks of `max_batch_size` so that
/// each chunk maps to one backend call (one QPU batch for CUNQA, one Aer call
/// for local).
pub struct VqcOracle {
    /// Parameterised circuit template (ansatz parameters unbound).
    pub circuit: CircuitSource,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    pub expectation_fn: Py<PyAny>,
}

impl EvaluationOracle for VqcOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Bind each candidate to the circuit template. For native circuits
        // this loop never touches Python.
        let bound: Vec<BoundCircuit> = candidates
            .iter()
            .map(|params| self.circuit.bind(params))
            .collect();

        // Submit circuits in backend-sized batches and collect expectations.
        // Local runs the whole batch in one Aer call (parallel experiments);
        // CUNQA caps each call at n_qpus (one circuit per QPU).
        let batch_size = self.backend.max_batch_size(bound.len()).max(1);
        let mut results = Vec::with_capacity(candidates.len());
        for chunk in bound.chunks(batch_size) {
            let ev = run_and_expect(
                self.backend.as_ref(),
                chunk,
                &self.config,
                &self.expectation_fn,
            );
            results.extend(ev);
        }
        results
    }
}
