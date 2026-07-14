use crate::evaluation::{run_and_expect, CircuitSource, EvaluationOracle, InterruptState};
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
    /// Shared cancellation state. When Ctrl+C (or any exception raised inside
    /// the Python `expectation_function`) is observed mid-run, it is captured
    /// here and the entry point re-raises it as the original exception (see
    /// [`InterruptState`]).
    pub interrupt: Arc<InterruptState>,
}

impl EvaluationOracle for VqcOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // A Ctrl+C (or a Python error) was already captured on an earlier batch:
        // do no further work. Returning a finite, correctly-sized placeholder
        // keeps the C-5 length/finiteness contract while the optimizer spins
        // cheaply through its remaining generations; the entry point discards
        // this outcome and re-raises the captured exception.
        if self.interrupt.is_interrupted() {
            return vec![0.0; candidates.len()];
        }

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
            match run_and_expect(
                self.backend.as_ref(),
                chunk,
                &self.config,
                &self.expectation_fn,
            ) {
                Ok(ev) => results.extend(ev),
                // Capture the original exception (e.g. `KeyboardInterrupt`) and
                // stop; the entry point turns it back into that exception.
                Err(err) => {
                    self.interrupt.capture(err);
                    return vec![0.0; candidates.len()];
                }
            }
        }
        results
    }
}
