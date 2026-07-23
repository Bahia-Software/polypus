use crate::evaluation::{
    run_and_evaluate, CircuitSource, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_optimizers::GradientOracle;
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
    /// Shared with the `train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for VqcOracle {
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

impl GradientOracle for VqcOracle {
    /// Exact parameter-shift gradient (noiseless limit; an unbiased estimator
    /// under shot noise). The VQC fitness is the raw expectation computed by
    /// `expectation_fn` — linear in each shifted expectation, no `Loss` on top —
    /// so shifting the whole candidate by `±π/2` and delegating to the existing
    /// `evaluate_batch` is exact by linearity. The caller is responsible for
    /// that precondition holding (contract C-5's `GradientOracle`): if a
    /// user-supplied `expectation_fn` is a nonlinear function of the raw
    /// expectation, the value is no longer the exact gradient.
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        // A prior failure short-circuits, mirroring `evaluate_batch`. No error
        // recording is needed here: any failure inside the delegated
        // `evaluate_batch` already records itself through the normal
        // `EvaluationOracle` path.
        if self.errors.failed() {
            return vec![0.0; dims];
        }
        polypus_optimizers::linear_parameter_shift_gradient(self, theta, dims)
    }

    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.gradient_batch(theta, param_index + 1)[param_index]
    }
}

impl VqcOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        // Bind each candidate to the circuit template. For native circuits
        // this loop never touches Python.
        let bound: Vec<BoundCircuit> = candidates
            .iter()
            .map(|params| self.circuit.bind(params))
            .collect::<Result<_, _>>()?;

        // Submit circuits in backend-sized batches and collect expectations.
        // Local runs the whole batch in one Aer call (parallel experiments);
        // CUNQA caps each call at n_qpus (one circuit per QPU).
        let batch_size = self.backend.max_batch_size(bound.len()).max(1);
        let mut results = Vec::with_capacity(candidates.len());
        for chunk in bound.chunks(batch_size) {
            let ev = run_and_evaluate(
                self.backend.as_ref(),
                chunk,
                &self.config,
                &self.expectation_fn,
            )?;
            results.extend(ev);
        }
        Ok(results)
    }
}
