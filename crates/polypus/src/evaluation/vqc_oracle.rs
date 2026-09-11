use crate::evaluation::{
    CircuitSource, CostObservable, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
use crate::infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, ExecutionConfig, Planner, QuantumBackend,
};
use std::sync::Arc;

/// Oracle for standard VQC training.
///
/// Holds a single parameterised circuit template ([`CircuitSource`]). For each
/// candidate parameter vector `θ`, it binds `θ` to the template, runs the
/// resulting circuit through the backend, and returns the expectation value
/// computed by `observable`.
///
/// With a [`CircuitSource::Native`] template the per-candidate binding is pure
/// Rust (no GIL); with [`CircuitSource::Qiskit`] it calls Python's
/// `assign_parameters` as before.
///
/// The oracle owns only the *what* (bind candidates, build tasks, validate C-5);
/// the [`Planner`] owns the *how* (waves, concurrency cap, between-wave
/// `check_signals`, cancellation).
pub struct VqcOracle {
    /// Parameterised circuit template (ansatz parameters unbound).
    pub circuit: CircuitSource,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    /// Owns how the bound circuits are executed and reduced (waves, concurrency).
    pub planner: Arc<dyn Planner>,
    pub observable: Arc<dyn CostObservable>,
    /// Cooperative cancellation, shared with the run's `Scheduler`/entry point.
    pub cancel: CancelToken,
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
                self.errors.record(e, &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }
}

impl VqcOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        // Bind each candidate eagerly to the template (native binding is GIL-free;
        // Qiskit re-acquires the GIL internally). The concurrency/memory cap is the
        // Planner's job (`max_concurrency`), not this loop.
        let bound: Vec<BoundCircuit> = candidates
            .iter()
            .map(|params| self.circuit.bind(params))
            .collect::<Result<_, _>>()?;

        // One 2D task per candidate — uniform shots in training. `shots` comes
        // from the task, the single source of truth the Planner reads.
        let tasks: Vec<CircuitTask> = bound
            .iter()
            .map(|circuit| CircuitTask {
                circuit,
                shots: self.config.shots,
            })
            .collect();

        // Delegate execution + reduction to the Planner: it owns the waves, the
        // concurrency cap, the between-wave `check_signals` and the shot merge.
        // This dissolves the former per-chunk `run_and_evaluate` loop.
        let values = self.planner.evaluate(
            self.backend.as_ref(),
            &tasks,
            self.observable.as_ref(),
            &self.config,
            &self.cancel,
        )?;

        // Contract C-5 stays in the oracle (its contract with the optimizer):
        // exactly one finite value per candidate. A short/long batch would index
        // out of bounds inside the pure-Rust optimizer; a NaN/inf would silently
        // poison it. Reported as a `Result`, never a panic (this runs under
        // `OracleErrorSlot`).
        if values.len() != candidates.len() {
            return Err(EvaluationError::WrongLength {
                expected: candidates.len(),
                got: values.len(),
            });
        }
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{BackendConfig, BackendError, OptLevel};
    use polypus_circuit::{GateParam, ParameterizedCircuit};
    use polypus_observable::ObservableError;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Native test double for the reduction step, replacing the former Python
    /// `expectation_values` seam: it returns the synthetic count the mock backend
    /// encodes under key "1" as the expectation, so a mis-ordered result vector is
    /// still detectable without touching Python.
    struct KeyOneObservable;

    impl CostObservable for KeyOneObservable {
        fn expectation_batch(
            &self,
            counts: &[HashMap<String, u64>],
        ) -> Result<Vec<f64>, ObservableError> {
            Ok(counts.iter().map(|c| c["1"] as f64).collect())
        }
    }

    /// A [`QuantumBackend`] that records the OpenQASM 2.0 text of every circuit
    /// handed to each `run_circuits` call, so a test can assert *how* the oracle
    /// chunked and ordered the work.
    ///
    /// With `fail = true` the call errors out before returning any counts, which
    /// is also before `run_and_evaluate` reaches `Python::with_gil` — that is what
    /// lets the chunking/short-circuit tests below run against a bare interpreter.
    struct MockBackend {
        batch_size: usize,
        fail: bool,
        /// One entry per `run_circuits` call: the QASM of each circuit in it.
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl MockBackend {
        fn new(batch_size: usize, fail: bool) -> Self {
            Self {
                batch_size,
                fail,
                calls: Mutex::new(Vec::new()),
            }
        }

        /// The per-call chunk sizes observed so far.
        fn chunk_sizes(&self) -> Vec<usize> {
            self.locked_calls().iter().map(Vec::len).collect()
        }

        /// Every circuit seen, flattened across calls in submission order.
        fn submitted_qasm(&self) -> Vec<String> {
            self.locked_calls().iter().flatten().cloned().collect()
        }

        fn locked_calls(&self) -> std::sync::MutexGuard<'_, Vec<Vec<String>>> {
            self.calls.lock().unwrap_or_else(|p| p.into_inner())
        }
    }

    impl QuantumBackend for MockBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &ExecutionConfig,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            let mut calls = self.locked_calls();
            // Index of the first circuit of this chunk within the whole batch;
            // it is what the synthetic counts encode, so a mis-ordered result
            // vector is detectable.
            let offset: usize = calls.iter().map(Vec::len).sum();
            calls.push(qcs.iter().map(qasm_of).collect());
            drop(calls);

            if self.fail {
                return Err(BackendError::Conversion("mock failure".to_string()));
            }
            // Encode each candidate's batch position in the "1" count (read back by
            // `KeyOneObservable`) and park the rest of the shots under "0", so the
            // synthetic result still conserves `config.shots` (contract C-3, now
            // validated centrally in `run_and_evaluate`). `offset + i` stays well
            // under the 16 shots for these 5 candidates.
            Ok((0..qcs.len())
                .map(|i| {
                    let ones = (offset + i) as u64;
                    HashMap::from([
                        ("1".to_string(), ones),
                        ("0".to_string(), u64::from(config.shots) - ones),
                    ])
                })
                .collect())
        }

        fn max_batch_size(&self, _total: usize) -> usize {
            self.batch_size
        }

        fn capabilities(&self) -> crate::infrastructure::BackendCapabilities {
            // The Planner waves at this size, reproducing the old chunking that
            // these tests assert on.
            crate::infrastructure::BackendCapabilities {
                max_concurrency: self.batch_size,
                supports_shot_distribution: true,
            }
        }
    }

    /// The tests only ever bind native templates, so every circuit the mock sees
    /// is `Native`; the other arms exist to keep the match exhaustive.
    fn qasm_of(circuit: &BoundCircuit) -> String {
        match circuit {
            BoundCircuit::Native(cc) => cc.to_qasm2(),
            BoundCircuit::Qasm2(qasm) => qasm.clone(),
            BoundCircuit::Qiskit(_) => panic!("the mock never receives a Qiskit circuit"),
        }
    }

    /// One-parameter template, so each candidate binds to a distinguishable
    /// circuit (a different `ry` angle in the emitted QASM).
    fn template() -> CircuitSource {
        CircuitSource::Native(
            ParameterizedCircuit::new(1)
                .ry(0, GateParam::Param(0))
                .measure_all(),
        )
    }

    fn config() -> Arc<ExecutionConfig> {
        Arc::new(ExecutionConfig {
            id: "vqc-oracle-test".to_string(),
            shots: 16,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        })
    }

    /// Five one-dimensional candidates with distinct angles.
    fn candidates() -> Vec<Vec<f64>> {
        (0..5).map(|i| vec![0.1 * (i as f64 + 1.0)]).collect()
    }

    /// Build an oracle over `backend`, reducing counts with [`KeyOneObservable`]
    /// (the native stand-in for the former Python `expectation_values` seam, so
    /// the tests need no installed package and only a bare interpreter for
    /// `run_and_evaluate`'s signal check).
    fn oracle(backend: Arc<MockBackend>) -> VqcOracle {
        VqcOracle {
            circuit: template(),
            config: config(),
            backend,
            planner: Arc::new(crate::infrastructure::SequentialPlanner),
            observable: Arc::new(KeyOneObservable),
            cancel: crate::infrastructure::CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    #[test]
    fn first_chunk_is_sized_by_max_batch_size_and_a_failure_short_circuits() {
        pyo3::prepare_freethreaded_python();
        let backend = Arc::new(MockBackend::new(3, true));
        let oracle = oracle(Arc::clone(&backend));

        let values = oracle.evaluate_batch(&candidates());

        assert_eq!(
            values,
            vec![0.0; 5],
            "a failed evaluation must yield one finite sentinel per candidate (contract C-5)"
        );
        assert!(
            oracle.errors.failed(),
            "the backend failure must be recorded in the shared slot"
        );
        assert_eq!(
            backend.chunk_sizes(),
            vec![3],
            "the first chunk must be `max_batch_size` circuits, and the `?` must \
             short-circuit the chunk loop instead of submitting the remainder"
        );
    }

    #[test]
    fn a_recorded_failure_short_circuits_later_batches_without_touching_the_backend() {
        pyo3::prepare_freethreaded_python();
        let backend = Arc::new(MockBackend::new(3, true));
        let oracle = oracle(Arc::clone(&backend));

        let first = oracle.evaluate_batch(&candidates());
        let second = oracle.evaluate_batch(&candidates());

        assert_eq!(first, vec![0.0; 5]);
        assert_eq!(second, vec![0.0; 5]);
        assert_eq!(
            backend.chunk_sizes().len(),
            1,
            "once a failure is recorded, `evaluate_batch` must return sentinels \
             without calling the backend again"
        );
    }

    #[test]
    fn multiple_successful_chunks_preserve_candidate_order() {
        // `run_and_evaluate` still touches the GIL once for its signal check, so
        // the interpreter must be initialised even though the reduction is native.
        pyo3::prepare_freethreaded_python();

        let backend = Arc::new(MockBackend::new(3, false));
        let oracle = oracle(Arc::clone(&backend));
        let candidates = candidates();

        let values = oracle.evaluate_batch(&candidates);

        assert!(
            !oracle.errors.failed(),
            "a fully successful evaluation must record no error"
        );
        // The mock encodes each circuit's position in the whole batch as its
        // count and the shim turns that into the expectation value, so this is
        // exactly "candidate i's result landed at index i".
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            backend.chunk_sizes(),
            vec![3, 2],
            "5 candidates at max_batch_size 3 must be submitted as 3 + 2"
        );
        // …and the circuits themselves reached the backend in candidate order.
        let expected: Vec<String> = candidates
            .iter()
            .map(|params| qasm_of(&template().bind(params).expect("binding succeeds")))
            .collect();
        assert_eq!(backend.submitted_qasm(), expected);
    }
}
