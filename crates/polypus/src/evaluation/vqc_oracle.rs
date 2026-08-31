use crate::evaluation::{
    run_and_evaluate, CircuitSource, EvaluationError, EvaluationOracle, OracleErrorSlot,
};
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

        // Defense-in-depth (contract C-5): `run_and_evaluate` already guarantees
        // exactly `chunk.len()` values per chunk, and the chunks partition
        // `bound` (== `candidates`) exactly, so this can only ever hold. It is
        // kept as an explicit, self-documenting invariant at the point where the
        // per-candidate results are finally assembled — do not "simplify" it away
        // on the assumption the centralized check is enough. As with that path,
        // report it as a `Result` rather than panicking (rule 4: FFI errors are
        // `PyErr`/`Result`, and this runs under `OracleErrorSlot`).
        if results.len() != candidates.len() {
            return Err(EvaluationError::WrongLength {
                expected: candidates.len(),
                got: results.len(),
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::{BackendConfig, BackendError, OptLevel};
    use polypus_circuit::{GateParam, ParameterizedCircuit};
    use pyo3::types::PyModule;
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::sync::Mutex;

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
            _config: &ExecutionConfig,
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
            Ok((0..qcs.len())
                .map(|i| HashMap::from([("1".to_string(), (offset + i) as u64)]))
                .collect())
        }

        fn max_batch_size(&self, _total: usize) -> usize {
            self.batch_size
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

    /// Build an oracle over `backend`. `expectation_fn` is `None` — the fake seam
    /// installed by the success test ignores it, and the failing-backend tests
    /// never get far enough to call it — but constructing the `Py<PyAny>` still
    /// needs an initialised interpreter (bare CPython, no packages).
    fn oracle(backend: Arc<MockBackend>) -> VqcOracle {
        let expectation_fn = Python::with_gil(|py| py.None());
        VqcOracle {
            circuit: template(),
            config: config(),
            backend,
            expectation_fn,
            errors: OracleErrorSlot::new(),
        }
    }

    /// Register a trivial, Qiskit-free stand-in for the `polypus_python` seam in
    /// `sys.modules`.
    ///
    /// `run_and_evaluate` imports `polypus_python` as soon as a chunk's backend
    /// call succeeds, but that pip package imports Qiskit at module load and the
    /// Rust suite runs against a bare interpreter by design (ENGINEERING.md §3).
    /// Injecting the shim into `sys.modules` makes `PyModule::import` resolve it
    /// there without ever touching `sys.path`, so no package has to be installed.
    fn install_fake_seam(py: Python<'_>) -> PyResult<()> {
        let source = CString::new(
            "def expectation_values(counts, fn):\n    return [float(c[\"1\"]) for c in counts]\n",
        )
        .expect("shim source has no interior NUL");
        let file_name = CString::new("polypus_python.py").expect("no interior NUL");
        let module_name = CString::new("polypus_python").expect("no interior NUL");
        let module = PyModule::from_code(py, &source, &file_name, &module_name)?;
        py.import("sys")?
            .getattr("modules")?
            .set_item("polypus_python", module)
    }

    /// Remove the shim again so it cannot leak into any other test in this
    /// binary (nothing else in the Rust suite imports `polypus_python`, but a
    /// process-global `sys.modules` entry is exactly the kind of cross-test
    /// coupling that makes execution order matter).
    fn remove_fake_seam(py: Python<'_>) -> PyResult<()> {
        py.import("sys")?
            .getattr("modules")?
            .del_item("polypus_python")
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| install_fake_seam(py).expect("the shim installs"));

        let backend = Arc::new(MockBackend::new(3, false));
        let oracle = oracle(Arc::clone(&backend));
        let candidates = candidates();

        let values = oracle.evaluate_batch(&candidates);

        // Uninstall before asserting so a failed assertion cannot leave the shim
        // behind for another test in this binary.
        Python::with_gil(|py| remove_fake_seam(py).expect("the shim uninstalls"));

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
