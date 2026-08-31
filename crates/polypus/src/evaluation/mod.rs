pub mod error;
pub mod qml_oracle;
pub mod vqc_oracle;

pub use error::EvaluationError;
pub use qml_oracle::QmlOracle;
pub use vqc_oracle::VqcOracle;

use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::ParameterizedCircuit;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use std::sync::{Arc, Mutex};

/// Thread-safe holder for the first error an oracle hits during `optimize`.
///
/// The optimizer traits ([`EvaluationOracle`] and
/// [`VarianceOracle`](polypus_optimizers::VarianceOracle)) return plain
/// `f64`/`Vec<f64>` — a pure-crate contract this crate cannot change — so a
/// Python-side failure mid-optimization cannot be returned through the trait.
/// Instead the oracle records it here and yields a finite sentinel; the entry
/// point inspects the slot after `optimize` returns and surfaces the error as a
/// `PyErr` (contract C-5 keeps oracle outputs finite regardless).
#[derive(Clone, Default)]
pub struct OracleErrorSlot(Arc<Mutex<Option<EvaluationError>>>);

impl OracleErrorSlot {
    /// A fresh, empty slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `err` as the failure, keeping the *first* one recorded.
    ///
    /// `run_id` is the effective [`ExecutionConfig::id`] of the run whose oracle
    /// failed, threaded in from the call site because this slot holds no run
    /// metadata of its own. The failure is logged at `error!` here, as it is
    /// recorded: from this point on the oracle only yields sentinel values, so
    /// without this record the log would simply go quiet until `optimize()`
    /// returns and the entry point raises.
    pub fn record(&self, err: EvaluationError, run_id: &str) {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        // Log only when this call actually stores the error, so the message never
        // claims to have "recorded" a failure it silently dropped.
        if guard.is_none() {
            // Formatting the `Python(PyErr)` variant reacquires the GIL through
            // `PyErr`'s own `Display`, and this can run with the GIL released
            // (the optimizers run inside `allow_threads`). Safe either way:
            // `Python::with_gil` is re-entrant — the same guarantee `cunqa.rs`
            // relies on to acquire the GIL from within a `Drop`.
            log::error!("run {run_id}: oracle evaluation failed: {err}");
            *guard = Some(err);
        }
    }

    /// Whether a failure has been recorded (lets callers short-circuit further
    /// work once evaluation is doomed).
    pub fn failed(&self) -> bool {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    /// Take the recorded failure, if any.
    pub fn take(&self) -> Option<EvaluationError> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// A parameterised circuit template, in one of the representations Polypus
/// supports as optimisation targets.
///
/// The variant determines *where* per-candidate parameter binding happens:
///
/// - [`Qiskit`](CircuitSource::Qiskit): `assign_parameters` is called on the
///   Python object — requires the GIL for every candidate.
/// - [`Native`](CircuitSource::Native): binding + OpenQASM 2.0 generation run
///   in pure Rust — **no GIL**, so candidates can be bound without holding the
///   interpreter lock (binding itself is still sequential today, not
///   parallel — see `VqcOracle::try_evaluate`) and the only remaining Python
///   touchpoint is the simulator call itself.
#[derive(Debug)]
pub enum CircuitSource {
    /// A Qiskit `QuantumCircuit` with unbound `Parameter`s.
    Qiskit(Py<PyAny>),
    /// A native Rust circuit from `polypus-circuit`.
    Native(ParameterizedCircuit),
}

impl CircuitSource {
    /// Bind one candidate parameter vector, producing an executable circuit.
    ///
    /// Returns an [`EvaluationError`] on failure (wrong parameter count, a
    /// Python error binding a Qiskit circuit) rather than panicking, so the
    /// failure can cross the FFI as a typed exception. Entry points still
    /// validate `dimensions` up front, so a failure here is normally
    /// unreachable — but it is reported, never a panic.
    pub fn bind(&self, params: &[f64]) -> Result<BoundCircuit, EvaluationError> {
        match self {
            CircuitSource::Qiskit(circuit) => Ok(BoundCircuit::Qiskit(assign_parameters_qiskit(
                circuit, params,
            )?)),
            // Pure Rust: no GIL anywhere on this path. The bound circuit keeps
            // its native structure so the statevector backend can simulate it
            // directly; Python backends serialise it to OpenQASM 2.0 on demand.
            CircuitSource::Native(circuit) => Ok(BoundCircuit::Native(
                circuit
                    .assign_parameters(params)
                    .map_err(EvaluationError::Binding)?,
            )),
        }
    }

    /// Number of free parameters, when it can be known without Python
    /// (`None` for Qiskit circuits — querying them needs the GIL and is done
    /// at the entry points instead).
    pub fn num_params(&self) -> Option<usize> {
        match self {
            CircuitSource::Qiskit(_) => None,
            CircuitSource::Native(c) => Some(c.num_params),
        }
    }
}

/// Bind `params` to a copy of a Qiskit `circuit` and return the bound circuit.
///
/// Any Python error (constructing the kwargs, calling `assign_parameters`) is
/// returned as [`EvaluationError::Python`] — carried verbatim so the caller can
/// re-raise it with its original type across the FFI.
pub(crate) fn assign_parameters_qiskit(
    circuit: &Py<PyAny>,
    params: &[f64],
) -> Result<Py<PyAny>, EvaluationError> {
    Python::with_gil(|py| {
        let qc = circuit
            .clone_ref(py)
            .into_pyobject(py)
            .map_err(|e| EvaluationError::Python(e.into()))?;
        let kwargs = [("inplace", false)]
            .into_py_dict(py)
            .map_err(EvaluationError::Python)?;
        Ok(qc
            .call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .map_err(EvaluationError::Python)?
            .unbind())
    })
}

/// Contract between optimization algorithms and quantum circuit evaluation.
///
/// Re-exported from the pure-Rust [`polypus_optimizers`] crate, where the trait
/// now lives (it is the optimizers' input contract). Re-exporting here keeps the
/// `crate::evaluation::EvaluationOracle` path — used by [`VqcOracle`] and
/// [`QmlOracle`] — resolving unchanged.
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
pub use polypus_optimizers::EvaluationOracle;

/// Execute a batch of bound circuits through `backend` and extract expectation
/// values using the Python `expectation_fn`.
///
/// This is the **single place** in the codebase that calls
/// `polypus_python.expectation_values`, eliminating the duplication that
/// previously existed across DE, PSO, QNG, and the orchestration layer.
///
/// Returns an [`EvaluationError`] on any failure: a backend error is wrapped;
/// a raised Python exception (import, a pending `KeyboardInterrupt`, or one
/// thrown by `expectation_values` / the user callback) is carried verbatim; and
/// a data-conversion failure across the Rust↔Python boundary — the native
/// backend results failing to convert into a Python `list[dict]`, or a
/// wrong-shaped `expectation_values` return value, neither of which is a raised
/// exception — becomes [`EvaluationError::Conversion`]. Never a panic.
pub(crate) fn run_and_evaluate(
    backend: &dyn QuantumBackend,
    qcs: &[BoundCircuit],
    config: &ExecutionConfig,
    expectation_fn: &Py<PyAny>,
) -> Result<Vec<f64>, EvaluationError> {
    let counts = backend.run_circuits(qcs, config)?;
    Python::with_gil(|py| {
        // Turn a pending SIGINT (Ctrl+C) into a `KeyboardInterrupt` at this safe
        // per-batch boundary. The optimizer entry points release the GIL around
        // `optimize()`, which lets other Python threads run but does NOT by
        // itself process signals: CPython only acts on a pending signal while
        // the main thread runs Python bytecode or when `PyErr_CheckSignals` is
        // called explicitly. This is that explicit call, so a long native-backend
        // run stays interruptible instead of ignoring Ctrl+C until it finishes
        // (see docs/ENGINEERING.md §3). The KeyboardInterrupt is carried verbatim
        // via `EvaluationError::Python` and re-raised as itself by the entry point.
        py.check_signals().map_err(EvaluationError::Python)?;
        // Convert the native counts back into a Python `list[dict]` for the
        // Python `expectation_values` function. Once expectation computation is
        // also native this round-trip disappears entirely. `counts` is our own
        // Rust-native value, so a failure here is a Rust-side conversion problem
        // (realistically allocation failure), not a raised Python exception.
        let py_counts = counts.into_pyobject(py).map_err(|e| {
            EvaluationError::Conversion(format!(
                "failed to convert the native backend results into a Python list[dict]: {e}"
            ))
        })?;
        let values = PyModule::import(py, "polypus_python")
            .map_err(EvaluationError::Python)?
            .call_method("expectation_values", (py_counts, expectation_fn), None)
            .map_err(EvaluationError::Python)?
            // `expectation_values` returned successfully; a wrong-shaped value
            // is a Rust-side conversion failure, not a raised Python exception.
            .extract::<Vec<f64>>()
            .map_err(|e| {
                EvaluationError::Conversion(format!(
                    "expected expectation_values() to return list[float]: {e}"
                ))
            })?;

        // Contract C-5: the Python-backed oracle must return exactly one finite
        // f64 per submitted circuit. This is the single choke point that calls
        // `polypus_python.expectation_values`, so validating here protects every
        // oracle: a short list would otherwise index out of bounds inside the
        // pure-Rust optimizer (an uncatchable `PanicException` across the FFI),
        // and a NaN/inf would silently poison the optimizer and yield a bogus
        // result with no error at all.
        if values.len() != qcs.len() {
            return Err(EvaluationError::WrongLength {
                expected: qcs.len(),
                got: values.len(),
            });
        }
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(values)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // `OracleErrorSlot` is plain `Arc<Mutex<Option<EvaluationError>>>` wrapping:
    // none of it touches `Py<...>`/`PyErr`, so these tests need no interpreter
    // (ENGINEERING.md §3). `WrongLength`/`NonFinite` are the two variants that
    // can be constructed without Python, and their `Display` is distinguishable,
    // which is what lets the first-error-wins assertions below tell them apart.

    fn wrong_length() -> EvaluationError {
        EvaluationError::WrongLength {
            expected: 4,
            got: 2,
        }
    }

    fn non_finite() -> EvaluationError {
        EvaluationError::NonFinite {
            index: 7,
            value: f64::NAN,
        }
    }

    /// Run id threaded into `record`. It only names the run in the `error!` line
    /// the slot emits as it stores the failure (#88); the slot itself keeps no
    /// run metadata, so it never affects what is stored or returned.
    const RUN_ID: &str = "oracle-error-slot-test";

    #[test]
    fn new_slot_is_empty() {
        let slot = OracleErrorSlot::new();
        assert!(!slot.failed(), "a fresh slot must not report a failure");
        assert!(slot.take().is_none(), "a fresh slot must hold no error");
    }

    #[test]
    fn default_slot_is_empty() {
        let slot = OracleErrorSlot::default();
        assert!(!slot.failed());
        assert!(slot.take().is_none());
    }

    #[test]
    fn record_marks_the_slot_as_failed() {
        let slot = OracleErrorSlot::new();
        slot.record(wrong_length(), RUN_ID);
        assert!(slot.failed(), "record() must make failed() true");
    }

    #[test]
    fn record_keeps_the_first_error() {
        let slot = OracleErrorSlot::new();
        slot.record(wrong_length(), RUN_ID);
        slot.record(non_finite(), RUN_ID);
        let kept = slot.take().expect("an error was recorded");
        assert!(
            matches!(kept, EvaluationError::WrongLength { .. }),
            "the first recorded error must win, got: {kept}"
        );
    }

    #[test]
    fn take_returns_the_error_and_clears_the_slot() {
        let slot = OracleErrorSlot::new();
        slot.record(non_finite(), RUN_ID);
        let taken = slot.take().expect("an error was recorded");
        assert!(matches!(taken, EvaluationError::NonFinite { .. }));
        assert!(!slot.failed(), "take() must clear the slot");
        assert!(slot.take().is_none(), "a second take() must yield None");
    }

    #[test]
    fn clone_shares_the_same_slot() {
        // The QML oracle hands a clone to each worker thread; they must all see
        // (and write to) the same underlying slot.
        let slot = OracleErrorSlot::new();
        let handed_out = slot.clone();
        handed_out.record(wrong_length(), RUN_ID);
        assert!(slot.failed(), "a clone must share the original's storage");
    }

    #[test]
    fn concurrent_records_keep_exactly_one_error() {
        // This type exists to be shared across the QML oracle's worker threads
        // (see `qml_oracle.rs`), so the first-error-wins property must hold
        // under contention, not just sequentially.
        let slot = OracleErrorSlot::new();
        let start = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let slot = slot.clone();
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    slot.record(
                        EvaluationError::NonFinite {
                            index: i,
                            value: f64::INFINITY,
                        },
                        RUN_ID,
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no worker may panic");
        }

        assert!(slot.failed(), "at least one writer must have recorded");
        let first = slot.take().expect("exactly one error survives");
        assert!(matches!(first, EvaluationError::NonFinite { .. }));
        assert!(
            slot.take().is_none(),
            "only one error is ever stored, however many writers raced"
        );
    }
}
