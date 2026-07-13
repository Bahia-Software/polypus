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
    pub fn record(&self, err: EvaluationError) {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
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
///   in pure Rust — **no GIL**, so candidates can be bound truly in parallel
///   and the only remaining Python touchpoint is the simulator call itself.
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
/// Returns an [`EvaluationError`] on any failure: a backend error is wrapped,
/// and a Python error (import, `expectation_values`, extraction) is carried
/// verbatim — never a panic.
pub(crate) fn run_and_evaluate(
    backend: &dyn QuantumBackend,
    qcs: &[BoundCircuit],
    config: &ExecutionConfig,
    expectation_fn: &Py<PyAny>,
) -> Result<Vec<f64>, EvaluationError> {
    let counts = backend.run_circuits(qcs, config)?;
    Python::with_gil(|py| {
        // Convert the native counts back into a Python `list[dict]` for the
        // Python `expectation_values` function. Once expectation computation is
        // also native this round-trip disappears entirely.
        let py_counts = counts.into_pyobject(py).map_err(EvaluationError::Python)?;
        PyModule::import(py, "polypus_python")
            .map_err(EvaluationError::Python)?
            .call_method("expectation_values", (py_counts, expectation_fn), None)
            .map_err(EvaluationError::Python)?
            .extract::<Vec<f64>>()
            .map_err(EvaluationError::Python)
    })
}
