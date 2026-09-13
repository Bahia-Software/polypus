//! # polypus-evaluation
//!
//! **How a candidate is evaluated** — the oracles that turn a parameter vector
//! into a scalar fitness. Holds [`VqcOracle`], [`QmlOracle`],
//! [`PyVarianceOracle`], the Python-callback observable [`PyCallbackObservable`],
//! the [`CircuitSource`] binding boundary and the evaluation error type.
//!
//! This crate touches `pyo3` (Qiskit binding under the GIL, Python callbacks) but
//! — like the rest of the workspace below the edge — defines no `#[pyclass]` and
//! no `From<_> for PyErr`: turning an [`EvaluationError`] into a typed `polypus.*`
//! exception is the `polypus` edge's job
//! (`polypus::exceptions::evaluation_error_to_pyerr`).

pub mod error;
pub mod py_callback_observable;
pub mod qml_oracle;
pub mod variance_oracle;
pub mod vqc_oracle;

pub use error::EvaluationError;
pub use py_callback_observable::PyCallbackObservable;
pub use qml_oracle::QmlOracle;
pub use variance_oracle::PyVarianceOracle;
pub use vqc_oracle::VqcOracle;

/// Re-export the native cost-observable seam so `crate::CostObservable`
/// resolves alongside the oracles that consume it.
pub use polypus_observable::CostObservable;

use polypus_circuit::ParameterizedCircuit;
use polypus_infrastructure::BoundCircuit;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

/// Re-export the type-erased error slot from the pure `polypus-scheduler` crate,
/// where it now lives — it is shared with `dispatch_optimizer`, and the scheduler
/// crate cannot depend on this pyo3-touching one. The oracles below box their
/// [`EvaluationError`] into it (`slot.record(Box::new(err), id)`); the FFI edge
/// downcasts it back to re-raise the original exception. Re-exported here so the
/// existing `crate::OracleErrorSlot` path keeps resolving.
pub use polypus_scheduler::OracleErrorSlot;

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
/// `crate::EvaluationOracle` path — used by [`VqcOracle`] and
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
