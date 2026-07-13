//! Error type for the orchestration / execution backend layer.
//!
//! # Granularity decision
//!
//! `crates/polypus` uses **two** hand-written error enums rather than one per
//! module: [`BackendError`] here (backend construction, circuit execution,
//! infrastructure selection, Rust↔Python conversion) and
//! [`EvaluationError`](crate::evaluation::EvaluationError) for the optimizer
//! oracle path. The feature-gated `QmioError` keeps its own rich enum (verified
//! against the wire protocol) and is *wrapped* by `BackendError::Qmio` instead
//! of being flattened. This mirrors the existing per-crate `error.rs` style
//! (`polypus-circuit`, `polypus-optimizers`) while keeping the number of types
//! the seam has to thread small.
//!
//! Every variant is a genuinely fallible interaction (Python call, IO, data
//! conversion), never a pure invariant — so replacing the previous
//! `unwrap()`/`expect()`/`panic!` sites with this `Result` is what lets the FFI
//! boundary map a failure to a `PyErr` instead of unwinding across it
//! (ENGINEERING.md §9).

use std::fmt;

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

use crate::exceptions::{
    BackendError as PyBackendError, CunqaError as PyCunqaError,
    NativeCircuitError as PyNativeCircuitError,
};

/// Failure of a quantum-execution backend or of backend construction.
///
/// Mirrors the hand-written style of
/// [`CircuitError`](polypus_circuit::CircuitError) and the crate's own
/// `QmioError`: no `thiserror`, a `match`-based [`fmt::Display`] and an empty
/// [`std::error::Error`] impl.
///
/// `Clone`/`Eq` are intentionally omitted: [`BackendError::Seam`] carries a
/// [`PyErr`], which is neither `Clone` nor `Eq`.
#[derive(Debug)]
pub enum BackendError {
    /// The requested infrastructure name is not recognised. Surfaces as
    /// `ValueError` to honour contract C-1 (unknown infrastructure).
    UnknownInfrastructure {
        /// The rejected infrastructure string.
        name: String,
    },
    /// A backend was asked to run a circuit representation it cannot execute
    /// (e.g. a Qiskit `QuantumCircuit` on a GIL-free backend).
    UnsupportedCircuit(String),
    /// A native (pure-Rust) circuit failed to parse or to simulate.
    NativeCircuit(String),
    /// A CUNQA-specific failure originating in the Rust layer (family-handle
    /// extraction, allocation bookkeeping).
    Cunqa(String),
    /// Converting data across the Rust↔Python boundary failed (our side of the
    /// call — building kwargs, converting counts, …).
    Conversion(String),
    /// A Python exception raised by the `polypus_python` execution seam
    /// (`connect_to_infrastructure` / `run_qcs` /
    /// `disconnect_from_infrastructure`).
    ///
    /// Carried verbatim so its original type is preserved when it crosses back
    /// into Python: contract C-1 requires an unknown infrastructure to be a
    /// `ValueError` and an unexpected/missing kwarg to be a `TypeError`, and
    /// both are raised on the Python side of the seam.
    Seam(PyErr),
    /// A failure on the QMIO network/serialisation path.
    #[cfg(feature = "qmio")]
    Qmio(super::qmio::QmioError),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnknownInfrastructure { name } => {
                write!(f, "unknown infrastructure '{name}'")
            }
            BackendError::UnsupportedCircuit(m) => write!(f, "{m}"),
            BackendError::NativeCircuit(m) => write!(f, "{m}"),
            BackendError::Cunqa(m) => write!(f, "CUNQA backend error: {m}"),
            BackendError::Conversion(m) => {
                write!(f, "data conversion across the Python boundary failed: {m}")
            }
            BackendError::Seam(err) => write!(f, "polypus_python seam error: {err}"),
            #[cfg(feature = "qmio")]
            BackendError::Qmio(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<BackendError> for PyErr {
    fn from(err: BackendError) -> PyErr {
        match err {
            // Re-raise the original Python exception unchanged so contract C-1's
            // documented ValueError/TypeError failure modes are preserved.
            BackendError::Seam(py_err) => py_err,
            BackendError::UnknownInfrastructure { name } => PyValueError::new_err(format!(
                "unknown infrastructure '{name}'; expected \"local\", \"cunqa\" or \"qmio\""
            )),
            BackendError::UnsupportedCircuit(m) => PyNativeCircuitError::new_err(m),
            BackendError::NativeCircuit(m) => PyNativeCircuitError::new_err(m),
            BackendError::Cunqa(m) => PyCunqaError::new_err(m),
            BackendError::Conversion(m) => PyBackendError::new_err(m),
            #[cfg(feature = "qmio")]
            BackendError::Qmio(qmio_err) => {
                crate::exceptions::QmioError::new_err(qmio_err.to_string())
            }
        }
    }
}
