//! Error type for the optimizer-oracle / expectation-evaluation path.
//!
//! See [`crate::infrastructure::error`] for the crate-wide granularity
//! decision. This enum wraps a [`BackendError`] (the underlying execution
//! failure), a [`CircuitError`] (native parameter binding) or a raw [`PyErr`]
//! (a Python callback/conversion), all reachable while an optimizer drives an
//! oracle across the FFI.

use std::fmt;

use polypus_circuit::CircuitError;
use polypus_observable::ObservableError;
use pyo3::PyErr;

use crate::exceptions::EvaluationError as PyEvaluationError;
use crate::infrastructure::BackendError;

/// A failure encountered while evaluating a candidate parameter vector.
///
/// The optimizer traits ([`EvaluationOracle`](polypus_optimizers::EvaluationOracle),
/// [`VarianceOracle`](polypus_optimizers::VarianceOracle)) return plain
/// `f64`/`Vec<f64>` and cannot carry a `Result` across the FFI, so an oracle
/// records its first failure of this type in an
/// [`OracleErrorSlot`](crate::evaluation::OracleErrorSlot) and the entry point
/// surfaces it after `optimize` returns.
///
/// `Clone`/`Eq` are omitted: the [`EvaluationError::Python`] variant carries a
/// [`PyErr`].
#[derive(Debug)]
pub enum EvaluationError {
    /// The underlying execution backend failed.
    Backend(BackendError),
    /// Native parameter binding failed (wrong count, non-finite value, …).
    Binding(CircuitError),
    /// Native cost-observable evaluation failed (bad bitstring width/char, or a
    /// callback observable's error carried in [`ObservableError::External`]).
    Observable(ObservableError),
    /// A Python callback or conversion on the evaluation path raised. Carried
    /// verbatim so the original exception type is preserved across the FFI.
    Python(PyErr),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Observable(err) => write!(f, "expectation evaluation failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<BackendError> for EvaluationError {
    fn from(err: BackendError) -> Self {
        EvaluationError::Backend(err)
    }
}

impl From<ObservableError> for EvaluationError {
    fn from(err: ObservableError) -> Self {
        EvaluationError::Observable(err)
    }
}

impl From<EvaluationError> for PyErr {
    fn from(err: EvaluationError) -> PyErr {
        match err {
            EvaluationError::Backend(backend_err) => backend_err.into(),
            EvaluationError::Binding(circuit_err) => {
                PyEvaluationError::new_err(circuit_err.to_string())
            }
            EvaluationError::Observable(obs_err) => match obs_err {
                // A callback observable boxes its `PyErr` here; recover it so the
                // original Python exception type re-raises verbatim across the FFI.
                ObservableError::External(boxed) => match boxed.downcast::<PyErr>() {
                    Ok(py_err) => *py_err,
                    Err(other) => PyEvaluationError::new_err(other.to_string()),
                },
                // Native evaluation failures (bad bitstring, invalid construction)
                // map to the typed evaluation exception.
                other => PyEvaluationError::new_err(other.to_string()),
            },
            // Preserve the original Python exception type raised by the callback.
            EvaluationError::Python(py_err) => py_err,
        }
    }
}
