//! Error type for the optimizer-oracle / expectation-evaluation path.
//!
//! See [`crate::infrastructure::error`] for the crate-wide granularity
//! decision. This enum wraps a [`BackendError`] (the underlying execution
//! failure), a [`CircuitError`] (native parameter binding) or a raw [`PyErr`]
//! (a Python callback/conversion), all reachable while an optimizer drives an
//! oracle across the FFI.

use std::fmt;

use polypus_circuit::CircuitError;
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
    /// A Python callback or conversion on the evaluation path raised. Carried
    /// verbatim so the original exception type is preserved across the FFI.
    Python(PyErr),
    /// A Rust-originated infrastructure failure on the QML evaluation path
    /// (Tokio runtime construction, or a worker task panic surfaced as a
    /// `JoinError`). Never a Python exception, so unlike `Python` it must not be
    /// re-raised verbatim.
    Runtime(String),
    /// Converting data across the Rust↔Python boundary on the evaluation path
    /// failed (e.g. `expectation_values`'s return value isn't `list[float]`).
    /// Unlike `Python`, this never originated in a raised Python exception, so
    /// it must not be re-raised verbatim.
    Conversion(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::Runtime(m) => write!(f, "QML evaluation runtime error: {m}"),
            EvaluationError::Conversion(m) => {
                write!(f, "data conversion across the Python boundary failed: {m}")
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<BackendError> for EvaluationError {
    fn from(err: BackendError) -> Self {
        EvaluationError::Backend(err)
    }
}

impl From<EvaluationError> for PyErr {
    fn from(err: EvaluationError) -> PyErr {
        match err {
            EvaluationError::Backend(backend_err) => backend_err.into(),
            EvaluationError::Binding(circuit_err) => {
                PyEvaluationError::new_err(circuit_err.to_string())
            }
            // Preserve the original Python exception type raised by the callback.
            EvaluationError::Python(py_err) => py_err,
            // A Rust-side infrastructure failure: surface as the typed
            // polypus.EvaluationError, not PyO3's generic RuntimeError.
            EvaluationError::Runtime(m) => PyEvaluationError::new_err(m),
            // A Rust-side data-conversion failure: surface as the typed
            // polypus.EvaluationError, not the TypeError PyO3's extract() emits.
            EvaluationError::Conversion(m) => PyEvaluationError::new_err(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyRuntimeError, PyTypeError};
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;

    /// A QML infrastructure failure (Tokio runtime construction or a
    /// `spawn_blocking` worker panic surfaced as a `JoinError`) is modelled by
    /// [`EvaluationError::Runtime`]. Forcing either condition deterministically
    /// from a test is neither viable nor portable — OS resource exhaustion for
    /// the runtime, and `evaluate_qml_single` is deliberately written not to
    /// panic — so instead we pin the *mapping*: `Runtime` must cross the FFI as
    /// the typed `polypus.EvaluationError`, never PyO3's generic
    /// `RuntimeError`. (Scope decision documented in the PR for issue #81.)
    #[test]
    fn runtime_variant_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err: PyErr = EvaluationError::Runtime("worker panicked".to_string()).into();
            assert!(
                err.value(py).is_instance_of::<PyEvaluationError>(),
                "Runtime must surface as polypus.EvaluationError"
            );
            // ...and specifically not PyO3's generic RuntimeError, which is what
            // the pre-fix code raised for this Rust-side infrastructure failure.
            assert!(
                !err.value(py).is_instance_of::<PyRuntimeError>(),
                "Runtime must not surface as the generic RuntimeError"
            );
            assert!(
                err.to_string().contains("worker panicked"),
                "the descriptive message must be preserved"
            );
        });
    }

    /// A wrong-shaped `expectation_values` return value is modelled by
    /// [`EvaluationError::Conversion`]: a Rust-side data-conversion failure, not
    /// a raised Python exception. It must cross the FFI as the typed
    /// `polypus.EvaluationError`, never as the generic `TypeError` that PyO3's
    /// `extract()` would otherwise emit. (End-to-end coverage lives in the
    /// Python suite; this pins the mapping in isolation.)
    #[test]
    fn conversion_variant_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err: PyErr = EvaluationError::Conversion("not list[float]".to_string()).into();
            assert!(
                err.value(py).is_instance_of::<PyEvaluationError>(),
                "Conversion must surface as polypus.EvaluationError"
            );
            // ...and specifically not the plain TypeError that extract() emits,
            // which is what the pre-fix code let through verbatim.
            assert!(
                !err.value(py).is_instance_of::<PyTypeError>(),
                "Conversion must not surface as the generic TypeError"
            );
            assert!(
                err.to_string().contains("not list[float]"),
                "the descriptive message must be preserved"
            );
        });
    }
}
