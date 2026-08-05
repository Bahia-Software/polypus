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
    /// The Python-backed oracle returned a different number of expectation
    /// values than circuits were submitted in this call (contract C-5).
    WrongLength { expected: usize, got: usize },
    /// The Python-backed oracle returned a non-finite expectation value
    /// (contract C-5 requires every output to be a finite f64).
    NonFinite { index: usize, value: f64 },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::WrongLength { expected, got } => write!(
                f,
                "oracle returned the wrong number of expectation values: expected {expected} (one per submitted circuit) but got {got} (contract C-5)"
            ),
            EvaluationError::NonFinite { index, value } => write!(
                f,
                "oracle returned a non-finite expectation value {value} at index {index}; contract C-5 requires every output to be a finite f64"
            ),
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
            wrong_length @ EvaluationError::WrongLength { .. } => {
                PyEvaluationError::new_err(wrong_length.to_string())
            }
            non_finite @ EvaluationError::NonFinite { .. } => {
                PyEvaluationError::new_err(non_finite.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests are deliberately Python-runtime-free (ENGINEERING.md §3): the
    // `polypus` crate's test suite runs without an initialized interpreter, so
    // we exercise `Display` only and never construct a `PyErr` / call `.into()`.

    #[test]
    fn wrong_length_display_names_both_lengths() {
        let msg = EvaluationError::WrongLength {
            expected: 4,
            got: 2,
        }
        .to_string();
        assert!(msg.contains('4'), "expected length missing from: {msg}");
        assert!(msg.contains('2'), "got length missing from: {msg}");
    }

    #[test]
    fn non_finite_display_names_index_and_value() {
        let msg = EvaluationError::NonFinite {
            index: 3,
            value: f64::NAN,
        }
        .to_string();
        assert!(msg.contains('3'), "offending index missing from: {msg}");
        assert!(msg.contains("NaN"), "offending value missing from: {msg}");
    }
}
