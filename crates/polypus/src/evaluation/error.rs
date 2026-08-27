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

    // The `Display` tests below deliberately construct no `PyErr` at all, so
    // they run with no interpreter whatsoever; the `PyErr`-mapping tests further
    // down need `prepare_freethreaded_python()` but still no installed package
    // (see the note above them).

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

    // The mapping tests below do construct a `PyErr`, which the `Display` tests
    // above deliberately avoid. That is still Python-runtime-free in the sense
    // ENGINEERING.md §3 means it: `prepare_freethreaded_python()` +
    // `is_instance_of` need a bare CPython interpreter and no installed package
    // (neither Qiskit nor `polypus_python`), exactly what CI provides — the same
    // thing `crates/polypus/tests/running_quantum_circuits_local.rs` already does.

    /// Assert `err` crosses the FFI as `polypus.EvaluationError` and keeps its
    /// message.
    fn assert_maps_to_evaluation_error(err: EvaluationError, expected_message: &str) {
        pyo3::prepare_freethreaded_python();
        let py_err: PyErr = err.into();
        pyo3::Python::with_gil(|py| {
            assert!(
                py_err.is_instance_of::<PyEvaluationError>(py),
                "wrong exception class for: {py_err}"
            );
            assert!(
                py_err.is_instance_of::<crate::exceptions::PolypusError>(py),
                "EvaluationError must stay catchable as PolypusError: {py_err}"
            );
            assert!(
                py_err.to_string().contains(expected_message),
                "message lost in translation: {py_err}"
            );
        });
    }

    #[test]
    fn binding_maps_to_evaluation_error() {
        // Normally unreachable: the entry points validate `dimensions` against
        // the circuit's free-parameter count before any candidate is bound (see
        // `CircuitSource::bind`). Constructed directly so the "unlikely" path is
        // still proven to be a typed exception rather than a panic (§9).
        assert_maps_to_evaluation_error(
            EvaluationError::Binding(CircuitError::WrongNumberOfParams {
                expected: 3,
                got: 1,
            }),
            "circuit declares 3 free parameter(s) but 1 value(s) were provided",
        );
    }

    #[test]
    fn wrong_length_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvaluationError::WrongLength {
                expected: 4,
                got: 2,
            },
            "contract C-5",
        );
    }

    #[test]
    fn non_finite_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvaluationError::NonFinite {
                index: 3,
                value: f64::NAN,
            },
            "contract C-5",
        );
    }

    #[test]
    fn backend_variant_delegates_to_the_backend_mapping() {
        // `EvaluationError::Backend` must not retype the wrapped failure: a
        // CUNQA error surfacing through an oracle is still a `polypus.CunqaError`.
        pyo3::prepare_freethreaded_python();
        let py_err: PyErr =
            EvaluationError::Backend(BackendError::Cunqa("qraise failed".to_string())).into();
        pyo3::Python::with_gil(|py| {
            assert!(py_err.is_instance_of::<crate::exceptions::CunqaError>(py));
            assert!(
                !py_err.is_instance_of::<PyEvaluationError>(py),
                "a wrapped backend failure must keep its own class"
            );
        });
    }
}
