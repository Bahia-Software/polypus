//! Error type for the optimizer-oracle / expectation-evaluation path.
//!
//! See [`polypus_infrastructure::error`] for the crate-wide granularity
//! decision. This enum wraps a [`BackendError`] (the underlying execution
//! failure), a [`CircuitError`] (native parameter binding) or a raw [`PyErr`]
//! (a Python callback/conversion), all reachable while an optimizer drives an
//! oracle across the FFI.

use std::fmt;

use polypus_circuit::CircuitError;
use polypus_observable::ObservableError;
use pyo3::PyErr;

use polypus_infrastructure::{BackendError, InfrastructureError};

/// A failure encountered while evaluating a candidate parameter vector.
///
/// The optimizer traits ([`EvaluationOracle`](polypus_optimizers::EvaluationOracle),
/// [`VarianceOracle`](polypus_optimizers::VarianceOracle)) return plain
/// `f64`/`Vec<f64>` and cannot carry a `Result` across the FFI, so an oracle
/// records its first failure of this type in an
/// [`OracleErrorSlot`](crate::OracleErrorSlot) and the entry point
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
    /// The Python-backed oracle returned a different number of expectation
    /// values than circuits were submitted in this call (contract C-5).
    WrongLength { expected: usize, got: usize },
    /// The Python-backed oracle returned a non-finite expectation value
    /// (contract C-5 requires every output to be a finite f64).
    NonFinite { index: usize, value: f64 },
    /// The Python `variance_function` (QNG) returned an invalid QFIM diagonal
    /// element: a `NaN`/infinite value or a negative one. A variance must be a
    /// finite, non-negative number — zero is allowed (Tikhonov regularisation
    /// keeps the QNG division well-posed). Rejected at the callback boundary so a
    /// bad value cannot silently corrupt the natural-gradient update.
    InvalidVariance {
        /// The parameter index the callback was evaluated at.
        param_index: usize,
        /// The offending value returned by `variance_function`.
        value: f64,
    },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Observable(err) => write!(f, "expectation evaluation failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::Runtime(m) => write!(f, "QML evaluation runtime error: {m}"),
            EvaluationError::Conversion(m) => {
                write!(f, "data conversion across the Python boundary failed: {m}")
            }
            EvaluationError::WrongLength { expected, got } => write!(
                f,
                "oracle returned the wrong number of expectation values: expected {expected} (one per submitted circuit) but got {got} (contract C-5)"
            ),
            EvaluationError::NonFinite { index, value } => write!(
                f,
                "oracle returned a non-finite expectation value {value} at index {index}; contract C-5 requires every output to be a finite f64"
            ),
            EvaluationError::InvalidVariance { param_index, value } => write!(
                f,
                "variance_function returned an invalid value {value} for parameter index {param_index}; a QFIM diagonal element must be a finite, non-negative number"
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

impl From<ObservableError> for EvaluationError {
    fn from(err: ObservableError) -> Self {
        EvaluationError::Observable(err)
    }
}

impl From<InfrastructureError> for EvaluationError {
    fn from(err: InfrastructureError) -> Self {
        match err {
            // A backend failure (including a between-wave interrupt, which now
            // arrives as `Backend(BackendError::External(boxed PyErr))`) and an
            // observable failure map straight through, exactly as the former
            // `run_and_evaluate` returned them.
            InfrastructureError::Backend(e) => EvaluationError::Backend(e),
            InfrastructureError::Observable(e) => EvaluationError::Observable(e),
            // A cooperative cancel surfaces as a KeyboardInterrupt, the same class
            // a SIGINT would (unreachable while nothing sets the token).
            InfrastructureError::Cancelled => EvaluationError::Python(
                pyo3::exceptions::PyKeyboardInterrupt::new_err("the run was cancelled"),
            ),
            // A planner/backend mismatch is a construction-time check, not reached
            // through the oracle; surface it as the typed evaluation error.
            InfrastructureError::IncompatiblePlanner(m) => EvaluationError::Runtime(m),
        }
    }
}

// `EvaluationError` deliberately implements no `From<_> for PyErr`: mapping it to
// the typed `polypus.*` exception hierarchy is the `polypus` FFI edge's job
// (`polypus::exceptions::evaluation_error_to_pyerr`), which owns those
// `#[pyclass]` types. The `Python`/`Observable(External)` variants still carry a
// `PyErr` verbatim so the edge can re-raise the original exception unchanged.

#[cfg(test)]
mod tests {
    use super::*;

    // The FFI mapping of these variants to the typed `polypus.*` exception
    // classes now lives at the `polypus` edge (`exceptions::evaluation_error_to_pyerr`)
    // and is tested there; this crate owns only the pure-Rust `Display`, which is
    // what those Python messages are built from — so it is pinned here.

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
