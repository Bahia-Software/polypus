//! Error type for the optimizer-oracle / expectation-evaluation path.
//!
//! See [`crate::infrastructure::error`] for the crate-wide granularity
//! decision. This enum wraps a [`BackendError`] (the underlying execution
//! failure), a [`CircuitError`] (native parameter binding), a raw [`PyErr`]
//! (a Python callback/conversion) or an [`OptimizerError`] raised by a helper of
//! `polypus-optimizers` called from inside an oracle, all reachable while an
//! optimizer drives an oracle across the FFI.

use std::fmt;

use polypus_circuit::CircuitError;
use polypus_optimizers::OptimizerError;
use polypus_qml::{QmlError, ValidationError};
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
    /// A native QML problem failed to bind a candidate into circuits or to turn
    /// measurement counts into a fitness (contract C-8). Reached only on the
    /// [`NativeQmlOracle`](crate::evaluation::NativeQmlOracle) path.
    Qml(QmlError),
    /// A derived native QML problem failed its own construction validation —
    /// today only carving a minibatch out of the full problem
    /// (`QmlProblem::subset`), which rejects an empty index set rather than
    /// building a zero-sample problem whose mean fitness would be `NaN`
    /// (contract C-8). Reached only on the minibatch path of the two native QML
    /// oracles.
    Validation(ValidationError),
    /// A `polypus-optimizers` helper called from *inside* an oracle failed its
    /// own contract check — today only
    /// [`linear_parameter_shift_gradient`](polypus_optimizers::linear_parameter_shift_gradient),
    /// which length-checks what the oracle's `evaluate_batch` handed back
    /// (contract C-5) instead of indexing it blindly. The
    /// [`GradientOracle`](polypus_optimizers::GradientOracle) trait method that
    /// calls it returns a plain `Vec<f64>`, so this slot is the only route that
    /// failure has to Python. Note that the *same* [`OptimizerError`] returned by
    /// `Optimizer::optimize` itself surfaces as a `ValueError` at the bindings
    /// boundary, exactly as [`EvaluationError::Validation`] is an evaluation
    /// failure here but a `ValueError` when raised while *building* a problem:
    /// what a failure recorded mid-evaluation means to the caller is "the oracle
    /// broke", not "your argument was invalid".
    Optimizer(OptimizerError),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::Qml(err) => write!(f, "QML evaluation error: {err}"),
            EvaluationError::Validation(err) => write!(f, "QML validation error: {err}"),
            EvaluationError::Optimizer(err) => write!(f, "optimizer oracle error: {err}"),
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<BackendError> for EvaluationError {
    fn from(err: BackendError) -> Self {
        EvaluationError::Backend(err)
    }
}

impl From<QmlError> for EvaluationError {
    fn from(err: QmlError) -> Self {
        EvaluationError::Qml(err)
    }
}

impl From<ValidationError> for EvaluationError {
    fn from(err: ValidationError) -> Self {
        EvaluationError::Validation(err)
    }
}

impl From<OptimizerError> for EvaluationError {
    fn from(err: OptimizerError) -> Self {
        EvaluationError::Optimizer(err)
    }
}

impl From<EvaluationError> for PyErr {
    fn from(err: EvaluationError) -> PyErr {
        match err {
            EvaluationError::Backend(backend_err) => backend_err.into(),
            EvaluationError::Binding(circuit_err) => {
                PyEvaluationError::new_err(circuit_err.to_string())
            }
            // A native QML failure maps to the same evaluation exception as a
            // native binding failure — both are Rust-side evaluation errors, and
            // so is a derived problem failing its construction validation
            // mid-evaluation (unlike the same `ValidationError` raised while
            // *building* a problem at the bindings boundary, which is a
            // `ValueError` there).
            EvaluationError::Qml(qml_err) => PyEvaluationError::new_err(qml_err.to_string()),
            EvaluationError::Validation(validation_err) => {
                PyEvaluationError::new_err(validation_err.to_string())
            }
            // Same reasoning: an optimizer-helper contract violation detected
            // while an oracle was evaluating is an evaluation failure to the
            // caller, so it joins the others under the evaluation exception
            // rather than the `ValueError` the bindings raise for the identical
            // error coming out of `Optimizer::optimize`.
            EvaluationError::Optimizer(optimizer_err) => {
                PyEvaluationError::new_err(optimizer_err.to_string())
            }
            // Preserve the original Python exception type raised by the callback.
            EvaluationError::Python(py_err) => py_err,
        }
    }
}
