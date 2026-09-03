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
use polypus_observable::ObservableError;
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
    /// Native cost-observable evaluation failed (bad bitstring width/char, or a
    /// callback observable's error carried in [`ObservableError::External`]).
    Observable(ObservableError),
    /// A Python callback or conversion on the evaluation path raised. Carried
    /// verbatim so the original exception type is preserved across the FFI.
    Python(PyErr),
    /// A native QML problem failed to bind a candidate into circuits or to turn
    /// measurement counts into a fitness (contract C-10). Reached only on the
    /// [`NativeQmlOracle`](crate::evaluation::NativeQmlOracle) path.
    Qml(QmlError),
    /// A derived native QML problem failed its own construction validation —
    /// today only carving a minibatch out of the full problem
    /// (`QmlProblem::subset`), which rejects an empty index set rather than
    /// building a zero-sample problem whose mean fitness would be `NaN`
    /// (contract C-10). Reached only on the minibatch path of the two native QML
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
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::Backend(err) => write!(f, "{err}"),
            EvaluationError::Binding(err) => write!(f, "circuit binding failed: {err}"),
            EvaluationError::Observable(err) => write!(f, "expectation evaluation failed: {err}"),
            EvaluationError::Python(err) => write!(f, "Python evaluation error: {err}"),
            EvaluationError::Qml(err) => write!(f, "QML evaluation error: {err}"),
            EvaluationError::Validation(err) => write!(f, "QML validation error: {err}"),
            EvaluationError::Optimizer(err) => write!(f, "optimizer oracle error: {err}"),
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
            // A Rust-side infrastructure failure: surface as the typed
            // polypus.EvaluationError, not PyO3's generic RuntimeError.
            EvaluationError::Runtime(m) => PyEvaluationError::new_err(m),
            // A Rust-side data-conversion failure: surface as the typed
            // polypus.EvaluationError, not the TypeError PyO3's extract() emits.
            EvaluationError::Conversion(m) => PyEvaluationError::new_err(m),
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
    use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError};
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;

    // Two kinds of test live in this module. The `*_display_*` ones construct no
    // `PyErr` at all, so they need no interpreter whatsoever; every other test
    // pins a variant's `PyErr` mapping and calls `prepare_freethreaded_python()`
    // first. Both stay inside ENGINEERING.md §3: `prepare_freethreaded_python()`
    // + `is_instance_of` need a bare CPython interpreter and no installed
    // package (neither Qiskit nor `polypus_python`), which is exactly what CI
    // provides — the same thing
    // `crates/polypus/tests/running_quantum_circuits_local.rs` already does.

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

    /// The native backend results failing to convert into a Python `list[dict]`
    /// is also modelled by [`EvaluationError::Conversion`]. This call site
    /// differs from the `expectation_values` one above: `counts` is our own
    /// Rust-native `Vec<HashMap<String, u64>>`, never something a Python
    /// callback handed back, so there is no original raised exception to
    /// preserve — realistically only allocation failure can trip it, which
    /// isn't practical to provoke from a test. So we pin the *mapping* for this
    /// site's message: it must surface as the typed `polypus.EvaluationError`,
    /// never as the `MemoryError` the pre-fix `EvaluationError::Python` arm
    /// would have re-raised verbatim. (Issue #98, follow-up to #81.)
    #[test]
    fn counts_conversion_failure_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let msg = "failed to convert the backend results into a Python list[dict]: OOM";
            let err: PyErr = EvaluationError::Conversion(msg.to_string()).into();
            assert!(
                err.value(py).is_instance_of::<PyEvaluationError>(),
                "a counts conversion failure must surface as polypus.EvaluationError"
            );
            // ...and specifically not the MemoryError the pre-fix code would
            // have carried across verbatim via `EvaluationError::Python`.
            assert!(
                !err.value(py).is_instance_of::<PyMemoryError>(),
                "a counts conversion failure must not surface as a raised MemoryError"
            );
            assert!(
                err.to_string()
                    .contains("backend results into a Python list[dict]"),
                "the descriptive message must be preserved"
            );
        });
    }

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
