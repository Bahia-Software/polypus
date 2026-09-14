//! Custom Python exception hierarchy for the Polypus bindings.
//!
//! The orchestration/FFI crate turns its internal `Result` error enums
//! ([`BackendError`](crate::infrastructure::BackendError),
//! [`EvaluationError`](crate::evaluation::EvaluationError)) into typed Python
//! exceptions so that a failure reaching Python is a catchable, documented
//! class instead of a `pyo3_runtime.PanicException` (or a process abort).
//!
//! The hierarchy is rooted at `PolypusError` so user code can catch every
//! Polypus-originated failure with a single `except polypus.PolypusError`.
//! Domain subclasses mirror the layers that raise them:
//!
//! ```text
//! Exception
//! └── PolypusError
//!     ├── BackendError            # execution/orchestration layer
//!     │   ├── CunqaError          # CUNQA distributed-QPU backend
//!     │   ├── QmioError           # QMIO real-QPU network path
//!     │   └── NativeCircuitError  # pure-Rust circuit / simulator path
//!     └── EvaluationError         # optimizer oracle / expectation evaluation
//! ```
//!
//! Contract C-1 (see `docs/CONTRACTS.md`) keeps its documented failure modes:
//! an *unknown infrastructure* still surfaces as `ValueError` and a *bad kwarg*
//! at the `polypus_python` seam still surfaces as `TypeError`, because
//! [`BackendError::Seam`](crate::infrastructure::BackendError::Seam) re-raises
//! the original Python exception verbatim. The classes below are raised for the
//! Rust-originated runtime failures that previously *panicked*.

use crate::infrastructure::BackendError as InfraBackendError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyKeyboardInterrupt, PyValueError};
use pyo3::PyErr;

create_exception!(
    polypus,
    PolypusError,
    PyException,
    "Base class for every Polypus runtime error."
);
create_exception!(
    polypus,
    BackendError,
    PolypusError,
    "A quantum-execution backend failed at runtime."
);
create_exception!(
    polypus,
    CunqaError,
    BackendError,
    "A failure originating in the CUNQA distributed-QPU backend."
);
create_exception!(
    polypus,
    QmioError,
    BackendError,
    "A failure on the QMIO real-QPU network/serialisation path."
);
create_exception!(
    polypus,
    NativeCircuitError,
    BackendError,
    "The native (pure-Rust) circuit or statevector-simulator path failed."
);
create_exception!(
    polypus,
    EvaluationError,
    PolypusError,
    "An oracle/expectation-evaluation failure during training."
);

/// Map a backend-layer [`BackendError`](crate::infrastructure::BackendError) to
/// the typed `polypus.*` Python exception it should surface as.
///
/// This is the FFI edge's sole responsibility: `polypus-infrastructure`
/// deliberately implements no `From<_> for PyErr`, so the whole
/// backend→exception-class decision is made here, once. `run_quantum_circuit`
/// and the oracle path reach it directly or via
/// [`infrastructure_error_to_pyerr`](crate::algorithms::orchestration::infrastructure_error_to_pyerr).
/// Contract C-1's documented failure modes are preserved: a `Seam` error
/// re-raises the original Python exception verbatim (keeping its
/// `ValueError`/`TypeError` type), and an unknown infrastructure is a `ValueError`.
pub(crate) fn backend_error_to_pyerr(err: InfraBackendError) -> PyErr {
    match err {
        // Re-raise the original Python exception unchanged (contract C-1).
        InfraBackendError::Seam(py_err) => py_err,
        InfraBackendError::UnknownInfrastructure { name } => PyValueError::new_err(format!(
            "unknown infrastructure '{name}'; expected \"local\", \"cunqa\" or \"qmio\""
        )),
        InfraBackendError::InvalidCircuitCount { expected, got } => {
            PyValueError::new_err(format!("expected exactly {expected} circuit(s), got {got}"))
        }
        InfraBackendError::UnsupportedCircuit(m) => NativeCircuitError::new_err(m),
        // A backend/contract violation on our own results: the typed backend base
        // class, not a native-circuit or seam error.
        InfraBackendError::InvalidResults(m) => {
            BackendError::new_err(format!("backend returned invalid results: {m}"))
        }
        InfraBackendError::NativeCircuit(m) => NativeCircuitError::new_err(m),
        InfraBackendError::Cunqa(m) => CunqaError::new_err(m),
        InfraBackendError::Conversion(m) => BackendError::new_err(m),
        #[cfg(feature = "qmio")]
        InfraBackendError::Qmio(qmio_err) => QmioError::new_err(qmio_err.to_string()),
    }
}

/// Map an evaluation-layer [`EvaluationError`](crate::evaluation::EvaluationError)
/// to the typed `polypus.*` Python exception it should surface as.
///
/// The FFI edge's counterpart to [`backend_error_to_pyerr`]:
/// `polypus-evaluation` implements no `From<_> for PyErr`, so this is where an
/// oracle failure becomes an exception. `Python`/callback-boxed variants re-raise
/// their original Python exception verbatim; a wrapped `Backend` failure keeps its
/// own class (delegates to [`backend_error_to_pyerr`]); everything else surfaces
/// as `polypus.EvaluationError`.
pub(crate) fn evaluation_error_to_pyerr(err: crate::evaluation::EvaluationError) -> PyErr {
    use crate::evaluation::EvaluationError as EvalErr;
    match err {
        EvalErr::Backend(backend_err) => backend_error_to_pyerr(backend_err),
        EvalErr::Binding(circuit_err) => EvaluationError::new_err(circuit_err.to_string()),
        EvalErr::Observable(obs_err) => match obs_err {
            // A callback observable boxes its `PyErr` here; recover it so the
            // original Python exception type re-raises verbatim across the FFI.
            polypus_observable::ObservableError::External(boxed) => {
                match boxed.downcast::<PyErr>() {
                    Ok(py_err) => *py_err,
                    Err(other) => EvaluationError::new_err(other.to_string()),
                }
            }
            // Native evaluation failures (bad bitstring, invalid construction) map
            // to the typed evaluation exception.
            other => EvaluationError::new_err(other.to_string()),
        },
        // Preserve the original Python exception type raised by the callback.
        EvalErr::Python(py_err) => py_err,
        // Rust-side failures: surface as the typed polypus.EvaluationError, not
        // PyO3's generic RuntimeError / the TypeError extract() would emit.
        EvalErr::Runtime(m) => EvaluationError::new_err(m),
        EvalErr::Conversion(m) => EvaluationError::new_err(m),
        wrong_length @ EvalErr::WrongLength { .. } => {
            EvaluationError::new_err(wrong_length.to_string())
        }
        non_finite @ EvalErr::NonFinite { .. } => EvaluationError::new_err(non_finite.to_string()),
        invalid_variance @ EvalErr::InvalidVariance { .. } => {
            EvaluationError::new_err(invalid_variance.to_string())
        }
    }
}

/// Convert a planner's `InfrastructureError` into a `PyErr` at the FFI edge.
///
/// `InfrastructureError` deliberately implements no `From<_> for PyErr`; this is
/// the one place `run_quantum_circuit` / the training flows perform that
/// conversion, mapping each variant to the class it always surfaced as: a backend
/// error keeps its own class (via [`backend_error_to_pyerr`]), a Python exception
/// (a SIGINT from the planner's `check_signals`) re-raises verbatim, and a
/// cooperative cancel is a `KeyboardInterrupt`.
pub(crate) fn infrastructure_error_to_pyerr(
    err: crate::infrastructure::InfrastructureError,
) -> PyErr {
    use crate::infrastructure::InfrastructureError as InfraErr;
    match err {
        InfraErr::Backend(e) => backend_error_to_pyerr(e),
        InfraErr::Observable(e) => EvaluationError::new_err(e.to_string()),
        InfraErr::Python(e) => e,
        InfraErr::Cancelled => PyKeyboardInterrupt::new_err("the run was cancelled"),
        InfraErr::IncompatiblePlanner(m) => PyValueError::new_err(m),
    }
}

/// Register the exception hierarchy on the extension module so Python can both
/// see (`polypus.BackendError`) and catch these classes.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    let py = m.py();
    m.add("PolypusError", py.get_type::<PolypusError>())?;
    m.add("BackendError", py.get_type::<BackendError>())?;
    m.add("CunqaError", py.get_type::<CunqaError>())?;
    m.add("QmioError", py.get_type::<QmioError>())?;
    m.add("NativeCircuitError", py.get_type::<NativeCircuitError>())?;
    m.add("EvaluationError", py.get_type::<EvaluationError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::PyTypeInfo;

    // These pin the FFI mapping now that it lives here (relocated from
    // `polypus-infrastructure`'s `error.rs` when the backend layer moved to its
    // own crate). The variants are constructed directly — the public entry points
    // reject the offending input long before it could reach them, so this is
    // defense-in-depth on the class each variant surfaces as. `is_instance_of`
    // needs an initialised interpreter but no installed package (ENGINEERING §3).

    /// Assert `err` crosses the FFI as an instance of the Python class `E`, that
    /// it is catchable as `polypus.PolypusError`, and that its message survives.
    fn assert_maps_to<E: PyTypeInfo>(err: InfraBackendError, expected_message: &str) {
        pyo3::prepare_freethreaded_python();
        let py_err = backend_error_to_pyerr(err);
        Python::with_gil(|py| {
            assert!(
                py_err.is_instance_of::<E>(py),
                "wrong exception class for: {py_err}"
            );
            assert!(
                py_err.is_instance_of::<PolypusError>(py),
                "every polypus.* class must stay catchable as PolypusError: {py_err}"
            );
            assert!(
                py_err.to_string().contains(expected_message),
                "message lost in translation: {py_err}"
            );
        });
    }

    #[test]
    fn unsupported_circuit_maps_to_native_circuit_error() {
        assert_maps_to::<NativeCircuitError>(
            InfraBackendError::UnsupportedCircuit(
                "the native statevector backend cannot execute a Qiskit QuantumCircuit".to_string(),
            ),
            "cannot execute a Qiskit QuantumCircuit",
        );
    }

    #[test]
    fn native_circuit_maps_to_native_circuit_error() {
        assert_maps_to::<NativeCircuitError>(
            InfraBackendError::NativeCircuit("could not parse OpenQASM 2.0".to_string()),
            "could not parse OpenQASM 2.0",
        );
    }

    #[test]
    fn conversion_maps_to_the_backend_error_base_class() {
        assert_maps_to::<BackendError>(
            InfraBackendError::Conversion("counts were not convertible".to_string()),
            "counts were not convertible",
        );
    }

    #[test]
    fn cunqa_maps_to_cunqa_error() {
        assert_maps_to::<CunqaError>(
            InfraBackendError::Cunqa("injected release failure".to_string()),
            "injected release failure",
        );
    }

    #[test]
    fn provider_errors_stay_catchable_as_backend_error() {
        // The hierarchy is what lets `except polypus.BackendError` catch every
        // backend-layer failure regardless of which provider raised it.
        pyo3::prepare_freethreaded_python();
        let cunqa = backend_error_to_pyerr(InfraBackendError::Cunqa("x".to_string()));
        let native = backend_error_to_pyerr(InfraBackendError::UnsupportedCircuit("y".to_string()));
        Python::with_gil(|py| {
            assert!(cunqa.is_instance_of::<BackendError>(py));
            assert!(native.is_instance_of::<BackendError>(py));
        });
    }
}

#[cfg(test)]
mod evaluation_mapping_tests {
    // Relocated from polypus-evaluation's error.rs when EvaluationError moved to
    // its own crate: the FFI mapping now lives here (`evaluation_error_to_pyerr`).
    // `prepare_freethreaded_python()` + `is_instance_of` need a bare CPython
    // interpreter and no installed package (ENGINEERING §3).
    use super::*;
    use crate::evaluation::EvaluationError as EvalErr;
    use polypus_circuit::CircuitError;
    use polypus_infrastructure::BackendError;
    use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError};
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    #[test]
    fn runtime_variant_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = evaluation_error_to_pyerr(EvalErr::Runtime("worker panicked".to_string()));
            assert!(
                err.value(py).is_instance_of::<EvaluationError>(),
                "Runtime must surface as polypus.EvaluationError"
            );
            assert!(
                !err.value(py).is_instance_of::<PyRuntimeError>(),
                "Runtime must not surface as the generic RuntimeError"
            );
            assert!(err.to_string().contains("worker panicked"));
        });
    }

    #[test]
    fn conversion_variant_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = evaluation_error_to_pyerr(EvalErr::Conversion("not list[float]".to_string()));
            assert!(err.value(py).is_instance_of::<EvaluationError>());
            assert!(
                !err.value(py).is_instance_of::<PyTypeError>(),
                "Conversion must not surface as the generic TypeError"
            );
            assert!(err.to_string().contains("not list[float]"));
        });
    }

    #[test]
    fn counts_conversion_failure_maps_to_typed_evaluation_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let msg = "failed to convert the backend results into a Python list[dict]: OOM";
            let err = evaluation_error_to_pyerr(EvalErr::Conversion(msg.to_string()));
            assert!(err.value(py).is_instance_of::<EvaluationError>());
            assert!(
                !err.value(py).is_instance_of::<PyMemoryError>(),
                "a counts conversion failure must not surface as a raised MemoryError"
            );
            assert!(err
                .to_string()
                .contains("backend results into a Python list[dict]"));
        });
    }

    /// Assert `err` crosses the FFI as `polypus.EvaluationError`, stays catchable
    /// as `PolypusError`, and keeps its message.
    fn assert_maps_to_evaluation_error(err: EvalErr, expected_message: &str) {
        pyo3::prepare_freethreaded_python();
        let py_err = evaluation_error_to_pyerr(err);
        Python::with_gil(|py| {
            assert!(
                py_err.is_instance_of::<EvaluationError>(py),
                "wrong exception class for: {py_err}"
            );
            assert!(
                py_err.is_instance_of::<PolypusError>(py),
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
        assert_maps_to_evaluation_error(
            EvalErr::Binding(CircuitError::WrongNumberOfParams {
                expected: 3,
                got: 1,
            }),
            "circuit declares 3 free parameter(s) but 1 value(s) were provided",
        );
    }

    #[test]
    fn wrong_length_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvalErr::WrongLength {
                expected: 4,
                got: 2,
            },
            "contract C-5",
        );
    }

    #[test]
    fn non_finite_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvalErr::NonFinite {
                index: 3,
                value: f64::NAN,
            },
            "contract C-5",
        );
    }

    #[test]
    fn invalid_variance_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvalErr::InvalidVariance {
                param_index: 1,
                value: -1.0,
            },
            "finite, non-negative",
        );
    }

    #[test]
    fn backend_variant_delegates_to_the_backend_mapping() {
        // `EvaluationError::Backend` must not retype the wrapped failure: a CUNQA
        // error surfacing through an oracle is still a `polypus.CunqaError`.
        pyo3::prepare_freethreaded_python();
        let py_err = evaluation_error_to_pyerr(EvalErr::Backend(BackendError::Cunqa(
            "qraise failed".to_string(),
        )));
        Python::with_gil(|py| {
            assert!(py_err.is_instance_of::<CunqaError>(py));
            assert!(
                !py_err.is_instance_of::<EvaluationError>(py),
                "a wrapped backend failure must keep its own class"
            );
        });
    }
}
