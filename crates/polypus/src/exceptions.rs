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
//!     ├── BackendError                # execution/orchestration layer
//!     │   ├── CunqaError              # CUNQA distributed-QPU backend
//!     │   ├── QmioError               # QMIO real-QPU network path
//!     │   ├── NativeCircuitError      # pure-Rust circuit / simulator path
//!     │   └── InsufficientMemoryError # a statevector exceeds the memory budget
//!     └── EvaluationError             # optimizer oracle / expectation evaluation
//! ```
//!
//! Contract C-1 (see `docs/CONTRACTS.md`) keeps its documented failure modes:
//! an *unknown infrastructure* still surfaces as `ValueError` and a *bad kwarg*
//! at the `polypus_python` seam still surfaces as `TypeError`, because a seam
//! exception is carried type-erased in
//! [`BackendError::External`](crate::infrastructure::BackendError::External), as a
//! [`DisplaySafePyErr`](crate::infrastructure::DisplaySafePyErr), and
//! `external_to_pyerr` re-raises the original Python exception verbatim. The
//! one exception is an exception raised by Qiskit (issue #218): it becomes a
//! `BackendError` chaining the original as `__cause__` — or an
//! `EvaluationError` when Qiskit fails while preparing or binding a Qiskit
//! circuit before any backend runs — so a Qiskit failure is catchable as
//! `PolypusError` like any other (see `wrap_qiskit_error`). The classes below
//! are raised for the Rust-originated runtime failures that previously
//! *panicked*.

use crate::infrastructure::BackendError as InfraBackendError;
use crate::infrastructure::DisplaySafePyErr;
use pyo3::exceptions::{PyException, PyKeyboardInterrupt, PyTypeError, PyValueError};
use pyo3::{create_exception, intern, PyErr};

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
    InsufficientMemoryError,
    BackendError,
    "A statevector would not fit in the memory budget, so the run was refused before \
     starting instead of being killed by the out-of-memory killer. The budget is \
     POLYPUS_MEM_BUDGET when set (e.g. POLYPUS_MEM_BUDGET=64G), else the detected \
     RAM/cgroup limit minus a safety reserve."
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
/// [`infrastructure_error_to_pyerr`](crate::exceptions::infrastructure_error_to_pyerr).
/// Contract C-1's documented failure modes are preserved: a seam exception boxed
/// in [`External`](InfraBackendError::External) re-raises the original Python
/// exception verbatim (keeping its `ValueError`/`TypeError` type, via
/// `external_to_pyerr`) unless Qiskit raised it, which makes it a
/// `polypus.BackendError` (`qiskit_error_to_pyerr`); an unknown infrastructure
/// is a `ValueError`.
pub(crate) fn backend_error_to_pyerr(err: InfraBackendError) -> PyErr {
    match err {
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
        // A backend that stopped responding mid-call (Fase-2 subprocess-bridge
        // finding): the typed backend base class.
        InfraBackendError::Unresponsive(m) => {
            BackendError::new_err(format!("the backend stopped responding: {m}"))
        }
        // A backend call aborted by an external signal (a terminal Ctrl+C reaching a
        // subprocess worker via the process group, say) is a cancellation whatever
        // its source, so it raises `KeyboardInterrupt` — the same class a between-wave
        // `check_signals` cancel (`InfrastructureError::Cancelled`) produces. This is
        // what makes an interrupt observed *inside* a blocked backend call surface
        // identically to one observed between waves.
        InfraBackendError::Aborted(m) => {
            PyKeyboardInterrupt::new_err(format!("the run was aborted: {m}"))
        }
        // A statevector that cannot fit in a known memory budget (issue #215): its
        // own `polypus.BackendError` subclass, so it stays catchable as
        // `PolypusError` (contract C-1) while being distinguishable from a crash.
        InfraBackendError::InsufficientMemory(e) => insufficient_memory_to_pyerr(&e),
        // The pyo3-free contract carries any provider/Python failure type-erased
        // here; recover its original class so contract C-1 holds (a seam
        // `ValueError`/`TypeError`, a `KeyboardInterrupt`, or a `polypus.QmioError`
        // all re-raise as themselves).
        InfraBackendError::External(boxed) => external_to_pyerr(boxed),
    }
}

/// The `polypus.InsufficientMemoryError` for a refused statevector, shared by the
/// backend mapping above and by `polypus.statevector`, which checks the budget
/// itself before allocating.
pub(crate) fn insufficient_memory_to_pyerr(
    err: &crate::infrastructure::InsufficientMemory,
) -> PyErr {
    InsufficientMemoryError::new_err(err.to_string())
}

/// Recover the concrete class of a type-erased
/// [`BackendError::External`](InfraBackendError::External) payload.
///
/// A Polypus Python backend boxes a [`DisplaySafePyErr`] here (a
/// `polypus_python` seam exception, or a `KeyboardInterrupt` from
/// `check_signals` / a Qiskit width read), whose original exception re-raises
/// verbatim unless Qiskit raised it ([`qiskit_error_to_pyerr`]); the QMIO backend
/// boxes its own `QmioError`; anything else is a third-party provider error.
///
/// Only that carrier is recovered. A bare `PyErr` boxed here is treated like any
/// other provider error (`polypus.BackendError` with its message): the pyo3-free
/// layers may format the box while detached, which a bare `PyErr` cannot survive
/// at interpreter shutdown, so boxing one is a bug that tests must catch rather
/// than a path the edge quietly keeps working. This is the FFI-edge
/// counterpart of the old `BackendError::Seam`/`BackendError::Qmio` variants,
/// preserving contract C-1's verbatim re-raise now that the boxing is generic
/// and pyo3-free.
fn external_to_pyerr(boxed: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    // A boxed Python exception re-raises verbatim, keeping its original class —
    // unless Qiskit raised it (issue #218), which joins the polypus hierarchy.
    let boxed = match boxed.downcast::<DisplaySafePyErr>() {
        Ok(py_err) => return qiskit_error_to_pyerr(py_err.into_inner()),
        Err(other) => other,
    };
    // The QMIO backend boxes its own error; surface it as the typed class.
    #[cfg(feature = "qmio")]
    let boxed = match boxed.downcast::<polypus_infrastructure::qmio::QmioError>() {
        Ok(qmio_err) => return QmioError::new_err(qmio_err.to_string()),
        Err(other) => other,
    };
    // Any other provider error: the typed backend base class, message preserved.
    BackendError::new_err(boxed.to_string())
}

/// Whether `module` (a class's `__module__`) belongs to a Qiskit package:
/// `qiskit` itself or a `qiskit_*` distribution such as `qiskit_aer`.
fn is_qiskit_module(module: &str) -> bool {
    let root = module.split('.').next().unwrap_or(module);
    root == "qiskit" || root.starts_with("qiskit_")
}

/// Whether `err` was raised by Qiskit: its class, or any class in its MRO, is
/// defined in a `qiskit*` module (`qiskit.exceptions.QiskitError`,
/// `qiskit_aer.AerError`, `qiskit.qasm2.QASM2ParseError`, a user subclass of
/// any of them, …).
///
/// Decided from the class names alone, so Qiskit is never imported: a backend
/// that does not use it (QMIO, a subprocess provider) never needs it installed.
/// This is the one definition of "a Qiskit exception"; every point where one
/// can cross into Python goes through [`wrap_qiskit_error`]
/// ([`qiskit_error_to_pyerr`] or [`qiskit_error_to_evaluation_error`]).
pub(crate) fn is_qiskit_exception(py: pyo3::Python<'_>, err: &PyErr) -> bool {
    use pyo3::types::{PyAnyMethods, PyTupleMethods, PyTypeMethods};
    err.get_type(py).mro().iter().any(|class| {
        class
            .getattr(intern!(py, "__module__"))
            .and_then(|module| module.extract::<String>())
            .is_ok_and(|module| is_qiskit_module(&module))
    })
}

/// The `module.qualname: message` text of a Qiskit exception, as it appears in
/// the `polypus.*` exception that wraps it (and in
/// `polypus.backend_compatibility`'s Aer reasons, which quote the same
/// failure).
pub(crate) fn describe_python_error(py: pyo3::Python<'_>, err: &PyErr) -> String {
    use pyo3::types::{PyAnyMethods, PyTypeMethods};
    let class = err
        .get_type(py)
        .fully_qualified_name()
        .map_or_else(|_| "<unknown exception>".to_string(), |n| n.to_string());
    let message = err
        .value(py)
        .str()
        .map_or_else(|_| "<unprintable message>".to_string(), |m| m.to_string());
    format!("{class}: {message}")
}

/// Contract C-1 as amended by issue #218, on the execution seam: an exception
/// raised by Qiskit reaches the caller as `polypus.BackendError` (see
/// [`wrap_qiskit_error`] for the rule); any other exception is returned
/// unchanged.
pub(crate) fn qiskit_error_to_pyerr(err: PyErr) -> PyErr {
    wrap_qiskit_error::<BackendError>(err)
}

/// The same rule as [`qiskit_error_to_pyerr`] for a Qiskit failure while
/// preparing or binding a Qiskit circuit for an optimizer, before any backend
/// runs (`qml.train`/`qml.predict` composing and binding their circuits, the
/// oracles' `assign_parameters`): it becomes `polypus.EvaluationError`.
pub(crate) fn qiskit_error_to_evaluation_error(err: PyErr) -> PyErr {
    wrap_qiskit_error::<EvaluationError>(err)
}

/// Raise an exception Qiskit raised as the `polypus.*` class `E`, with the
/// Qiskit class's qualified name and its message as the message
/// ([`describe_python_error`]) and the original chained as `__cause__`, so its
/// traceback is kept. Any other exception is returned unchanged.
///
/// A Qiskit class that is also a `ValueError`, `TypeError` or
/// `KeyboardInterrupt` is returned unchanged too: those are C-1's typed failure
/// modes (and a cancellation), and a caller catching them must keep catching
/// them.
///
/// Needs the interpreter; when it cannot be reached (it is finalizing), the
/// exception is returned unchanged rather than attaching unsafely (ENGINEERING
/// §9).
fn wrap_qiskit_error<E: pyo3::PyTypeInfo>(err: PyErr) -> PyErr {
    let wrapper = crate::infrastructure::attach_or(
        || None,
        |py| {
            let preserved = err.is_instance_of::<PyValueError>(py)
                || err.is_instance_of::<PyTypeError>(py)
                || err.is_instance_of::<PyKeyboardInterrupt>(py);
            (!preserved && is_qiskit_exception(py, &err))
                .then(|| PyErr::new::<E, _>(describe_python_error(py, &err)))
        },
    );
    match wrapper {
        Some(wrapper) => {
            crate::infrastructure::attach_or(|| (), |py| wrapper.set_cause(py, Some(err)));
            wrapper
        }
        None => err,
    }
}

/// Map a native cost-observable
/// [`ObservableError`](polypus_observable::ObservableError) to the `PyErr` it
/// should surface as.
///
/// A callback observable boxes its own exception, as a [`DisplaySafePyErr`], into
/// the [`External`](polypus_observable::ObservableError::External) variant;
/// recover it so the *original* Python exception type re-raises verbatim across
/// the FFI instead of being flattened into a `polypus.EvaluationError`. As in
/// `external_to_pyerr`, a bare boxed `PyErr` is not recovered. Every other
/// variant is a native evaluation failure (bad bitstring, invalid construction)
/// and surfaces as the typed `polypus.EvaluationError`.
///
/// Both [`evaluation_error_to_pyerr`] and [`infrastructure_error_to_pyerr`]
/// route their `Observable` arm through here so the downcast lives in one place.
fn observable_error_to_pyerr(err: polypus_observable::ObservableError) -> PyErr {
    use polypus_observable::ObservableError as ObsErr;
    match err {
        // A callback observable boxes its exception here; recover it so the
        // original Python exception type re-raises verbatim across the FFI.
        ObsErr::External(boxed) => match boxed.downcast::<DisplaySafePyErr>() {
            Ok(py_err) => py_err.into_inner(),
            Err(other) => EvaluationError::new_err(other.to_string()),
        },
        // Native evaluation failures (bad bitstring, invalid construction) map
        // to the typed evaluation exception.
        other => EvaluationError::new_err(other.to_string()),
    }
}

/// Map an evaluation-layer [`EvaluationError`](crate::evaluation::EvaluationError)
/// to the typed `polypus.*` Python exception it should surface as.
///
/// The FFI edge's counterpart to [`backend_error_to_pyerr`]:
/// `polypus-evaluation` implements no `From<_> for PyErr`, so this is where an
/// oracle failure becomes an exception. `Python`/callback-boxed variants re-raise
/// their original Python exception verbatim; a `Qiskit` binding failure becomes
/// `polypus.EvaluationError` when Qiskit raised it
/// ([`qiskit_error_to_evaluation_error`]); a wrapped `Backend` failure keeps its
/// own class (delegates to [`backend_error_to_pyerr`]); everything else surfaces
/// as `polypus.EvaluationError`.
pub(crate) fn evaluation_error_to_pyerr(err: crate::evaluation::EvaluationError) -> PyErr {
    use crate::evaluation::EvaluationError as EvalErr;
    match err {
        EvalErr::Backend(backend_err) => backend_error_to_pyerr(backend_err),
        EvalErr::Binding(circuit_err) => EvaluationError::new_err(circuit_err.to_string()),
        EvalErr::Observable(obs_err) => observable_error_to_pyerr(obs_err),
        // Preserve the original Python exception type raised by the callback.
        EvalErr::Python(py_err) => py_err.into_inner(),
        // Qiskit's own binding call: a Qiskit exception joins the hierarchy as
        // `polypus.EvaluationError` (issue #218); anything else stays verbatim.
        EvalErr::Qiskit(py_err) => qiskit_error_to_evaluation_error(py_err.into_inner()),
        // Rust-side failures: surface as the typed polypus.EvaluationError, not
        // PyO3's generic RuntimeError / the TypeError extract() would emit.
        EvalErr::Runtime(m) => EvaluationError::new_err(m),
        EvalErr::Conversion(m) => EvaluationError::new_err(m),
        wrong_length @ EvalErr::WrongLength { .. } => {
            EvaluationError::new_err(wrong_length.to_string())
        }
        non_finite @ EvalErr::NonFinite { .. } => EvaluationError::new_err(non_finite.to_string()),
        non_finite_score @ EvalErr::NonFiniteScore { .. } => {
            EvaluationError::new_err(non_finite_score.to_string())
        }
        label_count @ EvalErr::LabelCount { .. } => {
            EvaluationError::new_err(label_count.to_string())
        }
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
/// error keeps its own class (via [`backend_error_to_pyerr`]), an observable
/// failure goes through [`observable_error_to_pyerr`] so a Python exception
/// raised inside a callback observable re-raises with its original class, a
/// Python exception (a SIGINT from the planner's `check_signals`) re-raises
/// verbatim, and a cooperative cancel is a `KeyboardInterrupt`.
pub(crate) fn infrastructure_error_to_pyerr(
    err: crate::infrastructure::InfrastructureError,
) -> PyErr {
    use crate::infrastructure::InfrastructureError as InfraErr;
    match err {
        InfraErr::Backend(e) => backend_error_to_pyerr(e),
        // Share the callback-downcast with `evaluation_error_to_pyerr`: an
        // `External` variant carries a Python exception raised inside a callback
        // observable, and its original class must re-raise verbatim rather than
        // being discarded into a generic `polypus.EvaluationError`.
        InfraErr::Observable(e) => observable_error_to_pyerr(e),
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
    m.add(
        "InsufficientMemoryError",
        py.get_type::<InsufficientMemoryError>(),
    )?;
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
        pyo3::Python::initialize();
        let py_err = backend_error_to_pyerr(err);
        Python::attach(|py| {
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
    fn insufficient_memory_maps_to_insufficient_memory_error() {
        use crate::infrastructure::{BudgetSource, InsufficientMemory};
        // A `BackendError` subclass: catchable as `polypus.BackendError` and, via
        // `assert_maps_to`, as `polypus.PolypusError` (contract C-1).
        let refusal = || {
            InfraBackendError::InsufficientMemory(InsufficientMemory {
                num_qubits: 20,
                required_bytes: 16 << 20,
                budget_bytes: 1 << 20,
                source: BudgetSource::Explicit,
            })
        };
        assert_maps_to::<InsufficientMemoryError>(refusal(), "20-qubit statevector");
        assert_maps_to::<BackendError>(refusal(), "POLYPUS_MEM_BUDGET");
    }

    #[test]
    fn unresponsive_maps_to_the_backend_error_base_class() {
        // A backend that stopped responding mid-call (the Fase-2 subprocess-bridge
        // finding) surfaces as the typed backend base class, message preserved.
        assert_maps_to::<BackendError>(
            InfraBackendError::Unresponsive("worker died (signal 9)".to_string()),
            "the backend stopped responding: worker died (signal 9)",
        );
    }

    #[test]
    fn aborted_maps_to_keyboard_interrupt() {
        // A backend call aborted by an external signal (a terminal Ctrl+C reaching a
        // subprocess worker via the process group) is a cancellation whatever its
        // source, so it surfaces as a native `KeyboardInterrupt` — NOT a `polypus.*`
        // class — exactly like `InfrastructureError::Cancelled` does. Assert directly
        // rather than via `assert_maps_to`, which requires `PolypusError` catchability.
        pyo3::Python::initialize();
        let py_err = backend_error_to_pyerr(InfraBackendError::Aborted("Ctrl+C".to_string()));
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyKeyboardInterrupt>(py),
                "an aborted backend call must surface as KeyboardInterrupt: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<PolypusError>(py),
                "KeyboardInterrupt is a native Python exception, not a polypus.* class"
            );
            assert!(py_err.to_string().contains("the run was aborted: Ctrl+C"));
        });
    }

    #[test]
    fn external_provider_error_maps_to_the_backend_error_base_class() {
        // A non-Python, non-QMIO provider error boxed in `External` surfaces as the
        // typed backend base class with its message (the seam/KeyboardInterrupt/QMIO
        // recovery paths are covered by `external_backend_error_reraises_verbatim`
        // and the QMIO tests).
        #[derive(Debug)]
        struct Provider(&'static str);
        impl std::fmt::Display for Provider {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "provider failure: {}", self.0)
            }
        }
        impl std::error::Error for Provider {}
        assert_maps_to::<BackendError>(
            InfraBackendError::External(Box::new(Provider("device offline"))),
            "provider failure: device offline",
        );
    }

    /// Instantiate `class` from a bare-interpreter stand-in for Qiskit's
    /// exceptions (no package needed, ENGINEERING §3): the classes only claim a
    /// `qiskit*` `__module__`, which is all `is_qiskit_exception` reads.
    pub(super) fn fake_error(py: Python<'_>, class: &str, message: &str) -> PyErr {
        let module = pyo3::types::PyModule::from_code(
            py,
            c"class QiskitError(Exception): pass\n\
              QiskitError.__module__ = 'qiskit.exceptions'\n\
              class AerError(QiskitError): pass\n\
              AerError.__module__ = 'qiskit_aer.aererror'\n\
              class UserError(QiskitError): pass\n\
              UserError.__module__ = 'user_code'\n\
              class QiskitValueError(QiskitError, ValueError): pass\n\
              class QiskitKeyboardInterrupt(QiskitError, KeyboardInterrupt): pass\n\
              class NotQiskit(Exception): pass\n\
              NotQiskit.__module__ = 'qiskitlike'\n",
            c"fake_qiskit.py",
            c"fake_qiskit",
        )
        .expect("the stand-in module compiles");
        let value = module
            .getattr(class)
            .and_then(|c| c.call1((message,)))
            .expect("the stand-in class instantiates");
        PyErr::from_value(value)
    }

    #[test]
    fn qiskit_modules_are_recognised_by_their_root_package() {
        for module in ["qiskit", "qiskit.exceptions", "qiskit.qasm2.exceptions"] {
            assert!(is_qiskit_module(module), "{module}");
        }
        for module in ["qiskit_aer", "qiskit_aer.aererror", "qiskit_ibm_runtime.x"] {
            assert!(is_qiskit_module(module), "{module}");
        }
        for module in [
            "builtins",
            "qiskitlike",
            "my_qiskit",
            "polypus",
            "user.qiskit",
        ] {
            assert!(!is_qiskit_module(module), "{module}");
        }
    }

    #[test]
    fn qiskit_exception_is_detected_through_the_mro() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            for class in ["QiskitError", "AerError", "UserError", "QiskitValueError"] {
                assert!(
                    is_qiskit_exception(py, &fake_error(py, class, "x")),
                    "{class}"
                );
            }
            assert!(!is_qiskit_exception(py, &fake_error(py, "NotQiskit", "x")));
            let runtime = pyo3::exceptions::PyRuntimeError::new_err("x");
            assert!(!is_qiskit_exception(py, &runtime));
        });
    }

    #[test]
    fn external_qiskit_error_maps_to_backend_error_with_its_cause() {
        pyo3::Python::initialize();
        for (class, qualified) in [
            ("QiskitError", "qiskit.exceptions.QiskitError"),
            ("AerError", "qiskit_aer.aererror.AerError"),
            // Defined outside qiskit: only its MRO makes it a Qiskit error.
            ("UserError", "user_code.UserError"),
        ] {
            let original = Python::attach(|py| fake_error(py, class, "no counts"));
            let original_value = Python::attach(|py| original.value(py).clone().unbind());
            let expected = format!("{qualified}: no counts");
            let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
                DisplaySafePyErr::from(original),
            )));
            Python::attach(|py| {
                assert!(
                    py_err.get_type(py).is(py.get_type::<BackendError>()),
                    "exactly polypus.BackendError, not a subclass: {py_err}"
                );
                assert!(py_err.is_instance_of::<PolypusError>(py), "{py_err}");
                assert!(
                    py_err.to_string().contains(&expected),
                    "the Qiskit class and message must survive: {py_err}"
                );
                let cause = py_err.cause(py).expect("the Qiskit original is chained");
                assert!(
                    cause.value(py).is(original_value.bind(py)),
                    "__cause__ must be the original exception object"
                );
            });
        }
    }

    #[test]
    fn qiskit_error_that_is_also_a_c1_type_is_preserved() {
        pyo3::Python::initialize();
        let value_error = Python::attach(|py| fake_error(py, "QiskitValueError", "bad kwarg"));
        let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
            DisplaySafePyErr::from(value_error),
        )));
        Python::attach(|py| {
            assert!(py_err.is_instance_of::<PyValueError>(py), "{py_err}");
            assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
        });
        let interrupt = Python::attach(|py| fake_error(py, "QiskitKeyboardInterrupt", "stop"));
        let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
            DisplaySafePyErr::from(interrupt),
        )));
        Python::attach(|py| {
            assert!(py_err.is_instance_of::<PyKeyboardInterrupt>(py), "{py_err}");
            assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
        });
    }

    #[test]
    fn non_qiskit_python_errors_are_reraised_verbatim() {
        // C-1: anything Qiskit did not raise keeps its class — the seam's own
        // `ValueError`/`TypeError`, a `RuntimeError`, or a class whose module
        // merely looks like Qiskit's.
        pyo3::Python::initialize();
        let cases: [(PyErr, &str); 4] = [
            (
                pyo3::exceptions::PyRuntimeError::new_err("boom"),
                "RuntimeError",
            ),
            (PyTypeError::new_err("bad kwarg"), "TypeError"),
            (
                PyValueError::new_err("unknown infrastructure"),
                "ValueError",
            ),
            (
                Python::attach(|py| fake_error(py, "NotQiskit", "x")),
                "NotQiskit",
            ),
        ];
        for (err, class) in cases {
            let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
                DisplaySafePyErr::from(err),
            )));
            Python::attach(|py| {
                assert_eq!(py_err.get_type(py).name().unwrap().to_string(), class);
                assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
                assert!(py_err.cause(py).is_none(), "nothing is chained: {py_err}");
            });
        }
    }

    #[test]
    fn display_safe_py_err_reraises_the_original_exception() {
        // The carrier is unwrapped at the edge: the very exception object boxed by
        // the backend is what Python receives, with its class, not a copy.
        pyo3::Python::initialize();
        let original = PyTypeError::new_err("bad kwarg");
        let original_value = Python::attach(|py| original.value(py).clone().unbind());
        let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
            DisplaySafePyErr::from(original),
        )));
        Python::attach(|py| {
            assert!(py_err.is_instance_of::<PyTypeError>(py), "{py_err}");
            assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
            assert!(py_err.value(py).is(original_value.bind(py)));
        });
    }

    #[test]
    fn bare_py_err_in_external_is_not_reraised_verbatim() {
        // Only the attach-safe carrier is recovered (ENGINEERING §9): a bare
        // `PyErr` boxed by mistake becomes a plain provider error, so its C-1
        // class is lost and the C-1 tests catch the mistake instead of it
        // passing silently.
        assert_maps_to::<BackendError>(
            InfraBackendError::External(Box::new(PyValueError::new_err("bad kwarg"))),
            "ValueError: bad kwarg",
        );
        let py_err = backend_error_to_pyerr(InfraBackendError::External(Box::new(
            PyValueError::new_err("bad kwarg"),
        )));
        Python::attach(|py| assert!(!py_err.is_instance_of::<PyValueError>(py)));
    }

    #[test]
    fn provider_errors_stay_catchable_as_backend_error() {
        // The hierarchy is what lets `except polypus.BackendError` catch every
        // backend-layer failure regardless of which provider raised it.
        pyo3::Python::initialize();
        let cunqa = backend_error_to_pyerr(InfraBackendError::Cunqa("x".to_string()));
        let native = backend_error_to_pyerr(InfraBackendError::UnsupportedCircuit("y".to_string()));
        Python::attach(|py| {
            assert!(cunqa.is_instance_of::<BackendError>(py));
            assert!(native.is_instance_of::<BackendError>(py));
        });
    }
}

#[cfg(test)]
mod evaluation_mapping_tests {
    // Relocated from polypus-evaluation's error.rs when EvaluationError moved to
    // its own crate: the FFI mapping now lives here (`evaluation_error_to_pyerr`).
    // `Python::initialize()` + `is_instance_of` need a bare CPython
    // interpreter and no installed package (ENGINEERING §3).
    use super::*;
    use crate::evaluation::EvaluationError as EvalErr;
    use polypus_circuit::CircuitError;
    use polypus_infrastructure::BackendError;
    use polypus_observable::ObservableError;
    use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyZeroDivisionError};
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    #[test]
    fn runtime_variant_maps_to_typed_evaluation_error() {
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        Python::attach(|py| {
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
        pyo3::Python::initialize();
        let py_err = evaluation_error_to_pyerr(err);
        Python::attach(|py| {
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
    fn non_finite_score_maps_to_evaluation_error_naming_the_row() {
        assert_maps_to_evaluation_error(
            EvalErr::NonFiniteScore {
                sample: 5,
                value: f64::NEG_INFINITY,
            },
            "x_train row 5",
        );
    }

    #[test]
    fn label_count_maps_to_evaluation_error() {
        assert_maps_to_evaluation_error(
            EvalErr::LabelCount {
                labels: 2,
                samples: 3,
            },
            "contract C-8",
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
        pyo3::Python::initialize();
        let py_err = evaluation_error_to_pyerr(EvalErr::Backend(BackendError::Cunqa(
            "qraise failed".to_string(),
        )));
        Python::attach(|py| {
            assert!(py_err.is_instance_of::<CunqaError>(py));
            assert!(
                !py_err.is_instance_of::<EvaluationError>(py),
                "a wrapped backend failure must keep its own class"
            );
        });
    }

    #[test]
    fn observable_non_external_maps_to_evaluation_error() {
        // A native observable failure has no original Python class to preserve,
        // so it surfaces as the typed `polypus.EvaluationError` with its message.
        assert_maps_to_evaluation_error(
            EvalErr::Observable(ObservableError::Invalid("coupling i == j".to_string())),
            "coupling i == j",
        );
    }

    #[test]
    fn qiskit_binding_error_maps_to_evaluation_error_with_its_cause() {
        // Issue #218: Qiskit failing to bind a circuit's parameters is an
        // evaluation failure — no backend ran — so it becomes exactly
        // `polypus.EvaluationError`, with the Qiskit class in the message and
        // the original chained.
        pyo3::Python::initialize();
        let original = Python::attach(|py| super::tests::fake_error(py, "QiskitError", "bad"));
        let original_value = Python::attach(|py| original.value(py).clone().unbind());
        let py_err = evaluation_error_to_pyerr(EvalErr::Qiskit(original.into()));
        Python::attach(|py| {
            assert!(
                py_err.get_type(py).is(py.get_type::<EvaluationError>()),
                "exactly polypus.EvaluationError: {py_err}"
            );
            assert!(py_err.is_instance_of::<PolypusError>(py));
            assert!(
                py_err
                    .to_string()
                    .contains("qiskit.exceptions.QiskitError: bad"),
                "{py_err}"
            );
            let cause = py_err.cause(py).expect("the Qiskit original is chained");
            assert!(cause.value(py).is(original_value.bind(py)));
        });
    }

    #[test]
    fn qiskit_variant_keeps_non_qiskit_and_c1_typed_errors() {
        // Same preservation rule as the seam: a `ValueError` raised by the binding
        // call (Qiskit's "Mismatching number of values") or a Qiskit class that is
        // also a `TypeError`-like C-1 type keeps its class.
        pyo3::Python::initialize();
        let cases: [(PyErr, &str); 3] = [
            (PyValueError::new_err("Mismatching number"), "ValueError"),
            (PyTypeError::new_err("Cannot assign object"), "TypeError"),
            (
                Python::attach(|py| super::tests::fake_error(py, "QiskitValueError", "x")),
                "QiskitValueError",
            ),
        ];
        for (err, class) in cases {
            let py_err = evaluation_error_to_pyerr(EvalErr::Qiskit(err.into()));
            Python::attach(|py| {
                assert_eq!(py_err.get_type(py).name().unwrap().to_string(), class);
                assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
            });
        }
    }

    #[test]
    fn python_variant_stays_verbatim_even_for_a_qiskit_class() {
        // A user callback can raise anything, Qiskit classes included; the
        // `Python` variant carries callbacks and is never retyped.
        pyo3::Python::initialize();
        let py_err = evaluation_error_to_pyerr(EvalErr::Python(
            Python::attach(|py| super::tests::fake_error(py, "QiskitError", "from a callback"))
                .into(),
        ));
        Python::attach(|py| {
            assert_eq!(
                py_err.get_type(py).name().unwrap().to_string(),
                "QiskitError"
            );
            assert!(!py_err.is_instance_of::<PolypusError>(py), "{py_err}");
            assert!(py_err.cause(py).is_none());
        });
    }

    #[test]
    fn observable_external_preserves_the_original_python_class() {
        // Twin of the `infrastructure_error_to_pyerr` fix: a Python exception
        // boxed by a callback observable must re-raise with its original class,
        // not be flattened into `polypus.EvaluationError`.
        pyo3::Python::initialize();
        let py_err =
            evaluation_error_to_pyerr(EvalErr::Observable(ObservableError::External(Box::new(
                DisplaySafePyErr::from(PyZeroDivisionError::new_err("callback divided by zero")),
            ))));
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyZeroDivisionError>(py),
                "the callback's original class must survive the FFI: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<EvaluationError>(py),
                "the original class must not be flattened into EvaluationError"
            );
            assert!(py_err.to_string().contains("callback divided by zero"));
        });
    }

    #[test]
    fn bare_py_err_in_observable_external_is_not_reraised_verbatim() {
        // Same rule as `External` on the backend side: only a boxed
        // `DisplaySafePyErr` keeps its class; a bare `PyErr` is flattened.
        assert_maps_to_evaluation_error(
            EvalErr::Observable(ObservableError::External(Box::new(
                PyZeroDivisionError::new_err("callback divided by zero"),
            ))),
            "ZeroDivisionError: callback divided by zero",
        );
    }
}

#[cfg(test)]
mod infrastructure_mapping_tests {
    //! New coverage that closes the gap this issue fixed. Unlike the two modules
    //! above, nothing was relocated here: `infrastructure_error_to_pyerr` simply
    //! had no dedicated tests, and its `Observable` arm silently discarded the
    //! original class of a Python exception raised inside a callback observable
    //! (mapping it to a generic `polypus.EvaluationError`). These pin every
    //! `InfrastructureError` variant's target class and message.
    //!
    //! Note the hierarchy split: `Backend` and non-`External` `Observable` land
    //! inside the `polypus.PolypusError` hierarchy, but `Cancelled`
    //! (`KeyboardInterrupt`), `IncompatiblePlanner` (`ValueError`), a verbatim
    //! `Python` exception, and `Observable(External)` are *native* Python
    //! exceptions by design — so the `PolypusError`-catchable assertion is made
    //! only where it is actually true.
    use super::*;
    use crate::infrastructure::InfrastructureError as InfraErr;
    use polypus_observable::ObservableError as ObsErr;
    use pyo3::exceptions::{
        PyKeyboardInterrupt, PyRuntimeError, PyValueError, PyZeroDivisionError,
    };
    use pyo3::prelude::*;
    use pyo3::PyTypeInfo;

    /// Assert `err` crosses the FFI as `E`, stays catchable as `PolypusError`,
    /// and keeps its message. Valid only for variants whose target class is in
    /// the `polypus.*` hierarchy (`Backend`, non-`External` `Observable`); the
    /// native Python targets are asserted inline, because asserting
    /// `PolypusError`-catchability on them would be a false check.
    fn assert_maps_to<E: PyTypeInfo>(err: InfraErr, expected_message: &str) {
        pyo3::Python::initialize();
        let py_err = infrastructure_error_to_pyerr(err);
        Python::attach(|py| {
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
    fn backend_variant_delegates_to_the_backend_mapping() {
        // `InfrastructureError::Backend` must not retype the wrapped failure: a
        // CUNQA error surfacing through the planner is still a `polypus.CunqaError`.
        pyo3::Python::initialize();
        let py_err = infrastructure_error_to_pyerr(InfraErr::Backend(InfraBackendError::Cunqa(
            "qraise failed".to_string(),
        )));
        Python::attach(|py| {
            assert!(py_err.is_instance_of::<CunqaError>(py));
            assert!(
                !py_err.is_instance_of::<EvaluationError>(py),
                "a wrapped backend failure must keep its own class"
            );
        });
    }

    #[test]
    fn observable_non_external_maps_to_evaluation_error() {
        // A native observable failure has no original Python class to preserve,
        // so it surfaces as the typed `polypus.EvaluationError`.
        assert_maps_to::<EvaluationError>(
            InfraErr::Observable(ObsErr::Invalid("coupling i == j".to_string())),
            "coupling i == j",
        );
    }

    #[test]
    fn observable_external_preserves_the_original_python_class() {
        // The central regression test for this issue: a Python exception boxed by
        // a callback observable must re-raise with its original class, not be
        // discarded into a generic `polypus.EvaluationError`. `PyZeroDivisionError`
        // is deliberately outside the `polypus.*` hierarchy so the check is
        // unambiguous.
        pyo3::Python::initialize();
        let py_err =
            infrastructure_error_to_pyerr(InfraErr::Observable(ObsErr::External(Box::new(
                DisplaySafePyErr::from(PyZeroDivisionError::new_err("callback divided by zero")),
            ))));
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyZeroDivisionError>(py),
                "a callback's ZeroDivisionError must re-raise with its original class: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<EvaluationError>(py),
                "the original class must not be flattened into EvaluationError"
            );
            assert!(py_err.to_string().contains("callback divided by zero"));
        });
    }

    #[test]
    fn external_backend_error_reraises_verbatim() {
        // A `check_signals` SIGINT (or any planner-raised Python exception) now
        // arrives as `Backend(BackendError::External(boxed DisplaySafePyErr))`; its
        // boxed original class must re-raise verbatim across the FFI.
        pyo3::Python::initialize();
        let py_err = infrastructure_error_to_pyerr(InfraErr::Backend(InfraBackendError::External(
            Box::new(DisplaySafePyErr::from(PyRuntimeError::new_err(
                "planner boom",
            ))),
        )));
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyRuntimeError>(py),
                "a planner Python exception must re-raise with its original class: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<PolypusError>(py),
                "a verbatim Python exception is not a polypus.* class"
            );
            assert!(py_err.to_string().contains("planner boom"));
        });
    }

    #[test]
    fn cancelled_maps_to_keyboard_interrupt() {
        // A cooperative cancel between waves surfaces as the same class a SIGINT
        // would: a native `KeyboardInterrupt`, not a `polypus.*` class.
        pyo3::Python::initialize();
        let py_err = infrastructure_error_to_pyerr(InfraErr::Cancelled);
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyKeyboardInterrupt>(py),
                "a cooperative cancel must surface as KeyboardInterrupt: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<PolypusError>(py),
                "KeyboardInterrupt is a native Python exception, not a polypus.* class"
            );
            assert!(py_err.to_string().contains("the run was cancelled"));
        });
    }

    #[test]
    fn incompatible_planner_maps_to_value_error() {
        // A configuration mismatch checked up front surfaces as a native
        // `ValueError`, message preserved.
        pyo3::Python::initialize();
        let py_err = infrastructure_error_to_pyerr(InfraErr::IncompatiblePlanner(
            "shot distribution unsupported".to_string(),
        ));
        Python::attach(|py| {
            assert!(
                py_err.is_instance_of::<PyValueError>(py),
                "an incompatible planner must surface as ValueError: {py_err}"
            );
            assert!(
                !py_err.is_instance_of::<PolypusError>(py),
                "ValueError is a native Python exception, not a polypus.* class"
            );
            assert!(py_err.to_string().contains("shot distribution unsupported"));
        });
    }
}
