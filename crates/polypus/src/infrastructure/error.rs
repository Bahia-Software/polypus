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
    /// An algorithm was handed a number of circuits it cannot operate on
    /// (e.g. `DistributeByShotsRun` requires exactly one circuit: it replicates
    /// that circuit across the QPUs and splits the shots, so an empty `qcs`
    /// would panic on `qcs[0]` and extra circuits would be silently dropped).
    /// Input-parameter validation, so it surfaces as `ValueError` — the same
    /// mapping as [`UnknownInfrastructure`](Self::UnknownInfrastructure).
    InvalidCircuitCount {
        /// The exact number of circuits the algorithm requires.
        expected: usize,
        /// The number of circuits actually supplied.
        got: usize,
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
            BackendError::InvalidCircuitCount { expected, got } => {
                write!(f, "expected exactly {expected} circuit(s), got {got}")
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
            BackendError::InvalidCircuitCount { expected, got } => {
                PyValueError::new_err(format!("expected exactly {expected} circuit(s), got {got}"))
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exceptions::PolypusError;
    use pyo3::prelude::*;
    use pyo3::PyTypeInfo;

    // The variants exercised here are *defense in depth*: the public entry
    // points reject the offending input before it can reach the code that builds
    // them (a Qiskit circuit never gets past `run_quantum_circuit`/`train` to the
    // native backend; a malformed seam response is not reachable without
    // monkeypatching `polypus_python`). Rather than contort an end-to-end
    // scenario to reach them, the enum variant is constructed directly and only
    // the mapping is asserted — the same pattern as
    // `running_quantum_circuits_local.rs::distribute_rejects_empty_qcs`.
    //
    // `is_instance_of` needs an initialised interpreter but no installed
    // package, so this stays inside the Python-runtime-free rule of
    // ENGINEERING.md §3: bare CPython is exactly what CI provides.

    /// Assert `err` crosses the FFI as an instance of the Python class `E`, that
    /// it is catchable as `polypus.PolypusError`, and that its message survives.
    fn assert_maps_to<E: PyTypeInfo>(err: BackendError, expected_message: &str) {
        pyo3::prepare_freethreaded_python();
        let py_err: PyErr = err.into();
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
        assert_maps_to::<PyNativeCircuitError>(
            BackendError::UnsupportedCircuit(
                "the native statevector backend cannot execute a Qiskit QuantumCircuit".to_string(),
            ),
            "cannot execute a Qiskit QuantumCircuit",
        );
    }

    #[test]
    fn native_circuit_maps_to_native_circuit_error() {
        assert_maps_to::<PyNativeCircuitError>(
            BackendError::NativeCircuit("could not parse OpenQASM 2.0".to_string()),
            "could not parse OpenQASM 2.0",
        );
    }

    #[test]
    fn conversion_maps_to_the_backend_error_base_class() {
        assert_maps_to::<PyBackendError>(
            BackendError::Conversion("counts were not convertible".to_string()),
            "counts were not convertible",
        );
    }

    #[test]
    fn cunqa_maps_to_cunqa_error() {
        assert_maps_to::<PyCunqaError>(
            BackendError::Cunqa("injected release failure".to_string()),
            "injected release failure",
        );
    }

    #[test]
    fn provider_errors_stay_catchable_as_backend_error() {
        // The hierarchy is what lets `except polypus.BackendError` catch every
        // backend-layer failure regardless of which provider raised it.
        pyo3::prepare_freethreaded_python();
        let cunqa: PyErr = BackendError::Cunqa("x".to_string()).into();
        let native: PyErr = BackendError::UnsupportedCircuit("y".to_string()).into();
        Python::with_gil(|py| {
            assert!(cunqa.is_instance_of::<PyBackendError>(py));
            assert!(native.is_instance_of::<PyBackendError>(py));
        });
    }

    #[test]
    fn display_carries_the_variant_context() {
        // `Display` is what the `PyErr` message is built from for the wrapped
        // variants, so it is pinned here too.
        assert_eq!(
            BackendError::Conversion("boom".to_string()).to_string(),
            "data conversion across the Python boundary failed: boom"
        );
        assert_eq!(
            BackendError::Cunqa("boom".to_string()).to_string(),
            "CUNQA backend error: boom"
        );
        assert_eq!(
            BackendError::UnsupportedCircuit("boom".to_string()).to_string(),
            "boom"
        );
    }
}
