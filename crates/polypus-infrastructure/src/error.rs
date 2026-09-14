//! Error type for the orchestration / execution backend layer.
//!
//! # Granularity decision
//!
//! `crates/polypus` uses **two** hand-written error enums rather than one per
//! module: [`BackendError`] here (backend construction, circuit execution,
//! infrastructure selection, Rust↔Python conversion) and `EvaluationError`
//! (in the `polypus` crate) for the optimizer oracle path. The feature-gated `QmioError` keeps its own rich enum (verified
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

use polypus_observable::ObservableError;
use pyo3::PyErr;

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
    /// A backend returned measurement results that violate the execution
    /// contract: the wrong number of count maps, an empty map (no measurements
    /// for a circuit that ran), a shot total that does not match the request
    /// (contract C-3 shot conservation), or a malformed non-bitstring key.
    /// Surfaced here rather than silently reduced downstream — an empty map, in
    /// particular, would otherwise become a `0.0` fitness with no error at all.
    InvalidResults(String),
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
            BackendError::InvalidResults(m) => write!(f, "backend returned invalid results: {m}"),
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

// `BackendError` deliberately implements no `From<_> for PyErr`: mapping it to
// the typed `polypus.*` exception hierarchy is the sole responsibility of the
// `polypus` FFI edge (`polypus::exceptions::backend_error_to_pyerr`), which owns
// those `#[pyclass]` types. Keeping the conversion out of this crate is what lets
// `polypus-infrastructure` stay a plain library with no exception-class coupling
// (ENGINEERING.md §9). The `Seam`/`Python` variants still carry a `PyErr`
// verbatim so the edge can re-raise the original Python exception unchanged.

/// A failure while a [`Planner`](super::Planner) executes circuits on a backend:
/// the backend itself failed, a cost-observable failed while reducing counts to
/// expectations, or a Python exception was raised — a `KeyboardInterrupt` from
/// the `check_signals` the planner runs between waves (ENGINEERING §3).
///
/// This is the planner's own error type. Per the crate's granularity decision it
/// deliberately does **not** implement `From<_> for PyErr`: the evaluation oracle
/// wraps it into an `EvaluationError` and `run_quantum_circuit` converts it at the
/// FFI edge (both in the `polypus` crate), keeping the exception hierarchy in one
/// place.
///
/// `Clone`/`Eq` are omitted: [`Python`](Self::Python) carries a [`PyErr`].
#[derive(Debug)]
pub enum InfrastructureError {
    /// The execution backend failed.
    Backend(BackendError),
    /// A native cost-observable failed while reducing counts to expectations.
    Observable(ObservableError),
    /// A Python exception raised inside the planner (a `check_signals` SIGINT
    /// between waves). Carried verbatim so its original type re-raises.
    Python(PyErr),
    /// The run was cooperatively cancelled between waves via the planner's
    /// [`CancelToken`](super::CancelToken): a wave is atomic, so cancellation
    /// takes effect at the next wave boundary, never mid-wave. Surfaces as a
    /// `KeyboardInterrupt`, the same class a SIGINT would.
    Cancelled,
    /// A [`Planner`](super::Planner)'s requirements are not met by the backend it
    /// was paired with (e.g. shot distribution requested from a backend that does
    /// not support it). A configuration error, checked up front. Surfaces as a
    /// `ValueError`.
    IncompatiblePlanner(String),
}

impl fmt::Display for InfrastructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfrastructureError::Backend(err) => write!(f, "{err}"),
            InfrastructureError::Observable(err) => {
                write!(f, "expectation evaluation failed: {err}")
            }
            InfrastructureError::Python(err) => write!(f, "{err}"),
            InfrastructureError::Cancelled => write!(f, "the run was cancelled"),
            InfrastructureError::IncompatiblePlanner(m) => {
                write!(f, "planner is incompatible with the backend: {m}")
            }
        }
    }
}

impl std::error::Error for InfrastructureError {}

impl From<BackendError> for InfrastructureError {
    fn from(err: BackendError) -> Self {
        InfrastructureError::Backend(err)
    }
}

impl From<ObservableError> for InfrastructureError {
    fn from(err: ObservableError) -> Self {
        InfrastructureError::Observable(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The FFI mapping of these variants to the typed `polypus.*` exception
    // classes now lives at the `polypus` edge (`exceptions::backend_error_to_pyerr`)
    // and is tested there; this crate owns only the pure-Rust `Display`, which is
    // what those Python messages are built from — so it is pinned here.

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
