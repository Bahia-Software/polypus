//! Error types for the execution-backend layer — **pyo3-free**.
//!
//! # Granularity
//!
//! Two hand-written enums, no `thiserror`, matching the workspace's per-crate
//! `error.rs` style (`polypus-circuit`, `polypus-optimizers`): [`BackendError`]
//! (backend construction, circuit execution, data conversion) and
//! [`InfrastructureError`] (a [`Planner`](crate::Planner) running circuits on a
//! backend).
//!
//! # No PyO3, and no `From<_> for PyErr`
//!
//! This crate carries **no** PyO3 dependency, so — unlike the pre-extraction
//! `polypus-infrastructure::BackendError` — there is no `Seam(PyErr)` variant.
//! A failure that originated in Python (or in any provider SDK) crosses this
//! boundary type-erased in [`BackendError::External`], a
//! `Box<dyn std::error::Error + Send + Sync>`. The Polypus FFI edge
//! (`polypus::exceptions`) downcasts it back to the original `PyErr` and
//! re-raises it verbatim, exactly as the old `Seam` variant did — the difference
//! is that the boxing is generic, so a third-party backend boxes *its own* error
//! type here without this crate ever naming it. Mapping either enum to a Python
//! exception stays the sole responsibility of the FFI edge.

use std::error::Error;
use std::fmt;

use polypus_observable::ObservableError;

/// Failure of a quantum-execution backend or of backend construction.
///
/// `Clone`/`Eq` are intentionally omitted: [`BackendError::External`] carries a
/// boxed [`Error`], which is neither `Clone` nor `Eq`.
#[derive(Debug)]
pub enum BackendError {
    /// The requested infrastructure/backend name is not recognised. Surfaces as
    /// `ValueError` at the FFI edge (contract C-1, unknown infrastructure).
    UnknownInfrastructure {
        /// The rejected infrastructure string.
        name: String,
    },
    /// An algorithm was handed a number of circuits it cannot operate on
    /// (e.g. a shot-distributing planner requires exactly one circuit). Surfaces
    /// as `ValueError`.
    InvalidCircuitCount {
        /// The exact number of circuits the algorithm requires.
        expected: usize,
        /// The number of circuits actually supplied.
        got: usize,
    },
    /// A backend was asked to run a circuit representation it cannot execute
    /// (e.g. a [`Foreign`](crate::BoundCircuit::Foreign) provider object on a
    /// GIL-free backend).
    UnsupportedCircuit(String),
    /// A backend returned measurement results that violate the execution
    /// contract: the wrong number of count maps, an empty map, a shot total that
    /// does not match the request (contract C-3), or a malformed non-bitstring
    /// key. Surfaced here rather than silently reduced downstream.
    InvalidResults(String),
    /// A native (pure-Rust) circuit failed to parse or to simulate.
    NativeCircuit(String),
    /// A CUNQA-specific failure originating in the Rust layer (family-handle
    /// extraction, allocation bookkeeping).
    Cunqa(String),
    /// Converting data across a boundary failed (building kwargs, converting
    /// counts, serialising a payload, …).
    Conversion(String),
    /// A backend stopped responding mid-call — it crashed, was killed, or hung
    /// (e.g. a subprocess worker that died with a pending request, or one still
    /// alive but wedged past its liveness deadline). This is the Fase-2 finding
    /// made explicit: distinct from a clean provider error in
    /// [`External`](Self::External), because the caller may choose to *retry with
    /// a fresh worker* rather than treat it as a definitive failure.
    Unresponsive(String),
    /// Any provider-specific failure, type-erased.
    ///
    /// The Polypus Python backends box a `PyErr` here so the FFI edge can
    /// downcast and re-raise the original Python exception verbatim (keeping its
    /// `ValueError`/`TypeError`/`KeyboardInterrupt` class); the QMIO backend
    /// boxes its own `QmioError`; a third-party backend boxes whatever error type
    /// it likes. This is what keeps `polypus-backend` free of any provider or
    /// PyO3 coupling in its error surface.
    External(Box<dyn Error + Send + Sync>),
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
            BackendError::Conversion(m) => write!(f, "data conversion failed: {m}"),
            BackendError::Unresponsive(m) => write!(f, "the backend stopped responding: {m}"),
            BackendError::External(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            BackendError::External(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// A failure while a [`Planner`](crate::Planner) executes circuits on a backend:
/// the backend itself failed, a cost-observable failed while reducing counts to
/// expectations, the run was cooperatively cancelled, or the planner/backend
/// pairing is invalid.
///
/// **pyo3-free.** The pre-extraction type had a `Python(PyErr)` variant carrying
/// the `KeyboardInterrupt` from the between-wave `check_signals`. That signal
/// check is now injected via the [`Interrupt`](crate::Interrupt) guard on the
/// [`CancelToken`](crate::CancelToken); a pending interrupt arrives here as
/// [`Backend`](Self::Backend)`(`[`BackendError::External`]`(boxed PyErr))` and is
/// re-raised verbatim at the FFI edge — so no PyO3 type appears on this enum.
///
/// `Clone`/`Eq` are omitted: [`Backend`](Self::Backend) may carry an
/// [`External`](BackendError::External) box.
#[derive(Debug)]
pub enum InfrastructureError {
    /// The execution backend failed.
    Backend(BackendError),
    /// A native cost-observable failed while reducing counts to expectations.
    Observable(ObservableError),
    /// The run was cooperatively cancelled between waves via the planner's
    /// [`CancelToken`](crate::CancelToken): a wave is atomic, so cancellation
    /// takes effect at the next wave boundary, never mid-wave. Surfaces as a
    /// `KeyboardInterrupt` at the FFI edge, the same class a SIGINT would.
    Cancelled,
    /// A [`Planner`](crate::Planner)'s requirements are not met by the backend it
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

    #[test]
    fn display_carries_the_variant_context() {
        assert_eq!(
            BackendError::Conversion("boom".to_string()).to_string(),
            "data conversion failed: boom"
        );
        assert_eq!(
            BackendError::Cunqa("boom".to_string()).to_string(),
            "CUNQA backend error: boom"
        );
        assert_eq!(
            BackendError::UnsupportedCircuit("boom".to_string()).to_string(),
            "boom"
        );
        assert_eq!(
            BackendError::Unresponsive("worker died (signal 9)".to_string()).to_string(),
            "the backend stopped responding: worker died (signal 9)"
        );
    }

    /// The `External` box round-trips: a boxed error Displays through and can be
    /// downcast back to its concrete type (the pattern the FFI edge uses to
    /// recover the original `PyErr`).
    #[test]
    fn external_is_transparent_and_downcastable() {
        #[derive(Debug)]
        struct Provider(&'static str);
        impl fmt::Display for Provider {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "provider says {}", self.0)
            }
        }
        impl std::error::Error for Provider {}

        let err = BackendError::External(Box::new(Provider("nope")));
        assert_eq!(err.to_string(), "provider says nope");
        // The edge recovers the concrete type via `source()`.
        let source = std::error::Error::source(&err).expect("External has a source");
        assert!(source.downcast_ref::<Provider>().is_some());
    }

    #[test]
    fn backend_error_converts_into_infrastructure_error() {
        let infra: InfrastructureError = BackendError::Cunqa("x".to_string()).into();
        assert!(matches!(infra, InfrastructureError::Backend(_)));
    }
}
