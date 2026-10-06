//! The backend/planner error types now live in the pyo3-free `polypus-backend`
//! crate; this module re-exports them so existing `crate::error::…` paths keep
//! resolving. Mapping either enum to a typed `polypus.*` Python exception remains
//! the FFI edge's sole responsibility (`polypus::exceptions`).
//!
//! A Python exception from the `polypus_python` seam is carried verbatim in
//! [`BackendError::External`] (boxed `PyErr`) — the edge downcasts and re-raises
//! it, as `polypus.BackendError` if Qiskit raised it (contract C-1) — rather than in a PyO3-typed variant this crate cannot host without pulling
//! PyO3 into the contract. The feature-gated QMIO backend likewise boxes its own
//! `QmioError` into [`BackendError::External`], and so does this crate's own
//! [`EntropyError`] (a failed OS-entropy read while drawing a default seed).

pub use polypus_backend::error::{BackendError, InfrastructureError};

use std::error::Error;
use std::fmt;

/// The OS entropy source could not be read, so no default seed could be drawn
/// ([`random_seed`](crate::execution_config::random_seed)).
///
/// This crate's own error, not a [`BackendError`] variant: it reaches the
/// contract type-erased in [`BackendError::External`] (the same route as the
/// QMIO backend's `QmioError`), which the FFI edge surfaces as
/// `polypus.BackendError` — never as a panic. The underlying RNG error is kept
/// as [`source`](Error::source) and also quoted in the [`Display`](fmt::Display)
/// message, because that message is all a Python caller sees.
#[derive(Debug)]
pub struct EntropyError {
    source: Box<dyn Error + Send + Sync + 'static>,
}

impl EntropyError {
    /// Wrap the RNG failure that prevented drawing a seed.
    pub fn new(source: impl Error + Send + Sync + 'static) -> Self {
        EntropyError {
            source: Box::new(source),
        }
    }
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OS entropy source unavailable: cannot draw a default seed ({})",
            self.source
        )
    }
}

impl Error for EntropyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// An entropy failure crosses the backend contract type-erased, like any other
/// provider error this crate defines.
impl From<EntropyError> for BackendError {
    fn from(err: EntropyError) -> Self {
        BackendError::External(Box::new(err))
    }
}
