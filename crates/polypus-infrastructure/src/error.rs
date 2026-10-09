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
//! [`EntropyError`] (a failed OS-entropy read while drawing a default seed or
//! the random bytes of a run id).

pub use polypus_backend::error::{BackendError, InfrastructureError};

use std::error::Error;
use std::fmt;

/// The OS entropy source could not be read, so no default seed
/// ([`random_seed`](crate::execution_config::random_seed)) or run-id bytes
/// ([`random_id_bytes`](crate::execution_config::random_id_bytes)) could be
/// drawn.
///
/// This crate's own error, not a [`BackendError`] variant: it reaches the
/// contract type-erased in [`BackendError::External`] (the same route as the
/// QMIO backend's `QmioError`), which the FFI edge surfaces as
/// `polypus.BackendError` — never as a panic. The underlying RNG error is kept
/// as [`source`](Error::source) and also quoted in the [`Display`](fmt::Display)
/// message, together with what was being drawn, because that message is all a
/// Python caller sees.
#[derive(Debug)]
pub struct EntropyError {
    purpose: Purpose,
    source: Box<dyn Error + Send + Sync + 'static>,
}

/// What a failed draw was for; only shapes the [`Display`](fmt::Display) message.
#[derive(Debug, Clone, Copy)]
enum Purpose {
    DefaultSeed,
    RunId,
}

impl EntropyError {
    /// Wrap the RNG failure that prevented drawing a default seed.
    pub fn new(source: impl Error + Send + Sync + 'static) -> Self {
        EntropyError {
            purpose: Purpose::DefaultSeed,
            source: Box::new(source),
        }
    }

    /// Wrap the RNG failure that prevented drawing the random bytes of a run
    /// id's UUID v4 suffix.
    pub fn for_run_id(source: impl Error + Send + Sync + 'static) -> Self {
        EntropyError {
            purpose: Purpose::RunId,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.purpose {
            Purpose::DefaultSeed => "a default seed",
            Purpose::RunId => "a run id",
        };
        write!(
            f,
            "OS entropy source unavailable: cannot draw {what} ({})",
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
