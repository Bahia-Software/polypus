//! The backend/planner error types now live in the pyo3-free `polypus-backend`
//! crate; this module re-exports them so existing `crate::error::…` paths keep
//! resolving. Mapping either enum to a typed `polypus.*` Python exception remains
//! the FFI edge's sole responsibility (`polypus::exceptions`).
//!
//! A Python exception from the `polypus_python` seam is carried verbatim in
//! [`BackendError::External`] (boxed `PyErr`) — the edge downcasts and re-raises
//! it — rather than in a PyO3-typed variant this crate cannot host without pulling
//! PyO3 into the contract. The feature-gated QMIO backend likewise boxes its own
//! `QmioError` into [`BackendError::External`].

pub use polypus_backend::error::{BackendError, InfrastructureError};
