//! The `trace` argument of `train` and `run_quantum_circuit` (contract C-12):
//! the edge's side of a run's call trace.
//!
//! The trace itself — its schema, how its file is created, written, capped and
//! closed — is [`polypus_orchestration::trace`], which needs no Python. This
//! module holds the two steps around a run: [`open`] creates the trace before
//! the backend exists, so its clock counts the backend's start-up too, and
//! [`close`] writes its `end` line once the run has returned, with the status
//! the outcome Python sees calls for ([`status_of`]); [`close_on_error`] does
//! the same for a step that fails between the two.
//!
//! The trace is never in the run's way: a path that cannot be created leaves
//! the run untraced, with one warning, and neither step changes the run's
//! result or its error. Both write to the file, so both do it with the GIL
//! released (docs/ENGINEERING.md §3).

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;

use crate::orchestration::trace::{CallTracer, RunStatus, TraceHeader, DEFAULT_MAX_BYTES};

/// The call trace of a run, if `explicit` names a path for it.
///
/// Without a path, `None`. With one, the trace file is created there, capped
/// at [`DEFAULT_MAX_BYTES`], with `header` as its first line; see
/// [`CallTracer::create`], which never overwrites a file, follows a symbolic
/// link or creates a directory. If that fails — something is already at the
/// path, its directory does not exist, the process may not write there — the
/// run goes on without a trace, after one warning that names the run, the
/// path and the error.
///
/// It writes to the file: call it with the GIL released.
pub(crate) fn open(explicit: Option<PathBuf>, header: TraceHeader) -> Option<Arc<CallTracer>> {
    let path = explicit?;
    let id = header.id.clone();
    match CallTracer::create(&path, header, DEFAULT_MAX_BYTES) {
        Ok(tracer) => Some(tracer),
        Err(err) => {
            log::warn!(
                "run {id}: call trace off: cannot create {}: {err}",
                path.display()
            );
            None
        }
    }
}

/// The status of a run's `end` line, from what the run returns to Python.
///
/// A result is [`RunStatus::Ok`]. A `KeyboardInterrupt` is
/// [`RunStatus::Cancelled`]: a Ctrl+C reaches the edge as one, the
/// `KeyboardInterrupt` of `check_signals` carried inside a backend error (not
/// as a cancelled run), and so does a cancelled run
/// (`InfrastructureError::Cancelled`) or an aborted one (`BackendError::Aborted`,
/// an interpreter that can no longer be reached). Any other exception is
/// [`RunStatus::Error`].
pub(crate) fn status_of<T>(py: Python<'_>, out: &PyResult<T>) -> RunStatus {
    match out {
        Ok(_) => RunStatus::Ok,
        Err(err) => error_status(py, err),
    }
}

/// Closes the run's trace, if it has one, with the status of `out`
/// ([`status_of`]), the outcome the run returns to Python.
///
/// The status is read with the GIL held; the `end` line is written and the
/// file closed with it released. Never fails: see [`CallTracer::finish`].
pub(crate) fn close<T>(py: Python<'_>, tracer: Option<&Arc<CallTracer>>, out: &PyResult<T>) {
    if let Some(tracer) = tracer {
        finish(py, tracer, status_of(py, out));
    }
}

/// Closes the run's trace, if it has one, as [`close`] does for a run that
/// failed with `err`, and hands `err` back unchanged: the `map_err` of a step
/// that fails after [`open`], such as creating the backend.
pub(crate) fn close_on_error(
    py: Python<'_>,
    tracer: Option<&Arc<CallTracer>>,
    err: PyErr,
) -> PyErr {
    if let Some(tracer) = tracer {
        finish(py, tracer, error_status(py, &err));
    }
    err
}

/// The status of a run that failed with `err`: see [`status_of`].
fn error_status(py: Python<'_>, err: &PyErr) -> RunStatus {
    if err.is_instance_of::<PyKeyboardInterrupt>(py) {
        RunStatus::Cancelled
    } else {
        RunStatus::Error
    }
}

/// Writes the `end` line of `tracer` with `status`, with the GIL released.
fn finish(py: Python<'_>, tracer: &CallTracer, status: RunStatus) {
    py.detach(|| tracer.finish(status));
}

#[cfg(test)]
mod tests {
    use polypus_infrastructure::{seam_error, BackendError, InfrastructureError};
    use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};

    use super::*;

    #[test]
    fn run_status_follows_the_python_error() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(status_of(py, &Ok::<(), PyErr>(())), RunStatus::Ok);
            assert_eq!(
                status_of::<()>(py, &Err(PyKeyboardInterrupt::new_err("^C"))),
                RunStatus::Cancelled
            );
            let cancelled =
                crate::exceptions::infrastructure_error_to_pyerr(InfrastructureError::Cancelled);
            assert_eq!(status_of::<()>(py, &Err(cancelled)), RunStatus::Cancelled);
            assert_eq!(
                status_of::<()>(py, &Err(PyValueError::new_err("boom"))),
                RunStatus::Error
            );
        });
    }

    /// A Ctrl+C reaches the edge inside a backend error, not as a cancelled
    /// run, and an aborted backend call as a `KeyboardInterrupt` of its own:
    /// both close the trace as cancelled, and `close_on_error` hands the error
    /// back as it came.
    #[test]
    fn an_interrupt_inside_a_backend_error_is_cancelled() {
        Python::initialize();
        Python::attach(|py| {
            let ctrl_c = crate::exceptions::infrastructure_error_to_pyerr(
                InfrastructureError::Backend(seam_error(PyKeyboardInterrupt::new_err("^C"))),
            );
            assert_eq!(status_of::<()>(py, &Err(ctrl_c)), RunStatus::Cancelled);
            let aborted =
                crate::exceptions::backend_error_to_pyerr(BackendError::Aborted("gone".into()));
            assert_eq!(status_of::<()>(py, &Err(aborted)), RunStatus::Cancelled);
            let err = close_on_error(py, None, PyValueError::new_err("boom"));
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert_eq!(err.value(py).to_string(), "boom");
        });
    }
}
