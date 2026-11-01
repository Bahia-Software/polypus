//! The `trace` argument of `train` and `run_quantum_circuit` (contract C-12)
//! and the variables [`CALL_TRACE_ENV`] and [`CALL_TRACE_MAX_MB_ENV`]: the
//! edge's side of a run's call trace.
//!
//! The trace itself — its schema, how its file is created, written, capped and
//! closed — is [`polypus_orchestration::trace`], which needs no Python. This
//! module holds the two steps around a run: [`open`] creates the trace before
//! the backend exists, so its clock counts the backend's start-up too, and
//! [`close`] writes its `end` line once the run has returned, with the status
//! the outcome Python sees calls for ([`status_of`]); [`close_on_error`] does
//! the same for a step that fails between the two.
//!
//! A run is traced when its `trace` argument names a path or, without one,
//! when [`CALL_TRACE_ENV`] names a directory: each run then writes its trace
//! there, in a file of its own named by [`trace_file_name`].
//! [`CALL_TRACE_MAX_MB_ENV`] caps the size of every trace ([`parse_max_mb`]).
//!
//! The trace is never in the run's way: a path that cannot be created leaves
//! the run untraced, with one warning, and neither step changes the run's
//! result or its error. Both write to the file, so both do it with the GIL
//! released (docs/ENGINEERING.md §3).

use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;

use crate::orchestration::trace::{CallTracer, RunStatus, TraceHeader, DEFAULT_MAX_BYTES};

/// The variable that, for a run whose `trace` argument names no path, names
/// the directory its call trace is written to. Unset or empty, it is off.
pub(crate) const CALL_TRACE_ENV: &str = "POLYPUS_CALL_TRACE";

/// The variable that caps the size of a call trace, in whole MiB
/// ([`parse_max_mb`]); unset or invalid, the cap is [`DEFAULT_MAX_BYTES`].
pub(crate) const CALL_TRACE_MAX_MB_ENV: &str = "POLYPUS_CALL_TRACE_MAX_MB";

/// The longest id a trace file's name keeps whole ([`trace_file_name`]).
const MAX_FILE_ID_CHARS: usize = 100;

/// The `n` of the next trace file [`CALL_TRACE_ENV`] names in this process.
static NEXT_ENV_TRACE: AtomicU64 = AtomicU64::new(0);

/// The call trace of a run, if it has one.
///
/// The path is `explicit`, the run's `trace` argument, if it names one.
/// Otherwise, if [`CALL_TRACE_ENV`] is set and not empty, it names the
/// directory and the file is [`trace_file_name`]`(file_id, pid, n)` in it,
/// with `pid` this process's id and `n` the number of traces the variable
/// named before in the process, counted from 0. Otherwise there is no trace,
/// and nothing is logged.
///
/// The trace file is created with `header` as its first line, capped at the
/// size [`CALL_TRACE_MAX_MB_ENV`] sets ([`DEFAULT_MAX_BYTES`] if it is unset
/// or invalid; an invalid value is warned about once per process); see
/// [`CallTracer::create`], which never overwrites a file, follows a symbolic
/// link or creates a directory. If that fails — something is already at the
/// path, its directory does not exist, the process may not write there — the
/// run goes on without a trace, after one warning that names the run, the
/// path and the error.
///
/// It writes to the file: call it with the GIL released.
pub(crate) fn open(
    explicit: Option<PathBuf>,
    file_id: Option<&str>,
    header: TraceHeader,
) -> Option<Arc<CallTracer>> {
    let path = match explicit {
        Some(path) => path,
        None => {
            let dir = env::var_os(CALL_TRACE_ENV).filter(|dir| !dir.is_empty())?;
            let n = NEXT_ENV_TRACE.fetch_add(1, Ordering::Relaxed);
            PathBuf::from(dir).join(trace_file_name(file_id, std::process::id(), n))
        }
    };
    let id = header.id.clone();
    match CallTracer::create(&path, header, max_bytes()) {
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

/// The name of a trace file in the directory [`CALL_TRACE_ENV`] names:
/// `<id>-<pid>-<n>.jsonl`, the same for the same arguments.
///
/// `<id>` is `id`, or `run` if it is `None` or empty, with every character
/// outside `[A-Za-z0-9._-]` replaced by `_`, so the name stays inside the
/// directory whatever the id holds. An id longer than 100 characters keeps
/// its first 100, followed by `-` and eight hexadecimal digits of a hash of
/// the whole id (the low 32 bits of the 64-bit FNV-1a of its UTF-8 bytes), so
/// two ids that share their first 100 characters still get different names.
pub(crate) fn trace_file_name(id: Option<&str>, pid: u32, n: u64) -> String {
    let id = id.filter(|id| !id.is_empty()).unwrap_or("run");
    let kept: String = id
        .chars()
        .take(MAX_FILE_ID_CHARS)
        .map(file_name_char)
        .collect();
    if id.chars().count() > MAX_FILE_ID_CHARS {
        let hash = fnv1a_64(id.as_bytes()) & 0xffff_ffff;
        format!("{kept}-{hash:08x}-{pid}-{n}.jsonl")
    } else {
        format!("{kept}-{pid}-{n}.jsonl")
    }
}

/// The size cap [`CALL_TRACE_MAX_MB_ENV`] sets, in bytes, from its value
/// `raw`: a whole number of MiB (2^20 bytes, as `POLYPUS_MEM_BUDGET` counts
/// them), at least 1, written in ASCII digits alone, without a sign, spaces
/// or a unit.
///
/// `None` for any other `raw`, or for a size that does not fit in a `u64`.
pub(crate) fn parse_max_mb(raw: &str) -> Option<u64> {
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    raw.parse::<u64>()
        .ok()
        .filter(|&mib| mib >= 1)?
        .checked_mul(1 << 20)
}

/// The size cap of a trace: the one [`CALL_TRACE_MAX_MB_ENV`] sets
/// ([`parse_max_mb`]), or [`DEFAULT_MAX_BYTES`] if it is unset or invalid.
///
/// The variable is read at every call. An invalid value is logged with
/// `log::warn!` the first time one is seen in the process, and ignored.
fn max_bytes() -> u64 {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let Some(raw) = env::var_os(CALL_TRACE_MAX_MB_ENV) else {
        return DEFAULT_MAX_BYTES;
    };
    // A value that is not UTF-8 is kept, lossily, so it is reported as invalid.
    let raw = raw.to_string_lossy();
    parse_max_mb(&raw).unwrap_or_else(|| {
        if !WARNED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "ignoring {CALL_TRACE_MAX_MB_ENV}={raw:?}: expected a whole number of MiB >= 1; \
                 using {} MiB",
                DEFAULT_MAX_BYTES >> 20
            );
        }
        DEFAULT_MAX_BYTES
    })
}

/// `c` if a trace file's name may hold it (`[A-Za-z0-9._-]`), else `_`.
fn file_name_char(c: char) -> char {
    if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
        c
    } else {
        '_'
    }
}

/// The 64-bit FNV-1a hash of `bytes`.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
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

    #[test]
    fn trace_file_name_is_sanitized() {
        assert_eq!(
            trace_file_name(Some("run.1_a-B"), 42, 0),
            "run.1_a-B-42-0.jsonl"
        );
        assert_eq!(
            trace_file_name(Some("../../x"), 42, 1),
            ".._.._x-42-1.jsonl"
        );
        assert_eq!(
            trace_file_name(Some("naïve id"), 42, 2),
            "na_ve_id-42-2.jsonl"
        );
        assert_eq!(trace_file_name(Some(""), 42, 3), "run-42-3.jsonl");
        assert_eq!(trace_file_name(None, 42, 4), "run-42-4.jsonl");
        let long = "a".repeat(300);
        let name = trace_file_name(Some(&long), 42, 5);
        assert_eq!(&name[..100], "a".repeat(100));
        assert_eq!(name.len(), 100 + 1 + 8 + "-42-5.jsonl".len());
        assert!(
            name[100..].starts_with('-') && name[101..109].chars().all(|c| c.is_ascii_hexdigit())
        );
        assert_eq!(name, trace_file_name(Some(&long), 42, 5), "deterministic");
        let other = trace_file_name(Some(&format!("{}b", "a".repeat(299))), 42, 5);
        assert_ne!(
            name, other,
            "ids sharing their first 100 characters get different names"
        );
        for n in [&name, &other] {
            assert!(n
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')));
        }
    }

    /// The hash that tells long ids apart is FNV-1a, so a name is the same
    /// in every build and on every platform.
    #[test]
    fn file_name_hash_is_fnv1a_64() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
        let long = "a".repeat(300);
        let hash = fnv1a_64(long.as_bytes()) & 0xffff_ffff;
        assert_eq!(
            trace_file_name(Some(&long), 42, 5),
            format!("{}-{hash:08x}-42-5.jsonl", "a".repeat(100))
        );
    }

    #[test]
    fn max_mb_parses_whole_mebibytes() {
        assert_eq!(parse_max_mb("1"), Some(1 << 20));
        assert_eq!(parse_max_mb("512"), Some(512 << 20));
        // The largest cap whose size in bytes fits in 64 bits: 2^44 - 1 MiB.
        assert_eq!(parse_max_mb("17592186044415"), Some(((1 << 44) - 1) << 20));
        for bad in [
            "",
            "0",
            "-1",
            "1.5",
            "abc",
            " 1",
            "1MB",
            "17592186044416",
            "18446744073709551615",
        ] {
            assert_eq!(parse_max_mb(bad), None, "{bad:?}");
        }
    }
}
