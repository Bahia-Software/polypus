//! Attaching to the Python interpreter from code that may run when it is not
//! available.
//!
//! [`Python::attach`] is unsafe to rely on when the calling thread may not be
//! able to attach: during a GC traversal and before the interpreter is
//! initialized it panics, and once the interpreter has started shutting down it
//! either panics or attaches to a finalizing interpreter, which can hang the
//! thread. Shutdown looks like "not initialized" to PyO3 on every Python
//! version, because CPython clears its `initialized` flag right after marking
//! itself finalizing. `attach` panics there only if the process has not yet
//! made a fresh (non-reused) attach, since PyO3 checks initialization once.
//! Otherwise it goes ahead and attaches to the finalizing interpreter. That is
//! fine in straight-line code running inside a Python call, but not in two
//! places this workspace has:
//!
//! - **cleanup reached from `Drop`**, which can run at interpreter shutdown, where
//!   a panic aborts the process if another panic is already unwinding;
//! - **callbacks invoked while the thread is detached** (inside
//!   [`Python::detach`]), such as a signal-polling hook: `detach` resets the
//!   thread's attachment, so an `attach` in there is a *fresh* attach and gets
//!   the same checks, even though the surrounding Python call is still running.
//!   A daemon thread doing this at shutdown could panic across the FFI boundary,
//!   or hang.
//!
//! Such code attaches through [`attach_or`] (or [`attach_for_cleanup`]) instead,
//! choosing explicitly what happens when the interpreter cannot be reached.
//! Attaching is re-entrant only when the thread is *already* attached: then the
//! existing attachment is reused and none of the checks can fail.
//!
//! [`Python::try_attach`], which these helpers use, reports all of those states
//! as unavailable on every supported Python version. The one gap is a
//! check-then-attach race (shutdown starting between PyO3's check and the
//! attach itself), which PyO3 documents as best effort.

use std::fmt;

use pyo3::prelude::*;

/// Run `f` attached to the interpreter, or return `unavailable()` when the
/// interpreter cannot be attached to (finalizing, in a GC traversal, or not
/// initialized). Never panics on its own account.
///
/// When `f` returns an error that holds a [`PyErr`], remember that formatting a
/// `PyErr` (its `Display`) attaches again with [`Python::attach`]. If that error
/// may be formatted after this call returns, possibly while detached or at
/// shutdown, use [`attach_for_cleanup`] (or format the error inside `f`).
pub fn attach_or<R>(
    unavailable: impl FnOnce() -> R,
    f: impl for<'py> FnOnce(Python<'py>) -> R,
) -> R {
    Python::try_attach(f).unwrap_or_else(unavailable)
}

/// Why an [`attach_for_cleanup`] call failed. Holds no Python object: it can be
/// formatted, logged or dropped without attaching to the interpreter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupError {
    /// The interpreter could not be attached to, so the cleanup did not run.
    InterpreterUnavailable,
    /// The cleanup ran and raised; the message was formatted while attached.
    Failed(String),
}

impl fmt::Display for CleanupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CleanupError::InterpreterUnavailable => write!(
                f,
                "the Python interpreter is unavailable (not initialized, finalizing or in a \
                 GC traversal), so the cleanup did not run"
            ),
            CleanupError::Failed(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CleanupError {}

/// Run a cleanup step attached to the interpreter, for code that may run from
/// `Drop`. Never panics on its own account, and never lets a [`PyErr`] out: a
/// Python exception raised by `f` is formatted into [`CleanupError::Failed`]
/// while still attached, so the caller can log it at any time, including after
/// the interpreter has begun shutting down.
pub fn attach_for_cleanup(
    f: impl for<'py> FnOnce(Python<'py>) -> PyResult<()>,
) -> Result<(), CleanupError> {
    attach_or(
        || Err(CleanupError::InterpreterUnavailable),
        |py| f(py).map_err(|e| CleanupError::Failed(describe(&e))),
    )
}

/// Format `err` for a log line without panicking. `ToString::to_string` panics
/// when a `Display` impl returns an error, and a `PyErr`'s can (it fails when
/// the exception type's qualified name cannot be read, e.g. under memory
/// pressure). On the `Drop` path that panic could abort the process, so a
/// failed format falls back to a fixed message instead.
fn describe(err: &impl fmt::Display) -> String {
    use fmt::Write;
    let mut out = String::new();
    match write!(out, "{err}") {
        Ok(()) => out,
        Err(fmt::Error) => "<unformattable Python exception>".to_string(),
    }
}

/// Run the test `name` (its path relative to the crate root) alone in a fresh
/// copy of the current test binary, and fail if it fails there.
///
/// Used by the tests that need a process in which the interpreter was *never*
/// initialized: other tests in the same binary call [`Python::initialize`], and
/// that cannot be undone. The child is marked with [`FRESH_PROCESS_ENV`], which
/// the test checks to know it is the child.
#[cfg(test)]
pub(crate) fn run_in_fresh_process(name: &str) {
    let exe = std::env::current_exe().expect("the test binary path is available");
    let output = std::process::Command::new(exe)
        .args(["--exact", name, "--test-threads=1", "--nocapture"])
        .env(FRESH_PROCESS_ENV, "1")
        .output()
        .expect("the test binary can be re-executed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child run failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 passed"),
        "the child must actually run the test, not filter it out:\n{stdout}"
    );
}

/// Set in the child process started by [`run_in_fresh_process`].
#[cfg(test)]
pub(crate) const FRESH_PROCESS_ENV: &str = "POLYPUS_FRESH_PROCESS_CHILD";

/// Whether this process is a child started by [`run_in_fresh_process`].
#[cfg(test)]
pub(crate) fn is_fresh_process() -> bool {
    std::env::var_os(FRESH_PROCESS_ENV).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;

    /// With a live interpreter, `attach_or` runs `f` and ignores `unavailable`.
    #[test]
    fn attach_or_runs_f_when_the_interpreter_is_available() {
        Python::initialize();
        let value = attach_or(|| "unavailable", |_py| "attached");
        assert_eq!(value, "attached");
    }

    /// A Python exception raised by the cleanup comes back as an owned message,
    /// formatted while attached, not as a `PyErr`.
    #[test]
    fn attach_for_cleanup_formats_a_python_error_while_attached() {
        Python::initialize();
        let result = attach_for_cleanup(|_py| Err(PyValueError::new_err("release refused")));
        assert_eq!(
            result,
            Err(CleanupError::Failed(
                "ValueError: release refused".to_string()
            ))
        );
    }

    /// An error whose `Display` fails is described with the fixed fallback
    /// instead of panicking the way `to_string()` would. A real `PyErr` whose
    /// formatting fails cannot be produced reliably (on Python ≥ 3.11 PyO3 reads
    /// the type's qualified name straight from the type object), so a stand-in
    /// `Display` exercises the same branch.
    #[test]
    fn describe_falls_back_when_display_fails() {
        struct Unformattable;
        impl fmt::Display for Unformattable {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                Err(fmt::Error)
            }
        }
        assert_eq!(describe(&Unformattable), "<unformattable Python exception>");
        assert_eq!(describe(&"release refused"), "release refused");
    }

    /// Without an interpreter neither helper panics nor runs its closure; each
    /// takes its "unavailable" branch. "Not initialized" is the only unavailable
    /// state a test can produce deterministically. It is also how a shutting-down
    /// interpreter looks to PyO3, but a real shutdown cannot be reproduced from a
    /// Rust test. The test runs in a fresh process that never initializes Python.
    #[test]
    fn helpers_take_the_unavailable_branch_without_an_interpreter() {
        if !is_fresh_process() {
            run_in_fresh_process(
                "attach::tests::helpers_take_the_unavailable_branch_without_an_interpreter",
            );
            return;
        }
        let mut ran = false;
        let value = attach_or(
            || "unavailable",
            |_py| {
                ran = true;
                "attached"
            },
        );
        assert_eq!(value, "unavailable");
        let result = attach_for_cleanup(|_py| {
            ran = true;
            Ok(())
        });
        assert_eq!(result, Err(CleanupError::InterpreterUnavailable));
        assert!(!ran, "no closure may run without an interpreter");
    }
}
