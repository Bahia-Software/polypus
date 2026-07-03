//! Python bindings to install the process-wide Polypus logger.
//!
//! The Rust `log::*` macros used across the workspace (optimizers, backends, …)
//! are **no-ops until a global logger sink is installed**. These bindings let
//! Python callers install that sink so the internal log records are actually
//! emitted. Without calling `init_logger` once per process, no log output (and
//! no log file) is ever produced.
//!
//! By default (`init_logger()` with no `file`/`console`) output goes to a
//! **unique per-run file** under the log directory, because Polypus is used
//! almost always from Python — notebooks, background processes and job queues,
//! where stdout is frequently not captured and log output would otherwise be
//! lost. Pass `console=True` to opt back into stdout for local debugging.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::logger::{self, LogFormat, LogLevel, LogTarget, LoggerBuilder};

/// Tracks whether *these bindings* have already installed the global logger, so
/// a second call can no-op **without** opening (and orphaning) a fresh log file
/// before `log::set_boxed_logger` rejects it. The underlying `log` crate allows
/// exactly one global sink per process; this is a fast-path guard in front of
/// that one-shot install.
static LOGGER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Parse the public level string into a [`LogLevel`].
fn parse_level(level: &str) -> PyResult<LogLevel> {
    match level.to_ascii_lowercase().as_str() {
        "off" => Ok(LogLevel::Off),
        "error" => Ok(LogLevel::Error),
        "warn" | "warning" => Ok(LogLevel::Warn),
        "info" => Ok(LogLevel::Info),
        "debug" => Ok(LogLevel::Debug),
        "trace" => Ok(LogLevel::Trace),
        other => Err(PyValueError::new_err(format!(
            "unknown log level '{other}'; expected one of \
             off/error/warn/info/debug/trace"
        ))),
    }
}

/// Parse the public format string into a [`LogFormat`].
fn parse_format(format: &str) -> PyResult<LogFormat> {
    match format.to_ascii_lowercase().as_str() {
        "text" => Ok(LogFormat::Text),
        "json" => Ok(LogFormat::Json),
        other => Err(PyValueError::new_err(format!(
            "unknown log format '{other}'; expected \"text\" or \"json\""
        ))),
    }
}

/// Create the parent directories of `path` if they don't exist yet.
///
/// `OpenOptions` (used by the logger to open the file) does not create missing
/// parents, so we do it here before the logger tries to open the file.
fn create_parent_dirs(path: &Path) -> PyResult<()> {
    logger::ensure_parent_dir(path).map_err(|e| {
        PyValueError::new_err(format!("failed to create log directory for {path:?}: {e}"))
    })
}

/// Emit a Python `UserWarning` that a logger is already installed.
fn warn_logger_already_installed(py: Python<'_>) -> PyResult<()> {
    let warnings = PyModule::import(py, "warnings")?;
    warnings.call_method1(
        "warn",
        (
            "a Polypus logger is already installed; ignoring this init_logger call. \
          The logger can be configured only once per process — restart the \
          interpreter/kernel to change its level, format, or target.",
        ),
    )?;
    Ok(())
}

/// Install the process-wide Polypus logger.
///
/// Call this once, early in the program (before `train` / `run_quantum_circuit`),
/// otherwise the internal Rust `log::*` records are silently discarded.
///
/// Returns the path of the log file that was written to, or `None` when logging
/// to the console or when a logger was already installed (see the note on
/// re-initialization below).
///
/// # Arguments
/// * `level` — `off`/`error`/`warn`/`info`/`debug`/`trace`. This is the
///   **runtime** level: it can only lower the compile-time ceiling, never raise
///   it (see "Compile-time level ceiling" below). Optimizer progress is emitted
///   at `debug`, so to capture it pass `"debug"` **and** build with
///   `--no-default-features --features debug-logs` — the default `info-logs`
///   build compiles `debug!`/`trace!` out entirely, so `level="debug"` alone
///   captures nothing.
/// * `format` — `"text"` or `"json"`.
/// * `file` — explicit path to a log file, used **verbatim** (no timestamp is
///   appended). Missing parent directories are created; the file is opened in
///   append mode. Mutually exclusive with `console` and `run_name`.
/// * `run_name` — base name for the auto-generated default file (default
///   `polypus`), producing `<run_name>_<YYYYMMDD_HHMMSS>_<pid>_<counter>.log`. It
///   only names that default file, so it is valid **only** when neither `file`
///   nor `console` is given; passing it alongside either raises `ValueError`
///   instead of being silently ignored.
/// * `console` — log to stdout instead of a file (for local, interactive
///   debugging). Mutually exclusive with `file` and `run_name`.
/// * `timestamp` / `thread_id` / `module_path` — per-line annotations.
///
/// # Default target
/// With neither `file` nor `console`, logs go to a **unique per-run file** under
/// the log directory (`POLYPUS_LOG_DIR` if set, else `logs/` relative to the
/// current working directory), named
/// `<run_name>_<YYYYMMDD_HHMMSS>_<pid>_<counter>.log`. This never collides
/// between concurrent or repeated runs and is never lost to an uncaptured stdout.
///
/// # Compile-time level ceiling
/// `level` is a *runtime* filter and can only lower the ceiling fixed at build
/// time by `polypus`'s `*-logs` Cargo features (default `info-logs`, forwarded to
/// `polypus-logger`). Under the default build, `debug!`/`trace!` records are
/// compiled out, so `init_logger(level="debug")` will **not** capture debug
/// output — including optimizer per-generation progress. `log` treats its
/// `max_level_*` features as mutually exclusive (a hard compile error if more
/// than one is active), so raising the ceiling means **disabling** the default
/// first, not adding to it:
/// `maturin develop --release --no-default-features --features "extension-module debug-logs"`.
/// Adding `debug-logs` on top of the default `info-logs` *without*
/// `--no-default-features` leaves both active at once and fails to build.
///
/// # Re-initialization
/// A global logger can be installed only once per process. A second call (e.g.
/// re-running a Jupyter cell, or repeated calls across tests in one interpreter)
/// is a **no-op with a `UserWarning`**, leaving the first sink in place; restart
/// the interpreter/kernel to change the level, format or target.
#[pyfunction]
#[pyo3(signature = (
    level = "info",
    format = "text",
    file = None,
    run_name = None,
    console = false,
    timestamp = true,
    thread_id = false,
    module_path = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn init_logger(
    py: Python<'_>,
    level: &str,
    format: &str,
    file: Option<String>,
    run_name: Option<&str>,
    console: bool,
    timestamp: bool,
    thread_id: bool,
    module_path: bool,
) -> PyResult<Option<String>> {
    // Parse/validate everything that can fail cheaply *before* claiming the
    // install slot or touching the filesystem. `resolve_log_target` also rejects
    // contradictory target options (e.g. `run_name` together with `file` or
    // `console`), surfacing them as `ValueError` rather than dropping the input.
    let level = parse_level(level)?;
    let format = parse_format(format)?;
    let target = logger::resolve_log_target(file.map(PathBuf::from), console, run_name)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Claim the one install slot. If a logger is already installed (Jupyter
    // re-run, repeated test calls), no-op with a warning *without* opening a
    // fresh file — otherwise every re-run would orphan an empty log file before
    // `set_boxed_logger` rejected it.
    if LOGGER_INSTALLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        warn_logger_already_installed(py)?;
        return Ok(None);
    }

    // We hold the slot: create the parent directory for the resolved file path
    // and remember it to return to the caller.
    let file_path = match &target {
        LogTarget::File(path) => {
            create_parent_dirs(path)?;
            Some(path.display().to_string())
        }
        _ => None,
    };

    let installed = LoggerBuilder::new()
        .level(level)
        .format(format)
        .target(target)
        .include_timestamp(timestamp)
        .include_thread_id(thread_id)
        .include_module_path(module_path)
        .init();

    match installed {
        Ok(()) => Ok(file_path),
        Err(_) => {
            // We won our own guard but `log` still rejected the install: a sink
            // was registered from outside these bindings (e.g. a direct Rust
            // `LoggerBuilder::init`). Keep the guard set (a sink does exist) and
            // degrade to a no-op with a warning rather than raising.
            warn_logger_already_installed(py)?;
            Ok(None)
        }
    }
}
