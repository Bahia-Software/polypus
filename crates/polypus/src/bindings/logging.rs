//! Python bindings to install the process-wide Polypus logger.
//!
//! The Rust `log::*` macros used across the workspace (optimizers, backends, …)
//! are **no-ops until a global logger sink is installed**. These bindings let
//! Python callers install that sink so the internal log records are actually
//! emitted — to stdout/stderr or to a file. Without calling one of these once
//! per process, no log output (and no log file) is ever produced.

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::logger::{self, LogFormat, LogLevel, LogTarget, LoggerBuilder};

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

/// Shared error raised when a global logger was already installed.
///
/// The `log` crate only allows one global sink per process, so a second
/// `init_*` call is a caller error rather than something we can recover from.
fn already_installed_err() -> PyErr {
    PyRuntimeError::new_err(
        "a global logger is already installed; the Polypus logger can only be \
         initialized once per process",
    )
}

/// Install the process-wide Polypus logger.
///
/// Call this once, early in the program (before `train` / `run_quantum_circuit`),
/// otherwise the internal Rust `log::*` records are silently discarded.
///
/// # Arguments
/// * `level` — `off`/`error`/`warn`/`info`/`debug`/`trace` (optimizer progress is
///   emitted at `debug`, so pass `"debug"` to capture it).
/// * `format` — `"text"` or `"json"`.
/// * `file` — path to a log file; when omitted, logs go to stdout. Missing parent
///   directories are created automatically. The file is opened in append mode.
/// * `timestamp` / `thread_id` / `module_path` — per-line annotations.
#[pyfunction]
#[pyo3(signature = (
    level = "info",
    format = "text",
    file = None,
    timestamp = true,
    thread_id = false,
    module_path = false,
))]
pub fn init_logger(
    level: &str,
    format: &str,
    file: Option<String>,
    timestamp: bool,
    thread_id: bool,
    module_path: bool,
) -> PyResult<()> {
    let target = match file {
        Some(path) => {
            let path = PathBuf::from(path);
            // `OpenOptions` does not create missing parent directories, so make
            // sure they exist before the logger tries to open the file.
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        PyValueError::new_err(format!(
                            "failed to create log directory {parent:?}: {e}"
                        ))
                    })?;
                }
            }
            LogTarget::File(path)
        }
        None => LogTarget::Stdout,
    };

    LoggerBuilder::new()
        .level(parse_level(level)?)
        .format(parse_format(format)?)
        .target(target)
        .include_timestamp(timestamp)
        .include_thread_id(thread_id)
        .include_module_path(module_path)
        .init()
        .map_err(|_| already_installed_err())
}

/// Install a per-run file logger under `logs/<name>_<timestamp>.log`.
///
/// Convenience for experiment scripts: each run gets its own fully annotated
/// text log file instead of overwriting the previous one. Logs at `info` level;
/// use [`init_logger`] with `level="debug"` to also capture optimizer progress.
#[pyfunction]
#[pyo3(signature = (name = None))]
pub fn init_experiment_logger(name: Option<&str>) -> PyResult<()> {
    logger::init_experiment_logger(name).map_err(|_| already_installed_err())
}
