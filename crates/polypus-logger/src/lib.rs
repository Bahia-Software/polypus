//! Shared logging sink for the Polypus workspace.
//!
//! This crate provides a concrete [`log::Log`] implementation ([`Logger`]) plus
//! a [`LoggerBuilder`] to configure it (level, text/JSON format, stdout/stderr/
//! file target, timestamp/thread/module annotations) and install it as the
//! process-wide logger.
//!
//! # Design: facade vs. sink
//!
//! Library crates (`polypus-circuit`, `polypus-sim`, `polypus-physics`,
//! `polypus-optimizers`, …) **must not** depend on this crate. They emit log
//! records through the [`log`] facade (`log::info!`, `log::debug!`, …), which is
//! a no-op until a logger is installed. Only the *application* layer — the
//! `polypus` crate, binaries, examples or benches — depends on `polypus-logger`
//! to install the sink **once per process** via [`LoggerBuilder::init`].
//!
//! # Example
//!
//! ```no_run
//! use polypus_logger::{LoggerBuilder, LogLevel, LogFormat, LogTarget};
//!
//! LoggerBuilder::new()
//!     .level(LogLevel::Info)
//!     .format(LogFormat::Text)
//!     .target(LogTarget::Stdout)
//!     .init()
//!     .expect("a global logger was already installed");
//!
//! log::info!("logger ready");
//! ```

use chrono::Local;
use log::{LevelFilter, Metadata, Record, SetLoggerError};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;

/// Process-local counter that disambiguates default log file names created
/// within the *same* process (and even the same clock-second), complementing
/// the OS pid which disambiguates *across* concurrent processes. See
/// [`default_log_path`].
static LOG_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Level of detail for the logs.
///
/// Ordered from least to most verbose (`Off` < `Error` < ... < `Trace`). The
/// derived `Ord` relies on this order: `Logger::enabled` compares a record's
/// level against the configured one to decide whether to emit it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

/// Format of the log messages.
///
/// `Json` is serialized with `serde_json`, so message content is always
/// escaped correctly and the output is guaranteed to be valid JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
}

/// Target for the log output.
///
/// `File` holds the destination path; the actual file handle is opened once
/// by `LoggerBuilder::build` and kept alive on the resulting `Logger`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogTarget {
    Stdout,
    Stderr,
    File(PathBuf),
}

/// Configuration for the logger.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Minimum severity that gets logged; anything less severe is dropped.
    pub level: LogLevel,
    pub format: LogFormat,
    pub target: LogTarget,
    /// Whether to prefix each line with a `%Y-%m-%d %H:%M:%S` local timestamp.
    pub include_timestamp: bool,
    /// Whether to include the emitting thread's `ThreadId`.
    pub include_thread_id: bool,
    /// Whether to include the originating module path (e.g. `polypus::logger`).
    pub include_module_path: bool,
}

/// Default
impl Default for LoggerConfig {
    fn default() -> Self {
        LoggerConfig {
            level: LogLevel::Info,
            format: LogFormat::Text,
            target: LogTarget::Stdout,
            include_timestamp: true,
            include_thread_id: false,
            include_module_path: false,
        }
    }
}

/// Principal struct for the logger.
///
/// Implements `log::Log` so it can be registered as the global logger via
/// `LoggerBuilder::init`.
pub struct Logger {
    config: LoggerConfig,
    /// Opened once at build time for `LogTarget::File`; `None` for other targets
    /// or if opening the file failed. Reusing a single handle (behind a `Mutex`)
    /// avoids an open/close syscall per log line and keeps concurrent writes
    /// from different threads from interleaving mid-line.
    file: Option<Mutex<File>>,
    /// Set once the first file-write error has been reported to stderr. Because
    /// `log::Log::log` returns `()`, a write failure (full disk, broken pipe,
    /// revoked permissions) cannot be propagated to the caller; we surface it
    /// out-of-band on stderr, but only **once**, so a persistent failure doesn't
    /// emit one error line per dropped record.
    write_error_reported: AtomicBool,
    /// Same one-shot guard for the (effectively unreachable) JSON serialization
    /// fallback, so it can't flood stderr either.
    json_error_reported: AtomicBool,
}

impl Logger {
    /// Report a file-write failure to stderr exactly once. `log::Log::log`
    /// returns `()`, so stderr is the only channel left to surface the error
    /// without dropping it silently — but we report only the first occurrence to
    /// avoid flooding stderr when the underlying fault (e.g. a full disk)
    /// persists across every subsequent record.
    fn report_write_error(&self, path: &Path, err: &std::io::Error) {
        if !self.write_error_reported.swap(true, Ordering::Relaxed) {
            eprintln!(
                "polypus-logger: failed to write to log file {path:?}: {err}; \
                 suppressing further write-error reports"
            );
        }
    }

    /// Report a JSON serialization failure to stderr exactly once. Serializing a
    /// `Map<String, String>` cannot actually fail, so this is defensive; if it
    /// ever does, we still fall back to the raw message rather than dropping the
    /// record.
    fn report_serialize_error(&self, err: &serde_json::Error) {
        if !self.json_error_reported.swap(true, Ordering::Relaxed) {
            eprintln!(
                "polypus-logger: failed to serialize a log record to JSON: {err}; \
                 falling back to the raw message and suppressing further reports"
            );
        }
    }
}

/// Builder for the logger. Configures a `LoggerConfig` step by step and turns
/// it into a usable `Logger` via `build`, or registers it globally via `init`.
///
/// This is the low-level API: the default [`LogTarget`] is [`LogTarget::Stdout`]
/// and an explicit `LogTarget::File(path)` is used **verbatim** (no timestamp is
/// appended). For a safe per-run default that never collides between runs and
/// never loses output to an uncaptured stdout, use the high-level entry points
/// ([`init_experiment_logger`], or `resolve_log_target(None, false, name)` +
/// [`default_log_path`]).
pub struct LoggerBuilder {
    config: LoggerConfig,
}

impl LoggerBuilder {
    /// Create a new logger builder with default configuration
    pub fn new() -> Self {
        LoggerBuilder {
            config: LoggerConfig::default(),
        }
    }

    /// Set the log level
    pub fn level(mut self, level: LogLevel) -> Self {
        self.config.level = level;
        self
    }

    /// Set the log format
    pub fn format(mut self, format: LogFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set the log target
    pub fn target(mut self, target: LogTarget) -> Self {
        self.config.target = target;
        self
    }

    /// Set whether to include timestamp in logs
    pub fn include_timestamp(mut self, include: bool) -> Self {
        self.config.include_timestamp = include;
        self
    }

    /// Set whether to include thread ID in logs
    pub fn include_thread_id(mut self, include: bool) -> Self {
        self.config.include_thread_id = include;
        self
    }

    /// Set whether to include module path in logs
    pub fn include_module_path(mut self, include: bool) -> Self {
        self.config.include_module_path = include;
        self
    }

    /// Build the logger with the specified configuration.
    ///
    /// For a `File` target, the file is opened here (once).
    /// If opening fails, we warn immediately and fall back to `None`;
    /// `Logger::log` then degrades to printing to stderr instead of
    /// retrying the failed open on every single line.
    pub fn build(self) -> Logger {
        let file = match &self.config.target {
            LogTarget::File(path) => {
                match OpenOptions::new().create(true).append(true).open(path) {
                    Ok(file) => Some(Mutex::new(file)),
                    Err(e) => {
                        eprintln!("Failed to open log file {:?}: {}", path, e);
                        None
                    }
                }
            }
            _ => None,
        };

        Logger {
            config: self.config,
            file,
            write_error_reported: AtomicBool::new(false),
            json_error_reported: AtomicBool::new(false),
        }
    }

    /// Build the logger and install it as the process-wide logger used by the
    /// `log` crate's macros (`log::info!`, etc.).
    ///
    /// This can only succeed once per process — `log::set_boxed_logger` errors
    /// if a global logger is already registered.
    pub fn init(self) -> Result<(), SetLoggerError> {
        let logger = self.build();

        // The `log` facade filters records against this before they ever reach
        // `Logger::log`, so levels above it are cheap to skip process-wide.
        let max_level_filter: LevelFilter = logger.config.level.into();

        log::set_boxed_logger(Box::new(logger)).map(|()| log::set_max_level(max_level_filter))
    }
}

impl Default for LoggerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Translate LogLevel to Rust's log::Level
impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Off => LevelFilter::Off,
            LogLevel::Error => LevelFilter::Error,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Info => LevelFilter::Info,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Trace => LevelFilter::Trace,
        }
    }
}

impl log::Log for Logger {
    /// Check if a message must be processed.
    ///
    /// `log::Level` is ordered from most to least severe (`Error` < ... < `Trace`),
    /// so "at or more severe than the configured level" is `record level <= configured level`.
    /// The `log` facade already checks the global max level before calling `log`,
    /// but this per-instance check is what makes `Logger` correct if ever used
    /// directly (not just through the global logger).
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Into::<LevelFilter>::into(self.config.level)
    }

    /// Build and show the log message
    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            // Recolect basic data, only computing what the configuration actually needs
            let timestamp = self
                .config
                .include_timestamp
                .then(|| Local::now().format("%Y-%m-%d %H:%M:%S").to_string());
            let thread_id = self
                .config
                .include_thread_id
                .then(|| format!("{:?}", thread::current().id()));
            let module = self
                .config
                .include_module_path
                .then(|| record.module_path().unwrap_or("unknown"));
            let level = record.level().to_string();
            let message = record.args().to_string();

            // Format the log message based on the configuration
            let output = match self.config.format {
                LogFormat::Text => {
                    let mut text = String::new();
                    if let Some(timestamp) = &timestamp {
                        text.push_str(&format!("[{}] ", timestamp));
                    }
                    if let Some(thread_id) = &thread_id {
                        text.push_str(&format!("[{}] ", thread_id));
                    }
                    text.push_str(&format!("[{}] ", level));
                    if let Some(module) = &module {
                        text.push_str(&format!("[{}] ", module));
                    }
                    text.push_str(&format!(" {}", message));
                    text
                }
                LogFormat::Json => {
                    // Built with serde_json so that quotes, backslashes,
                    // newlines, etc. in the message are escaped correctly
                    // and the result is always valid JSON.
                    let mut map = serde_json::Map::new();
                    if let Some(timestamp) = &timestamp {
                        map.insert(
                            "timestamp".to_string(),
                            serde_json::Value::String(timestamp.clone()),
                        );
                    }
                    map.insert("level".to_string(), serde_json::Value::String(level));
                    if let Some(thread_id) = &thread_id {
                        map.insert(
                            "thread_id".to_string(),
                            serde_json::Value::String(thread_id.clone()),
                        );
                    }
                    if let Some(module) = &module {
                        map.insert(
                            "module".to_string(),
                            serde_json::Value::String(module.to_string()),
                        );
                    }
                    map.insert(
                        "message".to_string(),
                        serde_json::Value::String(message.clone()),
                    );

                    serde_json::to_string(&map).unwrap_or_else(|e| {
                        self.report_serialize_error(&e);
                        message
                    })
                }
            };

            // Send message to the appropriate target
            match &self.config.target {
                LogTarget::Stdout => println!("{}", output),
                LogTarget::Stderr => eprintln!("{}", output),
                LogTarget::File(path) => match &self.file {
                    Some(mutex) => {
                        // Locking a single shared handle serializes writes from
                        // concurrent threads, so lines can't interleave. Recover
                        // a poisoned lock (a prior panic while holding it does
                        // not corrupt the file) rather than silently dropping
                        // the record.
                        let mut file = match mutex.lock() {
                            Ok(guard) => guard,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        if let Err(e) = writeln!(file, "{}", output) {
                            // Don't discard the error (`let _ = ...`): surface it
                            // on stderr once. See `report_write_error`.
                            self.report_write_error(path, &e);
                        }
                    }
                    None => {
                        // The file failed to open at build time (already reported
                        // once by `build`); fall back to stderr so the record
                        // isn't silently dropped.
                        eprintln!("{}", output);
                    }
                },
            }
        }
    }

    /// Flush the file target. `File` is currently unbuffered, so at the moment
    /// this is effectively a no-op, but implementing it keeps `flush` correct if
    /// a user-space buffer (e.g. `BufWriter`) is introduced later. Stdout/stderr
    /// flush on their own line-by-line.
    fn flush(&self) {
        if let (Some(mutex), LogTarget::File(path)) = (&self.file, &self.config.target) {
            let mut file = match mutex.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Err(e) = file.flush() {
                self.report_write_error(path, &e);
            }
        }
    }
}

/// Directory for auto-generated log files.
///
/// Honors the `POLYPUS_LOG_DIR` environment variable when it is set and
/// non-empty (so batch jobs, notebooks and array/grid runs can redirect logs
/// without touching code), falling back to `logs/` relative to the current
/// working directory.
pub fn default_log_dir() -> PathBuf {
    resolve_log_dir(std::env::var_os("POLYPUS_LOG_DIR"))
}

/// Pure core of [`default_log_dir`], split out so the precedence can be unit
/// tested without mutating the (process-global) environment.
fn resolve_log_dir(env_dir: Option<std::ffi::OsString>) -> PathBuf {
    match env_dir {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from("logs"),
    }
}

/// Assemble a default log file name from its varying parts.
///
/// Split out as a pure function so the uniqueness guarantee is unit-testable
/// (distinct `pid`s or `counter`s must yield distinct names) without spawning
/// real processes.
fn build_log_file_name(base: &str, timestamp: &str, pid: u32, counter: u64) -> String {
    format!("{base}_{timestamp}_{pid}_{counter}.log")
}

/// Compute a unique per-run log file path under [`default_log_dir`].
///
/// The file name is `<base>_<YYYYMMDD_HHMMSS>_<pid>_<counter>.log` (base
/// defaults to `polypus`). Uniqueness holds on two axes:
///
/// - the OS **pid** distinguishes concurrent processes (grid runs, `pytest-xdist`,
///   array jobs launched in the same second), and
/// - a process-local monotonic **counter** distinguishes repeated calls within a
///   single process, even in the same clock second.
///
/// The leading timestamp keeps files chronologically sortable. This never opens
/// or creates anything; the caller decides when to create the directory/file.
pub fn default_log_path(base_name: Option<&str>) -> PathBuf {
    let base = base_name.unwrap_or("polypus");
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let pid = std::process::id();
    let counter = LOG_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    default_log_dir().join(build_log_file_name(base, &timestamp, pid, counter))
}

/// Error returned by [`resolve_log_target`] when the requested output options
/// are mutually exclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    /// Both an explicit `file` and `console` output were requested.
    FileAndConsole,
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetError::FileAndConsole => write!(
                f,
                "cannot log to both an explicit file and the console; \
                 pass either `file` or `console`, not both"
            ),
        }
    }
}

impl std::error::Error for TargetError {}

/// Resolve the [`LogTarget`] for the high-level entry points from their output
/// options.
///
/// Rules:
/// - an explicit `file` path is used **verbatim** (no timestamp appended);
/// - `console` selects stdout (for local, interactive debugging);
/// - with **neither**, output goes to a unique per-run file under
///   [`default_log_dir`] (see [`default_log_path`]), named from `default_name`.
///   This is the safe default: it never collides between runs and is never lost
///   to an uncaptured stdout (the common failure mode from notebooks, background
///   jobs and job queues).
///
/// `file` together with `console` is contradictory and returns
/// [`TargetError::FileAndConsole`]. `default_name` only affects the
/// auto-generated default path.
pub fn resolve_log_target(
    file: Option<PathBuf>,
    console: bool,
    default_name: Option<&str>,
) -> Result<LogTarget, TargetError> {
    match (file, console) {
        (Some(_), true) => Err(TargetError::FileAndConsole),
        (Some(path), false) => Ok(LogTarget::File(path)),
        (None, true) => Ok(LogTarget::Stdout),
        (None, false) => Ok(LogTarget::File(default_log_path(default_name))),
    }
}

/// Convenience entry point for experiment runs: installs a global logger that
/// writes timestamped, fully-annotated text logs to a **unique per-run file**
/// under [`default_log_dir`], so concurrent or repeated runs never share a file.
///
/// The name is `<name>_<YYYYMMDD_HHMMSS>_<pid>_<counter>.log` (see
/// [`default_log_path`]): the OS pid keeps concurrent processes apart, and the
/// process-local counter keeps repeated calls apart — a plain second-resolution
/// timestamp is *not* enough for jobs launched in the same second.
pub fn init_experiment_logger(experiment_name: Option<&str>) -> Result<(), SetLoggerError> {
    let file_path = default_log_path(experiment_name);

    // `OpenOptions` (used by `build`) does not create missing parents.
    if let Some(parent) = file_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Failed to create log directory {parent:?}: {e}");
            }
        }
    }

    // Informational, on stderr so it never pollutes a program's stdout data.
    eprintln!("polypus-logger: writing logs to {file_path:?}");

    LoggerBuilder::new()
        .level(LogLevel::Info)
        .format(LogFormat::Text)
        .target(LogTarget::File(file_path))
        .include_timestamp(true)
        .include_thread_id(true)
        .include_module_path(true)
        .init()
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;
    use std::ffi::OsString;

    macro_rules! log_message {
        ($logger:expr, $message:expr) => {
            $logger.log(
                &Record::builder()
                    .args(format_args!("{}", $message))
                    .level(log::Level::Info)
                    .module_path(Some("polypus_logger::tests"))
                    .build(),
            )
        };
    }

    #[test]
    fn text_format_respects_flags() {
        let logger = LoggerBuilder::new()
            .format(LogFormat::Text)
            .target(LogTarget::Stdout)
            .include_timestamp(false)
            .include_thread_id(false)
            .include_module_path(false)
            .build();

        // Nothing to assert on stdout output directly, but ensure the config that
        // drives text formatting keeps the flags it was given.
        assert!(!logger.config.include_timestamp);
        assert!(!logger.config.include_thread_id);
        assert!(!logger.config.include_module_path);
    }

    /// Regression test for the bug where `Json` always included `thread_id`
    /// and `module` regardless of their flags, unlike `Text`.
    #[test]
    fn json_format_respects_flags_and_is_valid() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "polypus_logger_test_json_{}.log",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let logger = LoggerBuilder::new()
            .format(LogFormat::Json)
            .target(LogTarget::File(path.clone()))
            .include_timestamp(false)
            .include_thread_id(false)
            .include_module_path(true)
            .build();

        log_message!(logger, "hello");

        let contents = std::fs::read_to_string(&path).unwrap();
        let line = contents.lines().next().unwrap();
        let value: serde_json::Value = serde_json::from_str(line).unwrap();

        assert!(value.get("timestamp").is_none());
        assert!(value.get("thread_id").is_none());
        assert!(value.get("module").is_some());
        assert_eq!(value.get("message").unwrap(), "hello");

        let _ = std::fs::remove_file(&path);
    }

    /// Regression test for the manual `.replace("\"", "\\\"")` escaping, which
    /// left backslashes and newlines unescaped and produced invalid JSON.
    #[test]
    fn json_format_escapes_special_characters() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "polypus_logger_test_escape_{}.log",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let logger = LoggerBuilder::new()
            .format(LogFormat::Json)
            .target(LogTarget::File(path.clone()))
            .build();

        log_message!(
            logger,
            "line with \"quotes\", a \\ backslash and a\nnewline"
        );

        let contents = std::fs::read_to_string(&path).unwrap();
        let line = contents.lines().next().unwrap();
        let value: serde_json::Value = serde_json::from_str(line).unwrap();

        assert_eq!(
            value.get("message").unwrap(),
            "line with \"quotes\", a \\ backslash and a\nnewline"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Ensures the `File` target keeps a single open handle across multiple
    /// `log` calls, appending each line rather than truncating on every write.
    #[test]
    fn file_target_opens_once_and_appends_across_calls() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "polypus_logger_test_append_{}.log",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let logger = LoggerBuilder::new()
            .format(LogFormat::Text)
            .target(LogTarget::File(path.clone()))
            .build();

        log_message!(logger, "first");
        log_message!(logger, "second");
        log_message!(logger, "third");

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();

        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("first"));
        assert!(lines[1].contains("second"));
        assert!(lines[2].contains("third"));

        let _ = std::fs::remove_file(&path);
    }

    // --- default naming / target resolution (Problems 1 & 2) ----------------

    /// `POLYPUS_LOG_DIR` takes precedence; empty or unset falls back to `logs/`.
    /// Tested through the pure `resolve_log_dir` so it doesn't race on the
    /// process-global environment with other (parallel) tests.
    #[test]
    fn log_dir_honors_env_override_else_defaults_to_logs() {
        assert_eq!(resolve_log_dir(None), PathBuf::from("logs"));
        assert_eq!(
            resolve_log_dir(Some(OsString::from(""))),
            PathBuf::from("logs")
        );
        assert_eq!(
            resolve_log_dir(Some(OsString::from("/var/log/polypus"))),
            PathBuf::from("/var/log/polypus")
        );
    }

    /// The default name is unique across concurrent processes (distinct pids)
    /// and across rapid calls within one process (distinct counters, even at the
    /// same clock second). This is the core of Problem 1.
    #[test]
    fn default_log_name_is_unique_across_pids_and_calls() {
        let ts = "20260702_143512";
        // Different pids (concurrent processes) => different names.
        assert_ne!(
            build_log_file_name("polypus", ts, 100, 0),
            build_log_file_name("polypus", ts, 200, 0),
        );
        // Same pid + same second, next counter (rapid same-process calls)
        // => still different names.
        assert_ne!(
            build_log_file_name("polypus", ts, 100, 0),
            build_log_file_name("polypus", ts, 100, 1),
        );
        // Sanity check on the shape.
        assert_eq!(
            build_log_file_name("exp", ts, 42, 7),
            "exp_20260702_143512_42_7.log"
        );
    }

    /// Two rapid consecutive `default_log_path` calls in the same process must
    /// not collide (the monotonic counter guarantees it), mirroring two quick
    /// consecutive initializations.
    #[test]
    fn default_log_path_does_not_collide_on_consecutive_calls() {
        let p1 = default_log_path(Some("exp"));
        let p2 = default_log_path(Some("exp"));
        assert_ne!(p1, p2);
        let name = p1.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("exp_") && name.ends_with(".log"));
    }

    /// With neither `file` nor `console`, the default target is a **file**, not
    /// stdout — the fix for Problem 2 (log output lost to an uncaptured stdout).
    #[test]
    fn default_target_is_file_not_stdout() {
        // `resolve_log_target(None, false, None)` is the no-arg default.
        let target = resolve_log_target(None, false, None).unwrap();
        assert!(matches!(target, LogTarget::File(_)));
    }

    /// `console = true` opts back into stdout for local debugging, and an
    /// explicit file path is used verbatim.
    #[test]
    fn console_selects_stdout_and_explicit_file_is_verbatim() {
        assert_eq!(
            resolve_log_target(None, true, None).unwrap(),
            LogTarget::Stdout
        );
        assert_eq!(
            resolve_log_target(Some(PathBuf::from("logs/x.log")), false, None).unwrap(),
            LogTarget::File(PathBuf::from("logs/x.log"))
        );
    }

    /// Requesting both an explicit file and console is contradictory.
    #[test]
    fn file_and_console_is_rejected() {
        assert_eq!(
            resolve_log_target(Some(PathBuf::from("x.log")), true, None),
            Err(TargetError::FileAndConsole)
        );
    }

    /// If the file can't be opened at build time (missing parent directory), the
    /// logger degrades gracefully: it keeps no handle, doesn't panic when
    /// logging, and doesn't create the file behind our back.
    #[test]
    fn file_open_failure_degrades_gracefully() {
        let path = std::env::temp_dir()
            .join(format!("polypus_missing_dir_{}", std::process::id()))
            .join("sub")
            .join("f.log");
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());

        let logger = LoggerBuilder::new()
            .format(LogFormat::Text)
            .target(LogTarget::File(path.clone()))
            .build();

        assert!(logger.file.is_none());
        // Must not panic and must not create the file.
        log_message!(logger, "goes to stderr fallback");
        assert!(!path.exists());
    }
}
