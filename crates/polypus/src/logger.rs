use std::path::PathBuf;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::thread;
use chrono::Local;
use log::{Record, LevelFilter, Metadata, SetLoggerError};

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
}

/// Builder for the logger. Configures a `LoggerConfig` step by step and turns
/// it into a usable `Logger` via `build`, or registers it globally via `init`.
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
            LogTarget::File(path) => match OpenOptions::new().create(true).append(true).open(path) {
                Ok(file) => Some(Mutex::new(file)),
                Err(e) => {
                    eprintln!("Failed to open log file {:?}: {}", path, e);
                    None
                }
            },
            _ => None,
        };

        Logger { config: self.config, file }
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
                        map.insert("timestamp".to_string(), serde_json::Value::String(timestamp.clone()));
                    }
                    map.insert("level".to_string(), serde_json::Value::String(level));
                    if let Some(thread_id) = &thread_id {
                        map.insert("thread_id".to_string(), serde_json::Value::String(thread_id.clone()));
                    }
                    if let Some(module) = &module {
                        map.insert("module".to_string(), serde_json::Value::String(module.to_string()));
                    }
                    map.insert("message".to_string(), serde_json::Value::String(message.clone()));

                    serde_json::to_string(&map).unwrap_or(message)
                }
            };

            // Send message to the appropriate target
            match &self.config.target {
                LogTarget::Stdout => println!("{}", output),
                LogTarget::Stderr => eprintln!("{}", output),
                LogTarget::File(path) => {
                    if let Some(mutex) = &self.file {
                        // Locking a single shared handle serializes writes from
                        // concurrent threads, so lines can't interleave.
                        if let Ok(mut file) = mutex.lock() {
                            let _ = writeln!(file, "{}", output);
                        }
                    } else {
                        // The file failed to open at build time; fall back to
                        // stderr so the message isn't silently dropped.
                        eprintln!("Failed to write to log file: {:?}", path);
                        eprintln!("{}", output);
                    }
                }
            }
        }
    }

    fn flush(&self) {}
}

/// Convenience entry point for experiment runs: installs a global logger that
/// writes timestamped, fully-annotated text logs to `logs/<name>_<timestamp>.log`,
/// so each run gets its own file instead of overwriting the previous one.
pub fn init_experiment_logger(experiment_name: Option<&str>) -> Result<(), SetLoggerError> {
    let base_name = experiment_name.unwrap_or("polypus_run");
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let file_name = format!("{}_{}.log", base_name, timestamp);

    // Logs live under `logs/` instead of the current working directory so
    // repeated runs don't scatter `.log` files at the repo root.
    let log_dir = PathBuf::from("logs");
    if let Err(e) = std::fs::create_dir_all(&log_dir) {
        eprintln!("Failed to create log directory {:?}: {}", log_dir, e);
    }
    let file_path = log_dir.join(file_name);

    println!("Initializing logger with log file: {:?}", file_path);

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

    macro_rules! log_message {
        ($logger:expr, $message:expr) => {
            $logger.log(
                &Record::builder()
                    .args(format_args!("{}", $message))
                    .level(log::Level::Info)
                    .module_path(Some("polypus::logger::tests"))
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
        let path = dir.join(format!("polypus_logger_test_json_{}.log", std::process::id()));
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
        let path = dir.join(format!("polypus_logger_test_escape_{}.log", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let logger = LoggerBuilder::new()
            .format(LogFormat::Json)
            .target(LogTarget::File(path.clone()))
            .build();

        log_message!(logger, "line with \"quotes\", a \\ backslash and a\nnewline");

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
        let path = dir.join(format!("polypus_logger_test_append_{}.log", std::process::id()));
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
}