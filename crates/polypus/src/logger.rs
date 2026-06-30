use std::path::PathBuf;
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;
use chrono::Local;
use log::{Record, LevelFilter, Metadata, SetLoggerError};

/// Level of detail for the logs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

/// Format of the log messages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
}

/// Target for the log output
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogTarget {
    Stdout,
    Stderr,
    File(PathBuf),
}

/// Configuration for the logger
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    pub target: LogTarget,
    pub include_timestamp: bool,
    pub include_thread_id: bool,
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

/// Principal struct for the logger
pub struct Logger {
    config: LoggerConfig,
}

/// Builder for the logger
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

    /// Build the logger with the specified configuration
    pub fn build(self) -> Logger {
        Logger { config: self.config }
    }

    pub fn init(self) -> Result<(), SetLoggerError> {
        let logger = self.build();

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
    /// Check if a message must be processed
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Into::<LevelFilter>::into(self.config.level)
    }

    /// Build and shoq the log message
    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            // Recolect basic data
            let now = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let level = record.level().to_string();
            let module = record.module_path().unwrap_or("unknown");
            let message = record.args().to_string();
            let thread_id = format!("{:?}", thread::current().id());

            // Format the log message based on the configuration
            let output = match self.config.format {
                LogFormat::Text => {
                    let mut text = String::new();
                    if self.config.include_timestamp {
                        text.push_str(&format!("[{}] ", now));
                    }
                    if self.config.include_thread_id {
                        text.push_str(&format!("[{}] ", thread_id));
                    }
                    text.push_str(&format!("[{}] ", level));
                    if self.config.include_module_path {
                        text.push_str(&format!("[{}] ", module));
                    }
                    text.push_str(&format!(" {}", message));
                    text
                }
                LogFormat::Json => {
                    let safe_message = message.replace("\"", "\\\"");

                    format!(
                        r#"{{"timestamp": "{}", "level": "{}", "thread_id": "{}", "module": "{}", "message": "{}"}}"#,
                        now, level, thread_id, module, safe_message
                    )
                }
            };

            // Send message to the appropriate target
            match &self.config.target {
                LogTarget::Stdout => println!("{}", output),
                LogTarget::Stderr => eprintln!("{}", output),
                LogTarget::File(path) => {
                    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                        let _ = writeln!(file, "{}", output);
                    } else {
                        eprintln!("Failed to write to log file: {:?}", path);
                        eprintln!("{}", output);
                    }
                }
            }
        }
    }

    fn flush(&self) {}
}

pub fn init_experiment_logger(experiment_name: Option<&str>) -> Result<(), SetLoggerError> {
    let base_name = experiment_name.unwrap_or("polypus_run");
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let file_name = format!("{}_{}.log", base_name, timestamp);

    println!("Initializing logger with log file: {}", file_name);

    LoggerBuilder::new()
        .level(LogLevel::Info)
        .format(LogFormat::Text)
        .target(LogTarget::File(std::path::PathBuf::from(file_name)))
        .include_timestamp(true)
        .include_thread_id(true)
        .include_module_path(true)
        .init()
}