use polypus::logger::{self, LogFormat, LogLevel, LogTarget, LoggerBuilder};

fn main() {
    println!("Starting the logger example...");

    // No explicit file/console: resolves to a unique per-run file under
    // `logs/` (or `POLYPUS_LOG_DIR`), named `my_experiment_<timestamp>_<pid>_<counter>.log`.
    let target = logger::resolve_log_target(None, false, Some("my_experiment"))
        .expect("file and console were not both requested");
    if let LogTarget::File(path) = &target {
        logger::ensure_parent_dir(path).expect("failed to create the log directory");
        println!("Logging to {path:?}");
    }

    LoggerBuilder::new()
        .level(LogLevel::Info)
        .format(LogFormat::Text)
        .target(target)
        .include_timestamp(true)
        .include_thread_id(true)
        .include_module_path(true)
        .init()
        .expect("a global logger was already installed");

    log::error!("This is an error message");
    log::warn!("This is a warning message");
    log::info!("This is an info message");
    log::debug!("This is a debug message");
    log::trace!("This is a trace message");

    println!("Logger example finished.");
}
