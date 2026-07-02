use polypus::logger;

fn main() {
    println!("Starting the logger example...");

    logger::init_experiment_logger(Some("my_experiment")).unwrap();

    log::error!("This is an error message");
    log::warn!("This is a warning message");
    log::info!("This is an info message");
    log::debug!("This is a debug message");
    log::trace!("This is a trace message");

    println!("Logger example finished.");
}
