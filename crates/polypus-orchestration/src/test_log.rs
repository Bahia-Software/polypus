//! A test logger shared by this crate's unit tests. `log::set_logger` is global
//! per process, so it is installed once for the whole test binary; each test
//! reads back only the lines naming its own run id.

use std::sync::{Mutex, Once};

/// Records every `WARN` and `ERROR` line it receives.
struct Capture(Mutex<Vec<String>>);

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let line = record.args().to_string();
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(line);
        }
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture(Mutex::new(Vec::new()));

/// Install [`CAPTURE`] (once) and return the lines logged so far for `run_id`.
pub(crate) fn logged_for(run_id: &str) -> Vec<String> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&CAPTURE).expect("the only logger this test binary installs");
        log::set_max_level(log::LevelFilter::Warn);
    });
    let prefix = format!("run {run_id}:");
    CAPTURE
        .0
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .filter(|line| line.starts_with(&prefix))
        .cloned()
        .collect()
}
