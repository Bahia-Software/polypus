//! A test logger shared by this crate's unit tests. `log::set_logger` is global
//! per process, so it is installed once for the whole test binary; each test
//! reads back only the lines naming its own run id.
//!
//! A line logged before the logger is installed is lost, so a test that reads
//! logs calls [`capture_logs`] first, before the code under test can log.
//! Installing it on the first read instead made such a test pass in the full
//! suite, where an earlier test had installed it, and fail when run alone.

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
static INSTALL: Once = Once::new();

/// Install [`CAPTURE`] as the process's logger, once. Every test that reads
/// [`logged_for`] calls it first, before anything it runs can log.
pub(crate) fn capture_logs() {
    INSTALL.call_once(|| {
        log::set_logger(&CAPTURE).expect("the only logger this test binary installs");
        log::set_max_level(log::LevelFilter::Warn);
    });
}

/// The lines logged so far for `run_id`.
///
/// # Panics
///
/// If [`capture_logs`] has not run yet: the lines would have been lost.
pub(crate) fn logged_for(run_id: &str) -> Vec<String> {
    assert!(
        INSTALL.is_completed(),
        "call test_log::capture_logs() at the start of the test, before anything logs"
    );
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
