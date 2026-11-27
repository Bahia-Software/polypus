//! A test logger shared by this crate's unit tests. `log::set_logger` is global
//! per process, so it is installed once for the whole test binary; each test
//! reads back only the lines naming its own run id.
//!
//! A line logged before the logger is installed is lost, so a test that reads
//! logs calls [`capture_logs`] first, before the code under test can log.
//! Installing it on the first read instead made such a test pass in the full
//! suite, where an earlier test had installed it, and fail when run alone.
//!
//! [`probe_contention`] makes the same logger check that a record is emitted
//! with no lock held (docs/ENGINEERING.md §9). It is inert unless a test turns
//! it on, and only for the records of that test's own thread.

use std::cell::RefCell;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

/// Records every `WARN` and `ERROR` line it receives, with its level.
struct Capture(Mutex<Vec<(log::Level, String)>>);

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            run_probe();
            let line = (record.level(), record.args().to_string());
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
    logged_with_levels_for(run_id)
        .into_iter()
        .map(|(_, line)| line)
        .collect()
}

/// The lines logged so far for `run_id`, each with its level.
///
/// # Panics
///
/// If [`capture_logs`] has not run yet: the lines would have been lost.
pub(crate) fn logged_with_levels_for(run_id: &str) -> Vec<(log::Level, String)> {
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
        .filter(|(_, line)| line.starts_with(&prefix))
        .cloned()
        .collect()
}

/// How long a probe waits for its contender before calling it blocked: far
/// more than an uncontended lock takes even on a loaded runner.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The probe [`probe_contention`] installs on the test's thread.
struct Probe {
    contender: Arc<dyn Fn() + Send + Sync>,
    records: usize,
    blocked: usize,
}

thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

/// What [`probe_contention`] saw.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Contention {
    /// Records the test's thread emitted while probed.
    pub(crate) records: usize,
    /// Of those, the ones whose contender was still waiting after
    /// [`PROBE_TIMEOUT`]: emitted while holding what the contender needs.
    pub(crate) blocked: usize,
}

/// Run `body` with every `WARN`/`ERROR` record this thread emits standing in for
/// a sink that blocks: before the record is captured, `contender` runs on a
/// fresh thread and the record waits for it, for up to [`PROBE_TIMEOUT`]. A
/// contender that takes the lock the code under test held while logging cannot
/// finish then, so the record counts as blocked; after the timeout the record
/// goes on, the lock is released and the contender finishes, so a failing test
/// fails instead of hanging. This is the deadlock the Python log bridge would
/// cause, with the GIL in the contender's place.
pub(crate) fn probe_contention(
    contender: impl Fn() + Send + Sync + 'static,
    body: impl FnOnce(),
) -> Contention {
    capture_logs();
    PROBE.with(|probe| {
        *probe.borrow_mut() = Some(Probe {
            contender: Arc::new(contender),
            records: 0,
            blocked: 0,
        });
    });
    body();
    let probe = PROBE.with(|probe| probe.borrow_mut().take());
    let probe = probe.expect("the probe stays installed until the body returns");
    Contention {
        records: probe.records,
        blocked: probe.blocked,
    }
}

/// The probing half of [`Capture::log`]: a no-op on a thread without a probe.
fn run_probe() {
    PROBE.with(|probe| {
        let mut probe = probe.borrow_mut();
        let Some(probe) = probe.as_mut() else {
            return;
        };
        probe.records += 1;
        let contender = Arc::clone(&probe.contender);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            contender();
            let _ = done.send(());
        });
        if finished.recv_timeout(PROBE_TIMEOUT).is_err() {
            probe.blocked += 1;
        }
    });
}
