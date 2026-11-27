//! The gate-parallel threshold's once-per-process resolution emits its records
//! after leaving its one-time initialiser (docs/ENGINEERING.md §9).
//!
//! A log sink may block: the Python bridge (`polypus.log_to_python()`) waits
//! for the GIL, and the bindings ask for the resolution with the GIL held
//! (`warn_if_using_default_threshold`), so a record emitted inside the
//! initialiser would deadlock the two. The probe logger below stands in for
//! that sink, with a second thread asking for the resolution in the place of
//! the thread holding the GIL. `log::set_logger` is global per process, so this
//! lives in its own test binary with a single test.

#![cfg(feature = "parallel")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use log::{LevelFilter, Log, Metadata, Record};

/// How long a record waits for the second thread before calling it blocked:
/// far more than reading a memoised value takes, even on a loaded runner.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// On every record of the calibration module, asks for the resolution from a
/// fresh thread and waits for the answer, for up to [`PROBE_TIMEOUT`]. If the
/// record was emitted inside the initialiser, that thread waits for it to
/// finish and the record counts as blocked; after the timeout the record goes
/// on, the initialiser finishes and so does the thread, so the test fails
/// instead of hanging.
struct Probe {
    records: AtomicUsize,
    blocked: AtomicUsize,
}

impl Log for Probe {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        if !record.target().starts_with("polypus_sim::calibration") {
            return;
        }
        self.records.fetch_add(1, Ordering::SeqCst);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = polypus_sim::resolved_fallback_reason();
            let _ = done.send(());
        });
        if finished.recv_timeout(PROBE_TIMEOUT).is_err() {
            self.blocked.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn flush(&self) {}
}

static PROBE: Probe = Probe {
    records: AtomicUsize::new(0),
    blocked: AtomicUsize::new(0),
};

#[test]
fn the_resolution_logs_after_leaving_its_initialiser() {
    log::set_logger(&PROBE).expect("the only logger this test binary installs");
    log::set_max_level(LevelFilter::Trace);

    // Whatever the cache on this machine holds, the resolution logs which way
    // it went: a cached threshold, an uncalibrated key or a fallback.
    let _ = polypus_sim::resolved_fallback_reason();

    assert!(
        PROBE.records.load(Ordering::SeqCst) >= 1,
        "the resolution emits at least one record, or this test checks nothing"
    );
    assert_eq!(
        PROBE.blocked.load(Ordering::SeqCst),
        0,
        "a record was emitted inside the resolution's initialiser"
    );
}
