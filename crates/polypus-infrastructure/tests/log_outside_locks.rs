//! Registering the built-in backends emits its records outside any one-time
//! initialiser (docs/ENGINEERING.md §9).
//!
//! A log sink may block: the Python bridge (`polypus.log_to_python()`) waits
//! for the GIL, and the bindings call `register_builtin_backends` both with the
//! GIL held (resolving a target) and without it (inside a run). A record
//! emitted inside a `Once` there would deadlock the two. The probe logger below
//! stands in for that sink, with a second thread registering the built-ins in
//! the place of the thread holding the GIL. `log::set_logger` is global per
//! process, so this lives in its own test binary with a single test. The
//! built-ins it registers exist only on Unix.

#![cfg(unix)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use log::{LevelFilter, Log, Metadata, Record};
use polypus_infrastructure::{is_registered, register_builtin_backends};

/// How long a record waits for the second thread before calling it blocked:
/// far more than an uncontended lock takes, even on a loaded runner.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// On every record of the backend registry, registers the built-ins from a
/// fresh thread and waits for it, for up to [`PROBE_TIMEOUT`]. If the record
/// was emitted inside a one-time initialiser (or with the registry locked),
/// that thread waits for it and the record counts as blocked; after the
/// timeout the record goes on and so does the thread, so the test fails
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
        if record.target() != "polypus_backend::registry" {
            return;
        }
        self.records.fetch_add(1, Ordering::SeqCst);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            register_builtin_backends();
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
fn the_builtin_backends_are_registered_outside_a_one_time_initialiser() {
    log::set_logger(&PROBE).expect("the only logger this test binary installs");
    log::set_max_level(LevelFilter::Trace);

    register_builtin_backends();

    assert!(is_registered("subprocess"));
    assert!(
        PROBE.records.load(Ordering::SeqCst) >= 1,
        "registering emits a record, or this test checks nothing"
    );
    assert_eq!(
        PROBE.blocked.load(Ordering::SeqCst),
        0,
        "a registration record was emitted inside a one-time initialiser"
    );
}
