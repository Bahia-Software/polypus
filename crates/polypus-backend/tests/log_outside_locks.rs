//! The memory-limit detection and the backend registry emit their records with
//! no lock held and outside their one-time initialiser (docs/ENGINEERING.md §9).
//!
//! A log sink may block: the Python bridge (`polypus.log_to_python()`) waits
//! for the GIL, and the bindings reach `active_budget` and `is_registered` with
//! the GIL held (`backend_compatibility`, resolving a registered target), so a
//! record emitted while holding what those need would deadlock the two. The
//! probe logger below stands in for that sink, with a second thread calling
//! the same accessor in the place of the thread holding the GIL.
//! `log::set_logger` is global per process, so this lives in its own test
//! binary.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Once};
use std::time::Duration;

use log::{LevelFilter, Log, Metadata, Record};
use polypus_backend::{
    active_budget, is_registered, register_backend, BackendBuildContext, BackendError,
    QuantumBackend,
};

/// How long a record waits for the second thread before calling it blocked:
/// far more than an uncontended lock takes, even on a loaded runner.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// What the probe saw for one module.
struct Seen {
    records: AtomicUsize,
    blocked: AtomicUsize,
}

impl Seen {
    const fn new() -> Self {
        Self {
            records: AtomicUsize::new(0),
            blocked: AtomicUsize::new(0),
        }
    }

    /// Run `contender` on a fresh thread and wait for it, for up to
    /// [`PROBE_TIMEOUT`]. If the record was emitted while holding what
    /// `contender` needs, it cannot finish and the record counts as blocked;
    /// after the timeout the record goes on, the lock is released and the
    /// thread finishes, so a test fails instead of hanging.
    fn probe(&self, contender: fn()) {
        self.records.fetch_add(1, Ordering::SeqCst);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            contender();
            let _ = done.send(());
        });
        if finished.recv_timeout(PROBE_TIMEOUT).is_err() {
            self.blocked.fetch_add(1, Ordering::SeqCst);
        }
    }
}

static BUDGET: Seen = Seen::new();
static REGISTRY: Seen = Seen::new();

struct Probe;

impl Log for Probe {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        match record.target() {
            "polypus_backend::mem_budget" => BUDGET.probe(|| {
                let _ = active_budget();
            }),
            "polypus_backend::registry" => REGISTRY.probe(|| {
                let _ = is_registered("log-outside-locks");
            }),
            _ => {}
        }
    }

    fn flush(&self) {}
}

fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&Probe).expect("the only logger this test binary installs");
        log::set_max_level(LevelFilter::Trace);
    });
}

#[test]
fn the_memory_limit_is_logged_after_its_detection() {
    install();
    // The first call detects the limit, once per process, and logs what it
    // found (a limit, or none).
    let _ = active_budget();
    assert!(
        BUDGET.records.load(Ordering::SeqCst) >= 1,
        "the detection emits a record, or this test checks nothing"
    );
    assert_eq!(
        BUDGET.blocked.load(Ordering::SeqCst),
        0,
        "a record was emitted inside the detection's initialiser"
    );
}

#[test]
fn a_registration_is_logged_with_the_registry_unlocked() {
    install();
    register_backend(
        "log-outside-locks",
        |_: &BackendBuildContext| -> Result<Arc<dyn QuantumBackend>, BackendError> {
            Err(BackendError::UnknownInfrastructure {
                name: "log-outside-locks".to_string(),
            })
        },
    );
    assert!(is_registered("log-outside-locks"));
    assert!(
        REGISTRY.records.load(Ordering::SeqCst) >= 1,
        "the registration emits a record, or this test checks nothing"
    );
    assert_eq!(
        REGISTRY.blocked.load(Ordering::SeqCst),
        0,
        "a record was emitted while holding the registry's write lock"
    );
}
