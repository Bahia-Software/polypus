//! The battery, applied to a bundled reference backend that is deliberately built to
//! satisfy *every* check — including one rigged into each fault. This CI-guards the
//! suite itself: it proves a fully-conformant backend passes with **zero failures
//! and zero skips**, so a real backend's skips/failures are meaningful signal and not
//! an artefact of the battery.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use polypus_backend::{
    BackendCapabilities, BackendError, BoundCircuit, Counts, QuantumBackend, RunParams,
};
use polypus_backend_conformance::{Conformance, Fault, Status};

/// How the reference backend behaves — one healthy path and one per fault.
#[derive(Clone, Copy)]
enum Mode {
    /// Returns width-appropriate deterministic counts; declares a finite cap of 2 so
    /// the concurrency and multi-wave checks are exercised (not skipped).
    Healthy,
    /// Always reports it stopped responding.
    Unresponsive,
    /// Blocks until cancelled, then reports the call was aborted; opts into the
    /// cancellation watcher.
    Aborting,
    /// Same abort behaviour, but *forgets* to opt into the watcher
    /// (`wants_cancel_watcher()` stays false) — the descuido the battery must catch.
    AbortingNoWatcher,
    /// Always reports a clean provider failure.
    CleanError,
}

struct ReferenceBackend {
    mode: Mode,
    cancelled: AtomicBool,
}

impl ReferenceBackend {
    fn arced(mode: Mode) -> Arc<dyn QuantumBackend> {
        Arc::new(ReferenceBackend {
            mode,
            cancelled: AtomicBool::new(false),
        })
    }

    /// All shots on the all-zeros bitstring of the circuit's width (a `Foreign`
    /// circuit has no native width and is rejected before this is called).
    fn zeros_counts(circuit: &BoundCircuit, shots: u32) -> Counts {
        let width = circuit.native_qubit_width().unwrap_or(1).max(1);
        HashMap::from([("0".repeat(width), u64::from(shots))])
    }
}

impl QuantumBackend for ReferenceBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        match self.mode {
            Mode::Unresponsive => {
                return Err(BackendError::Unresponsive(
                    "reference worker died".to_string(),
                ))
            }
            Mode::CleanError => {
                return Err(BackendError::External(
                    "reference provider rejected the job".into(),
                ))
            }
            Mode::Aborting | Mode::AbortingNoWatcher => {
                // Block like a real long call, watching for the out-of-band cancel.
                let deadline = Instant::now() + Duration::from_secs(5);
                while Instant::now() < deadline {
                    if self.cancelled.load(Ordering::SeqCst) {
                        return Err(BackendError::Aborted("cancelled by signal".to_string()));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                return Err(BackendError::Unresponsive(
                    "reference abort deadline elapsed without a cancel".to_string(),
                ));
            }
            Mode::Healthy => {}
        }
        // A Foreign circuit is unsupported by this pure-Rust backend.
        if let Some(bad) = qcs.iter().find(|c| c.is_foreign()) {
            let _ = bad;
            return Err(BackendError::UnsupportedCircuit(
                "the reference backend cannot run a Foreign circuit".to_string(),
            ));
        }
        Ok(qcs
            .iter()
            .map(|qc| Self::zeros_counts(qc, params.shots))
            .collect())
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn wants_cancel_watcher(&self) -> bool {
        matches!(self.mode, Mode::Aborting)
    }

    fn capabilities(&self) -> BackendCapabilities {
        // A finite cap so the declared-concurrency and multi-wave checks run.
        BackendCapabilities {
            max_concurrency: 2,
            supports_shot_distribution: true,
        }
    }
}

#[test]
fn a_fully_conformant_reference_backend_passes_everything() {
    let report = Conformance::new("reference", || Ok(ReferenceBackend::arced(Mode::Healthy)))
        .shots(600)
        .fault(Fault::Unresponsive, || {
            Ok(ReferenceBackend::arced(Mode::Unresponsive))
        })
        .fault(Fault::Aborted, || {
            Ok(ReferenceBackend::arced(Mode::Aborting))
        })
        .fault(Fault::CleanError, || {
            Ok(ReferenceBackend::arced(Mode::CleanError))
        })
        .run();

    println!("{report}");
    // A fully-rigged conformant backend must pass every check with no skips.
    assert_eq!(
        report.failed_count(),
        0,
        "reference backend should fail nothing:\n{report}"
    );
    assert_eq!(
        report.skipped_count(),
        0,
        "with all faults supplied, nothing should be skipped:\n{report}"
    );
    assert!(report.is_conformant());
}

#[test]
fn omitting_a_fault_factory_skips_not_fails() {
    // A healthy-only harness: the error-classification checks skip, and the backend
    // still counts as conformant (skips are not failures).
    let report = Conformance::new("reference-no-faults", || {
        Ok(ReferenceBackend::arced(Mode::Healthy))
    })
    .shots(300)
    .run();

    assert!(
        report.is_conformant(),
        "skips must not break conformance:\n{report}"
    );
    assert!(
        report.skipped_count() >= 3,
        "the three fault checks should skip:\n{report}"
    );
    // The aborted check specifically must be a skip, not a pass or fail.
    let aborted = report
        .checks
        .iter()
        .find(|c| c.name == "interrupted_call_is_aborted")
        .expect("the aborted check is always present");
    assert!(matches!(aborted.status, Status::Skipped(_)));
}

#[test]
fn aborting_backend_without_the_watcher_optin_is_non_conformant() {
    // A backend that aborts correctly on a direct cancel() but leaves
    // wants_cancel_watcher() at its false default would never be cancelled in
    // production (Scheduler::run_cancellable only spawns the watcher for opted-in
    // backends). Supplying Fault::Aborted asserts end-to-end cancellability, so the
    // battery must fail it despite the abort itself working.
    let report = Conformance::new("aborts-but-no-watcher", || {
        Ok(ReferenceBackend::arced(Mode::Healthy))
    })
    .shots(200)
    .abort_grace(Duration::from_millis(150))
    .fault(Fault::Aborted, || {
        Ok(ReferenceBackend::arced(Mode::AbortingNoWatcher))
    })
    .run();

    assert!(
        !report.is_conformant(),
        "an aborting backend that never opts into the watcher must be non-conformant:\n{report}"
    );
    let aborted = report
        .checks
        .iter()
        .find(|c| c.name == "interrupted_call_is_aborted")
        .unwrap();
    match &aborted.status {
        Status::Failed(msg) => assert!(
            msg.contains("wants_cancel_watcher"),
            "the failure should name wants_cancel_watcher, got: {msg}"
        ),
        other => panic!("expected a Failed status naming wants_cancel_watcher, got {other:?}"),
    }
}

#[test]
fn a_backend_that_misclassifies_a_dead_worker_fails() {
    // A backend whose "unresponsive" factory actually returns a clean error is
    // non-conformant: the battery must catch the misclassification.
    let report = Conformance::new("mislabeller", || Ok(ReferenceBackend::arced(Mode::Healthy)))
        .shots(200)
        .fault(Fault::Unresponsive, || {
            // Wrong: a dead worker reported as a clean provider error.
            Ok(ReferenceBackend::arced(Mode::CleanError))
        })
        .run();

    assert!(
        !report.is_conformant(),
        "a misclassified fault must fail:\n{report}"
    );
    let check = report
        .checks
        .iter()
        .find(|c| c.name == "unresponsive_backend_is_classified")
        .unwrap();
    assert!(matches!(check.status, Status::Failed(_)));
}
