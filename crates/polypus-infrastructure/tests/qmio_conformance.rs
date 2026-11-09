//! The conformance battery against the QMIO backend; the other built-in backends are in
//! `conformance.rs`. `QmioBackend::new` reads the `QMIO_*` tuning variables from the
//! process environment and this test sets them, so it is the only test in its binary:
//! mutating the environment cannot race any other test.
#![cfg(feature = "qmio")]

use std::sync::Arc;

use polypus_backend::QuantumBackend;
use polypus_backend_conformance::{Conformance, Fault, Status};
use polypus_infrastructure::qmio::{QmioBackend, QmioProgramFormat};

/// Assert a named check has the expected status, with the report in the message.
fn assert_status(report: &polypus_backend_conformance::Report, name: &str, want_passed: bool) {
    let check = report
        .checks
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("check {name} not found in report:\n{report}"));
    let passed = matches!(check.status, Status::Passed);
    assert_eq!(
        passed, want_passed,
        "check {name} status {:?} (want passed={want_passed}):\n{report}",
        check.status
    );
}

/// QMIO has no simulator mode, so only its error classification is exercised — against
/// an **unreachable endpoint**, with a short timeout so the test is fast. Documents
/// that a `Foreign` circuit is rejected as `UnsupportedCircuit` (finding QMIO-1) and that a
/// wire timeout is a clean `External` error (QMIO exhausts its own retries,
/// so `Unresponsive` is not applicable and is not supplied).
///
/// Run it explicitly when touching the QMIO backend:
/// ```text
/// cargo test -p polypus-infrastructure --features qmio --test qmio_conformance -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: builds the qmio feature and attempts a socket; run explicitly (see docs/backends.md)"]
fn qmio_error_classification() {
    // Fast-fail: short receive timeout, no retries, so an unreachable endpoint returns
    // promptly instead of retrying with backoff.
    std::env::set_var("QMIO_RECV_TIMEOUT_MS", "150");
    std::env::set_var("QMIO_MAX_RETRIES", "0");
    std::env::set_var("QMIO_RETRY_BACKOFF_MS", "0");

    let make = || {
        QmioBackend::new(
            "tcp://127.0.0.1:59999".to_string(), // nothing is listening here
            QmioProgramFormat::OpenQasm,
            0,
            None,
            "binary_count".to_string(),
        )
        .map(|b| Arc::new(b) as Arc<dyn QuantumBackend>)
        .map_err(|e| e.to_string())
    };

    let report = Conformance::new("qmio(unreachable)", make)
        // A wire failure is a clean, definitive provider error → External.
        .fault(Fault::CleanError, make)
        // Deliberately NO Unresponsive factory: QMIO exhausts its own retries and then
        // reports a definitive External, so Unresponsive is not applicable.
        .run_error_classification();
    println!("\n{report}");

    // QMIO does classify its wire failure as a clean (External) error.
    assert_status(&report, "clean_provider_error_is_classified", true);
    // Finding QMIO-1 (fixed): a Foreign circuit is `UnsupportedCircuit`, not `External`.
    assert_status(&report, "foreign_circuit_rejected_as_unsupported", true);
}
