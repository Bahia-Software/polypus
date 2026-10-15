//! A worker that writes non-protocol bytes to stdout (an SDK printing a banner on
//! import) is rejected at once with a clear error — not after the read timeout, and
//! without reserving the gigabytes its "length" would ask for — and a worker whose
//! reply exceeds `max_frame_bytes` is refused the same way.

mod common;

use std::time::{Duration, Instant};

use polypus_backend::{BackendError, OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::{BridgeError, SubprocessBackend};

/// Peak virtual memory of this process in kB (`VmPeak`), where `/proc` exposes it.
fn vm_peak_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmPeak:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// A generous read timeout, so a pass proves the error came from the frame check and
/// not from the timeout.
const LONG_TIMEOUT_MS: u64 = 60_000;

fn assert_fast(started: Instant) {
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "must fail immediately, not at the read timeout; took {:?}",
        started.elapsed()
    );
}

#[test]
fn a_banner_on_stdout_fails_the_handshake_fast_with_a_clear_error() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    let before = vm_peak_kb();
    let started = Instant::now();
    let err = match SubprocessBackend::spawn(common::config(
        &python,
        LONG_TIMEOUT_MS,
        &[("WORKER_STDOUT_BANNER", "Hello from vendor SDK v1.2")],
    )) {
        Err(e) => e,
        Ok(_) => panic!("a worker polluting stdout must not complete the handshake"),
    };
    // "Hell" read as a length is ~1.8 GB; nothing near that may have been reserved.
    if let (Some(before), Some(after)) = (before, vm_peak_kb()) {
        assert!(
            after.saturating_sub(before) < 512 * 1024,
            "VmPeak grew from {before} kB to {after} kB: the announced length was allocated"
        );
    }
    assert_fast(started);
    match &err {
        BridgeError::Protocol(m) => {
            assert!(
                m.contains("\"Hell\""),
                "should quote the offending bytes: {m}"
            );
            assert!(m.contains("stdout"), "should point at stdout: {m}");
        }
        other => panic!("expected BridgeError::Protocol, got {other:?}"),
    }
    assert!(!err.to_string().contains("hung"), "not a timeout: {err}");
}

#[test]
fn a_reply_above_the_frame_cap_is_refused_with_a_clear_error_and_kills_the_worker() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // The handshake reply is 29 bytes and fits; the counts of 200 distinct bitstrings
    // do not fit in 256.
    let mut config = common::config(&python, LONG_TIMEOUT_MS, &[("WORKER_SPREAD_KEYS", "200")]);
    config.max_frame_bytes = 256;
    let backend = SubprocessBackend::spawn(config).expect("the small ready frame is under the cap");
    let params = RunParams {
        id: "frame-cap".to_string(),
        shots: 1000,
        seed: None,
        opt_level: OptLevel::default(),
    };

    let started = Instant::now();
    let err = backend
        .run_circuits(&[common::native(8)], &params)
        .expect_err("an oversized frame must be refused");
    assert_fast(started);
    let message = err.to_string();
    assert!(
        matches!(err, BackendError::External(_)),
        "a protocol violation is not an Unresponsive/timeout error: {err:?}"
    );
    assert!(
        message.contains("max_frame_bytes") && message.contains("256-byte limit"),
        "should name the limit and the option: {message}"
    );

    // The stream is unusable afterwards: a later call fails fast instead of hanging.
    let started = Instant::now();
    assert!(backend.run_circuits(&[common::native(8)], &params).is_err());
    assert_fast(started);
}

#[test]
fn a_reply_under_a_small_cap_still_works() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // The same cap as above, but a two-key counts reply (~70 bytes) fits under it.
    // (A frame exactly at the limit is pinned by the `read_frame` unit tests.)
    let mut config = common::config(&python, LONG_TIMEOUT_MS, &[]);
    config.max_frame_bytes = 256;
    let backend = SubprocessBackend::spawn(config).expect("spawns");
    let params = RunParams {
        id: "frame-cap-ok".to_string(),
        shots: 100,
        seed: None,
        opt_level: OptLevel::default(),
    };
    let counts = backend
        .run_circuits(&[common::native(2)], &params)
        .expect("a reply under the cap is delivered");
    assert_eq!(counts.len(), 1);
    backend.close();
}
