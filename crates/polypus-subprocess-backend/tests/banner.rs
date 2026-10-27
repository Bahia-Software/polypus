//! A worker that writes non-protocol bytes to stdout (an SDK printing a banner on
//! import) is rejected at once with a clear error — not after the read timeout, and
//! without reserving the gigabytes its "length" would ask for.
//!
//! This test must stay alone in its binary. It checks the process-wide `VmPeak`, and
//! every thread that allocates may get its own glibc malloc arena, which reserves
//! 64 MiB of address space. With other tests spawning workers in parallel threads,
//! those reservations rise past the threshold without any large allocation (#465).
//! Alone, only this test's few threads count, which keeps the growth well below it.

mod common;

use std::time::{Duration, Instant};

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
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "must fail immediately, not at the read timeout; took {:?}",
        started.elapsed()
    );
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
