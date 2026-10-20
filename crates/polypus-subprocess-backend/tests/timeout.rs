//! A worker that is alive but wedged (never replies) trips the bridge's read
//! timeout and maps to [`BackendError::Unresponsive`] — the Fase-2 open risk, closed.
//! Without the timeout, the bridge's `recv` would block forever.

mod common;

use std::time::Instant;

use polypus_backend::{BackendError, OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

fn params() -> RunParams {
    RunParams {
        id: "bridge-timeout".to_string(),
        shots: 128,
        seed: None,
        opt_level: OptLevel::default(),
    }
}

#[test]
fn hung_worker_trips_the_read_timeout() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // The worker sleeps 10 s on the run (far past the 300 ms read timeout below), so
    // it is alive but never replies — a mute-QPU / hung-SDK stand-in.
    let backend = SubprocessBackend::spawn(common::config(
        &python,
        300, // 300 ms read timeout
        &[("WORKER_HANG_MS", "10000")],
    ))
    .expect("worker spawns and handshakes before the run");

    let started = Instant::now();
    let err = backend
        .run_circuits(&[common::native(2)], &params())
        .expect_err("a hung worker must time out, not block forever");
    let elapsed = started.elapsed();

    match err {
        BackendError::Unresponsive(msg) => {
            assert!(
                msg.contains("no response within"),
                "expected a timeout message, got: {msg}"
            );
        }
        other => panic!("expected Unresponsive, got: {other:?}"),
    }
    // It returned promptly (near the 300 ms deadline), not after the worker's 10 s.
    assert!(
        elapsed.as_secs() < 5,
        "timeout should fire near the deadline, took {elapsed:?}"
    );
}
