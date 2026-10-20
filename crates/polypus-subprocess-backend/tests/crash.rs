//! A worker that dies mid-call (EOF) is detected and mapped to
//! [`BackendError::Unresponsive`] — never a hang, never a panic.

mod common;

use polypus_backend::{BackendError, OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

fn params() -> RunParams {
    RunParams {
        id: "bridge-crash".to_string(),
        shots: 128,
        seed: None,
        opt_level: OptLevel::default(),
    }
}

#[test]
fn worker_that_crashes_mid_call_is_unresponsive() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // The worker self-destructs 50 ms into the run (os._exit, no reply).
    let backend = SubprocessBackend::spawn(common::config(
        &python,
        30_000,
        &[("WORKER_CRASH_AFTER_MS", "50")],
    ))
    .expect("worker spawns and handshakes before the run");

    let err = backend
        .run_circuits(&[common::native(2)], &params())
        .expect_err("a worker that dies mid-call must surface an error, not hang");
    match err {
        BackendError::Unresponsive(msg) => {
            assert!(
                msg.contains("worker process ended"),
                "expected a 'worker process ended' message, got: {msg}"
            );
        }
        other => panic!("expected Unresponsive, got: {other:?}"),
    }
}
