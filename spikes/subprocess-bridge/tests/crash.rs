//! Point 2b: the subprocess dies mid-call. Two ways in — an external SIGKILL
//! and the worker crashing itself (os._exit) — must both surface as an
//! explicit, typed error (never a hang, never a Rust panic), and the bridge
//! must be recoverable by spawning a fresh worker.
//!
//! This is the direct input to Fase 3's "backend stopped responding" error
//! variant: `BridgeError::WorkerDied` here maps to `BackendError::External`.

use std::time::Duration;

use subprocess_bridge_spike::{python_bin, worker_script, BridgeError, Circuit, Request, Worker};

fn one_circuit() -> Vec<Circuit> {
    vec![Circuit {
        qasm: "x".into(),
        n_qubits: 1,
    }]
}

#[test]
fn external_sigkill_midcall_is_detected_and_recoverable() {
    let mut w = Worker::spawn(&python_bin(), &worker_script(), false).expect("spawn");
    w.send(&Request::run("pending", one_circuit(), 100, 2000))
        .expect("send");
    std::thread::sleep(Duration::from_millis(150));

    // Kill the worker while the request is pending.
    w.kill().expect("SIGKILL the worker");

    // Rust detects EOF on the pipe and reports a typed error — no hang.
    let err = w.recv().expect_err("must be an error, not a reply");
    assert!(matches!(err, BridgeError::WorkerDied { .. }), "got {err:?}");
    eprintln!("detected: {err}");

    // Recoverable: a brand-new worker serves requests normally.
    let mut w2 = Worker::spawn(&python_bin(), &worker_script(), false).expect("respawn");
    let resp = w2
        .call(&Request::run("after-recovery", one_circuit(), 10, 0))
        .expect("recovered");
    assert_eq!(resp.op, "result");
}

#[test]
fn worker_self_crash_midcall_is_detected() {
    // WORKER_CRASH_AFTER_MS makes the worker os._exit(137) partway through a
    // run — a stand-in for a segfault or an unhandled SDK abort. Set on the
    // child only, so it can't leak to other workers under parallel tests.
    let mut w = Worker::spawn_with_env(
        &python_bin(),
        &worker_script(),
        false,
        &[("WORKER_CRASH_AFTER_MS", "150")],
    )
    .expect("spawn");

    let err = w
        .call(&Request::run("boom", one_circuit(), 100, 1000))
        .expect_err("worker self-crashes, no reply");
    match err {
        BridgeError::WorkerDied { wait_status } => {
            eprintln!("self-crash reaped as: {wait_status}");
            assert!(wait_status.contains("137") || wait_status.contains("signal"));
        }
        other => panic!("expected WorkerDied, got {other:?}"),
    }
}
