//! `cancel()` reaches a worker blocked mid-run: called from another thread while a
//! `run_circuits` is in flight, it signals the worker (SIGINT), which aborts and
//! stays alive and reusable — the out-of-band abort the planner's watcher drives.

mod common;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use polypus_backend::{OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

fn params(id: &str) -> RunParams {
    RunParams {
        id: id.to_string(),
        shots: 256,
        seed: None,
        opt_level: OptLevel::default(),
    }
}

#[test]
fn cancel_aborts_an_in_flight_run_and_leaves_the_worker_reusable() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // The worker takes ~3 s per run (interruptible), with a generous read timeout so
    // the timeout is NOT what ends the call — the cancel is.
    let backend = Arc::new(
        SubprocessBackend::spawn(common::config(
            &python,
            30_000,
            &[("WORKER_RUN_DELAY_MS", "3000")],
        ))
        .expect("worker spawns and handshakes"),
    );

    let runner = Arc::clone(&backend);
    let handle = thread::spawn(move || {
        let started = Instant::now();
        let res = runner.run_circuits(&[common::native(2)], &params("cancel-run"));
        (res, started.elapsed())
    });

    // Let the run get in flight, then cancel out-of-band from this thread.
    thread::sleep(Duration::from_millis(400));
    backend.cancel();

    let (res, elapsed) = handle.join().expect("runner thread must not panic");
    let err = res.expect_err("a cancelled run must return an error, not counts");
    assert!(
        err.to_string().contains("aborted"),
        "expected an abort error, got: {err}"
    );
    // It aborted well before the worker's 3 s delay elapsed.
    assert!(
        elapsed < Duration::from_millis(2500),
        "cancel should abort promptly, took {elapsed:?}"
    );

    // The worker survived the cancellation and services a fresh run.
    let out = backend
        .run_circuits(&[common::native(1)], &params("after-cancel"))
        .expect("the worker stays alive and reusable after an abort");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].values().sum::<u64>(), 256);

    backend.close();
}
