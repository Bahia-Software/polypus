//! `PR_SET_PDEATHSIG` is delivered when the *thread* that forked the child dies, not
//! when the process does (`prctl(2)`). A backend built from a short-lived thread (a
//! pool worker, a Python `threading.Thread`) must therefore still have a live worker
//! after that thread is gone: the bridge forks from its own long-lived spawner thread.

#![cfg(target_os = "linux")]

mod common;

use std::thread;
use std::time::Duration;

use polypus_backend::{OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

fn params() -> RunParams {
    RunParams {
        id: "bridge-spawn-thread".to_string(),
        shots: 256,
        seed: Some(3),
        opt_level: OptLevel::default(),
    }
}

#[test]
fn backend_built_on_an_ephemeral_thread_survives_that_thread() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    // Build the backend (spawn + handshake) on a thread that exits right after.
    let config = common::config(&python, 30_000, &[]);
    let backend = thread::spawn(move || SubprocessBackend::spawn(config))
        .join()
        .expect("the constructor thread does not panic")
        .expect("worker spawns and handshakes");

    // Give the kernel time to deliver a (wrongly) thread-bound SIGKILL.
    thread::sleep(Duration::from_millis(200));

    let out = backend
        .run_circuits(&[common::native(2)], &params())
        .expect("the worker must outlive the thread that built the backend");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].values().sum::<u64>(), 256);

    backend.close();
}
