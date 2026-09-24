//! The bridge spawns the reference worker, runs circuits, gets counts back, and
//! reuses the same worker for a second call — the normal round trip.

mod common;

use polypus_backend::{
    CancelToken, CircuitTask, OptLevel, Planner, QuantumBackend, RunParams, SequentialPlanner,
};
use polypus_subprocess_backend::SubprocessBackend;

fn params(shots: u32) -> RunParams {
    RunParams {
        id: "bridge-happy".to_string(),
        shots,
        seed: Some(7),
        opt_level: OptLevel::default(),
    }
}

#[test]
fn round_trip_and_worker_reuse() {
    let python = match common::resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };
    let backend = SubprocessBackend::spawn(common::config(&python, 30_000, &[]))
        .expect("worker spawns and handshakes");

    // First call.
    let circuits = [common::native(2), common::native(3)];
    let out = backend
        .run_circuits(&circuits, &params(1024))
        .expect("first run returns counts");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].values().sum::<u64>(), 1024);
    assert_eq!(out[1].values().sum::<u64>(), 1024);

    // Second call on the same (reused) worker.
    let out2 = backend
        .run_circuits(&[common::native(1)], &params(100))
        .expect("second run reuses the worker");
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].values().sum::<u64>(), 100);

    // And it drives the shipped planner just like a built-in backend.
    let a = common::native(2);
    let tasks = vec![CircuitTask {
        circuit: &a,
        shots: 64,
    }];
    let planned = SequentialPlanner
        .execute(&backend, &tasks, &params(64), &CancelToken::default())
        .expect("the sequential planner runs the subprocess backend");
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].values().sum::<u64>(), 64);

    backend.close();
}
