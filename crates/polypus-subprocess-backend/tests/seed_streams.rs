//! What a seed means across the calls of one run (contract C-7, #411): the bridge
//! sends the worker the base seed on its first call and, on call `n`, a seed
//! derived from the base seed and `n`, so every call of a run gets its own
//! sampling stream and a fresh bridge with the same seed replays the sequence.

mod common;

use polypus_backend::{OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::SubprocessBackend;

fn params(seed: Option<u64>) -> RunParams {
    RunParams {
        id: "bridge-seeds".to_string(),
        shots: 16,
        seed,
        opt_level: OptLevel::default(),
    }
}

/// The seeds the worker received in `calls` calls of one bridge, read from the
/// log the reference worker appends to under `WORKER_SEED_LOG`.
fn worker_seeds(python: &str, seed: Option<u64>, calls: usize, tag: &str) -> Vec<String> {
    let log = std::env::temp_dir().join(format!(
        "polypus-seed-streams-{}-{tag}.log",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&log);
    let log_path = log.to_str().expect("a UTF-8 temp path");
    let backend = SubprocessBackend::spawn(common::config(
        python,
        30_000,
        &[("WORKER_SEED_LOG", log_path)],
    ))
    .expect("worker spawns and handshakes");
    for _ in 0..calls {
        backend
            .run_circuits(&[common::native(1), common::native(2)], &params(seed))
            .expect("the run returns counts");
    }
    backend.close();
    let seeds = std::fs::read_to_string(&log)
        .expect("the worker wrote its seed log")
        .lines()
        .map(str::to_string)
        .collect();
    let _ = std::fs::remove_file(&log);
    seeds
}

#[test]
fn every_call_gets_its_own_seed_and_a_seed_replays_them() {
    let Some(python) = common::resolve_python() else {
        eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
        return;
    };
    let seeds = worker_seeds(&python, Some(7), 4, "a");
    assert_eq!(seeds.len(), 4);
    assert_eq!(seeds[0], "7", "the first call carries the base seed");
    let distinct: std::collections::HashSet<_> = seeds.iter().collect();
    assert_eq!(distinct.len(), 4, "{seeds:?}");
    assert_eq!(worker_seeds(&python, Some(7), 4, "b"), seeds);
}

#[test]
fn no_seed_stays_absent() {
    let Some(python) = common::resolve_python() else {
        eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)");
        return;
    };
    assert_eq!(worker_seeds(&python, None, 3, "none"), ["None"; 3]);
}
