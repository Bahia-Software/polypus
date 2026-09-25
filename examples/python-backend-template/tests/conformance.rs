//! The acceptance test for the Python template: `worker.py`, driven through the real
//! subprocess bridge, passes the Polypus conformance battery. Skipped (not failed) on
//! a host without a Python interpreter, so it never breaks a Python-less `cargo test`.

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use polypus_backend::QuantumBackend;
use polypus_backend_conformance::{Conformance, Fault};
use polypus_subprocess_backend::{SubprocessBackend, SubprocessConfig};

/// Locate a Python interpreter, or `None` to skip.
fn resolve_python() -> Option<String> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("POLYPUS_BRIDGE_PYTHON") {
        candidates.push(p);
    }
    candidates.push("python3".to_string());
    candidates.push("python".to_string());
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("-c")
            .arg("pass")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn worker_path() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/worker.py").to_string()
}

/// A bridge config launching `worker.py` with the given child env (the fault hooks).
fn config(python: &str, env: &[(&str, &str)]) -> SubprocessConfig {
    SubprocessConfig {
        command: vec![python.to_string(), worker_path()],
        cwd: None,
        env: env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        // Generous, so the *timeout* is never what ends a call in these tests — the
        // crash and cancel paths end them first.
        recv_timeout: Duration::from_secs(30),
        arm_pdeathsig: true,
    }
}

/// A battery factory that spawns a fresh worker with `env`.
fn spawn_with(
    python: String,
    env: &'static [(&'static str, &'static str)],
) -> impl Fn() -> Result<Arc<dyn QuantumBackend>, String> {
    move || {
        SubprocessBackend::spawn(config(&python, env))
            .map(|b| Arc::new(b) as Arc<dyn QuantumBackend>)
            .map_err(|e| e.to_string())
    }
}

#[test]
fn the_python_worker_passes_the_conformance_battery() {
    let python = match resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Python interpreter (set POLYPUS_BRIDGE_PYTHON)");
            return;
        }
    };

    let report = Conformance::new("subprocess(worker.py)", spawn_with(python.clone(), &[]))
        .shots(1024)
        // Drive each fault through the bridge:
        //  - crash mid-call  -> worker dies (EOF)      -> Unresponsive
        //  - a long call     -> cancel() sends SIGINT  -> Aborted
        //  - WORKER_ERROR    -> worker replies `error` -> External (clean provider error)
        .fault(
            Fault::Unresponsive,
            spawn_with(python.clone(), &[("WORKER_CRASH_AFTER_MS", "50")]),
        )
        .fault(
            Fault::Aborted,
            spawn_with(python.clone(), &[("WORKER_RUN_DELAY_MS", "3000")]),
        )
        .fault(
            Fault::CleanError,
            spawn_with(python.clone(), &[("WORKER_ERROR", "device rejected the job")]),
        )
        .run();

    println!("{report}");
    report.assert_conformant();
    // The bridge declares unbounded concurrency, so only that one check skips.
    for name in [
        "unresponsive_backend_is_classified",
        "clean_provider_error_is_classified",
        "interrupted_call_is_aborted",
    ] {
        let check = report.checks.iter().find(|c| c.name == name).unwrap();
        assert!(
            matches!(check.status, polypus_backend_conformance::Status::Passed),
            "{name} should pass, got {:?}",
            check.status
        );
    }
}
