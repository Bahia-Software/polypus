//! Shared helpers for the subprocess-bridge integration tests. Each test file is a
//! separate binary (separate process), so the per-test worker env vars set below via
//! [`SubprocessConfig::env`] never race across files.
// Each test binary uses a different subset of these helpers; silence per-binary
// dead-code warnings for the ones a given binary does not call.
#![allow(dead_code)]

use std::process::Command;
use std::time::Duration;

use polypus_backend::BoundCircuit;
use polypus_circuit::ParameterizedCircuit;
use polypus_subprocess_backend::SubprocessConfig;

/// Resolve a working Python interpreter, or `None` to skip the test on a host
/// without one (so the pyo3-free `cargo test` job never fails for lack of Python).
pub fn resolve_python() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("POLYPUS_BRIDGE_PYTHON") {
        candidates.push(p);
    }
    candidates.push("python3".to_string());
    candidates.push("python".to_string());
    candidates.into_iter().find(|cand| {
        Command::new(cand)
            .arg("-c")
            .arg("pass")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Path to the shipped reference worker.
pub fn worker_script() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/python/worker_template.py").to_string()
}

/// A config launching the reference worker with the given read timeout and child env.
pub fn config(python: &str, timeout_ms: u64, env: &[(&str, &str)]) -> SubprocessConfig {
    SubprocessConfig {
        command: vec![python.to_string(), worker_script()],
        cwd: None,
        env: env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        recv_timeout: Duration::from_millis(timeout_ms),
        arm_pdeathsig: true,
    }
}

/// A native, fully-bound circuit of `n` qubits with a terminal measurement (so it
/// serialises to a QASM2 program the worker echoes counts for).
pub fn native(n: usize) -> BoundCircuit {
    BoundCircuit::Native(
        ParameterizedCircuit::new(n)
            .measure_all()
            .assign_parameters(&[])
            .unwrap(),
    )
}
