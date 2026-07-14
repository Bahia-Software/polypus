use polypus::algorithms::{
    AlgorithmArgs, AlgorithmDifferentialEvolution, AlgorithmPSO, AlgorithmQNG, AlgorithmSingleRun,
    DistributeByShotsRun,
};
use polypus::circuit::ParameterizedCircuit;
use polypus::infrastructure::{
    BackendConfig, BackendError, BoundCircuit, ExecutionConfig, Infrastructure, OptLevel,
};
use polypus::AlgorithmTrait;
use pyo3::prelude::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// AlgorithmTrait metadata — name() and description() are pure Rust, no Python
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn single_run_name() {
    assert_eq!(AlgorithmSingleRun.name(), "Single Run Algorithm");
}

#[test]
fn single_run_description_is_non_empty() {
    assert!(!AlgorithmSingleRun.description().is_empty());
}

#[test]
fn distribute_by_shots_name() {
    assert_eq!(
        DistributeByShotsRun.name(),
        "Distribute By Shots Run Algorithm"
    );
}

#[test]
fn distribute_by_shots_description_is_non_empty() {
    assert!(!DistributeByShotsRun.description().is_empty());
}

#[test]
fn differential_evolution_name() {
    assert_eq!(
        AlgorithmDifferentialEvolution.name(),
        "Differential Evolution"
    );
}

#[test]
fn differential_evolution_description_is_non_empty() {
    assert!(!AlgorithmDifferentialEvolution.description().is_empty());
}

#[test]
fn pso_name() {
    assert_eq!(AlgorithmPSO.name(), "Particle Swarm Optimization");
}

#[test]
fn pso_description_is_non_empty() {
    assert!(!AlgorithmPSO.description().is_empty());
}

#[test]
fn qng_name() {
    assert_eq!(AlgorithmQNG.name(), "Quantum Natural Gradient");
}

#[test]
fn qng_description_is_non_empty() {
    assert!(!AlgorithmQNG.description().is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure::from_str — pure Rust parsing, no Python
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn infrastructure_from_str_local() {
    assert!(matches!(
        Infrastructure::from_str("local"),
        Ok(Infrastructure::Local)
    ));
}

#[test]
fn infrastructure_from_str_cunqa() {
    assert!(matches!(
        Infrastructure::from_str("cunqa"),
        Ok(Infrastructure::Cunqa)
    ));
}

#[test]
fn infrastructure_from_str_unknown_is_typed_error() {
    // An unknown infrastructure is now a typed `Result` error (surfaced across
    // the FFI as a `ValueError`, contract C-1), never a panic.
    match Infrastructure::from_str("unknown_backend") {
        Err(BackendError::UnknownInfrastructure { name }) => {
            assert_eq!(name, "unknown_backend");
        }
        _ => panic!("expected an UnknownInfrastructure error, not a panic"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistributeByShotsRun — shot conservation (contract C-3)
//
// The native (pure-Rust) statevector backend is used so the orchestration path
// runs without a live Python runtime or Qiskit; `DistributeByShotsRun::run`
// still builds a `PyDict` for its return value, so the interpreter is
// initialised with `prepare_freethreaded_python()` — the same pattern used by
// the qmio backend's tests.
// ─────────────────────────────────────────────────────────────────────────────

/// Build `AlgorithmArgs` for a native Bell circuit on the pure-Rust backend.
fn native_bell_args(shots: u32, n_qpus: u32, id: &str) -> AlgorithmArgs {
    let bell = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .assign_parameters(&[])
        .expect("bell circuit has no free parameters");
    AlgorithmArgs {
        config: ExecutionConfig {
            id: id.to_string(),
            shots,
            n_qpus,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            // Fixed seed: these tests assert only shot conservation, and a fixed
            // seed keeps the native backend's sampling deterministic across runs.
            seed: Some(7),
        },
        qcs: vec![BoundCircuit::Native(bell)],
    }
}

/// Run the distribute-by-shots algorithm and return the merged counts.
fn distributed_counts(shots: u32, n_qpus: u32, id: &str) -> HashMap<String, u64> {
    pyo3::prepare_freethreaded_python();
    let result = DistributeByShotsRun
        .run(native_bell_args(shots, n_qpus, id))
        .expect("distribute-by-shots must succeed on the native backend");
    Python::with_gil(|py| {
        result
            .extract::<HashMap<String, u64>>(py)
            .expect("distribute-by-shots must return a dict[str, int]")
    })
}

#[test]
fn distribute_conserves_shots_when_not_divisible() {
    // 1000 shots over 3 QPUs: remainder 1. The old `shots /= n_qpus` executed
    // only 999 shots; the total must now be conserved exactly.
    let counts = distributed_counts(1000, 3, "c3-uneven");
    let total: u64 = counts.values().sum();
    assert_eq!(
        total, 1000,
        "shots must be conserved on uneven distribution"
    );
    for key in counts.keys() {
        assert!(key == "00" || key == "11", "unexpected Bell outcome {key}");
    }
}

#[test]
fn distribute_conserves_shots_when_fewer_shots_than_qpus() {
    // 5 shots over 8 QPUs: base 0, remainder 5. The old logic executed 0 shots
    // (5 / 8 == 0); the total must now be exactly 5, spread one-per-QPU over the
    // first 5 QPUs.
    let counts = distributed_counts(5, 8, "c3-degenerate");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 5, "shots must be conserved when shots < n_qpus");
}

#[test]
fn distribute_conserves_shots_when_divisible() {
    // Exact division still holds (regression guard for the even case).
    let counts = distributed_counts(400, 4, "c3-even");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 400);
}
