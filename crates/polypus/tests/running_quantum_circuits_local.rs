use polypus::algorithms::{AlgorithmDifferentialEvolution, AlgorithmPSO, AlgorithmQNG};
use polypus::circuit::ParameterizedCircuit;
use polypus::infrastructure::{
    BackendConfig, BackendError, BoundCircuit, ExecutionConfig, Infrastructure,
    InfrastructureError, NativeStatevectorBackend, OptLevel, QuantumBackend,
    ShotDistributingPlanner,
};
use polypus::scheduler::{Resources, RunCircuitFlow, Scheduler};
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer metadata — name()/description() are inherent pure-Rust methods
// ─────────────────────────────────────────────────────────────────────────────

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
// Shot distribution (contract C-3) through the Phase-6 orchestration path:
// `Scheduler::run(RunCircuitFlow)` over a `ShotDistributingPlanner`, exactly as
// `run_quantum_circuit(n_qpus > 1)` builds it at the edge.
//
// The native (pure-Rust) statevector backend is used so this runs without a live
// Python runtime or Qiskit; the planner re-acquires the GIL only for its
// between-wave `check_signals`, so the interpreter is initialised with
// `prepare_freethreaded_python()` (the same pattern the qmio tests use).
// ─────────────────────────────────────────────────────────────────────────────

fn native_bell_circuit() -> BoundCircuit {
    let bell = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .assign_parameters(&[])
        .expect("bell circuit has no free parameters");
    BoundCircuit::Native(bell)
}

fn native_config(shots: u32, n_qpus: u32, id: &str) -> ExecutionConfig {
    ExecutionConfig {
        id: id.to_string(),
        shots,
        n_qpus,
        infrastructure: "local".to_string(),
        backend_config: BackendConfig::LocalNative { fusion: true },
        opt_level: OptLevel::default(),
        // Fixed seed: these tests assert only shot conservation, and a fixed seed
        // keeps the native backend's sampling deterministic across runs.
        seed: Some(7),
    }
}

/// Distribute `shots` of a Bell circuit across `n_qpus` on the native backend and
/// return the merged counts, driving the real `Scheduler`/`RunCircuitFlow` path.
fn distributed_counts(shots: u32, n_qpus: u32, id: &str) -> HashMap<String, u64> {
    pyo3::prepare_freethreaded_python();
    let backend: Arc<dyn QuantumBackend> = Arc::new(NativeStatevectorBackend::new(7));
    let resources = Resources::new(
        backend,
        Some(Arc::new(ShotDistributingPlanner)),
        Arc::new(native_config(shots, n_qpus, id)),
    )
    .expect("the native backend supports shot distribution");
    let scheduler = Scheduler::ephemeral(resources);
    let mut merged = scheduler
        .run(RunCircuitFlow {
            circuits: vec![native_bell_circuit()],
            shots,
        })
        .expect("distribute-by-shots must succeed on the native backend");
    // The shot-distributing planner merges its replicas into exactly one map.
    merged.pop().unwrap_or_default()
}

#[test]
fn distribute_conserves_shots_when_not_divisible() {
    // 1000 shots over 3 QPUs: remainder 1. The total must be conserved exactly.
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
    // 5 shots over 8 QPUs: base 0, remainder 5 — spread one-per-QPU over the first
    // 5 replicas, total exactly 5.
    let counts = distributed_counts(5, 8, "c3-degenerate");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 5, "shots must be conserved when shots < n_qpus");
}

#[test]
fn distribute_conserves_shots_when_divisible() {
    let counts = distributed_counts(400, 4, "c3-even");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 400);
}

#[test]
fn distribute_rejects_multiple_circuits() {
    // The shot-distributing planner operates on exactly one circuit; more than one
    // is a typed error (surfaced as a ValueError at the FFI edge), never a silent
    // truncation. The `run_quantum_circuit` edge only ever passes one circuit, so
    // this guards the planner directly.
    pyo3::prepare_freethreaded_python();
    let backend: Arc<dyn QuantumBackend> = Arc::new(NativeStatevectorBackend::new(7));
    let resources = Resources::new(
        backend,
        Some(Arc::new(ShotDistributingPlanner)),
        Arc::new(native_config(100, 2, "multi")),
    )
    .unwrap();
    let scheduler = Scheduler::ephemeral(resources);
    let err = scheduler
        .run(RunCircuitFlow {
            circuits: vec![native_bell_circuit(), native_bell_circuit()],
            shots: 100,
        })
        .expect_err("more than one circuit must be a typed error, not silently truncated");
    assert!(
        matches!(
            err,
            InfrastructureError::Backend(BackendError::InvalidCircuitCount {
                expected: 1,
                got: 2
            })
        ),
        "expected InvalidCircuitCount, got {err:?}"
    );
}
