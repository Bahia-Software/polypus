//! Proof that a third party can implement a working [`QuantumBackend`] against
//! `polypus-backend` **alone** — no PyO3, no Qiskit, no Polypus runtime.
//!
//! This file deliberately imports only `polypus_backend` (and `polypus_circuit`,
//! for a native circuit to feed it): if the extraction ever re-introduced a PyO3
//! leak into the contract, this integration test would fail to compile, because
//! `polypus-backend`'s dependency tree carries no `pyo3` (verified in CI with
//! `cargo tree`). It is the executable form of the Fase-3 acceptance criterion "a
//! fifth impl that uses neither PyO3 nor Qiskit compiles against polypus-backend
//! and runs".
//!
//! The backend itself is intentionally trivial — a deterministic "echo" QPU that
//! returns a single all-zeros bitstring with the requested shot count — so the
//! test exercises the *contract*, not any real simulation.

use std::collections::HashMap;

use polypus_backend::{
    BackendCapabilities, BackendError, BoundCircuit, CancelToken, CircuitTask, Counts, OptLevel,
    Planner, QuantumBackend, RunParams, SequentialPlanner, ShotDistributingPlanner,
};
use polypus_circuit::ParameterizedCircuit;

/// A minimal third-party backend: it "runs" each circuit by returning all shots on
/// the all-zeros bitstring (width taken from the circuit's native qubit width, or 1
/// if unknown). Pure Rust, `Send + Sync`, zero external dependencies.
struct EchoBackend;

impl EchoBackend {
    fn zeros_counts(circuit: &BoundCircuit, shots: u32) -> Counts {
        let width = circuit.native_qubit_width().unwrap_or(1).max(1);
        HashMap::from([("0".repeat(width), u64::from(shots))])
    }
}

impl QuantumBackend for EchoBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        // A backend that could not handle a representation would return
        // `BackendError::UnsupportedCircuit`; this one accepts all of them.
        Ok(qcs
            .iter()
            .map(|qc| Self::zeros_counts(qc, params.shots))
            .collect())
    }
}

fn native(n: usize) -> BoundCircuit {
    BoundCircuit::Native(
        ParameterizedCircuit::new(n)
            .measure_all()
            .assign_parameters(&[])
            .unwrap(),
    )
}

fn params(shots: u32) -> RunParams {
    RunParams {
        id: "third-party".to_string(),
        shots,
        seed: None,
        opt_level: OptLevel::default(),
    }
}

/// The third-party backend runs directly through the trait.
#[test]
fn third_party_backend_runs_via_the_trait() {
    let backend = EchoBackend;
    let circuits = [native(3), native(2)];
    let out = backend.run_circuits(&circuits, &params(128)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].get("000"), Some(&128));
    assert_eq!(out[1].get("00"), Some(&128));
}

/// And it composes with the shipped [`SequentialPlanner`] exactly like a built-in
/// backend would — wave sizing, ordering, the between-wave checkpoint and result
/// validation all apply to it unchanged.
#[test]
fn third_party_backend_drives_the_sequential_planner() {
    let backend = EchoBackend;
    let a = native(2);
    let b = native(2);
    let tasks = vec![
        CircuitTask {
            circuit: &a,
            shots: 64,
        },
        CircuitTask {
            circuit: &b,
            shots: 64,
        },
    ];
    let out = SequentialPlanner
        .execute(&backend, &tasks, &params(64), &CancelToken::default())
        .expect("the planner runs a third-party backend");
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|c| c.values().sum::<u64>() == 64));
}

/// It also works under the shot-distributing planner (via the default
/// `run_shots_distributed`), conserving the total across replicas (contract C-3).
#[test]
fn third_party_backend_distributes_shots() {
    let backend = EchoBackend;
    let circuit = native(1);
    let tasks = vec![CircuitTask {
        circuit: &circuit,
        shots: 100,
    }];
    let out = ShotDistributingPlanner::new(3)
        .execute(&backend, &tasks, &params(100), &CancelToken::default())
        .expect("shot distribution works for a third-party backend");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].values().sum::<u64>(), 100);
}

/// The default capabilities are inherited unless overridden — a third party need
/// implement only `run_circuits`.
#[test]
fn third_party_backend_inherits_default_capabilities() {
    let caps: BackendCapabilities = EchoBackend.capabilities();
    assert_eq!(caps.max_concurrency, usize::MAX);
    assert!(caps.supports_shot_distribution);
}
