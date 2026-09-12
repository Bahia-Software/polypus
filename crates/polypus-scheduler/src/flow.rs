//! [`Flow`] — a unit of orchestratable work — and [`RunCircuitFlow`], the
//! run-a-batch-of-circuits flow that builds no oracle and so lives here in the
//! pure scheduler crate (the training flows that build oracles live in
//! `polypus-evaluation`).

use polypus_infrastructure::{BoundCircuit, CancelToken, CircuitTask, Counts, InfrastructureError};

use crate::resources::Resources;

/// A unit of orchestratable work run by [`Scheduler::run`](crate::Scheduler::run).
///
/// `Scheduler::run` is **monomorphic** over `Flow` (never `dyn Flow`, plan §10.8),
/// so each flow keeps its own associated `Output`/`Error` types rather than being
/// erased behind a trait object.
pub trait Flow {
    /// What a successful run produces (e.g. counts, or an optimization outcome).
    type Output;
    /// How a run fails.
    type Error;
    /// Run to completion against `resources`, honouring `cancel` at wave
    /// boundaries. Takes `self` by value: a flow is consumed by its single run.
    fn run(self, resources: &Resources, cancel: &CancelToken) -> Result<Self::Output, Self::Error>;
}

/// Runs a batch of bound circuits through the resources' planner and returns the
/// counts — one [`Counts`] per circuit, in input order.
///
/// The single-vs-distributed decision is **which planner is in
/// [`Resources`]** (chosen by the `polypus` edge when it parses kwargs), not this
/// flow. Converting the counts to a Python object stays at the edge, so this flow
/// — and the whole crate — needs no `pyo3`.
pub struct RunCircuitFlow {
    /// The bound (parameter-free) circuits to run.
    pub circuits: Vec<BoundCircuit>,
    /// Shots per circuit.
    pub shots: u32,
}

impl Flow for RunCircuitFlow {
    type Output = Vec<Counts>;
    type Error = InfrastructureError;

    fn run(
        self,
        resources: &Resources,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        let tasks: Vec<CircuitTask> = self
            .circuits
            .iter()
            .map(|circuit| CircuitTask {
                circuit,
                shots: self.shots,
            })
            .collect();
        resources.planner.execute(
            resources.backend.as_ref(),
            &tasks,
            &resources.config,
            cancel,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Resources, Scheduler};
    use polypus_infrastructure::{
        BackendConfig, BackendError, ExecutionConfig, OptLevel, Planner, PlannerRequirements,
        QuantumBackend,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    fn config() -> ExecutionConfig {
        ExecutionConfig {
            id: "flow-test".to_string(),
            shots: 500,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(2024),
        }
    }

    /// A backend `RunCircuitFlow` hands to the planner. `EchoPlanner` ignores it,
    /// so it only needs to exist and pair (its default capabilities do).
    struct StubBackend;
    impl QuantumBackend for StubBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _config: &ExecutionConfig,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs.iter().map(|_| HashMap::new()).collect())
        }
    }

    /// A GIL-free planner that returns one counts map per task, keyed by input
    /// index and valued by the task's shots — so we can assert `RunCircuitFlow`
    /// built the right tasks, in order, with the flow's shots. Using it (rather
    /// than a real planner) keeps the test — and this crate — free of the
    /// interpreter its between-wave `check_signals` would need.
    struct EchoPlanner;
    impl Planner for EchoPlanner {
        fn requirements(&self) -> PlannerRequirements {
            PlannerRequirements {
                needs_shot_distribution: false,
                min_concurrency: 1,
            }
        }
        fn execute(
            &self,
            _backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            _config: &ExecutionConfig,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            Ok(tasks
                .iter()
                .enumerate()
                .map(|(i, t)| HashMap::from([(format!("task{i}"), u64::from(t.shots))]))
                .collect())
        }
    }

    /// The scheduler runs a `RunCircuitFlow`: it builds one task per circuit with
    /// the flow's shots, delegates to the resources' planner, and returns its
    /// counts one-per-circuit in input order — the machinery the `polypus` edge
    /// will use in Phase 6.
    #[test]
    fn scheduler_runs_a_run_circuit_flow_in_order() {
        let resources =
            Resources::new(Arc::new(StubBackend), Some(Arc::new(EchoPlanner)), config()).unwrap();
        let scheduler = Scheduler::ephemeral(resources);

        let circuits = vec![
            BoundCircuit::Qasm2("a".to_string()),
            BoundCircuit::Qasm2("b".to_string()),
            BoundCircuit::Qasm2("c".to_string()),
        ];
        let counts = scheduler
            .run(RunCircuitFlow {
                circuits,
                shots: 500,
            })
            .expect("the flow must run");

        // One counts map per circuit, in order, each carrying the flow's shots.
        assert_eq!(counts.len(), 3);
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(c.get(&format!("task{i}")), Some(&500));
        }
        scheduler.close();
    }
}
