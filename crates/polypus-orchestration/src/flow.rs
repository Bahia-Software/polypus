//! [`Flow`] — a unit of orchestratable work — and the flows that implement it:
//! [`RunCircuitFlow`] (run a batch of circuits) and [`TrainFlow`] (variational
//! training). A training flow needs a candidate-scoring oracle, but building that
//! oracle is `pyo3`-touching domain work; so [`TrainFlow`] carries an
//! [`OracleFactory`] — implemented in `polypus-evaluation` — and asks it to build
//! the oracle at run time, keeping this crate `pyo3`-free while every *order* lives
//! here with the [`Scheduler`](crate::Scheduler) that runs it.

use polypus_infrastructure::{BoundCircuit, CancelToken, CircuitTask, Counts, InfrastructureError};
use polypus_optimizers::{EvaluationOracle, OptimizationOutcome};

use crate::dispatch::{dispatch_optimizer, Method, OracleError, OracleErrorSlot};
use crate::scheduler::Resources;

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

/// Builds the candidate-scoring oracle a [`TrainFlow`] optimizes.
///
/// This is the seam that lets the training *order* live here with the
/// [`Scheduler`](crate::Scheduler) while the `pyo3`-touching work of assembling the
/// oracle (binding circuit templates, wiring observables) stays in
/// `polypus-evaluation`. [`TrainFlow`] holds a factory; when the scheduler runs the
/// flow, it hands the factory the run's [`Resources`], the per-run [`CancelToken`]
/// and the shared [`OracleErrorSlot`], and the factory returns the ready oracle.
///
/// `build` consumes the factory (`self`) — a flow builds its oracle exactly once —
/// and [`TrainFlow`] is generic over the concrete factory (never `dyn`), so the
/// flow keeps the factory's auto-derived thread markers and the `polypus` edge can
/// move it across `py.allow_threads` without this crate ever naming `pyo3`.
pub trait OracleFactory {
    /// Assemble the oracle from the run context. Called once, inside
    /// [`Flow::run`] — after the [`Scheduler`](crate::Scheduler) has created the
    /// [`CancelToken`], so the oracle captures the exact token this run cancels on.
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Box<dyn EvaluationOracle>;
}

/// Variational training as a [`Flow`]: build the oracle from an [`OracleFactory`],
/// then drive `dispatch_optimizer`. One flow for every training mode (VQC, QML, …)
/// — the mode is entirely in *which* factory it carries, so this crate needs no
/// `pyo3` and no knowledge of circuit templates or observables. Replaces the former
/// per-mode `TrainVqcFlow`/`TrainQmlFlow` that lived in `polypus-evaluation`.
pub struct TrainFlow<F: OracleFactory> {
    /// Builds the candidate-scoring oracle at run time (VQC/QML/… lives here).
    pub factory: F,
    /// Chosen optimizer + its config (built at the edge from the `polypus.DE/PSO/QNG`
    /// object; `Qng` carries its variance oracle).
    pub method: Method,
    /// Free-parameter count validated at the edge.
    pub dimensions: u32,
    /// Optimizer RNG seed (contract C-7).
    pub seed: u64,
    /// Shared failure slot: the oracle records into it, `dispatch_optimizer` takes
    /// it, and the edge downcasts + re-raises.
    pub errors: OracleErrorSlot,
}

impl<F: OracleFactory> Flow for TrainFlow<F> {
    type Output = OptimizationOutcome;
    type Error = OracleError;

    fn run(
        self,
        resources: &Resources,
        cancel: &CancelToken,
    ) -> Result<OptimizationOutcome, OracleError> {
        // Build the oracle now (not at the edge): only here do the run's `Resources`
        // and the per-run `CancelToken` both exist, so the oracle captures the exact
        // token this run cancels on — while the *order* still lives with the
        // scheduler that runs it.
        let oracle = self.factory.build(resources, cancel, &self.errors);
        dispatch_optimizer(
            self.method,
            oracle,
            self.dimensions,
            &self.errors,
            self.seed,
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
        let resources = Resources::new(
            Arc::new(StubBackend),
            Some(Arc::new(EchoPlanner)),
            Arc::new(config()),
        )
        .unwrap();
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

    /// A stub oracle: every candidate scores `0.0`. It ignores the backend/planner
    /// entirely, so the optimizer runs without touching the (unused) `Resources`.
    struct StubOracle;
    impl EvaluationOracle for StubOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            vec![0.0; candidates.len()]
        }
    }

    /// An `OracleFactory` that flips a shared flag when consumed, then returns a
    /// `StubOracle` — so the test can prove the flow built its oracle *through the
    /// factory* rather than receiving it pre-built from the edge.
    struct StubFactory(Arc<std::sync::atomic::AtomicBool>);
    impl OracleFactory for StubFactory {
        fn build(
            self,
            _resources: &Resources,
            _cancel: &CancelToken,
            _errors: &OracleErrorSlot,
        ) -> Box<dyn EvaluationOracle> {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Box::new(StubOracle)
        }
    }

    /// `TrainFlow` builds its oracle through the [`OracleFactory`] and optimizes it:
    /// the factory is consumed exactly once, at run time (not at the edge), and the
    /// run returns the optimizer's outcome over that oracle — the seam the `polypus`
    /// edge relies on.
    #[test]
    fn train_flow_builds_its_oracle_through_the_factory() {
        use crate::DeConfig;
        use std::sync::atomic::{AtomicBool, Ordering};

        let built = Arc::new(AtomicBool::new(false));
        let resources = Resources::new(
            Arc::new(StubBackend),
            Some(Arc::new(EchoPlanner)),
            Arc::new(config()),
        )
        .unwrap();
        let scheduler = Scheduler::ephemeral(resources);

        let flow = TrainFlow {
            factory: StubFactory(Arc::clone(&built)),
            method: Method::De(DeConfig {
                generations: 2,
                population_size: 6,
                tolerance: 0.0,
                patience: 10,
            }),
            dimensions: 3,
            seed: 7,
            errors: OracleErrorSlot::new(),
        };
        // `OracleError` is not `Debug` (it may carry a boxed `PyErr`), so match
        // rather than `.expect`.
        let outcome = match scheduler.run(flow) {
            Ok(outcome) => outcome,
            Err(_) => panic!("the train flow must run"),
        };

        // The factory was consumed by the flow, at run time.
        assert!(built.load(Ordering::SeqCst));
        // The optimizer saw the stub oracle: every candidate scores 0.0.
        assert_eq!(outcome.best_fitness, 0.0);
        scheduler.close();
    }
}
