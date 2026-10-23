//! [`Flow`] — a unit of orchestratable work — and the flows that implement it:
//! [`RunCircuitFlow`] (run a batch of circuits) and [`TrainFlow`] (variational
//! training). A training flow needs a candidate-scoring oracle, but building that
//! oracle is `pyo3`-touching domain work; so [`TrainFlow`] carries an
//! [`OracleFactory`] — implemented in `polypus-evaluation` — and asks it to build
//! the oracle at run time, keeping this crate `pyo3`-free while every *order* lives
//! here with the [`Scheduler`](crate::Scheduler) that runs it.

use std::sync::Arc;

use polypus_infrastructure::{BoundCircuit, CancelToken, CircuitTask, Counts, InfrastructureError};
use polypus_optimizers::{EvaluationOracle, OptimizationOutcome, OptimizerError};

use crate::dispatch::{dispatch_optimizer, Method, OracleError, OracleErrorSlot, Oracles};
use crate::final_batch::{final_batch, EliteRecorder};
use crate::scheduler::Resources;
use crate::walltime::{watched, SystemClock};

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
/// counts — one [`Counts`] per circuit, in input order, or, when the planner splits
/// one circuit's shots across replicas, one per replica in apportionment order
/// (contract C-3). It calls [`Planner::execute_replicas`](polypus_infrastructure::Planner::execute_replicas),
/// so the per-replica breakdown reaches the edge unmerged; the merged total is
/// computed there.
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
        resources.planner.execute_replicas(
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
/// move it across `py.detach` without this crate ever naming `pyo3`.
pub trait OracleFactory {
    /// Assemble the run's [`Oracles`] from the run context. Called once, inside
    /// [`Flow::run`] — after the [`Scheduler`](crate::Scheduler) has created the
    /// [`CancelToken`], so the oracles capture the exact token this run cancels on.
    /// A factory with only an evaluation oracle returns it `.into()`.
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles;
}

/// Variational training as a [`Flow`]: build the oracle from an [`OracleFactory`],
/// then drive `dispatch_optimizer`. One flow for every training mode (VQC, QML, …)
/// — the mode is entirely in *which* factory it carries, so this crate needs no
/// `pyo3` and no knowledge of circuit templates or observables. Replaces the former
/// per-mode `TrainVqcFlow`/`TrainQmlFlow` that lived in `polypus-evaluation`.
pub struct TrainFlow<F: OracleFactory> {
    /// Builds the candidate-scoring oracle at run time (VQC/QML/… lives here).
    pub factory: F,
    /// Chosen optimizer + its config (built at the edge from the
    /// `polypus.DE/PSO/QNG/CMAES/Adam/SPSA/Rotosolve` object; `Qng` carries its
    /// variance oracle).
    pub method: Method,
    /// Free-parameter count validated at the edge.
    pub dimensions: u32,
    /// Optimizer RNG seed (contract C-7).
    pub seed: u64,
    /// Optional starting point of the search, `dimensions` values (contract
    /// C-5), validated at the edge and again by the optimizer; `None` starts as
    /// before.
    pub initial_point: Option<Vec<f64>>,
    /// Shots per circuit of the final batch (contract C-5): once the optimizer
    /// has returned, its best candidates
    /// ([`Method::final_candidates`] of them, `best_params` first) are
    /// re-evaluated in one more batch with these shots, through
    /// [`EvaluationOracle::evaluate_batch_with_shots`], and the outcome's
    /// [`final_evaluation`](TrainOutcome::final_evaluation) holds the result.
    /// The optimizer's outcome does not change. `Some(0)` is an
    /// `InvalidConfig` error before anything runs; `None` sends no final batch
    /// and runs exactly as before.
    ///
    /// The batch comes after every batch the optimizer sent, so a seeded run is
    /// the same up to it with and without it. It is not part of the CUNQA
    /// walltime projection, and it is sent whatever stopped the optimizer, but
    /// not after an error or an interrupt.
    pub final_shots: Option<u32>,
    /// Shared failure slot: the oracle records into it, `dispatch_optimizer` takes
    /// it, and the edge downcasts + re-raises.
    pub errors: OracleErrorSlot,
}

/// What a [`TrainFlow`] run produces.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TrainOutcome {
    /// The optimizer's outcome, as it returned it: the final batch changes
    /// none of its fields (contract C-5).
    pub optimization: OptimizationOutcome,
    /// The final batch, when the flow had [`final_shots`](TrainFlow::final_shots).
    pub final_evaluation: Option<FinalEvaluation>,
}

/// The final batch of a [`TrainFlow`] with
/// [`final_shots`](TrainFlow::final_shots): the optimizer's best candidates,
/// re-evaluated once after it stopped.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FinalEvaluation {
    /// The candidate with the highest re-evaluated value; between equal
    /// values the earlier one, so `best_params` wins a tie. A `NaN` never wins
    /// over a number; if every value is `NaN`, it is `best_params`.
    pub params: Vec<f64>,
    /// The value of [`params`](Self::params) in the final batch.
    pub fitness: f64,
    /// The re-evaluated candidates, in the order they were sent:
    /// `best_params`, then the best other distinct candidates the optimizer
    /// scored, best first, up to [`Method::final_candidates`].
    pub candidates: Vec<Vec<f64>>,
    /// Their values in the final batch, in the same order.
    pub values: Vec<f64>,
    /// The shots of every circuit of the final batch.
    pub shots: u32,
}

impl<F: OracleFactory> Flow for TrainFlow<F> {
    type Output = TrainOutcome;
    type Error = OracleError;

    fn run(self, resources: &Resources, cancel: &CancelToken) -> Result<TrainOutcome, OracleError> {
        // A Rust caller can pass anything; the Python edge rejects it earlier.
        if self.final_shots == Some(0) {
            return Err(OracleError::Config(OptimizerError::InvalidConfig {
                parameter: "final_shots",
                reason: "must be >= 1, got 0".to_string(),
            }));
        }
        // Build the oracle now (not at the edge): only here do the run's `Resources`
        // and the per-run `CancelToken` both exist, so the oracle captures the exact
        // token this run cancels on — while the *order* still lives with the
        // scheduler that runs it.
        let oracles = self.factory.build(resources, cancel, &self.errors);
        // Only a backend whose allocation expires (CUNQA) gets the overrun watch,
        // which times every oracle of the run as one watch.
        let oracles = match resources.backend.walltime() {
            Some(walltime) => watched(
                oracles,
                walltime,
                &self.method,
                self.dimensions,
                &resources.config.id,
                SystemClock,
            ),
            None => oracles,
        };
        let Some(final_shots) = self.final_shots else {
            let optimization = dispatch_optimizer(
                self.method,
                oracles,
                self.dimensions,
                &self.errors,
                self.seed,
                self.initial_point,
            )?;
            return Ok(TrainOutcome {
                optimization,
                final_evaluation: None,
            });
        };
        // The optimizer scores through a recorder of its best candidates; the
        // final batch goes to the (watched) oracle itself, after the run.
        let count = self.method.final_candidates();
        let Oracles {
            evaluation,
            gradient,
        } = oracles;
        let evaluation: Arc<dyn EvaluationOracle> = Arc::from(evaluation);
        let elite = Arc::new(EliteRecorder::new(Arc::clone(&evaluation), count));
        // An optimizer error or an interrupt returns here: no final batch.
        let optimization = dispatch_optimizer(
            self.method,
            Oracles {
                evaluation: Box::new(Arc::clone(&elite)),
                gradient,
            },
            self.dimensions,
            &self.errors,
            self.seed,
            self.initial_point,
        )?;
        let candidates = elite.final_candidates(&optimization.best_params, count);
        let final_evaluation = final_batch(
            evaluation.as_ref(),
            candidates,
            final_shots,
            &self.errors,
            &resources.config.id,
        )?;
        Ok(TrainOutcome {
            optimization,
            final_evaluation: Some(final_evaluation),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Resources, Scheduler};
    use polypus_infrastructure::{
        BackendError, OptLevel, Planner, PlannerRequirements, QuantumBackend, RunParams,
    };
    use polypus_optimizers::EvaluationOracle;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn config() -> RunParams {
        RunParams {
            id: "flow-test".to_string(),
            shots: 500,
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
            _config: &RunParams,
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
            _config: &RunParams,
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

    /// A planner whose `execute` and `execute_replicas` answer differently (merged
    /// vs. per-replica), so a test can tell which one `RunCircuitFlow` called.
    struct ReplicaPlanner;
    impl Planner for ReplicaPlanner {
        fn requirements(&self) -> PlannerRequirements {
            PlannerRequirements {
                needs_shot_distribution: false,
                min_concurrency: 1,
            }
        }
        fn execute(
            &self,
            _backend: &dyn QuantumBackend,
            _tasks: &[CircuitTask<'_>],
            _config: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            Ok(vec![HashMap::from([("0".to_string(), 500)])])
        }
        fn execute_replicas(
            &self,
            _backend: &dyn QuantumBackend,
            _tasks: &[CircuitTask<'_>],
            _config: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            Ok(vec![
                HashMap::from([("0".to_string(), 250)]),
                HashMap::from([("0".to_string(), 250)]),
            ])
        }
    }

    /// `RunCircuitFlow` returns the planner's *per-replica* counts, not its merged
    /// total, so `run_quantum_circuit(n_qpus > 1)` can expose one dict per replica
    /// (issue #211, contract C-3).
    #[test]
    fn run_circuit_flow_returns_the_per_replica_counts() {
        let resources = Resources::new(
            Arc::new(StubBackend),
            Some(Arc::new(ReplicaPlanner)),
            Arc::new(config()),
        )
        .unwrap();
        let scheduler = Scheduler::ephemeral(resources);
        let counts = scheduler
            .run(RunCircuitFlow {
                circuits: vec![BoundCircuit::Qasm2("a".to_string())],
                shots: 500,
            })
            .expect("the flow must run");
        assert_eq!(counts.len(), 2);
        assert!(counts.iter().all(|c| c.get("0") == Some(&250)));
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
        ) -> Oracles {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Box::new(StubOracle).into()
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
                strategy: polypus_optimizers::DeStrategy::Rand1Bin,
            }),
            dimensions: 3,
            seed: 7,
            initial_point: None,
            final_shots: None,
            errors: OracleErrorSlot::new(),
        };
        // `OracleError` does not implement `Debug`, so match rather than
        // `.expect`.
        let outcome = match scheduler.run(flow) {
            Ok(outcome) => outcome,
            Err(_) => panic!("the train flow must run"),
        };

        // The factory was consumed by the flow, at run time.
        assert!(built.load(Ordering::SeqCst));
        // The optimizer saw the stub oracle: every candidate scores 0.0.
        assert_eq!(outcome.optimization.best_fitness, 0.0);
        assert_eq!(outcome.final_evaluation, None);
        scheduler.close();
    }

    /// A backend whose allocation expires after `walltime` (CUNQA-like) or never.
    struct ExpiringBackend(Option<std::time::Duration>);
    impl QuantumBackend for ExpiringBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _config: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs.iter().map(|_| HashMap::new()).collect())
        }
        fn walltime(&self) -> Option<std::time::Duration> {
            self.0
        }
    }

    /// Every batch takes a millisecond of real time.
    struct SleepyOracle;
    impl EvaluationOracle for SleepyOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            std::thread::sleep(std::time::Duration::from_millis(1));
            vec![0.0; candidates.len()]
        }
    }

    struct SleepyFactory;
    impl OracleFactory for SleepyFactory {
        fn build(
            self,
            _resources: &Resources,
            _cancel: &CancelToken,
            _errors: &OracleErrorSlot,
        ) -> Oracles {
            Box::new(SleepyOracle).into()
        }
    }

    /// Train over `backend` and return the warnings logged for the run.
    fn train_and_collect_warnings(
        run_id: &str,
        walltime: Option<std::time::Duration>,
    ) -> Vec<String> {
        use crate::test_log::{capture_logs, logged_for};
        capture_logs();
        let resources = Resources::new(
            Arc::new(ExpiringBackend(walltime)),
            Some(Arc::new(EchoPlanner)),
            Arc::new(RunParams {
                id: run_id.to_string(),
                ..config()
            }),
        )
        .unwrap();
        let scheduler = Scheduler::ephemeral(resources);
        let flow = TrainFlow {
            factory: SleepyFactory,
            method: Method::De(crate::DeConfig {
                generations: 3,
                population_size: 6,
                tolerance: 0.0,
                patience: 10,
                strategy: polypus_optimizers::DeStrategy::Rand1Bin,
            }),
            dimensions: 2,
            seed: 7,
            initial_point: None,
            final_shots: None,
            errors: OracleErrorSlot::new(),
        };
        assert!(scheduler.run(flow).is_ok(), "the train flow must run");
        scheduler.close();
        logged_for(run_id)
    }

    /// A backend with a walltime the run cannot meet gets exactly one warning.
    #[test]
    fn train_flow_warns_when_the_run_outlives_the_backend_walltime() {
        let warnings = train_and_collect_warnings(
            "flow-walltime-exceeded",
            Some(std::time::Duration::from_nanos(1)),
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("walltime"), "{warnings:?}");
    }

    /// Without a walltime (native, local, QMIO) the oracle is not watched at all.
    #[test]
    fn train_flow_does_not_watch_a_backend_without_walltime() {
        assert!(train_and_collect_warnings("flow-walltime-none", None).is_empty());
    }

    // ── final_shots ───────────────────────────────────────────────────────────

    /// One call the optimizer or the flow made: its candidates, the shots it
    /// asked for (`None`: the run's) and the values returned.
    type Call = (Vec<Vec<f64>>, Option<u32>, Vec<f64>);

    /// Scores `-Σ (xᵢ − 1)²` plus a noise drawn from the number of candidates
    /// scored before, as the native backend seeds circuit `k` of a run from
    /// `k`: a batch sent earlier or later than without the final batch would
    /// change every value after it. It records every call, records an error
    /// in the slot from call `fail_at` on, and cancels the run's token at
    /// call `cancel_at`, then fails as the planner does once cancelled.
    struct NoisyOracle {
        calls: Arc<std::sync::Mutex<Vec<Call>>>,
        scored: std::sync::atomic::AtomicU64,
        errors: OracleErrorSlot,
        cancel: CancelToken,
        fail_at: Option<usize>,
        cancel_at: Option<usize>,
    }

    #[derive(Debug)]
    struct Failed(&'static str);
    impl std::fmt::Display for Failed {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }
    impl std::error::Error for Failed {}

    impl NoisyOracle {
        fn score(&self, candidates: &[Vec<f64>], shots: Option<u32>) -> Vec<f64> {
            use std::sync::atomic::Ordering;
            let call = self.calls.lock().unwrap().len();
            if self.cancel_at == Some(call) {
                self.cancel.cancel();
            }
            let values: Vec<f64> = if self.cancel.is_cancelled() {
                self.errors
                    .record(Box::new(Failed("cancelled")), "flow-test");
                vec![0.0; candidates.len()]
            } else if self.fail_at.is_some_and(|at| call >= at) {
                self.errors.record(Box::new(Failed("backend")), "flow-test");
                vec![0.0; candidates.len()]
            } else {
                candidates
                    .iter()
                    .map(|c| {
                        let k = self.scored.fetch_add(1, Ordering::Relaxed);
                        let noise = (k.wrapping_mul(2_654_435_761) % 1000) as f64 * 1e-4;
                        noise - c.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>()
                    })
                    .collect()
            };
            self.calls
                .lock()
                .unwrap()
                .push((candidates.to_vec(), shots, values.clone()));
            values
        }
    }

    impl EvaluationOracle for NoisyOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.score(candidates, None)
        }
        fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
            self.score(candidates, Some(shots))
        }
    }

    #[derive(Default)]
    struct NoisyFactory {
        calls: Arc<std::sync::Mutex<Vec<Call>>>,
        built: Arc<std::sync::atomic::AtomicBool>,
        fail_at: Option<usize>,
        cancel_at: Option<usize>,
    }

    impl OracleFactory for NoisyFactory {
        fn build(
            self,
            _resources: &Resources,
            cancel: &CancelToken,
            errors: &OracleErrorSlot,
        ) -> Oracles {
            self.built.store(true, std::sync::atomic::Ordering::SeqCst);
            Box::new(NoisyOracle {
                calls: self.calls,
                scored: std::sync::atomic::AtomicU64::new(0),
                errors: errors.clone(),
                cancel: cancel.clone(),
                fail_at: self.fail_at,
                cancel_at: self.cancel_at,
            })
            .into()
        }
    }

    struct UnitVariance;
    impl polypus_optimizers::VarianceOracle for UnitVariance {
        fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
            1.0
        }
    }

    /// A seeded random search over `[-π, π)²` as an external ask/tell
    /// optimizer: `rows` points per `ask()`, never ending on its own.
    struct RandomAskTell {
        state: u64,
        rows: usize,
    }

    impl RandomAskTell {
        /// SplitMix64, mapped to `[-π, π)`.
        fn next(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * unit - 1.0) * std::f64::consts::PI
        }
    }

    impl polypus_optimizers::AskTell for RandomAskTell {
        fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, polypus_optimizers::AskTellError> {
            Ok(Some(
                (0..self.rows)
                    .map(|_| vec![self.next(), self.next()])
                    .collect(),
            ))
        }
        fn tell(
            &mut self,
            _candidates: &[Vec<f64>],
            _values: &[f64],
        ) -> Result<(), polypus_optimizers::AskTellError> {
            Ok(())
        }
    }

    /// A small run of every optimizer, with its name.
    fn every_method() -> Vec<(&'static str, Method)> {
        use crate::{
            AdamConfig, CmaesConfig, DeConfig, ExternalConfig, PsoConfig, QngConfig,
            RotosolveConfig, SpsaConfig,
        };
        use polypus_optimizers::{AdamRule, DeStrategy};
        let bounds = (-std::f64::consts::PI, std::f64::consts::PI);
        let mut methods: Vec<(&'static str, Method)> =
            [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade]
                .into_iter()
                .map(|strategy| {
                    (
                        strategy.as_str(),
                        Method::De(DeConfig {
                            generations: 4,
                            population_size: 6,
                            tolerance: 0.0,
                            patience: 100,
                            strategy,
                        }),
                    )
                })
                .collect();
        methods.extend([
            (
                "pso",
                Method::Pso(PsoConfig {
                    generations: 4,
                    population_size: 6,
                    bounds,
                    inertia_weight: 0.5,
                    cognitive_weight: 1.0,
                    social_weight: 1.0,
                    tolerance: 0.0,
                }),
            ),
            (
                "cmaes",
                Method::Cmaes(CmaesConfig {
                    generations: 4,
                    population_size: None,
                    sigma0: 1.0,
                    tolerance: 0.0,
                    patience: 100,
                    restarts: 0,
                }),
            ),
            (
                "qng",
                Method::Qng(
                    QngConfig {
                        max_iters: 3,
                        learning_rate: 0.1,
                        finite_difference_step: 0.1,
                        bounds,
                        tikhonov_reg: 0.05,
                    },
                    Box::new(UnitVariance),
                ),
            ),
            (
                "adam",
                Method::Adam(AdamConfig {
                    max_iters: 3,
                    learning_rate: 0.1,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    bounds,
                    tolerance: 0.0,
                    patience: 100,
                    finite_difference_step: 0.2,
                    rule: AdamRule::Adam,
                }),
            ),
            (
                "spsa",
                Method::Spsa(SpsaConfig {
                    max_iters: 3,
                    resamplings: 1,
                    perturbation: 0.1,
                    target_step: 0.63,
                    alpha: 0.602,
                    gamma: 0.101,
                    eval_interval: 2,
                }),
            ),
            (
                "rotosolve",
                Method::Rotosolve(RotosolveConfig {
                    sweeps: 2,
                    reevaluate_every: 3,
                    tolerance: 0.0,
                    patience: 1,
                }),
            ),
            (
                "external",
                Method::External(
                    ExternalConfig {
                        max_evaluations: 20,
                    },
                    Box::new(RandomAskTell { state: 5, rows: 4 }),
                ),
            ),
        ]);
        methods
    }

    /// The run of `method` over `factory`, on a backend with `walltime`.
    fn noisy_run(
        method: Method,
        factory: NoisyFactory,
        final_shots: Option<u32>,
        walltime: Option<std::time::Duration>,
    ) -> Result<TrainOutcome, OracleError> {
        let resources = Resources::new(
            Arc::new(ExpiringBackend(walltime)),
            Some(Arc::new(EchoPlanner)),
            Arc::new(config()),
        )
        .unwrap();
        let scheduler = Scheduler::ephemeral(resources);
        let result = scheduler.run(TrainFlow {
            factory,
            method,
            dimensions: 2,
            seed: 11,
            initial_point: None,
            final_shots,
            errors: OracleErrorSlot::new(),
        });
        scheduler.close();
        result
    }

    /// The `count` best distinct finite candidates of `calls`, computed
    /// independently of `EliteRecorder`: each candidate's best value, ties
    /// broken by the first evaluation that reached it.
    fn expected_elite(calls: &[Call], count: usize) -> Vec<Vec<f64>> {
        let mut best: Vec<(Vec<u64>, Vec<f64>, f64, usize)> = Vec::new();
        let scored = calls
            .iter()
            .flat_map(|(candidates, _, values)| candidates.iter().zip(values));
        for (order, (candidate, &value)) in scored.enumerate() {
            if !value.is_finite() {
                continue;
            }
            let bits: Vec<u64> = candidate.iter().map(|x| x.to_bits()).collect();
            match best.iter_mut().find(|(b, ..)| *b == bits) {
                Some(entry) if value > entry.2 => {
                    entry.2 = value;
                    entry.3 = order;
                }
                Some(_) => {}
                None => best.push((bits, candidate.clone(), value, order)),
            }
        }
        best.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then(a.3.cmp(&b.3)));
        best.into_iter().take(count).map(|e| e.1).collect()
    }

    /// With `final_shots`, the run is the run without them, call for call,
    /// then one more batch with those shots: `best_params` and the best other
    /// candidates, five for a population method and one for a local method,
    /// with the highest re-evaluated value reported. With the noise drawn
    /// from the candidates scored before (the native backend's seeding), this
    /// is the seeding guarantee: everything before the final batch is
    /// bit-identical with and without it.
    #[test]
    fn final_shots_add_one_batch_after_an_unchanged_run() {
        for walltime in [None, Some(std::time::Duration::from_secs(3600))] {
            for (name, method) in every_method() {
                let count = method.final_candidates();
                let without = NoisyFactory::default();
                let calls_without = Arc::clone(&without.calls);
                let Ok(plain) = noisy_run(method, without, None, walltime) else {
                    panic!("{name}: the run without final_shots must succeed");
                };
                assert_eq!(plain.final_evaluation, None, "{name}");

                let (_, method) = every_method()
                    .into_iter()
                    .find(|(n, _)| *n == name)
                    .unwrap();
                let with = NoisyFactory::default();
                let calls_with = Arc::clone(&with.calls);
                let Ok(outcome) = noisy_run(method, with, Some(4096), walltime) else {
                    panic!("{name}: the run with final_shots must succeed");
                };
                // The optimizer's outcome and every batch it sent are unchanged.
                assert_eq!(outcome.optimization, plain.optimization, "{name}");
                let calls_without = calls_without.lock().unwrap().clone();
                let mut calls_with = calls_with.lock().unwrap().clone();
                let (candidates, shots, values) = calls_with.pop().expect("a final batch");
                assert_eq!(calls_with, calls_without, "{name}");
                assert!(calls_with.iter().all(|(_, s, _)| s.is_none()), "{name}");
                // One more batch, with the requested shots.
                assert_eq!(shots, Some(4096), "{name}");
                let best_params = &outcome.optimization.best_params;
                assert_eq!(&candidates[0], best_params, "{name}");
                let mut expected = vec![best_params.clone()];
                for c in expected_elite(&calls_without, count) {
                    if expected.len() < count && !expected.contains(&c) {
                        expected.push(c);
                    }
                }
                assert_eq!(candidates, expected, "{name}");
                assert_eq!(candidates.len(), count, "{name}");
                let evaluation = outcome.final_evaluation.expect("a final evaluation");
                assert_eq!(evaluation.candidates, candidates, "{name}");
                assert_eq!(evaluation.values, values, "{name}");
                assert_eq!(evaluation.shots, 4096, "{name}");
                let top = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(evaluation.fitness, top, "{name}");
                let winner = values.iter().position(|&v| v == top).unwrap();
                assert_eq!(evaluation.params, candidates[winner], "{name}");
            }
        }
    }

    /// An oracle failure or an interrupt during the run is the run's error,
    /// and no final batch is sent.
    #[test]
    fn an_error_or_an_interrupt_skips_the_final_batch() {
        for (fail_at, cancel_at) in [(Some(2), None), (None, Some(2))] {
            for (name, method) in every_method() {
                let factory = NoisyFactory {
                    fail_at,
                    cancel_at,
                    ..NoisyFactory::default()
                };
                let calls = Arc::clone(&factory.calls);
                let result = noisy_run(method, factory, Some(4096), None);
                assert!(
                    matches!(result, Err(OracleError::Evaluation(_))),
                    "{name}: the failure must be the run's"
                );
                let calls = calls.lock().unwrap();
                assert!(calls.len() > 2, "{name}");
                assert!(calls.iter().all(|(_, s, _)| s.is_none()), "{name}");
            }
        }
    }

    /// The optimizer's own error skips it too, and the error is unchanged.
    #[test]
    fn an_optimizer_error_skips_the_final_batch() {
        let factory = NoisyFactory::default();
        let calls = Arc::clone(&factory.calls);
        let method = Method::De(crate::DeConfig {
            generations: 3,
            population_size: 2,
            tolerance: 0.0,
            patience: 10,
            strategy: polypus_optimizers::DeStrategy::Rand1Bin,
        });
        let result = noisy_run(method, factory, Some(4096), None);
        assert!(matches!(
            result,
            Err(OracleError::Config(
                OptimizerError::PopulationTooSmall { .. }
            ))
        ));
        assert!(calls.lock().unwrap().is_empty());
    }

    /// The final batch fails as any batch does: the run returns its error.
    #[test]
    fn a_failing_final_batch_is_the_runs_error() {
        // Count the optimizer's calls, then fail the one after them.
        let probe = NoisyFactory::default();
        let calls = Arc::clone(&probe.calls);
        let (_, method) = every_method().remove(0);
        assert!(noisy_run(method, probe, None, None).is_ok());
        let sent = calls.lock().unwrap().len();
        let factory = NoisyFactory {
            fail_at: Some(sent),
            ..NoisyFactory::default()
        };
        let calls = Arc::clone(&factory.calls);
        let (_, method) = every_method().remove(0);
        let result = noisy_run(method, factory, Some(4096), None);
        assert!(matches!(result, Err(OracleError::Evaluation(_))));
        assert_eq!(calls.lock().unwrap().last().unwrap().1, Some(4096));
    }

    /// `final_shots = Some(0)` is rejected before the oracle is built.
    #[test]
    fn zero_final_shots_are_rejected_before_anything_runs() {
        let factory = NoisyFactory::default();
        let built = Arc::clone(&factory.built);
        let (_, method) = every_method().remove(0);
        let result = noisy_run(method, factory, Some(0), None);
        assert!(matches!(
            result,
            Err(OracleError::Config(OptimizerError::InvalidConfig {
                parameter: "final_shots",
                ..
            }))
        ));
        assert!(!built.load(std::sync::atomic::Ordering::SeqCst));
    }
}
