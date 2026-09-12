//! [`Resources`] — a backend paired with its planner and run config — and the
//! thin [`Scheduler`] that runs a [`Flow`] over them.

use std::sync::Arc;

use polypus_infrastructure::{
    CancelToken, ExecutionConfig, InfrastructureError, Planner, QuantumBackend,
};

use crate::flow::Flow;

/// The bound execution context for a flow: a backend, the planner paired with it,
/// and the run configuration. Built once — validating the pairing up front — and
/// handed to [`Scheduler::run`].
pub struct Resources {
    /// The execution backend (native, Aer, CUNQA, QMIO).
    pub backend: Arc<dyn QuantumBackend>,
    /// The planner paired with `backend` (its `default_planner()` unless the edge
    /// chose another, e.g. the shot-distributing planner).
    pub planner: Arc<dyn Planner>,
    /// The run configuration (id, shots, seed, backend config, …).
    pub config: ExecutionConfig,
}

impl Resources {
    /// Pair `backend` with `planner` — or the backend's `default_planner()` when
    /// `None` — and **validate the pairing up front** (plan §4.2): a planner whose
    /// requirements the backend cannot meet (e.g. shot distribution on a backend
    /// that does not support it) fails here, at construction, rather than deep in
    /// a run.
    pub fn new(
        backend: Arc<dyn QuantumBackend>,
        planner: Option<Arc<dyn Planner>>,
        config: ExecutionConfig,
    ) -> Result<Self, InfrastructureError> {
        let planner = planner.unwrap_or_else(|| backend.default_planner());
        planner.requirements().check(&backend.capabilities())?;
        Ok(Self {
            backend,
            planner,
            config,
        })
    }
}

/// Orchestrates a [`Flow`] over [`Resources`].
///
/// A **thin seam** in this iteration (`run` is almost `flow.run(&resources,
/// &cancel)`), introduced now so the explicit Python `Scheduler` of §7.2 (context
/// manager, no-leak `qraise` teardown) is purely additive. The real lifecycle
/// (construction/teardown, session close) is deferred to §7.2.
pub struct Scheduler {
    resources: Resources,
}

impl Scheduler {
    /// A single-run scheduler owning `resources` for the duration of one flow.
    pub fn ephemeral(resources: Resources) -> Self {
        Self { resources }
    }

    /// Run `flow` to completion. Monomorphic (`F: Flow`, never `dyn Flow`) so the
    /// flow keeps its own associated `Output`/`Error`. A fresh [`CancelToken`] is
    /// created per run (plan §4.2). The GIL boundary (`allow_threads`) is the
    /// caller's responsibility (`polypus`), never the scheduler's (ENGINEERING §3).
    pub fn run<F: Flow>(&self, flow: F) -> Result<F::Output, F::Error> {
        let cancel = CancelToken::default();
        flow.run(&self.resources, &cancel)
    }

    /// Release the backend's held resources (SLURM jobs, sessions, reservations).
    pub fn close(&self) {
        self.resources.backend.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_infrastructure::{
        BackendCapabilities, BackendConfig, BackendError, BoundCircuit, Counts, ExecutionConfig,
        OptLevel, SequentialPlanner, ShotDistributingPlanner,
    };
    use std::collections::HashMap;

    fn config() -> ExecutionConfig {
        ExecutionConfig {
            id: "scheduler-test".to_string(),
            shots: 1,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative,
            opt_level: OptLevel::default(),
            seed: Some(7),
        }
    }

    /// A backend that cannot split shots, used to exercise the pairing rejection.
    struct NoDistributionBackend;
    impl QuantumBackend for NoDistributionBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _config: &ExecutionConfig,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs.iter().map(|_| HashMap::new()).collect())
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                max_concurrency: 4,
                supports_shot_distribution: false,
            }
        }
    }

    #[test]
    fn new_defaults_the_planner_and_accepts_a_compatible_pairing() {
        // A native backend + its default SequentialPlanner is always compatible.
        let backend = Arc::new(NoDistributionBackend);
        let resources = Resources::new(backend, None, config())
            .expect("SequentialPlanner (default) needs no shot distribution");
        // The default planner was filled in and validated.
        assert!(resources.planner.requirements().min_concurrency <= 4);
    }

    #[test]
    fn new_rejects_a_planner_the_backend_cannot_satisfy() {
        // The shot-distributing planner requires a backend that supports it; this
        // one does not, so the pairing must fail at construction, not at run time.
        let backend = Arc::new(NoDistributionBackend);
        // `Resources` holds trait objects and is not `Debug`, so assert on the
        // `Result` directly rather than via `expect_err`.
        let result = Resources::new(backend, Some(Arc::new(ShotDistributingPlanner)), config());
        assert!(
            matches!(result, Err(InfrastructureError::IncompatiblePlanner(_))),
            "shot distribution on a non-supporting backend must be rejected at construction",
        );
    }

    #[test]
    fn explicit_sequential_planner_is_accepted() {
        let backend = Arc::new(NoDistributionBackend);
        assert!(Resources::new(backend, Some(Arc::new(SequentialPlanner)), config()).is_ok());
    }
}
