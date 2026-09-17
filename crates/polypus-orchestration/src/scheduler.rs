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
    /// The run configuration (id, shots, seed, backend config, …). An `Arc` so a
    /// training [`Flow`] can share it with the oracle it builds without cloning it
    /// under the GIL (the config holds a `Py<PyAny>` noise model).
    pub config: Arc<ExecutionConfig>,
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
        config: Arc<ExecutionConfig>,
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

    /// Run `flow` to completion with a fresh, private [`CancelToken`] — the common
    /// case, where the caller keeps no handle on cancellation. A convenience over
    /// [`run_cancellable`](Self::run_cancellable) with a default token (plan §4.2).
    pub fn run<F: Flow>(&self, flow: F) -> Result<F::Output, F::Error> {
        self.run_cancellable(flow, &CancelToken::default())
    }

    /// Run `flow` to completion against a caller-supplied [`CancelToken`], so a
    /// clone of that same token held elsewhere — including on another thread — can
    /// request cancellation while this run is in flight.
    ///
    /// [`CancelToken`] is an `Arc<AtomicBool>` inside, so the caller keeps a cheap
    /// clone, hands this method a reference to it, and calls
    /// [`CancelToken::cancel`] from wherever it likes; the planner observes the
    /// request at the next wave boundary and aborts the run with
    /// [`InfrastructureError::Cancelled`] (surfaced through the flow's own error
    /// type). A wave is atomic, so cancellation stops the *next* wave from
    /// launching — it never interrupts a wave mid-flight.
    ///
    /// Monomorphic (`F: Flow`, never `dyn Flow`) so the flow keeps its own
    /// associated `Output`/`Error`. The GIL boundary (`allow_threads`) is the
    /// caller's responsibility (`polypus`), never the scheduler's (ENGINEERING §3).
    pub fn run_cancellable<F: Flow>(
        &self,
        flow: F,
        cancel: &CancelToken,
    ) -> Result<F::Output, F::Error> {
        flow.run(&self.resources, cancel)
    }

    /// Release the backend's held resources (SLURM jobs, sessions, reservations).
    pub fn close(&self) {
        self.resources.backend.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunCircuitFlow;
    use polypus_infrastructure::{
        BackendCapabilities, BackendConfig, BackendError, BoundCircuit, CircuitTask, Counts,
        ExecutionConfig, OptLevel, PlannerRequirements, SequentialPlanner, ShotDistributingPlanner,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Mutex};
    use std::thread;

    fn config() -> ExecutionConfig {
        ExecutionConfig {
            id: "scheduler-test".to_string(),
            shots: 1,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: BackendConfig::LocalNative { fusion: true },
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
        let resources = Resources::new(backend, None, Arc::new(config()))
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
        let result = Resources::new(
            backend,
            Some(Arc::new(ShotDistributingPlanner)),
            Arc::new(config()),
        );
        assert!(
            matches!(result, Err(InfrastructureError::IncompatiblePlanner(_))),
            "shot distribution on a non-supporting backend must be rejected at construction",
        );
    }

    #[test]
    fn explicit_sequential_planner_is_accepted() {
        let backend = Arc::new(NoDistributionBackend);
        assert!(Resources::new(
            backend,
            Some(Arc::new(SequentialPlanner)),
            Arc::new(config())
        )
        .is_ok());
    }

    /// The deterministic hand-off a [`TwoWavePlanner`] performs between its two
    /// waves, so another thread can cancel with no `sleep`-based racing: the
    /// planner announces it has reached the between-wave checkpoint over `reached`
    /// and then blocks on `resume` until the other thread has acted (in the
    /// cancellation test, called `cancel()`). This mirrors the `mpsc` rendezvous
    /// used by `polypus-infrastructure`'s `simulated_rep_server_end_to_end` test.
    struct Rendezvous {
        /// Signalled once the planner is parked at the between-wave checkpoint.
        reached: mpsc::Sender<()>,
        /// Unblocks the planner once the other thread has acted. Wrapped in a
        /// `Mutex` because `mpsc::Receiver` is `!Sync` while a `Planner` must be
        /// `Sync` to live behind `Arc<dyn Planner>` (`mpsc::Sender` needs no such
        /// wrapper — it is already `Sync` when its payload is `Send`).
        resume: Mutex<mpsc::Receiver<()>>,
    }

    /// A GIL-free planner that runs two synthetic waves and honours `cancel`
    /// between them — the same between-wave check `SequentialPlanner` and
    /// `ShotDistributingPlanner` make, but without the interpreter their
    /// `check_signals` needs, keeping this crate `pyo3`-free. It records how many
    /// waves actually ran (so a test can prove the second never launched) and,
    /// when given a [`Rendezvous`], parks at the between-wave checkpoint so a
    /// separate thread can cancel deterministically before the second wave.
    struct TwoWavePlanner {
        rendezvous: Option<Rendezvous>,
        waves_run: Arc<AtomicUsize>,
    }

    impl Planner for TwoWavePlanner {
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
            _config: &ExecutionConfig,
            cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            for wave in 0..2 {
                // The identical between-wave cancellation check the real planners
                // make: a set token stops the *next* wave from launching.
                if cancel.is_cancelled() {
                    return Err(InfrastructureError::Cancelled);
                }
                self.waves_run.fetch_add(1, Ordering::SeqCst);
                // After the first wave, hand the between-wave window to another
                // thread and block until it has acted, so the next `is_cancelled()`
                // read is ordered strictly *after* that thread's `cancel()` — a
                // deterministic hand-off, never a timing race.
                if wave == 0 {
                    if let Some(rendezvous) = &self.rendezvous {
                        rendezvous.reached.send(()).unwrap();
                        rendezvous.resume.lock().unwrap().recv().unwrap();
                    }
                }
            }
            Ok(Vec::new())
        }
    }

    fn two_circuits() -> Vec<BoundCircuit> {
        vec![
            BoundCircuit::Qasm2("a".to_string()),
            BoundCircuit::Qasm2("b".to_string()),
        ]
    }

    /// A [`CancelToken`] obtained *before* a run, cloned to a second thread, and
    /// cancelled from there while the run is blocked mid-flight, stops the run at
    /// the next wave boundary with [`InfrastructureError::Cancelled`] — the token
    /// the doc comment promises is "shared between the caller and a planner".
    ///
    /// Determinism (no `sleep`): the planner runs wave 0, then parks at the
    /// between-wave checkpoint (`reached.send`) and blocks (`resume.recv`). Only
    /// then does the canceller thread wake (`reached.recv`), call `cancel()`, and
    /// release the planner (`resume.send`). The planner's next `is_cancelled()`
    /// read is therefore ordered strictly after `cancel()`, so it must observe the
    /// request and abort before wave 1 ever runs.
    #[test]
    fn run_cancellable_stops_at_the_next_wave_when_cancelled_from_another_thread() {
        let (reached_tx, reached_rx) = mpsc::channel::<()>();
        let (resume_tx, resume_rx) = mpsc::channel::<()>();
        let waves_run = Arc::new(AtomicUsize::new(0));

        let planner = TwoWavePlanner {
            rendezvous: Some(Rendezvous {
                reached: reached_tx,
                resume: Mutex::new(resume_rx),
            }),
            waves_run: Arc::clone(&waves_run),
        };
        let resources = Resources::new(
            Arc::new(NoDistributionBackend),
            Some(Arc::new(planner)),
            Arc::new(config()),
        )
        .expect("the two-wave planner needs no shot distribution");
        let scheduler = Scheduler::ephemeral(resources);

        // The caller holds the token before the run and shares a clone with the
        // thread that will cancel it.
        let cancel = CancelToken::default();
        let canceller_token = cancel.clone();
        let canceller = thread::spawn(move || {
            // Block until the run is parked at the between-wave checkpoint, then
            // cancel and let it proceed to its next `is_cancelled()` check.
            reached_rx
                .recv()
                .expect("the planner must reach the checkpoint");
            canceller_token.cancel();
            resume_tx
                .send(())
                .expect("the planner must still be waiting");
        });

        let result = scheduler.run_cancellable(
            RunCircuitFlow {
                circuits: two_circuits(),
                shots: 100,
            },
            &cancel,
        );
        canceller
            .join()
            .expect("the canceller thread must not panic");

        // `InfrastructureError` is not `PartialEq`, so match rather than compare.
        assert!(
            matches!(result, Err(InfrastructureError::Cancelled)),
            "a cross-thread cancel must abort the run with Cancelled",
        );
        // And it stopped *early*: only the first wave ran; the second never launched.
        assert_eq!(
            waves_run.load(Ordering::SeqCst),
            1,
            "cancellation must stop the next wave from launching",
        );
        scheduler.close();
    }

    /// The happy path of the new entry point: with a token that is never
    /// cancelled, `run_cancellable` drives the flow through both waves to a normal
    /// `Ok` — regression cover so exposing the handle did not change the
    /// uncancelled behaviour `run` has always had.
    #[test]
    fn run_cancellable_completes_when_its_token_is_never_cancelled() {
        let waves_run = Arc::new(AtomicUsize::new(0));
        let planner = TwoWavePlanner {
            rendezvous: None,
            waves_run: Arc::clone(&waves_run),
        };
        let resources = Resources::new(
            Arc::new(NoDistributionBackend),
            Some(Arc::new(planner)),
            Arc::new(config()),
        )
        .expect("the two-wave planner needs no shot distribution");
        let scheduler = Scheduler::ephemeral(resources);

        let result = scheduler.run_cancellable(
            RunCircuitFlow {
                circuits: two_circuits(),
                shots: 100,
            },
            &CancelToken::default(),
        );

        assert!(result.is_ok(), "an uncancelled run must complete normally");
        assert_eq!(
            waves_run.load(Ordering::SeqCst),
            2,
            "both waves must run when the token is never cancelled",
        );
        scheduler.close();
    }
}
