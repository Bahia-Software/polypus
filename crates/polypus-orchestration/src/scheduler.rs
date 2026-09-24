//! [`Resources`] — a backend paired with its planner and run config — and the
//! thin [`Scheduler`] that runs a [`Flow`] over them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use polypus_infrastructure::{
    CancelToken, InfrastructureError, Planner, QuantumBackend, RunParams,
};

use crate::flow::Flow;

/// Poll interval of the [`CancelWatcher`]: how often it checks the token. Small
/// enough that cancellation latency is imperceptible next to QPU latency, large
/// enough to cost nothing over a long run.
const CANCEL_WATCHER_POLL: Duration = Duration::from_millis(50);

/// A background thread that calls [`QuantumBackend::cancel`] the moment a run's
/// [`CancelToken`] flips — the out-of-band abort a backend blocked mid-wave needs
/// (the cooperative between-wave check cannot reach a call already in flight).
///
/// Spawned by [`Scheduler::run_cancellable`] **only** for backends that opt in via
/// [`QuantumBackend::wants_cancel_watcher`], so the built-in backends (whose calls
/// are self-terminating) never pay for a thread. Its `Drop` stops and joins the
/// thread, so it is torn down when the run returns — normally or by unwind.
struct CancelWatcher {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CancelWatcher {
    fn spawn(backend: Arc<dyn QuantumBackend>, cancel: CancelToken) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                // React only to an explicit cooperative cancel (someone holding the
                // `CancelToken` called `.cancel()` from another thread). Signal the
                // backend once, then exit — a second signal is the backend's own
                // concern (the subprocess bridge latches it).
                //
                // NOTE — deliberately does NOT poll the token's `Interrupt` guard to
                // catch a real Ctrl+C. The guard is `py.check_signals()`, which PyO3
                // documents as a guaranteed **no-op on any non-main thread** (it
                // "does nothing yet still returns Ok(())"; pyo3 marker.rs). This
                // watcher runs on a `std::thread::spawn`ed thread, which is never the
                // Python main thread, so polling it here would silently never fire.
                // A terminal Ctrl+C instead reaches a subprocess backend directly via
                // the OS process group and comes back as `BackendError::Aborted` →
                // `KeyboardInterrupt`; do not reintroduce a guard-poll here.
                if cancel.is_cancelled() {
                    backend.cancel();
                    return;
                }
                std::thread::sleep(CANCEL_WATCHER_POLL);
            }
        });
        CancelWatcher {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for CancelWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The bound execution context for a flow: a backend, the planner paired with it,
/// and the run configuration. Built once — validating the pairing up front — and
/// handed to [`Scheduler::run`].
pub struct Resources {
    /// The execution backend (native, Aer, CUNQA, QMIO).
    pub backend: Arc<dyn QuantumBackend>,
    /// The planner paired with `backend` (its `default_planner()` unless the edge
    /// chose another, e.g. the shot-distributing planner).
    pub planner: Arc<dyn Planner>,
    /// The per-call run parameters a backend/planner reads (id, shots, seed,
    /// opt_level — the pyo3-free [`RunParams`]). An `Arc` so a training [`Flow`] can
    /// share it with the oracle it builds without re-cloning. The construction-time
    /// config (noise model, `n_qpus`, backend config) was already consumed when the
    /// edge built the backend, so it does not travel here.
    pub config: Arc<RunParams>,
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
        config: Arc<RunParams>,
    ) -> Result<Self, InfrastructureError> {
        let planner = planner.unwrap_or_else(|| backend.default_planner());
        planner.requirements().check(&backend.capabilities())?;
        // A shot-distributing planner splits into a fixed replica count; a backend
        // with a fixed execution-unit allocation (CUNQA's `n_qpus`) must match it,
        // or the run would leave QPUs idle / over-subscribe them. Both values default
        // to `None` (native/local/QMIO backends, non-distributing planners), so this
        // only fires for the genuinely dangerous CUNQA mismatch — the two `n_qpus`
        // are otherwise a coincidence the type system does not enforce.
        if let (Some(planner_replicas), Some(backend_replicas)) =
            (planner.required_replicas(), backend.replica_count())
        {
            if planner_replicas != backend_replicas {
                return Err(InfrastructureError::IncompatiblePlanner(format!(
                    "the shot-distributing planner splits shots across {planner_replicas} \
                     replica(s), but the backend allocated {backend_replicas} execution unit(s); \
                     they must match"
                )));
            }
        }
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

    /// Run `flow` to completion with a fresh, private, **guard-less**
    /// [`CancelToken`] — a convenience for callers that keep no handle on
    /// cancellation (its only users today are the crate/integration tests and
    /// simple Rust embeddings).
    ///
    /// **This method does NOT observe host interrupts.** The default token carries
    /// no [`Interrupt`](polypus_infrastructure::Interrupt) guard, so the planner's
    /// between-wave interrupt check is a no-op and a Ctrl+C (`SIGINT`) will *not*
    /// abort the run. That is deliberate — a pure-Rust caller has no CPython signal
    /// state to poll — but it means **any real, interruptible run must use
    /// [`run_cancellable`](Self::run_cancellable) with a token built via
    /// `CancelToken::with_interrupt(...)`** instead. The Polypus Python edge does
    /// exactly this (its `SignalInterrupt` guard backs the token it passes to
    /// `run_cancellable`); this `run` shortcut is not used on any Python-facing
    /// path, precisely so Ctrl+C is never silently lost there.
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
        // Backends that can block for a long time on an external resource opt into an
        // out-of-band abort: a watcher thread that calls `backend.cancel()` when this
        // run's token flips, unblocking a call parked mid-wave. Built-in backends do
        // not opt in, so no thread is spawned for them. The watcher is torn down when
        // `_watcher` drops, i.e. when this run returns (normally or by unwind).
        let _watcher = if self.resources.backend.wants_cancel_watcher() {
            Some(CancelWatcher::spawn(
                Arc::clone(&self.resources.backend),
                cancel.clone(),
            ))
        } else {
            None
        };
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
        BackendCapabilities, BackendError, BoundCircuit, CircuitTask, Counts, OptLevel,
        PlannerRequirements, RunParams, SequentialPlanner, ShotDistributingPlanner,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Mutex};
    use std::thread;

    fn config() -> RunParams {
        RunParams {
            id: "scheduler-test".to_string(),
            shots: 1,
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
            _config: &RunParams,
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

    /// A backend that supports shot distribution and advertises a fixed replica
    /// count (CUNQA-shaped), so the `Resources::new` replica cross-check can be
    /// exercised without a live CUNQA allocation.
    struct FixedReplicaBackend(u32);
    impl QuantumBackend for FixedReplicaBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _config: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs.iter().map(|_| HashMap::new()).collect())
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                max_concurrency: self.0 as usize,
                supports_shot_distribution: true,
            }
        }
        fn replica_count(&self) -> Option<u32> {
            Some(self.0)
        }
    }

    #[test]
    fn new_rejects_a_replica_count_mismatch() {
        // A shot-distributing planner splitting into 3 replicas paired with a backend
        // that allocated 4 execution units is a mis-sized configuration, rejected up
        // front rather than silently leaving a QPU idle.
        let backend = Arc::new(FixedReplicaBackend(4));
        let result = Resources::new(
            backend,
            Some(Arc::new(ShotDistributingPlanner::new(3))),
            Arc::new(config()),
        );
        assert!(
            matches!(result, Err(InfrastructureError::IncompatiblePlanner(_))),
            "a planner/backend replica-count mismatch must be rejected at construction",
        );
    }

    #[test]
    fn new_accepts_a_matching_replica_count() {
        let backend = Arc::new(FixedReplicaBackend(4));
        assert!(
            Resources::new(
                backend,
                Some(Arc::new(ShotDistributingPlanner::new(4))),
                Arc::new(config()),
            )
            .is_ok(),
            "matching replica counts must pair successfully",
        );
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
            Some(Arc::new(ShotDistributingPlanner::new(1))),
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
            _config: &RunParams,
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

    /// A backend that opts into the cancel watcher and **blocks in `run_circuits`**
    /// until its `cancel()` is invoked — modelling a subprocess worker parked in
    /// `recv()`. Proves the watcher (spawned by `run_cancellable`) reaches a call
    /// already in flight mid-wave, which the cooperative between-wave check cannot.
    struct BlockingCancelBackend {
        gate: Arc<(Mutex<bool>, std::sync::Condvar)>,
        cancel_called: Arc<AtomicBool>,
    }
    impl QuantumBackend for BlockingCancelBackend {
        fn run_circuits(
            &self,
            _qcs: &[BoundCircuit],
            _config: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            // Park until cancel() opens the gate — like a worker blocked in recv().
            let (lock, cv) = &*self.gate;
            let mut opened = lock.lock().unwrap_or_else(|p| p.into_inner());
            while !*opened {
                opened = cv.wait(opened).unwrap_or_else(|p| p.into_inner());
            }
            // Unblocked by the abort: report it as the bridge does on an `aborted`
            // reply, so the planner's cancel-translation yields `Cancelled`.
            Err(BackendError::External("aborted by cancel signal".into()))
        }
        fn wants_cancel_watcher(&self) -> bool {
            true
        }
        fn cancel(&self) {
            self.cancel_called.store(true, Ordering::SeqCst);
            let (lock, cv) = &*self.gate;
            *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
            cv.notify_all();
        }
    }

    /// End-to-end: a backend blocked mid-wave is aborted by the watcher when the
    /// run's token is cancelled from another thread — `cancel()` is invoked, the
    /// blocked call unblocks, and the run ends as `Cancelled` (not a hang).
    #[test]
    fn watcher_cancels_a_backend_blocked_mid_wave() {
        let cancel_called = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(BlockingCancelBackend {
            gate: Arc::new((Mutex::new(false), std::sync::Condvar::new())),
            cancel_called: Arc::clone(&cancel_called),
        });
        let resources = Resources::new(
            backend,
            Some(Arc::new(SequentialPlanner)),
            Arc::new(config()),
        )
        .expect("blocking backend pairs with the sequential planner");
        let scheduler = Scheduler::ephemeral(resources);

        let cancel = CancelToken::default();
        let canceller = cancel.clone();
        let handle = thread::spawn(move || {
            // Let the run reach its blocked run_circuits, then cancel.
            thread::sleep(std::time::Duration::from_millis(150));
            canceller.cancel();
        });

        let result = scheduler.run_cancellable(
            RunCircuitFlow {
                circuits: two_circuits(),
                shots: 100,
            },
            &cancel,
        );
        handle.join().expect("canceller thread must not panic");

        assert!(
            cancel_called.load(Ordering::SeqCst),
            "the watcher must have invoked backend.cancel()",
        );
        assert!(
            matches!(result, Err(InfrastructureError::Cancelled)),
            "a watcher-driven mid-wave abort must surface as Cancelled",
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
