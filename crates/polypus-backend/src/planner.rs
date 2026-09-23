//! Execution planning — *how* a set of circuits is run on a backend.
//!
//! The oracle owns the *what* (bind candidates, reduce counts → fitness); the
//! [`Planner`] owns the *how*: split the work into waves capped by the backend's
//! concurrency, run each wave, honour cancellation and host interrupts between
//! waves, merge shot-split results, and preserve input order. Two consumers share
//! one trait: a circuit run calls [`Planner::execute`] (wants counts) and the
//! training oracle calls [`Planner::evaluate`] (wants one `f64` per circuit).
//!
//! Two concrete planners reproduce today's two execution paths exactly:
//! [`SequentialPlanner`] and [`ShotDistributingPlanner`].
//!
//! **pyo3-free interrupt discipline.** Between waves a planner runs the host's
//! interrupt check via the [`Interrupt`] guard optionally carried on the
//! [`CancelToken`]. The Polypus edge backs that guard with `py.check_signals()`
//! so a pending Ctrl+C aborts the run at the next wave boundary; a pure-Rust
//! caller uses a guard-less token and the check is a no-op. This is what lets the
//! planner live in a crate that never names PyO3.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use polypus_observable::CostObservable;

use crate::error::{BackendError, InfrastructureError};
use crate::{validate_run_results, BoundCircuit, QuantumBackend, RunParams};

/// Native measurement counts for one circuit: bitstring → count.
pub type Counts = HashMap<String, u64>;

/// One unit of work: a bound circuit and the shots it needs. `shots` is the
/// single source of truth — a planner reads it here, never from `params`.
pub struct CircuitTask<'a> {
    /// The bound (parameter-free) circuit to run.
    pub circuit: &'a BoundCircuit,
    /// Shots for this circuit (`u32`, matching [`RunParams::shots`]).
    pub shots: u32,
}

/// A host-runtime interrupt check run at each wave boundary.
///
/// The Polypus FFI edge supplies an implementation backed by `py.check_signals()`
/// (so a pending Ctrl+C becomes an `Err` carrying the `KeyboardInterrupt`, boxed
/// in [`BackendError::External`] and re-raised verbatim at the edge). Pure-Rust
/// callers attach no guard, and the check is a no-op. This is the seam that keeps
/// the planner — and this whole crate — free of PyO3.
pub trait Interrupt: Send + Sync {
    /// Return `Err` to abort the run at this wave boundary.
    fn poll(&self) -> Result<(), BackendError>;
}

/// Cooperative cancellation shared between the caller and a planner, plus an
/// optional host [`Interrupt`] guard.
///
/// A wave is atomic, so a set token stops the *next* wave from launching; it never
/// interrupts a wave mid-flight. (Aborting a wave already in flight is a separate,
/// out-of-band affordance — see [`QuantumBackend::cancel`].) Cheap to clone: an
/// `Arc<AtomicBool>` and an `Option<Arc<dyn Interrupt>>`.
#[derive(Clone, Default)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
    interrupt: Option<Arc<dyn Interrupt>>,
}

impl CancelToken {
    /// A token with a host [`Interrupt`] guard attached (the Polypus edge passes a
    /// `py.check_signals()`-backed guard here). Cooperative cancellation starts
    /// un-requested.
    pub fn with_interrupt(interrupt: Arc<dyn Interrupt>) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            interrupt: Some(interrupt),
        }
    }

    /// Request cancellation. Takes effect at the next wave boundary.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Run the host interrupt check (a no-op when no guard is attached). Planners
    /// call this at every wave boundary, after the cooperative-cancel check. A
    /// pending interrupt is mapped to [`InfrastructureError::Backend`] so its boxed
    /// original exception re-raises verbatim at the FFI edge.
    pub fn poll_interrupt(&self) -> Result<(), InfrastructureError> {
        if let Some(guard) = &self.interrupt {
            guard.poll().map_err(InfrastructureError::Backend)?;
        }
        Ok(())
    }
}

/// What a backend can do, so a planner can size its waves.
#[derive(Debug)]
pub struct BackendCapabilities {
    /// Most circuits to run concurrently in one wave (native/local: a memory
    /// budget; CUNQA: `n_qpus`; QMIO: 1).
    pub max_concurrency: usize,
    /// Whether the backend can split one circuit's shots across replicas and
    /// merge them (contract C-3).
    pub supports_shot_distribution: bool,
}

/// What a planner needs of its backend, contrasted with [`BackendCapabilities`]
/// when the two are paired.
pub struct PlannerRequirements {
    /// The planner splits one circuit's shots across replicas.
    pub needs_shot_distribution: bool,
    /// The minimum `max_concurrency` the planner can operate with.
    pub min_concurrency: usize,
}

impl PlannerRequirements {
    /// Reject an incompatible pairing up front, so a bad combination fails at
    /// construction rather than deep in a run.
    pub fn check(&self, caps: &BackendCapabilities) -> Result<(), InfrastructureError> {
        if self.needs_shot_distribution && !caps.supports_shot_distribution {
            return Err(InfrastructureError::IncompatiblePlanner(
                "this planner distributes a circuit's shots across replicas, which the backend \
                 does not support"
                    .to_string(),
            ));
        }
        if caps.max_concurrency < self.min_concurrency {
            return Err(InfrastructureError::IncompatiblePlanner(format!(
                "this planner needs a concurrency of at least {}, but the backend offers {}",
                self.min_concurrency, caps.max_concurrency
            )));
        }
        Ok(())
    }
}

/// Plans *how* a batch of [`CircuitTask`]s runs on a backend. Object-safe
/// (`&self`, no generics) so it can be held as `Arc<dyn Planner>`.
pub trait Planner: Send + Sync {
    /// What this planner needs of its backend.
    fn requirements(&self) -> PlannerRequirements;

    /// The number of replicas this planner splits one circuit's shots across, when
    /// that is a fixed property of the planner ([`ShotDistributingPlanner`] returns
    /// its `n_qpus`). `None` (the default) means the planner does not distribute
    /// shots across a fixed replica count.
    ///
    /// Paired with [`QuantumBackend::replica_count`] where a planner and backend are
    /// combined (`Resources::new`): if both report a
    /// fixed count and they disagree, the pairing is rejected, so a shot-distributing
    /// planner can never be silently mis-sized against the backend's allocation.
    fn required_replicas(&self) -> Option<u32> {
        None
    }

    /// Run every task and return merged counts, one per task, in input order.
    /// Owns the wave loop, the concurrency cap, the between-wave cancellation /
    /// interrupt check and, if shots are split, the C-3 merge.
    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError>;

    /// Run every task and reduce it to one `f64`. The default preserves today's
    /// numerics exactly (`execute`, then `expectation_batch`).
    fn evaluate(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        reducer: &dyn CostObservable,
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<f64>, InfrastructureError> {
        let counts = self.execute(backend, tasks, params, cancel)?;
        reducer
            .expectation_batch(&counts)
            .map_err(InfrastructureError::Observable)
    }
}

/// Runs the tasks in sequential, atomic waves of `≤ max_concurrency` circuits,
/// one `run_circuits` call per wave, merging in input order — the default planner
/// for every backend.
pub struct SequentialPlanner;

impl Planner for SequentialPlanner {
    fn requirements(&self) -> PlannerRequirements {
        PlannerRequirements {
            needs_shot_distribution: false,
            min_concurrency: 1,
        }
    }

    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        // An empty batch yields zero `tasks.chunks(wave)` iterations, so the wave
        // loop below — including its `cancel.is_cancelled()` and between-wave
        // interrupt check — never runs. Check both explicitly here so the "check at
        // least once per `execute` call" guarantee holds regardless of batch size,
        // honouring a pending Ctrl+C/cancellation even when there is nothing to run.
        if tasks.is_empty() {
            if cancel.is_cancelled() {
                return Err(InfrastructureError::Cancelled);
            }
            cancel.poll_interrupt()?;
            return Ok(Vec::new());
        }
        // Size the wave with the *batch-aware* cap: native/local derive it from a
        // statevector memory budget scaled by this batch's widest circuit
        // (`capabilities_for`), so a high-qubit batch is split into several waves
        // instead of one. Sizing can read a circuit width through the provider SDK
        // (local's Qiskit `num_qubits`), so it is fallible: an interrupt raised
        // during that read propagates here verbatim, never swallowed.
        let wave = backend.capabilities_for(tasks)?.max_concurrency.max(1);
        let mut out: Vec<Counts> = Vec::with_capacity(tasks.len());
        for chunk in tasks.chunks(wave) {
            if cancel.is_cancelled() {
                return Err(InfrastructureError::Cancelled);
            }
            // Tasks are uniform-shots in this phase, so one `RunParams` carries the
            // whole wave. `shots` still comes from the task, the single source of
            // truth.
            let shots = chunk[0].shots;
            let circuits: Vec<BoundCircuit> = chunk.iter().map(|t| t.circuit.duplicate()).collect();
            let mut cfg = params.clone();
            cfg.shots = shots;
            let counts = backend
                .run_circuits(&circuits, &cfg)
                .map_err(InfrastructureError::Backend)?;
            validate_run_results(&counts, chunk.len(), shots)
                .map_err(InfrastructureError::Backend)?;
            // Honour a pending interrupt/Ctrl+C after each wave's run.
            cancel.poll_interrupt()?;
            out.extend(counts);
        }
        Ok(out)
    }
}

/// Runs *one* circuit by splitting its shots across `n_qpus` replicas and merging
/// them (contract C-3). The replicas run in waves of `≤ max_concurrency` (QMIO: 1,
/// CUNQA: `n_qpus`), with a between-wave interrupt check so a distributed-shots run
/// stays interruptible on slow/real hardware, just like [`SequentialPlanner`].
/// Opt-in; the `polypus` edge selects it for a single-circuit run with `n_qpus > 1`.
///
/// The replica count travels on the planner itself (`n_qpus`), not on the per-call
/// [`RunParams`] — it is a property of *how the run is distributed*, decided when
/// the planner is chosen, so the per-call parameter surface stays minimal.
pub struct ShotDistributingPlanner {
    /// Number of replicas to apportion the shots across.
    n_qpus: u32,
}

impl ShotDistributingPlanner {
    /// A planner that splits one circuit's shots across `n_qpus` replicas.
    pub fn new(n_qpus: u32) -> Self {
        Self { n_qpus }
    }
}

impl Planner for ShotDistributingPlanner {
    fn requirements(&self) -> PlannerRequirements {
        PlannerRequirements {
            needs_shot_distribution: true,
            min_concurrency: 1,
        }
    }

    fn required_replicas(&self) -> Option<u32> {
        Some(self.n_qpus)
    }

    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        // This planner distributes a single circuit's shots; reject any other
        // count with a typed error rather than panicking or silently dropping
        // circuits.
        if tasks.len() != 1 {
            return Err(InfrastructureError::Backend(
                BackendError::InvalidCircuitCount {
                    expected: 1,
                    got: tasks.len(),
                },
            ));
        }
        let task = &tasks[0];
        let shots = task.shots;
        let n_qpus = self.n_qpus;
        // Apportion the shots (base + one extra on the first `remainder` replicas),
        // conserving the total exactly (contract C-3); `n_qpus >= 1` and
        // `shots >= 1` are guaranteed by the Python boundary.
        let base = shots / n_qpus;
        let remainder = shots % n_qpus;
        let shot_batches: Vec<u32> = (0..n_qpus)
            .map(|i| if i < remainder { base + 1 } else { base })
            .collect();
        // Cap the in-flight replicas at the backend's concurrency (QMIO: 1,
        // CUNQA: `n_qpus`; unbounded backends run the whole array in one chunk),
        // running the interrupt check after every chunk so a slow/real-hardware run
        // stays interruptible between waves. Chunking is numerically transparent:
        // `run_shots_distributed` seeds each replica from a per-backend contiguous
        // block, independent of how the batch is chunked (C-7), and the C-3 merge
        // below accumulates across chunks.
        let wave = backend.capabilities().max_concurrency.max(1);
        let mut merged: Counts = HashMap::new();
        for chunk in shot_batches.chunks(wave) {
            if cancel.is_cancelled() {
                return Err(InfrastructureError::Cancelled);
            }
            let counts_vec = backend
                .run_shots_distributed(task.circuit, chunk, params)
                .map_err(InfrastructureError::Backend)?;
            // Merge this chunk's replicas into the running result (the merge, C-3,
            // lives in the planner).
            for counts in counts_vec {
                for (k, v) in counts {
                    *merged.entry(k).or_insert(0) += v;
                }
            }
            // Honour a pending interrupt/Ctrl+C after each chunk's run.
            cancel.poll_interrupt()?;
        }
        // Validate once over the fully-merged map: shot conservation (C-3) is a
        // property of the total, unaffected by how many backend calls produced it.
        validate_run_results(std::slice::from_ref(&merged), 1, shots)
            .map_err(InfrastructureError::Backend)?;
        Ok(vec![merged])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptLevel;
    use std::sync::Mutex;

    /// A backend that records the length of every `shot_batches` slice
    /// `ShotDistributingPlanner` hands it (via `run_shots_distributed`), so a test
    /// can prove how many checkpoints a run was split into and that each respects
    /// the concurrency cap. Its `max_concurrency` is configurable, and it can
    /// optionally cancel a shared `CancelToken` on its first call to model a Ctrl+C
    /// arriving mid-run.
    struct RecordingBackend {
        max_concurrency: usize,
        call_sizes: Mutex<Vec<usize>>,
        cancel_on_first_call: Option<CancelToken>,
    }

    impl RecordingBackend {
        fn new(max_concurrency: usize) -> Self {
            Self {
                max_concurrency,
                call_sizes: Mutex::new(Vec::new()),
                cancel_on_first_call: None,
            }
        }

        fn cancelling(max_concurrency: usize, token: CancelToken) -> Self {
            Self {
                max_concurrency,
                call_sizes: Mutex::new(Vec::new()),
                cancel_on_first_call: Some(token),
            }
        }

        fn call_sizes(&self) -> Vec<usize> {
            self.call_sizes
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }
    }

    impl QuantumBackend for RecordingBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs
                .iter()
                .map(|_| HashMap::from([("0".to_string(), u64::from(params.shots))]))
                .collect())
        }

        fn run_shots_distributed(
            &self,
            _qc: &BoundCircuit,
            shot_batches: &[u32],
            _params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            let mut sizes = self.call_sizes.lock().unwrap_or_else(|p| p.into_inner());
            let first_call = sizes.is_empty();
            sizes.push(shot_batches.len());
            drop(sizes);
            if first_call {
                if let Some(token) = &self.cancel_on_first_call {
                    token.cancel();
                }
            }
            Ok(shot_batches
                .iter()
                .map(|&s| HashMap::from([("0".to_string(), u64::from(s))]))
                .collect())
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                max_concurrency: self.max_concurrency,
                supports_shot_distribution: true,
            }
        }
    }

    fn params(shots: u32) -> RunParams {
        RunParams {
            id: "shot-dist-test".to_string(),
            shots,
            seed: Some(7),
            opt_level: OptLevel::default(),
        }
    }

    fn one_task(circuit: &BoundCircuit, shots: u32) -> Vec<CircuitTask<'_>> {
        vec![CircuitTask { circuit, shots }]
    }

    #[test]
    fn small_cap_splits_into_multiple_checkpoints() {
        let backend = RecordingBackend::new(1); // QMIO-shaped cap.
        let circuit = BoundCircuit::Qasm2(String::new());
        let (shots, n_qpus) = (100u32, 4u32);
        let out = ShotDistributingPlanner::new(n_qpus)
            .execute(
                &backend,
                &one_task(&circuit, shots),
                &params(shots),
                &CancelToken::default(),
            )
            .expect("execute succeeds");

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].values().sum::<u64>(), u64::from(shots));

        let sizes = backend.call_sizes();
        assert!(
            sizes.len() > 1,
            "expected multiple checkpoints, got {sizes:?}"
        );
        assert!(
            sizes.iter().all(|&s| s <= 1),
            "each chunk must respect max_concurrency = 1: {sizes:?}"
        );
        assert_eq!(sizes.iter().sum::<usize>(), n_qpus as usize);
    }

    #[test]
    fn cancellation_between_checkpoints_stops_before_the_next_chunk() {
        let cancel = CancelToken::default();
        let backend = RecordingBackend::cancelling(1, cancel.clone());
        let circuit = BoundCircuit::Qasm2(String::new());
        let (shots, n_qpus) = (100u32, 4u32);

        let err = ShotDistributingPlanner::new(n_qpus)
            .execute(
                &backend,
                &one_task(&circuit, shots),
                &params(shots),
                &cancel,
            )
            .expect_err("a cancellation between chunks must abort the run");
        assert!(matches!(err, InfrastructureError::Cancelled));
        assert_eq!(backend.call_sizes().len(), 1);
    }

    #[test]
    fn unbounded_cap_runs_in_a_single_call() {
        let backend = RecordingBackend::new(usize::MAX);
        let circuit = BoundCircuit::Qasm2(String::new());
        let (shots, n_qpus) = (100u32, 4u32);
        let out = ShotDistributingPlanner::new(n_qpus)
            .execute(
                &backend,
                &one_task(&circuit, shots),
                &params(shots),
                &CancelToken::default(),
            )
            .expect("execute succeeds");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].values().sum::<u64>(), u64::from(shots));
        assert_eq!(backend.call_sizes(), vec![n_qpus as usize]);
    }

    #[test]
    fn cap_equal_to_n_qpus_runs_in_a_single_call() {
        let n_qpus = 4u32;
        let backend = RecordingBackend::new(n_qpus as usize); // CUNQA: cap == n_qpus.
        let circuit = BoundCircuit::Qasm2(String::new());
        let shots = 100u32;
        let out = ShotDistributingPlanner::new(n_qpus)
            .execute(
                &backend,
                &one_task(&circuit, shots),
                &params(shots),
                &CancelToken::default(),
            )
            .expect("execute succeeds");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].values().sum::<u64>(), u64::from(shots));
        assert_eq!(backend.call_sizes(), vec![n_qpus as usize]);
    }

    #[test]
    fn requirements_reject_shot_distribution_when_unsupported() {
        let caps = BackendCapabilities {
            max_concurrency: 8,
            supports_shot_distribution: false,
        };
        let err = ShotDistributingPlanner::new(4)
            .requirements()
            .check(&caps)
            .unwrap_err();
        assert!(matches!(err, InfrastructureError::IncompatiblePlanner(_)));
    }

    #[test]
    fn requirements_accept_shot_distribution_when_supported() {
        let caps = BackendCapabilities {
            max_concurrency: 8,
            supports_shot_distribution: true,
        };
        assert!(ShotDistributingPlanner::new(4)
            .requirements()
            .check(&caps)
            .is_ok());
        assert!(SequentialPlanner.requirements().check(&caps).is_ok());
    }

    /// An empty batch must still honour the cancellation checkpoint (issue #162):
    /// `tasks.chunks(wave)` yields no iterations, so without the explicit early
    /// check `execute` would return `Ok(vec![])` without looking at the token.
    #[test]
    fn empty_batch_still_honours_cancellation() {
        let cancel = CancelToken::default();
        cancel.cancel();
        let err = SequentialPlanner
            .execute(&StubBackend, &[], &params(8), &cancel)
            .expect_err("a pre-cancelled empty batch must return Cancelled");
        assert!(matches!(err, InfrastructureError::Cancelled));
    }

    /// The between-wave interrupt guard fires: a token whose guard errors aborts
    /// `execute`, and the error re-surfaces as `Backend(External(..))` (the shape
    /// the FFI edge downcasts to re-raise a `KeyboardInterrupt` verbatim).
    #[test]
    fn interrupt_guard_aborts_between_waves() {
        struct BoomGuard;
        impl Interrupt for BoomGuard {
            fn poll(&self) -> Result<(), BackendError> {
                Err(BackendError::External("simulated Ctrl+C".into()))
            }
        }
        let cancel = CancelToken::with_interrupt(Arc::new(BoomGuard));
        let circuit = BoundCircuit::Qasm2(String::new());
        let tasks = one_task(&circuit, 8);
        let err = SequentialPlanner
            .execute(&StubBackend, &tasks, &params(8), &cancel)
            .expect_err("the interrupt guard must abort the run");
        assert!(matches!(
            err,
            InfrastructureError::Backend(BackendError::External(_))
        ));
    }

    /// A backend whose `capabilities_for` fails must have that error propagate out
    /// of `execute`, not be swallowed. `run_circuits` is never reached.
    #[test]
    fn execute_propagates_a_capabilities_for_error() {
        struct CapsFail;
        impl QuantumBackend for CapsFail {
            fn run_circuits(
                &self,
                _qcs: &[BoundCircuit],
                _params: &RunParams,
            ) -> Result<Vec<Counts>, BackendError> {
                panic!("run_circuits must not be reached when capabilities_for fails");
            }
            fn capabilities_for(
                &self,
                _tasks: &[CircuitTask<'_>],
            ) -> Result<BackendCapabilities, InfrastructureError> {
                Err(InfrastructureError::Backend(BackendError::External(
                    "width read interrupted".into(),
                )))
            }
        }
        let circuit = BoundCircuit::Qasm2(String::new());
        let tasks = one_task(&circuit, 8);
        let err = SequentialPlanner
            .execute(&CapsFail, &tasks, &params(8), &CancelToken::default())
            .unwrap_err();
        assert!(matches!(
            err,
            InfrastructureError::Backend(BackendError::External(_))
        ));
    }

    /// A minimal backend for the tests that only need `execute` to reach (or skip)
    /// a wave: it returns one shot-conserving counts map per circuit.
    struct StubBackend;
    impl QuantumBackend for StubBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Ok(qcs
                .iter()
                .map(|_| HashMap::from([("0".to_string(), u64::from(params.shots))]))
                .collect())
        }
    }
}
