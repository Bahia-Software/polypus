//! Execution planning — *how* a set of circuits is run on a backend.
//!
//! The oracle owns the *what* (bind candidates, reduce counts → fitness); the
//! [`Planner`] owns the *how*: split the work into waves capped by the backend's
//! concurrency, run each wave, honour Ctrl+C between waves, merge shot-split
//! results, and preserve input order. Two consumers share one trait:
//! `run_quantum_circuit` calls [`Planner::execute`] (wants counts) and the
//! training oracle calls [`Planner::evaluate`] (wants one `f64` per circuit).
//!
//! Two concrete planners reproduce today's two execution paths exactly:
//! [`SequentialPlanner`] (the former `AlgorithmSingleRun` + the oracle's chunk
//! loop) and [`ShotDistributingPlanner`] (the former `DistributeByShotsRun`).
//!
//! **GIL discipline (ENGINEERING §3):** a planner is invoked with the GIL
//! *released*; it re-acquires it only for the `check_signals` between waves. It
//! must never assume it holds the GIL. This module is the seed of
//! `polypus-infrastructure`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use polypus_observable::CostObservable;
use pyo3::Python;

use super::error::InfrastructureError;
use super::{validate_run_results, BackendError, BoundCircuit, ExecutionConfig, QuantumBackend};

/// Native measurement counts for one circuit: bitstring → count.
pub type Counts = HashMap<String, u64>;

/// One unit of work: a bound circuit and the shots it needs. `shots` is the
/// single source of truth — a planner reads it here, never from `config`.
pub struct CircuitTask<'a> {
    /// The bound (parameter-free) circuit to run.
    pub circuit: &'a BoundCircuit,
    /// Shots for this circuit (`u32`, matching [`ExecutionConfig::shots`]).
    pub shots: u32,
}

/// Cooperative cancellation shared between the caller and a planner. A wave is
/// atomic, so a set token stops the *next* wave from launching; it never
/// interrupts a wave mid-flight. Cheap to clone (an `Arc<AtomicBool>`).
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Request cancellation. Takes effect at the next wave boundary.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// What a backend can do, so a planner can size its waves. Absorbs the former
/// `max_batch_size`.
pub struct BackendCapabilities {
    /// Most circuits to run concurrently in one wave (native/local: a memory
    /// budget; CUNQA: `n_qpus`; QMIO: 1).
    pub max_concurrency: usize,
    /// Whether the backend can split one circuit's shots across replicas and
    /// merge them (contract C-3). Every backend implements `run_shots_distributed`
    /// (a default replicating path, or a reuse-the-evolution override), so this is
    /// true across the board.
    pub supports_shot_distribution: bool,
}

/// What a planner needs of its backend, contrasted with [`BackendCapabilities`]
/// when the two are paired (in `Resources::new`, a later phase).
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

    /// Run every task and return merged counts, one per task, in input order.
    /// Owns the wave loop, the concurrency cap, the between-wave `check_signals`
    /// (ENGINEERING §3) and cancellation. If shots are split, the C-3 merge
    /// happens here so the caller always sees one [`Counts`] per circuit.
    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        config: &ExecutionConfig,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError>;

    /// Run every task and reduce it to one `f64`. The default preserves today's
    /// numerics exactly (`execute`, then `expectation_batch` — the former
    /// `run_and_evaluate`). Reduce-at-source is a deferred override (§7.1).
    fn evaluate(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        reducer: &dyn CostObservable,
        config: &ExecutionConfig,
        cancel: &CancelToken,
    ) -> Result<Vec<f64>, InfrastructureError> {
        let counts = self.execute(backend, tasks, config, cancel)?;
        reducer
            .expectation_batch(&counts)
            .map_err(InfrastructureError::Observable)
    }
}

/// Runs the tasks in sequential, atomic waves of `≤ max_concurrency` circuits,
/// one `run_circuits` call per wave, merging in input order — the default
/// planner for every backend, reproducing today's `AlgorithmSingleRun` and the
/// oracle's per-chunk loop.
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
        config: &ExecutionConfig,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        // Size the wave with the *batch-aware* cap: native/local derive it from a
        // statevector memory budget scaled by this batch's widest circuit
        // (`capabilities_for`), so a high-qubit batch is split into several waves
        // instead of one. The between-wave `py.check_signals()` below (ENGINEERING
        // §3) therefore runs once per wave, keeping a big generation interruptible.
        // Backends with a static cap inherit the default, which delegates to the
        // batch-agnostic `capabilities()` — unchanged behaviour for them.
        let wave = backend.capabilities_for(tasks).max_concurrency.max(1);
        let mut out: Vec<Counts> = Vec::with_capacity(tasks.len());
        for chunk in tasks.chunks(wave) {
            if cancel.is_cancelled() {
                return Err(InfrastructureError::Cancelled);
            }
            // Tasks are uniform-shots in this phase (training and single runs), so
            // one config carries the whole wave. `shots` still comes from the
            // task, the single source of truth.
            let shots = chunk[0].shots;
            let circuits: Vec<BoundCircuit> = chunk.iter().map(|t| t.circuit.duplicate()).collect();
            let mut cfg = config.clone();
            cfg.shots = shots;
            let counts = backend
                .run_circuits(&circuits, &cfg)
                .map_err(InfrastructureError::Backend)?;
            validate_run_results(&counts, chunk.len(), shots)
                .map_err(InfrastructureError::Backend)?;
            // Honour a pending Ctrl+C after each wave's run (as the former
            // `run_and_evaluate` / `AlgorithmSingleRun` did after `run_circuits`).
            Python::with_gil(|py| py.check_signals()).map_err(InfrastructureError::Python)?;
            out.extend(counts);
        }
        Ok(out)
    }
}

/// Runs *one* circuit by splitting its shots across `n_qpus` replicas and merging
/// them (contract C-3) — reproducing `DistributeByShotsRun`. Opt-in; the
/// `polypus` edge selects it for `run_quantum_circuit` with `n_qpus > 1`.
pub struct ShotDistributingPlanner;

impl Planner for ShotDistributingPlanner {
    fn requirements(&self) -> PlannerRequirements {
        PlannerRequirements {
            needs_shot_distribution: true,
            min_concurrency: 1,
        }
    }

    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        config: &ExecutionConfig,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        // This planner distributes a single circuit's shots; reject any other
        // count with a typed error (as `DistributeByShotsRun` did) rather than
        // panicking or silently dropping circuits.
        if tasks.len() != 1 {
            return Err(InfrastructureError::Backend(
                BackendError::InvalidCircuitCount {
                    expected: 1,
                    got: tasks.len(),
                },
            ));
        }
        if cancel.is_cancelled() {
            return Err(InfrastructureError::Cancelled);
        }
        let task = &tasks[0];
        let shots = task.shots;
        let n_qpus = config.n_qpus;
        // Apportion the shots (base + one extra on the first `remainder` replicas),
        // conserving the total exactly (contract C-3); `n_qpus >= 1` and
        // `shots >= 1` are guaranteed by the Python boundary.
        let base = shots / n_qpus;
        let remainder = shots % n_qpus;
        let shot_batches: Vec<u32> = (0..n_qpus)
            .map(|i| if i < remainder { base + 1 } else { base })
            .collect();
        let counts_vec = backend
            .run_shots_distributed(task.circuit, &shot_batches, config)
            .map_err(InfrastructureError::Backend)?;
        // Merge the replicas into one result (the merge, C-3, lives in the planner).
        let mut merged: Counts = HashMap::new();
        for counts in counts_vec {
            for (k, v) in counts {
                *merged.entry(k).or_insert(0) += v;
            }
        }
        validate_run_results(std::slice::from_ref(&merged), 1, shots)
            .map_err(InfrastructureError::Backend)?;
        Python::with_gil(|py| py.check_signals()).map_err(InfrastructureError::Python)?;
        Ok(vec![merged])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_reject_shot_distribution_when_unsupported() {
        let caps = BackendCapabilities {
            max_concurrency: 8,
            supports_shot_distribution: false,
        };
        let err = ShotDistributingPlanner
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
        assert!(ShotDistributingPlanner.requirements().check(&caps).is_ok());
        // The sequential planner needs no distribution and any concurrency ≥ 1.
        assert!(SequentialPlanner.requirements().check(&caps).is_ok());
    }

    // --- Issue #147: batch-aware wave sizing for native/local -----------------
    //
    // These tests prove `SequentialPlanner::execute` now sizes its waves with the
    // *batch-aware* cap (`capabilities_for`), so a high-qubit batch on the native
    // or local backend is split into several memory-capped waves — with the
    // between-wave `py.check_signals()` running once per wave — instead of one
    // uninterruptible wave.
    //
    // The cap decision is taken by the *real* native/local backend
    // (`WaveSpy::capabilities_for` delegates to it), but the circuits are never
    // simulated: a `WaveSpy` intercepts `run_circuits`, records each wave's size
    // and returns dummy valid counts. That is what lets us use a 30-qubit circuit
    // (16 GiB statevector => cap 1 under the default budget, independent of the
    // thread count) without ever allocating 2^30 amplitudes or mutating the
    // `POLYPUS_MEM_BUDGET` env var (a known flakiness source this project avoids).
    use crate::{BackendCapabilities, NativeStatevectorBackend};
    use polypus_circuit::ParameterizedCircuit;
    use std::sync::Mutex;

    /// Wraps a real backend to borrow its batch-aware cap while stubbing out
    /// execution: `run_circuits` records the wave size and returns one valid
    /// (non-empty, shot-conserving) counts map per circuit, so nothing is ever
    /// simulated. Optionally cancels a shared token from inside the first wave.
    struct WaveSpy<'b> {
        inner: &'b dyn QuantumBackend,
        calls: Mutex<Vec<usize>>,
        cancel_after_first: Option<CancelToken>,
    }

    impl QuantumBackend for WaveSpy<'_> {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &ExecutionConfig,
        ) -> Result<Vec<Counts>, BackendError> {
            self.calls.lock().unwrap().push(qcs.len());
            if let Some(token) = &self.cancel_after_first {
                token.cancel();
            }
            Ok(qcs
                .iter()
                .map(|_| Counts::from([("0".to_string(), u64::from(config.shots))]))
                .collect())
        }

        // The whole point: waves are sized by the real backend's batch-aware cap.
        fn capabilities_for(&self, tasks: &[CircuitTask<'_>]) -> BackendCapabilities {
            self.inner.capabilities_for(tasks)
        }
    }

    fn config() -> ExecutionConfig {
        ExecutionConfig {
            id: "wave-test".to_string(),
            shots: 8,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: crate::BackendConfig::LocalNative { fusion: true },
            opt_level: crate::OptLevel::default(),
            seed: Some(7),
        }
    }

    /// Four 30-qubit tasks, referencing one zero-gate circuit (never simulated).
    fn wide_batch(circuit: &BoundCircuit) -> Vec<CircuitTask<'_>> {
        (0..4).map(|_| CircuitTask { circuit, shots: 8 }).collect()
    }

    /// Acceptance criterion 1 (native): a high-qubit batch is split into several
    /// waves respecting the memory-derived cap, not run as one wave. Cap 1 (30
    /// qubits under the 16 GiB default) ⇒ one circuit per wave ⇒ four waves.
    #[test]
    fn execute_splits_high_qubit_native_batch_into_memory_capped_waves() {
        pyo3::prepare_freethreaded_python();
        if rayon::current_num_threads() <= 1 {
            return; // Needs real parallelism for the budget to throttle below it.
        }
        let native = NativeStatevectorBackend::new(0);
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(30)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks = wide_batch(&wide);
        let spy = WaveSpy {
            inner: &native,
            calls: Mutex::new(Vec::new()),
            cancel_after_first: None,
        };

        let out = SequentialPlanner
            .execute(&spy, &tasks, &config(), &CancelToken::default())
            .unwrap();

        assert_eq!(out.len(), 4);
        assert_eq!(
            *spy.calls.lock().unwrap(),
            vec![1, 1, 1, 1],
            "cap 1 must split the batch into four single-circuit waves, not one wave of four"
        );
    }

    /// Acceptance criterion 1 (local): same split, cap taken from the real
    /// `LocalBackend` (whose `capabilities_for` reads widths through the GIL).
    #[test]
    fn execute_splits_high_qubit_local_batch_into_memory_capped_waves() {
        pyo3::prepare_freethreaded_python();
        if std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            <= 1
        {
            return; // Needs real parallelism for the budget to throttle below it.
        }
        let local =
            crate::LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(30)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks = wide_batch(&wide);
        let spy = WaveSpy {
            inner: &local,
            calls: Mutex::new(Vec::new()),
            cancel_after_first: None,
        };

        let out = SequentialPlanner
            .execute(&spy, &tasks, &config(), &CancelToken::default())
            .unwrap();

        assert_eq!(out.len(), 4);
        assert_eq!(
            *spy.calls.lock().unwrap(),
            vec![1, 1, 1, 1],
            "cap 1 must split the batch into four single-circuit waves, not one wave of four"
        );
    }

    /// Acceptance criterion 2 (native): the wave boundary — where
    /// `py.check_signals()` runs — is honoured *between* waves, not only once at
    /// the end. Cancelling from inside the first wave stops the second from ever
    /// launching: only one `run_circuits` call is recorded and `execute` returns
    /// `Cancelled`. Were the whole batch one wave, cancellation set mid-run would
    /// have no effect and all four circuits would run.
    #[test]
    fn execute_signal_checks_between_native_waves_not_only_at_the_end() {
        pyo3::prepare_freethreaded_python();
        if rayon::current_num_threads() <= 1 {
            return; // Needs real parallelism for the budget to throttle below it.
        }
        let native = NativeStatevectorBackend::new(0);
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(30)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks = wide_batch(&wide);
        let token = CancelToken::default();
        let spy = WaveSpy {
            inner: &native,
            calls: Mutex::new(Vec::new()),
            cancel_after_first: Some(token.clone()),
        };

        let err = SequentialPlanner
            .execute(&spy, &tasks, &config(), &token)
            .unwrap_err();

        assert!(matches!(err, InfrastructureError::Cancelled));
        assert_eq!(
            *spy.calls.lock().unwrap(),
            vec![1],
            "the second wave must not launch after cancellation at the first wave boundary"
        );
    }

    /// Acceptance criterion 2 (local): same between-wave boundary, with the cap
    /// taken from the real `LocalBackend`.
    #[test]
    fn execute_signal_checks_between_local_waves_not_only_at_the_end() {
        pyo3::prepare_freethreaded_python();
        if std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            <= 1
        {
            return; // Needs real parallelism for the budget to throttle below it.
        }
        let local =
            crate::LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(30)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks = wide_batch(&wide);
        let token = CancelToken::default();
        let spy = WaveSpy {
            inner: &local,
            calls: Mutex::new(Vec::new()),
            cancel_after_first: Some(token.clone()),
        };

        let err = SequentialPlanner
            .execute(&spy, &tasks, &config(), &token)
            .unwrap_err();

        assert!(matches!(err, InfrastructureError::Cancelled));
        assert_eq!(
            *spy.calls.lock().unwrap(),
            vec![1],
            "the second wave must not launch after cancellation at the first wave boundary"
        );
    }
}
