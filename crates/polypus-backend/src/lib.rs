//! # polypus-backend
//!
//! The **pyo3-free execution-backend contract** for Polypus. A third party can
//! implement a quantum backend against this crate alone — no PyO3, no Qiskit, no
//! Polypus runtime — and have it usable by every algorithm in Polypus.
//!
//! What lives here (and *only* here, so the dependency tree stays minimal):
//!
//! - [`QuantumBackend`] — the trait a backend implements.
//! - [`BoundCircuit`] / [`ForeignCircuit`] — the circuit representations a backend
//!   receives (native, OpenQASM 2.0, or an opaque provider object).
//! - [`RunParams`] — the per-call execution parameters a backend reads.
//! - [`BackendError`] / [`InfrastructureError`] — the error types, carrying any
//!   provider/Python failure type-erased in [`BackendError::External`].
//! - [`BackendCapabilities`], [`CircuitTask`], [`Counts`], the
//!   [`Transpiler`]/[`OptLevel`]/[`TranspileOptions`] seam, the statevector memory
//!   budget ([`max_statevector_concurrency`]), and the [`Planner`].
//!
//! What deliberately does **not** live here: the concrete Polypus backends
//! (native/Aer/CUNQA/QMIO), the `Infrastructure` factory and the Qiskit/ZeroMQ
//! machinery — those stay in `polypus-infrastructure`, which re-exports this
//! crate's API so existing intra-workspace import paths keep resolving.
//!
//! See `docs/backends.md` for the "how to write your own backend" guide and the
//! stability commitment.

use std::collections::HashMap;
use std::sync::Arc;

pub mod circuit;
pub mod error;
pub mod mem_budget;
pub mod params;
pub mod planner;
pub mod transpiler;

pub use circuit::{BoundCircuit, ForeignCircuit};
pub use error::{BackendError, InfrastructureError};
pub use mem_budget::max_statevector_concurrency;
pub use params::RunParams;
pub use planner::{
    BackendCapabilities, CancelToken, CircuitTask, Counts, Interrupt, Planner, PlannerRequirements,
    SequentialPlanner, ShotDistributingPlanner,
};
pub use transpiler::{IdentityTranspiler, OptLevel, TranspileOptions, Transpiler};

/// Wave size a memory-budgeted backend (native/local) should report from
/// [`capabilities_for`](QuantumBackend::capabilities_for) for a batch of
/// `batch_len` circuits whose widest is `widest_qubits`, running on `cores`
/// threads.
///
/// The rule: split the batch into planner-visible waves **only when the whole
/// batch cannot be held in the statevector memory budget at once** — i.e. the
/// backend would otherwise process it as several *sequential memory windows*
/// inside one uninterruptible `run_circuits` call. When the batch fits, report
/// [`usize::MAX`] so it stays a single wave (the backend still parallelises
/// internally up to its own cap), preserving the single-call contract that keeps
/// Aer efficient.
///
/// The "does it fit" test uses the **pure memory limit**
/// `max_statevector_concurrency(widest, usize::MAX)` — how many statevectors the
/// budget holds, with the thread count removed — **not** the thread-capped cap.
/// That is what makes the decision independent of `cores`: a thread-based gate
/// (`cap >= cores`) degenerates when `cores == 1`, where the cap always equals
/// `cores` regardless of qubit count and would wrongly report a single wave for a
/// memory-heavy batch. When a split *is* needed, the reported cap is the ordinary
/// thread-and-budget bound `max_statevector_concurrency(widest, cores)`.
pub fn wave_concurrency(widest_qubits: usize, cores: usize, batch_len: usize) -> usize {
    // Pure memory limit: how many `widest_qubits` statevectors the budget holds,
    // independent of the core count (pass `usize::MAX` as the thread bound).
    let budget_concurrency = max_statevector_concurrency(widest_qubits, usize::MAX);
    if budget_concurrency >= batch_len {
        // The whole batch fits under the budget at once — one wave.
        usize::MAX
    } else {
        // The batch cannot all be held at once: expose the real cap so those
        // windows become interruptible planner waves.
        max_statevector_concurrency(widest_qubits, cores)
    }
}

/// Contract for quantum circuit execution backends.
///
/// A backend is completely agnostic to the algorithm calling it; it only knows how
/// to execute a batch of bound (parameter-free) circuits and return counts.
/// Implementing this trait for a new provider (IBM, IQM, a subprocess bridge, …)
/// is sufficient to make it available to every algorithm in Polypus without
/// touching any algorithm code.
pub trait QuantumBackend: Send + Sync {
    /// Execute a slice of bound circuits.
    ///
    /// Returns native measurement counts — one map per circuit. Keeping the return
    /// type native (rather than a Python object) means non-Python backends never
    /// have to touch a GIL, which is essential for HPC-scale distribution.
    ///
    /// A failure is returned as a [`BackendError`] (never a panic): a provider or
    /// Python exception is carried type-erased in [`BackendError::External`] so the
    /// FFI edge can re-raise it verbatim, and a backend that stopped responding
    /// mid-call is [`BackendError::Unresponsive`].
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError>;

    /// Run a single circuit `qc` under a per-replica shot distribution, returning
    /// one counts map per entry of `shot_batches` (replica `i` runs
    /// `shot_batches[i]` shots). The caller has already apportioned the shots so
    /// the summed counts conserve the total exactly (contract C-3).
    ///
    /// The method exists so a backend that can *reuse* one circuit evolution across
    /// many shot batches can override it. This **default** reproduces the historical
    /// behaviour for backends that cannot: it replicates `qc` and forwards it to
    /// [`run_circuits`](Self::run_circuits), grouping consecutive replicas that
    /// request the same shot count into one uniform-shots batch (contract C-7). A
    /// zero-shot entry submits nothing and yields an empty map.
    fn run_shots_distributed(
        &self,
        qc: &BoundCircuit,
        shot_batches: &[u32],
        params: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        let mut out: Vec<HashMap<String, u64>> = Vec::with_capacity(shot_batches.len());
        let mut i = 0;
        while i < shot_batches.len() {
            let shots = shot_batches[i];
            let mut j = i + 1;
            while j < shot_batches.len() && shot_batches[j] == shots {
                j += 1;
            }
            if shots > 0 {
                let qcs: Vec<BoundCircuit> = (i..j).map(|_| qc.duplicate()).collect();
                let mut cfg = params.clone();
                cfg.shots = shots;
                out.extend(self.run_circuits(&qcs, &cfg)?);
            } else {
                out.extend((i..j).map(|_| HashMap::new()));
            }
            i = j;
        }
        Ok(out)
    }

    /// Release any held resources (SLURM jobs, cloud sessions, QPU reservations, …).
    fn close(&self) {}

    /// **Asynchronously abort an in-flight [`run_circuits`](Self::run_circuits)
    /// call**, invoked from *another thread* while a call is blocked.
    ///
    /// The cooperative [`CancelToken`] only takes effect *between* waves (a wave is
    /// atomic). Once a `run_circuits` call has started, nothing stops it until it
    /// returns — which is fine for the built-in backends, whose calls are either a
    /// fast CPU loop (native) or a blocking Python/network call that returns on its
    /// own. A backend that blocks for a long time on an external resource — the
    /// subprocess bridge (a worker parked in `recv()`), a cloud session — overrides
    /// this to signal that resource (e.g. send a signal to the child process) so
    /// the blocked call unblocks and returns [`BackendError::Unresponsive`] or a
    /// provider error. Polling a token from inside such a backend cannot work: the
    /// thread is blocked and never reaches a poll point; hence this out-of-band
    /// hook.
    ///
    /// The default is a no-op — correct for every backend whose calls are
    /// self-terminating. Wiring a planner watcher thread that calls this when a
    /// [`CancelToken`] flips mid-wave is deferred to the subprocess-bridge phase,
    /// which is the first backend that needs it; the hook is defined now so that
    /// addition is purely additive.
    fn cancel(&self) {}

    /// What this backend can do, so a [`Planner`] can size its execution waves.
    ///
    /// The default is unbounded concurrency with shot distribution supported. A
    /// backend overrides this to cap concurrency: CUNQA at `n_qpus`, QMIO at 1.
    ///
    /// This **batch-agnostic** form is the one paired against a planner's
    /// requirements up front (`Resources::new`) and the one mocks override, so its
    /// signature and behaviour are frozen. A backend whose real cap depends on the
    /// batch does **not** express it here — it overrides
    /// [`capabilities_for`](Self::capabilities_for) instead.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_concurrency: usize::MAX,
            supports_shot_distribution: true,
        }
    }

    /// Batch-aware capabilities: what this backend can do *for this specific batch
    /// of tasks*, so a [`Planner`] can size its execution waves against a cap that
    /// may depend on the batch itself.
    ///
    /// The default **ignores the batch and delegates to
    /// [`capabilities`](Self::capabilities)**, which is exactly correct for every
    /// backend whose cap is static, and for every mock. Only the native and local
    /// backends override it, capping concurrency by a statevector memory budget
    /// scaled by the batch's widest circuit — a cap unknowable without the batch.
    ///
    /// **Fallible on purpose.** Sizing a wave can require reading a circuit's width
    /// through a provider SDK (local reads a Qiskit `num_qubits`), and that read can
    /// be aborted by a host interrupt (a pending Ctrl+C). Returning a `Result` lets
    /// that interrupt propagate (as an [`InfrastructureError`]) instead of being
    /// swallowed into a silent "width unknown". The default and the GIL-free
    /// backends never error, so they simply return `Ok`.
    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        let _ = tasks;
        Ok(self.capabilities())
    }

    /// The sensible default planner for this backend: the atomic-wave
    /// [`SequentialPlanner`], used by every backend. The `polypus` edge opts into
    /// the [`ShotDistributingPlanner`] for single-circuit shot distribution.
    fn default_planner(&self) -> Arc<dyn Planner> {
        Arc::new(SequentialPlanner)
    }

    /// The number of parallel execution units (QPUs) this backend has allocated,
    /// when that is a fixed property of the backend — CUNQA returns its `n_qpus`.
    /// `None` means "not applicable / unbounded" (native, local, QMIO).
    ///
    /// This exists so a [`ShotDistributingPlanner`], which splits one circuit's
    /// shots across a caller-chosen replica count ([`Planner::required_replicas`]),
    /// can be **validated against the backend it is paired with** where the two are
    /// combined (`Resources::new` in `polypus-orchestration`): a planner that would
    /// split into a different number of replicas than the backend allocated is a
    /// configuration mistake (idle or over-subscribed QPUs) and is rejected up front
    /// rather than silently mis-sized. Defaulting to `None` keeps the check a no-op
    /// for every backend whose replica count is not a fixed integer, so the valid
    /// "QMIO with `n_qpus > 1` replicas run sequentially" case is unaffected.
    fn replica_count(&self) -> Option<u32> {
        None
    }
}

/// Centrally validate the measurement-count maps a backend returned for a batch,
/// before anything downstream consumes them.
///
/// Checks, in order: exactly one map per submitted circuit; every map non-empty;
/// every map's counts summing to `expected_shots` (contract C-3 shot
/// conservation); every key a non-empty bitstring (`0`/`1` only). Any violation is
/// a backend/contract bug, returned as [`BackendError::InvalidResults`].
///
/// `expected_shots` is the per-map shot count: `params.shots` for a normal batch,
/// or the total for a merged shot-distributed result.
pub fn validate_run_results(
    counts: &[HashMap<String, u64>],
    expected_circuits: usize,
    expected_shots: u32,
) -> Result<(), BackendError> {
    if counts.len() != expected_circuits {
        return Err(BackendError::InvalidResults(format!(
            "expected one counts map per circuit ({expected_circuits}), got {}",
            counts.len()
        )));
    }
    let expected_shots = u64::from(expected_shots);
    for (i, map) in counts.iter().enumerate() {
        if map.is_empty() {
            return Err(BackendError::InvalidResults(format!(
                "empty counts map for circuit {i} ({expected_shots} shot(s) requested); an empty \
                 result has no measurement outcomes and cannot be reduced to a fitness"
            )));
        }
        let total: u64 = map.values().copied().sum();
        if total != expected_shots {
            return Err(BackendError::InvalidResults(format!(
                "counts for circuit {i} sum to {total} shot(s) but {expected_shots} were requested \
                 (contract C-3 shot conservation)"
            )));
        }
        if let Some(bad) = map
            .keys()
            .find(|k| k.is_empty() || !k.bytes().all(|b| b == b'0' || b == b'1'))
        {
            return Err(BackendError::InvalidResults(format!(
                "counts for circuit {i} contain a non-bitstring key {bad:?} (expected a string of \
                 0/1 outcomes)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod result_validation_tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn accepts_a_well_formed_batch() {
        let batch = vec![counts(&[("00", 5), ("11", 5)]), counts(&[("01", 10)])];
        assert!(validate_run_results(&batch, 2, 10).is_ok());
    }

    #[test]
    fn rejects_wrong_result_count() {
        let batch = vec![counts(&[("0", 4)])];
        let err = validate_run_results(&batch, 2, 4).unwrap_err();
        assert!(matches!(err, BackendError::InvalidResults(_)));
        assert!(err.to_string().contains("one counts map per circuit"));
    }

    #[test]
    fn rejects_empty_map() {
        let err = validate_run_results(&[HashMap::new()], 1, 8).unwrap_err();
        assert!(err.to_string().contains("empty counts map"));
    }

    #[test]
    fn rejects_shot_non_conservation() {
        let batch = vec![counts(&[("0", 3), ("1", 4)])];
        let err = validate_run_results(&batch, 1, 8).unwrap_err();
        assert!(err.to_string().contains("C-3"));
    }

    #[test]
    fn rejects_non_bitstring_key() {
        let batch = vec![counts(&[("0x2", 8)])];
        let err = validate_run_results(&batch, 1, 8).unwrap_err();
        assert!(err.to_string().contains("non-bitstring"));
    }
}

#[cfg(test)]
mod wave_concurrency_tests {
    use super::*;

    // These pin the wave-sizing rule shared by native/local `capabilities_for`,
    // deterministically — the core count is an explicit parameter. They assume the
    // default 16 GiB budget (`POLYPUS_MEM_BUDGET` unset).

    #[test]
    fn single_core_high_qubit_batch_reports_a_finite_cap_not_unbounded() {
        assert_eq!(wave_concurrency(30, 1, 4), 1);
        assert_eq!(wave_concurrency(30, 32, 4), 1);
    }

    #[test]
    fn a_batch_that_fits_the_budget_stays_a_single_wave_on_any_core_count() {
        assert_eq!(wave_concurrency(2, 1, 200), usize::MAX);
        assert_eq!(wave_concurrency(2, 32, 200), usize::MAX);
        assert_eq!(wave_concurrency(30, 1, 1), usize::MAX);
        assert_eq!(wave_concurrency(30, 32, 1), usize::MAX);
    }

    #[test]
    fn split_reports_the_thread_and_budget_bound() {
        assert_eq!(wave_concurrency(30, 8, 4), 1);
        assert_eq!(wave_concurrency(28, 8, 10), 4);
        assert_eq!(wave_concurrency(28, 2, 10), 2);
    }
}
