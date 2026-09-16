use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub mod cunqa;
pub mod error;
pub mod execution_config;
pub mod local;
pub mod mem_budget;
pub mod native;
pub mod planner;
#[cfg(feature = "qmio")]
pub mod qmio;
pub mod transpiler;

pub use cunqa::CunqaBackend;
pub use error::{BackendError, InfrastructureError};
pub use execution_config::{BackendConfig, ExecutionConfig};
pub use local::LocalBackend;
pub use mem_budget::max_statevector_concurrency;
pub use native::NativeStatevectorBackend;
pub use planner::{
    BackendCapabilities, CancelToken, CircuitTask, Counts, Planner, PlannerRequirements,
    SequentialPlanner, ShotDistributingPlanner,
};
#[cfg(feature = "qmio")]
pub use qmio::QmioBackend;
pub use transpiler::{IdentityTranspiler, OptLevel, TranspileOptions, Transpiler};

/// Process-wide count of backend cleanup (`close`/`Drop`) failures.
///
/// A `Drop` must never panic, so a failed teardown is logged and counted here
/// rather than propagated. A per-instance flag would be useless — nobody holds
/// the instance once `Drop` returns — so the consultable state lives in this
/// process-wide counter, exposed to Python as `polypus.backend_cleanup_failures()`.
static CLEANUP_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Record that a backend's resource cleanup failed. Called from `close`/`Drop`.
pub fn record_cleanup_failure() {
    CLEANUP_FAILURES.fetch_add(1, Ordering::SeqCst);
}

/// Number of backend cleanup failures recorded so far this process (diagnostic).
pub fn cleanup_failure_count() -> u64 {
    CLEANUP_FAILURES.load(Ordering::SeqCst)
}

/// Supported quantum execution infrastructures.
#[derive(Debug)]
pub enum Infrastructure {
    Local,
    Cunqa,
    /// CESGA QMIO real QPU (see `QmioBackend`). The variant always exists so
    /// that selecting `"qmio"` produces a clear "requires `--features qmio`"
    /// error rather than an `Unknown infrastructure` panic when the feature is
    /// disabled; the backend itself is only built with the feature on.
    Qmio,
}

impl Infrastructure {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, BackendError> {
        match s {
            "local" => Ok(Infrastructure::Local),
            "cunqa" => Ok(Infrastructure::Cunqa),
            "qmio" => Ok(Infrastructure::Qmio),
            other => Err(BackendError::UnknownInfrastructure {
                name: other.to_string(),
            }),
        }
    }

    /// Instantiate the appropriate backend for the given execution config.
    ///
    /// Dispatch is driven solely by [`ExecutionConfig::backend_config`], so the
    /// chosen backend and its parameters can never desync. Adding a new backend
    /// (IBM, IQM, …) means adding one arm here and implementing
    /// [`QuantumBackend`] for the new type — no algorithm code changes.
    pub fn create_backend(
        config: &ExecutionConfig,
    ) -> Result<Arc<dyn QuantumBackend>, BackendError> {
        match &config.backend_config {
            BackendConfig::Local {
                backend,
                sim_method,
                noise_model,
            } => Ok(Arc::new(LocalBackend::new(
                backend.clone(),
                sim_method.clone(),
                noise_model
                    .as_ref()
                    .map(|nm| Python::with_gil(|py| nm.clone_ref(py))),
            ))),
            BackendConfig::Cunqa {
                backend,
                sim_method,
                nodes,
                cores_per_qpu,
            } => Ok(Arc::new(CunqaBackend::new(
                config.n_qpus,
                *nodes,
                &config.id,
                *cores_per_qpu,
                backend.clone(),
                sim_method.clone(),
            )?)),
            // The Python-facing layer resolves a concrete seed whenever the
            // native backend runs; the entropy fallback here only guards a
            // directly-built config that left `seed` unset (e.g. tests), so
            // an omitted seed still yields independent noise, never a panic.
            BackendConfig::LocalNative { fusion } => Ok(Arc::new(
                NativeStatevectorBackend::new(
                    config.seed.unwrap_or_else(execution_config::random_seed),
                )
                .with_fusion(*fusion),
            )),
            #[cfg(feature = "qmio")]
            BackendConfig::Qmio {
                endpoint,
                program_format,
                optimization,
                repetition_period,
                res_format,
            } => Ok(Arc::new(
                QmioBackend::new(
                    endpoint.clone(),
                    *program_format,
                    *optimization,
                    *repetition_period,
                    res_format.clone(),
                )
                .map_err(BackendError::Qmio)?,
            )),
        }
    }
}

/// A fully bound (parameter-free) circuit ready for execution, in one of the
/// representations Polypus supports.
///
/// Backends receive a slice of these and decide how to consume each variant:
/// Python-based backends convert `Qasm2` to a `QuantumCircuit` inside the
/// Python layer; future native or wire-protocol backends can submit the QASM
/// text directly without ever touching the GIL.
///
/// Being an enum (rather than `Py<PyAny>`) makes the contract explicit: adding
/// a new circuit representation is a compile-time-checked change in every
/// backend, not a runtime surprise.
#[derive(Debug)]
pub enum BoundCircuit {
    /// A bound Qiskit `QuantumCircuit` (Python object).
    Qiskit(Py<PyAny>),
    /// An OpenQASM 2.0 program produced by the native Rust circuit layer.
    Qasm2(String),
    /// A fully bound native circuit from `polypus-circuit`. Carries the circuit
    /// structure directly so the native statevector backend can simulate it
    /// without any OpenQASM round-trip or GIL; Python-based backends serialise
    /// it to OpenQASM 2.0 on demand in [`to_py_object`](Self::to_py_object).
    Native(polypus_circuit::ConcreteCircuit),
}

impl BoundCircuit {
    /// Convert to the Python object expected by `polypus_python.run_qcs`:
    /// the Qiskit circuit as-is, or the QASM program as a `str` (the Python
    /// layer parses/forwards it per infrastructure).
    pub fn to_py_object(&self, py: Python<'_>) -> Result<Py<PyAny>, BackendError> {
        let conv = |e: PyErr| BackendError::Conversion(e.to_string());
        match self {
            BoundCircuit::Qiskit(qc) => Ok(qc.clone_ref(py)),
            BoundCircuit::Qasm2(qasm) => Ok(qasm
                .into_pyobject(py)
                .map_err(|e| conv(e.into()))?
                .into_any()
                .unbind()),
            // Native circuits reach a Python backend (Aer/CUNQA) as OpenQASM 2.0,
            // exactly like the `Qasm2` variant; the conversion is pure Rust.
            BoundCircuit::Native(circuit) => Ok(circuit
                .to_qasm2()
                .into_pyobject(py)
                .map_err(|e| conv(e.into()))?
                .into_any()
                .unbind()),
        }
    }

    /// Cheap copy. Only the `Qiskit` variant needs the GIL (reference-count
    /// bump); the `Qasm2` and `Native` variants are plain Rust clones.
    pub fn duplicate(&self) -> BoundCircuit {
        match self {
            BoundCircuit::Qiskit(qc) => {
                Python::with_gil(|py| BoundCircuit::Qiskit(qc.clone_ref(py)))
            }
            BoundCircuit::Qasm2(qasm) => BoundCircuit::Qasm2(qasm.clone()),
            BoundCircuit::Native(circuit) => BoundCircuit::Native(circuit.clone()),
        }
    }

    /// Rewrite the *native domain* of this circuit with `transpiler`, returning a
    /// transpiled `BoundCircuit`. This is the composition point shared by the
    /// Python-backed backends ([`LocalBackend`], [`CunqaBackend`]): each applies
    /// its injected [`Transpiler`] to every circuit just before submission.
    ///
    /// The transpiler is intentionally confined to the GIL-free native domain:
    ///
    /// - [`Native`](Self::Native) is transpiled directly on its
    ///   [`ConcreteCircuit`](polypus_circuit::ConcreteCircuit).
    /// - [`Qasm2`](Self::Qasm2) is parsed back to a
    ///   [`ConcreteCircuit`](polypus_circuit::ConcreteCircuit),
    ///   transpiled, and re-emitted as OpenQASM 2.0. This is *best-effort*: if the
    ///   QASM cannot be parsed (an unsupported construct, a non-native program),
    ///   the original text is returned untouched rather than panicking, so a
    ///   backend that already accepted that QASM keeps working unchanged.
    /// - [`Qiskit`](Self::Qiskit) is returned as-is: Aer/CUNQA transpile Qiskit
    ///   circuits internally, and rewriting one here would require reading its
    ///   gates through the GIL, crossing the deliberate native/Python boundary.
    pub fn transpiled(&self, transpiler: &dyn Transpiler, opts: &TranspileOptions) -> BoundCircuit {
        // A guaranteed no-op transpiler changes nothing, so skip the parse →
        // transpile → re-emit round trip (Qasm2) and the trait-dispatch clone
        // (Native) entirely, keeping only the cheap representation-preserving
        // copy. `Qiskit` is already a passthrough via `duplicate`.
        if transpiler.is_identity() {
            return match self {
                BoundCircuit::Native(cc) => BoundCircuit::Native(cc.clone()),
                BoundCircuit::Qasm2(qasm) => BoundCircuit::Qasm2(qasm.clone()),
                BoundCircuit::Qiskit(_) => self.duplicate(),
            };
        }
        match self {
            BoundCircuit::Native(cc) => BoundCircuit::Native(transpiler.transpile(cc, opts)),
            BoundCircuit::Qasm2(qasm) => {
                match polypus_circuit::ParameterizedCircuit::from_qasm2(qasm)
                    .and_then(|pc| pc.assign_parameters(&[]))
                {
                    Ok(cc) => BoundCircuit::Qasm2(transpiler.transpile(&cc, opts).to_qasm2()),
                    // Unparseable QASM is left intact (best-effort, no new panic).
                    Err(_) => BoundCircuit::Qasm2(qasm.clone()),
                }
            }
            BoundCircuit::Qiskit(_) => self.duplicate(),
        }
    }
}

/// Wave size a memory-budgeted backend (native/local) should report from
/// [`capabilities_for`](QuantumBackend::capabilities_for) for a batch of
/// `batch_len` circuits whose widest is `widest_qubits`, running on `cores`
/// threads.
///
/// The rule: split the batch into planner-visible waves **only when the whole
/// batch cannot be held in the statevector memory budget at once** — i.e. the
/// backend would otherwise process it as several *sequential memory windows*
/// inside one uninterruptible `run_circuits` call, which is exactly what issue
/// #147 fixes by letting the `Planner` run a `py.check_signals()` between waves.
/// When the batch fits, report [`usize::MAX`] so it stays a single wave (the
/// backend still parallelises internally up to its own cap), preserving the
/// single-call contract that keeps Aer efficient
/// (`tests/python/test_qml_concurrency.py`).
///
/// The "does it fit" test uses the **pure memory limit**
/// `max_statevector_concurrency(widest, usize::MAX)` — how many statevectors the
/// budget holds, with the thread count removed — **not** the thread-capped cap.
/// That is what makes the decision independent of `cores`: a thread-based gate
/// (`cap >= cores`) degenerates when `cores == 1`, where the cap always equals
/// `cores` regardless of qubit count and would wrongly report a single wave for a
/// memory-heavy batch (issue #147, the single-core case). When a split *is*
/// needed, the reported cap is the ordinary thread-and-budget bound
/// `max_statevector_concurrency(widest, cores)`, the concurrency the backend will
/// actually use.
pub(crate) fn wave_concurrency(widest_qubits: usize, cores: usize, batch_len: usize) -> usize {
    // Pure memory limit: how many `widest_qubits` statevectors the budget holds,
    // independent of the core count (pass `usize::MAX` as the thread bound).
    let budget_concurrency = max_statevector_concurrency(widest_qubits, usize::MAX);
    if budget_concurrency >= batch_len {
        // The whole batch fits under the budget at once — one wave; the backend
        // parallelises it internally up to its own cap.
        usize::MAX
    } else {
        // The batch cannot all be held at once, so it would be processed in
        // several memory windows: expose the real cap so those windows become
        // interruptible planner waves.
        max_statevector_concurrency(widest_qubits, cores)
    }
}

/// Contract for quantum circuit execution backends.
///
/// A backend is completely agnostic to the algorithm calling it; it only knows
/// how to execute a batch of bound (parameter-free) circuits and return counts.
///
/// Implementing this trait for a new provider (IBM, IQM, CUNQA, …) is sufficient
/// to make it available to every algorithm in Polypus without touching any
/// algorithm code.
pub trait QuantumBackend: Send + Sync {
    /// Execute a slice of bound circuits.
    ///
    /// Returns native measurement counts — one `HashMap<bitstring, count>` per
    /// circuit. Keeping the return type native (rather than a Python object)
    /// means non-Python backends (IBM, IQM, a future native MPI scheduler) never
    /// have to touch the GIL, which is essential for HPC-scale distribution.
    ///
    /// A failure is returned as a [`BackendError`] (never a panic): a Python
    /// exception from the `polypus_python` seam is carried verbatim in
    /// [`BackendError::Seam`] so it re-raises with its original type, and
    /// Rust-originated failures map to the typed `polypus.*` exception hierarchy
    /// at the FFI boundary (in `polypus::exceptions`, the crate's edge).
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError>;

    /// Run a single circuit `qc` under a per-replica shot distribution,
    /// returning one counts map per entry of `shot_batches` (replica `i` runs
    /// `shot_batches[i]` shots). The caller has already apportioned the shots —
    /// e.g. the `polypus` edge's shot-distribution algorithm splitting a total
    /// across `n_qpus`, one extra shot on the first `shots % n_qpus` replicas — so
    /// the summed counts conserve the total exactly (contract C-3).
    ///
    /// The method exists so a backend that can *reuse* one circuit evolution
    /// across many shot batches can override it and avoid re-simulating the
    /// identical circuit once per replica (see [`NativeStatevectorBackend`]).
    /// This **default** reproduces the historical behaviour for backends that
    /// cannot: it replicates `qc` and forwards it to
    /// [`run_circuits`](Self::run_circuits), grouping consecutive replicas that
    /// request the same shot count into one uniform-shots batch — Aer/CUNQA
    /// submit a whole batch per call and re-seed per call, so this grouping
    /// leaves their per-run results unchanged (contract C-7). A zero-shot entry
    /// submits nothing and yields an empty map, so no zero-shot circuit is ever
    /// sent and the total is still conserved.
    fn run_shots_distributed(
        &self,
        qc: &BoundCircuit,
        shot_batches: &[u32],
        config: &ExecutionConfig,
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
                let mut cfg = config.clone();
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

    /// What this backend can do, so a [`Planner`] can size its execution waves.
    ///
    /// The default is unbounded concurrency (the whole batch in one wave, matching
    /// the previous default `max_batch_size`) with shot distribution supported. A
    /// backend overrides this to cap concurrency: CUNQA at `n_qpus`, QMIO at 1.
    ///
    /// This **batch-agnostic** form is the one paired against a planner's
    /// requirements up front (`Resources::new`) and the one mocks override, so its
    /// signature and behaviour are frozen. A backend whose real cap depends on the
    /// batch does **not** express it here — it overrides
    /// [`capabilities_for`](Self::capabilities_for) instead (see there for why).
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
    /// **Why this exists alongside [`capabilities`](Self::capabilities).** The
    /// batch-agnostic `capabilities()` is called *without* a batch — by
    /// `Resources::new` to validate the planner/backend pairing before any run, and
    /// by test mocks that override it — so its signature must not change. But
    /// [`NativeStatevectorBackend`] and [`LocalBackend`] cap their real
    /// concurrency by a *statevector memory budget scaled by the batch's widest
    /// circuit* (see `mem_budget`): a cap that is unknowable without the batch.
    /// Before this method existed they inherited the unbounded default, so
    /// [`SequentialPlanner::execute`] treated their entire batch as one wave and
    /// ran `py.check_signals()` only once, at the end — an ininterruptible
    /// training generation (ENGINEERING §3). This additive method lets those
    /// backends expose the batch-derived cap the wave loop needs, without
    /// perturbing the frozen `capabilities()` seam.
    ///
    /// The default **ignores the batch and delegates to
    /// [`capabilities`](Self::capabilities)**, which is exactly correct for every
    /// backend whose cap is static — [`CunqaBackend`] (`n_qpus`), `QmioBackend`
    /// (1, behind the `qmio` feature) — and for every mock. Only
    /// [`NativeStatevectorBackend`] and [`LocalBackend`] override it, reusing the
    /// very cap arithmetic their `run_circuits` already applies internally, so the
    /// wave size the planner picks matches the memory bound the backend would
    /// enforce anyway.
    ///
    /// **Fallible on purpose.** Sizing a wave can require reading a circuit's width
    /// through the GIL ([`LocalBackend`] reads a Qiskit `num_qubits`), and that
    /// `getattr` runs Python bytecode that CPython may abort with a
    /// `KeyboardInterrupt` for a pending Ctrl+C. Returning a `Result` lets that
    /// interrupt propagate verbatim (as [`InfrastructureError::Python`]) instead of
    /// being swallowed into a silent "width unknown" — which would clear the
    /// pending signal so the planner's own `check_signals` never fired, leaving a
    /// run unresponsive to Ctrl+C. The default and the GIL-free backends never
    /// error, so they simply return `Ok`.
    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        let _ = tasks;
        Ok(self.capabilities())
    }

    /// The sensible default planner for this backend: the atomic-wave
    /// [`SequentialPlanner`], used by every backend. The `polypus` edge opts into
    /// the [`ShotDistributingPlanner`] for `run_quantum_circuit` shot distribution.
    fn default_planner(&self) -> Arc<dyn Planner> {
        Arc::new(SequentialPlanner)
    }
}

/// Centrally validate the measurement-count maps a backend returned for a batch,
/// before anything downstream consumes them.
///
/// Checks, in order: exactly one map per submitted circuit; every map non-empty;
/// every map's counts summing to `expected_shots` (contract C-3 shot
/// conservation); every key a non-empty bitstring (`0`/`1` only). Any violation
/// is a backend/contract bug, returned as [`BackendError::InvalidResults`] so it
/// surfaces as a typed diagnostic — rather than an empty or short result being
/// silently reduced to a `0.0` fitness or indexing out of bounds in the
/// optimizer.
///
/// `expected_shots` is the per-map shot count: `config.shots` for a normal batch
/// ([`run_circuits`](QuantumBackend::run_circuits)), or the total for a merged
/// shot-distributed result. This is the fase-1 result-frontier half of C-3; the
/// per-wave merge check moves into the `Planner` in a later phase.
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
        // 3 + 4 = 7 shots, but 8 were requested.
        let batch = vec![counts(&[("0", 3), ("1", 4)])];
        let err = validate_run_results(&batch, 1, 8).unwrap_err();
        assert!(err.to_string().contains("C-3"));
    }

    #[test]
    fn rejects_non_bitstring_key() {
        // Valid count and shot total, but "0x2" is not a bitstring.
        let batch = vec![counts(&[("0x2", 8)])];
        let err = validate_run_results(&batch, 1, 8).unwrap_err();
        assert!(err.to_string().contains("non-bitstring"));
    }
}

#[cfg(test)]
mod wave_concurrency_tests {
    use super::*;

    // These pin the wave-sizing rule shared by native/local `capabilities_for`,
    // deterministically — the core count is an explicit parameter, so no case
    // depends on the host having (or not having) more than one CPU. They assume
    // the default 16 GiB budget (`POLYPUS_MEM_BUDGET` unset), like the rest of the
    // suite: a 30-qubit statevector is exactly 16 GiB (budget holds one), a
    // 2-qubit one is 64 bytes (budget holds hundreds of millions).

    /// Regression for issue #147's single-core case. On a 1-core host the
    /// thread-capped cap always equals the core count, so the old `cap >= cores`
    /// gate reported `usize::MAX` (one uninterruptible wave) for *any* qubit count.
    /// A 4-circuit batch of 30-qubit circuits cannot fit in the budget (holds 1),
    /// so even with `cores == 1` the real cap (1) must be reported — not unbounded.
    #[test]
    fn single_core_high_qubit_batch_reports_a_finite_cap_not_unbounded() {
        assert_eq!(wave_concurrency(30, 1, 4), 1);
        // Same on a many-core host: still throttled to the memory cap.
        assert_eq!(wave_concurrency(30, 32, 4), 1);
    }

    /// A batch that fits in the budget stays a single wave (`usize::MAX`) whatever
    /// the core count — including one core, so a low-qubit population is never
    /// exploded into per-circuit backend calls
    /// (`tests/python/test_qml_concurrency.py`).
    #[test]
    fn a_batch_that_fits_the_budget_stays_a_single_wave_on_any_core_count() {
        assert_eq!(wave_concurrency(2, 1, 200), usize::MAX);
        assert_eq!(wave_concurrency(2, 32, 200), usize::MAX);
        // A single high-qubit circuit fits (the budget holds exactly one), so it
        // is one wave regardless — nothing to split.
        assert_eq!(wave_concurrency(30, 1, 1), usize::MAX);
        assert_eq!(wave_concurrency(30, 32, 1), usize::MAX);
    }

    /// When the batch does not fit, the reported cap is the ordinary
    /// thread-and-budget bound (the concurrency the backend will actually use):
    /// bounded by the budget at high qubits and by the cores when the budget is
    /// slack but the batch is larger than what fits.
    #[test]
    fn split_reports_the_thread_and_budget_bound() {
        // 30 qubits: budget holds 1, so the cap is 1 whatever the core count.
        assert_eq!(wave_concurrency(30, 8, 4), 1);
        // 28 qubits: budget holds 4; with 8 cores the cap is min(8, 4) = 4, and
        // the 10-circuit batch (> 4) does not fit, so it splits at 4.
        assert_eq!(wave_concurrency(28, 8, 10), 4);
        // 28 qubits with only 2 cores: the cap is min(2, 4) = 2.
        assert_eq!(wave_concurrency(28, 2, 10), 2);
    }
}
