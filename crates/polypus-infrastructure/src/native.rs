//! Native statevector backend: runs circuits in pure Rust via `polypus-sim`,
//! without ever touching the Python interpreter or Qiskit.
//!
//! This is the local counterpart to [`LocalBackend`](crate::LocalBackend)
//! (Qiskit Aer). It is selected with `backend="polypus"` and consumes a
//! [`BoundCircuit::Native`] directly — no OpenQASM round-trip, no GIL — which is
//! what makes the native circuit path pay off end-to-end. An OpenQASM 2.0
//! string is also accepted (parsed in Rust); a Qiskit `QuantumCircuit` is not,
//! since reading its gates would require the interpreter.

use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{
    max_statevector_concurrency, BackendCapabilities, BoundCircuit, CircuitTask,
    InfrastructureError, MemBudget, QuantumBackend, RunParams,
};
use polypus_backend::mem_budget::{active_budget, check_fits};
use polypus_circuit::{ConcreteCircuit, GateInstruction, ParameterizedCircuit};
use polypus_sim::{sample_projected, SimError, Simulator, StatevectorSimulator, MAX_CLBITS};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Seed for the circuit at global position `index` of a backend seeded with
/// `base_seed` ([`polypus_backend::stream_seed`], contract C-7).
fn circuit_seed(base_seed: u64, index: u64) -> u64 {
    polypus_backend::stream_seed(base_seed, index)
}

/// Local, noiseless statevector backend backed by `polypus-sim`.
///
/// Sampling is seeded from an explicit base seed and a per-circuit counter,
/// mixed with SplitMix64 (circuit `k` takes output `k` of a SplitMix64 stream
/// started at the mixed base seed), so two backends built with the same seed
/// reproduce the same counts while distinct circuits in a batch, and backends
/// with neighbouring seeds, get independent shot noise. The seed is
/// supplied by the caller (`ExecutionConfig::seed`, resolved at the
/// Python-facing boundary from a user value or an OS-entropy draw) and is
/// **fully decoupled from the run `id`**: an omitted seed yields genuine,
/// independent noise across runs instead of the `id`-derived repetition that was
/// previously mistaken for shot noise.
///
/// The backend *composes* a [`Transpiler`] (the rewriting *strategy*) and runs
/// it on every native circuit before simulating, passing the per-run
/// [`TranspileOptions`] (the *tuning*) derived from the [`RunParams`].
/// It defaults to the no-op [`IdentityTranspiler`]; inject another strategy with
/// [`with_transpiler`](Self::with_transpiler).
pub struct NativeStatevectorBackend {
    simulator: StatevectorSimulator,
    transpiler: Box<dyn Transpiler>,
    base_seed: u64,
    counter: AtomicU64,
}

impl NativeStatevectorBackend {
    /// Create a backend whose sampling stream starts from `seed`, using the
    /// no-op [`IdentityTranspiler`] (behavior identical to having no transpiler).
    pub fn new(seed: u64) -> Self {
        Self::with_transpiler(seed, Box::new(IdentityTranspiler))
    }

    /// Create a backend with a custom transpilation *strategy* injected by
    /// composition. The sampling stream starts from `seed`.
    ///
    /// This is the extension point for hardware-aware rewriting: pass any
    /// `Box<dyn Transpiler>` without changing the backend or any algorithm code.
    pub fn with_transpiler(seed: u64, transpiler: Box<dyn Transpiler>) -> Self {
        NativeStatevectorBackend {
            simulator: StatevectorSimulator::new(),
            transpiler,
            base_seed: seed,
            counter: AtomicU64::new(0),
        }
    }

    /// Check, without simulating anything, that this backend can execute
    /// `circuit`: the step [`run_circuits`](QuantumBackend::run_circuits) takes
    /// first, with the error it would fail with — an OpenQASM 2.0 program the
    /// importer rejects (`reset`, `if`, a gate after a measurement, a syntax
    /// error; contract C-4) is [`BackendError::NativeCircuit`], a Qiskit circuit
    /// is [`BackendError::UnsupportedCircuit`]. GIL-free and side-effect free:
    /// unlike execution, it writes no log record for a rejected circuit.
    ///
    /// Structural only: the qubit ceiling and the memory budget, which depend on
    /// the host, are not checked; [`check_resources`](Self::check_resources)
    /// does.
    pub fn check_circuit(circuit: &BoundCircuit) -> Result<(), BackendError> {
        native_source(circuit).map(|_| ())
    }

    /// Check, without allocating anything, that this backend can hold
    /// `circuit`'s statevector, with the error a run fails with: a circuit wider
    /// than the qubit ceiling ([`polypus_sim::MAX_QUBITS`], every backend's) is
    /// [`BackendError::NativeCircuit`], one whose statevector does not fit in
    /// the memory budget (issue #215, read from `POLYPUS_MEM_BUDGET` or the host
    /// at each call) is [`BackendError::InsufficientMemory`]. A circuit
    /// [`check_circuit`](Self::check_circuit) rejects fails with that error
    /// instead. GIL-free, and silent like `check_circuit`.
    pub fn check_resources(circuit: &BoundCircuit) -> Result<(), BackendError> {
        let num_qubits = native_source(circuit)?.circuit.num_qubits();
        let max = polypus_sim::MAX_QUBITS;
        if num_qubits > max {
            return Err(simulation_failed(&SimError::TooManyQubits {
                requested: num_qubits,
                max,
            }));
        }
        check_fits(num_qubits, active_budget()).map_err(BackendError::from)
    }

    /// Override the injected [`StatevectorSimulator::fusion`] flag (`true` by
    /// default, matching [`StatevectorSimulator::default`]). Backs
    /// [`BackendConfig::LocalNative`](crate::BackendConfig::LocalNative)'s
    /// `fusion` field, which lets a Python caller opt out of gate fusion for a
    /// strictly gate-by-gate simulation of exactly the circuit as written.
    pub fn with_fusion(mut self, fusion: bool) -> Self {
        self.simulator.fusion = fusion;
        self
    }

    /// Derive the executable [`ConcreteCircuit`] from a [`BoundCircuit`] and
    /// apply the injected transpiler — all GIL-free. Shared by
    /// [`simulate_one`](Self::simulate_one) and
    /// [`run_shots_distributed`](Self::run_shots_distributed) so both paths
    /// handle the `Native`/`Qasm2`/`Foreign` variants and transpile identically.
    ///
    /// Returns a borrow of `circuit` whenever possible instead of an owned
    /// clone: the `Native` variant borrows directly, and when the transpiler
    /// is a guaranteed no-op ([`Transpiler::is_identity`]) that borrow is
    /// returned as-is, skipping the trait-dispatch clone entirely (the default
    /// hot path). Only the `Qasm2` variant (which must be parsed into a local)
    /// and a non-identity transpiler (which must produce a new circuit) force
    /// an owned result. The declared classical width travels alongside, so both
    /// paths key their counts identically (contract C-3).
    fn concrete_circuit<'a>(
        &self,
        circuit: &'a BoundCircuit,
        opts: &TranspileOptions,
    ) -> Result<NativeSource<'a>, BackendError> {
        // The pre-check (`check_circuit`) shares `native_source` but must stay
        // silent: a rejection is only an error when it stops a run, so it is
        // logged here, on the execution path.
        let source = native_source(circuit).inspect_err(|e| {
            if let BackendError::NativeCircuit(message) = e {
                log::error!("{message}");
            }
        })?;

        // Transpile the native circuit (GIL-free) before simulating. When the
        // transpiler is a guaranteed no-op, keep the borrow and skip the
        // trait-dispatch clone entirely.
        if self.transpiler.is_identity() {
            Ok(source)
        } else {
            Ok(NativeSource {
                circuit: Cow::Owned(self.transpiler.transpile(&source.circuit, opts)),
                declared_clbits: source.declared_clbits,
            })
        }
    }

    /// Run one bound circuit and return Aer-compatible bitstring counts.
    ///
    /// The bitstring width and bit order match Qiskit's: little-endian qubit
    /// indexing with the highest classical bit on the left, so the counts are
    /// interchangeable with those from the Aer backend.
    fn simulate_one(
        &self,
        circuit: &BoundCircuit,
        shots: u32,
        seed: u64,
        opts: &TranspileOptions,
        parallel_threshold: usize,
    ) -> Result<HashMap<String, u64>, BackendError> {
        let source = self.concrete_circuit(circuit, opts)?;
        // Simulate with the gate-parallel threshold the caller picked: `run_circuits` raises it to
        // `usize::MAX` (gate kernels off) for a population batch whose across-circuit `par_iter`
        // already owns the cores, and keeps the default for a lone circuit. Gate-parallel is
        // numerically identical to sequential, so this trades only speed, never results.
        let simulator = StatevectorSimulator {
            max_qubits: self.simulator.max_qubits,
            parallel_threshold,
            fusion: self.simulator.fusion,
        };
        let raw = simulator
            .run_and_sample(&source.circuit, shots as usize, seed)
            .map_err(|e| {
                log::error!("native statevector simulation failed: {e}");
                simulation_failed(&e)
            })?;
        Ok(format_counts(&source.circuit, source.declared_clbits, raw))
    }

    /// Refuse, before anything is allocated, a circuit whose statevector cannot
    /// fit in the memory budget (issue #215): a run that would otherwise be killed
    /// by the OOM-killer fails with [`BackendError::InsufficientMemory`] instead.
    ///
    /// A width above the simulator's qubit ceiling is left to the simulator, so
    /// such a circuit keeps failing with the ceiling error it always did.
    fn check_memory(&self, num_qubits: usize) -> Result<(), BackendError> {
        self.check_memory_against(num_qubits, active_budget())
    }

    /// [`check_memory`](Self::check_memory) against an explicit `budget`, so the
    /// decision is testable without touching the process environment.
    fn check_memory_against(
        &self,
        num_qubits: usize,
        budget: MemBudget,
    ) -> Result<(), BackendError> {
        if num_qubits > self.simulator.max_qubits {
            return Ok(());
        }
        check_fits(num_qubits, budget).map_err(|e| {
            log::error!("native backend refused a circuit: {e}");
            BackendError::from(e)
        })
    }

    /// Run `qcs` while holding at most `cap` statevectors in memory at once.
    ///
    /// This is a pure resource bound: the per-circuit seed index is reserved as
    /// one contiguous block up front (`circuit_seed(base_seed, block_start + i)`), so a
    /// circuit's seed — and therefore its counts — is independent of how the batch
    /// is chunked, and gate-vs-circuit parallelism is numerically identical. The
    /// result is byte-for-byte the historical single unbounded `par_iter` for
    /// every `cap`; only peak memory and speed change.
    fn run_batch_with_cap(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
        cap: usize,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        // Tuning travels as an argument; the strategy is the injected field.
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        // Reserve a contiguous block of seeds for the whole batch, so each circuit
        // is sampled independently and deterministically regardless of order OR of
        // which window runs it.
        let start = self.counter.fetch_add(qcs.len() as u64, Ordering::Relaxed);
        let threads = rayon::current_num_threads();
        let cap = cap.max(1);

        // Fast path: when the budget permits the full pool-wide concurrency
        // (`cap >= threads`, i.e. `threads` statevectors fit) or the batch is a
        // single window, evaluate the whole batch in one `par_iter` — byte-for-byte
        // the previous behaviour (rayon already bounds in-flight tasks to the pool).
        if cap >= threads || cap >= qcs.len() {
            let pt = self.gate_parallel_threshold(qcs.len(), threads);
            return self.simulate_window(qcs, config.shots, &opts, start, pt);
        }

        // Budget-throttled path: process the batch in windows of `cap` so at most
        // `cap` statevectors are alive at once. Windows run in input order and each
        // window's seed offset keeps the global seed block contiguous; an `Err`
        // short-circuits at the first failing window (the lowest-index failure, as
        // one `par_iter` over the whole batch would also surface first).
        let mut out = Vec::with_capacity(qcs.len());
        let mut offset = 0usize;
        while offset < qcs.len() {
            let end = (offset + cap).min(qcs.len());
            let pt = self.gate_parallel_threshold(end - offset, threads);
            let block = self.simulate_window(
                &qcs[offset..end],
                config.shots,
                &opts,
                start + offset as u64,
                pt,
            )?;
            out.extend(block);
            offset = end;
        }
        Ok(out)
    }

    /// Simulate one window in parallel ACROSS circuits, seeding circuit `i` of the
    /// window with `circuit_seed(base_seed, global_start + i)`. A DE generation submits its
    /// whole population this way — the embarrassingly-parallel population axis.
    /// `simulate_one` is pure Rust (no GIL) and `&self`-only; rayon's indexed
    /// `collect` preserves order and short-circuits on the first `Err`.
    fn simulate_window(
        &self,
        window: &[BoundCircuit],
        shots: u32,
        opts: &TranspileOptions,
        global_start: u64,
        parallel_threshold: usize,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        window
            .par_iter()
            .enumerate()
            .map(|(i, qc)| {
                let seed = circuit_seed(self.base_seed, global_start.wrapping_add(i as u64));
                self.simulate_one(qc, shots, seed, opts, parallel_threshold)
            })
            .collect()
    }

    /// Gate-vs-circuit auto-parallel threshold (native.rs' historical rule): gate
    /// kernels compete with the across-circuit `par_iter` on the same rayon pool,
    /// so once a window has at least as many circuits as pool threads, evolve each
    /// circuit's gates SEQUENTIALLY (`usize::MAX`) to avoid nested-rayon
    /// contention; a smaller window keeps the default threshold and lets rayon's
    /// nesting fill idle cores. Numerically identical either way — speed only.
    fn gate_parallel_threshold(&self, window_len: usize, threads: usize) -> usize {
        if window_len >= threads {
            usize::MAX
        } else {
            self.simulator.parallel_threshold
        }
    }

    /// Run a batch of bound circuits and return the **exact** basis-state
    /// probabilities of each (`|amplitude|²`), one map per circuit in submission
    /// order, keyed like [`run_circuits`](QuantumBackend::run_circuits)'s counts
    /// (contract C-3).
    ///
    /// The exact counterpart of `run_circuits`, deliberately an **inherent**
    /// method and not part of [`QuantumBackend`]: "exact" has no meaning for a
    /// noisy simulator or real hardware, only for this noiseless statevector
    /// backend. It is what the native QML exact mode runs.
    ///
    /// - **No sampling.** `config.shots` is ignored and no seed is drawn: the
    ///   result is a pure function of the circuits, bit for bit, and the
    ///   backend's sampling stream is left untouched. States of probability
    ///   exactly zero are omitted, as an unobserved bitstring is from counts;
    ///   the values sum to 1 up to rounding.
    /// - **Full-register read-out only.** A circuit with no measurement is read
    ///   on every qubit, and one that ends in `measure_all` on every classical
    ///   bit, at the width `run_circuits` keys its counts. A circuit that
    ///   measures individual qubits is [`BackendError::UnsupportedCircuit`]:
    ///   its distribution would have to be marginalised over the unmeasured
    ///   qubits.
    /// - **Bounded memory.** The same statevector budget as `run_circuits`: a
    ///   widest circuit that cannot fit even alone is refused
    ///   ([`BackendError::InsufficientMemory`]) before anything runs, and the
    ///   batch runs in windows of at most as many circuits as fit at once. Each
    ///   result map holds up to `2^n` entries, so the output, unlike counts,
    ///   grows with the width of the circuits rather than with the shots.
    pub fn run_circuits_exact(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, f64>>, BackendError> {
        let widest = representative_qubits(qcs);
        self.check_memory(widest)?;
        let cap = max_statevector_concurrency(widest, rayon::current_num_threads());
        self.run_exact_with_cap(qcs, config, cap)
    }

    /// [`run_circuits_exact`](Self::run_circuits_exact) holding at most `cap`
    /// statevectors at once: the windowing of
    /// [`run_batch_with_cap`](Self::run_batch_with_cap) without the seeds. A
    /// pure resource bound, so the result is the same for every `cap`.
    fn run_exact_with_cap(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
        cap: usize,
    ) -> Result<Vec<HashMap<String, f64>>, BackendError> {
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        let threads = rayon::current_num_threads();
        let cap = cap.max(1);
        // Same fast path as the sampled batch: when every thread's statevector
        // fits, rayon already bounds the circuits in flight.
        let window = if cap >= threads {
            qcs.len().max(1)
        } else {
            cap
        };
        let mut out = Vec::with_capacity(qcs.len());
        for chunk in qcs.chunks(window) {
            let pt = self.gate_parallel_threshold(chunk.len(), threads);
            let block: Vec<HashMap<String, f64>> = chunk
                .par_iter()
                .map(|qc| self.simulate_one_exact(qc, &opts, pt))
                .collect::<Result<_, _>>()?;
            out.extend(block);
        }
        Ok(out)
    }

    /// Exact probabilities of one bound circuit, keyed as its counts would be.
    fn simulate_one_exact(
        &self,
        circuit: &BoundCircuit,
        opts: &TranspileOptions,
        parallel_threshold: usize,
    ) -> Result<HashMap<String, f64>, BackendError> {
        let source = self.concrete_circuit(circuit, opts)?;
        let concrete = source.circuit.as_ref();
        if concrete
            .gates()
            .iter()
            .any(|gate| matches!(gate, GateInstruction::Measure { .. }))
        {
            return Err(BackendError::UnsupportedCircuit(
                "exact probabilities need a full-register read-out (measure_all, or no \
                 measurement); this circuit measures individual qubits"
                    .to_string(),
            ));
        }
        let simulator = StatevectorSimulator {
            max_qubits: self.simulator.max_qubits,
            parallel_threshold,
            fusion: self.simulator.fusion,
        };
        let sv = simulator.run(concrete).map_err(|e| {
            log::error!("native statevector simulation failed: {e}");
            BackendError::NativeCircuit(format!("native statevector simulation failed: {e}"))
        })?;
        let width = counts_width(concrete, source.declared_clbits);
        Ok(sv
            .probabilities()
            .into_iter()
            .enumerate()
            .filter(|&(_, p)| p != 0.0)
            .map(|(state, p)| (bitstring(state, width), p))
            .collect())
    }
}

/// What the native backend runs for one [`BoundCircuit`]: the circuit, and the
/// classical width its OpenQASM program declares (the sum of its `creg`s),
/// taken before any transpiler runs, so a transpiler that builds a new
/// [`ConcreteCircuit`] cannot drop it. See [`format_counts`].
struct NativeSource<'a> {
    circuit: Cow<'a, ConcreteCircuit>,
    declared_clbits: Option<usize>,
}

/// The error a simulation failure `e` reaches the caller as.
fn simulation_failed(e: &SimError) -> BackendError {
    BackendError::NativeCircuit(format!("native statevector simulation failed: {e}"))
}

/// The [`ConcreteCircuit`] a [`BoundCircuit`] stands for, obtained without
/// touching Python: a `Native` circuit is borrowed, a `Qasm2` program is parsed
/// by the OpenQASM importer (keeping the classical width it declares), and a
/// Qiskit `Foreign` circuit is rejected. Shared by execution
/// ([`NativeStatevectorBackend::concrete_circuit`]) and the structural
/// pre-check ([`NativeStatevectorBackend::check_circuit`]), so both accept and
/// reject exactly the same circuits with the same message. It logs nothing:
/// execution logs a rejection, the pre-check does not.
fn native_source(circuit: &BoundCircuit) -> Result<NativeSource<'_>, BackendError> {
    let source = match circuit {
        BoundCircuit::Native(cc) => Ok(NativeSource {
            circuit: Cow::Borrowed(cc),
            declared_clbits: cc.declared_clbits(),
        }),
        BoundCircuit::Qasm2(qasm) => ParameterizedCircuit::from_qasm2(qasm)
            .and_then(|pc| {
                Ok(NativeSource {
                    circuit: Cow::Owned(pc.assign_parameters(&[])?),
                    declared_clbits: pc.declared_clbits(),
                })
            })
            .map_err(|e| {
                BackendError::NativeCircuit(format!(
                    "native backend could not parse OpenQASM 2.0: {e}"
                ))
            }),
        BoundCircuit::Foreign(_) => Err(BackendError::UnsupportedCircuit(
            "the native statevector backend cannot execute a Qiskit QuantumCircuit; \
             pass a polypus.Circuit or an OpenQASM 2.0 string, or select backend=\"aer\""
                .to_string(),
        )),
    }?;
    check_counts_width(&source)?;
    Ok(source)
}

/// Refuse, before anything is simulated, a measured circuit whose counts would
/// be wider than [`MAX_CLBITS`] bits (issue #346): the simulator keys outcomes
/// by a `usize`, and every distinct outcome becomes a `width`-byte bitstring,
/// so a wide declared register (`creg c[65535];`) would otherwise cost memory
/// in proportion to it. A measurement-free circuit is `num_qubits` wide, which
/// the qubit ceiling already bounds.
fn check_counts_width(source: &NativeSource<'_>) -> Result<(), BackendError> {
    let circuit = source.circuit.as_ref();
    if circuit.num_clbits() == 0 {
        return Ok(());
    }
    match counts_width(circuit, source.declared_clbits) {
        width if width > MAX_CLBITS => Err(BackendError::NativeCircuit(format!(
            "the native statevector backend supports classical registers of at most \
             {MAX_CLBITS} bits, but this circuit's is {width} bits wide; \
             select backend=\"aer\" for a wider register"
        ))),
        _ => Ok(()),
    }
}

/// The widest circuit in a batch, used to size the memory budget (plan §4.5):
/// the native backend budgets for its largest statevector. Each circuit's width
/// comes from the shared, GIL-free [`BoundCircuit::native_qubit_width`] — `Native`
/// circuits expose it directly, a `Qasm2` program is parsed for it, and a `Qiskit`
/// circuit (which this backend rejects at execution) contributes nothing. An empty
/// or all-unsupported batch yields 0 — a one-amplitude budget that keeps full
/// thread concurrency and lets `simulate_one` surface the real per-circuit error.
fn representative_qubits(qcs: &[BoundCircuit]) -> usize {
    qcs.iter()
        .filter_map(BoundCircuit::native_qubit_width)
        .max()
        .unwrap_or(0)
}

/// Format raw basis-state counts as Aer-compatible bitstrings: little-endian
/// qubit indexing with the highest classical bit on the left (contract C-3).
///
/// The width is the qubit count for a measurement-free circuit (a
/// full-register read-out). Otherwise it is the classical bits the OpenQASM
/// program declared (`declared_clbits`, the sum of its `creg`s), as on Aer,
/// or the implicit register size `num_clbits()` for a circuit without declared
/// registers — whichever is larger; the bits no measurement writes read 0.
///
/// `num_clbits()` is 0 exactly when the circuit has no `Measure`/`MeasureAll`
/// instruction: it is derived from those instructions, never from declared
/// registers. So a `creg` declared but unmeasured does not change the width —
/// pinned by `unmeasured_circuit_is_num_qubits_wide_whatever_its_cregs`.
fn format_counts(
    concrete: &ConcreteCircuit,
    declared_clbits: Option<usize>,
    raw: HashMap<usize, u64>,
) -> HashMap<String, u64> {
    let width = counts_width(concrete, declared_clbits);
    raw.into_iter()
        .map(|(state, count)| (bitstring(state, width), count))
        .collect()
}

/// The width [`format_counts`] keys a circuit's counts at.
fn counts_width(concrete: &ConcreteCircuit, declared_clbits: Option<usize>) -> usize {
    match concrete.num_clbits() {
        0 => concrete.num_qubits(),
        c => declared_clbits.map_or(c, |declared| declared.max(c)),
    }
}

/// `state` as a `width`-character bitstring, most significant bit first. Built
/// by hand rather than with `format!("{:0w$b}")`, whose width is limited to
/// `u16::MAX`: bits above the key's own width read 0, at any `width`.
fn bitstring(state: usize, width: usize) -> String {
    (0..width)
        .rev()
        .map(|i| {
            if i < MAX_CLBITS && (state >> i) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

impl QuantumBackend for NativeStatevectorBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        // Bound the peak memory to a statevector budget (plan §4.5, P1-memory):
        // the real cost of a population batch is the concurrent statevectors
        // (`2^n * 16` bytes each), not the cheap `BoundCircuit` list, so cap how
        // many run at once by the batch's widest circuit. This is a pure resource
        // bound — counts are unchanged (see `run_batch_with_cap`). A widest circuit
        // that cannot fit even alone is refused before any window starts.
        let widest = representative_qubits(qcs);
        self.check_memory(widest)?;
        let cap = max_statevector_concurrency(widest, rayon::current_num_threads());
        self.run_batch_with_cap(qcs, config, cap)
    }

    /// Report the wave size the planner should use for this batch, derived from
    /// the same statevector memory budget [`run_circuits`](Self::run_circuits)
    /// enforces internally — the batch's widest circuit fed to
    /// [`max_statevector_concurrency`] against `rayon::current_num_threads()`,
    /// read off the task slice without cloning any circuit. GIL-free:
    /// `Native`/`Qasm2` widths need no interpreter, and a `Qiskit` circuit
    /// (rejected at execution) is ignored.
    ///
    /// The wave size is capped **only when the whole batch cannot be held in the
    /// memory budget at once** (the high-qubit regime): there `run_circuits` would
    /// otherwise process the batch as several sequential memory windows inside
    /// *one* call, so exposing the cap lets the planner run those windows as waves
    /// with a `py.check_signals()` between them (issue #147). When the batch fits,
    /// the cap is reported as unbounded so it stays a single wave — matching
    /// `run_batch_with_cap`'s fast path; the backend still parallelises internally
    /// up to the thread count. See `wave_concurrency`
    /// for why the fit test uses the pure memory limit rather than a thread-count
    /// gate (which would degenerate on a single-core host, issue #147's 1-thread
    /// case).
    ///
    /// Always `Ok`: this backend reads widths GIL-free (from the `Native` struct
    /// or by parsing `Qasm2`), so unlike [`LocalBackend`](crate::LocalBackend) it can never be
    /// interrupted mid-read. The `Result` exists only to satisfy the trait.
    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        let widest = tasks
            .iter()
            .filter_map(|t| t.circuit.native_qubit_width())
            .max()
            .unwrap_or(0);
        Ok(BackendCapabilities {
            max_concurrency: crate::wave_concurrency(
                widest,
                rayon::current_num_threads(),
                tasks.len(),
            ),
            supports_shot_distribution: true,
        })
    }

    /// Single-evolution fast path: `polypus-sim` separates evolution from
    /// sampling, so the shared circuit is evolved **once** and every shot batch
    /// samples from that same statevector — instead of re-running the identical
    /// evolution once per replica as the default (via `run_circuits`) would.
    ///
    /// Reproducibility is preserved exactly (contract C-7): the per-batch seeds
    /// come from the *same* contiguous-block scheme `run_circuits` uses
    /// (`counter.fetch_add` + the SplitMix64 per-circuit seed), and
    /// [`sample_projected`] with a given seed yields byte-identical counts to a
    /// full `run_and_sample`. Shot conservation is preserved too (contract
    /// C-3): a zero-shot batch samples nothing and returns an empty map.
    fn run_shots_distributed(
        &self,
        qc: &BoundCircuit,
        shot_batches: &[u32],
        config: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        // Evolve the shared circuit exactly once; the statevector is reused for
        // every batch's sampling.
        let source = self.concrete_circuit(qc, &opts)?;
        let concrete = source.circuit.as_ref();
        self.check_memory(concrete.num_qubits())?;
        let sv = self.simulator.run(concrete).map_err(|e| {
            log::error!("native statevector simulation failed: {e}");
            simulation_failed(&e)
        })?;
        // Reserve a contiguous seed block, one seed per batch, from the same
        // scheme `run_circuits` uses so a given base seed reproduces identical
        // counts regardless of which path ran the batches.
        let start = self
            .counter
            .fetch_add(shot_batches.len() as u64, Ordering::Relaxed);
        shot_batches
            .iter()
            .enumerate()
            .map(|(i, &shots)| {
                let seed = circuit_seed(self.base_seed, start.wrapping_add(i as u64));
                let raw = sample_projected(concrete, &sv, shots as usize, seed).map_err(|e| {
                    log::error!("native statevector sampling failed: {e}");
                    BackendError::NativeCircuit(format!("native statevector sampling failed: {e}"))
                })?;
                Ok(format_counts(concrete, source.declared_clbits, raw))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionConfig, OptLevel};
    use polypus_circuit::{GateInstruction, ParameterizedCircuit};
    use std::sync::atomic::AtomicU8;
    use std::sync::Arc;

    fn bell() -> ConcreteCircuit {
        ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    fn config_with(opt_level: OptLevel) -> ExecutionConfig {
        ExecutionConfig {
            id: "abc".to_string(),
            shots: 500,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: crate::BackendConfig::LocalNative { fusion: true },
            opt_level,
            // No explicit seed: exercises the OS-entropy fallback path in
            // `Infrastructure::create_backend`. Tests that build the backend
            // directly pass their seed to `new`/`with_transpiler`, so this field
            // is unused by them.
            seed: None,
        }
    }

    /// 3-qubit uniform superposition (H on every qubit) with a full read-out.
    /// Its counts spread over eight bitstrings, so two independent samplings
    /// differ with overwhelming probability — the circuit used to assert that
    /// distinct/omitted seeds produce distinct counts without statistical
    /// flakiness (a 2-outcome Bell state would collide far too often).
    fn uniform3() -> ConcreteCircuit {
        ParameterizedCircuit::new(3)
            .h(0)
            .h(1)
            .h(2)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    /// Records the last [`OptLevel`] it was asked to honor, proving the level set
    /// on [`ExecutionConfig`] reaches `transpile` as an argument.
    struct RecordingTranspiler {
        seen: Arc<AtomicU8>,
    }
    impl Transpiler for RecordingTranspiler {
        fn transpile(&self, circuit: &ConcreteCircuit, opts: &TranspileOptions) -> ConcreteCircuit {
            self.seen.store(opts.level as u8, Ordering::SeqCst);
            circuit.clone()
        }
    }

    /// Rewrites the circuit by appending a `barrier` (a simulation no-op),
    /// demonstrating the composition injection point works end-to-end through
    /// `run_circuits` without altering measurement outcomes.
    struct BarrierTranspiler;
    impl Transpiler for BarrierTranspiler {
        fn transpile(
            &self,
            circuit: &ConcreteCircuit,
            _opts: &TranspileOptions,
        ) -> ConcreteCircuit {
            let mut gates = circuit.gates().to_vec();
            gates.push(GateInstruction::Barrier(Vec::new()));
            circuit
                .try_with_gates(gates)
                .expect("a trailing barrier keeps the circuit valid")
        }
    }

    #[test]
    fn with_fusion_toggles_the_simulator_flag() {
        assert!(NativeStatevectorBackend::new(0).simulator.fusion);
        assert!(
            !NativeStatevectorBackend::new(0)
                .with_fusion(false)
                .simulator
                .fusion
        );
        assert!(
            NativeStatevectorBackend::new(0)
                .with_fusion(true)
                .simulator
                .fusion
        );
    }

    /// End-to-end: `create_backend` threads `BackendConfig::LocalNative`'s
    /// `fusion` field all the way to the simulator it constructs, for both
    /// values — not just the default. A same-seed, same-circuit comparison
    /// against the fusion-enabled backend must still agree (fusion changes
    /// performance, never the result), proving `fusion: false` reached a
    /// working (not just a differently-configured) simulator.
    #[test]
    fn create_backend_threads_fusion_through_local_native() {
        use crate::Infrastructure;

        let mut cfg = config_with(OptLevel::default());
        // Pinned so the only variable between the two branches is `fusion` --
        // `config_with`'s default `seed: None` would draw independent
        // OS-entropy seeds per `create_backend` call, which would make the
        // two branches' counts differ for a reason unrelated to fusion.
        cfg.seed = Some(2024);
        cfg.backend_config = crate::BackendConfig::LocalNative { fusion: true };
        let fused = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg.run_params())
            .unwrap();

        cfg.backend_config = crate::BackendConfig::LocalNative { fusion: false };
        let unfused = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg.run_params())
            .unwrap();

        assert_eq!(fused, unfused);
    }

    #[test]
    fn bell_counts_only_correlated_outcomes() {
        let backend = NativeStatevectorBackend::new(0);
        let counts = backend
            .simulate_one(
                &BoundCircuit::Native(bell()),
                2000,
                7,
                &TranspileOptions::default(),
                backend.simulator.parallel_threshold,
            )
            .unwrap();
        let total: u64 = counts.values().sum();
        assert_eq!(total, 2000);
        for key in counts.keys() {
            assert!(key == "00" || key == "11", "unexpected outcome {key}");
        }
    }

    /// Non-regression: the default backend (identity transpiler, default opt
    /// level) produces exactly the same counts as one built with an explicit
    /// [`IdentityTranspiler`] — introducing the transpiler changes nothing.
    #[test]
    fn identity_default_matches_explicit_identity() {
        let opts = TranspileOptions::default();
        let default_backend = NativeStatevectorBackend::new(0);
        let explicit_backend =
            NativeStatevectorBackend::with_transpiler(0, Box::new(IdentityTranspiler));
        let circuit = BoundCircuit::Native(bell());
        assert_eq!(
            default_backend
                .simulate_one(
                    &circuit,
                    1000,
                    42,
                    &opts,
                    default_backend.simulator.parallel_threshold
                )
                .unwrap(),
            explicit_backend
                .simulate_one(
                    &circuit,
                    1000,
                    42,
                    &opts,
                    explicit_backend.simulator.parallel_threshold
                )
                .unwrap(),
        );
    }

    #[test]
    fn accepts_qasm2_strings() {
        let qasm = ParameterizedCircuit::new(1)
            .x(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
            .try_to_qasm2()
            .unwrap();
        let backend = NativeStatevectorBackend::new(0);
        let counts = backend
            .simulate_one(
                &BoundCircuit::Qasm2(qasm),
                128,
                1,
                &TranspileOptions::default(),
                backend.simulator.parallel_threshold,
            )
            .unwrap();
        // X|0> = |1>: every shot reads "1".
        assert_eq!(counts.get("1"), Some(&128));
    }

    /// The Qasm2 path (parsed in Rust) and the Native path yield identical counts
    /// for the same circuit and seed under the identity transpiler.
    #[test]
    fn qasm2_path_matches_native_path() {
        let backend = NativeStatevectorBackend::new(0);
        let opts = TranspileOptions::default();
        let native = backend
            .simulate_one(
                &BoundCircuit::Native(bell()),
                1000,
                5,
                &opts,
                backend.simulator.parallel_threshold,
            )
            .unwrap();
        let qasm = backend
            .simulate_one(
                &BoundCircuit::Qasm2(bell().try_to_qasm2().unwrap()),
                1000,
                5,
                &opts,
                backend.simulator.parallel_threshold,
            )
            .unwrap();
        assert_eq!(native, qasm);
    }

    /// Regression (issue #83): under the identity transpiler, the `Qasm2` path
    /// of `transpiled` returns the *exact input bytes* without parsing and
    /// re-serializing. The input carries a comment (dropped on parse) and
    /// non-canonical spacing, so a parse + re-emit round trip provably changes
    /// the bytes — asserting byte-identity therefore proves the parse was
    /// skipped, not merely that the QASM happens to be stable.
    #[test]
    fn qasm2_identity_path_returns_input_bytes_without_reserializing() {
        let original = "OPENQASM 2.0;\n\
             include \"qelib1.inc\";\n\
             // this comment is dropped when the QASM is parsed\n\
             qreg q[2];\n\
             h q[0];\n\
             cx q[0], q[1];\n"
            .to_string();

        // Premise check: a naive parse + re-emit really would change the bytes,
        // so the byte-identity assertion below is meaningful.
        let reserialized = ParameterizedCircuit::from_qasm2(&original)
            .and_then(|pc| pc.assign_parameters(&[]))
            .unwrap()
            .try_to_qasm2()
            .unwrap();
        assert_ne!(
            reserialized, original,
            "test premise: this QASM must not round-trip byte-identically"
        );

        let out = BoundCircuit::Qasm2(original.clone())
            .transpiled(&IdentityTranspiler, &TranspileOptions::default());
        match out {
            BoundCircuit::Qasm2(text) => assert_eq!(text, original),
            _ => panic!("expected the original Qasm2 variant to be preserved"),
        }
    }

    /// An unparseable / unsupported QASM string is passed through untouched by
    /// the transpile helper instead of panicking (best-effort contract).
    #[test]
    fn unparseable_qasm_passes_through_without_panic() {
        let original = "this is definitely not valid openqasm".to_string();
        let out = BoundCircuit::Qasm2(original.clone())
            .transpiled(&IdentityTranspiler, &TranspileOptions::default());
        match out {
            BoundCircuit::Qasm2(text) => assert_eq!(text, original),
            _ => panic!("expected the original Qasm2 variant to be preserved"),
        }
    }

    /// End-to-end: the `opt_level` set in [`ExecutionConfig`] reaches the
    /// injected transpiler's `transpile` as a [`TranspileOptions`] argument.
    #[test]
    fn opt_level_reaches_transpiler() {
        let seen = Arc::new(AtomicU8::new(0xFF));
        let backend = NativeStatevectorBackend::with_transpiler(
            0,
            Box::new(RecordingTranspiler {
                seen: Arc::clone(&seen),
            }),
        );
        let cfg = config_with(OptLevel::Heavy);
        backend
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg.run_params())
            .unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), OptLevel::Heavy as u8);
    }

    /// End-to-end: an injected non-identity strategy is actually applied inside
    /// `run_circuits`. The appended `barrier` is a simulation no-op, so the
    /// counts remain the correlated Bell outcomes — the rewrite is exercised via
    /// the injection point, not observed as wrong results.
    #[test]
    fn injected_strategy_runs_through_run_circuits() {
        let backend = NativeStatevectorBackend::with_transpiler(0, Box::new(BarrierTranspiler));
        let cfg = config_with(OptLevel::default());
        let counts = backend
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg.run_params())
            .unwrap();
        assert_eq!(counts.len(), 1);
        let total: u64 = counts[0].values().sum();
        assert_eq!(total, u64::from(cfg.shots));
        for key in counts[0].keys() {
            assert!(key == "00" || key == "11", "unexpected outcome {key}");
        }
    }

    /// P1-memory (plan §4.5): capping concurrent statevectors must be a pure
    /// resource bound — the counts are byte-identical to the single unbounded
    /// `par_iter` for *every* cap, because the per-circuit seed is fixed up front
    /// (`circuit_seed(base_seed, block_start + i)`) and gate-vs-circuit parallelism is
    /// numerically identical. The batch is sized past the thread pool so the full
    /// path takes the `usize::MAX` gate-threshold branch while the small-cap
    /// windows take the default branch, pinning that this threshold flip changes
    /// nothing. Each backend gets a fresh seed block, so all runs start at 0.
    #[test]
    fn capped_batches_match_the_full_batch_byte_for_byte() {
        let cfg = config_with(OptLevel::default());
        let n = rayon::current_num_threads() + 3;
        // Identical circuits, but each index draws its own seed, so the output
        // vector is order- and seed-sensitive: a mis-seeded or reordered window
        // would change it.
        let batch: Vec<BoundCircuit> = (0..n).map(|_| BoundCircuit::Native(uniform3())).collect();

        // Reference: whole batch in one window (cap >= len => fast path).
        let full = NativeStatevectorBackend::new(2024)
            .run_batch_with_cap(&batch, &cfg.run_params(), usize::MAX)
            .unwrap();
        assert_eq!(full.len(), n);

        for cap in [1usize, 2, 3, n - 1, n, n + 100] {
            let capped = NativeStatevectorBackend::new(2024)
                .run_batch_with_cap(&batch, &cfg.run_params(), cap)
                .unwrap();
            assert_eq!(
                capped, full,
                "cap={cap} must yield byte-identical counts to the full batch"
            );
        }
    }

    /// Issue #215: a circuit whose statevector does not fit in a *known* budget is
    /// refused with `InsufficientMemory` (never started), a fallback budget never
    /// refuses, and a width above the qubit ceiling is left to the simulator's own
    /// ceiling error. The budget is passed explicitly, so this is independent of
    /// the host and of `POLYPUS_MEM_BUDGET`.
    #[test]
    fn check_memory_refuses_only_what_a_known_budget_cannot_hold() {
        use polypus_backend::BudgetSource;
        let backend = NativeStatevectorBackend::new(0);
        let one_mib = |source| MemBudget {
            bytes: 1 << 20,
            source,
        };
        for source in [BudgetSource::Explicit, BudgetSource::Detected] {
            let err = backend
                .check_memory_against(20, one_mib(source))
                .unwrap_err();
            assert!(
                matches!(&err, BackendError::InsufficientMemory(e)
                    if e.num_qubits == 20 && e.required_bytes == 16 << 20 && e.source == source),
                "expected InsufficientMemory, got {err:?}"
            );
            // 16 qubits = 1 MiB: fits exactly.
            assert!(backend.check_memory_against(16, one_mib(source)).is_ok());
            // Above the ceiling: the simulator's TooManyQubits stays the error.
            assert!(backend
                .check_memory_against(polypus_sim::MAX_QUBITS + 1, one_mib(source))
                .is_ok());
        }
        assert!(backend
            .check_memory_against(30, one_mib(BudgetSource::Fallback))
            .is_ok());
    }

    /// The ceiling error is still what a circuit above `MAX_QUBITS` gets from
    /// `run_circuits` (the memory check does not pre-empt it).
    #[test]
    fn run_circuits_above_the_ceiling_keeps_the_ceiling_error() {
        let backend = NativeStatevectorBackend::new(0);
        let too_wide = BoundCircuit::Native(
            ParameterizedCircuit::new(polypus_sim::MAX_QUBITS + 1)
                .measure_all()
                .assign_parameters(&[])
                .unwrap(),
        );
        let err = backend
            .run_circuits(
                &[too_wide],
                &RunParams {
                    id: "ceiling".to_string(),
                    shots: 8,
                    seed: Some(1),
                    opt_level: OptLevel::default(),
                },
            )
            .unwrap_err();
        assert!(
            matches!(&err, BackendError::NativeCircuit(m) if m.contains("qubits")),
            "expected the ceiling error, got {err:?}"
        );
    }

    /// `check_resources` reports the ceiling error a run fails with, and lets a
    /// circuit within the ceiling and the budget through.
    #[test]
    fn check_resources_reports_the_ceiling_error_of_a_run() {
        let backend = NativeStatevectorBackend::new(0);
        let too_wide = BoundCircuit::Native(
            ParameterizedCircuit::new(polypus_sim::MAX_QUBITS + 1)
                .measure_all()
                .assign_parameters(&[])
                .unwrap(),
        );
        let checked = NativeStatevectorBackend::check_resources(&too_wide).unwrap_err();
        let run = backend
            .run_circuits(
                &[too_wide],
                &RunParams {
                    id: "ceiling".to_string(),
                    shots: 8,
                    seed: Some(1),
                    opt_level: OptLevel::default(),
                },
            )
            .unwrap_err();
        assert!(matches!(&checked, BackendError::NativeCircuit(_)));
        assert_eq!(checked.to_string(), run.to_string());
        NativeStatevectorBackend::check_resources(&BoundCircuit::Native(bell())).unwrap();
    }

    /// Issue #147: a high-qubit batch that cannot fit in the memory budget is
    /// reported with the memory cap `run_circuits` enforces internally, so the
    /// planner can split it into memory-safe waves. A 40-qubit statevector is
    /// 16 TiB — beyond any budget this test can meet, detected or explicit (issue
    /// #215 made the default host-dependent, so the historical 30 qubits vs. a
    /// fixed 16 GiB no longer pins the outcome) — so only one fits at a time, and a
    /// 4-circuit batch cannot be held at once ⇒ cap 1 — **regardless of the host
    /// core count** (this is the single-core regression from issue #147, so the
    /// test deliberately does not guard on the thread count). The arithmetic for an
    /// explicit budget is pinned by `polypus-backend`'s `wave_concurrency_tests`.
    /// The circuit is zero-gate and never simulated, so this costs nothing. Crucially the
    /// batch-agnostic `capabilities()` is left at its unbounded default, proving
    /// the change is purely additive.
    #[test]
    fn capabilities_for_exposes_the_memory_cap_leaving_capabilities_unchanged() {
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(40)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks: Vec<CircuitTask> = (0..4)
            .map(|_| CircuitTask {
                circuit: &wide,
                shots: 8,
            })
            .collect();
        let backend = NativeStatevectorBackend::new(0);

        let cap = backend.capabilities_for(&tasks).unwrap().max_concurrency;
        assert_eq!(
            cap, 1,
            "a 40-qubit statevector (16 TiB) admits one at a time under any budget"
        );
        assert!(
            cap < tasks.len(),
            "the cap must force more than one wave for this batch"
        );

        // The frozen, batch-agnostic seam is untouched.
        assert_eq!(backend.capabilities().max_concurrency, usize::MAX);
        assert!(backend.capabilities().supports_shot_distribution);
    }

    /// A low-qubit batch fits comfortably in the memory budget, so it is reported
    /// as a **single unbounded wave** — the planner hands the whole batch to the
    /// backend in one call (which parallelises internally), with no per-wave
    /// chunking overhead — regardless of the host core count.
    #[test]
    fn capabilities_for_keeps_a_single_wave_at_low_qubits() {
        let small =
            BoundCircuit::Native(ParameterizedCircuit::new(4).assign_parameters(&[]).unwrap());
        let tasks: Vec<CircuitTask> = (0..8)
            .map(|_| CircuitTask {
                circuit: &small,
                shots: 8,
            })
            .collect();
        let cap = NativeStatevectorBackend::new(0)
            .capabilities_for(&tasks)
            .unwrap()
            .max_concurrency;
        assert_eq!(cap, usize::MAX);
    }

    /// Native counterpart of local.rs's `widest_qubits_picks_the_largest_circuit`:
    /// `representative_qubits` budgets for the largest statevector in a mixed batch.
    /// The `Native`/`Qasm2` widths are read GIL-free; a `Qiskit` circuit — which
    /// this backend rejects at execution — contributes nothing, so it must not raise
    /// the budget; an empty batch yields 0.
    #[test]
    fn representative_qubits_picks_the_largest_circuit() {
        use pyo3::prelude::*;

        let small = ParameterizedCircuit::new(2)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();
        let big = ParameterizedCircuit::new(7)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();

        // Include a Qiskit variant: the native backend cannot read it, so despite
        // standing in for an arbitrarily wide circuit it contributes nothing.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let batch = vec![
                BoundCircuit::Native(small),
                BoundCircuit::Qasm2(big.try_to_qasm2().unwrap()),
                crate::QiskitCircuit::into_bound(py.None()),
            ];
            assert_eq!(representative_qubits(&batch), 7);
        });
        assert_eq!(representative_qubits(&[]), 0);
    }

    /// Acceptance criterion (defect #1), positive half: the *same explicit
    /// seed* reproduces byte-identical counts across two independent backends,
    /// regardless of the run `id` (which no longer feeds the RNG).
    #[test]
    fn same_seed_reproduces_counts() {
        let cfg = config_with(OptLevel::default());
        let a = NativeStatevectorBackend::new(2024);
        let b = NativeStatevectorBackend::new(2024);
        let batch = vec![BoundCircuit::Native(bell())];
        assert_eq!(
            a.run_circuits(&batch, &cfg.run_params()).unwrap(),
            b.run_circuits(&batch, &cfg.run_params()).unwrap()
        );
    }

    /// Contract C-7 (#411): every call of a run gets its own stream. Two calls of
    /// one backend with the same batch sample differently, and a fresh backend
    /// with the same seed replays the sequence.
    #[test]
    fn consecutive_calls_get_their_own_streams() {
        let cfg = config_with(OptLevel::default());
        let batch = vec![BoundCircuit::Native(uniform3())];
        let run = |backend: &NativeStatevectorBackend| {
            backend.run_circuits(&batch, &cfg.run_params()).unwrap()
        };
        let a = NativeStatevectorBackend::new(2024);
        let (first, second) = (run(&a), run(&a));
        assert_ne!(first, second);
        let b = NativeStatevectorBackend::new(2024);
        assert_eq!((run(&b), run(&b)), (first, second));
    }

    /// Issue #348: neighbouring base seeds must not share sampling streams
    /// shifted by one circuit. With the former `base_seed + index` scheme,
    /// circuit `k` of seed `s` replayed circuit `k-1` of seed `s+1`.
    #[test]
    fn neighbouring_seeds_do_not_share_shifted_streams() {
        for s in [0u64, 1, 42, 2024, u64::MAX - 1] {
            for k in 1..64u64 {
                assert_ne!(circuit_seed(s, k), circuit_seed(s.wrapping_add(1), k - 1));
            }
        }

        let cfg = config_with(OptLevel::default());
        let batch: Vec<BoundCircuit> = (0..4).map(|_| BoundCircuit::Native(uniform3())).collect();
        let a = NativeStatevectorBackend::new(42)
            .run_circuits(&batch, &cfg.run_params())
            .unwrap();
        let b = NativeStatevectorBackend::new(43)
            .run_circuits(&batch, &cfg.run_params())
            .unwrap();
        for k in 1..batch.len() {
            assert_ne!(
                a[k],
                b[k - 1],
                "circuit {k} of seed 42 == circuit {} of seed 43",
                k - 1
            );
        }
    }

    /// The per-circuit seed map is pinned: a change would silently move every
    /// count for a fixed seed (C-7), so it must be deliberate.
    #[test]
    fn circuit_seed_is_pinned() {
        // SplitMix64 seeded with state 0: first output (reference value).
        assert_eq!(circuit_seed(0, 0), 0xE220_A839_7B1D_CDAF);
        let seeds: std::collections::HashSet<u64> = (0..1000).map(|k| circuit_seed(7, k)).collect();
        assert_eq!(seeds.len(), 1000);
    }

    /// Distinct seeds produce distinct counts: the seed genuinely drives the
    /// sampling stream (guards against the seed being ignored).
    #[test]
    fn distinct_seeds_produce_distinct_counts() {
        let cfg = config_with(OptLevel::default());
        let a = NativeStatevectorBackend::new(1);
        let b = NativeStatevectorBackend::new(2);
        let batch = vec![BoundCircuit::Native(uniform3())];
        assert_ne!(
            a.run_circuits(&batch, &cfg.run_params()).unwrap(),
            b.run_circuits(&batch, &cfg.run_params()).unwrap()
        );
    }

    /// Acceptance criterion (defect #1), negative half: with **no explicit
    /// seed**, two runs sharing the *same `id`* must NOT reproduce each other —
    /// the shot noise is real noise, not an artefact of hashing the `id`.
    /// `create_backend` draws a fresh OS-entropy seed for each `LocalNative`
    /// backend when `config.seed` is `None`. Asserting inequality of the full
    /// eight-outcome counts dict (not a single statistic) keeps this
    /// non-flaky. This is the exact inversion of the removed
    /// `seeding_is_reproducible_per_id`, which encoded the bug.
    /// Fast path (defect #3): `run_shots_distributed` conserves the total shots
    /// across batches (C-3) and is deterministic for a fixed base seed (C-7).
    #[test]
    fn run_shots_distributed_conserves_shots_and_is_deterministic() {
        let cfg = config_with(OptLevel::default());
        let batches = [3u32, 3, 2]; // 8 shots over 3 "QPUs", uneven split.
        let out = NativeStatevectorBackend::new(2024)
            .run_shots_distributed(&BoundCircuit::Native(bell()), &batches, &cfg.run_params())
            .unwrap();
        assert_eq!(out.len(), batches.len());
        let total: u64 = out.iter().flat_map(|m| m.values()).sum();
        assert_eq!(total, u64::from(batches.iter().sum::<u32>()));
        // Same base seed reproduces byte-identical batches.
        let again = NativeStatevectorBackend::new(2024)
            .run_shots_distributed(&BoundCircuit::Native(bell()), &batches, &cfg.run_params())
            .unwrap();
        assert_eq!(out, again);
        for m in &out {
            for k in m.keys() {
                assert!(k == "00" || k == "11", "unexpected outcome {k}");
            }
        }
    }

    /// Fast path equivalence (C-7): the single-evolution path yields
    /// byte-identical counts to sampling each batch through `run_circuits` with
    /// the same contiguous seed block (batch `i` uses seed index `i`). The
    /// optimisation changes performance, never results.
    #[test]
    fn run_shots_distributed_matches_run_circuits_per_batch() {
        let batches = [5u32, 5, 4];
        let fast = NativeStatevectorBackend::new(99)
            .run_shots_distributed(
                &BoundCircuit::Native(bell()),
                &batches,
                &config_with(OptLevel::default()).run_params(),
            )
            .unwrap();

        // Reference: one backend seeded 99, one single-circuit `run_circuits`
        // call per batch, so call `i` draws seed index `i` — the exact seed the
        // fast path reserves for replica `i`.
        let reference_backend = NativeStatevectorBackend::new(99);
        let reference: Vec<_> = batches
            .iter()
            .map(|&shots| {
                let mut cfg = config_with(OptLevel::default());
                cfg.shots = shots;
                reference_backend
                    .run_circuits(&[BoundCircuit::Native(bell())], &cfg.run_params())
                    .unwrap()
                    .remove(0)
            })
            .collect();

        assert_eq!(fast, reference);
    }

    /// A zero-shot batch samples nothing (empty map), so shots stay conserved
    /// even when `shots < n_qpus` (some replicas get `base = 0`).
    #[test]
    fn run_shots_distributed_zero_batch_is_empty() {
        let cfg = config_with(OptLevel::default());
        let out = NativeStatevectorBackend::new(1)
            .run_shots_distributed(
                &BoundCircuit::Native(bell()),
                &[1u32, 0, 0],
                &cfg.run_params(),
            )
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[1].is_empty() && out[2].is_empty());
        let total: u64 = out.iter().flat_map(|m| m.values()).sum();
        assert_eq!(total, 1);
    }

    /// Regression (issue #218, C-3): a circuit with no measurement instruction
    /// is read out on the full quantum register, `num_qubits` wide, whatever
    /// classical registers it declares — narrower, as wide or wider than the
    /// quantum one — on both the batch and the shot-distributed paths.
    #[test]
    fn unmeasured_circuit_is_num_qubits_wide_whatever_its_cregs() {
        let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n";
        let cfg = config_with(OptLevel::default());
        for cregs in [
            "",
            "creg c[1];\n",
            "creg c[3];\n",
            "creg c[5];\n",
            "creg a[2];\ncreg b[4];\n",
        ] {
            let qasm = format!("{header}{cregs}x q[0];\nx q[2];\n");
            let circuit = BoundCircuit::Qasm2(qasm);
            let backend = NativeStatevectorBackend::new(3);
            let batch = backend
                .run_circuits(std::slice::from_ref(&circuit), &cfg.run_params())
                .unwrap();
            assert_eq!(
                batch,
                vec![HashMap::from([("101".to_string(), 500)])],
                "{cregs:?}"
            );
            let distributed = backend
                .run_shots_distributed(&circuit, &[7, 5], &cfg.run_params())
                .unwrap();
            assert_eq!(
                distributed,
                vec![
                    HashMap::from([("101".to_string(), 7)]),
                    HashMap::from([("101".to_string(), 5)]),
                ],
                "{cregs:?}"
            );
        }
    }

    /// `format_counts` itself: the width follows the measurement instructions,
    /// so the same raw state formats `num_qubits` wide without them and
    /// `num_clbits` wide with them.
    #[test]
    fn format_counts_width_follows_measurement_instructions() {
        let unmeasured = ParameterizedCircuit::new(3)
            .x(0)
            .assign_parameters(&[])
            .unwrap();
        let raw = HashMap::from([(1usize, 4u64)]);
        assert_eq!(
            format_counts(&unmeasured, None, raw.clone()),
            HashMap::from([("001".to_string(), 4)])
        );
        let measured = ParameterizedCircuit::new(3)
            .x(0)
            .measure(0, 4)
            .assign_parameters(&[])
            .unwrap();
        assert_eq!(
            format_counts(&measured, None, raw),
            HashMap::from([("00001".to_string(), 4)])
        );
    }

    /// Regression (issue #251, C-3): a measured circuit's width is the declared
    /// classical width when there is one and it is larger than `num_clbits`;
    /// an equal one changes nothing, and a measurement-free circuit stays
    /// `num_qubits` wide whatever it declares.
    #[test]
    fn format_counts_width_follows_the_declared_clbits() {
        let raw = HashMap::from([(1usize, 4u64)]);
        let measured = ParameterizedCircuit::new(3)
            .x(0)
            .measure(0, 4)
            .assign_parameters(&[])
            .unwrap();
        assert_eq!(
            format_counts(&measured, Some(7), raw.clone()),
            HashMap::from([("0000001".to_string(), 4)])
        );
        assert_eq!(
            format_counts(&measured, Some(5), raw.clone()),
            HashMap::from([("00001".to_string(), 4)])
        );
        let unmeasured = ParameterizedCircuit::new(3)
            .x(0)
            .assign_parameters(&[])
            .unwrap();
        assert_eq!(
            format_counts(&unmeasured, Some(7), raw),
            HashMap::from([("001".to_string(), 4)])
        );
    }

    /// Regression (issue #251, C-3): an OpenQASM program with measurements is
    /// keyed at the classical width it declares (the sum of its `creg`s), as on
    /// Aer, not at `max(cbit) + 1` — on the batch and the shot-distributed
    /// paths alike, including a program the importer collapses to `MeasureAll`.
    #[test]
    fn measured_qasm_is_keyed_at_its_declared_clbit_width() {
        let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
        let cfg = config_with(OptLevel::default());
        let cases = [
            (
                "qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n",
                "010",
            ),
            (
                "qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[0];\n",
                "001",
            ),
            (
                "qreg q[2];\ncreg a[2];\ncreg b[3];\nx q[1];\nmeasure q[1] -> a[1];\n",
                "00010",
            ),
            (
                "qreg q[2];\ncreg c[4];\nx q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
                "0010",
            ),
        ];
        for (body, key) in cases {
            let qasm = format!("{header}{body}");
            let circuit = BoundCircuit::Qasm2(qasm);
            let backend = NativeStatevectorBackend::new(5);
            let batch = backend
                .run_circuits(std::slice::from_ref(&circuit), &cfg.run_params())
                .unwrap();
            assert_eq!(
                batch,
                vec![HashMap::from([(key.to_string(), 500)])],
                "{body:?}"
            );
            let distributed = backend
                .run_shots_distributed(&circuit, &[7, 5], &cfg.run_params())
                .unwrap();
            assert_eq!(
                distributed,
                vec![
                    HashMap::from([(key.to_string(), 7)]),
                    HashMap::from([(key.to_string(), 5)]),
                ],
                "{body:?}"
            );
        }
        // The last case really is the collapsed `MeasureAll` shape.
        let collapsed =
            ParameterizedCircuit::from_qasm2(&format!("{header}{}", cases[3].0)).unwrap();
        assert_eq!(collapsed.gates().last(), Some(&GateInstruction::MeasureAll));
    }

    /// A `Native` circuit has no declared registers: its counts keep the
    /// implicit `max(cbit) + 1` width on both paths (issue #251 changes only
    /// OpenQASM input).
    #[test]
    fn native_circuit_keeps_the_implicit_clbit_width() {
        let cfg = config_with(OptLevel::default());
        let circuit = BoundCircuit::Native(
            ParameterizedCircuit::new(2)
                .x(0)
                .measure(0, 1)
                .assign_parameters(&[])
                .unwrap(),
        );
        let backend = NativeStatevectorBackend::new(5);
        let batch = backend
            .run_circuits(std::slice::from_ref(&circuit), &cfg.run_params())
            .unwrap();
        assert_eq!(batch, vec![HashMap::from([("10".to_string(), 500)])]);
        let distributed = backend
            .run_shots_distributed(&circuit, &[7, 5], &cfg.run_params())
            .unwrap();
        assert_eq!(
            distributed,
            vec![
                HashMap::from([("10".to_string(), 7)]),
                HashMap::from([("10".to_string(), 5)]),
            ]
        );
    }

    /// An imported circuit keeps the width its source declares through
    /// binding, even through a transpiler that rebuilds it without that width.
    #[test]
    fn imported_native_circuit_keeps_its_declared_clbit_width() {
        struct RebuildingTranspiler;
        impl Transpiler for RebuildingTranspiler {
            fn transpile(
                &self,
                circuit: &ConcreteCircuit,
                _: &TranspileOptions,
            ) -> ConcreteCircuit {
                ConcreteCircuit::new(circuit.num_qubits(), circuit.gates().to_vec())
            }
        }
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n";
        let circuit = BoundCircuit::Native(
            ParameterizedCircuit::from_qasm2(qasm)
                .unwrap()
                .assign_parameters(&[])
                .unwrap(),
        );
        let cfg = config_with(OptLevel::default());
        for backend in [
            NativeStatevectorBackend::new(5),
            NativeStatevectorBackend::with_transpiler(5, Box::new(RebuildingTranspiler)),
        ] {
            let batch = backend
                .run_circuits(std::slice::from_ref(&circuit), &cfg.run_params())
                .unwrap();
            assert_eq!(batch, vec![HashMap::from([("010".to_string(), 500)])]);
            let distributed = backend
                .run_shots_distributed(&circuit, &[3], &cfg.run_params())
                .unwrap();
            assert_eq!(distributed, vec![HashMap::from([("010".to_string(), 3)])]);
        }
    }

    /// A non-identity transpiler rebuilds the circuit; the declared width must
    /// survive it, so the injected strategy cannot change the keys.
    #[test]
    fn declared_clbit_width_survives_a_transpiler() {
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n";
        let circuit = BoundCircuit::Qasm2(qasm.to_string());
        let cfg = config_with(OptLevel::default());
        let backend = NativeStatevectorBackend::with_transpiler(5, Box::new(BarrierTranspiler));
        let batch = backend
            .run_circuits(std::slice::from_ref(&circuit), &cfg.run_params())
            .unwrap();
        assert_eq!(batch, vec![HashMap::from([("010".to_string(), 500)])]);
        let distributed = backend
            .run_shots_distributed(&circuit, &[3], &cfg.run_params())
            .unwrap();
        assert_eq!(distributed, vec![HashMap::from([("010".to_string(), 3)])]);
    }

    /// `check_circuit` accepts and rejects exactly what execution does, with the
    /// same error, and simulates nothing.
    /// The native backend's verdict on `circuit` on all three paths — the
    /// pre-check, the batch and the shot-distributed one — which must agree.
    fn native_rejection(circuit: &BoundCircuit) -> String {
        let cfg = config_with(OptLevel::default());
        let backend = NativeStatevectorBackend::new(0);
        let checked = NativeStatevectorBackend::check_circuit(circuit);
        let run = backend.run_circuits(std::slice::from_ref(circuit), &cfg.run_params());
        let distributed = backend.run_shots_distributed(circuit, &[7, 5], &cfg.run_params());
        let messages: Vec<String> = [checked, run.map(|_| ()), distributed.map(|_| ())]
            .into_iter()
            .map(|result| match result {
                Err(BackendError::NativeCircuit(m)) => m,
                other => panic!("expected a NativeCircuit error, got {other:?}"),
            })
            .collect();
        assert!(messages.iter().all(|m| *m == messages[0]), "{messages:?}");
        messages[0].clone()
    }

    /// Issues #345 (b) and #346: a valid program measuring into `c[70]` used
    /// to shift the bit out of the `usize` key (wrong counts without overflow
    /// checks, a panic with them), and one declaring `creg c[65536]` panicked
    /// formatting its counts. Both are typed errors now, before simulating.
    #[test]
    fn a_classical_register_wider_than_the_counts_key_is_a_typed_error() {
        let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n";
        let cases = [
            (
                format!("{header}creg c[71];\nx q[0];\nmeasure q[0] -> c[70];\n"),
                71,
            ),
            (
                format!("{header}creg c[65536];\nx q[0];\nmeasure q[0] -> c[0];\n"),
                65536,
            ),
        ];
        for (qasm, width) in cases {
            let message = native_rejection(&BoundCircuit::Qasm2(qasm));
            assert!(
                message.contains(&format!("is {width} bits wide")),
                "{message}"
            );
            assert!(
                message.contains(&format!("at most {MAX_CLBITS} bits")),
                "{message}"
            );
        }
        // A native circuit, with no declared register, the same.
        let native = ParameterizedCircuit::new(1)
            .x(0)
            .measure(0, 70)
            .assign_parameters(&[])
            .unwrap();
        let message = native_rejection(&BoundCircuit::Native(native));
        assert!(message.contains("is 71 bits wide"), "{message}");
    }

    /// The widest register the counts key holds runs, with the measured bit in
    /// its own place.
    #[test]
    fn a_classical_register_as_wide_as_the_counts_key_is_sampled_in_place() {
        let qasm = format!(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[{MAX_CLBITS}];\n\
             x q[0];\nmeasure q[0] -> c[{}];\n",
            MAX_CLBITS - 1
        );
        let cfg = config_with(OptLevel::default());
        let counts = NativeStatevectorBackend::new(0)
            .run_circuits(&[BoundCircuit::Qasm2(qasm)], &cfg.run_params())
            .unwrap();
        let key = format!("1{}", "0".repeat(MAX_CLBITS - 1));
        assert_eq!(counts[0].len(), 1);
        assert_eq!(counts[0].get(&key).copied(), Some(u64::from(cfg.shots)));
    }

    /// `bitstring` never panics on a width `format!` would refuse (`u16::MAX`
    /// and above), and bits beyond the key read 0.
    #[test]
    fn bitstring_pads_any_width_without_panicking() {
        assert_eq!(bitstring(0b101, 5), "00101");
        assert_eq!(bitstring(usize::MAX, 2), "11");
        let wide = bitstring(usize::MAX, 70_000);
        assert_eq!(wide.len(), 70_000);
        assert_eq!(wide.matches('1').count(), MAX_CLBITS);
        assert!(wide.ends_with('1') && wide.starts_with('0'));
    }

    #[test]
    fn check_circuit_matches_execution() {
        let header = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n";
        let cfg = config_with(OptLevel::default());
        let backend = NativeStatevectorBackend::new(0);
        let cases = [
            (format!("{header}h q[0];\nmeasure q -> c;\n"), None),
            (
                format!("{header}reset q[0];\n"),
                Some("'reset' is not supported"),
            ),
            (
                format!("{header}measure q[0] -> c[0];\nx q[0];\n"),
                Some("after it was measured"),
            ),
            (
                format!("{header}if(c==1) x q[0];\n"),
                Some("'if' statements are not supported"),
            ),
            (
                "not openqasm".to_string(),
                Some("could not parse OpenQASM 2.0"),
            ),
        ];
        for (qasm, rejection) in cases {
            let circuit = BoundCircuit::Qasm2(qasm);
            let checked = NativeStatevectorBackend::check_circuit(&circuit);
            let run = backend.run_circuits(std::slice::from_ref(&circuit), &cfg.run_params());
            match rejection {
                None => {
                    assert!(checked.is_ok());
                    assert!(run.is_ok());
                }
                Some(fragment) => {
                    let checked = checked.expect_err("rejected by the check").to_string();
                    let run = run.expect_err("rejected by execution").to_string();
                    assert!(checked.contains(fragment), "{checked}");
                    assert_eq!(checked, run);
                }
            }
        }
        pyo3::Python::initialize();
        let foreign = pyo3::Python::attach(|py| crate::QiskitCircuit::into_bound(py.None()));
        assert!(matches!(
            NativeStatevectorBackend::check_circuit(&foreign),
            Err(BackendError::UnsupportedCircuit(_))
        ));
    }

    #[test]
    fn omitted_seed_differs_across_calls_for_same_id() {
        use crate::Infrastructure;
        let cfg = config_with(OptLevel::default()); // id "abc", seed: None
        let batch = vec![BoundCircuit::Native(uniform3())];
        let a = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&batch, &cfg.run_params())
            .unwrap();
        let b = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&batch, &cfg.run_params())
            .unwrap();
        assert_ne!(
            a, b,
            "no-seed runs with the same id must produce independent noise"
        );
    }

    /// Micro-benchmark (perf evidence for issue #83, not a correctness check —
    /// hence `#[ignore]`d and assertion-free). It isolates the exact cost the
    /// identity-transpiler fix eliminates, side by side within one binary, so
    /// there is no build-to-build noise and no reliance on the pre-fix code
    /// still existing on disk.
    ///
    /// The full-training benchmarks (`bench_native_vs_qiskit.py`,
    /// `bench_batching.py`) can't detect this fix: at 4–8 qubits through a DE
    /// loop, GIL crossings, Aer/Qiskit overhead and optimizer bookkeeping dwarf
    /// one or two `ConcreteCircuit` clones. This benchmark never calls the
    /// simulator (qubit count stays at 8); gate *count* — which drives clone and
    /// QASM cost — is what we scale.
    ///
    /// Run with:
    /// ```text
    /// LD_LIBRARY_PATH=~/miniconda3/lib cargo test -p polypus --lib \
    ///   infrastructure::native::tests::bench_identity_path_avoids_clone_and_qasm_roundtrip \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "perf micro-benchmark: prints timings, run explicitly with --ignored --nocapture"]
    fn bench_identity_path_avoids_clone_and_qasm_roundtrip() {
        use std::borrow::Cow;
        use std::hint::black_box;
        use std::time::Instant;

        /// Gate *count* (not qubit count) drives clone/QASM cost, so we keep
        /// qubits small and pile on gates.
        fn big_circuit(n_gates: usize) -> ConcreteCircuit {
            let mut pc = ParameterizedCircuit::new(8);
            for i in 0..n_gates {
                pc = pc.h(i % 8).cx(i % 8, (i + 1) % 8).rz(i % 8, 0.123);
            }
            pc.assign_parameters(&[]).unwrap()
        }

        const N: usize = 500;
        const N_GATES: usize = 20_000;
        let opts = TranspileOptions::default();
        let circuit = big_circuit(N_GATES);
        let qasm = circuit.try_to_qasm2().unwrap();

        // --- Native path: two clones (old) vs. borrow (new) ---
        let t0 = Instant::now();
        for _ in 0..N {
            // Exactly what `simulate_one` used to do: clone up front, then the
            // IdentityTranspiler clones again inside `transpile`.
            let a = circuit.clone();
            let b = IdentityTranspiler.transpile(&a, &opts);
            black_box(&b);
        }
        let native_old_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;

        let t0 = Instant::now();
        for _ in 0..N {
            // The `is_identity()` branch: borrow, zero clones.
            let c: Cow<'_, ConcreteCircuit> = if IdentityTranspiler.is_identity() {
                Cow::Borrowed(&circuit)
            } else {
                Cow::Owned(IdentityTranspiler.transpile(&circuit, &opts))
            };
            black_box(c.as_ref());
        }
        let native_new_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;

        // --- Qasm2 path: parse+transpile+re-emit (old) vs. String clone (new) ---
        let t0 = Instant::now();
        for _ in 0..N {
            // Exactly what `BoundCircuit::transpiled` used to do for Qasm2.
            let parsed = ParameterizedCircuit::from_qasm2(&qasm)
                .and_then(|pc| pc.assign_parameters(&[]))
                .unwrap();
            let out = IdentityTranspiler
                .transpile(&parsed, &opts)
                .try_to_qasm2()
                .unwrap();
            black_box(&out);
        }
        let qasm_old_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;

        let t0 = Instant::now();
        for _ in 0..N {
            // The `is_identity()` short-circuit: a cheap String clone.
            let out = qasm.clone();
            black_box(&out);
        }
        let qasm_new_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;

        // Extrapolation to the training scale used by bench_native_vs_qiskit.py
        // (pop=20, gens=10 => 200 candidate submissions per run).
        const SUBMISSIONS_PER_RUN: f64 = 200.0;
        let native_saved_ms = (native_old_us - native_new_us) * SUBMISSIONS_PER_RUN / 1e3;
        let qasm_saved_ms = (qasm_old_us - qasm_new_us) * SUBMISSIONS_PER_RUN / 1e3;

        println!(
            "\nissue #83 identity-path micro-benchmark \
             ({N_GATES} gates, 8 qubits, mean over {N} calls):"
        );
        println!(
            "  native clone+transpile (old): {native_old_us:8.2} us/call\n\
             native borrow          (new): {native_new_us:8.2} us/call\n\
             speedup                     : {:8.1}x",
            native_old_us / native_new_us
        );
        println!(
            "  qasm parse+transpile+emit (old): {qasm_old_us:8.2} us/call\n\
             qasm String clone         (new): {qasm_new_us:8.2} us/call\n\
             speedup                        : {:8.1}x",
            qasm_old_us / qasm_new_us
        );
        println!(
            "  extrapolated saved per training run ({} submissions): \
             native {native_saved_ms:.2} ms, qasm {qasm_saved_ms:.2} ms",
            SUBMISSIONS_PER_RUN as u64
        );
    }

    // ── Exact probabilities (run_circuits_exact) ──────────────────────────────

    fn exact(qcs: &[BoundCircuit]) -> Vec<HashMap<String, f64>> {
        NativeStatevectorBackend::new(0)
            .run_circuits_exact(qcs, &config_with(OptLevel::default()).run_params())
            .unwrap()
    }

    #[test]
    fn exact_bell_probabilities() {
        let probs = exact(&[BoundCircuit::Native(bell())]);
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].len(), 2, "{:?}", probs[0]);
        assert!((probs[0]["00"] - 0.5).abs() < 1e-12);
        assert!((probs[0]["11"] - 0.5).abs() < 1e-12);
    }

    /// A circuit whose eight outcomes all have different, non-trivial weights.
    fn skewed3() -> ConcreteCircuit {
        ParameterizedCircuit::new(3)
            .ry(0, 0.3)
            .ry(1, 1.1)
            .cx(0, 2)
            .ry(2, 2.2)
            .rz(1, 0.7)
            .h(1)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    #[test]
    fn exact_probabilities_sum_to_one_and_match_sampled_frequencies() {
        let circuit = BoundCircuit::Native(skewed3());
        let probs = exact(std::slice::from_ref(&circuit)).remove(0);
        assert!((probs.values().sum::<f64>() - 1.0).abs() < 1e-12);
        let shots = 400_000;
        let backend = NativeStatevectorBackend::new(11);
        let counts = backend
            .simulate_one(
                &circuit,
                shots,
                5,
                &TranspileOptions::default(),
                backend.simulator.parallel_threshold,
            )
            .unwrap();
        // Same keys as the counts (C-3), and frequencies within a few standard
        // deviations (σ ≤ 1/(2·√shots) ≈ 8e-4).
        for (key, &count) in &counts {
            let p = probs[key];
            assert!((count as f64 / shots as f64 - p).abs() < 5e-3, "{key}: {p}");
        }
        assert!(probs.keys().all(|key| key.len() == 3));
    }

    #[test]
    fn exact_results_are_deterministic_and_leave_the_seed_stream_alone() {
        let batch = vec![
            BoundCircuit::Native(skewed3()),
            BoundCircuit::Native(bell()),
        ];
        let params = config_with(OptLevel::default()).run_params();
        let backend = NativeStatevectorBackend::new(3);
        let first = backend.run_circuits_exact(&batch, &params).unwrap();
        let again = NativeStatevectorBackend::new(99)
            .run_circuits_exact(&batch, &params)
            .unwrap();
        assert_eq!(first, again);
        // Sampling after an exact run gives what it gives on a fresh backend.
        let sampled = backend.run_circuits(&batch, &params).unwrap();
        let fresh = NativeStatevectorBackend::new(3)
            .run_circuits(&batch, &params)
            .unwrap();
        assert_eq!(sampled, fresh);
    }

    #[test]
    fn exact_windows_do_not_change_the_result() {
        let batch: Vec<BoundCircuit> = (0..5)
            .map(|i| {
                BoundCircuit::Native(
                    ParameterizedCircuit::new(3)
                        .ry(0, 0.2 * i as f64)
                        .cx(0, 1)
                        .ry(2, 0.5)
                        .measure_all()
                        .assign_parameters(&[])
                        .unwrap(),
                )
            })
            .collect();
        let backend = NativeStatevectorBackend::new(0);
        let params = config_with(OptLevel::default()).run_params();
        let unbounded = backend
            .run_exact_with_cap(&batch, &params, usize::MAX)
            .unwrap();
        for cap in [1, 2, 3] {
            assert_eq!(
                backend.run_exact_with_cap(&batch, &params, cap).unwrap(),
                unbounded,
                "cap {cap}"
            );
        }
    }

    #[test]
    fn exact_measurement_free_circuit_is_read_on_every_qubit() {
        let circuit = ParameterizedCircuit::new(3)
            .x(2)
            .assign_parameters(&[])
            .unwrap();
        let probs = exact(&[BoundCircuit::Native(circuit)]);
        assert_eq!(probs[0], HashMap::from([("100".to_string(), 1.0)]));
    }

    #[test]
    fn exact_qasm2_matches_native() {
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n\
                    h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n";
        assert_eq!(
            exact(&[BoundCircuit::Qasm2(qasm.to_string())]),
            exact(&[BoundCircuit::Native(bell())])
        );
    }

    #[test]
    fn exact_rejects_a_partial_read_out() {
        let circuit = ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .measure(0, 0)
            .assign_parameters(&[])
            .unwrap();
        let err = NativeStatevectorBackend::new(0)
            .run_circuits_exact(
                &[BoundCircuit::Native(circuit)],
                &config_with(OptLevel::default()).run_params(),
            )
            .unwrap_err();
        match err {
            BackendError::UnsupportedCircuit(message) => {
                assert!(message.contains("measure_all"), "{message}")
            }
            other => panic!("expected UnsupportedCircuit, got {other:?}"),
        }
    }
}
