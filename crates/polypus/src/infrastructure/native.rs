//! Native statevector backend: runs circuits in pure Rust via `polypus-sim`,
//! without ever touching the Python interpreter or Qiskit.
//!
//! This is the local counterpart to [`LocalBackend`](crate::infrastructure::LocalBackend)
//! (Qiskit Aer). It is selected with `backend="polypus"` and consumes a
//! [`BoundCircuit::Native`] directly — no OpenQASM round-trip, no GIL — which is
//! what makes the native circuit path pay off end-to-end. An OpenQASM 2.0
//! string is also accepted (parsed in Rust); a Qiskit `QuantumCircuit` is not,
//! since reading its gates would require the interpreter.

use crate::infrastructure::error::BackendError;
use crate::infrastructure::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::{ConcreteCircuit, ParameterizedCircuit};
use polypus_sim::{Simulator, StatevectorSimulator};
use std::borrow::Cow;
use std::collections::HashMap;

/// Local, noiseless statevector backend backed by `polypus-sim`.
///
/// Sampling is seeded from an explicit base seed plus, per circuit, a hash of
/// the circuit's *own content* (its OpenQASM 2.0 text) and its position within
/// the submitted batch — never from any mutable state shared between calls. Two
/// backends built with the same seed reproduce the same counts, while distinct
/// circuits in a batch (e.g. the same ansatz bound to different `θ`) still get
/// independent shot noise because their content differs. Crucially, this makes
/// [`run_circuits`](QuantumBackend::run_circuits) **safe to call concurrently on
/// a shared instance**: each circuit's seed is a pure function of its content and
/// its index, so overlapping calls can never race for seed assignment (the bug
/// that a previous shared atomic counter had). The seed is supplied by the caller
/// (`ExecutionConfig::seed`, resolved at the Python-facing boundary from a user
/// value or an OS-entropy draw) and is **fully decoupled from the run `id`**: an
/// omitted seed yields genuine, independent noise across runs instead of the
/// `id`-derived repetition that was previously mistaken for shot noise.
///
/// The backend *composes* a [`Transpiler`] (the rewriting *strategy*) and runs
/// it on every native circuit before simulating, passing the per-run
/// [`TranspileOptions`] (the *tuning*) derived from the [`ExecutionConfig`].
/// It defaults to the no-op [`IdentityTranspiler`]; inject another strategy with
/// [`with_transpiler`](Self::with_transpiler).
pub struct NativeStatevectorBackend {
    simulator: StatevectorSimulator,
    transpiler: Box<dyn Transpiler>,
    base_seed: u64,
}

/// FNV-1a hash of `bytes`, used **only** to derive a per-circuit sampling seed
/// from the circuit's own content — it is neither a general-purpose PRNG nor a
/// hash-map hasher.
///
/// Written out by hand (offset basis `0xcbf29ce484222325`, prime
/// `0x100000001b3`, XOR-then-multiply per byte) rather than reaching for
/// `std::hash::DefaultHasher` because the standard library does **not** guarantee
/// `DefaultHasher` yields the same value across compiler versions — the very
/// reproducibility hazard that led `polypus-qml` to ship its own `SplitMix64`
/// instead of `rand` (crate decision D6). FNV-1a is a fixed, public-domain
/// algorithm, so the seeds derived here stay byte-stable forever.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The content key a circuit's seed is derived from: its own textual form, so
/// the seed depends only on the circuit itself and never on shared state.
///
/// A [`BoundCircuit::Qiskit`] never reaches here — [`simulate_one`] rejects that
/// variant before a seed is ever needed — so its empty key is unreachable in
/// practice and only present to keep the match total.
fn seed_content_key(circuit: &BoundCircuit) -> String {
    match circuit {
        BoundCircuit::Native(cc) => cc.to_qasm2(),
        BoundCircuit::Qasm2(s) => s.clone(),
        BoundCircuit::Qiskit(_) => String::new(),
    }
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
        }
    }

    /// Resolve a [`BoundCircuit`] to a transpiled [`ConcreteCircuit`] and the
    /// bitstring width its read-out uses, without touching Python.
    ///
    /// This is the shared front half of both execution paths — sampled
    /// ([`simulate_one`](Self::simulate_one)) and exact
    /// ([`simulate_one_exact`](Self::simulate_one_exact)) — factored out so the
    /// two paths cannot drift on circuit resolution, transpilation or width:
    ///
    /// 1. turn the variant into a [`ConcreteCircuit`] (a `Qiskit` circuit is
    ///    rejected — reading its gates would require the interpreter);
    /// 2. run the injected [`Transpiler`] with the per-run [`TranspileOptions`],
    ///    **unless** it is a guaranteed no-op ([`Transpiler::is_identity`]) — in
    ///    which case the circuit is borrowed straight through, avoiding the
    ///    identity clone entirely (the default [`IdentityTranspiler`] path);
    /// 3. compute the read-out width — the classical-register width, falling
    ///    back to the qubit count when the circuit has no measurements
    ///    (full-register read-out convention, matching C-3).
    ///
    /// The result is a [`Cow`]: `Borrowed` for a native circuit under an identity
    /// transpiler (zero copies), `Owned` when the circuit had to be parsed from
    /// OpenQASM or genuinely rewritten by a non-identity strategy.
    ///
    /// The width is computed **after** transpiling, so a strategy that changes
    /// the register width is reflected.
    fn resolve_and_transpile<'a>(
        &self,
        circuit: &'a BoundCircuit,
        opts: &TranspileOptions,
    ) -> Result<(Cow<'a, ConcreteCircuit>, usize), BackendError> {
        // Obtain a ConcreteCircuit without touching Python. A native circuit is
        // borrowed; an OpenQASM string must be parsed into an owned circuit.
        let source: Cow<'a, ConcreteCircuit> = match circuit {
            BoundCircuit::Native(cc) => Cow::Borrowed(cc),
            BoundCircuit::Qasm2(qasm) => Cow::Owned(
                ParameterizedCircuit::from_qasm2(qasm)
                    .and_then(|pc| pc.assign_parameters(&[]))
                    .map_err(|e| {
                        log::error!("native backend could not parse OpenQASM 2.0: {e}");
                        BackendError::NativeCircuit(format!(
                            "native backend could not parse OpenQASM 2.0: {e}"
                        ))
                    })?,
            ),
            BoundCircuit::Qiskit(_) => {
                return Err(BackendError::UnsupportedCircuit(
                    "the native statevector backend cannot execute a Qiskit QuantumCircuit; \
                     pass a polypus.Circuit or an OpenQASM 2.0 string, or select backend=\"aer\""
                        .to_string(),
                ))
            }
        };

        // Transpile the native circuit (GIL-free) before simulating — but only
        // when the strategy actually rewrites. Under the default
        // IdentityTranspiler this would be an identity clone, so borrow straight
        // through instead of cloning.
        let concrete: Cow<'a, ConcreteCircuit> = if self.transpiler.is_identity() {
            source
        } else {
            Cow::Owned(self.transpiler.transpile(source.as_ref(), opts))
        };

        // Bitstring length = classical register width (qubit count when the
        // circuit has no measurements, mirroring a full-register read-out).
        let width = match concrete.num_clbits() {
            0 => concrete.num_qubits,
            c => c,
        };
        Ok((concrete, width))
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
    ) -> Result<HashMap<String, u64>, BackendError> {
        let (concrete, width) = self.resolve_and_transpile(circuit, opts)?;

        let raw = self
            .simulator
            .run_and_sample(concrete.as_ref(), shots as usize, seed)
            .map_err(|e| {
                log::error!("native statevector simulation failed: {e}");
                BackendError::NativeCircuit(format!("native statevector simulation failed: {e}"))
            })?;

        Ok(raw
            .into_iter()
            .map(|(state, count)| (format!("{:0w$b}", state, w = width), count))
            .collect())
    }

    /// Run one bound circuit and return the **exact** probability of every
    /// computational basis state, `|amplitude|²` — never sampled, so no RNG and
    /// no shot noise are involved.
    ///
    /// The returned map is keyed by the same Qiskit little-endian bitstrings as
    /// [`simulate_one`](Self::simulate_one)'s counts, so the two are
    /// interchangeable everywhere except the value type (`f64` probability vs.
    /// `u64` count). Because the state is read exactly, two calls on the same
    /// circuit return byte-identical maps.
    ///
    /// # Errors
    ///
    /// - [`BackendError::UnsupportedCircuit`] for a **partial** measurement
    ///   (`0 < num_clbits < num_qubits`): reading a strict subset of the
    ///   register would require marginalising the exact distribution over the
    ///   unmeasured qubits, which is out of scope here. Only full-register
    ///   read-out (no measurements, or `MeasureAll` — the sole shape
    ///   `polypus-qml` emits) is supported.
    /// - [`BackendError::NativeCircuit`] if the statevector simulation fails.
    fn simulate_one_exact(
        &self,
        circuit: &BoundCircuit,
        opts: &TranspileOptions,
    ) -> Result<HashMap<String, f64>, BackendError> {
        let (concrete, width) = self.resolve_and_transpile(circuit, opts)?;
        // Precondition: the exact mode only supports a full-register read-out
        // (0 classical bits = read the whole state, or `num_clbits ==
        // num_qubits` = MeasureAll — the only case polypus-qml generates). A
        // PARTIAL measurement (0 < num_clbits < num_qubits) would require
        // marginalising the exact distribution over the unmeasured qubits, which
        // is not implemented here.
        let num_clbits = concrete.num_clbits();
        if num_clbits != 0 && num_clbits != concrete.num_qubits {
            return Err(BackendError::UnsupportedCircuit(format!(
                "exact mode only supports full-register read-out (0 or {} classical bits), \
                 got {num_clbits}",
                concrete.num_qubits
            )));
        }
        let sv = self.simulator.run(concrete.as_ref()).map_err(|e| {
            log::error!("native statevector simulation failed: {e}");
            BackendError::NativeCircuit(format!("native statevector simulation failed: {e}"))
        })?;
        let probs = sv.probabilities();
        Ok(probs
            .into_iter()
            .enumerate()
            .map(|(state, p)| (format!("{:0w$b}", state, w = width), p))
            .collect())
    }

    /// Run a batch of bound circuits and return the **exact** basis-state
    /// probabilities of each, one map per circuit in submission order.
    ///
    /// This is the exact counterpart of the sampled
    /// [`run_circuits`](QuantumBackend::run_circuits), but is an **inherent**
    /// method — deliberately *not* part of the [`QuantumBackend`] trait, because
    /// "exact" has no physical meaning for a noisy Aer backend or for real
    /// hardware (QMIO/CUNQA); it is exclusive to this pure-statevector backend
    /// and is reached only through the native `qml.train` exact path.
    ///
    /// `config.shots` and `config.seed` are **ignored**: there is no sampling,
    /// so there is nothing to seed and no shot budget to spend. Only
    /// [`ExecutionConfig::opt_level`] is read, to build the [`TranspileOptions`].
    pub fn run_circuits_exact(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Result<Vec<HashMap<String, f64>>, BackendError> {
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        qcs.iter()
            .map(|qc| self.simulate_one_exact(qc, &opts))
            .collect()
    }
}

impl QuantumBackend for NativeStatevectorBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
        // Tuning travels as an argument; the strategy is the injected field.
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        // Derive each circuit's seed from its own content plus its batch
        // position — no shared mutable state — so distinct circuits sample
        // independently and deterministically while concurrent calls on this
        // shared instance can never race for seed assignment.
        qcs.iter()
            .enumerate()
            .map(|(i, qc)| {
                let content = seed_content_key(qc);
                let seed = self
                    .base_seed
                    .wrapping_add(fnv1a(content.as_bytes()))
                    .wrapping_add(i as u64);
                self.simulate_one(qc, config.shots, seed, &opts)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::OptLevel;
    use polypus_circuit::{GateInstruction, ParameterizedCircuit};
    use std::sync::atomic::{AtomicU8, Ordering};
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
            backend_config: crate::infrastructure::BackendConfig::LocalNative,
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
            let mut out = circuit.clone();
            out.gates.push(GateInstruction::Barrier(Vec::new()));
            out
        }
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
                .simulate_one(&circuit, 1000, 42, &opts)
                .unwrap(),
            explicit_backend
                .simulate_one(&circuit, 1000, 42, &opts)
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
            .to_qasm2();
        let backend = NativeStatevectorBackend::new(0);
        let counts = backend
            .simulate_one(
                &BoundCircuit::Qasm2(qasm),
                128,
                1,
                &TranspileOptions::default(),
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
            .simulate_one(&BoundCircuit::Native(bell()), 1000, 5, &opts)
            .unwrap();
        let qasm = backend
            .simulate_one(&BoundCircuit::Qasm2(bell().to_qasm2()), 1000, 5, &opts)
            .unwrap();
        assert_eq!(native, qasm);
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
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg)
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
            .run_circuits(&[BoundCircuit::Native(bell())], &cfg)
            .unwrap();
        assert_eq!(counts.len(), 1);
        let total: u64 = counts[0].values().sum();
        assert_eq!(total, u64::from(cfg.shots));
        for key in counts[0].keys() {
            assert!(key == "00" || key == "11", "unexpected outcome {key}");
        }
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
            a.run_circuits(&batch, &cfg).unwrap(),
            b.run_circuits(&batch, &cfg).unwrap()
        );
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
            a.run_circuits(&batch, &cfg).unwrap(),
            b.run_circuits(&batch, &cfg).unwrap()
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
    #[test]
    fn omitted_seed_differs_across_calls_for_same_id() {
        use crate::infrastructure::Infrastructure;
        let cfg = config_with(OptLevel::default()); // id "abc", seed: None
        let batch = vec![BoundCircuit::Native(uniform3())];
        let a = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&batch, &cfg)
            .unwrap();
        let b = Infrastructure::create_backend(&cfg)
            .unwrap()
            .run_circuits(&batch, &cfg)
            .unwrap();
        assert_ne!(
            a, b,
            "no-seed runs with the same id must produce independent noise"
        );
    }

    /// Regression for the seed-assignment race (root cause of the phase-4
    /// emergency sequential fallback): many threads call `run_circuits`
    /// **concurrently on the same backend instance**, each with a distinct
    /// circuit (an `Ry` at a thread-specific angle, standing in for distinct
    /// optimizer candidates). Because every circuit's seed is derived purely
    /// from its own content plus its batch index — with no shared mutable state
    /// — the full set of results must be byte-identical across two independent
    /// runs at the same base seed, no matter how the OS scheduled the threads.
    /// With the old shared atomic counter, the seed each circuit received
    /// depended on the (racy) arrival order, so the two runs would diverge.
    #[test]
    fn concurrent_run_circuits_on_shared_instance_is_reproducible() {
        fn candidate(angle: f64) -> BoundCircuit {
            BoundCircuit::Native(
                ParameterizedCircuit::new(1)
                    .ry(0, angle)
                    .measure_all()
                    .assign_parameters(&[])
                    .unwrap(),
            )
        }

        // Each element is one thread's distinct candidate circuit.
        let angles: Vec<f64> = (0..8).map(|k| 0.1 + 0.37 * k as f64).collect();

        // Run every candidate concurrently on one shared backend, collecting
        // (angle-index, counts) so the outcome is order-independent.
        let run_once = || -> Vec<(usize, HashMap<String, u64>)> {
            let backend = NativeStatevectorBackend::new(2024);
            let cfg = config_with(OptLevel::default());
            std::thread::scope(|scope| {
                let handles: Vec<_> = angles
                    .iter()
                    .enumerate()
                    .map(|(i, &angle)| {
                        let backend = &backend;
                        let cfg = &cfg;
                        scope.spawn(move || {
                            let counts = backend
                                .run_circuits(&[candidate(angle)], cfg)
                                .unwrap()
                                .pop()
                                .unwrap();
                            (i, counts)
                        })
                    })
                    .collect();
                let mut out: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                out.sort_by_key(|(i, _)| *i);
                out
            })
        };

        assert_eq!(
            run_once(),
            run_once(),
            "concurrent runs on a shared instance must be byte-identical despite thread scheduling"
        );
    }

    /// Exact mode over a Bell state returns exactly `{"00": 0.5, "11": 0.5}` —
    /// no statistical tolerance, because the probabilities are read from the
    /// statevector, not sampled.
    #[test]
    fn run_circuits_exact_bell_is_exact() {
        let backend = NativeStatevectorBackend::new(0);
        let cfg = config_with(OptLevel::default());
        let mut probs = backend
            .run_circuits_exact(&[BoundCircuit::Native(bell())], &cfg)
            .unwrap();
        assert_eq!(probs.len(), 1);
        let probs = probs.pop().unwrap();
        // Only the two correlated outcomes carry weight, each exactly 1/2.
        assert!((probs.get("00").copied().unwrap_or(0.0) - 0.5).abs() < 1e-12);
        assert!((probs.get("11").copied().unwrap_or(0.0) - 0.5).abs() < 1e-12);
        assert!((probs.get("01").copied().unwrap_or(0.0)).abs() < 1e-12);
        assert!((probs.get("10").copied().unwrap_or(0.0)).abs() < 1e-12);
    }

    /// A partial measurement (fewer `measure` than qubits) is rejected in exact
    /// mode with [`BackendError::UnsupportedCircuit`]: marginalising the exact
    /// distribution over the unmeasured qubits is out of scope.
    #[test]
    fn run_circuits_exact_rejects_partial_measurement() {
        // 3 qubits, but only qubit 0 is measured → num_clbits() == 1.
        let partial = ParameterizedCircuit::new(3)
            .h(0)
            .measure(0, 0)
            .assign_parameters(&[])
            .unwrap();
        let backend = NativeStatevectorBackend::new(0);
        let cfg = config_with(OptLevel::default());
        let err = backend
            .run_circuits_exact(&[BoundCircuit::Native(partial)], &cfg)
            .unwrap_err();
        assert!(
            matches!(err, BackendError::UnsupportedCircuit(_)),
            "expected UnsupportedCircuit, got {err:?}"
        );
    }

    /// Determinism: two calls to `run_circuits_exact` on the same circuit return
    /// byte-identical maps. There is no RNG on this path, so this must hold
    /// trivially — but it is verified, not assumed.
    #[test]
    fn run_circuits_exact_is_bit_identical_across_calls() {
        let backend = NativeStatevectorBackend::new(0);
        let cfg = config_with(OptLevel::default());
        let batch = vec![BoundCircuit::Native(uniform3())];
        assert_eq!(
            backend.run_circuits_exact(&batch, &cfg).unwrap(),
            backend.run_circuits_exact(&batch, &cfg).unwrap()
        );
    }
}
