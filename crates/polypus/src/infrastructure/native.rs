//! Native statevector backend: runs circuits in pure Rust via `polypus-sim`,
//! without ever touching the Python interpreter or Qiskit.
//!
//! This is the local counterpart to [`LocalBackend`](crate::infrastructure::LocalBackend)
//! (Qiskit Aer). It is selected with `backend="polypus"` and consumes a
//! [`BoundCircuit::Native`] directly — no OpenQASM round-trip, no GIL — which is
//! what makes the native circuit path pay off end-to-end. An OpenQASM 2.0
//! string is also accepted (parsed in Rust); a Qiskit `QuantumCircuit` is not,
//! since reading its gates would require the interpreter.

use crate::infrastructure::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use polypus_circuit::{ConcreteCircuit, ParameterizedCircuit};
use polypus_sim::StatevectorSimulator;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Local, noiseless statevector backend backed by `polypus-sim`.
///
/// Sampling is seeded deterministically from the run `id` plus a per-circuit
/// counter, so repeated runs with the same `id` reproduce the same counts while
/// distinct circuits in a batch still get independent shot noise.
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
    counter: AtomicU64,
}

impl NativeStatevectorBackend {
    /// Create a backend whose sampling stream is derived from `id`, using the
    /// no-op [`IdentityTranspiler`] (behavior identical to having no transpiler).
    pub fn new(id: &str) -> Self {
        Self::with_transpiler(id, Box::new(IdentityTranspiler))
    }

    /// Create a backend with a custom transpilation *strategy* injected by
    /// composition. The sampling stream is still derived from `id`.
    ///
    /// This is the extension point for hardware-aware rewriting: pass any
    /// `Box<dyn Transpiler>` without changing the backend or any algorithm code.
    pub fn with_transpiler(id: &str, transpiler: Box<dyn Transpiler>) -> Self {
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        NativeStatevectorBackend {
            simulator: StatevectorSimulator::new(),
            transpiler,
            base_seed: hasher.finish(),
            counter: AtomicU64::new(0),
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
    ) -> HashMap<String, u64> {
        // Obtain a ConcreteCircuit without touching Python.
        let concrete: ConcreteCircuit = match circuit {
            BoundCircuit::Native(cc) => cc.clone(),
            BoundCircuit::Qasm2(qasm) => ParameterizedCircuit::from_qasm2(qasm)
                .and_then(|pc| pc.assign_parameters(&[]))
                .unwrap_or_else(|e| {
                    log::error!("native backend could not parse OpenQASM 2.0: {e}");
                    panic!("native backend could not parse OpenQASM 2.0: {e}");
                }),
            BoundCircuit::Qiskit(_) => panic!(
                "the native statevector backend cannot execute a Qiskit QuantumCircuit; \
                 pass a polypus.Circuit or an OpenQASM 2.0 string, or select backend=\"aer\""
            ),
        };

        // Transpile the native circuit (GIL-free) before simulating. With the
        // default IdentityTranspiler this is a clone and changes nothing.
        let concrete = self.transpiler.transpile(&concrete, opts);

        let raw = self
            .simulator
            .run_and_sample(&concrete, shots as usize, seed)
            .unwrap_or_else(|e| {
                log::error!("native statevector simulation failed: {e}");
                panic!("native statevector simulation failed: {e}");
            });

        // Bitstring length = classical register width (qubit count when the
        // circuit has no measurements, mirroring a full-register read-out).
        let width = match concrete.num_clbits() {
            0 => concrete.num_qubits,
            c => c,
        };
        raw.into_iter()
            .map(|(state, count)| (format!("{:0w$b}", state, w = width), count))
            .collect()
    }
}

impl QuantumBackend for NativeStatevectorBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Vec<HashMap<String, u64>> {
        // Tuning travels as an argument; the strategy is the injected field.
        let opts = TranspileOptions {
            level: config.opt_level,
        };
        // Reserve a contiguous block of seeds for this batch so each circuit is
        // sampled independently and deterministically, regardless of order.
        let start = self.counter.fetch_add(qcs.len() as u64, Ordering::Relaxed);
        qcs.iter()
            .enumerate()
            .map(|(i, qc)| {
                let seed = self.base_seed.wrapping_add(start).wrapping_add(i as u64);
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
            backend_config: crate::infrastructure::BackendConfig::LocalNative,
            opt_level,
        }
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
        let backend = NativeStatevectorBackend::new("test");
        let counts = backend.simulate_one(
            &BoundCircuit::Native(bell()),
            2000,
            7,
            &TranspileOptions::default(),
        );
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
        let default_backend = NativeStatevectorBackend::new("bell");
        let explicit_backend =
            NativeStatevectorBackend::with_transpiler("bell", Box::new(IdentityTranspiler));
        let circuit = BoundCircuit::Native(bell());
        assert_eq!(
            default_backend.simulate_one(&circuit, 1000, 42, &opts),
            explicit_backend.simulate_one(&circuit, 1000, 42, &opts),
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
        let backend = NativeStatevectorBackend::new("t");
        let counts = backend.simulate_one(
            &BoundCircuit::Qasm2(qasm),
            128,
            1,
            &TranspileOptions::default(),
        );
        // X|0> = |1>: every shot reads "1".
        assert_eq!(counts.get("1"), Some(&128));
    }

    /// The Qasm2 path (parsed in Rust) and the Native path yield identical counts
    /// for the same circuit and seed under the identity transpiler.
    #[test]
    fn qasm2_path_matches_native_path() {
        let backend = NativeStatevectorBackend::new("eq");
        let opts = TranspileOptions::default();
        let native = backend.simulate_one(&BoundCircuit::Native(bell()), 1000, 5, &opts);
        let qasm = backend.simulate_one(&BoundCircuit::Qasm2(bell().to_qasm2()), 1000, 5, &opts);
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
            "lvl",
            Box::new(RecordingTranspiler {
                seen: Arc::clone(&seen),
            }),
        );
        let cfg = config_with(OptLevel::Heavy);
        backend.run_circuits(&[BoundCircuit::Native(bell())], &cfg);
        assert_eq!(seen.load(Ordering::SeqCst), OptLevel::Heavy as u8);
    }

    /// End-to-end: an injected non-identity strategy is actually applied inside
    /// `run_circuits`. The appended `barrier` is a simulation no-op, so the
    /// counts remain the correlated Bell outcomes — the rewrite is exercised via
    /// the injection point, not observed as wrong results.
    #[test]
    fn injected_strategy_runs_through_run_circuits() {
        let backend = NativeStatevectorBackend::with_transpiler("inj", Box::new(BarrierTranspiler));
        let cfg = config_with(OptLevel::default());
        let counts = backend.run_circuits(&[BoundCircuit::Native(bell())], &cfg);
        assert_eq!(counts.len(), 1);
        let total: u64 = counts[0].values().sum();
        assert_eq!(total, u64::from(cfg.shots));
        for key in counts[0].keys() {
            assert!(key == "00" || key == "11", "unexpected outcome {key}");
        }
    }

    #[test]
    fn seeding_is_reproducible_per_id() {
        let cfg = config_with(OptLevel::default());
        let a = NativeStatevectorBackend::new(&cfg.id);
        let b = NativeStatevectorBackend::new(&cfg.id);
        let batch = vec![BoundCircuit::Native(bell())];
        assert_eq!(a.run_circuits(&batch, &cfg), b.run_circuits(&batch, &cfg));
    }
}
