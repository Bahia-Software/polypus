//! Native statevector backend: runs circuits in pure Rust via `polypus-sim`,
//! without ever touching the Python interpreter or Qiskit.
//!
//! This is the local counterpart to [`LocalBackend`](crate::infrastructure::LocalBackend)
//! (Qiskit Aer). It is selected with `backend="polypus"` and consumes a
//! [`BoundCircuit::Native`] directly — no OpenQASM round-trip, no GIL — which is
//! what makes the native circuit path pay off end-to-end. An OpenQASM 2.0
//! string is also accepted (parsed in Rust); a Qiskit `QuantumCircuit` is not,
//! since reading its gates would require the interpreter.

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
pub struct NativeStatevectorBackend {
    simulator: StatevectorSimulator,
    base_seed: u64,
    counter: AtomicU64,
}

impl NativeStatevectorBackend {
    /// Create a backend whose sampling stream is derived from `id`.
    pub fn new(id: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        NativeStatevectorBackend {
            simulator: StatevectorSimulator::new(),
            base_seed: hasher.finish(),
            counter: AtomicU64::new(0),
        }
    }

    /// Run one bound circuit and return Aer-compatible bitstring counts.
    ///
    /// The bitstring width and bit order match Qiskit's: little-endian qubit
    /// indexing with the highest classical bit on the left, so the counts are
    /// interchangeable with those from the Aer backend.
    fn simulate_one(&self, circuit: &BoundCircuit, shots: u32, seed: u64) -> HashMap<String, u64> {
        // Obtain a ConcreteCircuit without touching Python.
        let concrete: ConcreteCircuit = match circuit {
            BoundCircuit::Native(cc) => cc.clone(),
            BoundCircuit::Qasm2(qasm) => ParameterizedCircuit::from_qasm2(qasm)
                .and_then(|pc| pc.assign_parameters(&[]))
                .unwrap_or_else(|e| panic!("native backend could not parse OpenQASM 2.0: {e}")),
            BoundCircuit::Qiskit(_) => panic!(
                "the native statevector backend cannot execute a Qiskit QuantumCircuit; \
                 pass a polypus.Circuit or an OpenQASM 2.0 string, or select backend=\"aer\""
            ),
        };

        let raw = self
            .simulator
            .run_and_sample(&concrete, shots as usize, seed)
            .unwrap_or_else(|e| panic!("native statevector simulation failed: {e}"));

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
        // Reserve a contiguous block of seeds for this batch so each circuit is
        // sampled independently and deterministically, regardless of order.
        let start = self.counter.fetch_add(qcs.len() as u64, Ordering::Relaxed);
        qcs.iter()
            .enumerate()
            .map(|(i, qc)| {
                let seed = self.base_seed.wrapping_add(start).wrapping_add(i as u64);
                self.simulate_one(qc, config.shots, seed)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::ParameterizedCircuit;

    fn bell() -> ConcreteCircuit {
        ParameterizedCircuit::new(2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .assign_parameters(&[])
            .unwrap()
    }

    #[test]
    fn bell_counts_only_correlated_outcomes() {
        let backend = NativeStatevectorBackend::new("test");
        let counts = backend.simulate_one(&BoundCircuit::Native(bell()), 2000, 7);
        let total: u64 = counts.values().sum();
        assert_eq!(total, 2000);
        for key in counts.keys() {
            assert!(key == "00" || key == "11", "unexpected outcome {key}");
        }
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
        let counts = backend.simulate_one(&BoundCircuit::Qasm2(qasm), 128, 1);
        // X|0> = |1>: every shot reads "1".
        assert_eq!(counts.get("1"), Some(&128));
    }

    #[test]
    fn seeding_is_reproducible_per_id() {
        let cfg = ExecutionConfig {
            id: "abc".to_string(),
            shots: 500,
            n_qpus: 1,
            infrastructure: "local".to_string(),
            backend_config: crate::infrastructure::BackendConfig::LocalNative,
        };
        let a = NativeStatevectorBackend::new(&cfg.id);
        let b = NativeStatevectorBackend::new(&cfg.id);
        let batch = vec![BoundCircuit::Native(bell())];
        assert_eq!(a.run_circuits(&batch, &cfg), b.run_circuits(&batch, &cfg));
    }
}
