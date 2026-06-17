//! The runnable simulator: turns a [`ConcreteCircuit`] into a [`Statevector`]
//! (and, optionally, sampled measurement counts).

use crate::error::SimError;
use crate::rng::SplitMix64;
use crate::statevector::Statevector;
use polypus_circuit::{ConcreteCircuit, GateInstruction};
use std::collections::HashMap;

/// Abstraction over simulation backends. A future density-matrix or GPU
/// backend can implement this same contract.
pub trait Simulator {
    /// Evolve `|0…0⟩` through `circuit` and return the final state.
    fn run(&self, circuit: &ConcreteCircuit) -> Result<Statevector, SimError>;
}

/// Dense statevector backend.
///
/// Cheap to construct and clone; holds only configuration. Defaults to
/// [`MAX_QUBITS`](crate::MAX_QUBITS) and the crate's parallel threshold.
#[derive(Debug, Clone)]
pub struct StatevectorSimulator {
    /// Reject circuits needing more than this many qubits.
    pub max_qubits: usize,
    /// Qubit count at or above which gates use the parallel kernels (only with
    /// the `parallel` feature).
    pub parallel_threshold: usize,
}

impl Default for StatevectorSimulator {
    fn default() -> Self {
        StatevectorSimulator {
            max_qubits: crate::MAX_QUBITS,
            parallel_threshold: crate::DEFAULT_PARALLEL_THRESHOLD,
        }
    }
}

impl StatevectorSimulator {
    /// A simulator with default limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `circuit`, then draw `shots` measurements seeded by `seed`.
    ///
    /// Keys of the returned map are classical-register values. Qubits are
    /// mapped to classical bits by the circuit's `Measure`/`MeasureAll`
    /// instructions; if the circuit measures nothing, every qubit is reported
    /// (key = full basis state), matching the "measure all" convention.
    ///
    /// # Errors
    ///
    /// Propagates any [`SimError`] from [`run`](Self::run).
    pub fn run_and_sample(
        &self,
        circuit: &ConcreteCircuit,
        shots: usize,
        seed: u64,
    ) -> Result<HashMap<usize, u64>, SimError> {
        let sv = self.run(circuit)?;
        let mut rng = SplitMix64::new(seed);
        let raw = sv.sample(shots, &mut rng);

        // Collect the qubit → classical-bit mapping declared by the circuit.
        let mut measured: Vec<(usize, usize)> = Vec::new();
        let mut measure_all = false;
        for gate in &circuit.gates {
            match gate {
                GateInstruction::Measure { qubit, cbit } => measured.push((*qubit, *cbit)),
                GateInstruction::MeasureAll => measure_all = true,
                _ => {}
            }
        }

        // No measurements: report the full basis state directly.
        if !measure_all && measured.is_empty() {
            return Ok(raw);
        }
        if measure_all {
            for q in 0..sv.num_qubits() {
                measured.push((q, q));
            }
        }

        // Project each sampled basis state onto the classical register.
        let mut counts = HashMap::new();
        for (state, c) in raw {
            let mut key = 0usize;
            for &(qubit, cbit) in &measured {
                if (state >> qubit) & 1 == 1 {
                    key |= 1usize << cbit;
                }
            }
            *counts.entry(key).or_insert(0) += c;
        }
        Ok(counts)
    }
}

impl Simulator for StatevectorSimulator {
    fn run(&self, circuit: &ConcreteCircuit) -> Result<Statevector, SimError> {
        if circuit.num_qubits > self.max_qubits {
            return Err(SimError::TooManyQubits {
                requested: circuit.num_qubits,
                max: self.max_qubits,
            });
        }
        let mut sv = Statevector::new(circuit.num_qubits)?;
        sv.set_parallel_threshold(self.parallel_threshold);
        for gate in &circuit.gates {
            sv.apply(gate)?;
        }
        Ok(sv)
    }
}
