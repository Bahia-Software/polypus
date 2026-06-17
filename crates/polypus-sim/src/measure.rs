//! Measurement read-outs: probabilities, Pauli-Z expectation values, and
//! shot sampling. These live in a separate `impl` block to keep
//! [`statevector`](crate::statevector) focused on evolution.

use crate::rng::SplitMix64;
use crate::statevector::Statevector;
use std::collections::HashMap;

impl Statevector {
    /// Probability of every computational basis state, `|amplitude|²`, indexed
    /// like [`amplitudes`](Self::amplitudes). Sums to 1 up to rounding.
    pub fn probabilities(&self) -> Vec<f64> {
        self.data.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Probability of a single basis state, or `0.0` if the index is out of
    /// range.
    pub fn probability(&self, basis_state: usize) -> f64 {
        self.data.get(basis_state).map_or(0.0, |a| a.norm_sqr())
    }

    /// Expectation value `⟨Z_{q0} Z_{q1} … ⟩` of a Pauli-Z string on the given
    /// qubits. Each basis state contributes `±|amp|²` with the sign set by the
    /// parity of the selected bits. An empty list returns the total
    /// probability (`1` for a normalized state).
    pub fn expectation_z(&self, qubits: &[usize]) -> f64 {
        let mut mask = 0usize;
        for &q in qubits {
            mask |= 1usize << q;
        }
        let mut acc = 0.0;
        for (i, amp) in self.data.iter().enumerate() {
            let parity = (i & mask).count_ones() & 1;
            let sign = if parity == 0 { 1.0 } else { -1.0 };
            acc += sign * amp.norm_sqr();
        }
        acc
    }

    /// Draw `shots` measurements of all qubits, returning a map from basis
    /// state to how many times it was observed. Deterministic for a given
    /// `rng` seed.
    ///
    /// Sampling uses the inverse-CDF method: one prefix-sum pass over the
    /// probabilities, then a binary search per shot.
    pub fn sample(&self, shots: usize, rng: &mut SplitMix64) -> HashMap<usize, u64> {
        let mut counts = HashMap::new();
        if shots == 0 || self.data.is_empty() {
            return counts;
        }

        // Cumulative distribution over basis states.
        let mut cdf = Vec::with_capacity(self.data.len());
        let mut acc = 0.0;
        for amp in &self.data {
            acc += amp.norm_sqr();
            cdf.push(acc);
        }
        let total = acc; // ≈ 1.0; used to absorb normalization drift.
        let last = self.data.len() - 1;

        for _ in 0..shots {
            let r = rng.next_f64() * total;
            // First index whose cumulative probability is ≥ r.
            let idx = cdf.partition_point(|&c| c < r).min(last);
            *counts.entry(idx).or_insert(0) += 1;
        }
        counts
    }
}
