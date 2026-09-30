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
    /// Sampling is inverse-CDF: shot `k` draws `r_k = rng.next_f64() · total`
    /// (with `total` the summed probability, absorbing normalization drift) and
    /// lands on the first basis state whose cumulative probability is `≥ r_k`.
    /// Two equivalent strategies implement it, and the one with the **smaller
    /// extra memory** is chosen (issue #215):
    ///
    /// - `shots < 2^n` (the usual case at high `n`): the `r_k` are drawn, sorted,
    ///   and matched against the cumulative sum in a second sequential sweep of
    ///   the amplitudes — `O(shots)` extra memory, instead of a `2^n`-element CDF
    ///   that, at 30 qubits, was 8 GiB next to the 16 GiB statevector.
    /// - `shots ≥ 2^n`: the classical prefix-sum CDF plus a binary search per shot,
    ///   whose `2^n`-element buffer is then the smaller one.
    ///
    /// Both yield **identical counts** for the same `rng` state: the cumulative
    /// sums are accumulated in the same sequential order (so they are the same
    /// `f64`s), the `r_k` are drawn in the same order (the RNG stream is unchanged,
    /// contract C-7), and each `r_k` gets the same index. `rng` is advanced by
    /// exactly `shots` draws either way.
    pub fn sample(&self, shots: usize, rng: &mut SplitMix64) -> HashMap<usize, u64> {
        if shots == 0 || self.data.is_empty() {
            return HashMap::new();
        }
        if shots >= self.data.len() {
            self.sample_with_cdf(shots, rng)
        } else {
            self.sample_sorted(shots, rng)
        }
    }

    /// `total` = the summed probability, accumulated in index order — the same
    /// order (and hence the same rounding) as every cumulative sum below.
    fn total_probability(&self) -> f64 {
        let mut acc = 0.0;
        for amp in &self.data {
            acc += amp.norm_sqr();
        }
        acc
    }

    /// [`sample`](Self::sample)'s `shots ≥ 2^n` strategy: materialise the CDF and
    /// binary-search it once per shot. Extra memory: `8 · 2^n` bytes.
    fn sample_with_cdf(&self, shots: usize, rng: &mut SplitMix64) -> HashMap<usize, u64> {
        let mut cdf = Vec::with_capacity(self.data.len());
        let mut acc = 0.0;
        for amp in &self.data {
            acc += amp.norm_sqr();
            cdf.push(acc);
        }
        let total = acc;
        let last = self.data.len() - 1;

        let mut counts = HashMap::new();
        for _ in 0..shots {
            let r = rng.next_f64() * total;
            // First index whose cumulative probability is ≥ r.
            let idx = cdf.partition_point(|&c| c < r).min(last);
            *counts.entry(idx).or_insert(0) += 1;
        }
        counts
    }

    /// [`sample`](Self::sample)'s `shots < 2^n` strategy: draw every `r` first, in
    /// shot order, sort them, and assign them in one forward sweep that rebuilds
    /// each cumulative sum on the fly. Extra memory: `8 · shots` bytes.
    ///
    /// Equivalent to [`sample_with_cdf`](Self::sample_with_cdf) because the
    /// cumulative sums `c_0 ≤ c_1 ≤ …` are the very same `f64`s (same sequential
    /// accumulation), and `partition_point(|c| c < r)` on that monotone sequence is
    /// the number of leading `c_i < r` — a count that only grows with `r`. So one
    /// cursor walking the sums while `c_i < r`, over the `r` in ascending order,
    /// lands every `r` on the index the binary search would; an `r` past the end
    /// (rounding drift) is clamped to the last index, as there.
    fn sample_sorted(&self, shots: usize, rng: &mut SplitMix64) -> HashMap<usize, u64> {
        let total = self.total_probability();
        let last = self.data.len() - 1;

        let mut draws: Vec<f64> = (0..shots).map(|_| rng.next_f64() * total).collect();
        // A NaN draw (a NaN or infinite `total`) fails `c < r` for every `c`, so
        // the binary search puts it at index 0; `total_cmp` sorts NaN to the end,
        // where the sweep below routes it the same way.
        draws.sort_unstable_by(f64::total_cmp);

        let mut counts = HashMap::new();
        let mut amps = self.data.iter();
        let mut idx = 0usize;
        // Cumulative probability through index `idx`, once `idx` has been reached.
        let mut acc = match amps.next() {
            Some(first) => 0.0 + first.norm_sqr(),
            None => return counts,
        };
        for r in draws {
            if r.is_nan() {
                *counts.entry(0).or_insert(0) += 1;
                continue;
            }
            while acc < r && idx < last {
                match amps.next() {
                    Some(amp) => {
                        acc += amp.norm_sqr();
                        idx += 1;
                    }
                    None => break,
                }
            }
            // `acc ≥ r`, or `idx == last` (every sum < r: clamped as in the CDF path).
            *counts.entry(idx).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::C64;

    /// The pre-#215 implementation, kept verbatim as the oracle: a `2^n`-element
    /// CDF and a binary search per shot.
    fn reference_sample(
        sv: &Statevector,
        shots: usize,
        rng: &mut SplitMix64,
    ) -> HashMap<usize, u64> {
        let mut counts = HashMap::new();
        if shots == 0 || sv.data.is_empty() {
            return counts;
        }
        let mut cdf = Vec::with_capacity(sv.data.len());
        let mut acc = 0.0;
        for amp in &sv.data {
            acc += amp.norm_sqr();
            cdf.push(acc);
        }
        let total = acc;
        let last = sv.data.len() - 1;
        for _ in 0..shots {
            let r = rng.next_f64() * total;
            let idx = cdf.partition_point(|&c| c < r).min(last);
            *counts.entry(idx).or_insert(0) += 1;
        }
        counts
    }

    /// A statevector with exactly these amplitudes (normalized or not).
    fn state(amplitudes: Vec<C64>) -> Statevector {
        let n = amplitudes.len().trailing_zeros() as usize;
        assert_eq!(
            1usize << n,
            amplitudes.len(),
            "dimension must be a power of two"
        );
        let mut sv = Statevector::new(n).expect("small test state");
        sv.data = amplitudes;
        sv
    }

    fn real(values: &[f64]) -> Vec<C64> {
        values.iter().map(|&v| C64::new(v, 0.0)).collect()
    }

    /// Every strategy — the dispatching `sample`, and each path forced — agrees
    /// with the oracle for `shots` draws from `seed`, and leaves the RNG in the
    /// same state.
    fn assert_equivalent(sv: &Statevector, shots: usize, seed: u64, label: &str) {
        let mut oracle_rng = SplitMix64::new(seed);
        let expected = reference_sample(sv, shots, &mut oracle_rng);
        let next_after = oracle_rng.next_u64();

        type Strategy = fn(&Statevector, usize, &mut SplitMix64) -> HashMap<usize, u64>;
        let strategies: [(&str, Strategy); 3] = [
            ("sample", |sv, s, r| sv.sample(s, r)),
            ("cdf", |sv, s, r| {
                if s == 0 {
                    HashMap::new()
                } else {
                    sv.sample_with_cdf(s, r)
                }
            }),
            ("sorted", |sv, s, r| {
                if s == 0 {
                    HashMap::new()
                } else {
                    sv.sample_sorted(s, r)
                }
            }),
        ];
        for (name, strategy) in strategies {
            let mut rng = SplitMix64::new(seed);
            let got = strategy(sv, shots, &mut rng);
            assert_eq!(
                got, expected,
                "{label}: {name} differs (shots={shots}, seed={seed})"
            );
            assert_eq!(
                rng.next_u64(),
                next_after,
                "{label}: {name} consumed a different number of draws"
            );
        }
        assert_eq!(
            expected.values().sum::<u64>(),
            shots as u64,
            "{label}: shots lost"
        );
    }

    fn shot_sweep(dim: usize) -> Vec<usize> {
        let mut shots = vec![
            0,
            1,
            2,
            3,
            dim.saturating_sub(1),
            dim,
            dim + 1,
            4 * dim + 3,
            1000,
        ];
        shots.retain(|&s| s != usize::MAX);
        shots
    }

    const SEEDS: [u64; 5] = [0, 1, 7, 12345, u64::MAX];

    fn sweep(sv: &Statevector, label: &str) {
        for shots in shot_sweep(sv.dim()) {
            for seed in SEEDS {
                assert_equivalent(sv, shots, seed, label);
            }
        }
    }

    #[test]
    fn bell_state_matches_the_oracle() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        sweep(&state(real(&[h, 0.0, 0.0, h])), "bell");
    }

    #[test]
    fn uniform_superposition_matches_the_oracle() {
        for n in [1usize, 3, 6, 10] {
            let dim = 1usize << n;
            let a = 1.0 / (dim as f64).sqrt();
            sweep(
                &state(vec![C64::new(a, 0.0); dim]),
                &format!("uniform n={n}"),
            );
        }
    }

    #[test]
    fn mass_on_the_last_index_matches_the_oracle() {
        let mut amps = vec![C64::new(0.0, 0.0); 16];
        amps[15] = C64::new(0.0, 1.0);
        sweep(&state(amps), "last-index");
    }

    #[test]
    fn trailing_zero_amplitudes_match_the_oracle() {
        // The CDF plateaus at `total` before the end: no shot may land on the
        // zero-probability tail, even when `r` rounds up to `total`.
        sweep(
            &state(real(&[0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            "trailing zeros",
        );
        sweep(
            &state(real(&[0.0, 1.0, 0.0, 0.0])),
            "single interior basis state",
        );
    }

    #[test]
    fn unnormalized_drift_matches_the_oracle() {
        // Norm slightly above and below 1, as long gate sequences leave it.
        let above: Vec<f64> = [0.5, 0.5, 0.5, 0.5]
            .iter()
            .map(|v| v * (1.0 + 3e-12))
            .collect();
        let below: Vec<f64> = [0.5, 0.5, 0.5, 0.5]
            .iter()
            .map(|v| v * (1.0 - 3e-12))
            .collect();
        sweep(&state(real(&above)), "drift above");
        sweep(&state(real(&below)), "drift below");
        // Grossly unnormalized: sampling is relative to `total`.
        sweep(
            &state(real(&[3.0, 1.0, 2.0, 0.5, 0.0, 7.0, 0.0, 0.1])),
            "unnormalized",
        );
    }

    #[test]
    fn ties_between_draws_and_cumulative_sums_match_the_oracle() {
        // Dyadic probabilities make many cumulative sums exactly representable, so
        // `r == acc` ties happen; with duplicated amplitudes they also share
        // values. Probabilities 1/4, 1/4, 0, 1/2 and friends.
        sweep(
            &state(real(&[0.5, 0.5, 0.0, std::f64::consts::FRAC_1_SQRT_2])),
            "dyadic",
        );
        let mut amps = vec![C64::new(0.0, 0.0); 64];
        for (i, a) in amps.iter_mut().enumerate().step_by(3) {
            *a = C64::new(if i % 2 == 0 { 0.25 } else { 0.125 }, 0.0);
        }
        sweep(&state(amps), "dyadic sparse");
    }

    #[test]
    fn an_exact_tie_lands_on_the_first_index_reaching_it() {
        // Force r == acc: with total = 1 and a draw of exactly 0.5, both paths must
        // pick index 0 (acc_0 = 0.5 >= 0.5), never index 1.
        let sv = state(real(&[
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ]));
        let cdf = [
            sv.data[0].norm_sqr(),
            sv.data[0].norm_sqr() + sv.data[1].norm_sqr(),
        ];
        let r = cdf[0];
        assert_eq!(cdf.partition_point(|&c| c < r), 0);
        // And the random sweep over many seeds exercises the same comparison.
        sweep(&sv, "half-half");
    }

    #[test]
    fn a_zero_state_sends_every_shot_to_index_zero() {
        // total == 0, so every r == 0 and the first cumulative sum (0) is >= r.
        sweep(&state(vec![C64::new(0.0, 0.0); 8]), "zero state");
    }

    #[test]
    fn non_finite_states_match_the_oracle() {
        // Not reachable from a unitary evolution, but the two paths must still
        // agree rather than diverge silently: NaN and infinite amplitudes.
        let nan = state(real(&[0.5, f64::NAN, 0.5, 0.5]));
        let inf = state(real(&[0.5, f64::INFINITY, 0.5, 0.5]));
        for shots in [1, 3, 4, 9] {
            for seed in SEEDS {
                assert_equivalent(&nan, shots, seed, "nan");
                assert_equivalent(&inf, shots, seed, "inf");
            }
        }
    }

    #[test]
    fn a_large_state_with_few_shots_matches_the_oracle() {
        // The regime the sorted path exists for: shots << 2^n.
        let dim = 1usize << 14;
        let amps: Vec<C64> = (0..dim)
            .map(|i| C64::new(((i * 7919) % 101) as f64, ((i * 104729) % 37) as f64 - 18.0))
            .collect();
        let sv = state(amps);
        for shots in [1, 10, 1000, dim - 1] {
            for seed in SEEDS {
                assert_equivalent(&sv, shots, seed, "large random");
            }
        }
    }
}
