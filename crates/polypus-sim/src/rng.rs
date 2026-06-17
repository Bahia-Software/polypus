//! A small, dependency-free SplitMix64 PRNG for deterministic shot sampling.
//!
//! SplitMix64 (Steele, Lea & Flood, 2014) is a public-domain mixing function.
//! It is not cryptographic, but it is fast, has a 2^64 period, and passes the
//! statistical tests relevant for Monte-Carlo measurement sampling. Using it
//! keeps the crate free of an external RNG dependency.

/// Deterministic 64-bit SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator seeded with `seed`. The same seed always produces the
    /// same stream, which makes sampled measurement counts reproducible.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Return the next 64-bit value and advance the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Return a uniformly distributed `f64` in `[0, 1)`.
    ///
    /// Uses the top 53 bits of a generated word for full double precision.
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53 scaling over the 53 high bits → [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f64_stays_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }
}
