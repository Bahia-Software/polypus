//! The workspace's seeded pseudo-random generator: SplitMix64.
//!
//! SplitMix64 (Steele, Lea & Flood, 2014) is a public-domain mixing function.
//! It is not cryptographic, but it is fast, has a 2^64 period, and passes the
//! statistical tests relevant for what Polypus draws from it: measurement shots
//! in `polypus-sim`, dataset splits and minibatches in `polypus-qml`, the
//! per-call seeds of `polypus-backend` (contract C-7) and the probe sequence of
//! the expression checksums in [`expr`](crate::expr). Every one of them uses
//! this definition, so the stream behind a fixed seed is defined in one place.
//!
//! Carrying the generator in the repository, instead of depending on an RNG
//! crate, keeps this crate dependency-free and freezes the exact stream: an
//! external crate's algorithm can change between versions and silently move
//! every seeded result, whereas this code cannot change under us.

/// Deterministic 64-bit SplitMix64 generator.
///
/// Output `k` (from `0`) of a generator built with `new(s)` is
/// [`mix`](Self::mix)`(s + (k + 1) · `[`GAMMA`](Self::GAMMA)`)`, wrapping.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// The increment added to the state on every step (the golden-ratio
    /// gamma, `0x9E37_79B9_7F4A_7C15`).
    pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Create a generator seeded with `seed`. The same seed always produces the
    /// same stream, which makes everything drawn from it reproducible.
    #[inline]
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// The SplitMix64 output finaliser: a bijective 64-bit mixer, with
    /// `mix(0) == 0`. [`next_u64`](Self::next_u64) returns it applied to the
    /// advanced state.
    #[inline]
    pub fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Return the next 64-bit value and advance the state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        Self::mix(self.state)
    }

    /// Return a uniformly distributed `f64` in `[0, 1)`.
    ///
    /// Uses the top 53 bits of a generated word for full double precision.
    #[inline]
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

    /// The first outputs of SplitMix64 seeded with `1234567`, from the
    /// reference implementation (Vigna, `splitmix64.c`).
    #[test]
    fn the_stream_matches_the_reference_outputs() {
        let mut rng = SplitMix64::new(1234567);
        let got: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        assert_eq!(
            got,
            [
                6_457_827_717_110_365_317,
                3_203_168_211_198_807_973,
                9_817_491_932_198_370_423,
                4_593_380_528_125_082_431,
                16_408_922_859_458_223_821,
            ]
        );
    }

    /// The map to `f64` is pinned over several steps: the IEEE-754 bits of the
    /// first draws with seed 7, from an independent implementation.
    #[test]
    fn next_f64_is_pinned() {
        let mut rng = SplitMix64::new(7);
        let got: Vec<u64> = (0..4).map(|_| rng.next_f64().to_bits()).collect();
        assert_eq!(
            got,
            [
                0x3FD8_F2F8_7916_4C82,
                0x3F91_30F3_5FD0_F180,
                0x3FEC_D308_1017_5625,
                0x3FE2_A75D_6E0C_E7C5,
            ]
        );
    }

    #[test]
    fn mix_fixes_zero() {
        assert_eq!(SplitMix64::mix(0), 0);
    }

    /// One step from state `x` is the finaliser of `x + GAMMA`, wrapping: the
    /// identity `stream_seed` (polypus-backend) and the checksum probe rely on.
    #[test]
    fn one_step_is_the_mix_of_the_advanced_state() {
        for x in [0, 1, 42, 1234567, u64::MAX - SplitMix64::GAMMA, u64::MAX] {
            assert_eq!(
                SplitMix64::new(x).next_u64(),
                SplitMix64::mix(x.wrapping_add(SplitMix64::GAMMA)),
                "state {x:#x}"
            );
        }
    }
}
