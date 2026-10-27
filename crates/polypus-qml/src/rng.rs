//! A small, dependency-free SplitMix64 PRNG plus a Fisher–Yates shuffle.
//!
//! SplitMix64 (Steele, Lea & Flood, 2014) is a public-domain mixing function.
//! It is not cryptographic, but it is fast, has a 2^64 period, and passes the
//! statistical tests relevant for what this crate needs: shuffling rows for a
//! dataset split now, and (in a later phase) drawing uniform floats for
//! parameter initialization. Carrying our own generator (identical algorithm to
//! the one in `polypus-sim`) keeps the crate free of an external RNG dependency
//! and, more importantly, freezes the exact byte stream inside this repository:
//! an external crate's internal algorithm can change between versions and break
//! the byte-for-byte reproducibility promised by the engineering guidelines,
//! whereas this code cannot change under us.
//!
//! Phase 1 exposes only [`SplitMix64::next_u64`] and [`shuffle`]; the
//! `[0, 1)`-float draw that parameter initialization needs arrives with its
//! consumer in a later phase, so it is not added here yet (an unused method
//! would only be dead code today).

/// Deterministic 64-bit SplitMix64 generator.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator seeded with `seed`. The same seed always produces the
    /// same stream, which makes dataset splits and parameter initialization
    /// reproducible.
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Return the next 64-bit value and advance the state.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Shuffle `indices` in place into a uniformly random permutation using `rng`.
///
/// This is the Fisher–Yates shuffle in its modern (Durstenfeld) form: walking
/// from the last position down to the second, each element is swapped with one
/// chosen uniformly from itself and every position before it. After the pass
/// every permutation is equally likely, and the result is a permutation of the
/// original multiset by construction — the algorithm only ever swaps existing
/// elements, so nothing is added, dropped or duplicated.
///
/// The bounded index `j ∈ [0, i]` is drawn as `next_u64() % (i + 1)`. The
/// modulo introduces a bias vanishingly small for the slice lengths a dataset
/// split ever sees, and — the property that matters here — it is fully
/// deterministic for a given seed. Slices of length 0 and 1 are already
/// trivially shuffled: the loop body never runs, which is the correct no-op.
pub(crate) fn shuffle(indices: &mut [usize], rng: &mut SplitMix64) {
    // `i` is the position being placed; draw its partner from `[0, i]`.
    for i in (1..indices.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn shuffle_is_deterministic_for_a_seed() {
        let mut a: Vec<usize> = (0..64).collect();
        let mut b: Vec<usize> = (0..64).collect();
        let mut rng_a = SplitMix64::new(123);
        let mut rng_b = SplitMix64::new(123);
        shuffle(&mut a, &mut rng_a);
        shuffle(&mut b, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_is_a_valid_permutation() {
        for len in [0usize, 1, 2, 3, 8, 33, 100] {
            let original: Vec<usize> = (0..len).collect();
            let mut shuffled = original.clone();
            let mut rng = SplitMix64::new(len as u64 + 1);
            shuffle(&mut shuffled, &mut rng);
            // Same multiset of values: sorting must recover the input exactly.
            let mut sorted = shuffled.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, original);
            assert_eq!(shuffled.len(), len);
        }
    }
}
