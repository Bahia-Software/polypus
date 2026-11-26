//! A Fisher–Yates shuffle over the workspace's SplitMix64.
//!
//! The generator is [`polypus_circuit::rng::SplitMix64`], the one the
//! simulator samples shots with; this module adds what this crate draws from
//! it: shuffled rows for a dataset split and for each minibatch. Using that
//! in-repository generator rather than an RNG crate freezes the exact stream:
//! an external crate's internal algorithm can change between versions and
//! break the byte-for-byte reproducibility promised by the engineering
//! guidelines, whereas this code cannot change under us.

use polypus_circuit::rng::SplitMix64;

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
