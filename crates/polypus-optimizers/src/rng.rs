//! Injectable RNG source for the optimizers.
//!
//! The default (`None` seed) uses [`rand::rng`], preserving the exact
//! non-deterministic behaviour of the original optimizers. Passing a seed
//! selects a reproducible [`StdRng`] instead. Both variants delegate every
//! [`RngCore`] method to the wrapped generator, so the algorithm bodies consume
//! the RNG identically regardless of the source — only the construction differs.

use rand::rngs::{StdRng, ThreadRng};
use rand::{rng, RngCore, SeedableRng};

/// RNG used by the optimizers, chosen at run start from an optional seed.
pub(crate) enum OptRng {
    /// Non-deterministic thread-local generator (the default).
    Thread(ThreadRng),
    /// Deterministic generator seeded from an explicit `u64`.
    ///
    /// Boxed because [`StdRng`] is much larger than [`ThreadRng`]; the
    /// allocation happens once per optimization run, never in a hot loop.
    Seeded(Box<StdRng>),
}

impl OptRng {
    /// Build the RNG: `None` → [`rand::rng`]; `Some(seed)` → seeded
    /// [`StdRng`].
    pub(crate) fn from_seed(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => OptRng::Seeded(Box::new(StdRng::seed_from_u64(s))),
            None => OptRng::Thread(rng()),
        }
    }
}

/// Build the run RNG from an optional seed and hand it to `run`.
///
/// Centralises the identical `seed → OptRng::from_seed → run` dispatch that
/// every optimizer's `optimize` implementation performs (DE, PSO, and QNG),
/// keeping the seed-to-RNG wiring in one place next to [`OptRng`]. The closure
/// receives the freshly built generator by mutable reference and returns the
/// optimization outcome, which is passed straight through.
pub(crate) fn with_seeded_rng<T>(seed: Option<u64>, run: impl FnOnce(&mut OptRng) -> T) -> T {
    let mut rng = OptRng::from_seed(seed);
    run(&mut rng)
}

impl RngCore for OptRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        match self {
            OptRng::Thread(r) => r.next_u32(),
            OptRng::Seeded(r) => r.next_u32(),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        match self {
            OptRng::Thread(r) => r.next_u64(),
            OptRng::Seeded(r) => r.next_u64(),
        }
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            OptRng::Thread(r) => r.fill_bytes(dest),
            OptRng::Seeded(r) => r.fill_bytes(dest),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_seeded_rng_is_deterministic() {
        // A fixed seed reproduces the exact RNG stream — the property the
        // optimizers' determinism tests ultimately rely on.
        let a = with_seeded_rng(Some(42), |rng| rng.next_u64());
        let b = with_seeded_rng(Some(42), |rng| rng.next_u64());
        assert_eq!(a, b);
    }

    #[test]
    fn with_seeded_rng_distinct_seeds_differ() {
        let a = with_seeded_rng(Some(1), |rng| rng.next_u64());
        let b = with_seeded_rng(Some(2), |rng| rng.next_u64());
        assert_ne!(a, b);
    }

    #[test]
    fn with_seeded_rng_passes_closure_value_through() {
        let value = with_seeded_rng(Some(7), |_| 99u32);
        assert_eq!(value, 99);
    }

    /// Pins the absolute seeded output of every sampling method the optimizers
    /// draw from in `src/` (DE, PSO, QNG). The determinism tests above only
    /// compare one run against another, so a `rand` bump that changed the
    /// `StdRng` stream or a sampling algorithm would pass them silently; this
    /// one fails instead. Values captured with rand 0.9.4; `f64`s are compared
    /// by bits, with no tolerance.
    #[test]
    fn seeded_stream_is_pinned() {
        use rand::seq::IndexedRandom;
        use rand::Rng;

        let raw: [u64; 3] =
            with_seeded_rng(Some(42), |rng| std::array::from_fn(|_| rng.next_u64()));
        assert_eq!(
            raw,
            [
                9713269763989775522,
                10011513049433592189,
                11740708795755607249
            ]
        );

        // PSO: `random::<f64>()` for the r1/r2 weights.
        let unit: [u64; 3] = with_seeded_rng(Some(42), |rng| {
            std::array::from_fn(|_| rng.random::<f64>().to_bits())
        });
        assert_eq!(
            unit,
            [
                4602918027047224548,
                4603063653651445162,
                4603907987511953958
            ]
        );

        // DE/PSO/QNG: `random_range` over an `f64` interval for initialisation.
        let ranged: [u64; 3] = with_seeded_rng(Some(42), |rng| {
            std::array::from_fn(|_| rng.random_range(-1.5..2.5f64).to_bits())
        });
        assert_eq!(
            ranged,
            [
                4603635650670957456,
                4604218157087839912,
                4607388955664946252
            ]
        );

        let indices: [usize; 3] = with_seeded_rng(Some(42), |rng| {
            std::array::from_fn(|_| rng.random_range(0..10usize))
        });
        assert_eq!(indices, [1, 5, 2]);

        // DE: `random_bool(0.7)` for the crossover mask.
        let flips: [bool; 16] = with_seeded_rng(Some(42), |rng| {
            std::array::from_fn(|_| rng.random_bool(0.7))
        });
        assert_eq!(
            flips,
            [
                true, true, true, true, true, true, false, false, true, true, false, true, true,
                true, true, true
            ]
        );

        // DE: three distinct donors chosen from the other population members.
        let ids: Vec<usize> = (0..10).collect();
        let donors: Vec<usize> = with_seeded_rng(Some(42), |rng| {
            ids.choose_multiple(rng, 3).cloned().collect()
        });
        assert_eq!(donors, [1, 4, 2]);
    }
}
