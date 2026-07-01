//! Injectable RNG source for the optimizers.
//!
//! The default (`None` seed) uses [`rand::thread_rng`], preserving the exact
//! non-deterministic behaviour of the original optimizers. Passing a seed
//! selects a reproducible [`StdRng`] instead. Both variants delegate every
//! [`RngCore`] method to the wrapped generator, so the algorithm bodies consume
//! the RNG identically regardless of the source — only the construction differs.

use rand::rngs::{StdRng, ThreadRng};
use rand::{thread_rng, RngCore, SeedableRng};

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
    /// Build the RNG: `None` → [`rand::thread_rng`]; `Some(seed)` → seeded
    /// [`StdRng`].
    pub(crate) fn from_seed(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => OptRng::Seeded(Box::new(StdRng::seed_from_u64(s))),
            None => OptRng::Thread(thread_rng()),
        }
    }
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

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            OptRng::Thread(r) => r.try_fill_bytes(dest),
            OptRng::Seeded(r) => r.try_fill_bytes(dest),
        }
    }
}
