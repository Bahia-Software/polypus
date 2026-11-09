//! What a seed means across the calls of one run (contract C-7).
//!
//! [`RunParams::seed`](crate::RunParams::seed) is the run's base seed on every
//! call. A backend gives each call its own sampling stream derived from it, so
//! two generations of a training run never share shot noise and the same base
//! seed replays the same sequence. The native backend seeds each circuit with
//! [`stream_seed`]`(seed, k)`, `k` its index in the run; a backend that passes
//! one seed per call to an SDK (Aer, CUNQA, the subprocess bridge) takes it from
//! [`CallSeeds`].

use std::sync::atomic::{AtomicU64, Ordering};

/// SplitMix64 increment (the golden-ratio gamma).
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 output finaliser: a bijective 64-bit mixer.
fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Output `index` of a SplitMix64 stream whose state starts at the mixed
/// `base_seed`.
///
/// A plain `base_seed + index` made neighbouring base seeds share streams shifted
/// by one (index `k` of seed `s` was index `k-1` of seed `s+1`, issue #348).
/// Mixing the base seed first puts distinct base seeds on unrelated stream
/// positions, so repetitions run with seeds `0, 1, 2, ...` are independent. The
/// map is a pure function of `(base_seed, index)`, so the same seed reproduces.
pub fn stream_seed(base_seed: u64, index: u64) -> u64 {
    let state = splitmix64_mix(base_seed);
    splitmix64_mix(state.wrapping_add(index.wrapping_add(1).wrapping_mul(SPLITMIX64_GAMMA)))
}

/// Numbers the calls a backend receives and hands out each call's seed.
///
/// Call `0` gets the base seed itself, so a run of one call (a
/// `run_quantum_circuit`) is seeded exactly as the SDK would be with that seed;
/// call `n >= 1` gets [`stream_seed`]`(base, n)`. Call `next` once per call
/// the backend receives (or per submission, if it sends circuits one at a
/// time), seeded or not; the planners call a backend sequentially, so that
/// order is the run's. The default
/// [`run_shots_distributed`](crate::QuantumBackend::run_shots_distributed)
/// makes one `run_circuits` call per shot group, so each group counts.
#[derive(Debug, Default)]
pub struct CallSeeds {
    calls: AtomicU64,
}

impl CallSeeds {
    /// A counter at call `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Count a call and return its seed: `None` for `None`, `base` for the
    /// first call, `stream_seed(base, n)` for call `n`.
    pub fn next(&self, base: Option<u64>) -> Option<u64> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        base.map(|base| if n == 0 { base } else { stream_seed(base, n) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The map is pinned: a change would silently move every count for a fixed
    /// seed (C-7), so it must be deliberate.
    #[test]
    fn stream_seed_is_pinned() {
        assert_eq!(splitmix64_mix(0), 0);
        // SplitMix64 seeded with state 0: first output (reference value).
        assert_eq!(stream_seed(0, 0), 0xE220_A839_7B1D_CDAF);
        let seeds: HashSet<u64> = (0..1000).map(|k| stream_seed(7, k)).collect();
        assert_eq!(seeds.len(), 1000);
    }

    #[test]
    fn the_first_call_gets_the_base_seed_and_later_ones_their_own() {
        let seeds = CallSeeds::new();
        let got: Vec<_> = (0..100).map(|_| seeds.next(Some(42)).unwrap()).collect();
        assert_eq!(got[0], 42);
        assert_eq!(
            got[1..],
            (1..100).map(|n| stream_seed(42, n)).collect::<Vec<_>>()
        );
        assert_eq!(got.iter().collect::<HashSet<_>>().len(), 100);
    }

    #[test]
    fn the_same_base_seed_replays_the_sequence() {
        let (a, b) = (CallSeeds::new(), CallSeeds::new());
        for _ in 0..10 {
            assert_eq!(a.next(Some(7)), b.next(Some(7)));
        }
    }

    #[test]
    fn no_seed_stays_none_and_still_counts() {
        let seeds = CallSeeds::new();
        assert_eq!(seeds.next(None), None);
        assert_eq!(seeds.next(Some(9)), Some(stream_seed(9, 1)));
    }
}
