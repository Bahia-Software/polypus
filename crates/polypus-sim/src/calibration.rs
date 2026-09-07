//! Machine-aware calibration of the gate-parallelism threshold.
//!
//! [`DEFAULT_PARALLEL_THRESHOLD`](crate::DEFAULT_PARALLEL_THRESHOLD) is a single
//! hardware-oblivious constant, but the qubit count at which spreading one gate
//! across a rayon pool actually beats running it sequentially depends on the
//! *machine* — core count, memory bandwidth, cache — not just on `n`. Pinning it
//! at a constant makes the native simulator go parallel too late on a big node
//! (measurably slower than Qiskit Aer around n = 12–18) or too eagerly on a
//! small one. This module measures that crossover once, caches it on disk keyed
//! by a hardware fingerprint, and reuses it across processes — the same "wisdom"
//! mechanism FFTW uses.
//!
//! Everything here is gated behind the `parallel` feature: without the rayon
//! kernels there is no parallel path to calibrate, so the whole notion is moot
//! (and the `serde`/`serde_json` dependencies it needs stay out of the default
//! build).
//!
//! ## Who calibrates, and when
//!
//! Measurement is an **explicit** act — [`calibrate_and_cache`], driven by
//! `install.sh` or `polypus.calibrate_parallel_threshold()`. The per-process
//! runtime path ([`resolve_threshold`]) never measures: it reads the cache once
//! and, on a miss or a hardware mismatch, degrades to the static default rather
//! than stalling a user's first circuit for a second while it times kernels.
//! That keeps the fallback exactly as fast as today's behaviour and makes a
//! calibration the caller's deliberate choice (e.g. the first line of a SLURM
//! script, before any circuit runs).
//!
//! ## Visibility
//!
//! Diagnostics go through the `log` facade (this crate never depends on
//! `polypus-logger`) and stay silent until the application installs a sink —
//! from Python, `polypus.init_logger()` once per process.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{gates, kernels, C64, DEFAULT_PARALLEL_THRESHOLD, MAX_QUBITS};

/// Qubit counts probed during calibration, ascending. The smallest is the floor
/// below which the parallel path is never worth it on any machine we have seen;
/// probing stops at 18 because by then a single gate's `2^n` work dwarfs the
/// pool overhead on every machine, so one that has not crossed over by 18 will
/// not at a size anyone simulates densely (the ceiling is [`MAX_QUBITS`]).
const CALIBRATION_SIZES: [usize; 5] = [10, 12, 14, 16, 18];

/// Samples taken per (size, path) measurement, keeping the **median**. Odd, so
/// the median is a real sample; `>= 9` because on a shared machine 5 samples
/// picked the wrong crossover — the parallel path lost a scheduling race it wins
/// on the median.
const CALIBRATION_SAMPLES: usize = 9;

/// Lower bound on the wall-clock duration of one timing sample. Each sample
/// applies the kernel as many times as needed to fill this window, so clock
/// granularity and one-off scheduler jitter are amortised instead of dominating.
const SAMPLE_WINDOW: Duration = Duration::from_millis(3);

/// Parallel is chosen only when it is at least this many times faster than
/// sequential — a comfortable margin, not a photo finish. The crossover must pay
/// for the pool overhead *and* leave headroom, or a slightly noisy tie would
/// latch parallelism on and cost time on every later gate. `1.25` ⇒ 25% faster.
const MIN_SPEEDUP: f64 = 1.25;

/// Schema version of the on-disk cache. A file written by a different layout is
/// ignored (treated as absent) rather than mis-parsed.
const CACHE_SCHEMA: u32 = 1;

/// Threshold meaning "never take the gate-parallel path": one past the qubit
/// ceiling, so no runnable circuit (`n <= MAX_QUBITS`) reaches it. Chosen when
/// parallel never wins by [`MIN_SPEEDUP`] anywhere in the probed range — a safe
/// outcome (equivalent to disabling gate-level parallelism) we have never seen
/// hurt any thread count.
const PARALLELISM_DISABLED: usize = MAX_QUBITS + 1;

// The disabling sentinel must be unreachable by any runnable circuit
// (`n <= MAX_QUBITS`); enforced at compile time so it can never silently regress.
const _: () = assert!(PARALLELISM_DISABLED > MAX_QUBITS);

/// The measured crossover plus the context needed to report it.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Qubit count at or above which the gate-parallel path should be used.
    pub threshold: usize,
    /// `rayon::current_num_threads()` observed during calibration — the hardware
    /// fingerprint the cache is keyed by.
    pub num_threads: usize,
    /// Wall-clock time the measurement itself took.
    pub duration: Duration,
}

/// Outcome of [`calibrate_and_cache`]: the decided threshold plus what happened
/// to the on-disk cache, so a caller (the Python binding, `install.sh`) can
/// report whether it recomputed or reused, and whether persistence succeeded.
#[derive(Debug, Clone)]
pub struct CalibrationOutcome {
    /// Qubit count at or above which the gate-parallel path should be used.
    pub threshold: usize,
    /// Rayon threads detected — the cache key.
    pub num_threads: usize,
    /// Measurement time; [`Duration::ZERO`] when `reused_cache` is true.
    pub duration: Duration,
    /// True when a cache already valid for the current hardware was reused
    /// without re-measuring (only possible with `force = false`).
    pub reused_cache: bool,
    /// Absolute path of the cache file, when one could be resolved.
    pub cache_path: Option<String>,
    /// True when this call persisted a freshly measured result to disk.
    pub cache_written: bool,
}

/// On-disk cache payload. Small and forward-guarded by `schema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachedCalibration {
    schema: u32,
    threshold: usize,
    num_threads: usize,
}

/// One size's median timings, nanoseconds per gate application.
struct SizeTiming {
    n: usize,
    seq_ns: f64,
    par_ns: f64,
}

/// Pick the smallest probed size at which parallel beats sequential by at least
/// [`MIN_SPEEDUP`]; fall back to [`PARALLELISM_DISABLED`] if none does. Pure over
/// its input, so the decision is unit-tested without touching the clock.
fn select_threshold(timings: &[SizeTiming]) -> usize {
    for t in timings {
        if t.par_ns > 0.0 && t.seq_ns / t.par_ns >= MIN_SPEEDUP {
            return t.n;
        }
    }
    PARALLELISM_DISABLED
}

/// Median (per-application) nanoseconds to apply a dense 1-qubit gate to a `2^n`
/// buffer on the given path. One buffer is reused across samples; applying `H`
/// repeatedly keeps amplitudes bounded (`H·H = I`), so there is no drift to
/// renormalise between samples.
fn measure_median_ns(n: usize, parallel: bool) -> f64 {
    let dim = 1usize << n;
    let mut data = vec![C64::new(0.0, 0.0); dim];
    data[0] = C64::new(1.0, 0.0);
    let gate = gates::h();
    let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
    for _ in 0..CALIBRATION_SAMPLES {
        samples.push(sample_apply_ns(&mut data, n, &gate, parallel));
    }
    median(&mut samples)
}

/// One timing sample: apply the kernel enough times to fill [`SAMPLE_WINDOW`],
/// then return nanoseconds per application. The repeat count is discovered by
/// doubling, so a single window constant fits every size and machine.
fn sample_apply_ns(data: &mut [C64], n: usize, gate: &[[C64; 2]; 2], parallel: bool) -> f64 {
    let mut reps: u32 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            // Call the kernel directly with the explicit `parallel` flag: this
            // measures the two kernel paths in isolation, without routing through
            // `Statevector` and the very threshold we are calibrating.
            kernels::apply_1q(data, n, 0, gate, parallel);
        }
        let elapsed = start.elapsed();
        // Keep the optimiser from hoisting the timed loop: the buffer is observed
        // here and carried into the next sample.
        std::hint::black_box(&data[0]);
        if elapsed >= SAMPLE_WINDOW {
            return elapsed.as_secs_f64() * 1e9 / f64::from(reps);
        }
        reps = reps.saturating_mul(2);
    }
}

/// Median of `samples` (sorted in place). Never called with an empty slice
/// ([`CALIBRATION_SAMPLES`] `>= 1`).
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Measure the gate-parallelism crossover on this machine.
///
/// Probes [`CALIBRATION_SIZES`], timing the dense 1-qubit kernel on both paths
/// (median of [`CALIBRATION_SAMPLES`] samples each), and returns the smallest
/// size where parallel is at least [`MIN_SPEEDUP`]× faster — or a
/// parallelism-disabling threshold if none is. **Pure measurement**: it does not
/// read or write the cache (see [`calibrate_and_cache`]) and, with the default
/// parameters, takes well under a second.
///
/// Diagnostics go through the `log` facade and need `polypus.init_logger()` to
/// be visible.
pub fn calibrate_parallel_threshold() -> CalibrationResult {
    let start = Instant::now();
    let timings: Vec<SizeTiming> = CALIBRATION_SIZES
        .iter()
        .map(|&n| SizeTiming {
            n,
            seq_ns: measure_median_ns(n, false),
            par_ns: measure_median_ns(n, true),
        })
        .collect();
    let threshold = select_threshold(&timings);
    let num_threads = rayon::current_num_threads();
    let duration = start.elapsed();
    log::info!(
        "calibrated gate-parallel threshold = {threshold} qubit(s) for {num_threads} thread(s) \
         in {duration:?}"
    );
    CalibrationResult {
        threshold,
        num_threads,
        duration,
    }
}

/// Directory holding Polypus caches: `$XDG_CACHE_HOME/polypus` when that is set
/// and non-empty, otherwise `$HOME/.cache/polypus`. No `dirs`/`directories`
/// crate is added for this single lookup — the workspace has none and this does
/// not justify one. `None` when neither variable is usable (an unusual, headless
/// environment); the caller then runs uncached rather than guessing a path.
fn cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("polypus"));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home).join(".cache").join("polypus"));
        }
    }
    None
}

/// Absolute path of the cache file, when a cache directory can be resolved.
fn cache_file() -> Option<PathBuf> {
    Some(cache_dir()?.join("parallel_threshold.json"))
}

/// Parse a cache file. Any problem — missing, unreadable, malformed, wrong
/// schema — yields `None` (run uncached), never an error: a broken cache must
/// degrade to the default, never break simulation.
fn read_cache_from(path: &Path) -> Option<CachedCalibration> {
    let bytes = std::fs::read(path).ok()?;
    let cached: CachedCalibration = serde_json::from_slice(&bytes).ok()?;
    if cached.schema != CACHE_SCHEMA {
        return None;
    }
    Some(cached)
}

/// Serialise `cached` to `path`, creating parent directories. Errors (read-only
/// filesystem, container, CI) propagate so the caller can fall back to the
/// default without persistence — a failed write is never fatal.
fn write_cache_to(path: &Path, cached: &CachedCalibration) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(cached).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Read the cache at its resolved location, or `None` if there is no location.
fn read_cache() -> Option<CachedCalibration> {
    read_cache_from(&cache_file()?)
}

/// Why the runtime resolver fell back to the static
/// [`DEFAULT_PARALLEL_THRESHOLD`] instead of a calibrated value.
///
/// Public because a caller outside this crate (the `polypus` bindings layer)
/// surfaces it to the user — see [`resolved_fallback_reason`], which returns
/// `Some(reason)` on a fallback and `None` when the threshold came from a cache
/// valid for this hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// No usable calibration cache was found for this machine.
    NotCalibrated,
    /// A cache existed but was made for a different thread count (the machine, or
    /// `RAYON_NUM_THREADS`, changed since calibration).
    HardwareChanged {
        /// Thread count the stale cache was calibrated for.
        cached_threads: usize,
    },
}

/// The runtime resolution, pure over `(cache, current thread count)` so it is
/// unit-tested without disk or timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    Cached(usize),
    Fallback(FallbackReason),
}

/// Decide the threshold from a (possibly absent) cache and the current thread
/// count: use the cache only when its fingerprint matches this machine.
fn decide(cache: Option<CachedCalibration>, current_threads: usize) -> Decision {
    match cache {
        Some(c) if c.num_threads == current_threads => Decision::Cached(c.threshold),
        Some(c) => Decision::Fallback(FallbackReason::HardwareChanged {
            cached_threads: c.num_threads,
        }),
        None => Decision::Fallback(FallbackReason::NotCalibrated),
    }
}

/// The once-per-process resolution: the threshold in force, plus *why* it was
/// chosen (`None` = it came from a cache valid for this hardware; `Some` = a
/// fallback to the static default, with the reason).
///
/// Both fields are computed together in one place so a caller reading the reason
/// ([`resolved_fallback_reason`]) can never observe a threshold that disagrees
/// with it — the two answers come from the same cache read and the same decision.
#[derive(Debug, Clone, Copy)]
struct Resolution {
    threshold: usize,
    fallback: Option<FallbackReason>,
}

/// Process-wide memo of the resolution: filled once, reused for the life of the
/// process.
static RESOLUTION: OnceLock<Resolution> = OnceLock::new();

/// Resolve (once per process) and memoise the threshold together with its
/// fallback reason.
///
/// It reads the on-disk cache and compares its hardware fingerprint
/// (`rayon::current_num_threads()`) with this machine's:
/// * match → the calibrated threshold, no fallback;
/// * mismatch → [`DEFAULT_PARALLEL_THRESHOLD`] plus a `warn!` and
///   [`FallbackReason::HardwareChanged`];
/// * no cache → [`DEFAULT_PARALLEL_THRESHOLD`] plus an `info!` and
///   [`FallbackReason::NotCalibrated`].
///
/// It never calibrates here: measurement is an explicit act
/// ([`calibrate_and_cache`], run by `install.sh` or
/// `polypus.calibrate_parallel_threshold()`), so simulation never pays a
/// surprise stall and a cache miss degrades safely to today's behaviour. The
/// once-per-process guard mirrors the `LOGGER_INSTALLED` guard in the bindings —
/// resolve once, never per circuit or per gate. The `log` records here are only
/// visible with `polypus.init_logger()`; the bindings additionally surface a
/// fallback through Python's `warnings`, which is visible by default.
fn resolution() -> Resolution {
    *RESOLUTION.get_or_init(|| {
        let current = rayon::current_num_threads();
        match decide(read_cache(), current) {
            Decision::Cached(threshold) => {
                log::debug!(
                    "using cached gate-parallel threshold {threshold} qubit(s) for \
                     {current} thread(s)"
                );
                Resolution {
                    threshold,
                    fallback: None,
                }
            }
            Decision::Fallback(reason) => {
                match reason {
                    FallbackReason::HardwareChanged { cached_threads } => log::warn!(
                        "gate-parallel calibration was made for {cached_threads} thread(s) but \
                         this machine has {current}; using the default threshold \
                         {DEFAULT_PARALLEL_THRESHOLD}. Recalibrate with \
                         polypus.calibrate_parallel_threshold(force=True)."
                    ),
                    FallbackReason::NotCalibrated => log::info!(
                        "no gate-parallel calibration cached; using the default threshold \
                         {DEFAULT_PARALLEL_THRESHOLD}. Run install.sh or \
                         polypus.calibrate_parallel_threshold() to tune it for this machine \
                         (requires polypus.init_logger() to be visible)."
                    ),
                }
                Resolution {
                    threshold: DEFAULT_PARALLEL_THRESHOLD,
                    fallback: Some(reason),
                }
            }
        }
    })
}

/// Threshold the default [`StatevectorSimulator`](crate::StatevectorSimulator)
/// uses, resolved **once per process** and memoised (see [`resolution`]).
pub(crate) fn resolve_threshold() -> usize {
    resolution().threshold
}

/// Why the process-wide threshold fell back to the static default, or `None`
/// when it came from a cache valid for this hardware.
///
/// Shares the same once-per-process resolution as [`resolve_threshold`] (reading
/// it here triggers that resolution if it has not happened yet), so the reason
/// always matches the threshold actually in force. Intended for the `polypus`
/// bindings, which turn a `Some(..)` into a default-visible Python warning.
pub fn resolved_fallback_reason() -> Option<FallbackReason> {
    resolution().fallback
}

/// The threshold to reuse without measuring, or `None` if a fresh measurement is
/// required. Pure, so the `force`/reuse logic is unit-tested directly.
fn threshold_to_reuse(
    force: bool,
    cache: Option<CachedCalibration>,
    current_threads: usize,
) -> Option<usize> {
    if force {
        return None;
    }
    match cache {
        Some(c) if c.num_threads == current_threads => Some(c.threshold),
        _ => None,
    }
}

/// Calibrate if needed and persist the result — the entry point behind
/// `polypus.calibrate_parallel_threshold(force=...)` and `install.sh`.
///
/// With `force = false` a cache already valid for the current hardware (same
/// rayon thread count) is reused as-is: nothing is measured and `duration` is
/// zero. Otherwise the crossover is measured and written to the cache. A cache
/// that cannot be written (read-only filesystem, container, CI) is **not** an
/// error: the measured threshold is still returned, with `cache_written = false`,
/// so this process runs calibrated even though the next one will not benefit.
///
/// Intended to run **before any simulation** — e.g. the first line of a SLURM
/// script — because the per-process resolver ([`resolve_threshold`]) reads the
/// cache once and memoises it, so a calibration performed after the first
/// simulator is built will only be picked up by later processes.
pub fn calibrate_and_cache(force: bool) -> CalibrationOutcome {
    let current = rayon::current_num_threads();
    let cache_path = cache_file().map(|p| p.display().to_string());

    if let Some(threshold) = threshold_to_reuse(force, read_cache(), current) {
        return CalibrationOutcome {
            threshold,
            num_threads: current,
            duration: Duration::ZERO,
            reused_cache: true,
            cache_path,
            cache_written: false,
        };
    }

    let result = calibrate_parallel_threshold();
    let cached = CachedCalibration {
        schema: CACHE_SCHEMA,
        threshold: result.threshold,
        num_threads: result.num_threads,
    };
    let cache_written = match cache_file() {
        Some(path) => match write_cache_to(&path, &cached) {
            Ok(()) => true,
            Err(e) => {
                log::warn!(
                    "could not persist gate-parallel calibration to {}: {e}",
                    path.display()
                );
                false
            }
        },
        None => {
            log::warn!(
                "no cache directory (set XDG_CACHE_HOME or HOME) to persist gate-parallel \
                 calibration; this process is calibrated but the result was not saved"
            );
            false
        }
    };

    CalibrationOutcome {
        threshold: result.threshold,
        num_threads: result.num_threads,
        duration: result.duration,
        reused_cache: false,
        cache_path,
        cache_written,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached(threshold: usize, num_threads: usize) -> CachedCalibration {
        CachedCalibration {
            schema: CACHE_SCHEMA,
            threshold,
            num_threads,
        }
    }

    /// A unique temp directory per test/process, so the disk tests never collide
    /// with each other or with the real user cache.
    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("polypus-cal-{tag}-{}", std::process::id()))
    }

    #[test]
    fn select_threshold_picks_smallest_size_that_wins_by_margin() {
        // Parallel loses at 10/12, then wins comfortably from 14 up.
        let timings = vec![
            SizeTiming {
                n: 10,
                seq_ns: 100.0,
                par_ns: 120.0,
            },
            SizeTiming {
                n: 12,
                seq_ns: 100.0,
                par_ns: 105.0,
            },
            SizeTiming {
                n: 14,
                seq_ns: 100.0,
                par_ns: 50.0,
            },
            SizeTiming {
                n: 16,
                seq_ns: 100.0,
                par_ns: 25.0,
            },
            SizeTiming {
                n: 18,
                seq_ns: 100.0,
                par_ns: 20.0,
            },
        ];
        assert_eq!(select_threshold(&timings), 14);
    }

    #[test]
    fn select_threshold_requires_a_comfortable_margin_not_a_tie() {
        // A near-tie (only ~4% faster) must NOT latch parallelism on.
        let timings = vec![SizeTiming {
            n: 10,
            seq_ns: 100.0,
            par_ns: 96.0,
        }];
        assert_eq!(select_threshold(&timings), PARALLELISM_DISABLED);
    }

    #[test]
    fn select_threshold_disables_when_parallel_never_wins() {
        let timings: Vec<SizeTiming> = CALIBRATION_SIZES
            .iter()
            .map(|&n| SizeTiming {
                n,
                seq_ns: 100.0,
                par_ns: 130.0,
            })
            .collect();
        assert_eq!(select_threshold(&timings), PARALLELISM_DISABLED);
    }

    #[test]
    fn decide_uses_cache_only_on_matching_fingerprint() {
        assert_eq!(decide(Some(cached(14, 8)), 8), Decision::Cached(14));
        assert_eq!(
            decide(Some(cached(14, 8)), 32),
            Decision::Fallback(FallbackReason::HardwareChanged { cached_threads: 8 })
        );
        assert_eq!(
            decide(None, 8),
            Decision::Fallback(FallbackReason::NotCalibrated)
        );
    }

    #[test]
    fn reuse_requires_no_force_and_matching_fingerprint() {
        // force=True always recalibrates, even with a perfectly valid cache.
        assert_eq!(threshold_to_reuse(true, Some(cached(14, 8)), 8), None);
        // Matching fingerprint reuses without measuring.
        assert_eq!(threshold_to_reuse(false, Some(cached(14, 8)), 8), Some(14));
        // A different thread count recalibrates.
        assert_eq!(threshold_to_reuse(false, Some(cached(14, 8)), 32), None);
        // No cache recalibrates.
        assert_eq!(threshold_to_reuse(false, None, 8), None);
    }

    #[test]
    fn cache_round_trips_through_disk() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("parallel_threshold.json");
        let original = cached(16, 12);
        write_cache_to(&path, &original).expect("temp dir is writable");
        let read = read_cache_from(&path).expect("just-written cache parses");
        assert_eq!(read, original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_cache_rejects_wrong_schema() {
        let dir = temp_dir("schema");
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, br#"{"schema":999,"threshold":14,"num_threads":8}"#).unwrap();
        assert_eq!(read_cache_from(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_cache_tolerates_missing_and_garbage() {
        let missing = temp_dir("missing").join("none.json");
        assert_eq!(read_cache_from(&missing), None);

        let dir = temp_dir("garbage");
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, b"not json at all").unwrap();
        assert_eq!(read_cache_from(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_cache_errors_when_directory_is_unwritable() {
        // The parent is an existing *file*, so `create_dir_all` must fail: this
        // is the read-only-filesystem / container case, surfaced as an `Err`
        // (never a panic) that `calibrate_and_cache` turns into
        // `cache_written = false`.
        let file = temp_dir("file");
        std::fs::write(&file, b"x").unwrap();
        let path = file.join("sub").join("parallel_threshold.json");
        assert!(write_cache_to(&path, &cached(14, 8)).is_err());
        let _ = std::fs::remove_file(&file);
    }
}
