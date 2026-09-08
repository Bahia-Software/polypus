//! Machine-aware calibration of the gate-parallelism threshold.
//!
//! [`DEFAULT_PARALLEL_THRESHOLD`](crate::DEFAULT_PARALLEL_THRESHOLD) is a single
//! hardware-oblivious constant, but the qubit count at which spreading one gate
//! across a rayon pool actually beats running it sequentially depends on the
//! *machine* — core count, memory bandwidth, cache — not just on `n`. Pinning it
//! at a constant makes the native simulator go parallel too late on a big node
//! (measurably slower than Qiskit Aer around n = 12–18) or too eagerly on a
//! small one. This module measures that crossover, caches it on disk keyed by
//! rayon thread count — one entry per thread count the machine is calibrated at,
//! since a single SLURM node hands its jobs different CPU allotments — and reuses
//! it across processes, the same "wisdom" mechanism FFTW uses.
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
//! and, when it holds no entry for this thread count, degrades to the static
//! default rather than stalling a user's first circuit for a second timing kernels.
//! That keeps the fallback exactly as fast as today's behaviour and makes a
//! calibration the caller's deliberate choice (e.g. the first line of a SLURM
//! script, before any circuit runs).
//!
//! ## How the crossover is measured
//!
//! Each *session* sweeps [`CALIBRATION_SIZES`], timing a small fixed sequence
//! that exercises **every** kernel family once — dense and diagonal 1-qubit,
//! diagonal and dense 2-qubit, controlled 1-qubit (see [`apply_probe_sequence`])
//! — rather than one gate repeated. A real circuit dispatches to all five
//! kernels (a QFT, for instance, is dominated by the diagonal 2-qubit `cp`, not
//! by the 1-qubit `H` the probe used to time), so measuring a single gate gave a
//! crossover for the wrong kernel.
//!
//! Robustness comes in two layers, both a median. *Within* a session, every
//! `(size, path)` timing is the median of [`CALIBRATION_SAMPLES`] samples.
//! *Across* sessions, [`calibrate_parallel_threshold`] runs
//! [`CALIBRATION_SESSIONS`] independent sessions and takes the median of the
//! thresholds they chose — the same statistical principle one level up, so a
//! single noisy session cannot swing the result. That extra layer is what let
//! the per-session margin ([`MIN_SPEEDUP`]) shrink from a wide 25% — needed when
//! one session was the whole decision — to a tighter 15%, without a slightly
//! unlucky run latching parallelism on. Time budget: `CALIBRATION_SESSIONS`
//! sessions × well under a second each stays comfortably below the ~10s that
//! passes unnoticed inside `install.sh` (measured, not assumed — see the crate
//! benchmarks).
//!
//! ## Visibility
//!
//! Diagnostics go through the `log` facade (this crate never depends on
//! `polypus-logger`) and stay silent until the application installs a sink —
//! from Python, `polypus.init_logger()` once per process.

use std::collections::HashMap;
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

/// Parallel is chosen (within one session) only when it is at least this many
/// times faster than sequential — a margin, not a photo finish: the crossover
/// must pay for the pool overhead *and* leave headroom, or a slightly noisy tie
/// would latch parallelism on and cost time on every later gate. `1.15` ⇒ 15%
/// faster. This is tighter than the earlier 25% because a single session is no
/// longer the whole decision — [`CALIBRATION_SESSIONS`] sessions are combined by
/// median (see [`calibrate_parallel_threshold`]), so one unlucky near-tie can no
/// longer swing the result and the margin need not absorb that risk alone.
const MIN_SPEEDUP: f64 = 1.15;

/// Independent calibration sessions whose chosen thresholds are combined by
/// **median** (see [`median_threshold`]). Odd for the same reason
/// [`CALIBRATION_SAMPLES`] is: the median is then one real session's decision,
/// not an interpolation. `3` is enough for one noisy session to be outvoted
/// while keeping the total time budget well under `install.sh`'s ~10s window.
const CALIBRATION_SESSIONS: usize = 3;

/// Schema version of the on-disk cache. A file written by a different layout is
/// ignored (treated as absent) rather than mis-parsed. Bumped to `2` when the
/// payload changed from a single flat `(threshold, num_threads)` to a
/// thread-count → threshold map: a `schema = 1` file is therefore treated as
/// absent, costing at worst one ~1s recalibration of the current size — no
/// migration path is warranted for a cache that is cheap to rebuild.
const CACHE_SCHEMA: u32 = 2;

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
///
/// [`entries`](Self::entries) maps a rayon thread count to the gate-parallel
/// threshold calibrated for it, e.g. `{"8": 16, "32": 18}` on disk. One entry
/// per distinct thread count the machine has been calibrated at — a single
/// SLURM node hands its jobs different CPU allotments (`--cpus-per-task`), and
/// `rayon::current_num_threads()` honours the cgroup limit, so the *same* node
/// legitimately calibrates several thread counts. A flat single-entry payload
/// (the `schema = 1` layout) would let each new job size overwrite the last, so
/// every size change would look like a hardware change and fall back to the
/// static default; a map keeps every size's result side by side instead.
///
/// **Known limitation (single-node assumption):** the only key is the thread
/// count. If this cluster ever grew heterogeneous nodes, two different nodes that
/// happened to run with the same thread count would share — and clobber — one
/// entry, since nothing here distinguishes them. There is exactly one node today
/// (`sinfo` confirms), so this is not resolved; a richer fingerprint (CPU model,
/// cache sizes) would be the fix if that changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachedCalibration {
    schema: u32,
    /// Rayon thread count → calibrated gate-parallel threshold.
    entries: HashMap<usize, usize>,
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

/// Median of the thresholds chosen by several independent sessions (sorted in
/// place). Pure over its input, so the cross-session combination is unit-tested
/// without the clock — the outer analogue of [`median`] over samples. Never
/// called with an empty slice ([`CALIBRATION_SESSIONS`] `>= 1`).
fn median_threshold(thresholds: &mut [usize]) -> usize {
    thresholds.sort_unstable();
    thresholds[thresholds.len() / 2]
}

/// Apply one instance of the representative probe sequence — exactly one gate
/// per kernel family — so calibration times the *mix* of kernels a real circuit
/// dispatches to, not a single gate repeated. The dispatch table in
/// [`statevector`](crate::statevector) routes every `GateInstruction` to one of
/// five kernel functions; this sequence hits each once:
///
/// * `apply_1q`            — dense 1-qubit    → `H` on q0
/// * `apply_diagonal_1q`   — diagonal 1-qubit → `Z` on q0
/// * `apply_diagonal_2q`   — diagonal 2-qubit → `Cz` on (q0, q1)
/// * `apply_controlled_1q` — controlled 1-qubit → `Cx` (control q1, target q0)
/// * `apply_2q`            — dense 2-qubit    → `Swap` on (q0, q1)
///
/// The gates are chosen only as representatives of their kernel, not tied to any
/// named algorithm: the point is to characterise the machine for *anything* that
/// runs, not just QFT/Hadamards. Every gate is unitary, so applying the sequence
/// repeatedly across samples keeps the buffer normalised (`|amp| <= 1`) with no
/// drift to renormalise — the same property the old `H`-only probe leaned on
/// (`H·H = I`). Needs `n >= 2` for the two-qubit kernels, which every
/// [`CALIBRATION_SIZES`] entry satisfies. Kernels are called directly with the
/// explicit `parallel` flag so the two paths are measured in isolation, without
/// routing through `Statevector` and the very threshold being calibrated (which
/// would add instruction-decode overhead and contaminate the pure kernel timing).
fn apply_probe_sequence(data: &mut [C64], n: usize, parallel: bool) {
    kernels::apply_1q(data, n, 0, &gates::h(), parallel);
    let (z0, z1) = gates::z();
    kernels::apply_diagonal_1q(data, 0, z0, z1, parallel);
    kernels::apply_diagonal_2q(data, 0, 1, gates::cz_diag(), parallel);
    kernels::apply_controlled_1q(data, n, 1, 0, &gates::x(), parallel);
    kernels::apply_2q(data, n, 0, 1, &gates::swap(), parallel);
}

/// Median (per-sequence) nanoseconds to apply the representative probe sequence
/// ([`apply_probe_sequence`]) to a `2^n` buffer on the given path. One buffer is
/// reused across samples; the sequence is unitary, so amplitudes stay bounded
/// with no drift to renormalise between samples.
fn measure_median_ns(n: usize, parallel: bool) -> f64 {
    let dim = 1usize << n;
    let mut data = vec![C64::new(0.0, 0.0); dim];
    data[0] = C64::new(1.0, 0.0);
    let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
    for _ in 0..CALIBRATION_SAMPLES {
        samples.push(sample_sequence_ns(&mut data, n, parallel));
    }
    median(&mut samples)
}

/// One timing sample: apply the probe sequence enough times to fill
/// [`SAMPLE_WINDOW`], then return nanoseconds per sequence. The repeat count is
/// discovered by doubling, so a single window constant fits every size and
/// machine. Both paths use the same unit (ns per sequence), so the crossover
/// decision in [`select_threshold`] — a ratio — is unaffected by the change from
/// per-gate to per-sequence timing.
fn sample_sequence_ns(data: &mut [C64], n: usize, parallel: bool) -> f64 {
    let mut reps: u32 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            apply_probe_sequence(data, n, parallel);
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

/// One calibration session: probe [`CALIBRATION_SIZES`] on both paths (median of
/// [`CALIBRATION_SAMPLES`] samples each, over the representative
/// [`apply_probe_sequence`]) and pick the crossover. This is the unit the outer
/// cross-session median in [`calibrate_parallel_threshold`] is taken over.
fn measure_session_threshold() -> usize {
    let timings: Vec<SizeTiming> = CALIBRATION_SIZES
        .iter()
        .map(|&n| SizeTiming {
            n,
            seq_ns: measure_median_ns(n, false),
            par_ns: measure_median_ns(n, true),
        })
        .collect();
    select_threshold(&timings)
}

/// Measure the gate-parallelism crossover on this machine.
///
/// Runs `CALIBRATION_SESSIONS` independent sessions — each probing
/// `CALIBRATION_SIZES` with the representative kernel sequence
/// (`apply_probe_sequence`, median of `CALIBRATION_SAMPLES` samples per
/// point) and choosing the smallest size where parallel is at least
/// `MIN_SPEEDUP`× faster — and returns the **median** of the thresholds they
/// chose, so one noisy session cannot swing the result (a parallelism-disabling
/// threshold is returned when the sessions agree nothing wins). **Pure
/// measurement**: it does not read or write the cache (see [`calibrate_and_cache`])
/// and, with the default parameters, takes well under `install.sh`'s ~10s window.
///
/// Diagnostics go through the `log` facade and need `polypus.init_logger()` to
/// be visible.
pub fn calibrate_parallel_threshold() -> CalibrationResult {
    let start = Instant::now();
    let mut thresholds: Vec<usize> = (0..CALIBRATION_SESSIONS)
        .map(|_| measure_session_threshold())
        .collect();
    let threshold = median_threshold(&mut thresholds);
    let num_threads = rayon::current_num_threads();
    let duration = start.elapsed();
    log::info!(
        "calibrated gate-parallel threshold = {threshold} qubit(s) for {num_threads} thread(s) \
         over {CALIBRATION_SESSIONS} session(s) in {duration:?}"
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

/// Fold a freshly measured `num_threads → threshold` result into whatever the
/// cache already holds, inserting or updating **only** that thread count's entry
/// and preserving every other. `existing = None` — no cache, or one whose schema
/// this build does not understand (e.g. the flat `schema = 1` layout) — starts a
/// fresh map. Pure over its inputs, so the "calibrating size B must not drop size
/// A" invariant is unit-tested without touching the clock or disk.
fn merged_cache(
    existing: Option<CachedCalibration>,
    num_threads: usize,
    threshold: usize,
) -> CachedCalibration {
    let mut entries = existing.map(|c| c.entries).unwrap_or_default();
    entries.insert(num_threads, threshold);
    CachedCalibration {
        schema: CACHE_SCHEMA,
        entries,
    }
}

/// Why the runtime resolver fell back to the static
/// `DEFAULT_PARALLEL_THRESHOLD` instead of a calibrated value.
///
/// Public because a caller outside this crate (the `polypus` bindings layer)
/// surfaces it to the user — see [`resolved_fallback_reason`], which returns
/// `Some(reason)` on a fallback and `None` when the threshold came from a cache
/// valid for this hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// No usable calibration cache was found for this machine at all — the file
    /// is absent, unreadable, of an unknown schema, or holds no entries. This is
    /// the *serious* case (`warn!` + a Python `UserWarning`): the machine has
    /// never been calibrated. Contrast a cache that simply lacks an entry for the
    /// current thread count while holding others — a routine event on a shared
    /// node whose jobs get different CPU allotments — which is only logged at
    /// `info!` and is **not** reported as a fallback (see `resolution`).
    NotCalibrated,
    /// A cache existed but was made for a different thread count.
    ///
    /// **Currently unreachable** under the on-disk map layout: since the cache
    /// keys thresholds by thread count, a mismatched thread count is simply an
    /// absent entry (handled as the informational size-uncalibrated case), not a
    /// fallback. The variant is retained because it is part of the crate's public
    /// surface and because a future, richer hardware fingerprint (see the
    /// single-node limitation on `CachedCalibration`) would revive a genuine
    /// "same thread count, different machine" mismatch that belongs here.
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
    /// The cache holds thresholds for other thread counts but none for the
    /// current one — routine on a shared node whose jobs get different CPU
    /// allotments. The static default is used, but this is informational
    /// (`info!`), not a warnable fallback: see [`resolution`].
    SizeUncalibrated,
    Fallback(FallbackReason),
}

/// Decide the threshold from a (possibly absent) cache and the current thread
/// count: use the entry calibrated for this thread count when the map has one.
///
/// A cache with entries but none for `current_threads` is
/// [`SizeUncalibrated`](Decision::SizeUncalibrated) (a new job size, routine),
/// distinct from a cache that is absent or holds no entries at all
/// ([`NotCalibrated`](FallbackReason::NotCalibrated), the serious case).
fn decide(cache: Option<CachedCalibration>, current_threads: usize) -> Decision {
    match cache {
        Some(c) => match c.entries.get(&current_threads) {
            Some(&threshold) => Decision::Cached(threshold),
            None if c.entries.is_empty() => Decision::Fallback(FallbackReason::NotCalibrated),
            None => Decision::SizeUncalibrated,
        },
        None => Decision::Fallback(FallbackReason::NotCalibrated),
    }
}

/// The once-per-process resolution: the threshold in force, plus *why* it was
/// chosen (`None` = no warnable fallback — either a cache entry valid for this
/// thread count, or the routine informational case where other thread counts are
/// calibrated but this one is not; `Some` = the serious fallback to the static
/// default, with the reason).
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
/// It reads the on-disk cache and looks up the current thread count
/// (`rayon::current_num_threads()`) in it:
/// * entry present → the calibrated threshold, no fallback;
/// * entries exist but not for this thread count → [`DEFAULT_PARALLEL_THRESHOLD`]
///   plus an `info!` and **no** fallback reason. A shared SLURM node serves jobs
///   sized by `--cpus-per-task`, so a not-yet-seen thread count is routine, not a
///   problem worth a default-visible warning — the caller runs
///   `polypus.calibrate_parallel_threshold()` per job to fill it in;
/// * no cache at all (absent, unreadable, unknown schema, or empty) →
///   [`DEFAULT_PARALLEL_THRESHOLD`] plus a `warn!` and
///   [`FallbackReason::NotCalibrated`] — the serious case, reserved for a machine
///   that has never been calibrated.
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
            Decision::SizeUncalibrated => {
                // Routine on a shared node: other thread counts are calibrated,
                // this one is simply new. Informational only — no warnable
                // fallback, so the bindings raise no Python warning for it.
                log::info!(
                    "no gate-parallel calibration cached for {current} thread(s) (other thread \
                     counts are calibrated); using the default threshold \
                     {DEFAULT_PARALLEL_THRESHOLD}. Run polypus.calibrate_parallel_threshold() to \
                     tune this thread count too (requires polypus.init_logger() to be visible)."
                );
                Resolution {
                    threshold: DEFAULT_PARALLEL_THRESHOLD,
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
                    FallbackReason::NotCalibrated => log::warn!(
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

/// Why the process-wide threshold fell back to the static default in a way worth
/// surfacing, or `None` when there is nothing to warn about — either a cache
/// entry valid for this thread count, or the routine case where the machine is
/// calibrated for other thread counts but not this one (informational only).
///
/// Shares the same once-per-process resolution as `resolve_threshold` (reading
/// it here triggers that resolution if it has not happened yet), so the reason
/// always matches the threshold actually in force. Intended for the `polypus`
/// bindings, which turn a `Some(..)` into a default-visible Python warning.
pub fn resolved_fallback_reason() -> Option<FallbackReason> {
    resolution().fallback
}

/// The threshold to reuse without measuring, or `None` if a fresh measurement is
/// required. Reuse is per thread count: only the entry calibrated for
/// `current_threads` counts as a hit, so a new job size on an already-calibrated
/// node recalibrates just that size. Pure, so the `force`/reuse logic is
/// unit-tested directly.
fn threshold_to_reuse(
    force: bool,
    cache: Option<CachedCalibration>,
    current_threads: usize,
) -> Option<usize> {
    if force {
        return None;
    }
    cache?.entries.get(&current_threads).copied()
}

/// Calibrate if needed and persist the result — the entry point behind
/// `polypus.calibrate_parallel_threshold(force=...)` and `install.sh`.
///
/// With `force = false` a cache entry already valid for the current thread count
/// is reused as-is: nothing is measured and `duration` is zero. Otherwise the
/// crossover is measured and merged into the cache — the freshly measured thread
/// count's entry is inserted or updated while every other thread count's entry is
/// preserved (see `merged_cache`), so calibrating one job size never drops
/// another's. A cache that cannot be written (read-only filesystem, container,
/// CI) is **not** an error: the measured threshold is still returned, with
/// `cache_written = false`, so this process runs calibrated even though the next
/// one will not benefit.
///
/// Intended to run **before any simulation** — e.g. the first line of a SLURM
/// script — because the per-process resolver (`resolve_threshold`) reads the
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
    // Re-read and merge so this thread count's entry is updated in place without
    // clobbering the thresholds calibrated for other thread counts on this node.
    let cached = merged_cache(read_cache(), result.num_threads, result.threshold);
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

    /// Build a cache from `(num_threads, threshold)` entries — the on-disk map,
    /// spelled out for a test.
    fn cached(entries: &[(usize, usize)]) -> CachedCalibration {
        CachedCalibration {
            schema: CACHE_SCHEMA,
            entries: entries.iter().copied().collect(),
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
        // A win below the 15% margin (here ~10% faster) must NOT latch
        // parallelism on: it would not pay for the pool overhead with headroom.
        let timings = vec![SizeTiming {
            n: 10,
            seq_ns: 100.0,
            par_ns: 91.0,
        }];
        assert_eq!(select_threshold(&timings), PARALLELISM_DISABLED);
    }

    #[test]
    fn select_threshold_latches_just_past_the_margin() {
        // A win just *past* the 15% margin (here ~16.6% faster) does latch: this
        // pins the tighter margin down from the winning side, complementing the
        // near-tie test above.
        let timings = vec![SizeTiming {
            n: 12,
            seq_ns: 100.0,
            par_ns: 85.0,
        }];
        assert_eq!(select_threshold(&timings), 12);
    }

    #[test]
    fn median_threshold_returns_the_middle_session_decision() {
        // Odd count: the median is one real session's choice, order-independent.
        assert_eq!(median_threshold(&mut [12, 18, 14]), 14);
        assert_eq!(median_threshold(&mut [14, 12, 18]), 14);
        // Unanimous sessions return that value.
        assert_eq!(median_threshold(&mut [16, 16, 16]), 16);
        // A single dissenting session (e.g. one that disabled parallelism) is
        // outvoted by the two that agreed.
        assert_eq!(median_threshold(&mut [12, 12, PARALLELISM_DISABLED]), 12);
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
    fn decide_uses_the_entry_for_the_current_thread_count() {
        let cache = cached(&[(8, 16), (32, 18)]);
        // A thread count with an entry uses it.
        assert_eq!(decide(Some(cache.clone()), 8), Decision::Cached(16));
        assert_eq!(decide(Some(cache.clone()), 32), Decision::Cached(18));
        // A thread count absent from a populated cache is the routine
        // size-uncalibrated case, NOT a warnable fallback.
        assert_eq!(decide(Some(cache), 16), Decision::SizeUncalibrated);
        // A cache with no entries at all, and no cache at all, are both the
        // serious "never calibrated" fallback.
        assert_eq!(
            decide(Some(cached(&[])), 8),
            Decision::Fallback(FallbackReason::NotCalibrated)
        );
        assert_eq!(
            decide(None, 8),
            Decision::Fallback(FallbackReason::NotCalibrated)
        );
    }

    #[test]
    fn reuse_requires_no_force_and_an_entry_for_the_current_thread_count() {
        let cache = cached(&[(8, 14)]);
        // force=True always recalibrates, even with a perfectly valid entry.
        assert_eq!(threshold_to_reuse(true, Some(cache.clone()), 8), None);
        // An entry for the current thread count reuses without measuring.
        assert_eq!(threshold_to_reuse(false, Some(cache.clone()), 8), Some(14));
        // A thread count without an entry recalibrates (only that size).
        assert_eq!(threshold_to_reuse(false, Some(cache), 32), None);
        // No cache recalibrates.
        assert_eq!(threshold_to_reuse(false, None, 8), None);
    }

    #[test]
    fn merged_cache_updates_one_entry_and_preserves_the_rest() {
        // The direct regression test for the reported bug: calibrating thread
        // count B must not drop the entry calibrated for A. Calibrate A (8→16),
        // then B (32→18), and confirm A is still recoverable.
        let after_a = merged_cache(None, 8, 16);
        let after_b = merged_cache(Some(after_a), 32, 18);
        assert_eq!(decide(Some(after_b.clone()), 8), Decision::Cached(16));
        assert_eq!(decide(Some(after_b.clone()), 32), Decision::Cached(18));

        // Recalibrating A (force, same thread count) updates only A's entry and
        // leaves B untouched.
        let after_a_again = merged_cache(Some(after_b), 8, 20);
        assert_eq!(decide(Some(after_a_again.clone()), 8), Decision::Cached(20));
        assert_eq!(decide(Some(after_a_again), 32), Decision::Cached(18));
    }

    #[test]
    fn merged_cache_starts_fresh_from_an_unreadable_or_absent_cache() {
        // A `None` (absent, or an incompatible older schema `read_cache` rejected)
        // yields a single-entry map, never an error and never a lost write.
        let fresh = merged_cache(None, 8, 16);
        assert_eq!(decide(Some(fresh), 8), Decision::Cached(16));
    }

    #[test]
    fn cache_round_trips_through_disk() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("parallel_threshold.json");
        let original = cached(&[(12, 16), (8, 14)]);
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
        std::fs::write(&path, br#"{"schema":999,"entries":{"8":14}}"#).unwrap();
        assert_eq!(read_cache_from(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_cache_treats_the_old_flat_schema_as_absent() {
        // A `schema = 1` file (the pre-map flat layout) must be treated as absent,
        // not migrated and not an error: the worst case is one ~1s recalibration
        // of the current size. Both the schema guard and the shape mismatch reject
        // it; either way the result is `None`.
        let dir = temp_dir("flat-schema");
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, br#"{"schema":1,"threshold":18,"num_threads":32}"#).unwrap();
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
        assert!(write_cache_to(&path, &cached(&[(8, 14)])).is_err());
        let _ = std::fs::remove_file(&file);
    }
}
