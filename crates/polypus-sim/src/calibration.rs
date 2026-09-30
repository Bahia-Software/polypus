//! Machine-aware calibration of the gate-parallelism threshold.
//!
//! [`DEFAULT_PARALLEL_THRESHOLD`](crate::DEFAULT_PARALLEL_THRESHOLD) is a single
//! hardware-oblivious constant, but the qubit count at which spreading one gate
//! across a rayon pool actually beats running it sequentially depends on the
//! *machine* — core count, memory bandwidth, cache — not just on `n`. Pinning it
//! at a constant makes the native simulator go parallel too late on a big node
//! (measurably slower than Qiskit Aer around n = 12–18) or too eagerly on a
//! small one. This module measures that crossover, caches it on disk keyed by
//! rayon thread count *and* a CPU fingerprint — one entry per (hardware, thread
//! count) pair, since a single SLURM node hands its jobs different CPU allotments
//! and a cluster with a shared `$HOME` hands the same cache file to different
//! nodes — and reuses it across processes, the same "wisdom" mechanism FFTW uses.
//!
//! ## Concurrent writers
//!
//! Many processes may calibrate against one cache file at once (dozens of SLURM
//! ranks importing `polypus` on a shared `$HOME`). Every write is atomic — a
//! temp file in the cache directory renamed over the old one — so a reader sees
//! either the previous or the new complete file, never a torn one, and needs no
//! lock. The read-merge-write that folds one entry into the map additionally
//! runs under an exclusive advisory lock on a sibling `.lock` file, so two
//! writers cannot both read the old map and drop each other's entry. See
//! [`persist_entry`]. On network filesystems that lock is only as good as the
//! mount's lock support (see [`acquire_cache_lock`]); where it is missing, the
//! atomic rename still holds and the worst case is one lost entry, never a
//! corrupt file.
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

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
///
/// Bumped to `3` when the map key changed from a bare thread count (`"8"`) to a
/// hardware-qualified key (`"8|<cpu fingerprint>"`, see [`cache_key`]). The JSON
/// shape is unchanged, but a `schema = 2` entry does not say which hardware it
/// measured — on a shared `$HOME` it may be another node's — so it cannot be
/// trusted under the new meaning and is dropped rather than carried along as a
/// key nothing will ever match again. Same cost as the 1→2 bump: at worst one
/// recalibration per (hardware, thread count).
const CACHE_SCHEMA: u32 = 3;

/// Longest a writer waits for the cache lock before giving up on it (see
/// [`acquire_cache_lock`]). The lock only covers a read-merge-write of a few
/// hundred bytes — never the measurement — so even dozens of queued ranks clear
/// it in milliseconds; a wait this long means a stale lock (e.g. an NFS lock
/// server that lost track of a dead client), and a calibration must not hang an
/// `import polypus` on that.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);

/// Pause between non-blocking lock attempts while waiting for [`LOCK_TIMEOUT`].
const LOCK_POLL: Duration = Duration::from_millis(10);

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
    /// `rayon::current_num_threads()` observed during calibration — one half of
    /// the cache key, alongside the CPU fingerprint (see `cache_key`).
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
    /// Rayon threads detected — one half of the cache key, alongside the CPU
    /// fingerprint (see `cache_key`).
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
/// [`entries`](Self::entries) maps a hardware-qualified key ([`cache_key`]: the
/// rayon thread count plus a CPU fingerprint) to the gate-parallel threshold
/// calibrated for it, e.g.
/// `{"8|Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz|49152 KB|2": 16}` on disk.
///
/// The thread count is in the key because a single SLURM node hands its jobs
/// different CPU allotments (`--cpus-per-task`), and `rayon::current_num_threads()`
/// honours the cgroup limit, so the *same* node legitimately calibrates several
/// thread counts. A flat single-entry payload (the `schema = 1` layout) would let
/// each new job size overwrite the last; a map keeps every size's result side by
/// side instead.
///
/// The CPU fingerprint is in the key because a cluster with a shared `$HOME`
/// hands this one file to every node: two heterogeneous nodes running at the
/// same thread count must not share — and clobber — one entry (the `schema = 2`
/// limitation, keyed by thread count alone).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachedCalibration {
    schema: u32,
    /// [`cache_key`] → calibrated gate-parallel threshold.
    entries: HashMap<String, usize>,
}

/// The canonical cache key for `num_threads` rayon threads on hardware described
/// by `cpu` (a [`cpu_fingerprint`]): `"{num_threads}|{cpu}"`, or the bare thread
/// count when no fingerprint is available (non-Linux targets, an unreadable
/// `/proc/cpuinfo`). That fallback is exactly as specific as the `schema = 2`
/// key, never less — the heterogeneous shared-`$HOME` cluster this guards
/// against is a Linux one. Pure, so the key shape is unit-tested directly.
fn cache_key(num_threads: usize, cpu: Option<&str>) -> String {
    match cpu {
        Some(cpu) => format!("{num_threads}|{cpu}"),
        None => num_threads.to_string(),
    }
}

/// Extract a stable hardware fingerprint from the text of `/proc/cpuinfo`:
/// `"{model}|{cache size}|{sockets}"`.
///
/// * **model** — the first `model name` (x86, most distributions' ARM kernels);
///   failing that, `CPU implementer`/`CPU part` (other ARM kernels), which
///   identify the core design; `?` when neither is present.
/// * **cache size** — the first `cache size` (x86 reports the last-level cache
///   here); `?` when absent.
/// * **sockets** — the number of distinct `physical id` values; `0` when the
///   kernel does not report them (it is only a key component, so an honest
///   "unknown" is better than a guess).
///
/// Whitespace runs are collapsed and `|` is replaced so a field can never forge
/// the key's separator. Returns `None` when nothing identifying is found at all,
/// so [`cache_key`] falls back to the thread count rather than keying every
/// unknown machine as the same `?|?|0` hardware. Pure over its input, so it is
/// unit-tested with canned `cpuinfo` text.
fn parse_cpuinfo(text: &str) -> Option<String> {
    let mut model: Option<String> = None;
    let mut implementer: Option<String> = None;
    let mut part: Option<String> = None;
    let mut cache: Option<String> = None;
    let mut sockets: HashSet<String> = HashSet::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = normalize_field(value);
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "model name" if model.is_none() => model = Some(value),
            "CPU implementer" if implementer.is_none() => implementer = Some(value),
            "CPU part" if part.is_none() => part = Some(value),
            "cache size" if cache.is_none() => cache = Some(value),
            "physical id" => {
                sockets.insert(value);
            }
            _ => {}
        }
    }
    let model = model.or_else(|| match (implementer, part) {
        (Some(implementer), Some(part)) => Some(format!("arm {implementer}:{part}")),
        _ => None,
    });
    if model.is_none() && cache.is_none() && sockets.is_empty() {
        return None;
    }
    Some(format!(
        "{}|{}|{}",
        model.as_deref().unwrap_or("?"),
        cache.as_deref().unwrap_or("?"),
        sockets.len()
    ))
}

/// Trim, collapse internal whitespace runs to one space, and replace the key
/// separator `|`, so one `cpuinfo` value is a canonical key component.
fn normalize_field(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('|', "/")
}

/// This machine's CPU fingerprint ([`parse_cpuinfo`] over `/proc/cpuinfo`),
/// read once per process and memoised. `None` on non-Linux targets and whenever
/// the file cannot be read or identifies nothing — [`cache_key`] then degrades
/// to the thread count alone, never to an error.
fn cpu_fingerprint() -> Option<&'static str> {
    static FINGERPRINT: OnceLock<Option<String>> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            if !cfg!(target_os = "linux") {
                return None;
            }
            let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            parse_cpuinfo(&text)
        })
        .as_deref()
}

/// The cache key for `num_threads` rayon threads on *this* machine.
fn hardware_key(num_threads: usize) -> String {
    cache_key(num_threads, cpu_fingerprint())
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

/// Serialise `cached` to `path` **atomically**. The parent directory must
/// already exist ([`persist_entry`] creates it, once, before taking the lock), so
/// the critical section does no directory work.
///
/// The JSON goes to a uniquely named temp file in the *same* directory (a rename
/// is only atomic within one filesystem, so not `std::env::temp_dir()`), is
/// flushed to disk, and is then renamed over `path`. A concurrent reader
/// therefore sees the previous complete file or the new one, never a truncated
/// or half-written one — which a plain `std::fs::write` (truncate, then write)
/// allowed. Atomicity alone does not stop two writers from losing each other's
/// entry; that is [`persist_entry`]'s lock. Errors (read-only filesystem,
/// container, CI) propagate so the caller can fall back to the default without
/// persistence — a failed write is never fatal — and the temp file is removed
/// on the way out.
fn write_cache_to(path: &Path, cached: &CachedCalibration) -> std::io::Result<()> {
    let json = serde_json::to_vec_pretty(cached).map_err(std::io::Error::other)?;
    let tmp = temp_path_for(path);
    let written = write_new_file(&tmp, &json).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        // Best-effort cleanup: the write error is what the caller needs; a temp
        // file that cannot be removed either (e.g. it was never created) is only
        // worth a debug record, not a second error.
        if let Err(e) = std::fs::remove_file(&tmp) {
            log::debug!(
                "could not remove calibration temp file {}: {e}",
                tmp.display()
            );
        }
    }
    written
}

/// Create `path` (failing if it already exists, so a name collision can never
/// clobber another writer's temp file), write `bytes`, and flush them to disk
/// before the caller renames it into place.
fn write_new_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// A temp-file name next to `path`, unique per process *and* per call: pid,
/// wall-clock nanoseconds and a process-wide counter. The pid alone would not
/// do — two threads of one process write concurrently, and two nodes sharing
/// `$HOME` can run processes with the same pid. The leading dot keeps it out of
/// casual `ls` output should a crash ever strand one.
fn temp_path_for(path: &Path) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .map_or_else(|| "cache".into(), |n| n.to_string_lossy());
    path.with_file_name(format!(".{name}.{}.{nanos}.{seq}.tmp", std::process::id()))
}

/// The sibling lock file guarding `path`'s read-merge-write: `<name>.lock`. A
/// separate file, not the data file itself, because the data file is replaced
/// by rename on every write — a lock held on the old inode would not exclude a
/// writer that opened the new one. It is never deleted: removing a lock file
/// races with a writer that has just opened it.
fn lock_path_for(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    path.with_file_name(name)
}

/// Take an exclusive advisory lock on `lock_path` (flock(2) on Unix, LockFileEx
/// on Windows, via `fs4`), waiting at most `timeout`. The lock is held for as
/// long as the returned `File` lives and released when it is dropped (or when
/// the process dies, so a crashed writer never wedges the cache).
///
/// `None` means "proceed without the lock": the lock file cannot be opened, the
/// filesystem does not support locking (some Lustre/NFS mounts return
/// `ENOLCK`/`ENOSYS`), or `timeout` passed. The caller then still writes — the
/// rename keeps the file whole, so the worst case is the lost update the lock
/// exists to prevent (one entry recalibrated later), never corruption, a hang,
/// or a skipped write that would make every later import recalibrate.
///
/// **Network filesystems — the shared-`$HOME` case this lock exists for.** On
/// Linux NFS the kernel emulates `flock` with byte-range locks forwarded to the
/// server (in-protocol on NFSv4; through NLM on NFSv3, which needs `rpc.lockd`/
/// `rpc.statd` working on client and server), so on a correctly configured mount
/// it does coordinate across nodes. Without a reachable lock daemon the call
/// usually fails with `ENOLCK` — the `None` path above. But on an NFS mount with
/// `nolock` or `local_lock=flock`/`local_lock=all`, or a Lustre mount with
/// `localflock`, `flock` **succeeds while being local to each client**: ranks on
/// one node still exclude each other, two *nodes* do not. That cannot be
/// detected from here (the call reports success), and the outcome is the same
/// as the `None` path: the rename still keeps the file whole, and the worst
/// cross-node result is one lost entry, recalibrated later by the node that
/// lost it. The import-time secondary-rank skip (`python/polypus/__init__.py`)
/// keeps simultaneous cross-node writers rare to begin with.
fn acquire_cache_lock(lock_path: &Path, timeout: Duration) -> Option<File> {
    let file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
    {
        Ok(file) => file,
        Err(e) => {
            log::warn!(
                "could not open calibration cache lock {}: {e}; writing without it",
                lock_path.display()
            );
            return None;
        }
    };
    let deadline = Instant::now() + timeout;
    loop {
        // Fully qualified: std's inherent `File::try_lock` (Rust 1.89, newer
        // than the MSRV) would otherwise shadow the trait method on new toolchains.
        match fs4::FileExt::try_lock(&file) {
            Ok(()) => return Some(file),
            Err(fs4::TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(LOCK_POLL);
            }
            Err(fs4::TryLockError::WouldBlock) => {
                log::warn!(
                    "timed out after {timeout:?} waiting for calibration cache lock {}; \
                     writing without it",
                    lock_path.display()
                );
                return None;
            }
            Err(fs4::TryLockError::Error(e)) => {
                log::warn!(
                    "could not lock calibration cache {} (filesystem without lock support?): \
                     {e}; writing without it",
                    lock_path.display()
                );
                return None;
            }
        }
    }
}

/// Fold `key → threshold` into the cache at `path`, under the cache lock.
///
/// The whole read-merge-write runs while holding the exclusive lock on the
/// sibling `.lock` file ([`acquire_cache_lock`], waiting at most `lock_timeout`),
/// so two processes calibrating at once — two ranks, two nodes on a shared
/// `$HOME` — serialise: the second one reads a map that already holds the
/// first one's entry, instead of both reading the old map and the last rename
/// silently dropping the other's entry. Only this short critical section is
/// locked, never the measurement. Readers ([`read_cache_from`]) take no lock:
/// the atomic rename in [`write_cache_to`] already guarantees them a complete
/// file. The cache directory is created here, before the lock — the lock file
/// lives in it — so nothing inside the critical section touches directories.
/// Errors propagate exactly like [`write_cache_to`]'s.
fn persist_entry(
    path: &Path,
    key: String,
    threshold: usize,
    lock_timeout: Duration,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Held until the end of this function; dropping it releases the lock.
    let _lock = acquire_cache_lock(&lock_path_for(path), lock_timeout);
    let cached = merged_cache(read_cache_from(path), key, threshold);
    write_cache_to(path, &cached)
}

/// Read the cache at its resolved location, or `None` if there is no location.
fn read_cache() -> Option<CachedCalibration> {
    read_cache_from(&cache_file()?)
}

/// Fold a freshly measured `key → threshold` result ([`cache_key`]) into whatever
/// the cache already holds, inserting or updating **only** that key's entry and
/// preserving every other. `existing = None` — no cache, or one whose schema
/// this build does not understand (e.g. the flat `schema = 1` layout, or the
/// thread-count-only `schema = 2`) — starts a fresh map. Pure over its inputs,
/// so the "calibrating B must not drop A" invariant is unit-tested without
/// touching the clock or disk.
fn merged_cache(
    existing: Option<CachedCalibration>,
    key: String,
    threshold: usize,
) -> CachedCalibration {
    let mut entries = existing.map(|c| c.entries).unwrap_or_default();
    entries.insert(key, threshold);
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
    /// **Currently unreachable** under the on-disk map layout: the cache keys
    /// thresholds by (thread count, CPU fingerprint), so a mismatched thread
    /// count *or* different hardware is simply an absent entry (handled as the
    /// informational size-uncalibrated case), not a fallback — on a shared-`$HOME`
    /// cluster, a node seeing only other nodes' entries is as routine as a new
    /// job size. The variant is retained because it is part of the crate's public
    /// surface.
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
    /// The cache holds thresholds for other keys but none for the current one —
    /// routine on a shared node whose jobs get different CPU allotments, and on
    /// a shared-`$HOME` cluster whose other nodes calibrated first. The static
    /// default is used, but this is informational (`info!`), not a warnable
    /// fallback: see [`resolution`].
    SizeUncalibrated,
    Fallback(FallbackReason),
}

/// Decide the threshold from a (possibly absent) cache and the current
/// [`cache_key`]: use the entry calibrated for this key when the map has one.
///
/// A cache with entries but none for `current_key` is
/// [`SizeUncalibrated`](Decision::SizeUncalibrated) (a new job size or a node not
/// yet calibrated, routine), distinct from a cache that is absent or holds no
/// entries at all ([`NotCalibrated`](FallbackReason::NotCalibrated), the serious
/// case).
fn decide(cache: Option<CachedCalibration>, current_key: &str) -> Decision {
    match cache {
        Some(c) => match c.entries.get(current_key) {
            Some(&threshold) => Decision::Cached(threshold),
            None if c.entries.is_empty() => Decision::Fallback(FallbackReason::NotCalibrated),
            None => Decision::SizeUncalibrated,
        },
        None => Decision::Fallback(FallbackReason::NotCalibrated),
    }
}

/// The once-per-process resolution: the threshold in force, plus *why* it was
/// chosen (`None` = no warnable fallback — either a cache entry valid for this
/// hardware and thread count, or the routine informational case where other keys
/// are calibrated but this one is not; `Some` = the serious fallback to the
/// static default, with the reason).
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
/// It reads the on-disk cache and looks up this machine's [`cache_key`] (the
/// current `rayon::current_num_threads()` plus the CPU fingerprint) in it:
/// * entry present → the calibrated threshold, no fallback;
/// * entries exist but not for this key → [`DEFAULT_PARALLEL_THRESHOLD`] plus an
///   `info!` and **no** fallback reason. A shared SLURM node serves jobs sized by
///   `--cpus-per-task`, and a shared `$HOME` serves every node one file, so a
///   not-yet-seen key is routine, not a problem worth a default-visible warning —
///   the caller runs `polypus.calibrate_parallel_threshold()` per job/node to
///   fill it in;
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
        let key = hardware_key(current);
        match decide(read_cache(), &key) {
            Decision::Cached(threshold) => {
                log::debug!(
                    "using cached gate-parallel threshold {threshold} qubit(s) for \
                     {current} thread(s) (cache key {key:?})"
                );
                Resolution {
                    threshold,
                    fallback: None,
                }
            }
            Decision::SizeUncalibrated => {
                // Routine on a shared node or shared $HOME: other thread counts
                // or nodes are calibrated, this one is simply new. Informational
                // only — no warnable fallback, so the bindings raise no Python
                // warning for it.
                log::info!(
                    "no gate-parallel calibration cached for {current} thread(s) on this \
                     hardware (cache key {key:?}; other keys are calibrated); using the default \
                     threshold {DEFAULT_PARALLEL_THRESHOLD}. Run \
                     polypus.calibrate_parallel_threshold() to tune this one too (requires \
                     polypus.init_logger() to be visible)."
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
/// entry valid for this hardware and thread count, or the routine case where
/// other thread counts or nodes are calibrated but not this one (informational
/// only).
///
/// Shares the same once-per-process resolution as `resolve_threshold` (reading
/// it here triggers that resolution if it has not happened yet), so the reason
/// always matches the threshold actually in force. Intended for the `polypus`
/// bindings, which turn a `Some(..)` into a default-visible Python warning.
pub fn resolved_fallback_reason() -> Option<FallbackReason> {
    resolution().fallback
}

/// The threshold to reuse without measuring, or `None` if a fresh measurement is
/// required. Reuse is per [`cache_key`]: only the entry calibrated for
/// `current_key` counts as a hit, so a new job size on an already-calibrated
/// node — or a node another one calibrated beside on a shared `$HOME` —
/// recalibrates just that key. Pure, so the `force`/reuse logic is unit-tested
/// directly.
fn threshold_to_reuse(
    force: bool,
    cache: Option<CachedCalibration>,
    current_key: &str,
) -> Option<usize> {
    if force {
        return None;
    }
    cache?.entries.get(current_key).copied()
}

/// Calibrate if needed and persist the result — the entry point behind
/// `polypus.calibrate_parallel_threshold(force=...)` and `install.sh`.
///
/// With `force = false` a cache entry already valid for the current hardware and
/// thread count is reused as-is: nothing is measured and `duration` is zero.
/// Otherwise the crossover is measured and merged into the cache — the freshly
/// measured key's entry is inserted or updated while every other key's entry is
/// preserved (see `merged_cache`), so calibrating one job size or node never
/// drops another's, even when several processes calibrate at once (the merge is
/// lock-protected and the write atomic, see `persist_entry`). This runs on
/// whichever process calls it, including a secondary MPI/SLURM rank: skipping
/// those is the job of the *implicit* import-time auto-calibration in
/// `python/polypus/__init__.py`, not of this explicit entry point. A cache that
/// cannot be written (read-only filesystem, container,
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

    if let Some(threshold) = threshold_to_reuse(force, read_cache(), &hardware_key(current)) {
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
    // Re-read and merge under the cache lock so this key's entry is updated in
    // place without clobbering the thresholds calibrated for other thread counts
    // or nodes — including ones another process wrote while we were measuring.
    let key = hardware_key(result.num_threads);
    let cache_written = match cache_file() {
        Some(path) => match persist_entry(&path, key, result.threshold, LOCK_TIMEOUT) {
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

    /// Build a cache from `(key, threshold)` entries — the on-disk map, spelled
    /// out for a test.
    fn cached(entries: &[(&str, usize)]) -> CachedCalibration {
        CachedCalibration {
            schema: CACHE_SCHEMA,
            entries: entries.iter().map(|&(k, t)| (k.to_owned(), t)).collect(),
        }
    }

    /// Two fingerprinted keys with the *same* thread count on different CPUs —
    /// the heterogeneous shared-`$HOME` pair the schema-3 key must keep apart.
    const XEON_8: &str = "8|Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz|49152 KB|2";
    const EPYC_8: &str = "8|AMD EPYC 7763 64-Core Processor|512 KB|2";
    const XEON_32: &str = "32|Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz|49152 KB|2";

    /// A two-socket x86 `/proc/cpuinfo` excerpt (two logical CPUs shown, one per
    /// socket), with the kernel's padding kept verbatim.
    const CPUINFO_X86_2S: &str = "\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz
cache size\t: 49152 KB
physical id\t: 0
core id\t\t: 0

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz
cache size\t: 49152 KB
physical id\t: 1
core id\t\t: 0
";

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
    fn decide_uses_the_entry_for_the_current_key() {
        let cache = cached(&[(XEON_8, 16), (XEON_32, 18)]);
        // A key with an entry uses it.
        assert_eq!(decide(Some(cache.clone()), XEON_8), Decision::Cached(16));
        assert_eq!(decide(Some(cache.clone()), XEON_32), Decision::Cached(18));
        // A key absent from a populated cache — here the same thread count on
        // *different hardware* — is the routine size-uncalibrated case, NOT a
        // warnable fallback, and never borrows the other CPU's threshold.
        assert_eq!(decide(Some(cache), EPYC_8), Decision::SizeUncalibrated);
        // A cache with no entries at all, and no cache at all, are both the
        // serious "never calibrated" fallback.
        assert_eq!(
            decide(Some(cached(&[])), XEON_8),
            Decision::Fallback(FallbackReason::NotCalibrated)
        );
        assert_eq!(
            decide(None, XEON_8),
            Decision::Fallback(FallbackReason::NotCalibrated)
        );
    }

    #[test]
    fn reuse_requires_no_force_and_an_entry_for_the_current_key() {
        let cache = cached(&[(XEON_8, 14)]);
        // force=True always recalibrates, even with a perfectly valid entry.
        assert_eq!(threshold_to_reuse(true, Some(cache.clone()), XEON_8), None);
        // An entry for the current key reuses without measuring.
        assert_eq!(
            threshold_to_reuse(false, Some(cache.clone()), XEON_8),
            Some(14)
        );
        // A thread count without an entry recalibrates (only that size) ...
        assert_eq!(
            threshold_to_reuse(false, Some(cache.clone()), XEON_32),
            None
        );
        // ... and so does the same thread count on different hardware.
        assert_eq!(threshold_to_reuse(false, Some(cache), EPYC_8), None);
        // No cache recalibrates.
        assert_eq!(threshold_to_reuse(false, None, XEON_8), None);
    }

    #[test]
    fn merged_cache_updates_one_entry_and_preserves_the_rest() {
        // Calibrating key B must not drop the entry calibrated for A: calibrate
        // A (8 threads on a Xeon → 16), then B (8 threads on an EPYC → 18, the
        // clobbering pair under the old thread-count-only key), and confirm both.
        let after_a = merged_cache(None, XEON_8.to_owned(), 16);
        let after_b = merged_cache(Some(after_a), EPYC_8.to_owned(), 18);
        assert_eq!(decide(Some(after_b.clone()), XEON_8), Decision::Cached(16));
        assert_eq!(decide(Some(after_b.clone()), EPYC_8), Decision::Cached(18));

        // Recalibrating A (force, same key) updates only A's entry and leaves B
        // untouched.
        let after_a_again = merged_cache(Some(after_b), XEON_8.to_owned(), 20);
        assert_eq!(
            decide(Some(after_a_again.clone()), XEON_8),
            Decision::Cached(20)
        );
        assert_eq!(decide(Some(after_a_again), EPYC_8), Decision::Cached(18));
    }

    #[test]
    fn merged_cache_starts_fresh_from_an_unreadable_or_absent_cache() {
        // A `None` (absent, or an incompatible older schema `read_cache` rejected)
        // yields a single-entry map, never an error and never a lost write.
        let fresh = merged_cache(None, XEON_8.to_owned(), 16);
        assert_eq!(decide(Some(fresh), XEON_8), Decision::Cached(16));
    }

    #[test]
    fn cache_key_qualifies_the_thread_count_with_the_cpu_fingerprint() {
        assert_eq!(cache_key(8, Some("cpu A|1 KB|1")), "8|cpu A|1 KB|1");
        // Same thread count, different hardware ⇒ different keys: the
        // acceptance criterion the schema-3 key exists for.
        assert_ne!(
            cache_key(8, Some("cpu A|1 KB|1")),
            cache_key(8, Some("cpu B|1 KB|1"))
        );
        // Same hardware, different thread count ⇒ different keys, as before.
        assert_ne!(
            cache_key(8, Some("cpu A|1 KB|1")),
            cache_key(32, Some("cpu A|1 KB|1"))
        );
        // No fingerprint (non-Linux, unreadable /proc/cpuinfo) degrades to the
        // bare thread count — as specific as the schema-2 key, never an error.
        assert_eq!(cache_key(8, None), "8");
    }

    #[test]
    fn parse_cpuinfo_reads_model_cache_and_socket_count() {
        // Padding collapsed, one model/cache taken, two distinct physical ids.
        assert_eq!(
            parse_cpuinfo(CPUINFO_X86_2S).as_deref(),
            Some("Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz|49152 KB|2")
        );
        // One socket reporting many logical CPUs still counts as one socket.
        let one_socket = CPUINFO_X86_2S.replace("physical id\t: 1", "physical id\t: 0");
        assert_eq!(
            parse_cpuinfo(&one_socket).as_deref(),
            Some("Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz|49152 KB|1")
        );
    }

    #[test]
    fn parse_cpuinfo_tells_different_cpus_apart() {
        let epyc = CPUINFO_X86_2S
            .replace(
                "Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz",
                "AMD EPYC 7763 64-Core Processor",
            )
            .replace("49152 KB", "512 KB");
        assert_ne!(parse_cpuinfo(CPUINFO_X86_2S), parse_cpuinfo(&epyc));
    }

    #[test]
    fn parse_cpuinfo_falls_back_to_arm_implementer_and_part() {
        // aarch64 kernels commonly print no `model name`, `cache size` or
        // `physical id`: the core design (implementer:part) identifies it.
        let arm = "processor\t: 0\nBogoMIPS\t: 50.00\nCPU implementer\t: 0x41\n\
                   CPU architecture: 8\nCPU variant\t: 0x3\nCPU part\t: 0xd0c\n";
        assert_eq!(parse_cpuinfo(arm).as_deref(), Some("arm 0x41:0xd0c|?|0"));
    }

    #[test]
    fn parse_cpuinfo_returns_none_when_nothing_identifies_the_cpu() {
        // Empty, or nothing recognisable: no fingerprint, so the key degrades to
        // the thread count instead of lumping every unknown machine together.
        assert_eq!(parse_cpuinfo(""), None);
        assert_eq!(parse_cpuinfo("processor\t: 0\nflags\t: fpu vme\n"), None);
    }

    #[test]
    fn parse_cpuinfo_cannot_forge_the_key_separator() {
        // A `|` inside a field must not shift the key's components.
        let text = "model name\t: weird | cpu\ncache size\t: 1 KB\nphysical id\t: 0\n";
        assert_eq!(parse_cpuinfo(text).as_deref(), Some("weird / cpu|1 KB|1"));
    }

    #[test]
    fn cache_round_trips_through_disk() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        let original = cached(&[(XEON_32, 16), (XEON_8, 14), ("8", 12)]);
        write_cache_to(&path, &original).expect("temp dir is writable");
        let read = read_cache_from(&path).expect("just-written cache parses");
        assert_eq!(read, original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_cache_is_a_rename_that_leaves_no_temp_file() {
        // Overwriting an existing cache goes through a temp file + rename: the
        // result is the new content, and no `.tmp` sibling is left behind.
        let dir = temp_dir("atomic");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        write_cache_to(&path, &cached(&[(XEON_8, 14)])).expect("temp dir is writable");
        write_cache_to(&path, &cached(&[(XEON_8, 16)])).expect("temp dir is writable");
        assert_eq!(read_cache_from(&path), Some(cached(&[(XEON_8, 16)])));
        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("temp dir is readable")
            .map(|e| {
                e.expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["parallel_threshold.json".to_owned()]);
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
    fn read_cache_treats_the_thread_count_only_schema_as_absent() {
        // A `schema = 2` file has the same JSON shape but keys by thread count
        // alone, so its entries may be another node's: treated as absent (one
        // recalibration), never trusted under the hardware-qualified meaning.
        let dir = temp_dir("schema-2");
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, br#"{"schema":2,"entries":{"8":14,"32":18}}"#).unwrap();
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
        // `cache_written = false` — for the raw write (whose temp file cannot be
        // created in a missing directory) and the locked merge alike.
        let file = temp_dir("file");
        std::fs::write(&file, b"x").unwrap();
        let path = file.join("sub").join("parallel_threshold.json");
        assert!(write_cache_to(&path, &cached(&[(XEON_8, 14)])).is_err());
        assert!(persist_entry(&path, XEON_8.to_owned(), 14, LOCK_TIMEOUT).is_err());
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn persist_entry_merges_into_the_existing_cache() {
        let dir = temp_dir("persist");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        let path = dir.join("parallel_threshold.json");
        persist_entry(&path, XEON_8.to_owned(), 16, LOCK_TIMEOUT).expect("writable");
        persist_entry(&path, EPYC_8.to_owned(), 18, LOCK_TIMEOUT).expect("writable");
        assert_eq!(
            read_cache_from(&path),
            Some(cached(&[(XEON_8, 16), (EPYC_8, 18)]))
        );
        // The sibling lock file exists (and is never deleted) next to the cache.
        assert!(lock_path_for(&path).is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lock_is_exclusive_and_times_out_instead_of_hanging() {
        // While one handle holds the lock, a second (independent open, as another
        // process would have) cannot take it and gives up after its timeout
        // rather than blocking forever; once released, it can.
        let dir = temp_dir("lock");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        std::fs::create_dir_all(&dir).unwrap();
        let lock = dir.join("parallel_threshold.json.lock");
        let held = acquire_cache_lock(&lock, LOCK_TIMEOUT).expect("uncontended lock");
        let start = Instant::now();
        assert!(acquire_cache_lock(&lock, Duration::from_millis(50)).is_none());
        assert!(start.elapsed() >= Duration::from_millis(50));
        drop(held);
        assert!(acquire_cache_lock(&lock, LOCK_TIMEOUT).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn persist_entry_still_writes_when_the_lock_cannot_be_taken() {
        // A stuck lock (stale NFS lock, another writer wedged) must degrade to an
        // unlocked atomic write — the entry lands, the file stays whole — never a
        // hang and never a skipped write.
        let dir = temp_dir("stuck-lock");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        let _held = acquire_cache_lock(&lock_path_for(&path), LOCK_TIMEOUT).expect("lock");
        persist_entry(&path, XEON_8.to_owned(), 16, Duration::from_millis(20))
            .expect("degraded write still succeeds");
        assert_eq!(read_cache_from(&path), Some(cached(&[(XEON_8, 16)])));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_writers_never_lose_an_entry_or_corrupt_the_file() {
        // The regression test for the lost-update race: two writers (independent
        // OS threads, each opening its own handles, as two SLURM ranks would)
        // hammer the *same* cache path, each merging its own distinct keys. With
        // an unlocked read-merge-write, both would read the same old map and the
        // last rename would drop the other's entry; under the lock every entry
        // must survive, and the file must parse after every step.
        const PER_WRITER: usize = 50;
        let dir = temp_dir("concurrent");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        let path = dir.join("parallel_threshold.json");
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            for writer in ["rank-a", "rank-b"] {
                let (path, barrier) = (&path, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    for i in 0..PER_WRITER {
                        persist_entry(path, format!("{i}|{writer}"), i, LOCK_TIMEOUT)
                            .expect("temp dir is writable");
                        assert!(
                            read_cache_from(path).is_some(),
                            "cache must parse after every write"
                        );
                    }
                });
            }
        });
        let cache = read_cache_from(&path).expect("final cache parses");
        assert_eq!(cache.entries.len(), 2 * PER_WRITER, "an entry was lost");
        for writer in ["rank-a", "rank-b"] {
            for i in 0..PER_WRITER {
                assert_eq!(cache.entries.get(&format!("{i}|{writer}")), Some(&i));
            }
        }
        // Every temp file was renamed away; only the cache and its lock remain.
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .expect("temp dir is readable")
            .map(|e| {
                e.expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "parallel_threshold.json".to_owned(),
                "parallel_threshold.json.lock".to_owned()
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reader_never_sees_a_half_written_cache() {
        // The regression test for reading the file mid-write: a writer rewrites
        // a multi-kilobyte cache over and over while a reader parses it as fast
        // as it can, taking no lock (as `read_cache_from` does not). With the old
        // truncate-then-write, the reader could see an empty or truncated file;
        // with temp file + rename, every read is a complete, valid cache.
        let dir = temp_dir("reader");
        let _ = std::fs::remove_dir_all(&dir); // debris from an aborted earlier run
        let path = dir.join("parallel_threshold.json");
        std::fs::create_dir_all(&dir).unwrap();
        let big = |t: usize| -> CachedCalibration {
            let mut c = cached(&[]);
            for i in 0..200 {
                c.entries.insert(format!("{i}|{XEON_8}"), t);
            }
            c
        };
        write_cache_to(&path, &big(0)).expect("temp dir is writable");
        let done = std::sync::atomic::AtomicBool::new(false);
        let reads = std::thread::scope(|s| {
            let reader = s.spawn(|| {
                let mut reads = 0usize;
                while !done.load(Ordering::Acquire) {
                    let bytes = std::fs::read(&path).expect("the cache always exists");
                    let parsed: Result<CachedCalibration, _> = serde_json::from_slice(&bytes);
                    assert!(
                        parsed.is_ok(),
                        "reader saw a torn cache ({} bytes)",
                        bytes.len()
                    );
                    reads += 1;
                }
                reads
            });
            for t in 1..=200 {
                write_cache_to(&path, &big(t)).expect("temp dir is writable");
            }
            done.store(true, Ordering::Release);
            reader.join().expect("reader thread")
        });
        assert!(reads > 0, "the reader must actually have raced the writer");
        assert_eq!(read_cache_from(&path), Some(big(200)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
