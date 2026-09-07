//! Python binding for machine-aware gate-parallelism calibration.
//!
//! `polypus.calibrate_parallel_threshold(force=False)` measures the qubit count
//! at which the native statevector simulator's gate kernels should switch to the
//! rayon parallel path on *this* machine, caches it on disk keyed by the rayon
//! thread count, and returns what it decided. It mirrors `polypus.init_logger`:
//! a one-shot, process-level setup call, meant to run once — the first line of a
//! SLURM script, or inside `install.sh` — **before** any circuits are simulated,
//! because the per-process resolver reads the cache once and memoises it.
//!
//! Like every other library diagnostic, its `log::*` output stays silent until
//! `polypus.init_logger()` installs a sink.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Calibrate (or reuse) the gate-parallelism threshold for this machine.
///
/// With `force=False` (default) a cache already valid for the current hardware
/// (same rayon thread count) is reused unchanged and nothing is measured; with
/// `force=True` the crossover is re-measured and the cache overwritten.
///
/// Returns a `dict` describing the outcome:
/// * `threshold` (int) — qubit count at/above which gates go parallel;
/// * `num_threads` (int) — rayon threads detected (the cache key);
/// * `duration_secs` (float) — measurement time (`0.0` when reused);
/// * `reused_cache` (bool) — whether a valid cache was reused as-is;
/// * `cache_path` (str | None) — where the cache lives / would live;
/// * `cache_written` (bool) — whether this call persisted a fresh result.
///
/// A cache that cannot be written (read-only container/CI) is **not** an error:
/// the measured threshold is still returned with `cache_written=False`, so the
/// call is safe to run unconditionally from an installer. Diagnostics require
/// `polypus.init_logger()` to be visible.
#[pyfunction]
#[pyo3(signature = (force = false))]
pub fn calibrate_parallel_threshold(py: Python<'_>, force: bool) -> PyResult<PyObject> {
    // Pure-Rust CPU work (rayon timing loops): release the GIL for it, mirroring
    // the other bindings so it cannot stall other Python threads
    // (docs/ENGINEERING.md §3).
    let outcome = py.allow_threads(|| polypus_sim::calibrate_and_cache(force));

    let dict = PyDict::new(py);
    dict.set_item("threshold", outcome.threshold)?;
    dict.set_item("num_threads", outcome.num_threads)?;
    dict.set_item("duration_secs", outcome.duration.as_secs_f64())?;
    dict.set_item("reused_cache", outcome.reused_cache)?;
    dict.set_item("cache_path", outcome.cache_path)?;
    dict.set_item("cache_written", outcome.cache_written)?;
    Ok(dict.into_any().unbind())
}
