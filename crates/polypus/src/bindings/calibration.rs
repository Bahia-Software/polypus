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
//!
//! ## Default-visible fallback warning
//!
//! When a process runs the native statevector path against a cache calibrated
//! for *different* hardware (`FallbackReason::HardwareChanged`), `polypus-sim`
//! emits a `log::warn!` — but that is silent unless the user installed a logger,
//! which most do not. So [`warn_if_using_default_threshold`] additionally raises
//! a Python `UserWarning`, which the interpreter prints to stderr **by default**
//! with no setup. This does not break the "the pure Rust core never prints, only
//! logs" rule (that binds `polypus-sim`/`polypus-circuit`): this lives in the
//! bindings layer, which already talks to the interpreter directly.
//!
//! Since issue #176, `import polypus` auto-calibrates once per machine, so the
//! `FallbackReason::NotCalibrated` case no longer raises a `UserWarning` (it
//! would only nag on a read-only cache dir, where the result is still correct);
//! it is demoted to a logger-only `info!`. See [`warn_if_using_default_threshold`].

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use polypus_sim::FallbackReason;

/// Guards the one-per-process Python fallback warning. An explicit guard — the
/// same pattern as `LOGGER_INSTALLED` in `logging.rs` — rather than relying on
/// Python's own duplicate-warning filter, whose behaviour a caller can
/// reconfigure (`warnings.simplefilter("always")`, `-W`, pytest) and which we do
/// not want to depend on guessing right.
static THRESHOLD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

/// Raise a one-time, default-visible `UserWarning` if this process resolved the
/// gate-parallel threshold to the static default *because the cache was
/// calibrated for different hardware* (`FallbackReason::HardwareChanged`). The
/// `NotCalibrated` case is handled quietly (a logger-only `info!`) since import
/// auto-calibrates the machine (issue #176) — see the module docs.
///
/// Called at the entry points that actually run the native statevector backend
/// (`polypus.statevector`, and `run_quantum_circuit` only when it selects the
/// native backend). No-op — and cheap — when the threshold came from a valid
/// cache, or after the first warning in a process. The message is actionable: it
/// names `polypus.calibrate_parallel_threshold()` so a user can fix it, matching
/// the `log` records `polypus-sim` emits for the same conditions.
///
/// Uses `polypus_sim::resolved_fallback_reason()`, which shares the one
/// per-process resolution the simulator uses, so the warning can never disagree
/// with the threshold actually in force.
pub(crate) fn warn_if_using_default_threshold(py: Python<'_>) -> PyResult<()> {
    let Some(reason) = polypus_sim::resolved_fallback_reason() else {
        return Ok(());
    };
    let cached_threads = match reason {
        FallbackReason::HardwareChanged { cached_threads } => cached_threads,
        // `NotCalibrated` no longer raises a default-visible `UserWarning`. Since
        // issue #176, `import polypus` auto-calibrates once per machine, so on
        // the happy path this reason does not occur at all. The one case that
        // still reaches it — a read-only cache dir (containers/CI), where
        // auto-calibration measures but cannot persist — is exactly where a
        // stderr warning on every native run would be noise, not signal (the
        // result is numerically identical, just possibly untuned). Demote it to
        // an `info!` record, visible only to callers who installed a logger,
        // matching the record `polypus-sim` already emits for the same reason.
        FallbackReason::NotCalibrated => {
            log::info!(
                "polypus is using the default gate-parallel threshold (not calibrated on this \
                 machine, e.g. a read-only cache dir). Results are unaffected; call \
                 polypus.calibrate_parallel_threshold() to tune performance."
            );
            return Ok(());
        }
    };
    // Claim the one-shot slot: at most one `HardwareChanged` warning per process,
    // and safe if two threads reach here at once. Only claimed once we know we
    // are actually going to warn, so the `NotCalibrated` info path above never
    // consumes it.
    if THRESHOLD_WARNING_EMITTED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let message = format!(
        "polypus was calibrated for {cached_threads} thread(s) but this machine has {}; \
         using the default gate-parallel threshold, which may not be optimal for this \
         hardware. Call polypus.calibrate_parallel_threshold(force=True) to recalibrate.",
        rayon::current_num_threads()
    );
    let warnings = PyModule::import(py, "warnings")?;
    // Default category (UserWarning): shown on stderr by default, unlike a `log`
    // record. `stacklevel=2` points the warning at the caller of the native
    // entry point rather than at this helper.
    let kwargs = PyDict::new(py);
    kwargs.set_item("stacklevel", 2)?;
    warnings.call_method("warn", (message,), Some(&kwargs))?;
    Ok(())
}

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
