//! Python binding for machine-aware gate-parallelism calibration.
//!
//! `polypus.calibrate_parallel_threshold(force=False)` measures the qubit count
//! at which the native statevector simulator's gate kernels should switch to the
//! rayon parallel path on *this* machine, caches it on disk keyed by the rayon
//! thread count *and* a CPU fingerprint (so nodes sharing one `$HOME` keep
//! separate entries; see `polypus_sim::calibration`), and returns what it
//! decided. It mirrors `polypus.init_logger`:
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
//! Since issue #176 the first native statevector run of a process auto-calibrates
//! the machine (lazily since #351; see [`prepare_native_threshold`]), so the
//! `FallbackReason::NotCalibrated` case no longer raises a `UserWarning` (it
//! would only nag on a read-only cache dir, where the result is still correct);
//! it is demoted to a logger-only `info!`. See [`warn_if_using_default_threshold`].
//!
//! ## Lazy auto-calibration
//!
//! A `pip install` of a published wheel has no post-install hook, and the CI
//! machine that builds the wheel is not the one that runs it, so the threshold
//! is measured on the runtime hardware, once per machine (#176). It used to run
//! at `import polypus`, which cost CPU and a cache write on every cold import,
//! even in processes that never simulate (#351). It now runs on the first call
//! that resolves the threshold — `polypus.statevector`, and `run_quantum_circuit`
//! or `train` on the native backend — just before the simulator reads the cache.
//! `import polypus` measures nothing and writes nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use polypus_sim::FallbackReason;

/// Guards the one-per-process Python fallback warning. An explicit guard — the
/// same pattern as `LOGGER_INSTALLED` in `logging.rs` — rather than relying on
/// Python's own duplicate-warning filter, whose behaviour a caller can
/// reconfigure (`warnings.simplefilter("always")`, `-W`, pytest) and which we do
/// not want to depend on guessing right.
static THRESHOLD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

/// Guards the once-per-process implicit calibration (see
/// [`autocalibrate_once`]).
static AUTOCALIBRATION: Once = Once::new();

/// Opt-out for the implicit calibration (CI, containers, runs that must pin the
/// static default threshold).
const OPTOUT_VAR: &str = "POLYPUS_NO_AUTOCALIBRATE";

/// Launcher rank variables: a non-zero rank skips the implicit calibration
/// (issue #214).
const RANK_VARS: [&str; 2] = ["SLURM_PROCID", "OMPI_COMM_WORLD_RANK"];

/// Whether the implicit calibration may run, given an environment lookup.
///
/// `false` when `POLYPUS_NO_AUTOCALIBRATE` is `1`/`true`/`yes`/`on` (any case),
/// or when `SLURM_PROCID` or `OMPI_COMM_WORLD_RANK` is set to anything other
/// than empty or `0`: otherwise every rank of a large `srun`/`mpirun` would
/// measure at once, contending for the CPUs being timed and for the shared
/// cache file. Rank 0, or no launcher at all, calibrates. Pure, so the policy
/// is unit-tested without touching the process environment.
fn autocalibration_allowed(var: impl Fn(&str) -> Option<String>) -> bool {
    let normalized = |name: &str| var(name).unwrap_or_default().trim().to_ascii_lowercase();
    if matches!(normalized(OPTOUT_VAR).as_str(), "1" | "true" | "yes" | "on") {
        return false;
    }
    RANK_VARS.iter().all(|name| {
        let rank = normalized(name);
        rank.is_empty() || rank == "0"
    })
}

/// Calibrate (or reuse) this machine's gate-parallel threshold, at most once per
/// process and only when [`autocalibration_allowed`] permits it.
///
/// `force = false`: a cache entry valid for this hardware is reused in ~0s; a
/// cold or hardware-changed cache is measured (<1s) and persisted. A read-only
/// cache dir is not an error (the result is just not persisted). Calibration is
/// a pure performance optimisation — the parallel and sequential paths are
/// numerically identical — so a panic inside it is contained and the simulator
/// falls back to the static default. The GIL is released for the measurement,
/// including while another thread waits for it to finish.
fn autocalibrate_once(py: Python<'_>) {
    py.detach(|| {
        AUTOCALIBRATION.call_once(|| {
            let env = |name: &str| std::env::var_os(name).map(|v| v.to_string_lossy().into_owned());
            if autocalibration_allowed(env) {
                let _ = std::panic::catch_unwind(|| polypus_sim::calibrate_and_cache(false));
            }
        })
    });
}

/// Prepare the gate-parallel threshold before a native statevector run: run the
/// implicit calibration on the first such call of the process, then raise the
/// fallback warning if the threshold still resolved to the static default
/// because of a stale cache.
///
/// Must run before the run builds its simulator: the simulator resolves the
/// threshold from the cache once per process, so a calibration after that
/// point would not be seen until the next process.
pub(crate) fn prepare_native_threshold(py: Python<'_>) -> PyResult<()> {
    autocalibrate_once(py);
    warn_if_using_default_threshold(py)
}

/// Raise a one-time, default-visible `UserWarning` if this process resolved the
/// gate-parallel threshold to the static default *because the cache was
/// calibrated for different hardware* (`FallbackReason::HardwareChanged`). The
/// `NotCalibrated` case is handled quietly (a logger-only `info!`) since the
/// first native run auto-calibrates the machine (issue #176) — see the module
/// docs.
///
/// Called through [`prepare_native_threshold`] at the entry points that actually
/// run the native statevector backend (`polypus.statevector`, and
/// `run_quantum_circuit`/`train` only when they select the native backend). No-op — and cheap — when the threshold came from a valid
/// cache, or after the first warning in a process. The message is actionable: it
/// names `polypus.calibrate_parallel_threshold()` so a user can fix it, matching
/// the `log` records `polypus-sim` emits for the same conditions.
///
/// Uses `polypus_sim::resolved_fallback_reason()`, which shares the one
/// per-process resolution the simulator uses, so the warning can never disagree
/// with the threshold actually in force.
fn warn_if_using_default_threshold(py: Python<'_>) -> PyResult<()> {
    let Some(reason) = polypus_sim::resolved_fallback_reason() else {
        return Ok(());
    };
    let cached_threads = match reason {
        FallbackReason::HardwareChanged { cached_threads } => cached_threads,
        // `NotCalibrated` no longer raises a default-visible `UserWarning`. Since
        // issue #176, the first native run auto-calibrates once per machine, so on
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
/// With `force=False` (default) a cache entry already valid for the current
/// hardware (same CPU fingerprint *and* rayon thread count) is reused unchanged
/// and nothing is measured; with `force=True` the crossover is re-measured and
/// that one entry replaced — entries for other thread counts or other nodes are
/// preserved. The write is atomic and lock-protected, so concurrent calls (e.g.
/// several ranks or nodes on a shared `$HOME`) never tear the file. Unlike the
/// implicit calibration of the first native run, this explicit call runs on every
/// rank it is invoked on, secondary SLURM/OpenMPI ranks included, and ignores
/// `POLYPUS_NO_AUTOCALIBRATE`.
///
/// Returns a `dict` describing the outcome:
/// * `threshold` (int) — qubit count at/above which gates go parallel;
/// * `num_threads` (int) — rayon threads detected (one half of the cache key,
///   alongside the CPU fingerprint, which is not returned);
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
pub fn calibrate_parallel_threshold(py: Python<'_>, force: bool) -> PyResult<Py<PyAny>> {
    // Pure-Rust CPU work (rayon timing loops): release the GIL for it, mirroring
    // the other bindings so it cannot stall other Python threads
    // (docs/ENGINEERING.md §3).
    let outcome = py.detach(|| polypus_sim::calibrate_and_cache(force));

    let dict = PyDict::new(py);
    dict.set_item("threshold", outcome.threshold)?;
    dict.set_item("num_threads", outcome.num_threads)?;
    dict.set_item("duration_secs", outcome.duration.as_secs_f64())?;
    dict.set_item("reused_cache", outcome.reused_cache)?;
    dict.set_item("cache_path", outcome.cache_path)?;
    dict.set_item("cache_written", outcome.cache_written)?;
    Ok(dict.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::autocalibration_allowed;
    use std::collections::HashMap;

    fn allowed(vars: &[(&str, &str)]) -> bool {
        let env: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        autocalibration_allowed(|name| env.get(name).cloned())
    }

    #[test]
    fn calibrates_without_optout_or_launcher() {
        assert!(allowed(&[]));
        assert!(allowed(&[("POLYPUS_NO_AUTOCALIBRATE", "0")]));
        assert!(allowed(&[("POLYPUS_NO_AUTOCALIBRATE", "")]));
    }

    #[test]
    fn optout_values_skip_calibration() {
        for value in ["1", "true", "TRUE", "yes", "on", " On "] {
            assert!(
                !allowed(&[("POLYPUS_NO_AUTOCALIBRATE", value)]),
                "opt-out value {value:?} must skip calibration"
            );
        }
    }

    #[test]
    fn secondary_ranks_skip_calibration() {
        for var in ["SLURM_PROCID", "OMPI_COMM_WORLD_RANK"] {
            for value in ["1", "7", " 12 "] {
                assert!(!allowed(&[(var, value)]), "{var}={value:?} must skip");
            }
        }
    }

    #[test]
    fn rank_zero_or_empty_rank_calibrates() {
        assert!(allowed(&[("SLURM_PROCID", "0")]));
        assert!(allowed(&[("OMPI_COMM_WORLD_RANK", "0")]));
        assert!(allowed(&[
            ("SLURM_PROCID", "0"),
            ("OMPI_COMM_WORLD_RANK", "0")
        ]));
        assert!(allowed(&[("SLURM_PROCID", "")]));
        assert!(!allowed(&[
            ("SLURM_PROCID", "0"),
            ("OMPI_COMM_WORLD_RANK", "2")
        ]));
    }
}
