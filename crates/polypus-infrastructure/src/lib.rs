//! # polypus-infrastructure
//!
//! The concrete Polypus execution backends and the factory that builds them. The
//! provider-agnostic, **pyo3-free** contract they implement — the
//! [`QuantumBackend`] trait, [`BoundCircuit`], [`RunParams`], the error types, the
//! [`Planner`], the transpiler seam and the memory budget — lives in the
//! `polypus-backend` crate and is re-exported here, so every existing
//! `polypus_infrastructure::…` / `crate::…` import keeps resolving.
//!
//! What this crate adds on top of `polypus-backend`:
//!
//! - the four backends: [`NativeStatevectorBackend`], [`LocalBackend`],
//!   [`CunqaBackend`] and (behind `--features qmio`) `QmioBackend`;
//! - the [`Infrastructure`] factory ([`Infrastructure::create_backend`]);
//! - the Qiskit boundary ([`QiskitCircuit`], [`to_py_object`]) — where PyO3
//!   re-enters after the clean contract — and the construction-time
//!   [`ExecutionConfig`]/[`BackendConfig`];
//! - the process-wide backend-cleanup failure counter exposed to Python.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;

pub mod circuit;
pub mod cunqa;
pub mod error;
pub mod execution_config;
pub mod local;
pub mod native;
#[cfg(feature = "qmio")]
pub mod qmio;
pub mod transpiler;

// --- re-export the pyo3-free contract so downstream import paths are unchanged ---
pub use polypus_backend::{
    create_registered_backend, is_registered, max_statevector_concurrency, register_backend,
    registered_names, validate_run_results, wave_concurrency, BackendBuildContext,
    BackendCapabilities, BackendError, BackendFactory, BoundCircuit, CancelToken, CircuitTask,
    Counts, ForeignCircuit, IdentityTranspiler, InfrastructureError, Interrupt, OptLevel, Planner,
    PlannerRequirements, QuantumBackend, RunParams, SequentialPlanner, ShotDistributingPlanner,
    TranspileOptions, Transpiler,
};

// --- this crate's own additions ---
pub use circuit::{as_qiskit, seam_error, to_py_object, QiskitCircuit};
pub use cunqa::CunqaBackend;
pub use execution_config::{BackendConfig, ExecutionConfig};
pub use local::LocalBackend;
pub use native::NativeStatevectorBackend;
#[cfg(feature = "qmio")]
pub use qmio::QmioBackend;

/// Process-wide count of backend cleanup (`close`/`Drop`) failures.
///
/// A `Drop` must never panic, so a failed teardown is logged and counted here
/// rather than propagated. A per-instance flag would be useless — nobody holds
/// the instance once `Drop` returns — so the consultable state lives in this
/// process-wide counter, exposed to Python as `polypus.backend_cleanup_failures()`.
static CLEANUP_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Record that a backend's resource cleanup failed. Called from `close`/`Drop`.
pub fn record_cleanup_failure() {
    CLEANUP_FAILURES.fetch_add(1, Ordering::SeqCst);
}

/// Number of backend cleanup failures recorded so far this process (diagnostic).
pub fn cleanup_failure_count() -> u64 {
    CLEANUP_FAILURES.load(Ordering::SeqCst)
}

/// Register the backends Polypus ships through the runtime registry, once per
/// process. Idempotent (a `Once` guard), and called at the top of
/// [`Infrastructure::create_backend`] so a name-driven build always finds them.
///
/// - `"subprocess"` — the Python subprocess bridge (`polypus-subprocess-backend`),
///   always available (pyo3-free).
/// - `"qmio"` — the CESGA QMIO QPU, only with `--features qmio`; without it, the
///   name stays unregistered and selecting it yields the actionable
///   "requires --features qmio" error at the edge.
///
/// A third party's own `register_backend(...)` call composes with these: last
/// registration wins, so an embedder can even override a built-in name.
pub fn register_builtin_backends() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Never clobber a name an embedder already registered: registering is
        // last-wins, so an embedder who registered their own `"subprocess"` (or
        // `"qmio"`) *before* the first `create_backend` keeps it. Only fill in a name
        // that is still free.
        if !is_registered(polypus_subprocess_backend::BACKEND_NAME) {
            polypus_subprocess_backend::register();
        }
        #[cfg(feature = "qmio")]
        if !is_registered("qmio") {
            register_backend("qmio", qmio::qmio_factory);
        }
    });
}

/// Supported quantum execution infrastructures.
#[derive(Debug)]
pub enum Infrastructure {
    Local,
    Cunqa,
    /// CESGA QMIO real QPU (see `QmioBackend`). The variant always exists so
    /// that selecting `"qmio"` produces a clear "requires `--features qmio`"
    /// error rather than an `Unknown infrastructure` panic when the feature is
    /// disabled; the backend itself is only built with the feature on.
    Qmio,
}

impl Infrastructure {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, BackendError> {
        match s {
            "local" => Ok(Infrastructure::Local),
            "cunqa" => Ok(Infrastructure::Cunqa),
            "qmio" => Ok(Infrastructure::Qmio),
            other => Err(BackendError::UnknownInfrastructure {
                name: other.to_string(),
            }),
        }
    }

    /// Instantiate the appropriate backend for the given execution config.
    ///
    /// Dispatch is driven solely by [`ExecutionConfig::backend_config`], so the
    /// chosen backend and its parameters can never desync. Adding a new backend
    /// (IBM, IQM, …) means adding one arm here and implementing
    /// [`QuantumBackend`] for the new type — no algorithm code changes.
    pub fn create_backend(
        config: &ExecutionConfig,
    ) -> Result<Arc<dyn QuantumBackend>, BackendError> {
        // Ensure Polypus's registry-migrated backends (subprocess, qmio) — and any
        // the embedder registered before us — are discoverable by name.
        register_builtin_backends();
        match &config.backend_config {
            BackendConfig::Local {
                backend,
                sim_method,
                noise_model,
            } => Ok(Arc::new(LocalBackend::new(
                backend.clone(),
                sim_method.clone(),
                noise_model
                    .as_ref()
                    .map(|nm| Python::with_gil(|py| nm.clone_ref(py))),
            ))),
            BackendConfig::Cunqa {
                backend,
                sim_method,
                nodes,
                cores_per_qpu,
            } => Ok(Arc::new(CunqaBackend::new(
                config.n_qpus,
                *nodes,
                &config.id,
                *cores_per_qpu,
                backend.clone(),
                sim_method.clone(),
            )?)),
            // The Python-facing layer resolves a concrete seed whenever the
            // native backend runs; the entropy fallback here only guards a
            // directly-built config that left `seed` unset (e.g. tests), so
            // an omitted seed still yields independent noise, never a panic.
            BackendConfig::LocalNative { fusion } => Ok(Arc::new(
                NativeStatevectorBackend::new(
                    config.seed.unwrap_or_else(execution_config::random_seed),
                )
                .with_fusion(*fusion),
            )),
            // A registry-dispatched backend (QMIO, the subprocess bridge, or a
            // third party's own). The typed construction fields are gone; the factory
            // reads its configuration from the pyo3-free `BackendBuildContext`. A
            // QMIO wire error still crosses the contract type-erased in
            // `BackendError::External` (the factory boxes it), and the FFI edge
            // downcasts it back to raise the typed `polypus.QmioError`.
            BackendConfig::Registered { name, options } => {
                let ctx = BackendBuildContext {
                    id: config.id.clone(),
                    shots: config.shots,
                    n_qpus: config.n_qpus,
                    seed: config.seed,
                    opt_level: config.opt_level,
                    options: options.clone(),
                };
                create_registered_backend(name, &ctx)
            }
        }
    }
}
