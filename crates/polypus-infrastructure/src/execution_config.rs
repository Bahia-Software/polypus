use std::collections::HashMap;

use pyo3::prelude::*;

use polypus_backend::{OptLevel, RunParams};

/// **Construction-time** execution configuration: everything the
/// [`Infrastructure`](crate::Infrastructure) factory needs to *build* a backend,
/// including the provider-specific [`BackendConfig`] (Aer noise model, CUNQA node
/// count) and the `n_qpus` replica count.
///
/// This is deliberately distinct from [`RunParams`], the small, pyo3-free struct a
/// backend reads on *every* [`run_circuits`](crate::QuantumBackend::run_circuits)
/// call. The audit (plan Fase 1) confirmed no backend reads `backend_config`,
/// `infrastructure` or `n_qpus` at execution time — they are consumed once, here,
/// at construction (or, for `n_qpus`, by the shot-distributing planner it is passed
/// to). Deriving the per-call view is [`run_params`](Self::run_params).
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Unique identifier for this run (logging, temp files, SLURM job names).
    ///
    /// Already validated when it comes from a Python entry point: `train` /
    /// `qml.train` restrict the caller-supplied *prefix* to `[A-Za-z0-9._-]`,
    /// non-empty and at most 64 characters (contract C-9, `validate_id` in
    /// `crate::bindings`) before appending the UUID v4 suffix, precisely because
    /// this string reaches CUNQA's SLURM `family_name`/`family_id` and the temp
    /// file / log stream names above. Constructing an `ExecutionConfig` directly
    /// from Rust bypasses that check — keep the same charset if the value can
    /// reach an external tool.
    pub id: String,
    /// Number of shots per circuit.
    pub shots: u32,
    /// Number of QPUs to target.
    pub n_qpus: u32,
    /// Human-readable infrastructure label (`"local"`, `"cunqa"`, …).
    ///
    /// Informational only: backend dispatch is driven by [`BackendConfig`],
    /// which is the single source of truth for *which* backend runs.
    pub infrastructure: String,
    /// Provider-specific configuration.
    pub backend_config: BackendConfig,
    /// Optimization effort for the backend's transpiler.
    ///
    /// Travels to [`crate::Transpiler::transpile`] as a
    /// [`TranspileOptions`](crate::TranspileOptions) *argument*
    /// (the *tuning*), while the transpilation *strategy* is injected into the
    /// backend by composition. Defaults to [`OptLevel::Light`]; with the default
    /// [`IdentityTranspiler`](crate::IdentityTranspiler) it has
    /// no effect on results.
    pub opt_level: OptLevel,
    /// Explicit RNG seed for shot sampling.
    ///
    /// Consumed by every backend that samples shots itself: the native
    /// statevector backend
    /// ([`NativeStatevectorBackend`](crate::NativeStatevectorBackend))
    /// seeds its per-circuit sampling stream directly, while [`Local`](BackendConfig::Local)
    /// and [`Cunqa`](BackendConfig::Cunqa) forward it to the underlying Aer
    /// (`seed_simulator`) and CUNQA `run(..., seed=...)` calls respectively.
    /// The Python-facing layer resolves it to `Some` whenever a seed-consuming
    /// backend runs (user-supplied value or a fresh OS-entropy draw), so the
    /// effective seed can be reported back in the run manifest (contract
    /// C-7). `None` means "no explicit seed", and each backend falls back to
    /// its own unseeded default. Decoupled from [`id`](Self::id), which is
    /// only a logging/temp-file/SLURM label.
    pub seed: Option<u64>,
}

impl ExecutionConfig {
    /// Project the per-call view a backend and planner actually read: the pyo3-free
    /// [`RunParams`]. Drops the construction-only fields (`backend_config`,
    /// `infrastructure`, `n_qpus`), which no `run_circuits` consults.
    pub fn run_params(&self) -> RunParams {
        RunParams {
            id: self.id.clone(),
            shots: self.shots,
            seed: self.seed,
            opt_level: self.opt_level,
        }
    }
}

/// Draw a fresh 64-bit seed from OS entropy.
///
/// Used as the default when no explicit seed is supplied, so an omitted seed
/// produces genuine (independent) shot noise across runs rather than repeating a
/// value derived from the run [`id`](ExecutionConfig::id).
pub fn random_seed() -> u64 {
    use rand::RngCore;
    rand::rngs::OsRng.next_u64()
}

/// Provider-specific configuration.
///
/// Each variant declares exactly the fields its backend needs. Supporting a new
/// provider (IBM, IQM, an HPC scheduler, …) means adding one variant here and
/// one [`crate::QuantumBackend`] implementation — existing
/// variants, backends, and every algorithm stay untouched.
#[derive(Debug)]
pub enum BackendConfig {
    /// Local Qiskit Aer simulator.
    Local {
        /// Backend/device class name forwarded to Python (e.g. `"AerSimulator"`).
        backend: String,
        /// Aer simulation method: `"automatic"`, `"statevector"`, `"matrix_product_state"`, …
        sim_method: String,
        /// Optional Qiskit `NoiseModel` forwarded to the Aer backend.
        noise_model: Option<Py<PyAny>>,
    },
    /// Local pure-Rust statevector simulator (`polypus-sim`).
    ///
    /// Selected with `backend="polypus"`. Runs entirely in Rust (no GIL, no
    /// Qiskit) on a [`crate::NativeStatevectorBackend`]. It is
    /// noiseless by construction, so its only provider-specific field is
    /// `fusion`; the shot count and run id travel in [`ExecutionConfig`].
    LocalNative {
        /// Whether the backend may fuse gates (diagonal-run and dense
        /// connected-component fusion — see
        /// [`polypus_sim::StatevectorSimulator::fusion`]) before applying
        /// them. Defaults to `true`; `false` runs the circuit strictly
        /// gate-by-gate, exactly as written, for a caller that wants a pure
        /// simulation unaffected by the fusion heuristics.
        fusion: bool,
    },
    /// CUNQA distributed QPU platform (SLURM-managed HPC).
    Cunqa {
        /// Backend/device class name forwarded to CUNQA.
        backend: String,
        /// Simulation method for CUNQA's simulated QPUs.
        sim_method: String,
        /// Number of cluster nodes to reserve.
        nodes: u32,
        /// CPU cores reserved per QPU.
        cores_per_qpu: u32,
    },
    /// A backend built through the **runtime registry**
    /// ([`polypus_backend::register_backend`]), selected by `name`. This is how a
    /// third-party backend — and Polypus's own registry-migrated backends (QMIO, the
    /// subprocess bridge) — are dispatched without a dedicated typed variant here.
    ///
    /// `options` is the provider-specific configuration as string key/value pairs,
    /// handed to the factory in the [`BackendBuildContext`](polypus_backend::BackendBuildContext).
    /// It replaced the former hardcoded `Qmio { … }` variant: QMIO now registers a
    /// factory (`qmio::qmio_factory`) that reads its `endpoint`/`program_format`/…
    /// from here. Only the endpoints that carry a `Py<PyAny>` (Aer's noise model)
    /// still need a typed variant; everything pyo3-free flows through this one.
    Registered {
        /// The registered backend name (e.g. `"qmio"`, `"subprocess"`, or a third
        /// party's own).
        name: String,
        /// Provider-specific configuration, as documented per backend.
        options: HashMap<String, String>,
    },
}

/// Manual [`Clone`]: the only non-`Clone` field is `BackendConfig::Local`'s
/// optional Qiskit `NoiseModel`, whose reference count must be bumped under the
/// GIL via `clone_ref` (the same pattern
/// [`Infrastructure::create_backend`](crate::Infrastructure::create_backend)
/// uses). Cloning a config is what lets an orchestration algorithm derive a
/// per-batch config that differs only in `shots` without mutating the caller's.
impl Clone for BackendConfig {
    fn clone(&self) -> Self {
        match self {
            BackendConfig::Local {
                backend,
                sim_method,
                noise_model,
            } => BackendConfig::Local {
                backend: backend.clone(),
                sim_method: sim_method.clone(),
                noise_model: noise_model
                    .as_ref()
                    .map(|nm| Python::with_gil(|py| nm.clone_ref(py))),
            },
            BackendConfig::LocalNative { fusion } => BackendConfig::LocalNative { fusion: *fusion },
            BackendConfig::Cunqa {
                backend,
                sim_method,
                nodes,
                cores_per_qpu,
            } => BackendConfig::Cunqa {
                backend: backend.clone(),
                sim_method: sim_method.clone(),
                nodes: *nodes,
                cores_per_qpu: *cores_per_qpu,
            },
            BackendConfig::Registered { name, options } => BackendConfig::Registered {
                name: name.clone(),
                options: options.clone(),
            },
        }
    }
}

/// Representation of the program sent to the QMIO QPU.
///
/// The legible/compiled axis applies to QIR (a `.ll` text module vs assembled
/// `.bc` bitcode); OpenQASM has no standard binary form, so it is always text.
#[cfg(feature = "qmio")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QmioProgramFormat {
    /// OpenQASM text (`ConcreteCircuit::to_qasm2`, with the header retargeted to
    /// match the QMIO compiler — see `infrastructure::qmio`).
    OpenQasm,
    /// QIR Base Profile LLVM IR text (`ConcreteCircuit::to_qir`).
    QirText,
    /// Assembled QIR LLVM bitcode (`ConcreteCircuit::to_qir_bitcode`, needs
    /// `llvm-as` on `PATH`); travels as Python `bytes`.
    QirBitcode,
}
