use pyo3::prelude::*;

/// Provider-agnostic execution parameters, fully decoupled from circuit data.
///
/// Only fields that *every* backend needs live here. Anything provider-specific
/// (Aer simulation method, CUNQA node count, future IBM token, …) belongs in
/// [`BackendConfig`], so adding a new provider never widens this struct.
///
/// Passed to [`crate::infrastructure::QuantumBackend::run_circuits`] alongside
/// the circuits, so the backend knows *how* and *where* to run them without
/// coupling to algorithm logic.
#[derive(Debug)]
pub struct ExecutionConfig {
    /// Unique identifier for this run (logging, temp files, SLURM job names).
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
}

/// Provider-specific configuration.
///
/// Each variant declares exactly the fields its backend needs. Supporting a new
/// provider (IBM, IQM, an HPC scheduler, …) means adding one variant here and
/// one [`crate::infrastructure::QuantumBackend`] implementation — existing
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
    /// Qiskit) on a [`crate::infrastructure::NativeStatevectorBackend`]. It is
    /// noiseless by construction, so it carries no provider-specific fields; the
    /// shot count and run id travel in [`ExecutionConfig`].
    LocalNative,
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
}
