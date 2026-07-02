use pyo3::prelude::*;

use crate::infrastructure::transpiler::OptLevel;

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
    /// Optimization effort for the backend's transpiler.
    ///
    /// Travels to [`crate::infrastructure::Transpiler::transpile`] as a
    /// [`TranspileOptions`](crate::infrastructure::TranspileOptions) *argument*
    /// (the *tuning*), while the transpilation *strategy* is injected into the
    /// backend by composition. Defaults to [`OptLevel::Light`]; with the default
    /// [`IdentityTranspiler`](crate::infrastructure::IdentityTranspiler) it has
    /// no effect on results.
    pub opt_level: OptLevel,
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
    /// CESGA QMIO real QPU, reached directly over its ZeroMQ REQ endpoint.
    ///
    /// Unlike [`Local`](Self::Local)/[`Cunqa`](Self::Cunqa), this backend speaks
    /// the QMIO wire protocol (pickle over ZMQ) entirely from Rust, so the
    /// `Native`/`Qasm2` execution path never touches the Python interpreter.
    /// Only compiled with `--features qmio`.
    #[cfg(feature = "qmio")]
    Qmio {
        /// ZMQ REQ endpoint of the QMIO server (e.g. `"tcp://10.133.29.226:5556"`).
        endpoint: String,
        /// Representation of the program submitted to the QPU.
        program_format: QmioProgramFormat,
        /// Tket optimisation level (`0`/`1`/`2` → Tket `$value` `1`/`18`/`30`).
        optimization: u8,
        /// Repetition period (`None` = server default).
        repetition_period: Option<f64>,
        /// Results format requested from the server (`"binary_count"` by default).
        res_format: String,
    },
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
