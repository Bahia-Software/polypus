//! Per-call execution parameters handed to a backend on every
//! [`run_circuits`](crate::QuantumBackend::run_circuits).
//!
//! [`RunParams`] is the *provider-agnostic, pyo3-free* configuration a backend
//! actually reads at execution time. It deliberately excludes everything that is
//! consumed only when a backend is *constructed* (the Aer noise model, the CUNQA
//! node count / `n_qpus`, the informational infrastructure label): those live in
//! the construction-time config the factory owns (in `polypus-infrastructure`),
//! never on this per-call struct. Keeping the per-call surface this small is what
//! lets a third-party backend depend on `polypus-backend` alone — no PyO3, no
//! Qiskit — and still receive everything it needs to run a batch.

use crate::transpiler::OptLevel;

/// The parameters a backend reads on every execution call.
///
/// Audited (plan Fase 1) as the *only* fields any backend consults inside
/// `run_circuits`/`run_shots_distributed`: the run `id`, the `shots`, the
/// sampling `seed`, and the transpiler `opt_level`. The number of replicas
/// (`n_qpus`) is **not** here — a shot-distributing planner carries it itself
/// (see `ShotDistributingPlanner`) — and neither is the provider-specific backend
/// configuration, which is consumed once at construction.
#[derive(Debug, Clone)]
pub struct RunParams {
    /// Unique identifier for this run (logging, temp files, SLURM job names).
    ///
    /// Already validated when it comes from a Python entry point: `train` /
    /// `qml.train` restrict the caller-supplied *prefix* to `[A-Za-z0-9._-]`,
    /// non-empty and at most 64 characters (contract C-9) before appending the
    /// UUID v4 suffix, precisely because this string reaches CUNQA's SLURM
    /// `family_name`/`family_id` and the temp file / log stream names. Building a
    /// `RunParams` directly from Rust bypasses that check — keep the same charset
    /// if the value can reach an external tool.
    pub id: String,
    /// Number of shots per circuit.
    pub shots: u32,
    /// Explicit RNG seed for shot sampling (contract C-7), or `None` for each
    /// backend's own unseeded default. The Python-facing layer resolves it to
    /// `Some` whenever a seed-consuming backend runs (user value or a fresh
    /// OS-entropy draw), so the effective seed can be reported in the manifest.
    pub seed: Option<u64>,
    /// Optimization effort for the backend's transpiler, travelling to
    /// [`Transpiler::transpile`](crate::Transpiler::transpile) as the tuning
    /// argument. Defaults to [`OptLevel::Light`]; with the no-op
    /// [`IdentityTranspiler`](crate::IdentityTranspiler) it has no effect on
    /// results.
    pub opt_level: OptLevel,
}
