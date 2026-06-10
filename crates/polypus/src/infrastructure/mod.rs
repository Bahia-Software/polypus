use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub mod execution_config;
pub mod local;
pub mod cunqa;

pub use execution_config::{ExecutionConfig, BackendConfig};
pub use local::LocalBackend;
pub use cunqa::CunqaBackend;

/// Supported quantum execution infrastructures.
pub enum Infrastructure {
    Local,
    Cunqa,
}

impl Infrastructure {
    pub fn from_str(s: &str) -> Self {
        match s {
            "local" => Infrastructure::Local,
            "cunqa" => Infrastructure::Cunqa,
            _ => panic!("Unknown infrastructure: {}", s),
        }
    }

    /// Instantiate the appropriate backend for the given execution config.
    ///
    /// Dispatch is driven solely by [`ExecutionConfig::backend_config`], so the
    /// chosen backend and its parameters can never desync. Adding a new backend
    /// (IBM, IQM, …) means adding one arm here and implementing
    /// [`QuantumBackend`] for the new type — no algorithm code changes.
    pub fn create_backend(config: &ExecutionConfig) -> Arc<dyn QuantumBackend> {
        match &config.backend_config {
            BackendConfig::Local { backend, sim_method, noise_model } => {
                Arc::new(LocalBackend::new(
                    backend.clone(),
                    sim_method.clone(),
                    noise_model
                        .as_ref()
                        .map(|nm| Python::with_gil(|py| nm.clone_ref(py))),
                ))
            }
            BackendConfig::Cunqa { backend, sim_method, nodes, cores_per_qpu } => {
                Arc::new(CunqaBackend::new(
                    config.n_qpus,
                    *nodes,
                    &config.id,
                    *cores_per_qpu,
                    backend.clone(),
                    sim_method.clone(),
                ))
            }
        }
    }
}

/// Contract for quantum circuit execution backends.
///
/// A backend is completely agnostic to the algorithm calling it; it only knows
/// how to execute a batch of bound (parameter-free) circuits and return counts.
///
/// Implementing this trait for a new provider (IBM, IQM, CUNQA, …) is sufficient
/// to make it available to every algorithm in Polypus without touching any
/// algorithm code.
pub trait QuantumBackend: Send + Sync {
    /// Execute a slice of bound circuits.
    ///
    /// Returns native measurement counts — one `HashMap<bitstring, count>` per
    /// circuit. Keeping the return type native (rather than a Python object)
    /// means non-Python backends (IBM, IQM, a future native MPI scheduler) never
    /// have to touch the GIL, which is essential for HPC-scale distribution.
    fn run_circuits(&self, qcs: &[Py<PyAny>], config: &ExecutionConfig) -> Vec<HashMap<String, u64>>;

    /// Maximum number of circuits to submit per [`run_circuits`](Self::run_circuits)
    /// call, given the `total` circuits to evaluate.
    ///
    /// Local simulation can take the whole batch at once: Aer's C++ engine runs
    /// the experiments in parallel across cores and releases the GIL, which is
    /// the only real parallelism available locally. Distributed backends such as
    /// CUNQA are bounded by the number of physical QPUs and therefore cap the
    /// batch at `n_qpus` (one circuit per QPU per call).
    fn max_batch_size(&self, total: usize) -> usize {
        total
    }

    /// Release any held resources (SLURM jobs, cloud sessions, QPU reservations, …).
    fn close(&self) {}
}