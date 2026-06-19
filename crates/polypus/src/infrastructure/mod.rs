use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub mod execution_config;
pub mod local;
pub mod cunqa;
pub mod native;
#[cfg(feature = "qmio")]
pub mod qmio;

pub use execution_config::{ExecutionConfig, BackendConfig};
pub use local::LocalBackend;
pub use cunqa::CunqaBackend;
pub use native::NativeStatevectorBackend;
#[cfg(feature = "qmio")]
pub use qmio::QmioBackend;

/// Supported quantum execution infrastructures.
pub enum Infrastructure {
    Local,
    Cunqa,
    /// CESGA QMIO real QPU (see [`QmioBackend`]). The variant always exists so
    /// that selecting `"qmio"` produces a clear "requires `--features qmio`"
    /// error rather than an `Unknown infrastructure` panic when the feature is
    /// disabled; the backend itself is only built with the feature on.
    Qmio,
}

impl Infrastructure {
    pub fn from_str(s: &str) -> Self {
        match s {
            "local" => Infrastructure::Local,
            "cunqa" => Infrastructure::Cunqa,
            "qmio" => Infrastructure::Qmio,
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
            BackendConfig::LocalNative => {
                Arc::new(NativeStatevectorBackend::new(&config.id))
            }
            #[cfg(feature = "qmio")]
            BackendConfig::Qmio {
                endpoint,
                program_format,
                optimization,
                repetition_period,
                res_format,
            } => Arc::new(QmioBackend::new(
                endpoint.clone(),
                *program_format,
                *optimization,
                *repetition_period,
                res_format.clone(),
            )),
        }
    }
}

/// A fully bound (parameter-free) circuit ready for execution, in one of the
/// representations Polypus supports.
///
/// Backends receive a slice of these and decide how to consume each variant:
/// Python-based backends convert `Qasm2` to a `QuantumCircuit` inside the
/// Python layer; future native or wire-protocol backends can submit the QASM
/// text directly without ever touching the GIL.
///
/// Being an enum (rather than `Py<PyAny>`) makes the contract explicit: adding
/// a new circuit representation is a compile-time-checked change in every
/// backend, not a runtime surprise.
pub enum BoundCircuit {
    /// A bound Qiskit `QuantumCircuit` (Python object).
    Qiskit(Py<PyAny>),
    /// An OpenQASM 2.0 program produced by the native Rust circuit layer.
    Qasm2(String),
    /// A fully bound native circuit from `polypus-circuit`. Carries the circuit
    /// structure directly so the native statevector backend can simulate it
    /// without any OpenQASM round-trip or GIL; Python-based backends serialise
    /// it to OpenQASM 2.0 on demand in [`to_py_object`](Self::to_py_object).
    Native(polypus_circuit::ConcreteCircuit),
}

impl BoundCircuit {
    /// Convert to the Python object expected by `polypus_python.run_qcs`:
    /// the Qiskit circuit as-is, or the QASM program as a `str` (the Python
    /// layer parses/forwards it per infrastructure).
    pub fn to_py_object(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            BoundCircuit::Qiskit(qc) => qc.clone_ref(py),
            BoundCircuit::Qasm2(qasm) => qasm
                .into_pyobject(py)
                .expect("Failed to convert QASM string to Python")
                .into_any()
                .unbind(),
            // Native circuits reach a Python backend (Aer/CUNQA) as OpenQASM 2.0,
            // exactly like the `Qasm2` variant; the conversion is pure Rust.
            BoundCircuit::Native(circuit) => circuit
                .to_qasm2()
                .into_pyobject(py)
                .expect("Failed to convert QASM string to Python")
                .into_any()
                .unbind(),
        }
    }

    /// Cheap copy. Only the `Qiskit` variant needs the GIL (reference-count
    /// bump); the `Qasm2` and `Native` variants are plain Rust clones.
    pub fn duplicate(&self) -> BoundCircuit {
        match self {
            BoundCircuit::Qiskit(qc) => {
                Python::with_gil(|py| BoundCircuit::Qiskit(qc.clone_ref(py)))
            }
            BoundCircuit::Qasm2(qasm) => BoundCircuit::Qasm2(qasm.clone()),
            BoundCircuit::Native(circuit) => BoundCircuit::Native(circuit.clone()),
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
    fn run_circuits(&self, qcs: &[BoundCircuit], config: &ExecutionConfig) -> Vec<HashMap<String, u64>>;

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