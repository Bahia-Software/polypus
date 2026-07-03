use crate::infrastructure::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// CunqaBackend: runs quantum circuits on the CUNQA distributed QPU platform.
pub struct CunqaBackend {
    family: Option<Py<PyAny>>,
    /// Guards against double-release of the QPU allocation (explicit `close()`
    /// followed by `Drop`, or vice versa).
    closed: AtomicBool,
    /// Backend/device class name forwarded to CUNQA.
    backend: String,
    /// Simulation method for CUNQA's simulated QPUs.
    sim_method: String,
    /// Number of physical QPUs in the allocation. Bounds how many circuits can
    /// be dispatched per `run_circuits` call (one circuit per QPU).
    n_qpus: u32,
    /// Native-circuit transpiler applied before submission. CUNQA's simulated
    /// QPUs transpile internally, so this defaults to the no-op
    /// [`IdentityTranspiler`] and the observable behavior is unchanged.
    transpiler: Box<dyn Transpiler>,
}

impl QuantumBackend for CunqaBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Vec<HashMap<String, u64>> {
        Python::with_gil(|py| {
            // Native circuits are transpiled in pure Rust before submission;
            // Qiskit circuits pass through untouched and every native circuit
            // travels to Python as OpenQASM 2.0.
            let opts = TranspileOptions {
                level: config.opt_level,
            };
            let qcs_pylist = PyList::empty(py);
            for qc in qcs {
                let qc = qc.transpiled(self.transpiler.as_ref(), &opts);
                qcs_pylist.append(qc.to_py_object(py)).unwrap();
            }

            let module = PyModule::import(py, "polypus_python").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("family_id", &config.id).unwrap();
            kwargs.set_item("backend", &self.backend).unwrap();
            kwargs.set_item("qcs", qcs_pylist).unwrap();
            kwargs.set_item("shots", config.shots).unwrap();
            kwargs.set_item("sim_method", &self.sim_method).unwrap();

            module
                .call_method("run_qcs", ("cunqa",), Some(&kwargs))
                .unwrap_or_else(|e| {
                    // Surface the failure before the panic PyO3 turns into a
                    // Python exception (mirrors the QMIO backend's error paths).
                    log::error!("CUNQA circuit execution failed: {e}");
                    panic!("Error running circuits on CUNQA: {e}");
                })
                .extract::<Vec<HashMap<String, u64>>>()
                .expect("run_qcs must return list[dict[str, int]]")
        })
    }

    fn max_batch_size(&self, _total: usize) -> usize {
        // CUNQA dispatches one circuit per QPU, so a single call can carry at
        // most `n_qpus` circuits.
        self.n_qpus as usize
    }

    fn close(&self) {
        // Idempotent: only the first call actually releases the allocation.
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.drop_qpus();
    }
}

/// RAII guarantee: QPUs are released even if the algorithm panics or returns
/// early. This is essential for the HPC use case, where leaking a SLURM
/// allocation would keep nodes reserved for the full requested walltime.
impl Drop for CunqaBackend {
    fn drop(&mut self) {
        self.close();
    }
}

impl CunqaBackend {
    pub fn new(
        n_qpus: u32,
        nodes: u32,
        id: &str,
        cores_per_qpu: u32,
        backend: String,
        sim_method: String,
    ) -> Self {
        let mut backend = CunqaBackend {
            family: None,
            closed: AtomicBool::new(false),
            backend,
            sim_method,
            n_qpus,
            transpiler: Box::new(IdentityTranspiler),
        };
        backend.raise_qpus(n_qpus, nodes, id, cores_per_qpu);
        backend
    }

    fn raise_qpus(&mut self, n_qpus: u32, nodes: u32, id: &str, cores_per_qpu: u32) {
        // Allocating QPUs reserves an HPC (SLURM) allocation — a rare, coarse
        // lifecycle event an operator wants to see at the default level.
        log::info!(
            "Raising QPUs in CUNQA: n_qpus={n_qpus}, nodes={nodes}, id={id}, cores_per_qpu={cores_per_qpu}"
        );
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("n", n_qpus).unwrap();
            kwargs.set_item("t", "10:00:00").unwrap();
            kwargs.set_item("n_nodes", nodes).unwrap();
            kwargs.set_item("family_name", id).unwrap();
            kwargs.set_item("cores_per_qpu", cores_per_qpu).unwrap();

            let module = PyModule::import(py, "polypus_python").unwrap();
            let connection = module
                .call_method("connect_to_infrastructure", ("cunqa",), Some(&kwargs))
                .unwrap_or_else(|e| {
                    log::error!("CUNQA QPU allocation failed: {e}");
                    panic!("Error raising QPUs in CUNQA: {e}");
                });
            let family: Py<PyAny> = connection
                .extract()
                .expect("Error extracting family from CUNQA");
            log::info!("QPUs raised successfully");
            self.family = Some(family);
        });
    }

    fn drop_qpus(&self) {
        // Nothing to release if the allocation was never established. Guarding
        // here is important because `drop_qpus` runs from `Drop`, and a panic
        // while unwinding would abort the entire process.
        let Some(family) = self.family.as_ref() else {
            return;
        };
        log::info!("Dropping QPUs");
        Python::with_gil(|py| {
            let module = PyModule::import(py, "polypus_python").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("family", family.clone_ref(py)).unwrap();
            module
                .call_method("disconnect_from_infrastructure", ("cunqa",), Some(&kwargs))
                .unwrap_or_else(|e| {
                    log::error!("CUNQA QPU release failed: {e}");
                    panic!("Error dropping QPUs in CUNQA: {e}");
                });
            log::info!("QPUs dropped successfully");
        });
    }
}
