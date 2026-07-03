use crate::infrastructure::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// LocalBackend: runs quantum circuits on the local machine using Qiskit AerSimulator.
pub struct LocalBackend {
    /// Backend/device class name forwarded to Python (e.g. `"AerSimulator"`).
    backend: String,
    /// Aer simulation method.
    sim_method: String,
    /// Optional Qiskit `NoiseModel`.
    noise_model: Option<Py<PyAny>>,
    /// Native-circuit transpiler applied before submission. Aer transpiles
    /// internally, so this defaults to the no-op [`IdentityTranspiler`] and the
    /// observable behavior is unchanged.
    transpiler: Box<dyn Transpiler>,
}

impl LocalBackend {
    pub fn new(backend: String, sim_method: String, noise_model: Option<Py<PyAny>>) -> Self {
        LocalBackend {
            backend,
            sim_method,
            noise_model,
            transpiler: Box::new(IdentityTranspiler),
        }
    }
}

impl QuantumBackend for LocalBackend {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        config: &ExecutionConfig,
    ) -> Vec<HashMap<String, u64>> {
        Python::with_gil(|py| {
            // Native circuits are transpiled in pure Rust before submission;
            // Qiskit circuits pass through untouched (Aer transpiles them) and
            // every native circuit travels to Python as OpenQASM 2.0.
            let opts = TranspileOptions {
                level: config.opt_level,
            };
            let qcs_pylist = PyList::empty(py);
            for qc in qcs {
                let qc = qc.transpiled(self.transpiler.as_ref(), &opts);
                qcs_pylist.append(qc.to_py_object(py)).unwrap();
            }

            let module = PyModule::import(py, "polypus_python").unwrap();
            let connection = module
                .call_method("connect_to_infrastructure", ("local",), None)
                .unwrap_or_else(|e| {
                    // Surface the failure before the panic PyO3 turns into a
                    // Python exception (mirrors the QMIO/CUNQA error paths).
                    log::error!("local infrastructure connection failed: {e}");
                    panic!("Error connecting to local infrastructure: {e}");
                });
            let connection_str = connection.extract::<String>().unwrap();

            let kwargs = PyDict::new(py);
            kwargs.set_item("id", &config.id).unwrap();
            kwargs.set_item("backend", &self.backend).unwrap();
            kwargs.set_item("qcs", qcs_pylist).unwrap();
            kwargs.set_item("shots", config.shots).unwrap();
            kwargs.set_item("sim_method", &self.sim_method).unwrap();
            if let Some(nm) = &self.noise_model {
                kwargs.set_item("noise_model", nm.clone_ref(py)).unwrap();
            }

            module
                .call_method("run_qcs", (connection_str,), Some(&kwargs))
                .unwrap_or_else(|e| {
                    log::error!("local circuit execution failed: {e}");
                    panic!("Error running run_qcs: {e}");
                })
                .extract::<Vec<HashMap<String, u64>>>()
                .expect("run_qcs must return list[dict[str, int]]")
        })
    }
}
