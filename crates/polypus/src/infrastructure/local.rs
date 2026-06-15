use crate::infrastructure::{BoundCircuit, QuantumBackend, ExecutionConfig};
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
}

impl LocalBackend {
    pub fn new(backend: String, sim_method: String, noise_model: Option<Py<PyAny>>) -> Self {
        LocalBackend { backend, sim_method, noise_model }
    }
}

impl QuantumBackend for LocalBackend {
    fn run_circuits(&self, qcs: &[BoundCircuit], config: &ExecutionConfig) -> Vec<HashMap<String, u64>> {
        Python::with_gil(|py| {
            // Qiskit circuits pass through as-is; native circuits travel as
            // OpenQASM 2.0 strings and are parsed by the Python layer.
            let qcs_pylist = PyList::empty(py);
            for qc in qcs {
                qcs_pylist.append(qc.to_py_object(py)).unwrap();
            }

            let module = PyModule::import(py, "polypus_python").unwrap();
            let connection = module
                .call_method("connect_to_infrastructure", ("local",), None)
                .expect("Error connecting to local infrastructure");
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
                .expect("Error running run_qcs")
                .extract::<Vec<HashMap<String, u64>>>()
                .expect("run_qcs must return list[dict[str, int]]")
        })
    }
}