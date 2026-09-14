use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{max_statevector_concurrency, BoundCircuit, ExecutionConfig, QuantumBackend};
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
    ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
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
                qcs_pylist
                    .append(qc.to_py_object(py)?)
                    .map_err(|e| BackendError::Conversion(e.to_string()))?;
            }

            let module = PyModule::import(py, "polypus_python").map_err(BackendError::Seam)?;
            let connection = module
                .call_method("connect_to_infrastructure", ("local",), None)
                .map_err(|e| {
                    // Surface the failure before it crosses the FFI as a Python
                    // exception (mirrors the QMIO/CUNQA error paths).
                    log::error!("local infrastructure connection failed: {e}");
                    BackendError::Seam(e)
                })?;
            // The call above succeeded; a wrong-shaped return value is our
            // Rust-side conversion failure, not a seam exception (contract C-1).
            let connection_str = connection.extract::<String>().map_err(|e| {
                BackendError::Conversion(format!(
                    "expected connect_to_infrastructure(\"local\") to return str: {e}"
                ))
            })?;

            let kwargs = PyDict::new(py);
            let conv = |e: PyErr| BackendError::Conversion(e.to_string());
            kwargs.set_item("id", &config.id).map_err(conv)?;
            kwargs.set_item("backend", &self.backend).map_err(conv)?;
            kwargs.set_item("qcs", qcs_pylist).map_err(conv)?;
            kwargs.set_item("shots", config.shots).map_err(conv)?;
            kwargs
                .set_item("sim_method", &self.sim_method)
                .map_err(conv)?;
            // Bound Aer's concurrent experiments to the statevector memory budget
            // (plan §4.5, P1-memory): Aer's `max_parallel_experiments=0` default
            // ("auto") spawns one process per experiment and OOMs at high qubit
            // counts. This is a pure resource knob — Aer seeds each experiment
            // deterministically, so the counts are unchanged for any bound.
            let cores = std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1);
            let max_parallel_experiments =
                max_statevector_concurrency(widest_qubits(qcs, py), cores);
            kwargs
                .set_item("max_parallel_experiments", max_parallel_experiments)
                .map_err(conv)?;
            if let Some(nm) = &self.noise_model {
                kwargs
                    .set_item("noise_model", nm.clone_ref(py))
                    .map_err(conv)?;
            }
            // Forwarded to Aer's `seed_simulator` on the Python side (contract
            // C-7); `None` is simply omitted rather than sent as `seed=None`,
            // so Aer's own unseeded default behavior is unchanged.
            if let Some(seed) = config.seed {
                kwargs.set_item("seed", seed).map_err(conv)?;
            }

            let result = module
                .call_method("run_qcs", (connection_str,), Some(&kwargs))
                .map_err(|e| {
                    log::error!("local circuit execution failed: {e}");
                    BackendError::Seam(e)
                })?;
            // As above: `run_qcs` returned successfully, so a wrong-shaped
            // value is a Rust-side conversion failure, not a seam exception.
            result.extract::<Vec<HashMap<String, u64>>>().map_err(|e| {
                BackendError::Conversion(format!(
                    "expected run_qcs() to return list[dict[str, int]] (contract C-1): {e}"
                ))
            })
        })
    }
}

/// The widest circuit in the batch, used to size Aer's `max_parallel_experiments`
/// against the statevector memory budget (plan §4.5). Each variant's width is
/// read where it is cheapest: `Native` exposes it directly, `Qasm2` is parsed for
/// it, and a `Qiskit` circuit is read through the GIL (`num_qubits`), which the
/// caller already holds. An empty or unreadable batch yields 0 — a one-amplitude
/// budget that leaves Aer's parallelism at the core count.
fn widest_qubits(qcs: &[BoundCircuit], py: Python<'_>) -> usize {
    qcs.iter()
        .filter_map(|qc| match qc {
            BoundCircuit::Native(cc) => Some(cc.num_qubits),
            BoundCircuit::Qasm2(qasm) => polypus_circuit::ParameterizedCircuit::from_qasm2(qasm)
                .ok()
                .map(|pc| pc.num_qubits),
            BoundCircuit::Qiskit(obj) => obj.bind(py).getattr("num_qubits").ok()?.extract().ok(),
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::ParameterizedCircuit;

    /// `widest_qubits` budgets for the largest statevector in a mixed batch: a
    /// too-small `n` would under-bound Aer and reintroduce the OOM. The `Native`
    /// and `Qasm2` arms are GIL-free; the GIL is only held to satisfy the
    /// signature (the `Qiskit` arm is exercised end-to-end by the Python suite).
    #[test]
    fn widest_qubits_picks_the_largest_circuit() {
        pyo3::prepare_freethreaded_python();
        let small = ParameterizedCircuit::new(2)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();
        let big = ParameterizedCircuit::new(7)
            .h(0)
            .measure_all()
            .assign_parameters(&[])
            .unwrap();
        let batch = vec![
            BoundCircuit::Native(small),
            BoundCircuit::Qasm2(big.to_qasm2()),
        ];
        Python::with_gil(|py| {
            assert_eq!(widest_qubits(&batch, py), 7);
            assert_eq!(widest_qubits(&[], py), 0);
        });
    }
}
