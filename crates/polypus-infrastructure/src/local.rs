use crate::error::BackendError;
use crate::transpiler::{IdentityTranspiler, TranspileOptions, Transpiler};
use crate::{
    max_statevector_concurrency, BackendCapabilities, BoundCircuit, CircuitTask, ExecutionConfig,
    QuantumBackend,
};
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

    /// Report the wave size the planner should use for this batch, derived from
    /// the same statevector memory budget [`run_circuits`](Self::run_circuits)
    /// hands Aer as `max_parallel_experiments` — the batch's widest circuit fed to
    /// [`max_statevector_concurrency`] against `available_parallelism()`, read off
    /// the task slice without cloning any circuit. The GIL is acquired solely to
    /// read a `Qiskit` circuit's `num_qubits` (as `run_circuits` does);
    /// `Native`/`Qasm2` widths need no interpreter.
    ///
    /// The wave size is capped **only when the whole batch cannot be held in the
    /// memory budget at once** (the high-qubit regime): there splitting into waves
    /// lets the planner run a `py.check_signals()` between them (issue #147). When
    /// the batch fits, the cap is reported as unbounded so the whole population
    /// reaches Aer as a **single** call — Aer parallelises the experiments
    /// internally up to `max_parallel_experiments`, so splitting would only add
    /// per-call overhead (see `tests/python/test_qml_concurrency.py`). See
    /// `wave_concurrency` for why the fit test uses the
    /// pure memory limit rather than a core-count gate (which would degenerate on a
    /// single-core host, issue #147's 1-thread case).
    fn capabilities_for(&self, tasks: &[CircuitTask<'_>]) -> BackendCapabilities {
        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let widest = Python::with_gil(|py| {
            tasks
                .iter()
                .filter_map(|t| circuit_qubits(t.circuit, py))
                .max()
                .unwrap_or(0)
        });
        BackendCapabilities {
            max_concurrency: crate::wave_concurrency(widest, cores, tasks.len()),
            supports_shot_distribution: true,
        }
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
        .filter_map(|qc| circuit_qubits(qc, py))
        .max()
        .unwrap_or(0)
}

/// Qubit width of a single circuit, read where it is cheapest: `Native` exposes it
/// directly, `Qasm2` is parsed for it, and a `Qiskit` circuit is read through the
/// GIL (`num_qubits`), which the caller already holds. Shared by [`widest_qubits`]
/// (over the batch `run_circuits` receives) and
/// [`LocalBackend::capabilities_for`] (over the planner's task slice), so both
/// size Aer's memory budget by the identical per-circuit rule.
fn circuit_qubits(qc: &BoundCircuit, py: Python<'_>) -> Option<usize> {
    match qc {
        BoundCircuit::Native(cc) => Some(cc.num_qubits),
        BoundCircuit::Qasm2(qasm) => polypus_circuit::ParameterizedCircuit::from_qasm2(qasm)
            .ok()
            .map(|pc| pc.num_qubits),
        BoundCircuit::Qiskit(obj) => obj.bind(py).getattr("num_qubits").ok()?.extract().ok(),
    }
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

    /// Issue #147: a high-qubit batch that cannot fit in the memory budget is
    /// reported with the memory cap `run_circuits` hands Aer as
    /// `max_parallel_experiments`, so the planner can split it into memory-safe
    /// waves. A 30-qubit statevector is 16 GiB, so under the default 16 GiB budget
    /// only one fits at a time, and a 4-circuit batch cannot be held at once ⇒
    /// cap 1 — **regardless of the host core count** (the single-core regression
    /// from issue #147, so the test deliberately does not guard on the core
    /// count). The circuit is zero-gate and never submitted to Aer, so this costs
    /// nothing. The batch-agnostic `capabilities()` is left at its unbounded
    /// default (purely additive).
    #[test]
    fn capabilities_for_exposes_the_memory_cap_leaving_capabilities_unchanged() {
        pyo3::prepare_freethreaded_python();
        let wide = BoundCircuit::Native(
            ParameterizedCircuit::new(30)
                .assign_parameters(&[])
                .unwrap(),
        );
        let tasks: Vec<CircuitTask> = (0..4)
            .map(|_| CircuitTask {
                circuit: &wide,
                shots: 8,
            })
            .collect();
        let backend =
            LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);

        let cap = backend.capabilities_for(&tasks).max_concurrency;
        assert_eq!(
            cap, 1,
            "a 30-qubit statevector (16 GiB) admits one at a time under the 16 GiB default"
        );
        assert!(
            cap < tasks.len(),
            "the cap must force more than one wave for this batch"
        );

        // The frozen, batch-agnostic seam is untouched.
        assert_eq!(backend.capabilities().max_concurrency, usize::MAX);
        assert!(backend.capabilities().supports_shot_distribution);
    }

    /// A low-qubit batch is reported as a **single unbounded wave**, so the whole
    /// population reaches Aer in one call (issue #147 must not regress the
    /// single-batch contract of `tests/python/test_qml_concurrency.py`).
    #[test]
    fn capabilities_for_keeps_a_single_wave_at_low_qubits() {
        pyo3::prepare_freethreaded_python();
        let small =
            BoundCircuit::Native(ParameterizedCircuit::new(2).assign_parameters(&[]).unwrap());
        let tasks: Vec<CircuitTask> = (0..64)
            .map(|_| CircuitTask {
                circuit: &small,
                shots: 8,
            })
            .collect();
        let backend =
            LocalBackend::new("AerSimulator".to_string(), "statevector".to_string(), None);
        assert_eq!(
            backend.capabilities_for(&tasks).max_concurrency,
            usize::MAX,
            "a low-qubit population must stay a single wave"
        );
    }
}
