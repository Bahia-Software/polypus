pub mod vqc_oracle;
pub mod qml_oracle;

pub use vqc_oracle::VqcOracle;
pub use qml_oracle::QmlOracle;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use polypus_circuit::ParameterizedCircuit;
use crate::infrastructure::{BoundCircuit, QuantumBackend, ExecutionConfig};

/// A parameterised circuit template, in one of the representations Polypus
/// supports as optimisation targets.
///
/// The variant determines *where* per-candidate parameter binding happens:
///
/// - [`Qiskit`](CircuitSource::Qiskit): `assign_parameters` is called on the
///   Python object — requires the GIL for every candidate.
/// - [`Native`](CircuitSource::Native): binding + OpenQASM 2.0 generation run
///   in pure Rust — **no GIL**, so candidates can be bound truly in parallel
///   and the only remaining Python touchpoint is the simulator call itself.
pub enum CircuitSource {
    /// A Qiskit `QuantumCircuit` with unbound `Parameter`s.
    Qiskit(Py<PyAny>),
    /// A native Rust circuit from `polypus-circuit`.
    Native(ParameterizedCircuit),
}

impl CircuitSource {
    /// Bind one candidate parameter vector, producing an executable circuit.
    ///
    /// # Panics
    ///
    /// Panics if binding fails (wrong number of parameters, Python error).
    /// Entry points validate `dimensions` against the template up front, so a
    /// failure here is a programming error, consistent with the rest of the
    /// evaluation layer.
    pub fn bind(&self, params: &[f64]) -> BoundCircuit {
        match self {
            CircuitSource::Qiskit(circuit) => {
                BoundCircuit::Qiskit(assign_parameters_qiskit(circuit, params))
            }
            // Pure Rust: no GIL anywhere on this path. The bound circuit keeps
            // its native structure so the statevector backend can simulate it
            // directly; Python backends serialise it to OpenQASM 2.0 on demand.
            CircuitSource::Native(circuit) => BoundCircuit::Native(
                circuit
                    .assign_parameters(params)
                    .unwrap_or_else(|e| panic!("Error binding native circuit: {e}")),
            ),
        }
    }

    /// Number of free parameters, when it can be known without Python
    /// (`None` for Qiskit circuits — querying them needs the GIL and is done
    /// at the entry points instead).
    pub fn num_params(&self) -> Option<usize> {
        match self {
            CircuitSource::Qiskit(_) => None,
            CircuitSource::Native(c) => Some(c.num_params),
        }
    }
}

/// Bind `params` to a copy of a Qiskit `circuit` and return the bound circuit.
pub(crate) fn assign_parameters_qiskit(circuit: &Py<PyAny>, params: &[f64]) -> Py<PyAny> {
    Python::with_gil(|py| {
        let qc = circuit
            .clone_ref(py)
            .into_pyobject(py)
            .expect("Failed to get circuit as PyObject");
        let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
        qc.call_method("assign_parameters", (params.to_vec(),), Some(&kwargs))
            .expect("Error assigning parameters to circuit")
            .unbind()
    })
}

/// Contract between optimization algorithms and quantum circuit evaluation.
///
/// An oracle encapsulates everything needed to translate a parameter vector
/// into a scalar fitness value: the circuit template (or training circuits),
/// the backend, and the expectation function.
///
/// Algorithms only call [`EvaluationOracle::evaluate_batch`] and have no
/// knowledge of circuits, QPUs, infrastructure, or training modes.
///
/// To add a new evaluation strategy (e.g. noisy readout mitigation, hardware
/// native gates, …) implement this trait without touching any algorithm.
pub trait EvaluationOracle: Send + Sync {
    /// Evaluate a batch of candidate parameter vectors.
    ///
    /// Returns one fitness value per candidate. Higher is better
    /// (algorithms maximise the expectation value).
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64>;
}

/// Execute a batch of bound circuits through `backend` and extract expectation
/// values using the Python `expectation_fn`.
///
/// This is the **single place** in the codebase that calls
/// `polypus_python.expectation_values`, eliminating the duplication that
/// previously existed across DE, PSO, QNG, and the orchestration layer.
pub(crate) fn run_and_expect(
    backend: &dyn QuantumBackend,
    qcs: &[BoundCircuit],
    config: &ExecutionConfig,
    expectation_fn: &Py<PyAny>,
) -> Vec<f64> {
    let counts = backend.run_circuits(qcs, config);
    Python::with_gil(|py| {
        // Convert the native counts back into a Python `list[dict]` for the
        // Python `expectation_values` function. Once expectation computation is
        // also native this round-trip disappears entirely.
        let py_counts = counts
            .into_pyobject(py)
            .expect("Failed to convert counts to a Python object");
        PyModule::import(py, "polypus_python")
            .expect("Failed to import polypus_python")
            .call_method("expectation_values", (py_counts, expectation_fn), None)
            .expect("Error computing expectation values")
            .extract::<Vec<f64>>()
            .expect("Failed to extract expectation values as Vec<f64>")
    })
}
