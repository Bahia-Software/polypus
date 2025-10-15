use pyo3::prelude::*;
use pyo3::types::{PyInt, PyAny, PyModule};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

// use crate::algorithms::{AlgorithmArgs, AlgorithmDifferentialEvolutionArgs, AlgorithmSingleRun, AlgorithmTrait};
use crate::algorithms::{AlgorithmArgs, AlgorithmTrait, AlgorithmSingleRun};
use crate::algorithms::{AlgorithmDifferentialEvolutionArgs, AlgorithmDifferentialEvolution};

/// Function to run a quantum circuit called from Python.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1))]
pub fn run_quantum_circuit<'py>(
	qc: Bound<'py, PyAny>,
	shots: Option<Bound<'py, PyInt>>,
	infrastructure: Option<String>,
	n_qpus: Option<u32>,
) -> pyo3::PyObject {
	let n_qpus = n_qpus.unwrap_or(1);
	let infrastructure = infrastructure.unwrap_or_else(|| "local".to_string());
	let id: String = format!("run_{}_{}", n_qpus, infrastructure);
	// setup_logger(&format!("logger_{}.log", id));
	let args = AlgorithmArgs {
		id: id.clone(),
		qc,
		shots,
		n_qpus: Some(n_qpus),
		infrastructure,
		backend: "AerSimulator".to_string(),
		nodes: 1,
	};
	let algorithm = AlgorithmSingleRun;
	algorithm.run(args)
}

/// Function to raise QPUs in an infrastructure called from Python.
#[pyfunction]
pub fn raise_qpus() {
	// Placeholder for raising QPUs if needed.
	log::debug!("raise_qpus function called, but not implemented yet.");
}

/// Function to train a QAOA using differential evolution called from Python.
#[pyfunction(signature = (qc, shots=None, n_qpus=None, expectation_function=None, generations=None, population_size=None, dimensions=None, infrastructure=None, nodes=None, id=None, tolerance=0.01))]
pub fn differential_evolution<'py>(
	qc: Bound<'py, PyAny>,
	shots: Option<Bound<'py, PyInt>>,
	n_qpus: Option<u32>,
	expectation_function: Option<Bound<'py, PyAny>>,
	generations: Option<usize>,
	population_size: Option<usize>,
	dimensions: Option<usize>,
	infrastructure: Option<String>,
	nodes: Option<u32>,
	id: Option<String>,
	tolerance: Option<f64>,
) -> pyo3::PyObject {
	// setup_logger(&format!("logger_{}.log", id.as_ref().unwrap_or(&"default_id".to_string())));
	let args = AlgorithmArgs {
		id: id.unwrap_or_else(|| "default_id".to_string()),
		qc,
		shots,
		n_qpus: Some(n_qpus.expect("nqpus required")),
		infrastructure: infrastructure.unwrap_or_else(|| "local".to_string()),
		backend: "AerSimulator".to_string(),
		nodes: nodes.unwrap_or(1),
	};
	let differential_evolution_args = AlgorithmDifferentialEvolutionArgs {
		base: args,
		population_size,
		generations,
		dimensions,
		expectation_function,
		tolerance,
	};
	let algorithm = AlgorithmDifferentialEvolution;
	algorithm.run(differential_evolution_args)
}

#[pymodule]
fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
	m.add_function(wrap_pyfunction!(raise_qpus, m)?)?;
	m.add_function(wrap_pyfunction!(differential_evolution, m)?)?;
	Ok(())
}
