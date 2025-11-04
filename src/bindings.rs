use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

// use crate::algorithms::{AlgorithmArgs, AlgorithmDifferentialEvolutionArgs, AlgorithmSingleRun, AlgorithmTrait};
use crate::algorithms::{AlgorithmArgs, AlgorithmTrait, AlgorithmSingleRun, DistributeByShotsRun};
use crate::algorithms::vqc::{AlgorithmDifferentialEvolutionArgs, AlgorithmDifferentialEvolution};

/// Function to run a quantum circuit called from Python.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1))]
pub fn run_quantum_circuit<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	infrastructure: String,
	n_qpus: u32,
) -> pyo3::PyObject {
	println!("run_quantum_circuit called with qc: {:?}, shots: {}, infrastructure: {}, n_qpus: {}", qc, shots, infrastructure, n_qpus);
	let n_qpus = n_qpus;
	let infrastructure = infrastructure;
	let id: String = format!("run_{}_{}", n_qpus, infrastructure);
	let args = AlgorithmArgs {
		id: id.clone(),
		qcs: vec![qc.unbind()],
		shots,
		n_qpus,
		infrastructure,
		backend: "AerSimulator".to_string(),
		nodes: 1,
	};

	let algorithm: Box<dyn AlgorithmTrait<Args=AlgorithmArgs, AlgorithmReturnType=PyObject>> = if n_qpus == 1 {
        Box::new(AlgorithmSingleRun)
    } else {
        Box::new(DistributeByShotsRun)
    };

	algorithm.run(args)
}

/// Function to train a QAOA using differential evolution called from Python.
#[pyfunction(signature = (qc, shots, n_qpus, expectation_function, generations, population_size, dimensions, infrastructure, nodes, id, tolerance=0.01))]
pub fn differential_evolution<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	n_qpus: u32,
	expectation_function: Bound<'py, PyAny>,
	generations: u32,
	population_size: u32,
	dimensions: u32,
	infrastructure: String,
	nodes: u32,
	id: String,
	tolerance: f64,
) -> pyo3::PyObject {
	// setup_logger(&format!("logger_{}.log", id.as_ref().unwrap_or(&"default_id".to_string())));
	let args = AlgorithmArgs {
		id: id.clone(),
		qcs: vec![qc.unbind()],
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend: "AerSimulator".to_string(),
		nodes,
	};

	let differential_evolution_args = AlgorithmDifferentialEvolutionArgs {
		base: args,
		population_size,
		generations,
		dimensions,
		expectation_function: expectation_function.unbind(),
		tolerance,
	};
	let algorithm = AlgorithmDifferentialEvolution;
	algorithm.run(differential_evolution_args)
}

#[pymodule]
fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
	m.add_function(wrap_pyfunction!(differential_evolution, m)?)?;
	Ok(())
}
