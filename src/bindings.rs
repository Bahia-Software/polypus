use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

// use crate::algorithms::{AlgorithmArgs, AlgorithmDifferentialEvolutionArgs, AlgorithmSingleRun, AlgorithmTrait};
use crate::algorithms::{AlgorithmArgs, AlgorithmTrait, AlgorithmSingleRun, DistributeByShotsRun};
use crate::algorithms::vqc::{AlgorithmDifferentialEvolutionArgs, AlgorithmDifferentialEvolution};
use crate::algorithms::vqc::{AlgorithmPSOArgs, AlgorithmPSO};
use crate::algorithms::vqc::{AlgorithmQNGArgs, AlgorithmQNG};

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
		cores_per_qpu: 2,
	};

	let algorithm: Box<dyn AlgorithmTrait<Args=AlgorithmArgs, AlgorithmReturnType=PyObject>> = if n_qpus == 1 {
        Box::new(AlgorithmSingleRun)
    } else {
        Box::new(DistributeByShotsRun)
    };

	algorithm.run(args)
}

/// Function to train a QAOA using differential evolution called from Python.
#[pyfunction(signature = (qc, shots, n_qpus, expectation_function, generations, population_size, dimensions, infrastructure, nodes, cores_per_qpu, id, tolerance=0.01))]
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
	cores_per_qpu: u32,
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
		cores_per_qpu,
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

/// Function to train a QAOA using Particle Swarm Optimization called from Python.
#[pyfunction(signature = (qc, shots, n_qpus, expectation_function, generations, population_size, dimensions, bounds, infrastructure, nodes, cores_per_qpu, id, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0, tolerance=0.01))]
pub fn particle_swarm_optimization<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	n_qpus: u32,
	expectation_function: Bound<'py, PyAny>,
	generations: u32,
	population_size: u32,
	dimensions: u32,
	bounds: (f64, f64),
	infrastructure: String,
	nodes: u32,
	cores_per_qpu: u32,
	id: String,
	inertia_weight: f64,
	cognitive_weight: f64,
	social_weight: f64,
	tolerance: f64,
) -> pyo3::PyObject {
	let args = AlgorithmArgs {
		id: id.clone(),
		qcs: vec![qc.unbind()],
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend: "AerSimulator".to_string(),
		nodes,
		cores_per_qpu,
	};

	let pso_args = AlgorithmPSOArgs {
		base: args,
		population_size,
		generations,
		dimensions,
		bounds,
		inertia_weight,
		cognitive_weight,
		social_weight,
		expectation_function: expectation_function.unbind(),
		tolerance,
	};
	let algorithm = AlgorithmPSO;
	algorithm.run(pso_args)
}

/// Function to train a QAOA using Quantum Natural Gradient called from Python.
#[pyfunction(signature = (qc, shots, n_qpus, expectation_function, variance_function, max_iters, dimensions, bounds, infrastructure, nodes, cores_per_qpu, id, learning_rate=0.1, finite_difference_step=0.1, tikhonov_reg=0.05))]
pub fn quantum_natural_gradient<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	n_qpus: u32,
	expectation_function: Bound<'py, PyAny>,
	variance_function: Bound<'py, PyAny>,
	max_iters: u32,
	dimensions: u32,
	bounds: (f64, f64),
	infrastructure: String,
	nodes: u32,
	cores_per_qpu: u32,
	id: String,
	learning_rate: f64,
	finite_difference_step: f64,
	tikhonov_reg: f64,
) -> pyo3::PyObject {
	let args = AlgorithmArgs {
		id: id.clone(),
		qcs: vec![qc.unbind()],
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend: "AerSimulator".to_string(),
		nodes,
		cores_per_qpu,
	};
	let qng_args = AlgorithmQNGArgs {
		base: args,
		max_iters,
		learning_rate,
		finite_difference_step,
		bounds,
		dimensions,
		expectation_function: expectation_function.unbind(),
		variance_function: variance_function.unbind(),
		tikhonov_reg,
	};
	let algorithm = AlgorithmQNG;
	algorithm.run(qng_args)
}

#[pymodule]
fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
	m.add_function(wrap_pyfunction!(differential_evolution, m)?)?;
	m.add_function(wrap_pyfunction!(particle_swarm_optimization, m)?)?;
	m.add_function(wrap_pyfunction!(quantum_natural_gradient, m)?)?;
	Ok(())
}
