use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

pub mod de;
pub mod pso;
pub mod qng;

use de::DE;
use pso::PSO;
use qng::QNG;

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

/// Unified entry point: train a variational quantum circuit with a chosen optimizer.
///
/// `method` must be an instance of `DE`, `PSO`, or `QNG`.
///
/// Example::
///
///     result = polypus.train(
///         qc, polypus.DE(generations=200, population_size=50),
///         shots=1024, n_qpus=1, dimensions=4,
///         expectation_function=my_cost,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
///     )
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id))]
pub fn train<'py>(
	qc: Bound<'py, PyAny>,
	method: Bound<'py, PyAny>,
	shots: u32,
	n_qpus: u32,
	dimensions: u32,
	expectation_function: Bound<'py, PyAny>,
	infrastructure: String,
	nodes: u32,
	cores_per_qpu: u32,
	id: String,
) -> PyResult<PyObject> {
	let base = AlgorithmArgs {
		id: id.clone(),
		qcs: vec![qc.unbind()],
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend: "AerSimulator".to_string(),
		nodes,
		cores_per_qpu,
	};

	if let Ok(de) = method.extract::<PyRef<DE>>() {
		let args = AlgorithmDifferentialEvolutionArgs {
			base,
			population_size: de.population_size,
			generations: de.generations,
			dimensions,
			expectation_function: expectation_function.unbind(),
			tolerance: de.tolerance,
		};
		return Ok(AlgorithmDifferentialEvolution.run(args));
	}

	if let Ok(pso) = method.extract::<PyRef<PSO>>() {
		let args = AlgorithmPSOArgs {
			base,
			population_size: pso.population_size,
			generations: pso.generations,
			dimensions,
			bounds: pso.bounds,
			inertia_weight: pso.inertia_weight,
			cognitive_weight: pso.cognitive_weight,
			social_weight: pso.social_weight,
			expectation_function: expectation_function.unbind(),
			tolerance: pso.tolerance,
		};
		return Ok(AlgorithmPSO.run(args));
	}

	if let Ok(qng) = method.extract::<PyRef<QNG>>() {
		let args = AlgorithmQNGArgs {
			base,
			max_iters: qng.max_iters,
			learning_rate: qng.learning_rate,
			finite_difference_step: qng.finite_difference_step,
			bounds: qng.bounds,
			dimensions,
			expectation_function: expectation_function.unbind(),
			variance_function: qng.variance_function.clone_ref(method.py()),
			tikhonov_reg: qng.tikhonov_reg,
		};
		return Ok(AlgorithmQNG.run(args));
	}

	Err(pyo3::exceptions::PyTypeError::new_err(
		"method must be an instance of polypus.DE, polypus.PSO, or polypus.QNG"
	))
}

#[pymodule]
pub fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<DE>()?;
	m.add_class::<PSO>()?;
	m.add_class::<QNG>()?;
	m.add_function(wrap_pyfunction!(train, m)?)?;
	m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
	Ok(())
}
