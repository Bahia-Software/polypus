use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, IntoPyDict};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

pub mod de;
pub mod pso;
pub mod qng;

use de::DE;
use pso::PSO;
use qng::QNG;

use crate::algorithms::{AlgorithmArgs, AlgorithmTrait, AlgorithmSingleRun, DistributeByShotsRun, TrainMode};
use crate::algorithms::vqc::{AlgorithmDifferentialEvolutionArgs, AlgorithmDifferentialEvolution};
use crate::algorithms::vqc::{AlgorithmPSOArgs, AlgorithmPSO};
use crate::algorithms::vqc::{AlgorithmQNGArgs, AlgorithmQNG};

/// Function to run a quantum circuit called from Python.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1, sim_method="automatic", noise_model=None))]
pub fn run_quantum_circuit<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	infrastructure: String,
	n_qpus: u32,
	sim_method: &str,
	noise_model: Option<Bound<'py, PyAny>>,
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
		sim_method: sim_method.to_string(),
		noise_model: noise_model.map(|nm| nm.unbind()),
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
/// Example:
///
/// ```ignore
///     result = polypus.train(
///         qc, polypus.DE(generations=200, population_size=50),
///         shots=1024, n_qpus=1, dimensions=4,
///         expectation_function=my_cost,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
///     )
/// ```
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None))]
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
	sim_method: &str,
	noise_model: Option<Bound<'py, PyAny>>,
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
		sim_method: sim_method.to_string(),
		noise_model: noise_model.map(|nm| nm.unbind()),
	};

	if let Ok(de) = method.extract::<PyRef<DE>>() {
		let args = AlgorithmDifferentialEvolutionArgs {
			base,
			population_size: de.population_size,
			generations: de.generations,
			dimensions,
			expectation_function: expectation_function.unbind(),
			tolerance: de.tolerance,
			mode: TrainMode::Vqc,
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
			mode: TrainMode::Vqc,
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
			mode: TrainMode::Vqc,
		};
		return Ok(AlgorithmQNG.run(args));
	}

	Err(pyo3::exceptions::PyTypeError::new_err(
		"method must be an instance of polypus.DE, polypus.PSO, or polypus.QNG"
	))
}

/// QML entry point: train a data-encoding VQC where `feature_map` encodes each
/// training sample and `ansatz` holds the trainable weights.
///
/// Internally, this function:
/// 1. Composes `feature_map` and `ansatz` into a single circuit.
/// 2. Pre-binds each row of `x_train` to the feature-map parameters, producing
///    one partially-bound circuit per training sample.
/// 3. Delegates to the chosen optimizer with `TrainMode::Qml`, so that for
///    every candidate parameter vector θ the optimizer binds θ to all training
///    circuits, runs them, and averages the expectation values into a single
///    fitness value.
///
/// Example:
///
/// ```ignore
///     result = polypus.qml.train(
///         feature_map, ansatz, X_train,
///         polypus.PSO(generations=50, population_size=20, bounds=(0, np.pi)),
///         shots=1024, n_qpus=4, dimensions=12,
///         expectation_function=my_loss,
///         infrastructure="local", nodes=1, cores_per_qpu=2, id="qml_run",
///     )
/// ```
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None))]
pub fn qml_train<'py>(
	feature_map: Bound<'py, PyAny>,
	ansatz: Bound<'py, PyAny>,
	x_train: Bound<'py, PyAny>,
	method: Bound<'py, PyAny>,
	shots: u32,
	n_qpus: u32,
	dimensions: u32,
	expectation_function: Bound<'py, PyAny>,
	infrastructure: String,
	nodes: u32,
	cores_per_qpu: u32,
	id: String,
	sim_method: &str,
	noise_model: Option<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
	let py = feature_map.py();

	// 1. Compose feature_map + ansatz
	let composed = feature_map.call_method1("compose", (&ansatz,))?;

	// 2. Add measurements if the composed circuit has no classical bits.
	//    Qiskit's AerSimulator requires classical bits to return counts.
	let num_clbits: usize = composed.getattr("num_clbits")?.extract()?;
	if num_clbits == 0 {
		composed.call_method0("measure_all")?;
	}

	// 3. Collect feature-map parameters in their canonical (sorted-by-name) order
	let fm_params = feature_map.getattr("parameters")?;
	let builtins = PyModule::import(py, "builtins")?;
	let fm_params_list = builtins.call_method1("list", (&fm_params,))?;

	// 4. Pre-bind each training sample to the feature-map parameters.
	//    We pass a dict so Qiskit performs *partial* binding, leaving the ansatz
	//    parameters unbound for the optimizer to fill in later.
	let kwargs_assign = [("inplace", false)].into_py_dict(py)?;
	let mut qcs: Vec<Py<PyAny>> = Vec::new();
	for row_result in x_train.try_iter()? {
		let row = row_result?;
		let param_dict = PyDict::new(py);
		for (param, val) in fm_params_list.try_iter()?.zip(row.try_iter()?) {
			param_dict.set_item(param?, val?)?;
		}
		let bound_qc = composed
			.call_method("assign_parameters", (&param_dict,), Some(&kwargs_assign))?
			.unbind();
		qcs.push(bound_qc);
	}

	if qcs.is_empty() {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"x_train must contain at least one training sample",
		));
	}

	let base = AlgorithmArgs {
		id: id.clone(),
		qcs,
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend: "AerSimulator".to_string(),
		nodes,
		cores_per_qpu,
		sim_method: sim_method.to_string(),
		noise_model: noise_model.map(|nm| nm.unbind()),
	};

	if let Ok(de) = method.extract::<PyRef<DE>>() {
		let args = AlgorithmDifferentialEvolutionArgs {
			base,
			population_size: de.population_size,
			generations: de.generations,
			dimensions,
			expectation_function: expectation_function.unbind(),
			tolerance: de.tolerance,
			mode: TrainMode::Qml,
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
			mode: TrainMode::Qml,
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
			variance_function: qng.variance_function.clone_ref(py),
			tikhonov_reg: qng.tikhonov_reg,
			mode: TrainMode::Qml,
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

	// qml submodule — exposes polypus.qml.train()
	let py = m.py();
	let qml = PyModule::new(py, "qml")?;
	qml.add_function(wrap_pyfunction!(qml_train, &qml)?)?;
	m.add_submodule(&qml)?;
	// Register in sys.modules so `import polypus.qml` also works
	let sys = PyModule::import(py, "sys")?;
	sys.getattr("modules")?.set_item("polypus.qml", &qml)?;

	Ok(())
}
