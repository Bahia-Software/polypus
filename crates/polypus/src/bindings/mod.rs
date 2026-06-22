use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, IntoPyDict};
use pyo3::Bound;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

pub mod circuit;
pub mod de;
pub mod pso;
pub mod qng;

use circuit::{statevector, Circuit, Param};
use de::DE;
use pso::PSO;
use qng::QNG;

use std::sync::Arc;
use crate::algorithms::{AlgorithmArgs, AlgorithmTrait, AlgorithmSingleRun, DistributeByShotsRun};
use crate::algorithms::vqc::{AlgorithmDifferentialEvolutionArgs, AlgorithmDifferentialEvolution};
use crate::algorithms::vqc::{AlgorithmPSOArgs, AlgorithmPSO};
use crate::algorithms::vqc::{AlgorithmQNGArgs, AlgorithmQNG};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, BackendConfig, Infrastructure, OptLevel};
#[cfg(feature = "qmio")]
use crate::infrastructure::execution_config::QmioProgramFormat;
use crate::evaluation::{CircuitSource, EvaluationOracle, VqcOracle, QmlOracle};

/// Map the public `infrastructure` + `backend` strings and provider parameters
/// into a typed [`BackendConfig`]. Centralising this keeps the string→variant
/// mapping in one place and guarantees the config matches the selected backend.
///
/// `backend` selects the *device* within an infrastructure. For `local`:
/// `"aer"` (default) runs Qiskit Aer; `"polypus"` runs the pure-Rust native
/// statevector simulator. The choice is ignored for CUNQA, which manages its
/// own simulated QPUs.
fn build_backend_config(
	infrastructure: &str,
	backend: &str,
	sim_method: &str,
	noise_model: Option<Py<PyAny>>,
	nodes: u32,
	cores_per_qpu: u32,
) -> PyResult<BackendConfig> {
	match Infrastructure::from_str(infrastructure) {
		Infrastructure::Local => match backend {
			"aer" | "AerSimulator" => Ok(BackendConfig::Local {
				backend: "AerSimulator".to_string(),
				sim_method: sim_method.to_string(),
				noise_model,
			}),
			"polypus" | "statevector" | "polypus_statevector" => {
				if noise_model.is_some() {
					return Err(pyo3::exceptions::PyValueError::new_err(
						"the native 'polypus' backend is a noiseless statevector simulator \
						 and does not accept a noise_model; use backend=\"aer\"",
					));
				}
				Ok(BackendConfig::LocalNative)
			}
			other => Err(pyo3::exceptions::PyValueError::new_err(format!(
				"unknown local backend '{other}'; expected \"aer\" or \"polypus\""
			))),
		},
		Infrastructure::Cunqa => Ok(BackendConfig::Cunqa {
			backend: "AerSimulator".to_string(),
			sim_method: sim_method.to_string(),
			nodes,
			cores_per_qpu,
		}),
		Infrastructure::Qmio => build_qmio_backend_config(backend),
	}
}

/// Build the [`BackendConfig::Qmio`] for the CESGA QMIO QPU.
///
/// The endpoint is read from the `ZMQ_SERVER` environment variable (the same
/// variable the reference `qmio` Python client uses), falling back to the
/// documented CESGA address. The public `backend` argument selects the program
/// representation submitted to the QPU. Only available with `--features qmio`;
/// otherwise it returns an actionable error instead of silently degrading.
#[cfg(feature = "qmio")]
fn build_qmio_backend_config(backend: &str) -> PyResult<BackendConfig> {
	// Default endpoint documented by CESGA; overridden by ZMQ_SERVER when set.
	const DEFAULT_QMIO_ENDPOINT: &str = "tcp://10.133.29.226:5556";
	let endpoint =
		std::env::var("ZMQ_SERVER").unwrap_or_else(|_| DEFAULT_QMIO_ENDPOINT.to_string());
	let program_format = match backend {
		// `"aer"` is the entry-point default, so treat it (and the explicit
		// aliases) as OpenQASM for the QMIO path.
		"aer" | "qmio" | "openqasm" | "qasm" => QmioProgramFormat::OpenQasm,
		"qir" | "qir_text" => QmioProgramFormat::QirText,
		"qir_bitcode" | "qir_compiled" => QmioProgramFormat::QirBitcode,
		other => {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"unknown qmio program format '{other}'; expected \"openqasm\", \"qir\", or \"qir_bitcode\""
			)))
		}
	};
	Ok(BackendConfig::Qmio {
		endpoint,
		program_format,
		// Sensible defaults; the optimisation level / results format are not yet
		// exposed as Python kwargs (kept extensible in BackendConfig::Qmio).
		optimization: 1,
		repetition_period: None,
		res_format: "binary_count".to_string(),
	})
}

/// Without the `qmio` feature, selecting the QMIO infrastructure fails with a
/// clear, actionable message instead of pulling a ZeroMQ stack into every build.
#[cfg(not(feature = "qmio"))]
fn build_qmio_backend_config(_backend: &str) -> PyResult<BackendConfig> {
	Err(pyo3::exceptions::PyValueError::new_err(
		"the 'qmio' infrastructure requires compiling polypus with --features qmio",
	))
}

/// Whether `backend` selects the pure-Rust native statevector simulator, which
/// (unlike Aer) cannot consume a Qiskit `QuantumCircuit`.
fn is_native_backend(backend: &str) -> bool {
	matches!(backend, "polypus" | "statevector" | "polypus_statevector")
}

/// Interpret the `qc` argument of an entry point as a parameterised circuit
/// template. A `polypus.Circuit` becomes [`CircuitSource::Native`] (binding
/// will run GIL-free); any other object is assumed to be a Qiskit
/// `QuantumCircuit`, preserving the original behaviour.
fn extract_circuit_source(qc: &Bound<'_, PyAny>) -> CircuitSource {
	if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
		CircuitSource::Native(native.native().clone())
	} else {
		CircuitSource::Qiskit(qc.clone().unbind())
	}
}

/// Interpret the `qc` argument of `run_quantum_circuit` as a fully bound,
/// executable circuit:
/// - `polypus.Circuit` → native circuit (must have no free parameters),
/// - `str` → raw OpenQASM 2.0 program,
/// - anything else → Qiskit `QuantumCircuit` (original behaviour).
fn extract_bound_circuit(qc: &Bound<'_, PyAny>) -> PyResult<BoundCircuit> {
	if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
		let concrete = native.native().assign_parameters(&[]).map_err(|e| {
			pyo3::exceptions::PyValueError::new_err(format!(
				"circuit has unbound parameters and cannot be executed directly: {e}. \
				 Bind values first via to_qasm2(params) or use polypus.train"
			))
		})?;
		return Ok(BoundCircuit::Native(concrete));
	}
	if let Ok(qasm) = qc.extract::<String>() {
		return Ok(BoundCircuit::Qasm2(qasm));
	}
	Ok(BoundCircuit::Qiskit(qc.clone().unbind()))
}

/// Function to run a quantum circuit called from Python.
///
/// `qc` may be a Qiskit `QuantumCircuit`, a `polypus.Circuit` (fully bound),
/// or an OpenQASM 2.0 string. `backend` selects the local device: `"aer"`
/// (default) or the pure-Rust `"polypus"` statevector simulator.
#[pyfunction(signature=(qc, shots, infrastructure, n_qpus=1, sim_method="automatic", noise_model=None, backend="aer"))]
pub fn run_quantum_circuit<'py>(
	qc: Bound<'py, PyAny>,
	shots: u32,
	infrastructure: String,
	n_qpus: u32,
	sim_method: &str,
	noise_model: Option<Bound<'py, PyAny>>,
	backend: &str,
) -> PyResult<pyo3::PyObject> {
	println!("run_quantum_circuit called with qc: {:?}, shots: {}, infrastructure: {}, n_qpus: {}, backend: {}", qc, shots, infrastructure, n_qpus, backend);
	let bound_qc = extract_bound_circuit(&qc)?;
	if is_native_backend(backend) {
		if let BoundCircuit::Qiskit(_) = &bound_qc {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
				 pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"",
			));
		}
	}
	// The QMIO path serialises circuits to QASM/QIR in Rust (GIL-free) and cannot
	// read a Qiskit QuantumCircuit, whose gates are only accessible via Python.
	if infrastructure == "qmio" {
		if let BoundCircuit::Qiskit(_) = &bound_qc {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
				 serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM \
				 2.0 string",
			));
		}
	}
	let id = format!("run_{}_{}", n_qpus, infrastructure);
	let backend_config = build_backend_config(
		&infrastructure,
		backend,
		sim_method,
		noise_model.map(|nm| nm.unbind()),
		1,
		2,
	)?;
	let config = ExecutionConfig {
		id,
		shots,
		n_qpus,
		infrastructure,
		backend_config,
		opt_level: OptLevel::default(),
	};
	let args = AlgorithmArgs {
		qcs: vec![bound_qc],
		config,
	};

	let algorithm: Box<dyn AlgorithmTrait<Args=AlgorithmArgs, AlgorithmReturnType=PyObject>> = if n_qpus == 1 {
		Box::new(AlgorithmSingleRun)
	} else {
		Box::new(DistributeByShotsRun)
	};

	Ok(algorithm.run(args))
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
#[pyfunction(signature = (qc, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer"))]
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
	backend: &str,
) -> PyResult<PyObject> {
	// Native circuits know their parameter count — catch a mismatch with the
	// requested optimisation dimensions before any QPU work starts.
	let circuit_source = extract_circuit_source(&qc);
	if let Some(num_params) = circuit_source.num_params() {
		if num_params != dimensions as usize {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"dimensions ({dimensions}) does not match the circuit's free parameters ({num_params})"
			)));
		}
	}
	// The native statevector backend runs pure-Rust circuits only; a Qiskit
	// template can't be simulated without the interpreter.
	if is_native_backend(backend) {
		if let CircuitSource::Qiskit(_) = &circuit_source {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"the native 'polypus' backend requires a native polypus.Circuit; \
				 got a Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"",
			));
		}
	}
	// QMIO serialises GIL-free, so it likewise needs a native circuit, not a
	// Qiskit template that can only be read through the interpreter.
	if infrastructure == "qmio" {
		if let CircuitSource::Qiskit(_) = &circuit_source {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
				 serialization); got a Qiskit circuit. Build it with polypus.Circuit",
			));
		}
	}
	let backend_config = build_backend_config(
		&infrastructure,
		backend,
		sim_method,
		noise_model.map(|nm| nm.unbind()),
		nodes,
		cores_per_qpu,
	)?;
	let config = Arc::new(ExecutionConfig {
		id: id.clone(),
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend_config,
		opt_level: OptLevel::default(),
	});
	let backend = Infrastructure::create_backend(&config);
	let oracle: Box<dyn EvaluationOracle> = Box::new(VqcOracle {
		circuit: circuit_source,
		config: Arc::clone(&config),
		backend,
		expectation_fn: expectation_function.unbind(),
	});

	if let Ok(de) = method.extract::<PyRef<DE>>() {
		let args = AlgorithmDifferentialEvolutionArgs {
			oracle,
			population_size: de.population_size,
			generations: de.generations,
			dimensions,
			tolerance: de.tolerance,
		};
		return Ok(AlgorithmDifferentialEvolution.run(args));
	}

	if let Ok(pso) = method.extract::<PyRef<PSO>>() {
		let args = AlgorithmPSOArgs {
			oracle,
			population_size: pso.population_size,
			generations: pso.generations,
			dimensions,
			bounds: pso.bounds,
			inertia_weight: pso.inertia_weight,
			cognitive_weight: pso.cognitive_weight,
			social_weight: pso.social_weight,
			tolerance: pso.tolerance,
		};
		return Ok(AlgorithmPSO.run(args));
	}

	if let Ok(qng) = method.extract::<PyRef<QNG>>() {
		let args = AlgorithmQNGArgs {
			oracle,
			max_iters: qng.max_iters,
			learning_rate: qng.learning_rate,
			finite_difference_step: qng.finite_difference_step,
			bounds: qng.bounds,
			dimensions,
			variance_function: qng.variance_function.clone_ref(method.py()),
			tikhonov_reg: qng.tikhonov_reg,
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
#[pyfunction(name = "train", signature = (feature_map, ansatz, x_train, method, shots, n_qpus, dimensions, expectation_function, infrastructure, nodes, cores_per_qpu, id, sim_method="automatic", noise_model=None, backend="aer"))]
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
	backend: &str,
) -> PyResult<PyObject> {
	// QML composes Qiskit feature maps and ansätze, so it is inherently a
	// Qiskit path; the native statevector backend cannot consume a Qiskit
	// `QuantumCircuit`. Accept `backend` for API symmetry but reject native.
	if is_native_backend(backend) {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"the native 'polypus' backend is not supported for qml.train (feature maps \
			 and ansätze are Qiskit circuits); use backend=\"aer\"",
		));
	}
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

	// QML composes Qiskit feature maps and ansätze, so it is inherently a
	// Qiskit path (native backend already rejected above): `backend` can only be
	// an Aer variant here.
	let backend_config = build_backend_config(
		&infrastructure,
		backend,
		sim_method,
		noise_model.map(|nm| nm.unbind()),
		nodes,
		cores_per_qpu,
	)?;
	let config = Arc::new(ExecutionConfig {
		id: id.clone(),
		shots,
		n_qpus,
		infrastructure: infrastructure.clone(),
		backend_config,
		opt_level: OptLevel::default(),
	});
	let backend = Infrastructure::create_backend(&config);
	let oracle: Box<dyn EvaluationOracle> = Box::new(QmlOracle {
		training_circuits: qcs,
		config: Arc::clone(&config),
		backend,
		expectation_fn: expectation_function.unbind(),
	});

	if let Ok(de) = method.extract::<PyRef<DE>>() {
		let args = AlgorithmDifferentialEvolutionArgs {
			oracle,
			population_size: de.population_size,
			generations: de.generations,
			dimensions,
			tolerance: de.tolerance,
		};
		return Ok(AlgorithmDifferentialEvolution.run(args));
	}

	if let Ok(pso) = method.extract::<PyRef<PSO>>() {
		let args = AlgorithmPSOArgs {
			oracle,
			population_size: pso.population_size,
			generations: pso.generations,
			dimensions,
			bounds: pso.bounds,
			inertia_weight: pso.inertia_weight,
			cognitive_weight: pso.cognitive_weight,
			social_weight: pso.social_weight,
			tolerance: pso.tolerance,
		};
		return Ok(AlgorithmPSO.run(args));
	}

	if let Ok(qng) = method.extract::<PyRef<QNG>>() {
		let args = AlgorithmQNGArgs {
			oracle,
			max_iters: qng.max_iters,
			learning_rate: qng.learning_rate,
			finite_difference_step: qng.finite_difference_step,
			bounds: qng.bounds,
			dimensions,
			variance_function: qng.variance_function.clone_ref(py),
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
	m.add_class::<Circuit>()?;
	m.add_class::<Param>()?;
	m.add_function(wrap_pyfunction!(train, m)?)?;
	m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
	m.add_function(wrap_pyfunction!(statevector, m)?)?;

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
