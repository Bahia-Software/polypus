//! PyO3 bindings for `polypus-qml`: the `polypus.qml.Model` / `polypus.qml.Dataset`
//! wrappers and the type-dispatching `polypus.qml.train` entry point.
//!
//! `qml.train` inspects its first argument at run time (decision A of the phase-4
//! plan): a native [`Model`] takes the pure-Rust path (a [`NativeQmlOracle`] over
//! any simulated backend, `polypus` included); anything else is a Qiskit
//! `QuantumCircuit` feature map and takes the original Qiskit/Aer path unchanged.
//!
//! The shared plumbing (seed resolution, `ExecutionConfig`/backend construction,
//! the DE/PSO/QNG/Adam dispatch, `finish_optimization`) lives in the parent
//! [`bindings`](super) module and is reached through `super::`; only the
//! QML-specific wrappers and the native path live here.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyModule};
use std::sync::Arc;

use polypus_qml::{
    Decision, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout,
    RotationAxis, ValidationError,
};

use super::{
    build_backend_config, finish_optimization, is_native_backend, method_seed,
    resolve_optimizer_seed, unique_id, validate_cunqa_allocation, validate_shots_and_qpus,
};
use crate::bindings::adam::Adam;
use crate::bindings::de::DE;
use crate::bindings::pso::PSO;
use crate::bindings::qng::{PyVarianceOracle, QNG};
use crate::evaluation::{EvaluationOracle, NativeQmlOracle, OracleErrorSlot, QmlOracle};
use crate::infrastructure::{ExecutionConfig, Infrastructure, OptLevel};
use polypus_optimizers::{
    AlgorithmAdam, AlgorithmAdamArgs, AlgorithmDifferentialEvolution,
    AlgorithmDifferentialEvolutionArgs, AlgorithmPSO, AlgorithmPSOArgs, AlgorithmQNG,
    AlgorithmQNGArgs, GradientOracle, Optimizer,
};

/// Map a `polypus-qml` [`ValidationError`] (a construction/compilation failure —
/// `Model.readout`, `compile`, `QmlProblem::new`) onto a Python `ValueError`.
///
/// This is a free function, not a `From<ValidationError> for PyErr`: both types
/// are foreign to this crate, so the orphan rule forbids the impl. It mirrors
/// `to_py_err` in [`circuit`](super::circuit), which maps `CircuitError` the same
/// way for the same reason.
fn validation_to_py_err(e: ValidationError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse a rotation-axis string into a [`RotationAxis`]. Strict: an unrecognised
/// value is a `ValueError` listing the valid options (decision D).
fn parse_axis(axis: &str) -> PyResult<RotationAxis> {
    match axis {
        "rx" => Ok(RotationAxis::Rx),
        "ry" => Ok(RotationAxis::Ry),
        "rz" => Ok(RotationAxis::Rz),
        other => Err(PyValueError::new_err(format!(
            "unknown rotation axis '{other}'; expected \"rx\", \"ry\" or \"rz\""
        ))),
    }
}

/// Parse a single-qubit Pauli string into a [`Pauli`]. Accepts `"x"`/`"y"`/`"z"`;
/// `compile` rejects X/Y readout with its own `UnsupportedPauli`, so this does
/// not duplicate that check (decision D).
fn parse_pauli(pauli: &str) -> PyResult<Pauli> {
    match pauli {
        "x" => Ok(Pauli::X),
        "y" => Ok(Pauli::Y),
        "z" => Ok(Pauli::Z),
        other => Err(PyValueError::new_err(format!(
            "unknown Pauli '{other}'; expected \"x\", \"y\" or \"z\""
        ))),
    }
}

/// Parse a loss string into a [`Loss`]. Strict: an unrecognised value is a
/// `ValueError` listing the valid options (decision D).
fn parse_loss(loss: &str) -> PyResult<Loss> {
    match loss {
        "squared_error" => Ok(Loss::SquaredError),
        "binary_cross_entropy" => Ok(Loss::BinaryCrossEntropy),
        "hinge" => Ok(Loss::Hinge),
        "categorical_cross_entropy" => Ok(Loss::CategoricalCrossEntropy),
        other => Err(PyValueError::new_err(format!(
            "unknown loss '{other}'; expected \"squared_error\", \"binary_cross_entropy\", \"hinge\" or \"categorical_cross_entropy\""
        ))),
    }
}

/// Parse a decision string into a [`Decision`].
///
/// `threshold` is **required** for `decision="threshold"` and **rejected** for
/// every other decision (decision D — we choose the strict reading: passing a
/// `threshold` where it has no effect is an error, not a silently-ignored value).
fn parse_decision(decision: &str, threshold: Option<f64>) -> PyResult<Decision> {
    let stray = |name: &str| {
        PyValueError::new_err(format!(
            "threshold is only valid with decision=\"threshold\", not \"{name}\""
        ))
    };
    match decision {
        "sign" => threshold.map_or(Ok(Decision::Sign), |_| Err(stray("sign"))),
        "argmax" => threshold.map_or(Ok(Decision::Argmax), |_| Err(stray("argmax"))),
        "raw" => threshold.map_or(Ok(Decision::Raw), |_| Err(stray("raw"))),
        "threshold" => threshold.map(Decision::Threshold).ok_or_else(|| {
            PyValueError::new_err("decision=\"threshold\" requires a threshold value")
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown decision '{other}'; expected \"sign\", \"threshold\", \"argmax\" or \"raw\""
        ))),
    }
}

/// A quantum model builder, mirroring [`QuantumModel`] for Python.
///
/// [`QuantumModel`]'s builder methods consume `self` (`fn layer(self, ..) -> Self`),
/// which a `#[pyclass]` cannot expose directly. So the wrapper holds
/// `inner: Option<QuantumModel>` and each method takes the model out, applies the
/// consuming builder call, and puts the result back — the model is always `Some`
/// between calls (decision B). Methods return `self` for chaining, exactly like
/// `polypus.Circuit`.
#[pyclass(module = "polypus.qml", name = "Model")]
pub struct Model {
    inner: Option<QuantumModel>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(num_qubits: usize) -> Self {
        Model {
            inner: Some(QuantumModel::new(num_qubits)),
        }
    }

    /// Append an angle encoder over `axis` (`"rx"`/`"ry"`/`"rz"`).
    fn angle_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        axis: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let axis = parse_axis(axis)?;
        let model = slf
            .inner
            .take()
            .expect("Model.inner is always Some between calls");
        slf.inner = Some(model.angle_encoder(axis));
        Ok(slf)
    }

    /// Append a hardware-efficient ansatz with `reps` repetitions.
    fn hardware_efficient(
        mut slf: PyRefMut<'_, Self>,
        reps: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let model = slf
            .inner
            .take()
            .expect("Model.inner is always Some between calls");
        slf.inner = Some(model.hardware_efficient(reps));
        Ok(slf)
    }

    /// Attach the readout: the `observables` to measure plus the `decision` rule.
    ///
    /// Each observable is a list of `(pauli, position)` factors of a single Pauli
    /// string with implicit coefficient `1.0` (no weighted sums from Python in
    /// v1 — decision C). `decision` is `"sign"`/`"threshold"`/`"argmax"`/`"raw"`;
    /// `threshold` is required for `"threshold"` and rejected otherwise.
    #[pyo3(signature = (observables, decision, threshold=None))]
    fn readout<'py>(
        mut slf: PyRefMut<'py, Self>,
        observables: Vec<Vec<(String, usize)>>,
        decision: &str,
        threshold: Option<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let decision = parse_decision(decision, threshold)?;
        let mut parsed = Vec::with_capacity(observables.len());
        for factors in observables {
            let mut terms = Vec::with_capacity(factors.len());
            for (pauli, position) in factors {
                terms.push((position, parse_pauli(&pauli)?));
            }
            let string = PauliString::new(terms).map_err(validation_to_py_err)?;
            // Single-term observable, coefficient 1.0 (decision C).
            parsed.push(Observable::new(vec![(1.0, string)]).map_err(validation_to_py_err)?);
        }
        let readout = Readout::new(parsed, decision).map_err(validation_to_py_err)?;
        let model = slf
            .inner
            .take()
            .expect("Model.inner is always Some between calls");
        slf.inner = Some(model.readout(readout));
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(_) => "Model(...)".to_string(),
            None => "Model(<in flight>)".to_string(),
        }
    }
}

/// A validated supervised dataset, mirroring `polypus-qml`'s [`Dataset`].
///
/// Built from a 2-D feature matrix `x` and a label vector `y`, both extracted via
/// PyO3's sequence protocol — so plain Python lists and NumPy arrays both work
/// with no `numpy` dependency of our own.
///
/// [`Dataset`]: polypus_qml::Dataset
#[pyclass(module = "polypus.qml", name = "Dataset")]
pub struct Dataset {
    pub(crate) inner: polypus_qml::Dataset,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<Self> {
        let rows: Vec<Vec<f64>> = x.extract()?;
        let labels: Vec<f64> = y.extract()?;
        let inner =
            polypus_qml::Dataset::from_rows(&rows, &labels).map_err(validation_to_py_err)?;
        Ok(Dataset { inner })
    }

    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(num_samples={}, num_features={})",
            self.inner.num_samples(),
            self.inner.num_features()
        )
    }
}

/// QML entry point: train a variational quantum model with a chosen optimizer.
///
/// The first argument is inspected at run time (decision A):
///
/// - a native `polypus.qml.Model` → the **native path**: the second argument
///   must be a `polypus.qml.Dataset`, `loss` is required, and training runs
///   through a [`NativeQmlOracle`] on any simulated backend (`polypus`, `aer`,
///   `cunqa`). `x_train`/`expectation_function` are rejected (the samples and
///   fitness come from the `Dataset` + `loss` + the model's readout), and
///   `dimensions`, if given, must equal the compiled model's parameter count.
/// - anything else (a Qiskit `QuantumCircuit` feature map) → the **Qiskit path**,
///   unchanged: `x_train`, `dimensions` and `expectation_function` are required.
///
/// `seed` follows the same precedence as [`train`](super::train) and returns a
/// [`TrainResult`](super::TrainResult). On the native path the same seed also
/// drives shot sampling on every simulated backend, so the run reproduces
/// byte-for-byte. Ctrl+C interrupts promptly, and a user callback's exception
/// propagates verbatim (Qiskit path).
///
/// Example (native path):
///
/// ```python
/// ds = polypus.qml.Dataset(X, y)
/// model = (polypus.qml.Model(num_qubits=4)
///          .angle_encoder(axis="ry")
///          .hardware_efficient(reps=2)
///          .readout(observables=[[("z", 0)]], decision="sign"))
/// result = polypus.qml.train(model, ds, method=polypus.DE(...), loss="hinge",
///                            shots=1024, infrastructure="local",
///                            backend="polypus", id="qml", seed=7)
/// ```
///
/// `nodes`/`cores_per_qpu` size the SLURM allocation and matter only for
/// `infrastructure="cunqa"`; they default to `1`. `x_train`, `dimensions`,
/// `expectation_function` and `loss` default to `None`; several kwargs that were
/// positional (`method`, `shots`, `n_qpus`, `infrastructure`, `id`) gain defaults
/// too, which the shared `#[pyfunction]` signature requires once an earlier
/// positional argument (`x_train`) becomes optional — a backward-compatible
/// relaxation, since every existing caller passes them explicitly.
#[pyfunction(name = "train", signature = (
    feature_map,
    ansatz,
    x_train=None,
    method=None,
    shots=1024,
    n_qpus=1,
    dimensions=None,
    expectation_function=None,
    infrastructure="local".to_string(),
    nodes=1,
    cores_per_qpu=1,
    id="qml".to_string(),
    sim_method="automatic",
    noise_model=None,
    backend="aer",
    seed=None,
    loss=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn qml_train<'py>(
    feature_map: Bound<'py, PyAny>,
    ansatz: Bound<'py, PyAny>,
    x_train: Option<Bound<'py, PyAny>>,
    method: Option<Bound<'py, PyAny>>,
    shots: u32,
    n_qpus: u32,
    dimensions: Option<u32>,
    expectation_function: Option<Bound<'py, PyAny>>,
    infrastructure: String,
    nodes: u32,
    cores_per_qpu: u32,
    id: String,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    loss: Option<&str>,
) -> PyResult<PyObject> {
    validate_shots_and_qpus(shots, n_qpus)?;
    validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu)?;
    // `method` is required on both paths; it only has a default because the
    // signature needs one once `x_train` (an earlier positional) is optional.
    let method = method.ok_or_else(|| {
        PyTypeError::new_err(
            "method is required: pass an instance of polypus.DE, polypus.PSO, polypus.QNG, or polypus.Adam",
        )
    })?;

    // Dispatch by the type of the first argument (decision A).
    if let Ok(model) = feature_map.extract::<PyRef<'_, Model>>() {
        qml_train_native(
            model,
            &ansatz,
            x_train,
            &method,
            shots,
            n_qpus,
            dimensions,
            expectation_function,
            &infrastructure,
            nodes,
            cores_per_qpu,
            id,
            sim_method,
            noise_model,
            backend,
            seed,
            loss,
        )
    } else {
        qml_train_qiskit(
            &feature_map,
            &ansatz,
            x_train,
            &method,
            shots,
            n_qpus,
            dimensions,
            expectation_function,
            &infrastructure,
            nodes,
            cores_per_qpu,
            id,
            sim_method,
            noise_model,
            backend,
            seed,
        )
    }
}

/// The native (pure-Rust) path: a compiled `polypus-qml` model + dataset + loss,
/// trained through a [`NativeQmlOracle`] on any simulated backend.
#[allow(clippy::too_many_arguments)]
fn qml_train_native(
    model: PyRef<'_, Model>,
    ansatz: &Bound<'_, PyAny>,
    x_train: Option<Bound<'_, PyAny>>,
    method: &Bound<'_, PyAny>,
    shots: u32,
    n_qpus: u32,
    dimensions: Option<u32>,
    expectation_function: Option<Bound<'_, PyAny>>,
    infrastructure: &str,
    nodes: u32,
    cores_per_qpu: u32,
    id: String,
    sim_method: &str,
    noise_model: Option<Bound<'_, PyAny>>,
    backend: &str,
    seed: Option<u64>,
    loss: Option<&str>,
) -> PyResult<PyObject> {
    let py = model.py();
    // Reject kwargs that belong to the Qiskit path: on the native path the
    // samples (and their labels) travel inside the Dataset, and the fitness is
    // defined by `loss` + the model's readout, not a user callback.
    if x_train.is_some() {
        return Err(PyValueError::new_err(
            "x_train is not accepted with a native Model + Dataset; the training samples \
             (and their labels) travel inside the Dataset",
        ));
    }
    if expectation_function.is_some() {
        return Err(PyValueError::new_err(
            "expectation_function is not accepted with a native Model + Dataset; the fitness \
             is defined by loss=... together with the model's readout",
        ));
    }
    // The second argument must be a native Dataset on this path.
    let dataset = ansatz.extract::<PyRef<'_, Dataset>>().map_err(|_| {
        PyTypeError::new_err(
            "with a native polypus.qml.Model the second argument must be a polypus.qml.Dataset",
        )
    })?;
    // `loss` is required on the native path.
    let loss = parse_loss(loss.ok_or_else(|| {
        PyValueError::new_err(
            "loss is required with a native Model + Dataset (e.g. loss=\"hinge\")",
        )
    })?)?;

    // Compile the model against the dataset's feature count. We only hold a
    // borrow of the model via `PyRef`, so clone the inner builder before the
    // consuming `compile` call.
    let compiled = model
        .inner
        .as_ref()
        .expect("Model.inner is always Some between calls")
        .clone()
        .compile(dataset.inner.num_features())
        .map_err(validation_to_py_err)?;
    let num_params = compiled.num_params();

    // `dimensions`, if supplied, must match the compiled model's parameter count
    // (same spirit as the Qiskit path's dimension check).
    if let Some(dimensions) = dimensions {
        if dimensions as usize != num_params {
            return Err(PyValueError::new_err(format!(
                "dimensions ({dimensions}) does not match the model's trainable parameters \
                 ({num_params})"
            )));
        }
    }
    let dimensions = num_params as u32;

    let problem =
        QmlProblem::new(compiled, dataset.inner.clone(), loss).map_err(validation_to_py_err)?;

    // From here on, the seed / config / backend / dispatch is exactly the shared
    // logic `train` and the Qiskit path use — only the oracle and `dimensions`
    // differ (decision E/G).
    let effective_seed = resolve_optimizer_seed(seed, method_seed(method));
    let backend_config = build_backend_config(
        infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
    )?;
    let effective_id = unique_id(&id);
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.to_string(),
        backend_config,
        opt_level: OptLevel::default(),
        seed: Some(effective_seed),
    });
    let backend = Infrastructure::create_backend(&config)?;
    let errors = OracleErrorSlot::new();
    // One concrete oracle behind an `Arc`, handed to `dispatch_optimizer` as two
    // independent trait-object boxes via the `Arc<T>` blanket impls: the same
    // NativeQmlOracle scores fitness (EvaluationOracle) and its exact
    // parameter-shift gradient (GradientOracle) for the gradient optimizers
    // (QNG, Adam).
    let oracle = Arc::new(NativeQmlOracle {
        problem,
        config: Arc::clone(&config),
        backend,
        errors: errors.clone(),
    });
    let eval_oracle: Box<dyn EvaluationOracle> = Box::new(Arc::clone(&oracle));
    let gradient_oracle: Box<dyn GradientOracle> = Box::new(oracle);

    dispatch_optimizer(
        py,
        method,
        (eval_oracle, gradient_oracle),
        dimensions,
        &errors,
        effective_seed,
        effective_id,
    )
}

/// The Qiskit/Aer path: compose the feature map with the ansatz, pre-bind each
/// training sample's feature-map parameters, and train through a [`QmlOracle`].
/// Behaviour is unchanged from the pre-phase-4 inline `qml_train`.
#[allow(clippy::too_many_arguments)]
fn qml_train_qiskit(
    feature_map: &Bound<'_, PyAny>,
    ansatz: &Bound<'_, PyAny>,
    x_train: Option<Bound<'_, PyAny>>,
    method: &Bound<'_, PyAny>,
    shots: u32,
    n_qpus: u32,
    dimensions: Option<u32>,
    expectation_function: Option<Bound<'_, PyAny>>,
    infrastructure: &str,
    nodes: u32,
    cores_per_qpu: u32,
    id: String,
    sim_method: &str,
    noise_model: Option<Bound<'_, PyAny>>,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    // On the Qiskit path these three are required (they default to None only so
    // the shared signature can make `x_train` optional for the native path).
    let x_train = x_train.ok_or_else(|| {
        PyValueError::new_err("x_train is required with a Qiskit feature_map / ansatz")
    })?;
    let dimensions = dimensions.ok_or_else(|| {
        PyValueError::new_err("dimensions is required with a Qiskit feature_map / ansatz")
    })?;
    let expectation_function = expectation_function.ok_or_else(|| {
        PyValueError::new_err("expectation_function is required with a Qiskit feature_map / ansatz")
    })?;

    let effective_seed = resolve_optimizer_seed(seed, method_seed(method));
    // QML composes Qiskit feature maps and ansätze, so it is inherently a
    // Qiskit path; the native statevector backend cannot consume a Qiskit
    // `QuantumCircuit`. Accept `backend` for API symmetry but reject native.
    if is_native_backend(backend) {
        return Err(PyValueError::new_err(
            "the native 'polypus' backend cannot consume a Qiskit feature_map / ansatz; \
             pass a polypus.qml.Model for the native path, or use backend=\"aer\"",
        ));
    }
    let py = feature_map.py();

    // 1. Compose feature_map + ansatz
    let composed = feature_map.call_method1("compose", (ansatz,))?;

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
        return Err(PyValueError::new_err(
            "x_train must contain at least one training sample",
        ));
    }

    let backend_config = build_backend_config(
        infrastructure,
        backend,
        sim_method,
        noise_model.map(|nm| nm.unbind()),
        nodes,
        cores_per_qpu,
    )?;
    let effective_id = unique_id(&id);
    let config = Arc::new(ExecutionConfig {
        id: effective_id.clone(),
        shots,
        n_qpus,
        infrastructure: infrastructure.to_string(),
        backend_config,
        opt_level: OptLevel::default(),
        seed: Some(effective_seed),
    });
    let backend = Infrastructure::create_backend(&config)?;
    let errors = OracleErrorSlot::new();
    // Same Arc + two-boxes pattern as the native path: the QmlOracle scores
    // fitness and, for the gradient optimizers (QNG, Adam), its parameter-shift
    // gradient (exact by linearity of the mean expectation — this path has no
    // nonlinear loss).
    let oracle = Arc::new(QmlOracle {
        training_circuits: qcs,
        config: Arc::clone(&config),
        backend,
        expectation_fn: expectation_function.unbind(),
        errors: errors.clone(),
    });
    let eval_oracle: Box<dyn EvaluationOracle> = Box::new(Arc::clone(&oracle));
    let gradient_oracle: Box<dyn GradientOracle> = Box::new(oracle);

    dispatch_optimizer(
        py,
        method,
        (eval_oracle, gradient_oracle),
        dimensions,
        &errors,
        effective_seed,
        effective_id,
    )
}

/// Run `oracle` under the optimizer named by `method` (DE/PSO/QNG/Adam),
/// releasing the GIL for the whole `optimize()` call (see `train` and
/// ENGINEERING §3) and surfacing any recorded oracle error afterwards.
///
/// Shared verbatim by both `qml.train` paths — the only difference between them
/// is which oracle and `dimensions` are passed in.
///
/// `gradient_oracle` is the same underlying oracle as `oracle` (built from one
/// `Arc` via the blanket impls); it is consumed by the gradient optimizers
/// (QNG, Adam) and ignored by DE/PSO, which are gradient-free.
fn dispatch_optimizer(
    py: Python<'_>,
    method: &Bound<'_, PyAny>,
    oracles: (Box<dyn EvaluationOracle>, Box<dyn GradientOracle>),
    dimensions: u32,
    errors: &OracleErrorSlot,
    effective_seed: u64,
    effective_id: String,
) -> PyResult<PyObject> {
    // The two boxes are the same underlying oracle (one Arc, two blanket-impl
    // facets): the evaluation box for DE/PSO/QNG/Adam fitness, the gradient box
    // for QNG and Adam. Passed as a pair to keep the argument count in check.
    let (oracle, gradient_oracle) = oracles;
    if let Ok(de) = method.extract::<PyRef<DE>>() {
        let args = AlgorithmDifferentialEvolutionArgs {
            oracle,
            population_size: de.population_size,
            generations: de.generations,
            dimensions,
            tolerance: de.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            py,
            py.allow_threads(|| AlgorithmDifferentialEvolution.optimize(args)),
            errors,
            effective_seed,
            effective_id,
        );
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
            seed: Some(effective_seed),
        };
        return finish_optimization(
            py,
            py.allow_threads(|| AlgorithmPSO.optimize(args)),
            errors,
            effective_seed,
            effective_id,
        );
    }

    if let Ok(qng) = method.extract::<PyRef<QNG>>() {
        let args = AlgorithmQNGArgs {
            oracle,
            gradient_oracle,
            max_iters: qng.max_iters,
            learning_rate: qng.learning_rate,
            bounds: qng.bounds,
            dimensions,
            tolerance: qng.tolerance,
            variance_oracle: Box::new(PyVarianceOracle {
                variance_function: qng.variance_function.clone_ref(py),
                errors: errors.clone(),
            }),
            tikhonov_reg: qng.tikhonov_reg,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            py,
            py.allow_threads(|| AlgorithmQNG.optimize(args)),
            errors,
            effective_seed,
            effective_id,
        );
    }

    if let Ok(adam) = method.extract::<PyRef<Adam>>() {
        // Same two facets of the one Arc as QNG: the evaluation box scores
        // fitness, the gradient box supplies the exact parameter-shift gradient.
        // No VarianceOracle — Adam's adaptive step comes from the gradient moments.
        let args = AlgorithmAdamArgs {
            oracle,
            gradient_oracle,
            max_iters: adam.max_iters,
            learning_rate: adam.learning_rate,
            beta1: adam.beta1,
            beta2: adam.beta2,
            epsilon: adam.epsilon,
            bounds: adam.bounds,
            dimensions,
            tolerance: adam.tolerance,
            seed: Some(effective_seed),
        };
        return finish_optimization(
            py,
            py.allow_threads(|| AlgorithmAdam.optimize(args)),
            errors,
            effective_seed,
            effective_id,
        );
    }

    Err(PyTypeError::new_err(
        "method must be an instance of polypus.DE, polypus.PSO, polypus.QNG, or polypus.Adam",
    ))
}
