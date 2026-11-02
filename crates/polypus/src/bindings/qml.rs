//! The native API of `polypus.qml` (plan D4-R): a [`Model`] built from an
//! `Encoder` that loads each sample, an `Ansatz` with the trainable layers and a
//! `Readout`, trained with [`Model::fit`]; the `Dataset` it trains on; and the
//! Pauli readout helpers `Z`/`X`/`Y`, `PauliTerm` and `Observable`.
//! `qml.train`/`qml.predict` stay the Qiskit form and reject these pieces.
//!
//! Each class wraps a `polypus-qml` type and validates as it is built: every
//! option is a string checked against its closed set (plan F6), and every
//! builder call on an `Encoder`/`Ansatz` re-checks the layers that do not need
//! the data (plan F1), so a mistake surfaces at the line that made it rather
//! than inside `fit`. What does depend on the data (the feature count) is
//! checked when the model meets a dataset.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use numpy::ndarray::{Array2, Array3, ShapeError};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{
    PyFileNotFoundError, PyOSError, PyPermissionError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};
use polypus_qml::{
    AngleEncoder, CompiledModel, ConvBlock, ConvLayer, Decision, Entanglement, Entangler,
    ExpectationJacobian, HardwareEfficientAnsatz, IqpEncoder, JacobianWrt, KeepRule, Layer, Loss,
    Observable, Pairing, Pauli, PauliString, PoolBlock, PoolLayer, QmlError, QmlProblem,
    QuantumModel, Readout, RotationAxis, ValidationError,
};

use super::callback::validate_callback;
use super::circuit::Circuit;
use super::{
    build_backend_config, counts_to_pylist, extract_labels, finish_optimization,
    interruptible_token, is_native_backend, is_rotosolve, log_training_start, method_from_pyclass,
    method_seed, resolve_optimizer_seed, seed_or_entropy, training_observer, unique_id,
    validate_cunqa_allocation, validate_final_shots, validate_id, validate_method,
    validate_shots_and_qpus, OptionsArg, RunResult,
};
use crate::evaluation::{
    candidate_window_size, Label, MinibatchConfig, OracleErrorSlot, QmlOracleFactory, QmlSource,
};
use crate::exceptions::configuration_error;
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{
    BoundCircuit, CancelToken, Counts, ExecutionConfig, Infrastructure, InfrastructureError,
    NativeStatevectorBackend, OptLevel, QuantumBackend, RunParams,
};
use crate::orchestration::{Resources, RunCircuitFlow, Scheduler, TrainFlow};

/// A `polypus-qml` [`ValidationError`] as a Python `ValueError`. A free function
/// because both types are foreign here (the orphan rule forbids a `From`), as
/// `to_py_err` in [`circuit`](super::circuit) does for `CircuitError`.
fn validation_to_py_err(e: ValidationError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A [`ValidationError`] found while `Model.fit`, `TrainedModel.predict` or
/// `run` checks its arguments (the rows, the labels, the model against them),
/// as a `polypus.ConfigurationError`, which every argument check of an entry
/// point raises (#418). The builders keep [`validation_to_py_err`], a plain
/// `ValueError`, as `polypus.Circuit`'s do.
fn validation_to_config_err(e: ValidationError) -> PyErr {
    configuration_error(e.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Option strings (plan F6): each parser accepts exactly its closed set and
// names it in the error. The stubs spell the same sets as `Literal`s.
// ─────────────────────────────────────────────────────────────────────────────

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

fn parse_entangler(entangler: &str) -> PyResult<Entangler> {
    match entangler {
        "cx" => Ok(Entangler::Cx),
        "cz" => Ok(Entangler::Cz),
        other => Err(PyValueError::new_err(format!(
            "unknown entangler '{other}'; expected \"cx\" or \"cz\""
        ))),
    }
}

fn parse_entanglement(entanglement: &str) -> PyResult<Entanglement> {
    match entanglement {
        "linear" => Ok(Entanglement::Linear),
        "circular" => Ok(Entanglement::Circular),
        "full" => Ok(Entanglement::Full),
        other => Err(PyValueError::new_err(format!(
            "unknown entanglement '{other}'; expected \"linear\", \"circular\" or \"full\""
        ))),
    }
}

fn parse_conv_block(block: &str) -> PyResult<ConvBlock> {
    match block {
        "basic" => Ok(ConvBlock::Basic),
        "cartan" => Ok(ConvBlock::Cartan),
        other => Err(PyValueError::new_err(format!(
            "unknown conv block '{other}'; expected \"basic\" or \"cartan\""
        ))),
    }
}

fn parse_pairing(pairing: &str) -> PyResult<Pairing> {
    match pairing {
        "even_pairs" => Ok(Pairing::EvenPairs),
        "odd_pairs" => Ok(Pairing::OddPairs),
        "alternating" => Ok(Pairing::Alternating),
        other => Err(PyValueError::new_err(format!(
            "unknown pairing '{other}'; expected \"even_pairs\", \"odd_pairs\" or \"alternating\""
        ))),
    }
}

fn parse_pool_block(block: &str) -> PyResult<PoolBlock> {
    match block {
        "basic" => Ok(PoolBlock::Basic),
        other => Err(PyValueError::new_err(format!(
            "unknown pool block '{other}'; expected \"basic\""
        ))),
    }
}

fn parse_keep_rule(keep: &str) -> PyResult<KeepRule> {
    match keep {
        "even_positions" => Ok(KeepRule::EvenPositions),
        "odd_positions" => Ok(KeepRule::OddPositions),
        other => Err(PyValueError::new_err(format!(
            "unknown keep rule '{other}'; expected \"even_positions\" or \"odd_positions\""
        ))),
    }
}

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

fn parse_loss(loss: &str) -> PyResult<Loss> {
    match loss {
        "squared_error" => Ok(Loss::SquaredError),
        "binary_cross_entropy" => Ok(Loss::BinaryCrossEntropy),
        "hinge" => Ok(Loss::Hinge),
        "categorical_cross_entropy" => Ok(Loss::CategoricalCrossEntropy),
        "expectation" => Ok(Loss::Expectation),
        other => Err(configuration_error(format!(
            "unknown loss '{other}'; expected \"squared_error\", \"binary_cross_entropy\", \
             \"hinge\", \"categorical_cross_entropy\" or \"expectation\""
        ))),
    }
}

/// Check, before anything runs, that `polypus.Rotosolve` can train this native
/// model, as `train` and `qml.train` check a circuit (#398). Its closed-form
/// step assumes a fitness that is a sinusoid of each parameter, which takes two
/// things: `loss="expectation"`, the one loss that is linear in each sample's
/// measured distribution, and every trainable parameter the angle of exactly
/// one single-Pauli rotation (`ParameterizedCircuit::check_single_frequency`,
/// which a convolution or pooling layer fails: it shares each parameter across
/// qubit pairs). One sample's template stands for all, since the trainable
/// gates never depend on the data; a dataset is never empty.
fn check_rotosolve_model(
    compiled: &CompiledModel,
    dataset: &polypus_qml::Dataset,
    loss: Loss,
) -> PyResult<()> {
    if loss != Loss::Expectation {
        return Err(configuration_error(format!(
            "polypus.Rotosolve trains a native model only with loss=\"expectation\": with \
             loss=\"{}\" the fitness is not linear in the measured distribution, so it is not \
             a sinusoid of each parameter; train this model with another optimizer",
            loss_name(loss)
        )));
    }
    let template = compiled
        .template_for(dataset.sample(0))
        .map_err(qml_to_config_err)?;
    template.check_single_frequency().map_err(|problem| {
        configuration_error(format!(
            "polypus.Rotosolve cannot train this model: {problem}. Rotosolve needs every \
             trainable parameter to be the angle of exactly one rotation, and a convolution or \
             pooling layer shares each of its parameters across qubit pairs; train this model \
             with another optimizer"
        ))
    })
}

/// The Python spelling of a [`Loss`], the inverse of [`parse_loss`].
fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::SquaredError => "squared_error",
        Loss::BinaryCrossEntropy => "binary_cross_entropy",
        Loss::Hinge => "hinge",
        Loss::CategoricalCrossEntropy => "categorical_cross_entropy",
        Loss::Expectation => "expectation",
    }
}

/// Parse a decision. `threshold` is required by `"threshold"` and rejected by
/// every other decision: a value that would have no effect is an error, not
/// silently ignored.
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
            "unknown decision '{other}'; expected \"raw\", \"sign\", \"threshold\" or \"argmax\""
        ))),
    }
}

/// The Python spelling of a [`Decision`], without its threshold.
fn decision_name(decision: Decision) -> &'static str {
    match decision {
        Decision::Sign => "sign",
        Decision::Threshold(_) => "threshold",
        Decision::Argmax => "argmax",
        Decision::Raw => "raw",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Readout observables
// ─────────────────────────────────────────────────────────────────────────────

/// Decode `[(pauli, position), …]` into a [`PauliString`]: the one place the
/// Python spelling of a Pauli string is parsed, shared by every observable form.
fn parse_pauli_string(factors: &[(String, usize)]) -> PyResult<PauliString> {
    let mut terms = Vec::with_capacity(factors.len());
    for (pauli, position) in factors {
        terms.push((*position, parse_pauli(pauli)?));
    }
    PauliString::new(terms).map_err(validation_to_py_err)
}

/// The observable a coefficient-free product denotes: one Pauli string with
/// coefficient `1.0`.
fn unit_observable(factors: &[(String, usize)]) -> PyResult<Observable> {
    let string = parse_pauli_string(factors)?;
    Observable::new(vec![(1.0, string)]).map_err(validation_to_py_err)
}

/// A product of Pauli factors on distinct qubits, written `Z(0)`, `Z(0) @ Z(1)`
/// or `X(1) @ Y(2)`: one term with coefficient `1.0`. A weighted sum is an
/// [`Observable`](PyObservable); there is deliberately no `+` or `*` here.
///
/// The positions are logical: they index the qubits still active when the
/// readout is measured (after any pooling). `@` validates at once, so
/// `Z(0) @ Z(0)` is a `ValueError` where it is written.
#[pyclass(module = "polypus.qml", name = "PauliTerm", frozen)]
pub struct PauliTerm {
    /// The factors in the order written.
    factors: Vec<(String, usize)>,
}

impl PauliTerm {
    fn single(pauli: &str, position: usize) -> Self {
        PauliTerm {
            factors: vec![(pauli.to_string(), position)],
        }
    }
}

#[pymethods]
impl PauliTerm {
    fn __matmul__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        let mut factors = self.factors.clone();
        factors.extend(other.factors.iter().cloned());
        parse_pauli_string(&factors)?;
        Ok(PauliTerm { factors })
    }

    fn __repr__(&self) -> String {
        let factors: Vec<String> = self
            .factors
            .iter()
            .map(|(pauli, position)| format!("{}({position})", pauli.to_uppercase()))
            .collect();
        factors.join(" @ ")
    }
}

/// `polypus.qml.Z(position)`: Pauli `Z` on one logical qubit.
#[pyfunction(name = "Z")]
pub fn pauli_z(position: usize) -> PauliTerm {
    PauliTerm::single("z", position)
}

/// `polypus.qml.X(position)`: Pauli `X` on one logical qubit. A readout may mix
/// bases only if it still fits one measurement basis, which `train` checks.
#[pyfunction(name = "X")]
pub fn pauli_x(position: usize) -> PauliTerm {
    PauliTerm::single("x", position)
}

/// `polypus.qml.Y(position)`: Pauli `Y` on one logical qubit, with the same
/// single-basis rule as [`pauli_x`].
#[pyfunction(name = "Y")]
pub fn pauli_y(position: usize) -> PauliTerm {
    PauliTerm::single("y", position)
}

/// A weighted sum of Pauli products, `O = Σ cᵢ·Pᵢ`, for a readout that needs
/// coefficients or several terms (a cost Hamiltonian, say). Built from
/// `[(coefficient, term), …]`, each term a [`PauliTerm`] or a
/// `[(pauli, position), …]` list. A non-finite coefficient, an unknown Pauli or
/// a position repeated within a term is a `ValueError`.
#[pyclass(module = "polypus.qml", name = "Observable", frozen)]
pub struct PyObservable {
    inner: Observable,
}

#[pymethods]
impl PyObservable {
    #[new]
    fn new(terms: Vec<(f64, Bound<'_, PyAny>)>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(terms.len());
        for (coefficient, term) in terms {
            parsed.push((coefficient, parse_pauli_string(&term_factors(&term)?)?));
        }
        let inner = Observable::new(parsed).map_err(validation_to_py_err)?;
        Ok(PyObservable { inner })
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.terms.len()
    }

    fn __repr__(&self) -> String {
        format!("Observable(num_terms={})", self.inner.terms.len())
    }
}

/// The factors of one coefficient-free term: a [`PauliTerm`] or the
/// `[(pauli, position), …]` list it stands for.
fn term_factors(term: &Bound<'_, PyAny>) -> PyResult<Vec<(String, usize)>> {
    if let Ok(term) = term.cast::<PauliTerm>() {
        return Ok(term.get().factors.clone());
    }
    term.extract::<Vec<(String, usize)>>().map_err(|_| {
        PyTypeError::new_err(
            "a Pauli term must be a polypus.qml.PauliTerm (e.g. Z(0) @ Z(1)) or a list of \
             (pauli, position) factors (e.g. [(\"z\", 0), (\"z\", 1)])",
        )
    })
}

/// One readout observable: an [`Observable`](PyObservable) as it is, or a
/// coefficient-free term with coefficient `1.0`.
fn extract_observable(observable: &Bound<'_, PyAny>) -> PyResult<Observable> {
    if let Ok(weighted) = observable.cast::<PyObservable>() {
        return Ok(weighted.get().inner.clone());
    }
    let factors = term_factors(observable).map_err(|_| {
        PyTypeError::new_err(
            "each observable must be a polypus.qml.PauliTerm (e.g. Z(0) @ Z(1)), a \
             polypus.qml.Observable, or a list of (pauli, position) factors",
        )
    })?;
    unit_observable(&factors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoder and Ansatz
// ─────────────────────────────────────────────────────────────────────────────

/// Check `layers` over `num_qubits` for every error that does not depend on the
/// data, and return the number of trainable parameters they reserve.
///
/// The layers are compiled with a `Z(0)` readout and zero features. No layer's
/// parameter count depends on the feature count (encoders reserve none; the
/// trainable layers size theirs from the active qubits), and zero features trips
/// no feature-count check, so what this raises is exactly the structural errors
/// (a misplaced amplitude encoder, pooling below two qubits, …) and the count it
/// returns is the real one. A model without trainable parameters yet is not an
/// error here: an `Encoder` never has any, and an `Ansatz` may still be growing.
fn check_layers(num_qubits: usize, layers: &[Layer]) -> PyResult<usize> {
    // Nothing appended yet: a valid state to build from, which `compile` would
    // report as an empty model.
    if layers.is_empty() {
        return Ok(0);
    }
    let probe = Observable::new(vec![(
        1.0,
        PauliString::new(vec![(0, Pauli::Z)]).map_err(validation_to_py_err)?,
    )])
    .and_then(|observable| Readout::new(vec![observable], Decision::Raw))
    .map_err(validation_to_py_err)?;
    let model = layers
        .iter()
        .cloned()
        .fold(QuantumModel::new(num_qubits), QuantumModel::layer)
        .readout(probe);
    match model.compile(0) {
        Ok(compiled) => Ok(compiled.num_params()),
        Err(ValidationError::NoTrainableParams) => Ok(0),
        Err(e) => Err(validation_to_py_err(e)),
    }
}

/// `num_qubits` must be at least 1; shared by `Encoder` and `Ansatz`.
fn check_num_qubits(class: &str, num_qubits: usize) -> PyResult<()> {
    if num_qubits == 0 {
        return Err(PyValueError::new_err(format!(
            "{class} needs at least one qubit, got num_qubits=0"
        )));
    }
    Ok(())
}

/// The layers of an `Encoder` or `Ansatz` and their qubit count, with the
/// append-and-check step both share.
#[derive(Clone)]
struct Layers {
    num_qubits: usize,
    layers: Vec<Layer>,
}

impl Layers {
    /// Append `layer` if the extended stack still passes [`check_layers`];
    /// otherwise raise and leave the stack as it was.
    fn push(&mut self, layer: Layer) -> PyResult<()> {
        let mut layers = self.layers.clone();
        layers.push(layer);
        check_layers(self.num_qubits, &layers)?;
        self.layers = layers;
        Ok(())
    }
}

/// The data-encoding part of a native model: the layers that load each sample,
/// in the `feature_map` slot of `qml.train` and `qml.predict`.
///
/// `Encoder(num_qubits)` declares its width, which must match the `Ansatz`'s
/// (plan F1). Layers are appended by chaining, and each call checks what it can
/// without the data:
///
/// ```python
/// encoder = polypus.qml.Encoder(4).angle_encoder("ry")
/// ```
///
/// An encoder consumes no trainable parameters; the feature count it can carry
/// is checked against the dataset when training starts.
#[pyclass(module = "polypus.qml", name = "Encoder")]
pub struct Encoder {
    stack: Layers,
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        check_num_qubits("Encoder", num_qubits)?;
        Ok(Encoder {
            stack: Layers {
                num_qubits,
                layers: Vec::new(),
            },
        })
    }

    /// Append an angle encoder: one rotation about `axis` (`"rx"`/`"ry"`/`"rz"`)
    /// per feature, feature `j` on qubit `j`.
    fn angle_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        axis: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::AngleEncoder(AngleEncoder::new(parse_axis(axis)?));
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append an amplitude encoder, which loads a sample (`≤ 2^num_qubits`
    /// features) into the amplitudes of the state. It prepares the state from
    /// `|0…0⟩`, so it must be the first layer of the model.
    fn amplitude_encoder(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.stack
            .push(Layer::AmplitudeEncoder(polypus_qml::AmplitudeEncoder))?;
        Ok(slf)
    }

    /// Append a basis encoder: `X` on qubit `j` when feature `j` is `1.0`. Every
    /// feature must be `0.0` or `1.0`, checked where a sample is encoded.
    fn basis_encoder(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.stack.push(Layer::Basis(polypus_qml::BasisEncoder))?;
        Ok(slf)
    }

    /// Append an IQP (`ZZFeatureMap`) encoder over the `entanglement` pattern
    /// (`"linear"`/`"circular"`/`"full"`, default `"full"`).
    #[pyo3(signature = (entanglement="full"))]
    fn iqp_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        entanglement: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::Iqp(IqpEncoder {
            entanglement: parse_entanglement(entanglement)?,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.stack.num_qubits
    }

    #[getter]
    fn num_layers(&self) -> usize {
        self.stack.layers.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoder(num_qubits={}, num_layers={})",
            self.stack.num_qubits,
            self.stack.layers.len()
        )
    }
}

/// The trainable part of a native model, in the `ansatz` slot of `qml.train`
/// and `qml.predict`.
///
/// `Ansatz(num_qubits)` declares its width, which must match the `Encoder`'s
/// (plan F1). Besides its trainable layers it can re-encode the data between
/// them (data re-uploading) with the angle, basis and IQP encoders; the
/// amplitude encoder must come first, so it belongs in the `Encoder`.
///
/// ```python
/// ansatz = polypus.qml.Ansatz(4).conv("basic").pool("basic").hardware_efficient(1)
/// ```
///
/// Each call checks what it can without the data; for example, pooling below
/// two active qubits raises `ValueError` at the `.pool(...)` that does it.
#[pyclass(module = "polypus.qml", name = "Ansatz")]
pub struct Ansatz {
    stack: Layers,
}

#[pymethods]
impl Ansatz {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        check_num_qubits("Ansatz", num_qubits)?;
        Ok(Ansatz {
            stack: Layers {
                num_qubits,
                layers: Vec::new(),
            },
        })
    }

    /// Append a hardware-efficient block with `reps` repetitions. The defaults
    /// are Qiskit's `TwoLocal` ones: `rotations=["ry", "rz"]`, `entangler="cx"`,
    /// `entanglement="linear"` and a final rotation layer.
    #[pyo3(signature = (
        reps, rotations=None, entangler="cx", entanglement="linear",
        final_rotation_layer=true,
    ))]
    fn hardware_efficient<'py>(
        mut slf: PyRefMut<'py, Self>,
        reps: usize,
        rotations: Option<Vec<String>>,
        entangler: &str,
        entanglement: &str,
        final_rotation_layer: bool,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let rotations = match rotations {
            Some(axes) => axes
                .iter()
                .map(|axis| parse_axis(axis))
                .collect::<PyResult<Vec<_>>>()?,
            // Read off the Rust constructor so the two defaults never drift.
            None => HardwareEfficientAnsatz::new(reps).rotations,
        };
        let layer = Layer::HardwareEfficient(HardwareEfficientAnsatz {
            reps,
            rotations,
            entangler: parse_entangler(entangler)?,
            entanglement: parse_entanglement(entanglement)?,
            final_rotation_layer,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append Qiskit's `RealAmplitudes` preset: one `Ry` per rotation block,
    /// linear `CX` entanglement and a final rotation layer.
    fn real_amplitudes(mut slf: PyRefMut<'_, Self>, reps: usize) -> PyResult<PyRefMut<'_, Self>> {
        let layer = Layer::HardwareEfficient(HardwareEfficientAnsatz::real_amplitudes(reps));
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append a QCNN convolution: one `block` (`"basic"`, 4 parameters, or
    /// `"cartan"`, 3) shared by every pair of active qubits that `pairing`
    /// selects (`"even_pairs"`/`"odd_pairs"`/`"alternating"`, default
    /// `"alternating"`).
    #[pyo3(signature = (block, pairing="alternating"))]
    fn conv<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        pairing: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::Conv(ConvLayer {
            block: parse_conv_block(block)?,
            pairing: parse_pairing(pairing)?,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append a QCNN pooling layer: each adjacent pair of active qubits is
    /// reduced to one through the shared `block` (`"basic"`), `keep`
    /// (`"even_positions"`/`"odd_positions"`, default `"even_positions"`)
    /// deciding which of the two stays active.
    #[pyo3(signature = (block, keep="even_positions"))]
    fn pool<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        keep: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::Pool(PoolLayer {
            block: parse_pool_block(block)?,
            keep: parse_keep_rule(keep)?,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Re-upload the data with an angle encoder, as [`Encoder::angle_encoder`].
    fn angle_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        axis: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::AngleEncoder(AngleEncoder::new(parse_axis(axis)?));
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Re-upload the data with a basis encoder, as [`Encoder::basis_encoder`].
    fn basis_encoder(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.stack.push(Layer::Basis(polypus_qml::BasisEncoder))?;
        Ok(slf)
    }

    /// Re-upload the data with an IQP encoder, as [`Encoder::iqp_encoder`].
    #[pyo3(signature = (entanglement="full"))]
    fn iqp_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        entanglement: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::Iqp(IqpEncoder {
            entanglement: parse_entanglement(entanglement)?,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.stack.num_qubits
    }

    #[getter]
    fn num_layers(&self) -> usize {
        self.stack.layers.len()
    }

    /// The trainable parameters the layers reserve: the `dimensions` of a
    /// training run and the length of its `best_params`.
    #[getter]
    fn num_params(&self) -> PyResult<usize> {
        check_layers(self.stack.num_qubits, &self.stack.layers)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Ansatz(num_qubits={}, num_layers={}, num_params={})",
            self.stack.num_qubits,
            self.stack.layers.len(),
            self.num_params()?
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Readout
// ─────────────────────────────────────────────────────────────────────────────

/// What a native model measures and how that becomes a prediction: the third
/// piece of a [`Model`].
///
/// `Readout(observables, decision="raw", threshold=None)`:
///
/// - `observables`: what is measured, each a [`PauliTerm`] (`Z(0)`,
///   `Z(0) @ Z(1)`), an [`Observable`](PyObservable) (a weighted sum) or a
///   `[(pauli, position), …]` list. Positions are logical: they index the qubits
///   still active after any pooling. `"argmax"` reads one per class (at least
///   two); the other decisions predict from the first.
/// - `decision`: how the readout becomes a prediction: `"raw"` (the expectation
///   itself), `"sign"`, `"threshold"` (with `threshold`) or `"argmax"`.
///
/// How a prediction is scored against a label is a training choice, the `loss`
/// of [`Model::fit`].
#[pyclass(module = "polypus.qml", name = "Readout", frozen)]
pub struct PyReadout {
    pub(crate) inner: Readout,
}

#[pymethods]
impl PyReadout {
    #[new]
    #[pyo3(signature = (observables, decision="raw", threshold=None))]
    fn new(
        observables: Vec<Bound<'_, PyAny>>,
        decision: &str,
        threshold: Option<f64>,
    ) -> PyResult<Self> {
        let decision = parse_decision(decision, threshold)?;
        let observables = observables
            .iter()
            .map(extract_observable)
            .collect::<PyResult<Vec<_>>>()?;
        let inner = Readout::new(observables, decision).map_err(validation_to_py_err)?;
        Ok(PyReadout { inner })
    }

    #[getter]
    fn decision(&self) -> &'static str {
        decision_name(self.inner.decision)
    }

    #[getter]
    fn threshold(&self) -> Option<f64> {
        match self.inner.decision {
            Decision::Threshold(threshold) => Some(threshold),
            _ => None,
        }
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.observables.len()
    }

    fn __repr__(&self) -> String {
        let threshold = self
            .threshold()
            .map(|t| format!(", threshold={t}"))
            .unwrap_or_default();
        format!(
            "Readout(num_observables={}, decision=\"{}\"{threshold})",
            self.inner.observables.len(),
            self.decision()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Training or test samples for a native model, in the `x_train` slot of
/// `qml.train` (then without `y_train`) or the `x` slot of `qml.predict`.
///
/// `Dataset(x, y=None)` validates the rows (non-empty, equal length, finite)
/// and, if given, one finite label per row. Without `y` the dataset is
/// *unlabeled* (plan F4): it trains only with `loss="expectation"`, and the mark
/// survives a split. Labels are `float`s; class indices for
/// `"categorical_cross_entropy"` are whole numbers.
#[pyclass(module = "polypus.qml", name = "Dataset")]
pub struct PyDataset {
    pub(crate) inner: polypus_qml::Dataset,
    pub(crate) labeled: bool,
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (x, y=None))]
    fn new(x: Bound<'_, PyAny>, y: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let rows: Vec<Vec<f64>> = x.extract()?;
        let (labels, labeled) = match y {
            Some(y) => (y.extract::<Vec<f64>>()?, true),
            // Unlabeled: zeros satisfy the pure crate, which always carries
            // labels; `labeled` keeps them from ever reaching a supervised loss.
            None => (vec![0.0; rows.len()], false),
        };
        let inner =
            polypus_qml::Dataset::from_rows(&rows, &labels).map_err(validation_to_py_err)?;
        Ok(PyDataset { inner, labeled })
    }

    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Whether the dataset was built with labels.
    #[getter]
    fn labeled(&self) -> bool {
        self.labeled
    }

    /// The labels, or `None` for an unlabeled dataset.
    #[getter]
    fn labels(&self) -> Option<Vec<f64>> {
        self.labeled.then(|| self.inner.labels().to_vec())
    }

    /// Split into `(train, test)`: the samples are shuffled with `seed` and
    /// `floor(num_samples * test_fraction)` of them go to the test set.
    /// `test_fraction` lies in the open interval `(0, 1)`. An explicit `seed`
    /// reproduces the split; `None` draws one from OS entropy (contract C-7).
    /// Both halves keep this dataset's labeled or unlabeled mark.
    #[pyo3(signature = (test_fraction, seed=None))]
    fn train_test_split(&self, test_fraction: f64, seed: Option<u64>) -> PyResult<(Self, Self)> {
        let seed = seed_or_entropy(seed, random_seed)?;
        let (train, test) = self
            .inner
            .train_test_split(test_fraction, seed)
            .map_err(validation_to_py_err)?;
        let labeled = self.labeled;
        Ok((
            PyDataset {
                inner: train,
                labeled,
            },
            PyDataset {
                inner: test,
                labeled,
            },
        ))
    }

    /// Min–max scale every feature in place into `[lo, hi]`, with ranges taken
    /// from this dataset. A constant feature maps to `lo`. `[0, π]` suits angle
    /// encoding.
    fn scale_features_to(&mut self, lo: f64, hi: f64) {
        self.inner.scale_features_to(lo, hi);
    }

    /// The `(min, max)` of each feature: the scaler to freeze on a training set
    /// and replay on a test set with [`scale_features_with`](Self::scale_features_with).
    fn feature_ranges(&self) -> Vec<(f64, f64)> {
        self.inner.feature_ranges()
    }

    /// Min–max scale every feature in place into `[lo, hi]` with the given
    /// `ranges` (one per feature, else `ValueError`). Values outside a range map
    /// linearly outside `[lo, hi]`, as a frozen scaler should.
    fn scale_features_with(&mut self, ranges: Vec<(f64, f64)>, lo: f64, hi: f64) -> PyResult<()> {
        self.inner
            .scale_features_with(&ranges, lo, hi)
            .map_err(validation_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(num_samples={}, num_features={}{})",
            self.inner.num_samples(),
            self.inner.num_features(),
            if self.labeled { "" } else { ", unlabeled" }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model and Model.fit
// ─────────────────────────────────────────────────────────────────────────────

/// Whether `layer` loads the sample. A `match` rather than a `matches!` so a new
/// layer kind has to be classified here.
fn loads_data(layer: &Layer) -> bool {
    match layer {
        Layer::AngleEncoder(_) | Layer::AmplitudeEncoder(_) | Layer::Iqp(_) | Layer::Basis(_) => {
            true
        }
        Layer::HardwareEfficient(_) | Layer::Conv(_) | Layer::Pool(_) => false,
    }
}

/// A native QML model: an [`Encoder`] that loads each sample, an [`Ansatz`]
/// with the trainable layers and a [`Readout`](PyReadout) that turns the
/// measurement into a prediction (plan D4-R).
///
/// `Model(encoder, ansatz, readout)` copies the three pieces, so building on
/// them afterwards does not change the model, and checks everything that does
/// not depend on the data: the `Encoder` and `Ansatz` have the same width, some
/// layer loads the data, the layers reserve trainable parameters, and the
/// readout's positions exist among the qubits still active at the end. What
/// depends on the data (how many features the encoders can carry) is checked
/// when the model meets a dataset.
///
/// The model holds no trained parameters: [`fit`](Self::fit) returns them in a
/// `TrainResult`.
#[pyclass(module = "polypus.qml", name = "Model", frozen)]
pub struct Model {
    /// Encoder layers, then ansatz layers, with the readout.
    spec: QuantumModel,
    num_params: usize,
}

#[pymethods]
impl Model {
    #[new]
    fn new(
        encoder: PyRef<'_, Encoder>,
        ansatz: PyRef<'_, Ansatz>,
        readout: PyRef<'_, PyReadout>,
    ) -> PyResult<Self> {
        let (encoder, ansatz) = (&encoder.stack, &ansatz.stack);
        if encoder.num_qubits != ansatz.num_qubits {
            return Err(PyValueError::new_err(format!(
                "the Encoder has {} qubits but the Ansatz has {}; they must match",
                encoder.num_qubits, ansatz.num_qubits
            )));
        }
        if !encoder.layers.iter().chain(&ansatz.layers).any(loads_data) {
            return Err(PyValueError::new_err(
                "no layer loads the data: add an encoder to the Encoder (or re-upload the data \
                 in the Ansatz)",
            ));
        }
        let spec = encoder
            .layers
            .iter()
            .chain(&ansatz.layers)
            .cloned()
            .fold(QuantumModel::new(encoder.num_qubits), QuantumModel::layer)
            .readout(readout.inner.clone());
        // Zero features trips no feature-count check and changes no parameter
        // count (see `check_layers`), so this raises exactly the structural
        // errors and returns the real count.
        let num_params = spec
            .clone()
            .compile(0)
            .map_err(validation_to_py_err)?
            .num_params();
        Ok(Model { spec, num_params })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.spec.num_qubits()
    }

    /// The trainable parameters: the length of `best_params`.
    #[getter]
    fn num_params(&self) -> usize {
        self.num_params
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(num_qubits={}, num_layers={}, num_params={})",
            self.spec.num_qubits(),
            self.spec.num_layers(),
            self.num_params
        )
    }

    /// Bind trained parameters (typically a `TrainResult.best_params`) into a
    /// [`TrainedModel`] for inference. There must be exactly `num_params` of
    /// them, all finite. The model itself is unchanged.
    fn bind(&self, params: Vec<f64>) -> PyResult<TrainedModel> {
        let inner = polypus_qml::TrainedModel::new(self.spec.clone(), params)
            .map_err(validation_to_py_err)?;
        Ok(TrainedModel { inner })
    }

    /// The circuit the model runs for the one sample `x` (a flat sequence of
    /// floats), as a `polypus.Circuit` to draw or export: the features fixed,
    /// then the readout's basis change and the terminal `measure_all`.
    ///
    /// Without `params` the trainable angles stay free (`num_params` of them,
    /// named `theta_0`, `theta_1`, … in the order of `best_params`); with
    /// `params` it is `self.bind(params).circuit(x)`, every angle fixed, and
    /// `params` is checked as `bind` checks it. `x` is checked as one row of
    /// `fit` is: a feature count the encoders cannot carry, or a value that is
    /// not finite, is a `ValueError`.
    #[pyo3(signature = (x, params=None))]
    fn circuit(&self, x: &Bound<'_, PyAny>, params: Option<Vec<f64>>) -> PyResult<Circuit> {
        match params {
            None => sample_circuit(&self.spec, x, None),
            Some(params) => self.bind(params)?.circuit(x),
        }
    }

    /// Train the model on `x` (rows of features, or a `Dataset`) and, for
    /// supervised training, the labels `y`, and return a `TrainResult`.
    ///
    /// - **Labels and loss.** With labels (`y`, or a labeled `Dataset`, never
    ///   both) `loss` is required: `"squared_error"`, `"binary_cross_entropy"`,
    ///   `"hinge"`, or `"categorical_cross_entropy"` for an `"argmax"` readout.
    ///   Without labels the run maximises the readout's first observable
    ///   (`loss=None` or `"expectation"`); a supervised loss is then an error.
    /// - **`method`** is any optimizer (`DE`, `PSO`, `QNG`, `CMAES`, `Adam`,
    ///   `SPSA`, `Rotosolve`, `External`); Adam gets the exact parameter-shift
    ///   gradient. `Rotosolve` needs `loss="expectation"` and every trainable
    ///   parameter in one rotation.
    /// - **`exact=True`** uses exact probabilities instead of shots (then `shots`
    ///   does not matter), on `infrastructure="local"` with the native simulator
    ///   only.
    /// - **`batch_size`** scores each evaluation on a fresh, seed-determined
    ///   subset of that many samples, `1 <= batch_size < num_samples`; the
    ///   reported `best_fitness` is always on the whole training set (C-5).
    /// - **`initial_point`** starts the optimizer from a given point, such as
    ///   the `params` of a `TrainedModel` read back with `qml.load` to keep
    ///   training it: `num_params` finite numbers, with `train`'s rules (C-5).
    /// - **`final_shots`** re-evaluates the best candidates with that many
    ///   shots once the optimizer has stopped, as in `train`, on every training
    ///   sample even with `batch_size`, into `TrainResult.final_params` and
    ///   `final_fitness`. `fit` refuses it with `exact=True`, which uses no shots.
    /// - **`callback`** is called once per iteration of the optimizer with a
    ///   `polypus.TrainProgress`, as in `polypus.train`: returning `True` stops
    ///   the run, which returns the `TrainResult` it has reached
    ///   (`converged=False`), and the `final_shots` batch is still sent. With
    ///   `batch_size`, its `best_fitness` is a minibatch estimate, except for
    ///   Adam.
    /// - **Execution** takes the arguments of `polypus.train`. `backend=None`
    ///   is the native simulator on `infrastructure="local"` and Aer otherwise
    ///   (CUNQA's QPUs run Aer); an explicit `backend` wins. `seed` drives the
    ///   optimizer, the shots and the minibatches (C-7), and `id` is a prefix to
    ///   which a UUID is appended (C-9).
    ///
    /// Every check runs before a backend exists and raises
    /// `polypus.ConfigurationError` (a `ValueError`), except a value of the
    /// wrong type (a `final_shots` that is not an integer, a `callback` that is
    /// neither callable nor `None`), a `TypeError`. The fitness is the
    /// negated mean loss over the samples, which the optimizers maximise.
    #[pyo3(signature = (
        x, y=None, *, method, loss=None, shots=1024, infrastructure="local", n_qpus=1,
        nodes=1, cores_per_qpu=2, walltime=None, options=None, sim_method="automatic",
        noise_model=None, backend=None, seed=None, id="qml", exact=false, batch_size=None,
        initial_point=None, final_shots=None, callback=None,
    ))]
    fn fit<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        y: Option<&Bound<'py, PyAny>>,
        method: &Bound<'py, PyAny>,
        loss: Option<&str>,
        shots: u32,
        infrastructure: &str,
        n_qpus: u32,
        nodes: u32,
        cores_per_qpu: u32,
        walltime: Option<String>,
        options: Option<OptionsArg>,
        sim_method: &str,
        noise_model: Option<Bound<'py, PyAny>>,
        backend: Option<&str>,
        seed: Option<u64>,
        id: &str,
        exact: bool,
        batch_size: Option<usize>,
        initial_point: Option<Vec<f64>>,
        final_shots: Option<Bound<'py, PyAny>>,
        callback: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let start = Instant::now();
        let py = x.py();
        validate_shots_and_qpus(shots, n_qpus)?;
        validate_cunqa_allocation(infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
        validate_id(id)?;
        let loss = loss.map(parse_loss).transpose()?;
        let effective_seed = resolve_optimizer_seed(seed, method_seed(method), random_seed)?;
        let dimensions = u32::try_from(self.num_params).map_err(|_| {
            configuration_error(format!(
                "the model has {} trainable parameters, more than an optimizer can search",
                self.num_params
            ))
        })?;
        // The optimizer and its point, then `final_shots` and `callback`, in
        // `train`'s order (C-5).
        validate_method(method, dimensions, initial_point.as_deref())?;
        let final_shots = validate_final_shots(final_shots.as_ref())?;
        let callback = validate_callback(callback.as_ref())?;
        let backend = resolve_backend(backend, infrastructure);
        if exact && !(infrastructure == "local" && is_native_backend(backend)) {
            return Err(configuration_error(
                "exact=True needs infrastructure=\"local\" and backend=\"polypus\": only the \
                 native simulator computes exact probabilities",
            ));
        }
        check_exact_final_shots(exact, final_shots)?;

        let (dataset, labeled) = native_dataset(x, y)?;
        let loss = resolve_loss(loss, labeled)?;
        let compiled = self
            .spec
            .clone()
            .compile(dataset.num_features())
            .map_err(validation_to_config_err)?;
        // Rotosolve's step is exact only on single-frequency models with a
        // linear fitness: checked before any backend, and so any CUNQA
        // allocation, exists, as `train` checks a circuit.
        if is_rotosolve(method) {
            check_rotosolve_model(&compiled, &dataset, loss)?;
        }
        let problem = QmlProblem::new(compiled, dataset, loss).map_err(validation_to_config_err)?;
        let num_samples = problem.num_circuits();
        // A batch as large as the training set is the run without minibatches.
        let minibatch = match batch_size {
            Some(size) if size == 0 || size >= num_samples => {
                return Err(configuration_error(format!(
                    "batch_size must satisfy 1 <= batch_size < {num_samples} (the number of \
                     training samples), got {size}"
                )));
            }
            Some(size) => Some(MinibatchConfig::new(size, effective_seed)),
            None => None,
        };
        let exact = exact.then(|| Arc::new(NativeStatevectorBackend::new(effective_seed)));

        // Native circuits never ask for fusion here: `None` keeps the backend's
        // default.
        let backend_config = build_backend_config(
            infrastructure,
            backend,
            sim_method,
            noise_model.map(|nm| nm.unbind()),
            nodes,
            cores_per_qpu,
            walltime.as_deref(),
            None,
            options.map(OptionsArg::into_inner),
        )?;
        let effective_id = unique_id(id)?;
        let config = Arc::new(ExecutionConfig {
            id: effective_id.clone(),
            shots,
            n_qpus,
            infrastructure: infrastructure.to_string(),
            backend_config,
            opt_level: OptLevel::default(),
            seed: Some(effective_seed),
        });
        // Converted before the backend exists, as in `train`, so a failure here
        // cannot leave it (a CUNQA `qraise`) unclosed.
        let errors = OracleErrorSlot::new();
        let method = method_from_pyclass(method, dimensions, &errors, &effective_id)?;
        log_training_start(
            &effective_id,
            infrastructure,
            backend,
            n_qpus,
            shots,
            effective_seed,
        );
        let backend = Infrastructure::create_backend(&config)
            .map_err(crate::exceptions::backend_error_to_pyerr)?;
        let scheduler = scheduler_over(backend, config.run_params())
            .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
        let flow = TrainFlow {
            factory: QmlOracleFactory {
                source: QmlSource::Native {
                    problem: Arc::new(problem),
                    exact,
                },
                minibatch,
            },
            method,
            dimensions,
            seed: effective_seed,
            errors: errors.clone(),
            initial_point,
            final_shots,
            observer: training_observer(callback, &errors, &effective_id),
        };
        // GIL released for the run, as in `train`; the interrupt token keeps
        // Ctrl+C prompt between waves.
        let token = interruptible_token();
        let result = py.detach(|| scheduler.run_cancellable(flow, &token));
        scheduler.close();
        finish_optimization(py, result, effective_seed, effective_id, start)
    }
}

/// One scheduler over `backend`, with its default planner, for a run that
/// owns the backend. Should the pairing fail (`Resources::new`), the backend
/// is closed before the error returns, so no allocation it holds (a CUNQA
/// `qraise`) outlives the call.
fn scheduler_over(
    backend: Arc<dyn QuantumBackend>,
    params: RunParams,
) -> Result<Scheduler, InfrastructureError> {
    match Resources::new(Arc::clone(&backend), None, Arc::new(params)) {
        Ok(resources) => Ok(Scheduler::ephemeral(resources)),
        Err(e) => {
            backend.close();
            Err(e)
        }
    }
}

/// The native API's backend when none is named (plan F5): the native simulator
/// on `local`, Aer elsewhere (CUNQA's QPUs run Aer; a registry backend ignores
/// it).
fn resolve_backend<'a>(backend: Option<&'a str>, infrastructure: &str) -> &'a str {
    backend.unwrap_or(if infrastructure == "local" {
        "polypus"
    } else {
        "aer"
    })
}

/// `fit`'s `final_shots` with `exact=True`: exact mode computes probabilities
/// and takes no shots, so a re-evaluation with more of them would only repeat
/// the exact values. Refused before any backend exists (C-11), after
/// `final_shots` itself has been checked.
fn check_exact_final_shots(exact: bool, final_shots: Option<u32>) -> PyResult<()> {
    if exact && final_shots.is_some() {
        return Err(configuration_error(
            "final_shots does not apply with exact=True: exact mode computes exact \
             probabilities and uses no shots. Drop final_shots, or exact=True to train \
             with shots",
        ));
    }
    Ok(())
}

/// The training set and whether it has labels: a [`Dataset`](PyDataset) as it
/// is (then `y` is an error), or rows plus an optional `y`, read with the same
/// label checks as `qml.train`'s `y_train`.
fn native_dataset(
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<(polypus_qml::Dataset, bool)> {
    if let Ok(dataset) = x.cast::<PyDataset>() {
        if y.is_some() {
            return Err(configuration_error(
                "x is a polypus.qml.Dataset, which carries its own labels; pass y only with \
                 plain rows",
            ));
        }
        let dataset = dataset.try_borrow()?;
        return Ok((dataset.inner.clone(), dataset.labeled));
    }
    let rows: Vec<Vec<f64>> = x.extract()?;
    let labels = y
        .map(|y| {
            extract_labels(y).map(|labels| {
                labels
                    .into_iter()
                    .map(|label| match label {
                        Label::Class(class) => class as f64,
                        Label::Real(value) => value,
                    })
                    .collect::<Vec<_>>()
            })
        })
        .transpose()?;
    let labeled = labels.is_some();
    // Unlabeled: zeros satisfy the pure crate; `labeled` keeps a supervised loss
    // from ever training against them (plan F3).
    let labels = labels.unwrap_or_else(|| vec![0.0; rows.len()]);
    let dataset =
        polypus_qml::Dataset::from_rows(&rows, &labels).map_err(validation_to_config_err)?;
    Ok((dataset, labeled))
}

/// The loss a run trains with (plans F2 and F3): with labels, a supervised loss,
/// which must be named; without labels, `expectation`. Every other pairing is a
/// `polypus.ConfigurationError`, so labels are never ignored and a supervised loss never trains
/// against made-up labels.
fn resolve_loss(loss: Option<Loss>, labeled: bool) -> PyResult<Loss> {
    match (loss, labeled) {
        (None | Some(Loss::Expectation), false) => Ok(Loss::Expectation),
        (Some(loss), false) => Err(configuration_error(format!(
            "loss=\"{}\" needs labels: pass y (or a labeled Dataset), or use \
             loss=\"expectation\" to maximise the readout",
            loss_name(loss)
        ))),
        (None, true) => Err(configuration_error(
            "the training set has labels, so fit needs a loss, e.g. loss=\"hinge\"",
        )),
        (Some(Loss::Expectation), true) => Err(configuration_error(
            "loss=\"expectation\" does not use labels: drop y (or use a Dataset without \
             labels), or choose a supervised loss",
        )),
        (Some(loss), true) => Ok(loss),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainedModel and ExactResult
// ─────────────────────────────────────────────────────────────────────────────

/// The exact basis-state probabilities of each row, from
/// `TrainedModel.run(..., exact=True)` (plan H3): no shots, so no seed and no
/// sampling noise.
#[pyclass(module = "polypus.qml", name = "ExactResult", frozen)]
pub struct ExactResult {
    /// One `dict[str, float]` per row, keyed like counts (C-3); basis states of
    /// probability exactly zero are left out.
    #[pyo3(get)]
    pub probabilities: Py<PyAny>,
    /// Run identifier used for logging.
    #[pyo3(get)]
    pub id: String,
    /// Always the native simulator (`"polypus"`).
    #[pyo3(get)]
    pub backend: String,
    /// Always `"local"`.
    #[pyo3(get)]
    pub infrastructure: String,
}

#[pymethods]
impl ExactResult {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ExactResult(rows={}, id='{}', backend='{}', infrastructure='{}')",
            self.probabilities.bind(py).len()?,
            self.id,
            self.backend,
            self.infrastructure
        ))
    }
}

/// The execution arguments `TrainedModel.predict` and `run`, and `qml.expval`
/// and `qml.jacobian`, share.
struct Execution<'a, 'py> {
    shots: u32,
    infrastructure: &'a str,
    n_qpus: u32,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<String>,
    options: Option<OptionsArg>,
    sim_method: &'a str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: Option<&'a str>,
    seed: Option<u64>,
    exact: bool,
}

impl<'a> Execution<'a, '_> {
    /// The checks that need no data: `shots` and `n_qpus`, the CUNQA
    /// allocation, then `exact` against the backend, which it resolves
    /// (`backend=None` by infrastructure) and returns. Each a
    /// `polypus.ConfigurationError`.
    fn check(&self) -> PyResult<&'a str> {
        validate_shots_and_qpus(self.shots, self.n_qpus)?;
        validate_cunqa_allocation(
            self.infrastructure,
            self.nodes,
            self.cores_per_qpu,
            self.walltime.as_deref(),
        )?;
        let backend = resolve_backend(self.backend, self.infrastructure);
        if self.exact && !(self.infrastructure == "local" && is_native_backend(backend)) {
            return Err(configuration_error(
                "exact=True needs infrastructure=\"local\" and backend=\"polypus\": only the \
                 native simulator computes exact probabilities",
            ));
        }
        Ok(backend)
    }

    /// The run's [`ExecutionConfig`] for `backend` (from [`check`](Self::check)),
    /// with an `id` of `id_prefix` plus a UUID and the seed: `None` on `qmio`,
    /// where an explicit one is a `polypus.ConfigurationError`, else the
    /// explicit one or one drawn from OS entropy (C-7).
    fn into_config(self, backend: &str, id_prefix: &str) -> PyResult<ExecutionConfig> {
        // As in `run_quantum_circuit` (C-7): `qmio` is hardware, so a seed there
        // would be silently ineffective.
        let seed = if self.infrastructure == "qmio" {
            if self.seed.is_some() {
                return Err(configuration_error(
                    "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
                ));
            }
            None
        } else {
            Some(seed_or_entropy(self.seed, random_seed)?)
        };
        let backend_config = build_backend_config(
            self.infrastructure,
            backend,
            self.sim_method,
            self.noise_model.map(|nm| nm.unbind()),
            self.nodes,
            self.cores_per_qpu,
            self.walltime.as_deref(),
            None,
            self.options.map(OptionsArg::into_inner),
        )?;
        Ok(ExecutionConfig {
            id: unique_id(id_prefix)?,
            shots: self.shots,
            n_qpus: self.n_qpus,
            infrastructure: self.infrastructure.to_string(),
            backend_config,
            opt_level: OptLevel::default(),
            seed,
        })
    }
}

/// One run of a [`TrainedModel`] over some rows: one result per row, the
/// compiled model that reads them, and the run's manifest.
struct RowsRun {
    compiled: CompiledModel,
    results: RowResults,
    id: String,
    seed: Option<u64>,
    backend: String,
    infrastructure: String,
}

enum RowResults {
    Counts(Vec<Counts>),
    Exact(Vec<HashMap<String, f64>>),
}

/// A `polypus-qml` [`QmlError`] from reading a result as a `ValueError`: at
/// inference it is always about the input.
fn qml_to_py_err(e: QmlError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A [`QmlError`] from binding a row before anything runs, as a
/// `polypus.ConfigurationError`, like [`validation_to_config_err`].
fn qml_to_config_err(e: QmlError) -> PyErr {
    configuration_error(e.to_string())
}

/// A [`Model`] bound to trained parameters, from [`Model::bind`]: what runs
/// inference. Immutable: binding other parameters gives another `TrainedModel`
/// and leaves the `Model` as it was.
///
/// [`predict`](Self::predict) returns the readout's decisions, and
/// [`run`](Self::run) the raw counts (or exact probabilities). Both take the
/// rows as rows of features or a `Dataset` (its labels are not used) and the
/// execution arguments of `Model.fit`, all keyword-only; the seed is drawn when
/// omitted and must be omitted on `qmio`, real hardware Polypus cannot seed.
#[pyclass(module = "polypus.qml", name = "TrainedModel", frozen)]
pub struct TrainedModel {
    /// The crate's trained model, which also defines the save format.
    inner: polypus_qml::TrainedModel,
}

impl TrainedModel {
    /// Bind every row of `x` to the parameters and run the circuits: sampled
    /// through the scheduler, or, with `exact`, as exact probabilities on the
    /// native simulator. Every check runs before a backend exists and raises
    /// `polypus.ConfigurationError`.
    fn run_rows(&self, x: &Bound<'_, PyAny>, exec: Execution<'_, '_>) -> PyResult<RowsRun> {
        let py = x.py();
        let start = Instant::now();
        let backend = exec.check()?;
        // The labels play no part in inference.
        let (rows, _) = native_dataset(x, None)?;
        let compiled = self
            .inner
            .compile(rows.num_features())
            .map_err(validation_to_config_err)?;
        let circuits = (0..rows.num_samples())
            .map(|i| {
                compiled
                    .bind(rows.sample(i), self.inner.params())
                    .map(BoundCircuit::Native)
                    .map_err(qml_to_config_err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (infrastructure, n_qpus, shots, exact) =
            (exec.infrastructure, exec.n_qpus, exec.shots, exec.exact);
        let config = exec.into_config(backend, &format!("qml_{n_qpus}_{infrastructure}"))?;
        let (id, seed) = (config.id.clone(), config.seed);
        log::info!(
            "qml inference {id} starting: rows={}, infrastructure={infrastructure}, \
             backend={backend}, n_qpus={n_qpus}, shots={shots}, exact={exact}, seed={seed:?}",
            circuits.len(),
        );
        let results = if exact {
            let native = NativeStatevectorBackend::new(seed.unwrap_or_default());
            let params = config.run_params();
            py.detach(|| native.run_circuits_exact(&circuits, &params))
                .map(RowResults::Exact)
                .map_err(crate::exceptions::backend_error_to_pyerr)?
        } else {
            // One run, GIL released, the rows in waves of the backend's
            // concurrency; Ctrl+C is honoured between waves.
            py.detach(move || -> Result<Vec<Counts>, InfrastructureError> {
                let backend = Infrastructure::create_backend(&config)
                    .map_err(InfrastructureError::Backend)?;
                let scheduler = scheduler_over(backend, config.run_params())?;
                let out = scheduler
                    .run_cancellable(RunCircuitFlow { circuits, shots }, &interruptible_token());
                scheduler.close();
                out
            })
            .map(RowResults::Counts)
            .map_err(crate::exceptions::infrastructure_error_to_pyerr)?
        };
        log::info!(
            "qml inference {id} completed: duration={:?}",
            start.elapsed()
        );
        Ok(RowsRun {
            compiled,
            results,
            id,
            seed,
            backend: backend.to_string(),
            infrastructure: infrastructure.to_string(),
        })
    }
}

#[pymethods]
impl TrainedModel {
    /// The bound parameters, in the order `Model.fit` returned them.
    #[getter]
    fn params(&self) -> Vec<f64> {
        self.inner.params().to_vec()
    }

    /// The model the parameters are bound to.
    #[getter]
    fn model(&self) -> Model {
        Model {
            spec: self.inner.spec().clone(),
            num_params: self.inner.num_params(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainedModel(num_qubits={}, num_params={})",
            self.inner.spec().num_qubits(),
            self.inner.num_params()
        )
    }

    /// Save to `path` (a `str` or `pathlib.Path`) as JSON, replacing any file
    /// there: `{"format": "polypus.qml.TrainedModel", "version": 1, "spec": …,
    /// "params": […]}`, the format the `polypus-qml` crate reads too. The file
    /// is written beside `path` and then renamed over it, so a failure never
    /// leaves a partial model. Floats are written so they read back bit for
    /// bit. A disk failure raises `OSError` naming the path.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let json = serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("cannot serialize the model: {e}")))?;
        write_atomically(&path, json.as_bytes()).map_err(|e| io_to_py_err(&path, e))
    }

    /// The readout's decision for each row of `x`: `int` for `"sign"` and
    /// `"threshold"` (±1) and for `"argmax"` (the class index), `float` for
    /// `"raw"` (the expectation itself). With `exact=True` the decisions come
    /// from exact probabilities instead of shots.
    #[pyo3(signature = (
        x, *, shots=1024, infrastructure="local", n_qpus=1, nodes=1, cores_per_qpu=2,
        walltime=None, options=None, sim_method="automatic", noise_model=None, backend=None,
        seed=None, exact=false,
    ))]
    fn predict<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        shots: u32,
        infrastructure: &str,
        n_qpus: u32,
        nodes: u32,
        cores_per_qpu: u32,
        walltime: Option<String>,
        options: Option<OptionsArg>,
        sim_method: &str,
        noise_model: Option<Bound<'py, PyAny>>,
        backend: Option<&str>,
        seed: Option<u64>,
        exact: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let run = self.run_rows(
            x,
            Execution {
                shots,
                infrastructure,
                n_qpus,
                nodes,
                cores_per_qpu,
                walltime,
                options,
                sim_method,
                noise_model,
                backend,
                seed,
                exact,
            },
        )?;
        let values = match &run.results {
            RowResults::Counts(counts) => counts
                .iter()
                .map(|counts| run.compiled.predict_from_counts(counts))
                .collect::<Result<Vec<_>, _>>(),
            RowResults::Exact(probabilities) => probabilities
                .iter()
                .map(|p| run.compiled.predict_from_probabilities(p))
                .collect::<Result<Vec<_>, _>>(),
        }
        .map_err(qml_to_py_err)?;
        let py = x.py();
        let decisions = PyList::empty(py);
        for value in values {
            // A discrete decision is a whole number in `f64`; Python gets it as
            // the `int` it is.
            match self.inner.decision() {
                Decision::Raw => decisions.append(value)?,
                Decision::Sign | Decision::Threshold(_) | Decision::Argmax => {
                    decisions.append(value as i64)?
                }
            }
        }
        Ok(decisions)
    }

    /// The raw results for each row of `x`: a `RunResult` whose `counts` holds
    /// one dict per row (`merged_counts` is `None`: the rows are different
    /// circuits), or, with `exact=True`, an `ExactResult` of probabilities.
    #[pyo3(signature = (
        x, *, shots=1024, infrastructure="local", n_qpus=1, nodes=1, cores_per_qpu=2,
        walltime=None, options=None, sim_method="automatic", noise_model=None, backend=None,
        seed=None, exact=false,
    ))]
    fn run<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        shots: u32,
        infrastructure: &str,
        n_qpus: u32,
        nodes: u32,
        cores_per_qpu: u32,
        walltime: Option<String>,
        options: Option<OptionsArg>,
        sim_method: &str,
        noise_model: Option<Bound<'py, PyAny>>,
        backend: Option<&str>,
        seed: Option<u64>,
        exact: bool,
    ) -> PyResult<Py<PyAny>> {
        let run = self.run_rows(
            x,
            Execution {
                shots,
                infrastructure,
                n_qpus,
                nodes,
                cores_per_qpu,
                walltime,
                options,
                sim_method,
                noise_model,
                backend,
                seed,
                exact,
            },
        )?;
        let py = x.py();
        match run.results {
            RowResults::Counts(counts) => Py::new(
                py,
                RunResult {
                    counts: counts_to_pylist(py, &counts)?.into_any().unbind(),
                    merged_counts: py.None(),
                    id: run.id,
                    seed: run.seed,
                    backend: run.backend,
                    infrastructure: run.infrastructure,
                },
            )
            .map(Py::into_any),
            RowResults::Exact(probabilities) => {
                let list = PyList::empty(py);
                for row in &probabilities {
                    let dict = PyDict::new(py);
                    let mut keys: Vec<&String> = row.keys().collect();
                    keys.sort();
                    for key in keys {
                        dict.set_item(key, row[key])?;
                    }
                    list.append(dict)?;
                }
                Py::new(
                    py,
                    ExactResult {
                        probabilities: list.into_any().unbind(),
                        id: run.id,
                        backend: run.backend,
                        infrastructure: run.infrastructure,
                    },
                )
                .map(Py::into_any)
            }
        }
    }

    /// The circuit `run` executes for the one sample `x`, every angle fixed
    /// (`num_params == 0`): the same as `self.model.circuit(x, self.params)`.
    fn circuit(&self, x: &Bound<'_, PyAny>) -> PyResult<Circuit> {
        sample_circuit(self.inner.spec(), x, Some(self.inner.params()))
    }
}

/// The circuit `spec` runs for the one sample `x`, as `run_rows` builds it for
/// a row: compiled for `x`'s feature count and with the template's free angles
/// or, with `params` (already checked against the model), bound to them. `x`
/// goes through the checks a row does, through the same `Dataset`.
fn sample_circuit(
    spec: &QuantumModel,
    x: &Bound<'_, PyAny>,
    params: Option<&[f64]>,
) -> PyResult<Circuit> {
    let x: Vec<f64> = x.extract().map_err(|e: PyErr| {
        if e.is_instance_of::<PyTypeError>(x.py()) {
            let err = PyTypeError::new_err("x must be one sample, a flat sequence of floats");
            err.set_cause(x.py(), Some(e));
            err
        } else {
            e
        }
    })?;
    let rows = polypus_qml::Dataset::from_rows(std::slice::from_ref(&x), &[0.0])
        .map_err(validation_to_py_err)?;
    let compiled = spec
        .clone()
        .compile(rows.num_features())
        .map_err(validation_to_py_err)?;
    let inner = match params {
        None => compiled.template_for(rows.sample(0)),
        Some(params) => compiled
            .bind(rows.sample(0), params)
            .and_then(|bound| parameterized_from_concrete(bound).map_err(QmlError::from)),
    }
    .map_err(qml_to_py_err)?;
    Ok(Circuit { inner })
}

/// `concrete` as a [`ParameterizedCircuit`], the type a `polypus.Circuit`
/// wraps: the same qubits and gates, every angle fixed, so no parameters.
/// Every gate goes through `try_push`, which re-checks it (C-4), so a gate that
/// could not be pushed is an error rather than a circuit the builder would
/// refuse.
fn parameterized_from_concrete(
    concrete: ConcreteCircuit,
) -> Result<ParameterizedCircuit, CircuitError> {
    let mut circuit = ParameterizedCircuit::new(concrete.num_qubits());
    for gate in concrete.into_gates() {
        circuit.try_push(gate)?;
    }
    Ok(circuit)
}

/// Write `bytes` to a temporary file beside `path`, then rename it over `path`,
/// so `path` is either the old file or the complete new one. The temporary
/// file is removed if anything fails.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "the path names no file")
    })?;
    let mut temporary = name.to_os_string();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = path.with_file_name(temporary);
    let result = std::fs::write(&temporary, bytes).and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        // Best effort: the original error is the one worth reporting.
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// An I/O error on `path` as the matching `OSError` subclass, naming the path.
fn io_to_py_err(path: &Path, e: std::io::Error) -> PyErr {
    let message = format!("{}: {e}", path.display());
    match e.kind() {
        std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(message),
        std::io::ErrorKind::PermissionDenied => PyPermissionError::new_err(message),
        _ => PyOSError::new_err(message),
    }
}

/// `polypus.qml.load(path)`: read a [`TrainedModel`] saved by
/// `TrainedModel.save` (or by the `polypus-qml` crate). The file is checked as
/// it is read: its format and version, then the model and parameters exactly
/// as `Model` and `Model.bind` check them, so a corrupt or edited file is a
/// `ValueError` and never an inconsistent model. Loading only parses JSON; it
/// never runs code. A missing or unreadable file raises `OSError`.
#[pyfunction(name = "load")]
pub fn load(path: PathBuf) -> PyResult<TrainedModel> {
    let json = std::fs::read_to_string(&path).map_err(|e| io_to_py_err(&path, e))?;
    let inner = serde_json::from_str::<polypus_qml::TrainedModel>(&json).map_err(|e| {
        PyValueError::new_err(format!(
            "{} is not a valid polypus.qml.TrainedModel file: {e}",
            path.display()
        ))
    })?;
    Ok(TrainedModel { inner })
}

// ─────────────────────────────────────────────────────────────────────────────
// qml.expval and qml.jacobian: the loss-free primitives
// ─────────────────────────────────────────────────────────────────────────────

/// `wrt` of `qml.jacobian`, a closed set (C-11, "Options").
fn parse_wrt(wrt: &str) -> PyResult<JacobianWrt> {
    match wrt {
        "params" => Ok(JacobianWrt::Params),
        "inputs" => Ok(JacobianWrt::Inputs),
        "both" => Ok(JacobianWrt::Both),
        other => Err(configuration_error(format!(
            "unknown wrt '{other}'; expected \"params\", \"inputs\" or \"both\""
        ))),
    }
}

/// The results of one block, one per circuit, in the block's order.
enum BlockResults {
    Counts(Vec<Counts>),
    Exact(Vec<HashMap<String, f64>>),
}

/// What runs every block of one primitive call: one backend behind one
/// scheduler, so one connection (one `qraise` on CUNQA) for the whole call, or,
/// with `exact=True`, the native simulator's exact path.
enum BlockRunner {
    Sampled {
        scheduler: Scheduler,
        max_concurrency: usize,
        shots: u32,
    },
    Exact {
        native: NativeStatevectorBackend,
        params: RunParams,
    },
}

impl BlockRunner {
    /// One scheduler over `backend` ([`scheduler_over`], which closes the
    /// backend if it cannot be built).
    fn sampled(
        backend: Arc<dyn QuantumBackend>,
        params: RunParams,
        shots: u32,
    ) -> Result<Self, InfrastructureError> {
        let max_concurrency = backend.capabilities().max_concurrency;
        Ok(BlockRunner::Sampled {
            scheduler: scheduler_over(backend, params)?,
            max_concurrency,
            shots,
        })
    }

    /// The backend's batch-agnostic concurrency; "no limit" for the exact
    /// path, whose parallelism is the host's.
    fn max_concurrency(&self) -> usize {
        match self {
            BlockRunner::Sampled {
                max_concurrency, ..
            } => *max_concurrency,
            BlockRunner::Exact { .. } => usize::MAX,
        }
    }

    fn run(
        &self,
        circuits: Vec<BoundCircuit>,
        cancel: &CancelToken,
    ) -> Result<BlockResults, InfrastructureError> {
        match self {
            BlockRunner::Sampled {
                scheduler, shots, ..
            } => scheduler
                .run_cancellable(
                    RunCircuitFlow {
                        circuits,
                        shots: *shots,
                    },
                    cancel,
                )
                .map(BlockResults::Counts),
            BlockRunner::Exact { native, params } => native
                .run_circuits_exact(&circuits, params)
                .map(BlockResults::Exact)
                .map_err(InfrastructureError::Backend),
        }
    }

    fn close(&self) {
        if let BlockRunner::Sampled { scheduler, .. } = self {
            scheduler.close();
        }
    }
}

/// Why a primitive call stopped once running.
#[derive(Debug)]
enum PrimitiveError {
    /// The backend, a cancellation or a Ctrl+C.
    Infrastructure(InfrastructureError),
    /// Binding or reading a block.
    Qml(QmlError),
    /// A block outside the samples: a bookkeeping bug, never the caller's.
    Internal(String),
}

impl From<QmlError> for PrimitiveError {
    fn from(e: QmlError) -> Self {
        PrimitiveError::Qml(e)
    }
}

impl PrimitiveError {
    fn into_py_err(self) -> PyErr {
        match self {
            PrimitiveError::Infrastructure(e) => {
                crate::exceptions::infrastructure_error_to_pyerr(e)
            }
            PrimitiveError::Qml(e) => qml_to_py_err(e),
            PrimitiveError::Internal(message) => {
                PyRuntimeError::new_err(format!("internal error: {message}"))
            }
        }
    }

    fn block_out_of_range(samples: &Range<usize>, num_samples: usize) -> Self {
        PrimitiveError::Internal(format!(
            "block {samples:?} lies outside the {num_samples} sample(s)"
        ))
    }
}

/// One primitive call's state: its circuits block by block, and what it keeps
/// from their results, appended in sample order.
trait Primitive {
    fn num_samples(&self) -> usize;
    fn circuits_per_sample(&self) -> usize;
    /// The circuits of the samples at `samples`, sample-major.
    fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError>;
    /// Read the results of [`circuits`](Self::circuits)`(samples)`.
    fn read(&mut self, samples: Range<usize>, results: BlockResults) -> Result<(), PrimitiveError>;
}

/// The detached part of a primitive call: create its one runner, size the
/// blocks from the runner's concurrency (`block_size`), then run every block on
/// it, in sample order: check for a cancellation or a Ctrl+C (as the planner
/// does between waves; the exact path has no planner), bind the block's
/// circuits, run them, read them and drop them before the next. Once created,
/// the runner is closed on every path out, error and cancellation included.
fn run_in_blocks(
    primitive: &mut impl Primitive,
    create: impl FnOnce() -> Result<BlockRunner, InfrastructureError>,
    block_size: impl FnOnce(usize) -> usize,
    cancel: &CancelToken,
) -> Result<(), PrimitiveError> {
    let runner = create().map_err(PrimitiveError::Infrastructure)?;
    let block = block_size(runner.max_concurrency()).max(1);
    log::debug!(
        "qml primitive: {} sample(s) in blocks of {block}",
        primitive.num_samples()
    );
    let outcome = run_blocks_on(&runner, primitive, block, cancel);
    runner.close();
    outcome
}

fn run_blocks_on(
    runner: &BlockRunner,
    primitive: &mut impl Primitive,
    block: usize,
    cancel: &CancelToken,
) -> Result<(), PrimitiveError> {
    let num_samples = primitive.num_samples();
    let mut start = 0;
    while start < num_samples {
        if cancel.is_cancelled() {
            return Err(PrimitiveError::Infrastructure(
                InfrastructureError::Cancelled,
            ));
        }
        cancel
            .poll_interrupt()
            .map_err(PrimitiveError::Infrastructure)?;
        let end = start.saturating_add(block).min(num_samples);
        let circuits = primitive
            .circuits(start..end)?
            .into_iter()
            .map(BoundCircuit::Native)
            .collect();
        let results = runner
            .run(circuits, cancel)
            .map_err(PrimitiveError::Infrastructure)?;
        primitive.read(start..end, results)?;
        start = end;
    }
    Ok(())
}

/// `qml.expval`: one circuit per sample, every readout observable read from it.
struct Expval {
    compiled: CompiledModel,
    /// One template per sample, built (and so checked) before any backend.
    templates: Vec<ParameterizedCircuit>,
    params: Vec<f64>,
    /// `(num_samples, num_observables)`, row-major, filled block by block.
    values: Vec<f64>,
}

impl Primitive for Expval {
    fn num_samples(&self) -> usize {
        self.templates.len()
    }

    fn circuits_per_sample(&self) -> usize {
        1
    }

    fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError> {
        let templates = self
            .templates
            .get(samples.clone())
            .ok_or_else(|| PrimitiveError::block_out_of_range(&samples, self.templates.len()))?;
        Ok(templates
            .iter()
            .map(|template| template.assign_parameters(&self.params))
            .collect::<Result<Vec<_>, _>>()
            .map_err(QmlError::from)?)
    }

    fn read(&mut self, samples: Range<usize>, results: BlockResults) -> Result<(), PrimitiveError> {
        let got = match &results {
            BlockResults::Counts(counts) => counts.len(),
            BlockResults::Exact(probabilities) => probabilities.len(),
        };
        if got != samples.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: samples.len(),
                got,
            }
            .into());
        }
        match results {
            BlockResults::Counts(counts) => {
                for counts in &counts {
                    self.values
                        .extend(self.compiled.expectations_from_counts(counts)?);
                }
            }
            BlockResults::Exact(probabilities) => {
                for probabilities in &probabilities {
                    self.values.extend(
                        self.compiled
                            .expectations_from_probabilities(probabilities)?,
                    );
                }
            }
        }
        Ok(())
    }
}

/// `qml.jacobian`: `2 · shifts` circuits per sample, read into the parts
/// `wrt` asks for.
struct JacobianRun {
    plan: ExpectationJacobian,
    params: Vec<f64>,
    /// `(num_samples, num_observables, num_params)`, row-major.
    by_params: Vec<f64>,
    /// `(num_samples, num_observables, num_features)`, row-major.
    by_inputs: Vec<f64>,
}

impl Primitive for JacobianRun {
    fn num_samples(&self) -> usize {
        self.plan.num_samples()
    }

    fn circuits_per_sample(&self) -> usize {
        self.plan.circuits_per_sample()
    }

    fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError> {
        let block = self
            .plan
            .block(samples.clone())
            .ok_or_else(|| PrimitiveError::block_out_of_range(&samples, self.plan.num_samples()))?;
        Ok(block.circuits(&self.params)?)
    }

    fn read(&mut self, samples: Range<usize>, results: BlockResults) -> Result<(), PrimitiveError> {
        let block = self
            .plan
            .block(samples.clone())
            .ok_or_else(|| PrimitiveError::block_out_of_range(&samples, self.plan.num_samples()))?;
        let jacobian = match &results {
            BlockResults::Counts(counts) => block.jacobian_from_counts(counts)?,
            BlockResults::Exact(probabilities) => {
                block.jacobian_from_probabilities(probabilities)?
            }
        };
        let (by_params, by_inputs) = jacobian.into_parts();
        self.by_params.extend(by_params.unwrap_or_default());
        self.by_inputs.extend(by_inputs.unwrap_or_default());
        Ok(())
    }
}

/// `model` as a native [`Model`], or the `TypeError` `qml.<entry>` raises for
/// anything else.
fn primitive_model<'py>(model: &Bound<'py, PyAny>, entry: &str) -> PyResult<Bound<'py, Model>> {
    if let Ok(model) = model.cast::<Model>() {
        return Ok(model.clone());
    }
    let hint = if model.is_instance_of::<TrainedModel>() {
        "; for a TrainedModel, pass trained.model and trained.params"
    } else {
        ""
    };
    Err(PyTypeError::new_err(format!(
        "qml.{entry} takes a polypus.qml.Model as model, got {}{hint}",
        model.get_type().name()?
    )))
}

/// The checks both primitives make after the execution arguments, in order and
/// all before any backend exists: the rows of `x` (rows or a `Dataset`, whose
/// labels are not used) against the model, then `params` as `Model.bind` checks
/// them. Each a `polypus.ConfigurationError`. Returns the compiled model, the
/// rows and the parameters.
fn primitive_inputs(
    model: &Model,
    x: &Bound<'_, PyAny>,
    params: Vec<f64>,
) -> PyResult<(CompiledModel, polypus_qml::Dataset, Vec<f64>)> {
    let (rows, _) = native_dataset(x, None)?;
    let compiled = model
        .spec
        .clone()
        .compile(rows.num_features())
        .map_err(validation_to_config_err)?;
    let bound = polypus_qml::TrainedModel::new(model.spec.clone(), params)
        .map_err(validation_to_config_err)?;
    Ok((compiled, rows, bound.params().to_vec()))
}

/// Run `primitive` on the backend `exec` names (`backend`, from
/// [`Execution::check`]), with the GIL released for the whole block loop.
/// Draws the seed and the id (logged, not returned: pass a `seed` to replay).
fn run_primitive<P: Primitive + Send>(
    py: Python<'_>,
    entry: &str,
    primitive: &mut P,
    exec: Execution<'_, '_>,
    backend: &str,
) -> PyResult<()> {
    let start = Instant::now();
    let (infrastructure, n_qpus, shots, exact) =
        (exec.infrastructure, exec.n_qpus, exec.shots, exec.exact);
    let config = exec.into_config(backend, &format!("qml_{entry}_{n_qpus}_{infrastructure}"))?;
    let (id, seed) = (config.id.clone(), config.seed);
    let circuits_per_sample = primitive.circuits_per_sample();
    log::info!(
        "qml {entry} {id} starting: rows={}, circuits={}, infrastructure={infrastructure}, \
         backend={backend}, n_qpus={n_qpus}, shots={shots}, exact={exact}, seed={seed:?}",
        primitive.num_samples(),
        primitive.num_samples() * circuits_per_sample,
    );
    py.detach(|| {
        let create = || -> Result<BlockRunner, InfrastructureError> {
            if exact {
                return Ok(BlockRunner::Exact {
                    native: NativeStatevectorBackend::new(seed.unwrap_or_default()),
                    params: config.run_params(),
                });
            }
            let backend =
                Infrastructure::create_backend(&config).map_err(InfrastructureError::Backend)?;
            BlockRunner::sampled(backend, config.run_params(), shots)
        };
        run_in_blocks(
            primitive,
            create,
            // The `QmlOracle`'s window policy, with rows for candidates.
            |backend| candidate_window_size(backend, circuits_per_sample),
            &interruptible_token(),
        )
    })
    .map_err(PrimitiveError::into_py_err)?;
    log::info!("qml {entry} {id} completed: duration={:?}", start.elapsed());
    Ok(())
}

/// `values` as the `(rows, cols)` array it holds row-major, or why not.
fn array2(rows: usize, cols: usize, values: Vec<f64>) -> Result<Array2<f64>, ShapeError> {
    Array2::from_shape_vec((rows, cols), values)
}

/// `values` as the `(samples, observables, columns)` array it holds row-major,
/// or why not.
fn array3(
    samples: usize,
    observables: usize,
    columns: usize,
    values: Vec<f64>,
) -> Result<Array3<f64>, ShapeError> {
    Array3::from_shape_vec((samples, observables, columns), values)
}

/// A flat result that does not fill its array: a bookkeeping bug.
fn shape_to_py_err(e: ShapeError) -> PyErr {
    PyRuntimeError::new_err(format!("internal error assembling the result: {e}"))
}

/// `polypus.qml.expval(model, x, params, *, …)`: the expectation of every
/// readout observable for each row of `x`, as a `float64` `numpy.ndarray` of
/// shape `(rows, observables)`, C-contiguous, in row order.
///
/// - **`model`** is a `polypus.qml.Model` (anything else is a `TypeError`);
///   **`x`** is rows of features or a `Dataset` (its labels are not used);
///   **`params`** is a list, tuple or NumPy array of exactly `num_params` finite
///   values, as `Model.bind` takes them.
/// - **Every observable**, whatever the readout's decision: one column per class
///   for `"argmax"`, the weighted sum for an `Observable`.
/// - **Execution** takes `TrainedModel.run`'s arguments with its rules: with
///   `exact=True` the exact expectations (`infrastructure="local"`,
///   `backend="polypus"`), otherwise each one estimated from `shots` shots;
///   `backend=None` by infrastructure; `seed` drawn when omitted and refused on
///   `qmio`. The seed is logged, not returned: pass one to replay.
/// - **Runs** one circuit per row, in blocks of rows over one backend for the
///   whole call, with the GIL released; Ctrl+C stops it between blocks.
///
/// Every check runs before a backend exists and raises
/// `polypus.ConfigurationError` (a `ValueError`).
#[pyfunction(name = "expval", signature = (
    model, x, params, *, shots=1024, infrastructure="local", n_qpus=1, nodes=1,
    cores_per_qpu=2, walltime=None, options=None, sim_method="automatic", noise_model=None,
    backend=None, seed=None, exact=false,
))]
pub fn expval<'py>(
    model: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    params: Vec<f64>,
    shots: u32,
    infrastructure: &str,
    n_qpus: u32,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<String>,
    options: Option<OptionsArg>,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: Option<&str>,
    seed: Option<u64>,
    exact: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let py = x.py();
    let exec = Execution {
        shots,
        infrastructure,
        n_qpus,
        nodes,
        cores_per_qpu,
        walltime,
        options,
        sim_method,
        noise_model,
        backend,
        seed,
        exact,
    };
    let model = primitive_model(model, "expval")?;
    let backend = exec.check()?;
    let (compiled, rows, params) = primitive_inputs(model.get(), x, params)?;
    let templates = (0..rows.num_samples())
        .map(|i| compiled.template_for(rows.sample(i)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(qml_to_config_err)?;
    let (num_samples, num_observables) = (templates.len(), compiled.num_observables());
    let mut primitive = Expval {
        compiled,
        templates,
        params,
        values: Vec::with_capacity(num_samples * num_observables),
    };
    run_primitive(py, "expval", &mut primitive, exec, backend)?;
    let values = array2(num_samples, num_observables, primitive.values).map_err(shape_to_py_err)?;
    Ok(values.into_pyarray(py))
}

/// `polypus.qml.jacobian(model, x, params, *, wrt="params", …)`: the Jacobian
/// of every readout observable's expectation for each row of `x`, by the
/// parameter-shift rule, as `float64` `numpy.ndarray`s, C-contiguous, in row
/// order. The derivative of the expectations, not of a loss: compose the loss
/// with the chain rule.
///
/// - **`wrt="params"`** gives shape `(rows, observables, num_params)`;
///   **`"inputs"`** gives `(rows, observables, features)`, for angle and IQP
///   encoders (an amplitude or basis encoder has no derivative in the sample:
///   a `polypus.ConfigurationError` naming it); **`"both"`** gives the tuple
///   `(by_params, by_inputs)` from one set of circuits. Any other `wrt` is a
///   `polypus.ConfigurationError`.
/// - **Circuits:** `2` per shifted gate per row: every gate that reads a
///   trainable parameter (a parameter shared by several gates counts once per
///   gate) for `"params"`, every encoder gate that reads the row for
///   `"inputs"`, both for `"both"`. No unshifted circuit.
/// - **`model`**, **`x`**, **`params`** and the execution arguments are
///   [`expval`]'s, with its rules. With `exact=True` the result is the
///   Jacobian itself; with shots, an unbiased estimate from `shots` shots per
///   circuit.
///
/// Every check runs before a backend exists and raises
/// `polypus.ConfigurationError` (a `ValueError`).
#[pyfunction(name = "jacobian", signature = (
    model, x, params, *, wrt="params", shots=1024, infrastructure="local", n_qpus=1, nodes=1,
    cores_per_qpu=2, walltime=None, options=None, sim_method="automatic", noise_model=None,
    backend=None, seed=None, exact=false,
))]
pub fn jacobian<'py>(
    model: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    params: Vec<f64>,
    wrt: &str,
    shots: u32,
    infrastructure: &str,
    n_qpus: u32,
    nodes: u32,
    cores_per_qpu: u32,
    walltime: Option<String>,
    options: Option<OptionsArg>,
    sim_method: &str,
    noise_model: Option<Bound<'py, PyAny>>,
    backend: Option<&str>,
    seed: Option<u64>,
    exact: bool,
) -> PyResult<Py<PyAny>> {
    let py = x.py();
    let exec = Execution {
        shots,
        infrastructure,
        n_qpus,
        nodes,
        cores_per_qpu,
        walltime,
        options,
        sim_method,
        noise_model,
        backend,
        seed,
        exact,
    };
    let model = primitive_model(model, "jacobian")?;
    let wrt = parse_wrt(wrt)?;
    let backend = exec.check()?;
    let (compiled, rows, params) = primitive_inputs(model.get(), x, params)?;
    let samples: Vec<&[f64]> = (0..rows.num_samples()).map(|i| rows.sample(i)).collect();
    let plan = compiled
        .jacobian(&samples, wrt)
        .map_err(qml_to_config_err)?;
    let (n, k) = (plan.num_samples(), plan.num_observables());
    let (num_params, num_features) = (plan.num_params(), plan.num_features());
    let mut primitive = JacobianRun {
        plan,
        params,
        by_params: Vec::new(),
        by_inputs: Vec::new(),
    };
    run_primitive(py, "jacobian", &mut primitive, exec, backend)?;
    let array = |columns: usize, values: Vec<f64>| {
        array3(n, k, columns, values)
            .map(|a| a.into_pyarray(py).into_any())
            .map_err(shape_to_py_err)
    };
    let JacobianRun {
        by_params,
        by_inputs,
        ..
    } = primitive;
    Ok(match wrt {
        JacobianWrt::Params => array(num_params, by_params)?.unbind(),
        JacobianWrt::Inputs => array(num_features, by_inputs)?.unbind(),
        JacobianWrt::Both => PyTuple::new(
            py,
            [
                array(num_params, by_params)?,
                array(num_features, by_inputs)?,
            ],
        )?
        .into_any()
        .unbind(),
    })
}

/// `qml.train` and `qml.predict` run Qiskit circuits. A native piece in their
/// `feature_map` or `ansatz` is a `TypeError` that says so (plan D4-R), rather
/// than a Qiskit error from composing it.
pub(crate) fn reject_native_pieces(
    entry: &str,
    feature_map: &Bound<'_, PyAny>,
    ansatz: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let native =
        |obj: &Bound<'_, PyAny>| obj.is_instance_of::<Encoder>() || obj.is_instance_of::<Ansatz>();
    if native(feature_map) || native(ansatz) {
        return Err(PyTypeError::new_err(format!(
            "qml.{entry} takes Qiskit QuantumCircuits as feature_map and ansatz; a \
             polypus.qml.Encoder or Ansatz belongs in a polypus.qml.Model, trained with \
             Model.fit"
        )));
    }
    Ok(())
}

/// Add the native classes and the Pauli helpers to the `polypus.qml` module.
pub(crate) fn register(qml: &Bound<'_, PyModule>) -> PyResult<()> {
    qml.add_class::<Encoder>()?;
    qml.add_class::<Ansatz>()?;
    qml.add_class::<PyReadout>()?;
    qml.add_class::<Model>()?;
    qml.add_class::<TrainedModel>()?;
    qml.add_class::<ExactResult>()?;
    qml.add_function(wrap_pyfunction!(load, qml)?)?;
    qml.add_function(wrap_pyfunction!(expval, qml)?)?;
    qml.add_function(wrap_pyfunction!(jacobian, qml)?)?;
    qml.add_class::<PyDataset>()?;
    qml.add_class::<PauliTerm>()?;
    qml.add_class::<PyObservable>()?;
    qml.add_function(wrap_pyfunction!(pauli_z, qml)?)?;
    qml.add_function(wrap_pyfunction!(pauli_x, qml)?)?;
    qml.add_function(wrap_pyfunction!(pauli_y, qml)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_loss_round_trips_through_its_name() {
        for loss in [
            Loss::SquaredError,
            Loss::BinaryCrossEntropy,
            Loss::Hinge,
            Loss::CategoricalCrossEntropy,
            Loss::Expectation,
        ] {
            assert_eq!(parse_loss(loss_name(loss)).unwrap(), loss);
        }
    }

    #[test]
    fn every_decision_round_trips_through_its_name() {
        for (decision, threshold) in [
            (Decision::Raw, None),
            (Decision::Sign, None),
            (Decision::Argmax, None),
            (Decision::Threshold(0.25), Some(0.25)),
        ] {
            assert_eq!(
                parse_decision(decision_name(decision), threshold).unwrap(),
                decision
            );
        }
    }

    #[test]
    fn a_threshold_is_required_by_threshold_and_rejected_elsewhere() {
        Python::initialize();
        assert!(parse_decision("threshold", None).is_err());
        for decision in ["raw", "sign", "argmax"] {
            let err = parse_decision(decision, Some(0.5)).unwrap_err();
            assert!(err.to_string().contains(decision), "{err}");
        }
    }

    #[test]
    fn unknown_options_name_the_valid_ones() {
        Python::initialize();
        let err = parse_axis("rq").unwrap_err().to_string();
        assert!(err.contains("\"rx\", \"ry\" or \"rz\""), "{err}");
        let err = parse_loss("l2").unwrap_err().to_string();
        assert!(err.contains("\"expectation\""), "{err}");
        assert!(parse_pairing("pairs").is_err());
        assert!(parse_keep_rule("left").is_err());
        assert!(parse_pool_block("cartan").is_err());
        assert!(parse_conv_block("dense").is_err());
        assert!(parse_entangler("cy").is_err());
        assert!(parse_entanglement("ring").is_err());
        assert!(parse_pauli("i").is_err());
    }

    #[test]
    fn check_layers_counts_parameters_without_the_data() {
        Python::initialize();
        // Encoders reserve none, and an ansatz still being built may have none.
        assert_eq!(
            check_layers(
                4,
                &[Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry))]
            )
            .unwrap(),
            0
        );
        assert_eq!(check_layers(4, &[]).unwrap(), 0);
        // A basic convolution has 4 parameters, shared by every pair.
        assert_eq!(
            check_layers(4, &[Layer::Conv(ConvLayer::new(ConvBlock::Basic))]).unwrap(),
            4
        );
        assert_eq!(
            check_layers(
                2,
                &[Layer::HardwareEfficient(
                    HardwareEfficientAnsatz::real_amplitudes(1)
                )]
            )
            .unwrap(),
            4
        );
    }

    #[test]
    fn resolve_loss_pairs_losses_with_labels() {
        Python::initialize();
        assert_eq!(resolve_loss(None, false).unwrap(), Loss::Expectation);
        assert_eq!(
            resolve_loss(Some(Loss::Expectation), false).unwrap(),
            Loss::Expectation
        );
        assert_eq!(resolve_loss(Some(Loss::Hinge), true).unwrap(), Loss::Hinge);
        assert!(resolve_loss(Some(Loss::Hinge), false).is_err());
        assert!(resolve_loss(None, true).is_err());
        assert!(resolve_loss(Some(Loss::Expectation), true).is_err());
    }

    #[test]
    fn final_shots_is_refused_only_with_exact() {
        Python::initialize();
        assert!(check_exact_final_shots(false, None).is_ok());
        assert!(check_exact_final_shots(false, Some(4096)).is_ok());
        assert!(check_exact_final_shots(true, None).is_ok());
        let err = check_exact_final_shots(true, Some(4096))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("final_shots does not apply with exact=True"),
            "{err}"
        );
    }

    #[test]
    fn the_default_backend_follows_the_infrastructure() {
        assert_eq!(resolve_backend(None, "local"), "polypus");
        assert_eq!(resolve_backend(None, "cunqa"), "aer");
        assert_eq!(resolve_backend(None, "qmio"), "aer");
        assert_eq!(resolve_backend(Some("aer"), "local"), "aer");
    }

    #[test]
    fn only_encoders_load_data() {
        assert!(loads_data(&Layer::AngleEncoder(AngleEncoder::new(
            RotationAxis::Ry
        ))));
        assert!(loads_data(&Layer::AmplitudeEncoder(
            polypus_qml::AmplitudeEncoder
        )));
        assert!(loads_data(&Layer::Iqp(IqpEncoder::new())));
        assert!(loads_data(&Layer::Basis(polypus_qml::BasisEncoder)));
        assert!(!loads_data(&Layer::HardwareEfficient(
            HardwareEfficientAnsatz::new(1)
        )));
        assert!(!loads_data(&Layer::Conv(ConvLayer::new(ConvBlock::Basic))));
        assert!(!loads_data(&Layer::Pool(PoolLayer::new(PoolBlock::Basic))));
    }

    #[test]
    fn check_layers_raises_the_structural_errors() {
        Python::initialize();
        // The amplitude encoder must come first.
        let misplaced = [
            Layer::Basis(polypus_qml::BasisEncoder),
            Layer::AmplitudeEncoder(polypus_qml::AmplitudeEncoder),
        ];
        assert!(check_layers(2, &misplaced).is_err());
        // Pooling one active qubit.
        let pool = || Layer::Pool(PoolLayer::new(PoolBlock::Basic));
        assert!(check_layers(2, &[pool()]).is_ok());
        assert!(check_layers(2, &[pool(), pool()]).is_err());
    }

    #[test]
    fn a_bound_circuit_converts_gate_for_gate() {
        use polypus_circuit::GateInstruction;

        // A `Y` readout, so the bound circuit carries the basis change too.
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Y)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let compiled = QuantumModel::new(2)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(2)
            .unwrap();
        let (x, theta) = ([0.3, 1.2], [0.1, -0.2, 0.3, -0.4]);
        let bound = compiled.bind(&x, &theta).unwrap();

        let circuit = parameterized_from_concrete(bound.clone()).unwrap();
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_params(), 0);
        assert_eq!(circuit.gates(), bound.gates());
        assert_eq!(circuit.num_clbits(), 2);
        assert_eq!(circuit.gates().last(), Some(&GateInstruction::MeasureAll));
        assert!(circuit
            .gates()
            .windows(2)
            .any(|w| w == [GateInstruction::Sdg(0), GateInstruction::H(0)]));
        // Binding nothing gives back the very circuit the model runs.
        assert_eq!(circuit.assign_parameters(&[]).unwrap(), bound);
    }

    // ── qml.expval / qml.jacobian ────────────────────────────────────────────

    #[test]
    fn wrt_is_a_closed_set_naming_the_valid_values() {
        Python::initialize();
        assert_eq!(parse_wrt("params").unwrap(), JacobianWrt::Params);
        assert_eq!(parse_wrt("inputs").unwrap(), JacobianWrt::Inputs);
        assert_eq!(parse_wrt("both").unwrap(), JacobianWrt::Both);
        let err = parse_wrt("theta").unwrap_err().to_string();
        assert!(err.contains("\"params\", \"inputs\" or \"both\""), "{err}");
    }

    #[test]
    fn flat_results_become_c_contiguous_arrays_or_an_error() {
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let a = array3(2, 3, 2, values.clone()).unwrap();
        assert_eq!(a.shape(), &[2, 3, 2]);
        assert!(a.is_standard_layout());
        // Row-major: [s][o][p] is element (s·3 + o)·2 + p.
        assert_eq!(a[[1, 2, 0]], 10.0);
        let b = array2(4, 3, values.clone()).unwrap();
        assert_eq!(b.shape(), &[4, 3]);
        assert!(b.is_standard_layout());
        // A length that does not fill the shape is an error, never a panic.
        assert!(array3(2, 3, 3, values.clone()).is_err());
        assert!(array2(5, 3, values).is_err());
        // An empty part (no features, no rows) is a valid empty array.
        assert_eq!(array3(2, 3, 0, Vec::new()).unwrap().shape(), &[2, 3, 0]);
    }

    /// A 3-qubit model with an angle encoder, a re-upload and an IQP encoder,
    /// read out as `⟨Z₀⟩` and `⟨Z₁Z₂⟩`, over 5 rows.
    fn primitive_model_and_rows() -> (CompiledModel, Vec<Vec<f64>>, Vec<f64>) {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
                Observable::new(vec![(
                    1.0,
                    PauliString::new(vec![(1, Pauli::Z), (2, Pauli::Z)]).unwrap(),
                )])
                .unwrap(),
            ],
            Decision::Argmax,
        )
        .unwrap();
        let compiled = QuantumModel::new(3)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1)))
            .layer(Layer::Iqp(IqpEncoder::new()))
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(3)
            .unwrap();
        let rows = (0..5)
            .map(|i| (0..3).map(|j| 0.3 + 0.21 * (i * 3 + j) as f64).collect())
            .collect();
        let theta = (0..compiled.num_params())
            .map(|p| -0.7 + 0.13 * p as f64)
            .collect();
        (compiled, rows, theta)
    }

    fn jacobian_run(wrt: JacobianWrt) -> JacobianRun {
        let (compiled, rows, theta) = primitive_model_and_rows();
        JacobianRun {
            plan: compiled.jacobian(&rows, wrt).unwrap(),
            params: theta,
            by_params: Vec::new(),
            by_inputs: Vec::new(),
        }
    }

    fn expval_run() -> Expval {
        let (compiled, rows, theta) = primitive_model_and_rows();
        let templates = rows
            .iter()
            .map(|row| compiled.template_for(row).unwrap())
            .collect();
        Expval {
            compiled,
            templates,
            params: theta,
            values: Vec::new(),
        }
    }

    fn run_params(seed: u64, shots: u32) -> RunParams {
        RunParams {
            id: "qml-primitive-test".to_string(),
            shots,
            opt_level: OptLevel::default(),
            seed: Some(seed),
        }
    }

    /// Run `primitive` sampled on a native backend seeded with `seed`, in
    /// blocks of `block`.
    fn sampled_on_native(primitive: &mut impl Primitive, seed: u64, block: usize) {
        run_in_blocks(
            primitive,
            || {
                BlockRunner::sampled(
                    Arc::new(NativeStatevectorBackend::new(seed)),
                    run_params(seed, 256),
                    256,
                )
            },
            |_| block,
            &CancelToken::default(),
        )
        .unwrap();
    }

    #[test]
    fn a_seeded_native_result_does_not_depend_on_the_block_size() {
        // The native backend's circuit counter keeps running across the blocks,
        // so every circuit gets the seed of its position in the whole call.
        let mut whole = jacobian_run(JacobianWrt::Both);
        sampled_on_native(&mut whole, 7, usize::MAX);
        assert!(!whole.by_params.is_empty() && !whole.by_inputs.is_empty());
        for block in [1, 2, 3] {
            let mut blocked = jacobian_run(JacobianWrt::Both);
            sampled_on_native(&mut blocked, 7, block);
            assert_eq!(blocked.by_params, whole.by_params, "block {block}");
            assert_eq!(blocked.by_inputs, whole.by_inputs, "block {block}");
        }
        let mut other_seed = jacobian_run(JacobianWrt::Both);
        sampled_on_native(&mut other_seed, 8, usize::MAX);
        assert_ne!(other_seed.by_params, whole.by_params);

        let mut whole = expval_run();
        sampled_on_native(&mut whole, 11, usize::MAX);
        for block in [1, 2, 4] {
            let mut blocked = expval_run();
            sampled_on_native(&mut blocked, 11, block);
            assert_eq!(blocked.values, whole.values, "block {block}");
        }
    }

    #[test]
    fn rows_come_back_in_row_order_whatever_the_blocks() {
        // Exactly, each row's values are those of the row on its own.
        let (compiled, rows, theta) = primitive_model_and_rows();
        for block in [1, 2, 5] {
            let mut run = jacobian_run(JacobianWrt::Params);
            let mut expval = expval_run();
            for primitive in [&mut run as &mut dyn PrimitiveDyn, &mut expval] {
                primitive.run_exact(block);
            }
            let k = compiled.num_observables();
            let per_row = k * compiled.num_params();
            for (i, row) in rows.iter().enumerate() {
                let alone = compiled
                    .jacobian(std::slice::from_ref(row), JacobianWrt::Params)
                    .unwrap();
                let circuits: Vec<BoundCircuit> = alone
                    .circuits(&theta)
                    .unwrap()
                    .into_iter()
                    .map(BoundCircuit::Native)
                    .collect();
                let probabilities = NativeStatevectorBackend::new(0)
                    .run_circuits_exact(&circuits, &run_params(0, 1))
                    .unwrap();
                let expected = alone.jacobian_from_probabilities(&probabilities).unwrap();
                assert_eq!(
                    Some(&run.by_params[i * per_row..(i + 1) * per_row]),
                    expected.params(),
                    "block {block}, row {i}"
                );
                let bound = BoundCircuit::Native(compiled.bind(row, &theta).unwrap());
                let probabilities = NativeStatevectorBackend::new(0)
                    .run_circuits_exact(&[bound], &run_params(0, 1))
                    .unwrap();
                assert_eq!(
                    expval.values[i * k..(i + 1) * k],
                    compiled
                        .expectations_from_probabilities(&probabilities[0])
                        .unwrap(),
                    "block {block}, row {i}"
                );
            }
        }
    }

    /// Object-safe shim so one loop can drive both primitives exactly.
    trait PrimitiveDyn {
        fn run_exact(&mut self, block: usize);
    }

    impl<P: Primitive> PrimitiveDyn for P {
        fn run_exact(&mut self, block: usize) {
            run_in_blocks(
                self,
                || {
                    Ok(BlockRunner::Exact {
                        native: NativeStatevectorBackend::new(0),
                        params: run_params(0, 1),
                    })
                },
                |_| block,
                &CancelToken::default(),
            )
            .unwrap();
        }
    }

    /// A native backend that counts the calls it receives and its closes.
    struct CountingBackend {
        inner: NativeStatevectorBackend,
        runs: Arc<AtomicUsize>,
        closes: Arc<AtomicUsize>,
    }

    impl QuantumBackend for CountingBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            params: &RunParams,
        ) -> Result<Vec<Counts>, crate::infrastructure::BackendError> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            self.inner.run_circuits(qcs, params)
        }

        fn close(&self) {
            self.closes.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Counters {
        creates: AtomicUsize,
        runs: Arc<AtomicUsize>,
        closes: Arc<AtomicUsize>,
    }

    impl Counters {
        fn new() -> Self {
            Counters {
                creates: AtomicUsize::new(0),
                runs: Arc::new(AtomicUsize::new(0)),
                closes: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn create(&self) -> Result<BlockRunner, InfrastructureError> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            BlockRunner::sampled(
                Arc::new(CountingBackend {
                    inner: NativeStatevectorBackend::new(3),
                    runs: Arc::clone(&self.runs),
                    closes: Arc::clone(&self.closes),
                }),
                run_params(3, 64),
                64,
            )
        }

        fn get(&self) -> (usize, usize, usize) {
            (
                self.creates.load(Ordering::SeqCst),
                self.runs.load(Ordering::SeqCst),
                self.closes.load(Ordering::SeqCst),
            )
        }
    }

    #[test]
    fn several_blocks_share_one_backend_created_and_closed_once() {
        let counters = Counters::new();
        let mut run = jacobian_run(JacobianWrt::Params);
        run_in_blocks(
            &mut run,
            || counters.create(),
            |_| 2,
            &CancelToken::default(),
        )
        .unwrap();
        // 5 rows in blocks of 2: three blocks, each one call on the one backend.
        assert_eq!(counters.get(), (1, 3, 1));
    }

    /// A backend whose own default planner it cannot serve (shot
    /// distribution it does not support), so no scheduler can be built over
    /// it; it counts its closes.
    struct UnpairableBackend {
        closes: Arc<AtomicUsize>,
    }

    impl QuantumBackend for UnpairableBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _params: &RunParams,
        ) -> Result<Vec<Counts>, crate::infrastructure::BackendError> {
            Ok(qcs.iter().map(|_| Counts::new()).collect())
        }

        fn capabilities(&self) -> crate::infrastructure::BackendCapabilities {
            crate::infrastructure::BackendCapabilities {
                max_concurrency: 1,
                supports_shot_distribution: false,
            }
        }

        fn default_planner(&self) -> Arc<dyn crate::infrastructure::Planner> {
            Arc::new(crate::infrastructure::ShotDistributingPlanner::new(2))
        }

        fn close(&self) {
            self.closes.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_scheduler_that_cannot_be_built_closes_its_backend() {
        // What `run_rows` and `fit` call, and `BlockRunner::sampled` through it:
        // each failure closes the backend exactly once and returns the error.
        let closes = Arc::new(AtomicUsize::new(0));
        let backend = || {
            Arc::new(UnpairableBackend {
                closes: Arc::clone(&closes),
            })
        };
        match scheduler_over(backend(), run_params(1, 8)) {
            Err(InfrastructureError::IncompatiblePlanner(_)) => {}
            Err(other) => panic!("expected IncompatiblePlanner, got {other:?}"),
            Ok(_) => panic!("the pairing must fail"),
        }
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        match BlockRunner::sampled(backend(), run_params(1, 8), 8) {
            Err(InfrastructureError::IncompatiblePlanner(_)) => {}
            Err(other) => panic!("expected IncompatiblePlanner, got {other:?}"),
            Ok(_) => panic!("the pairing must fail"),
        }
        assert_eq!(closes.load(Ordering::SeqCst), 2);
    }

    /// Wraps a primitive: counts the blocks it builds, and cancels `token`
    /// once the first block is read, or fails building the second.
    struct Interfering<P> {
        inner: P,
        token: CancelToken,
        fail_second: bool,
        built: std::cell::Cell<usize>,
    }

    impl<P: Primitive> Primitive for Interfering<P> {
        fn num_samples(&self) -> usize {
            self.inner.num_samples()
        }

        fn circuits_per_sample(&self) -> usize {
            self.inner.circuits_per_sample()
        }

        fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError> {
            self.built.set(self.built.get() + 1);
            if self.fail_second && self.built.get() == 2 {
                return Err(QmlError::EmptyCounts.into());
            }
            self.inner.circuits(samples)
        }

        fn read(
            &mut self,
            samples: Range<usize>,
            results: BlockResults,
        ) -> Result<(), PrimitiveError> {
            self.inner.read(samples, results)?;
            self.token.cancel();
            Ok(())
        }
    }

    #[test]
    fn a_cancellation_between_blocks_stops_before_the_next_and_still_closes() {
        let counters = Counters::new();
        let token = CancelToken::default();
        let mut run = Interfering {
            inner: jacobian_run(JacobianWrt::Params),
            token: token.clone(),
            fail_second: false,
            built: std::cell::Cell::new(0),
        };
        let err = run_in_blocks(&mut run, || counters.create(), |_| 2, &token).unwrap_err();
        assert!(
            matches!(
                err,
                PrimitiveError::Infrastructure(InfrastructureError::Cancelled)
            ),
            "{err:?}"
        );
        // The first block ran; the second was never built nor run.
        assert_eq!(run.built.get(), 1);
        assert_eq!(counters.get(), (1, 1, 1));
        // Only the first block's rows were read.
        let per_row = run.inner.plan.num_observables() * run.inner.plan.num_params();
        assert_eq!(run.inner.by_params.len(), 2 * per_row);
    }

    #[test]
    fn an_error_in_a_later_block_still_closes_the_backend() {
        let counters = Counters::new();
        let mut run = Interfering {
            inner: expval_run(),
            token: CancelToken::default(),
            fail_second: true,
            built: std::cell::Cell::new(0),
        };
        // Its own token: the one it cancels is not the run's.
        let err = run_in_blocks(
            &mut run,
            || counters.create(),
            |_| 2,
            &CancelToken::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, PrimitiveError::Qml(QmlError::EmptyCounts)),
            "{err:?}"
        );
        assert_eq!(counters.get(), (1, 1, 1));
    }

    #[test]
    fn a_backend_that_cannot_be_created_closes_nothing() {
        let mut run = expval_run();
        let err = run_in_blocks(
            &mut run,
            || {
                Err(InfrastructureError::Backend(
                    crate::infrastructure::BackendError::UnknownInfrastructure {
                        name: "nowhere".to_string(),
                    },
                ))
            },
            |_| 2,
            &CancelToken::default(),
        )
        .unwrap_err();
        assert!(matches!(err, PrimitiveError::Infrastructure(_)), "{err:?}");
        assert!(run.values.is_empty());
    }

    #[test]
    fn an_invalid_concrete_circuit_is_an_error_not_a_panic() {
        use polypus_circuit::GateInstruction;

        // A gate after the terminal measurement breaks C-4.
        let invalid = ConcreteCircuit::new_unchecked(
            1,
            vec![GateInstruction::MeasureAll, GateInstruction::H(0)],
        );
        assert!(matches!(
            parameterized_from_concrete(invalid),
            Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
        ));
    }
}
