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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use numpy::ndarray::{Array2, Array3, ShapeError};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{
    PyFileNotFoundError, PyOSError, PyOverflowError, PyPermissionError, PyRuntimeError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyModule, PyTuple, PyType};

use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};
use polypus_qml::{
    AngleEncoder, CompiledModel, ConvBlock, ConvLayer, Decision, Entanglement, Entangler,
    ExactState, ExpectationJacobian, FeatureMap, FeatureStates, FidelityKernel,
    HardwareEfficientAnsatz, IqpEncoder, JacobianWrt, KeepRule, Layer, Loss, Observable, Pairing,
    Pauli, PauliString, PoolBlock, PoolLayer, QmlError, QmlProblem, QuantumModel, Readout,
    RotationAxis, SimplifiedTwoDesignAnsatz, StronglyEntanglingAnsatz, ValidationError,
};

use super::backend_config::{
    config_from_strings, validate_cunqa_allocation, validate_id, validate_shots_and_qpus,
};
use super::callback::validate_callback;
use super::circuit::Circuit;
use super::extract::{extract_labels, RealReader};
use super::interrupt::interruptible_token;
use super::optimizer::{
    is_rotosolve, method_from_pyclass, training_observer, validate_final_shots, validate_method,
};
use super::options::OptionsArg;
use super::results::{counts_to_pylist, finish_optimization, log_training_start, RunResult};
use super::seeds::{
    check_seed_for_backend, method_seed, resolve_optimizer_seed, seed_or_entropy, unique_id,
};
use crate::evaluation::{
    candidate_window_size, Label, MinibatchConfig, OracleErrorSlot, QmlOracleFactory, QmlSource,
};
use crate::exceptions::configuration_error;
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{
    AdjointTask, BackendError, BoundCircuit, CancelToken, Counts, ExecutionConfig, Infrastructure,
    InfrastructureError, NativeStatevectorBackend, OptLevel, QuantumBackend, RunParams,
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

/// An integer (or an `__index__` one) that does not fit in a `usize`, as
/// [`extract_count`] finds it: the integer (what `__index__` returns, for an
/// object that is not an `int`) as Python prints it, and its sign, which picks
/// the message.
struct OutOfRangeCount {
    shown: String,
    negative: bool,
}

impl OutOfRangeCount {
    /// Why a count cannot be this value, the text of [`CountArg`]'s
    /// `ValueError`.
    fn message(&self) -> String {
        let shown = &self.shown;
        if self.negative {
            format!("expected a non-negative integer, got {shown}")
        } else {
            format!("expected an integer of at most {}, got {shown}", usize::MAX)
        }
    }
}

/// `ob` as a `usize`, the conversion every count-like argument of the native
/// API shares (contract C-11): the count when it fits, an [`OutOfRangeCount`]
/// for an integer outside `usize`, which each caller refuses in its own words
/// instead of PyO3's `OverflowError`, and PyO3's own error for anything else
/// (a `TypeError` for another type).
///
/// A `bool` is not a count, although Python makes it an `int`: it is a
/// `TypeError`, as `final_shots` treats one. An object with `__index__` (a
/// NumPy integer, say) is the integer it returns: an out-of-range one is
/// described by that integer, never compared or printed as the object itself,
/// which may support neither.
fn extract_count(ob: &Bound<'_, PyAny>) -> PyResult<Result<usize, OutOfRangeCount>> {
    let py = ob.py();
    if ob.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "expected an integer, got a bool ({})",
            ob.str()?
        )));
    }
    match ob.extract::<usize>() {
        Ok(count) => Ok(Ok(count)),
        Err(err) if err.is_instance_of::<PyOverflowError>(py) => {
            let index = PyModule::import(py, "operator")?.call_method1("index", (ob,))?;
            Ok(Err(OutOfRangeCount {
                shown: index.str()?.to_string(),
                negative: index.lt(0)?,
            }))
        }
        Err(err) => Err(err),
    }
}

/// A count argument, such as `reps` or `num_qubits`: a Python `int` that fits
/// in a `usize`. A negative or oversized one is a `ValueError` (contract
/// C-11), not the `OverflowError` PyO3's own conversion raises; a `bool` or
/// any other type is a `TypeError`. On Python 3.11 and later, PyO3 adds a
/// note naming the argument to whatever this raises (it shows in the
/// traceback, not in `str(e)`).
///
/// A local newtype, like [`AngleArg`](super::circuit::AngleArg), because
/// `FromPyObject` and `usize` are both foreign to this crate.
struct CountArg(usize);

impl<'a, 'py> FromPyObject<'a, 'py> for CountArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match extract_count(&ob)? {
            Ok(count) => Ok(CountArg(count)),
            Err(out_of_range) => Err(PyValueError::new_err(out_of_range.message())),
        }
    }
}

/// `Model.fit`'s `batch_size`: an integer that fits in a `usize`, or one that
/// does not (negative or oversized), kept so that `fit` refuses it where it
/// refuses `0` or a size as large as the training set, with the same
/// `polypus.ConfigurationError` (contract C-11) rather than PyO3's
/// `OverflowError`. A `bool` or any other type is a `TypeError`.
enum BatchSizeArg {
    Size(usize),
    OutOfRange(OutOfRangeCount),
}

impl<'a, 'py> FromPyObject<'a, 'py> for BatchSizeArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(match extract_count(&ob)? {
            Ok(size) => BatchSizeArg::Size(size),
            Err(out_of_range) => BatchSizeArg::OutOfRange(out_of_range),
        })
    }
}

impl std::fmt::Display for BatchSizeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchSizeArg::Size(size) => write!(f, "{size}"),
            BatchSizeArg::OutOfRange(out_of_range) => f.write_str(&out_of_range.shown),
        }
    }
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

/// `polypus.qml.Z(position)`: Pauli `Z` on one logical qubit. The position is a
/// count argument: a negative or oversized one is a `ValueError` (C-11).
#[pyfunction(name = "Z")]
fn pauli_z(position: CountArg) -> PauliTerm {
    PauliTerm::single("z", position.0)
}

/// `polypus.qml.X(position)`: Pauli `X` on one logical qubit. A readout may mix
/// bases only if it still fits one measurement basis, which `train` checks.
#[pyfunction(name = "X")]
fn pauli_x(position: CountArg) -> PauliTerm {
    PauliTerm::single("x", position.0)
}

/// `polypus.qml.Y(position)`: Pauli `Y` on one logical qubit, with the same
/// single-basis rule as [`pauli_x`].
#[pyfunction(name = "Y")]
fn pauli_y(position: CountArg) -> PauliTerm {
    PauliTerm::single("y", position.0)
}

/// A weighted sum of Pauli products, `O = Σ cᵢ·Pᵢ`, for a readout that needs
/// coefficients or several terms (a cost Hamiltonian, say). Built from
/// `[(coefficient, term), …]`, each term a [`PauliTerm`] or a
/// `[(pauli, position), …]` list. A non-finite coefficient, an unknown Pauli, a
/// position repeated within a term or a negative or oversized position is a
/// `ValueError`.
#[pyclass(module = "polypus.qml", name = "Observable", frozen)]
pub struct PyObservable {
    inner: Observable,
}

#[pymethods]
impl PyObservable {
    #[new]
    fn new(terms: Vec<(f64, Bound<'_, PyAny>)>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(terms.len());
        for (at, (coefficient, term)) in terms.iter().enumerate() {
            let factors = term_factors(term, ("term", at))?;
            parsed.push((*coefficient, parse_pauli_string(&factors)?));
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
///
/// A list is read factor by factor, in order, and the first bad factor decides
/// the error: one that is not a `(str, int)` pair is the `TypeError` naming the
/// forms, and a position that is a count out of range (negative, or larger
/// than a `usize`) is a `ValueError`, as `Z(-1)` is (contract C-11), naming the
/// factor and `whose`: the term of an `Observable` or the observable of a
/// `Readout` the list is, as `("term", 1)` or `("observable", 1)`.
/// An unknown Pauli or a repeated position is found later, by
/// [`parse_pauli_string`], once every factor has been read. Any other exception
/// (one raised inside a custom `__index__`, say) propagates unchanged.
fn term_factors(term: &Bound<'_, PyAny>, whose: (&str, usize)) -> PyResult<Vec<(String, usize)>> {
    if let Ok(term) = term.cast::<PauliTerm>() {
        return Ok(term.get().factors.clone());
    }
    let not_a_term = || {
        PyTypeError::new_err(
            "a Pauli term must be a polypus.qml.PauliTerm (e.g. Z(0) @ Z(1)) or a list of \
             (pauli, position) factors (e.g. [(\"z\", 0), (\"z\", 1)])",
        )
    };
    let py = term.py();
    let items = term
        .extract::<Vec<Bound<'_, PyAny>>>()
        .map_err(|_| not_a_term())?;
    let mut factors = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let (pauli, position) = item
            .extract::<(String, Bound<'_, PyAny>)>()
            .map_err(|_| not_a_term())?;
        match extract_count(&position) {
            Ok(Ok(position)) => factors.push((pauli, position)),
            Ok(Err(out_of_range)) => {
                let (kind, at) = whose;
                return Err(PyValueError::new_err(format!(
                    "the position of factor {index} ({pauli:?}) of {kind} {at} is out of \
                     range: {}",
                    out_of_range.message()
                )));
            }
            Err(err) if err.is_instance_of::<PyTypeError>(py) => return Err(not_a_term()),
            Err(err) => return Err(err),
        }
    }
    Ok(factors)
}

/// Observable `at` of a readout: an [`Observable`](PyObservable) as it is, or
/// a coefficient-free term with coefficient `1.0`. A term that is not one of
/// the three forms is the `TypeError` naming them; any other error of
/// [`term_factors`] (an out-of-range position's `ValueError`) is its own.
fn extract_observable(observable: &Bound<'_, PyAny>, at: usize) -> PyResult<Observable> {
    if let Ok(weighted) = observable.cast::<PyObservable>() {
        return Ok(weighted.get().inner.clone());
    }
    let factors = term_factors(observable, ("observable", at)).map_err(|err| {
        if err.is_instance_of::<PyTypeError>(observable.py()) {
            PyTypeError::new_err(
                "each observable must be a polypus.qml.PauliTerm (e.g. Z(0) @ Z(1)), a \
                 polypus.qml.Observable, or a list of (pauli, position) factors",
            )
        } else {
            err
        }
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

/// `num_qubits` must be at least 1 and at most
/// [`MAX_MODEL_QUBITS`](polypus_qml::MAX_MODEL_QUBITS), which `compile` would
/// refuse anyway, but only once a layer is added; shared by `Encoder` and
/// `Ansatz`.
fn check_num_qubits(class: &str, num_qubits: usize) -> PyResult<()> {
    if num_qubits == 0 {
        return Err(PyValueError::new_err(format!(
            "{class} needs at least one qubit, got num_qubits=0"
        )));
    }
    let max = polypus_qml::MAX_MODEL_QUBITS;
    if num_qubits > max {
        return Err(PyValueError::new_err(format!(
            "{class} takes at most {max} qubits, got num_qubits={num_qubits}"
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
    fn new(num_qubits: CountArg) -> PyResult<Self> {
        let CountArg(num_qubits) = num_qubits;
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
    fn new(num_qubits: CountArg) -> PyResult<Self> {
        let CountArg(num_qubits) = num_qubits;
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
        reps: CountArg,
        rotations: Option<Vec<String>>,
        entangler: &str,
        entanglement: &str,
        final_rotation_layer: bool,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let CountArg(reps) = reps;
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
    fn real_amplitudes(
        mut slf: PyRefMut<'_, Self>,
        reps: CountArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let CountArg(reps) = reps;
        let layer = Layer::HardwareEfficient(HardwareEfficientAnsatz::real_amplitudes(reps));
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append PennyLane's `StronglyEntanglingLayers`: `reps` layers, each a
    /// general rotation `Rot(φ, θ, ω)`, applied as `Rz(φ)`, then `Ry(θ)`, then
    /// `Rz(ω)`, on every active qubit (three parameters per qubit) followed by
    /// an `entangler` (`"cx"`, the default, or `"cz"`) from every position `i`
    /// to `(i + r) mod n`. `ranges` gives each layer's `r`, one per layer and
    /// each in `1..=n-1` for the `n` qubits active at this point; `None` is
    /// PennyLane's default, `r = (layer mod (n-1)) + 1`. A single active qubit
    /// gets no entanglers.
    #[pyo3(signature = (reps, ranges=None, entangler="cx"))]
    fn strongly_entangling<'py>(
        mut slf: PyRefMut<'py, Self>,
        reps: CountArg,
        ranges: Option<Vec<CountArg>>,
        entangler: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = Layer::StronglyEntangling(StronglyEntanglingAnsatz {
            reps: reps.0,
            ranges: ranges.map(|ranges| ranges.into_iter().map(|range| range.0).collect()),
            entangler: parse_entangler(entangler)?,
        });
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append PennyLane's `SimplifiedTwoDesign`: an `Ry` on every active
    /// qubit, then `reps` blocks that, over the even pairs `(0,1),(2,3),…` and
    /// then the odd pairs `(1,2),(3,4),…` of the `n` qubits active at this
    /// point, apply `CZ` followed by an `Ry` on each qubit of the pair. That is
    /// `n + 2·(n−1)·reps` parameters; a single active qubit gets only the
    /// initial `Ry`, whatever `reps`.
    fn simplified_two_design(
        mut slf: PyRefMut<'_, Self>,
        reps: CountArg,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let CountArg(reps) = reps;
        let layer = Layer::SimplifiedTwoDesign(SimplifiedTwoDesignAnsatz::new(reps));
        slf.stack.push(layer)?;
        Ok(slf)
    }

    /// Append a QCNN convolution: one `block` (`"basic"`, 4 parameters, or
    /// `"cartan"`, 3) shared by every pair of active qubits that `pairing`
    /// selects (`"even_pairs"`/`"odd_pairs"`/`"alternating"`, default
    /// `"alternating"`). The pairing must select at least one pair of the
    /// qubits active where the layer sits: `"odd_pairs"` needs at least three,
    /// the others at least two; otherwise this raises `ValueError` and leaves
    /// the `Ansatz` as it was.
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
            .enumerate()
            .map(|(at, observable)| extract_observable(observable, at))
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
/// `Dataset(x, y=None)` validates the rows (non-empty, at least one feature,
/// equal length, finite) and, if given, one finite label per row. A complex
/// value in `x` or `y` (Python's or NumPy's, whatever its imaginary part) is a
/// `TypeError` naming its position, never a value cut to its real part.
/// Without `y` the dataset is *unlabeled* (plan F4): it trains only with
/// `loss="expectation"`, and the mark survives a split. Labels are `float`s;
/// class indices for `"categorical_cross_entropy"` are whole numbers.
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
        let reals = RealReader::new(x.py())?;
        let rows = reals.rows(&x, "x", |e| e)?;
        let (labels, labeled) = match y {
            Some(y) => (reals.values(&y, "y", |e| e)?, true),
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
        Layer::HardwareEfficient(_)
        | Layer::StronglyEntangling(_)
        | Layer::SimplifiedTwoDesign(_)
        | Layer::Conv(_)
        | Layer::Pool(_) => false,
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
/// `TrainResult`. It pickles (so `copy.deepcopy` works) as its JSON spec, the
/// one `TrainedModel.save` writes, rebuilt through the checks `qml.load` runs.
#[pyclass(module = "polypus.qml", name = "Model", frozen)]
pub struct Model {
    /// Encoder layers, then ansatz layers, with the readout.
    spec: QuantumModel,
    num_params: usize,
    num_observables: usize,
}

impl Model {
    /// The model over `spec`, with the counts its compilation without data
    /// gives. Zero features trips no feature-count check and changes no
    /// parameter count (see `check_layers`), so this raises exactly the
    /// structural errors and returns the real counts.
    fn checked(spec: QuantumModel) -> PyResult<Self> {
        let compiled = spec.clone().compile(0).map_err(validation_to_py_err)?;
        Ok(Model {
            num_params: compiled.num_params(),
            num_observables: compiled.num_observables(),
            spec,
        })
    }
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
        Self::checked(spec)
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

    /// The readout's observables: the columns of `qml.expval` (one per class
    /// for an `"argmax"` readout).
    #[getter]
    fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Pickle as `Model._from_spec(json)`, the JSON of the spec.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let json = serde_json::to_string(&slf.get().spec)
            .map_err(|e| PyValueError::new_err(format!("cannot serialize the model: {e}")))?;
        Ok((slf.get_type().getattr("_from_spec")?, (json,)))
    }

    /// Unpickle: rebuild the model from the JSON of its spec, checked as
    /// `qml.load` checks a saved one, so a corrupt or edited spec is a
    /// `ValueError` and never an inconsistent model. Parsing JSON runs no code.
    #[classmethod]
    fn _from_spec(_cls: &Bound<'_, PyType>, spec: &str) -> PyResult<Self> {
        let spec = serde_json::from_str::<QuantumModel>(spec).map_err(|e| {
            PyValueError::new_err(format!("not a valid polypus.qml.Model spec: {e}"))
        })?;
        Self::checked(spec)
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
    /// not finite, is a `ValueError`; a complex value (Python's or NumPy's,
    /// whatever its imaginary part) is a `TypeError` naming its position.
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
    ///   optimizer, the shots and the minibatches (C-7); on Aer and CUNQA it
    ///   must be below `2**63` (`polypus.ConfigurationError` otherwise, #393).
    ///   `id` is a prefix to which a UUID is appended (C-9).
    ///
    /// Every check runs before a backend exists and raises
    /// `polypus.ConfigurationError` (a `ValueError`), except a value of the
    /// wrong type (a value of `x` or `y` that is not a real number, a complex
    /// one included whatever its imaginary part; a `final_shots` that is not an
    /// integer; a `callback` that is neither callable nor `None`), a
    /// `TypeError`. The fitness is the negated mean loss over the samples,
    /// which the optimizers maximise.
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
        batch_size: Option<BatchSizeArg>,
        initial_point: Option<Vec<f64>>,
        final_shots: Option<Bound<'py, PyAny>>,
        callback: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let start = Instant::now();
        let py = x.py();
        // Read Python's log levels before the GIL is released below (`python_log`).
        super::python_log::refresh_levels(py);
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
        if exact && !(infrastructure == "local" && super::targets::is_native_device(backend)) {
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
        // A batch as large as the training set is the run without minibatches;
        // a negative or oversized integer is refused here too, with the same
        // message (C-11).
        let minibatch = match batch_size {
            Some(BatchSizeArg::Size(size)) if (1..num_samples).contains(&size) => {
                Some(MinibatchConfig::new(size, effective_seed))
            }
            Some(size) => {
                return Err(configuration_error(format!(
                    "batch_size must satisfy 1 <= batch_size < {num_samples} (the number of \
                     training samples), got {size}"
                )));
            }
            None => None,
        };
        let exact = exact.then(|| Arc::new(NativeStatevectorBackend::new(effective_seed)));

        // Native circuits never ask for fusion here: `None` keeps the backend's
        // default.
        let backend_config = config_from_strings(
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
        check_seed_for_backend(Some(effective_seed), &backend_config)?;
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
            trace: None,
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
/// label checks as `qml.train`'s `y_train`. A complex value in the rows is a
/// `TypeError` naming its row and column (see [`RealReader`]).
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
    let rows = RealReader::new(x.py())?.rows(x, "x", |e| e)?;
    let labels = y
        .map(|y| {
            extract_labels(y, "y").map(|labels| {
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

/// The execution arguments `TrainedModel.predict` and `run`, and `qml.expval`,
/// `qml.jacobian` and `qml.vjp`, share.
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
        if self.exact
            && !(self.infrastructure == "local" && super::targets::is_native_device(backend))
        {
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
        let backend_config = config_from_strings(
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
        check_seed_for_backend(seed, &backend_config)?;
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
    /// `predict(exact=True)`'s decisions, each row read straight from its
    /// statevector ([`exact_predictions`], issues #645, #628).
    Predictions(Vec<f64>),
}

/// What [`TrainedModel::run_rows`] keeps of an exact run: the maps
/// (`run(exact=True)` returns them) or only each row's decision
/// (`predict(exact=True)`).
#[derive(Clone, Copy)]
enum RowRead {
    Results,
    Predictions,
}

/// Why an exact inference run stopped: the backend, or reading a row.
#[derive(Debug)]
enum ExactRowError {
    Backend(BackendError),
    Qml(QmlError),
}

impl From<BackendError> for ExactRowError {
    fn from(e: BackendError) -> Self {
        ExactRowError::Backend(e)
    }
}

impl ExactRowError {
    fn into_py_err(self) -> PyErr {
        match self {
            ExactRowError::Backend(e) => crate::exceptions::backend_error_to_pyerr(e),
            ExactRowError::Qml(e) => qml_to_py_err(e),
        }
    }
}

/// `compiled`'s exact decision for each of `circuits`, in order, each row read
/// straight from its statevector (issues #645, #628): no map of its
/// probabilities is built, and only the decision is kept. The same values, bit
/// for bit, as `predict_from_probabilities` over `run_circuits_exact`'s maps.
fn exact_predictions(
    native: &NativeStatevectorBackend,
    params: &RunParams,
    compiled: &CompiledModel,
    circuits: &[BoundCircuit],
) -> Result<Vec<f64>, ExactRowError> {
    native.run_circuits_exact_on_states(circuits, params, |amplitudes, width| {
        let state = ExactState::new(amplitudes.len(), width, |s| amplitudes[s].norm_sqr());
        compiled
            .predict_from_state(&state)
            .map_err(ExactRowError::Qml)
    })
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
    /// native simulator, kept as they are (`read` = [`RowRead::Results`]) or
    /// read into each row's decision as soon as it is computed
    /// ([`RowRead::Predictions`], issue #645). Every check runs before a
    /// backend exists and raises `polypus.ConfigurationError`.
    fn run_rows(
        &self,
        x: &Bound<'_, PyAny>,
        exec: Execution<'_, '_>,
        read: RowRead,
    ) -> PyResult<RowsRun> {
        let py = x.py();
        // Read Python's log levels before the GIL is released below (`python_log`).
        super::python_log::refresh_levels(py);
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
            match read {
                RowRead::Results => py
                    .detach(|| native.run_circuits_exact(&circuits, &params))
                    .map(RowResults::Exact)
                    .map_err(crate::exceptions::backend_error_to_pyerr)?,
                RowRead::Predictions => py
                    .detach(|| exact_predictions(&native, &params, &compiled, &circuits))
                    .map(RowResults::Predictions)
                    .map_err(ExactRowError::into_py_err)?,
            }
        } else {
            // One run, GIL released, the rows in waves of the backend's
            // concurrency; Ctrl+C is honoured between waves.
            py.detach(move || -> Result<Vec<Counts>, InfrastructureError> {
                let backend = Infrastructure::create_backend(&config)
                    .map_err(InfrastructureError::Backend)?;
                let scheduler = scheduler_over(backend, config.run_params())?;
                let out = scheduler.run_cancellable(
                    RunCircuitFlow {
                        circuits,
                        shots,
                        trace: None,
                    },
                    &interruptible_token(),
                );
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
    fn model(&self) -> PyResult<Model> {
        // The counts come from compiling the spec without data (the
        // `TrainedModel` does not keep `num_observables`); that spec was checked
        // when the `TrainedModel` was built, so the error cannot happen.
        Model::checked(self.inner.spec().clone())
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
            RowRead::Predictions,
        )?;
        // `RowRead::Predictions` reads exact rows into `Predictions`, so an
        // `Exact` result does not reach here; read it the same way regardless.
        let values = match run.results {
            RowResults::Counts(counts) => counts
                .iter()
                .map(|counts| run.compiled.predict_from_counts(counts))
                .collect::<Result<Vec<_>, _>>(),
            RowResults::Exact(probabilities) => probabilities
                .iter()
                .map(|p| run.compiled.predict_from_probabilities(p))
                .collect::<Result<Vec<_>, _>>(),
            RowResults::Predictions(values) => Ok(values),
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
            RowRead::Results,
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
                // Each row's map is consumed as its dict is built, so the Rust
                // maps and their Python copies are not all alive at once: the
                // peak is about the larger of the two, not their sum (#628).
                // The copies themselves are not counted against the budget.
                let list = PyList::empty(py);
                for row in probabilities {
                    let dict = PyDict::new(py);
                    let mut entries: Vec<(String, f64)> = row.into_iter().collect();
                    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    for (key, p) in entries {
                        dict.set_item(key, p)?;
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
            RowResults::Predictions(_) => Err(PyRuntimeError::new_err(
                "internal error: run() read its rows as decisions",
            )),
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
/// goes through the checks a row does, through the same `Dataset`. A complex
/// value in `x` is the `TypeError` naming its position that a row raises, not
/// the generic "one sample" one, which covers what is not a flat sequence of
/// real numbers.
fn sample_circuit(
    spec: &QuantumModel,
    x: &Bound<'_, PyAny>,
    params: Option<&[f64]>,
) -> PyResult<Circuit> {
    let x = RealReader::new(x.py())?.values(x, "x", |e: PyErr| {
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
// qml.expval, qml.jacobian and qml.vjp: the loss-free primitives
// ─────────────────────────────────────────────────────────────────────────────

/// `wrt` of `qml.jacobian` and `qml.vjp`, a closed set (C-11, "Options").
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
    /// On the exact path, each circuit's readout expectations, read straight
    /// from its statevector ([`Primitive::exact_readout`], issues #645, #628).
    Expectations(Vec<Vec<f64>>),
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

    /// Run `circuits`. On the exact path with a `readout`, each circuit is
    /// read into that model's expectations straight from its statevector, with
    /// no map built, and only those are kept (issues #645, #628); without one,
    /// the maps are returned.
    fn run(
        &self,
        circuits: Vec<BoundCircuit>,
        cancel: &CancelToken,
        readout: Option<&CompiledModel>,
    ) -> Result<BlockResults, PrimitiveError> {
        match self {
            BlockRunner::Sampled {
                scheduler, shots, ..
            } => scheduler
                .run_cancellable(
                    RunCircuitFlow {
                        circuits,
                        shots: *shots,
                        trace: None,
                    },
                    cancel,
                )
                .map(BlockResults::Counts)
                .map_err(PrimitiveError::Infrastructure),
            BlockRunner::Exact { native, params } => match readout {
                Some(model) => native
                    .run_circuits_exact_on_states(&circuits, params, |amplitudes, width| {
                        let state =
                            ExactState::new(amplitudes.len(), width, |s| amplitudes[s].norm_sqr());
                        model
                            .expectations_from_state(&state)
                            .map_err(PrimitiveError::Qml)
                    })
                    .map(BlockResults::Expectations),
                None => native
                    .run_circuits_exact(&circuits, params)
                    .map(BlockResults::Exact)
                    .map_err(PrimitiveError::from),
            },
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

impl From<BackendError> for PrimitiveError {
    fn from(e: BackendError) -> Self {
        PrimitiveError::Infrastructure(InfrastructureError::Backend(e))
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

    /// Adjoint derivatives that do not answer the runs they were asked for,
    /// one row per run: a bookkeeping bug, never the caller's.
    fn derivative_rows_mismatch(expected: usize, got: usize) -> Self {
        PrimitiveError::Internal(format!(
            "the adjoint runs gave {got} row(s) of derivatives for {expected} run(s)"
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
    /// The model whose readout expectations are all this primitive reads from
    /// a circuit on the exact path: each circuit is then read straight from
    /// its statevector, with no map built ([`BlockResults::Expectations`],
    /// issues #645, #628). `None` (the default) hands [`read`](Self::read) the
    /// maps.
    fn exact_readout(&self) -> Option<&CompiledModel> {
        None
    }
    /// Run the block `samples` on the exact native backend by another route
    /// than [`circuits`](Self::circuits), reading its results as it goes, or
    /// `None` to run the circuits (the default). The Jacobian's route is the
    /// adjoint (#503); `cancel` must be honoured as between blocks.
    fn run_exact_block(
        &mut self,
        _native: &NativeStatevectorBackend,
        _samples: Range<usize>,
        _cancel: &CancelToken,
    ) -> Option<Result<(), PrimitiveError>> {
        None
    }
    /// Run **every** sample on the exact native backend at once, by a route
    /// that bounds its own memory and honours `cancel` inside (the kernel's
    /// inner products, which would otherwise simulate each state again for
    /// every block), or `None` to run in blocks (the default). Tried before
    /// the first block.
    fn run_exact_all(
        &mut self,
        _native: &NativeStatevectorBackend,
        _cancel: &CancelToken,
    ) -> Option<Result<(), PrimitiveError>> {
        None
    }
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
    if let BlockRunner::Exact { native, .. } = runner {
        if num_samples > 0 {
            check_cancel(cancel)?;
            if let Some(outcome) = primitive.run_exact_all(native, cancel) {
                return outcome;
            }
        }
    }
    let mut start = 0;
    while start < num_samples {
        check_cancel(cancel)?;
        let end = start.saturating_add(block).min(num_samples);
        if let BlockRunner::Exact { native, .. } = runner {
            if let Some(outcome) = primitive.run_exact_block(native, start..end, cancel) {
                outcome?;
                start = end;
                continue;
            }
        }
        let circuits = primitive
            .circuits(start..end)?
            .into_iter()
            .map(BoundCircuit::Native)
            .collect();
        let results = runner.run(circuits, cancel, primitive.exact_readout())?;
        primitive.read(start..end, results)?;
        start = end;
    }
    Ok(())
}

/// A cancellation or a Ctrl+C, as the planner checks them between waves.
fn check_cancel(cancel: &CancelToken) -> Result<(), PrimitiveError> {
    if cancel.is_cancelled() {
        return Err(PrimitiveError::Infrastructure(
            InfrastructureError::Cancelled,
        ));
    }
    cancel
        .poll_interrupt()
        .map_err(PrimitiveError::Infrastructure)
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
            BlockResults::Expectations(readings) => readings.len(),
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
            BlockResults::Expectations(readings) => {
                let expected = self.compiled.num_observables();
                for reading in readings {
                    if reading.len() != expected {
                        return Err(QmlError::CountsLengthMismatch {
                            expected,
                            got: reading.len(),
                        }
                        .into());
                    }
                    self.values.extend(reading);
                }
            }
        }
        Ok(())
    }

    fn exact_readout(&self) -> Option<&CompiledModel> {
        Some(&self.compiled)
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
            BlockResults::Expectations(readings) => block.jacobian_from_expectations(readings)?,
        };
        self.keep(jacobian);
        Ok(())
    }

    fn exact_readout(&self) -> Option<&CompiledModel> {
        Some(self.plan.model())
    }

    /// The exact Jacobian by adjoint differentiation (#503): one run per sample
    /// and observable, two statevectors each, instead of `2 · shifts` circuits
    /// per sample. When the memory budget holds one statevector of the width
    /// but not two, `None`: the shifted circuits run instead (the backend logs
    /// it) — unless the model has none, being one only the adjoint
    /// differentiates (#561): then the memory's `NoExactGradient`, before
    /// anything runs. A Ctrl+C or a cancellation stops a run in flight
    /// ([`watched`]).
    fn run_exact_block(
        &mut self,
        native: &NativeStatevectorBackend,
        samples: Range<usize>,
        cancel: &CancelToken,
    ) -> Option<Result<(), PrimitiveError>> {
        self.run_exact_block_with(native, samples, cancel, |width| native.adjoint_fits(width))
    }
}

impl JacobianRun {
    /// [`run_exact_block`](Primitive::run_exact_block) with `adjoint_fits`
    /// answering whether the adjoint's two statevectors of a width fit the
    /// memory budget: the backend's
    /// [`adjoint_fits`](NativeStatevectorBackend::adjoint_fits), or a test's
    /// answer (the budget is the process's, which a test must not set).
    fn run_exact_block_with(
        &mut self,
        native: &NativeStatevectorBackend,
        samples: Range<usize>,
        cancel: &CancelToken,
        adjoint_fits: impl FnOnce(usize) -> Result<bool, BackendError>,
    ) -> Option<Result<(), PrimitiveError>> {
        let Some(block) = self.plan.block(samples.clone()) else {
            return Some(Err(PrimitiveError::block_out_of_range(
                &samples,
                self.plan.num_samples(),
            )));
        };
        let runs = match block.adjoint_runs(&self.params) {
            Ok(runs) => runs,
            Err(e) => return Some(Err(e.into())),
        };
        let derivatives = match run_exact_adjoint(native, &self.plan, &runs, cancel, adjoint_fits)?
        {
            Ok(derivatives) => derivatives,
            Err(e) => return Some(Err(e)),
        };
        Some(
            block
                .jacobian_from_adjoint(&derivatives)
                .map(|jacobian| self.keep(jacobian))
                .map_err(PrimitiveError::from),
        )
    }

    /// Append a block's Jacobian, in sample order.
    fn keep(&mut self, jacobian: polypus_qml::Jacobian) {
        let (by_params, by_inputs) = jacobian.into_parts();
        self.by_params.extend(by_params.unwrap_or_default());
        self.by_inputs.extend(by_inputs.unwrap_or_default());
    }
}

/// `runs` on the exact native backend by adjoint differentiation, one row of
/// derivatives per run, or `None` when the memory budget holds one statevector
/// of their width but not two ([`NativeStatevectorBackend::adjoint_fits`],
/// which logs it): the caller then runs the shifted circuits. A model only the
/// adjoint differentiates (#561) has none, so there `plan` answers instead
/// with the memory's `NoExactGradient`, before anything runs
/// ([`ExpectationJacobian::check_shift_fallback`]): the one place both
/// `qml.jacobian` and `qml.vjp` decide it. A Ctrl+C or a cancellation stops
/// the runs in flight ([`watched`]).
///
/// `adjoint_fits` answers whether the adjoint's two statevectors of a width fit
/// the memory budget: the backend's
/// [`adjoint_fits`](NativeStatevectorBackend::adjoint_fits), or a test's
/// answer (the budget is the process's, which a test must not set).
fn run_exact_adjoint(
    native: &NativeStatevectorBackend,
    plan: &ExpectationJacobian,
    runs: &[polypus_qml::AdjointRun],
    cancel: &CancelToken,
    adjoint_fits: impl FnOnce(usize) -> Result<bool, BackendError>,
) -> Option<Result<Vec<Vec<f64>>, PrimitiveError>> {
    let width = runs
        .iter()
        .map(|run| run.circuit.num_qubits())
        .max()
        .unwrap_or(0);
    match adjoint_fits(width) {
        Ok(true) => {}
        // The shifted circuits instead, if the model has them.
        Ok(false) => {
            return plan
                .check_shift_fallback(width)
                .err()
                .map(|e| Err(e.into()))
        }
        Err(e) => return Some(Err(PrimitiveError::Infrastructure(e.into()))),
    }
    let tasks: Vec<AdjointTask<'_, _>> = runs
        .iter()
        .map(|run| AdjointTask {
            circuit: &run.circuit,
            wrt: &run.wrt,
            weight: move |b: usize| run.observable.weight(b),
        })
        .collect();
    let derivatives = watched(cancel, |stop| {
        native.run_adjoint(&tasks, Some(&|| stop.load(Ordering::Relaxed)))
    });
    Some(match derivatives {
        Ok(Ok(derivatives)) => Ok(derivatives),
        Ok(Err(e)) => Err(PrimitiveError::Infrastructure(e.into())),
        Err(e) => Err(e),
    })
}

/// The largest weight bound ([`DiagonalObservable::weight_bound`]) of a
/// combined observable `Σ_o v_o·O_o` that `qml.vjp` hands the native simulator
/// (C-11). Past it, or when it is not finite, the sample takes
/// `qml.jacobian`'s runs and contracts them.
///
/// Why a quarter of `f64::MAX`. In the simulator nothing grows past the bound
/// `W`: `λ = O|ψ⟩` has every amplitude within `W` and `‖λ‖ ≤ W`; each gate of
/// the sweep back is a unitary matrix-vector product, its outputs within the
/// norm of its inputs (Cauchy–Schwarz on each row); and each derivative,
/// `Im ⟨λ|P|ψ⟩` with no further factor, is within `‖λ‖·‖ψ‖ ≤ W`. So the
/// simulator never overflows below the limit. What can still grow is what the
/// crate does with the derivatives afterwards: the sum over the gates that
/// share a parameter, and the chain-rule factors of the inputs, which the
/// contraction also computes, in another order. The factor 4 is room for a
/// few of those, not a guarantee: with more than 4 gates on one parameter, or
/// chain-rule factors past 4, a product near `f64::MAX` can overflow on the
/// combined path where the contraction would not, or the other way round. A
/// cotangent within a few orders of magnitude of `f64::MAX` is not a practical
/// case; the limit is there so that the simulator never sees a weight it
/// refuses.
///
/// [`DiagonalObservable::weight_bound`]: polypus_qml::DiagonalObservable::weight_bound
const MAX_COMBINED_WEIGHT: f64 = f64::MAX / 4.0;

/// `qml.vjp`: the Jacobian's product with a cotangent. Its circuits, and its
/// reading of their results, are `qml.jacobian`'s, the Jacobian of each block
/// contracted with the block's cotangent; exactly, one adjoint run per sample
/// instead (#559).
struct VjpRun {
    plan: ExpectationJacobian,
    params: Vec<f64>,
    /// `(num_samples, num_observables)`, row-major: the cotangent `v`.
    cotangent: Vec<f64>,
    /// `num_params` values (empty unless `wrt` asks for them): the product's
    /// rows read so far, summed in row order, so the sum does not depend on
    /// the blocks.
    by_params: Vec<f64>,
    /// `(num_samples, num_features)`, row-major.
    by_inputs: Vec<f64>,
    /// The adjoint runs handed to the native backend so far.
    adjoint_runs: usize,
}

impl VjpRun {
    fn new(plan: ExpectationJacobian, params: Vec<f64>, cotangent: Vec<f64>) -> Self {
        let by_params = match plan.wrt() {
            JacobianWrt::Params | JacobianWrt::Both => vec![0.0; plan.num_params()],
            JacobianWrt::Inputs => Vec::new(),
        };
        VjpRun {
            plan,
            params,
            cotangent,
            by_params,
            by_inputs: Vec::new(),
            adjoint_runs: 0,
        }
    }

    /// The block `samples` and its rows of the cotangent.
    fn block(
        &self,
        samples: Range<usize>,
    ) -> Result<(polypus_qml::JacobianBlock<'_>, &[f64]), PrimitiveError> {
        let k = self.plan.num_observables();
        let out_of_range = || PrimitiveError::block_out_of_range(&samples, self.plan.num_samples());
        let block = self.plan.block(samples.clone()).ok_or_else(out_of_range)?;
        let cotangent = self
            .cotangent
            .get(samples.start * k..samples.end * k)
            .ok_or_else(out_of_range)?;
        Ok((block, cotangent))
    }

    /// Add a block's product, in sample order: its parameter rows into the
    /// running sum, its input rows after the others.
    fn keep(&mut self, vjp: polypus_qml::Vjp) {
        let num_params = vjp.num_params().max(1);
        let (by_params, by_inputs) = vjp.into_parts();
        for row in by_params.unwrap_or_default().chunks_exact(num_params) {
            for (sum, &value) in self.by_params.iter_mut().zip(row) {
                *sum += value;
            }
        }
        self.by_inputs.extend(by_inputs.unwrap_or_default());
    }

    /// The exact product of the block by adjoint differentiation, one [`Vjp`]
    /// per sample in order, or `None` for the shifted circuits (see
    /// [`run_exact_adjoint`]).
    ///
    /// [`Vjp`]: polypus_qml::Vjp
    fn exact_block(
        &mut self,
        native: &NativeStatevectorBackend,
        samples: Range<usize>,
        cancel: &CancelToken,
    ) -> Option<Result<Vec<polypus_qml::Vjp>, PrimitiveError>> {
        self.exact_block_with(native, samples, cancel, |width| native.adjoint_fits(width))
    }

    /// [`exact_block`](Self::exact_block) with the memory decision of
    /// [`run_exact_adjoint`].
    fn exact_block_with(
        &mut self,
        native: &NativeStatevectorBackend,
        samples: Range<usize>,
        cancel: &CancelToken,
        adjoint_fits: impl FnOnce(usize) -> Result<bool, BackendError>,
    ) -> Option<Result<Vec<polypus_qml::Vjp>, PrimitiveError>> {
        let (runs, combined) = match self.exact_runs(samples.clone()) {
            Ok(found) => found,
            Err(e) => return Some(Err(e)),
        };
        let derivatives = match run_exact_adjoint(native, &self.plan, &runs, cancel, adjoint_fits)?
        {
            Ok(derivatives) => derivatives,
            Err(e) => return Some(Err(e)),
        };
        let vjps = self.read_exact(samples, &combined, &derivatives);
        self.adjoint_runs += runs.len();
        log::debug!(
            "qml vjp: {} adjoint run(s), {} in all",
            runs.len(),
            self.adjoint_runs
        );
        Some(vjps)
    }

    /// The block's adjoint runs, sample after sample, and for each sample
    /// whether it takes the combined observable. The route is **decided per
    /// sample**, so a sample's product does not depend on the others' — nor on
    /// the blocks, which follow the host's cores (C-11): one run, the
    /// cotangent folded into its observable; or, when that observable cannot
    /// be weighed safely — a non-finite value in the sample's cotangent (the
    /// simulator refuses a non-finite weight), or a weight bound past
    /// `MAX_COMBINED_WEIGHT` — the Jacobian's runs, one per observable, to be
    /// contracted: the cotangent then reaches the product as it would through
    /// the contraction of `qml.jacobian`. The bounds come before any circuit
    /// is bound, so each sample's runs are built once. `<=` is false for a
    /// `NaN` bound too.
    fn exact_runs(
        &self,
        samples: Range<usize>,
    ) -> Result<(Vec<polypus_qml::AdjointRun>, Vec<bool>), PrimitiveError> {
        let (block, cotangent) = self.block(samples.clone())?;
        let bounds = block.vjp_weight_bounds(cotangent)?;
        let mut runs = Vec::with_capacity(block.num_vjp_runs());
        let mut combined = Vec::with_capacity(bounds.len());
        for (s, bound) in samples.zip(bounds) {
            let (one, v) = self.block(s..s + 1)?;
            let safe = bound <= MAX_COMBINED_WEIGHT;
            runs.extend(if safe {
                one.vjp_runs(&self.params, v)?
            } else {
                one.adjoint_runs(&self.params)?
            });
            combined.push(safe);
        }
        Ok((runs, combined))
    }

    /// Each sample's product, in order, from the derivatives of
    /// [`exact_runs`](Self::exact_runs)' runs: its one row through the
    /// combined observable, or its `num_observables` rows read as its
    /// Jacobian and contracted with its cotangent. One row of `derivatives`
    /// per run, no more and no fewer, or a bookkeeping error before any is
    /// read: a row too many or too few would shift every sample after it.
    fn read_exact(
        &self,
        samples: Range<usize>,
        combined: &[bool],
        derivatives: &[Vec<f64>],
    ) -> Result<Vec<polypus_qml::Vjp>, PrimitiveError> {
        let k = self.plan.num_observables();
        let expected: usize = combined.iter().map(|&c| if c { 1 } else { k }).sum();
        if derivatives.len() != expected {
            return Err(PrimitiveError::derivative_rows_mismatch(
                expected,
                derivatives.len(),
            ));
        }
        let mut rest = derivatives;
        samples
            .zip(combined)
            .map(|(s, &combined)| {
                let (one, v) = self.block(s..s + 1)?;
                let take = if combined { 1 } else { k };
                let (rows, after) = rest.split_at(take.min(rest.len()));
                rest = after;
                let vjp = if combined {
                    one.vjp_from_adjoint(rows)
                } else {
                    one.jacobian_from_adjoint(rows)
                        .and_then(|jacobian| jacobian.vjp(v))
                };
                Ok(vjp?)
            })
            .collect()
    }
}

impl Primitive for VjpRun {
    fn num_samples(&self) -> usize {
        self.plan.num_samples()
    }

    fn circuits_per_sample(&self) -> usize {
        self.plan.circuits_per_sample()
    }

    fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError> {
        let (block, _) = self.block(samples)?;
        Ok(block.circuits(&self.params)?)
    }

    fn read(&mut self, samples: Range<usize>, results: BlockResults) -> Result<(), PrimitiveError> {
        let (block, cotangent) = self.block(samples)?;
        let jacobian = match &results {
            BlockResults::Counts(counts) => block.jacobian_from_counts(counts)?,
            BlockResults::Exact(probabilities) => {
                block.jacobian_from_probabilities(probabilities)?
            }
            BlockResults::Expectations(readings) => block.jacobian_from_expectations(readings)?,
        };
        let vjp = jacobian.vjp(cotangent)?;
        self.keep(vjp);
        Ok(())
    }

    fn exact_readout(&self) -> Option<&CompiledModel> {
        Some(self.plan.model())
    }

    /// The exact product by adjoint differentiation (#559): one run per sample
    /// whatever the number of observables, two statevectors each. When the
    /// memory budget holds one statevector of the width but not two, `None`:
    /// the shifted circuits run instead, as for the Jacobian.
    fn run_exact_block(
        &mut self,
        native: &NativeStatevectorBackend,
        samples: Range<usize>,
        cancel: &CancelToken,
    ) -> Option<Result<(), PrimitiveError>> {
        let vjps = self.exact_block(native, samples, cancel)?;
        Some(vjps.map(|vjps| vjps.into_iter().for_each(|vjp| self.keep(vjp))))
    }
}

/// `qml.kernel_matrix`: one sample per pair `(i, j)` of the kernel, one
/// compute-uncompute circuit each, read into one value per pair; or, exactly,
/// the whole matrix from the native backend.
struct KernelRun {
    kernel: FidelityKernel,
    /// The `rows × columns` matrix, row-major: the native backend's whole
    /// result ([`run_exact_all`](Primitive::run_exact_all)), or allocated by
    /// the first block read and filled block by block, each pair's value
    /// written at its entry (a symmetric kernel's at its mirror too), so the
    /// pairs' values are never held beside it.
    matrix: Option<Vec<f64>>,
    /// The pairs whose values are in `matrix`.
    filled: usize,
}

impl KernelRun {
    fn new(kernel: FidelityKernel) -> Self {
        KernelRun {
            kernel,
            matrix: None,
            filled: 0,
        }
    }

    /// The `rows × columns` matrix, row-major, once every pair is in it (a
    /// kernel with no pairs has an empty one).
    fn into_matrix(self) -> Result<Vec<f64>, QmlError> {
        if self.filled != self.kernel.num_pairs() {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.kernel.num_pairs(),
                got: self.filled,
            });
        }
        match self.matrix {
            Some(matrix) => Ok(matrix),
            None => self.kernel.zero_matrix(),
        }
    }
}

impl Primitive for KernelRun {
    fn num_samples(&self) -> usize {
        self.kernel.num_pairs()
    }

    fn circuits_per_sample(&self) -> usize {
        1
    }

    fn circuits(&self, samples: Range<usize>) -> Result<Vec<ConcreteCircuit>, PrimitiveError> {
        match self.kernel.circuits(samples.clone()) {
            Some(circuits) => Ok(circuits?),
            None => Err(PrimitiveError::block_out_of_range(
                &samples,
                self.kernel.num_pairs(),
            )),
        }
    }

    fn read(&mut self, samples: Range<usize>, results: BlockResults) -> Result<(), PrimitiveError> {
        let got = match &results {
            BlockResults::Counts(counts) => counts.len(),
            BlockResults::Exact(probabilities) => probabilities.len(),
            BlockResults::Expectations(_) => {
                return Err(PrimitiveError::Internal(
                    "the kernel reads probabilities, not expectations".to_string(),
                ))
            }
        };
        if got != samples.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: samples.len(),
                got,
            }
            .into());
        }
        let values = match results {
            BlockResults::Counts(counts) => counts
                .iter()
                .map(|counts| self.kernel.fidelity_from_counts(counts))
                .collect::<Result<Vec<_>, _>>()?,
            BlockResults::Exact(probabilities) => probabilities
                .iter()
                .map(|probabilities| self.kernel.fidelity_from_probabilities(probabilities))
                .collect::<Result<Vec<_>, _>>()?,
            BlockResults::Expectations(_) => {
                return Err(PrimitiveError::Internal(
                    "the kernel reads probabilities, not expectations".to_string(),
                ))
            }
        };
        let matrix = match &mut self.matrix {
            Some(matrix) => matrix,
            None => self.matrix.insert(self.kernel.zero_matrix()?),
        };
        self.kernel.place(matrix, samples.start, &values)?;
        self.filled += values.len();
        Ok(())
    }

    /// The exact kernel in one call to the native backend, which picks its
    /// route by the memory budget with the matrix counted: inner products of
    /// statevectors, each state simulated once when they all fit and in tiles
    /// otherwise, or, when the budget holds one statevector beside the matrix
    /// but not two, one compute-uncompute circuit per pair (the backend logs
    /// it). Never `None`, so the circuits never run in blocks here. A Ctrl+C
    /// or a cancellation stops it in flight ([`watched`]).
    fn run_exact_all(
        &mut self,
        native: &NativeStatevectorBackend,
        cancel: &CancelToken,
    ) -> Option<Result<(), PrimitiveError>> {
        let kernel = &self.kernel;
        let columns = (!kernel.is_symmetric()).then(|| kernel.column_states());
        let matrix = watched(cancel, |stop| {
            native.fidelity_matrix(
                kernel.row_states(),
                columns,
                Some(&|| stop.load(Ordering::Relaxed)),
            )
        });
        Some(match matrix {
            Ok(Ok(matrix)) => {
                self.matrix = Some(matrix);
                self.filled = self.kernel.num_pairs();
                Ok(())
            }
            Ok(Err(e)) => Err(PrimitiveError::Infrastructure(e.into())),
            Err(e) => Err(e),
        })
    }
}

/// How often [`watched`] looks for a cancellation or a Ctrl+C: the simulator's
/// own floor between two calls of its cancellation hook, so a Ctrl+C is
/// honoured within about a frame.
const WATCH_INTERVAL: Duration = Duration::from_millis(25);

/// Run `work` on a scoped thread while this thread watches `cancel`: its
/// cooperative cancellation and its interrupt guard (at the edge, a pending
/// Ctrl+C: the guard attaches and calls `check_signals`, which only works on
/// the thread that called into Polypus — this one — and is a no-op on the
/// rayon threads an adjoint batch or the kernel's inner products run on). On
/// either, `stop` is set for
/// `work`'s cancellation hook to see, and once `work` returns, the reason is the
/// result: the same `InfrastructureError` a check between blocks gives, so the
/// edge raises the same `KeyboardInterrupt`. A signal the guard consumed is
/// raised even if `work` finished meanwhile, since it cannot be raised again.
///
/// The watching costs nothing while `work` runs (a timed park) and nothing
/// after: `work`'s thread unparks this one as it ends.
fn watched<T: Send>(
    cancel: &CancelToken,
    work: impl FnOnce(&AtomicBool) -> T + Send,
) -> Result<T, PrimitiveError> {
    let stop = AtomicBool::new(false);
    let watcher = std::thread::current();
    let mut reason: Option<InfrastructureError> = None;
    let outcome = std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = work(&stop);
            watcher.unpark();
            result
        });
        while !worker.is_finished() {
            std::thread::park_timeout(WATCH_INTERVAL);
            if reason.is_none() && !worker.is_finished() {
                reason = if cancel.is_cancelled() {
                    Some(InfrastructureError::Cancelled)
                } else {
                    cancel.poll_interrupt().err()
                };
                if reason.is_some() {
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
        worker.join()
    });
    match (reason, outcome) {
        (Some(reason), _) => Err(PrimitiveError::Infrastructure(reason)),
        (None, Ok(result)) => Ok(result),
        (None, Err(_)) => Err(PrimitiveError::Internal(
            "the watched work's thread panicked".to_string(),
        )),
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
    // Read Python's log levels before the GIL is released below (`python_log`).
    super::python_log::refresh_levels(py);
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
///   `backend=None` by infrastructure; `seed` drawn when omitted, refused on
///   `qmio`, and below `2**63` on Aer and CUNQA. The seed is logged, not
///   returned: pass one to replay.
/// - **Runs** one circuit per row, in blocks of rows over one backend for the
///   whole call, with the GIL released; Ctrl+C stops it between blocks.
///
/// Every check runs before a backend exists and raises
/// `polypus.ConfigurationError` (a `ValueError`), except an `x` that is not
/// rows of real numbers: a `TypeError` (a complex value, Python's or NumPy's,
/// whatever its imaginary part, names its position).
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
/// `polypus.ConfigurationError` (a `ValueError`), except an `x` that is not
/// rows of real numbers: a `TypeError` (a complex value, Python's or NumPy's,
/// whatever its imaginary part, names its position).
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

/// A shape as Python prints a tuple: `(3, 2)`, `(3,)`, `()`.
fn shape_repr(shape: &[usize]) -> String {
    match shape {
        [only] => format!("({only},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `qml.vjp`'s `cotangent` as its `rows · observables` values, row-major: the
/// check that follows `params` (C-11). Its values are read as `x`'s are (each
/// a real number, a `bool` included), so the two take the same values:
///
/// - **Structure first**, through `numpy.asarray(cotangent)`: rows as lists or
///   tuples, or a NumPy array. Ragged rows (NumPy's `ValueError`) or a shape
///   other than `(rows, observables)` is a `polypus.ConfigurationError` naming
///   the expected shape. A complex dtype is checked before the shape, and is
///   the `TypeError` `x` raises for a complex value rather than losing the
///   imaginary part in the conversion. Any other exception of NumPy's (a
///   `MemoryError`, say) propagates as it is.
/// - **Then the values**: a real or boolean dtype is converted to `float64`
///   in one go; any other (strings, objects such as `None` or a complex
///   number) is read value by value as `x` is, through [`RealReader`], so a
///   value that is not a real number is the `TypeError` `x` raises for it.
///   Non-finite values are taken as they are.
fn cotangent_values(
    cotangent: &Bound<'_, PyAny>,
    rows: usize,
    observables: usize,
) -> PyResult<Vec<f64>> {
    let py = cotangent.py();
    let expected = shape_repr(&[rows, observables]);
    let numpy = py.import("numpy")?;
    let as_is = numpy.call_method1("asarray", (cotangent,)).map_err(|e| {
        if e.is_instance_of::<PyValueError>(py) {
            configuration_error(format!(
                "cotangent must be a (rows, observables) = {expected} array of real \
                 numbers: {}",
                e.value(py)
            ))
        } else {
            e
        }
    })?;
    let dtype = as_is.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    if kind == "c" {
        return Err(PyTypeError::new_err(format!(
            "cotangent must hold real numbers, as x does, got a complex array ({}); the \
             imaginary part is never dropped, even when it is zero",
            dtype.str()?
        )));
    }
    let shape: Vec<usize> = as_is.getattr("shape")?.extract()?;
    if shape != [rows, observables] {
        return Err(configuration_error(format!(
            "cotangent must have shape (rows, observables) = {expected}, got {}",
            shape_repr(&shape)
        )));
    }
    if matches!(kind.as_str(), "f" | "i" | "u" | "b") {
        let array = numpy
            .call_method1("asarray", (as_is, "float64"))?
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        return Ok(array.as_array().iter().copied().collect());
    }
    // Strings, objects, dates: value by value, with `x`'s `TypeError` for the
    // first that is not a real number. `tolist` keeps an object array's own
    // values, so a NumPy complex scalar reaches the reader as it is.
    let values = RealReader::new(py)?.rows(&as_is.call_method0("tolist")?, "cotangent", |e| {
        if e.is_instance_of::<PyTypeError>(py) {
            PyTypeError::new_err(format!(
                "cotangent must hold real numbers, as x does: {}",
                e.value(py)
            ))
        } else {
            e
        }
    })?;
    Ok(values.into_iter().flatten().collect())
}

/// `polypus.qml.vjp(model, x, params, cotangent, *, wrt="params", …)`: the
/// vector-Jacobian product of [`jacobian`] with `cotangent`, `v`, of shape
/// `(rows, observables)` — what a reverse-mode backward pass needs — as
/// `float64` `numpy.ndarray`s, C-contiguous:
///
/// - **`wrt="params"`** gives `Σ_r,o v[r,o]·J_θ[r,o,:]`, shape `(num_params,)`,
///   summed over the rows; **`"inputs"`** gives `Σ_o v[r,o]·J_x[r,o,:]` per
///   row, `(rows, features)`, for angle and IQP encoders (an amplitude or basis
///   encoder is [`jacobian`]'s `polypus.ConfigurationError`); **`"both"`** the
///   tuple `(by_params, by_inputs)`, each identical to its separate call. Any
///   other `wrt` is a `polypus.ConfigurationError`.
/// - **`cotangent`** is rows of real values or a 2-D NumPy array (a complex
///   value is `x`'s `TypeError`), of shape exactly `(rows, observables)` (otherwise a
///   `polypus.ConfigurationError`), its values read as `x`'s are (a string or
///   `None` is the same `TypeError`) and taken as they are: a `NaN` or an
///   infinity propagates to the product as it does through the contraction of
///   [`jacobian`].
/// - **Circuits:** with `exact=True`, one adjoint differentiation per row,
///   whatever the number of observables (of `Σ_o v[r,o]·O_o`); a row whose
///   combined observable cannot be weighed safely (a non-finite value in its
///   row of `cotangent`, or a weight bound past `f64::MAX / 4`) takes
///   [`jacobian`]'s runs, one per observable, decided row by row so no row
///   depends on another; and when the memory budget holds one statevector but
///   not two,
///   [`jacobian`]'s shifted circuits. With shots, [`jacobian`]'s circuits, the
///   same with the same `seed`: the result is its contraction, to rounding.
/// - **`model`**, **`x`**, **`params`** and the execution arguments are
///   [`jacobian`]'s, with its rules.
///
/// Every check runs before a backend exists and raises
/// `polypus.ConfigurationError` (a `ValueError`), except an `x` that is not
/// rows of real numbers ([`expval`]'s `TypeError`) and a cotangent value that
/// is not a real number: the `TypeError` `x` raises for one.
#[pyfunction(name = "vjp", signature = (
    model, x, params, cotangent, *, wrt="params", shots=1024, infrastructure="local", n_qpus=1,
    nodes=1, cores_per_qpu=2, walltime=None, options=None, sim_method="automatic",
    noise_model=None, backend=None, seed=None, exact=false,
))]
pub fn vjp<'py>(
    model: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    params: Vec<f64>,
    cotangent: &Bound<'py, PyAny>,
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
    let model = primitive_model(model, "vjp")?;
    let wrt = parse_wrt(wrt)?;
    let backend = exec.check()?;
    let (compiled, rows, params) = primitive_inputs(model.get(), x, params)?;
    let cotangent = cotangent_values(cotangent, rows.num_samples(), compiled.num_observables())?;
    let samples: Vec<&[f64]> = (0..rows.num_samples()).map(|i| rows.sample(i)).collect();
    let plan = compiled
        .jacobian(&samples, wrt)
        .map_err(qml_to_config_err)?;
    let (n, num_features) = (plan.num_samples(), plan.num_features());
    let mut primitive = VjpRun::new(plan, params, cotangent);
    run_primitive(py, "vjp", &mut primitive, exec, backend)?;
    let VjpRun {
        by_params,
        by_inputs,
        ..
    } = primitive;
    let by_params = || by_params.into_pyarray(py).into_any();
    let by_inputs = || {
        array2(n, num_features, by_inputs)
            .map(|a| a.into_pyarray(py).into_any())
            .map_err(shape_to_py_err)
    };
    Ok(match wrt {
        JacobianWrt::Params => by_params().unbind(),
        JacobianWrt::Inputs => by_inputs()?.unbind(),
        JacobianWrt::Both => PyTuple::new(py, [by_params(), by_inputs()?])?
            .into_any()
            .unbind(),
    })
}

/// `encoder` as a native [`Encoder`], or the `TypeError` `qml.kernel_matrix`
/// raises for anything else, naming what it takes.
fn kernel_encoder<'py>(encoder: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Encoder>> {
    if let Ok(encoder) = encoder.cast::<Encoder>() {
        return Ok(encoder.clone());
    }
    let hint = if encoder.is_instance_of::<Ansatz>() {
        "; trainable layers have no place in a fidelity kernel (trainable kernels are not \
         supported)"
    } else if encoder.is_instance_of::<Model>() || encoder.is_instance_of::<TrainedModel>() {
        "; pass the polypus.qml.Encoder the model was built from"
    } else {
        ""
    };
    Err(PyTypeError::new_err(format!(
        "qml.kernel_matrix takes a polypus.qml.Encoder as encoder, got {}{hint}",
        encoder.get_type().name()?
    )))
}

/// The rows of `x` (rows or a `Dataset`, whose labels are not used) for
/// `qml.kernel_matrix`, checked as `qml.expval` checks its rows: a complex
/// value is a `TypeError` naming `name` and its position (see
/// [`RealReader`]); then empty, zero-width, ragged or non-finite rows, each a
/// `polypus.ConfigurationError` naming `name`.
fn kernel_rows(x: &Bound<'_, PyAny>, name: &str) -> PyResult<polypus_qml::Dataset> {
    if let Ok(dataset) = x.cast::<PyDataset>() {
        return Ok(dataset.try_borrow()?.inner.clone());
    }
    let rows = RealReader::new(x.py())?.rows(x, name, |e| e)?;
    polypus_qml::Dataset::from_rows(&rows, &vec![0.0; rows.len()])
        .map_err(|e| configuration_error(format!("{name}: {e}")))
}

/// The states of every row of `rows` under `feature_map`, each rejection (a
/// zero sample for amplitude encoding, a non-binary feature for basis
/// encoding) a `polypus.ConfigurationError` naming `name`.
fn kernel_states(
    feature_map: &FeatureMap,
    rows: &polypus_qml::Dataset,
    name: &str,
) -> PyResult<FeatureStates> {
    let samples: Vec<&[f64]> = (0..rows.num_samples()).map(|i| rows.sample(i)).collect();
    feature_map
        .states(&samples)
        .map_err(|e| configuration_error(format!("{name}: {e}")))
}

/// `polypus.qml.kernel_matrix(encoder, x1, x2=None, *, …)`: the fidelity
/// kernel `K[i, j] = |⟨φ(x1_i)|φ(x2_j)⟩|²` of the states `encoder` prepares,
/// as a `float64` `numpy.ndarray` of shape `(len(x1), len(x2))`, C-contiguous,
/// ready for `sklearn.svm.SVC(kernel="precomputed")`.
///
/// - **`encoder`** is a `polypus.qml.Encoder` (anything else is a
///   `TypeError`); **`x1`** and **`x2`** are rows of features or `Dataset`s
///   (labels are not used), as wide as each other.
/// - **`x2=None`** is the Gram matrix of `x1` with itself: only the upper
///   triangle, diagonal included, is computed, `N(N+1)/2` circuits, and
///   mirrored, so the matrix is symmetric bit for bit. The diagonal is
///   executed, not set to `1`. An explicit `x2` computes all `N·M` entries,
///   even if it equals `x1`.
/// - **Execution** takes `qml.expval`'s arguments with its rules. With
///   `exact=True`, the inner products of the native simulator's statevectors
///   (each state simulated once when they fit in the memory budget); otherwise
///   each entry is the frequency of `0…0` over `shots` shots of the circuit
///   `U(x1_i)` then `U(x2_j)†` — an unbiased estimate, and a matrix that may
///   not be positive semi-definite.
///
/// Every check runs before a backend exists and raises
/// `polypus.ConfigurationError` (a `ValueError`), except an `x1` or `x2` that
/// is not rows of real numbers: a `TypeError` (a complex value, Python's or
/// NumPy's, whatever its imaginary part, names its position).
#[pyfunction(name = "kernel_matrix", signature = (
    encoder, x1, x2=None, *, shots=1024, infrastructure="local", n_qpus=1, nodes=1,
    cores_per_qpu=2, walltime=None, options=None, sim_method="automatic", noise_model=None,
    backend=None, seed=None, exact=false,
))]
pub fn kernel_matrix<'py>(
    encoder: &Bound<'py, PyAny>,
    x1: &Bound<'py, PyAny>,
    x2: Option<&Bound<'py, PyAny>>,
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
    let py = x1.py();
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
    let encoder = kernel_encoder(encoder)?;
    let backend = exec.check()?;
    let (feature_map, rows) = {
        // A copy: the encoder can be built on afterwards without changing it.
        let stack = encoder.try_borrow()?.stack.clone();
        let rows1 = kernel_rows(x1, "x1")?;
        if stack.layers.is_empty() {
            return Err(configuration_error(
                "the Encoder has no layers: add an encoder (angle_encoder, iqp_encoder, \
                 amplitude_encoder or basis_encoder) to load the rows",
            ));
        }
        let feature_map = FeatureMap::compile(stack.num_qubits, stack.layers, rows1.num_features())
            .map_err(|e| configuration_error(format!("x1: {e}")))?;
        let rows = kernel_states(&feature_map, &rows1, "x1")?;
        (feature_map, rows)
    };
    let columns = x2
        .map(|x2| {
            let rows2 = kernel_rows(x2, "x2")?;
            if rows2.num_features() != feature_map.num_features() {
                return Err(configuration_error(format!(
                    "x2 has {} feature(s) per row but x1 has {}; the kernel compares rows of \
                     the same width",
                    rows2.num_features(),
                    feature_map.num_features()
                )));
            }
            kernel_states(&feature_map, &rows2, "x2")
        })
        .transpose()?;
    let kernel = FidelityKernel::from_states(rows, columns).map_err(qml_to_config_err)?;
    let (num_rows, num_columns) = (kernel.num_rows(), kernel.num_columns());
    let mut primitive = KernelRun::new(kernel);
    run_primitive(py, "kernel_matrix", &mut primitive, exec, backend)?;
    let matrix = primitive.into_matrix().map_err(qml_to_py_err)?;
    let matrix = array2(num_rows, num_columns, matrix).map_err(shape_to_py_err)?;
    Ok(matrix.into_pyarray(py))
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
    qml.add_function(wrap_pyfunction!(vjp, qml)?)?;
    qml.add_function(wrap_pyfunction!(kernel_matrix, qml)?)?;
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
    use polypus_qml::SizeLimit;
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
        // Three per qubit per layer, after pooling too (4 → 2 qubits).
        let strongly = || Layer::StronglyEntangling(StronglyEntanglingAnsatz::new(2));
        assert_eq!(check_layers(3, &[strongly()]).unwrap(), 18);
        assert_eq!(
            check_layers(
                4,
                &[Layer::Pool(PoolLayer::new(PoolBlock::Basic)), strongly()]
            )
            .unwrap(),
            3 + 12
        );
        // n + 2·(n−1)·reps: an initial Ry per qubit, two per pair per block.
        let simplified = |reps| Layer::SimplifiedTwoDesign(SimplifiedTwoDesignAnsatz::new(reps));
        assert_eq!(check_layers(3, &[simplified(2)]).unwrap(), 11);
        assert_eq!(check_layers(4, &[simplified(2)]).unwrap(), 16);
        assert_eq!(check_layers(5, &[simplified(1)]).unwrap(), 13);
        assert_eq!(check_layers(3, &[simplified(0)]).unwrap(), 3);
        // After pooling 4 → 2 qubits: the pooling's 3, then 2 + 2·1·2.
        assert_eq!(
            check_layers(
                4,
                &[Layer::Pool(PoolLayer::new(PoolBlock::Basic)), simplified(2)]
            )
            .unwrap(),
            3 + 6
        );
        // One qubit has no pairs: one parameter, however many blocks.
        assert_eq!(check_layers(1, &[simplified(usize::MAX)]).unwrap(), 1);
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
        assert!(!loads_data(&Layer::StronglyEntangling(
            StronglyEntanglingAnsatz::new(1)
        )));
        assert!(!loads_data(&Layer::SimplifiedTwoDesign(
            SimplifiedTwoDesignAnsatz::new(1)
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
        // A strongly entangling range must lie in 1..=n-1 for the qubits still
        // active: 3 is valid on 4 qubits, not on the 2 left after pooling.
        let ranges = |ranges: Vec<usize>| {
            Layer::StronglyEntangling(StronglyEntanglingAnsatz {
                ranges: Some(ranges),
                ..StronglyEntanglingAnsatz::new(1)
            })
        };
        assert!(check_layers(4, &[ranges(vec![3])]).is_ok());
        assert!(check_layers(4, &[pool(), ranges(vec![3])]).is_err());
        assert!(check_layers(4, &[ranges(vec![0])]).is_err());
        assert!(check_layers(4, &[ranges(vec![1, 1])]).is_err());
        // 3 × 4 qubits × reps parameters that do not fit: an error, not a panic.
        let huge = Layer::StronglyEntangling(StronglyEntanglingAnsatz::new(usize::MAX / 4));
        assert!(check_layers(4, &[huge]).is_err());
        // 2 × 3 pairs × reps parameters that do not fit either.
        let huge = Layer::SimplifiedTwoDesign(SimplifiedTwoDesignAnsatz::new(usize::MAX / 4));
        assert!(check_layers(4, &[huge]).is_err());
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

    #[test]
    fn check_layers_refuses_a_model_past_the_size_limits() {
        Python::initialize();
        let too_large =
            |layer: usize, reason| ValidationError::ModelTooLarge { layer, reason }.to_string();
        let message = |layers: &[Layer]| {
            Python::attach(|py| match check_layers(2, layers) {
                Ok(count) => panic!("expected an error, got {count} parameter(s)"),
                Err(err) => {
                    assert!(err.is_instance_of::<PyValueError>(py), "{err}");
                    err.value(py).to_string()
                }
            })
        };
        // A count that does not fit in a usize.
        let huge = Layer::HardwareEfficient(HardwareEfficientAnsatz::new(usize::MAX));
        assert_eq!(message(&[huge]), too_large(0, SizeLimit::Overflow));
        let huge = Layer::SimplifiedTwoDesign(SimplifiedTwoDesignAnsatz::new(usize::MAX));
        assert_eq!(message(&[huge]), too_large(0, SizeLimit::Overflow));
        // One that fits but passes the budget: 5·2^40 + 4 gates.
        let budget = |gates| SizeLimit::Budget {
            gates,
            max_gates: 4_000_000,
        };
        let big = Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1 << 40));
        assert_eq!(
            message(std::slice::from_ref(&big)),
            too_large(0, budget(5_497_558_138_884))
        );
        // Entanglers alone, after the 5 gates of a `real_amplitudes(1)`.
        let entanglers_only = Layer::HardwareEfficient(HardwareEfficientAnsatz {
            rotations: vec![],
            ..HardwareEfficientAnsatz::new(1 << 40)
        });
        let small = Layer::HardwareEfficient(HardwareEfficientAnsatz::real_amplitudes(1));
        assert_eq!(
            message(&[small, entanglers_only]),
            too_large(1, budget(1_099_511_627_781))
        );
    }

    #[test]
    fn check_num_qubits_refuses_zero_and_more_than_the_limit() {
        Python::initialize();
        Python::attach(|py| {
            let message = |num_qubits| {
                let err = check_num_qubits("Ansatz", num_qubits).unwrap_err();
                assert!(err.is_instance_of::<PyValueError>(py), "{err}");
                err.value(py).to_string()
            };
            assert_eq!(
                message(0),
                "Ansatz needs at least one qubit, got num_qubits=0"
            );
            assert_eq!(
                message(1_000_001),
                "Ansatz takes at most 1000000 qubits, got num_qubits=1000001"
            );
            assert!(message(usize::MAX).contains(&usize::MAX.to_string()));
            assert!(check_num_qubits("Ansatz", 1).is_ok());
            assert!(check_num_qubits("Ansatz", 1_000_000).is_ok());
        });
    }

    #[test]
    fn a_count_argument_is_a_value_error_when_out_of_range() {
        Python::initialize();
        Python::attach(|py| {
            let extract = |source: &str| {
                let value = py.eval(&std::ffi::CString::new(source).unwrap(), None, None)?;
                value.extract::<CountArg>().map(|count| count.0)
            };
            assert_eq!(extract("3").unwrap(), 3);
            assert_eq!(extract("2**64 - 1").unwrap(), usize::MAX);
            assert_eq!(extract(&index_only("3")).unwrap(), 3);
            let (index_minus_one, index_huge) = (index_only("-1"), index_only("2**70"));
            for (source, message) in [
                ("-1", "expected a non-negative integer, got -1"),
                (
                    "-2**70",
                    "expected a non-negative integer, got -1180591620717411303424",
                ),
                ("2**64", "got 18446744073709551616"),
                // An object with only `__index__`, neither comparable nor
                // printed as its integer: described by the integer it returns.
                (
                    index_minus_one.as_str(),
                    "expected a non-negative integer, got -1",
                ),
                (index_huge.as_str(), "got 1180591620717411303424"),
            ] {
                let err = extract(source).unwrap_err();
                assert!(err.is_instance_of::<PyValueError>(py), "{source}: {err}");
                assert!(
                    err.value(py).to_string().contains(message),
                    "{source}: {err}"
                );
            }
            // Another type keeps its `TypeError`, and a `bool`, although an
            // `int` to Python, is one too.
            for source in ["'3'", "3.0", "None", "True", "False"] {
                let err = extract(source).unwrap_err();
                assert!(err.is_instance_of::<PyTypeError>(py), "{source}: {err}");
            }
            for (source, shown) in [("True", "True"), ("False", "False")] {
                let message = extract(source).unwrap_err().value(py).to_string();
                assert_eq!(
                    message,
                    format!("expected an integer, got a bool ({shown})")
                );
            }
        });
    }

    /// A Python expression for an object whose only integer protocol is
    /// `__index__`, returning `value`: no `__lt__`, and a `str` that is not
    /// the integer.
    fn index_only(value: &str) -> String {
        format!("type('IndexOnly', (), {{'__index__': lambda self: {value}}})()")
    }

    #[test]
    fn a_batch_size_out_of_range_is_kept_for_fit_to_refuse() {
        Python::initialize();
        Python::attach(|py| {
            let extract = |source: &str| {
                let value = py.eval(&std::ffi::CString::new(source).unwrap(), None, None)?;
                value.extract::<BatchSizeArg>()
            };
            // In range for a `usize`: kept as the size, which `fit` then checks
            // against the training set.
            for (source, size) in [("0", 0), ("4", 4), ("2**64 - 1", usize::MAX)] {
                match extract(source).unwrap() {
                    BatchSizeArg::Size(got) => assert_eq!(got, size, "{source}"),
                    BatchSizeArg::OutOfRange(_) => panic!("{source}: out of range"),
                }
            }
            // Outside it: kept, never PyO3's `OverflowError`, and shown as
            // Python prints it in `fit`'s message.
            let index_minus_one = index_only("-1");
            for (source, shown) in [
                ("-1", "-1"),
                ("2**64", "18446744073709551616"),
                (index_minus_one.as_str(), "-1"),
            ] {
                let arg = extract(source).unwrap();
                assert!(
                    matches!(arg, BatchSizeArg::OutOfRange(_)),
                    "{source}: in range"
                );
                assert_eq!(arg.to_string(), shown);
            }
            // Another type keeps its `TypeError`, a `bool` included.
            for source in ["'2'", "2.0", "True", "False"] {
                let err = extract(source).err().unwrap();
                assert!(err.is_instance_of::<PyTypeError>(py), "{source}: {err}");
            }
        });
    }

    #[test]
    fn a_pauli_position_out_of_range_is_a_value_error_naming_it() {
        Python::initialize();
        Python::attach(|py| {
            let eval = |source: &str| {
                py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
                    .unwrap()
            };
            let factors = |source: &str| term_factors(&eval(source), ("term", 2));
            let observable = |source: &str| extract_observable(&eval(source), 3).map(|_| ());

            assert_eq!(
                factors("[('z', 0), ('x', 2**64 - 1)]").unwrap(),
                vec![("z".to_string(), 0), ("x".to_string(), usize::MAX)]
            );
            let index_minus_one = format!("[('z', {})]", index_only("-1"));
            for (source, message) in [
                (
                    "[('z', -1)]",
                    "factor 0 (\"z\") of {} is out of range: expected a \
                     non-negative integer, got -1",
                ),
                (
                    "[('z', 0), ('x', 2**64)]",
                    "factor 1 (\"x\") of {} is out of range: expected an integer of at \
                     most 18446744073709551615, got 18446744073709551616",
                ),
                // The first bad factor decides: the position before the pair.
                (
                    "[('z', -1), ('z',)]",
                    "factor 0 (\"z\") of {} is out of range: expected a \
                     non-negative integer, got -1",
                ),
                // A Pauli is shown escaped.
                ("[('\"', -1)]", "factor 0 (\"\\\"\") of {} is out of range"),
                // An `__index__` object is described by its integer.
                (
                    index_minus_one.as_str(),
                    "factor 0 (\"z\") of {} is out of range: expected a \
                     non-negative integer, got -1",
                ),
            ] {
                for (err, whose) in [
                    (factors(source).unwrap_err(), "term 2"),
                    (observable(source).unwrap_err(), "observable 3"),
                ] {
                    let message = format!("the position of {}", message.replace("{}", whose));
                    assert!(err.is_instance_of::<PyValueError>(py), "{source}: {err}");
                    assert!(
                        err.value(py).to_string().contains(&message),
                        "{source}: {err} (expected {message})"
                    );
                }
            }
            // Anything else that is not a list of `(str, int)` pairs keeps the
            // `TypeError` naming the forms, also when it comes first.
            for source in [
                "[('z',)]",
                "[('z', '0')]",
                "[('z', 0.5)]",
                "'zz'",
                "[('z',), ('z', -1)]",
                "[('z', True)]",
            ] {
                let err = factors(source).unwrap_err();
                assert!(err.is_instance_of::<PyTypeError>(py), "{source}: {err}");
                assert!(err.value(py).to_string().contains("a Pauli term must be"));
                let err = observable(source).unwrap_err();
                assert!(err.is_instance_of::<PyTypeError>(py), "{source}: {err}");
                assert!(err
                    .value(py)
                    .to_string()
                    .contains("each observable must be"));
            }
        });
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
                let shift = alone.jacobian_from_probabilities(&probabilities).unwrap();
                // #503: the exact Jacobian is the adjoint one; the row alone
                // is still the reference bit for bit, and it is the shift's to
                // 1e-9.
                let got = &run.by_params[i * per_row..(i + 1) * per_row];
                assert_eq!(
                    Some(got),
                    adjoint_jacobian(&alone, &theta).params(),
                    "block {block}, row {i}"
                );
                assert_close(got, shift.params().unwrap());
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

    /// #645: on the exact path, `qml.expval` and the shifted circuits of
    /// `qml.jacobian` and `qml.vjp` are read into the model's expectations as
    /// each circuit's map is built, so the runner hands them expectations, not
    /// maps; what they keep is what reading the maps gives, bit for bit
    /// (`{:?}` prints every `f64` exactly, a `-0.0` included).
    #[test]
    fn exact_primitives_read_expectations_that_match_the_maps_bit_for_bit() {
        fn read_both_ways<P: Primitive>(mut by_readings: P, mut by_maps: P) -> (P, P) {
            let runner = BlockRunner::Exact {
                native: NativeStatevectorBackend::new(0),
                params: run_params(0, 1),
            };
            let cancel = CancelToken::default();
            let all = 0..by_maps.num_samples();
            let circuits = |primitive: &P| -> Vec<BoundCircuit> {
                primitive
                    .circuits(all.clone())
                    .unwrap()
                    .into_iter()
                    .map(BoundCircuit::Native)
                    .collect()
            };
            assert!(by_readings.exact_readout().is_some());
            let readings = runner
                .run(circuits(&by_readings), &cancel, by_readings.exact_readout())
                .unwrap();
            assert!(matches!(readings, BlockResults::Expectations(_)));
            by_readings.read(all.clone(), readings).unwrap();
            let maps = runner.run(circuits(&by_maps), &cancel, None).unwrap();
            assert!(matches!(maps, BlockResults::Exact(_)));
            by_maps.read(all, maps).unwrap();
            (by_readings, by_maps)
        }
        let exactly = |a: &[f64], b: &[f64]| assert_eq!(format!("{a:?}"), format!("{b:?}"));

        let (a, b) = read_both_ways(expval_run(), expval_run());
        assert!(!a.values.is_empty());
        exactly(&a.values, &b.values);
        let compiled = model_with_observables(2);
        let v = cotangent(5, 2);
        for wrt in [JacobianWrt::Params, JacobianWrt::Inputs, JacobianWrt::Both] {
            let (a, b) = read_both_ways(jacobian_run(wrt), jacobian_run(wrt));
            assert!(!(a.by_params.is_empty() && a.by_inputs.is_empty()));
            exactly(&a.by_params, &b.by_params);
            exactly(&a.by_inputs, &b.by_inputs);
            let (a, b) = read_both_ways(
                vjp_run(&compiled, wrt, v.clone()),
                vjp_run(&compiled, wrt, v.clone()),
            );
            assert!(!(a.by_params.is_empty() && a.by_inputs.is_empty()));
            exactly(&a.by_params, &b.by_params);
            exactly(&a.by_inputs, &b.by_inputs);
        }
    }

    /// #645: `predict(exact=True)` reads each row's decision as its map is
    /// built; the decisions are what the maps give, bit for bit.
    /// With a raw readout the decision is `⟨O₀⟩` itself, so "bit for bit"
    /// compares real values, not class indices.
    #[test]
    fn exact_predictions_are_the_maps_decisions_bit_for_bit() {
        let (argmax, rows, _) = primitive_model_and_rows();
        for compiled in [argmax, model_with_observables(1)] {
            let theta: Vec<f64> = (0..compiled.num_params())
                .map(|p| -0.7 + 0.13 * p as f64)
                .collect();
            let circuits: Vec<BoundCircuit> = rows
                .iter()
                .map(|row| BoundCircuit::Native(compiled.bind(row, &theta).unwrap()))
                .collect();
            let native = NativeStatevectorBackend::new(0);
            let params = run_params(0, 1);
            let read = exact_predictions(&native, &params, &compiled, &circuits).unwrap();
            let from_maps: Vec<f64> = native
                .run_circuits_exact(&circuits, &params)
                .unwrap()
                .iter()
                .map(|map| compiled.predict_from_probabilities(map).unwrap())
                .collect();
            assert_eq!(read.len(), rows.len());
            assert_eq!(format!("{read:?}"), format!("{from_maps:?}"));
        }
    }

    /// The kernel reads probabilities: it keeps the maps, and refuses
    /// expectations as a bookkeeping error rather than misreading them.
    #[test]
    fn the_kernel_keeps_reading_maps_on_the_exact_path() {
        let mut run = kernel_run(true);
        assert!(run.exact_readout().is_none());
        assert!(matches!(
            run.read(0..1, BlockResults::Expectations(vec![vec![1.0]])),
            Err(PrimitiveError::Internal(_))
        ));
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

    /// `jacobian`'s exact Jacobian by adjoint differentiation, computed apart
    /// from the block runner: its adjoint runs on a fresh native backend.
    fn adjoint_jacobian(jacobian: &ExpectationJacobian, theta: &[f64]) -> polypus_qml::Jacobian {
        let runs = jacobian.adjoint_runs(theta).unwrap();
        let tasks: Vec<AdjointTask<'_, _>> = runs
            .iter()
            .map(|run| AdjointTask {
                circuit: &run.circuit,
                wrt: &run.wrt,
                weight: move |b: usize| run.observable.weight(b),
            })
            .collect();
        let derivatives = NativeStatevectorBackend::new(0)
            .run_adjoint(&tasks, None)
            .unwrap();
        jacobian.jacobian_from_adjoint(&derivatives).unwrap()
    }

    /// Two exact derivatives agree to rounding.
    fn assert_close(got: &[f64], shift: &[f64]) {
        assert_eq!(got.len(), shift.len());
        for (k, (g, s)) in got.iter().zip(shift).enumerate() {
            assert!((g - s).abs() < 1e-9, "entry {k}: {g} vs the shift's {s}");
        }
    }

    /// A model only the adjoint differentiates (#561), its templates written
    /// by hand (`test-support`: no layer emits these gates): the model, its
    /// rows and `θ`, the Jacobian plan over them (`wrt` the parameters), and
    /// the position of the first gate the shift has no rule for (`Cp`).
    fn adjoint_only_plan() -> (
        CompiledModel,
        Vec<Vec<f64>>,
        Vec<f64>,
        ExpectationJacobian,
        usize,
    ) {
        use polypus_circuit::{GateInstruction as G, GateParam, ParameterizedCircuit};
        let (compiled, rows, theta) = primitive_model_and_rows();
        let mut first = 0;
        let templates: Vec<ParameterizedCircuit> = rows
            .iter()
            .map(|row| {
                let mut gates = compiled.template_for(row).unwrap().gates().to_vec();
                assert_eq!(gates.pop(), Some(G::MeasureAll));
                first = gates.len();
                gates.extend([
                    G::Cp {
                        q0: 0,
                        q1: 1,
                        theta: GateParam::Param(0),
                    },
                    G::Cu3 {
                        control: 2,
                        target: 0,
                        theta: GateParam::Param(1),
                        phi: GateParam::Fixed(0.4),
                        lam: GateParam::Param(0),
                    },
                    G::H(0),
                    G::MeasureAll,
                ]);
                let mut template = ParameterizedCircuit::new(3);
                for gate in gates {
                    template.try_push(gate).unwrap();
                }
                template
            })
            .collect();
        let plan = ExpectationJacobian::from_templates_for_tests(
            &compiled,
            templates,
            JacobianWrt::Params,
        )
        .unwrap();
        (compiled, rows, theta, plan, first)
    }

    /// The memory's `NoExactGradient` of a model only the adjoint
    /// differentiates, at three qubits, naming the `Cp` at `first`.
    fn assert_memory_refusal(outcome: Option<Result<(), PrimitiveError>>, first: usize) {
        match outcome {
            Some(Err(PrimitiveError::Qml(QmlError::NoExactGradient { reason }))) => {
                let instruction = format!("instruction {first} (Cp");
                for needle in [
                    instruction.as_str(),
                    "two statevectors of 3 qubits",
                    "exact",
                ] {
                    assert!(reason.contains(needle), "{needle:?} missing: {reason}");
                }
            }
            other => panic!("expected the memory's NoExactGradient, got {other:?}"),
        }
    }

    /// #561: `qml.jacobian(exact=True)` on a model only the adjoint
    /// differentiates. With room for two statevectors it takes the adjoint;
    /// without, the block stops with the memory's `NoExactGradient` before
    /// anything runs or is read — where a model of Pauli rotations falls back
    /// to its shifted circuits (`None`). The decision is the injected one: the
    /// budget is the process's, which a test must not set.
    #[test]
    fn the_exact_jacobian_of_a_model_only_the_adjoint_differentiates_cannot_fall_back() {
        let (_, rows, theta, plan, first) = adjoint_only_plan();
        let mut run = JacobianRun {
            plan,
            params: theta.clone(),
            by_params: Vec::new(),
            by_inputs: Vec::new(),
        };
        let native = NativeStatevectorBackend::new(0);
        let cancel = CancelToken::default();
        let all = 0..rows.len();
        assert_memory_refusal(
            run.run_exact_block_with(&native, all.clone(), &cancel, |_| Ok(false))
                .map(|outcome| outcome.map(|_| ())),
            first,
        );
        assert!(run.by_params.is_empty(), "nothing may be read");

        // With room for two statevectors, the adjoint Jacobian.
        assert!(matches!(
            run.run_exact_block_with(&native, all, &cancel, |_| Ok(true)),
            Some(Ok(()))
        ));
        let apart = adjoint_jacobian(&run.plan, &theta);
        assert_eq!(Some(run.by_params.as_slice()), apart.params());

        // A model of Pauli rotations falls back to its shifted circuits.
        let mut pauli = jacobian_run(JacobianWrt::Params);
        assert!(pauli
            .run_exact_block_with(&native, 0..1, &cancel, |_| Ok(false))
            .is_none());
        assert!(pauli.by_params.is_empty());
    }

    /// #561, Diego's carry-over from #577: `qml.vjp(exact=True)` goes through
    /// the same `run_exact_adjoint` as the Jacobian, so a model only the
    /// adjoint differentiates cannot fall back to the shifted circuits there
    /// either — before this check it returned `None` and the shifted circuits
    /// were asked for, failing with the shift's message, to a user already in
    /// exact mode. Without room the block stops with the memory's
    /// `NoExactGradient` before anything runs; with room it is the adjoint
    /// Jacobian's contraction, to rounding (the combined observable sums in
    /// another order); and a model of Pauli rotations still falls back.
    #[test]
    fn the_exact_vjp_of_a_model_only_the_adjoint_differentiates_cannot_fall_back() {
        let (compiled, rows, theta, plan, first) = adjoint_only_plan();
        let v = cotangent(rows.len(), plan.num_observables());
        let mut run = VjpRun::new(plan, theta.clone(), v.clone());
        let native = NativeStatevectorBackend::new(0);
        let cancel = CancelToken::default();
        let all = 0..rows.len();
        assert_memory_refusal(
            run.exact_block_with(&native, all.clone(), &cancel, |_| Ok(false))
                .map(|outcome| outcome.map(|_| ())),
            first,
        );
        assert_eq!(run.adjoint_runs, 0, "nothing may run");

        // With room for two statevectors: one run per sample, and the product
        // is the adjoint Jacobian's contraction.
        let vjps = match run.exact_block_with(&native, all, &cancel, |_| Ok(true)) {
            Some(Ok(vjps)) => vjps,
            other => panic!("expected the products, got {other:?}"),
        };
        assert_eq!(vjps.len(), rows.len());
        let contracted = adjoint_jacobian(&run.plan, &theta).vjp(&v).unwrap();
        let num_params = contracted.num_params();
        let expected = contracted.params().expect("wrt asks for the parameters");
        for (s, vjp) in vjps.iter().enumerate() {
            let got = vjp.params().expect("wrt asks for the parameters");
            assert_close(got, &expected[s * num_params..(s + 1) * num_params]);
        }
        assert!(
            expected.iter().any(|d| d.abs() > 1e-4),
            "the comparison would be vacuous"
        );

        // A model of Pauli rotations falls back to its shifted circuits.
        let mut pauli = vjp_run(
            &compiled,
            JacobianWrt::Params,
            cotangent(rows.len(), compiled.num_observables()),
        );
        assert!(pauli
            .exact_block_with(&native, 0..1, &cancel, |_| Ok(false))
            .is_none());
    }

    /// `qml.jacobian(exact=True)` takes the adjoint (#503): with both parts,
    /// the Jacobian is the parameter shift's to rounding, and the block size
    /// changes nothing, bit for bit.
    #[test]
    fn the_exact_jacobian_is_the_adjoint_one_whatever_the_blocks() {
        let (compiled, rows, theta) = primitive_model_and_rows();
        let plan = compiled.jacobian(&rows, JacobianWrt::Both).unwrap();
        let circuits: Vec<BoundCircuit> = plan
            .circuits(&theta)
            .unwrap()
            .into_iter()
            .map(BoundCircuit::Native)
            .collect();
        let probabilities = NativeStatevectorBackend::new(0)
            .run_circuits_exact(&circuits, &run_params(0, 1))
            .unwrap();
        let shift = plan.jacobian_from_probabilities(&probabilities).unwrap();
        let adjoint = adjoint_jacobian(&plan, &theta);
        for block in [usize::MAX, 1, 2, 3] {
            let mut run = jacobian_run(JacobianWrt::Both);
            run.run_exact(block);
            assert_eq!(
                Some(run.by_params.as_slice()),
                adjoint.params(),
                "block {block}"
            );
            assert_eq!(
                Some(run.by_inputs.as_slice()),
                adjoint.inputs(),
                "block {block}"
            );
            assert_close(&run.by_params, shift.params().unwrap());
            assert_close(&run.by_inputs, shift.inputs().unwrap());
        }
    }

    /// An interrupt guard that lets its first `ok` polls through and then
    /// reports an interrupt, counting every poll.
    struct InterruptAfter {
        ok: usize,
        polls: AtomicUsize,
    }

    impl crate::infrastructure::Interrupt for InterruptAfter {
        fn poll(&self) -> Result<(), crate::infrastructure::BackendError> {
            if self.polls.fetch_add(1, Ordering::SeqCst) < self.ok {
                Ok(())
            } else {
                Err(crate::infrastructure::BackendError::Aborted(
                    "test interrupt".to_string(),
                ))
            }
        }
    }

    /// A Ctrl+C (the edge's interrupt guard) stops an adjoint batch in flight,
    /// with the error a check between blocks gives: one block, whose check
    /// before it lets the run start, then the watcher's first look interrupts
    /// it. The batch must outlive that first look (`WATCH_INTERVAL`, 25 ms), or
    /// the watcher never asks and the call ends `Ok`: uninterrupted, this one (16
    /// qubits, a 4-layer ansatz, 32 rows, both parts, one block) took 7.0 s in a
    /// debug `cargo test` on a 16-CPU node, 280 times that, so the margin holds
    /// on a much faster machine; interrupted, the test takes milliseconds.
    #[test]
    fn a_ctrl_c_stops_an_adjoint_jacobian_in_flight() {
        let compiled = QuantumModel::new(16)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(4)))
            .readout(
                Readout::new(
                    vec![Observable::new(vec![(
                        1.0,
                        PauliString::new(vec![(0, Pauli::Z)]).unwrap(),
                    )])
                    .unwrap()],
                    Decision::Raw,
                )
                .unwrap(),
            )
            .compile(16)
            .unwrap();
        let rows: Vec<Vec<f64>> = (0..32)
            .map(|i| (0..16).map(|j| 0.1 * (i + j) as f64).collect())
            .collect();
        let theta: Vec<f64> = (0..compiled.num_params())
            .map(|p| 0.01 * p as f64)
            .collect();
        let mut run = JacobianRun {
            plan: compiled.jacobian(&rows, JacobianWrt::Both).unwrap(),
            params: theta,
            by_params: Vec::new(),
            by_inputs: Vec::new(),
        };
        let guard = Arc::new(InterruptAfter {
            ok: 1,
            polls: AtomicUsize::new(0),
        });
        let token = CancelToken::with_interrupt(guard.clone());
        let err = run_in_blocks(
            &mut run,
            || {
                Ok(BlockRunner::Exact {
                    native: NativeStatevectorBackend::new(0),
                    params: run_params(0, 1),
                })
            },
            |_| usize::MAX,
            &token,
        )
        .unwrap_err();
        match err {
            PrimitiveError::Infrastructure(InfrastructureError::Backend(
                crate::infrastructure::BackendError::Aborted(message),
            )) => assert_eq!(message, "test interrupt"),
            other => panic!("expected the interrupt, got {other:?}"),
        }
        assert_eq!(guard.polls.load(Ordering::SeqCst), 2);
        assert!(run.by_params.is_empty() && run.by_inputs.is_empty());
    }

    // ── qml.vjp ──────────────────────────────────────────────────────────────

    /// `primitive_model_and_rows`'s model and rows read out by `k` observables
    /// (`⟨Z₀⟩`, `⟨Z₁Z₂⟩`, `⟨Z₂⟩`, …, cycling over the three qubits).
    fn model_with_observables(k: usize) -> CompiledModel {
        let strings = [vec![0], vec![1, 2], vec![2], vec![0, 1], vec![1]];
        let observables = (0..k)
            .map(|o| {
                let string = strings[o % strings.len()]
                    .iter()
                    .map(|&q| (q, Pauli::Z))
                    .collect();
                Observable::new(vec![(
                    1.0 + 0.1 * o as f64,
                    PauliString::new(string).unwrap(),
                )])
                .unwrap()
            })
            .collect();
        QuantumModel::new(3)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1)))
            .layer(Layer::Iqp(IqpEncoder::new()))
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(Readout::new(observables, Decision::Raw).unwrap())
            .compile(3)
            .unwrap()
    }

    /// A cotangent with neither zero nor repeated entries.
    fn cotangent(rows: usize, k: usize) -> Vec<f64> {
        (0..rows * k).map(|i| 0.9 - 0.23 * i as f64).collect()
    }

    fn vjp_run(compiled: &CompiledModel, wrt: JacobianWrt, v: Vec<f64>) -> VjpRun {
        let (_, rows, _) = primitive_model_and_rows();
        let theta = (0..compiled.num_params())
            .map(|p| -0.7 + 0.13 * p as f64)
            .collect();
        VjpRun::new(compiled.jacobian(&rows, wrt).unwrap(), theta, v)
    }

    /// `qml.jacobian`'s result for the same model, rows and `wrt`, contracted
    /// with `v` as the PyTorch backward pass did: `(Σ_r,o v·J_θ, Σ_o v·J_x)`.
    fn contracted_jacobian(run: &JacobianRun, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (n, k) = (run.plan.num_samples(), run.plan.num_observables());
        let (p, f) = (run.plan.num_params(), run.plan.num_features());
        let mut by_params = vec![0.0; if run.by_params.is_empty() { 0 } else { p }];
        let mut by_inputs = vec![0.0; if run.by_inputs.is_empty() { 0 } else { n * f }];
        for r in 0..n {
            for o in 0..k {
                let weight = v[r * k + o];
                for (c, sum) in by_params.iter_mut().enumerate() {
                    *sum += weight * run.by_params[(r * k + o) * p + c];
                }
                for c in 0..by_inputs.len() / n.max(1) {
                    by_inputs[r * f + c] += weight * run.by_inputs[(r * k + o) * f + c];
                }
            }
        }
        (by_params, by_inputs)
    }

    /// Exactly, `qml.vjp` hands `run_adjoint` one task per row, whatever the
    /// number of observables (`qml.jacobian` hands it one per row and
    /// observable), and its result is `qml.jacobian`'s contracted, whatever the
    /// blocks.
    #[test]
    fn the_exact_vjp_takes_one_adjoint_run_per_row_whatever_the_observables() {
        for k in [1, 2, 3, 5] {
            let compiled = model_with_observables(k);
            let v = cotangent(5, k);
            for wrt in [JacobianWrt::Params, JacobianWrt::Inputs, JacobianWrt::Both] {
                let mut jacobian = JacobianRun {
                    plan: compiled
                        .jacobian(&primitive_model_and_rows().1, wrt)
                        .unwrap(),
                    params: vjp_run(&compiled, wrt, v.clone()).params,
                    by_params: Vec::new(),
                    by_inputs: Vec::new(),
                };
                jacobian.run_exact(usize::MAX);
                let (by_params, by_inputs) = contracted_jacobian(&jacobian, &v);
                for block in [usize::MAX, 1, 2] {
                    let mut run = vjp_run(&compiled, wrt, v.clone());
                    run.run_exact(block);
                    assert_eq!(run.adjoint_runs, 5, "k = {k}, block {block}");
                    assert_eq!(run.by_params.len(), by_params.len());
                    assert_eq!(run.by_inputs.len(), by_inputs.len());
                    assert_close(&run.by_params, &by_params);
                    assert_close(&run.by_inputs, &by_inputs);
                }
            }
        }
    }

    /// The route the exact path takes when only one statevector fits — the
    /// Jacobian's shifted circuits run exactly and read by `read` — gives the
    /// adjoint's product; and with shots the circuits are `qml.jacobian`'s, so
    /// with the same seed the product is its contraction.
    #[test]
    fn the_shift_and_sampled_routes_contract_the_jacobian() {
        let compiled = model_with_observables(3);
        let v = cotangent(5, 3);
        let mut adjoint = vjp_run(&compiled, JacobianWrt::Both, v.clone());
        adjoint.run_exact(2);
        let mut shift = vjp_run(&compiled, JacobianWrt::Both, v.clone());
        for start in (0..5).step_by(2) {
            let samples = start..(start + 2).min(5);
            let circuits: Vec<BoundCircuit> = shift
                .circuits(samples.clone())
                .unwrap()
                .into_iter()
                .map(BoundCircuit::Native)
                .collect();
            let probabilities = NativeStatevectorBackend::new(0)
                .run_circuits_exact(&circuits, &run_params(0, 1))
                .unwrap();
            shift
                .read(samples, BlockResults::Exact(probabilities))
                .unwrap();
        }
        assert_eq!(shift.adjoint_runs, 0);
        assert_close(&shift.by_params, &adjoint.by_params);
        assert_close(&shift.by_inputs, &adjoint.by_inputs);

        let mut sampled = vjp_run(&compiled, JacobianWrt::Both, v.clone());
        sampled_on_native(&mut sampled, 7, 2);
        let mut jacobian = JacobianRun {
            plan: compiled
                .jacobian(&primitive_model_and_rows().1, JacobianWrt::Both)
                .unwrap(),
            params: sampled.params.clone(),
            by_params: Vec::new(),
            by_inputs: Vec::new(),
        };
        sampled_on_native(&mut jacobian, 7, 2);
        let (by_params, by_inputs) = contracted_jacobian(&jacobian, &v);
        assert_close(&sampled.by_params, &by_params);
        assert_close(&sampled.by_inputs, &by_inputs);
    }

    /// A non-finite cotangent is not refused: the row with one takes the
    /// Jacobian's runs (one per observable), every other row one run, and the
    /// value reaches the product as the contraction gives it.
    #[test]
    fn a_non_finite_cotangent_propagates_through_the_jacobians_runs() {
        let k = 2;
        let compiled = model_with_observables(k);
        let mut v = cotangent(5, k);
        v[3 * k] = f64::NAN; // row 3
        let mut run = vjp_run(&compiled, JacobianWrt::Both, v.clone());
        run.run_exact(2);
        // Rows 0, 1, 2 and 4 take one run each, row 3 takes k.
        assert_eq!(run.adjoint_runs, 4 + k);
        assert!(run.by_params.iter().all(|d| d.is_nan()));
        let f = compiled.num_features();
        for (r, row) in run.by_inputs.chunks(f).enumerate() {
            assert!(row.iter().all(|d| d.is_nan() == (r == 3)), "row {r}");
        }
    }

    /// `qml.vjp` exactly over the rows of `primitive_model_and_rows`, read out
    /// by two copies of `⟨Z₀⟩`, with row 1's cotangent `value` in both columns
    /// (so its combined weight bound is `2 · value`), in blocks of 2: the run,
    /// once checked against `qml.jacobian`'s contraction — the same finiteness,
    /// and within 1e-9 relative where finite. On the `combined` path, relative
    /// to `value`'s scale rather than to each entry: `assert_close`'s 1e-9 for
    /// a cotangent of order 1, scaled by this one's, since the rounding errors
    /// of the combined weight grow with `value` and an entry near zero is all
    /// rounding at that scale. Off it, the Jacobian is `qml.jacobian`'s own and
    /// each entry is held to its own magnitude.
    fn vjp_with_a_large_row(value: f64, combined: bool) -> VjpRun {
        let z0 = || {
            Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap()
        };
        let compiled = QuantumModel::new(3)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1)))
            .layer(Layer::Iqp(IqpEncoder::new()))
            .readout(Readout::new(vec![z0(), z0()], Decision::Argmax).unwrap())
            .compile(3)
            .unwrap();
        let mut v = cotangent(5, 2);
        v[2] = value; // row 1, both observables
        v[3] = value;
        assert!(v.iter().all(|x| x.is_finite()));
        let mut run = vjp_run(&compiled, JacobianWrt::Both, v.clone());
        run.run_exact(2);
        let mut jacobian = JacobianRun {
            plan: compiled
                .jacobian(&primitive_model_and_rows().1, JacobianWrt::Both)
                .unwrap(),
            params: run.params.clone(),
            by_params: Vec::new(),
            by_inputs: Vec::new(),
        };
        jacobian.run_exact(usize::MAX);
        let (by_params, by_inputs) = contracted_jacobian(&jacobian, &v);
        for (got, expected) in [(&run.by_params, &by_params), (&run.by_inputs, &by_inputs)] {
            assert_eq!(got.len(), expected.len());
            for (k, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.is_finite(),
                    e.is_finite(),
                    "{value}, entry {k}: {g} vs {e}"
                );
                if e.is_finite() {
                    let floor = if combined { value } else { 1.0 };
                    let scale = g.abs().max(e.abs()).max(floor);
                    assert!(
                        (g - e).abs() <= 1e-9 * scale,
                        "{value}, entry {k}: {g} vs {e}"
                    );
                }
            }
        }
        // Row 1's input product is large and finite.
        let f = compiled.num_features();
        assert!(
            run.by_inputs[f..2 * f]
                .iter()
                .any(|d| d.is_finite() && d.abs() > value / 1e3),
            "{value}"
        );
        run
    }

    /// Finite cotangent values can still overflow the combined weight: two
    /// copies of `⟨Z₀⟩` weighed by `1e308` each make a weight of `2e308`,
    /// which no statevector can be weighed by. That row takes the Jacobian's
    /// runs instead, and the product is the contraction, finite where it is.
    #[test]
    fn a_finite_cotangent_that_overflows_the_weight_takes_the_jacobians_runs() {
        let run = vjp_with_a_large_row(1e308, false);
        // Rows 0, 2, 3 and 4 take one run each, row 1 takes k = 2.
        assert_eq!(run.adjoint_runs, 4 + 2);
    }

    /// A finite weight bound past `MAX_COMBINED_WEIGHT` (`6e307 · 2 = 1.2e308`,
    /// below `f64::MAX`) takes the Jacobian's runs too; one under it (`1e307 ·
    /// 2 = 2e307`) stays on the combined path, one run per row. Both are the
    /// contraction.
    #[test]
    fn a_weight_bound_near_the_largest_float_takes_the_jacobians_runs() {
        const {
            assert!(2.0 * 6e307 > MAX_COMBINED_WEIGHT && (2.0_f64 * 6e307).is_finite());
            assert!(2.0 * 1e307 < MAX_COMBINED_WEIGHT);
        }
        assert_eq!(vjp_with_a_large_row(6e307, false).adjoint_runs, 4 + 2);
        assert_eq!(vjp_with_a_large_row(1e307, true).adjoint_runs, 5);
    }

    /// The route is decided row by row, never for a block: with a `NaN` in
    /// row 0's cotangent, or finite values that overflow row 1's combined
    /// weight, only that row takes the Jacobian's runs, so the product is the
    /// same bit for bit whatever the block size — that is, whatever the
    /// host's cores (DBFerPra's review of #577: blocks of 16 rows on one host
    /// and 8 on another gave rows 16–31 different last bits).
    #[test]
    fn the_route_is_decided_per_row_so_the_blocks_change_nothing() {
        let k = 3;
        let compiled = model_with_observables(k);
        let mut with_nan = cotangent(5, k);
        with_nan[0] = f64::NAN; // row 0
        let mut overflowing = cotangent(5, k);
        overflowing[k..2 * k].fill(1e308); // row 1
        let bits = |values: &[f64]| values.iter().map(|d| d.to_bits()).collect::<Vec<_>>();
        for v in [with_nan, overflowing] {
            let mut reference = vjp_run(&compiled, JacobianWrt::Both, v.clone());
            reference.run_exact(usize::MAX);
            for block in [1, 2, 3, 4] {
                let mut run = vjp_run(&compiled, JacobianWrt::Both, v.clone());
                run.run_exact(block);
                assert_eq!(run.adjoint_runs, 4 + k, "block {block}");
                assert_eq!(
                    bits(&run.by_inputs),
                    bits(&reference.by_inputs),
                    "block {block}"
                );
                assert_eq!(
                    bits(&run.by_params),
                    bits(&reference.by_params),
                    "block {block}"
                );
            }
            // The rows the value is not in are those of a finite cotangent's
            // route, one run each, bit for bit.
            let row = if v[0].is_nan() { 0 } else { 1 };
            let mut clean = v.clone();
            clean[row * k..(row + 1) * k].fill(0.5);
            let mut finite = vjp_run(&compiled, JacobianWrt::Inputs, clean);
            finite.run_exact(2);
            assert_eq!(finite.adjoint_runs, 5);
            let f = compiled.num_features();
            for r in (0..5).filter(|&r| r != row) {
                assert_eq!(
                    bits(&reference.by_inputs[r * f..(r + 1) * f]),
                    bits(&finite.by_inputs[r * f..(r + 1) * f]),
                    "row {r}"
                );
            }
        }
    }

    /// `read_exact` takes one row of derivatives per run, no more and no
    /// fewer: a row too many or too few is a bookkeeping error before any
    /// sample is read, rather than a product read from rows shifted by one.
    #[test]
    fn reading_the_exact_product_takes_exactly_one_row_per_run() {
        let k = 3;
        let compiled = model_with_observables(k);
        let mut v = cotangent(5, k);
        v[0] = f64::NAN; // row 0 takes k runs, the others one each
        let run = vjp_run(&compiled, JacobianWrt::Both, v);
        let (runs, combined) = run.exact_runs(0..5).unwrap();
        assert_eq!(runs.len(), 4 + k);
        let rows: Vec<Vec<f64>> = runs.iter().map(|r| vec![0.0; r.wrt.len()]).collect();
        assert_eq!(run.read_exact(0..5, &combined, &rows).unwrap().len(), 5);
        let mut extra = rows.clone();
        extra.push(rows[0].clone());
        let missing = &rows[..rows.len() - 1];
        for (what, derivatives) in [("one too many", extra.as_slice()), ("one too few", missing)] {
            let expected = format!(
                "{} row(s) of derivatives for {} run(s)",
                derivatives.len(),
                runs.len()
            );
            match run.read_exact(0..5, &combined, derivatives) {
                Err(PrimitiveError::Internal(message)) => {
                    assert!(message.contains(&expected), "{what}: {message}")
                }
                other => panic!("{what}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_cancellation_between_vjp_blocks_stops_before_the_next() {
        let counters = Counters::new();
        let token = CancelToken::default();
        let compiled = model_with_observables(2);
        let mut run = Interfering {
            inner: vjp_run(&compiled, JacobianWrt::Inputs, cotangent(5, 2)),
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
        assert_eq!(run.built.get(), 1);
        assert_eq!(counters.get(), (1, 1, 1));
        assert_eq!(run.inner.by_inputs.len(), 2 * compiled.num_features());
    }

    /// A Ctrl+C stops the adjoint runs of an exact `qml.vjp` block in flight,
    /// as it does `qml.jacobian`'s (see
    /// `a_ctrl_c_stops_an_adjoint_jacobian_in_flight` for the sizing).
    #[test]
    fn a_ctrl_c_stops_an_adjoint_vjp_in_flight() {
        let compiled = QuantumModel::new(16)
            .layer(Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)))
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(4)))
            .readout(
                Readout::new(
                    vec![Observable::new(vec![(
                        1.0,
                        PauliString::new(vec![(0, Pauli::Z)]).unwrap(),
                    )])
                    .unwrap()],
                    Decision::Raw,
                )
                .unwrap(),
            )
            .compile(16)
            .unwrap();
        let rows: Vec<Vec<f64>> = (0..32)
            .map(|i| (0..16).map(|j| 0.1 * (i + j) as f64).collect())
            .collect();
        let theta: Vec<f64> = (0..compiled.num_params())
            .map(|p| 0.01 * p as f64)
            .collect();
        let mut run = VjpRun::new(
            compiled.jacobian(&rows, JacobianWrt::Both).unwrap(),
            theta,
            vec![1.0; 32],
        );
        let guard = Arc::new(InterruptAfter {
            ok: 1,
            polls: AtomicUsize::new(0),
        });
        let token = CancelToken::with_interrupt(guard.clone());
        let err = run_in_blocks(
            &mut run,
            || {
                Ok(BlockRunner::Exact {
                    native: NativeStatevectorBackend::new(0),
                    params: run_params(0, 1),
                })
            },
            |_| usize::MAX,
            &token,
        )
        .unwrap_err();
        match err {
            PrimitiveError::Infrastructure(InfrastructureError::Backend(
                crate::infrastructure::BackendError::Aborted(message),
            )) => assert_eq!(message, "test interrupt"),
            other => panic!("expected the interrupt, got {other:?}"),
        }
        assert_eq!(guard.polls.load(Ordering::SeqCst), 2);
        assert!(run.by_params.iter().all(|&d| d == 0.0) && run.by_inputs.is_empty());
        assert_eq!(run.adjoint_runs, 0);
    }

    #[test]
    fn shapes_print_as_python_tuples() {
        assert_eq!(shape_repr(&[3, 2]), "(3, 2)");
        assert_eq!(shape_repr(&[3]), "(3,)");
        assert_eq!(shape_repr(&[]), "()");
        assert_eq!(shape_repr(&[1, 2, 3]), "(1, 2, 3)");
    }

    /// `watched` hands the work its stop flag: an interrupt seen by the watcher
    /// reaches a work that polls the flag, which returns early instead of
    /// running out its (here 10 s) course, and the interrupt is the result.
    #[test]
    fn the_watcher_stops_the_work_it_watches() {
        let guard = Arc::new(InterruptAfter {
            ok: 0,
            polls: AtomicUsize::new(0),
        });
        let token = CancelToken::with_interrupt(guard.clone());
        let saw_stop = AtomicBool::new(false);
        let start = Instant::now();
        let outcome = watched(&token, |stop| {
            while start.elapsed() < Duration::from_secs(10) {
                if stop.load(Ordering::Relaxed) {
                    saw_stop.store(true, Ordering::Relaxed);
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        assert!(
            saw_stop.load(Ordering::Relaxed),
            "the work never saw the stop"
        );
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(matches!(
            outcome,
            Err(PrimitiveError::Infrastructure(
                InfrastructureError::Backend(_)
            ))
        ));
        // A cooperative cancellation stops it the same way.
        let token = CancelToken::default();
        token.cancel();
        let outcome = watched(&token, |stop| {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        assert!(matches!(
            outcome,
            Err(PrimitiveError::Infrastructure(
                InfrastructureError::Cancelled
            ))
        ));
        // And a work that ends before the first look is just its result.
        let quick = watched(&CancelToken::default(), |_| 7);
        assert!(matches!(quick, Ok(7)));
    }

    // ── qml.kernel_matrix ────────────────────────────────────────────────────

    /// An IQP-then-`Rx` feature map on 3 qubits and 4 + 3 rows.
    fn kernel_map_and_rows() -> (FeatureMap, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let map = FeatureMap::compile(
            3,
            vec![
                Layer::Iqp(IqpEncoder::new()),
                Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Rx)),
            ],
            3,
        )
        .unwrap();
        let rows = |n: usize, shift: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| (0..3).map(|j| shift + 0.23 * (i * 3 + j) as f64).collect())
                .collect()
        };
        (map, rows(4, 0.1), rows(3, -0.6))
    }

    fn kernel_run(symmetric: bool) -> KernelRun {
        let (map, x1, x2) = kernel_map_and_rows();
        let x2 = (!symmetric).then_some(&x2[..]);
        KernelRun::new(FidelityKernel::new(&map, &x1, x2).unwrap())
    }

    /// The matrix of the compute-uncompute circuits of `kernel`, each run
    /// exactly on its own: the reference the native inner products are
    /// checked against.
    fn compute_uncompute_matrix(kernel: &FidelityKernel) -> Vec<f64> {
        let circuits: Vec<BoundCircuit> = kernel
            .circuits(0..kernel.num_pairs())
            .unwrap()
            .unwrap()
            .into_iter()
            .map(BoundCircuit::Native)
            .collect();
        let probabilities = NativeStatevectorBackend::new(0)
            .run_circuits_exact(&circuits, &run_params(0, 1))
            .unwrap();
        let values: Vec<f64> = probabilities
            .iter()
            .map(|p| kernel.fidelity_from_probabilities(p).unwrap())
            .collect();
        kernel.matrix(&values).unwrap()
    }

    #[test]
    fn a_seeded_kernel_does_not_depend_on_the_block_size() {
        for symmetric in [true, false] {
            let mut whole = kernel_run(symmetric);
            sampled_on_native(&mut whole, 5, usize::MAX);
            assert_eq!(whole.filled, whole.kernel.num_pairs());
            for block in [1, 2, 5] {
                let mut blocked = kernel_run(symmetric);
                sampled_on_native(&mut blocked, 5, block);
                assert_eq!(blocked.matrix, whole.matrix, "block {block}");
            }
            let mut other_seed = kernel_run(symmetric);
            sampled_on_native(&mut other_seed, 6, usize::MAX);
            assert_ne!(other_seed.matrix, whole.matrix);
            // The sampled matrix: each block written at its pairs' entries as
            // it is read, a symmetric one mirrored bit for bit, every entry a
            // frequency.
            let n2 = whole.kernel.num_columns();
            let matrix = whole.into_matrix().unwrap();
            for (k, value) in matrix.iter().enumerate() {
                assert!((0.0..=1.0).contains(value));
                if symmetric {
                    let (i, j) = (k / n2, k % n2);
                    assert_eq!(value.to_bits(), matrix[j * n2 + i].to_bits());
                }
            }
        }
    }

    #[test]
    fn the_exact_kernel_is_the_compute_uncompute_one_whatever_the_blocks() {
        for symmetric in [true, false] {
            let reference = compute_uncompute_matrix(&kernel_run(symmetric).kernel);
            let mut first: Option<Vec<f64>> = None;
            for block in [1, 2, usize::MAX] {
                let mut run = kernel_run(symmetric);
                run.run_exact(block);
                // One call to the native backend computed the whole matrix,
                // whatever the blocks: its result, bit for bit.
                let kernel = &run.kernel;
                let native = NativeStatevectorBackend::new(0)
                    .fidelity_matrix(
                        kernel.row_states(),
                        (!symmetric).then(|| kernel.column_states()),
                        None,
                    )
                    .unwrap();
                let matrix = run.into_matrix().unwrap();
                assert_eq!(matrix, native, "block {block}");
                for (k, (a, b)) in matrix.iter().zip(&reference).enumerate() {
                    // Two exact routes, a few dozen roundings each.
                    assert!((a - b).abs() < 1e-12, "entry {k}: {a} != {b}");
                }
                match &first {
                    None => first = Some(matrix),
                    Some(first) => assert_eq!(first, &matrix, "block {block}"),
                }
            }
        }
    }

    #[test]
    fn exact_rows_in_chunks_concatenate_to_the_whole_matrix() {
        let (map, x1, x2) = kernel_map_and_rows();
        let exact = |rows: &[Vec<f64>]| {
            let mut run = KernelRun::new(FidelityKernel::new(&map, rows, Some(&x2[..])).unwrap());
            run.run_exact(usize::MAX);
            run.into_matrix().unwrap()
        };
        let whole = exact(&x1);
        let chunks: Vec<f64> = [&x1[..1], &x1[1..3], &x1[3..]]
            .into_iter()
            .flat_map(exact)
            .collect();
        assert_eq!(chunks, whole);
    }

    #[test]
    fn a_kernel_runs_its_blocks_on_one_backend_created_and_closed_once() {
        let counters = Counters::new();
        let mut run = kernel_run(true);
        run_in_blocks(
            &mut run,
            || counters.create(),
            |_| 4,
            &CancelToken::default(),
        )
        .unwrap();
        // 4 rows: 10 pairs, so 10 circuits in blocks of 4: three calls.
        assert_eq!(run.filled, 10);
        assert_eq!(counters.get(), (1, 3, 1));
    }

    #[test]
    fn a_kernel_cancelled_or_failing_between_blocks_stops_and_still_closes() {
        let counters = Counters::new();
        let token = CancelToken::default();
        let mut run = Interfering {
            inner: kernel_run(false),
            token: token.clone(),
            fail_second: false,
            built: std::cell::Cell::new(0),
        };
        let err = run_in_blocks(&mut run, || counters.create(), |_| 5, &token).unwrap_err();
        assert!(
            matches!(
                err,
                PrimitiveError::Infrastructure(InfrastructureError::Cancelled)
            ),
            "{err:?}"
        );
        assert_eq!(run.built.get(), 1);
        assert_eq!(run.inner.filled, 5);
        assert_eq!(counters.get(), (1, 1, 1));

        let counters = Counters::new();
        let mut run = Interfering {
            inner: kernel_run(false),
            token: CancelToken::default(),
            fail_second: true,
            built: std::cell::Cell::new(0),
        };
        let err = run_in_blocks(
            &mut run,
            || counters.create(),
            |_| 5,
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
    fn a_kernel_block_outside_its_pairs_is_an_internal_error() {
        let run = kernel_run(true);
        assert!(matches!(
            run.circuits(9..11),
            Err(PrimitiveError::Internal(_))
        ));
        let mut run = kernel_run(true);
        assert!(matches!(
            run.read(0..2, BlockResults::Counts(vec![Counts::new()])),
            Err(PrimitiveError::Qml(QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }))
        ));
    }

    /// A Ctrl+C stops the exact inner products in flight, with the error a
    /// check between blocks gives: the check before them lets them start, then
    /// the watcher's first look interrupts them. The error alone cannot tell a
    /// stop in flight from work that ran to its end and was reported
    /// interrupted afterwards (`watched` returns the reason it saw either
    /// way), so the interrupted call is timed against the same call run to
    /// completion: a stop in flight takes a fraction of it. The work must
    /// outlive the watcher's first look (25 ms): uninterrupted, these 24
    /// states of 18 qubits (a full IQP map) and their 300 inner products took
    /// 3.5 s in a debug `cargo test` on a 16-CPU node, 140 times that. The
    /// comparison needs a run long enough to time: it is made when the whole
    /// run takes at least 16 looks of the watcher (400 ms), as it always does
    /// in debug, which CI runs; an optimized build can finish in less, too
    /// close to the watcher's look and the simulator's own cancellation cadence
    /// for a quarter of it to mean anything, so there it is skipped, and said
    /// so. The error, the polls and the empty results are checked always.
    #[test]
    fn a_ctrl_c_stops_an_exact_kernel_in_flight() {
        let map = FeatureMap::compile(18, vec![Layer::Iqp(IqpEncoder::new())], 18).unwrap();
        let rows: Vec<Vec<f64>> = (0..24)
            .map(|i| (0..18).map(|j| 0.05 * (i + j) as f64).collect())
            .collect();
        let exact = || {
            Ok(BlockRunner::Exact {
                native: NativeStatevectorBackend::new(0),
                params: run_params(0, 1),
            })
        };
        let mut whole = KernelRun::new(FidelityKernel::new(&map, &rows, None).unwrap());
        let started = Instant::now();
        run_in_blocks(&mut whole, exact, |_| usize::MAX, &CancelToken::default()).unwrap();
        let uninterrupted = started.elapsed();
        assert!(whole.matrix.is_some());

        let mut run = KernelRun::new(FidelityKernel::new(&map, &rows, None).unwrap());
        let guard = Arc::new(InterruptAfter {
            ok: 1,
            polls: AtomicUsize::new(0),
        });
        let token = CancelToken::with_interrupt(guard.clone());
        let started = Instant::now();
        let err = run_in_blocks(&mut run, exact, |_| usize::MAX, &token).unwrap_err();
        let interrupted = started.elapsed();
        match err {
            PrimitiveError::Infrastructure(InfrastructureError::Backend(
                crate::infrastructure::BackendError::Aborted(message),
            )) => assert_eq!(message, "test interrupt"),
            other => panic!("expected the interrupt, got {other:?}"),
        }
        assert_eq!(guard.polls.load(Ordering::SeqCst), 2);
        assert!(run.matrix.is_none() && run.filled == 0);
        if uninterrupted < 16 * WATCH_INTERVAL {
            eprintln!(
                "the whole run took {uninterrupted:?}, under {:?}: too short to time a stop \
                 in flight against, so that comparison is skipped",
                16 * WATCH_INTERVAL
            );
            return;
        }
        assert!(
            interrupted < uninterrupted / 4,
            "stopped after {interrupted:?}, the whole run takes {uninterrupted:?}: the \
             work did not see the stop"
        );
    }
}
