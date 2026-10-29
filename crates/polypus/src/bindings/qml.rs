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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{
    PyFileNotFoundError, PyOSError, PyPermissionError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};
use polypus_qml::{
    AngleEncoder, CompiledModel, ConvBlock, ConvLayer, Decision, Entanglement, Entangler,
    HardwareEfficientAnsatz, IqpEncoder, KeepRule, Layer, Loss, Observable, Pairing, Pauli,
    PauliString, PoolBlock, PoolLayer, QmlError, QmlProblem, QuantumModel, Readout, RotationAxis,
    ValidationError,
};

use super::circuit::Circuit;
use super::{
    build_backend_config, counts_to_pylist, extract_labels, finish_optimization,
    interruptible_token, is_native_backend, is_rotosolve, log_training_start, method_from_pyclass,
    method_seed, resolve_optimizer_seed, seed_or_entropy, unique_id, validate_cunqa_allocation,
    validate_id, validate_shots_and_qpus, OptionsArg, RunResult,
};
use crate::evaluation::{Label, MinibatchConfig, OracleErrorSlot, QmlOracleFactory, QmlSource};
use crate::exceptions::configuration_error;
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{
    BoundCircuit, Counts, ExecutionConfig, Infrastructure, InfrastructureError,
    NativeStatevectorBackend, OptLevel,
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
    /// - **Execution** takes the arguments of `polypus.train`. `backend=None`
    ///   is the native simulator on `infrastructure="local"` and Aer otherwise
    ///   (CUNQA's QPUs run Aer); an explicit `backend` wins. `seed` drives the
    ///   optimizer, the shots and the minibatches (C-7), and `id` is a prefix to
    ///   which a UUID is appended (C-9).
    ///
    /// Every check runs before a backend exists and raises
    /// `polypus.ConfigurationError` (a `ValueError`). The fitness is the
    /// negated mean loss over the samples, which the optimizers maximise.
    #[pyo3(signature = (
        x, y=None, *, method, loss=None, shots=1024, infrastructure="local", n_qpus=1,
        nodes=1, cores_per_qpu=2, walltime=None, options=None, sim_method="automatic",
        noise_model=None, backend=None, seed=None, id="qml", exact=false, batch_size=None,
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
    ) -> PyResult<Py<PyAny>> {
        let start = Instant::now();
        let py = x.py();
        validate_shots_and_qpus(shots, n_qpus)?;
        validate_cunqa_allocation(infrastructure, nodes, cores_per_qpu, walltime.as_deref())?;
        validate_id(id)?;
        let loss = loss.map(parse_loss).transpose()?;
        let effective_seed = resolve_optimizer_seed(seed, method_seed(method), random_seed)?;
        let backend = resolve_backend(backend, infrastructure);
        if exact && !(infrastructure == "local" && is_native_backend(backend)) {
            return Err(configuration_error(
                "exact=True needs infrastructure=\"local\" and backend=\"polypus\": only the \
                 native simulator computes exact probabilities",
            ));
        }

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
        let dimensions = u32::try_from(self.num_params).map_err(|_| {
            configuration_error(format!(
                "the model has {} trainable parameters, more than an optimizer can search",
                self.num_params
            ))
        })?;
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
        let errors = OracleErrorSlot::new();
        let method = method_from_pyclass(method, dimensions, &errors, &effective_id)?;
        let resources = Resources::new(backend, None, Arc::new(config.run_params()))
            .map_err(crate::exceptions::infrastructure_error_to_pyerr)?;
        let scheduler = Scheduler::ephemeral(resources);
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
            initial_point: None,
            final_shots: None,
        };
        // GIL released for the run, as in `train`; the interrupt token keeps
        // Ctrl+C prompt between waves.
        let token = interruptible_token();
        let result = py.detach(|| scheduler.run_cancellable(flow, &token));
        scheduler.close();
        finish_optimization(py, result, effective_seed, effective_id, start)
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

/// The execution arguments `TrainedModel.predict` and `run` share.
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
        validate_shots_and_qpus(exec.shots, exec.n_qpus)?;
        validate_cunqa_allocation(
            exec.infrastructure,
            exec.nodes,
            exec.cores_per_qpu,
            exec.walltime.as_deref(),
        )?;
        let backend = resolve_backend(exec.backend, exec.infrastructure);
        if exec.exact && !(exec.infrastructure == "local" && is_native_backend(backend)) {
            return Err(configuration_error(
                "exact=True needs infrastructure=\"local\" and backend=\"polypus\": only the \
                 native simulator computes exact probabilities",
            ));
        }
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
        // As in `run_quantum_circuit` (C-7): `qmio` is hardware, so a seed there
        // would be silently ineffective.
        let seed = if exec.infrastructure == "qmio" {
            if exec.seed.is_some() {
                return Err(configuration_error(
                    "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
                ));
            }
            None
        } else {
            Some(seed_or_entropy(exec.seed, random_seed)?)
        };
        let backend_config = build_backend_config(
            exec.infrastructure,
            backend,
            exec.sim_method,
            exec.noise_model.map(|nm| nm.unbind()),
            exec.nodes,
            exec.cores_per_qpu,
            exec.walltime.as_deref(),
            None,
            exec.options.map(OptionsArg::into_inner),
        )?;
        let id = unique_id(&format!("qml_{}_{}", exec.n_qpus, exec.infrastructure))?;
        let config = ExecutionConfig {
            id: id.clone(),
            shots: exec.shots,
            n_qpus: exec.n_qpus,
            infrastructure: exec.infrastructure.to_string(),
            backend_config,
            opt_level: OptLevel::default(),
            seed,
        };
        log::info!(
            "qml inference {id} starting: rows={}, infrastructure={}, backend={backend}, \
             n_qpus={}, shots={}, exact={}, seed={seed:?}",
            circuits.len(),
            exec.infrastructure,
            exec.n_qpus,
            exec.shots,
            exec.exact
        );
        let shots = exec.shots;
        let results = if exec.exact {
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
                let resources = Resources::new(backend, None, Arc::new(config.run_params()))?;
                let scheduler = Scheduler::ephemeral(resources);
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
            infrastructure: exec.infrastructure.to_string(),
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
