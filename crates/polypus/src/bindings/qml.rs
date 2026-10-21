//! The native pieces of `polypus.qml`: the `Encoder`, `Ansatz`, `Objective` and
//! `Dataset` that `qml.train` / `qml.predict` accept in place of Qiskit circuits
//! and a Python cost (plan D4, P-a), plus the Pauli readout helpers `Z`/`X`/`Y`,
//! `PauliTerm` and `Observable`.
//!
//! Each class wraps a `polypus-qml` type and validates as it is built: every
//! option is a string checked against its closed set (plan F6), and every
//! builder call on an `Encoder`/`Ansatz` re-checks the layers that do not need
//! the data (plan F1), so a mistake surfaces at the line that made it rather
//! than inside `train`. What does depend on the data (the feature count) is
//! checked when the pieces meet a dataset.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

use polypus_qml::{
    AngleEncoder, ConvBlock, ConvLayer, Decision, Entanglement, Entangler, HardwareEfficientAnsatz,
    IqpEncoder, KeepRule, Layer, Loss, Observable, Pairing, Pauli, PauliString, PoolBlock,
    PoolLayer, QmlProblem, QuantumModel, Readout, RotationAxis, ValidationError,
};

use super::{extract_labels, seed_or_entropy};
use crate::evaluation::Label;
use crate::infrastructure::execution_config::random_seed;

/// A `polypus-qml` [`ValidationError`] as a Python `ValueError`. A free function
/// because both types are foreign here (the orphan rule forbids a `From`), as
/// `to_py_err` in [`circuit`](super::circuit) does for `CircuitError`.
fn validation_to_py_err(e: ValidationError) -> PyErr {
    PyValueError::new_err(e.to_string())
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
        other => Err(PyValueError::new_err(format!(
            "unknown loss '{other}'; expected \"squared_error\", \"binary_cross_entropy\", \
             \"hinge\", \"categorical_cross_entropy\" or \"expectation\""
        ))),
    }
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
// Objective
// ─────────────────────────────────────────────────────────────────────────────

/// What a native model reads out and how it is scored, in the
/// `expectation_function` slot of `qml.train` and `qml.predict`.
///
/// `Objective(observables, decision="raw", loss=None, threshold=None)`:
///
/// - `observables`: what is measured, each a [`PauliTerm`] (`Z(0)`,
///   `Z(0) @ Z(1)`), an [`Observable`](PyObservable) (a weighted sum) or a
///   `[(pauli, position), …]` list. `"argmax"` reads one per class (at least
///   two); the other decisions predict from the first.
/// - `decision`: how the readout becomes a prediction: `"raw"` (the expectation
///   itself), `"sign"`, `"threshold"` (with `threshold`) or `"argmax"`.
/// - `loss`: how a prediction is scored against its label: `"squared_error"`,
///   `"binary_cross_entropy"`, `"hinge"`, `"categorical_cross_entropy"` (the one
///   loss for `"argmax"`), or `"expectation"`, which needs no labels and
///   maximises the first observable. With labels a loss is required; without
///   them, `None` means `"expectation"` (plan F2).
#[pyclass(module = "polypus.qml", name = "Objective", frozen)]
pub struct Objective {
    pub(crate) readout: Readout,
    pub(crate) loss: Option<Loss>,
}

#[pymethods]
impl Objective {
    #[new]
    #[pyo3(signature = (observables, decision="raw", loss=None, threshold=None))]
    fn new(
        observables: Vec<Bound<'_, PyAny>>,
        decision: &str,
        loss: Option<&str>,
        threshold: Option<f64>,
    ) -> PyResult<Self> {
        let decision = parse_decision(decision, threshold)?;
        let loss = loss.map(parse_loss).transpose()?;
        let observables = observables
            .iter()
            .map(extract_observable)
            .collect::<PyResult<Vec<_>>>()?;
        let readout = Readout::new(observables, decision).map_err(validation_to_py_err)?;
        // `argmax` pairs only with the categorical loss and every other decision
        // only with a scalar one (`QmlProblem::new`'s rule), checked here so the
        // mistake surfaces where it is written. `None` stands for `expectation`
        // (scalar) or a loss still to be named, and `argmax` has a single
        // possible loss, so `argmax` without one is caught here too.
        let effective = loss.unwrap_or(Loss::Expectation);
        if (decision == Decision::Argmax) != (effective == Loss::CategoricalCrossEntropy) {
            return Err(match loss {
                Some(loss) => validation_to_py_err(ValidationError::DecisionNotSupportedByLoss {
                    decision,
                    loss,
                }),
                None => PyValueError::new_err(
                    "decision=\"argmax\" needs loss=\"categorical_cross_entropy\"",
                ),
            });
        }
        Ok(Objective { readout, loss })
    }

    #[getter]
    fn decision(&self) -> &'static str {
        decision_name(self.readout.decision)
    }

    #[getter]
    fn threshold(&self) -> Option<f64> {
        match self.readout.decision {
            Decision::Threshold(threshold) => Some(threshold),
            _ => None,
        }
    }

    #[getter]
    fn loss(&self) -> Option<&'static str> {
        self.loss.map(loss_name)
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.readout.observables.len()
    }

    fn __repr__(&self) -> String {
        let threshold = self
            .threshold()
            .map(|t| format!(", threshold={t}"))
            .unwrap_or_default();
        let loss = self
            .loss()
            .map(|l| format!("\"{l}\""))
            .unwrap_or_else(|| "None".to_string());
        format!(
            "Objective(num_observables={}, decision=\"{}\", loss={loss}{threshold})",
            self.readout.observables.len(),
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
// qml.train: the native form, and the Dataset in the Qiskit form
// ─────────────────────────────────────────────────────────────────────────────

/// What each form takes, for the errors that reject a mix (plan G2).
const FORMS: &str = "the native form takes a polypus.qml.Encoder as feature_map, a \
                     polypus.qml.Ansatz as ansatz and a polypus.qml.Objective as \
                     expectation_function; the Qiskit form takes QuantumCircuits and a \
                     callable, CachedCost, SampleCost, Qubo or Ising";

/// The layers of a native `feature_map`/`ansatz` pair.
pub(crate) struct NativePieces {
    encoder: Layers,
    ansatz: Layers,
}

/// The native pieces if `feature_map` is an [`Encoder`] and `ansatz` an
/// [`Ansatz`], `None` if neither is native (the Qiskit form), and a
/// `ValueError` for anything in between: one native piece and one circuit, or
/// the two native pieces swapped.
pub(crate) fn native_pieces(
    feature_map: &Bound<'_, PyAny>,
    ansatz: &Bound<'_, PyAny>,
) -> PyResult<Option<NativePieces>> {
    let is_native =
        |obj: &Bound<'_, PyAny>| obj.is_instance_of::<Encoder>() || obj.is_instance_of::<Ansatz>();
    match (feature_map.cast::<Encoder>(), ansatz.cast::<Ansatz>()) {
        (Ok(encoder), Ok(trainable)) => Ok(Some(NativePieces {
            encoder: encoder.try_borrow()?.stack.clone(),
            ansatz: trainable.try_borrow()?.stack.clone(),
        })),
        _ if !is_native(feature_map) && !is_native(ansatz) => Ok(None),
        _ => Err(PyValueError::new_err(format!(
            "feature_map and ansatz must belong to the same form: {FORMS}"
        ))),
    }
}

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

/// The training set of the native form and whether it has labels: a
/// [`Dataset`](PyDataset) as it is (then `y_train` is an error), or rows plus
/// an optional `y_train`, read with the same label checks as the Qiskit form.
fn native_dataset(
    x_train: &Bound<'_, PyAny>,
    y_train: Option<&Bound<'_, PyAny>>,
) -> PyResult<(polypus_qml::Dataset, bool)> {
    if let Ok(dataset) = x_train.cast::<PyDataset>() {
        if y_train.is_some() {
            return Err(dataset_and_labels_err());
        }
        let dataset = dataset.try_borrow()?;
        return Ok((dataset.inner.clone(), dataset.labeled));
    }
    let rows: Vec<Vec<f64>> = x_train.extract()?;
    let labels = y_train
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
    let dataset = polypus_qml::Dataset::from_rows(&rows, &labels).map_err(validation_to_py_err)?;
    Ok((dataset, labeled))
}

fn dataset_and_labels_err() -> PyErr {
    PyValueError::new_err(
        "x_train is a polypus.qml.Dataset, which carries its own labels; pass y_train only \
         with plain rows",
    )
}

/// The loss a run trains with (plans F2 and F3): with labels, the Objective's
/// supervised loss, which it must name; without labels, `expectation`. Every
/// other pairing is a `ValueError`, so labels are never ignored and a
/// supervised loss never trains against made-up labels.
fn resolve_loss(loss: Option<Loss>, labeled: bool) -> PyResult<Loss> {
    match (loss, labeled) {
        (None | Some(Loss::Expectation), false) => Ok(Loss::Expectation),
        (Some(loss), false) => Err(PyValueError::new_err(format!(
            "loss=\"{}\" needs labels: pass y_train (or a labeled Dataset), or use \
             loss=\"expectation\" to maximise the readout",
            loss_name(loss)
        ))),
        (None, true) => Err(PyValueError::new_err(
            "the training set has labels, so the Objective needs a loss, e.g. \
             Objective([Z(0)], decision=\"sign\", loss=\"hinge\")",
        )),
        (Some(Loss::Expectation), true) => Err(PyValueError::new_err(
            "loss=\"expectation\" does not use labels: drop y_train (or use a Dataset \
             without labels), or choose a supervised loss",
        )),
        (Some(loss), true) => Ok(loss),
    }
}

/// The native form's problem: `Encoder` and `Ansatz` joined into one model
/// read out by the [`Objective`], compiled against the training set and scored
/// with the resolved loss. Every check runs here, before any backend exists.
pub(crate) fn native_problem(
    pieces: NativePieces,
    x_train: &Bound<'_, PyAny>,
    y_train: Option<&Bound<'_, PyAny>>,
    expectation_function: &Bound<'_, PyAny>,
) -> PyResult<Arc<QmlProblem>> {
    let objective = expectation_function.cast::<Objective>().map_err(|_| {
        PyTypeError::new_err(format!(
            "with a native Encoder and Ansatz, expectation_function must be a \
             polypus.qml.Objective ({FORMS})"
        ))
    })?;
    let objective = objective.get();
    let NativePieces { encoder, ansatz } = pieces;
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
    let (dataset, labeled) = native_dataset(x_train, y_train)?;
    let loss = resolve_loss(objective.loss, labeled)?;
    let compiled = encoder
        .layers
        .into_iter()
        .chain(ansatz.layers)
        .fold(QuantumModel::new(encoder.num_qubits), QuantumModel::layer)
        .readout(objective.readout.clone())
        .compile(dataset.num_features())
        .map_err(validation_to_py_err)?;
    let problem = QmlProblem::new(compiled, dataset, loss).map_err(validation_to_py_err)?;
    Ok(Arc::new(problem))
}

/// The Qiskit form's rejection of an [`Objective`] (plan G2).
pub(crate) fn reject_objective(expectation_function: &Bound<'_, PyAny>) -> PyResult<()> {
    if expectation_function.is_instance_of::<Objective>() {
        return Err(PyTypeError::new_err(format!(
            "a polypus.qml.Objective needs a native Encoder and Ansatz ({FORMS})"
        )));
    }
    Ok(())
}

/// A [`Dataset`](PyDataset) read for the Qiskit form: its rows as a Python list
/// to bind to the feature map, and its labels if it has them.
pub(crate) type QiskitDataset<'py> = (Bound<'py, PyList>, Option<Vec<Label>>);

/// A [`Dataset`](PyDataset) in the Qiskit form's `x_train` (plan G1), or `None`
/// when `x_train` is not one. Labels that are all whole numbers reach the
/// objective as `int` class labels, otherwise all as `float`, the rule
/// `y_train` follows.
pub(crate) fn qiskit_dataset<'py>(
    x_train: &Bound<'py, PyAny>,
    has_y_train: bool,
) -> PyResult<Option<QiskitDataset<'py>>> {
    let Ok(dataset) = x_train.cast::<PyDataset>() else {
        return Ok(None);
    };
    if has_y_train {
        return Err(dataset_and_labels_err());
    }
    let dataset = dataset.try_borrow()?;
    let inner = &dataset.inner;
    let rows = PyList::empty(x_train.py());
    for i in 0..inner.num_samples() {
        rows.append(inner.sample(i).to_vec())?;
    }
    let labels = dataset.labeled.then(|| {
        let values = inner.labels();
        // Below 2^53 every whole `f64` converts to `i64` exactly.
        let whole = values
            .iter()
            .all(|v| v.fract() == 0.0 && v.abs() < 2f64.powi(53));
        values
            .iter()
            .map(|&v| {
                if whole {
                    Label::Class(v as i64)
                } else {
                    Label::Real(v)
                }
            })
            .collect()
    });
    Ok(Some((rows, labels)))
}

/// Add the native classes and the Pauli helpers to the `polypus.qml` module.
pub(crate) fn register(qml: &Bound<'_, PyModule>) -> PyResult<()> {
    qml.add_class::<Encoder>()?;
    qml.add_class::<Ansatz>()?;
    qml.add_class::<Objective>()?;
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
}
