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
use std::collections::HashMap;
use std::sync::Arc;

use polypus_qml::{
    ConvBlock, ConvLayer, Decision, Entanglement, Entangler, HardwareEfficientAnsatz, IqpEncoder,
    KeepRule, Layer, Loss, Observable, Pairing, Pauli, PauliString, PoolBlock, PoolLayer,
    QmlProblem, QuantumModel, Readout, RotationAxis, ValidationError,
};

use super::{
    build_backend_config, finish_optimization, is_native_backend, method_seed,
    resolve_optimizer_seed, unique_id, validate_cunqa_allocation, validate_shots_and_qpus,
    TrainResult,
};
use crate::bindings::adam::Adam;
use crate::bindings::de::DE;
use crate::bindings::pso::PSO;
use crate::bindings::qng::{PyVarianceOracle, QNG};
use crate::evaluation::{
    EvaluationOracle, ExactNativeQmlOracle, MinibatchConfig, NativeQmlOracle, OracleErrorSlot,
    QmlOracle,
};
use crate::infrastructure::execution_config::random_seed;
use crate::infrastructure::{BackendError, BoundCircuit};
use crate::infrastructure::{ExecutionConfig, Infrastructure, NativeStatevectorBackend, OptLevel};
use polypus_optimizers::{
    AlgorithmAdam, AlgorithmAdamArgs, AlgorithmDifferentialEvolution,
    AlgorithmDifferentialEvolutionArgs, AlgorithmPSO, AlgorithmPSOArgs, AlgorithmQNG,
    AlgorithmQNGArgs, GradientOracle, OptimizationOutcome, Optimizer, OptimizerError,
};

/// The final-fitness recompute (design doc §17): re-score a parameter vector
/// against the **full** training set once optimization ends, replacing a
/// minibatched `best_fitness`. Aliased so the boxed owner, the borrowed
/// dispatcher argument and the oracle-branch tuple all name one unsized type.
type RecomputeFn = dyn Fn(&[f64]) -> f64;

/// The three things the native oracle branch hands to [`dispatch_optimizer`]:
/// the two trait-object facets of the one oracle `Arc` (fitness + gradient) and
/// the optional full-dataset recompute (design doc §17). Aliased to keep the
/// `let` binding's type readable.
type NativeOracleParts = (
    Box<dyn EvaluationOracle>,
    Box<dyn GradientOracle>,
    Option<Box<RecomputeFn>>,
);

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

/// Parse an entangling-gate string into an [`Entangler`]. Strict, like
/// [`parse_axis`]: an unrecognised value is a `ValueError` listing the options.
fn parse_entangler(entangler: &str) -> PyResult<Entangler> {
    match entangler {
        "cx" => Ok(Entangler::Cx),
        "cz" => Ok(Entangler::Cz),
        other => Err(PyValueError::new_err(format!(
            "unknown entangler '{other}'; expected \"cx\" or \"cz\""
        ))),
    }
}

/// Parse an entanglement-pattern string into an [`Entanglement`]. Shared by the
/// hardware-efficient ansatz and the IQP encoder, which both take one.
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

/// Parse a convolution-block string into a [`ConvBlock`].
fn parse_conv_block(block: &str) -> PyResult<ConvBlock> {
    match block {
        "basic" => Ok(ConvBlock::Basic),
        "cartan" => Ok(ConvBlock::Cartan),
        other => Err(PyValueError::new_err(format!(
            "unknown conv block '{other}'; expected \"basic\" or \"cartan\""
        ))),
    }
}

/// Parse a convolution pairing string into a [`Pairing`].
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

/// Parse a pooling-block string into a [`PoolBlock`].
///
/// A `match` even though [`PoolBlock`] has a single variant today: written open
/// to the variants the design doc anticipates, so adding one is a new arm rather
/// than a rewrite from `if` to `match`.
fn parse_pool_block(block: &str) -> PyResult<PoolBlock> {
    match block {
        "basic" => Ok(PoolBlock::Basic),
        other => Err(PyValueError::new_err(format!(
            "unknown pool block '{other}'; expected \"basic\""
        ))),
    }
}

/// Parse a pooling keep-rule string into a [`KeepRule`].
fn parse_keep_rule(keep: &str) -> PyResult<KeepRule> {
    match keep {
        "even_positions" => Ok(KeepRule::EvenPositions),
        "odd_positions" => Ok(KeepRule::OddPositions),
        other => Err(PyValueError::new_err(format!(
            "unknown keep rule '{other}'; expected \"even_positions\" or \"odd_positions\""
        ))),
    }
}

/// Parse a single-qubit Pauli string into a [`Pauli`]. Accepts `"x"`/`"y"`/`"z"`;
/// `compile` validates the readout's measurement bases itself (an X/Y readout is
/// supported when the whole readout resolves to a single basis group, else a
/// typed `ObservableHasIncompatibleBases`/`ReadoutNeedsMultipleBasisGroups`), so
/// this does not duplicate that check (decision D).
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

/// Parse a list of `(pauli, position)` factors into a [`PauliString`].
///
/// The one place the Python spelling of a Pauli string is decoded: both
/// [`Model::readout`]'s bare form and [`PyObservable::new`]'s per-term lists go
/// through it, so the two can never drift apart in what they accept or in the
/// error they raise (an unknown Pauli from [`parse_pauli`], a repeated position
/// from `PauliString::new`).
fn parse_pauli_string(factors: Vec<(String, usize)>) -> PyResult<PauliString> {
    let mut terms = Vec::with_capacity(factors.len());
    for (pauli, position) in factors {
        terms.push((position, parse_pauli(&pauli)?));
    }
    PauliString::new(terms).map_err(validation_to_py_err)
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

/// A weighted sum of Pauli strings, `O = Σ cᵢ·Pᵢ`, mirroring `polypus-qml`'s
/// [`Observable`] for Python.
///
/// This is the **additive** way to spell a readout observable: `Model.readout`
/// still accepts the bare `[(pauli, position), …]` list it always has (one Pauli
/// string, coefficient `1.0`), and this type is what a caller reaches for when
/// they need more than one term — `0.5·Z₀ + 1.5·Z₀Z₁`, say, which the bare form
/// cannot express. The Rust `Observable` has supported weighted sums since
/// phase 1; only the Python spelling was missing (design doc §17).
///
/// The constructor takes `[(coefficient, term), …]`, where each `term` is
/// exactly the same `(pauli, position)` list `readout` accepts on its own — so
/// `Observable([(1.0, [("z", 0)])])` is the explicit spelling of the bare
/// `[("z", 0)]`, and builds the identical observable.
///
/// A non-finite coefficient (`NaN`/`inf`) is a `ValueError`
/// (`ValidationError::NonFiniteCoefficient`), never a panic; so is an unknown
/// Pauli or a position repeated inside one term.
///
/// The name is `PyObservable` in Rust only because `Observable` here is the
/// `polypus-qml` type it wraps; Python sees `polypus.qml.Observable`.
#[pyclass(module = "polypus.qml", name = "Observable")]
pub struct PyObservable {
    inner: Observable,
}

#[pymethods]
impl PyObservable {
    #[new]
    fn new(terms: Vec<(f64, Vec<(String, usize)>)>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(terms.len());
        for (coefficient, factors) in terms {
            parsed.push((coefficient, parse_pauli_string(factors)?));
        }
        let inner = Observable::new(parsed).map_err(validation_to_py_err)?;
        Ok(PyObservable { inner })
    }

    fn __repr__(&self) -> String {
        format!("Observable(num_terms={})", self.inner.terms.len())
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
///
/// **A `Model` instance is reusable without limit** — this is a deliberate
/// property, not an accident of the `Option` dance above. Compiling a model
/// (inside [`train`](qml_train) and [`TrainedModel::new`]) always clones
/// `self.inner` before the consuming [`QuantumModel::compile`] call, so the
/// original Python object is never consumed: the same `Model` can be trained
/// more than once, wrapped in more than one `TrainedModel`, and even extended
/// with further builder calls (`.layer(...)`, `.hardware_efficient(...)`, …)
/// after any of that — each use starts from an independent clone of whatever
/// the builder currently holds.
#[pyclass(module = "polypus.qml", name = "Model")]
pub struct Model {
    inner: Option<QuantumModel>,
}

impl Model {
    /// Apply one consuming [`QuantumModel`] builder call in place: take the model
    /// out, hand it to `build`, and put the result back.
    ///
    /// Every builder method funnels through here, so the `inner: Option<_>`
    /// dance — and the single place where the "always `Some` between calls"
    /// invariant is asserted — is written once instead of once per method.
    fn apply(slf: &mut Self, build: impl FnOnce(QuantumModel) -> QuantumModel) {
        let model = slf
            .inner
            .take()
            .expect("Model.inner is always Some between calls");
        slf.inner = Some(build(model));
    }
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
        Model::apply(&mut slf, |model| model.angle_encoder(axis));
        Ok(slf)
    }

    /// Append an amplitude encoder, which loads a sample into the amplitudes of
    /// the state.
    ///
    /// It takes no configuration, and must be the model's **first** layer over
    /// enough qubits for the feature count (`num_features <= 2^num_qubits`);
    /// `compile` enforces both, so a misplaced or undersized one is a `ValueError`
    /// there rather than a check duplicated at this boundary.
    fn amplitude_encoder(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        Model::apply(&mut slf, |model| model.amplitude_encoder());
        Ok(slf)
    }

    /// Append an IQP / `ZZFeatureMap` feature encoder over the `entanglement`
    /// pattern (`"linear"`/`"circular"`/`"full"`).
    ///
    /// The `"full"` default is the original `ZZFeatureMap` connectivity, i.e. the
    /// one `IqpEncoder::new` picks. The encoder consumes no `θ`.
    #[pyo3(signature = (entanglement="full"))]
    fn iqp_encoder<'py>(
        mut slf: PyRefMut<'py, Self>,
        entanglement: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let encoder = IqpEncoder {
            entanglement: parse_entanglement(entanglement)?,
        };
        Model::apply(&mut slf, |model| model.layer(Layer::Iqp(encoder)));
        Ok(slf)
    }

    /// Append a QCNN convolution layer: one shared `block` (`"basic"`/`"cartan"`)
    /// applied to every pair of active qubits chosen by `pairing`
    /// (`"even_pairs"`/`"odd_pairs"`/`"alternating"`).
    ///
    /// Parameters are **shared** across pairs, so the layer's `θ` count depends
    /// on the block alone (4 for `"basic"`, 3 for `"cartan"`), never on the qubit
    /// count. `"alternating"` — all even pairs then all odd — is the
    /// Cong–Choi–Lukin default, matching `ConvLayer::new`.
    #[pyo3(signature = (block, pairing="alternating"))]
    fn conv<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        pairing: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = ConvLayer {
            block: parse_conv_block(block)?,
            pairing: parse_pairing(pairing)?,
        };
        Model::apply(&mut slf, |model| model.layer(Layer::Conv(layer)));
        Ok(slf)
    }

    /// Append a QCNN unitary pooling layer: adjacent pairs of active qubits are
    /// each reduced to one, `keep` (`"even_positions"`/`"odd_positions"`) deciding
    /// which position of each pair survives.
    ///
    /// The discarded qubit's information flows into the retained one through the
    /// shared `block` (`"basic"`), after which it leaves the active set and
    /// receives no further gate — pooling without the mid-circuit measurement
    /// that contract C-4 forbids. `"even_positions"` (retain the lower position)
    /// is the default, matching `PoolLayer::new`.
    #[pyo3(signature = (block, keep="even_positions"))]
    fn pool<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        keep: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer = PoolLayer {
            block: parse_pool_block(block)?,
            keep: parse_keep_rule(keep)?,
        };
        Model::apply(&mut slf, |model| model.layer(Layer::Pool(layer)));
        Ok(slf)
    }

    /// Append a hardware-efficient ansatz with `reps` repetitions.
    ///
    /// All five of the Rust struct's fields are configurable, and every default
    /// is the `TwoLocal` default `HardwareEfficientAnsatz::new` uses — so
    /// `hardware_efficient(reps)` alone behaves exactly as it did before the
    /// kwargs existed: `rotations=["ry","rz"]`, `entangler="cx"`,
    /// `entanglement="linear"`, `final_rotation_layer=True`.
    ///
    /// `rotations` is a list of axes emitted per rotation block, in order
    /// (axis-major, qubit-minor, like Qiskit's `TwoLocal`). An empty list
    /// reserves no `θ` here, which `compile` catches model-wide as
    /// `NoTrainableParams` if no other layer contributes any — so, as with the
    /// readout's bases, that check is not duplicated at this boundary.
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
            // `None` means "keep the default", read off the Rust constructor
            // itself so the two can never drift apart.
            None => HardwareEfficientAnsatz::new(reps).rotations,
        };
        let ansatz = HardwareEfficientAnsatz {
            reps,
            rotations,
            entangler: parse_entangler(entangler)?,
            entanglement: parse_entanglement(entanglement)?,
            final_rotation_layer,
        };
        Model::apply(&mut slf, |model| {
            model.layer(Layer::HardwareEfficient(ansatz))
        });
        Ok(slf)
    }

    /// Append the `RealAmplitudes` preset: a single `Ry` per rotation block,
    /// linear `Cx` entanglement and a final rotation layer.
    ///
    /// A separate method rather than a flag on
    /// [`hardware_efficient`](Self::hardware_efficient), mirroring
    /// `HardwareEfficientAnsatz::real_amplitudes`: it is a preset that *fixes*
    /// four of the five fields, so exposing it as a kwarg would mean kwargs that
    /// silently override each other.
    fn real_amplitudes(mut slf: PyRefMut<'_, Self>, reps: usize) -> PyResult<PyRefMut<'_, Self>> {
        let ansatz = HardwareEfficientAnsatz::real_amplitudes(reps);
        Model::apply(&mut slf, |model| {
            model.layer(Layer::HardwareEfficient(ansatz))
        });
        Ok(slf)
    }

    /// Attach the readout: the `observables` to measure plus the `decision` rule.
    ///
    /// Each observable is **either** a bare list of `(pauli, position)` factors —
    /// one Pauli string with implicit coefficient `1.0`, the form this method has
    /// always accepted — **or** a [`polypus.qml.Observable`](PyObservable), the
    /// weighted sum `Σ cᵢ·Pᵢ` (design doc §17). The two are distinguished by
    /// *type*, never by guessing at the shape of the tuples: an element that is
    /// an `Observable` instance is used as such, and anything else is extracted
    /// as the bare form. The two forms may be mixed freely in one call — a
    /// multiclass `"argmax"` can spell one class bare and another weighted.
    ///
    /// `decision` is `"sign"`/`"threshold"`/`"argmax"`/`"raw"`; `threshold` is
    /// required for `"threshold"` and rejected otherwise.
    #[pyo3(signature = (observables, decision, threshold=None))]
    fn readout<'py>(
        mut slf: PyRefMut<'py, Self>,
        observables: Vec<Bound<'py, PyAny>>,
        decision: &str,
        threshold: Option<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let decision = parse_decision(decision, threshold)?;
        let mut parsed = Vec::with_capacity(observables.len());
        for observable in observables {
            // Type dispatch, in the same spirit as `qml.train`'s first-argument
            // dispatch: try the dedicated type first, fall back to the bare form.
            let observable = match observable.extract::<PyRef<'_, PyObservable>>() {
                Ok(weighted) => weighted.inner.clone(),
                Err(_) => {
                    let factors = observable.extract::<Vec<(String, usize)>>().map_err(|_| {
                        PyTypeError::new_err(
                            "each observable must be a list of (pauli, position) factors \
                             (e.g. [(\"z\", 0)]) or a polypus.qml.Observable",
                        )
                    })?;
                    let string = parse_pauli_string(factors)?;
                    // The bare form is exactly one term with coefficient 1.0.
                    Observable::new(vec![(1.0, string)]).map_err(validation_to_py_err)?
                }
            };
            parsed.push(observable);
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

    /// Split into `(train, test)`, shuffling sample indices with `seed` and
    /// cutting `test_fraction` of them off for the test set.
    ///
    /// `test_fraction` must lie in the **open** interval `(0, 1)` — either
    /// endpoint would leave a partition empty — and the test partition rounds
    /// **down** (`floor(num_samples * test_fraction)`). Both partitions are fresh
    /// `Dataset`s; this one is left untouched.
    ///
    /// `seed` follows the same convention as everywhere else at this seam (C-7):
    /// an explicit value makes the split reproducible byte for byte, and omitting
    /// it draws a fresh OS-entropy seed — the resolution the pure crate
    /// deliberately leaves to the bindings layer, which is why its own
    /// `train_test_split` takes a mandatory `u64`.
    #[pyo3(signature = (test_fraction, seed=None))]
    fn train_test_split(&self, test_fraction: f64, seed: Option<u64>) -> PyResult<(Self, Self)> {
        let (train, test) = self
            .inner
            .train_test_split(test_fraction, seed.unwrap_or_else(random_seed))
            .map_err(validation_to_py_err)?;
        Ok((Dataset { inner: train }, Dataset { inner: test }))
    }

    /// Min–max scale every feature **in place** into `[lo, hi]`, using ranges
    /// computed over this dataset. A constant feature (min equals max) has no
    /// range to normalize against and maps to `lo`. The convention recommended
    /// for angle encoding is `[0, π]`.
    fn scale_features_to(&mut self, lo: f64, hi: f64) {
        self.inner.scale_features_to(lo, hi);
    }

    /// The current `(min, max)` of each feature, in feature order — the scaler to
    /// freeze on a train set and replay on a test set with
    /// [`scale_features_with`](Self::scale_features_with).
    fn feature_ranges(&self) -> Vec<(f64, f64)> {
        self.inner.feature_ranges()
    }

    /// Apply the min–max scaling described by `ranges` **in place**, mapping into
    /// `[lo, hi]`.
    ///
    /// `ranges` is typically a train set's [`feature_ranges`](Self::feature_ranges)
    /// replayed on a test set so both are scaled identically; its length must
    /// equal `num_features`, otherwise a `ValueError`. Test values outside the
    /// supplied range map linearly and may land outside `[lo, hi]` — the intended
    /// behaviour of a frozen scaler, not an error.
    fn scale_features_with(&mut self, ranges: Vec<(f64, f64)>, lo: f64, hi: f64) -> PyResult<()> {
        self.inner
            .scale_features_with(&ranges, lo, hi)
            .map_err(validation_to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(num_samples={}, num_features={})",
            self.inner.num_samples(),
            self.inner.num_features()
        )
    }
}

/// A trained model saved for inference: a compiled `polypus.qml.Model` bound to
/// its optimal parameters `theta`, mirroring `polypus-qml`'s [`TrainedModel`].
///
/// Constructed from a `Model`, the `Dataset` it was trained on (its feature
/// count is what the model compiles against) and the optimizer's `best_params`.
/// [`save`](Self::save) writes it as JSON and [`load`](Self::load) reads it back;
/// the file carries only `{spec, num_features}` for the model, which
/// `polypus-qml` recompiles on load, so a corrupt or tampered file surfaces as a
/// `ValueError` rather than an inconsistent model (design doc §17).
///
/// [`predict`](Self::predict) is the end-to-end inference path: given a batch of
/// new samples it binds each to `θ`, runs them on a backend, and applies the
/// model's readout decision — all in one call.
/// [`predict_from_counts`](Self::predict_from_counts) is the lower-level entry
/// for a caller who obtained counts on their own.
///
/// [`TrainedModel`]: polypus_qml::TrainedModel
#[pyclass(module = "polypus.qml", name = "TrainedModel")]
pub struct TrainedModel {
    inner: polypus_qml::TrainedModel,
}

#[pymethods]
impl TrainedModel {
    #[new]
    fn new(
        model: PyRef<'_, Model>,
        dataset: PyRef<'_, Dataset>,
        theta: Vec<f64>,
    ) -> PyResult<Self> {
        let compiled = model
            .inner
            .as_ref()
            .expect("Model.inner is always Some between calls")
            .clone()
            .compile(dataset.inner.num_features())
            .map_err(validation_to_py_err)?;
        Ok(TrainedModel {
            inner: polypus_qml::TrainedModel {
                model: compiled,
                theta,
            },
        })
    }

    /// Write the trained model to `path` as pretty-printed JSON.
    fn save(&self, path: &str) -> PyResult<()> {
        let json = serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a trained model from a JSON file at `path`. The model is recompiled
    /// from its `{spec, num_features}`, so a malformed or tampered file fails
    /// here with a `ValueError` instead of yielding an inconsistent model.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let inner: polypus_qml::TrainedModel =
            serde_json::from_str(&contents).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TrainedModel { inner })
    }

    /// The optimal trainable parameters `θ`.
    #[getter]
    fn theta(&self) -> Vec<f64> {
        self.inner.theta.clone()
    }

    /// Infer a prediction from one sample's measurement `counts`, applying the
    /// model's readout decision (design doc §7.1). The caller supplies counts
    /// they obtained by running the bound circuit themselves.
    fn predict_from_counts(&self, counts: HashMap<String, u64>) -> PyResult<f64> {
        self.inner
            .model
            .predict_from_counts(&counts)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Infer a prediction from one sample's exact basis-state `probabilities` —
    /// the exact-mode mirror of
    /// [`predict_from_counts`](Self::predict_from_counts) (design doc §17).
    ///
    /// The lower-level entry for a caller who obtained exact probabilities on
    /// their own, as [`predict(..., exact=True)`](Self::predict) does internally.
    /// Since counts enter the readout only through their relative frequencies, a
    /// `probabilities` dict and a `counts` dict with the same distribution yield
    /// the same prediction.
    fn predict_from_probabilities(&self, probabilities: HashMap<String, f64>) -> PyResult<f64> {
        self.inner
            .model
            .predict_from_probabilities(&probabilities)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// End-to-end inference (design doc §17): predict on a batch of **new**
    /// samples `x` in one call — bind each to `θ`, run it on a backend, and apply
    /// the model's readout decision. `x` is always a list of samples
    /// (`List[List[float]]`), never a flat vector, so its shape is unambiguous.
    ///
    /// Seed resolution and backend construction follow
    /// [`run_quantum_circuit`](super::run_quantum_circuit), not the optimizer
    /// paths: there is no "method" object here, so the seed is resolved directly
    /// (`seed.unwrap_or_else(random_seed)`, with `infrastructure="qmio"` rejecting
    /// an explicit seed), and the backend is built the usual way
    /// (`build_backend_config` + `Infrastructure::create_backend`).
    ///
    /// `exact=True` reuses the **same** guard as
    /// [`qml.train`](super::qml_train): it requires `infrastructure="local"` and
    /// the native `backend="polypus"`, and reads exact basis-state probabilities
    /// straight off the statevector (`shots`/`seed` are then irrelevant — two
    /// calls are byte-identical). Any other combination rejects `exact=True` with
    /// a clear error rather than silently ignoring it.
    ///
    /// A sample with the wrong number of features surfaces as a `ValueError`
    /// (`QmlError::FeatureCountMismatch` from `bind`), never a panic. Predictions
    /// come back in the same order as `x`.
    #[pyo3(signature = (
        x, shots=1024, n_qpus=1, infrastructure="local".to_string(), nodes=1,
        cores_per_qpu=1, id="qml_predict".to_string(), sim_method="automatic",
        noise_model=None, backend="aer", seed=None, exact=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        shots: u32,
        n_qpus: u32,
        infrastructure: String,
        nodes: u32,
        cores_per_qpu: u32,
        id: String,
        sim_method: &str,
        noise_model: Option<Bound<'_, PyAny>>,
        backend: &str,
        seed: Option<u64>,
        exact: bool,
    ) -> PyResult<Vec<f64>> {
        // Same guards `qml.train` applies at the Python-facing boundary.
        validate_shots_and_qpus(shots, n_qpus)?;
        validate_cunqa_allocation(&infrastructure, nodes, cores_per_qpu)?;

        // Bind each new sample to the trained θ, producing one native circuit per
        // sample (C-8 (a)). A sample of the wrong feature count fails here with a
        // `QmlError::FeatureCountMismatch`, mapped to `ValueError` — `bind`
        // already validates, so there is no separate check to add.
        let bound: Vec<BoundCircuit> = x
            .iter()
            .map(|xi| {
                self.inner
                    .model
                    .bind(xi, &self.inner.theta)
                    .map(BoundCircuit::Native)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Resolve the shot-sampling seed exactly as `run_quantum_circuit` does
        // (not via a "method" object, which does not exist here): `qmio` is real
        // hardware and rejects an explicit seed; every simulated backend draws a
        // fresh OS-entropy seed when none is given.
        let effective_seed: Option<u64> = if infrastructure == "qmio" {
            if seed.is_some() {
                return Err(PyValueError::new_err(
                    "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
                ));
            }
            None
        } else {
            Some(seed.unwrap_or_else(random_seed))
        };

        // Build the execution config once and share it across both paths (same
        // pattern as `qml_train_native`: exact mode ignores the config's
        // backend_config/shots/seed, but the config is still built once).
        let backend_config = build_backend_config(
            &infrastructure,
            backend,
            sim_method,
            noise_model.map(|nm| nm.unbind()),
            nodes,
            cores_per_qpu,
        )?;
        let config = ExecutionConfig {
            id: unique_id(&id),
            shots,
            n_qpus,
            infrastructure: infrastructure.clone(),
            backend_config,
            opt_level: OptLevel::default(),
            seed: effective_seed,
        };

        if exact {
            // Exact mode is native-only, gated by the *same* condition
            // `qml.train` uses: local infrastructure + the native backend.
            if infrastructure != "local" || !is_native_backend(backend) {
                return Err(PyValueError::new_err(format!(
                    "exact mode (exact=True) requires the native statevector backend: \
                     infrastructure=\"local\" and backend=\"polypus\", got \
                     infrastructure=\"{infrastructure}\", backend=\"{backend}\""
                )));
            }
            // The exact read-out is an inherent method of the concrete backend
            // (not on the `QuantumBackend` trait). The seed is unused in exact
            // mode; a resolved value is passed only because the constructor
            // requires one (`effective_seed` is `Some` since infra is "local").
            let native_backend =
                NativeStatevectorBackend::new(effective_seed.unwrap_or_else(random_seed));
            // Whole batch at once, no chunking — same behaviour as
            // `ExactNativeQmlOracle`. Release the GIL around the blocking read-out
            // (ENGINEERING §3) and check signals once afterwards.
            let all_probs =
                py.allow_threads(|| native_backend.run_circuits_exact(&bound, &config))?;
            py.check_signals()?;
            all_probs
                .iter()
                .map(|probs| {
                    self.inner
                        .model
                        .predict_from_probabilities(probs)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect()
        } else {
            let backend = Infrastructure::create_backend(&config)?;
            // Chunk by the backend's max batch size, exactly as
            // `run_native_qml_counts` does. Release the GIL around the whole
            // (potentially slow) run and check signals once afterwards.
            let batch_size = backend.max_batch_size(bound.len()).max(1);
            let all_counts = py.allow_threads(|| {
                let mut all: Vec<HashMap<String, u64>> = Vec::with_capacity(bound.len());
                for chunk in bound.chunks(batch_size) {
                    // Counts come back one dict per circuit in submission order
                    // (C-3), so extending preserves the sample order.
                    all.extend(backend.run_circuits(chunk, &config)?);
                }
                Ok::<_, BackendError>(all)
            })?;
            py.check_signals()?;
            all_counts
                .iter()
                .map(|counts| {
                    self.inner
                        .model
                        .predict_from_counts(counts)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect()
        }
    }
}

/// What the **native** path of [`qml.train`](qml_train) returns: every field a
/// [`TrainResult`] carries plus the [`TrainedModel`] that run produced, already
/// built (contract C-7, design doc §17).
///
/// Training a native `Model` + `Dataset` used to hand back a bare
/// [`TrainResult`], which knows nothing about either — so predicting meant
/// rebuilding `TrainedModel(model, dataset, result.best_params)` by hand, passing
/// `model` and `dataset` in a second time even though the `train()` call had them
/// all along. `trained_model` closes that gap: it is built **eagerly**, at the
/// single point where the training run ends and the model, the dataset and the
/// optimizer's `best_params` are all in scope at once, so
/// `train(...).trained_model.predict(x_new, ...)` is the whole
/// train→predict flow.
///
/// This type is deliberately **not** related to [`TrainResult`] by inheritance
/// (no `#[pyclass(extends = TrainResult)]`): two independent types cost less
/// PyO3 machinery than an `isinstance` compatibility nobody needs today. The
/// other two entry points are untouched — the Qiskit path of `qml.train` and the
/// generic [`train`](super::train) still return a plain [`TrainResult`], since
/// neither has a `Model`/`Dataset` to wrap.
#[pyclass(module = "polypus.qml", name = "QmlTrainResult", frozen)]
pub struct QmlTrainResult {
    /// Best parameter vector found — the `θ` bound into
    /// [`trained_model`](Self::trained_model).
    #[pyo3(get)]
    pub best_params: Vec<f64>,
    /// Fitness of [`best_params`](Self::best_params) (higher is better).
    #[pyo3(get)]
    pub best_fitness: f64,
    /// Generations/iterations actually executed.
    #[pyo3(get)]
    pub iterations_run: usize,
    /// Whether the optimizer's convergence criterion was satisfied.
    #[pyo3(get)]
    pub converged: bool,
    /// Effective RNG seed that drove the optimizer and shot sampling (C-7).
    #[pyo3(get)]
    pub seed: u64,
    /// Effective run identifier (the caller-supplied `id` prefix plus a UUID v4).
    #[pyo3(get)]
    pub id: String,
    /// The trained model — the `Model` compiled against the `Dataset`'s feature
    /// count and bound to [`best_params`](Self::best_params) — ready for
    /// [`predict`](TrainedModel::predict) or [`save`](TrainedModel::save).
    #[pyo3(get)]
    pub trained_model: Py<TrainedModel>,
}

#[pymethods]
impl QmlTrainResult {
    fn __repr__(&self) -> String {
        // The trained model is summarised by its `θ` width rather than its own
        // repr, to keep this one line long; that width is `best_params.len()` by
        // construction (the model is built *from* `best_params`), so reporting it
        // needs no borrow of the wrapped pyclass.
        format!(
            "QmlTrainResult(id={:?}, best_fitness={}, iterations_run={}, converged={}, seed={}, \
             best_params={:?}, trained_model=TrainedModel(num_theta={}))",
            self.id,
            self.best_fitness,
            self.iterations_run,
            self.converged,
            self.seed,
            self.best_params,
            self.best_params.len()
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
/// `seed` follows the same precedence as [`train`](super::train). On the native
/// path the same seed also drives shot sampling on every simulated backend, so
/// the run reproduces byte-for-byte. Ctrl+C interrupts promptly, and a user
/// callback's exception propagates verbatim (Qiskit path).
///
/// **The return type follows the path** (contract C-7): the native path returns a
/// [`QmlTrainResult`] — the same six fields plus a ready-to-use
/// [`trained_model`](QmlTrainResult::trained_model) — while the Qiskit path
/// returns a plain [`TrainResult`](super::TrainResult), unchanged, having no
/// `Model`/`Dataset` to wrap. This mirrors the kwarg asymmetry the two paths
/// already have (`x_train`/`expectation_function` versus `loss`/`batch_size`).
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
///
/// `exact` (default `False`) selects the shot-free exact-expectation path
/// (design doc §17). It is supported **only** on the native Model+Dataset path
/// with `infrastructure="local"` and a native `backend` (`polypus`); any other
/// combination — including the Qiskit path — rejects `exact=True` with a clear
/// error rather than silently ignoring it. In exact mode `shots`/`seed` no
/// longer affect the result (there is no sampling), so two runs with the same
/// configuration produce byte-identical `best_params`.
///
/// `batch_size` (default `None`) enables deterministic minibatching on the
/// **native** path (design doc §17): each optimizer evaluation scores a fresh
/// pseudo-random subset of `batch_size` samples (derived from `seed` + a
/// per-oracle call counter, so the run stays reproducible under C-7), instead of
/// the whole training set. It must satisfy `1 <= batch_size < num_samples` — a
/// batch as large as the dataset is just the non-minibatch path with more code
/// and is rejected. The reported `best_fitness` is **not** a minibatch estimate:
/// after the optimizer finishes, it is recomputed once against the full dataset,
/// so it stays comparable to a non-minibatch run (contract C-5). `batch_size` is
/// rejected on the Qiskit path, which does not use a `QmlProblem`.
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
    exact=false,
    batch_size=None,
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
    exact: bool,
    batch_size: Option<usize>,
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
            exact,
            batch_size,
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
            exact,
            batch_size,
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
    exact: bool,
    batch_size: Option<usize>,
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

    // Validate `batch_size` once, here at the Python-facing boundary, against the
    // full problem's sample count — so `QmlProblem::minibatch_indices` need not
    // re-check its precondition on every call (design doc §17). A batch of `0`,
    // or one as large as (or larger than) the dataset, is rejected: the latter is
    // just the non-minibatch path with more code, never what the caller wants.
    if let Some(b) = batch_size {
        let n = problem.num_circuits();
        if b < 1 || b >= n {
            return Err(PyValueError::new_err(format!(
                "batch_size must satisfy 1 <= batch_size < num_samples ({n}); got {b}"
            )));
        }
    }

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
    // Deterministic minibatch config (design doc §17), seeded by the same
    // resolved seed that drives the optimizer and shot sampling, so a minibatched
    // run reproduces byte-for-byte (C-7). Built once and moved into whichever
    // oracle branch runs; `None` keeps the full-dataset path untouched.
    let minibatch = batch_size.map(|b| MinibatchConfig::new(b, effective_seed));
    let errors = OracleErrorSlot::new();
    // Build the pair of trait-object boxes `dispatch_optimizer` consumes, plus an
    // optional `recompute` closure. Both paths hand it the *same* concrete oracle
    // behind an `Arc`, exposed as two independent facets via the `Arc<T>` blanket
    // impls: the evaluation box scores fitness (EvaluationOracle) and the gradient
    // box supplies the exact parameter-shift gradient (GradientOracle) for
    // QNG/Adam. `recompute`, present only under minibatching, re-scores the final
    // `best_params` against the full dataset (via a third `Arc` facet calling the
    // oracle's inherent `evaluate_full`) so the reported `best_fitness` is honest
    // rather than the last iteration's minibatch estimate (design doc §17).
    let (eval_oracle, gradient_oracle, recompute): NativeOracleParts = if exact {
        // Exact mode (design doc §17) is native-only: require local
        // infrastructure and a native backend, then read exact expectations
        // straight off the statevector — never through the QuantumBackend
        // trait (which has no "exact" meaning for noisy Aer or real hardware).
        if infrastructure != "local" || !is_native_backend(backend) {
            return Err(PyValueError::new_err(format!(
                "exact mode (exact=True) requires the native statevector backend: \
                 infrastructure=\"local\" and backend=\"polypus\", got \
                 infrastructure=\"{infrastructure}\", backend=\"{backend}\""
            )));
        }
        // Build the concrete native backend directly. The exact read-out is
        // an inherent method of `NativeStatevectorBackend`, not on the
        // `QuantumBackend` trait, so `Infrastructure::create_backend` (which
        // returns `Arc<dyn QuantumBackend>`) cannot supply it. The seed is
        // never used in exact mode — there is no sampling — but the
        // constructor requires one, so we pass the resolved seed anyway.
        let backend = Arc::new(NativeStatevectorBackend::new(effective_seed));
        let oracle = Arc::new(ExactNativeQmlOracle {
            problem,
            config: Arc::clone(&config),
            backend,
            errors: errors.clone(),
            minibatch,
        });
        let recompute = recompute_full_fitness(batch_size, &oracle, &errors);
        (Box::new(Arc::clone(&oracle)), Box::new(oracle), recompute)
    } else {
        let backend = Infrastructure::create_backend(&config)?;
        let oracle = Arc::new(NativeQmlOracle {
            problem,
            config: Arc::clone(&config),
            backend,
            errors: errors.clone(),
            minibatch,
        });
        let recompute = recompute_full_fitness(batch_size, &oracle, &errors);
        (Box::new(Arc::clone(&oracle)), Box::new(oracle), recompute)
    };

    let result = dispatch_optimizer(
        py,
        method,
        (eval_oracle, gradient_oracle),
        recompute.as_deref(),
        dimensions,
        &errors,
        effective_seed,
        effective_id,
    )?;

    // `dispatch_optimizer` is shared verbatim with the Qiskit path and the generic
    // `train`, so it still produces the plain `TrainResult`; only this path
    // upgrades it to a `QmlTrainResult`, here at the one point where the model,
    // the dataset and the optimizer's `best_params` are all in scope (design doc
    // §17). `TrainResult` is `frozen`, so reading its fields needs no runtime
    // borrow — `Bound::get` hands out a plain `&TrainResult`. An `Err` from the
    // dispatch (a bad `method`, an oracle failure, Ctrl+C) is propagated by the
    // `?` above exactly as before: the error is already the right one.
    let outcome = result
        .bind(py)
        .downcast::<TrainResult>()
        .map_err(|_| {
            PyTypeError::new_err(
                "internal error: the optimizer dispatch did not return a TrainResult",
            )
        })?
        .get();
    // Reuse `TrainedModel::new` rather than re-deriving the compile step: it is
    // the single definition of "this model, compiled against this dataset's
    // feature count, bound to this θ", and it is what a caller building the
    // trained model by hand would have called.
    let trained_model = Py::new(
        py,
        TrainedModel::new(model, dataset, outcome.best_params.clone())?,
    )?;
    Py::new(
        py,
        QmlTrainResult {
            best_params: outcome.best_params.clone(),
            best_fitness: outcome.best_fitness,
            iterations_run: outcome.iterations_run,
            converged: outcome.converged,
            seed: outcome.seed,
            id: outcome.id.clone(),
            trained_model,
        },
    )
    .map(|result| result.into_any())
}

/// Build the final-fitness recompute closure for a minibatched native run
/// (design doc §17), or `None` when `batch_size` is off.
///
/// When present, `dispatch_optimizer` applies it to the optimizer's
/// `best_params` **after** the run, replacing the last iteration's minibatch
/// `best_fitness` with an honest full-dataset value (contract C-5). It captures
/// a dedicated `Arc` facet of the oracle so it can call the inherent
/// `evaluate_full`, and the shared error slot: a failure there is recorded and a
/// finite sentinel returned, so `finish_optimization` surfaces the real error
/// rather than a bogus fitness — the same discipline the trait paths use.
///
/// Generic over the concrete oracle type (`NativeQmlOracle` /
/// `ExactNativeQmlOracle`); both expose `evaluate_full` inherently.
fn recompute_full_fitness<O>(
    batch_size: Option<usize>,
    oracle: &Arc<O>,
    errors: &OracleErrorSlot,
) -> Option<Box<RecomputeFn>>
where
    O: FullDatasetEvaluator + Send + Sync + 'static,
{
    batch_size.map(|_| {
        let oracle = Arc::clone(oracle);
        let errors = errors.clone();
        let closure: Box<RecomputeFn> = Box::new(move |theta: &[f64]| {
            match oracle.evaluate_full(theta) {
                Ok(fitness) => fitness,
                // Record and return a finite sentinel; `finish_optimization` sees
                // the recorded error and raises it instead of using this value.
                Err(e) => {
                    errors.record(e);
                    0.0
                }
            }
        });
        closure
    })
}

/// The inherent `evaluate_full` shared by both native oracles, abstracted so
/// [`recompute_full_fitness`] can build one closure type over either. Not part of
/// the `EvaluationOracle` trait: full-dataset re-scoring is a native-path concern
/// (the Qiskit path never minibatches), not a general oracle capability.
trait FullDatasetEvaluator {
    fn evaluate_full(&self, theta: &[f64]) -> Result<f64, crate::evaluation::EvaluationError>;
}

impl FullDatasetEvaluator for NativeQmlOracle {
    fn evaluate_full(&self, theta: &[f64]) -> Result<f64, crate::evaluation::EvaluationError> {
        NativeQmlOracle::evaluate_full(self, theta)
    }
}

impl FullDatasetEvaluator for ExactNativeQmlOracle {
    fn evaluate_full(&self, theta: &[f64]) -> Result<f64, crate::evaluation::EvaluationError> {
        ExactNativeQmlOracle::evaluate_full(self, theta)
    }
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
    exact: bool,
    batch_size: Option<usize>,
) -> PyResult<PyObject> {
    // Exact mode is native-only (design doc §17): the Qiskit path has no
    // statevector of its own to read exactly, so reject rather than ignore it.
    if exact {
        return Err(PyValueError::new_err(
            "exact mode (exact=True) is only supported on the native Model+Dataset path; \
             pass a polypus.qml.Model with infrastructure=\"local\" and backend=\"polypus\"",
        ));
    }
    // Minibatching is native-only too (design doc §17): it selects a subset of a
    // `QmlProblem`'s precompiled templates, which the Qiskit path has none of —
    // it prebinds Qiskit circuits per sample. Reject rather than silently ignore,
    // mirroring the `exact` cross-path rejection above.
    if batch_size.is_some() {
        return Err(PyValueError::new_err(
            "batch_size (minibatching) is only supported on the native Model+Dataset path; \
             pass a polypus.qml.Model",
        ));
    }
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
        // The Qiskit path never minibatches (batch_size is rejected above), so
        // its reported best_fitness is already the full-dataset value.
        None,
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
///
/// `recompute`, present only for a minibatched native run (design doc §17),
/// re-scores the optimizer's `best_params` against the **full** dataset once the
/// run ends, replacing the last iteration's minibatch `best_fitness` — so the
/// reported fitness is comparable to a non-minibatch run (contract C-5). It is
/// applied uniformly at the single convergence point below, regardless of which
/// optimizer ran; the Qiskit path passes `None`.
#[allow(clippy::too_many_arguments)]
fn dispatch_optimizer(
    py: Python<'_>,
    method: &Bound<'_, PyAny>,
    oracles: (Box<dyn EvaluationOracle>, Box<dyn GradientOracle>),
    recompute: Option<&RecomputeFn>,
    dimensions: u32,
    errors: &OracleErrorSlot,
    effective_seed: u64,
    effective_id: String,
) -> PyResult<PyObject> {
    // The two boxes are the same underlying oracle (one Arc, two blanket-impl
    // facets): the evaluation box for DE/PSO/QNG/Adam fitness, the gradient box
    // for QNG and Adam. Passed as a pair to keep the argument count in check.
    let (oracle, gradient_oracle) = oracles;

    // Run the selected optimizer, converging every branch on a single raw
    // `Result<OptimizationOutcome, _>` so the final-fitness recompute and
    // `finish_optimization` happen in exactly one place (design doc §17). The GIL
    // is released around each `optimize()` exactly as before (ENGINEERING §3).
    let result: Result<OptimizationOutcome, OptimizerError> =
        if let Ok(de) = method.extract::<PyRef<DE>>() {
            let args = AlgorithmDifferentialEvolutionArgs {
                oracle,
                population_size: de.population_size,
                generations: de.generations,
                dimensions,
                tolerance: de.tolerance,
                seed: Some(effective_seed),
            };
            py.allow_threads(|| AlgorithmDifferentialEvolution.optimize(args))
        } else if let Ok(pso) = method.extract::<PyRef<PSO>>() {
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
            py.allow_threads(|| AlgorithmPSO.optimize(args))
        } else if let Ok(qng) = method.extract::<PyRef<QNG>>() {
            let args = AlgorithmQNGArgs {
                oracle,
                gradient_oracle,
                max_iters: qng.max_iters,
                learning_rate: qng.learning_rate,
                bounds: qng.bounds,
                dimensions,
                tolerance: qng.tolerance,
                patience: qng.patience,
                variance_oracle: Box::new(PyVarianceOracle {
                    variance_function: qng.variance_function.clone_ref(py),
                    errors: errors.clone(),
                }),
                tikhonov_reg: qng.tikhonov_reg,
                seed: Some(effective_seed),
            };
            py.allow_threads(|| AlgorithmQNG.optimize(args))
        } else if let Ok(adam) = method.extract::<PyRef<Adam>>() {
            // Same two facets of the one Arc as QNG: the evaluation box scores
            // fitness, the gradient box supplies the exact parameter-shift
            // gradient. No VarianceOracle — Adam's step comes from the moments.
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
                patience: adam.patience,
                seed: Some(effective_seed),
            };
            py.allow_threads(|| AlgorithmAdam.optimize(args))
        } else {
            return Err(PyTypeError::new_err(
            "method must be an instance of polypus.DE, polypus.PSO, polypus.QNG, or polypus.Adam",
        ));
        };

    // Final-fitness recompute (design doc §17): only when minibatching is active
    // (`recompute` is `Some`), the optimizer produced a valid outcome, and no
    // oracle error is pending. If the optimizer already recorded a failure the
    // `best_params` are meaningless, so skip the recompute and let
    // `finish_optimization` surface that error. A failure *inside* the recompute
    // records into the same slot and is surfaced there too.
    let result = match (result, recompute) {
        (Ok(mut outcome), Some(recompute)) if !errors.failed() => {
            outcome.best_fitness = recompute(&outcome.best_params);
            Ok(outcome)
        }
        (other, _) => other,
    };

    finish_optimization(py, result, errors, effective_seed, effective_id)
}
