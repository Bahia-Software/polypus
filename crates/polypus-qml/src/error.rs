//! Error types for the QML layer.
//!
//! Two enums, split by *when* they can be raised, following the design doc
//! (§10) and the repo style (`PhysicsError`/`OptimizerError`: one flat enum
//! each, hand-written `Display` and `std::error::Error`, no `thiserror`, named
//! fields):
//!
//! - [`ValidationError`] — construction/compilation failures, all detectable
//!   *without* runtime (execution) data: bad datasets, and — added in phase 2
//!   — the model-level invariants checked by `compile` (no qubits, empty model,
//!   no trainable parameters, not enough qubits for an encoder).
//! - [`QmlError`] — runtime failures raised while emitting or binding a
//!   circuit template. It wraps [`CircuitError`] (with a `From` impl so `?`
//!   propagates transparently from every `try_push`/`assign_parameters` call)
//!   and adds the feature-count check `template_for` performs before dispatch,
//!   plus — added in phase 3 — the counts-format checks the expectation
//!   estimator performs, and the per-layer `emit` failures (the amplitude
//!   encoder's zero-norm sample, the basis encoder's non-binary feature).
//!
//! Neither enum derives `Eq`: [`ValidationError::InvalidTestFraction`] carries
//! an `f64` (as do [`ValidationError::LabelDomain`], which carries a [`Loss`],
//! and the [`Decision`]-bearing variants), and so does
//! [`QmlError::NonBinaryFeature`], which reports the offending feature value.
//! `QmlError` did derive `Eq` while every payload of its was `Eq`; that variant
//! traded it for the ability to name the value it rejected, which is what makes
//! its message actionable. `PartialEq` — the one the tests and the `assert_eq!`
//! call sites actually use — is unchanged on both.

use std::fmt;

use polypus_circuit::CircuitError;

use crate::loss::Loss;
use crate::observables::Pauli;
use crate::readout::Decision;

/// Errors raised while constructing or validating QML data.
///
/// Every variant is a precondition violation detectable without runtime
/// (execution) data. Returning these instead of panicking is what lets a
/// future FFI seam map a bad input to a typed Python error rather than
/// unwinding across the boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// A dataset was constructed with no samples.
    EmptyDataset,
    /// A row's width differs from the first row's width (features must form a
    /// rectangular matrix). Reports the first offending row.
    RaggedRows {
        /// Index of the first row whose width differs.
        sample: usize,
        /// The expected width (the first row's width).
        expected: usize,
        /// The offending row's width.
        got: usize,
    },
    /// The number of labels does not match the number of rows.
    LabelCountMismatch {
        /// Number of feature rows provided.
        rows: usize,
        /// Number of labels provided.
        labels: usize,
    },
    /// A feature value is not finite (`NaN` or infinite). Mirrors C-2's
    /// `NonFiniteParam` policy: no `NaN` ever enters the system.
    NonFiniteFeature {
        /// Index of the sample holding the offending value.
        sample: usize,
        /// Index of the offending feature within that sample.
        index: usize,
    },
    /// A label value is not finite (`NaN` or infinite).
    NonFiniteLabel {
        /// Index of the sample holding the offending label.
        sample: usize,
    },
    /// A `test_fraction` outside the open interval `(0.0, 1.0)` was requested.
    /// The endpoints `0.0` and `1.0` are rejected too: either would leave one
    /// of the two partitions empty, which cannot train or evaluate.
    InvalidTestFraction {
        /// The rejected fraction.
        fraction: f64,
    },
    /// A feature-range slice supplied to
    /// [`Dataset::scale_features_with`](crate::Dataset::scale_features_with)
    /// has a length different from the dataset's feature count.
    FeatureCountMismatch {
        /// The dataset's feature count (the required length).
        expected: usize,
        /// The length of the supplied slice.
        got: usize,
    },
    /// A model was compiled with zero qubits. A circuit needs at least one
    /// qubit to carry any gate.
    NoQubits,
    /// A model was compiled with no layers. There is nothing to emit.
    EmptyModel,
    /// A model compiled to zero trainable parameters (e.g. only encoders, no
    /// ansatz). Training a model with `dimensions == 0` is meaningless, so it
    /// is rejected at compile time rather than discovered as an optimizer that
    /// "converges" trivially.
    NoTrainableParams,
    /// A layer needs more active qubits than are available at its position in
    /// the model. Raised by an encoder whose feature count exceeds the number
    /// of active qubits.
    NotEnoughQubits {
        /// The number of active qubits the layer requires.
        needed: usize,
        /// The number of active qubits available at this position.
        active: usize,
    },
    /// A model was compiled without a readout. Training and inference both need
    /// observables to read out, so a model with no [`Readout`](crate::Readout)
    /// is rejected at compile time. Attach one with
    /// [`QuantumModel::readout`](crate::QuantumModel::readout).
    MissingReadout,
    /// A readout observable references a logical qubit position that does not
    /// exist among the model's final active qubits (e.g. a `PauliString` on
    /// position 3 of a 2-qubit readout).
    ObservableQubitOutOfRange {
        /// The offending logical position.
        position: usize,
        /// The number of active qubits available to the readout.
        num_active: usize,
    },
    /// A pooling layer needs at least two active qubits to form a pair, but
    /// fewer are available at its position in the model. Raised by
    /// [`PoolLayer`](crate::PoolLayer)'s `plan`.
    PoolNeedsTwoQubits {
        /// The number of active qubits available at this position.
        active: usize,
    },
    /// An [`AmplitudeEncoder`](crate::AmplitudeEncoder) was placed anywhere but
    /// first in the model. It prepares a state from `|0…0⟩` rather than
    /// transforming an existing one, so it cannot compose on top of earlier
    /// layers (design doc §6.2).
    AmplitudeEncoderNotFirst,
    /// An encoder was asked to encode more features than its active qubits can
    /// address. Raised by [`AmplitudeEncoder`](crate::AmplitudeEncoder)'s
    /// `plan`, which can hold at most `2^k` amplitudes on `k` qubits.
    TooManyFeatures {
        /// The largest feature count the active qubits can encode (`2^k`).
        max: usize,
        /// The feature count requested.
        got: usize,
    },
    /// A [`PauliString`](crate::PauliString) was constructed with two factors
    /// on the same qubit position. Positions must be unique.
    DuplicatePauliPosition {
        /// The position referenced more than once.
        position: usize,
    },
    /// A single readout observable mixes measurement bases on the same qubit:
    /// two of its terms require different Paulis on the same logical position
    /// (e.g. `0.5·Z₀ + 0.5·X₀`), so no single basis change can measure them
    /// together. Reports the qubit and the two conflicting bases (design doc
    /// §7.2).
    ObservableHasIncompatibleBases {
        /// The logical position asked for in two different bases.
        position: usize,
        /// The basis the first term to touch this position requires.
        first: Pauli,
        /// The conflicting basis a later term requires on the same position.
        second: Pauli,
    },
    /// The readout's observables cannot all be measured under a single basis
    /// change: they partition into more than one basis group (e.g. one class in
    /// `Z` and another in `X` on the *same* qubit). Measuring them would need
    /// one circuit per group, and multi-circuit base grouping is not
    /// implemented yet (design doc §7.2) — so this is rejected rather than
    /// returning a silently mismeasured result. A single-group readout (any
    /// all-`Z` readout, or one whose classes share a compatible non-`Z` basis)
    /// is accepted and measured with one circuit.
    ReadoutNeedsMultipleBasisGroups {
        /// The number of distinct measurement bases the readout would require.
        groups: usize,
    },
    /// An [`Observable`](crate::Observable) was constructed with a non-finite
    /// coefficient (`NaN` or infinite). Reports the first offending term.
    NonFiniteCoefficient {
        /// Index of the offending term within the observable.
        term_index: usize,
    },
    /// A label falls outside the domain the chosen [`Loss`] requires
    /// (`BinaryCrossEntropy` needs `{0, 1}`, `Hinge` needs `{-1, +1}`). Reports
    /// the first offending sample.
    LabelDomain {
        /// The loss whose domain was violated.
        loss: Loss,
        /// A human-readable description of the expected domain.
        expected: &'static str,
        /// Index of the first sample whose label is out of domain.
        found_sample: usize,
    },
    /// A [`Decision`] is incompatible with the number of observables in the
    /// readout: `Argmax` needs at least two, the binary/regression decisions
    /// (`Sign`, `Threshold`, `Raw`) need at least one. Raised by
    /// [`Readout::new`](crate::Readout::new).
    DecisionObservableMismatch {
        /// The decision that could not be satisfied.
        decision: Decision,
        /// The number of observables supplied.
        num_observables: usize,
    },
    /// A [`Decision`] and a [`Loss`] were paired incompatibly. The pairing is
    /// bidirectional (design doc §17): [`Decision::Argmax`] is the multiclass
    /// decision and must pair with the multiclass
    /// [`Loss::CategoricalCrossEntropy`], while every scalar loss
    /// (`SquaredError`/`BinaryCrossEntropy`/`Hinge`) reads `⟨O₀⟩` alone and must
    /// pair with a scalar decision. Either mismatch — a scalar loss under
    /// `Argmax`, or `CategoricalCrossEntropy` under a non-`Argmax` decision —
    /// raises this. Raised by [`QmlProblem::new`](crate::QmlProblem::new).
    DecisionNotSupportedByLoss {
        /// The decision that could not be paired with `loss`.
        decision: Decision,
        /// The loss that could not be paired with `decision`.
        loss: Loss,
    },
    /// A categorical label is a valid non-negative integer (already checked by
    /// [`Loss::validate_label`]) but names a class `≥ num_classes`. Raised by
    /// [`QmlProblem::new`](crate::QmlProblem::new), which knows the number of
    /// observables (`= num_classes`) that [`Loss::validate_label`] does not.
    LabelClassOutOfRange {
        /// Index of the offending sample.
        sample: usize,
        /// The out-of-range class label.
        label: f64,
        /// The number of classes (readout observables) available.
        num_classes: usize,
    },
    /// Precompiling a training template failed while constructing a
    /// [`QmlProblem`](crate::QmlProblem). Wraps the underlying [`QmlError`] so
    /// the `?` inside `QmlProblem::new` can convert a template failure into a
    /// construction error. In practice unreachable — the dataset and model are
    /// already validated — but typed rather than assumed away with `expect`.
    Template(QmlError),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyDataset => {
                write!(f, "dataset is empty: at least one sample is required")
            }
            ValidationError::RaggedRows {
                sample,
                expected,
                got,
            } => write!(
                f,
                "ragged feature rows: sample {sample} has width {got}, expected {expected} (the first row's width)"
            ),
            ValidationError::LabelCountMismatch { rows, labels } => write!(
                f,
                "label count mismatch: {rows} feature row(s) but {labels} label(s)"
            ),
            ValidationError::NonFiniteFeature { sample, index } => write!(
                f,
                "non-finite feature at sample {sample}, index {index}"
            ),
            ValidationError::NonFiniteLabel { sample } => {
                write!(f, "non-finite label at sample {sample}")
            }
            ValidationError::InvalidTestFraction { fraction } => write!(
                f,
                "test_fraction must lie in the open interval (0.0, 1.0), got {fraction}"
            ),
            ValidationError::FeatureCountMismatch { expected, got } => write!(
                f,
                "feature-range count mismatch: dataset has {expected} feature(s), got {got} range(s)"
            ),
            ValidationError::NoQubits => {
                write!(f, "model has no qubits: at least one qubit is required")
            }
            ValidationError::EmptyModel => {
                write!(f, "model has no layers: at least one layer is required")
            }
            ValidationError::NoTrainableParams => write!(
                f,
                "model has no trainable parameters: add at least one ansatz layer"
            ),
            ValidationError::NotEnoughQubits { needed, active } => write!(
                f,
                "layer needs {needed} active qubit(s) but only {active} are available"
            ),
            ValidationError::MissingReadout => write!(
                f,
                "model has no readout: attach one with QuantumModel::readout before compiling"
            ),
            ValidationError::ObservableQubitOutOfRange {
                position,
                num_active,
            } => write!(
                f,
                "readout observable references logical qubit {position}, but only {num_active} qubit(s) are active"
            ),
            ValidationError::PoolNeedsTwoQubits { active } => write!(
                f,
                "pooling layer needs at least 2 active qubit(s) but only {active} are available"
            ),
            ValidationError::AmplitudeEncoderNotFirst => write!(
                f,
                "amplitude encoder must be the first layer: it prepares a state from |0…0⟩, not a composable transformation"
            ),
            ValidationError::TooManyFeatures { max, got } => write!(
                f,
                "too many features for amplitude encoding: {got} feature(s) exceed the {max} amplitude(s) the active qubits can hold"
            ),
            ValidationError::DuplicatePauliPosition { position } => write!(
                f,
                "Pauli string has more than one factor on position {position}"
            ),
            ValidationError::ObservableHasIncompatibleBases {
                position,
                first,
                second,
            } => write!(
                f,
                "observable mixes measurement bases on qubit {position}: one term requires {first:?} and another requires {second:?}; every observable must resolve to a single Pauli per qubit"
            ),
            ValidationError::ReadoutNeedsMultipleBasisGroups { groups } => write!(
                f,
                "readout needs {groups} distinct measurement bases (e.g. Z on one class and X on another for the same qubit): multi-circuit base grouping is not implemented yet (design doc §7.2), so this is rejected rather than silently mismeasured"
            ),
            ValidationError::NonFiniteCoefficient { term_index } => write!(
                f,
                "non-finite coefficient in observable term {term_index}"
            ),
            ValidationError::LabelDomain {
                loss,
                expected,
                found_sample,
            } => write!(
                f,
                "label domain violation for {loss:?}: sample {found_sample} is outside the expected domain {expected}"
            ),
            ValidationError::DecisionObservableMismatch {
                decision,
                num_observables,
            } => write!(
                f,
                "decision {decision:?} is incompatible with {num_observables} observable(s)"
            ),
            ValidationError::DecisionNotSupportedByLoss { decision, loss } => write!(
                f,
                "decision {decision:?} cannot be paired with loss {loss:?}: Argmax requires CategoricalCrossEntropy and every scalar loss requires a non-Argmax decision"
            ),
            ValidationError::LabelClassOutOfRange {
                sample,
                label,
                num_classes,
            } => write!(
                f,
                "label class out of range: sample {sample} names class {label} but only {num_classes} class(es) are available"
            ),
            ValidationError::Template(e) => write!(f, "template compilation failed: {e}"),
        }
    }
}

impl std::error::Error for ValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ValidationError::Template(e) => Some(e),
            _ => None,
        }
    }
}

impl From<QmlError> for ValidationError {
    fn from(e: QmlError) -> Self {
        ValidationError::Template(e)
    }
}

/// Errors raised while emitting or binding a circuit template at runtime.
///
/// Distinct from [`ValidationError`]: those are precondition violations caught
/// during construction/compilation, whereas these arise while turning a
/// compiled model plus a sample `x` into a [`ParameterizedCircuit`] or a
/// [`ConcreteCircuit`]. Every `try_push` and `assign_parameters` call inside an
/// `emit`/`template_for`/`bind` propagates its [`CircuitError`] here via `?`
/// and the [`From`] impl below, so an internal bookkeeping bug surfaces as a
/// typed error rather than a `panic!` crossing the FFI boundary.
///
/// [`ParameterizedCircuit`]: polypus_circuit::ParameterizedCircuit
/// [`ConcreteCircuit`]: polypus_circuit::ConcreteCircuit
#[derive(Debug, Clone, PartialEq)]
pub enum QmlError {
    /// A circuit operation failed (out-of-range qubit, non-finite angle, wrong
    /// number of bound parameters, …). Wraps the underlying [`CircuitError`].
    Circuit(CircuitError),
    /// A sample passed to
    /// [`template_for`](crate::CompiledModel::template_for) or
    /// [`bind`](crate::CompiledModel::bind) has a feature count different from
    /// the one the model was compiled for.
    FeatureCountMismatch {
        /// The feature count the model was compiled with.
        expected: usize,
        /// The length of the supplied sample.
        got: usize,
    },
    /// A counts map handed to the expectation estimator has keys whose width
    /// does not match, or a Pauli string references a position wider than the
    /// keys carry (see [`expectation_from_counts`](crate::observables)).
    CountsWidthMismatch {
        /// The expected bitstring width.
        expected: usize,
        /// The width actually found (an offending key's width, or the key
        /// width against which an out-of-range position was checked).
        got: usize,
    },
    /// A sample handed to the [`AmplitudeEncoder`](crate::AmplitudeEncoder) has
    /// zero L2 norm (every feature is zero), so there is no normalized state to
    /// prepare. Never "fixed" silently — a zero sample is a data error the
    /// caller must resolve (design doc §6.2).
    ///
    /// This is a unit variant: `emit` sees only the sample it is handed, not
    /// its index in the dataset, so it cannot report a `sample` position
    /// without widening the [`LayerOps::emit`](crate::model) signature for this
    /// single case.
    ZeroNormSample,
    /// A feature handed to the [`BasisEncoder`](crate::BasisEncoder) is neither
    /// `0.0` nor `1.0`, so it names no computational basis state. Never rounded
    /// or thresholded silently — a non-binary feature is a data error the caller
    /// must resolve (design doc §6.7).
    ///
    /// Unlike [`ZeroNormSample`](Self::ZeroNormSample) — which is about the
    /// whole sample's norm and so has no position to report — this variant does
    /// carry one: `emit` walks the features one at a time, so it knows exactly
    /// which failed. The index is within the sample, not within the dataset
    /// (`emit` never sees the sample's own position).
    NonBinaryFeature {
        /// Index of the offending feature within the sample.
        feature: usize,
        /// The rejected value.
        got: f64,
    },
    /// The counts map handed to the expectation estimator is empty (or records
    /// zero total shots): there is nothing to estimate an expectation from.
    EmptyCounts,
    /// A slice of counts maps handed to
    /// [`fitness_from_counts`](crate::QmlProblem::fitness_from_counts) has a
    /// length different from the number of training circuits.
    CountsLengthMismatch {
        /// The number of circuits (and hence counts maps) expected.
        expected: usize,
        /// The number of counts maps supplied.
        got: usize,
    },
    /// One per-sample entry of the `base_expectations` slice handed to
    /// [`param_gradient_categorical`](crate::QmlProblem::param_gradient_categorical)
    /// (or its exact mirror) carries a number of class expectations different
    /// from the readout's observable count.
    ///
    /// Distinct from [`CountsLengthMismatch`](Self::CountsLengthMismatch),
    /// which is about the *outer* length — how many samples the slice holds.
    /// This one is about the *inner* width of a single sample's
    /// class-expectation vector, which the categorical chain rule zips against
    /// the per-class shifted expectations: a wrong width there would silently
    /// truncate (or over-extend) the sum instead of failing, yielding a
    /// plausible but wrong gradient.
    ///
    /// Reported for the first offending sample, deterministically — the same
    /// convention as [`ValidationError::RaggedRows`].
    ClassCountMismatch {
        /// Index of the first offending sample.
        sample: usize,
        /// The number of classes expected (the readout's observable count).
        expected: usize,
        /// The length of the offending sample's expectation vector.
        got: usize,
    },
    /// [`Loss::evaluate`](crate::Loss::evaluate) or
    /// [`Loss::gradient`](crate::Loss::gradient) was reached with
    /// `Loss::CategoricalCrossEntropy`, which has no scalar form: it scores a
    /// whole per-class expectation vector, not a single `⟨O₀⟩`, and is served
    /// instead by the free `categorical_cross_entropy`/
    /// `categorical_cross_entropy_gradient` functions. `QmlProblem` always
    /// routes the categorical loss to those *before* reaching the scalar
    /// methods (see `fitness_from_counts`/`param_gradient`), so this should
    /// never actually surface — it is typed rather than assumed away with
    /// `unreachable!`, so an internal dispatch bug becomes a typed error
    /// instead of a panic crossing the FFI boundary.
    CategoricalLossHasNoScalarForm,
}

impl fmt::Display for QmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QmlError::Circuit(e) => write!(f, "circuit error: {e}"),
            QmlError::FeatureCountMismatch { expected, got } => write!(
                f,
                "feature count mismatch: model expects {expected} feature(s) per sample, got {got}"
            ),
            QmlError::CountsWidthMismatch { expected, got } => write!(
                f,
                "counts width mismatch: expected bitstrings of width {expected}, got {got}"
            ),
            QmlError::ZeroNormSample => write!(
                f,
                "amplitude encoding requires a non-zero sample: the L2 norm is zero, so there is no state to prepare"
            ),
            QmlError::NonBinaryFeature { feature, got } => write!(
                f,
                "basis encoding requires every feature to be exactly 0.0 or 1.0: feature {feature} is {got}"
            ),
            QmlError::EmptyCounts => {
                write!(
                    f,
                    "counts map is empty: nothing to estimate an expectation from"
                )
            }
            QmlError::CountsLengthMismatch { expected, got } => write!(
                f,
                "counts length mismatch: expected {expected} counts map(s), got {got}"
            ),
            QmlError::ClassCountMismatch {
                sample,
                expected,
                got,
            } => write!(
                f,
                "class count mismatch: sample {sample} carries {got} class expectation(s), expected {expected} (one per readout observable)"
            ),
            QmlError::CategoricalLossHasNoScalarForm => write!(
                f,
                "internal error: CategoricalCrossEntropy has no scalar evaluate/gradient form; \
                 QmlProblem should always route it through the categorical path instead"
            ),
        }
    }
}

impl std::error::Error for QmlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            QmlError::Circuit(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CircuitError> for QmlError {
    fn from(e: CircuitError) -> Self {
        QmlError::Circuit(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_offending_values() {
        let err = ValidationError::RaggedRows {
            sample: 2,
            expected: 3,
            got: 4,
        };
        let s = err.to_string();
        assert!(s.contains('2'));
        assert!(s.contains('3'));
        assert!(s.contains('4'));
    }

    #[test]
    fn invalid_test_fraction_displays_fraction() {
        let err = ValidationError::InvalidTestFraction { fraction: 1.5 };
        assert!(err.to_string().contains("1.5"));
    }

    #[test]
    fn error_trait_is_implemented() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&ValidationError::EmptyDataset);
        assert_error(&QmlError::FeatureCountMismatch {
            expected: 2,
            got: 1,
        });
    }

    #[test]
    fn new_validation_variants_display_their_values() {
        assert!(ValidationError::NoQubits.to_string().contains("qubit"));
        assert!(ValidationError::EmptyModel.to_string().contains("layer"));
        assert!(ValidationError::NoTrainableParams
            .to_string()
            .contains("trainable"));
        let s = ValidationError::NotEnoughQubits {
            needed: 4,
            active: 3,
        }
        .to_string();
        assert!(s.contains('4'));
        assert!(s.contains('3'));
    }

    #[test]
    fn qml_error_feature_count_mismatch_displays_values() {
        let s = QmlError::FeatureCountMismatch {
            expected: 5,
            got: 2,
        }
        .to_string();
        assert!(s.contains('5'));
        assert!(s.contains('2'));
    }

    #[test]
    fn phase3_validation_variants_display_their_values() {
        assert!(ValidationError::MissingReadout
            .to_string()
            .contains("readout"));
        let s = ValidationError::ObservableQubitOutOfRange {
            position: 3,
            num_active: 2,
        }
        .to_string();
        assert!(s.contains('3') && s.contains('2'));
        assert!(ValidationError::DuplicatePauliPosition { position: 5 }
            .to_string()
            .contains('5'));
        let s = ValidationError::ObservableHasIncompatibleBases {
            position: 1,
            first: Pauli::Z,
            second: Pauli::X,
        }
        .to_string();
        assert!(s.contains('1') && s.contains('Z') && s.contains('X'));
        let s = ValidationError::ReadoutNeedsMultipleBasisGroups { groups: 2 }.to_string();
        assert!(s.contains('2'));
        assert!(ValidationError::NonFiniteCoefficient { term_index: 4 }
            .to_string()
            .contains('4'));
        let s = ValidationError::LabelDomain {
            loss: Loss::Hinge,
            expected: "{-1.0, 1.0}",
            found_sample: 2,
        }
        .to_string();
        assert!(s.contains("Hinge") && s.contains('2'));
        assert!(ValidationError::DecisionNotSupportedByLoss {
            decision: Decision::Argmax,
            loss: Loss::Hinge,
        }
        .to_string()
        .contains("Argmax"));
    }

    #[test]
    fn categorical_validation_variants_display_their_values() {
        // DecisionNotSupportedByLoss now carries both the decision and the loss.
        let s = ValidationError::DecisionNotSupportedByLoss {
            decision: Decision::Argmax,
            loss: Loss::SquaredError,
        }
        .to_string();
        assert!(s.contains("Argmax") && s.contains("SquaredError"));
        // LabelClassOutOfRange reports the sample, the label and the class count.
        let s = ValidationError::LabelClassOutOfRange {
            sample: 4,
            label: 3.0,
            num_classes: 3,
        }
        .to_string();
        assert!(s.contains('4') && s.contains('3'));
    }

    #[test]
    fn pool_needs_two_qubits_displays_its_value() {
        assert!(ValidationError::PoolNeedsTwoQubits { active: 1 }
            .to_string()
            .contains('1'));
    }

    #[test]
    fn phase6_variants_display_their_values() {
        assert!(ValidationError::AmplitudeEncoderNotFirst
            .to_string()
            .contains("first"));
        let s = ValidationError::TooManyFeatures { max: 4, got: 5 }.to_string();
        assert!(s.contains('4') && s.contains('5'));
        assert!(QmlError::ZeroNormSample.to_string().contains("norm"));
    }

    #[test]
    fn qml_counts_variants_display_their_values() {
        let s = QmlError::CountsWidthMismatch {
            expected: 3,
            got: 4,
        }
        .to_string();
        assert!(s.contains('3') && s.contains('4'));
        assert!(QmlError::EmptyCounts.to_string().contains("empty"));
        let s = QmlError::CountsLengthMismatch {
            expected: 8,
            got: 7,
        }
        .to_string();
        assert!(s.contains('8') && s.contains('7'));
        assert!(QmlError::CategoricalLossHasNoScalarForm
            .to_string()
            .contains("scalar"));
    }

    #[test]
    fn class_count_mismatch_displays_sample_and_widths() {
        // All three halves must be in the message: the sample says *which*
        // expectation vector to fix, the widths say what was wrong with it.
        let s = QmlError::ClassCountMismatch {
            sample: 1,
            expected: 3,
            got: 2,
        }
        .to_string();
        assert!(s.contains('1'), "missing the sample index: {s}");
        assert!(s.contains('3'), "missing the expected class count: {s}");
        assert!(s.contains('2'), "missing the offending width: {s}");
        // Its wording must not read as the *outer* (per-sample) mismatch that
        // `CountsLengthMismatch` reports — the two are distinct failures.
        assert!(
            s != QmlError::CountsLengthMismatch {
                expected: 3,
                got: 2,
            }
            .to_string()
        );
    }

    #[test]
    fn non_binary_feature_displays_its_index_and_value() {
        // Both halves must be in the message: the index says *which* feature to
        // fix, the value says what was wrong with it.
        let s = QmlError::NonBinaryFeature {
            feature: 2,
            got: 0.5,
        }
        .to_string();
        assert!(s.contains('2'), "missing the feature index: {s}");
        assert!(s.contains("0.5"), "missing the offending value: {s}");
        // A value outside [0, 1] is reported the same way, not special-cased.
        let s = QmlError::NonBinaryFeature {
            feature: 0,
            got: -1.5,
        }
        .to_string();
        assert!(s.contains("-1.5"), "missing the offending value: {s}");
    }

    #[test]
    fn qml_error_converts_into_validation_error_via_from() {
        // The `?` inside `QmlProblem::new` relies on this From to turn a
        // template failure into a construction error.
        let inner = QmlError::EmptyCounts;
        let err: ValidationError = inner.clone().into();
        assert_eq!(err, ValidationError::Template(inner.clone()));
        assert!(err.to_string().contains(&inner.to_string()));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn qml_error_wraps_circuit_error_via_from() {
        let inner = CircuitError::QubitOutOfRange {
            qubit: 3,
            num_qubits: 2,
        };
        let err: QmlError = inner.clone().into();
        assert_eq!(err, QmlError::Circuit(inner.clone()));
        // Display forwards the inner message; `source` exposes the cause.
        assert!(err.to_string().contains(&inner.to_string()));
        assert!(std::error::Error::source(&err).is_some());
    }
}
