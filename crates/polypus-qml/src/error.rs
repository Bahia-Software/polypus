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
//!   estimator performs.
//!
//! `ValidationError` deliberately omits `Eq` because
//! [`ValidationError::InvalidTestFraction`] carries an `f64`, which is not
//! `Eq` (nor is [`ValidationError::LabelDomain`], which carries a [`Loss`], or
//! the [`Decision`]-bearing variants); `QmlError` derives `Eq` (all its
//! payloads are `Eq`, and `CircuitError` is too). The full error catalogue of
//! §10 is still completed incrementally — the remaining
//! amplitude-encoding/pooling variants arrive with the phases that can raise
//! them.

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
    /// A [`PauliString`](crate::PauliString) was constructed with two factors
    /// on the same qubit position. Positions must be unique.
    DuplicatePauliPosition {
        /// The position referenced more than once.
        position: usize,
    },
    /// A readout observable uses a Pauli the v1 readout does not support. Only
    /// `Z` (computational-basis) readout exists today; `X`/`Y` arrive with the
    /// base-grouping phase (design doc §7.2).
    UnsupportedPauli {
        /// The unsupported Pauli.
        pauli: Pauli,
        /// The logical position it was requested on.
        position: usize,
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
    /// A [`Decision`] cannot be trained by any v1 [`Loss`]. `Argmax` is
    /// multiclass but every v1 loss operates on `⟨O₀⟩` alone, so the pairing
    /// would silently optimise one observable while inference reads them all.
    /// Raised by [`QmlProblem::new`](crate::QmlProblem::new) (design doc §8).
    DecisionNotSupportedByLoss {
        /// The decision that no v1 loss supports.
        decision: Decision,
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
            ValidationError::DuplicatePauliPosition { position } => write!(
                f,
                "Pauli string has more than one factor on position {position}"
            ),
            ValidationError::UnsupportedPauli { pauli, position } => write!(
                f,
                "unsupported Pauli {pauli:?} on position {position}: the v1 readout only supports Z (computational basis)"
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
            ValidationError::DecisionNotSupportedByLoss { decision } => write!(
                f,
                "decision {decision:?} is not trainable by any v1 loss (all operate on a single observable)"
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
            QmlError::EmptyCounts => {
                write!(f, "counts map is empty: nothing to estimate an expectation from")
            }
            QmlError::CountsLengthMismatch { expected, got } => write!(
                f,
                "counts length mismatch: expected {expected} counts map(s), got {got}"
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
        let s = ValidationError::UnsupportedPauli {
            pauli: Pauli::X,
            position: 1,
        }
        .to_string();
        assert!(s.contains('X') && s.contains('1'));
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
        }
        .to_string()
        .contains("Argmax"));
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
