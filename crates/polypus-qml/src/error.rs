//! Error types for the QML layer.
//!
//! This crate does **not** depend on `polypus-circuit` yet (that dependency
//! arrives with `model.rs` in a later phase), so there is deliberately no
//! runtime `QmlError` and no `From<CircuitError>` conversion here. Phase 1 only
//! constructs and validates data, so a single construction/validation enum is
//! all that is reachable. The remaining variants of the full catalogue
//! (compilation, emission, readout) are introduced by the phases that can
//! actually raise them — they are intentionally not anticipated here.
//!
//! Style follows `PhysicsError`/`OptimizerError`: one flat enum, hand-written
//! `Display` and `std::error::Error` (no `thiserror`), named fields, and no
//! premature `From` impls. `Eq` is intentionally omitted because
//! [`ValidationError::InvalidTestFraction`] carries an `f64`, which is not
//! `Eq`.

use std::fmt;

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
        }
    }
}

impl std::error::Error for ValidationError {}

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
    }
}
