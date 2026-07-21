//! Loss functions and their per-sample contribution (design doc §8).
//!
//! A [`Loss`] does two things: it constrains the label domain
//! ([`Loss::validate_label`], checked once when the training problem is built)
//! and it scores a prediction against a label ([`Loss::evaluate`]). Every loss
//! operates on the **raw** expectation `⟨O₀⟩`, never on the output of a
//! [`Decision`](crate::Decision) — the decision is a separate inference step.
//!
//! The optimizers *maximise*, so [`QmlProblem`](crate::QmlProblem) turns the
//! mean loss into `fitness = −mean_loss`. C-5 requires finite fitness; every
//! loss below is finite for finite inputs (`BinaryCrossEntropy` clamps its
//! probability away from the `log` singularities), and non-finite predictions
//! are already impossible upstream (the dataset and circuits reject them).

use crate::error::ValidationError;

/// Small margin keeping `BinaryCrossEntropy`'s probability strictly inside
/// `(0, 1)`, so `ln(p)` and `ln(1 − p)` stay finite even when `⟨O₀⟩` saturates
/// at `±1`.
const BCE_EPSILON: f64 = 1e-9;

/// A training loss (design doc §8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    /// Squared error `(pred − label)²`. Regression / continuous labels; accepts
    /// any finite label.
    SquaredError,
    /// Binary cross-entropy with `p = clamp((1 + pred)/2, ε, 1 − ε)`. Requires
    /// labels in `{0, 1}`.
    BinaryCrossEntropy,
    /// Hinge loss `max(0, 1 − label·pred)`. Requires labels in `{−1, +1}`.
    Hinge,
}

impl Loss {
    /// Check that `label` (of sample index `sample`) lies in this loss's
    /// domain. `SquaredError` accepts any finite label (already guaranteed by
    /// the [`Dataset`](crate::Dataset)); `BinaryCrossEntropy` requires exactly
    /// `{0.0, 1.0}` and `Hinge` exactly `{−1.0, 1.0}`, otherwise
    /// [`ValidationError::LabelDomain`].
    pub(crate) fn validate_label(&self, label: f64, sample: usize) -> Result<(), ValidationError> {
        let ok = match self {
            Loss::SquaredError => true,
            Loss::BinaryCrossEntropy => label == 0.0 || label == 1.0,
            Loss::Hinge => label == -1.0 || label == 1.0,
        };
        if ok {
            return Ok(());
        }
        let expected = match self {
            Loss::SquaredError => "any finite value",
            Loss::BinaryCrossEntropy => "{0.0, 1.0}",
            Loss::Hinge => "{-1.0, 1.0}",
        };
        Err(ValidationError::LabelDomain {
            loss: *self,
            expected,
            found_sample: sample,
        })
    }

    /// The per-sample loss of `prediction` (`⟨O₀⟩`, raw) against `label`.
    /// Always finite for finite inputs.
    pub(crate) fn evaluate(&self, prediction: f64, label: f64) -> f64 {
        match self {
            Loss::SquaredError => (prediction - label).powi(2),
            Loss::BinaryCrossEntropy => {
                let p = ((1.0 + prediction) / 2.0).clamp(BCE_EPSILON, 1.0 - BCE_EPSILON);
                -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
            }
            Loss::Hinge => (1.0 - label * prediction).max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_error_accepts_any_finite_label() {
        assert!(Loss::SquaredError.validate_label(2.5, 0).is_ok());
        assert!(Loss::SquaredError.validate_label(-100.0, 0).is_ok());
    }

    #[test]
    fn bce_requires_zero_or_one() {
        assert!(Loss::BinaryCrossEntropy.validate_label(0.0, 0).is_ok());
        assert!(Loss::BinaryCrossEntropy.validate_label(1.0, 0).is_ok());
        let err = Loss::BinaryCrossEntropy.validate_label(-1.0, 4).unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelDomain {
                loss: Loss::BinaryCrossEntropy,
                expected: "{0.0, 1.0}",
                found_sample: 4,
            }
        );
    }

    #[test]
    fn hinge_requires_plus_minus_one() {
        assert!(Loss::Hinge.validate_label(-1.0, 0).is_ok());
        assert!(Loss::Hinge.validate_label(1.0, 0).is_ok());
        let err = Loss::Hinge.validate_label(0.0, 2).unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelDomain {
                loss: Loss::Hinge,
                expected: "{-1.0, 1.0}",
                found_sample: 2,
            }
        );
    }

    #[test]
    fn squared_error_value() {
        assert!((Loss::SquaredError.evaluate(0.5, 1.0) - 0.25).abs() < 1e-12);
        assert!(Loss::SquaredError.evaluate(1.0, 1.0).abs() < 1e-12);
    }

    #[test]
    fn hinge_value() {
        // Correct, confident: 1 − 1·1 = 0.
        assert!(Loss::Hinge.evaluate(1.0, 1.0).abs() < 1e-12);
        // Wrong side: 1 − (−1)·1 = 2.
        assert!((Loss::Hinge.evaluate(1.0, -1.0) - 2.0).abs() < 1e-12);
        // Inside the margin: 1 − 1·0.5 = 0.5.
        assert!((Loss::Hinge.evaluate(0.5, 1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bce_is_finite_even_at_saturation() {
        // pred = 1 (label 1) would send p → 1 and ln(1 − p) → −∞ without the
        // clamp; assert it stays finite.
        let l = Loss::BinaryCrossEntropy.evaluate(1.0, 1.0);
        assert!(l.is_finite());
        let l = Loss::BinaryCrossEntropy.evaluate(-1.0, 0.0);
        assert!(l.is_finite());
        // pred = 0 → p = 0.5 → −ln(0.5) for either label.
        let l = Loss::BinaryCrossEntropy.evaluate(0.0, 1.0);
        assert!((l - 0.5f64.ln().abs()).abs() < 1e-9);
    }
}
