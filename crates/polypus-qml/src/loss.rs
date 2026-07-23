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

    /// The derivative of the per-sample loss with respect to `prediction`
    /// (`d loss / d ⟨O₀⟩`), the chain-rule factor a parameter-shift gradient
    /// multiplies against the shift of the raw expectation (design doc §17,
    /// [`QmlProblem::param_gradient`](crate::QmlProblem::param_gradient)). Every
    /// branch below is the analytic derivative of the matching arm of
    /// [`evaluate`](Self::evaluate) and is finite for finite inputs.
    pub(crate) fn gradient(&self, prediction: f64, label: f64) -> f64 {
        match self {
            Loss::SquaredError => 2.0 * (prediction - label),
            Loss::BinaryCrossEntropy => {
                // `evaluate` clamps `(1 + pred)/2` into `[ε, 1 − ε]`. Where the
                // clamp is inactive the loss is `−[y·ln p + (1−y)·ln(1−p)]` with
                // `p = (1 + pred)/2`, whose derivative w.r.t. `pred` is
                // `(p − label) / (2 p (1 − p))`. Where the clamp saturates, `p`
                // is pinned constant, so the true slope of the clamped loss is
                // exactly `0` there.
                let raw = (1.0 + prediction) / 2.0;
                let p = raw.clamp(BCE_EPSILON, 1.0 - BCE_EPSILON);
                if raw > BCE_EPSILON && raw < 1.0 - BCE_EPSILON {
                    (p - label) / (2.0 * p * (1.0 - p))
                } else {
                    0.0
                }
            }
            // `max(0, 1 − label·pred)`: slope `−label` inside the margin, `0`
            // outside it. At the exact break-point (`1 − label·pred == 0`) the
            // subgradient `0` is chosen (matches `evaluate`'s `max(_, 0.0)`).
            Loss::Hinge => {
                if 1.0 - label * prediction > 0.0 {
                    -label
                } else {
                    0.0
                }
            }
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
        let err = Loss::BinaryCrossEntropy
            .validate_label(-1.0, 4)
            .unwrap_err();
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

    /// Central finite difference of `evaluate`, valid only away from
    /// break-points/saturation (where the true derivative is discontinuous).
    fn fd_gradient(loss: Loss, pred: f64, label: f64) -> f64 {
        let h = 1e-6;
        (loss.evaluate(pred + h, label) - loss.evaluate(pred - h, label)) / (2.0 * h)
    }

    #[test]
    fn squared_error_gradient_matches_finite_difference() {
        // Smooth everywhere, so any point works.
        for &(pred, label) in &[(0.3, -0.7), (-1.2, 0.0), (2.5, 2.5), (0.0, 1.0)] {
            let analytic = Loss::SquaredError.gradient(pred, label);
            let fd = fd_gradient(Loss::SquaredError, pred, label);
            assert!(
                (analytic - fd).abs() < 1e-4,
                "SquaredError grad {analytic} vs fd {fd} at pred={pred}, label={label}"
            );
        }
    }

    #[test]
    fn bce_gradient_matches_finite_difference_where_unsaturated() {
        // Points where (1+pred)/2 stays well inside (ε, 1−ε), so the clamp is
        // inactive and the finite-difference comparison is valid.
        for &(pred, label) in &[(0.2, 1.0), (-0.4, 0.0), (0.5, 1.0), (-0.6, 0.0)] {
            let analytic = Loss::BinaryCrossEntropy.gradient(pred, label);
            let fd = fd_gradient(Loss::BinaryCrossEntropy, pred, label);
            assert!(
                (analytic - fd).abs() < 1e-4,
                "BCE grad {analytic} vs fd {fd} at pred={pred}, label={label}"
            );
        }
    }

    #[test]
    fn hinge_gradient_matches_finite_difference_away_from_kink() {
        // Inside the margin (1 − label·pred > 0) and outside it (< 0); avoid the
        // exact kink where the finite difference straddles the discontinuity.
        for &(pred, label) in &[(0.3, 1.0), (-0.5, 1.0), (0.5, -1.0), (2.0, 1.0)] {
            let analytic = Loss::Hinge.gradient(pred, label);
            let fd = fd_gradient(Loss::Hinge, pred, label);
            assert!(
                (analytic - fd).abs() < 1e-4,
                "Hinge grad {analytic} vs fd {fd} at pred={pred}, label={label}"
            );
        }
    }

    #[test]
    fn bce_gradient_is_zero_in_saturated_region() {
        // (1+pred)/2 ≥ 1 − ε → clamp active → slope pinned to exactly 0.
        assert_eq!(Loss::BinaryCrossEntropy.gradient(1.0, 1.0), 0.0);
        // (1+pred)/2 ≤ ε → clamp active → exactly 0.
        assert_eq!(Loss::BinaryCrossEntropy.gradient(-1.0, 0.0), 0.0);
    }

    #[test]
    fn hinge_gradient_is_zero_outside_margin_and_at_kink() {
        // Outside the margin (1 − label·pred < 0): flat, exactly 0.
        assert_eq!(Loss::Hinge.gradient(2.0, 1.0), 0.0);
        // Exactly at the break-point (1 − label·pred == 0): 0, not −label.
        assert_eq!(Loss::Hinge.gradient(1.0, 1.0), 0.0);
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
