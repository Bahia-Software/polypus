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
    /// Multiclass softmax cross-entropy over **all** the readout's observables
    /// (design doc §17). Unlike the three losses above — which score the single
    /// raw expectation `⟨O₀⟩` — this one consumes the whole expectation vector
    /// `[⟨O₀⟩, …, ⟨O_{k−1}⟩]` and an integer class index `y ∈ {0, …, k−1}`, so
    /// it is served by the free [`categorical_cross_entropy`] /
    /// [`categorical_cross_entropy_gradient`] functions rather than the scalar
    /// [`evaluate`](Self::evaluate) / [`gradient`](Self::gradient) methods.
    /// Requires a non-negative integer label (the class index); the upper bound
    /// `< num_classes` is checked separately by
    /// [`QmlProblem::new`](crate::QmlProblem::new), which knows `k`.
    CategoricalCrossEntropy,
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
            // A class index: a non-negative integer. The upper bound
            // (`< num_classes`) is not knowable here — `validate_label` has no
            // `k` — so `QmlProblem::new` checks it separately once it knows the
            // number of observables.
            Loss::CategoricalCrossEntropy => label >= 0.0 && label.fract() == 0.0,
        };
        if ok {
            return Ok(());
        }
        let expected = match self {
            Loss::SquaredError => "any finite value",
            Loss::BinaryCrossEntropy => "{0.0, 1.0}",
            Loss::Hinge => "{-1.0, 1.0}",
            Loss::CategoricalCrossEntropy => "a non-negative integer class index",
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
            // `CategoricalCrossEntropy` scores the whole expectation vector, not
            // a single `⟨O₀⟩`, so it has no scalar form. It is served by the free
            // `categorical_cross_entropy` function, and `QmlProblem` routes the
            // categorical path there *before* reaching this scalar method (see
            // `fitness_from_counts`), so this arm is unreachable by construction.
            Loss::CategoricalCrossEntropy => unreachable!(
                "CategoricalCrossEntropy uses the free categorical_cross_entropy function, \
                 not the scalar Loss::evaluate path"
            ),
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
            // See `evaluate`: the categorical path is served by the free
            // `categorical_cross_entropy_gradient` function and never reaches
            // this scalar method (`param_gradient` routes it away first).
            Loss::CategoricalCrossEntropy => unreachable!(
                "CategoricalCrossEntropy uses the free categorical_cross_entropy_gradient \
                 function, not the scalar Loss::gradient path"
            ),
        }
    }
}

/// The softmax cross-entropy of an expectation vector `expectations` against an
/// integer class `label` (design doc §17):
///
/// `CE(z, y) = −z_y + ln(Σ_k exp(z_k))`
///
/// where `z = expectations` are the raw per-class expectations `[⟨O₀⟩, …]` and
/// `y = label` is the class index. The standard log-sum-exp stability trick
/// (subtract `max(z)` before exponentiating) keeps `exp` from overflowing for
/// large expectations; the subtracted constant cancels exactly, so the result
/// is unchanged. A free function, not a [`Loss`] method: it only makes sense for
/// the vector-valued categorical variant (see the [`Loss`] enum docs).
pub(crate) fn categorical_cross_entropy(expectations: &[f64], label: usize) -> f64 {
    // log-sum-exp with the max shifted out: ln Σ exp(z_k)
    //   = m + ln Σ exp(z_k − m),  m = max_k z_k.
    let max = expectations
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = expectations.iter().map(|&z| (z - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    log_sum_exp - expectations[label]
}

/// The gradient of [`categorical_cross_entropy`] with respect to each component
/// of the expectation vector (design doc §17):
///
/// `∂CE/∂z_j = softmax(z)_j − [j == y]`
///
/// Returns one value per class, in class order. Uses the same log-sum-exp
/// stability shift as [`categorical_cross_entropy`]. A free function for the
/// same reason.
pub(crate) fn categorical_cross_entropy_gradient(expectations: &[f64], label: usize) -> Vec<f64> {
    let max = expectations
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = expectations.iter().map(|&z| (z - max).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    exps.iter()
        .enumerate()
        .map(|(j, &e)| e / sum_exp - if j == label { 1.0 } else { 0.0 })
        .collect()
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
    fn categorical_cross_entropy_requires_non_negative_integer_label() {
        assert!(Loss::CategoricalCrossEntropy.validate_label(0.0, 0).is_ok());
        assert!(Loss::CategoricalCrossEntropy.validate_label(3.0, 0).is_ok());
        // Negative: rejected.
        let err = Loss::CategoricalCrossEntropy
            .validate_label(-1.0, 5)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelDomain {
                loss: Loss::CategoricalCrossEntropy,
                expected: "a non-negative integer class index",
                found_sample: 5,
            }
        );
        // Non-integer: rejected.
        let err = Loss::CategoricalCrossEntropy
            .validate_label(1.5, 2)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelDomain {
                loss: Loss::CategoricalCrossEntropy,
                expected: "a non-negative integer class index",
                found_sample: 2,
            }
        );
    }

    #[test]
    fn categorical_cross_entropy_value_matches_hand_computation() {
        // k=3, z=[1.0, 2.0, 0.5], y=1.
        //   Σ exp = e + e² + e^0.5 = 11.7560592
        //   ln Σ exp = 2.4643696
        //   CE = −z_1 + ln Σ exp = −2.0 + 2.4643696 = 0.4643696
        let z = [1.0, 2.0, 0.5];
        let ce = categorical_cross_entropy(&z, 1);
        assert!((ce - 0.4643696).abs() < 1e-6, "CE={ce}");
    }

    #[test]
    fn categorical_cross_entropy_stability_shift_is_exact() {
        // Adding a constant to every logit leaves CE unchanged (the log-sum-exp
        // shift cancels), and a large logit must not overflow to inf/NaN.
        let z = [1.0, 2.0, 0.5];
        let shifted = [1.0 + 1000.0, 2.0 + 1000.0, 0.5 + 1000.0];
        let a = categorical_cross_entropy(&z, 1);
        let b = categorical_cross_entropy(&shifted, 1);
        assert!(b.is_finite(), "CE overflowed: {b}");
        assert!((a - b).abs() < 1e-9, "shift changed CE: {a} vs {b}");
    }

    #[test]
    fn categorical_cross_entropy_gradient_matches_finite_difference() {
        // Central finite difference of `categorical_cross_entropy` per component
        // vs the analytic softmax−onehot gradient, at k=3, z=[1.0, 2.0, 0.5],
        // y=1 (the same hand case). Verifies ∂CE/∂z_j = softmax(z)_j − [j==y].
        let z = [1.0, 2.0, 0.5];
        let label = 1usize;
        let analytic = categorical_cross_entropy_gradient(&z, label);
        assert_eq!(analytic.len(), z.len());
        let h = 1e-6;
        for j in 0..z.len() {
            let mut zp = z;
            let mut zm = z;
            zp[j] += h;
            zm[j] -= h;
            let fd = (categorical_cross_entropy(&zp, label)
                - categorical_cross_entropy(&zm, label))
                / (2.0 * h);
            assert!(
                (analytic[j] - fd).abs() < 1e-4,
                "component {j}: analytic {} vs fd {fd}",
                analytic[j]
            );
        }
        // Invariant: the softmax−onehot gradient sums to exactly 0 (Σ softmax = 1).
        let sum: f64 = analytic.iter().sum();
        assert!(sum.abs() < 1e-12, "gradient does not sum to zero: {sum}");
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
