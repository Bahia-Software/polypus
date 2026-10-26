//! [`FiniteDifferenceGradient`]: a [`GradientOracle`] for any [`EvaluationOracle`].

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};
use std::ops::Range;

/// Check a finite-difference step as [`FiniteDifferenceGradient::new`] does,
/// without an oracle: finite and `> 0` ([`OptimizerError::InvalidConfig`] for
/// `finite_difference_step` otherwise).
pub fn check_step(step: f64) -> Result<(), OptimizerError> {
    if !step.is_finite() || step <= 0.0 {
        return Err(OptimizerError::InvalidConfig {
            parameter: "finite_difference_step",
            reason: format!("must be finite and > 0, got {step}"),
        });
    }
    Ok(())
}

/// Central finite-difference gradient of an [`EvaluationOracle`]'s fitness.
///
/// For callers that can only score candidates, such as a circuit the library
/// cannot look inside: `∂f/∂θᵢ ≈ (f(θ + h·eᵢ) − f(θ − h·eᵢ)) / (2h)`, with the
/// step `h` given at construction. It is an *approximation* (truncation error
/// `O(h²)`, and shot noise in `f` is amplified by `1/h`); an exact gradient,
/// when the caller has one, is a [`GradientOracle`] of its own.
///
/// [`gradient_batch`](GradientOracle::gradient_batch) scores every shifted
/// candidate in **one** [`evaluate_batch`](EvaluationOracle::evaluate_batch)
/// call of `2·dims` candidates, laid out `[θ+h·e₀, θ−h·e₀, θ+h·e₁, θ−h·e₁, …]`,
/// so a backend sees a single batch per gradient. If the wrapped oracle breaks
/// its length contract, or a parameter index is out of range for `θ` (`dims`
/// greater than its length), `gradient_batch` returns an empty vector, which
/// the optimizer reports as [`OptimizerError::OracleLengthMismatch`], and
/// [`gradient`](GradientOracle::gradient) returns `NaN`, without a panic; it is
/// logged at `error` level either way, and an index out of range calls no
/// oracle.
///
/// To score candidates with the same oracle that feeds the optimizer, share it
/// through an [`Arc`](std::sync::Arc), which is an [`EvaluationOracle`] too.
#[derive(Debug, Clone)]
pub struct FiniteDifferenceGradient<O> {
    oracle: O,
    step: f64,
}

impl<O: EvaluationOracle> FiniteDifferenceGradient<O> {
    /// Wrap `oracle` with the finite-difference step `step`, which must be
    /// finite and `> 0` ([`OptimizerError::InvalidConfig`] otherwise).
    pub fn new(oracle: O, step: f64) -> Result<Self, OptimizerError> {
        check_step(step)?;
        Ok(FiniteDifferenceGradient { oracle, step })
    }

    /// The finite-difference step `h`.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Score the stencil around `theta` for `indices`, one batch, and return
    /// one component per index (empty if an index is out of range for `theta`
    /// or the oracle broke its length contract).
    fn stencil(&self, theta: &[f64], indices: Range<usize>) -> Vec<f64> {
        if indices.end > theta.len() {
            log::error!(
                "finite-difference gradient: parameter {} is out of range for a point of {} \
                 value(s)",
                indices.start.max(theta.len()),
                theta.len()
            );
            return Vec::new();
        }
        let h = self.step;
        let candidates: Vec<Vec<f64>> = indices
            .flat_map(|i| {
                let mut plus = theta.to_vec();
                let mut minus = theta.to_vec();
                plus[i] += h;
                minus[i] -= h;
                [plus, minus]
            })
            .collect();
        let values = self.oracle.evaluate_batch(&candidates);
        if values.len() != candidates.len() {
            log::error!(
                "finite-difference gradient: the oracle returned {} value(s) for {} candidate(s)",
                values.len(),
                candidates.len()
            );
            return Vec::new();
        }
        values
            .chunks_exact(2)
            .map(|pair| (pair[0] - pair[1]) / (2.0 * h))
            .collect()
    }
}

impl<O: EvaluationOracle> GradientOracle for FiniteDifferenceGradient<O> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.stencil(theta, param_index..param_index.saturating_add(1))
            .first()
            .copied()
            .unwrap_or(f64::NAN)
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.stencil(theta, 0..dims)
    }
}
