//! [`FiniteDifferenceGradient`]: a [`GradientOracle`] for any [`EvaluationOracle`].

use crate::error::OptimizerError;
use crate::objective::{EvaluationOracle, GradientOracle};

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
/// so a backend sees a single batch per gradient.
/// [`value_and_gradient`](GradientOracle::value_and_gradient), which
/// [`AlgorithmAdam`](crate::AlgorithmAdam) calls, puts `θ` in front of the same
/// stencil, `[θ, θ+h·e₀, θ−h·e₀, …]`, one call of `2·dims + 1` candidates, and
/// returns `θ`'s value with the gradient: one batch per Adam iteration. If the
/// wrapped oracle breaks its length contract, the gradient is an empty vector
/// (and `θ`'s value `NaN`), which the optimizer reports as
/// [`OptimizerError::OracleLengthMismatch`], and
/// [`gradient`](GradientOracle::gradient) returns `NaN`; the mismatch is logged
/// at `error` level either way.
///
/// Wrap the oracle that feeds the optimizer, shared through an
/// [`Arc`](std::sync::Arc) (an [`EvaluationOracle`] too): Adam reports the
/// value `value_and_gradient` returns for `θ` as a fitness it measured
/// (contract C-5), so it must be that oracle's.
#[derive(Debug, Clone)]
pub struct FiniteDifferenceGradient<O> {
    oracle: O,
    step: f64,
}

impl<O: EvaluationOracle> FiniteDifferenceGradient<O> {
    /// Wrap `oracle` with the finite-difference step `step`, which must be
    /// finite and `> 0` ([`OptimizerError::InvalidConfig`] otherwise).
    pub fn new(oracle: O, step: f64) -> Result<Self, OptimizerError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(OptimizerError::InvalidConfig {
                parameter: "finite_difference_step",
                reason: format!("must be finite and > 0, got {step}"),
            });
        }
        Ok(FiniteDifferenceGradient { oracle, step })
    }

    /// The finite-difference step `h`.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Score the stencil around `theta` for `indices` in one batch, after
    /// `theta` itself when `with_theta`, and return `theta`'s value (`NaN` when
    /// it is not scored) and one component per index. If the oracle broke its
    /// length contract, the value is `NaN` and the components are empty.
    fn stencil(
        &self,
        theta: &[f64],
        with_theta: bool,
        indices: impl Iterator<Item = usize>,
    ) -> (f64, Vec<f64>) {
        let h = self.step;
        let candidates: Vec<Vec<f64>> = with_theta
            .then(|| theta.to_vec())
            .into_iter()
            .chain(indices.flat_map(|i| {
                let mut plus = theta.to_vec();
                let mut minus = theta.to_vec();
                plus[i] += h;
                minus[i] -= h;
                [plus, minus]
            }))
            .collect();
        let values = self.oracle.evaluate_batch(&candidates);
        if values.len() != candidates.len() {
            log::error!(
                "finite-difference gradient: the oracle returned {} value(s) for {} candidate(s)",
                values.len(),
                candidates.len()
            );
            return (f64::NAN, Vec::new());
        }
        let (centre, shifted) = values.split_at(usize::from(with_theta));
        let gradient = shifted
            .chunks_exact(2)
            .map(|pair| (pair[0] - pair[1]) / (2.0 * h))
            .collect();
        (centre.first().copied().unwrap_or(f64::NAN), gradient)
    }
}

impl<O: EvaluationOracle> GradientOracle for FiniteDifferenceGradient<O> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.stencil(theta, false, std::iter::once(param_index))
            .1
            .first()
            .copied()
            .unwrap_or(f64::NAN)
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.stencil(theta, false, 0..dims).1
    }

    /// `θ`'s value and the gradient from one batch, `[θ, θ+h·e₀, θ−h·e₀, …]`.
    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        Some(self.stencil(theta, true, 0..dims))
    }
}
