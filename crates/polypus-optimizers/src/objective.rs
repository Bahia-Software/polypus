//! Input contracts consumed by the optimizers.
//!
//! These traits are the *only* seam between an optimizer and the outside world.
//! Implementing them lets any scorer (statevector simulator, QPU, analytic
//! function, …) drive the optimizers without this crate knowing anything about
//! circuits or Python.

/// Contract between optimization algorithms and candidate evaluation.
///
/// An oracle encapsulates everything needed to translate a parameter vector
/// into a scalar fitness value. Algorithms only call
/// [`EvaluationOracle::evaluate_batch`] and have no knowledge of circuits,
/// QPUs, infrastructure, or training modes.
///
/// To add a new evaluation strategy (e.g. noisy readout mitigation, hardware
/// native gates, …) implement this trait without touching any algorithm.
pub trait EvaluationOracle: Send + Sync {
    /// Evaluate a batch of candidate parameter vectors.
    ///
    /// Returns one fitness value per candidate, in the same order. Higher is
    /// better (algorithms maximise the returned value).
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64>;

    /// Evaluate a batch as [`evaluate_batch`](EvaluationOracle::evaluate_batch)
    /// does, but measure every circuit with `shots` shots instead of the
    /// oracle's own.
    ///
    /// No optimizer calls it. A training run calls it once, after the
    /// optimizer has stopped, to re-evaluate the best candidates with more
    /// shots (`final_shots` of `TrainFlow` in `polypus-orchestration`).
    ///
    /// The default calls `evaluate_batch`, which is right for an oracle
    /// without shots (an analytic function, an exact simulator), whose values
    /// do not depend on them. A custom oracle that samples shots and keeps the
    /// default scores the batch with its usual shots, so a value re-evaluated
    /// through it is as noisy as the optimizer's own: override it to honour
    /// `shots`. A wrapper that forwards `evaluate_batch` to another oracle
    /// should forward this method too, as the `Arc` implementation below
    /// does; otherwise the inner oracle's override is skipped.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], _shots: u32) -> Vec<f64> {
        self.evaluate_batch(candidates)
    }
}

/// A shared oracle is an oracle: the same scorer can feed an optimizer and a
/// [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient) built over it.
impl<T: EvaluationOracle + ?Sized> EvaluationOracle for std::sync::Arc<T> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        (**self).evaluate_batch(candidates)
    }

    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        (**self).evaluate_batch_with_shots(candidates, shots)
    }
}

/// Diagonal of the Fubini–Study metric (quantum Fisher information matrix)
/// consumed by [`AlgorithmQNG`](crate::AlgorithmQNG).
///
/// This abstracts the single algorithm-specific callback that Quantum Natural
/// Gradient needs. Keeping it behind a trait means the QNG loop stays pure
/// Rust: a Python-backed implementation (or any other runtime) lives entirely
/// in the caller.
pub trait VarianceOracle: Send + Sync {
    /// `Var[H_a | theta]`: the diagonal QFIM element for parameter index `a`.
    fn variance(&self, theta: &[f64], param_index: usize) -> f64;

    /// Compute all `dims` diagonal QFIM elements for `theta` at once.
    ///
    /// The default implementation simply loops over [`VarianceOracle::variance`].
    /// Implementations backed by an external runtime (for example a Python
    /// callback that must be invoked under the GIL) should override this to
    /// amortise the per-call setup cost across the whole diagonal — acquiring
    /// the runtime once and evaluating every index in a tight loop.
    fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        (0..dims).map(|a| self.variance(theta, a)).collect()
    }
}

/// Gradient of the fitness, consumed by [`AlgorithmAdam`](crate::AlgorithmAdam).
///
/// The sign convention is [`EvaluationOracle`]'s: higher fitness is better, so
/// the gradient points uphill and an ascent step *adds* it. Like
/// [`VarianceOracle`], it keeps the one extra quantity a gradient method needs
/// behind a trait, so how it is computed (an exact parameter shift, a finite
/// difference, an analytic formula) stays with the caller. Every component must
/// be finite, as every fitness value must be. An implementation that scores
/// candidates to compute its gradient can also return the fitness at `θ` from
/// the same batch ([`value_and_gradient`](GradientOracle::value_and_gradient)).
pub trait GradientOracle: Send + Sync {
    /// `∂fitness/∂θ[param_index]` at `theta`.
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64;

    /// All `dims` components at once, in parameter order.
    ///
    /// The default implementation loops over [`GradientOracle::gradient`].
    /// Implementations that compute every component from one batch of circuits
    /// (a parameter shift, a finite-difference stencil) should override it.
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        (0..dims).map(|k| self.gradient(theta, k)).collect()
    }

    /// The fitness at `theta` and the `dims` gradient components there,
    /// measured in one call, or `None` when this oracle does not score `theta`
    /// itself.
    ///
    /// [`AlgorithmAdam`](crate::AlgorithmAdam) calls it once per iteration.
    /// With `None`, the default, Adam scores `theta` with its own
    /// [`EvaluationOracle`] and then calls
    /// [`gradient_batch`](GradientOracle::gradient_batch): two calls, which is
    /// all an implementation that only computes the gradient needs to support.
    /// One whose gradient is itself a batch of circuits (a finite-difference
    /// stencil, a parameter shift) can put `theta` in that batch and return its
    /// value here, so that an iteration is a single batch;
    /// [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient) does.
    ///
    /// The optimizer reports the value as a fitness it measured (contract C-5),
    /// so it must come from the optimizer's own [`EvaluationOracle`]: build the
    /// implementation over that oracle, shared through an
    /// [`Arc`](std::sync::Arc). The gradient follows
    /// [`gradient_batch`](GradientOracle::gradient_batch)'s contract. `None`
    /// must mean that nothing was scored, since a caller that counts batches
    /// does not count that call.
    ///
    /// A training run's CUNQA walltime projection (`polypus-orchestration`)
    /// cannot tell in advance whether an exact gradient overrides this, so it
    /// counts two calls per Adam iteration. With an override an iteration is
    /// one call, and the projection times two iterations as one: it
    /// overestimates, never underestimates.
    fn value_and_gradient(&self, _theta: &[f64], _dims: usize) -> Option<(f64, Vec<f64>)> {
        None
    }
}

/// A shared gradient oracle is a gradient oracle: one object can serve as both
/// the [`EvaluationOracle`] and the [`GradientOracle`] of a run.
impl<T: GradientOracle + ?Sized> GradientOracle for std::sync::Arc<T> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        (**self).gradient(theta, param_index)
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        (**self).gradient_batch(theta, dims)
    }

    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        (**self).value_and_gradient(theta, dims)
    }
}

#[cfg(test)]
mod tests {
    use super::EvaluationOracle;
    use std::sync::Arc;

    /// Scores a candidate by its first coordinate, with no shots at all.
    struct FirstCoordinate;
    impl EvaluationOracle for FirstCoordinate {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            candidates.iter().map(|c| c[0]).collect()
        }
    }

    /// Scores every candidate by the shots it was measured with: 100 when
    /// the oracle chooses them.
    struct ShotCounter;
    impl EvaluationOracle for ShotCounter {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.evaluate_batch_with_shots(candidates, 100)
        }
        fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
            vec![f64::from(shots); candidates.len()]
        }
    }

    #[test]
    fn the_default_scores_as_evaluate_batch_whatever_the_shots() {
        let candidates = vec![vec![0.5, 1.0], vec![-2.0, 3.0]];
        for shots in [1, 4096, u32::MAX] {
            assert_eq!(
                FirstCoordinate.evaluate_batch_with_shots(&candidates, shots),
                FirstCoordinate.evaluate_batch(&candidates)
            );
        }
    }

    #[test]
    fn a_shared_oracle_forwards_the_shots() {
        let candidates = vec![vec![0.0]; 3];
        let shared = Arc::new(ShotCounter);
        assert_eq!(shared.evaluate_batch(&candidates), vec![100.0; 3]);
        assert_eq!(
            shared.evaluate_batch_with_shots(&candidates, 4096),
            vec![4096.0; 3]
        );
        // The same through a trait object, as a training run holds its oracle.
        let erased: Arc<dyn EvaluationOracle> = shared;
        let boxed: Box<dyn EvaluationOracle> = Box::new(Arc::clone(&erased));
        assert_eq!(
            boxed.evaluate_batch_with_shots(&candidates, 7),
            vec![7.0; 3]
        );
    }
}
