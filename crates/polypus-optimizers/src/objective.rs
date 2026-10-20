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
}

/// A shared oracle is an oracle: the same scorer can feed an optimizer and a
/// [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient) built over it.
impl<T: EvaluationOracle + ?Sized> EvaluationOracle for std::sync::Arc<T> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        (**self).evaluate_batch(candidates)
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
/// be finite, as every fitness value must be.
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
}
