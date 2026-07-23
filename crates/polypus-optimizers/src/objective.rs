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

/// The gradient of an [`EvaluationOracle`]'s fitness with respect to each
/// trainable parameter, at a fixed `theta`. "Exact" below means an exact
/// mathematical identity in the noiseless limit — under real shot noise every
/// implementation remains an unbiased *estimator*, not a noise-free value.
/// An implementation is expected to be exact for whatever scalar its
/// companion [`EvaluationOracle`] computes — the caller is responsible for that
/// guarantee (this crate has no visibility into circuits or gate generators
/// to check it itself), exactly as [`VarianceOracle`]'s caller is responsible
/// for a mathematically sound QFIM diagonal.
pub trait GradientOracle: Send + Sync {
    /// ∂fitness/∂θ[param_index] at `theta` (same sign convention as
    /// [`EvaluationOracle`]: higher fitness is better, so this is the ascent
    /// direction, not its negation).
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64;

    /// All `dims` gradient components at once. Override when computing them
    /// together is cheaper than `dims` independent calls.
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        (0..dims).map(|k| self.gradient(theta, k)).collect()
    }
}

impl<T: EvaluationOracle + ?Sized> EvaluationOracle for std::sync::Arc<T> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.as_ref().evaluate_batch(candidates)
    }
}

impl<T: GradientOracle + ?Sized> GradientOracle for std::sync::Arc<T> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.as_ref().gradient(theta, param_index)
    }
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.as_ref().gradient_batch(theta, dims)
    }
}

/// Exact gradient (see the noiseless-limit caveat on [`GradientOracle`]) of an
/// [`EvaluationOracle`] whose fitness is already linear in each shifted
/// expectation — a raw expectation value, or an unweighted mean of several —
/// with no nonlinear composition (e.g. a `Loss`) on top. Under that
/// precondition, shifting the whole candidate by `±π/2` and combining is exact
/// by linearity, with no per-sample decomposition needed. **Not** valid for an
/// oracle whose fitness composes a nonlinear loss over per-sample
/// expectations (see `polypus_qml::QmlProblem::param_gradient` for that case).
pub fn linear_parameter_shift_gradient(
    oracle: &dyn EvaluationOracle,
    theta: &[f64],
    dims: usize,
) -> Vec<f64> {
    let candidates: Vec<Vec<f64>> = (0..dims)
        .flat_map(|i| {
            let mut tp = theta.to_vec();
            let mut tm = theta.to_vec();
            tp[i] += std::f64::consts::PI / 2.0;
            tm[i] -= std::f64::consts::PI / 2.0;
            [tp, tm]
        })
        .collect();
    let results = oracle.evaluate_batch(&candidates);
    (0..dims)
        .map(|i| (results[2 * i] - results[2 * i + 1]) / 2.0)
        .collect()
}
