use pyo3::prelude::*;

/// Adam optimizer configuration: Kingma and Ba's Adam with bias correction,
/// climbing the fitness along its gradient.
///
/// The gradient is exact when the oracle has one (a native QML model: the
/// parameter-shift rule); otherwise, in `train` and in a Qiskit `qml.train`,
/// it is a central finite difference of the fitness with step
/// `finite_difference_step`, one batch of `2·dimensions` circuits per
/// iteration. The settings are validated when `train`/`qml.train` starts,
/// before any circuit runs: an invalid one raises `ValueError` (contract C-5).
#[pyclass(module = "polypus")]
pub struct Adam {
    /// Iterations, each one gradient and one evaluation of the updated
    /// parameters. At least 1.
    #[pyo3(get, set)]
    pub max_iters: u32,
    /// Step size, finite and `> 0`. Defaults to `0.05`, not deep learning's
    /// `0.001`: the parameters are rotation angles in radians, not network
    /// weights, so a larger step is well scaled.
    #[pyo3(get, set)]
    pub learning_rate: f64,
    /// Decay of the first-moment estimate, in `[0, 1)`. Defaults to `0.9`.
    #[pyo3(get, set)]
    pub beta1: f64,
    /// Decay of the second-moment estimate, in `[0, 1)`. Defaults to `0.999`.
    #[pyo3(get, set)]
    pub beta2: f64,
    /// Added to the denominator for numerical stability, finite and `> 0`.
    /// Defaults to `1e-8`.
    #[pyo3(get, set)]
    pub epsilon: f64,
    /// Range the initial parameters are drawn from, finite with
    /// `lower < upper`. Defaults to `(-π, π)`. The run is not confined to it.
    #[pyo3(get, set)]
    pub bounds: (f64, f64),
    /// Minimum improvement of the best fitness over the last `patience`
    /// iterations below which the run ends (DE's rule, contract C-5).
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Iterations over which the best-fitness improvement is measured.
    /// Defaults to 20.
    #[pyo3(get, set)]
    pub patience: u32,
    /// Step of the central finite difference used when there is no exact
    /// gradient, finite and `> 0`. Defaults to `0.1`, QNG's default.
    #[pyo3(get, set)]
    pub finite_difference_step: f64,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (max_iters = 100, learning_rate = 0.05, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, bounds = (-std::f64::consts::PI, std::f64::consts::PI), tolerance = 0.01, patience = 20, finite_difference_step = 0.1, seed = None))]
    pub fn new(
        max_iters: u32,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        bounds: (f64, f64),
        tolerance: f64,
        patience: u32,
        finite_difference_step: f64,
        seed: Option<u64>,
    ) -> Self {
        Adam {
            max_iters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bounds,
            tolerance,
            patience,
            finite_difference_step,
            seed,
        }
    }
}
