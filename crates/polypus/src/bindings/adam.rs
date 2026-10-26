use polypus_optimizers::{AdamRule, OptimizerError};
use pyo3::prelude::*;

/// Adam optimizer configuration: Kingma and Ba's Adam with bias correction,
/// climbing the fitness along its gradient, or Polyak's heavy-ball momentum
/// with `rule="momentum"`.
///
/// Each iteration sends one batch of `2·dimensions + 1` circuits: the current
/// parameters, whose fitness is the iteration's, and the central finite
/// difference around them with step `finite_difference_step`, which gives the
/// gradient. `best_params` is the best of the parameters measured. The settings
/// are validated when `train`/`qml.train` starts, before any circuit runs: an
/// invalid one raises `ValueError` (contract C-5).
#[pyclass(module = "polypus")]
pub struct Adam {
    /// Iterations, each one batch of `2·dimensions + 1` circuits. At least 1.
    #[pyo3(get, set)]
    pub max_iters: u32,
    /// Step size, finite and `> 0`. Defaults to `0.1`, not deep learning's
    /// `0.001`: the parameters are rotation angles in radians, not network
    /// weights, so with `rule="adam"` each parameter moves about 0.1 rad per
    /// iteration. With `rule="momentum"` the step also scales with the
    /// gradient, so the rate that works depends on the scale of the cost.
    #[pyo3(get, set)]
    pub learning_rate: f64,
    /// Decay of the first-moment estimate, in `[0, 1)`; the momentum with
    /// `rule="momentum"`. Defaults to `0.9`.
    #[pyo3(get, set)]
    pub beta1: f64,
    /// Decay of the second-moment estimate, in `[0, 1)`. Defaults to `0.999`.
    /// Unused by `rule="momentum"`.
    #[pyo3(get, set)]
    pub beta2: f64,
    /// Added to the denominator for numerical stability, finite and `> 0`.
    /// Defaults to `1e-8`. Unused by `rule="momentum"`.
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
    /// Step of the central finite difference that estimates the gradient,
    /// finite and `> 0`. Defaults to `0.2`. Shot noise in the two fitness
    /// values is divided by `2·finite_difference_step`, so a small step ruins
    /// the gradient: Jones, Mineh and Montanaro (arXiv:2411.13742) found
    /// that `0.4` works and `0.01` does not.
    #[pyo3(get, set)]
    pub finite_difference_step: f64,
    /// How the gradient becomes a step. Python reads and writes it by name
    /// (`"adam"`, `"momentum"`); an unknown name raises `ValueError` as soon
    /// as it is passed or set, so a run never starts with one.
    pub rule: AdamRule,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

/// Parse a `rule` name; an unknown one is a `ValueError` naming the valid
/// names.
fn parse_rule(name: &str) -> PyResult<AdamRule> {
    name.parse()
        .map_err(|e: OptimizerError| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (max_iters = 100, learning_rate = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, bounds = (-std::f64::consts::PI, std::f64::consts::PI), tolerance = 0.01, patience = 20, finite_difference_step = 0.2, rule = "adam", seed = None))]
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
        rule: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Adam {
            max_iters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            bounds,
            tolerance,
            patience,
            finite_difference_step,
            rule: parse_rule(rule)?,
            seed,
        })
    }

    /// How the gradient becomes a step: `"adam"` (the default; Kingma and
    /// Ba's Adam, whose bias-corrected, moment-normalised step is about
    /// `learning_rate` per parameter whatever the scale of the cost) or
    /// `"momentum"` (Polyak's heavy ball, with `beta1` as the momentum: a
    /// velocity `beta1·v + learning_rate·gradient` is added to the
    /// parameters, so the `learning_rate` that works depends on the scale of
    /// the cost; `beta2` and `epsilon` are unused).
    #[getter]
    fn rule(&self) -> &'static str {
        self.rule.as_str()
    }

    #[setter]
    fn set_rule(&mut self, rule: &str) -> PyResult<()> {
        self.rule = parse_rule(rule)?;
        Ok(())
    }
}
