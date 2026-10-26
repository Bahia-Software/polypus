use pyo3::prelude::*;

/// SPSA optimizer configuration: Spall's simultaneous perturbation stochastic
/// approximation, climbing the fitness along a gradient estimated from two
/// circuits per random perturbation, whatever the number of parameters.
///
/// The run starts with one calibration batch of 50 circuits around the
/// starting point that sets the gain so that, with `resamplings=1`, the first
/// step moves each parameter by about `target_step`; with more perturbations
/// their average is smaller, so the first step is shorter. Each iteration then sends one batch of
/// `2·resamplings` circuits, plus the current parameters on the first
/// iteration and every `eval_interval` iterations after it; after the last
/// step, one more circuit measures the parameters it reached.
/// `best_params` is the best of the parameters measured. The run always makes
/// `max_iters` iterations and reports `converged=False`. The settings are
/// validated when `train`/`qml.train` starts, before any circuit runs: an
/// invalid one raises `ValueError` (contract C-5).
#[pyclass(module = "polypus", name = "SPSA")]
pub struct Spsa {
    /// Iterations, each one batch of `2·resamplings` circuits (plus one when
    /// the parameters are measured), after the calibration batch; the final
    /// parameters cost one circuit more. At least 1.
    #[pyo3(get, set)]
    pub max_iters: u32,
    /// Random perturbations averaged per iteration, K. At least 1; `1` is
    /// Spall's SPSA. Each costs two circuits of the iteration's batch, so a
    /// larger K fills more QPUs with the same number of batches.
    #[pyo3(get, set)]
    pub resamplings: u32,
    /// Perturbation size `c` in radians, finite and `> 0`: iteration k
    /// perturbs every parameter by `±c/k^gamma`. Defaults to `0.1`. Shot noise
    /// in the two fitness values is divided by the perturbation, so a small
    /// one ruins the estimate.
    #[pyo3(get, set)]
    pub perturbation: f64,
    /// Size in radians of the first step on each parameter with
    /// `resamplings=1`, finite and `> 0`, which the calibration batch turns
    /// into the gain `a`; with more perturbations the first step is shorter.
    /// Defaults to `0.63`. Later steps shrink as `1/k^alpha`.
    #[pyo3(get, set)]
    pub target_step: f64,
    /// Decay exponent of the step gain, in `(0.5, 1]`. Defaults to Spall's
    /// practical `0.602`.
    #[pyo3(get, set)]
    pub alpha: f64,
    /// Decay exponent of the perturbation, `> 0` with `alpha - gamma > 0.5`.
    /// Defaults to Spall's practical `0.101`.
    #[pyo3(get, set)]
    pub gamma: f64,
    /// Iterations between measurements of the parameters themselves, at
    /// least 1. Defaults to 20: the first iteration and every 20th after it
    /// add one circuit to their batch; the final parameters are measured
    /// whatever the interval.
    #[pyo3(get, set)]
    pub eval_interval: u32,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl Spsa {
    #[new]
    #[pyo3(signature = (max_iters = 300, resamplings = 1, perturbation = 0.1, target_step = 0.63, alpha = 0.602, gamma = 0.101, eval_interval = 20, seed = None))]
    pub fn new(
        max_iters: u32,
        resamplings: u32,
        perturbation: f64,
        target_step: f64,
        alpha: f64,
        gamma: f64,
        eval_interval: u32,
        seed: Option<u64>,
    ) -> Self {
        Spsa {
            max_iters,
            resamplings,
            perturbation,
            target_step,
            alpha,
            gamma,
            eval_interval,
            seed,
        }
    }
}
