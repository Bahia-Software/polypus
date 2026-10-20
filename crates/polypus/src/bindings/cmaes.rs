use pyo3::prelude::*;

/// CMA-ES optimizer configuration: the covariance matrix adaptation evolution
/// strategy, with optional IPOP restarts that double the population.
///
/// The mean and the samples live unwrapped in ℝ^d and every candidate is
/// wrapped into `[0, 2π)` before it is evaluated, like DE's. The settings are
/// validated when `train`/`qml.train` starts, before any circuit runs: an
/// invalid one raises `ValueError` (contract C-5).
#[pyclass(module = "polypus")]
#[allow(clippy::upper_case_acronyms)]
pub struct CMAES {
    /// Total generations over the first run and every restart; each is one
    /// batch of `population_size` circuits (doubled after each restart). At
    /// least 1.
    #[pyo3(get, set)]
    pub generations: u32,
    /// Population λ of the first run, at least 2; `None` (the default) is
    /// Hansen's `4 + ⌊3 ln d⌋` for `d = dimensions`. Each restart doubles it.
    #[pyo3(get, set)]
    pub population_size: Option<u32>,
    /// Initial step size σ₀ in radians, finite and in `(0, 2π]`; every restart
    /// starts from it again. Defaults to π/2. The spread of each angle stays
    /// within `max(σ₀, π/2)` during the run.
    #[pyo3(get, set)]
    pub sigma0: f64,
    /// Minimum improvement of a run's best fitness over its last `patience`
    /// generations below which the run ends (DE's rule, contract C-5).
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Generations over which a run's best-fitness improvement is measured.
    /// Defaults to 20.
    #[pyo3(get, set)]
    pub patience: u32,
    /// IPOP restarts allowed after the first run ends; each doubles the
    /// population and shares the `generations` budget. Defaults to 0.
    #[pyo3(get, set)]
    pub restarts: u32,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl CMAES {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = None, sigma0 = std::f64::consts::FRAC_PI_2, tolerance = 0.01, patience = 20, restarts = 0, seed = None))]
    pub fn new(
        generations: u32,
        population_size: Option<u32>,
        sigma0: f64,
        tolerance: f64,
        patience: u32,
        restarts: u32,
        seed: Option<u64>,
    ) -> Self {
        CMAES {
            generations,
            population_size,
            sigma0,
            tolerance,
            patience,
            restarts,
            seed,
        }
    }
}
