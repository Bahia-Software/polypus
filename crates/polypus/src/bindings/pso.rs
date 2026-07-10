use pyo3::prelude::*;

/// Particle Swarm Optimization configuration.
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct PSO {
    #[pyo3(get, set)]
    pub generations: u32,
    #[pyo3(get, set)]
    pub population_size: u32,
    #[pyo3(get, set)]
    pub bounds: (f64, f64),
    #[pyo3(get, set)]
    pub inertia_weight: f64,
    #[pyo3(get, set)]
    pub cognitive_weight: f64,
    #[pyo3(get, set)]
    pub social_weight: f64,
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl PSO {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = 50, bounds = (-std::f64::consts::PI, std::f64::consts::PI), inertia_weight = 0.5, cognitive_weight = 1.0, social_weight = 1.0, tolerance = 0.01, seed = None))]
    // Constructor mirrors PSO's full hyper-parameter set as Python kwargs; the
    // added `seed` pushes it one past the lint threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        generations: u32,
        population_size: u32,
        bounds: (f64, f64),
        inertia_weight: f64,
        cognitive_weight: f64,
        social_weight: f64,
        tolerance: f64,
        seed: Option<u64>,
    ) -> Self {
        PSO {
            generations,
            population_size,
            bounds,
            inertia_weight,
            cognitive_weight,
            social_weight,
            tolerance,
            seed,
        }
    }
}
