use pyo3::prelude::*;

/// Differential Evolution optimizer configuration.
#[pyclass]
pub struct DE {
    #[pyo3(get, set)]
    pub generations: u32,
    #[pyo3(get, set)]
    pub population_size: u32,
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
impl DE {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = 50, tolerance = 0.01, seed = None))]
    pub fn new(generations: u32, population_size: u32, tolerance: f64, seed: Option<u64>) -> Self {
        DE {
            generations,
            population_size,
            tolerance,
            seed,
        }
    }
}
