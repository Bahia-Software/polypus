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
}

#[pymethods]
impl DE {
    #[new]
    #[pyo3(signature = (generations = 100, population_size = 50, tolerance = 0.01))]
    pub fn new(generations: u32, population_size: u32, tolerance: f64) -> Self {
        DE {
            generations,
            population_size,
            tolerance,
        }
    }
}
