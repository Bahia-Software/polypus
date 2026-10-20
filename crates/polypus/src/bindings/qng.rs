use pyo3::prelude::*;

/// Quantum Natural Gradient optimizer configuration.
#[pyclass(module = "polypus")]
#[allow(clippy::upper_case_acronyms)]
pub struct QNG {
    #[pyo3(get, set)]
    pub max_iters: u32,
    #[pyo3(get, set)]
    pub bounds: (f64, f64),
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub finite_difference_step: f64,
    #[pyo3(get, set)]
    pub tikhonov_reg: f64,
    #[pyo3(get, set)]
    pub variance_function: Py<PyAny>,
    /// Optional RNG seed pinned on the optimizer object. Consumed by
    /// `train`/`qml.train` per the precedence rule (contract C-7): the explicit
    /// `seed` kwarg passed to the call wins; this field is the fallback; a fresh
    /// OS-entropy value is used when neither is set. `None` by default.
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl QNG {
    #[new]
    #[pyo3(signature = (variance_function, max_iters = 100, bounds = (-std::f64::consts::PI, std::f64::consts::PI), learning_rate = 0.1, finite_difference_step = 0.1, tikhonov_reg = 0.05, seed = None))]
    pub fn new(
        variance_function: Py<PyAny>,
        max_iters: u32,
        bounds: (f64, f64),
        learning_rate: f64,
        finite_difference_step: f64,
        tikhonov_reg: f64,
        seed: Option<u64>,
    ) -> Self {
        QNG {
            variance_function,
            max_iters,
            bounds,
            learning_rate,
            finite_difference_step,
            tikhonov_reg,
            seed,
        }
    }
}
