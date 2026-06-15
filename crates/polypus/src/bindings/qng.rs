use pyo3::prelude::*;

/// Quantum Natural Gradient optimizer configuration.
#[pyclass]
pub struct QNG {
	#[pyo3(get, set)] pub max_iters: u32,
	#[pyo3(get, set)] pub bounds: (f64, f64),
	#[pyo3(get, set)] pub learning_rate: f64,
	#[pyo3(get, set)] pub finite_difference_step: f64,
	#[pyo3(get, set)] pub tikhonov_reg: f64,
	#[pyo3(get, set)] pub variance_function: Py<PyAny>,
}

#[pymethods]
impl QNG {
	#[new]
	#[pyo3(signature = (variance_function, max_iters = 100, bounds = (-std::f64::consts::PI, std::f64::consts::PI), learning_rate = 0.1, finite_difference_step = 0.1, tikhonov_reg = 0.05))]
	pub fn new(
		variance_function: Py<PyAny>,
		max_iters: u32,
		bounds: (f64, f64),
		learning_rate: f64,
		finite_difference_step: f64,
		tikhonov_reg: f64,
	) -> Self {
		QNG { variance_function, max_iters, bounds, learning_rate, finite_difference_step, tikhonov_reg }
	}
}
