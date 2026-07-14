use crate::evaluation::InterruptState;
use polypus_optimizers::VarianceOracle;
use pyo3::prelude::*;
use std::sync::Arc;

/// Quantum Natural Gradient optimizer configuration.
#[pyclass]
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

/// PyO3 adapter that lets a Python `variance_function` satisfy the pure-Rust
/// [`VarianceOracle`] contract consumed by the QNG optimizer.
///
/// This is the single point where the GIL is touched on the variance path: the
/// pure optimizer stays Python-free and calls back through this trait object.
/// [`variance_diagonal`](VarianceOracle::variance_diagonal) is overridden to
/// acquire the GIL **once** and evaluate every dimension in a tight loop,
/// preserving the original implementation's single-acquisition semantics.
pub struct PyVarianceOracle {
    /// Python callable `fn(theta: list[float], a: int) -> float`.
    pub variance_function: Py<PyAny>,
    /// Cancellation state shared with the run's [`VqcOracle`](crate::evaluation::VqcOracle).
    /// QNG hits both oracles per iteration; sharing the state means a Ctrl+C (or
    /// a `variance_function` exception) observed on either path short-circuits
    /// the whole run and is re-raised by the entry point as the original
    /// exception (see [`InterruptState`]).
    pub interrupt: Arc<InterruptState>,
}

impl PyVarianceOracle {
    /// Call the Python `variance_function(theta, param_index)` under an already
    /// held GIL and extract the returned float, propagating any exception it
    /// raises rather than swallowing it into a panic with `.expect()`.
    fn call(&self, py: Python<'_>, theta: &[f64], param_index: usize) -> PyResult<f64> {
        self.variance_function
            .bind(py)
            .call1((theta.to_vec(), param_index as u32))?
            .extract()
    }
}

impl VarianceOracle for PyVarianceOracle {
    fn variance(&self, theta: &[f64], param_index: usize) -> f64 {
        if self.interrupt.is_interrupted() {
            return 0.0;
        }
        Python::with_gil(|py| match self.call(py, theta, param_index) {
            Ok(v) => v,
            Err(err) => {
                self.interrupt.capture(err);
                0.0
            }
        })
    }

    fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        // Once interrupted, return a finite placeholder diagonal without calling
        // Python. QNG adds Tikhonov regularisation to every element, so a zero
        // here can never cause a divide-by-zero; the outcome is discarded anyway.
        if self.interrupt.is_interrupted() {
            return vec![0.0; dims];
        }
        Python::with_gil(|py| {
            let mut diagonal = Vec::with_capacity(dims);
            for a in 0..dims {
                match self.call(py, theta, a) {
                    Ok(v) => diagonal.push(v),
                    Err(err) => {
                        // Capture the exception, stop calling Python, and pad the
                        // rest with placeholders (the entry point re-raises it).
                        self.interrupt.capture(err);
                        diagonal.resize(dims, 0.0);
                        break;
                    }
                }
            }
            diagonal
        })
    }
}
