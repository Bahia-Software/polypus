use crate::evaluation::{EvaluationError, OracleErrorSlot};
use polypus_optimizers::VarianceOracle;
use pyo3::prelude::*;

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
    /// Shared with the `train`/`qml.train` entry point: a failure calling the
    /// user's `variance_function` is recorded here and surfaced as a `PyErr`
    /// after `optimize` returns, since [`VarianceOracle`] cannot return a
    /// `Result`.
    pub errors: OracleErrorSlot,
}

impl PyVarianceOracle {
    /// Call the Python `variance_function(theta, param_index)` under an already
    /// held GIL, recording any failure and returning a finite sentinel so the
    /// optimizer never observes a panic (contract C-5 keeps outputs finite).
    fn call(&self, py: Python<'_>, theta: &[f64], param_index: usize) -> f64 {
        if self.errors.failed() {
            return 0.0;
        }
        match self.try_call(py, theta, param_index) {
            Ok(value) => value,
            Err(e) => {
                self.errors.record(e);
                0.0
            }
        }
    }

    /// Fallible core of [`call`](Self::call): invoke the user callback and
    /// extract the float, carrying any Python error verbatim.
    fn try_call(
        &self,
        py: Python<'_>,
        theta: &[f64],
        param_index: usize,
    ) -> Result<f64, EvaluationError> {
        self.variance_function
            .bind(py)
            .call1((theta.to_vec(), param_index as u32))
            .map_err(EvaluationError::Python)?
            .extract()
            .map_err(EvaluationError::Python)
    }
}

impl VarianceOracle for PyVarianceOracle {
    fn variance(&self, theta: &[f64], param_index: usize) -> f64 {
        Python::with_gil(|py| self.call(py, theta, param_index))
    }

    fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        Python::with_gil(|py| (0..dims).map(|a| self.call(py, theta, a)).collect())
    }
}
