use crate::{EvaluationError, OracleErrorSlot};
use polypus_optimizers::VarianceOracle;
use pyo3::prelude::*;

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
    /// Effective run id (`ExecutionConfig::id`) of the training run this adapter
    /// serves, so a recorded failure names its run in the log. Owned rather than
    /// borrowed: the adapter is boxed into the optimizer args and outlives the
    /// entry-point frame that built it.
    pub run_id: String,
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
                // Type-erase into the (pyo3-free) slot; the edge downcasts it back.
                self.errors.record(Box::new(e), &self.run_id);
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
        let value: f64 = self
            .variance_function
            .bind(py)
            .call1((theta.to_vec(), param_index as u32))
            .map_err(EvaluationError::Python)?
            .extract()
            .map_err(EvaluationError::Python)?;
        // A QFIM diagonal element must be finite and non-negative — zero is fine
        // (Tikhonov regularisation keeps the QNG division well-posed). Reject a
        // NaN/infinite/negative value here so it cannot silently corrupt the
        // natural-gradient update (θ ← θ − η·∇/qfim).
        if !value.is_finite() || value < 0.0 {
            return Err(EvaluationError::InvalidVariance { param_index, value });
        }
        Ok(value)
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
