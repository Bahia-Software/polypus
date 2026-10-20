use crate::{EvaluationError, OracleErrorSlot};
use polypus_infrastructure::{attach_or, InterpreterUnavailable};
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
            .map_err(|e| EvaluationError::Python(e.into()))?
            .extract()
            .map_err(|e: PyErr| EvaluationError::Python(e.into()))?;
        // A QFIM diagonal element must be finite and non-negative — zero is fine
        // (Tikhonov regularisation keeps the QNG division well-posed). Reject a
        // NaN/infinite/negative value here so it cannot silently corrupt the
        // natural-gradient update (θ ← θ − η·∇/qfim).
        if !value.is_finite() || value < 0.0 {
            return Err(EvaluationError::InvalidVariance { param_index, value });
        }
        Ok(value)
    }

    /// Record that the interpreter could not be attached to, so the entry point
    /// raises it once `optimize` returns.
    fn record_unavailable(&self) {
        self.errors.record(
            Box::new(EvaluationError::from(InterpreterUnavailable)),
            &self.run_id,
        );
    }
}

/// QNG calls these while the run is detached, so they attach with [`attach_or`]
/// (docs/ENGINEERING.md §9). The trait cannot return an error: an unreachable
/// interpreter is recorded in the error slot, like a callback failure, and the
/// finite sentinel is returned.
impl VarianceOracle for PyVarianceOracle {
    fn variance(&self, theta: &[f64], param_index: usize) -> f64 {
        attach_or(
            || {
                self.record_unavailable();
                0.0
            },
            |py| self.call(py, theta, param_index),
        )
    }

    fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        attach_or(
            || {
                self.record_unavailable();
                vec![0.0; dims]
            },
            |py| (0..dims).map(|a| self.call(py, theta, a)).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_infrastructure::{BackendError, InterpreterUnavailable};
    use pyo3::types::PyModule;

    /// QNG calls the variance oracle while the run is detached, and its trait
    /// cannot return an error. When the interpreter cannot be attached to, both
    /// methods record `InterpreterUnavailable` in the run's error slot and return
    /// the finite sentinel, without calling `variance_function` (which records
    /// every call). The entry point then raises the recorded error.
    #[test]
    fn records_the_failure_without_calling_python_when_attaching_is_forbidden() {
        pyo3::Python::initialize();
        let (module, variance_function) = Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"calls = []\ndef variance(theta, a):\n    calls.append(a)\n    return 1.0\n",
                c"variance_recorder.py",
                c"variance_recorder",
            )
            .expect("the recorder compiles");
            let function = module.getattr("variance").expect("the callback exists");
            (module.unbind(), function.unbind())
        });
        let errors = OracleErrorSlot::new();
        let oracle = PyVarianceOracle {
            variance_function,
            errors: errors.clone(),
            run_id: "variance-without-interpreter".to_string(),
        };
        let (single, diagonal) = crate::test_support::run_while_attaching_is_forbidden(move || {
            (
                oracle.variance(&[0.1, 0.2], 0),
                oracle.variance_diagonal(&[0.1, 0.2], 2),
            )
        });
        assert_eq!(single, 0.0);
        assert_eq!(diagonal, vec![0.0, 0.0]);
        let recorded = errors
            .take()
            .expect("the failure is recorded")
            .downcast::<EvaluationError>()
            .expect("an EvaluationError is recorded");
        match *recorded {
            EvaluationError::Backend(BackendError::External(boxed)) => {
                assert!(boxed.is::<InterpreterUnavailable>(), "{boxed}")
            }
            other => panic!("expected Backend(External(InterpreterUnavailable)), got {other:?}"),
        }
        Python::attach(|py| {
            let calls = module.bind(py).getattr("calls").unwrap();
            assert_eq!(
                calls.len().unwrap(),
                0,
                "variance_function must not be called"
            );
        });
    }
}
