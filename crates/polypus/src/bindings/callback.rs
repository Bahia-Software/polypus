//! The `callback` of `train`, `qml.train` and `qml.Model.fit` (issue #500):
//! called once per iteration of the optimizer with a [`TrainProgress`], it can
//! watch the run, stop it, or write a checkpoint.
//!
//! The pure side is [`TrainObserver`](crate::orchestration::TrainObserver),
//! which a training flow shows every iteration with the evaluations spent so
//! far. This module holds the Python side: [`validate_callback`], the read-only
//! [`TrainProgress`] record the callback receives, and [`PyTrainObserver`], the
//! observer that calls it.
//!
//! The run is detached from the interpreter, as every training run is. Each
//! call attaches with [`attach_or`] for that call only, as the external
//! optimizer's `ask()` and `tell()` do, and first checks for a pending signal,
//! so Ctrl+C still stops the run before the callback starts. An exception the
//! callback raises, or a value it returns that is neither a stop nor a go-on,
//! is recorded in the run's [`OracleErrorSlot`] as
//! [`EvaluationError::Python`], and the edge raises it as itself once the run
//! returns (contract C-1): no panic crosses the edge.

use std::ops::ControlFlow;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::evaluation::{EvaluationError, OracleErrorSlot};
use crate::infrastructure::{attach_or, InterpreterUnavailable};
use crate::orchestration::{TrainObserver, TrainProgress as Progress};

/// The state of a training run at the end of one iteration of its optimizer,
/// as the `callback` of `train`, `qml.train` and `qml.Model.fit` receives it.
///
/// Read-only: the run built it for this call, and `best_params` is a new list
/// on every access, so nothing the callback does to it reaches the run.
#[pyclass(module = "polypus", name = "TrainProgress", frozen)]
pub struct TrainProgress {
    /// Iterations completed, this one included: 1 on the first call, and
    /// `TrainResult.iterations_run` on the last. An iteration is a generation
    /// of DE, PSO or CMA-ES, an iteration of QNG, Adam or SPSA, a step of
    /// Rotosolve, and an ask/tell batch of `External`.
    #[pyo3(get)]
    pub iteration: usize,
    /// The best fitness so far (higher is better): the entry this iteration
    /// leaves in `TrainResult.fitness_history`, so it never falls. Under a
    /// minibatch (`Model.fit` with `batch_size`), a minibatch estimate, except
    /// for Adam, which tracks its best on every sample.
    #[pyo3(get)]
    pub best_fitness: f64,
    /// The parameters [`best_fitness`](Self::best_fitness) belongs to.
    #[pyo3(get)]
    pub best_params: Vec<f64>,
    /// Candidates the optimizer has had scored so far, every candidate of
    /// every batch. In `train` a candidate is one circuit; in `qml.train` and
    /// `Model.fit` it is one circuit per training sample (per minibatch
    /// sample with `batch_size`, and on every sample when a minibatched Adam
    /// measures its point). With an exact gradient (Adam in `Model.fit`), the
    /// point it measures counts and its shifted circuits do not. The calls of
    /// QNG's `variance_function` and the final batch of `final_shots` are not
    /// counted either.
    #[pyo3(get)]
    pub evaluations: u64,
}

#[pymethods]
impl TrainProgress {
    fn __repr__(&self) -> String {
        format!(
            "TrainProgress(iteration={}, best_fitness={}, evaluations={}, best_params={:?})",
            self.iteration, self.best_fitness, self.evaluations, self.best_params
        )
    }
}

/// Check the `callback` of `train`, `qml.train` or `Model.fit` before anything
/// runs: `None`, or a callable. Anything else is a `TypeError`, raised before
/// any backend or CUNQA allocation exists.
pub(crate) fn validate_callback(
    callback: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Py<PyAny>>> {
    match callback.filter(|callback| !callback.is_none()) {
        None => Ok(None),
        Some(callback) if callback.is_callable() => Ok(Some(callback.clone().unbind())),
        Some(callback) => Err(PyTypeError::new_err(format!(
            "callback must be a callable or None, got {}",
            shown(callback)
        ))),
    }
}

/// The [`TrainObserver`] of a run with a `callback`: each iteration it calls
/// `callback(progress)` with a new [`TrainProgress`], and stops the run when
/// the callback returns `True`.
///
/// It does not call Python once the run's error slot holds a failure (a
/// backend error, a Ctrl+C caught between waves, an earlier call's
/// exception): the run is over. The flow stops such a run before calling it,
/// and this check keeps the observer safe on its own.
pub(crate) struct PyTrainObserver {
    callback: Py<PyAny>,
    errors: OracleErrorSlot,
    run_id: String,
}

impl PyTrainObserver {
    pub(crate) fn new(callback: Py<PyAny>, errors: &OracleErrorSlot, run_id: &str) -> Self {
        Self {
            callback,
            errors: errors.clone(),
            run_id: run_id.to_string(),
        }
    }
}

impl TrainObserver for PyTrainObserver {
    fn observe(&mut self, progress: &Progress<'_>) -> ControlFlow<()> {
        if self.errors.failed() {
            return ControlFlow::Break(());
        }
        let callback = &self.callback;
        let answered = attach_or(
            || Err(EvaluationError::from(InterpreterUnavailable)),
            |py| {
                let call = || -> PyResult<ControlFlow<()>> {
                    py.check_signals()?;
                    let record = TrainProgress {
                        iteration: progress.iteration,
                        best_fitness: progress.best_fitness,
                        best_params: progress.best_params.to_vec(),
                        evaluations: progress.evaluations,
                    };
                    let returned = callback.bind(py).call1((record,))?;
                    answer(&returned)
                };
                call().map_err(|e| EvaluationError::Python(e.into()))
            },
        );
        match answered {
            Ok(answer) => answer,
            Err(err) => {
                self.errors
                    .record_as(Box::new(err), &self.run_id, "the training callback failed");
                ControlFlow::Break(())
            }
        }
    }
}

/// What the callback's return value asks for: `True` (or a NumPy `bool_` that
/// is true) stops the run, `None` or `False` lets it go on. Any other value,
/// a number or a list included, is a `TypeError` that says what it was: a
/// callback that meant something else by it would otherwise run on silently.
fn answer(returned: &Bound<'_, PyAny>) -> PyResult<ControlFlow<()>> {
    let stop = if returned.is_none() {
        false
    } else if let Ok(value) = returned.cast::<PyBool>() {
        value.is_true()
    } else if is_numpy_bool(returned) {
        returned.is_truthy()?
    } else {
        return Err(PyTypeError::new_err(format!(
            "the training callback must return True to stop the run, or None or False to \
             go on; it returned {}",
            shown(returned)
        )));
    };
    Ok(if stop {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    })
}

/// Whether `value` is a NumPy boolean scalar (`numpy.bool_`, named
/// `numpy.bool` from NumPy 2), recognised by its type without importing NumPy.
fn is_numpy_bool(value: &Bound<'_, PyAny>) -> bool {
    let kind = value.get_type();
    let module = kind.module().map(|m| m.to_string()).unwrap_or_default();
    let name = kind.name().map(|n| n.to_string()).unwrap_or_default();
    module == "numpy" && (name == "bool_" || name == "bool")
}

/// `value`'s `repr` and type for an error message, or a fixed text when the
/// `repr` fails.
fn shown(value: &Bound<'_, PyAny>) -> String {
    let kind = value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_string(), |n| n.to_string());
    match value.repr() {
        Ok(repr) => format!("{repr} (of type {kind})"),
        Err(_) => format!("an unprintable value (of type {kind})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;
    use std::ffi::CString;

    /// A module holding `source`, under a unique name.
    fn module<'py>(py: Python<'py>, source: &str, name: &str) -> Bound<'py, PyModule> {
        let code = CString::new(source).expect("no NUL in the source");
        let file = CString::new(format!("{name}.py")).expect("no NUL in the name");
        let name = CString::new(name).expect("no NUL in the name");
        PyModule::from_code(py, &code, &file, &name).expect("the test module compiles")
    }

    /// A callback that keeps every record it gets and returns `RETURNS`.
    const RECORDING: &str = "\
seen = []
RETURNS = None
def callback(progress):
    seen.append(progress)
    return RETURNS
";

    fn progress(params: &[f64]) -> Progress<'_> {
        Progress {
            iteration: 3,
            best_fitness: -0.25,
            best_params: params,
            evaluations: 42,
        }
    }

    /// Call the observer for `callback` from a detached thread, as a run does.
    fn observe(
        py: Python<'_>,
        callback: &Bound<'_, PyAny>,
        errors: &OracleErrorSlot,
    ) -> ControlFlow<()> {
        let mut observer = PyTrainObserver::new(callback.clone().unbind(), errors, "callback-test");
        py.detach(|| observer.observe(&progress(&[0.5, 1.5])))
    }

    /// The class of the exception recorded in `errors`.
    fn recorded_class(py: Python<'_>, errors: &OracleErrorSlot) -> (String, String) {
        let recorded = errors.take().expect("the failure is recorded");
        let recorded = recorded
            .downcast::<EvaluationError>()
            .expect("an EvaluationError");
        let EvaluationError::Python(err) = *recorded else {
            panic!("a Python exception");
        };
        let err = err.into_inner();
        (
            err.get_type(py).name().unwrap().to_string(),
            err.value(py).to_string(),
        )
    }

    #[test]
    fn the_callback_gets_the_iterations_record() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, RECORDING, "callback_record");
            let errors = OracleErrorSlot::new();
            let answer = observe(py, &module.getattr("callback").unwrap(), &errors);
            assert_eq!(answer, ControlFlow::Continue(()));
            assert!(!errors.failed());
            let seen = module.getattr("seen").unwrap();
            let record = seen.get_item(0).unwrap();
            assert_eq!(
                record
                    .getattr("iteration")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );
            assert_eq!(
                record
                    .getattr("best_fitness")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                -0.25
            );
            assert_eq!(
                record
                    .getattr("best_params")
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap(),
                vec![0.5, 1.5]
            );
            assert_eq!(
                record
                    .getattr("evaluations")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                42
            );
            assert_eq!(
                record.repr().unwrap().to_string(),
                "TrainProgress(iteration=3, best_fitness=-0.25, evaluations=42, \
                 best_params=[0.5, 1.5])"
            );
        });
    }

    /// The record is read-only, and `best_params` is a copy: mutating it
    /// changes neither the record nor the run.
    #[test]
    fn the_record_is_read_only_and_its_params_a_copy() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, RECORDING, "callback_read_only");
            let errors = OracleErrorSlot::new();
            let answer = observe(py, &module.getattr("callback").unwrap(), &errors);
            assert_eq!(answer, ControlFlow::Continue(()));
            let record = module.getattr("seen").unwrap().get_item(0).unwrap();
            for field in ["iteration", "best_fitness", "best_params", "evaluations"] {
                let err = record
                    .setattr(field, 0)
                    .expect_err("the record is read-only");
                assert!(
                    err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py),
                    "{field}"
                );
            }
            let params = record.getattr("best_params").unwrap();
            params.set_item(0, 99.0).unwrap();
            assert_eq!(
                record
                    .getattr("best_params")
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap(),
                vec![0.5, 1.5]
            );
        });
    }

    #[test]
    fn true_stops_and_none_or_false_go_on() {
        Python::initialize();
        Python::attach(|py| {
            let numpy = py.import("numpy").ok();
            let module = module(py, RECORDING, "callback_returns");
            let mut cases: Vec<(Bound<'_, PyAny>, ControlFlow<()>)> = vec![
                (py.None().into_bound(py), ControlFlow::Continue(())),
                (
                    PyBool::new(py, false).to_owned().into_any(),
                    ControlFlow::Continue(()),
                ),
                (
                    PyBool::new(py, true).to_owned().into_any(),
                    ControlFlow::Break(()),
                ),
            ];
            if let Some(numpy) = &numpy {
                let bool_ = numpy.getattr("bool_").unwrap();
                cases.push((bool_.call1((true,)).unwrap(), ControlFlow::Break(())));
                cases.push((bool_.call1((false,)).unwrap(), ControlFlow::Continue(())));
            }
            for (returned, expected) in cases {
                module.setattr("RETURNS", &returned).unwrap();
                let errors = OracleErrorSlot::new();
                let answer = observe(py, &module.getattr("callback").unwrap(), &errors);
                assert_eq!(answer, expected, "{returned:?}");
                assert!(!errors.failed(), "{returned:?}");
            }
        });
    }

    /// Any other return value is a `TypeError` that names it, recorded as the
    /// run's failure, and the run stops.
    #[test]
    fn any_other_return_value_is_a_type_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, RECORDING, "callback_bad_returns");
            for (value, shown) in [
                ("1", "1 (of type int)"),
                ("0", "0 (of type int)"),
                ("1.0", "1.0 (of type float)"),
                ("[True]", "[True] (of type list)"),
                ("'stop'", "'stop' (of type str)"),
            ] {
                let code = CString::new(value).unwrap();
                let returned = py.eval(&code, None, None).unwrap();
                module.setattr("RETURNS", returned).unwrap();
                let errors = OracleErrorSlot::new();
                let answer = observe(py, &module.getattr("callback").unwrap(), &errors);
                assert_eq!(answer, ControlFlow::Break(()), "{value}");
                let (class, message) = recorded_class(py, &errors);
                assert_eq!(class, "TypeError", "{value}");
                assert!(
                    message.ends_with(&format!("it returned {shown}")),
                    "{message}"
                );
            }
        });
    }

    /// An exception the callback raises is recorded as itself, class and
    /// message, `KeyboardInterrupt` included, and the run stops.
    #[test]
    fn an_exception_is_recorded_as_itself() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(
                py,
                "\
class Checkpoint(Exception):
    pass
def failing(progress):
    raise Checkpoint('disk full')
def interrupted(progress):
    raise KeyboardInterrupt
",
                "callback_raises",
            );
            for (name, class, message) in [
                ("failing", "Checkpoint", "disk full"),
                ("interrupted", "KeyboardInterrupt", ""),
            ] {
                let errors = OracleErrorSlot::new();
                let answer = observe(py, &module.getattr(name).unwrap(), &errors);
                assert_eq!(answer, ControlFlow::Break(()), "{name}");
                assert_eq!(
                    recorded_class(py, &errors),
                    (class.to_string(), message.to_string()),
                    "{name}"
                );
            }
        });
    }

    /// Once the slot holds a failure, the callback is not called.
    #[test]
    fn a_failed_run_does_not_call_the_callback() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, RECORDING, "callback_failed_run");
            let errors = OracleErrorSlot::new();
            errors.record(
                Box::new(EvaluationError::Runtime("backend failed".to_string())),
                "callback-test",
            );
            let answer = observe(py, &module.getattr("callback").unwrap(), &errors);
            assert_eq!(answer, ControlFlow::Break(()));
            assert_eq!(module.getattr("seen").unwrap().len().unwrap(), 0);
        });
    }

    #[test]
    fn only_a_callable_or_none_is_a_callback() {
        Python::initialize();
        Python::attach(|py| {
            assert!(validate_callback(None).unwrap().is_none());
            assert!(validate_callback(Some(&py.None().into_bound(py)))
                .unwrap()
                .is_none());
            let print = py.eval(c"print", None, None).unwrap();
            assert!(validate_callback(Some(&print)).unwrap().is_some());
            for value in [c"1", c"'callback'", c"[print]"] {
                let value = py.eval(value, None, None).unwrap();
                let err = validate_callback(Some(&value)).expect_err("not a callback");
                assert!(err.is_instance_of::<PyTypeError>(py), "{value:?}");
                assert!(
                    err.value(py)
                        .to_string()
                        .starts_with("callback must be a callable or None, got "),
                    "{value:?}"
                );
            }
        });
    }
}
