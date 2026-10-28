//! `polypus.External`: an external ask/tell optimizer run through Polypus's
//! scheduler (issue #399).
//!
//! The pure bridge, [`AlgorithmExternal`](polypus_optimizers::AlgorithmExternal),
//! asks a [`AskTell`] for batches, evaluates each one through the oracle and
//! tells the values back. This module holds the Python side: the `External`
//! configuration object and [`PyAskTell`], the [`AskTell`] that calls the user's
//! object's `ask()` and `tell()`.
//!
//! The run is detached from the interpreter, as every training run is. Each
//! `ask()` and `tell()` attaches with [`attach_or`] for that call only, on the
//! thread that called `train`, as the QNG variance oracle does; other Python
//! threads run while a batch is evaluated. An exception from either call, a
//! conversion that fails, or a pending Ctrl+C, is recorded in the run's
//! [`OracleErrorSlot`] as [`EvaluationError::Python`], and the edge raises it
//! as itself once the run returns (contract C-1): no panic crosses the edge.

use pyo3::exceptions::{PyAttributeError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList};
use pyo3::{PyTraverseError, PyVisit};

use crate::evaluation::{EvaluationError, OracleErrorSlot};
use crate::infrastructure::{attach_or, InterpreterUnavailable};
use polypus_optimizers::{AskTell, AskTellError};

/// Run an external ask/tell optimizer (Optuna, Nevergrad, pycma or your own
/// object) through Polypus's scheduler.
///
/// `optimizer` is any object with two methods:
///
/// - `ask()` returns the next batch of candidates: an iterable of rows (a list
///   of lists, a 2-D NumPy array), each row `dimensions` finite numbers. `None`
///   or an empty batch means the optimizer is done.
/// - `tell(candidates, values)` receives the rows of that batch, as a
///   `list[list[float]]`, and their fitness, as a `list[float]` in the same
///   order. Fitness is to be MAXIMISED, as everywhere in Polypus: a library
///   that minimises must be told `-value`. The adapters in `polypus.external`
///   wrap Optuna, Nevergrad and pycma this way.
///
/// Each `ask()` is one batch of the oracle, spread over the QPUs as any
/// optimizer's batch is. The run stops when `ask()` returns `None` or an empty
/// batch, or before a batch that would take the evaluations past
/// `max_evaluations`: that batch is neither evaluated nor told, so the budget
/// is never exceeded; a batch is read only up to one row past the budget
/// left, and a row up to one value past `dimensions`, so an endless iterator
/// does not hang the run. A row that is not
/// `dimensions` finite real numbers (a `bool` or a complex number is not one)
/// stops the run with `ValueError` naming its batch and row, before the batch
/// is evaluated. An exception raised in `ask()` or `tell()`,
/// `KeyboardInterrupt` included, stops the run and `train`/`qml.train` raises
/// it as itself.
///
/// `best_params` and `best_fitness` are the best candidate Polypus measured,
/// whatever the library reports as its own best; `fitness_history` has one
/// entry per batch, the best value measured up to it; `iterations_run` is the
/// number of batches evaluated and told; `converged` is True when the
/// optimizer ended the run with an empty `ask()` and False when the budget
/// did (contract C-5). With `final_shots`, the five best distinct candidates
/// are re-evaluated, as for DE, PSO and CMA-ES.
///
/// The library owns its seed and its start (contract C-7). `External` has no
/// `seed`: the `seed` of `train`/`qml.train` still fixes the shot sampling and
/// is reported, while the library's own seed fixes its candidates. Passing
/// `initial_point` raises `ValueError` before anything runs; give the start to
/// the library, or to the `initial_point` of an adapter.
///
/// `ask()` and `tell()` run on the thread that called `train`, holding the GIL
/// only during each call. The object is used, not copied: a second run with the
/// same `External` continues from the library's state, and `max_evaluations`
/// counts each run's evaluations on their own. Two runs at the same time must
/// not share one optimizer.
///
/// `max_evaluations` must be an integer `>= 1`: a smaller one raises
/// `ValueError`, and a value that is not an integer (a `bool` or a `float`
/// too) `TypeError`; an `optimizer` without callable `ask` and `tell` raises
/// `TypeError`. Both are checked here and whenever they are set.
///
/// `External` takes part in Python's garbage collection, so an optimizer that
/// refers back to its `External` (a cycle) is freed with it.
#[pyclass(module = "polypus")]
pub struct External {
    optimizer: Py<PyAny>,
    max_evaluations: u64,
}

#[pymethods]
impl External {
    #[new]
    #[pyo3(signature = (optimizer, max_evaluations))]
    pub fn new(optimizer: Bound<'_, PyAny>, max_evaluations: Bound<'_, PyAny>) -> PyResult<Self> {
        check_optimizer(&optimizer)?;
        let max_evaluations = parse_max_evaluations(&max_evaluations)?;
        Ok(External {
            optimizer: optimizer.unbind(),
            max_evaluations,
        })
    }

    /// The ask/tell object this configuration runs.
    #[getter]
    fn optimizer(&self, py: Python<'_>) -> Py<PyAny> {
        self.optimizer.clone_ref(py)
    }

    #[setter]
    fn set_optimizer(&mut self, optimizer: Bound<'_, PyAny>) -> PyResult<()> {
        check_optimizer(&optimizer)?;
        self.optimizer = optimizer.unbind();
        Ok(())
    }

    /// The most candidates a run evaluates, an integer `>= 1`.
    #[getter]
    fn max_evaluations(&self) -> u64 {
        self.max_evaluations
    }

    #[setter]
    fn set_max_evaluations(&mut self, max_evaluations: Bound<'_, PyAny>) -> PyResult<()> {
        self.max_evaluations = parse_max_evaluations(&max_evaluations)?;
        Ok(())
    }

    /// The garbage collector's traversal: the optimizer is the one Python
    /// object an `External` holds.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.optimizer)
    }

    /// Break a reference cycle through the optimizer: the collector calls it
    /// on an `External` it is about to free, and the optimizer is replaced by
    /// `None`.
    fn __clear__(&mut self) {
        let none = Python::attach(|py| py.None());
        drop(std::mem::replace(&mut self.optimizer, none));
    }
}

impl External {
    /// The configuration and the [`AskTell`] a run of this object dispatches:
    /// `max_evaluations`, and the optimizer adapted for the run `run_id` of a
    /// circuit with `dimensions` parameters, recording its failures in
    /// `errors`.
    pub(crate) fn for_run(
        &self,
        py: Python<'_>,
        dimensions: u32,
        errors: &OracleErrorSlot,
        run_id: &str,
    ) -> (u64, PyAskTell) {
        (
            self.max_evaluations,
            PyAskTell {
                optimizer: self.optimizer.clone_ref(py),
                errors: errors.clone(),
                run_id: run_id.to_string(),
                asked: 0,
                remaining: self.max_evaluations,
                dimensions: usize::try_from(dimensions).unwrap_or(usize::MAX),
            },
        )
    }
}

/// `TypeError` unless `optimizer` has callable `ask` and `tell` attributes. An
/// exception other than `AttributeError` raised while reading one (a property
/// that fails) propagates unchanged.
fn check_optimizer(optimizer: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = optimizer.py();
    for method in ["ask", "tell"] {
        let callable = match optimizer.getattr(method) {
            Ok(attribute) => attribute.is_callable(),
            Err(err) if err.is_instance_of::<PyAttributeError>(py) => false,
            Err(err) => return Err(err),
        };
        if !callable {
            let kind = optimizer
                .get_type()
                .name()
                .map_or_else(|_| "the object".to_string(), |name| format!("{name}"));
            return Err(PyTypeError::new_err(format!(
                "polypus.External needs an optimizer with callable ask() and tell() methods; \
                 {kind} has no callable {method}()"
            )));
        }
    }
    Ok(())
}

/// `max_evaluations` as a `u64`: an integer from 1 to 2⁶⁴ − 1, otherwise
/// `ValueError`, or `TypeError` for a value that is not an integer (see
/// [`positive_index`](super::positive_index)).
fn parse_max_evaluations(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    let index = super::positive_index(value, "max_evaluations")?;
    index
        .extract::<u64>()
        .map_err(|_| super::too_large(value, "max_evaluations", u64::MAX))
}

/// The [`AskTell`] that calls a Python object's `ask()` and `tell()`.
///
/// Neither method calls Python once the run's error slot holds a failure (a
/// backend error, a Ctrl+C caught between waves, an earlier call's exception):
/// the run is over, and the library is never told the oracle's sentinel
/// values. Each call first checks for a pending signal, so a Ctrl+C pressed
/// during a batch stops the run before the library's next call starts.
pub(crate) struct PyAskTell {
    optimizer: Py<PyAny>,
    errors: OracleErrorSlot,
    run_id: String,
    /// The batches `ask()` has returned: the number of the next one, 0-based,
    /// as the bridge numbers them in its messages.
    asked: usize,
    /// The evaluations left in the run's budget: `max_evaluations` less the
    /// candidates told. A batch is read up to one row past it, enough for the
    /// bridge to see that it does not fit.
    remaining: u64,
    /// The values a row must have, one per parameter. A row is read up to one
    /// value past them, enough to tell that it is too long.
    dimensions: usize,
}

/// What `ask`/`tell` return once the run has failed; the slot holds the cause.
const RUN_FAILED: &str = "the run has already failed";

impl PyAskTell {
    /// Record `err` as the run's failure and return the error the bridge stops
    /// with. The dispatch returns the recorded failure instead of the bridge's
    /// error, so this message is never what `train` raises.
    fn fail(&self, err: EvaluationError, call: &str) -> AskTellError {
        self.errors.record_as(
            Box::new(err),
            &self.run_id,
            &format!("the external optimizer's {call}() failed"),
        );
        format!("{call}() raised; the run's error is the exception it raised").into()
    }
}

impl AskTell for PyAskTell {
    fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
        if self.errors.failed() {
            return Err(RUN_FAILED.into());
        }
        let batch = self.asked;
        let limit = usize::try_from(self.remaining.saturating_add(1)).unwrap_or(usize::MAX);
        let dimensions = self.dimensions;
        let optimizer = &self.optimizer;
        let proposed = attach_or(
            || Err(EvaluationError::from(InterpreterUnavailable)),
            |py| {
                let call = || {
                    py.check_signals()?;
                    let proposed = optimizer.bind(py).call_method0("ask")?;
                    rows_of(&proposed, batch, limit, dimensions)
                };
                call().map_err(|e| EvaluationError::Python(e.into()))
            },
        );
        match proposed {
            Ok(rows) => {
                self.asked += 1;
                Ok(rows)
            }
            Err(err) => Err(self.fail(err, "ask")),
        }
    }

    fn tell(&mut self, candidates: &[Vec<f64>], values: &[f64]) -> Result<(), AskTellError> {
        if self.errors.failed() {
            return Err(RUN_FAILED.into());
        }
        // The bridge tells only what it evaluated, so these are spent.
        let told = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
        self.remaining = self.remaining.saturating_sub(told);
        let optimizer = &self.optimizer;
        let told = attach_or(
            || Err(EvaluationError::from(InterpreterUnavailable)),
            |py| {
                let call = || -> PyResult<()> {
                    py.check_signals()?;
                    let rows = candidates
                        .iter()
                        .map(|row| PyList::new(py, row))
                        .collect::<PyResult<Vec<_>>>()?;
                    let rows = PyList::new(py, rows)?;
                    let values = PyList::new(py, values)?;
                    optimizer.bind(py).call_method1("tell", (rows, values))?;
                    Ok(())
                };
                call().map_err(|e| EvaluationError::Python(e.into()))
            },
        );
        told.map_err(|err| self.fail(err, "tell"))
    }
}

/// The rows of the batch `ask()` returned, as `f64`s: `None` for `None`, else
/// the rows of the iterable, at most `limit` of them: a batch with more does
/// not fit the budget whatever its length, so an endless iterator is not read
/// to its end. Each row is read up to one value past `dimensions`, so an
/// endless row (an `itertools.repeat`) is not read to its end either: one
/// value more is a `ValueError` that says the row is too long. A pending
/// Ctrl+C is checked before each row. A result or a row that is not iterable,
/// or a value that is not a real number (see [`Reals::value`]), is a
/// `ValueError` naming the batch and the row; a row too short and the values'
/// finiteness are the bridge's checks, with the same message. An exception
/// other than the ones that say "not iterable" or "not a number" (one raised
/// inside a generator, a `KeyboardInterrupt`) propagates unchanged.
fn rows_of(
    proposed: &Bound<'_, PyAny>,
    batch: usize,
    limit: usize,
    dimensions: usize,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    if proposed.is_none() {
        return Ok(None);
    }
    let py = proposed.py();
    let reals = Reals::new(py)?;
    let mut rows = match proposed.try_iter() {
        Ok(rows) => rows,
        Err(err) if err.is_instance_of::<PyTypeError>(py) => {
            return Err(PyValueError::new_err(format!(
                "the external optimizer's ask() must return None or an iterable of rows, \
                 got {} for batch {batch} (0-based)",
                shown(proposed)
            )));
        }
        Err(err) => return Err(err),
    };
    let invalid = |row: usize, reason: String| {
        PyValueError::new_err(format!(
            "invalid candidate in row {row} of batch {batch} from the external optimizer's \
             ask() (both 0-based): {reason}"
        ))
    };
    let mut parsed = Vec::new();
    for r in 0..limit {
        py.check_signals()?;
        let Some(row) = rows.next() else {
            break;
        };
        let row = row?;
        let values = match row.try_iter() {
            Ok(values) => values,
            Err(err) if err.is_instance_of::<PyTypeError>(py) => {
                return Err(invalid(
                    r,
                    format!("it is {}, not a sequence of numbers", shown(&row)),
                ));
            }
            Err(err) => return Err(err),
        };
        let mut point = Vec::new();
        for (i, value) in values.enumerate() {
            if i == dimensions {
                return Err(invalid(
                    r,
                    format!(
                        "it has more than {dimensions} values, expected one per dimension \
                         ({dimensions})"
                    ),
                ));
            }
            match reals.value(&value?)? {
                Ok(x) => point.push(x),
                Err(reason) => {
                    return Err(invalid(r, format!("its value at index {i} is {reason}")));
                }
            }
        }
        parsed.push(point);
    }
    Ok(Some(parsed))
}

/// What a value of a row may be: a real number. `float()` would also take a
/// `bool` (Python's or NumPy's) and a NumPy complex number, dropping its
/// imaginary part with a warning; neither is a candidate's value.
struct Reals<'py> {
    real: Bound<'py, PyAny>,
    complex: Bound<'py, PyAny>,
    /// `numpy.bool_`, when NumPy is imported (a NumPy value cannot exist
    /// otherwise); NumPy is never imported here.
    numpy_bool: Option<Bound<'py, PyAny>>,
}

impl<'py> Reals<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let numbers = py.import("numbers")?;
        let modules = py.import("sys")?.getattr("modules")?;
        let numpy_bool = match modules.cast::<PyDict>() {
            Ok(modules) => match modules.get_item("numpy")? {
                Some(numpy) => numpy.getattr("bool_").ok(),
                None => None,
            },
            Err(_) => None,
        };
        Ok(Self {
            real: numbers.getattr("Real")?,
            complex: numbers.getattr("Complex")?,
            numpy_bool,
        })
    }

    /// `value` as an `f64`, or (`Ok(Err(reason))`) why it is not a real
    /// number: a `bool`, a complex number (one of `numbers.Complex` that is not
    /// `numbers.Real`), an `int` too large for a float, or anything `float()`
    /// does not take. Any other exception from the conversion propagates.
    fn value(&self, value: &Bound<'py, PyAny>) -> PyResult<Result<f64, String>> {
        let py = value.py();
        // A float, or a subclass such as `numpy.float64`: the common case.
        if value.is_instance_of::<PyFloat>() {
            return value.extract::<f64>().map(Ok);
        }
        let is_bool = value.is_instance_of::<PyBool>()
            || match &self.numpy_bool {
                Some(numpy_bool) => value.is_instance(numpy_bool)?,
                None => false,
            };
        if is_bool {
            return Ok(Err(format!("{}, a bool, not a number", shown(value))));
        }
        if value.is_instance(&self.complex)? && !value.is_instance(&self.real)? {
            return Ok(Err(format!(
                "{}, a complex number; every value must be real",
                shown(value)
            )));
        }
        match value.extract::<f64>() {
            Ok(x) => Ok(Ok(x)),
            Err(err) if err.is_instance_of::<PyTypeError>(py) => {
                Ok(Err(format!("{}, not a number", shown(value))))
            }
            Err(err) if err.is_instance_of::<PyOverflowError>(py) => {
                Ok(Err(format!("{}, too large for a float", shown(value))))
            }
            Err(err) => Err(err),
        }
    }
}

/// `value`'s `repr`, cut to 80 characters, for an error message.
fn shown(value: &Bound<'_, PyAny>) -> String {
    const LIMIT: usize = 80;
    let repr = value
        .repr()
        .map_or_else(|_| "an unprintable value".to_string(), |r| r.to_string());
    if repr.chars().count() <= LIMIT {
        return repr;
    }
    let cut: String = repr.chars().take(LIMIT).collect();
    format!("{cut}...")
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

    /// A scripted optimizer: `ask()` returns the next of `BATCHES` (then
    /// `None`), and `tell` keeps what it was given.
    const SCRIPTED: &str = "\
class Scripted:
    def __init__(self, batches):
        self.batches = list(batches)
        self.told = []
    def ask(self):
        return self.batches.pop(0) if self.batches else None
    def tell(self, candidates, values):
        self.told.append((candidates, values))
";

    fn py_ask_tell(optimizer: &Bound<'_, PyAny>, errors: &OracleErrorSlot) -> PyAskTell {
        let external = External::new(
            optimizer.clone(),
            10_i64.into_pyobject(optimizer.py()).unwrap().into_any(),
        )
        .expect("a valid External");
        external
            .for_run(optimizer.py(), 2, errors, "external-test")
            .1
    }

    /// `ask()`'s rows reach the bridge as `f64`s, whatever iterable holds them,
    /// and `tell` receives new lists of floats.
    #[test]
    fn rows_and_values_cross_as_lists_of_floats() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, SCRIPTED, "external_scripted");
            let optimizer = module
                .getattr("Scripted")
                .unwrap()
                .call1((vec![
                    py.eval(c"[(1, 2.5), [3.0, 4]]", None, None).unwrap(),
                    py.eval(c"iter([range(2)])", None, None).unwrap(),
                ],))
                .unwrap();
            let errors = OracleErrorSlot::new();
            let mut bridge = py_ask_tell(&optimizer, &errors);
            let first = py.detach(|| bridge.ask()).expect("ask succeeds");
            assert_eq!(first, Some(vec![vec![1.0, 2.5], vec![3.0, 4.0]]));
            py.detach(|| bridge.tell(&[vec![1.0, 2.5]], &[0.5]))
                .expect("tell succeeds");
            let second = py.detach(|| bridge.ask()).expect("ask succeeds");
            assert_eq!(second, Some(vec![vec![0.0, 1.0]]));
            assert_eq!(py.detach(|| bridge.ask()).expect("ask succeeds"), None);
            let told = optimizer.getattr("told").unwrap();
            assert_eq!(
                told.repr().unwrap().to_string(),
                "[([[1.0, 2.5]], [0.5])]",
                "tell gets a list of float lists and a list of floats"
            );
            assert!(!errors.failed());
        });
    }

    /// The failure `ask()` or `tell()` raised is recorded in the slot with its
    /// class, and the bridge only gets an error message.
    #[test]
    fn an_exception_is_recorded_with_its_class() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(
                py,
                "\
class Failing:
    def __init__(self, where):
        self.where = where
    def ask(self):
        if self.where == 'ask':
            raise LookupError('no candidates')
        return [[0.0]]
    def tell(self, candidates, values):
        raise KeyboardInterrupt
",
                "external_failing",
            );
            for (where_, class) in [("ask", "LookupError"), ("tell", "KeyboardInterrupt")] {
                let optimizer = module.getattr("Failing").unwrap().call1((where_,)).unwrap();
                let errors = OracleErrorSlot::new();
                let mut bridge = py_ask_tell(&optimizer, &errors);
                let result = py.detach(|| match bridge.ask() {
                    Err(err) => Err(err.to_string()),
                    Ok(_) => bridge.tell(&[vec![0.0]], &[1.0]).map_err(|e| e.to_string()),
                });
                let message = result.expect_err("the call raised");
                assert!(message.starts_with(&format!("{where_}()")), "{message}");
                let recorded = errors.take().expect("the exception is recorded");
                let recorded = recorded
                    .downcast::<EvaluationError>()
                    .expect("an EvaluationError");
                let EvaluationError::Python(err) = *recorded else {
                    panic!("a Python exception");
                };
                let err = err.into_inner();
                assert_eq!(
                    err.get_type(py).name().unwrap().to_string(),
                    class,
                    "{where_}"
                );
            }
        });
    }

    /// Once the slot holds a failure, neither method calls Python.
    #[test]
    fn a_failed_run_calls_neither_method() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, SCRIPTED, "external_failed_run");
            let optimizer = module
                .getattr("Scripted")
                .unwrap()
                .call1((vec![vec![vec![0.0]]],))
                .unwrap();
            let errors = OracleErrorSlot::new();
            errors.record(
                Box::new(EvaluationError::Runtime("backend failed".to_string())),
                "external-test",
            );
            let mut bridge = py_ask_tell(&optimizer, &errors);
            assert!(py.detach(|| bridge.ask()).is_err());
            assert!(py.detach(|| bridge.tell(&[vec![0.0]], &[1.0])).is_err());
            let batches = optimizer.getattr("batches").unwrap();
            assert_eq!(batches.len().unwrap(), 1, "ask() was not called");
            assert_eq!(optimizer.getattr("told").unwrap().len().unwrap(), 0);
        });
    }

    /// A result or a row that is not iterable, and a value that is not a
    /// real number (a `str`, a `bool`, a complex number, an `int` too large
    /// for a float), are `ValueError`s naming the batch and the row.
    #[test]
    fn rows_that_are_not_numbers_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let row_error = |reason: &str| {
                format!(
                    "invalid candidate in row 0 of batch 0 from the external optimizer's ask() \
                     (both 0-based): its value at index 1 is {reason}"
                )
            };
            let cases: [(&std::ffi::CStr, String); 7] = [
                (
                    c"3.5",
                    "the external optimizer's ask() must return None or an iterable of rows, \
                     got 3.5 for batch 0 (0-based)"
                        .to_string(),
                ),
                (
                    c"[[0.0, 1.0], 7]",
                    "invalid candidate in row 1 of batch 0 from the external optimizer's ask() \
                     (both 0-based): it is 7, not a sequence of numbers"
                        .to_string(),
                ),
                (c"[[0.0, 'x']]", row_error("'x', not a number")),
                (
                    c"[[0.0, 1j]]",
                    row_error("1j, a complex number; every value must be real"),
                ),
                (c"[[0.0, True]]", row_error("True, a bool, not a number")),
                (c"[[0.0, False]]", row_error("False, a bool, not a number")),
                (
                    c"[[0.0, 10**400]]",
                    row_error(&format!(
                        "{}..., too large for a float",
                        "1".to_string() + &"0".repeat(79)
                    )),
                ),
            ];
            for (source, message) in cases {
                let proposed = py.eval(source, None, None).unwrap();
                let err = rows_of(&proposed, 0, 10, 2).expect_err("not rows of numbers");
                assert!(err.is_instance_of::<PyValueError>(py), "{err}");
                assert_eq!(err.value(py).to_string(), message);
            }
            // Integers and fractions are real numbers.
            let proposed = py
                .eval(c"[[3, __import__('fractions').Fraction(1, 4)]]", None, None)
                .unwrap();
            assert_eq!(
                rows_of(&proposed, 0, 10, 2).unwrap(),
                Some(vec![vec![3.0, 0.25]])
            );
            // The batch number is the one given.
            let proposed = py.eval(c"[['a']]", None, None).unwrap();
            let err = rows_of(&proposed, 4, 10, 2).expect_err("not a number");
            assert!(err.value(py).to_string().contains("row 0 of batch 4"));
            // An exception raised while iterating keeps its class.
            let module = module(
                py,
                "def rows():\n    yield [0.0]\n    raise OSError('disk')\n",
                "external_generator",
            );
            let proposed = module.getattr("rows").unwrap().call0().unwrap();
            let err = rows_of(&proposed, 0, 10, 2).expect_err("the generator raised");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyOSError>(py),
                "{err}"
            );
        });
    }

    /// A batch is read up to `limit` rows, so an endless iterator stops there;
    /// `PyAskTell` sets it one past the budget left, which `tell` lowers.
    #[test]
    fn a_batch_is_read_up_to_one_row_past_the_budget() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(
                py,
                "\
import itertools

class Endless:
    def __init__(self):
        self.read = 0
    def rows(self):
        for _ in itertools.count():
            self.read += 1
            yield [0.5]
    def ask(self):
        return self.rows()
    def tell(self, candidates, values):
        pass
",
                "external_endless",
            );
            let optimizer = module.getattr("Endless").unwrap().call0().unwrap();
            let proposed = optimizer.call_method0("rows").unwrap();
            let rows = rows_of(&proposed, 0, 3, 1).unwrap().expect("rows");
            assert_eq!(rows.len(), 3);
            let read: usize = optimizer.getattr("read").unwrap().extract().unwrap();
            assert_eq!(read, 3, "no row past the limit is read");

            // A budget of 10: 11 rows, then 7 once 4 are told.
            let errors = OracleErrorSlot::new();
            let mut bridge = py_ask_tell(&optimizer, &errors);
            let first = py.detach(|| bridge.ask()).unwrap().expect("rows");
            assert_eq!(first.len(), 11);
            py.detach(|| bridge.tell(&first[..4], &[0.0; 4])).unwrap();
            let second = py.detach(|| bridge.ask()).unwrap().expect("rows");
            assert_eq!(second.len(), 7);
            assert!(!errors.failed());
        });
    }

    /// A row is read up to one value past `dimensions`, so an endless row
    /// (`itertools.repeat`) stops there as a row too long, before the bridge
    /// sees it; a row of `dimensions` values is read whole.
    #[test]
    fn a_row_is_read_up_to_one_value_past_the_dimensions() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(
                py,
                "\
import itertools

class Counted:
    def __init__(self):
        self.read = 0
    def __iter__(self):
        for _ in itertools.count():
            self.read += 1
            yield 0.5
",
                "external_endless_row",
            );
            let row = module.getattr("Counted").unwrap().call0().unwrap();
            let proposed = PyList::new(py, [&row]).unwrap();
            let err = rows_of(proposed.as_any(), 0, 10, 3).expect_err("the row is too long");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert_eq!(
                err.value(py).to_string(),
                "invalid candidate in row 0 of batch 0 from the external optimizer's ask() \
                 (both 0-based): it has more than 3 values, expected one per dimension (3)"
            );
            let read: usize = row.getattr("read").unwrap().extract().unwrap();
            assert_eq!(read, 4, "no value past dimensions + 1 is read");
            let repeat = py
                .eval(c"[__import__('itertools').repeat(0.5)]", None, None)
                .unwrap();
            assert!(rows_of(&repeat, 0, 10, 3).is_err());
            let exact = py.eval(c"[(0.5, 1.5, 2.5)]", None, None).unwrap();
            assert_eq!(
                rows_of(&exact, 0, 10, 3).unwrap(),
                Some(vec![vec![0.5, 1.5, 2.5]])
            );
        });
    }

    /// `max_evaluations` is an integer `>= 1` up to `u64::MAX` (`TypeError`
    /// when it is not an integer, `ValueError` when it is out of range); the
    /// optimizer needs callable `ask` and `tell`.
    #[test]
    fn the_configuration_is_checked_when_built_and_when_set() {
        Python::initialize();
        Python::attach(|py| {
            let module = module(py, SCRIPTED, "external_config");
            let optimizer = module.getattr("Scripted").unwrap().call1(((),)).unwrap();
            // An integer out of range is a `ValueError`, anything else a
            // `TypeError`, as an integer argument's.
            for (bad, is_an_integer) in [
                ("0", true),
                ("-3", true),
                ("2**64", true),
                ("True", false),
                ("4.0", false),
                ("'7'", false),
                ("None", false),
            ] {
                let code = CString::new(bad).unwrap();
                let value = py.eval(&code, None, None).unwrap();
                let err = External::new(optimizer.clone(), value)
                    .err()
                    .unwrap_or_else(|| panic!("{bad} must be rejected"));
                if is_an_integer {
                    assert!(err.is_instance_of::<PyValueError>(py), "{bad}: {err}");
                } else {
                    assert!(err.is_instance_of::<PyTypeError>(py), "{bad}: {err}");
                }
                assert!(
                    err.value(py)
                        .to_string()
                        .starts_with("max_evaluations must be"),
                    "{bad}: {err}"
                );
            }
            let largest = py.eval(c"2**64 - 1", None, None).unwrap();
            let external = External::new(optimizer.clone(), largest).expect("the largest u64");
            assert_eq!(external.max_evaluations, u64::MAX);

            let class = module.getattr("Scripted").unwrap();
            let builtins = PyModule::import(py, "builtins").unwrap();
            let object = builtins.getattr("object").unwrap().call0().unwrap();
            for not_an_optimizer in [object, 3_i32.into_pyobject(py).unwrap().into_any()] {
                let err = External::new(
                    not_an_optimizer,
                    5_i32.into_pyobject(py).unwrap().into_any(),
                )
                .err()
                .expect("no ask/tell");
                assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
                assert!(err.to_string().contains("callable ask()"), "{err}");
            }
            // The class itself has the methods, as functions: accepted here, it
            // fails when called, as any wrong `ask()` does.
            assert!(External::new(class, 5_i32.into_pyobject(py).unwrap().into_any()).is_ok());

            // The setters check the same, and leave the value unchanged.
            let external = Bound::new(
                py,
                External::new(
                    optimizer.clone(),
                    5_i32.into_pyobject(py).unwrap().into_any(),
                )
                .unwrap(),
            )
            .unwrap();
            let err = external.setattr("max_evaluations", 0).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = external.setattr("optimizer", 3).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            let read: u64 = external
                .getattr("max_evaluations")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(read, 5);
            external.setattr("max_evaluations", 9).unwrap();
            let read: u64 = external
                .getattr("max_evaluations")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(read, 9);
            assert!(external.getattr("optimizer").unwrap().is(&optimizer));
        });
    }

    type Probe = Box<dyn FnOnce() + Send>;

    /// A Python object whose `__traverse__` runs a probe once.
    #[pyclass]
    struct TraverseProbe {
        probe: std::sync::Mutex<Option<Probe>>,
    }

    #[pymethods]
    impl TraverseProbe {
        fn __traverse__(&self, _visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
            let probe = self.probe.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(probe) = probe {
                probe();
            }
            Ok(())
        }
    }

    /// Run `f` inside a GC traversal, where PyO3 forbids attaching to the
    /// interpreter, and return what it returned: how a test reaches the
    /// "interpreter unavailable" branch of code that needs a live Python
    /// object. Mirrors the test helper of `polypus-evaluation`, which is not
    /// visible from this crate. A panic in `f` is caught there and re-raised
    /// here, since one escaping `__traverse__` would abort the process.
    fn run_while_attaching_is_forbidden<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
    ) -> R {
        use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
        use std::sync::{Arc, Mutex};
        let outcome = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&outcome);
        let probe: Probe = Box::new(move || {
            let forbidden = Python::try_attach(|_py| ()).is_none();
            let result = catch_unwind(AssertUnwindSafe(f));
            *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some((forbidden, result));
        });
        Python::initialize();
        Python::attach(|py| {
            let object = Bound::new(
                py,
                TraverseProbe {
                    probe: Mutex::new(Some(probe)),
                },
            )
            .expect("the probe object is created");
            py.import("gc")
                .and_then(|gc| gc.call_method0("collect"))
                .expect("gc.collect() runs");
            drop(object);
        });
        let (forbidden, result) = outcome
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .expect("gc.collect() traversed the probe object");
        assert!(forbidden, "the probe must run where attaching is forbidden");
        result.unwrap_or_else(|panic| resume_unwind(panic))
    }

    /// When the interpreter cannot be attached to, `ask` and `tell` record
    /// `InterpreterUnavailable` (raised as `KeyboardInterrupt` at the edge)
    /// and fail, without calling the optimizer.
    #[test]
    fn an_unavailable_interpreter_is_recorded_without_calling_python() {
        use crate::infrastructure::BackendError;
        Python::initialize();
        let (module, optimizer) = Python::attach(|py| {
            let module = module(py, SCRIPTED, "external_unavailable");
            let optimizer = module
                .getattr("Scripted")
                .unwrap()
                .call1((vec![vec![vec![0.0]]],))
                .unwrap();
            (module.unbind(), optimizer.unbind())
        });
        for call in ["ask", "tell"] {
            let errors = OracleErrorSlot::new();
            let mut bridge = Python::attach(|py| py_ask_tell(optimizer.bind(py), &errors));
            let failed = run_while_attaching_is_forbidden(move || match call {
                "ask" => bridge.ask().is_err(),
                _ => bridge.tell(&[vec![0.0]], &[1.0]).is_err(),
            });
            assert!(failed, "{call} must fail");
            let recorded = errors
                .take()
                .expect("the failure is recorded")
                .downcast::<EvaluationError>()
                .expect("an EvaluationError is recorded");
            match *recorded {
                EvaluationError::Backend(BackendError::External(boxed)) => {
                    assert!(boxed.is::<InterpreterUnavailable>(), "{call}: {boxed}")
                }
                other => panic!("{call}: expected InterpreterUnavailable, got {other:?}"),
            }
        }
        Python::attach(|py| {
            let optimizer = optimizer.bind(py);
            assert_eq!(optimizer.getattr("batches").unwrap().len().unwrap(), 1);
            assert_eq!(optimizer.getattr("told").unwrap().len().unwrap(), 0);
            drop(module);
        });
    }
}
