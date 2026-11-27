//! The circuit, cost and label arguments of the entry points, read into
//! their pyo3-free forms.

use std::sync::Arc;

use pyo3::prelude::*;

use super::circuit::Circuit;
use super::observable::{CachedCost, Ising, Qubo, SampleCost};
use crate::evaluation::{
    CircuitSource, CostObservable, Label, PyCallbackObservable, PyLabelledCost, PySampleCost,
    SupervisedObjective,
};
use crate::infrastructure::BoundCircuit;

/// Interpret the `qc` argument of an entry point as a parameterised circuit
/// template. A `polypus.Circuit` becomes [`CircuitSource::Native`] (binding
/// will run GIL-free); any other object is assumed to be a Qiskit
/// `QuantumCircuit`, preserving the original behaviour.
pub(super) fn extract_circuit_source(qc: &Bound<'_, PyAny>) -> CircuitSource {
    if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
        CircuitSource::Native(native.native().clone())
    } else {
        CircuitSource::Qiskit(qc.clone().unbind())
    }
}

/// Start of the `ValueError` message for a circuit with free parameters, the
/// same for a `polypus.Circuit` and a Qiskit `QuantumCircuit`.
const UNBOUND_PARAMETERS: &str = "circuit has unbound parameters and cannot be executed directly";

/// Interpret the `qc` argument of `run_quantum_circuit` as a fully bound,
/// executable circuit:
/// - `polypus.Circuit` → native circuit (must have no free parameters),
/// - `str` → raw OpenQASM 2.0 program,
/// - anything else → Qiskit `QuantumCircuit` (must have no free parameters:
///   no backend binds them, and Aer would fail with its own error, issue #308).
pub(super) fn extract_bound_circuit(qc: &Bound<'_, PyAny>) -> PyResult<BoundCircuit> {
    if let Ok(native) = qc.extract::<PyRef<'_, Circuit>>() {
        let concrete = native.native().assign_parameters(&[]).map_err(|e| {
            crate::exceptions::configuration_error(format!(
                "{UNBOUND_PARAMETERS}: {e}. \
                 Bind values first via to_qasm2(params) or use polypus.train"
            ))
        })?;
        return Ok(BoundCircuit::Native(concrete));
    }
    if let Ok(qasm) = qc.extract::<String>() {
        return Ok(BoundCircuit::Qasm2(qasm));
    }
    let free = free_parameter_names(qc)?;
    if !free.is_empty() {
        return Err(crate::exceptions::configuration_error(format!(
            "{UNBOUND_PARAMETERS}: {}. \
             Bind values first via QuantumCircuit.assign_parameters or use polypus.train",
            free.join(", ")
        )));
    }
    // Anything else is a Qiskit `QuantumCircuit`: carry it opaquely through the
    // pyo3-free `BoundCircuit` enum via its `Foreign` escape hatch, so the Aer/CUNQA
    // backends still receive the native Qiskit object unchanged.
    Ok(crate::infrastructure::QiskitCircuit::into_bound(
        qc.clone().unbind(),
    ))
}

/// The names of the free parameters of a Qiskit `QuantumCircuit` (its
/// `parameters`), in Qiskit's order. An object without that attribute has none.
fn free_parameter_names(qc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let Ok(parameters) = qc.getattr(pyo3::intern!(qc.py(), "parameters")) else {
        return Ok(Vec::new());
    };
    parameters
        .try_iter()?
        .map(|parameter| Ok(parameter?.str()?.to_string()))
        .collect()
}

/// Interpret the `expectation_function` argument as a cost observable.
///
/// A `polypus.Qubo` / `polypus.Ising` becomes the corresponding native,
/// GIL-free evaluator; any other **callable** becomes a [`PyCallbackObservable`]
/// (the cost function is invoked once per unique bitstring per batch, then
/// aggregation runs in Rust). This is the dispatch mirror of
/// [`extract_circuit_source`], so the existing `expectation_function=<callable>`
/// API keeps working unchanged while native observables opt into the fast path.
pub(super) fn extract_cost_observable(obj: &Bound<'_, PyAny>) -> PyResult<Arc<dyn CostObservable>> {
    if let Ok(q) = obj.extract::<PyRef<'_, Qubo>>() {
        // Clone into the concrete Arc first, then coerce to the trait object
        // (`Arc::clone`'s generic is pinned by its `&Arc<T>` argument).
        let concrete = Arc::clone(&q.inner);
        let obs: Arc<dyn CostObservable> = concrete;
        return Ok(obs);
    }
    if let Ok(i) = obj.extract::<PyRef<'_, Ising>>() {
        let concrete = Arc::clone(&i.inner);
        let obs: Arc<dyn CostObservable> = concrete;
        return Ok(obs);
    }
    // A callable wrapped in polypus.CachedCost keeps a cross-generation memo;
    // a bare callable deduplicates only within each batch.
    if let Ok(cached) = obj.extract::<PyRef<'_, CachedCost>>() {
        let obs: Arc<dyn CostObservable> = Arc::new(PyCallbackObservable::new(
            cached.cost_fn.clone_ref(obj.py()),
            true,
        ));
        return Ok(obs);
    }
    if obj.is_callable() {
        let obs: Arc<dyn CostObservable> =
            Arc::new(PyCallbackObservable::new(obj.clone().unbind(), false));
        return Ok(obs);
    }
    // Not callable either; this only gives a better message than the one below.
    if obj.extract::<PyRef<'_, SampleCost>>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "polypus.SampleCost scores each training sample against its label, so it needs \
             labels: use it with polypus.qml.train(..., y_train=...). Without labels, pass a \
             callable (bitstring -> float)",
        ));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expectation_function must be a callable (bitstring -> float), a \
         polypus.CachedCost(callable), or a polypus.Qubo / polypus.Ising observable",
    ))
}

/// Interpret `qml.train`'s `expectation_function` when `y_train` is given: a
/// callable is a per-shot `f(bitstring, label)` ([`PyLabelledCost`]),
/// `polypus.CachedCost(f)` the same with a memo, and `polypus.SampleCost(g)` a
/// per-sample `g(counts, label)` ([`PySampleCost`]). A `Qubo`/`Ising` cannot read
/// labels, so it is a `TypeError` rather than silently training without them.
pub(super) fn extract_supervised_objective(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn SupervisedObjective>> {
    if let Ok(sample) = obj.extract::<PyRef<'_, SampleCost>>() {
        let objective: Arc<dyn SupervisedObjective> =
            Arc::new(PySampleCost::new(sample.cost_fn.clone_ref(obj.py())));
        return Ok(objective);
    }
    if let Ok(cached) = obj.extract::<PyRef<'_, CachedCost>>() {
        let objective: Arc<dyn SupervisedObjective> = Arc::new(PyLabelledCost::new(
            cached.cost_fn.clone_ref(obj.py()),
            true,
        ));
        return Ok(objective);
    }
    if obj.extract::<PyRef<'_, Qubo>>().is_ok() || obj.extract::<PyRef<'_, Ising>>().is_ok() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "a polypus.Qubo / polypus.Ising observable cannot read labels; with y_train, \
             expectation_function must be a callable (bitstring, label) -> float, a \
             polypus.CachedCost(callable) or a polypus.SampleCost(callable)",
        ));
    }
    if obj.is_callable() {
        let objective: Arc<dyn SupervisedObjective> =
            Arc::new(PyLabelledCost::new(obj.clone().unbind(), false));
        return Ok(objective);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "with y_train, expectation_function must be a callable (bitstring, label) -> float, \
         a polypus.CachedCost(callable) or a polypus.SampleCost((counts, label) -> float)",
    ))
}

/// Reads real numbers from Python as PyO3's `f64` extraction does (through
/// `float()`), except that a complex number is refused first, by its type,
/// whatever its imaginary part: Python's `complex`, anything with a NumPy
/// `dtype` of kind `c` (the scalars `complex64`, `complex128` and
/// `clongdouble`, and a 0-d array), or else one of `numbers.Complex` that is
/// not `numbers.Real`. NumPy's complex scalars (and a `clongdouble` 0-d array)
/// answer `float()` by dropping their imaginary part with a `ComplexWarning`,
/// so without this a complex value in a NumPy array was taken cut while the
/// same value in a list was a `TypeError` (#602). The refusal is a `TypeError` naming the value's
/// position; a `bool`, Python's or NumPy's, is not complex and reads as 0 or 1.
/// Shared by every native QML reader of `x`, `y` and the cotangent, and by
/// [`extract_labels`].
pub(super) struct RealReader<'py> {
    complex: Bound<'py, PyAny>,
    real: Bound<'py, PyAny>,
}

impl<'py> RealReader<'py> {
    pub(super) fn new(py: Python<'py>) -> PyResult<Self> {
        let numbers = py.import("numbers")?;
        Ok(Self {
            complex: numbers.getattr("Complex")?,
            real: numbers.getattr("Real")?,
        })
    }

    /// The `TypeError` naming `at` (as `x[0][1]`) when `value` is a complex
    /// number, before anything calls its `float()`.
    fn refuse_complex(
        &self,
        value: &Bound<'py, PyAny>,
        at: impl FnOnce() -> String,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyTypeError;
        use pyo3::types::{PyComplex, PyFloat, PyInt};

        // The common cases skip the ABCs: a float (`numpy.float64` is one), an
        // int or a bool; Python's `complex` (`numpy.complex128` is one).
        if value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>() {
            return Ok(());
        }
        let complex = if value.is_instance_of::<PyComplex>() {
            true
        } else if let Some(kind) = numpy_dtype_kind(value) {
            // A NumPy scalar or array: its dtype decides. A 0-d array is not
            // registered with `numbers.Complex`, so the ABCs would miss it.
            kind == "c"
        } else {
            value.is_instance(&self.complex)? && !value.is_instance(&self.real)?
        };
        if !complex {
            return Ok(());
        }
        // For the message only, so an unprintable value is not itself an error.
        let shown = value
            .repr()
            .map_or_else(|_| "a value".to_string(), |repr| repr.to_string());
        Err(PyTypeError::new_err(format!(
            "{} is {shown}, a complex number; it must be a real number (the imaginary part \
             is never dropped, even when it is zero)",
            at()
        )))
    }

    /// The flat sequence `values`, its `i`-th value named `name[i]`: the
    /// sequence read as PyO3 reads a `Vec<f64>` (a `str` or a non-sequence is
    /// its `TypeError`), then each value refused if complex, else read as an
    /// `f64`. `pyo3_err` maps PyO3's own errors (the sequence's and
    /// `float()`'s), never the refusal of a complex value. One value object at
    /// a time is converted, the sequence's items held only for this sequence.
    /// A NumPy array of a real or boolean dtype (kind `f`, `i`, `u` or `b`)
    /// holds no complex value, so its values skip the check: a `float32` or
    /// `int64` value would otherwise pay two ABC `isinstance` calls.
    pub(super) fn values(
        &self,
        values: &Bound<'py, PyAny>,
        name: &str,
        pyo3_err: impl Fn(PyErr) -> PyErr,
    ) -> PyResult<Vec<f64>> {
        let items: Vec<Bound<'py, PyAny>> = values.extract().map_err(&pyo3_err)?;
        let all_real = numpy_dtype_kind(values)
            .is_some_and(|kind| matches!(kind.as_str(), "f" | "i" | "u" | "b"));
        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if !all_real {
                    self.refuse_complex(item, || format!("{name}[{i}]"))?;
                }
                item.extract::<f64>().map_err(&pyo3_err)
            })
            .collect()
    }

    /// The rows of `x`, the value in row `r`, column `c` named `name[r][c]`:
    /// `x` and each row read as PyO3 reads a `Vec<Vec<f64>>`, each row through
    /// [`values`](Self::values) in turn, so a row's value objects never outlive
    /// it. `pyo3_err` is as for `values`.
    pub(super) fn rows(
        &self,
        x: &Bound<'py, PyAny>,
        name: &str,
        pyo3_err: impl Fn(PyErr) -> PyErr,
    ) -> PyResult<Vec<Vec<f64>>> {
        let rows: Vec<Bound<'py, PyAny>> = x.extract().map_err(&pyo3_err)?;
        rows.iter()
            .enumerate()
            .map(|(r, row)| self.values(row, &format!("{name}[{r}]"), &pyo3_err))
            .collect()
    }
}

/// The `dtype.kind` of a NumPy scalar or array (`"f"`, `"c"`, `"b"`, ...), or
/// `None` for anything without one, read by attribute so NumPy is never
/// imported.
fn numpy_dtype_kind(value: &Bound<'_, PyAny>) -> Option<String> {
    value
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract::<String>())
        .ok()
}

/// Normalise the labels `labels`, the argument `name` (`y_train`, or `y` for
/// `Model.fit`), into one [`Label`] per sample (contract C-8). If every
/// element is an integer (Python or NumPy ints and bools) they are `Class` labels,
/// otherwise all are `Real`. A non-number, a complex number (Python's or NumPy's,
/// whatever its imaginary part, refused by [`RealReader`] before its `float()`
/// drops that part) or a nested row is a `TypeError`, and `NaN`/`inf` a
/// `ValueError`, naming `name` and the index. The caller checks the count.
pub(super) fn extract_labels(labels: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Label>> {
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyString;

    enum Raw {
        Int(i64),
        Float(f64),
    }
    // For error messages only, so an unreadable name is not itself an error.
    let type_name = |item: &Bound<'_, PyAny>| {
        item.get_type()
            .name()
            .map_or_else(|_| "an unknown type".to_string(), |found| found.to_string())
    };
    let reals = RealReader::new(labels.py())?;
    let mut raw = Vec::new();
    for (idx, item) in labels.try_iter()?.enumerate() {
        let item = item?;
        if item.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(format!(
                "{name}[{idx}] is a str; labels must be numbers (int class labels or float \
                 targets) — encode class names as integers first"
            )));
        }
        // Checked before the numeric reads: NumPy converts a length-1 array to a
        // float, which would silently accept a column vector.
        if let Ok(len) = item.len() {
            return Err(PyTypeError::new_err(format!(
                "{name}[{idx}] is a sequence ({} of length {len}), not a single label; pass \
                 one number per sample (class indices rather than one-hot rows; \
                 y.ravel() for a column vector)",
                type_name(&item)
            )));
        }
        // Before any numeric read: `float()` would drop the imaginary part of a
        // NumPy complex label.
        reals.refuse_complex(&item, || format!("{name}[{idx}]"))?;
        // Through `__index__`: accepts Python and NumPy integers, never truncates
        // a float.
        if let Ok(value) = item.extract::<i64>() {
            raw.push(Raw::Int(value));
            continue;
        }
        // A NumPy bool has no `__index__`, but a boolean mask is a natural binary
        // `y`. Detected through `dtype.kind`, so NumPy is never imported.
        let is_numpy_bool = numpy_dtype_kind(&item).is_some_and(|kind| kind == "b");
        if is_numpy_bool {
            raw.push(Raw::Int(i64::from(item.is_truthy()?)));
            continue;
        }
        match item.extract::<f64>() {
            Ok(value) if value.is_finite() => raw.push(Raw::Float(value)),
            Ok(value) => {
                return Err(crate::exceptions::configuration_error(format!(
                    "{name}[{idx}] is {value}; labels must be finite numbers"
                )));
            }
            Err(_) => {
                return Err(PyTypeError::new_err(format!(
                    "{name}[{idx}] has type {}; labels must be numbers (int class labels or \
                     float targets)",
                    type_name(&item)
                )));
            }
        }
    }
    let all_integers = raw.iter().all(|r| matches!(r, Raw::Int(_)));
    Ok(raw
        .into_iter()
        .map(|r| match r {
            Raw::Int(value) if all_integers => Label::Class(value),
            Raw::Int(value) => Label::Real(value as f64),
            Raw::Float(value) => Label::Real(value),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::test_support::uniform3_qasm;
    use pyo3::types::{PyDict, PyString};

    #[test]
    fn extract_bound_circuit_reads_a_qasm_string() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qasm = uniform3_qasm();
            let bound = extract_bound_circuit(&PyString::new(py, &qasm).into_any())
                .expect("a str is an OpenQASM 2.0 program");
            match bound {
                BoundCircuit::Qasm2(text) => assert_eq!(text, qasm),
                _ => panic!("a str must become BoundCircuit::Qasm2"),
            }
        });
    }

    #[test]
    fn extract_bound_circuit_treats_anything_else_as_a_qiskit_circuit() {
        // The classification is "not a polypus.Circuit and not a str", so any
        // other object lands on the Qiskit arm — which is exactly why the
        // native/qmio guards downstream can reject it without importing Qiskit.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let bound = extract_bound_circuit(py.None().bind(py))
                .expect("a non-Circuit, non-str object is assumed to be a Qiskit circuit");
            // A Qiskit circuit rides through the pyo3-free enum's `Foreign` hatch.
            assert!(bound.is_foreign());
        });
    }

    #[test]
    fn extract_bound_circuit_rejects_a_qiskit_circuit_with_free_parameters() {
        // Stands in for a `QuantumCircuit` (only `parameters` is read), so the
        // test needs no Qiskit.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qc = py
                .eval(
                    c"type('QC', (), {'parameters': ['t', 'theta[0]']})()",
                    None,
                    None,
                )
                .expect("the stand-in builds");
            let err = match extract_bound_circuit(&qc) {
                Err(err) => err,
                Ok(_) => panic!("a circuit with free parameters cannot be executed directly"),
            };
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let message = err.to_string();
            assert!(message.contains("unbound parameters"), "{message}");
            assert!(message.contains("directly: t, theta[0]."), "{message}");
            let unbound = py
                .eval(c"type('QC', (), {'parameters': []})()", None, None)
                .expect("the stand-in builds");
            assert!(extract_bound_circuit(&unbound).is_ok_and(|b| b.is_foreign()));
        });
    }

    #[test]
    fn extract_bound_circuit_reads_a_fully_bound_native_circuit() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let circuit = Py::new(
                py,
                Circuit {
                    inner: polypus_circuit::ParameterizedCircuit::new(1)
                        .x(0)
                        .measure_all(),
                },
            )
            .expect("the pyclass instantiates");
            let bound = extract_bound_circuit(circuit.bind(py).as_any())
                .expect("a parameter-free polypus.Circuit is executable as-is");
            assert!(matches!(bound, BoundCircuit::Native(_)));
        });
    }

    #[test]
    fn extract_bound_circuit_rejects_a_native_circuit_with_free_parameters() {
        // An unbound `polypus.Circuit` cannot be executed directly; it must be a
        // clear ValueError pointing at `train`, not a panic inside binding.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let circuit = Py::new(
                py,
                Circuit {
                    inner: polypus_circuit::ParameterizedCircuit::new(1)
                        .ry(0, polypus_circuit::GateParam::Param(0))
                        .measure_all(),
                },
            )
            .expect("the pyclass instantiates");
            // `BoundCircuit` is not `Debug`, so unwrap the `Result` by hand
            // rather than widening a production type just for a test.
            let err = match extract_bound_circuit(circuit.bind(py).as_any()) {
                Err(err) => err,
                Ok(_) => panic!("an unbound template cannot be executed directly"),
            };
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                err.to_string().contains("unbound parameters"),
                "the message must explain what to do instead: {err}"
            );
        });
    }

    /// `extract_labels` over a Python expression (builtins only: no NumPy here).
    fn labels_of(py: Python<'_>, expr: &std::ffi::CStr) -> PyResult<Vec<Label>> {
        let y_train = py
            .eval(expr, None, None)
            .expect("the test expression evaluates");
        extract_labels(&y_train, "y_train")
    }

    /// The `(class, message)` of a rejected `y_train`.
    fn label_error(py: Python<'_>, expr: &std::ffi::CStr) -> (String, String) {
        let err = labels_of(py, expr).expect_err("these labels must be rejected");
        let class = err
            .get_type(py)
            .name()
            .expect("an exception type has a name")
            .to_string();
        (class, err.to_string())
    }

    #[test]
    fn extract_labels_keeps_all_integer_labels_as_classes() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                labels_of(py, c"[0, 2, -1, True]").expect("integers are labels"),
                [0, 2, -1, 1].map(Label::Class),
                "ints and bools stay class labels"
            );
            // y_train is only iterated, so a generator works too.
            assert_eq!(
                labels_of(py, c"(k % 2 for k in range(3))").expect("a generator is iterable"),
                [0, 1, 0].map(Label::Class)
            );
        });
    }

    #[test]
    fn extract_labels_makes_every_label_a_float_when_any_is_not_an_integer() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                labels_of(py, c"[0, 1.5, 2]").expect("mixed numbers are labels"),
                [0.0, 1.5, 2.0].map(Label::Real),
                "one float makes every label a float"
            );
            // The type decides, not the value.
            assert_eq!(
                labels_of(py, c"[1.0, 0.0]").expect("floats are labels"),
                [1.0, 0.0].map(Label::Real)
            );
        });
    }

    #[test]
    fn extract_labels_rejects_non_numbers_and_nested_rows_naming_the_index() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let (class, msg) = label_error(py, c"[0, 'cat']");
            assert_eq!(class, "TypeError");
            assert!(msg.contains("y_train[1] is a str"), "{msg}");

            let (class, msg) = label_error(py, c"[0, [1, 0]]");
            assert_eq!(class, "TypeError");
            assert!(
                msg.contains("y_train[1] is a sequence (list of length 2)"),
                "a one-hot row is not a label: {msg}"
            );

            let (class, msg) = label_error(py, c"[None]");
            assert_eq!(class, "TypeError");
            assert!(msg.contains("y_train[0] has type NoneType"), "{msg}");
        });
    }

    #[test]
    fn extract_labels_rejects_complex_numbers_naming_the_index() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // Python's `complex` (`numpy.complex128` is a subclass), even with
            // no imaginary part: the type decides, not the value.
            let (class, msg) = label_error(py, c"[0, 1+0j]");
            assert_eq!(class, "TypeError");
            assert!(
                msg.contains("y_train[1] is (1+0j), a complex number"),
                "{msg}"
            );

            // NumPy's `complex64` is no subclass of `complex`, only registered
            // with `numbers.Complex`, and its `float()` drops the imaginary
            // part (#602). NumPy is not installed where the Rust tests run, so
            // a registered stand-in plays it; the NumPy types themselves are
            // tested from Python.
            let scope = PyDict::new(py);
            py.run(
                c"import numbers\n\
                  class Complex64:\n\
                  \x20   def __float__(self):\n\
                  \x20       return 1.0\n\
                  \x20   def __repr__(self):\n\
                  \x20       return 'Complex64(1+0j)'\n\
                  numbers.Complex.register(Complex64)\n\
                  y = [1.5, Complex64()]\n",
                Some(&scope),
                None,
            )
            .expect("the stand-in is defined");
            let y = scope
                .get_item("y")
                .expect("the scope is a dict")
                .expect("y is defined");
            let err = extract_labels(&y, "y_train").expect_err("a complex label is refused");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(
                err.to_string()
                    .contains("y_train[1] is Complex64(1+0j), a complex number"),
                "{err}"
            );

            // A NumPy 0-d array (`numpy.array(1j)`) is not registered with
            // `numbers.Complex`, and a `clongdouble` one's `float()` drops the
            // imaginary part: its dtype's kind decides. A stand-in with only a
            // `dtype` plays it; the message names the argument it is given
            // (`y` for `Model.fit`). The same stand-in with a real dtype is a
            // label.
            py.run(
                c"from types import SimpleNamespace\n\
                  class ZeroDim:\n\
                  \x20   def __init__(self, kind):\n\
                  \x20       self.dtype = SimpleNamespace(kind=kind)\n\
                  \x20   def __float__(self):\n\
                  \x20       return 1.0\n\
                  \x20   def __repr__(self):\n\
                  \x20       return f'ZeroDim({self.dtype.kind!r})'\n\
                  y = [1.5, ZeroDim('c')]\n\
                  y_real = [1.5, ZeroDim('f')]\n",
                Some(&scope),
                None,
            )
            .expect("the stand-in is defined");
            let item = |name: &str| {
                scope
                    .get_item(name)
                    .expect("the scope is a dict")
                    .expect("the labels are defined")
            };
            let err = extract_labels(&item("y"), "y").expect_err("a complex dtype is refused");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(
                err.to_string()
                    .contains("y[1] is ZeroDim('c'), a complex number"),
                "{err}"
            );
            assert_eq!(
                extract_labels(&item("y_real"), "y").expect("a real dtype is a label"),
                [1.5, 1.0].map(Label::Real)
            );

            // A bool stays a class label, and a float stays a float.
            assert_eq!(
                labels_of(py, c"[True, False, 2]").expect("bools are labels"),
                [1, 0, 2].map(Label::Class)
            );
            assert_eq!(
                labels_of(py, c"[0.5, 1]").expect("reals are labels"),
                [0.5, 1.0].map(Label::Real)
            );
        });
    }

    #[test]
    fn extract_labels_rejects_non_finite_values_naming_the_index() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let (class, msg) = label_error(py, c"[0.5, float('nan')]");
            assert_eq!(class, "ConfigurationError");
            assert!(msg.contains("y_train[1] is NaN"), "{msg}");

            let (class, msg) = label_error(py, c"[float('-inf')]");
            assert_eq!(class, "ConfigurationError");
            assert!(msg.contains("y_train[0] is -inf"), "{msg}");
        });
    }

    #[test]
    fn extract_supervised_objective_accepts_callables_and_the_two_wrappers() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let callable = py
                .eval(c"lambda bitstring, label: 0.0", None, None)
                .expect("a lambda evaluates");
            let cached = Py::new(
                py,
                CachedCost {
                    cost_fn: callable.clone().unbind(),
                },
            )
            .expect("the pyclass instantiates");
            let sample = Py::new(
                py,
                SampleCost {
                    cost_fn: callable.clone().unbind(),
                },
            )
            .expect("the pyclass instantiates");
            for accepted in [
                callable,
                cached.into_bound(py).into_any(),
                sample.into_bound(py).into_any(),
            ] {
                assert!(
                    extract_supervised_objective(&accepted).is_ok(),
                    "{accepted:?} is a valid supervised objective"
                );
            }
        });
    }

    #[test]
    fn extract_supervised_objective_rejects_observables_that_cannot_read_labels() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let qubo = Py::new(
                py,
                Qubo {
                    inner: Arc::new(
                        polypus_observable::QuboObservable::new(
                            1,
                            vec![(0, 1.0)],
                            vec![],
                            0.0,
                            1.0,
                        )
                        .expect("a one-variable QUBO is valid"),
                    ),
                },
            )
            .expect("the pyclass instantiates");
            let err = extract_supervised_objective(qubo.bind(py).as_any())
                .err()
                .expect("a Qubo cannot read labels");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("cannot read labels"), "{err}");

            let not_callable = 42i64.into_pyobject(py).expect("an int converts").into_any();
            let err = extract_supervised_objective(&not_callable)
                .err()
                .expect("a non-callable is not an objective");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("with y_train"), "{err}");
        });
    }

    #[test]
    fn extract_cost_observable_rejects_a_sample_cost_without_labels() {
        // Without labels (polypus.train, or qml.train without y_train) a SampleCost
        // gets a TypeError that says why, not the generic message.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let sample = Py::new(
                py,
                SampleCost {
                    cost_fn: py
                        .eval(c"lambda counts, label: 0.0", None, None)
                        .expect("a lambda evaluates")
                        .unbind(),
                },
            )
            .expect("the pyclass instantiates");
            let err = extract_cost_observable(sample.bind(py).as_any())
                .err()
                .expect("a SampleCost needs labels");
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("y_train"), "{err}");
        });
    }
}
