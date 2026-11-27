//! Fixtures shared by the entry points' tests.

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};

use super::de::DE;
use super::qml_entry::qml_train;
use super::training::train;

/// 3-qubit uniform superposition as OpenQASM 2.0. Its eight equally likely
/// outcomes make "counts differ across unseeded runs" non-flaky (a 2-outcome
/// Bell state collides far too often to assert inequality reliably).
pub(super) fn uniform3_qasm() -> String {
    polypus_circuit::ParameterizedCircuit::new(3)
        .h(0)
        .h(1)
        .h(2)
        .measure_all()
        .assign_parameters(&[])
        .expect("no free parameters")
        .try_to_qasm2()
        .unwrap()
}

/// Register `name` as a backend whose factory counts its calls in `builds`
/// and fails, so a test can tell whether an entry point created it.
pub(super) fn register_sentinel_backend(
    name: &str,
    builds: &'static std::sync::atomic::AtomicUsize,
) {
    crate::infrastructure::register_backend(
        name,
        move |_ctx: &crate::infrastructure::BackendBuildContext| {
            builds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(crate::infrastructure::BackendError::Conversion(
                "the sentinel backend was created".to_string(),
            ))
        },
    );
}

/// A `polypus.DE` built through its Python signature, so these tests do not
/// depend on the order of `DE::new`'s arguments.
pub(super) fn de<'py>(py: Python<'py>, population_size: u32) -> Bound<'py, PyAny> {
    let kwargs = [("generations", 1), ("population_size", population_size)]
        .into_py_dict(py)
        .expect("the kwargs build");
    py.get_type::<DE>()
        .call((), Some(&kwargs))
        .expect("the pyclass instantiates")
}

/// `train` with every argument but these fixed: one dimension, a Qiskit-like
/// placeholder circuit (nothing reads it before the backend exists).
pub(super) fn train_on<'py>(
    py: Python<'py>,
    infrastructure: &str,
    method: Bound<'py, PyAny>,
    expectation_function: Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    train_with(py, infrastructure, method, expectation_function, None, None)
}

/// [`train_on`] with `final_shots` and `callback`, and no `trace`.
pub(super) fn train_with<'py>(
    py: Python<'py>,
    infrastructure: &str,
    method: Bound<'py, PyAny>,
    expectation_function: Bound<'py, PyAny>,
    final_shots: Option<Bound<'py, PyAny>>,
    callback: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    train(
        py.None().into_bound(py),
        method,
        16,
        1,
        1,
        expectation_function,
        Some(infrastructure.to_string()),
        Some(1),
        Some(1),
        Some("edge".to_string()),
        "automatic",
        None,
        "aer",
        Some(3),
        None,
        None,
        None,
        None,
        final_shots,
        callback,
        None,
        None,
    )
}

/// Stand-ins for a Qiskit feature map and ansatz: what `qml.train` reads of
/// them before the backend exists (`parameters`, `compose`, `num_clbits`,
/// `assign_parameters`).
pub(super) fn fake_qml_circuits(py: Python<'_>) -> (Bound<'_, PyAny>, Bound<'_, PyAny>) {
    let module = PyModule::from_code(
        py,
        c"class Circuit:
    num_clbits = 1
    def __init__(self, *parameters):
        self.parameters = list(parameters)
    def compose(self, other):
        return Circuit(*self.parameters, *other.parameters)
    def assign_parameters(self, values, inplace=False):
        return self
",
        c"fake_qml_circuits.py",
        c"fake_qml_circuits",
    )
    .expect("the module compiles");
    let circuit = module.getattr("Circuit").expect("the class exists");
    (
        circuit.call1(("x",)).expect("a feature map"),
        circuit.call1(("w",)).expect("an ansatz"),
    )
}

/// `qml.train` with every argument but these fixed: one feature, one weight,
/// two training rows.
pub(super) fn qml_train_on<'py>(
    py: Python<'py>,
    infrastructure: &str,
    method: Bound<'py, PyAny>,
    expectation_function: Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    qml_train_with(py, infrastructure, method, expectation_function, None, None)
}

/// [`qml_train_on`] with `final_shots` and `callback`.
pub(super) fn qml_train_with<'py>(
    py: Python<'py>,
    infrastructure: &str,
    method: Bound<'py, PyAny>,
    expectation_function: Bound<'py, PyAny>,
    final_shots: Option<Bound<'py, PyAny>>,
    callback: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (feature_map, ansatz) = fake_qml_circuits(py);
    let rows = py.eval(c"[[0.1], [0.2]]", None, None).expect("the rows");
    qml_train(
        feature_map,
        ansatz,
        rows,
        method,
        16,
        1,
        1,
        expectation_function,
        Some(infrastructure.to_string()),
        Some(1),
        Some(1),
        Some("edge".to_string()),
        "automatic",
        None,
        "aer",
        Some(3),
        None,
        None,
        None,
        None,
        final_shots,
        callback,
        None,
    )
}
