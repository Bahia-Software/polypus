//! What the four Python entry points (`run_quantum_circuit`, `train`,
//! `qml.train`, `qml.predict`) raise when a circuit cannot run where it is
//! sent, and in which order their checks fail.
//!
//! The messages are pinned in full and per entry point, so a shared helper
//! cannot change one without a test failing. A circuit that is accepted is
//! sent to a registered sentinel backend, whose creation fails with a known
//! message, or stopped by the next check, so nothing here needs Qiskit, Aer,
//! CUNQA or a QPU.

use super::*;
use pyo3::types::PyString;

/// The class name and the text (`str(exc)`) of `err`.
fn class_and_message(py: Python<'_>, err: &PyErr) -> (String, String) {
    let class = err
        .get_type(py)
        .name()
        .expect("an exception type has a name")
        .to_string();
    (class, err.value(py).to_string())
}

/// The `(class, message)` of a call that must fail.
fn failure(py: Python<'_>, result: PyResult<Py<PyAny>>) -> (String, String) {
    match result {
        Ok(_) => panic!("the call must fail"),
        Err(err) => class_and_message(py, &err),
    }
}

/// Assert that `result` failed with a `polypus.ConfigurationError` reading
/// exactly `message`.
fn assert_rejected(py: Python<'_>, result: PyResult<Py<PyAny>>, message: &str) {
    let (class, text) = failure(py, result);
    assert_eq!(class, "ConfigurationError", "{text}");
    assert_eq!(text, message);
}

/// Register `name` as a backend whose creation fails with
/// [`SENTINEL_MESSAGE`], so a call that reaches it passed every check.
fn register_failing_backend(name: &str) {
    crate::infrastructure::register_backend(
        name,
        |_ctx: &crate::infrastructure::BackendBuildContext| {
            Err(crate::infrastructure::BackendError::Conversion(
                SENTINEL_MESSAGE.to_string(),
            ))
        },
    );
}

/// What a backend registered with [`register_failing_backend`] fails with.
const SENTINEL_MESSAGE: &str = "the entry point reached the sentinel backend";

/// Assert that `result` failed creating the sentinel backend: the circuit
/// passed every check before it.
fn assert_reached_backend(py: Python<'_>, result: PyResult<Py<PyAny>>) {
    let (_, text) = failure(py, result);
    assert!(text.contains(SENTINEL_MESSAGE), "{text}");
}

/// A stand-in for a Qiskit `QuantumCircuit` with no free parameters: neither a
/// `polypus.Circuit` nor a `str`, which is all the entry points look at.
fn qiskit_circuit(py: Python<'_>) -> Bound<'_, PyAny> {
    py.eval(c"type('QC', (), {'parameters': []})()", None, None)
        .expect("the stand-in builds")
}

/// A `polypus.Circuit` wrapping `circuit`.
fn native_circuit(
    py: Python<'_>,
    circuit: polypus_circuit::ParameterizedCircuit,
) -> Bound<'_, PyAny> {
    Py::new(py, Circuit { inner: circuit })
        .expect("the pyclass instantiates")
        .into_bound(py)
        .into_any()
}

/// A one-qubit, parameter-free `polypus.Circuit`.
fn bound_native_circuit(py: Python<'_>) -> Bound<'_, PyAny> {
    native_circuit(
        py,
        polypus_circuit::ParameterizedCircuit::new(1)
            .x(0)
            .measure_all(),
    )
}

/// A one-parameter `polypus.Circuit` that `polypus.Rotosolve` refuses (a
/// controlled rotation), so a `train` that gets past the circuit checks stops
/// at the Rotosolve check, before any backend exists.
fn rotosolve_refused_circuit(py: Python<'_>) -> Bound<'_, PyAny> {
    native_circuit(
        py,
        polypus_circuit::ParameterizedCircuit::new(2)
            .crx(0, 1, polypus_circuit::GateParam::Param(0))
            .measure_all(),
    )
}

/// The start of the Rotosolve rejection of [`rotosolve_refused_circuit`].
const ROTOSOLVE_REFUSED: &str = "polypus.Rotosolve cannot train this circuit";

/// A `polypus.DE` built through its Python signature.
fn de_method(py: Python<'_>, population_size: u32) -> Bound<'_, PyAny> {
    let kwargs = [("generations", 1), ("population_size", population_size)]
        .into_py_dict(py)
        .expect("the kwargs build");
    py.get_type::<DE>()
        .call((), Some(&kwargs))
        .expect("the pyclass instantiates")
}

/// A `polypus.Rotosolve` with its defaults but the seed.
fn rotosolve_method(py: Python<'_>) -> Bound<'_, PyAny> {
    Py::new(py, Rotosolve::new(7, 3, 0.25, 4, None))
        .expect("the pyclass instantiates")
        .into_bound(py)
        .into_any()
}

/// `run_quantum_circuit(qc, shots, infrastructure, backend=backend, seed=seed)`.
fn run_case(
    qc: Bound<'_, PyAny>,
    shots: u32,
    infrastructure: &str,
    backend: &str,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    run_quantum_circuit(
        qc,
        shots,
        Some(infrastructure.to_string()),
        1,
        None,
        None,
        "automatic",
        None,
        backend,
        seed,
        None,
        None,
        None,
        None,
        None,
    )
}

/// `train` of a one-dimensional `qc` with `method`, a callable cost and
/// `callback`, on `infrastructure`/`backend`.
fn train_case<'py>(
    py: Python<'py>,
    qc: Bound<'py, PyAny>,
    method: Bound<'py, PyAny>,
    shots: u32,
    infrastructure: &str,
    backend: &str,
    callback: Option<Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let cost = py.eval(c"lambda bits: 0.0", None, None).unwrap();
    train(
        qc,
        method,
        shots,
        1,
        1,
        cost,
        Some(infrastructure.to_string()),
        Some(1),
        Some(1),
        Some("edge".to_string()),
        "automatic",
        None,
        backend,
        Some(3),
        None,
        None,
        None,
        None,
        None,
        callback,
        None,
        None,
    )
}

/// Stand-ins for a Qiskit feature map and ansatz with one parameter each:
/// what `qml.train` and `qml.predict` read of them before the backend exists
/// (`parameters`, `compose`, `num_clbits`, `assign_parameters`).
fn qml_circuits(py: Python<'_>) -> (Bound<'_, PyAny>, Bound<'_, PyAny>) {
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
        c"entry_qml_circuits.py",
        c"entry_qml_circuits",
    )
    .expect("the module compiles");
    let circuit = module.getattr("Circuit").expect("the class exists");
    (
        circuit.call1(("x",)).expect("a feature map"),
        circuit.call1(("w",)).expect("an ansatz"),
    )
}

/// `qml.train` of the [`qml_circuits`] on two rows, with `method` and
/// `dimensions`.
fn qml_train_case<'py>(
    py: Python<'py>,
    method: Bound<'py, PyAny>,
    shots: u32,
    dimensions: u32,
    infrastructure: &str,
    backend: &str,
) -> PyResult<Py<PyAny>> {
    let (feature_map, ansatz) = qml_circuits(py);
    let rows = py.eval(c"[[0.1], [0.2]]", None, None).unwrap();
    let cost = py.eval(c"lambda bits: 0.0", None, None).unwrap();
    qml_train(
        feature_map,
        ansatz,
        rows,
        method,
        shots,
        1,
        dimensions,
        cost,
        Some(infrastructure.to_string()),
        Some(1),
        Some(1),
        Some("edge".to_string()),
        "automatic",
        None,
        backend,
        Some(3),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

/// `qml.predict` of the [`qml_circuits`] on two rows, with `params`.
fn qml_predict_case(
    py: Python<'_>,
    params: Vec<f64>,
    shots: u32,
    infrastructure: &str,
    backend: &str,
) -> PyResult<Py<PyAny>> {
    let (feature_map, ansatz) = qml_circuits(py);
    let rows = py.eval(c"[[0.1], [0.2]]", None, None).unwrap();
    qml_predict(
        feature_map,
        ansatz,
        rows,
        params,
        shots,
        Some(infrastructure.to_string()),
        1,
        None,
        None,
        "automatic",
        None,
        backend,
        Some(3),
        None,
        None,
    )
}

const RUN_NATIVE: &str = "the native 'polypus' backend cannot execute a Qiskit QuantumCircuit; \
     pass a polypus.Circuit or an OpenQASM 2.0 string, or use backend=\"aer\"";
const RUN_QMIO: &str = "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize a Qiskit QuantumCircuit; pass a polypus.Circuit or an OpenQASM 2.0 string";
const TRAIN_NATIVE: &str = "the native 'polypus' backend requires a native polypus.Circuit; \
     got a Qiskit circuit. Build it with polypus.Circuit or use backend=\"aer\"";
const TRAIN_QMIO: &str = "the 'qmio' infrastructure requires a native polypus.Circuit (GIL-free \
     serialization); got a Qiskit circuit. Build it with polypus.Circuit";
const QML_TRAIN_NATIVE: &str = "the native 'polypus' backend is not supported for qml.train \
     (feature maps and ansätze are Qiskit circuits); use backend=\"aer\"";
const QML_PREDICT_NATIVE: &str = "the native 'polypus' backend is not supported for qml.predict \
     (feature maps and ansätze are Qiskit circuits); use backend=\"aer\"";
const QML_QMIO: &str = "the 'qmio' infrastructure runs entirely in Rust (GIL-free) and cannot \
     serialize the Qiskit circuits of a QML model";

/// `run_quantum_circuit`: a Qiskit circuit is refused on the native backend
/// and on QMIO with each one's message, and reaches any other backend; a
/// native circuit is accepted everywhere.
#[test]
fn run_quantum_circuit_refuses_a_qiskit_circuit_on_native_and_qmio() {
    register_failing_backend("sentinel-rule-run");
    Python::initialize();
    Python::attach(|py| {
        let qasm = PyString::new(py, "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nx q[0];\nmeasure q -> c;\n").into_any();

        // Native backend.
        assert_rejected(
            py,
            run_case(qiskit_circuit(py), 8, "local", "polypus", Some(1)),
            RUN_NATIVE,
        );
        for native in [bound_native_circuit(py), qasm.clone()] {
            let result = run_case(native, 8, "local", "polypus", Some(1))
                .expect("a native circuit runs on the native backend");
            let counts = result
                .bind(py)
                .getattr("merged_counts")
                .unwrap()
                .extract::<HashMap<String, u64>>()
                .unwrap();
            assert_eq!(counts, HashMap::from([("1".to_string(), 8)]));
        }

        // QMIO: a native circuit passes the rule and is stopped by the seed
        // check right after it, before anything is built.
        assert_rejected(
            py,
            run_case(qiskit_circuit(py), 8, "qmio", "openqasm", None),
            RUN_QMIO,
        );
        for native in [bound_native_circuit(py), qasm.clone()] {
            assert_rejected(
                py,
                run_case(native, 8, "qmio", "openqasm", Some(1)),
                "seed is not supported for the 'qmio' infrastructure (real quantum hardware)",
            );
        }

        // Any other backend: every circuit reaches it.
        for qc in [qiskit_circuit(py), bound_native_circuit(py), qasm] {
            assert_reached_backend(py, run_case(qc, 8, "sentinel-rule-run", "aer", Some(1)));
        }
    });
}

/// `run_quantum_circuit` checks, in this order: `shots`/`n_qpus`, that the
/// circuit is bound, the native backend, QMIO, and last the QMIO seed.
#[test]
fn run_quantum_circuit_checks_in_a_fixed_order() {
    Python::initialize();
    Python::attach(|py| {
        assert_rejected(
            py,
            run_case(qiskit_circuit(py), 0, "local", "polypus", Some(1)),
            "shots must be >= 1, got 0",
        );
        let unbound = py
            .eval(c"type('QC', (), {'parameters': ['t']})()", None, None)
            .unwrap();
        assert_rejected(
            py,
            run_case(unbound, 8, "local", "polypus", Some(1)),
            "circuit has unbound parameters and cannot be executed directly: t. Bind values \
             first via QuantumCircuit.assign_parameters or use polypus.train",
        );
        // `backend` names the native device whatever the infrastructure, and
        // the native check comes first.
        assert_rejected(
            py,
            run_case(qiskit_circuit(py), 8, "qmio", "polypus", Some(1)),
            RUN_NATIVE,
        );
        assert_rejected(
            py,
            run_case(qiskit_circuit(py), 8, "qmio", "openqasm", Some(1)),
            RUN_QMIO,
        );
    });
}

/// `train`: a Qiskit circuit is refused on the native backend and on QMIO
/// with each one's message, and reaches any other backend; a native circuit
/// is accepted everywhere.
#[test]
fn train_refuses_a_qiskit_circuit_on_native_and_qmio() {
    register_failing_backend("sentinel-rule-train");
    Python::initialize();
    Python::attach(|py| {
        for (infrastructure, backend, message) in [
            ("local", "polypus", TRAIN_NATIVE),
            ("qmio", "openqasm", TRAIN_QMIO),
        ] {
            assert_rejected(
                py,
                train_case(
                    py,
                    qiskit_circuit(py),
                    de_method(py, 4),
                    16,
                    infrastructure,
                    backend,
                    None,
                ),
                message,
            );
            // A native circuit passes the rule and is stopped by the
            // Rotosolve check right after it, before anything is built.
            let (class, text) = failure(
                py,
                train_case(
                    py,
                    rotosolve_refused_circuit(py),
                    rotosolve_method(py),
                    16,
                    infrastructure,
                    backend,
                    None,
                ),
            );
            assert_eq!(class, "ConfigurationError", "{text}");
            assert!(text.starts_with(ROTOSOLVE_REFUSED), "{text}");
        }
        for qc in [qiskit_circuit(py), rotosolve_refused_circuit(py)] {
            assert_reached_backend(
                py,
                train_case(
                    py,
                    qc,
                    de_method(py, 4),
                    16,
                    "sentinel-rule-train",
                    "aer",
                    None,
                ),
            );
        }
    });
}

/// `train` checks, in this order: `shots`/`n_qpus`, the optimizer, the
/// `callback`, the native backend and QMIO.
#[test]
fn train_checks_in_a_fixed_order() {
    Python::initialize();
    Python::attach(|py| {
        let not_callable = py.eval(c"'checkpoint.json'", None, None).unwrap();
        assert_rejected(
            py,
            train_case(
                py,
                qiskit_circuit(py),
                de_method(py, 3),
                0,
                "local",
                "polypus",
                Some(not_callable.clone()),
            ),
            "shots must be >= 1, got 0",
        );
        let (class, text) = failure(
            py,
            train_case(
                py,
                qiskit_circuit(py),
                de_method(py, 3),
                16,
                "local",
                "polypus",
                Some(not_callable.clone()),
            ),
        );
        assert_eq!(class, "ConfigurationError", "{text}");
        assert!(text.contains("population_size"), "{text}");
        let (class, text) = failure(
            py,
            train_case(
                py,
                qiskit_circuit(py),
                de_method(py, 4),
                16,
                "local",
                "polypus",
                Some(not_callable),
            ),
        );
        assert_eq!(class, "TypeError", "{text}");
        assert!(
            text.starts_with("callback must be a callable or None"),
            "{text}"
        );
        assert_rejected(
            py,
            train_case(
                py,
                qiskit_circuit(py),
                de_method(py, 4),
                16,
                "qmio",
                "polypus",
                None,
            ),
            TRAIN_NATIVE,
        );
    });
}

/// `qml.train` and `qml.predict` refuse the native backend and QMIO with
/// their own messages, whatever the model, and reach any other backend.
#[test]
fn qml_entry_points_refuse_native_and_qmio() {
    register_failing_backend("sentinel-rule-qml");
    Python::initialize();
    Python::attach(|py| {
        assert_rejected(
            py,
            qml_train_case(py, de_method(py, 4), 16, 1, "local", "polypus"),
            QML_TRAIN_NATIVE,
        );
        assert_rejected(
            py,
            qml_train_case(py, de_method(py, 4), 16, 1, "qmio", "openqasm"),
            QML_QMIO,
        );
        assert_reached_backend(
            py,
            qml_train_case(py, de_method(py, 4), 16, 1, "sentinel-rule-qml", "aer"),
        );

        assert_rejected(
            py,
            qml_predict_case(py, vec![0.5], 16, "local", "polypus"),
            QML_PREDICT_NATIVE,
        );
        assert_rejected(
            py,
            qml_predict_case(py, vec![0.5], 16, "qmio", "openqasm"),
            QML_QMIO,
        );
        assert_reached_backend(
            py,
            qml_predict_case(py, vec![0.5], 16, "sentinel-rule-qml", "aer"),
        );
    });
}

/// `qml.train` checks `shots`/`n_qpus` before the backend rules, and those
/// before `dimensions` and the optimizer; `qml.predict` checks them before
/// `params`.
#[test]
fn qml_entry_points_check_in_a_fixed_order() {
    Python::initialize();
    Python::attach(|py| {
        assert_rejected(
            py,
            qml_train_case(py, de_method(py, 4), 0, 1, "local", "polypus"),
            "shots must be >= 1, got 0",
        );
        // A wrong `dimensions` and a population too small, both after the rule.
        assert_rejected(
            py,
            qml_train_case(py, de_method(py, 3), 16, 2, "local", "polypus"),
            QML_TRAIN_NATIVE,
        );
        assert_rejected(
            py,
            qml_train_case(py, de_method(py, 3), 16, 2, "qmio", "polypus"),
            QML_TRAIN_NATIVE,
        );

        assert_rejected(
            py,
            qml_predict_case(py, vec![0.5], 0, "local", "polypus"),
            "shots must be >= 1, got 0",
        );
        // Two weights for a one-parameter ansatz, checked after the rule.
        assert_rejected(
            py,
            qml_predict_case(py, vec![0.5, 0.5], 16, "local", "polypus"),
            QML_PREDICT_NATIVE,
        );
        assert_rejected(
            py,
            qml_predict_case(py, vec![0.5, 0.5], 16, "qmio", "polypus"),
            QML_PREDICT_NATIVE,
        );
    });
}
