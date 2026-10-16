//! `polypus.backend_compatibility`: which local backend can run a circuit, and
//! why not (issue #218).
//!
//! The native `"polypus"` backend runs terminal-measurement circuits only
//! (contract C-4, `docs/adr/0001-terminal-measurements.md`) while Aer also runs
//! dynamic ones (`reset`, mid-circuit measurement, classical control), and Aer
//! alone runs Qiskit circuits. This check tells a caller, before switching
//! backends, what each one would reject. It is structural: it runs nothing,
//! connects to nothing and does not look at resources (the qubit ceiling, the
//! memory budget, a QPU's size).
//!
//! Each entry reuses what the backend itself decides. The native one calls
//! [`NativeStatevectorBackend::check_circuit`] (the OpenQASM importer, the same
//! call execution makes) and the entry point's Qiskit guard, and reports the
//! message the run would raise. The Aer one parses with the parser the Aer path
//! uses (`QuantumCircuit.from_qasm_str`). For a circuit that reaches Aer as
//! OpenQASM (a `polypus.Circuit` or a `str`) it does not compare instructions
//! with Aer's target: the local backend lowers what is outside Aer's basis
//! before running it (`polypus_python/local.py`, contract C-2), so an
//! instruction outside the basis is no reason to reject. For a Qiskit
//! `QuantumCircuit` the Aer entry replays that lowering: when an instruction is
//! outside the basis it calls the rewrite the run calls
//! (`_rewrite_into_aer_basis`), so one that cannot be lowered (an opaque
//! instruction without a definition, a `box`) is reported with the error the run
//! would raise. The rewrite is shared with `local.py`; the decision to rewrite
//! is copied from it, and `tests/python/test_backend_compatibility.py` keeps the
//! two in step.

use std::collections::HashSet;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::exceptions::{
    backend_error_to_pyerr, describe_python_error, qiskit_error_to_pyerr, BackendError,
};
use crate::infrastructure::{to_py_object, BoundCircuit, NativeStatevectorBackend};

/// Qiskit's classical control-flow operations: each makes a dynamic circuit.
const CONTROL_FLOW: [&str; 4] = ["if_else", "while_loop", "for_loop", "switch_case"];

/// For each local backend (`"aer"`, `"polypus"`), the reasons it would reject
/// `circuit`; an empty list means it accepts it.
///
/// `circuit` is a `polypus.Circuit`, an OpenQASM 2.0 `str` or a Qiskit
/// `QuantumCircuit`, classified exactly as `run_quantum_circuit` does. The check
/// is structural and runs nothing: a circuit accepted here can still fail for
/// lack of resources (qubits, memory).
///
/// - `"polypus"` rejects what the native backend rejects, with the message the
///   run would raise: an OpenQASM program with `reset`, `if` or a gate after a
///   measurement (C-4), one that does not parse, and any Qiskit
///   `QuantumCircuit`, for which the dynamic features found in it are listed
///   as well.
/// - `"aer"` rejects an OpenQASM program Qiskit cannot parse; if Qiskit Aer is
///   not installed, that is the reason. An instruction outside Aer's basis (such
///   as `ch` or a declared gate) is no reason: the local backend lowers it
///   (synthesis and basis translation) before running it. A Qiskit
///   `QuantumCircuit` with such an instruction is rewritten here by the run's
///   own rewrite, so one that cannot be lowered (an opaque instruction without
///   a definition, a `box`) is rejected with the error the run would raise
///   (`polypus.BackendError`'s message); checking it costs the rewrite the run
///   would pay, which a circuit in the basis does not. Like the run, it lowers
///   into the basis of the default `AerSimulator()` whatever `sim_method` or
///   `noise_model` a run is given (either one narrows the simulator's target,
///   not what Aer runs), with no cap on the width.
/// - A `polypus.Circuit` with free parameters cannot be run by either backend;
///   both entries give the same reason.
///
/// Raises `TypeError` for any other object.
///
/// ```python
/// report = polypus.backend_compatibility(qasm)
/// if not report["polypus"]:
///     result = polypus.run_quantum_circuit(qasm, shots=1000,
///                                          infrastructure="local", backend="polypus")
/// ```
#[pyfunction]
pub(crate) fn backend_compatibility<'py>(
    circuit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = circuit.py();
    let (aer, native) = reasons(circuit).map_err(qiskit_error_to_pyerr)?;
    let report = PyDict::new(py);
    report.set_item("aer", PyList::new(py, aer)?)?;
    report.set_item("polypus", PyList::new(py, native)?)?;
    Ok(report)
}

/// The `(aer, polypus)` rejection reasons for `circuit`.
fn reasons(circuit: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Vec<String>)> {
    let py = circuit.py();
    let bound = match super::extract_bound_circuit(circuit) {
        Ok(bound) => bound,
        // A `polypus.Circuit` with free parameters: no backend can run it.
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            let reason = message(py, &err);
            return Ok((vec![reason.clone()], vec![reason]));
        }
        Err(err) => return Err(err),
    };
    if bound.is_foreign() {
        let qc = quantum_circuit(circuit)?;
        let mut native = vec![super::NATIVE_REJECTS_QISKIT.to_string()];
        native.extend(dynamic_features(qc)?);
        return Ok((aer_quantum_circuit_rejections(qc)?, native));
    }
    let native = match NativeStatevectorBackend::check_circuit(&bound) {
        Ok(()) => Vec::new(),
        Err(err) => vec![message(py, &backend_error_to_pyerr(err))],
    };
    Ok((aer_qasm_rejections(py, &bound)?, native))
}

/// `str(err)`: what a caller reads from the exception a run would raise.
fn message(py: Python<'_>, err: &PyErr) -> String {
    err.value(py)
        .str()
        .map_or_else(|_| err.to_string(), |m| m.to_string())
}

/// `circuit` itself when it is a Qiskit `QuantumCircuit`, else the `TypeError`
/// for an object that is no circuit at all. Without Qiskit installed nothing
/// can be a `QuantumCircuit`.
fn quantum_circuit<'a, 'py>(circuit: &'a Bound<'py, PyAny>) -> PyResult<&'a Bound<'py, PyAny>> {
    let py = circuit.py();
    let is_circuit = PyModule::import(py, "qiskit")
        .and_then(|qiskit| qiskit.getattr("QuantumCircuit"))
        .and_then(|class| circuit.is_instance(&class))
        .unwrap_or(false);
    if is_circuit {
        Ok(circuit)
    } else {
        Err(PyTypeError::new_err(format!(
            "backend_compatibility expects a polypus.Circuit, an OpenQASM 2.0 str or a \
             qiskit QuantumCircuit, got {}",
            circuit.get_type().name()?
        )))
    }
}

/// The Aer reasons for a native or OpenQASM circuit: it reaches Aer as
/// OpenQASM 2.0 and is parsed there by `QuantumCircuit.from_qasm_str`, so a
/// parse failure is quoted as the run would raise it (`polypus.BackendError`'s
/// `module.Class: message`). A circuit that parses is accepted.
fn aer_qasm_rejections(py: Python<'_>, bound: &BoundCircuit) -> PyResult<Vec<String>> {
    let Some(from_qasm_str) = aer_parser(py) else {
        return Ok(vec![aer_missing(py)]);
    };
    let qasm = to_py_object(bound, py).map_err(backend_error_to_pyerr)?;
    match from_qasm_str.call1((qasm,)) {
        Ok(_) => Ok(Vec::new()),
        Err(err) => Ok(vec![describe_python_error(py, &err)]),
    }
}

/// `QuantumCircuit.from_qasm_str`, if Qiskit and Qiskit Aer are importable.
fn aer_parser(py: Python<'_>) -> Option<Bound<'_, PyAny>> {
    PyModule::import(py, "qiskit_aer").ok()?;
    PyModule::import(py, "qiskit")
        .and_then(|qiskit| qiskit.getattr("QuantumCircuit"))
        .and_then(|class| class.getattr("from_qasm_str"))
        .ok()
}

/// The reason given when Aer cannot run anything: it is not installed, or
/// `QuantumCircuit.from_qasm_str` is missing from what is.
fn aer_missing(py: Python<'_>) -> String {
    match PyModule::import(py, "qiskit_aer").and(PyModule::import(py, "qiskit")) {
        Ok(_) => "Qiskit Aer is not available".to_string(),
        Err(err) => format!(
            "Qiskit Aer is not installed ({}), so backend=\"aer\" cannot run",
            describe_python_error(py, &err)
        ),
    }
}

/// The Aer reasons for a Qiskit `QuantumCircuit`, decided as the local backend
/// prepares it for Aer (`_lower_to_aer_basis` in `polypus_python/local.py`,
/// contract C-2). A circuit whose instructions are all in the basis of the
/// default `AerSimulator()` reaches Aer as it is, so it is accepted without
/// being rewritten. One with an instruction outside it, at any depth
/// ([`outside_basis`]), is rewritten as the run rewrites it
/// (`_rewrite_into_aer_basis`; the result is a new circuit and is discarded, the
/// caller's is not modified). A failure follows the run's own rule
/// ([`qiskit_error_to_pyerr`]): what the run would raise as
/// `polypus.BackendError` (a Qiskit exception) is quoted as its message
/// (`module.Class: message`); anything else, a Qiskit class that is also a
/// `ValueError`, `TypeError` or `KeyboardInterrupt` included, propagates as the
/// run would raise it. The rewrite is looked up on every call.
fn aer_quantum_circuit_rejections(qc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let py = qc.py();
    let Some(aer_simulator) = aer_simulator_class(py) else {
        return Ok(vec![aer_missing(py)]);
    };
    let basis = aer_simulator
        .call0()?
        .getattr(intern!(py, "target"))?
        .getattr(intern!(py, "operation_names"))?;
    let supported = names(&basis)?;
    let control_flow = names(
        &PyModule::import(py, "qiskit.circuit")?.getattr(intern!(py, "CONTROL_FLOW_OP_NAMES"))?,
    )?;
    if !outside_basis(qc, &supported, &control_flow)? {
        return Ok(Vec::new());
    }
    let rewrite = PyModule::import(py, "polypus_python.local")?
        .getattr(intern!(py, "_rewrite_into_aer_basis"))?;
    match rewrite.call1((qc, basis)) {
        Ok(_) => Ok(Vec::new()),
        Err(err) => {
            let raised = qiskit_error_to_pyerr(err);
            if raised.is_instance_of::<BackendError>(py) {
                Ok(vec![message(py, &raised)])
            } else {
                Err(raised)
            }
        }
    }
}

/// `qiskit_aer.AerSimulator`, if Qiskit and Qiskit Aer are importable.
fn aer_simulator_class(py: Python<'_>) -> Option<Bound<'_, PyAny>> {
    PyModule::import(py, "qiskit").ok()?;
    PyModule::import(py, "qiskit_aer")
        .and_then(|aer| aer.getattr("AerSimulator"))
        .ok()
}

/// The strings in the Python iterable `iterable` (a set of names, or the keys
/// of a `count_ops()` dict).
fn names(iterable: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    iterable
        .try_iter()?
        .map(|name| name?.extract::<String>())
        .collect()
}

/// Whether the Qiskit circuit `qc` holds an instruction whose name is not in
/// `supported`, at any depth: `_outside_basis` in `polypus_python/local.py`,
/// rule for rule. `barrier` is accepted everywhere; the names come from
/// `count_ops()`, and `qc.data` is walked only when one of them is in
/// `control_flow` (Qiskit's `CONTROL_FLOW_OP_NAMES`), to look into the bodies
/// of those operations. A control-flow operation outside `supported` (for Aer,
/// today `box`) is outside by its name, before any body is looked at.
fn outside_basis(
    qc: &Bound<'_, PyAny>,
    supported: &HashSet<String>,
    control_flow: &HashSet<String>,
) -> PyResult<bool> {
    let py = qc.py();
    let used = names(&qc.call_method0(intern!(py, "count_ops"))?)?;
    if used
        .iter()
        .any(|name| name != "barrier" && !supported.contains(name))
    {
        return Ok(true);
    }
    if used.is_disjoint(control_flow) {
        return Ok(false);
    }
    for instruction in qc.getattr(intern!(py, "data"))?.try_iter()? {
        let operation = instruction?.getattr(intern!(py, "operation"))?;
        let name: String = operation.getattr(intern!(py, "name"))?.extract()?;
        if !control_flow.contains(&name) {
            continue;
        }
        for block in operation.getattr(intern!(py, "blocks"))?.try_iter()? {
            if outside_basis(&block?, supported, control_flow)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The dynamic-circuit features of the Qiskit circuit `qc` that a Polypus
/// circuit cannot express (C-4), phrased like the OpenQASM importer's own
/// rejections: `reset`, a gate on an already-measured qubit, classical control
/// flow and legacy `c_if` conditions. A `barrier`, and measuring a qubit again,
/// are allowed, exactly as in `polypus_circuit::terminal_measurement_violation`.
fn dynamic_features(qc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let py = qc.py();
    let mut measured = HashSet::new();
    let mut reasons: Vec<String> = Vec::new();
    let mut report = |reason: String| {
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    };
    for instruction in qc.getattr(intern!(py, "data"))?.try_iter()? {
        let instruction = instruction?;
        let operation = instruction.getattr(intern!(py, "operation"))?;
        let name: String = operation.getattr(intern!(py, "name"))?.extract()?;
        if CONTROL_FLOW.contains(&name.as_str()) {
            report(format!(
                "instruction '{name}' is classical control flow, which makes a dynamic \
                 circuit; Polypus circuits use terminal measurement (contract C-4, \
                 docs/adr/0001-terminal-measurements.md)"
            ));
            continue;
        }
        // Qiskit < 2 conditioned single gates with `c_if`; 2.x has no such
        // attribute, so this only fires on circuits built with an older Qiskit.
        if operation
            .getattr(intern!(py, "condition"))
            .is_ok_and(|condition| !condition.is_none())
        {
            report(format!(
                "instruction '{name}' is classically conditioned, which makes a dynamic \
                 circuit; Polypus circuits use terminal measurement (contract C-4, \
                 docs/adr/0001-terminal-measurements.md)"
            ));
        }
        for qubit in instruction.getattr(intern!(py, "qubits"))?.try_iter()? {
            let index: usize = qc
                .call_method1(intern!(py, "find_bit"), (qubit?,))?
                .getattr(intern!(py, "index"))?
                .extract()?;
            match name.as_str() {
                "measure" => {
                    measured.insert(index);
                }
                "barrier" => {}
                "reset" => report(format!(
                    "'reset' on qubit {index} is not supported: Polypus circuits use terminal \
                     measurement and no other non-unitary operation (contract C-4, \
                     docs/adr/0001-terminal-measurements.md)"
                )),
                _ if measured.contains(&index) => report(format!(
                    "'{name}' acts on qubit {index} after it was measured; Polypus circuits \
                     use terminal measurement (contract C-4)"
                )),
                _ => {}
            }
        }
    }
    Ok(reasons)
}
