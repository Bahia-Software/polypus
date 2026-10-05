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
//! uses (`QuantumCircuit.from_qasm_str`) and compares the instructions with Aer's
//! own target.

use std::collections::HashSet;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::exceptions::{backend_error_to_pyerr, describe_python_error, qiskit_error_to_pyerr};
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
/// - `"aer"` rejects an OpenQASM program Qiskit cannot parse and an
///   instruction outside Aer's basis (such as `ch` or a declared gate, which
///   need `qiskit.transpile` first). Aer is checked for its default simulation
///   method; if Qiskit Aer is not installed, that is the reason.
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
        return Ok((aer_rejections(qc)?, native));
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
/// `module.Class: message`); a parsed circuit is checked like a Qiskit one.
fn aer_qasm_rejections(py: Python<'_>, bound: &BoundCircuit) -> PyResult<Vec<String>> {
    let Some(from_qasm_str) = aer_parser(py) else {
        return Ok(vec![aer_missing(py)]);
    };
    let qasm = to_py_object(bound, py).map_err(backend_error_to_pyerr)?;
    match from_qasm_str.call1((qasm,)) {
        Ok(qc) => aer_rejections(&qc),
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

/// The reason given when Aer cannot run anything: it is not installed.
fn aer_missing(py: Python<'_>) -> String {
    match PyModule::import(py, "qiskit_aer").and(PyModule::import(py, "qiskit")) {
        Ok(_) => "Qiskit Aer is not available".to_string(),
        Err(err) => format!(
            "Qiskit Aer is not installed ({}), so backend=\"aer\" cannot run",
            describe_python_error(py, &err)
        ),
    }
}

/// Instructions of the Qiskit circuit `qc` that Aer's default target does not
/// support, one reason each, in order of first appearance (control-flow bodies
/// included). `barrier` is a directive Aer always accepts.
fn aer_rejections(qc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let py = qc.py();
    let Ok(aer) = PyModule::import(py, "qiskit_aer") else {
        return Ok(vec![aer_missing(py)]);
    };
    let supported: HashSet<String> = aer
        .getattr("AerSimulator")?
        .call0()?
        .getattr("target")?
        .getattr("operation_names")?
        .try_iter()?
        .map(|name| name?.extract::<String>())
        .collect::<PyResult<_>>()?;
    let mut unsupported = Vec::new();
    collect_unsupported(qc, &supported, &mut unsupported)?;
    Ok(unsupported
        .into_iter()
        .map(|name| {
            format!(
                "instruction '{name}' is not in Aer's basis: transpile the circuit first \
                 (qiskit.transpile(qc, AerSimulator()))"
            )
        })
        .collect())
}

fn collect_unsupported(
    qc: &Bound<'_, PyAny>,
    supported: &HashSet<String>,
    unsupported: &mut Vec<String>,
) -> PyResult<()> {
    let py = qc.py();
    for instruction in qc.getattr(intern!(py, "data"))?.try_iter()? {
        let operation = instruction?.getattr(intern!(py, "operation"))?;
        let name: String = operation.getattr(intern!(py, "name"))?.extract()?;
        if name != "barrier" && !supported.contains(&name) && !unsupported.contains(&name) {
            unsupported.push(name);
        }
        if let Ok(blocks) = operation.getattr(intern!(py, "blocks")) {
            for block in blocks.try_iter()? {
                collect_unsupported(&block?, supported, unsupported)?;
            }
        }
    }
    Ok(())
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
