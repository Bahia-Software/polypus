//! Integration tests for polypus-circuit: errors

use polypus_circuit::CircuitError;

#[test]
fn test_circuit_error_display_wrong_number() {
    let err = CircuitError::WrongNumberOfParams {
        expected: 2,
        got: 1,
    };

    let msg = format!("{err}");

    assert!(msg.contains("wrong number of parameter values"));
    assert!(msg.contains("2"));
    assert!(msg.contains("1"));
}

#[test]
fn test_circuit_error_display_param_index_out_of_bounds() {
    let err = CircuitError::ParamIndexOutOfBounds {
        index: 5,
        num_params: 2,
    };

    let msg = format!("{err}");

    assert!(msg.contains("parameter index 5"));
    assert!(msg.contains("2"));
}

#[test]
fn test_circuit_error_display_qubit_out_of_range() {
    let err = CircuitError::QubitOutOfRange {
        qubit: 5,
        num_qubits: 2,
    };

    let msg = format!("{err}");

    assert!(msg.contains("qubit index 5 out of range"));
    assert!(msg.contains("2"));
}

#[test]
fn test_circuit_error_display_identical_qubits() {
    let err = CircuitError::IdenticalQubits { qubit: 3 };

    let msg = format!("{err}");

    assert!(msg.contains("two-qubit gate requires distinct qubits"));
    assert!(msg.contains("3"));
}

#[test]
fn test_circuit_error_display_parse() {
    let err = CircuitError::Parse {
        line: 10,
        message: "unexpected token".to_string(),
    };

    let msg = format!("{err}");

    assert!(msg.contains("QASM parse error"));
    assert!(msg.contains("line 10"));
    assert!(msg.contains("unexpected token"));
}
