//! Integration tests for polypus-circuit: circuit
// ParametrizedCircuit

use polypus_circuit::{ CircuitError, ConcreteCircuit, GateInstruction, GateParam, Param, ParameterizedCircuit };

#[test]
fn test_basic_creation(){
    let qc = ParameterizedCircuit::new(3);

    assert_eq!(qc.num_qubits, 3);
    assert_eq!(qc.num_params, 0);
    assert_eq!(qc.gates.len(), 0);
}

#[test]
fn test_gate_basic_creation(){
    let qc = ParameterizedCircuit::new(2).h(0);

    assert_eq!(qc.num_qubits, 2);
    assert_eq!(qc.num_params, 0);
    assert_eq!(qc.gates.len(), 1);
    assert_eq!(qc.gates[0], GateInstruction::H(0));
    match qc.gates[0] {
        GateInstruction::H(q) => assert_eq!(q, 0),
        _ => panic!("Expected H"),
    }
}

#[test]
fn test_gate_param_basic_creation(){
    let qc = ParameterizedCircuit::new(2).rx(0, GateParam::Param(3));

    assert_eq!(qc.num_qubits, 2);
    assert_eq!(qc.num_params, 4); // Param(3) means we have 4 parameters (0, 1, 2, 3)
    assert_eq!(qc.gates.len(), 1);
    assert_eq!(qc.gates[0], GateInstruction::Rx { qubit: 0, theta: GateParam::Param(3) });
    match qc.gates[0] {
        GateInstruction::Rx { qubit, theta } => {
            assert_eq!(qubit, 0);
            assert_eq!(theta, GateParam::Param(3));
        }
        _ => panic!("Expected Rx"),
    }
}

#[test]
#[should_panic]
fn test_gate_basic_qubit_out_of_range(){
    let _qc = ParameterizedCircuit::new(2).h(5);
}

#[test]
#[should_panic]
fn test_gate_basic_two_same_qubit(){
    let _qc = ParameterizedCircuit::new(2).cx(0, 0);
}

#[test]
fn test_gate_assign_parameters(){
    let qc = ParameterizedCircuit::new(2).rx(0, Param(0)).ry(1, Param(1));
    assert_eq!(qc.num_params, 2);

    let concrete_qc = qc.assign_parameters(&[0.1, 0.2]).unwrap();
    assert_eq!(concrete_qc.gates[0], GateInstruction::Rx { qubit: 0, theta: GateParam::Fixed(0.1) });
    assert_eq!(concrete_qc.gates[1], GateInstruction::Ry { qubit: 1, theta: GateParam::Fixed(0.2) });
}

#[test]
fn test_gate_assign_parameters_wrong_number_of_params(){
    let qc = ParameterizedCircuit::new(2).rx(0, Param(0)).ry(1, Param(1));
    assert_eq!(qc.num_params, 2);

    let result = qc.assign_parameters(&[0.1]);
    assert_eq!(result, Err(CircuitError::WrongNumberOfParams { expected: 2, got: 1 }));
}