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

#[test]
fn test_num_clbits_no_measurements() {
    let qc = ParameterizedCircuit::new(2);

    assert_eq!(qc.num_clbits(), 0);
}

#[test]
fn test_num_clbits_single_measurement() {
    let qc = ParameterizedCircuit::new(4).measure(0, 3);

    assert_eq!(qc.num_clbits(), 4);
}

#[test]
fn test_num_clbits_multiple_measurements() {
    let qc = ParameterizedCircuit::new(6).measure(0, 3).measure(1, 5);

    assert_eq!(qc.num_clbits(), 6);
}

#[test]
fn test_num_clbits_measure_all() {
    let qc = ParameterizedCircuit::new(3).measure_all();

    assert_eq!(qc.num_clbits(), 3);
}

#[test]
fn test_num_clbits_measure_all_after_measurement() {
    let qc = ParameterizedCircuit::new(3).measure(0, 1).measure_all();

    assert_eq!(qc.num_clbits(), 3);
}

#[test]
fn test_num_clbits_single_zero() {
    let qc = ParameterizedCircuit::new(3).measure(0, 0);

    assert_eq!(qc.num_clbits(), 1);
}

#[test]
fn test_to_qasm2_basic() {
    let qc = ParameterizedCircuit::new(1).h(0);
    let qasm = qc.to_qasm2_with_params(&[]).unwrap();


    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[1];"));
    assert!(qasm.contains("h q[0];"));
    assert!(!qasm.contains("creg"))
}

#[test]
fn test_to_qasm2_with_params_basic() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0));
    let qasm = qc.to_qasm2_with_params(&[0.5]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[1];"));
    assert!(qasm.contains("rx("));
    assert!(qasm.contains("0.5"));
    assert!(qasm.contains("q[0];"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_to_qasm2_multiple_gates_measure() {
    let qc = ParameterizedCircuit::new(2).h(0).rx(0, Param(0)).ry(1, Param(1)).measure_all();
    let qasm = qc.to_qasm2_with_params(&[0.5, 1.0]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("h q[0];"));
    assert!(qasm.contains("rx("));
    assert!(qasm.contains("0.5"));
    assert!(qasm.contains("q[0];"));
    assert!(qasm.contains("ry("));
    assert!(qasm.contains("1.0"));
    assert!(qasm.contains("q[1];"));
    assert!(qasm.contains("creg c[2];")); 
    assert!(qasm.contains("measure q -> c;"));
}

#[test]
fn test_to_qasm2_multiple_gates_and_parameters() {
    let qc = ParameterizedCircuit::new(2).rx(0, Param(0)).ry(1, Param(1));
    let qasm = qc.to_qasm2_with_params(&[0.1, 0.2]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("rx("));
    assert!(qasm.contains("0.1"));
    assert!(qasm.contains("q[0];"));
    assert!(qasm.contains("ry("));
    assert!(qasm.contains("0.2"));
    assert!(qasm.contains("q[1];"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_to_qasm2_wrong_number_of_param() {
    
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0));
    let result = qc.to_qasm2_with_params(&[]);

    assert_eq!(result,
        Err(CircuitError::WrongNumberOfParams { expected: 1, got: 0 })
    );
}