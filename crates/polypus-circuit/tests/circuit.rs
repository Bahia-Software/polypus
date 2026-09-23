//! Integration tests for polypus-circuit: circuit
// ParameterizedCircuit

use polypus_circuit::{
    CircuitError, ConcreteCircuit, GateInstruction, GateParam, Param, ParameterizedCircuit,
};

#[test]
fn test_parameterized_circuit_basic_creation() {
    let qc = ParameterizedCircuit::new(3);

    assert_eq!(qc.num_qubits, 3);
    assert_eq!(qc.num_params, 0);
    assert_eq!(qc.gates.len(), 0);
}

#[test]
fn test_parameterized_circuit_gate_basic_creation() {
    let qc = ParameterizedCircuit::new(2).h(0).cz(0, 1);

    assert_eq!(qc.num_qubits, 2);
    assert_eq!(qc.num_params, 0);
    assert_eq!(qc.gates.len(), 2);
    assert_eq!(qc.gates[0], GateInstruction::H(0));
    assert_eq!(qc.gates[1], GateInstruction::Cz(0, 1));
}

#[test]
fn test_parameterized_circuit_gate_param_basic_creation() {
    let qc = ParameterizedCircuit::new(2).x(0).rx(0, GateParam::Param(3));

    assert_eq!(qc.num_qubits, 2);
    assert_eq!(qc.num_params, 4); // Param(3) means we have 4 parameters (0, 1, 2, 3)
    assert_eq!(qc.gates.len(), 2);
    assert_eq!(qc.gates[0], GateInstruction::X(0));
    assert_eq!(
        qc.gates[1],
        GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Param(3),
        }
    );
}

#[test]
fn test_parameterized_circuit_swap_builder_and_qasm() {
    let qc = ParameterizedCircuit::new(3).swap(0, 2);

    assert_eq!(qc.gates.len(), 1);
    assert_eq!(qc.gates[0], GateInstruction::Swap(0, 2));

    let qasm = qc.to_qasm2_with_params(&[]).unwrap();
    assert!(qasm.contains("swap q[0],q[2];"));
}

#[test]
#[should_panic]
fn test_parameterized_circuit_swap_same_qubit() {
    let _qc = ParameterizedCircuit::new(2).swap(1, 1);
}

#[test]
#[should_panic]
fn test_parameterized_circuit_swap_qubit_out_of_range() {
    let _qc = ParameterizedCircuit::new(2).swap(0, 5);
}

#[test]
#[should_panic]
fn test_parameterized_circuit_gate_basic_qubit_out_of_range() {
    let _qc = ParameterizedCircuit::new(2).y(5);
}

#[test]
#[should_panic]
fn test_parameterized_circuit_gate_basic_two_same_qubit() {
    let _qc = ParameterizedCircuit::new(2).cx(0, 0);
}

#[test]
fn test_parameterized_circuit_gate_assign_parameters() {
    let qc = ParameterizedCircuit::new(2)
        .t(0)
        .rz(0, Param(0))
        .ry(1, Param(1))
        .rzz(0, 1, Param(2));
    assert_eq!(qc.num_params, 3);

    let concrete_qc = qc.assign_parameters(&[0.1, 0.2, 0.3]).unwrap();
    assert_eq!(concrete_qc.gates[0], GateInstruction::T(0));
    assert_eq!(
        concrete_qc.gates[1],
        GateInstruction::Rz {
            qubit: 0,
            theta: GateParam::Fixed(0.1)
        }
    );
    assert_eq!(
        concrete_qc.gates[2],
        GateInstruction::Ry {
            qubit: 1,
            theta: GateParam::Fixed(0.2)
        }
    );
    assert_eq!(
        concrete_qc.gates[3],
        GateInstruction::Rzz {
            q0: 0,
            q1: 1,
            theta: GateParam::Fixed(0.3)
        }
    );
}

#[test]
fn test_parameterized_circuit_gate_assign_parameters_wrong_number_of_params() {
    let qc = ParameterizedCircuit::new(2).rx(0, Param(0)).ry(1, Param(1));
    assert_eq!(qc.num_params, 2);

    let result = qc.assign_parameters(&[0.1]);
    assert_eq!(
        result,
        Err(CircuitError::WrongNumberOfParams {
            expected: 2,
            got: 1
        })
    );
}

#[test]
fn test_parameterized_circuit_assign_parameters_param_index_out_of_bounds() {
    let mut qc = ParameterizedCircuit::new(1);
    qc.num_params = 1;
    qc.gates = vec![GateInstruction::Rx {
        qubit: 0,
        theta: GateParam::Param(5),
    }];

    let result = qc.assign_parameters(&[0.1]);

    assert_eq!(
        result,
        Err(CircuitError::ParamIndexOutOfBounds {
            index: 5,
            num_params: 1
        })
    );
}

#[test]
fn test_parameterized_circuit_assign_parameters_rejects_non_finite() {
    let qc = ParameterizedCircuit::new(2)
        .rx(0, Param(0))
        .u(1, Param(1), Param(2), Param(3));
    assert_eq!(qc.num_params, 4);

    // A caller-supplied non-finite value bound to a `Param` is rejected.
    assert_eq!(
        qc.assign_parameters(&[f64::NAN, 0.1, 0.2, 0.3]),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(
        qc.assign_parameters(&[0.1, f64::INFINITY, 0.2, 0.3]),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(
        qc.assign_parameters(&[0.1, 0.2, 0.3, f64::NEG_INFINITY]),
        Err(CircuitError::NonFiniteParam)
    );
}

#[test]
fn test_parameterized_circuit_num_clbits_no_measurements() {
    let qc = ParameterizedCircuit::new(2);

    assert_eq!(qc.num_clbits(), 0);
}

#[test]
fn test_parameterized_circuit_num_clbits_single_measurement() {
    let qc = ParameterizedCircuit::new(4).measure(0, 3);

    assert_eq!(qc.num_clbits(), 4);
}

#[test]
fn test_parameterized_circuit_num_clbits_multiple_measurements() {
    let qc = ParameterizedCircuit::new(6).measure(0, 3).measure(1, 5);

    assert_eq!(qc.num_clbits(), 6);
}

#[test]
fn test_parameterized_circuit_num_clbits_measure_all() {
    let qc = ParameterizedCircuit::new(3).measure_all();

    assert_eq!(qc.num_clbits(), 3);
}

#[test]
fn test_parameterized_circuit_num_clbits_measure_all_after_measurement() {
    let qc = ParameterizedCircuit::new(3).measure(0, 1).measure_all();

    assert_eq!(qc.num_clbits(), 3);
}

#[test]
fn test_parameterized_circuit_num_clbits_single_zero() {
    let qc = ParameterizedCircuit::new(3).measure(0, 0);

    assert_eq!(qc.num_clbits(), 1);
}

#[test]
fn test_parameterized_circuit_to_qasm2_basic() {
    let qc = ParameterizedCircuit::new(1).y(0);
    let qasm = qc.to_qasm2_with_params(&[]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[1];"));
    assert!(qasm.contains("y q[0];"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_parameterized_circuit_to_qasm2_with_params_basic() {
    let qc = ParameterizedCircuit::new(2)
        .sdg(0)
        .rx(0, Param(0))
        .rxx(0, 1, Param(1));
    let qasm = qc.to_qasm2_with_params(&[0.5, 1.0]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("sdg q[0];"));
    assert!(qasm.contains("rx("));
    assert!(qasm.contains("0.5"));
    assert!(qasm.contains("q[0];"));
    assert!(qasm.contains("rxx("));
    assert!(qasm.contains("1.0"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_parameterized_circuit_to_qasm2_multiple_gates_measure() {
    let qc = ParameterizedCircuit::new(2)
        .z(0)
        .barrier()
        .rx(0, Param(0))
        .ry(1, Param(1))
        .measure_all();
    let qasm = qc.to_qasm2_with_params(&[0.5, 1.0]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("z q[0];"));
    assert!(qasm.contains("barrier"));
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
fn test_parameterized_circuit_to_qasm2_multiple_gates_and_parameters() {
    let qc = ParameterizedCircuit::new(2)
        .tdg(0)
        .barrier_on(&[0, 1])
        .rx(0, Param(0))
        .ry(1, Param(1))
        .u(0, Param(2), Param(3), Param(4));
    let qasm = qc.to_qasm2_with_params(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("tdg q[0];"));
    assert!(qasm.contains("barrier"));
    assert!(qasm.contains("rx("));
    assert!(qasm.contains("0.1"));
    assert!(qasm.contains("q[0];"));
    assert!(qasm.contains("ry("));
    assert!(qasm.contains("0.2"));
    assert!(qasm.contains("q[1];"));
    assert!(qasm.contains("u3("));
    assert!(qasm.contains("0.3"));
    assert!(qasm.contains("0.4"));
    assert!(qasm.contains("0.5"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_parameterized_circuit_to_qasm2_wrong_number_of_params() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0));
    let result = qc.to_qasm2_with_params(&[]);

    assert_eq!(
        result,
        Err(CircuitError::WrongNumberOfParams {
            expected: 1,
            got: 0
        })
    );
}

#[test]
fn test_parameterized_circuit_to_qasm2_parser() {
    let qc = ParameterizedCircuit::new(2)
        .s(0)
        .rx(0, Param(0))
        .ry(1, Param(1))
        .measure_all();
    let qasm = qc.to_qasm2_with_params(&[0.5, 1.0]).unwrap();

    let qc2 = ParameterizedCircuit::from_qasm2(&qasm).unwrap();
    let qasm2 = qc2.to_qasm2_with_params(&[]).unwrap();

    assert_eq!(qasm, qasm2);
}

#[test]
fn test_parameterized_circuit_from_qasm2_invalid_syntax() {
    let result = ParameterizedCircuit::from_qasm2("not valid qasm");

    assert!(matches!(result, Err(CircuitError::Parse { .. })));
}

#[test]
fn test_parameterized_circuit_from_qasm2_qubit_out_of_range() {
    let qasm = r#"
    OPENQASM 2.0;
    qreg q[1];
    h q[5];
    "#;

    let result = ParameterizedCircuit::from_qasm2(qasm);

    assert!(matches!(result, Err(CircuitError::Parse { .. })));
}

#[test]
fn test_parameterized_circuit_from_qasm2_undeclared_register() {
    let qasm = r#"
    OPENQASM 2.0;
    h q[0];
    "#;

    let result = ParameterizedCircuit::from_qasm2(qasm);

    assert!(matches!(result, Err(CircuitError::Parse { .. })));
}

#[test]
fn test_parameterized_circuit_from_qasm2_unsupported_statement() {
    let qasm = r#"
    OPENQASM 2.0;
    qreg q[1];
    reset q[0];
    "#;

    let result = ParameterizedCircuit::from_qasm2(qasm);

    assert!(matches!(result, Err(CircuitError::Parse { .. })));
}

#[test]
fn test_parameterized_circuit_barrier_creation() {
    let qc = ParameterizedCircuit::new(3).barrier().barrier_on(&[0, 2]);

    assert_eq!(qc.gates.len(), 2);
    assert_eq!(qc.gates[0], GateInstruction::Barrier(vec![]));
    assert_eq!(qc.gates[1], GateInstruction::Barrier(vec![0, 2]));
}

#[test]
fn test_parameterized_circuit_try_push_basic() {
    let mut qc = ParameterizedCircuit::new(2);
    assert_eq!(qc.gates.len(), 0);

    let result = qc.try_push(GateInstruction::T(1));
    assert_eq!(qc.gates[0], GateInstruction::T(1));
    assert_eq!(qc.gates.len(), 1);
    assert_eq!(result, Ok(()));
}

#[test]
fn test_parameterized_circuit_try_push_out_of_range() {
    let mut qc = ParameterizedCircuit::new(1);
    let result = qc.try_push(GateInstruction::Sdg(5));

    assert_eq!(
        result,
        Err(CircuitError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 1
        })
    );
}

#[test]
fn test_parameterized_circuit_try_push_same_qubits() {
    let mut qc = ParameterizedCircuit::new(1);
    let result = qc.try_push(GateInstruction::Cz(0, 0));

    assert_eq!(result, Err(CircuitError::IdenticalQubits { qubit: 0 }));
}

#[test]
fn test_parameterized_circuit_try_push_track_params() {
    let mut qc = ParameterizedCircuit::new(2);
    let result = qc.try_push(GateInstruction::Rx {
        qubit: 0,
        theta: GateParam::Param(3),
    });

    assert_eq!(result, Ok(()));
    assert_eq!(qc.num_params, 4);
}

// Push-time C-4 state (contract C-4 enforced incrementally by `try_push`).

#[test]
fn try_push_rejects_unitary_on_a_measured_qubit() {
    let mut qc = ParameterizedCircuit::new(2);
    qc.try_push(GateInstruction::Measure { qubit: 0, cbit: 0 })
        .unwrap();

    assert_eq!(
        qc.try_push(GateInstruction::X(0)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
    );
    // The other qubit is untouched by the measurement, and re-measuring and
    // barriers remain legal on the measured one.
    assert_eq!(qc.try_push(GateInstruction::H(1)), Ok(()));
    assert_eq!(
        qc.try_push(GateInstruction::Measure { qubit: 0, cbit: 1 }),
        Ok(())
    );
    assert_eq!(qc.try_push(GateInstruction::Barrier(vec![0])), Ok(()));
    assert_eq!(
        qc.try_push(GateInstruction::Cx(1, 0)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
    );
}

#[test]
fn try_push_rejects_any_unitary_after_measure_all() {
    let mut qc = ParameterizedCircuit::new(3);
    qc.try_push(GateInstruction::MeasureAll).unwrap();

    for q in 0..3 {
        assert_eq!(
            qc.try_push(GateInstruction::H(q)),
            Err(CircuitError::QubitAlreadyMeasured { qubit: q })
        );
    }
    // Two-qubit gates report the first offending operand, as before.
    assert_eq!(
        qc.try_push(GateInstruction::Cx(2, 1)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 2 })
    );
    // `MeasureAll` covers indices past the register too: the C-4 check runs
    // before the range check, so this is "already measured", not "out of range".
    assert_eq!(
        qc.try_push(GateInstruction::H(9)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 9 })
    );
    // Measurements and barriers still go through.
    assert_eq!(qc.try_push(GateInstruction::MeasureAll), Ok(()));
    assert_eq!(qc.try_push(GateInstruction::Barrier(vec![])), Ok(()));
    assert_eq!(qc.gates.len(), 3);
}

/// A rejected push must leave the circuit — including the incremental C-4
/// state — exactly as it was, so later legal pushes still behave.
#[test]
fn try_push_rejections_do_not_disturb_the_measured_state() {
    let mut qc = ParameterizedCircuit::new(3);
    qc.try_push(GateInstruction::H(0)).unwrap();
    qc.try_push(GateInstruction::Measure { qubit: 1, cbit: 0 })
        .unwrap();

    // A mix of rejections: C-4, out of range, identical qubits, non-finite angle.
    assert_eq!(
        qc.try_push(GateInstruction::Cx(0, 1)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 1 })
    );
    assert_eq!(
        qc.try_push(GateInstruction::H(7)),
        Err(CircuitError::QubitOutOfRange {
            qubit: 7,
            num_qubits: 3
        })
    );
    assert_eq!(
        qc.try_push(GateInstruction::Cx(2, 2)),
        Err(CircuitError::IdenticalQubits { qubit: 2 })
    );
    assert_eq!(
        qc.try_push(GateInstruction::Rz {
            qubit: 2,
            theta: GateParam::Fixed(f64::NAN),
        }),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(qc.gates.len(), 2);

    // Unmeasured qubits are still usable, and measuring one more is tracked.
    assert_eq!(qc.try_push(GateInstruction::Cx(0, 2)), Ok(()));
    assert_eq!(
        qc.try_push(GateInstruction::Measure { qubit: 2, cbit: 1 }),
        Ok(())
    );
    assert_eq!(
        qc.try_push(GateInstruction::Cx(0, 2)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 2 })
    );
    assert_eq!(qc.try_push(GateInstruction::H(0)), Ok(()));
}

/// The C-4 state is derived from `gates`, not assumed empty: a circuit whose
/// instruction list was assembled without the builder (hand-written, or produced
/// by the QASM importer) is still checked against its own measurements.
#[test]
fn try_push_derives_the_measured_state_from_existing_gates() {
    let mut hand_assembled = ParameterizedCircuit::new(2);
    hand_assembled.gates = vec![
        GateInstruction::H(0),
        GateInstruction::Measure { qubit: 0, cbit: 0 },
    ];
    assert_eq!(
        hand_assembled.try_push(GateInstruction::X(0)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
    );
    assert_eq!(hand_assembled.try_push(GateInstruction::X(1)), Ok(()));

    // Same via the importer, whose `finish` synthesises `MeasureAll`.
    let mut imported = ParameterizedCircuit::from_qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\nmeasure q -> c;\n",
    )
    .unwrap();
    assert_eq!(imported.gates.last(), Some(&GateInstruction::MeasureAll));
    assert_eq!(
        imported.try_push(GateInstruction::H(1)),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 1 })
    );
}

#[test]
fn test_parameterized_circuit_push_basic() {
    let qc = ParameterizedCircuit::new(2).push(GateInstruction::H(1));

    assert_eq!(qc.gates.len(), 1);
    assert_eq!(qc.gates[0], GateInstruction::H(1));
}

#[test]
#[should_panic]
fn test_parameterized_circuit_push_out_of_range() {
    let _qc = ParameterizedCircuit::new(2).push(GateInstruction::H(5));
}

#[test]
#[should_panic]
fn test_parameterized_circuit_push_same_qubits() {
    let _qc = ParameterizedCircuit::new(2).push(GateInstruction::Cx(0, 0));
}

/// Mirror of `polypus-sim`'s `non_finite_angle_is_rejected`
/// (crates/polypus-sim/tests/gate_matrices.rs): the circuit layer rejects a
/// non-finite fixed angle with its own error type, matching the simulator's
/// reference behaviour (contract C-2).
#[test]
fn non_finite_angle_is_rejected() {
    let mut qc = ParameterizedCircuit::new(1);
    let err = qc
        .try_push(GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Fixed(f64::NAN),
        })
        .unwrap_err();
    assert_eq!(err, CircuitError::NonFiniteParam);
}

#[test]
fn test_parameterized_circuit_try_push_rejects_non_finite() {
    let mut qc = ParameterizedCircuit::new(1);

    // Single-angle gate: infinity is rejected as well as NaN.
    assert_eq!(
        qc.try_push(GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Fixed(f64::INFINITY),
        }),
        Err(CircuitError::NonFiniteParam)
    );

    // Multi-angle U gate: a non-finite value in any of the three slots is
    // rejected (NaN in phi, then -inf in theta).
    assert_eq!(
        qc.try_push(GateInstruction::U {
            qubit: 0,
            theta: GateParam::Fixed(1.0),
            phi: GateParam::Fixed(f64::NAN),
            lam: GateParam::Fixed(0.5),
        }),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(
        qc.try_push(GateInstruction::U {
            qubit: 0,
            theta: GateParam::Fixed(f64::NEG_INFINITY),
            phi: GateParam::Fixed(1.0),
            lam: GateParam::Fixed(0.5),
        }),
        Err(CircuitError::NonFiniteParam)
    );

    // A finite `Param` reference is unaffected — construction-time validation
    // only inspects `Fixed` angles.
    assert_eq!(
        qc.try_push(GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Param(0),
        }),
        Ok(())
    );

    // Only the valid gate made it into the circuit.
    assert_eq!(qc.gates.len(), 1);
}

/// A rejected push must not leak partial state: even when the valid slots of
/// the rejected gate reference free parameters, `num_params` is unchanged.
#[test]
fn test_rejected_push_leaves_num_params_untouched() {
    let mut qc = ParameterizedCircuit::new(1);
    assert_eq!(
        qc.try_push(GateInstruction::U {
            qubit: 0,
            theta: GateParam::Param(4),
            phi: GateParam::Fixed(f64::NAN),
            lam: GateParam::Param(7),
        }),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(qc.num_params, 0);
    assert!(qc.gates.is_empty());

    // Same for a structural rejection (out-of-range qubit) of a gate whose
    // angle is a free parameter.
    assert_eq!(
        qc.try_push(GateInstruction::Rzz {
            q0: 0,
            q1: 3,
            theta: GateParam::Param(2),
        }),
        Err(CircuitError::QubitOutOfRange {
            qubit: 3,
            num_qubits: 1
        })
    );
    assert_eq!(qc.num_params, 0);
    assert!(qc.gates.is_empty());
}

/// Structural checks run in a fixed order for every arity: range before
/// distinctness, each reporting the first offending operand in operand order.
#[test]
fn test_try_push_reports_range_before_repeated_qubits() {
    let mut qc = ParameterizedCircuit::new(2);
    // Both operands out of range (and equal): the range error wins.
    assert_eq!(
        qc.try_push(GateInstruction::Cx(5, 5)),
        Err(CircuitError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 2
        })
    );
    // In range but repeated.
    assert_eq!(
        qc.try_push(GateInstruction::Swap(1, 1)),
        Err(CircuitError::IdenticalQubits { qubit: 1 })
    );
    // A barrier may repeat a qubit; it is not a unitary.
    assert_eq!(qc.try_push(GateInstruction::Barrier(vec![1, 1])), Ok(()));
}

/// Regression for issue #38 (acceptance criterion): a non-finite angle must
/// never reach the QASM exporter as a literal `NaN`/`inf` string. Both a bound
/// free parameter and a hand-assembled fixed angle are rejected at export.
#[test]
fn test_export_rejects_non_finite_angle() {
    // Route 1: a free parameter bound to a non-finite value.
    let parameterized = ParameterizedCircuit::new(1).rx(0, Param(0));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            parameterized.to_qasm2_with_params(&[bad]),
            Err(CircuitError::NonFiniteParam)
        );
    }

    // Route 2: a circuit assembled by hand with a fixed non-finite angle,
    // bypassing the builder's construction-time guard — the exporter itself
    // still refuses it, so no `NaN`/`inf` literal is ever serialised.
    let mut hand_assembled = ParameterizedCircuit::new(1);
    hand_assembled.gates = vec![GateInstruction::Rx {
        qubit: 0,
        theta: GateParam::Fixed(f64::INFINITY),
    }];
    assert_eq!(
        hand_assembled.to_qasm2_with_params(&[]),
        Err(CircuitError::NonFiniteParam)
    );
}

// ConcreteCircuit
#[test]
fn test_concrete_circuit_num_clbits_no_measurements() {
    let qc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![],
    };

    assert_eq!(qc.num_clbits(), 0);
}

#[test]
fn test_concrete_circuit_num_clbits_single_measurement() {
    let qc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(1.0),
            },
            GateInstruction::Measure { qubit: 0, cbit: 0 },
        ],
    };

    assert_eq!(qc.num_clbits(), 1);
}

#[test]
fn test_concrete_circuit_num_clbits_multiple_measurements() {
    let qc = ConcreteCircuit {
        num_qubits: 5,
        gates: vec![
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(1.0),
            },
            GateInstruction::Measure { qubit: 0, cbit: 2 },
            GateInstruction::Measure { qubit: 4, cbit: 5 },
        ],
    };

    assert_eq!(qc.num_clbits(), 6);
}

#[test]
fn test_concrete_circuit_num_clbits_measure_all() {
    let qc = ConcreteCircuit {
        num_qubits: 5,
        gates: vec![
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(1.0),
            },
            GateInstruction::MeasureAll,
        ],
    };

    assert_eq!(qc.num_clbits(), 5);
}

#[test]
fn test_concrete_circuit_num_clbits_measure_and_measure_all() {
    let qc = ConcreteCircuit {
        num_qubits: 3,
        gates: vec![
            GateInstruction::Measure { qubit: 0, cbit: 10 },
            GateInstruction::MeasureAll,
        ],
    };

    assert_eq!(qc.num_clbits(), 11);
}

#[test]
fn test_concrete_circuit_to_qasm2_basic() {
    let qc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![GateInstruction::H(0)],
    };
    let qasm = qc.to_qasm2();

    assert!(qasm.starts_with("OPENQASM 2.0;"));
    assert!(qasm.contains("qreg q[2];"));
    assert!(qasm.contains("h q[0];"));
    assert!(!qasm.contains("creg"));
}

#[test]
fn test_concrete_circuit_to_qasm2_multiple_gates_measure() {
    let qc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![
            GateInstruction::H(0),
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(0.5),
            },
            GateInstruction::Ry {
                qubit: 1,
                theta: GateParam::Fixed(1.0),
            },
            GateInstruction::MeasureAll,
        ],
    };
    let qasm = qc.to_qasm2();

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
#[should_panic]
fn test_concrete_circuit_to_qasm2_panics_on_param() {
    let qc = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Param(0),
        }],
    };

    let _ = qc.to_qasm2();
}

// ── Tier-1 qelib1.inc gates: builder, validation and parameter binding ──────

#[test]
fn test_qelib1_builder_methods_push_one_instruction_each() {
    use GateInstruction as G;
    use GateParam::Fixed;
    let qc = ParameterizedCircuit::new(3)
        .sx(0)
        .sxdg(1)
        .cy(2, 0)
        .ch(1, 2)
        .csx(0, 1)
        .ccx(2, 1, 0)
        .cswap(0, 2, 1)
        .crx(1, 0, 0.1)
        .cry(2, 1, 0.2)
        .crz(0, 2, 0.3)
        .cu1(1, 2, 0.4)
        .cu3(2, 0, 0.5, 0.6, 0.7)
        .cu(0, 1, 0.8, 0.9, 1.0, 1.1);
    assert_eq!(
        qc.gates,
        [
            G::Sx(0),
            G::Sxdg(1),
            G::Cy(2, 0),
            G::Ch(1, 2),
            G::Csx(0, 1),
            G::Ccx(2, 1, 0),
            G::Cswap(0, 2, 1),
            G::Crx {
                control: 1,
                target: 0,
                theta: Fixed(0.1)
            },
            G::Cry {
                control: 2,
                target: 1,
                theta: Fixed(0.2)
            },
            G::Crz {
                control: 0,
                target: 2,
                theta: Fixed(0.3)
            },
            G::Cu1 {
                q0: 1,
                q1: 2,
                theta: Fixed(0.4)
            },
            G::Cu3 {
                control: 2,
                target: 0,
                theta: Fixed(0.5),
                phi: Fixed(0.6),
                lam: Fixed(0.7)
            },
            G::Cu {
                control: 0,
                target: 1,
                theta: Fixed(0.8),
                phi: Fixed(0.9),
                lam: Fixed(1.0),
                gamma: Fixed(1.1)
            },
        ]
    );
    assert_eq!(qc.num_params, 0);
}

#[test]
fn test_qelib1_param_tracking_covers_every_angle_slot() {
    // The highest index anywhere in a gate — including `cu`'s fourth slot —
    // sizes `num_params`.
    assert_eq!(
        ParameterizedCircuit::new(2).crx(0, 1, Param(3)).num_params,
        4
    );
    assert_eq!(
        ParameterizedCircuit::new(2)
            .cu3(0, 1, 0.1, Param(1), 0.2)
            .num_params,
        2
    );
    assert_eq!(
        ParameterizedCircuit::new(2)
            .cu(0, 1, 0.1, 0.2, 0.3, Param(6))
            .num_params,
        7
    );
}

#[test]
fn test_try_push_validates_three_qubit_gates() {
    let mut qc = ParameterizedCircuit::new(3);
    assert_eq!(
        qc.try_push(GateInstruction::Ccx(0, 1, 5)),
        Err(CircuitError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 3
        })
    );
    assert_eq!(
        qc.try_push(GateInstruction::Ccx(0, 2, 0)),
        Err(CircuitError::IdenticalQubits { qubit: 0 })
    );
    assert_eq!(
        qc.try_push(GateInstruction::Cswap(1, 2, 2)),
        Err(CircuitError::IdenticalQubits { qubit: 2 })
    );
    assert_eq!(
        qc.try_push(GateInstruction::Cu {
            control: 0,
            target: 1,
            theta: GateParam::Fixed(0.1),
            phi: GateParam::Fixed(0.2),
            lam: GateParam::Fixed(0.3),
            gamma: GateParam::Fixed(f64::NAN),
        }),
        Err(CircuitError::NonFiniteParam)
    );
    assert!(qc.gates.is_empty());
    assert_eq!(qc.try_push(GateInstruction::Cswap(2, 0, 1)), Ok(()));
}

/// Parameter binding for every parameterised Tier-1 gate: binding free
/// parameters yields exactly the circuit built with those angles fixed, both
/// as instructions and as exported QASM. This is the binding path the
/// orchestration benchmark measures.
#[test]
fn test_qelib1_bound_circuit_matches_fixed_angle_circuit() {
    let values = [0.3, -1.2, 2.5, 0.7];
    let parameterised = ParameterizedCircuit::new(3)
        .crx(0, 1, Param(0))
        .cry(2, 0, Param(1))
        .crz(1, 2, Param(2))
        .cu1(2, 1, Param(3))
        .cu3(0, 2, Param(2), Param(0), Param(1))
        .cu(1, 0, Param(3), Param(2), Param(1), Param(0));
    let fixed = ParameterizedCircuit::new(3)
        .crx(0, 1, values[0])
        .cry(2, 0, values[1])
        .crz(1, 2, values[2])
        .cu1(2, 1, values[3])
        .cu3(0, 2, values[2], values[0], values[1])
        .cu(1, 0, values[3], values[2], values[1], values[0]);
    assert_eq!(parameterised.num_params, 4);

    let bound = parameterised.assign_parameters(&values).unwrap();
    assert_eq!(bound.gates, fixed.gates);
    assert_eq!(
        parameterised.to_qasm2_with_params(&values).unwrap(),
        fixed.to_qasm2_with_params(&[]).unwrap()
    );
    assert_eq!(bound.to_qasm2(), fixed.to_qasm2_with_params(&[]).unwrap());
}
