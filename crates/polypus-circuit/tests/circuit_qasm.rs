//! Integration tests for polypus-circuit: builder, parameter binding and
//! OpenQASM 2.0 round-trips.

use polypus_circuit::{
    CircuitError, ConcreteCircuit, GateInstruction, GateParam, Param, ParameterizedCircuit,
};

// ─────────────────────────────────────────────────────────────────────────────
// QASM round-trip: build circuit → to_qasm2 → verify exact string
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fixed_circuit_round_trip_exact_string() {
    let qasm = ParameterizedCircuit::new(2)
        .h(0)
        .cx(0, 1)
        .rz(1, 0.25)
        .measure(0, 0)
        .measure(1, 1)
        .to_qasm2_with_params(&[])
        .unwrap();

    let expected = "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
rz(0.250000000000) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
";
    assert_eq!(qasm, expected);
}

#[test]
fn all_gate_variants_serialize_with_qelib1_names() {
    let qasm = ParameterizedCircuit::new(3)
        .h(0)
        .x(1)
        .y(2)
        .z(0)
        .s(1)
        .t(2)
        .sdg(0)
        .tdg(1)
        .rx(0, 0.1)
        .ry(1, 0.2)
        .rz(2, 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .rzz(0, 1, 0.4)
        .rxx(1, 2, 0.5)
        .u(0, 0.6, 0.7, 0.8)
        .barrier()
        .barrier_on(&[0, 2])
        .measure(0, 0)
        .to_qasm2_with_params(&[])
        .unwrap();

    let expected = "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[3];
creg c[1];
h q[0];
x q[1];
y q[2];
z q[0];
s q[1];
t q[2];
sdg q[0];
tdg q[1];
rx(0.100000000000) q[0];
ry(0.200000000000) q[1];
rz(0.300000000000) q[2];
cx q[0],q[1];
cz q[1],q[2];
rzz(0.400000000000) q[0],q[1];
rxx(0.500000000000) q[1],q[2];
u3(0.600000000000,0.700000000000,0.800000000000) q[0];
barrier q;
barrier q[0],q[2];
measure q[0] -> c[0];
";
    assert_eq!(qasm, expected);
}

#[test]
fn circuit_without_measurements_has_no_creg() {
    let qasm = ParameterizedCircuit::new(1)
        .h(0)
        .to_qasm2_with_params(&[])
        .unwrap();
    assert!(!qasm.contains("creg"));
    assert!(qasm.contains("qreg q[1];"));
}

// ─────────────────────────────────────────────────────────────────────────────
// assign_parameters: values correctly substituted
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn assign_parameters_substitutes_values() {
    let qc = ParameterizedCircuit::new(2)
        .rx(0, Param(0))
        .ry(1, Param(1))
        .rz(0, 0.5); // fixed angle must survive binding untouched

    assert_eq!(qc.num_params, 2);

    let bound = qc.assign_parameters(&[1.0, 2.0]).unwrap();
    assert_eq!(bound.num_qubits, 2);
    assert_eq!(
        bound.gates,
        vec![
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(1.0)
            },
            GateInstruction::Ry {
                qubit: 1,
                theta: GateParam::Fixed(2.0)
            },
            GateInstruction::Rz {
                qubit: 0,
                theta: GateParam::Fixed(0.5)
            },
        ]
    );
}

#[test]
fn assign_parameters_reuses_same_param_index() {
    let bound = ParameterizedCircuit::new(2)
        .rzz(0, 1, Param(0))
        .rx(0, Param(0))
        .rx(1, Param(0))
        .assign_parameters(&[0.7])
        .unwrap();

    let qasm = bound.to_qasm2();
    assert!(qasm.contains("rzz(0.700000000000) q[0],q[1];"));
    assert_eq!(qasm.matches("0.700000000000").count(), 3);
}

#[test]
fn to_qasm2_with_params_matches_assign_then_serialize() {
    let qc = ParameterizedCircuit::new(2)
        .h(0)
        .rzz(0, 1, Param(0))
        .rx(0, Param(1))
        .measure_all();

    let direct = qc.to_qasm2_with_params(&[0.3, 0.9]).unwrap();
    let two_step = qc.assign_parameters(&[0.3, 0.9]).unwrap().to_qasm2();
    assert_eq!(direct, two_step);
}

#[test]
fn u_gate_binds_all_three_angles() {
    let qasm = ParameterizedCircuit::new(1)
        .u(0, Param(0), 0.5, Param(1))
        .to_qasm2_with_params(&[0.1, 0.9])
        .unwrap();
    assert!(qasm.contains("u3(0.100000000000,0.500000000000,0.900000000000) q[0];"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn wrong_number_of_params_is_rejected() {
    let qc = ParameterizedCircuit::new(1).rx(0, Param(0)).ry(0, Param(1));
    assert_eq!(qc.num_params, 2);

    assert_eq!(
        qc.assign_parameters(&[0.1]),
        Err(CircuitError::WrongNumberOfParams {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(
        qc.to_qasm2_with_params(&[0.1, 0.2, 0.3]).unwrap_err(),
        CircuitError::WrongNumberOfParams {
            expected: 2,
            got: 3
        }
    );
}

#[test]
fn param_index_out_of_bounds_is_rejected() {
    // Assemble manually so num_params and gate indices disagree.
    let qc = ParameterizedCircuit {
        num_qubits: 1,
        num_params: 1,
        gates: vec![GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Param(5),
        }],
    };
    assert_eq!(
        qc.assign_parameters(&[0.1]),
        Err(CircuitError::ParamIndexOutOfBounds {
            index: 5,
            num_params: 1
        })
    );
}

#[test]
fn errors_display_human_readable_messages() {
    let e = CircuitError::WrongNumberOfParams {
        expected: 2,
        got: 1,
    };
    assert!(e.to_string().contains("2"));
    let e = CircuitError::ParamIndexOutOfBounds {
        index: 5,
        num_params: 1,
    };
    assert!(e.to_string().contains("5"));
}

#[test]
#[should_panic(expected = "qubit index 3 out of range")]
fn builder_panics_on_qubit_out_of_range() {
    let _ = ParameterizedCircuit::new(3).h(3);
}

#[test]
#[should_panic(expected = "distinct qubits")]
fn builder_panics_on_identical_two_qubit_operands() {
    let _ = ParameterizedCircuit::new(2).cx(1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement / classical register handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn measure_all_emits_register_level_measure() {
    let qasm = ParameterizedCircuit::new(2)
        .h(0)
        .measure_all()
        .to_qasm2_with_params(&[])
        .unwrap();
    assert!(qasm.contains("creg c[2];"));
    assert!(qasm.ends_with("measure q -> c;\n"));
}

#[test]
fn measure_all_expands_when_register_sizes_differ() {
    // An explicit measure into c[4] grows creg beyond num_qubits, so the
    // register-level `measure q -> c;` shorthand would be invalid QASM.
    let qasm = ParameterizedCircuit::new(2)
        .measure(0, 4)
        .measure_all()
        .to_qasm2_with_params(&[])
        .unwrap();
    assert!(qasm.contains("creg c[5];"));
    assert!(qasm.contains("measure q[0] -> c[4];"));
    assert!(qasm.contains("measure q[0] -> c[0];"));
    assert!(qasm.contains("measure q[1] -> c[1];"));
    assert!(!qasm.contains("measure q -> c;"));
}

#[test]
fn num_clbits_reflects_measurements() {
    assert_eq!(ParameterizedCircuit::new(3).num_clbits(), 0);
    assert_eq!(ParameterizedCircuit::new(3).measure(0, 1).num_clbits(), 2);
    assert_eq!(ParameterizedCircuit::new(3).measure_all().num_clbits(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// ConcreteCircuit invariants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "unbound Param")]
fn concrete_circuit_panics_on_unbound_param() {
    let qc = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Param(0),
        }],
    };
    let _ = qc.to_qasm2();
}

#[test]
#[should_panic(expected = "non-finite")]
fn concrete_circuit_panics_on_non_finite_fixed_angle() {
    // A hand-assembled ConcreteCircuit bypasses the builder's validation; the
    // exporter still refuses a non-finite fixed angle rather than serialising it.
    let qc = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Fixed(f64::NAN),
        }],
    };
    let _ = qc.to_qasm2();
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-world case: QAOA MaxCut on 4 qubits (ring graph), p = 1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn qaoa_maxcut_4_qubits() {
    let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
    let (gamma, beta) = (0.4, 0.8);

    let mut qc = ParameterizedCircuit::new(4);
    for q in 0..4 {
        qc = qc.h(q);
    }
    for &(a, b) in &edges {
        qc = qc.rzz(a, b, Param(0)); // cost layer: 2*gamma per edge convention left to caller
    }
    for q in 0..4 {
        qc = qc.rx(q, Param(1)); // mixer layer
    }
    let qc = qc.measure_all();

    assert_eq!(qc.num_qubits, 4);
    assert_eq!(qc.num_params, 2);
    assert_eq!(qc.num_clbits(), 4);

    let qasm = qc.to_qasm2_with_params(&[gamma, beta]).unwrap();

    let expected = "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
rzz(0.400000000000) q[0],q[1];
rzz(0.400000000000) q[1],q[2];
rzz(0.400000000000) q[2],q[3];
rzz(0.400000000000) q[3],q[0];
rx(0.800000000000) q[0];
rx(0.800000000000) q[1];
rx(0.800000000000) q[2];
rx(0.800000000000) q[3];
measure q -> c;
";
    assert_eq!(qasm, expected);
}
