//! Integration tests for polypus-circuit: composing one circuit onto chosen
//! qubits of another (`ParameterizedCircuit::try_compose` / `compose`).

use polypus_circuit::templates::qft;
use polypus_circuit::{
    CircuitError, GateInstruction as G, GateParam, Param, ParamExpr, ParameterizedCircuit as Pc,
};

#[test]
fn compose_maps_qubits_in_order() {
    let qc = Pc::new(4).x(0).compose(&Pc::new(2).h(0).cx(0, 1), &[3, 1]);
    assert_eq!(qc.gates(), [G::X(0), G::H(3), G::Cx(3, 1)]);
}

#[test]
fn compose_places_a_template_on_a_subregister() {
    let qc = Pc::new(5).compose(&qft(3).unwrap(), &[1, 2, 3]);
    assert_eq!(qc.gates().len(), 7);
    assert_eq!(qc.gates()[0], G::H(3));
    assert_eq!(qc.gates()[6], G::Swap(1, 3));
}

#[test]
fn compose_expands_measure_all_on_a_subset() {
    let qc = Pc::new(3).compose(&Pc::new(2).h(0).measure_all(), &[2, 0]);
    assert_eq!(
        qc.gates(),
        [
            G::H(2),
            G::Measure { qubit: 2, cbit: 0 },
            G::Measure { qubit: 0, cbit: 1 }
        ]
    );
}

#[test]
fn compose_with_the_identity_map_keeps_measure_all_and_full_barriers() {
    let sub = Pc::new(2).h(0).barrier().measure_all();
    assert_eq!(Pc::new(2).compose(&sub, &[0, 1]).gates(), sub.gates());
    assert_eq!(
        Pc::new(3).compose(&sub, &[0, 1]).gates()[1],
        G::Barrier(vec![0, 1])
    );
}

#[test]
fn compose_rejects_a_gate_on_a_measured_target() {
    let mut qc = Pc::new(2).measure(1, 0);
    assert_eq!(
        qc.try_compose(&Pc::new(1).x(0), &[1]),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 1 })
    );
}

#[test]
fn failed_compose_leaves_the_circuit_unchanged() {
    let mut qc = Pc::new(3).h(0).measure(2, 0);
    let before = qc.clone();
    assert!(qc
        .try_compose(&Pc::new(2).rx(0, Param(4)).x(1), &[0, 2])
        .is_err());
    assert_eq!(qc, before); // no gates, no num_params, no names
}

#[test]
fn compose_validates_the_map() {
    let sub = Pc::new(2).cx(0, 1);
    let mut qc = Pc::new(3);
    assert_eq!(
        qc.try_compose(&sub, &[0]),
        Err(CircuitError::InvalidQubitMap {
            reason: "expected 2 target qubits, got 1".into()
        })
    );
    assert_eq!(
        qc.try_compose(&sub, &[0, 3]),
        Err(CircuitError::QubitOutOfRange {
            qubit: 3,
            num_qubits: 3
        })
    );
    assert_eq!(
        qc.try_compose(&sub, &[1, 1]),
        Err(CircuitError::InvalidQubitMap {
            reason: "qubit 1 is targeted twice".into()
        })
    );
}

#[test]
fn compose_shares_parameters_by_index() {
    let qc = Pc::new(2)
        .ry(1, Param(0))
        .compose(&Pc::new(1).rx(0, Param(1)), &[0]);
    assert_eq!(qc.num_params(), 2);
    let bound = qc.assign_parameters(&[0.1, 0.2]).unwrap();
    assert_eq!(
        bound.gates()[1],
        G::Rx {
            qubit: 0,
            theta: GateParam::Fixed(0.2)
        }
    );
}

#[test]
fn compose_copies_expressions_into_the_target() {
    let mut sub = Pc::new(1);
    let theta = sub.add_expr(ParamExpr::param(0) * 2.0).unwrap();
    sub.try_push(G::Rx { qubit: 0, theta }).unwrap();
    let bound = Pc::new(2)
        .compose(&sub, &[1])
        .assign_parameters(&[0.25])
        .unwrap();
    assert_eq!(
        bound.gates(),
        [G::Rx {
            qubit: 1,
            theta: GateParam::Fixed(0.5)
        }]
    );
}

#[test]
fn compose_rejects_conflicting_parameter_names() {
    let header = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
    let ours = Pc::from_qasm3(&format!(
        "{header}input float gamma;\nqubit[1] q;\nrx(gamma) q[0];\n"
    ))
    .unwrap();
    let theirs = Pc::from_qasm3(&format!(
        "{header}input float beta;\nqubit[1] q;\nrx(beta) q[0];\n"
    ))
    .unwrap();
    let mut qc = ours.clone();
    assert_eq!(
        qc.try_compose(&theirs, &[0]),
        Err(CircuitError::ConflictingParamNames {
            index: 0,
            name: "beta".into()
        })
    );
    assert_eq!(qc, ours);
}

#[test]
fn compose_adopts_the_parameters_other_declares() {
    let header = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
    let theirs = Pc::from_qasm3(&format!(
        "{header}input float gamma;\ninput float beta;\nqubit[1] q;\nrx(gamma) q[0];\n"
    ))
    .unwrap();
    let qc = Pc::new(2).compose(&theirs, &[1]);
    assert_eq!(qc.num_params(), 2);
    assert_eq!(qc.param_names(), ["gamma", "beta"]);
}

#[test]
fn compose_keeps_the_wider_declared_classical_register() {
    let theirs = Pc::from_qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[3];\nmeasure q[0] -> c[0];\n",
    )
    .unwrap();
    assert_eq!(Pc::new(2).compose(&theirs, &[1]).clbits_width(), 3);
}

#[test]
fn compose_rejects_a_circuit_that_refers_outside_its_register() {
    let unchecked = Pc::from_parts_unchecked(1, 0, vec![G::X(3)]);
    assert_eq!(
        Pc::new(2).try_compose(&unchecked, &[0]),
        Err(CircuitError::QubitOutOfRange {
            qubit: 3,
            num_qubits: 1
        })
    );
}
