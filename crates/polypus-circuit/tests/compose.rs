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
fn compose_rejects_a_gate_on_a_qubit_other_measured_before_it() {
    // Only a circuit assembled unchecked acts on a qubit it measured.
    let measured_then_x = |first| Pc::from_parts_unchecked(1, 0, vec![first, G::X(0)]);
    // A measurement, and a MeasureAll expanded onto a subregister.
    for first in [G::Measure { qubit: 0, cbit: 0 }, G::MeasureAll] {
        let mut qc = Pc::new(2);
        assert_eq!(
            qc.try_compose(&measured_then_x(first), &[1]),
            Err(CircuitError::QubitAlreadyMeasured { qubit: 1 })
        );
        assert_eq!(qc, Pc::new(2));
        // Nor is qubit 1 left marked as measured.
        assert_eq!(qc.try_push(G::X(1)), Ok(()));
    }
    // A MeasureAll kept over the whole register.
    assert_eq!(
        Pc::new(1).try_compose(&measured_then_x(G::MeasureAll), &[0]),
        Err(CircuitError::QubitAlreadyMeasured { qubit: 0 })
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
    assert_eq!(
        qc.try_compose(&sub, &[1, 1]).unwrap_err().to_string(),
        "invalid qubit map for compose: qubit 1 is targeted twice"
    );
}

#[test]
#[should_panic(expected = "invalid qubit map for compose: expected 2 target qubits, got 1")]
fn compose_panics_where_try_compose_fails() {
    let _ = Pc::new(3).compose(&Pc::new(2), &[0]);
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
fn compose_copies_an_expression_next_to_other_angles() {
    let mut sub = Pc::new(1);
    let theta = sub.add_expr(ParamExpr::param(0) * 2.0).unwrap();
    sub.try_push(G::U {
        qubit: 0,
        theta,
        phi: GateParam::Fixed(0.5),
        lam: GateParam::Param(0),
    })
    .unwrap();
    let bound = Pc::new(2)
        .compose(&sub, &[1])
        .assign_parameters(&[0.25])
        .unwrap();
    assert_eq!(
        bound.gates(),
        [G::U {
            qubit: 1,
            theta: GateParam::Fixed(0.5),
            phi: GateParam::Fixed(0.5),
            lam: GateParam::Fixed(0.25)
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
    let err = qc.try_compose(&theirs, &[0]).unwrap_err();
    assert_eq!(
        err,
        CircuitError::ConflictingParamNames {
            index: 0,
            name: "beta".into()
        }
    );
    assert_eq!(
        err.to_string(),
        "cannot compose: the composed circuit names parameter 0 'beta', which the target \
         circuit names differently or gives to another parameter"
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

/// One instruction of every kind the OpenQASM 2.0 importer builds, on the five
/// distinct qubits `q`, ending with the measurement (contract C-4).
fn every_instruction(q: [usize; 5]) -> String {
    let [a, b, c, d, e] = q.map(|i| format!("q[{i}]"));
    format!(
        "h {a}; x {a}; y {a}; z {a}; s {a}; t {a}; sdg {a}; tdg {a}; id {a};\n\
         sx {a}; sxdg {a}; rx(0.1) {a}; ry(0.2) {a}; rz(0.3) {a}; p(0.4) {a};\n\
         u1(0.5) {a}; u2(0.1, 0.2) {a}; u3(0.1, 0.2, 0.3) {a};\n\
         u(0.1, 0.2, 0.3) {a}; U(0.1, 0.2, 0.3) {a}; u0(0.1) {a};\n\
         cx {a}, {b}; CX {b}, {a}; cz {a}, {b}; swap {a}, {b}; cy {a}, {b};\n\
         ch {a}, {b}; csx {a}, {b}; rzz(0.1) {a}, {b}; rxx(0.2) {a}, {b};\n\
         cp(0.3) {a}, {b}; cu1(0.4) {a}, {b}; crx(0.1) {a}, {b};\n\
         cry(0.2) {a}, {b}; crz(0.3) {b}, {a}; cu3(0.1, 0.2, 0.3) {a}, {b};\n\
         cu(0.1, 0.2, 0.3, 0.4) {a}, {b}; ccx {a}, {b}, {c};\n\
         cswap {c}, {a}, {b}; rccx {a}, {b}, {c}; rc3x {a}, {b}, {c}, {d};\n\
         c3x {d}, {c}, {b}, {a}; c3sqrtx {a}, {b}, {c}, {d};\n\
         c4x {a}, {b}, {c}, {d}, {e}; pair {b}, {e}; barrier {a}, {c};\n\
         measure {a} -> c[0];\n"
    )
}

#[test]
fn compose_maps_every_operand_of_every_instruction() {
    let program = |num_qubits: usize, q: [usize; 5]| {
        format!(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
             gate pair x, y {{ cx x, y; h y; }}\n\
             qreg q[{num_qubits}];\ncreg c[1];\n{}",
            every_instruction(q)
        )
    };
    let map = [6, 2, 4, 0, 5];
    let sub = Pc::from_qasm2(&program(5, [0, 1, 2, 3, 4])).unwrap();
    assert!(sub.gates().iter().any(|g| matches!(g, G::Custom(_))));
    let placed = Pc::from_qasm2(&program(7, map)).unwrap();
    assert_eq!(Pc::new(7).compose(&sub, &map).gates(), placed.gates());
}
