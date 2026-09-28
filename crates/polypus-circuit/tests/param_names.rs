//! Parameter names: every free parameter has one, given or by default, and
//! the names are part of the circuit.

use polypus_circuit::{GateInstruction, Param, ParamExpr, ParameterizedCircuit};

fn names(qc: &ParameterizedCircuit) -> Vec<String> {
    qc.param_names()
}

/// Every index `0..num_params` has a name, including the ones no gate uses,
/// whether the highest index comes from a `Param` or from an expression.
#[test]
fn default_names_cover_every_index() {
    assert!(names(&ParameterizedCircuit::new(1)).is_empty());
    let qc = ParameterizedCircuit::new(1).rx(0, Param(7));
    assert_eq!(
        names(&qc),
        ["theta_0", "theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6", "theta_7"]
    );
    let mut qc = ParameterizedCircuit::new(1);
    let angle = qc.add_expr(ParamExpr::param(3) * 2.0).unwrap();
    let qc = qc.rz(0, angle);
    assert_eq!(names(&qc), ["theta_0", "theta_1", "theta_2", "theta_3"]);
}

/// A default name that a declared gate of the circuit already uses — called
/// directly or only through another declaration — becomes the first free
/// `theta_<index>_<k>`.
#[test]
fn default_names_avoid_declared_gate_names() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate theta_1 a { x a; }\n\
               gate theta_1_1 a { theta_1 a; }\n\
               gate wrap a { theta_1_1 a; }\n\
               qreg q[1];\nwrap q[0];\n";
    let imported = ParameterizedCircuit::from_qasm2(src).unwrap();
    let mut qc = ParameterizedCircuit::new(1).rx(0, Param(2));
    assert_eq!(names(&qc), ["theta_0", "theta_1", "theta_2"]);
    qc.try_push(imported.gates[0].clone()).unwrap();
    assert_eq!(names(&qc), ["theta_0", "theta_1_2", "theta_2"]);
}

/// Names are a function of the circuit: the same circuit built twice, or
/// cloned, has the same names.
#[test]
fn names_are_deterministic_and_cloned() {
    let build = || {
        let mut qc = ParameterizedCircuit::new(2);
        let angle = qc
            .add_expr(ParamExpr::param(0) - ParamExpr::param(4))
            .unwrap();
        qc.rzz(0, 1, angle).rx(1, Param(2))
    };
    assert_eq!(names(&build()), names(&build()));
    assert_eq!(names(&build().clone()), names(&build()));
    assert_eq!(build(), build().clone());
}

/// Names belong to the circuit: a gate pushed from another circuit refers to
/// its parameter by index and takes the receiving circuit's name for it.
#[test]
fn pushed_gates_take_the_receiving_circuits_names() {
    let other = ParameterizedCircuit::new(1).ry(0, Param(0)).rz(0, Param(2));
    let mut qc = ParameterizedCircuit::new(1);
    qc.try_push(other.gates[1].clone()).unwrap();
    assert_eq!(names(&qc), names(&other));
    assert!(matches!(
        qc.gates[0],
        GateInstruction::Rz {
            theta: Param(2),
            ..
        }
    ));
}
