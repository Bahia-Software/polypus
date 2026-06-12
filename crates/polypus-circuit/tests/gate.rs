//! Integration tests for polypus-circuit: gates
// GateParam

use polypus_circuit::{ GateInstruction, GateParam };

#[test]
fn test_gateparam_creation(){
    let var_fixe = GateParam::Fixed(3.14);
    let var_param = GateParam::Param(42);

    assert_eq!(var_fixe, GateParam::Fixed(3.14));
    assert_eq!(var_param, GateParam::Param(42));
}

#[test]
fn test_gateparam_from_f64(){
    let param_from_from = GateParam::from(2.72);
    let param_from_into: GateParam = 2.72.into();

    assert_eq!(param_from_from, GateParam::Fixed(2.72));
    assert_eq!(param_from_into, GateParam::Fixed(2.72));
}

#[test]
fn test_gateparam_from_f64_limit_cases(){
    let param_cero = GateParam::from(0.0);
    let param_negativo = GateParam::from(-1.0);
    let param_grande = GateParam::from(1e10);
    let param_nan = GateParam::from(f64::NAN);
    let param_inf = GateParam::from(f64::INFINITY);


    assert_eq!(param_cero, GateParam::Fixed(0.0));
    assert_eq!(param_negativo, GateParam::Fixed(-1.0));
    assert_eq!(param_grande, GateParam::Fixed(1e10));
    match param_nan {
        GateParam::Fixed(v) => assert!(v.is_nan()),
        _ => panic!("Expected Fixed(NaN)"),
    }
    assert_eq!(param_inf, GateParam::Fixed(f64::INFINITY));
}

#[test]
fn test_copy() {
    let p1 = GateParam::Param(7);
    let p2 = p1;

    assert_eq!(p1, p2);
}

// GateInstruction

#[test]
fn test_single_qubit_gates_creation() {
    let h = GateInstruction::H(0);
    let x = GateInstruction::X(1);
    let y = GateInstruction::Y(2);
    let z = GateInstruction::Z(3);

    assert_eq!(h, GateInstruction::H(0));
    assert_eq!(x, GateInstruction::X(1));
    assert_eq!(y, GateInstruction::Y(2));
    assert_eq!(z, GateInstruction::Z(3));
}

#[test]
fn test_rotation_gates_creation() {
    let rx = GateInstruction::Rx {
        qubit: 0,
        theta: GateParam::Fixed(1.0),
    };

    match rx {
        GateInstruction::Rx { qubit, theta } => {
            assert_eq!(qubit, 0);
            assert_eq!(theta, GateParam::Fixed(1.0));
        }
        _ => panic!("Expected Rx"),
    }
}

#[test]
fn test_two_qubit_gates() {
    let cx = GateInstruction::Cx(0, 1);
    let cz = GateInstruction::Cz(2, 3);

    assert_eq!(cx, GateInstruction::Cx(0, 1));
    assert_eq!(cz, GateInstruction::Cz(2, 3));
}

#[test]
fn test_u_gate() {
    let u = GateInstruction::U {
        qubit: 0,
        theta: GateParam::Fixed(1.0),
        phi: GateParam::Fixed(2.0),
        lam: GateParam::Fixed(3.0),
    };

    match u {
        GateInstruction::U { qubit, theta, phi, lam } => {
            assert_eq!(qubit, 0);
            assert_eq!(theta, GateParam::Fixed(1.0));
            assert_eq!(phi, GateParam::Fixed(2.0));
            assert_eq!(lam, GateParam::Fixed(3.0));
        }
        _ => panic!("Expected U gate"),
    }
}