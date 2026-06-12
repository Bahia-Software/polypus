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

