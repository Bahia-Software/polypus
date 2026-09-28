//! Angle expressions over a circuit's free parameters ([`ParamExpr`],
//! [`ParameterizedCircuit::add_expr`], [`GateParam::Expr`]): construction,
//! canonical forms, binding semantics, exports of bound circuits, identity,
//! and resource bounds.

use polypus_circuit::expr::{Constant, Function};
use polypus_circuit::{
    CircuitError, ConcreteCircuit, GateInstruction, GateParam, Param, ParamExpr,
    ParameterizedCircuit,
};
use std::f64::consts::{E, FRAC_PI_2, PI, TAU};

fn x(i: usize) -> ParamExpr {
    ParamExpr::param(i)
}

/// The angle of a one-parameter gate.
fn theta(gate: &GateInstruction) -> GateParam {
    match gate {
        GateInstruction::Rx { theta, .. }
        | GateInstruction::Ry { theta, .. }
        | GateInstruction::Rz { theta, .. }
        | GateInstruction::Rzz { theta, .. } => *theta,
        other => panic!("not a one-parameter gate: {other:?}"),
    }
}

/// `rz(expr) q[0]` bound to `params`: the angle binding computes.
fn bind(expr: ParamExpr, params: &[f64]) -> Result<f64, CircuitError> {
    let mut qc = ParameterizedCircuit::new(1);
    let angle = qc.add_expr(expr)?;
    let bound = qc.rz(0, angle).assign_parameters(params)?;
    match theta(&bound.gates[0]) {
        GateParam::Fixed(v) => Ok(v),
        other => panic!("binding left {other:?}"),
    }
}

// ───────────────────────────── Canonical forms ────────────────────────────

/// A bare parameter is a `Param`, a constant expression is evaluated to a
/// `Fixed` at once, and only an expression over free parameters is stored.
#[test]
fn add_expr_returns_the_canonical_angle() {
    let mut qc = ParameterizedCircuit::new(1);
    assert_eq!(qc.add_expr(x(3)), Ok(Param(3)));
    assert_eq!(
        qc.add_expr(ParamExpr::constant(Constant::Pi) / 2.0),
        Ok(GateParam::Fixed(FRAC_PI_2))
    );
    assert_eq!(
        qc.add_expr(ParamExpr::from(-0.0)).map(|p| match p {
            GateParam::Fixed(v) => v.is_sign_negative() && v == 0.0,
            _ => false,
        }),
        Ok(true)
    );
    assert!(matches!(qc.add_expr(2.0 * x(0)), Ok(GateParam::Expr(_))));
    // Storing an expression does not declare its parameters by itself...
    assert_eq!(qc.num_params, 0);
}

/// ... pushing a gate that uses it does, exactly as `Param(i)` does.
#[test]
fn expressions_count_their_parameters_when_pushed() {
    let mut qc = ParameterizedCircuit::new(2);
    let angle = qc.add_expr(x(1) * x(4)).unwrap();
    assert_eq!(qc.num_params, 0);
    let qc = qc.rzz(0, 1, angle);
    assert_eq!(qc.num_params, 5);
    // Parameters 0, 2 and 3 are unused but still counted: binding takes five.
    assert!(qc.assign_parameters(&[0.0; 5]).is_ok());
    assert_eq!(
        qc.assign_parameters(&[0.0; 4]),
        Err(CircuitError::WrongNumberOfParams {
            expected: 5,
            got: 4
        })
    );
}

// ────────────────────────────── Binding ───────────────────────────────────

/// Binding computes exactly what the expression spells, in its own order:
/// `(a + b) + c` and `a + (b + c)` are different expressions.
#[test]
fn binding_evaluates_exactly_the_written_operations() {
    let (a, b, c) = (1e16, -1e16, 1.0);
    assert_eq!(bind((x(0) + x(1)) + x(2), &[a, b, c]), Ok((a + b) + c));
    assert_eq!(bind(x(0) + (x(1) + x(2)), &[a, b, c]), Ok(a + (b + c)));
    assert_ne!((a + b) + c, a + (b + c));

    // The QED-C and Qiskit feature-map shapes.
    assert_eq!(bind(-x(0), &[0.3]), Ok(-0.3));
    assert_eq!(bind(2.0 * x(0), &[0.3]), Ok(2.0 * 0.3));
    let pi = || ParamExpr::constant(Constant::Pi);
    let zz = (-pi() + x(0)) * (-pi() + x(1)) * 2.0;
    assert_eq!(bind(zz, &[0.4, 1.3]), Ok((-PI + 0.4) * (-PI + 1.3) * 2.0));
    // `**` and the functions.
    assert_eq!(bind(x(0).pow(3.0), &[1.5]), Ok(1.5f64.powf(3.0)));
    assert_eq!(bind(x(0).pow(-x(1)), &[2.0, 3.0]), Ok(2f64.powf(-3.0)));
    for (function, reference) in [
        (Function::Sin, f64::sin as fn(f64) -> f64),
        (Function::Cos, f64::cos),
        (Function::Tan, f64::tan),
        (Function::Arcsin, f64::asin),
        (Function::Arccos, f64::acos),
        (Function::Arctan, f64::atan),
        (Function::Exp, f64::exp),
        (Function::Ln, f64::ln),
        (Function::Sqrt, f64::sqrt),
    ] {
        assert_eq!(
            bind(x(0).apply(function), &[0.25]),
            Ok(reference(0.25)),
            "{function:?}"
        );
    }
    for (constant, value) in [
        (Constant::Pi, PI),
        (Constant::Tau, TAU),
        (Constant::Euler, E),
    ] {
        assert_eq!(bind(x(0) * constant, &[1.0]), Ok(value));
        assert_eq!(constant.value(), value);
    }
}

/// Only the final value is an angle: it must be finite, and a division by
/// zero is an error of its own. A non-finite intermediate whose result is
/// finite is accepted (documented on `assign_parameters`).
#[test]
fn binding_rejects_unusable_angles() {
    assert_eq!(bind(1.0 / x(0), &[0.0]), Err(CircuitError::DivisionByZero));
    assert_eq!(bind(1.0 / x(0), &[-0.0]), Err(CircuitError::DivisionByZero));
    assert_eq!(
        bind(x(0).apply(Function::Exp), &[1000.0]),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(
        bind(x(0).apply(Function::Ln), &[-1.0]),
        Err(CircuitError::NonFiniteParam)
    );
    // exp(1000) is infinite; its reciprocal is 0.
    assert_eq!(bind(1.0 / x(0).apply(Function::Exp), &[1000.0]), Ok(0.0));
    // arctan(exp(1000)) = arctan(inf) = pi/2.
    assert_eq!(
        bind(x(0).apply(Function::Exp).apply(Function::Arctan), &[1000.0]),
        Ok(FRAC_PI_2)
    );
    // A non-finite value bound to a parameter is rejected, even where the
    // expression would map it to a finite angle.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(bind(1.0 / x(0), &[bad]), Err(CircuitError::NonFiniteParam));
    }
}

/// A non-finite number cannot enter an expression, and a constant expression
/// is checked when it is evaluated, at `add_expr`.
#[test]
fn add_expr_rejects_unusable_numbers_and_constants() {
    let mut qc = ParameterizedCircuit::new(1);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            qc.add_expr(x(0) + bad),
            Err(CircuitError::NonFiniteParam),
            "{bad}"
        );
    }
    assert_eq!(
        qc.add_expr(ParamExpr::from(1.0) / 0.0),
        Err(CircuitError::DivisionByZero)
    );
    assert_eq!(
        qc.add_expr(ParamExpr::from(1000.0).apply(Function::Exp)),
        Err(CircuitError::NonFiniteParam)
    );
}

// ──────────────────────── Belonging to a circuit ──────────────────────────

/// An expression id only means something in the circuit that issued it: any
/// other circuit rejects it — at push time and, for gates assembled by hand,
/// at binding — unless it holds an identical expression at the same position.
#[test]
fn expressions_belong_to_their_circuit() {
    let mut a = ParameterizedCircuit::new(1);
    let two_x = a.add_expr(2.0 * x(0)).unwrap();

    let mut b = ParameterizedCircuit::new(1);
    b.add_expr(3.0 * x(0)).unwrap();
    let rx = GateInstruction::Rx {
        qubit: 0,
        theta: two_x,
    };
    assert_eq!(b.try_push(rx.clone()), Err(CircuitError::UnknownExpression));
    assert!(b.gates.is_empty(), "a rejected gate is not pushed");
    assert_eq!(
        ParameterizedCircuit::new(1).try_push(rx.clone()),
        Err(CircuitError::UnknownExpression)
    );

    let mut hand = ParameterizedCircuit::new(1);
    hand.gates = vec![rx.clone()];
    hand.num_params = 1;
    assert_eq!(
        hand.assign_parameters(&[0.5]).err(),
        Some(CircuitError::UnknownExpression)
    );
    assert_eq!(
        hand.to_qasm2_with_params(&[0.5]),
        Err(CircuitError::UnknownExpression)
    );
    assert_eq!(
        hand.to_qir_with_params(&[0.5]),
        Err(CircuitError::UnknownExpression)
    );

    // Identical content at the same position is the same expression.
    let mut same = ParameterizedCircuit::new(1);
    same.add_expr(2.0 * x(0)).unwrap();
    same.try_push(rx.clone()).unwrap();
    assert_eq!(
        same.assign_parameters(&[0.5]).unwrap().gates,
        [GateInstruction::Rx {
            qubit: 0,
            theta: GateParam::Fixed(1.0)
        }]
    );

    // A clone holds the same expressions.
    a.try_push(rx.clone()).unwrap();
    let mut copy = a.clone();
    copy.try_push(rx).unwrap();
    assert_eq!(copy.assign_parameters(&[0.25]).unwrap().gates.len(), 2);
}

/// A bound circuit holds no expressions, so exporting one that still refers
/// to an expression fails instead of emitting anything.
#[test]
fn concrete_circuits_hold_no_expressions() {
    let mut pc = ParameterizedCircuit::new(1);
    let angle = pc.add_expr(x(0) * x(0)).unwrap();
    let hand = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![GateInstruction::Ry {
            qubit: 0,
            theta: angle,
        }],
    };
    // The bitcode export fails before it needs `llvm-as`.
    assert_eq!(hand.to_qir_bitcode(), Err(CircuitError::UnknownExpression));
    let panicked = std::panic::catch_unwind(|| hand.to_qasm2()).is_err();
    assert!(panicked, "ConcreteCircuit::to_qasm2 documents a panic");
}

// ─────────────────────── Exports of bound circuits ────────────────────────

/// Every angle kind in one circuit: the OpenQASM 2.0 and QIR exports with
/// parameter values are those of the circuit the values bind to.
#[test]
fn bound_exports_evaluate_expressions() {
    let values = [0.4, -1.1];
    let mut qc = ParameterizedCircuit::new(3);
    let cost = qc.add_expr(-x(1)).unwrap();
    let mixer = qc.add_expr(2.0 * x(0)).unwrap();
    let u = qc
        .add_expr(x(0) - ParamExpr::constant(Constant::Pi) / 2.0)
        .unwrap();
    let qc = qc
        .h(0)
        .rzz(0, 1, cost)
        .rx(0, mixer)
        .cu3(2, 1, u, Param(1), 0.3)
        .u2(1, cost, mixer)
        .measure_all();

    let bound = qc.assign_parameters(&values).unwrap();
    let fixed = ParameterizedCircuit::new(3)
        .h(0)
        .rzz(0, 1, 1.1)
        .rx(0, 0.8)
        .cu3(2, 1, 0.4 - FRAC_PI_2, -1.1, 0.3)
        .u2(1, 1.1, 0.8)
        .measure_all();
    assert_eq!(bound.gates, fixed.gates);

    let qasm = qc.to_qasm2_with_params(&values).unwrap();
    assert_eq!(qasm, fixed.to_qasm2_with_params(&[]).unwrap());
    assert_eq!(qasm, bound.to_qasm2());
    // The export imports back to the bound circuit and is a fixed point.
    let imported = ParameterizedCircuit::from_qasm2(&qasm).unwrap();
    assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), qasm);

    assert_eq!(
        qc.to_qir_with_params(&values).unwrap(),
        fixed.to_qir_with_params(&[]).unwrap()
    );
    // An angle error surfaces from every export, not only from binding.
    assert_eq!(
        qc.to_qasm2_with_params(&[f64::NAN, 0.0]),
        Err(CircuitError::NonFiniteParam)
    );
    assert_eq!(
        qc.to_qir_with_params(&[0.0, f64::INFINITY]),
        Err(CircuitError::NonFiniteParam)
    );
}

// ─────────────────────────────── Identity ─────────────────────────────────

/// Circuits compare expressions by content, wherever they are stored.
#[test]
fn equality_compares_expressions_by_content() {
    let build = |unused_first: bool, factor: f64| {
        let mut qc = ParameterizedCircuit::new(1);
        if unused_first {
            qc.add_expr(x(0) + 7.0).unwrap();
        }
        let angle = qc.add_expr(factor * x(0)).unwrap();
        qc.rx(0, angle)
    };
    let plain = build(false, 2.0);
    let shifted = build(true, 2.0);
    // Different ids, same expression.
    assert_ne!(theta(&plain.gates[0]), theta(&shifted.gates[0]));
    assert_eq!(plain, shifted);
    assert_eq!(plain, plain.clone());
    assert_ne!(plain, build(false, 3.0));
    // An expression is not the parameter it wraps, nor its value.
    assert_ne!(plain, ParameterizedCircuit::new(1).rx(0, Param(0)));
}

#[test]
fn circuits_with_expressions_stay_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ParameterizedCircuit>();
    assert_send_sync::<ParamExpr>();
    assert_send_sync::<GateParam>();
}

// ──────────────────────────── Resource bounds ─────────────────────────────

/// Adversarial shapes are bounded where they are built and never recursed
/// over: a 999 999-node chain (the node limit is 1 000 000) is stored,
/// evaluated, cloned and dropped on a thread with a 256 KiB stack.
#[test]
fn long_expressions_are_handled_without_recursion() {
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut chain = x(0);
            for _ in 0..499_999 {
                chain = chain + 1.0;
            }
            let mut qc = ParameterizedCircuit::new(1);
            let angle = qc.add_expr(chain.clone()).unwrap();
            let qc = qc.rz(0, angle);
            let copy = qc.clone();
            assert_eq!(copy, qc);
            drop(qc);
            let bound = copy.assign_parameters(&[0.5]).unwrap();
            assert_eq!(theta(&bound.gates[0]), GateParam::Fixed(499_999.5));
            // A few nodes over the limit.
            let mut qc = ParameterizedCircuit::new(1);
            assert!(matches!(
                qc.add_expr(chain + 1.0 + 1.0),
                Err(CircuitError::InvalidExpression { .. })
            ));
        })
        .unwrap();
    worker.join().unwrap();
}

/// Nesting is bounded at 64 levels as OpenQASM writes the expression, the
/// depth at which the importers stop; up to it, deep shapes evaluate without
/// recursion.
#[test]
fn nesting_is_bounded_as_written() {
    // `x + (1 + (1 + (… + (1 + 1))))`: every right operand in parentheses,
    // `depth` of them.
    let right_nested = |depth: usize| {
        let mut e = ParamExpr::from(1.0) + 1.0;
        for _ in 1..depth {
            e = ParamExpr::from(1.0) + e;
        }
        x(0) + e
    };
    assert_eq!(bind(right_nested(64), &[0.5]), Ok(65.0 + 0.5));
    assert!(matches!(
        ParameterizedCircuit::new(1).add_expr(right_nested(65)),
        Err(CircuitError::InvalidExpression { .. })
    ));
    // A power tower nests one level per exponent.
    let tower = |levels: usize| {
        let mut e = x(0);
        for _ in 0..levels {
            e = ParamExpr::from(1.0).pow(e);
        }
        e
    };
    assert_eq!(bind(tower(64), &[3.0]), Ok(1.0));
    assert!(matches!(
        ParameterizedCircuit::new(1).add_expr(tower(65)),
        Err(CircuitError::InvalidExpression { .. })
    ));
    // Unary minus too.
    let negated = |levels: usize| {
        let mut e = x(0);
        for _ in 0..levels {
            e = -e;
        }
        e
    };
    assert_eq!(bind(negated(64), &[0.5]), Ok(0.5));
    assert!(matches!(
        ParameterizedCircuit::new(1).add_expr(negated(65)),
        Err(CircuitError::InvalidExpression { .. })
    ));
}
