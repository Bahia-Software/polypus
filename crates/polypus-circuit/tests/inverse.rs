//! Integration tests for `ConcreteCircuit::inverse`: the instruction-level
//! inverse of every kind, the reversed order, the negated angles, barriers,
//! the typed errors, and what the inverse keeps of its circuit. That each
//! inverse undoes its gate on a state is checked against the simulator, the
//! reference semantics, in `crates/polypus-sim/tests/inverse.rs`.

use polypus_circuit::{
    CircuitError, ConcreteCircuit, GateInstruction as G, GateParam, ParameterizedCircuit as Pc,
};
use std::f64::consts::FRAC_PI_2;

fn fixed(v: f64) -> GateParam {
    GateParam::Fixed(v)
}

/// The inverse of the one-instruction circuit `gate` on 5 qubits.
fn inverse_of(gate: G) -> Vec<G> {
    ConcreteCircuit::new(5, vec![gate])
        .inverse()
        .expect("an invertible instruction")
        .into_gates()
}

/// A call of a gate declared with a `gate` block.
fn declared_call() -> G {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate g(t) a,b { rz(t) a; cx a,b; }\nqreg q[2];\ng(0.25) q[1],q[0];\n";
    Pc::from_qasm2(src).unwrap().gates()[0].clone()
}

#[test]
fn self_inverse_gates_are_kept() {
    for gate in [
        G::H(0),
        G::X(1),
        G::Y(2),
        G::Z(3),
        G::Id(4),
        G::Cx(0, 1),
        G::Cz(1, 2),
        G::Swap(2, 3),
        G::Cy(3, 4),
        G::Ch(4, 0),
        G::Ccx(0, 1, 2),
        G::Cswap(1, 2, 3),
        G::Rccx(2, 3, 4),
        G::C3x(0, 1, 2, 3),
        G::C4x(0, 1, 2, 3, 4),
        G::U0 {
            qubit: 2,
            gamma: fixed(0.7),
        },
    ] {
        assert_eq!(inverse_of(gate.clone()), [gate]);
    }
}

#[test]
fn dagger_pairs_swap() {
    assert_eq!(inverse_of(G::S(1)), [G::Sdg(1)]);
    assert_eq!(inverse_of(G::Sdg(1)), [G::S(1)]);
    assert_eq!(inverse_of(G::T(2)), [G::Tdg(2)]);
    assert_eq!(inverse_of(G::Tdg(2)), [G::T(2)]);
    assert_eq!(inverse_of(G::Sx(3)), [G::Sxdg(3)]);
    assert_eq!(inverse_of(G::Sxdg(3)), [G::Sx(3)]);
}

#[test]
fn one_angle_gates_negate_their_angle() {
    let t = 0.3;
    let cases = [
        (
            G::Rx {
                qubit: 0,
                theta: fixed(t),
            },
            G::Rx {
                qubit: 0,
                theta: fixed(-t),
            },
        ),
        (
            G::Ry {
                qubit: 1,
                theta: fixed(t),
            },
            G::Ry {
                qubit: 1,
                theta: fixed(-t),
            },
        ),
        (
            G::Rz {
                qubit: 2,
                theta: fixed(t),
            },
            G::Rz {
                qubit: 2,
                theta: fixed(-t),
            },
        ),
        (
            G::Rzz {
                q0: 0,
                q1: 3,
                theta: fixed(t),
            },
            G::Rzz {
                q0: 0,
                q1: 3,
                theta: fixed(-t),
            },
        ),
        (
            G::Rxx {
                q0: 4,
                q1: 1,
                theta: fixed(t),
            },
            G::Rxx {
                q0: 4,
                q1: 1,
                theta: fixed(-t),
            },
        ),
        (
            G::Cp {
                q0: 0,
                q1: 1,
                theta: fixed(t),
            },
            G::Cp {
                q0: 0,
                q1: 1,
                theta: fixed(-t),
            },
        ),
        (
            G::Cu1 {
                q0: 1,
                q1: 0,
                theta: fixed(t),
            },
            G::Cu1 {
                q0: 1,
                q1: 0,
                theta: fixed(-t),
            },
        ),
        (
            G::Crx {
                control: 0,
                target: 2,
                theta: fixed(t),
            },
            G::Crx {
                control: 0,
                target: 2,
                theta: fixed(-t),
            },
        ),
        (
            G::Cry {
                control: 2,
                target: 0,
                theta: fixed(t),
            },
            G::Cry {
                control: 2,
                target: 0,
                theta: fixed(-t),
            },
        ),
        (
            G::Crz {
                control: 3,
                target: 4,
                theta: fixed(t),
            },
            G::Crz {
                control: 3,
                target: 4,
                theta: fixed(-t),
            },
        ),
        (
            G::P {
                qubit: 1,
                lam: fixed(t),
            },
            G::P {
                qubit: 1,
                lam: fixed(-t),
            },
        ),
        (
            G::U1 {
                qubit: 2,
                lam: fixed(t),
            },
            G::U1 {
                qubit: 2,
                lam: fixed(-t),
            },
        ),
    ];
    for (gate, inverse) in cases {
        assert_eq!(inverse_of(gate), [inverse]);
    }
}

#[test]
fn u_forms_negate_and_swap_phi_and_lambda_under_their_own_spelling() {
    let (t, p, l, g) = (0.1, 0.2, 0.3, 0.4);
    assert_eq!(
        inverse_of(G::U {
            qubit: 0,
            theta: fixed(t),
            phi: fixed(p),
            lam: fixed(l),
        }),
        [G::U {
            qubit: 0,
            theta: fixed(-t),
            phi: fixed(-l),
            lam: fixed(-p),
        }]
    );
    assert_eq!(
        inverse_of(G::UGate {
            qubit: 1,
            theta: fixed(t),
            phi: fixed(p),
            lam: fixed(l),
        }),
        [G::UGate {
            qubit: 1,
            theta: fixed(-t),
            phi: fixed(-l),
            lam: fixed(-p),
        }]
    );
    assert_eq!(
        inverse_of(G::Cu3 {
            control: 2,
            target: 3,
            theta: fixed(t),
            phi: fixed(p),
            lam: fixed(l),
        }),
        [G::Cu3 {
            control: 2,
            target: 3,
            theta: fixed(-t),
            phi: fixed(-l),
            lam: fixed(-p),
        }]
    );
    assert_eq!(
        inverse_of(G::Cu {
            control: 4,
            target: 0,
            theta: fixed(t),
            phi: fixed(p),
            lam: fixed(l),
            gamma: fixed(g),
        }),
        [G::Cu {
            control: 4,
            target: 0,
            theta: fixed(-t),
            phi: fixed(-l),
            lam: fixed(-p),
            gamma: fixed(-g),
        }]
    );
}

#[test]
fn u2_becomes_the_u3_of_its_inverse() {
    assert_eq!(
        inverse_of(G::U2 {
            qubit: 3,
            phi: fixed(0.2),
            lam: fixed(-0.6),
        }),
        [G::U {
            qubit: 3,
            theta: fixed(-FRAC_PI_2),
            phi: fixed(0.6),
            lam: fixed(-0.2),
        }]
    );
}

#[test]
fn order_four_gates_without_an_inverse_instruction_repeat_three_times() {
    for gate in [G::Csx(0, 1), G::Rc3x(0, 1, 2, 3), G::C3sqrtx(4, 3, 2, 1)] {
        assert_eq!(inverse_of(gate.clone()), [gate.clone(), gate.clone(), gate]);
    }
}

#[test]
fn the_inverse_runs_the_inverses_in_reverse_order() {
    let circuit = Pc::new(3)
        .h(0)
        .s(1)
        .rx(2, 0.5)
        .cx(0, 1)
        .csx(1, 2)
        .t(2)
        .assign_parameters(&[])
        .unwrap();
    let inverse = circuit.inverse().unwrap();
    assert_eq!(
        inverse.gates(),
        [
            G::Tdg(2),
            G::Csx(1, 2),
            G::Csx(1, 2),
            G::Csx(1, 2),
            G::Cx(0, 1),
            G::Rx {
                qubit: 2,
                theta: fixed(-0.5),
            },
            G::Sdg(1),
            G::H(0),
        ]
    );
}

#[test]
fn barriers_are_kept_in_their_reversed_place() {
    let circuit = Pc::new(3)
        .h(0)
        .barrier()
        .x(1)
        .barrier_on(&[0, 2])
        .s(2)
        .assign_parameters(&[])
        .unwrap();
    assert_eq!(
        circuit.inverse().unwrap().gates(),
        [
            G::Sdg(2),
            G::Barrier(vec![0, 2]),
            G::X(1),
            G::Barrier(vec![]),
            G::H(0),
        ]
    );
}

#[test]
fn the_inverse_of_the_inverse_is_the_circuit() {
    // Every invertible kind whose inverse is one instruction of its own
    // spelling: the round trip gives the same instructions back, angles bit for
    // bit (negation is exact).
    let circuit = Pc::new(5)
        .h(0)
        .x(1)
        .y(2)
        .z(3)
        .s(4)
        .t(0)
        .sdg(1)
        .tdg(2)
        .id(3)
        .rx(0, 0.25)
        .ry(1, 0.5)
        .rz(2, -1.5)
        .u(0, 0.1, 0.2, 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .swap(0, 2)
        .rzz(0, 2, 0.4)
        .rxx(1, 2, 2.0)
        .cp(0, 1, 0.75)
        .sx(1)
        .sxdg(2)
        .cy(2, 0)
        .ch(1, 2)
        .ccx(2, 0, 1)
        .cswap(1, 2, 0)
        .crx(1, 0, 0.6)
        .cry(2, 1, -0.6)
        .crz(0, 2, 1.1)
        .cu1(2, 1, 0.8)
        .cu3(1, 0, 0.2, -0.4, 0.9)
        .cu(0, 2, 0.3, 0.1, -0.1, 0.9)
        .u0(3, 0.7)
        .rccx(4, 0, 2)
        .c3x(2, 4, 0, 3)
        .c4x(1, 4, 0, 3, 2)
        .p(4, 0.2)
        .u1(3, 0.45)
        .push(G::UGate {
            qubit: 1,
            theta: fixed(0.1),
            phi: fixed(0.2),
            lam: fixed(-0.7),
        })
        .barrier()
        .assign_parameters(&[])
        .unwrap();
    let twice = circuit.inverse().unwrap().inverse().unwrap();
    assert_eq!(twice, circuit);
    assert_eq!(twice.gates(), circuit.gates());
}

#[test]
fn measurements_and_declared_gate_calls_are_typed_errors() {
    let measured = Pc::new(2)
        .h(0)
        .measure(0, 0)
        .assign_parameters(&[])
        .unwrap();
    assert_eq!(
        measured.inverse(),
        Err(CircuitError::NotInvertible {
            position: 1,
            instruction: "measure",
        })
    );
    let measured_all = Pc::new(2).h(0).x(1).measure_all();
    assert_eq!(
        measured_all.assign_parameters(&[]).unwrap().inverse(),
        Err(CircuitError::NotInvertible {
            position: 2,
            instruction: "measure_all",
        })
    );
    let called = ConcreteCircuit::new(2, vec![G::H(0), declared_call(), G::X(1)]);
    assert_eq!(
        called.inverse(),
        Err(CircuitError::NotInvertible {
            position: 1,
            instruction: "a call of a declared gate",
        })
    );
    // The first offending instruction is the one reported.
    let both = ConcreteCircuit::new(2, vec![declared_call(), G::MeasureAll]);
    assert!(matches!(
        both.inverse(),
        Err(CircuitError::NotInvertible { position: 0, .. })
    ));
    assert!(called
        .inverse()
        .unwrap_err()
        .to_string()
        .contains("instruction 1 is a call of a declared gate"));
}

#[test]
fn an_angle_that_is_not_fixed_is_a_typed_error_not_a_silent_copy() {
    let free = ConcreteCircuit::new_unchecked(
        1,
        vec![G::Rx {
            qubit: 0,
            theta: GateParam::Param(0),
        }],
    );
    assert_eq!(
        free.inverse(),
        Err(CircuitError::ParamIndexOutOfBounds {
            index: 0,
            num_params: 0,
        })
    );
    // A parameter-free gate that only copies its angle (u0) is re-validated
    // with the rest, so it cannot smuggle a free parameter through either.
    let idle = ConcreteCircuit::new_unchecked(
        1,
        vec![G::U0 {
            qubit: 0,
            gamma: GateParam::Param(2),
        }],
    );
    assert_eq!(
        idle.inverse(),
        Err(CircuitError::ParamIndexOutOfBounds {
            index: 2,
            num_params: 0,
        })
    );
}

#[test]
fn the_inverse_keeps_the_qubits_and_the_declared_classical_width() {
    // An imported program that declares three classical bits and measures
    // nothing: the width survives the inverse, so the inverse compares equal
    // only with a circuit of the same width.
    let src =
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[3];\nh q[1];\nrz(0.5) q[3];\n";
    let circuit = Pc::from_qasm2(src).unwrap().assign_parameters(&[]).unwrap();
    let inverse = circuit.inverse().unwrap();
    assert_eq!(inverse.num_qubits(), 4);
    assert_eq!(inverse.declared_clbits(), Some(3));
    assert_eq!(inverse.clbits_width(), 3);
    assert_eq!(
        inverse.gates(),
        [
            G::Rz {
                qubit: 3,
                theta: fixed(-0.5),
            },
            G::H(1),
        ]
    );
    let builder = Pc::new(4).h(1).assign_parameters(&[]).unwrap();
    assert_eq!(builder.inverse().unwrap().declared_clbits(), None);
}

#[test]
fn the_empty_circuit_is_its_own_inverse() {
    let empty = ConcreteCircuit::new(3, Vec::new());
    assert_eq!(empty.inverse().unwrap(), empty);
}
