//! Each gate produces the expected amplitudes on basis states, including
//! phases. Conventions are Qiskit little-endian (qubit 0 = least-significant
//! bit of the amplitude index).

use polypus_circuit::GateInstruction as G;
use polypus_circuit::GateParam::Fixed;
use polypus_sim::{Statevector, C64};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI};

fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-12
}

/// Build a state, apply one gate, return amplitudes.
fn after(n: usize, prep: &[G], gate: G) -> Vec<C64> {
    let mut sv = Statevector::new(n).unwrap();
    for g in prep {
        sv.apply(g).unwrap();
    }
    sv.apply(&gate).unwrap();
    sv.amplitudes().to_vec()
}

#[test]
fn hadamard_on_zero() {
    let a = after(1, &[], G::H(0));
    assert!(close(a[0], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[1], C64::new(FRAC_1_SQRT_2, 0.0)));
}

#[test]
fn pauli_x_flips() {
    let a = after(1, &[], G::X(0));
    assert!(close(a[0], C64::new(0.0, 0.0)));
    assert!(close(a[1], C64::new(1.0, 0.0)));
}

#[test]
fn pauli_y_on_zero() {
    // Y|0> = i|1>
    let a = after(1, &[], G::Y(0));
    assert!(close(a[0], C64::new(0.0, 0.0)));
    assert!(close(a[1], C64::new(0.0, 1.0)));
}

#[test]
fn pauli_z_on_one() {
    // Z|1> = -|1>
    let a = after(1, &[G::X(0)], G::Z(0));
    assert!(close(a[1], C64::new(-1.0, 0.0)));
}

#[test]
fn s_gate_on_one() {
    // S|1> = i|1>
    let a = after(1, &[G::X(0)], G::S(0));
    assert!(close(a[1], C64::new(0.0, 1.0)));
}

#[test]
fn sdg_gate_on_one() {
    let a = after(1, &[G::X(0)], G::Sdg(0));
    assert!(close(a[1], C64::new(0.0, -1.0)));
}

#[test]
fn t_gate_on_one() {
    // T|1> = e^{iπ/4}|1>
    let a = after(1, &[G::X(0)], G::T(0));
    assert!(close(a[1], C64::from_polar(1.0, FRAC_PI_4)));
}

#[test]
fn tdg_gate_on_one() {
    let a = after(1, &[G::X(0)], G::Tdg(0));
    assert!(close(a[1], C64::from_polar(1.0, -FRAC_PI_4)));
}

#[test]
fn rz_applies_global_phase_pair() {
    // Rz(θ)|0> = e^{-iθ/2}|0>
    let theta = 0.7;
    let a = after(
        1,
        &[],
        G::Rz {
            qubit: 0,
            theta: Fixed(theta),
        },
    );
    assert!(close(a[0], C64::from_polar(1.0, -theta / 2.0)));
    assert!(close(a[1], C64::new(0.0, 0.0)));
}

#[test]
fn rx_pi_on_zero() {
    // Rx(π)|0> = -i|1>
    let a = after(
        1,
        &[],
        G::Rx {
            qubit: 0,
            theta: Fixed(PI),
        },
    );
    assert!(close(a[0], C64::new(0.0, 0.0)));
    assert!(close(a[1], C64::new(0.0, -1.0)));
}

#[test]
fn ry_pi_on_zero() {
    // Ry(π)|0> = |1>
    let a = after(
        1,
        &[],
        G::Ry {
            qubit: 0,
            theta: Fixed(PI),
        },
    );
    assert!(close(a[0], C64::new(0.0, 0.0)));
    assert!(close(a[1], C64::new(1.0, 0.0)));
}

#[test]
fn u_reproduces_x() {
    // U(π, 0, π) == X
    let a = after(
        1,
        &[],
        G::U {
            qubit: 0,
            theta: Fixed(PI),
            phi: Fixed(0.0),
            lam: Fixed(PI),
        },
    );
    assert!(close(a[0], C64::new(0.0, 0.0)));
    assert!(close(a[1], C64::new(1.0, 0.0)));
}

#[test]
fn u_reproduces_hadamard() {
    // U(π/2, 0, π) == H
    let a = after(
        1,
        &[],
        G::U {
            qubit: 0,
            theta: Fixed(std::f64::consts::FRAC_PI_2),
            phi: Fixed(0.0),
            lam: Fixed(PI),
        },
    );
    assert!(close(a[0], C64::new(FRAC_1_SQRT_2, 0.0)));
    assert!(close(a[1], C64::new(FRAC_1_SQRT_2, 0.0)));
}

#[test]
fn cx_flips_target_when_control_set() {
    // |00> --X(0)--> |01_idx1> --CX(0,1)--> q1 flips -> index 3
    let a = after(2, &[G::X(0)], G::Cx(0, 1));
    assert!(close(a[3], C64::new(1.0, 0.0)));
    for (i, amp) in a.iter().enumerate() {
        if i != 3 {
            assert!(close(*amp, C64::new(0.0, 0.0)));
        }
    }
}

#[test]
fn cx_leaves_target_when_control_clear() {
    // control q0 = 0 -> no flip; state stays |0...>
    let a = after(2, &[], G::Cx(0, 1));
    assert!(close(a[0], C64::new(1.0, 0.0)));
}

#[test]
fn cz_phases_eleven() {
    // |11> -> -|11>
    let a = after(2, &[G::X(0), G::X(1)], G::Cz(0, 1));
    assert!(close(a[3], C64::new(-1.0, 0.0)));
}

#[test]
fn rzz_on_zero_zero_phase() {
    // Rzz(θ)|00> = e^{-iθ/2}|00> (bits equal)
    let theta = 0.9;
    let a = after(
        2,
        &[],
        G::Rzz {
            q0: 0,
            q1: 1,
            theta: Fixed(theta),
        },
    );
    assert!(close(a[0], C64::from_polar(1.0, -theta / 2.0)));
}

#[test]
fn rzz_on_zero_one_phase() {
    // |01> (q0=1, q1=0): bits differ -> e^{+iθ/2}
    let theta = 0.9;
    let a = after(
        2,
        &[G::X(0)],
        G::Rzz {
            q0: 0,
            q1: 1,
            theta: Fixed(theta),
        },
    );
    assert!(close(a[1], C64::from_polar(1.0, theta / 2.0)));
}

#[test]
fn rxx_on_zero_zero() {
    // Rxx(θ)|00> = cos(θ/2)|00> - i sin(θ/2)|11>
    let theta = 1.1;
    let a = after(
        2,
        &[],
        G::Rxx {
            q0: 0,
            q1: 1,
            theta: Fixed(theta),
        },
    );
    assert!(close(a[0], C64::new((theta / 2.0).cos(), 0.0)));
    assert!(close(a[3], C64::new(0.0, -(theta / 2.0).sin())));
}

#[test]
fn unbound_parameter_is_rejected() {
    use polypus_circuit::GateParam::Param;
    let mut sv = Statevector::new(1).unwrap();
    let err = sv
        .apply(&G::Rx {
            qubit: 0,
            theta: Param(0),
        })
        .unwrap_err();
    assert_eq!(err, polypus_sim::SimError::UnboundParameter { index: 0 });
}

#[test]
fn non_finite_angle_is_rejected() {
    let mut sv = Statevector::new(1).unwrap();
    let err = sv
        .apply(&G::Rx {
            qubit: 0,
            theta: Fixed(f64::NAN),
        })
        .unwrap_err();
    assert_eq!(err, polypus_sim::SimError::NonFiniteAmplitude);
}
