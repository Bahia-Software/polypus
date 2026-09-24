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
fn identity_leaves_every_amplitude_unchanged() {
    // A generic two-qubit state, so a mis-applied phase or swap would show.
    let prep = [
        G::H(0),
        G::Ry {
            qubit: 1,
            theta: Fixed(0.7),
        },
        G::Cx(0, 1),
        G::T(0),
    ];
    let mut before = Statevector::new(2).unwrap();
    for g in &prep {
        before.apply(g).unwrap();
    }
    for q in 0..2 {
        assert_eq!(after(2, &prep, G::Id(q)), before.amplitudes());
    }
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
fn swap_exchanges_the_two_qubits() {
    // |00> --X(0)--> |q1=0,q0=1> (index 1) --SWAP(0,1)--> |q1=1,q0=0> (index 2)
    let a = after(2, &[G::X(0)], G::Swap(0, 1));
    assert!(close(a[2], C64::new(1.0, 0.0)));
    for (i, amp) in a.iter().enumerate() {
        if i != 2 {
            assert!(close(*amp, C64::new(0.0, 0.0)));
        }
    }
}

#[test]
fn swap_is_symmetric_in_its_qubits() {
    // Same result regardless of argument order.
    let a = after(2, &[G::X(0)], G::Swap(0, 1));
    let b = after(2, &[G::X(0)], G::Swap(1, 0));
    assert!(a.iter().zip(&b).all(|(x, y)| close(*x, *y)));
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

// ── Tier-1 qelib1.inc gates: full unitaries against independent references ──
//
// Each gate's unitary is read off the simulator column by column and compared,
// entry by entry and including phases, with a reference matrix built here from
// the gate's definition (Qiskit conventions, little-endian basis). Operands are
// deliberately not ascending, so a swapped control/target is caught.

/// A `2^n × 2^n` matrix stored column by column: `m[col][row]`, where column
/// `col` is the image of the basis state `|col⟩`.
type Matrix = Vec<Vec<C64>>;

/// The unitary the simulator realises for `gate`.
fn unitary_of(n: usize, gate: &G) -> Matrix {
    (0..1usize << n)
        .map(|col| {
            let prep: Vec<G> = (0..n).filter(|q| (col >> q) & 1 == 1).map(G::X).collect();
            after(n, &prep, gate.clone())
        })
        .collect()
}

/// A reference matrix whose column `col` is `column(col)`, a list of
/// `(row, amplitude)` entries (all other rows zero).
fn reference(n: usize, column: impl Fn(usize) -> Vec<(usize, C64)>) -> Matrix {
    let dim = 1usize << n;
    (0..dim)
        .map(|col| {
            let mut out = vec![C64::new(0.0, 0.0); dim];
            for (row, amp) in column(col) {
                out[row] = amp;
            }
            out
        })
        .collect()
}

/// Reference: apply the 2×2 `u` to `target` iff every qubit in `controls` is 1.
fn controlled_reference(n: usize, controls: &[usize], target: usize, u: [[C64; 2]; 2]) -> Matrix {
    reference(n, |col| {
        if controls.iter().any(|c| (col >> c) & 1 == 0) {
            return vec![(col, C64::new(1.0, 0.0))];
        }
        let bit = (col >> target) & 1;
        u.iter()
            .enumerate()
            .map(|(out_bit, u_row)| ((col & !(1 << target)) | (out_bit << target), u_row[bit]))
            .collect()
    })
}

/// Reference: exchange qubits `a` and `b` iff `control` is 1.
fn cswap_reference(n: usize, control: usize, a: usize, b: usize) -> Matrix {
    reference(n, |col| {
        let row = if (col >> control) & 1 == 1 && (col >> a) & 1 != (col >> b) & 1 {
            col ^ (1 << a) ^ (1 << b)
        } else {
            col
        };
        vec![(row, C64::new(1.0, 0.0))]
    })
}

/// Reference: a single-qubit `u` on `target` of an `n`-qubit register.
fn one_qubit_reference(n: usize, target: usize, u: [[C64; 2]; 2]) -> Matrix {
    controlled_reference(n, &[], target, u)
}

fn assert_matrix_eq(actual: &Matrix, expected: &Matrix, what: &str) {
    for (col, (ca, ce)) in actual.iter().zip(expected).enumerate() {
        for (row, (a, e)) in ca.iter().zip(ce).enumerate() {
            assert!(
                close(*a, *e),
                "{what}: entry [{row}][{col}] is {a}, expected {e}"
            );
        }
    }
}

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Qiskit's `U(θ, φ, λ)`.
fn u_matrix(theta: f64, phi: f64, lam: f64) -> [[C64; 2]; 2] {
    let (co, si) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        [c(co, 0.0), -C64::from_polar(si, lam)],
        [C64::from_polar(si, phi), C64::from_polar(co, phi + lam)],
    ]
}

#[test]
fn sx_and_sxdg_match_their_matrices_and_compose_to_x() {
    let sx = [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]];
    let sxdg = [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]];
    assert_matrix_eq(
        &unitary_of(2, &G::Sx(1)),
        &one_qubit_reference(2, 1, sx),
        "sx",
    );
    assert_matrix_eq(
        &unitary_of(2, &G::Sxdg(0)),
        &one_qubit_reference(2, 0, sxdg),
        "sxdg",
    );
    // sx · sx = x (exactly, no phase).
    let mut sv = Statevector::new(1).unwrap();
    sv.apply(&G::Sx(0)).unwrap();
    sv.apply(&G::Sx(0)).unwrap();
    assert!(close(sv.amplitudes()[0], c(0.0, 0.0)));
    assert!(close(sv.amplitudes()[1], c(1.0, 0.0)));
}

#[test]
fn controlled_single_qubit_gates_match_their_references() {
    let s = FRAC_1_SQRT_2;
    let y = [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]];
    let h = [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]];
    let sx = [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]];
    for (control, target) in [(2, 0), (0, 2), (1, 0)] {
        assert_matrix_eq(
            &unitary_of(3, &G::Cy(control, target)),
            &controlled_reference(3, &[control], target, y),
            "cy",
        );
        assert_matrix_eq(
            &unitary_of(3, &G::Ch(control, target)),
            &controlled_reference(3, &[control], target, h),
            "ch",
        );
        assert_matrix_eq(
            &unitary_of(3, &G::Csx(control, target)),
            &controlled_reference(3, &[control], target, sx),
            "csx",
        );
    }
}

#[test]
fn controlled_rotations_match_their_references() {
    for &theta in &[0.3, -1.7, PI] {
        let (co, si) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        let rx = [[c(co, 0.0), c(0.0, -si)], [c(0.0, -si), c(co, 0.0)]];
        let ry = [[c(co, 0.0), c(-si, 0.0)], [c(si, 0.0), c(co, 0.0)]];
        let rz = [
            [C64::from_polar(1.0, -theta / 2.0), c(0.0, 0.0)],
            [c(0.0, 0.0), C64::from_polar(1.0, theta / 2.0)],
        ];
        let (control, target) = (2, 1);
        let t = Fixed(theta);
        assert_matrix_eq(
            &unitary_of(
                3,
                &G::Crx {
                    control,
                    target,
                    theta: t,
                },
            ),
            &controlled_reference(3, &[control], target, rx),
            "crx",
        );
        assert_matrix_eq(
            &unitary_of(
                3,
                &G::Cry {
                    control,
                    target,
                    theta: t,
                },
            ),
            &controlled_reference(3, &[control], target, ry),
            "cry",
        );
        assert_matrix_eq(
            &unitary_of(
                3,
                &G::Crz {
                    control,
                    target,
                    theta: t,
                },
            ),
            &controlled_reference(3, &[control], target, rz),
            "crz",
        );
    }
}

#[test]
fn cu1_is_exactly_cp() {
    for &theta in &[0.3, -2.1, FRAC_PI_4] {
        let cu1 = unitary_of(
            3,
            &G::Cu1 {
                q0: 2,
                q1: 0,
                theta: Fixed(theta),
            },
        );
        let cp = unitary_of(
            3,
            &G::Cp {
                q0: 2,
                q1: 0,
                theta: Fixed(theta),
            },
        );
        assert_matrix_eq(&cu1, &cp, "cu1 vs cp");
        let phase = [
            [c(1.0, 0.0), c(0.0, 0.0)],
            [c(0.0, 0.0), C64::from_polar(1.0, theta)],
        ];
        assert_matrix_eq(&cu1, &controlled_reference(3, &[2], 0, phase), "cu1");
    }
}

#[test]
fn cu3_and_cu_match_their_references_including_gamma() {
    for &(th, ph, la, ga) in &[(0.1, 0.2, 0.3, 0.4), (1.3, -0.7, 2.2, -1.1)] {
        let u = u_matrix(th, ph, la);
        let (control, target) = (0, 2);
        assert_matrix_eq(
            &unitary_of(
                3,
                &G::Cu3 {
                    control,
                    target,
                    theta: Fixed(th),
                    phi: Fixed(ph),
                    lam: Fixed(la),
                },
            ),
            &controlled_reference(3, &[control], target, u),
            "cu3",
        );
        // `cu` multiplies the controlled branch by e^{iγ}: a relative phase.
        let g = C64::from_polar(1.0, ga);
        let gu = u.map(|row| row.map(|e| e * g));
        assert_matrix_eq(
            &unitary_of(
                3,
                &G::Cu {
                    control,
                    target,
                    theta: Fixed(th),
                    phi: Fixed(ph),
                    lam: Fixed(la),
                    gamma: Fixed(ga),
                },
            ),
            &controlled_reference(3, &[control], target, gu),
            "cu",
        );
    }
}

#[test]
fn ccx_and_cswap_are_exact_permutations() {
    let x = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    for (a, b, t) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
        assert_matrix_eq(
            &unitary_of(3, &G::Ccx(a, b, t)),
            &controlled_reference(3, &[a, b], t, x),
            "ccx",
        );
        assert_matrix_eq(
            &unitary_of(3, &G::Cswap(a, b, t)),
            &cswap_reference(3, a, b, t),
            "cswap",
        );
    }
    // On a 4-qubit register, the untouched qubit stays untouched.
    assert_matrix_eq(
        &unitary_of(4, &G::Ccx(3, 1, 2)),
        &controlled_reference(4, &[3, 1], 2, x),
        "ccx on 4 qubits",
    );
}

#[test]
fn qelib1_gates_reject_unbound_and_non_finite_angles() {
    use polypus_circuit::GateParam::Param;
    let mut sv = Statevector::new(2).unwrap();
    assert_eq!(
        sv.apply(&G::Cu {
            control: 0,
            target: 1,
            theta: Fixed(0.1),
            phi: Fixed(0.2),
            lam: Fixed(0.3),
            gamma: Param(4),
        }),
        Err(polypus_sim::SimError::UnboundParameter { index: 4 })
    );
    assert_eq!(
        sv.apply(&G::Crz {
            control: 1,
            target: 0,
            theta: Fixed(f64::INFINITY),
        }),
        Err(polypus_sim::SimError::NonFiniteAmplitude)
    );
    // Nothing was applied: the state is still |00>.
    assert!(close(sv.amplitudes()[0], c(1.0, 0.0)));
}

// ── The multi-qubit qelib1.inc gates (rccx, rc3x, c3x, c3sqrtx, c4x, u0) ────

#[test]
fn multi_controlled_gates_match_their_references() {
    let x = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    let sx = [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]];
    for (a, b, cc, t) in [(0, 1, 2, 3), (3, 1, 0, 2), (2, 3, 1, 0)] {
        assert_matrix_eq(
            &unitary_of(4, &G::C3x(a, b, cc, t)),
            &controlled_reference(4, &[a, b, cc], t, x),
            "c3x",
        );
        assert_matrix_eq(
            &unitary_of(4, &G::C3sqrtx(a, b, cc, t)),
            &controlled_reference(4, &[a, b, cc], t, sx),
            "c3sqrtx",
        );
    }
    for (a, b, cc, d, t) in [(0, 1, 2, 3, 4), (4, 2, 0, 3, 1)] {
        assert_matrix_eq(
            &unitary_of(5, &G::C4x(a, b, cc, d, t)),
            &controlled_reference(5, &[a, b, cc, d], t, x),
            "c4x",
        );
    }
}

/// `rccx` is a Toffoli up to relative phases — Qiskit's `RCCXGate` matrix:
/// with controls `a`, `b` and target `c` (little-endian `a + 2b + 4c`),
/// |a b c⟩ = |1 1 0⟩ ↦ i|1 1 1⟩, |1 1 1⟩ ↦ −i|1 1 0⟩, |1 0 1⟩ ↦ −|1 0 1⟩, and
/// every other basis state is fixed.
#[test]
fn rccx_matches_qiskits_matrix_including_relative_phases() {
    for (a, b, t) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
        let reference = reference(3, |col| {
            let bit = |q: usize| (col >> q) & 1;
            match (bit(a), bit(b), bit(t)) {
                (1, 1, 0) => vec![(col | 1 << t, c(0.0, 1.0))],
                (1, 1, 1) => vec![(col & !(1 << t), c(0.0, -1.0))],
                (1, 0, 1) => vec![(col, c(-1.0, 0.0))],
                _ => vec![(col, c(1.0, 0.0))],
            }
        });
        assert_matrix_eq(&unitary_of(3, &G::Rccx(a, b, t)), &reference, "rccx");
    }
}

/// `rc3x` acts as a 3-controlled X on the |c0 c1 c2⟩ = |111⟩ subspace up to
/// relative phases: every basis state keeps its probability pattern of
/// `c3x` (unit modulus on the same entries). Its exact phases are checked
/// against Qiskit's `RC3XGate` in the Python equivalence tests.
#[test]
fn rc3x_has_the_permutation_structure_of_c3x() {
    let rc3x = unitary_of(4, &G::Rc3x(2, 0, 3, 1));
    let c3x = unitary_of(4, &G::C3x(2, 0, 3, 1));
    for (col_r, col_c) in rc3x.iter().zip(&c3x) {
        for (r, x) in col_r.iter().zip(col_c) {
            assert!((r.norm() - x.norm()).abs() < 1e-12);
        }
    }
}

/// The spellings of the generic single-qubit gate: `p`/`u1` are the phase
/// gate, `u2(φ,λ)` is `u3(π/2,φ,λ)`, and `u` is `u3` — each exactly.
#[test]
fn single_qubit_spellings_match_their_matrices() {
    let (th, ph, la) = (0.7, -1.3, 2.1);
    let phase = [
        [c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), C64::from_polar(1.0, la)],
    ];
    for gate in [
        G::P {
            qubit: 1,
            lam: Fixed(la),
        },
        G::U1 {
            qubit: 1,
            lam: Fixed(la),
        },
    ] {
        assert_matrix_eq(
            &unitary_of(2, &gate),
            &one_qubit_reference(2, 1, phase),
            "p/u1",
        );
    }
    assert_matrix_eq(
        &unitary_of(
            2,
            &G::U2 {
                qubit: 0,
                phi: Fixed(ph),
                lam: Fixed(la),
            },
        ),
        &one_qubit_reference(2, 0, u_matrix(std::f64::consts::FRAC_PI_2, ph, la)),
        "u2",
    );
    assert_matrix_eq(
        &unitary_of(
            2,
            &G::UGate {
                qubit: 1,
                theta: Fixed(th),
                phi: Fixed(ph),
                lam: Fixed(la),
            },
        ),
        &one_qubit_reference(2, 1, u_matrix(th, ph, la)),
        "u",
    );
}

#[test]
fn u0_is_the_identity() {
    let prep = [G::H(0), G::Cx(0, 1), G::T(1)];
    let mut before = Statevector::new(2).unwrap();
    for g in &prep {
        before.apply(g).unwrap();
    }
    let after_u0 = after(
        2,
        &prep,
        G::U0 {
            qubit: 1,
            gamma: Fixed(3.0),
        },
    );
    assert_eq!(after_u0, before.amplitudes());
}
