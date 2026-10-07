//! The matrices of the OpenQASM 3 profile, in full, with the native simulator
//! as Polypus's reference semantics:
//!
//! - **Spec conformance.** Every `stdgates.inc` gate and the builtin `U`, as
//!   `from_qasm3` reads them, against matrices computed from the
//!   specification's own definitions (`source/language/gates.rst` for `U`,
//!   `stdgates.inc` at tag `spec/v3.1.0` for the rest). `U`, `u2` and `u3`
//!   differ by exactly the global factors of the profile's Qiskit phase
//!   conventions; every other gate is equal.
//! - **Exported definitions.** Every gate definition `to_qasm3` writes for an
//!   instruction `stdgates.inc` lacks reproduces that instruction's matrix
//!   exactly — no global phase allowed.
//! - **Cross-dialect.** Declarations printed in the other dialect keep their
//!   matrices.
//!
//! Conventions are Qiskit's: qubit 0 is the least-significant bit, and a
//! gate's first argument is qubit 0 of its own matrix.

use polypus_circuit::{Fixed, GateInstruction as G, ParameterizedCircuit};
use polypus_sim::{Statevector, C64};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// A square matrix, row-major.
type Matrix = Vec<Vec<C64>>;

const TOLERANCE: f64 = 1e-12;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn phase(angle: f64) -> C64 {
    C64::from_polar(1.0, angle)
}

/// The unitary `gates` realise on `n` qubits, column by column.
fn unitary(n: usize, gates: &[G]) -> Matrix {
    let dim = 1 << n;
    let columns: Vec<Vec<C64>> = (0..dim)
        .map(|col| {
            let mut sv = Statevector::new(n).unwrap();
            for q in 0..n {
                if (col >> q) & 1 == 1 {
                    sv.apply(&G::X(q)).unwrap();
                }
            }
            for g in gates {
                sv.apply(g).unwrap();
            }
            sv.amplitudes().to_vec()
        })
        .collect();
    (0..dim)
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect()
}

/// The unitary of a circuit with no measurements.
fn circuit_unitary(qc: &ParameterizedCircuit) -> Matrix {
    unitary(qc.num_qubits, &qc.assign_parameters(&[]).unwrap().gates)
}

fn assert_close(what: &str, got: &Matrix, want: &Matrix) {
    assert_eq!(got.len(), want.len(), "{what}: size");
    for (r, (g, w)) in got.iter().zip(want).enumerate() {
        for (k, (a, b)) in g.iter().zip(w).enumerate() {
            assert!(
                (a - b).norm() < TOLERANCE,
                "{what}: entry ({r}, {k}) is {a}, expected {b}"
            );
        }
    }
}

fn scale(z: C64, m: &Matrix) -> Matrix {
    m.iter()
        .map(|row| row.iter().map(|x| z * x).collect())
        .collect()
}

/// `a · b`: `b` applied first.
fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    (0..n)
        .map(|r| {
            (0..n)
                .map(|k| (0..n).map(|j| a[r][j] * b[j][k]).sum())
                .collect()
        })
        .collect()
}

fn dagger(m: &Matrix) -> Matrix {
    let n = m.len();
    (0..n)
        .map(|r| (0..n).map(|k| m[k][r].conj()).collect())
        .collect()
}

fn diag(entries: &[C64]) -> Matrix {
    let n = entries.len();
    (0..n)
        .map(|r| {
            (0..n)
                .map(|k| if r == k { entries[r] } else { c(0.0, 0.0) })
                .collect()
        })
        .collect()
}

fn identity(dim: usize) -> Matrix {
    diag(&vec![c(1.0, 0.0); dim])
}

// ─────────────────── The specification's definitions ────────────────────

/// `U(θ, φ, λ)` as `gates.rst` defines it:
/// ½[[1+e^{iθ}, −i e^{iλ}(1−e^{iθ})], [i e^{iφ}(1−e^{iθ}), e^{i(φ+λ)}(1+e^{iθ})]].
fn spec_u(theta: f64, phi: f64, lam: f64) -> Matrix {
    let one = c(1.0, 0.0);
    let i = c(0.0, 1.0);
    let e = phase;
    vec![
        vec![
            0.5 * (one + e(theta)),
            0.5 * (-i * e(lam) * (one - e(theta))),
        ],
        vec![
            0.5 * (i * e(phi) * (one - e(theta))),
            0.5 * (e(phi + lam) * (one + e(theta))),
        ],
    ]
}

/// `ctrl @ g`: the new control is the first argument, qubit 0.
fn ctrl(g: &Matrix) -> Matrix {
    let m = g.len();
    let mut out = identity(2 * m);
    for row in 0..m {
        for col in 0..m {
            out[2 * row + 1][2 * col + 1] = g[row][col];
        }
    }
    out
}

/// `g` on the first qubit of two (the second untouched).
fn on_first_of_two(g: &Matrix) -> Matrix {
    let mut out = vec![vec![c(0.0, 0.0); 4]; 4];
    for high in 0..2 {
        for r in 0..2 {
            for k in 0..2 {
                out[2 * high + r][2 * high + k] = g[r][k];
            }
        }
    }
    out
}

/// A two-qubit gate with its arguments exchanged.
fn swap_arguments(g: &Matrix) -> Matrix {
    let flip = |i: usize| ((i & 1) << 1) | (i >> 1);
    (0..4)
        .map(|r| (0..4).map(|k| g[flip(r)][flip(k)]).collect())
        .collect()
}

/// The principal power `pow(k) @` of a diagonal unitary: each eigenphase in
/// (−π, π] scaled by `k`.
fn principal_power_of_diagonal(m: &Matrix, k: f64) -> Matrix {
    let entries: Vec<C64> = (0..m.len())
        .map(|i| {
            let mut angle = m[i][i].arg();
            if (angle + PI).abs() < 1e-15 {
                angle = PI;
            }
            m[i][i].norm().powf(k) * phase(k * angle)
        })
        .collect();
    diag(&entries)
}

/// The `stdgates.inc` gates, transcribed from their definitions at tag
/// `spec/v3.1.0` (`gphase(γ)` is the factor e^{iγ}).
struct Spec;

impl Spec {
    fn p(lam: f64) -> Matrix {
        // `ctrl @ gphase(λ) a`
        diag(&[c(1.0, 0.0), phase(lam)])
    }
    fn x() -> Matrix {
        scale(phase(-FRAC_PI_2), &spec_u(PI, 0.0, PI))
    }
    fn y() -> Matrix {
        scale(phase(-FRAC_PI_2), &spec_u(PI, FRAC_PI_2, FRAC_PI_2))
    }
    fn z() -> Matrix {
        Self::p(PI)
    }
    fn h() -> Matrix {
        scale(phase(-FRAC_PI_4), &spec_u(FRAC_PI_2, 0.0, PI))
    }
    fn s() -> Matrix {
        principal_power_of_diagonal(&Self::z(), 0.5)
    }
    fn sdg() -> Matrix {
        dagger(&Self::s())
    }
    fn t() -> Matrix {
        principal_power_of_diagonal(&Self::s(), 0.5)
    }
    fn tdg() -> Matrix {
        dagger(&Self::t())
    }
    fn sx() -> Matrix {
        // `pow(0.5) @ x`: h diagonalises x (x = h·z·h), so its principal
        // square root is h·pow(0.5)(z)·h.
        let h = Self::h();
        mul(&h, &mul(&Self::s(), &h))
    }
    fn rx(theta: f64) -> Matrix {
        scale(phase(-theta / 2.0), &spec_u(theta, -FRAC_PI_2, FRAC_PI_2))
    }
    fn ry(theta: f64) -> Matrix {
        scale(phase(-theta / 2.0), &spec_u(theta, 0.0, 0.0))
    }
    fn rz(lam: f64) -> Matrix {
        scale(phase(-lam / 2.0), &spec_u(0.0, 0.0, lam))
    }
    fn swap() -> Matrix {
        // `cx a, b; cx b, a; cx a, b;`
        let cx = ctrl(&Self::x());
        let xc = swap_arguments(&cx);
        mul(&cx, &mul(&xc, &cx))
    }
    fn cu(theta: f64, phi: f64, lam: f64, gamma: f64) -> Matrix {
        // `p(γ − θ/2) a; ctrl @ U(θ, φ, λ) a, b;`
        mul(
            &ctrl(&spec_u(theta, phi, lam)),
            &on_first_of_two(&Self::p(gamma - theta / 2.0)),
        )
    }
    /// `stdgates.inc`'s own `CX`: `ctrl @ U(π, 0, π)`.
    fn cx_as_defined() -> Matrix {
        ctrl(&spec_u(PI, 0.0, PI))
    }
    fn u2(phi: f64, lam: f64) -> Matrix {
        scale(
            phase(-(phi + lam + FRAC_PI_2) / 2.0),
            &spec_u(FRAC_PI_2, phi, lam),
        )
    }
    fn u3(theta: f64, phi: f64, lam: f64) -> Matrix {
        scale(phase(-(phi + lam + theta) / 2.0), &spec_u(theta, phi, lam))
    }
}

/// Qiskit's `U(θ, φ, λ)` (Polypus's `u`):
/// [[cos(θ/2), −e^{iλ} sin(θ/2)], [e^{iφ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)]].
fn qiskit_u(theta: f64, phi: f64, lam: f64) -> Matrix {
    let (s, co) = (theta / 2.0).sin_cos();
    vec![
        vec![c(co, 0.0), -phase(lam) * s],
        vec![phase(phi) * s, phase(phi + lam) * co],
    ]
}

/// The one-instruction circuit `statement` imports to, on `n` qubits.
fn read(n: usize, statement: &str) -> Matrix {
    let src = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\n{statement}\n");
    let qc = ParameterizedCircuit::from_qasm3(&src).unwrap_or_else(|e| panic!("{statement}: {e}"));
    assert_eq!(qc.gates.len(), 1, "{statement}");
    circuit_unitary(&qc)
}

/// Angle draws for the parameterised gates.
const DRAWS: [(f64, f64, f64, f64); 3] = [
    (0.3, -1.1, 2.2, 0.7),
    (-2.5, 0.4, -0.9, 1.9),
    (3.0, 2.9, -3.1, -0.2),
];

// ───────────────────────── Spec conformance ─────────────────────────────

#[test]
fn stdgates_gates_without_parameters_follow_the_specification_exactly() {
    let cases: [(&str, usize, Matrix); 16] = [
        ("x q[0];", 1, Spec::x()),
        ("y q[0];", 1, Spec::y()),
        ("z q[0];", 1, Spec::z()),
        ("h q[0];", 1, Spec::h()),
        ("s q[0];", 1, Spec::s()),
        ("sdg q[0];", 1, Spec::sdg()),
        ("t q[0];", 1, Spec::t()),
        ("tdg q[0];", 1, Spec::tdg()),
        ("sx q[0];", 1, Spec::sx()),
        ("id q[0];", 1, spec_u(0.0, 0.0, 0.0)),
        ("cx q[0], q[1];", 2, ctrl(&Spec::x())),
        ("cy q[0], q[1];", 2, ctrl(&Spec::y())),
        ("cz q[0], q[1];", 2, ctrl(&Spec::z())),
        ("ch q[0], q[1];", 2, ctrl(&Spec::h())),
        ("swap q[0], q[1];", 2, Spec::swap()),
        ("cswap q[0], q[1], q[2];", 3, ctrl(&Spec::swap())),
    ];
    for (statement, n, spec) in cases {
        assert_close(statement, &read(n, statement), &spec);
    }
    assert_close(
        "ccx",
        &read(3, "ccx q[0], q[1], q[2];"),
        &ctrl(&ctrl(&Spec::x())),
    );
}

#[test]
fn stdgates_gates_with_parameters_follow_the_specification_exactly() {
    for (t, f, l, g) in DRAWS {
        let cases: Vec<(String, usize, Matrix)> = vec![
            (format!("p({t}) q[0];"), 1, Spec::p(t)),
            (format!("phase({t}) q[0];"), 1, spec_u(0.0, 0.0, t)),
            (format!("u1({t}) q[0];"), 1, spec_u(0.0, 0.0, t)),
            (format!("rx({t}) q[0];"), 1, Spec::rx(t)),
            (format!("ry({t}) q[0];"), 1, Spec::ry(t)),
            (format!("rz({t}) q[0];"), 1, Spec::rz(t)),
            (format!("cp({t}) q[0], q[1];"), 2, ctrl(&Spec::p(t))),
            (
                format!("cphase({t}) q[0], q[1];"),
                2,
                ctrl(&spec_u(0.0, 0.0, t)),
            ),
            (format!("crx({t}) q[0], q[1];"), 2, ctrl(&Spec::rx(t))),
            (format!("cry({t}) q[0], q[1];"), 2, ctrl(&Spec::ry(t))),
            (format!("crz({t}) q[0], q[1];"), 2, ctrl(&Spec::rz(t))),
            (
                format!("cu({t}, {f}, {l}, {g}) q[0], q[1];"),
                2,
                Spec::cu(t, f, l, g),
            ),
        ];
        for (statement, n, spec) in cases {
            assert_close(&statement, &read(n, &statement), &spec);
        }
    }
}

/// Decision 1: `U`, `u2` and `u3` are read with Qiskit's matrices, which
/// differ from the specification's by exactly e^{−iθ/2} (`U`) and
/// e^{+i(φ+λ)/2} (`u2`, `u3`). The factor is the predicted one, never fitted.
#[test]
fn u_u2_and_u3_differ_from_the_specification_by_the_declared_factors() {
    for (t, f, l, _) in DRAWS {
        let u = read(1, &format!("U({t}, {f}, {l}) q[0];"));
        assert_close("U", &u, &scale(phase(-t / 2.0), &spec_u(t, f, l)));
        assert_close("U is Qiskit's", &u, &qiskit_u(t, f, l));

        let u3 = read(1, &format!("u3({t}, {f}, {l}) q[0];"));
        assert_close("u3", &u3, &scale(phase((f + l) / 2.0), &Spec::u3(t, f, l)));
        assert_close("u3 is Qiskit's", &u3, &qiskit_u(t, f, l));

        let u2 = read(1, &format!("u2({f}, {l}) q[0];"));
        assert_close("u2", &u2, &scale(phase((f + l) / 2.0), &Spec::u2(f, l)));
        assert_close("u2 is Qiskit's", &u2, &qiskit_u(FRAC_PI_2, f, l));
    }
}

/// `CX` is read as `cx`, as `standard_library.rst` describes it ("an alias
/// for cx"). `stdgates.inc` defines it as `ctrl @ U(π, 0, π)`, which under
/// the specification's `U` is controlled-(iX): not `cx` up to any global
/// phase. The profile follows the prose; this pins the inconsistency.
#[test]
fn cx_is_read_as_cx_although_stdgates_defines_a_different_matrix() {
    let read_cx = read(2, "CX q[0], q[1];");
    assert_close("CX", &read_cx, &ctrl(&Spec::x()));
    let defined = Spec::cx_as_defined();
    // The controlled block is i·X: a relative phase i against the identity
    // block, which no global phase removes.
    assert!((defined[3][1] - c(0.0, 1.0)).norm() < TOLERANCE);
    assert!((defined[0][0] - c(1.0, 0.0)).norm() < TOLERANCE);
    assert!((read_cx[3][1] - c(1.0, 0.0)).norm() < TOLERANCE);
}

// ──────────────────────── Exported definitions ──────────────────────────

/// Each instruction `stdgates.inc` lacks, on permuted operands: exported, it
/// is a call of a gate the output defines; read back, that call has exactly
/// the instruction's matrix.
#[test]
fn every_exported_definition_reproduces_its_instruction_exactly() {
    let t = 0.37;
    let cases: Vec<(usize, G)> = vec![
        (
            2,
            G::Rzz {
                q0: 1,
                q1: 0,
                theta: Fixed(t),
            },
        ),
        (
            2,
            G::Rxx {
                q0: 1,
                q1: 0,
                theta: Fixed(-t),
            },
        ),
        (1, G::Sxdg(0)),
        (2, G::Csx(1, 0)),
        (
            2,
            G::Cu1 {
                q0: 1,
                q1: 0,
                theta: Fixed(t),
            },
        ),
        (
            2,
            G::Cu3 {
                control: 1,
                target: 0,
                theta: Fixed(0.3),
                phi: Fixed(-1.1),
                lam: Fixed(2.2),
            },
        ),
        (
            1,
            G::U0 {
                qubit: 0,
                gamma: Fixed(t),
            },
        ),
        (3, G::Rccx(2, 0, 1)),
        (4, G::Rc3x(3, 1, 0, 2)),
        (4, G::C3x(2, 3, 0, 1)),
        (4, G::C3sqrtx(1, 3, 2, 0)),
        (5, G::C4x(4, 0, 3, 1, 2)),
    ];
    for (n, gate) in cases {
        let native = ParameterizedCircuit::new(n).push(gate.clone());
        let exported = native.to_qasm3().unwrap();
        let read_back = ParameterizedCircuit::from_qasm3(&exported).unwrap();
        assert!(
            matches!(read_back.gates.as_slice(), [G::Custom(_)]),
            "{gate:?}"
        );
        assert_close(
            &format!("{gate:?}"),
            &circuit_unitary(&read_back),
            &circuit_unitary(&native),
        );
    }
}

/// The whole vocabulary in one circuit (measurements aside) keeps its
/// unitary through the OpenQASM 3 profile: Polypus reads its own `U`, `u2`
/// and `u3` back exactly.
#[test]
fn the_vocabulary_keeps_its_unitary_through_openqasm3() {
    let qc = ParameterizedCircuit::new(4)
        .h(0)
        .x(1)
        .y(2)
        .z(3)
        .s(0)
        .t(1)
        .sdg(2)
        .tdg(3)
        .id(0)
        .rx(0, 0.25)
        .ry(1, -0.4)
        .rz(2, 1.5)
        .u(3, 0.1, 0.2, 0.3)
        .cx(0, 1)
        .cz(1, 2)
        .swap(0, 2)
        .rzz(0, 3, 0.6)
        .rxx(1, 2, 2.0)
        .cp(0, 1, 0.75)
        .sx(1)
        .sxdg(2)
        .cy(2, 0)
        .ch(1, 3)
        .csx(0, 2)
        .ccx(2, 0, 1)
        .cswap(1, 3, 0)
        .crx(1, 0, 0.8)
        .cry(2, 1, -0.6)
        .crz(0, 3, 1.1)
        .cu1(2, 1, 0.9)
        .cu3(1, 0, 0.2, -0.4, 0.5)
        .cu(0, 2, 0.3, 0.7, -0.1, 0.9)
        .u0(3, 0.5)
        .rccx(3, 0, 2)
        .rc3x(3, 1, 2, 0)
        .c3x(2, 3, 0, 1)
        .c3sqrtx(0, 3, 1, 2)
        .p(3, -0.3)
        .u1(2, 0.45)
        .u2(1, -0.2, 0.6)
        .push(G::UGate {
            qubit: 0,
            theta: Fixed(0.1),
            phi: Fixed(-0.3),
            lam: Fixed(-0.7),
        });
    let read_back = ParameterizedCircuit::from_qasm3(&qc.to_qasm3().unwrap()).unwrap();
    assert_close(
        "vocabulary",
        &circuit_unitary(&read_back),
        &circuit_unitary(&qc),
    );
}

// ─────────────────────────── Cross-dialect ──────────────────────────────

#[test]
fn declarations_keep_their_matrices_across_dialects() {
    // OpenQASM 2.0 → 3: `ln`, `^`, `u` and a helper-spelled `cu1` in a body.
    let from_2 = ParameterizedCircuit::from_qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate g(t) a,b { rz(ln(t)^2) a; cu1(-t) a,b; u(t,0,pi) b; rzz(t/3) a,b; }\n\
         qreg q[2];\ng(0.5) q[0],q[1];\n",
    )
    .unwrap();
    let to_3 = ParameterizedCircuit::from_qasm3(&from_2.to_qasm3().unwrap()).unwrap();
    assert_close("2 → 3", &circuit_unitary(&to_3), &circuit_unitary(&from_2));

    // OpenQASM 3 → 2.0: `log`, `**`, `U`, `CX`, `phase`, `cphase`, `tau`,
    // `euler`, renamed identifiers.
    let from_3 = ParameterizedCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n\
         gate rzz(Theta) _a, b { cx _a, b; rz(Theta ** 2 / 2.0 + log(2.0)) b; cx _a, b; }\n\
         gate q(x) a { U(x, -x, tau) a; phase(euler) a; }\n\
         gate wrap(t) a, b { rzz(t) a, b; q(-t) b; CX a, b; cphase(t) a, b; }\n\
         qubit[2] r;\nwrap(0.125) r[0], r[1];\n",
    )
    .unwrap();
    let to_2 =
        ParameterizedCircuit::from_qasm2(&from_3.to_qasm2_with_params(&[]).unwrap()).unwrap();
    assert_close("3 → 2", &circuit_unitary(&to_2), &circuit_unitary(&from_3));
}

/// A native `rzz` and a declared `rzz` with another body, in one circuit:
/// each export keeps them apart, so the unitary survives.
#[test]
fn a_native_and_a_declared_rzz_keep_their_matrices() {
    let mut qc = ParameterizedCircuit::from_qasm3(
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n\
         gate rzz(t) a, b { rx(t) a; ry(t) b; }\nqubit[3] q;\nrzz(0.5) q[0], q[1];\n",
    )
    .unwrap();
    qc.try_push(G::Rzz {
        q0: 1,
        q1: 2,
        theta: Fixed(0.25),
    })
    .unwrap();
    let want = circuit_unitary(&qc);
    let via_3 = ParameterizedCircuit::from_qasm3(&qc.to_qasm3().unwrap()).unwrap();
    assert_close("OpenQASM 3", &circuit_unitary(&via_3), &want);
    let via_2 = ParameterizedCircuit::from_qasm2(&qc.to_qasm2_with_params(&[]).unwrap()).unwrap();
    assert_close("OpenQASM 2.0", &circuit_unitary(&via_2), &want);
}
