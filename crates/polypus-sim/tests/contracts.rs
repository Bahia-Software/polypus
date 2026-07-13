//! Simulator-side enforcement of the `polypus-circuit` contracts
//! (see `docs/CONTRACTS.md`):
//!
//! - **C-2 · Gate vocabulary symmetry (QIR half).** Each non-trivial QIR
//!   decomposition (`rzz`, `rxx`, `cp`, `u3`) must realise the *same unitary*
//!   as the native gate up to a global phase, with the native simulator as the
//!   reference. The decompositions here mirror `polypus_circuit`'s `qir.rs`.
//! - **C-4 · Terminal measurement placement (simulator half).** The simulator
//!   rejects a circuit that operates on an already-measured qubit rather than
//!   silently treating the measurement as a no-op.

use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
use polypus_sim::{SimError, Simulator, Statevector, StatevectorSimulator, C64};
use std::f64::consts::{FRAC_PI_3, PI};

// ─────────────────── C-2 · QIR-vs-simulator equivalence ───────────────────

/// Column `k` of the unitary realised by `gates` on `n` qubits: prepare the
/// basis state |k> (via `x` gates), evolve, read the amplitudes.
fn column(n: usize, k: usize, gates: &[G]) -> Vec<C64> {
    let mut sv = Statevector::new(n).unwrap();
    for q in 0..n {
        if (k >> q) & 1 == 1 {
            sv.apply(&G::X(q)).unwrap();
        }
    }
    for g in gates {
        sv.apply(g).unwrap();
    }
    sv.amplitudes().to_vec()
}

/// Assert two gate sequences realise the same `n`-qubit unitary up to one
/// global phase (tolerance 1e-12). The native gate is the reference (C-2).
fn assert_equiv_up_to_global_phase(n: usize, native: &[G], decomposed: &[G]) {
    let dim = 1usize << n;
    let mut u_native = Vec::with_capacity(dim * dim);
    let mut u_decomp = Vec::with_capacity(dim * dim);
    for k in 0..dim {
        u_native.extend(column(n, k, native));
        u_decomp.extend(column(n, k, decomposed));
    }
    // Pin the global phase from the first sizeable native amplitude.
    let i = u_native
        .iter()
        .position(|z| z.norm() > 1e-9)
        .expect("native unitary is entirely zero");
    let phase = u_decomp[i] / u_native[i];
    assert!(
        (phase.norm() - 1.0).abs() < 1e-9,
        "phase factor is not unit-modulus ({})",
        phase.norm()
    );
    for (a, b) in u_native.iter().zip(&u_decomp) {
        assert!(
            (b - a * phase).norm() < 1e-12,
            "decomposition differs from the native gate by more than a global phase"
        );
    }
}

const ANGLES: [f64; 5] = [0.3, 0.7, 1.25, -2.0, FRAC_PI_3];

#[test]
fn c2_qir_rzz_decomposition_matches_native() {
    // rzz(θ) = cnot · rz(θ) · cnot
    for &t in &ANGLES {
        let native = [G::Rzz {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t),
            },
            G::Cx(0, 1),
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_rxx_decomposition_matches_native() {
    // rxx(θ) = (h⊗h) · cnot · rz(θ) · cnot · (h⊗h)
    for &t in &ANGLES {
        let native = [G::Rxx {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::H(0),
            G::H(1),
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t),
            },
            G::Cx(0, 1),
            G::H(0),
            G::H(1),
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_cp_decomposition_matches_native() {
    // cp(θ) = rz(θ/2) q0; cnot; rz(−θ/2) q1; cnot; rz(θ/2) q1
    // (audit item C3: the old `cz; rz; cz` collapsed to rz(θ) and was wrong).
    for &t in &ANGLES {
        let native = [G::Cp {
            q0: 0,
            q1: 1,
            theta: Fixed(t),
        }];
        let decomp = [
            G::Rz {
                qubit: 0,
                theta: Fixed(t / 2.0),
            },
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(-t / 2.0),
            },
            G::Cx(0, 1),
            G::Rz {
                qubit: 1,
                theta: Fixed(t / 2.0),
            },
        ];
        assert_equiv_up_to_global_phase(2, &native, &decomp);
    }
}

#[test]
fn c2_qir_u3_decomposition_matches_native() {
    // u3(θ,φ,λ) applied left-to-right is rz(λ), ry(θ), rz(φ).
    for &(th, ph, la) in &[(0.1, 0.2, 0.3), (1.0, -0.5, 2.0), (PI / 2.0, 0.0, PI)] {
        let native = [G::U {
            qubit: 0,
            theta: Fixed(th),
            phi: Fixed(ph),
            lam: Fixed(la),
        }];
        let decomp = [
            G::Rz {
                qubit: 0,
                theta: Fixed(la),
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(th),
            },
            G::Rz {
                qubit: 0,
                theta: Fixed(ph),
            },
        ];
        assert_equiv_up_to_global_phase(1, &native, &decomp);
    }
}

// ─────────────────────── C-4 · terminal measurement ───────────────────────

#[test]
fn c4_simulator_rejects_gate_after_measure() {
    let cc = ConcreteCircuit {
        num_qubits: 1,
        gates: vec![G::Measure { qubit: 0, cbit: 0 }, G::X(0)],
    };
    let err = StatevectorSimulator::new().run(&cc).unwrap_err();
    assert_eq!(err, SimError::GateAfterMeasure { qubit: 0 });
}

#[test]
fn c4_simulator_accepts_terminal_measurement() {
    let cc = ConcreteCircuit {
        num_qubits: 2,
        gates: vec![G::H(0), G::Cx(0, 1), G::MeasureAll],
    };
    let sv = StatevectorSimulator::new().run(&cc).unwrap();
    assert!((sv.norm() - 1.0).abs() < 1e-12);
}
