//! Adjoint differentiation (issues #503, #561): `StatevectorSimulator::adjoint_gradient`
//! against two independent references computed with `Simulator::run` — the
//! parameter-shift rule, gate by gate, for the Pauli rotations, and central
//! finite differences for every angle it differentiates — over every gate and
//! angle it differentiates, every instruction it steps back through, angles
//! shared by several gates, both kernel paths, and its typed errors.

use polypus_circuit::{
    ConcreteCircuit, GateInstruction as G, GateParam, GateParam::Fixed, GateParam::Param,
    ParameterizedCircuit,
};
use polypus_sim::{
    AdjointGradient, AngleRef, SimError, Simulator, SplitMix64, StatevectorSimulator, MAX_QUBITS,
};
use std::f64::consts::FRAC_PI_2;

/// Agreement with the references. The adjoint and the shift rule are both
/// exact, so they agree to rounding (~1e-15 here); central differences with
/// `FD_STEP` carry a truncation error of about `h²/6 · |E'''| ≈ 2e-11` (each
/// angle enters as `a + b·cos α + c·sin α`, with `|b|, |c| ≤ ‖O‖ ≤ 1`) and a
/// rounding error of about `ε/h ≈ 2e-11`, well inside it.
const TOLERANCE: f64 = 1e-9;
const FD_STEP: f64 = 1e-5;

fn simulator() -> StatevectorSimulator {
    StatevectorSimulator::new()
}

/// A diagonal observable with a pseudo-random weight in `[-1, 1)` on every
/// basis state (a table, which the test may afford; the simulator never
/// builds one).
fn random_weights(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..1usize << n)
        .map(|_| 2.0 * rng.next_f64() - 1.0)
        .collect()
}

/// `⟨O⟩` from the exact probabilities of `Simulator::run`: the reference
/// expectation, computed with none of the adjoint's machinery.
fn expectation(circuit: &ConcreteCircuit, weights: &[f64]) -> f64 {
    let sv = simulator().run(circuit).expect("a valid circuit");
    sv.probabilities()
        .iter()
        .zip(weights)
        .map(|(p, w)| p * w)
        .sum()
}

/// The angles of `gate` in `AngleRef`'s documented order: empty for a gate
/// with none, and for the call of a declared gate, which the adjoint does not
/// differentiate. Written out here, independently of the simulator; a kind
/// this list does not name panics, so a gate added to `polypus-circuit` fails
/// `every_instruction_kind_is_differentiated_through` until it is classified.
fn angles_mut(gate: &mut G) -> Vec<&mut GateParam> {
    match gate {
        G::Rx { theta, .. }
        | G::Ry { theta, .. }
        | G::Rz { theta, .. }
        | G::Rxx { theta, .. }
        | G::Rzz { theta, .. }
        | G::Crx { theta, .. }
        | G::Cry { theta, .. }
        | G::Crz { theta, .. }
        | G::Cp { theta, .. }
        | G::Cu1 { theta, .. } => vec![theta],
        G::P { lam, .. } | G::U1 { lam, .. } => vec![lam],
        G::U0 { gamma, .. } => vec![gamma],
        G::U {
            theta, phi, lam, ..
        }
        | G::UGate {
            theta, phi, lam, ..
        }
        | G::Cu3 {
            theta, phi, lam, ..
        } => vec![theta, phi, lam],
        G::U2 { phi, lam, .. } => vec![phi, lam],
        G::Cu {
            theta,
            phi,
            lam,
            gamma,
            ..
        } => vec![theta, phi, lam, gamma],
        G::H(_)
        | G::X(_)
        | G::Y(_)
        | G::Z(_)
        | G::S(_)
        | G::T(_)
        | G::Sdg(_)
        | G::Tdg(_)
        | G::Id(_)
        | G::Sx(_)
        | G::Sxdg(_)
        | G::Cx(..)
        | G::Cz(..)
        | G::Swap(..)
        | G::Cy(..)
        | G::Ch(..)
        | G::Csx(..)
        | G::Ccx(..)
        | G::Cswap(..)
        | G::Rccx(..)
        | G::Rc3x(..)
        | G::C3x(..)
        | G::C3sqrtx(..)
        | G::C4x(..)
        | G::Custom(_)
        | G::Barrier(_)
        | G::Measure { .. }
        | G::MeasureAll => vec![],
        other => panic!("{other:?} is not classified by this test"),
    }
}

fn num_angles(gate: &G) -> usize {
    angles_mut(&mut gate.clone()).len()
}

fn is_pauli_rotation(gate: &G) -> bool {
    matches!(
        gate,
        G::Rx { .. } | G::Ry { .. } | G::Rz { .. } | G::Rxx { .. } | G::Rzz { .. }
    )
}

/// `gate` with its angle `angle` moved by `delta`.
fn shifted(gate: &G, angle: usize, delta: f64) -> G {
    let mut gate = gate.clone();
    let mut angles = angles_mut(&mut gate);
    let param = angles
        .get_mut(angle)
        .unwrap_or_else(|| panic!("no angle {angle}"));
    match **param {
        Fixed(v) => **param = Fixed(v + delta),
        other => panic!("the reference shifts fixed angles only, not {other:?}"),
    }
    gate
}

/// `circuit` with the angles at `angles` all moved by `delta`.
fn moved(circuit: &ConcreteCircuit, angles: &[AngleRef], delta: f64) -> ConcreteCircuit {
    let mut gates = circuit.gates().to_vec();
    for at in angles {
        gates[at.position] = shifted(&gates[at.position], at.angle, delta);
    }
    ConcreteCircuit::try_new(circuit.num_qubits(), gates).expect("a valid circuit")
}

/// The parameter-shift derivative of a Pauli rotation's angle.
fn shift_rule(circuit: &ConcreteCircuit, at: AngleRef, weights: &[f64]) -> f64 {
    assert!(is_pauli_rotation(&circuit.gates()[at.position]));
    let plus = expectation(&moved(circuit, &[at], FRAC_PI_2), weights);
    let minus = expectation(&moved(circuit, &[at], -FRAC_PI_2), weights);
    (plus - minus) / 2.0
}

/// The central-difference derivative of `⟨O⟩` when the angles at `angles`
/// move together (one angle they share).
fn central_difference(circuit: &ConcreteCircuit, angles: &[AngleRef], weights: &[f64]) -> f64 {
    let plus = expectation(&moved(circuit, angles, FD_STEP), weights);
    let minus = expectation(&moved(circuit, angles, -FD_STEP), weights);
    (plus - minus) / (2.0 * FD_STEP)
}

fn adjoint(circuit: &ConcreteCircuit, wrt: &[AngleRef], weights: &[f64]) -> AdjointGradient {
    simulator()
        .adjoint_gradient(circuit, wrt, |b| weights[b], None)
        .expect("a differentiable circuit")
}

/// The angle of every `Rx`/`Ry`/`Rz`/`Rxx`/`Rzz` in `circuit`.
fn rotation_angles(circuit: &ConcreteCircuit) -> Vec<AngleRef> {
    circuit
        .gates()
        .iter()
        .enumerate()
        .filter(|(_, g)| is_pauli_rotation(g))
        .map(|(p, _)| AngleRef::new(p, 0))
        .collect()
}

/// Every angle of every gate of `circuit` the adjoint differentiates.
fn every_angle(circuit: &ConcreteCircuit) -> Vec<AngleRef> {
    circuit
        .gates()
        .iter()
        .enumerate()
        .flat_map(|(p, g)| (0..num_angles(g)).map(move |a| AngleRef::new(p, a)))
        .collect()
}

/// Every derivative against central differences, and the Pauli rotations'
/// also against the shift rule; the expectation against the probabilities.
fn assert_matches_references(circuit: &ConcreteCircuit, weights: &[f64], what: &str) {
    let wrt = every_angle(circuit);
    assert!(!wrt.is_empty(), "{what}: nothing to differentiate");
    let grad = adjoint(circuit, &wrt, weights);
    let reference = expectation(circuit, weights);
    assert!(
        (grad.expectation - reference).abs() < 1e-12,
        "{what}: expectation {} vs {reference}",
        grad.expectation
    );
    for (&at, &d) in wrt.iter().zip(&grad.derivatives) {
        let gate = &circuit.gates()[at.position];
        if is_pauli_rotation(gate) {
            let shift = shift_rule(circuit, at, weights);
            assert!(
                (d - shift).abs() < TOLERANCE,
                "{what}: {at:?} adjoint {d} vs shift {shift}"
            );
        }
        let fd = central_difference(circuit, &[at], weights);
        assert!(
            (d - fd).abs() < TOLERANCE,
            "{what}: {at:?} ({gate:?}) adjoint {d} vs finite difference {fd}"
        );
    }
}

/// The kinds of [`random_rotation`]: the five Pauli rotations, the gates only
/// the adjoint differentiates, and both.
const PAULI: [u64; 5] = [0, 1, 2, 3, 4];
const ADJOINT_ONLY: [u64; 13] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
const EVERY_KIND: [u64; 18] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

/// A differentiable gate of kind `kind` at random angles: `0..5` the Pauli
/// rotations `Rx, Ry, Rz, Rxx, Rzz`, then `Crx, Cry, Crz, Cp, Cu1, P, U1, U,
/// UGate, U2, Cu3, Cu, U0`.
fn random_rotation(kind: u64, n: usize, rng: &mut SplitMix64) -> G {
    let q = (rng.next_u64() % n as u64) as usize;
    let r = (q + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
    let theta = Fixed(6.0 * rng.next_f64() - 3.0);
    // Only the kinds with more than one angle draw more, so the circuits of
    // the one-angle kinds are the ones these tests always built.
    let mut angle = || Fixed(6.0 * rng.next_f64() - 3.0);
    match kind {
        0 => G::Rx { qubit: q, theta },
        1 => G::Ry { qubit: q, theta },
        2 => G::Rz { qubit: q, theta },
        3 => G::Rxx {
            q0: q,
            q1: r,
            theta,
        },
        4 => G::Rzz {
            q0: q,
            q1: r,
            theta,
        },
        5 => G::Crx {
            control: q,
            target: r,
            theta,
        },
        6 => G::Cry {
            control: q,
            target: r,
            theta,
        },
        7 => G::Crz {
            control: q,
            target: r,
            theta,
        },
        8 => G::Cp {
            q0: q,
            q1: r,
            theta,
        },
        9 => G::Cu1 {
            q0: q,
            q1: r,
            theta,
        },
        10 => G::P {
            qubit: q,
            lam: theta,
        },
        11 => G::U1 {
            qubit: q,
            lam: theta,
        },
        12 => G::U {
            qubit: q,
            theta,
            phi: angle(),
            lam: angle(),
        },
        13 => G::UGate {
            qubit: q,
            theta,
            phi: angle(),
            lam: angle(),
        },
        14 => G::U2 {
            qubit: q,
            phi: theta,
            lam: angle(),
        },
        15 => G::Cu3 {
            control: q,
            target: r,
            theta,
            phi: angle(),
            lam: angle(),
        },
        16 => G::Cu {
            control: q,
            target: r,
            theta,
            phi: angle(),
            lam: angle(),
            gamma: angle(),
        },
        _ => G::U0 {
            qubit: q,
            gamma: theta,
        },
    }
}

/// A random fixed (not differentiated) gate, from a pool that mixes dense,
/// diagonal, controlled, composite and parameterised-but-fixed gates.
fn random_fixed(n: usize, rng: &mut SplitMix64) -> G {
    let q = (rng.next_u64() % n as u64) as usize;
    let r = (q + 1) % n;
    let s = (q + 2) % n;
    let a = 6.0 * rng.next_f64() - 3.0;
    match rng.next_u64() % 14 {
        0 => G::H(q),
        1 => G::Sx(q),
        2 => G::T(q),
        3 => G::Sdg(q),
        4 => G::Cx(q, r),
        5 => G::Cz(q, r),
        6 => G::Swap(q, r),
        7 => G::Ch(q, r),
        8 => G::Csx(q, r),
        9 => G::Ccx(q, r, s),
        10 => G::U {
            qubit: q,
            theta: Fixed(a),
            phi: Fixed(0.3 * a),
            lam: Fixed(-0.7 * a),
        },
        11 => G::Cp {
            q0: q,
            q1: r,
            theta: Fixed(a),
        },
        12 => G::Cry {
            control: q,
            target: r,
            theta: Fixed(a),
        },
        _ => G::Rccx(q, r, s),
    }
}

/// `layers` × (one random fixed gate + one differentiable gate of a kind drawn
/// from `kinds`), after a layer of `H`s so the state is generic from the
/// start.
fn random_circuit(n: usize, layers: usize, kinds: &[u64], seed: u64) -> ConcreteCircuit {
    let mut rng = SplitMix64::new(seed);
    let mut gates: Vec<G> = (0..n).map(G::H).collect();
    for _ in 0..layers {
        gates.push(random_fixed(n, &mut rng));
        let kind = kinds[(rng.next_u64() % kinds.len() as u64) as usize];
        gates.push(random_rotation(kind, n, &mut rng));
    }
    gates.push(G::MeasureAll);
    ConcreteCircuit::try_new(n, gates).expect("a valid circuit")
}

/// Acceptance criterion 1: each of the five rotations, alone among fixed gates,
/// matches the shift rule and central differences.
#[test]
fn each_rotation_matches_the_shift_rule_and_finite_differences() {
    let n = 4;
    for kind in PAULI {
        for seed in 0..4 {
            let circuit = random_circuit(n, 12, &[kind], 100 * kind + seed);
            let weights = random_weights(n, seed);
            assert_matches_references(&circuit, &weights, &format!("kind {kind} seed {seed}"));
        }
    }
}

#[test]
fn mixed_rotations_on_a_wider_circuit_match_the_references() {
    let n = 6;
    for seed in 0..3 {
        let circuit = random_circuit(n, 30, &PAULI, 7_000 + seed);
        let weights = random_weights(n, 50 + seed);
        assert_matches_references(&circuit, &weights, &format!("seed {seed}"));
    }
}

/// `gate` between two layers that make every qubit a generic superposition —
/// before it, so a control is in superposition and a phase relative to the
/// control (`Cu`'s `γ`, the phase gates) is not a global one; after it, so
/// the observable sees those phases.
fn sandwiched(gate: G, n: usize, seed: u64) -> ConcreteCircuit {
    let mut rng = SplitMix64::new(seed);
    let layer = |rng: &mut SplitMix64| -> Vec<G> {
        (0..n)
            .flat_map(|q| {
                [
                    G::H(q),
                    G::Ry {
                        qubit: q,
                        theta: Fixed(2.0 * rng.next_f64() - 1.0),
                    },
                    G::Rz {
                        qubit: q,
                        theta: Fixed(2.0 * rng.next_f64() - 1.0),
                    },
                ]
            })
            .collect()
    };
    let mut gates = layer(&mut rng);
    gates.push(gate);
    gates.extend(layer(&mut rng));
    gates.push(G::MeasureAll);
    ConcreteCircuit::try_new(n, gates).expect("a valid circuit")
}

/// Acceptance criterion 1 (#561): every gate only the adjoint differentiates,
/// alone, with its control in superposition, for each of its angles, matches
/// central differences. Each derivative but `u0`'s is far from zero, so none
/// passes by both sides being `0` (`γ` and the phases included).
#[test]
fn each_adjoint_only_gate_matches_finite_differences_for_every_angle() {
    let n = 3;
    for kind in ADJOINT_ONLY {
        for seed in 0..3 {
            let mut rng = SplitMix64::new(1_000 * kind + seed);
            let gate = random_rotation(kind, n, &mut rng);
            let circuit = sandwiched(gate.clone(), n, seed);
            let position = 3 * n;
            let weights = random_weights(n, 40 + seed);
            let wrt: Vec<AngleRef> = (0..num_angles(&gate))
                .map(|a| AngleRef::new(position, a))
                .collect();
            let grad = adjoint(&circuit, &wrt, &weights);
            for (&at, &d) in wrt.iter().zip(&grad.derivatives) {
                let fd = central_difference(&circuit, &[at], &weights);
                assert!(
                    (d - fd).abs() < TOLERANCE,
                    "{gate:?} angle {}: adjoint {d} vs finite difference {fd}",
                    at.angle
                );
                if matches!(gate, G::U0 { .. }) {
                    assert_eq!(d.to_bits(), 0.0f64.to_bits(), "u0 does not depend on γ");
                } else {
                    assert!(
                        d.abs() > 1e-4,
                        "{gate:?} angle {}: {d} is vacuous",
                        at.angle
                    );
                }
            }
        }
    }
}

/// Acceptance criterion 1 (#561): the adjoint-only gates mixed with the Pauli
/// rotations and fixed gates, over several seeds, every angle of every
/// differentiable gate against the references.
#[test]
fn adjoint_only_gates_in_random_circuits_match_the_references() {
    let n = 4;
    for seed in 0..4 {
        let circuit = random_circuit(n, 16, &EVERY_KIND, 9_000 + seed);
        let weights = random_weights(n, 60 + seed);
        assert_matches_references(&circuit, &weights, &format!("seed {seed}"));
    }
}

/// A `Z`-parity observable (the readouts of `polypus-qml`): the weight is a
/// function of the basis state, never a table.
#[test]
fn a_pauli_z_product_observable_matches_the_references() {
    let n = 5;
    let circuit = random_circuit(n, 20, &PAULI, 31);
    let parity = |b: usize| {
        let z0z2 = if (b & 0b101).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let z4 = if b & 0b1_0000 == 0 { 1.0 } else { -1.0 };
        0.5 * z0z2 - 1.5 * z4
    };
    let weights: Vec<f64> = (0..1usize << n).map(parity).collect();
    assert_matches_references(&circuit, &weights, "Z products");
    let wrt = every_angle(&circuit);
    let from_function = simulator()
        .adjoint_gradient(&circuit, &wrt, parity, None)
        .expect("differentiable");
    assert_eq!(from_function, adjoint(&circuit, &wrt, &weights));
}

/// One angle read by three gates (as a convolution reuses its parameters): the
/// derivative of the shared angle is the sum of the three gates' derivatives,
/// and each of those is its own gate's shift derivative.
#[test]
fn rotations_sharing_an_angle_add_up_to_its_derivative() {
    let n = 3;
    let theta = Fixed(0.7);
    let circuit = ConcreteCircuit::try_new(
        n,
        vec![
            G::H(0),
            G::H(1),
            G::Ry { qubit: 0, theta },
            G::Cx(0, 1),
            G::Rzz {
                q0: 0,
                q1: 1,
                theta,
            },
            G::Sx(2),
            G::Rx { qubit: 2, theta },
            G::Ccx(2, 0, 1),
            G::Rxx {
                q0: 1,
                q1: 2,
                theta,
            },
            G::MeasureAll,
        ],
    )
    .expect("a valid circuit");
    let weights = random_weights(n, 9);
    let shared: Vec<AngleRef> = [2, 4, 6, 8].map(|p| AngleRef::new(p, 0)).to_vec();
    let grad = adjoint(&circuit, &shared, &weights);
    for (&at, &d) in shared.iter().zip(&grad.derivatives) {
        assert!((d - shift_rule(&circuit, at, &weights)).abs() < TOLERANCE);
    }
    let total: f64 = grad.derivatives.iter().sum();
    let fd = central_difference(&circuit, &shared, &weights);
    assert!((total - fd).abs() < TOLERANCE, "sum {total} vs {fd}");
    // Not just the first occurrence: every gate contributes.
    let first_only = grad.derivatives[0];
    assert!(
        (first_only - fd).abs() > 1e-3,
        "the gates' derivatives must all matter"
    );
}

/// Acceptance criterion 2 (#561): one angle read by several adjoint-only gates
/// — and by angles of different index within them (`U`'s `φ`, `Cu`'s `γ`,
/// `U2`'s `λ`) — adds up to the derivative of moving them all together.
#[test]
fn adjoint_only_gates_sharing_an_angle_add_up_to_its_derivative() {
    let n = 3;
    let a = 0.7;
    let circuit = ConcreteCircuit::try_new(
        n,
        vec![
            G::H(0),
            G::H(1),
            G::H(2),
            G::Cry {
                control: 0,
                target: 1,
                theta: Fixed(a),
            },
            G::U {
                qubit: 2,
                theta: Fixed(0.4),
                phi: Fixed(a),
                lam: Fixed(-0.3),
            },
            G::Cx(2, 0),
            G::Cu {
                control: 1,
                target: 0,
                theta: Fixed(1.1),
                phi: Fixed(-0.5),
                lam: Fixed(0.2),
                gamma: Fixed(a),
            },
            G::Ry {
                qubit: 1,
                theta: Fixed(a),
            },
            G::Cp {
                q0: 2,
                q1: 1,
                theta: Fixed(a),
            },
            G::U2 {
                qubit: 0,
                phi: Fixed(0.9),
                lam: Fixed(a),
            },
            G::H(1),
            G::MeasureAll,
        ],
    )
    .expect("a valid circuit");
    let weights = random_weights(n, 21);
    let shared = [
        AngleRef::new(3, 0),
        AngleRef::new(4, 1),
        AngleRef::new(6, 3),
        AngleRef::new(7, 0),
        AngleRef::new(8, 0),
        AngleRef::new(9, 1),
    ];
    let grad = adjoint(&circuit, &shared, &weights);
    for (&at, &d) in shared.iter().zip(&grad.derivatives) {
        let fd = central_difference(&circuit, &[at], &weights);
        assert!((d - fd).abs() < TOLERANCE, "{at:?}: {d} vs {fd}");
    }
    let total: f64 = grad.derivatives.iter().sum();
    let fd = central_difference(&circuit, &shared, &weights);
    assert!((total - fd).abs() < TOLERANCE, "sum {total} vs {fd}");
    for (i, &d) in grad.derivatives.iter().enumerate() {
        assert!(
            (d - fd).abs() > 1e-3,
            "gate {i} alone must not be the whole derivative"
        );
    }
}

/// A rotation on every qubit, then one instruction of every kind but `Measure`
/// (rejected; see below), each followed by a rotation, then `MeasureAll`.
fn every_kind_circuit() -> ConcreteCircuit {
    let declared = ParameterizedCircuit::from_qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate g(t) a,b,c { rz(t) a; cx a,b; sx b; rccx a,b,c; }\nqreg q[3];\ng(0.3) q[1],q[0],q[2];\n",
    )
    .expect("a valid declared gate")
    .gates()[0]
        .clone();
    let kinds = ParameterizedCircuit::new(5)
        .h(0)
        .x(1)
        .y(2)
        .z(0)
        .s(1)
        .t(2)
        .sdg(0)
        .tdg(1)
        .id(2)
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
        .csx(0, 2)
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
        .rc3x(3, 1, 4, 0)
        .c3x(2, 4, 0, 3)
        .c3sqrtx(4, 3, 1, 2)
        .c4x(1, 4, 0, 3, 2)
        .p(4, 0.2)
        .u1(3, 0.45)
        .u2(2, -0.2, 0.6)
        .push(G::UGate {
            qubit: 1,
            theta: Fixed(0.1),
            phi: Fixed(0.2),
            lam: Fixed(-0.7),
        })
        .push(declared)
        .barrier()
        .assign_parameters(&[])
        .expect("every angle is fixed");
    let mut gates: Vec<G> = (0..5)
        .map(|q| G::Ry {
            qubit: q,
            theta: Fixed(0.3 + 0.2 * q as f64),
        })
        .collect();
    for (i, gate) in kinds.gates().iter().enumerate() {
        gates.push(gate.clone());
        gates.push(G::Rx {
            qubit: i % 5,
            theta: Fixed(0.1 * i as f64 - 1.0),
        });
    }
    gates.push(G::MeasureAll);
    ConcreteCircuit::try_new(5, gates).expect("a valid circuit")
}

/// Acceptance criteria 2 (#503) and 3 (#561): the sweep back steps through the
/// inverse of every instruction kind, and every angle of every kind that has
/// one — all but the call of a declared gate — is differentiated, matching the
/// shift rule (Pauli rotations) or central differences (the rest). The kind
/// list must be complete, so a gate added to `polypus-circuit` fails here until
/// the adjoint supports it.
#[test]
fn every_instruction_kind_is_differentiated_through() {
    let circuit = every_kind_circuit();
    let mut seen = [false; G::KINDS];
    for gate in circuit.gates() {
        seen[gate.kind_index()] = true;
    }
    let wrt = every_angle(&circuit);
    let mut differentiated = [false; G::KINDS];
    for at in &wrt {
        differentiated[circuit.gates()[at.position].kind_index()] = true;
    }
    let weights = random_weights(5, 77);
    let grad = adjoint(&circuit, &wrt, &weights);
    for (&at, &d) in wrt.iter().zip(&grad.derivatives) {
        let gate = &circuit.gates()[at.position];
        let reference = if is_pauli_rotation(gate) {
            shift_rule(&circuit, at, &weights)
        } else {
            central_difference(&circuit, &[at], &weights)
        };
        assert!(
            (d - reference).abs() < TOLERANCE,
            "{at:?} ({gate:?}): adjoint {d} vs reference {reference}"
        );
    }
    // Every kind with an angle is differentiated: the 18 of the table, and
    // nothing else (the call of a declared gate is `NotDifferentiable`).
    let angle_kinds: Vec<usize> = (0..G::KINDS).filter(|&k| differentiated[k]).collect();
    assert_eq!(
        angle_kinds.len(),
        18,
        "kinds differentiated: {angle_kinds:?}"
    );

    // `Measure` is the one kind the adjoint refuses: the read-out is the full
    // register.
    let measured = ConcreteCircuit::try_new(
        2,
        vec![
            G::Ry {
                qubit: 0,
                theta: Fixed(0.2),
            },
            G::Measure { qubit: 1, cbit: 0 },
        ],
    )
    .expect("a valid circuit");
    seen[measured.gates()[1].kind_index()] = true;
    assert_eq!(
        simulator().adjoint_gradient(&measured, &[AngleRef::new(0, 0)], |_| 1.0, None),
        Err(SimError::PartialMeasurement { qubit: 1 })
    );

    let missing: Vec<usize> = (0..G::KINDS).filter(|&k| !seen[k]).collect();
    assert!(
        missing.is_empty(),
        "instruction kinds not exercised: {missing:?}"
    );
}

/// The derivatives follow `wrt`'s order, and an angle named twice gets its
/// derivative twice — also two angles of one gate, each its own.
#[test]
fn derivatives_follow_the_order_of_wrt() {
    let circuit = random_circuit(4, 10, &EVERY_KIND, 5);
    let weights = random_weights(4, 6);
    let wrt = every_angle(&circuit);
    let sorted = adjoint(&circuit, &wrt, &weights);
    let mut reversed_wrt = wrt.clone();
    reversed_wrt.reverse();
    reversed_wrt.push(wrt[2]);
    let reversed = adjoint(&circuit, &reversed_wrt, &weights);
    let mut expected: Vec<f64> = sorted.derivatives.iter().rev().copied().collect();
    expected.push(sorted.derivatives[2]);
    assert_eq!(reversed.derivatives, expected);
    assert_eq!(reversed.expectation, sorted.expectation);

    // One gate, its angles interleaved and repeated: each is its own.
    let u = ConcreteCircuit::try_new(
        1,
        vec![
            G::H(0),
            G::U {
                qubit: 0,
                theta: Fixed(0.8),
                phi: Fixed(-0.4),
                lam: Fixed(1.3),
            },
            G::Rx {
                qubit: 0,
                theta: Fixed(0.5),
            },
        ],
    )
    .expect("a valid circuit");
    let w = [0.25, -0.75];
    let each = adjoint(&u, &[0, 1, 2].map(|a| AngleRef::new(1, a)), &w);
    let mixed = adjoint(&u, &[2, 0, 2, 1, 0].map(|a| AngleRef::new(1, a)), &w);
    let d = &each.derivatives;
    assert_eq!(mixed.derivatives, vec![d[2], d[0], d[2], d[1], d[0]]);
    assert!(d[0] != d[1] && d[1] != d[2] && d[0] != d[2], "{d:?}");
}

#[test]
fn an_empty_wrt_gives_the_expectation_alone() {
    let circuit = random_circuit(3, 6, &[1], 2);
    let weights = random_weights(3, 3);
    let grad = adjoint(&circuit, &[], &weights);
    assert!(grad.derivatives.is_empty());
    assert!((grad.expectation - expectation(&circuit, &weights)).abs() < 1e-12);
}

/// The circuit is applied as written whatever the `fusion` flag says, and a
/// terminal `MeasureAll` reads the same as no measurement.
#[test]
fn neither_fusion_nor_a_terminal_measure_all_changes_the_result() {
    let circuit = random_circuit(4, 16, &EVERY_KIND, 12);
    let weights = random_weights(4, 13);
    let wrt = every_angle(&circuit);
    let fused = simulator()
        .adjoint_gradient(&circuit, &wrt, |b| weights[b], None)
        .expect("differentiable");
    let unfused = StatevectorSimulator {
        fusion: false,
        ..simulator()
    }
    .adjoint_gradient(&circuit, &wrt, |b| weights[b], None)
    .expect("differentiable");
    assert_eq!(fused, unfused);

    let mut gates = circuit.gates().to_vec();
    assert_eq!(gates.pop(), Some(G::MeasureAll));
    let unmeasured = ConcreteCircuit::try_new(4, gates).expect("a valid circuit");
    assert_eq!(adjoint(&unmeasured, &wrt, &weights), fused);
}

/// Acceptance criterion 4: the parallel kernels and reductions give the same
/// bits as the sequential ones, for any number of threads — the Pauli
/// rotations' kernel and the generator kernel of the other gates (#561).
#[cfg(feature = "parallel")]
#[test]
fn parallel_and_sequential_paths_are_bit_identical() {
    let n = 14;
    for (kinds, seed) in [(&PAULI[..], 2_024), (&EVERY_KIND[..], 2_025)] {
        let circuit = random_circuit(n, 24, kinds, seed);
        let wrt = every_angle(&circuit);
        let weight = |b: usize| ((b % 11) as f64 - 5.0) / 5.0;
        let run = |threshold: usize| {
            StatevectorSimulator {
                parallel_threshold: threshold,
                ..simulator()
            }
            .adjoint_gradient(&circuit, &wrt, weight, None)
            .expect("differentiable")
        };
        let bits = |g: &AdjointGradient| {
            let mut v = vec![g.expectation.to_bits()];
            v.extend(g.derivatives.iter().map(|d| d.to_bits()));
            v
        };
        let sequential = bits(&run(MAX_QUBITS + 1));
        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a small thread pool");
            let parallel = pool.install(|| bits(&run(0)));
            assert_eq!(parallel, sequential, "seed {seed}, {threads} threads");
        }
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// The qubit ceiling and every structural check come before the `2^n`
/// allocation: on a circuit at the ceiling (16 GiB) each error is immediate.
#[test]
fn the_checks_run_before_anything_is_allocated() {
    let wide = |gates: Vec<G>| ConcreteCircuit::new_unchecked(MAX_QUBITS, gates);
    let ry = G::Ry {
        qubit: 0,
        theta: Fixed(0.1),
    };
    let at = |position: usize, angle: usize| AngleRef::new(position, angle);
    let capped = StatevectorSimulator {
        max_qubits: 10,
        ..simulator()
    };
    assert_eq!(
        capped.adjoint_gradient(&wide(vec![ry.clone()]), &[at(0, 0)], |_| 1.0, None),
        Err(SimError::TooManyQubits {
            requested: MAX_QUBITS,
            max: 10
        })
    );
    let above = ConcreteCircuit::new_unchecked(MAX_QUBITS + 1, vec![]);
    assert_eq!(
        simulator().adjoint_gradient(&above, &[], |_| 1.0, None),
        Err(SimError::TooManyQubits {
            requested: MAX_QUBITS + 1,
            max: MAX_QUBITS
        })
    );
    let u = |phi: GateParam| G::U {
        qubit: 1,
        theta: Fixed(0.4),
        phi,
        lam: Fixed(0.2),
    };
    let u2 = G::U2 {
        qubit: 0,
        phi: Fixed(0.1),
        lam: Fixed(0.2),
    };
    let cases: Vec<(Vec<G>, Vec<AngleRef>, SimError)> = vec![
        (
            vec![ry.clone(), G::Measure { qubit: 3, cbit: 0 }],
            vec![at(0, 0)],
            SimError::PartialMeasurement { qubit: 3 },
        ),
        (
            vec![ry.clone()],
            vec![at(1, 0)],
            SimError::GateIndexOutOfRange {
                index: 1,
                num_gates: 1,
            },
        ),
        (
            vec![ry.clone(), G::H(0)],
            vec![at(0, 0), at(1, 0)],
            SimError::NotDifferentiable {
                index: 1,
                instruction: "H(0)".to_string(),
            },
        ),
        // #561: an angle the gate does not have, whatever its family.
        (
            vec![ry.clone()],
            vec![at(0, 1)],
            SimError::AngleIndexOutOfRange {
                index: 0,
                angle: 1,
                num_angles: 1,
                instruction: format!("{ry:?}"),
            },
        ),
        (
            vec![ry.clone(), u2.clone()],
            vec![at(1, 1), at(1, 2)],
            SimError::AngleIndexOutOfRange {
                index: 1,
                angle: 2,
                num_angles: 2,
                instruction: format!("{u2:?}"),
            },
        ),
        (
            vec![u(Fixed(0.3))],
            vec![at(0, 3)],
            SimError::AngleIndexOutOfRange {
                index: 0,
                angle: 3,
                num_angles: 3,
                instruction: format!("{:?}", u(Fixed(0.3))),
            },
        ),
        (
            vec![G::Rx {
                qubit: 2,
                theta: Fixed(f64::INFINITY),
            }],
            vec![at(0, 0)],
            SimError::NonFiniteAmplitude,
        ),
        (
            vec![G::Rzz {
                q0: 0,
                q1: 1,
                theta: Param(4),
            }],
            vec![at(0, 0)],
            SimError::UnboundParameter { index: 4 },
        ),
        // Every angle of a gate in `wrt` is resolved, not only the one named.
        (
            vec![u(Param(2))],
            vec![at(0, 0)],
            SimError::UnboundParameter { index: 2 },
        ),
        (
            vec![G::Cx(MAX_QUBITS, 0)],
            vec![],
            SimError::QubitIndexOutOfRange {
                qubit: MAX_QUBITS,
                num_qubits: MAX_QUBITS,
            },
        ),
        (
            vec![G::Cz(1, 1)],
            vec![],
            SimError::RepeatedQubit { qubit: 1 },
        ),
        (
            vec![G::MeasureAll, ry],
            vec![at(1, 0)],
            SimError::GateAfterMeasure { qubit: 0 },
        ),
    ];
    for (gates, wrt, expected) in cases {
        assert_eq!(
            simulator().adjoint_gradient(&wide(gates), &wrt, |_| 1.0, None),
            Err(expected)
        );
    }
}

/// The call of a declared gate is not differentiated, whatever its angles
/// (its body may use them in any way), nor is a gate without angles.
#[test]
fn a_declared_gate_and_a_gate_without_angles_are_not_differentiable() {
    let declared = ParameterizedCircuit::from_qasm2(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate g(t) a { rz(t) a; }\nqreg q[1];\ng(0.3) q[0];\n",
    )
    .expect("a valid declared gate")
    .gates()[0]
        .clone();
    for gate in [declared, G::Sx(0)] {
        let circuit = ConcreteCircuit::try_new(1, vec![gate.clone()]).expect("valid");
        assert_eq!(
            simulator().adjoint_gradient(&circuit, &[AngleRef::new(0, 0)], |_| 1.0, None),
            Err(SimError::NotDifferentiable {
                index: 0,
                instruction: format!("{gate:?}"),
            })
        );
    }
}

#[test]
fn a_non_finite_weight_is_a_typed_error() {
    let circuit = random_circuit(3, 4, &[0], 1);
    let wrt = rotation_angles(&circuit);
    assert_eq!(
        simulator().adjoint_gradient(
            &circuit,
            &wrt,
            |b| if b == 5 { f64::NAN } else { 1.0 },
            None
        ),
        Err(SimError::NonFiniteWeight)
    );
}
