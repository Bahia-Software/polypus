//! Adjoint differentiation (issue #503): `StatevectorSimulator::adjoint_gradient`
//! against two independent references computed with `Simulator::run` — the
//! parameter-shift rule, gate by gate, and central finite differences — over
//! every rotation it differentiates, every instruction it steps back through,
//! angles shared by several gates, both kernel paths, and its typed errors.

use polypus_circuit::{
    ConcreteCircuit, GateInstruction as G, GateParam::Fixed, GateParam::Param, ParameterizedCircuit,
};
use polypus_sim::{
    AdjointGradient, SimError, Simulator, SplitMix64, StatevectorSimulator, MAX_QUBITS,
};
use std::f64::consts::FRAC_PI_2;

/// Agreement with the references. The adjoint and the shift rule are both
/// exact, so they agree to rounding (~1e-15 here); central differences with
/// `FD_STEP` carry a truncation error of about `h²/6 · |E'''| ≈ 2e-11` (each
/// angle enters as `a + b·cos θ + c·sin θ` with `|b|, |c| ≤ ‖O‖ ≤ 1`) and a
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

/// `gate` with its rotation angle moved by `delta`.
fn shifted(gate: &G, delta: f64) -> G {
    let moved = |theta: &polypus_circuit::GateParam| match theta {
        Fixed(v) => Fixed(v + delta),
        other => panic!("the reference shifts fixed angles only, not {other:?}"),
    };
    match gate {
        G::Rx { qubit, theta } => G::Rx {
            qubit: *qubit,
            theta: moved(theta),
        },
        G::Ry { qubit, theta } => G::Ry {
            qubit: *qubit,
            theta: moved(theta),
        },
        G::Rz { qubit, theta } => G::Rz {
            qubit: *qubit,
            theta: moved(theta),
        },
        G::Rxx { q0, q1, theta } => G::Rxx {
            q0: *q0,
            q1: *q1,
            theta: moved(theta),
        },
        G::Rzz { q0, q1, theta } => G::Rzz {
            q0: *q0,
            q1: *q1,
            theta: moved(theta),
        },
        other => panic!("{other:?} is not a rotation"),
    }
}

/// `circuit` with the rotations at `positions` all moved by `delta`.
fn moved(circuit: &ConcreteCircuit, positions: &[usize], delta: f64) -> ConcreteCircuit {
    let mut gates = circuit.gates().to_vec();
    for &p in positions {
        gates[p] = shifted(&gates[p], delta);
    }
    ConcreteCircuit::try_new(circuit.num_qubits(), gates).expect("a valid circuit")
}

/// The parameter-shift derivative of the rotation at `position`.
fn shift_rule(circuit: &ConcreteCircuit, position: usize, weights: &[f64]) -> f64 {
    let plus = expectation(&moved(circuit, &[position], FRAC_PI_2), weights);
    let minus = expectation(&moved(circuit, &[position], -FRAC_PI_2), weights);
    (plus - minus) / 2.0
}

/// The central-difference derivative of `⟨O⟩` when the rotations at
/// `positions` move together (one angle they share).
fn central_difference(circuit: &ConcreteCircuit, positions: &[usize], weights: &[f64]) -> f64 {
    let plus = expectation(&moved(circuit, positions, FD_STEP), weights);
    let minus = expectation(&moved(circuit, positions, -FD_STEP), weights);
    (plus - minus) / (2.0 * FD_STEP)
}

fn adjoint(circuit: &ConcreteCircuit, wrt: &[usize], weights: &[f64]) -> AdjointGradient {
    simulator()
        .adjoint_gradient(circuit, wrt, |b| weights[b], None)
        .expect("a differentiable circuit")
}

/// The positions of every `Rx`/`Ry`/`Rz`/`Rxx`/`Rzz` in `circuit`.
fn rotation_positions(circuit: &ConcreteCircuit) -> Vec<usize> {
    circuit
        .gates()
        .iter()
        .enumerate()
        .filter(|(_, g)| {
            matches!(
                g,
                G::Rx { .. } | G::Ry { .. } | G::Rz { .. } | G::Rxx { .. } | G::Rzz { .. }
            )
        })
        .map(|(p, _)| p)
        .collect()
}

/// Every derivative against the shift rule and central differences, and the
/// expectation against the probabilities.
fn assert_matches_references(circuit: &ConcreteCircuit, weights: &[f64], what: &str) {
    let wrt = rotation_positions(circuit);
    assert!(!wrt.is_empty(), "{what}: nothing to differentiate");
    let grad = adjoint(circuit, &wrt, weights);
    let reference = expectation(circuit, weights);
    assert!(
        (grad.expectation - reference).abs() < 1e-12,
        "{what}: expectation {} vs {reference}",
        grad.expectation
    );
    for (&position, &d) in wrt.iter().zip(&grad.derivatives) {
        let shift = shift_rule(circuit, position, weights);
        let fd = central_difference(circuit, &[position], weights);
        assert!(
            (d - shift).abs() < TOLERANCE,
            "{what}: gate {position} adjoint {d} vs shift {shift}"
        );
        assert!(
            (d - fd).abs() < TOLERANCE,
            "{what}: gate {position} adjoint {d} vs finite difference {fd}"
        );
    }
}

/// A rotation of kind `kind` (0..5: `Rx, Ry, Rz, Rxx, Rzz`) at a random angle.
fn random_rotation(kind: u64, n: usize, rng: &mut SplitMix64) -> G {
    let q = (rng.next_u64() % n as u64) as usize;
    let r = (q + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
    let theta = Fixed(6.0 * rng.next_f64() - 3.0);
    match kind {
        0 => G::Rx { qubit: q, theta },
        1 => G::Ry { qubit: q, theta },
        2 => G::Rz { qubit: q, theta },
        3 => G::Rxx {
            q0: q,
            q1: r,
            theta,
        },
        _ => G::Rzz {
            q0: q,
            q1: r,
            theta,
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

/// `layers` × (one random fixed gate + one rotation of a kind drawn from
/// `kinds`), after a layer of `H`s so the state is generic from the start.
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

// contract test: needs human sign-off
/// Acceptance criterion 1: each of the five rotations, alone among fixed gates,
/// matches the shift rule and central differences.
#[test]
fn each_rotation_matches_the_shift_rule_and_finite_differences() {
    let n = 4;
    for kind in 0..5 {
        for seed in 0..4 {
            let circuit = random_circuit(n, 12, &[kind], 100 * kind + seed);
            let weights = random_weights(n, seed);
            assert_matches_references(&circuit, &weights, &format!("kind {kind} seed {seed}"));
        }
    }
}

// contract test: needs human sign-off
#[test]
fn mixed_rotations_on_a_wider_circuit_match_the_references() {
    let n = 6;
    for seed in 0..3 {
        let circuit = random_circuit(n, 30, &[0, 1, 2, 3, 4], 7_000 + seed);
        let weights = random_weights(n, 50 + seed);
        assert_matches_references(&circuit, &weights, &format!("seed {seed}"));
    }
}

/// A `Z`-parity observable (the readouts of `polypus-qml`): the weight is a
/// function of the basis state, never a table.
#[test]
fn a_pauli_z_product_observable_matches_the_references() {
    let n = 5;
    let circuit = random_circuit(n, 20, &[0, 1, 2, 3, 4], 31);
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
    let wrt = rotation_positions(&circuit);
    let from_function = simulator()
        .adjoint_gradient(&circuit, &wrt, parity, None)
        .expect("differentiable");
    assert_eq!(from_function, adjoint(&circuit, &wrt, &weights));
}

// contract test: needs human sign-off
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
    let shared = [2, 4, 6, 8];
    let grad = adjoint(&circuit, &shared, &weights);
    for (&position, &d) in shared.iter().zip(&grad.derivatives) {
        assert!((d - shift_rule(&circuit, position, &weights)).abs() < TOLERANCE);
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

/// A rotation on every qubit, then one instruction of every kind but `Measure`
/// (rejected; see below), each followed by a rotation, then `MeasureAll`.
fn every_kind_circuit() -> (ConcreteCircuit, Vec<usize>) {
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
    let circuit = ConcreteCircuit::try_new(5, gates).expect("a valid circuit");
    let wrt = rotation_positions(&circuit);
    (circuit, wrt)
}

// contract test: needs human sign-off
/// Acceptance criterion 2: the sweep back steps through the inverse of every
/// instruction kind, and the gradient of a circuit holding all of them is the
/// shift rule's. The kind list must be complete, so a gate added to
/// `polypus-circuit` fails here until the adjoint supports it.
#[test]
fn every_instruction_kind_is_differentiated_through() {
    let (circuit, wrt) = every_kind_circuit();
    let mut seen = [false; G::KINDS];
    for gate in circuit.gates() {
        seen[gate.kind_index()] = true;
    }
    let weights = random_weights(5, 77);
    let grad = adjoint(&circuit, &wrt, &weights);
    for (&position, &d) in wrt.iter().zip(&grad.derivatives) {
        let shift = shift_rule(&circuit, position, &weights);
        assert!(
            (d - shift).abs() < TOLERANCE,
            "gate {position} ({:?}): adjoint {d} vs shift {shift}",
            circuit.gates()[position]
        );
    }

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
        simulator().adjoint_gradient(&measured, &[0], |_| 1.0, None),
        Err(SimError::PartialMeasurement { qubit: 1 })
    );

    let missing: Vec<usize> = (0..G::KINDS).filter(|&k| !seen[k]).collect();
    assert!(
        missing.is_empty(),
        "instruction kinds not exercised: {missing:?}"
    );
}

/// The derivatives follow `wrt`'s order, and a position named twice gets its
/// derivative twice.
#[test]
fn derivatives_follow_the_order_of_wrt() {
    let circuit = random_circuit(4, 10, &[0, 1, 2, 3, 4], 5);
    let weights = random_weights(4, 6);
    let wrt = rotation_positions(&circuit);
    let sorted = adjoint(&circuit, &wrt, &weights);
    let mut reversed_wrt = wrt.clone();
    reversed_wrt.reverse();
    reversed_wrt.push(wrt[2]);
    let reversed = adjoint(&circuit, &reversed_wrt, &weights);
    let mut expected: Vec<f64> = sorted.derivatives.iter().rev().copied().collect();
    expected.push(sorted.derivatives[2]);
    assert_eq!(reversed.derivatives, expected);
    assert_eq!(reversed.expectation, sorted.expectation);
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
    let circuit = random_circuit(4, 16, &[0, 1, 2, 3, 4], 12);
    let weights = random_weights(4, 13);
    let wrt = rotation_positions(&circuit);
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

// contract test: needs human sign-off
/// Acceptance criterion 4: the parallel kernels and reductions give the same
/// bits as the sequential ones, for any number of threads.
#[cfg(feature = "parallel")]
#[test]
fn parallel_and_sequential_paths_are_bit_identical() {
    let n = 14;
    let circuit = random_circuit(n, 24, &[0, 1, 2, 3, 4], 2_024);
    let wrt = rotation_positions(&circuit);
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
        assert_eq!(parallel, sequential, "{threads} threads");
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
    let capped = StatevectorSimulator {
        max_qubits: 10,
        ..simulator()
    };
    assert_eq!(
        capped.adjoint_gradient(&wide(vec![ry.clone()]), &[0], |_| 1.0, None),
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
    let cases: Vec<(Vec<G>, Vec<usize>, SimError)> = vec![
        (
            vec![ry.clone(), G::Measure { qubit: 3, cbit: 0 }],
            vec![0],
            SimError::PartialMeasurement { qubit: 3 },
        ),
        (
            vec![ry.clone()],
            vec![1],
            SimError::GateIndexOutOfRange {
                index: 1,
                num_gates: 1,
            },
        ),
        (
            vec![ry.clone(), G::H(0)],
            vec![0, 1],
            SimError::NotDifferentiable {
                index: 1,
                instruction: "H(0)".to_string(),
            },
        ),
        (
            vec![G::Rx {
                qubit: 2,
                theta: Fixed(f64::INFINITY),
            }],
            vec![0],
            SimError::NonFiniteAmplitude,
        ),
        (
            vec![G::Rzz {
                q0: 0,
                q1: 1,
                theta: Param(4),
            }],
            vec![0],
            SimError::UnboundParameter { index: 4 },
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
            vec![1],
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

/// Only the five Pauli rotations are differentiated: every other gate with an
/// angle is `NotDifferentiable`, as `polypus-qml` classifies it.
#[test]
fn gates_with_angles_other_than_pauli_rotations_are_not_differentiable() {
    let others = [
        G::Cp {
            q0: 0,
            q1: 1,
            theta: Fixed(0.3),
        },
        G::P {
            qubit: 0,
            lam: Fixed(0.3),
        },
        G::U {
            qubit: 0,
            theta: Fixed(0.3),
            phi: Fixed(0.1),
            lam: Fixed(0.2),
        },
        G::Crx {
            control: 0,
            target: 1,
            theta: Fixed(0.3),
        },
    ];
    for gate in others {
        let circuit = ConcreteCircuit::try_new(2, vec![gate.clone()]).expect("valid");
        assert_eq!(
            simulator().adjoint_gradient(&circuit, &[0], |_| 1.0, None),
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
    let wrt = rotation_positions(&circuit);
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
