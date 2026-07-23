//! Semantic verification of the layer catalogue against the real
//! `polypus-sim` statevector simulator (design doc §13.2) — not just
//! instruction-list golden tests, but the actual quantum state and its
//! measured expectations.
//!
//! Five checks:
//! 1. the [`AngleEncoder`] produces the expected tensor-product state;
//! 2. the [`ConvBlock::Cartan`] block (including the synthesized `ryy`) matches
//!    an independently hand-applied gate sequence on every canonical basis
//!    state;
//! 3. `expectation_from_counts(sample(seed))` ≈ [`Statevector::expectation_z`]
//!    with statistical tolerance;
//! 4. after pooling, the expectation over a retained qubit is consistent with
//!    the global state;
//! 5. the `AmplitudeEncoder` prepares exactly the L2-normalized, zero-padded
//!    target state (real, imaginary part ~0) — the empirical guarantee of the
//!    Möttönen state preparation.
//!
//! `polypus-sim` enters only as a dev-dependency — the crate's public API never
//! executes a circuit.
//!
//! `expectation_from_counts` is `pub(crate)`, so these external tests reach it
//! through its public equivalent: [`Decision::Raw`] +
//! [`QmlProblem::predict_from_counts`], which invokes it internally.

use std::collections::HashMap;
use std::f64::consts::PI;

use polypus_circuit::{Fixed, GateInstruction};
use polypus_qml::{
    ConvBlock, ConvLayer, Dataset, Decision, HardwareEfficientAnsatz, Layer, Loss, Observable,
    Pauli, PauliString, PoolBlock, PoolLayer, QmlProblem, QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, SplitMix64, Statevector, StatevectorSimulator, C64};

/// Amplitude-comparison helper, same idiom as `polypus-sim`'s own tests.
fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-10
}

/// Convert `polypus-sim`'s `HashMap<state_index, count>` into the C-3 bitstring
/// format `expectation_from_counts` expects. Same pattern as
/// `crates/polypus/src/infrastructure/native.rs`: standard binary formatting,
/// no bit reversal — the character at `width - 1 - k` is qubit `k`.
fn to_bitstring_counts(raw: HashMap<usize, u64>, width: usize) -> HashMap<String, u64> {
    raw.into_iter()
        .map(|(state, count)| (format!("{state:0width$b}"), count))
        .collect()
}

/// A `⟨Z₀⟩` readout with the given decision, reused by several tests.
fn z0_readout(decision: Decision) -> Readout {
    Readout::new(
        vec![Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap()],
        decision,
    )
    .unwrap()
}

/// Apply a fixed-angle gate to a hand-built expected state, failing loudly (in
/// a test) if the simulator rejects it.
fn apply(sv: &mut Statevector, gate: GateInstruction) {
    sv.apply(&gate).expect("hand-applied gate is valid");
}

/// (1) The angle encoder builds the expected tensor-product state.
///
/// With `θ = 0` every ansatz `Ry` is the identity, so the only gates besides
/// the encoder are the fixed `Cx(0,1)` of `real_amplitudes(1)`.
#[test]
fn angle_encoder_is_the_expected_tensor_product() {
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(z0_readout(Decision::Raw))
        .compile(2)
        .unwrap();

    let x = [0.7, 1.1];
    let circuit = model.bind(&x, &[0.0; 4]).unwrap();
    let actual = StatevectorSimulator::new().run(&circuit).unwrap();

    let mut expected = Statevector::new(2).unwrap();
    apply(
        &mut expected,
        GateInstruction::Ry {
            qubit: 0,
            theta: Fixed(x[0]),
        },
    );
    apply(
        &mut expected,
        GateInstruction::Ry {
            qubit: 1,
            theta: Fixed(x[1]),
        },
    );
    apply(&mut expected, GateInstruction::Cx(0, 1));

    for (a, b) in actual.amplitudes().iter().zip(expected.amplitudes()) {
        assert!(close(*a, *b), "amplitude mismatch: {a} vs {b}");
    }
}

/// (2) The `Cartan` block, including the synthesized `ryy`, matches an
/// independently hand-applied gate sequence on every canonical basis state.
///
/// Each basis state is reached with the public API only: `AngleEncoder(Ry)`
/// with `x_j ∈ {0, π}` (`Ry(π)|0⟩ = |1⟩`). The expected state applies the
/// Cartan expansion `rxx, sdg,sdg,rxx,s,s, rzz` **directly** via
/// `Statevector::apply`, never through `ConvLayer::emit`.
#[test]
fn cartan_block_matches_hand_applied_synthesis() {
    let theta = [0.3, 0.5, 0.7];
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::Conv(ConvLayer::new(ConvBlock::Cartan)))
        .readout(z0_readout(Decision::Raw))
        .compile(2)
        .unwrap();

    for x in [[0.0, 0.0], [0.0, PI], [PI, 0.0], [PI, PI]] {
        let circuit = model.bind(&x, &theta).unwrap();
        let actual = StatevectorSimulator::new().run(&circuit).unwrap();

        let mut expected = Statevector::new(2).unwrap();
        // State preparation (the encoder).
        apply(
            &mut expected,
            GateInstruction::Ry {
                qubit: 0,
                theta: Fixed(x[0]),
            },
        );
        apply(
            &mut expected,
            GateInstruction::Ry {
                qubit: 1,
                theta: Fixed(x[1]),
            },
        );
        // Cartan core: rxx(θ0) · ryy(θ1) · rzz(θ2), ryy synthesized.
        apply(
            &mut expected,
            GateInstruction::Rxx {
                q0: 0,
                q1: 1,
                theta: Fixed(theta[0]),
            },
        );
        apply(&mut expected, GateInstruction::Sdg(0));
        apply(&mut expected, GateInstruction::Sdg(1));
        apply(
            &mut expected,
            GateInstruction::Rxx {
                q0: 0,
                q1: 1,
                theta: Fixed(theta[1]),
            },
        );
        apply(&mut expected, GateInstruction::S(0));
        apply(&mut expected, GateInstruction::S(1));
        apply(
            &mut expected,
            GateInstruction::Rzz {
                q0: 0,
                q1: 1,
                theta: Fixed(theta[2]),
            },
        );

        for (a, b) in actual.amplitudes().iter().zip(expected.amplitudes()) {
            assert!(close(*a, *b), "basis {x:?}: amplitude mismatch {a} vs {b}");
        }
    }
}

/// (3) The counts-based expectation estimate agrees with the exact
/// `Statevector::expectation_z` within statistical tolerance.
#[test]
fn counts_expectation_matches_exact_expectation() {
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(z0_readout(Decision::Raw))
        .compile(2)
        .unwrap();

    // Single-sample dataset; the label is irrelevant to a `Raw` readout, and
    // `SquaredError` accepts any finite label.
    let dataset = Dataset::from_rows(&[vec![0.7, 1.1]], &[0.0]).unwrap();
    let problem = QmlProblem::new(model, dataset, Loss::SquaredError).unwrap();

    let theta = [0.1, 0.2, 0.3, 0.4];
    let circuits = problem.bind_batch(&theta).unwrap();
    let circuit = &circuits[0];
    let sv = StatevectorSimulator::new().run(circuit).unwrap();

    let exact = sv.expectation_z(&[0]);

    let raw = sv.sample(200_000, &mut SplitMix64::new(0xC0FFEE));
    let counts = to_bitstring_counts(raw, circuit.num_qubits);
    let estimate = problem.predict_from_counts(&counts).unwrap();

    assert!(
        (exact - estimate).abs() < 0.01,
        "counts estimate {estimate} deviates from exact {exact}"
    );
}

/// (4) After pooling, the expectation over the retained qubit matches the exact
/// expectation on the corresponding physical qubit of the global state.
///
/// 3 qubits, `KeepRule::EvenPositions` (default): pair `(0,1)` retains physical
/// qubit 0, and physical qubit 2 stays active untouched. The readout on logical
/// position 0 resolves to physical qubit 0.
#[test]
fn pooling_expectation_is_consistent_with_the_global_state() {
    let model = QuantumModel::new(3)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::Pool(PoolLayer::new(PoolBlock::Basic)))
        .readout(z0_readout(Decision::Raw))
        .compile(3)
        .unwrap();

    let dataset = Dataset::from_rows(&[vec![0.5, 1.0, 1.5]], &[0.0]).unwrap();
    let problem = QmlProblem::new(model, dataset, Loss::SquaredError).unwrap();

    let theta = [0.1, 0.2, 0.3];
    let circuits = problem.bind_batch(&theta).unwrap();
    let circuit = &circuits[0];
    let sv = StatevectorSimulator::new().run(circuit).unwrap();

    // The readout's logical position 0 → physical qubit 0 after pooling.
    let exact = sv.expectation_z(&[0]);

    let raw = sv.sample(200_000, &mut SplitMix64::new(0xBADF00D));
    let counts = to_bitstring_counts(raw, circuit.num_qubits);
    let estimate = problem.predict_from_counts(&counts).unwrap();

    assert!(
        (exact - estimate).abs() < 0.01,
        "pooled counts estimate {estimate} deviates from exact {exact}"
    );
}

/// (5) The [`AmplitudeEncoder`] prepares exactly `x/‖x‖₂` (zero-padded to
/// `2^k`) as a *real* statevector.
///
/// A model needs a trainable layer to compile (an encoder alone reserves no
/// θ), so a `real_amplitudes(1)` ansatz is appended and bound at `θ = 0`. That
/// zeroes every ansatz `Ry`, but its fixed linear `Cx` chain is **not**
/// identity — so the chain is stripped back off the simulated state (each `Cx`
/// is its own inverse, undone in reverse order) to isolate the pure encoded
/// state, which is then compared amplitude-by-amplitude against the exact
/// normalized target. The whole circuit is `Ry`+`Cx` (real), so every
/// imaginary part must vanish — checked explicitly, since a non-trivial
/// imaginary component would signal a bug.
fn assert_amplitude_encoding(k: usize, x: &[f64]) {
    let model = QuantumModel::new(k)
        .amplitude_encoder()
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(z0_readout(Decision::Raw))
        .compile(x.len())
        .unwrap();

    let theta = vec![0.0; model.num_params()];
    let circuit = model.bind(x, &theta).unwrap();
    let mut sv = StatevectorSimulator::new().run(&circuit).unwrap();

    // Undo the ansatz's fixed linear Cx chain — Cx(0,1),…,Cx(k-2,k-1). Each Cx
    // is involutory, so re-applying them in reverse order inverts the chain,
    // leaving the pure amplitude-encoded state.
    for i in (0..k.saturating_sub(1)).rev() {
        apply(&mut sv, GateInstruction::Cx(i, i + 1));
    }

    // Exact target: x/‖x‖₂, zero-padded to 2^k (index j addresses |j⟩ under the
    // C-3 little-endian convention, matching the encoder's own indexing).
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dim = 1usize << k;
    let mut target = vec![0.0; dim];
    for (j, &xi) in x.iter().enumerate() {
        target[j] = xi / norm;
    }

    let amps = sv.amplitudes();
    assert_eq!(amps.len(), dim);
    for (j, a) in amps.iter().enumerate() {
        assert!(
            a.im.abs() < 1e-10,
            "amplitude {j} has a non-negligible imaginary part: {a}"
        );
        assert!(
            (a.re - target[j]).abs() < 1e-10,
            "amplitude {j}: got real {} vs expected {}",
            a.re,
            target[j]
        );
    }
}

#[test]
fn amplitude_encoder_prepares_the_normalized_state() {
    // k=2, an interior zero: the general multiplexed path with a zeroed leaf.
    assert_amplitude_encoding(2, &[1.0, 2.0, 0.0, 3.0]);
    // k=2, a whole zeroed sub-block ([0,0] at prefix 1): exercises the
    // null-mass branch (M_p == 0 ⇒ θ_p = 0) inside the level-angle computation.
    assert_amplitude_encoding(2, &[1.0, 2.0, 0.0, 0.0]);
    // k=3, eight positive values including a zero.
    assert_amplitude_encoding(3, &[0.5, 1.5, 2.0, 0.0, 1.0, 2.5, 3.0, 0.7]);
}
