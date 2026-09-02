//! Semantic verification of the layer catalogue against the real
//! `polypus-sim` statevector simulator (design doc §13.2) — not just
//! instruction-list golden tests, but the actual quantum state and its
//! measured expectations.
//!
//! Eight checks:
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
//!    Möttönen state preparation;
//! 6. the [`IqpEncoder`] produces the closed-form IQP state, amplitude by
//!    amplitude, for all three [`Entanglement`] patterns;
//! 7. the `IqpEncoder`'s `⟨X₀⟩` is the non-linear closed form
//!    `cos(x₀)·cos(x₀·x₁)` — the data non-linearity that motivates the layer;
//! 8. the `BasisEncoder` puts a binary sample on a **computational basis
//!    state** — a single unit amplitude, matching an independently
//!    hand-applied `X` on every set bit.
//!
//! `polypus-sim` enters only as a dev-dependency — the crate's public API never
//! executes a circuit.
//!
//! `expectation_from_counts` is `pub(crate)`, so these external tests reach it
//! through its public equivalent: [`Decision::Raw`] +
//! [`QmlProblem::predict_from_counts`], which invokes it internally.

use std::f64::consts::PI;

use polypus_circuit::{Fixed, GateInstruction};
use polypus_qml::{
    ConvBlock, ConvLayer, Dataset, Decision, Entanglement, Entangler, HardwareEfficientAnsatz,
    IqpEncoder, Layer, Loss, Observable, Pauli, PauliString, PoolBlock, PoolLayer, QmlProblem,
    QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, SplitMix64, Statevector, StatevectorSimulator, C64};

/// Amplitude-comparison helper, same idiom as `polypus-sim`'s own tests.
fn close(a: C64, b: C64) -> bool {
    (a - b).norm() < 1e-10
}

mod common;
use common::to_bitstring_counts;

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

/// A trainable layer that is the **exact identity** at `θ = 0`: `reps = 0` (so
/// no entangling block is ever emitted) plus a final `Rz` rotation layer. It
/// exists only to satisfy `compile`'s `NoTrainableParams` check, and
/// `rz(0) = diag(1, 1)`, so the simulated state is the encoder's alone — no
/// fixed `Cx` chain to strip back off (contrast with
/// [`assert_amplitude_encoding`], which has to undo one).
fn identity_ansatz() -> HardwareEfficientAnsatz {
    HardwareEfficientAnsatz {
        reps: 0,
        rotations: vec![RotationAxis::Rz],
        entangler: Entangler::Cx,
        entanglement: Entanglement::Linear,
        final_rotation_layer: true,
    }
}

/// (6) The [`IqpEncoder`] produces the closed-form IQP state.
///
/// The expected amplitudes are **not** built by replaying gates: they come from
/// the analytic form of `H⊗n` followed by the diagonal `Rz`/`Rzz` phases. With
/// `s_k = 2·b_k − 1` the sign of qubit `k` in basis state `|b⟩`,
///
/// ```text
/// ⟨b|ψ⟩ = 2^(−n/2) · exp( (i/2) · [ Σ_k s_k·x_k − Σ_(a,b)∈pairs s_a·s_b·x_a·x_b ] )
/// ```
///
/// because `H⊗n|0…0⟩` is the uniform superposition and each subsequent gate is
/// diagonal: `Rz(θ)` contributes `exp(i·s_k·θ/2)` (it is
/// `diag(e^(−iθ/2), e^(+iθ/2))`), and `Rzz(θ)` on `(a,b)` contributes
/// `exp(−i·s_a·s_b·θ/2)` (`e^(−iθ/2)` on equal bits, `e^(+iθ/2)` on differing
/// ones — and `s_a·s_b` is `+1` exactly when the bits are equal). The encoder's
/// `Rzz` angle is the product `x_a·x_b`, which is where the map's non-linearity
/// in the data comes from.
///
/// `pairs` is passed in literally per case, never taken from
/// `entanglement_pairs`, so the pattern is verified too and not assumed.
fn assert_iqp_state(
    num_qubits: usize,
    entanglement: Entanglement,
    x: &[f64],
    pairs: &[(usize, usize)],
) {
    let model = QuantumModel::new(num_qubits)
        .layer(Layer::Iqp(IqpEncoder { entanglement }))
        .layer(Layer::HardwareEfficient(identity_ansatz()))
        .readout(z0_readout(Decision::Raw))
        .compile(x.len())
        .unwrap();

    let theta = vec![0.0; model.num_params()];
    let circuit = model.bind(x, &theta).unwrap();
    let actual = StatevectorSimulator::new().run(&circuit).unwrap();

    let dim = 1usize << num_qubits;
    let amplitude = 1.0 / (dim as f64).sqrt();
    let amps = actual.amplitudes();
    assert_eq!(amps.len(), dim);
    for (index, got) in amps.iter().enumerate() {
        // s_k = +1 when bit k of `index` is set, −1 otherwise (the simulator's
        // convention: bit k of the state index is qubit k).
        let s = |k: usize| if index >> k & 1 == 1 { 1.0 } else { -1.0 };
        let single: f64 = (0..num_qubits).map(|k| s(k) * x[k]).sum();
        let interaction: f64 = pairs.iter().map(|&(a, b)| s(a) * s(b) * x[a] * x[b]).sum();
        let want = C64::from_polar(amplitude, (single - interaction) / 2.0);
        assert!(
            close(*got, want),
            "{entanglement:?}, x={x:?}, |{index:0num_qubits$b}⟩: got {got} vs expected {want}"
        );
    }
}

#[test]
fn iqp_encoder_matches_the_closed_form_iqp_state() {
    // Full on 2 qubits: the single pair (0,1).
    assert_iqp_state(2, Entanglement::Full, &[0.3, 0.5], &[(0, 1)]);
    assert_iqp_state(2, Entanglement::Full, &[1.3, -0.7], &[(0, 1)]);
    // Full on 3 qubits: every i < j.
    assert_iqp_state(
        3,
        Entanglement::Full,
        &[0.2, 0.4, 0.8],
        &[(0, 1), (0, 2), (1, 2)],
    );
    // Linear on 3 qubits: the chain only — a strictly smaller pair set, so a
    // mixed-up pattern could not pass both this case and the previous one.
    assert_iqp_state(3, Entanglement::Linear, &[0.2, 0.4, 0.8], &[(0, 1), (1, 2)]);
    // Circular on 3 qubits: the chain plus the wrap-around (2,0). `Rzz` is
    // symmetric and its angle is a product, so the pair's order is immaterial
    // to the state — only its presence is.
    assert_iqp_state(
        3,
        Entanglement::Circular,
        &[0.2, 0.4, 0.8],
        &[(0, 1), (1, 2), (2, 0)],
    );
    // 4 qubits, Full: six pairs, a feature at 0 (all its products vanish, so
    // its Rzz factors are identity) and a negative feature.
    assert_iqp_state(
        4,
        Entanglement::Full,
        &[0.5, -1.1, 0.0, 0.25],
        &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    );
}

/// (7) The `IqpEncoder`'s `⟨X₀⟩` on 2 qubits with `Entanglement::Full` is
/// `cos(x₀)·cos(x₀·x₁)`.
///
/// An `X` readout is what makes this non-trivial: the IQP state is `H⊗n`
/// followed by *diagonal* gates, so every basis-state probability stays `2^−n`
/// and **every** `Z` expectation is exactly `0` — the whole encoding lives in
/// the phases, which only a non-`Z` basis can see.
///
/// Derived from the amplitudes of check (6): with `ψ(b₁,b₀) = ½·e^(iφ)`,
/// `⟨X₀⟩ = Σ_(b₁) 2·Re(ψ*(b₁,0)·ψ(b₁,1)) = ½·[cos(Δφ|b₁=0) + cos(Δφ|b₁=1)]`,
/// where `Δφ = x₀·(1 − s₁·x₁)`. That is
/// `½·[cos(x₀(1+x₁)) + cos(x₀(1−x₁))] = cos(x₀)·cos(x₀·x₁)` by the
/// sum-to-product identity. Note the dependence on `x₀·x₁`: a bare product
/// kernel, which no `AngleEncoder` can produce.
#[test]
fn iqp_encoder_x_expectation_is_the_non_linear_closed_form() {
    let readout = Readout::new(
        vec![Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::X)]).unwrap())]).unwrap()],
        Decision::Raw,
    )
    .unwrap();
    let model = QuantumModel::new(2)
        .layer(Layer::Iqp(IqpEncoder::new()))
        .layer(Layer::HardwareEfficient(identity_ansatz()))
        .readout(readout)
        .compile(2)
        .unwrap();

    for x in [[0.3, 0.5], [1.3, -0.7], [0.0, 2.0], [2.5, 1.25]] {
        let theta = vec![0.0; model.num_params()];
        let circuit = model.bind(&x, &theta).unwrap();
        let sv = StatevectorSimulator::new().run(&circuit).unwrap();

        // The compiled model inserts the X→Z basis change (an `H` on qubit 0)
        // before the measurement, so a `Z` expectation on the *simulated* state
        // is the `X` expectation of the encoded one.
        let exact = sv.expectation_z(&[0]);
        let want = x[0].cos() * (x[0] * x[1]).cos();
        assert!(
            (exact - want).abs() < 1e-10,
            "x={x:?}: exact ⟨X₀⟩ {exact} vs closed form {want}"
        );

        // And the same value survives the counts path the trainer actually uses.
        let dataset = Dataset::from_rows(&[x.to_vec()], &[0.0]).unwrap();
        let problem = QmlProblem::new(model.clone(), dataset, Loss::SquaredError).unwrap();
        let bound = &problem.bind_batch(&theta).unwrap()[0];
        let sampled = StatevectorSimulator::new().run(bound).unwrap();
        let raw = sampled.sample(200_000, &mut SplitMix64::new(0xB0BACAFE));
        let counts = to_bitstring_counts(raw, bound.num_qubits);
        let estimate = problem.predict_from_counts(&counts).unwrap();
        assert!(
            (want - estimate).abs() < 0.01,
            "x={x:?}: counts estimate {estimate} deviates from closed form {want}"
        );
    }
}

/// (8) The [`BasisEncoder`] loads a binary sample onto a computational basis
/// state.
///
/// Same `θ = 0` trick as check (1): every `real_amplitudes(1)` `Ry` becomes the
/// identity, so the only gates besides the encoder's `X`s are the ansatz's fixed
/// linear `Cx` chain — hand-applied here too, rather than stripped, since the
/// point is to compare the *whole* circuit against an independently built state.
/// Every gate involved (`X`, `Cx`) is a basis permutation, so the result must
/// still be a single unit amplitude: that is asserted separately, because it is
/// what distinguishes basis encoding from every other encoder in the catalogue
/// (an angle or IQP encoder spreads the state over a superposition).
#[test]
fn basis_encoder_prepares_the_computational_basis_state() {
    let model = QuantumModel::new(3)
        .basis_encoder()
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(z0_readout(Decision::Raw))
        .compile(3)
        .unwrap();

    let x = [1.0, 0.0, 1.0];
    let circuit = model.bind(&x, &vec![0.0; model.num_params()]).unwrap();
    let actual = StatevectorSimulator::new().run(&circuit).unwrap();

    let mut expected = Statevector::new(3).unwrap();
    // The encoder: one X per feature equal to 1.0 — qubit 1 is left alone.
    apply(&mut expected, GateInstruction::X(0));
    apply(&mut expected, GateInstruction::X(2));
    // The ansatz's fixed linear Cx chain, all its Ry being identity at θ = 0.
    apply(&mut expected, GateInstruction::Cx(0, 1));
    apply(&mut expected, GateInstruction::Cx(1, 2));

    for (a, b) in actual.amplitudes().iter().zip(expected.amplitudes()) {
        assert!(close(*a, *b), "amplitude mismatch: {a} vs {b}");
    }

    // Every gate above is a basis permutation, so the state is one basis vector
    // — no superposition anywhere, which is the defining property of the layer.
    let occupied: Vec<usize> = actual
        .amplitudes()
        .iter()
        .enumerate()
        .filter(|(_, a)| a.norm() > 1e-10)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        occupied.len(),
        1,
        "basis encoding must leave a single occupied basis state, got {occupied:?}"
    );
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
