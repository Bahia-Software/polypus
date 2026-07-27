//! Semantic verification of X/Y-basis readout against the real `polypus-sim`
//! statevector simulator (design doc §7.2) — the same rigour as the layer
//! semantics tests: not "it compiles", but "the measured expectation equals the
//! Pauli it claims, computed independently".
//!
//! Each test compiles a model whose readout measures a non-`Z` Pauli. The
//! compiled `template_for` inserts the basis change (`H` for `X`; `Sdg` then `H`
//! for `Y`) before the terminal measurement, so feeding the resulting exact
//! basis-state probabilities to `predict_from_probabilities` yields `⟨P⟩` of the
//! pre-measurement state. The reference `⟨P⟩` is computed by hand from that same
//! pre-measurement state (obtained from a Z-readout twin, which emits no basis
//! change) via the operator definition — an independent path, never the
//! circuit's own `H`/`Sdg`.
//!
//! `polypus-sim` enters only as a dev-dependency — the crate's public API never
//! executes a circuit.

use std::collections::HashMap;

use polypus_qml::{
    CompiledModel, Decision, HardwareEfficientAnsatz, Layer, Observable, Pauli, PauliString,
    QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, Statevector, StatevectorSimulator, C64};

/// `⟨P₀P₁…⟩` for a product of single-qubit Paulis, computed straight from a full
/// statevector's amplitudes via the operator definition. Handles entanglement:
/// it sums over the whole register, not a per-qubit reduced state.
///
/// State index bit `q` (value `2^q`) is qubit `q` — the same convention
/// `Statevector::sample`/`probabilities` use. `X`/`Y` flip their qubit's bit;
/// `Y` carries `−i` when that bit is 0 and `+i` when it is 1; `Z` is diagonal
/// with sign `(−1)^bit`.
fn expect_pauli_string(amps: &[C64], factors: &[(usize, Pauli)]) -> f64 {
    let flip: usize = factors
        .iter()
        .filter(|(_, p)| matches!(p, Pauli::X | Pauli::Y))
        .fold(0usize, |acc, &(q, _)| acc ^ (1 << q));

    let mut total = C64::new(0.0, 0.0);
    for (i, amp_i) in amps.iter().enumerate() {
        let j = i ^ flip;
        let mut coeff = C64::new(1.0, 0.0);
        for &(q, pauli) in factors {
            let bit = (i >> q) & 1;
            coeff *= match pauli {
                Pauli::X => C64::new(1.0, 0.0),
                Pauli::Y if bit == 0 => C64::new(0.0, -1.0),
                Pauli::Y => C64::new(0.0, 1.0),
                Pauli::Z if bit == 0 => C64::new(1.0, 0.0),
                Pauli::Z => C64::new(-1.0, 0.0),
            };
        }
        total += amp_i.conj() * coeff * amps[j];
    }
    total.re
}

/// The exact basis-state probability map (`|amplitude|²` keyed by C-3 bitstring)
/// of a statevector, ready for `predict_from_probabilities`.
fn probabilities_map(sv: &Statevector, width: usize) -> HashMap<String, f64> {
    sv.probabilities()
        .into_iter()
        .enumerate()
        .map(|(state, p)| (format!("{state:0width$b}"), p))
        .collect()
}

/// Run a bound circuit and return its statevector.
fn run(model: &CompiledModel, x: &[f64], theta: &[f64]) -> Statevector {
    let circuit = model.bind(x, theta).unwrap();
    StatevectorSimulator::new().run(&circuit).unwrap()
}

fn observable(factors: &[(usize, Pauli)]) -> Observable {
    Observable::new(vec![(1.0, PauliString::new(factors.to_vec()).unwrap())]).unwrap()
}

/// An all-`Z` twin readout over the given positions: it emits no basis change,
/// so running it gives the pre-measurement state to compute references from.
fn z_twin(positions: &[usize]) -> Readout {
    let observables = positions
        .iter()
        .map(|&p| observable(&[(p, Pauli::Z)]))
        .collect();
    // Decision is irrelevant here — the twin is only ever run for its state.
    Readout::new(observables, Decision::Raw).unwrap()
}

/// (0) Regression: an all-`Z` readout emits a circuit byte-identical to a bare
/// computational-basis measurement — the basis-change insertion adds nothing.
/// A golden QASM comparison for a reference `angle_encoder + real_amplitudes(1)`
/// model pins the exact output, so any accidental basis-change gate (or gate
/// reordering) fails loudly.
#[test]
fn z_readout_qasm_is_unchanged() {
    let readout = Readout::new(vec![observable(&[(0, Pauli::Z)])], Decision::Sign).unwrap();
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(readout)
        .compile(2)
        .unwrap();
    let qasm = model
        .bind(&[0.5, 1.25], &[0.1, 0.2, 0.3, 0.4])
        .unwrap()
        .to_qasm2();

    let expected = "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
ry(0.500000000000) q[0];
ry(1.250000000000) q[1];
ry(0.100000000000) q[0];
ry(0.200000000000) q[1];
cx q[0],q[1];
ry(0.300000000000) q[0];
ry(0.400000000000) q[1];
measure q -> c;
";
    assert_eq!(qasm, expected);
}

/// (1) A `⟨X₀⟩` readout measures `X`, not `Z`. On a real (phase-free) state
/// `⟨X₀⟩` and `⟨Z₀⟩` differ sharply, so getting `Z` instead would fail loudly.
#[test]
fn x_readout_measures_x_not_z() {
    let x = [0.9];
    let theta = [0.3, 0.4];
    let build = |readout| {
        QuantumModel::new(1)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(1)
            .unwrap()
    };

    let x_model = build(Readout::new(vec![observable(&[(0, Pauli::X)])], Decision::Raw).unwrap());
    let pre = run(&build(z_twin(&[0])), &x, &theta);

    let measured = x_model
        .predict_from_probabilities(&probabilities_map(&run(&x_model, &x, &theta), 1))
        .unwrap();
    let reference_x = expect_pauli_string(pre.amplitudes(), &[(0, Pauli::X)]);
    let reference_z = expect_pauli_string(pre.amplitudes(), &[(0, Pauli::Z)]);

    assert!(
        (measured - reference_x).abs() < 1e-9,
        "measured {measured} != ⟨X₀⟩ {reference_x}"
    );
    // The discriminating check: the readout is genuinely X, not Z.
    assert!(
        (reference_x - reference_z).abs() > 0.5,
        "test is not discriminating: ⟨X₀⟩ {reference_x} ≈ ⟨Z₀⟩ {reference_z}"
    );
    assert!((measured - reference_z).abs() > 0.5);
}

/// (2) A `⟨Y₀⟩` readout measures `Y`, not `Z`. `Rz` rotations give the state a
/// phase, so `⟨Y₀⟩` is non-trivial and distinct from `⟨Z₀⟩`.
#[test]
fn y_readout_measures_y_not_z() {
    let x = [0.8];
    // HardwareEfficientAnsatz::new(1) = [Ry, Rz] × 2 blocks = 4 params; the Rz
    // gates put a phase on the state so ⟨Y₀⟩ ≠ 0.
    let theta = [0.5, 1.1, 0.7, 0.9];
    let build = |readout| {
        QuantumModel::new(1)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1)))
            .readout(readout)
            .compile(1)
            .unwrap()
    };

    let y_model = build(Readout::new(vec![observable(&[(0, Pauli::Y)])], Decision::Raw).unwrap());
    let pre = run(&build(z_twin(&[0])), &x, &theta);

    let measured = y_model
        .predict_from_probabilities(&probabilities_map(&run(&y_model, &x, &theta), 1))
        .unwrap();
    let reference_y = expect_pauli_string(pre.amplitudes(), &[(0, Pauli::Y)]);
    let reference_z = expect_pauli_string(pre.amplitudes(), &[(0, Pauli::Z)]);

    assert!(
        (measured - reference_y).abs() < 1e-9,
        "measured {measured} != ⟨Y₀⟩ {reference_y}"
    );
    assert!(
        reference_y.abs() > 0.1 && (reference_y - reference_z).abs() > 0.1,
        "test is not discriminating: ⟨Y₀⟩ {reference_y}, ⟨Z₀⟩ {reference_z}"
    );
}

/// (3) A single observable with `X` and `Y` on *different* qubits (`X₀·Y₁`)
/// applies the right basis change to each independently, and the parity
/// estimator yields the correct joint expectation on an entangled state.
#[test]
fn mixed_x_y_string_on_distinct_qubits() {
    let x = [0.7, 1.3];
    // 2-qubit HardwareEfficientAnsatz::new(1) = 2 × 2 × 2 = 8 params; the Cx
    // entangles the qubits, so ⟨X₀Y₁⟩ is not a product of local expectations.
    let theta = [0.2, 0.5, 0.9, 0.3, 1.2, 0.4, 0.6, 0.8];
    let build = |readout| {
        QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(1)))
            .readout(readout)
            .compile(2)
            .unwrap()
    };

    let string = [(0, Pauli::X), (1, Pauli::Y)];
    let model = build(Readout::new(vec![observable(&string)], Decision::Raw).unwrap());
    let pre = run(&build(z_twin(&[0, 1])), &x, &theta);

    let measured = model
        .predict_from_probabilities(&probabilities_map(&run(&model, &x, &theta), 2))
        .unwrap();
    let reference = expect_pauli_string(pre.amplitudes(), &string);

    assert!(
        (measured - reference).abs() < 1e-9,
        "measured {measured} != ⟨X₀Y₁⟩ {reference}"
    );
    // Non-trivial value, so the test actually exercises the joint estimator.
    assert!(reference.abs() > 0.05, "⟨X₀Y₁⟩ ~ 0: not discriminating");
}

/// (4) A multiclass `Argmax` whose classes share the *same* non-`Z` basis (both
/// `X`, on different qubits) compiles to one circuit and predicts the class with
/// the larger `⟨X⟩` — proving the single-group rule is not tied to
/// `Sign`/`Threshold`/`Raw`.
#[test]
fn argmax_over_shared_x_basis_classes() {
    let x = [1.4, 0.2];
    let theta = [0.6, 0.1, 0.9, 0.5];
    let build = |readout| {
        QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(2)
            .unwrap()
    };

    let model = build(
        Readout::new(
            vec![observable(&[(0, Pauli::X)]), observable(&[(1, Pauli::X)])],
            Decision::Argmax,
        )
        .unwrap(),
    );
    let pre = run(&build(z_twin(&[0, 1])), &x, &theta);

    let predicted = model
        .predict_from_probabilities(&probabilities_map(&run(&model, &x, &theta), 2))
        .unwrap();

    let x0 = expect_pauli_string(pre.amplitudes(), &[(0, Pauli::X)]);
    let x1 = expect_pauli_string(pre.amplitudes(), &[(1, Pauli::X)]);
    let expected_index = if x1 > x0 { 1.0 } else { 0.0 };

    assert_eq!(predicted, expected_index, "⟨X₀⟩ {x0}, ⟨X₁⟩ {x1}");
    // A clear margin, so the argmax is not decided by floating-point noise.
    assert!(
        (x0 - x1).abs() > 0.1,
        "test is not discriminating: ⟨X₀⟩ {x0} ≈ ⟨X₁⟩ {x1}"
    );
}
