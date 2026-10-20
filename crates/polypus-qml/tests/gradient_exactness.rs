//! The exact parameter-shift gradient must be the derivative of the fitness.
//!
//! In exact mode there is no shot noise: the gradient the crate assembles from
//! exact probabilities has to match a central finite difference of
//! [`QmlProblem::fitness_from_probabilities`] to near machine precision, for
//! every layer the crate offers.
//!
//! The two-term `±π/2` shift rule is only exact when a parameter enters the
//! circuit **once**, as the angle of a Pauli rotation `exp(−iθP/2)`. The
//! convolution and pooling layers reuse one `θ` slice across every qubit pair,
//! so shifting that `θ` as a whole moves several gates at once and the rule no
//! longer gives the derivative. The readouts below are chosen so the reused
//! gates sit inside the observable's light cone (`Z₁`, or a `Z` string over
//! every active qubit): a lone `Z₀` readout happens to see only one pair and
//! hides the error.
//!
//! See `docs/design/qml-resync-plan.md` §1.2 (the measured error) and §6, block 1.

use std::collections::HashMap;

use polypus_circuit::ConcreteCircuit;
use polypus_qml::{
    ConvBlock, Dataset, Decision, HardwareEfficientAnsatz, Layer, Loss, Observable, Pauli,
    PauliString, PoolBlock, QmlProblem, QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, StatevectorSimulator};

/// Central finite-difference step. With exact probabilities the truncation error
/// is `O(h²)` and the rounding error `O(ε/h)`, both far below `TOLERANCE`.
const STEP: f64 = 1e-5;
/// Agreement required between the shift-rule gradient and the finite difference.
const TOLERANCE: f64 = 1e-8;

const ROWS: [[f64; 4]; 3] = [
    [0.4, 1.1, 0.7, 2.0],
    [1.3, 0.2, 2.2, 0.9],
    [2.0, 1.7, 0.3, 1.2],
];

fn dataset(labels: &[f64]) -> Dataset {
    let rows: Vec<Vec<f64>> = ROWS.iter().map(|r| r.to_vec()).collect();
    Dataset::from_rows(&rows, labels).unwrap()
}

/// `Z` on every listed logical position, as one Pauli string.
fn z_string(positions: &[usize]) -> Observable {
    let terms = positions.iter().map(|&p| (p, Pauli::Z)).collect();
    Observable::new(vec![(1.0, PauliString::new(terms).unwrap())]).unwrap()
}

fn raw_readout(positions: &[usize]) -> Readout {
    Readout::new(vec![z_string(positions)], Decision::Raw).unwrap()
}

/// Two classes read out as `⟨Z_a⟩` and `⟨Z_b⟩`, for the categorical loss.
fn argmax_readout(a: usize, b: usize) -> Readout {
    Readout::new(vec![z_string(&[a]), z_string(&[b])], Decision::Argmax).unwrap()
}

/// Exact basis-state probabilities of `circuits`, keyed like the C-3 counts
/// (`width − 1 − k` is qubit `k`).
fn exact_probabilities(circuits: &[ConcreteCircuit]) -> Vec<HashMap<String, f64>> {
    let simulator = StatevectorSimulator::new();
    circuits
        .iter()
        .map(|circuit| {
            let width = match circuit.num_clbits() {
                0 => circuit.num_qubits(),
                clbits => clbits,
            };
            simulator
                .run(circuit)
                .unwrap()
                .probabilities()
                .into_iter()
                .enumerate()
                .map(|(state, p)| (format!("{state:0width$b}"), p))
                .collect()
        })
        .collect()
}

/// Exact probabilities of every training circuit at `theta`.
fn probabilities(problem: &QmlProblem, theta: &[f64]) -> Vec<HashMap<String, f64>> {
    exact_probabilities(&problem.bind_batch(theta).unwrap())
}

/// The gradient as the crate computes it: run every circuit of
/// `gradient_circuits` exactly and hand the results to
/// `gradient_from_probabilities`.
fn crate_gradient(problem: &QmlProblem, theta: &[f64]) -> Vec<f64> {
    let circuits = problem.gradient_circuits(theta).unwrap();
    problem
        .gradient_from_probabilities(&exact_probabilities(&circuits))
        .unwrap()
}

/// Central finite difference of the exact fitness.
fn finite_difference(problem: &QmlProblem, theta: &[f64]) -> Vec<f64> {
    (0..theta.len())
        .map(|k| {
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[k] += STEP;
            minus[k] -= STEP;
            let f_plus = problem
                .fitness_from_probabilities(&probabilities(problem, &plus))
                .unwrap();
            let f_minus = problem
                .fitness_from_probabilities(&probabilities(problem, &minus))
                .unwrap();
            (f_plus - f_minus) / (2.0 * STEP)
        })
        .collect()
}

/// Two parameter points that are neither zero nor symmetric, so no gradient
/// component vanishes by accident.
fn thetas(num_params: usize) -> [Vec<f64>; 2] {
    [
        (0..num_params).map(|i| 0.3 + 0.17 * i as f64).collect(),
        (0..num_params).map(|i| -1.1 + 0.41 * i as f64).collect(),
    ]
}

fn assert_gradient_is_exact(problem: &QmlProblem) {
    for theta in thetas(problem.num_params()) {
        let shift = crate_gradient(problem, &theta);
        let reference = finite_difference(problem, &theta);
        for (k, (s, r)) in shift.iter().zip(&reference).enumerate() {
            assert!(
                (s - r).abs() < TOLERANCE,
                "θ = {theta:?}: ∂fitness/∂θ[{k}] is {s} by parameter shift but {r} by finite difference"
            );
        }
    }
}

fn problem(model: QuantumModel, loss: Loss, labels: &[f64]) -> QmlProblem {
    QmlProblem::new(model.compile(4).unwrap(), dataset(labels), loss).unwrap()
}

const REAL_LABELS: [f64; 3] = [0.5, -0.3, 0.8];

fn encoded(num_qubits: usize) -> QuantumModel {
    QuantumModel::new(num_qubits).angle_encoder(RotationAxis::Ry)
}

// --- Every parameter used once: the shift rule is exact today. ---------------

#[test]
fn hardware_efficient_gradient_is_exact() {
    for readout in [raw_readout(&[1]), raw_readout(&[0, 1, 2, 3])] {
        let model = encoded(4).hardware_efficient(2).readout(readout);
        assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
    }
}

#[test]
fn stacked_iqp_encoders_around_an_ansatz_keep_the_gradient_exact() {
    let model = QuantumModel::new(4)
        .iqp_encoder()
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .iqp_encoder()
        .readout(raw_readout(&[0, 1, 2, 3]));
    assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
}

/// Data re-uploading is another encoder later in the layer list (design doc
/// §6.1): the features come back as fixed angles between trainable layers,
/// which untying must copy, not renumber.
#[test]
fn data_reuploading_keeps_the_gradient_exact() {
    for readout in [raw_readout(&[1]), raw_readout(&[0, 1, 2, 3])] {
        let model = encoded(4)
            .hardware_efficient(1)
            .angle_encoder(RotationAxis::Rx)
            .hardware_efficient(1)
            .readout(readout);
        assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
    }
}

// --- Parameters reused across qubit pairs (regression, plan §1.2). ------------

#[test]
fn basic_convolution_gradient_is_exact() {
    for readout in [raw_readout(&[1]), raw_readout(&[0, 1, 2, 3])] {
        let model = encoded(4).conv(ConvBlock::Basic).readout(readout);
        assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
    }
}

#[test]
fn cartan_convolution_gradient_is_exact() {
    for readout in [raw_readout(&[1]), raw_readout(&[0, 1, 2, 3])] {
        let model = encoded(4).conv(ConvBlock::Cartan).readout(readout);
        assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
    }
}

#[test]
fn pooling_gradient_is_exact() {
    // After pooling 4 qubits, 2 stay active: logical positions 0 and 1.
    for readout in [raw_readout(&[1]), raw_readout(&[0, 1])] {
        let model = encoded(4)
            .conv(ConvBlock::Basic)
            .pool(PoolBlock::Basic)
            .readout(readout);
        assert_gradient_is_exact(&problem(model, Loss::SquaredError, &REAL_LABELS));
    }
}

/// A QCNN that re-uploads the data after pooling: shared parameters on both
/// sides of a second encoder. An angle encoder needs a qubit per feature, so
/// the data has two features (pooling leaves two of the four qubits active).
#[test]
fn reuploading_after_pooling_keeps_the_gradient_exact() {
    let model = encoded(4)
        .conv(ConvBlock::Basic)
        .pool(PoolBlock::Basic)
        .angle_encoder(RotationAxis::Ry)
        .conv(ConvBlock::Cartan)
        .readout(raw_readout(&[0, 1]));
    let rows: Vec<Vec<f64>> = ROWS.iter().map(|r| r[..2].to_vec()).collect();
    let data = Dataset::from_rows(&rows, &REAL_LABELS).unwrap();
    let problem = QmlProblem::new(model.compile(2).unwrap(), data, Loss::SquaredError).unwrap();
    assert_gradient_is_exact(&problem);
}

#[test]
fn categorical_gradient_through_a_convolution_is_exact() {
    let model = encoded(4)
        .conv(ConvBlock::Basic)
        .readout(argmax_readout(1, 2));
    assert_gradient_is_exact(&problem(
        model,
        Loss::CategoricalCrossEntropy,
        &[0.0, 1.0, 1.0],
    ));
}
