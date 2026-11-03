//! The adjoint gradient (issue #503) must be the parameter-shift gradient.
//!
//! [`QmlProblem::adjoint_runs`] / [`QmlProblem::gradient_from_adjoint`] and
//! their Jacobian twins hand a statevector simulator one adjoint
//! differentiation per sample (and observable) instead of two shifted circuits
//! per trainable gate. Both are exact, so on every model of the catalogue — every
//! layer, the parameters shared by convolution and pooling, re-uploading, the
//! IQP encoder, every loss and readout, a minibatch — the adjoint result must
//! equal the shift rule's (itself checked against finite differences in
//! `gradient_exactness.rs` and `jacobian.rs`) to rounding.
//!
//! `polypus-sim` runs the circuits, as a dev-dependency only: the crate still
//! executes nothing.

use std::collections::HashMap;

use polypus_circuit::{CircuitError, ConcreteCircuit};
use polypus_qml::{
    AdjointRun, CompiledModel, ConvBlock, Dataset, Decision, Entanglement, HardwareEfficientAnsatz,
    IqpEncoder, Jacobian, JacobianWrt, Layer, Loss, Observable, Pauli, PauliString, PoolBlock,
    QmlError, QmlProblem, QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, StatevectorSimulator};

/// Both gradients are exact; they differ only in rounding (~1e-15 here).
const TOLERANCE: f64 = 1e-9;

const ROWS: [[f64; 4]; 3] = [
    [0.4, 1.1, 0.7, 2.0],
    [1.3, 0.2, 2.2, 0.9],
    [2.0, 1.7, 0.3, 1.2],
];

fn rows() -> Vec<Vec<f64>> {
    ROWS.iter().map(|r| r.to_vec()).collect()
}

fn string(positions: &[usize], pauli: Pauli) -> PauliString {
    PauliString::new(positions.iter().map(|&p| (p, pauli)).collect()).unwrap()
}

fn observable(positions: &[usize], pauli: Pauli) -> Observable {
    Observable::new(vec![(1.0, string(positions, pauli))]).unwrap()
}

fn readout(positions: &[usize], decision: Decision) -> Readout {
    Readout::new(vec![observable(positions, Pauli::Z)], decision).unwrap()
}

fn argmax(observables: Vec<Observable>) -> Readout {
    Readout::new(observables, Decision::Argmax).unwrap()
}

/// Exact basis-state probabilities of `circuits`, keyed like the C-3 counts.
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

/// Run every adjoint differentiation, one derivative row per run.
fn run_adjoint(runs: &[AdjointRun]) -> Vec<Vec<f64>> {
    let simulator = StatevectorSimulator::new();
    runs.iter()
        .map(|run| {
            simulator
                .adjoint_gradient(&run.circuit, &run.wrt, |b| run.observable.weight(b), None)
                .unwrap()
                .derivatives
        })
        .collect()
}

fn shift_gradient(problem: &QmlProblem, theta: &[f64]) -> Vec<f64> {
    let circuits = problem.gradient_circuits(theta).unwrap();
    problem
        .gradient_from_probabilities(&exact_probabilities(&circuits))
        .unwrap()
}

fn adjoint_gradient(problem: &QmlProblem, theta: &[f64]) -> Vec<f64> {
    let base = exact_probabilities(&problem.bind_batch(theta).unwrap());
    let runs = problem.adjoint_runs(theta, &base).unwrap();
    assert_eq!(runs.len(), problem.num_circuits(), "one run per sample");
    // Each run differentiates the very circuit that scores the sample.
    for (run, bound) in runs.iter().zip(problem.bind_batch(theta).unwrap()) {
        assert_eq!(run.circuit, bound);
    }
    problem.gradient_from_adjoint(&run_adjoint(&runs)).unwrap()
}

fn thetas(num_params: usize) -> [Vec<f64>; 2] {
    [
        (0..num_params).map(|i| 0.3 + 0.17 * i as f64).collect(),
        (0..num_params).map(|i| -1.1 + 0.41 * i as f64).collect(),
    ]
}

fn assert_same(what: &str, adjoint: &[f64], shift: &[f64]) {
    assert_eq!(adjoint.len(), shift.len(), "{what}");
    for (k, (a, s)) in adjoint.iter().zip(shift).enumerate() {
        assert!(
            (a - s).abs() < TOLERANCE,
            "{what}: component {k} is {a} by the adjoint but {s} by the shift rule"
        );
    }
}

fn assert_gradient_matches(problem: &QmlProblem, what: &str) {
    for theta in thetas(problem.num_params()) {
        assert_same(
            &format!("{what}, θ = {theta:?}"),
            &adjoint_gradient(problem, &theta),
            &shift_gradient(problem, &theta),
        );
    }
}

fn problem(model: QuantumModel, features: usize, loss: Loss, labels: &[f64]) -> QmlProblem {
    let rows: Vec<Vec<f64>> = ROWS.iter().map(|r| r[..features].to_vec()).collect();
    let dataset = Dataset::from_rows(&rows, labels).unwrap();
    QmlProblem::new(model.compile(features).unwrap(), dataset, loss).unwrap()
}

fn encoded(axis: RotationAxis) -> QuantumModel {
    QuantumModel::new(4).angle_encoder(axis)
}

fn real_amplitudes(reps: usize) -> Layer {
    Layer::HardwareEfficient(HardwareEfficientAnsatz::real_amplitudes(reps))
}

/// Every model shape of the gradient tests, each with the readout `readout`.
fn scalar_models(readout: Readout) -> Vec<(QuantumModel, usize, &'static str)> {
    vec![
        (
            encoded(RotationAxis::Ry)
                .hardware_efficient(2)
                .readout(readout.clone()),
            4,
            "hardware efficient",
        ),
        (
            encoded(RotationAxis::Rx)
                .conv(ConvBlock::Basic)
                .readout(readout.clone()),
            4,
            "basic convolution",
        ),
        (
            encoded(RotationAxis::Rz)
                .hardware_efficient(1)
                .conv(ConvBlock::Cartan)
                .readout(readout.clone()),
            4,
            "cartan convolution",
        ),
        (
            encoded(RotationAxis::Ry)
                .hardware_efficient(1)
                .angle_encoder(RotationAxis::Rx)
                .hardware_efficient(1)
                .readout(readout.clone()),
            4,
            "re-uploading",
        ),
        (
            QuantumModel::new(4)
                .iqp_encoder()
                .layer(real_amplitudes(1))
                .iqp_encoder()
                .readout(readout),
            4,
            "stacked IQP",
        ),
    ]
}

// contract test: needs human sign-off
/// Acceptance criterion 3, scalar losses: every model shape, every scalar
/// loss and decision, `Z` strings, the `X`/`Y` bases and a weighted
/// observable.
#[test]
fn the_adjoint_gradient_is_the_shift_gradient_for_every_scalar_loss() {
    let weighted = Observable::new(vec![
        (0.5, string(&[0], Pauli::Z)),
        (-1.5, string(&[1, 2], Pauli::Z)),
        (2.0, string(&[3], Pauli::Z)),
    ])
    .unwrap();
    let cases = [
        (
            Loss::SquaredError,
            readout(&[1], Decision::Raw),
            vec![0.5, -0.3, 0.8],
        ),
        (
            Loss::SquaredError,
            Readout::new(vec![observable(&[0, 2], Pauli::X)], Decision::Sign).unwrap(),
            vec![0.5, -0.3, 0.8],
        ),
        (
            Loss::BinaryCrossEntropy,
            readout(&[0, 1, 2, 3], Decision::Threshold(0.1)),
            vec![1.0, 0.0, 1.0],
        ),
        (
            Loss::Hinge,
            Readout::new(vec![observable(&[1], Pauli::Y)], Decision::Sign).unwrap(),
            vec![1.0, -1.0, 1.0],
        ),
        (
            Loss::Expectation,
            Readout::new(vec![weighted], Decision::Raw).unwrap(),
            vec![0.0, 0.0, 0.0],
        ),
    ];
    for (loss, readout, labels) in cases {
        for (model, features, name) in scalar_models(readout) {
            let problem = problem(model, features, loss, &labels);
            assert_gradient_matches(&problem, &format!("{loss:?}, {name}"));
        }
    }
}

// contract test: needs human sign-off
/// The categorical loss: one adjoint run per sample covers every class, with
/// the per-class chain-rule weights folded into its observable.
#[test]
fn the_adjoint_gradient_is_the_shift_gradient_for_the_categorical_loss() {
    let classes = argmax(vec![
        observable(&[0], Pauli::Z),
        observable(&[1, 2], Pauli::Z),
        observable(&[3], Pauli::Z),
    ]);
    for (model, features, name) in scalar_models(classes) {
        let problem = problem(
            model,
            features,
            Loss::CategoricalCrossEntropy,
            &[0.0, 2.0, 1.0],
        );
        assert_gradient_matches(&problem, &format!("categorical, {name}"));
    }
}

// contract test: needs human sign-off
/// Convolution and pooling share each parameter across qubit pairs: the
/// adjoint gives one derivative per gate, and the parameter's is their sum.
/// Summing only the first occurrence would miss the other pairs' gates, and
/// fail here (the shift gradient is the sum, checked against finite
/// differences in `gradient_exactness.rs`).
#[test]
fn parameters_shared_by_convolution_and_pooling_sum_every_gate() {
    let pooled = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .pool(PoolBlock::Basic)
        .readout(readout(&[0, 1], Decision::Raw));
    let pooling = problem(pooled, 4, Loss::SquaredError, &[0.5, -0.3, 0.8]);
    let theta = thetas(pooling.num_params())[0].clone();
    let base = exact_probabilities(&pooling.bind_batch(&theta).unwrap());
    let runs = pooling.adjoint_runs(&theta, &base).unwrap();
    assert!(
        runs[0].wrt.len() > pooling.num_params(),
        "the parameters must be shared for this test to mean anything"
    );
    assert_gradient_matches(&pooling, "pooling");

    // A QCNN that re-uploads the data after pooling: shared parameters on
    // both sides of a second encoder (two features: pooling leaves two qubits).
    let reuploaded = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .pool(PoolBlock::Basic)
        .angle_encoder(RotationAxis::Ry)
        .conv(ConvBlock::Cartan)
        .readout(readout(&[0, 1], Decision::Raw));
    let qcnn = problem(reuploaded, 2, Loss::SquaredError, &[0.5, -0.3, 0.8]);
    assert_gradient_matches(&qcnn, "re-uploading after pooling");
}

// contract test: needs human sign-off
/// A minibatch is a subset problem: its adjoint gradient is its shift
/// gradient, and different from the full problem's.
#[test]
fn a_minibatch_differentiates_its_own_samples() {
    let model = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .hardware_efficient(1)
        .readout(readout(&[1, 2], Decision::Raw));
    let full = problem(model, 4, Loss::SquaredError, &[0.5, -0.3, 0.8]);
    let batch = full.subset(&[2, 0]).unwrap();
    assert_gradient_matches(&batch, "minibatch");
    let theta = thetas(full.num_params())[1].clone();
    assert_ne!(
        adjoint_gradient(&batch, &theta),
        adjoint_gradient(&full, &theta)
    );
}

// ── The expectation Jacobian ─────────────────────────────────────────────────

fn shift_jacobian(
    model: &CompiledModel,
    samples: &[Vec<f64>],
    theta: &[f64],
    wrt: JacobianWrt,
) -> Jacobian {
    let jacobian = model.jacobian(samples, wrt).unwrap();
    let circuits = jacobian.circuits(theta).unwrap();
    jacobian
        .jacobian_from_probabilities(&exact_probabilities(&circuits))
        .unwrap()
}

fn adjoint_jacobian(
    model: &CompiledModel,
    samples: &[Vec<f64>],
    theta: &[f64],
    wrt: JacobianWrt,
) -> Jacobian {
    let jacobian = model.jacobian(samples, wrt).unwrap();
    let runs = jacobian.adjoint_runs(theta).unwrap();
    assert_eq!(runs.len(), jacobian.all().num_adjoint_runs());
    assert_eq!(runs.len(), samples.len() * model.num_observables());
    for run in &runs {
        assert_eq!(run.wrt.len(), jacobian.num_shifts());
    }
    jacobian.jacobian_from_adjoint(&run_adjoint(&runs)).unwrap()
}

fn assert_jacobian_matches(
    model: &CompiledModel,
    samples: &[Vec<f64>],
    wrt: JacobianWrt,
    what: &str,
) {
    for theta in thetas(model.num_params()) {
        let adjoint = adjoint_jacobian(model, samples, &theta, wrt);
        let shift = shift_jacobian(model, samples, &theta, wrt);
        assert_eq!(
            adjoint.params().is_some(),
            shift.params().is_some(),
            "{what}"
        );
        assert_eq!(
            adjoint.inputs().is_some(),
            shift.inputs().is_some(),
            "{what}"
        );
        if let (Some(a), Some(s)) = (adjoint.params(), shift.params()) {
            assert_same(&format!("{what}, params, θ = {theta:?}"), a, s);
        }
        if let (Some(a), Some(s)) = (adjoint.inputs(), shift.inputs()) {
            assert_same(&format!("{what}, inputs, θ = {theta:?}"), a, s);
        }
    }
}

/// The readouts of `jacobian.rs`: `Z` strings, the `X`/`Y` bases, a weighted
/// observable, and a three-class `Argmax`.
fn readouts() -> Vec<Readout> {
    let weighted = Observable::new(vec![
        (0.5, string(&[0], Pauli::Z)),
        (-1.5, string(&[1, 2], Pauli::Z)),
        (2.0, string(&[3], Pauli::Z)),
    ])
    .unwrap();
    vec![
        readout(&[1], Decision::Raw),
        readout(&[0, 1, 2, 3], Decision::Raw),
        Readout::new(vec![observable(&[0, 2], Pauli::X)], Decision::Sign).unwrap(),
        Readout::new(vec![observable(&[1], Pauli::Y)], Decision::Raw).unwrap(),
        Readout::new(vec![weighted], Decision::Raw).unwrap(),
        argmax(vec![
            observable(&[0], Pauli::Z),
            observable(&[1, 2], Pauli::Z),
            observable(&[3], Pauli::Z),
        ]),
    ]
}

// contract test: needs human sign-off
/// The parameter Jacobian, every model shape and readout.
#[test]
fn the_adjoint_parameter_jacobian_is_the_shift_jacobian() {
    for readout in readouts() {
        for (model, features, name) in scalar_models(readout) {
            let model = model.compile(features).unwrap();
            assert_jacobian_matches(&model, &rows(), JacobianWrt::Params, name);
        }
    }
}

// contract test: needs human sign-off
/// The input Jacobian, and both at once, for the angle and IQP encoders —
/// re-uploaded, and in every IQP entanglement, whose `Rzz(x_i·x_j)` gates
/// carry a chain-rule factor towards each of their two features.
#[test]
fn the_adjoint_input_jacobian_is_the_shift_jacobian() {
    for wrt in [JacobianWrt::Inputs, JacobianWrt::Both] {
        for readout in [
            readouts().remove(1),
            readouts().remove(3),
            readouts().remove(5),
        ] {
            let reuploaded = encoded(RotationAxis::Ry)
                .hardware_efficient(1)
                .angle_encoder(RotationAxis::Rx)
                .conv(ConvBlock::Basic)
                .angle_encoder(RotationAxis::Rz)
                .readout(readout.clone())
                .compile(4)
                .unwrap();
            assert_jacobian_matches(&reuploaded, &rows(), wrt, "re-uploaded angle encoders");
            for entanglement in [
                Entanglement::Linear,
                Entanglement::Circular,
                Entanglement::Full,
            ] {
                let iqp = || Layer::Iqp(IqpEncoder { entanglement });
                let model = QuantumModel::new(4)
                    .layer(iqp())
                    .layer(real_amplitudes(1))
                    .layer(iqp())
                    .readout(readout.clone())
                    .compile(4)
                    .unwrap();
                assert_jacobian_matches(&model, &rows(), wrt, &format!("IQP {entanglement:?}"));
            }
        }
    }
}

/// Blocks concatenate to the whole Jacobian, value for value, as they do for
/// the shifted circuits.
#[test]
fn adjoint_blocks_concatenate_to_the_whole_jacobian() {
    let model = QuantumModel::new(4)
        .iqp_encoder()
        .conv(ConvBlock::Basic)
        .readout(readouts().remove(5))
        .compile(4)
        .unwrap();
    let samples = rows();
    let theta = thetas(model.num_params())[0].clone();
    let jacobian = model.jacobian(&samples, JacobianWrt::Both).unwrap();
    let whole = jacobian
        .jacobian_from_adjoint(&run_adjoint(&jacobian.adjoint_runs(&theta).unwrap()))
        .unwrap();
    let mut params = Vec::new();
    let mut inputs = Vec::new();
    for block in jacobian.blocks(2) {
        let part = block
            .jacobian_from_adjoint(&run_adjoint(&block.adjoint_runs(&theta).unwrap()))
            .unwrap();
        params.extend_from_slice(part.params().unwrap());
        inputs.extend_from_slice(part.inputs().unwrap());
    }
    assert_eq!(whole.params().unwrap(), params.as_slice());
    assert_eq!(whole.inputs().unwrap(), inputs.as_slice());
}

// ── Errors ──────────────────────────────────────────────────────────────────

#[test]
fn wrong_lengths_are_typed_errors() {
    let model = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .readout(readout(&[1], Decision::Raw));
    let problem = problem(model, 4, Loss::SquaredError, &[0.5, -0.3, 0.8]);
    let theta = vec![0.2; problem.num_params()];
    let base = exact_probabilities(&problem.bind_batch(&theta).unwrap());

    // θ of the wrong length.
    assert_eq!(
        problem.adjoint_runs(&[0.1], &base).unwrap_err(),
        QmlError::Circuit(CircuitError::WrongNumberOfParams {
            expected: problem.num_params(),
            got: 1
        })
    );
    // Base results of the wrong length.
    assert_eq!(
        problem.adjoint_runs(&theta, &base[..2]).unwrap_err(),
        QmlError::CountsLengthMismatch {
            expected: 3,
            got: 2
        }
    );
    // Rows: the wrong number, then one of the wrong length.
    let runs = problem.adjoint_runs(&theta, &base).unwrap();
    let mut derivatives = run_adjoint(&runs);
    let gates = derivatives[0].len();
    assert_eq!(
        problem
            .gradient_from_adjoint(&derivatives[..2])
            .unwrap_err(),
        QmlError::CountsLengthMismatch {
            expected: 3,
            got: 2
        }
    );
    derivatives[1].pop();
    assert_eq!(
        problem.gradient_from_adjoint(&derivatives).unwrap_err(),
        QmlError::CountsLengthMismatch {
            expected: gates,
            got: gates - 1
        }
    );

    let compiled = encoded(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(readouts().remove(5))
        .compile(4)
        .unwrap();
    let jacobian = compiled.jacobian(&rows(), JacobianWrt::Params).unwrap();
    assert_eq!(
        jacobian.adjoint_runs(&[0.1]).unwrap_err(),
        QmlError::Circuit(CircuitError::WrongNumberOfParams {
            expected: compiled.num_params(),
            got: 1
        })
    );
    let theta = vec![0.2; compiled.num_params()];
    let mut rows = run_adjoint(&jacobian.adjoint_runs(&theta).unwrap());
    rows.pop();
    assert_eq!(
        jacobian.jacobian_from_adjoint(&rows).unwrap_err(),
        QmlError::CountsLengthMismatch {
            expected: jacobian.all().num_adjoint_runs(),
            got: jacobian.all().num_adjoint_runs() - 1
        }
    );
}

/// The amplitude and basis encoders still have no input Jacobian: the check
/// is at construction, before any run exists, whichever reader follows.
#[test]
fn an_encoder_without_an_input_derivative_is_refused_before_any_run() {
    let amplitude = QuantumModel::new(2)
        .amplitude_encoder()
        .hardware_efficient(1)
        .readout(readout(&[0], Decision::Raw))
        .compile(4)
        .unwrap();
    assert_eq!(
        amplitude
            .jacobian(&rows(), JacobianWrt::Inputs)
            .unwrap_err(),
        QmlError::NoInputGradient {
            encoder: "amplitude"
        }
    );
    // Its parameter Jacobian is fine, by either route.
    assert_jacobian_matches(
        &amplitude,
        &rows(),
        JacobianWrt::Params,
        "amplitude encoder",
    );
}
