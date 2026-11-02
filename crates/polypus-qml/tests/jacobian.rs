//! The loss-free expectation Jacobian ([`CompiledModel::jacobian`]) must be the
//! derivative of the expectations, and the fitness gradient `Model.fit` trains
//! with must be the chain rule over it.
//!
//! In exact mode there is no shot noise, so the Jacobian assembled from exact
//! probabilities has to match a central finite difference of
//! [`CompiledModel::expectations_from_probabilities`] to near machine
//! precision: with respect to the parameters for every layer and readout, and
//! with respect to the inputs for the angle and IQP encoders (re-uploaded or
//! not). Same `STEP` and `TOLERANCE` as `gradient_exactness.rs`.
//!
//! Contract C-11, "The expectation Jacobian (Rust)". Draft pending human
//! sign-off.

use std::collections::HashMap;

use polypus_circuit::{CircuitError, ConcreteCircuit, GateInstruction, GateParam};
use polypus_qml::{
    CompiledModel, ConvBlock, Dataset, Decision, Entanglement, HardwareEfficientAnsatz, IqpEncoder,
    Jacobian, JacobianWrt, Layer, Loss, Observable, Pauli, PauliString, PoolBlock, QmlError,
    QmlProblem, QuantumModel, Readout, RotationAxis,
};
use polypus_sim::{Simulator, StatevectorSimulator};

/// Central finite-difference step. With exact probabilities the truncation error
/// is `O(h²)` and the rounding error `O(ε/h)`, both far below `TOLERANCE`.
const STEP: f64 = 1e-5;
/// Agreement required between the shift-rule Jacobian and the finite difference.
const TOLERANCE: f64 = 1e-8;

const ROWS: [[f64; 4]; 3] = [
    [0.4, 1.1, 0.7, 2.0],
    [1.3, 0.2, 2.2, 0.9],
    [2.0, 1.7, 0.3, 1.2],
];

fn rows() -> Vec<Vec<f64>> {
    ROWS.iter().map(|r| r.to_vec()).collect()
}

/// `pauli` on every listed logical position, as one Pauli string.
fn string(positions: &[usize], pauli: Pauli) -> PauliString {
    PauliString::new(positions.iter().map(|&p| (p, pauli)).collect()).unwrap()
}

fn observable(positions: &[usize], pauli: Pauli) -> Observable {
    Observable::new(vec![(1.0, string(positions, pauli))]).unwrap()
}

fn raw(positions: &[usize]) -> Readout {
    Readout::new(vec![observable(positions, Pauli::Z)], Decision::Raw).unwrap()
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

/// Every observable's exact expectation for sample `x` at `theta`.
fn expectations(model: &CompiledModel, x: &[f64], theta: &[f64]) -> Vec<f64> {
    let circuit = model.bind(x, theta).unwrap();
    let probabilities = exact_probabilities(&[circuit]);
    model
        .expectations_from_probabilities(&probabilities[0])
        .unwrap()
}

/// The Jacobian as the crate computes it, in exact mode.
fn crate_jacobian(
    model: &CompiledModel,
    samples: &[Vec<f64>],
    theta: &[f64],
    wrt: JacobianWrt,
) -> Jacobian {
    let jacobian = model.jacobian(samples, wrt).unwrap();
    let circuits = jacobian.circuits(theta).unwrap();
    assert_eq!(circuits.len(), jacobian.num_circuits());
    jacobian
        .jacobian_from_probabilities(&exact_probabilities(&circuits))
        .unwrap()
}

/// Two parameter points that are neither zero nor symmetric, so no derivative
/// vanishes by accident.
fn thetas(num_params: usize) -> [Vec<f64>; 2] {
    [
        (0..num_params).map(|i| 0.3 + 0.17 * i as f64).collect(),
        (0..num_params).map(|i| -1.1 + 0.41 * i as f64).collect(),
    ]
}

fn assert_close(what: &str, shift: f64, reference: f64) {
    assert!(
        (shift - reference).abs() < TOLERANCE,
        "{what}: {shift} by parameter shift but {reference} by finite difference"
    );
}

/// `∂⟨O_o⟩/∂θ` against a central finite difference, every sample, observable
/// and parameter, at two points.
fn assert_param_jacobian_is_exact(model: &CompiledModel) {
    let samples = rows();
    for theta in thetas(model.num_params()) {
        let jacobian = crate_jacobian(model, &samples, &theta, JacobianWrt::Params);
        assert!(jacobian.inputs().is_none());
        for (s, x) in samples.iter().enumerate() {
            for p in 0..model.num_params() {
                let (mut plus, mut minus) = (theta.clone(), theta.clone());
                plus[p] += STEP;
                minus[p] -= STEP;
                let (e_plus, e_minus) = (
                    expectations(model, x, &plus),
                    expectations(model, x, &minus),
                );
                for o in 0..model.num_observables() {
                    let reference = (e_plus[o] - e_minus[o]) / (2.0 * STEP);
                    assert_close(
                        &format!("θ = {theta:?}, ∂⟨O_{o}⟩(x_{s})/∂θ_{p}"),
                        jacobian.param(s, o, p).unwrap(),
                        reference,
                    );
                }
            }
        }
    }
}

/// `∂⟨O_o⟩/∂x` against a central finite difference in each feature.
fn assert_input_jacobian_is_exact(model: &CompiledModel) {
    let samples = rows();
    for theta in thetas(model.num_params()) {
        let jacobian = crate_jacobian(model, &samples, &theta, JacobianWrt::Inputs);
        assert!(jacobian.params().is_none());
        for (s, x) in samples.iter().enumerate() {
            for f in 0..model.num_features() {
                let (mut plus, mut minus) = (x.clone(), x.clone());
                plus[f] += STEP;
                minus[f] -= STEP;
                let (e_plus, e_minus) = (
                    expectations(model, &plus, &theta),
                    expectations(model, &minus, &theta),
                );
                for o in 0..model.num_observables() {
                    let reference = (e_plus[o] - e_minus[o]) / (2.0 * STEP);
                    assert_close(
                        &format!("θ = {theta:?}, ∂⟨O_{o}⟩(x_{s})/∂x_{f}"),
                        jacobian.input(s, o, f).unwrap(),
                        reference,
                    );
                }
            }
        }
    }
}

fn encoded(axis: RotationAxis) -> QuantumModel {
    QuantumModel::new(4).angle_encoder(axis)
}

fn real_amplitudes(reps: usize) -> Layer {
    Layer::HardwareEfficient(HardwareEfficientAnsatz::real_amplitudes(reps))
}

/// Every readout the parameter Jacobian is checked under: `Z` strings, the
/// `X` and `Y` bases, a weighted multi-term observable and a multiclass
/// `Argmax` readout (which `QmlProblem` only takes with the categorical loss).
fn readouts() -> Vec<Readout> {
    let weighted = Observable::new(vec![
        (0.5, string(&[0], Pauli::Z)),
        (-1.5, string(&[1, 2], Pauli::Z)),
        (2.0, string(&[3], Pauli::Z)),
    ])
    .unwrap();
    vec![
        raw(&[1]),
        raw(&[0, 1, 2, 3]),
        Readout::new(vec![observable(&[0, 2], Pauli::X)], Decision::Sign).unwrap(),
        Readout::new(vec![observable(&[1], Pauli::Y)], Decision::Raw).unwrap(),
        Readout::new(vec![weighted], Decision::Raw).unwrap(),
        Readout::new(
            vec![
                observable(&[0], Pauli::Z),
                observable(&[1, 2], Pauli::Z),
                observable(&[3], Pauli::Z),
            ],
            Decision::Argmax,
        )
        .unwrap(),
    ]
}

// --- Criterion: parameter Jacobian, every layer and readout. ---------------

#[test]
fn the_parameter_jacobian_is_exact_for_every_angle_encoder() {
    for axis in [RotationAxis::Rx, RotationAxis::Ry, RotationAxis::Rz] {
        for readout in readouts() {
            let model = encoded(axis).hardware_efficient(1).readout(readout);
            assert_param_jacobian_is_exact(&model.compile(4).unwrap());
        }
    }
}

#[test]
fn the_parameter_jacobian_is_exact_for_both_ansatz_presets() {
    for readout in readouts() {
        let model = encoded(RotationAxis::Ry)
            .hardware_efficient(2)
            .layer(real_amplitudes(1))
            .readout(readout);
        assert_param_jacobian_is_exact(&model.compile(4).unwrap());
    }
}

/// Convolution and pooling share each parameter across qubit pairs: the
/// Jacobian must sum the per-gate shifts, as the fitness gradient does.
#[test]
fn the_parameter_jacobian_is_exact_for_shared_convolution_and_pooling_parameters() {
    for block in [ConvBlock::Basic, ConvBlock::Cartan] {
        for readout in readouts() {
            let model = encoded(RotationAxis::Ry).conv(block).readout(readout);
            assert_param_jacobian_is_exact(&model.compile(4).unwrap());
        }
    }
    // Pooling to two qubits, read out on what is left.
    for readout in [
        raw(&[0, 1]),
        Readout::new(
            vec![observable(&[0], Pauli::Z), observable(&[1], Pauli::Z)],
            Decision::Argmax,
        )
        .unwrap(),
    ] {
        let model = encoded(RotationAxis::Ry)
            .conv(ConvBlock::Basic)
            .pool(PoolBlock::Basic)
            .readout(readout);
        assert_param_jacobian_is_exact(&model.compile(4).unwrap());
    }
}

#[test]
fn the_parameter_jacobian_is_exact_with_reuploading_and_iqp() {
    for readout in readouts() {
        let reuploaded = encoded(RotationAxis::Ry)
            .hardware_efficient(1)
            .angle_encoder(RotationAxis::Rx)
            .hardware_efficient(1)
            .readout(readout.clone());
        assert_param_jacobian_is_exact(&reuploaded.compile(4).unwrap());
        let iqp = QuantumModel::new(4)
            .iqp_encoder()
            .layer(real_amplitudes(1))
            .iqp_encoder()
            .readout(readout);
        assert_param_jacobian_is_exact(&iqp.compile(4).unwrap());
    }
}

// --- Criterion: input Jacobian. -----------------------------------------------

#[test]
fn the_input_jacobian_is_exact_for_every_angle_encoder() {
    for axis in [RotationAxis::Rx, RotationAxis::Ry, RotationAxis::Rz] {
        for readout in [raw(&[0, 1, 2, 3]), readouts().remove(5)] {
            let model = encoded(axis).hardware_efficient(1).readout(readout);
            assert_input_jacobian_is_exact(&model.compile(4).unwrap());
        }
    }
}

/// The same feature read by two encoders: the derivative sums both.
#[test]
fn the_input_jacobian_is_exact_with_reuploading() {
    for readout in [raw(&[1]), raw(&[0, 1, 2, 3]), readouts().remove(2)] {
        let model = encoded(RotationAxis::Ry)
            .hardware_efficient(1)
            .angle_encoder(RotationAxis::Rx)
            .conv(ConvBlock::Basic)
            .angle_encoder(RotationAxis::Rz)
            .readout(readout);
        assert_input_jacobian_is_exact(&model.compile(4).unwrap());
    }
}

/// `Rz(x_i)` has factor `1`, `Rzz(x_i·x_j)` has `x_j` towards `x_i` and `x_i`
/// towards `x_j`: the chain rule the encoders record.
#[test]
fn the_input_jacobian_is_exact_for_iqp_in_every_entanglement() {
    for entanglement in [
        Entanglement::Linear,
        Entanglement::Circular,
        Entanglement::Full,
    ] {
        let iqp = || Layer::Iqp(IqpEncoder { entanglement });
        for readout in [raw(&[0, 1, 2, 3]), readouts().remove(4)] {
            let once = QuantumModel::new(4)
                .layer(iqp())
                .layer(real_amplitudes(1))
                .readout(readout.clone());
            assert_input_jacobian_is_exact(&once.compile(4).unwrap());
            let reuploaded = QuantumModel::new(4)
                .layer(iqp())
                .layer(real_amplitudes(1))
                .layer(iqp())
                .readout(readout);
            assert_input_jacobian_is_exact(&reuploaded.compile(4).unwrap());
        }
    }
}

/// Fewer features than qubits: the surplus qubits read nothing.
#[test]
fn the_input_jacobian_is_exact_with_fewer_features_than_qubits() {
    let samples: Vec<Vec<f64>> = ROWS.iter().map(|r| r[..2].to_vec()).collect();
    let model = QuantumModel::new(4)
        .iqp_encoder()
        .hardware_efficient(1)
        .readout(raw(&[0, 1, 2, 3]))
        .compile(2)
        .unwrap();
    let theta = thetas(model.num_params())[0].clone();
    let jacobian = crate_jacobian(&model, &samples, &theta, JacobianWrt::Inputs);
    for (s, x) in samples.iter().enumerate() {
        for f in 0..2 {
            let (mut plus, mut minus) = (x.clone(), x.clone());
            plus[f] += STEP;
            minus[f] -= STEP;
            let reference = (expectations(&model, &plus, &theta)[0]
                - expectations(&model, &minus, &theta)[0])
                / (2.0 * STEP);
            assert_close(
                &format!("∂⟨O⟩(x_{s})/∂x_{f}"),
                jacobian.input(s, 0, f).unwrap(),
                reference,
            );
        }
    }
}

#[test]
fn both_is_the_two_jacobians_of_separate_calls_bit_for_bit() {
    let samples = rows();
    for readout in readouts() {
        let model = QuantumModel::new(4)
            .iqp_encoder()
            .conv(ConvBlock::Basic)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(4)
            .unwrap();
        let theta = thetas(model.num_params())[1].clone();
        let both = crate_jacobian(&model, &samples, &theta, JacobianWrt::Both);
        let params = crate_jacobian(&model, &samples, &theta, JacobianWrt::Params);
        let inputs = crate_jacobian(&model, &samples, &theta, JacobianWrt::Inputs);
        assert_eq!(both.params(), params.params());
        assert_eq!(both.inputs(), inputs.inputs());
        assert!(both.params().is_some() && both.inputs().is_some());
    }
}

// --- Criterion: coherence with `Model.fit`'s gradient. ------------------------

/// `d loss / d ⟨O₀⟩` for each scalar loss, spelled out here as the closed
/// forms of `Loss::gradient` so the test does not borrow the crate's code.
fn loss_derivative(loss: Loss, e: f64, y: f64) -> f64 {
    const BCE_EPSILON: f64 = 1e-12;
    match loss {
        Loss::SquaredError => 2.0 * (e - y),
        Loss::BinaryCrossEntropy => {
            let p = (1.0 + e) / 2.0;
            assert!(p > BCE_EPSILON && p < 1.0 - BCE_EPSILON);
            (p - y) / (2.0 * p * (1.0 - p))
        }
        Loss::Hinge => {
            if 1.0 - y * e > 0.0 {
                -y
            } else {
                0.0
            }
        }
        Loss::Expectation => -1.0,
        Loss::CategoricalCrossEntropy => unreachable!("vector-valued, see below"),
    }
}

/// `softmax(z) − onehot(y)`, the categorical loss's derivative per class.
fn categorical_derivative(z: &[f64], y: usize) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter()
        .enumerate()
        .map(|(j, e)| e / sum - if j == y { 1.0 } else { 0.0 })
        .collect()
}

/// `−(1/n)·Σ_i Σ_o ∂loss/∂⟨O_o⟩(x_i) · J[i, o, :]`: the fitness gradient as the
/// chain rule over the Jacobian.
fn chain_rule(model: &CompiledModel, loss: Loss, labels: &[f64], theta: &[f64]) -> Vec<f64> {
    let samples = rows();
    let jacobian = crate_jacobian(model, &samples, theta, JacobianWrt::Params);
    let n = samples.len();
    let mut gradient = vec![0.0; model.num_params()];
    for (i, x) in samples.iter().enumerate() {
        let e = expectations(model, x, theta);
        let factors = match loss {
            Loss::CategoricalCrossEntropy => categorical_derivative(&e, labels[i] as usize),
            _ => {
                let mut factors = vec![0.0; e.len()];
                factors[0] = loss_derivative(loss, e[0], labels[i]);
                factors
            }
        };
        for (o, factor) in factors.iter().enumerate() {
            for (p, g) in gradient.iter_mut().enumerate() {
                *g += factor * jacobian.param(i, o, p).unwrap();
            }
        }
    }
    gradient.iter().map(|g| -g / n as f64).collect()
}

/// `QmlProblem`'s exact gradient, the one `Model.fit` trains with.
fn fit_gradient(model: &CompiledModel, loss: Loss, labels: &[f64], theta: &[f64]) -> Vec<f64> {
    let dataset = Dataset::from_rows(&rows(), labels).unwrap();
    let problem = QmlProblem::new(model.clone(), dataset, loss).unwrap();
    let circuits = problem.gradient_circuits(theta).unwrap();
    problem
        .gradient_from_probabilities(&exact_probabilities(&circuits))
        .unwrap()
}

#[test]
fn the_chain_rule_over_the_jacobian_is_the_fitness_gradient_for_every_loss() {
    let scalar = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .hardware_efficient(1);
    let cases = [
        (Loss::SquaredError, vec![0.5, -0.3, 0.8]),
        (Loss::BinaryCrossEntropy, vec![1.0, 0.0, 1.0]),
        (Loss::Hinge, vec![1.0, -1.0, 1.0]),
        (Loss::Expectation, vec![0.0, 0.0, 0.0]),
    ];
    for (loss, labels) in cases {
        let model = scalar.clone().readout(raw(&[1, 2])).compile(4).unwrap();
        for theta in thetas(model.num_params()) {
            let fit = fit_gradient(&model, loss, &labels, &theta);
            let chained = chain_rule(&model, loss, &labels, &theta);
            for (p, (f, c)) in fit.iter().zip(&chained).enumerate() {
                assert!(
                    (f - c).abs() < 1e-12,
                    "{loss:?}, θ = {theta:?}: ∂fitness/∂θ_{p} is {f} for fit, {c} by the chain rule"
                );
            }
        }
    }
    // The categorical loss scores every class.
    let labels = vec![0.0, 2.0, 1.0];
    let model = scalar.readout(readouts().remove(5)).compile(4).unwrap();
    for theta in thetas(model.num_params()) {
        let fit = fit_gradient(&model, Loss::CategoricalCrossEntropy, &labels, &theta);
        let chained = chain_rule(&model, Loss::CategoricalCrossEntropy, &labels, &theta);
        for (p, (f, c)) in fit.iter().zip(&chained).enumerate() {
            assert!(
                (f - c).abs() < 1e-12,
                "categorical, θ = {theta:?}: ∂fitness/∂θ_{p} is {f} for fit, {c} by the chain rule"
            );
        }
    }
}

// --- Criterion: circuit count. ---------------------------------------------

/// The gates of `x`'s template that read a trainable parameter: the
/// occurrences, counted on the circuit rather than by the crate.
fn param_occurrences(model: &CompiledModel, x: &[f64]) -> usize {
    model
        .template_for(x)
        .unwrap()
        .gates()
        .iter()
        .filter(|gate| {
            matches!(
                gate,
                GateInstruction::Rx {
                    theta: GateParam::Param(_),
                    ..
                } | GateInstruction::Ry {
                    theta: GateParam::Param(_),
                    ..
                } | GateInstruction::Rz {
                    theta: GateParam::Param(_),
                    ..
                } | GateInstruction::Rxx {
                    theta: GateParam::Param(_),
                    ..
                } | GateInstruction::Rzz {
                    theta: GateParam::Param(_),
                    ..
                }
            )
        })
        .count()
}

#[test]
fn the_circuit_count_is_two_per_shifted_gate_per_sample() {
    let samples = rows();
    let n = samples.len();
    // A basic convolution over 4 qubits: 4 parameters on 3 pairs, so 12
    // occurrences for 4 parameters. Then an angle re-upload (4 input gates)
    // and an IQP over 4 features in full (4 `Rz` + 6 `Rzz`, each `Rzz`
    // shifted once for both of its features).
    let model = encoded(RotationAxis::Ry)
        .conv(ConvBlock::Basic)
        .iqp_encoder()
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    let occ_params = param_occurrences(&model, &samples[0]);
    assert_eq!((occ_params, model.num_params()), (12, 4));
    let occ_inputs = 4 + 4 + 6;
    for (wrt, shifts) in [
        (JacobianWrt::Params, occ_params),
        (JacobianWrt::Inputs, occ_inputs),
        (JacobianWrt::Both, occ_params + occ_inputs),
    ] {
        let jacobian = model.jacobian(&samples, wrt).unwrap();
        assert_eq!(jacobian.num_shifts(), shifts, "{wrt:?}");
        assert_eq!(jacobian.circuits_per_sample(), 2 * shifts, "{wrt:?}");
        assert_eq!(jacobian.num_circuits(), n * 2 * shifts, "{wrt:?}");
        let theta = vec![0.1; model.num_params()];
        assert_eq!(
            jacobian.circuits(&theta).unwrap().len(),
            n * 2 * shifts,
            "{wrt:?}"
        );
    }
}

// --- Order and blocks. ---------------------------------------------------------

/// The circuits are sample-major, `+π/2` then `−π/2` per shifted gate in gate
/// order: the first two of each sample shift the first trainable gate.
#[test]
fn circuits_are_sample_major_with_the_plus_shift_first() {
    let samples = rows();
    let model = encoded(RotationAxis::Ry)
        .layer(real_amplitudes(1))
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    let theta: Vec<f64> = (0..model.num_params()).map(|i| 0.1 * i as f64).collect();
    let jacobian = model.jacobian(&samples, JacobianWrt::Params).unwrap();
    let circuits = jacobian.circuits(&theta).unwrap();
    let per_sample = jacobian.circuits_per_sample();
    let shift = std::f64::consts::FRAC_PI_2;
    for (s, x) in samples.iter().enumerate() {
        for (delta, circuit) in [shift, -shift]
            .iter()
            .zip(&circuits[s * per_sample..s * per_sample + 2])
        {
            let mut shifted = theta.clone();
            shifted[0] += delta;
            assert_eq!(*circuit, model.bind(x, &shifted).unwrap(), "sample {s}");
        }
    }
}

#[test]
fn blocks_concatenate_to_the_whole_jacobian() {
    let samples = rows();
    let model = QuantumModel::new(4)
        .iqp_encoder()
        .hardware_efficient(1)
        .readout(readouts().remove(5))
        .compile(4)
        .unwrap();
    let theta = thetas(model.num_params())[0].clone();
    let jacobian = model.jacobian(&samples, JacobianWrt::Both).unwrap();
    let whole = jacobian
        .jacobian_from_probabilities(&exact_probabilities(&jacobian.circuits(&theta).unwrap()))
        .unwrap();
    for size in [0, 1, 2, 3, 10] {
        let (mut params, mut inputs) = (Vec::new(), Vec::new());
        let mut blocks = 0;
        for block in jacobian.blocks(size) {
            let circuits = block.circuits(&theta).unwrap();
            assert_eq!(circuits.len(), block.num_circuits());
            let part = block
                .jacobian_from_probabilities(&exact_probabilities(&circuits))
                .unwrap();
            assert_eq!(part.num_samples(), block.num_samples());
            let (p, i) = part.into_parts();
            params.extend(p.unwrap());
            inputs.extend(i.unwrap());
            blocks += 1;
        }
        assert_eq!(blocks, samples.len().div_ceil(size.max(1)), "size {size}");
        assert_eq!(Some(params.as_slice()), whole.params(), "size {size}");
        assert_eq!(Some(inputs.as_slice()), whole.inputs(), "size {size}");
    }
    // A block by range is the same block; a range past the samples is `None`.
    let middle = jacobian.block(1..2).unwrap();
    let from_range = middle
        .jacobian_from_probabilities(&exact_probabilities(&middle.circuits(&theta).unwrap()))
        .unwrap();
    let per_sample = whole.num_observables() * model.num_params();
    assert_eq!(
        from_range.params(),
        whole.params().map(|p| &p[per_sample..2 * per_sample])
    );
    assert!(jacobian.block(2..4).is_none());
    assert_eq!(jacobian.block(3..3).map(|b| b.num_samples()), Some(0));
}

/// From counts the Jacobian is an estimator: counts proportional to the exact
/// probabilities give it back to the counts' resolution.
#[test]
fn jacobian_from_counts_reads_the_same_shifts() {
    let samples = rows();
    let model = encoded(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(raw(&[0, 1]))
        .compile(4)
        .unwrap();
    let theta = thetas(model.num_params())[0].clone();
    let jacobian = model.jacobian(&samples, JacobianWrt::Both).unwrap();
    let probabilities = exact_probabilities(&jacobian.circuits(&theta).unwrap());
    let counts: Vec<HashMap<String, u64>> = probabilities
        .iter()
        .map(|p| {
            p.iter()
                .map(|(k, v)| (k.clone(), (v * 1e9).round() as u64))
                .collect()
        })
        .collect();
    let exact = jacobian
        .jacobian_from_probabilities(&probabilities)
        .unwrap();
    let sampled = jacobian.jacobian_from_counts(&counts).unwrap();
    for (e, c) in exact
        .params()
        .unwrap()
        .iter()
        .chain(exact.inputs().unwrap())
        .zip(
            sampled
                .params()
                .unwrap()
                .iter()
                .chain(sampled.inputs().unwrap()),
        )
    {
        assert!((e - c).abs() < 1e-7, "{e} vs {c}");
    }
}

// --- Errors. -------------------------------------------------------------------

/// Amplitude and basis encoding have no derivative in the sample: the input
/// Jacobian is a typed error naming the encoder, the parameter one works.
#[test]
fn amplitude_and_basis_encoders_have_no_input_jacobian() {
    let amplitude = QuantumModel::new(2)
        .amplitude_encoder()
        .hardware_efficient(1)
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    let basis = QuantumModel::new(4)
        .angle_encoder(RotationAxis::Ry)
        .hardware_efficient(1)
        .basis_encoder()
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    let binary = vec![vec![1.0, 0.0, 1.0, 1.0], vec![0.0, 1.0, 0.0, 0.0]];
    for (model, samples, encoder) in [(amplitude, rows(), "amplitude"), (basis, binary, "basis")] {
        for wrt in [JacobianWrt::Inputs, JacobianWrt::Both] {
            assert_eq!(
                model.jacobian(&samples, wrt).unwrap_err(),
                QmlError::NoInputGradient { encoder },
                "{encoder}, {wrt:?}"
            );
        }
        let theta = thetas(model.num_params())[0].clone();
        let jacobian = model.jacobian(&samples, JacobianWrt::Params).unwrap();
        assert!(jacobian.circuits(&theta).is_ok(), "{encoder}");
        let error = QmlError::NoInputGradient { encoder }.to_string();
        assert!(error.contains(encoder), "{error}");
    }
}

/// `QmlProblem` refuses an `Argmax` readout without the categorical loss; the
/// Jacobian has no loss, so it takes one.
#[test]
fn an_argmax_readout_is_accepted_without_a_loss() {
    let model = encoded(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(readouts().remove(5))
        .compile(4)
        .unwrap();
    let dataset = Dataset::from_rows(&rows(), &[0.0, 0.0, 0.0]).unwrap();
    assert!(QmlProblem::new(model.clone(), dataset, Loss::SquaredError).is_err());
    let jacobian = model.jacobian(&rows(), JacobianWrt::Params).unwrap();
    assert_eq!(jacobian.num_observables(), 3);
}

#[test]
fn wrong_lengths_are_typed_errors() {
    let model = encoded(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    // A sample of the wrong width, at whichever position.
    let mut samples = rows();
    samples[2].pop();
    assert_eq!(
        model.jacobian(&samples, JacobianWrt::Params).unwrap_err(),
        QmlError::FeatureCountMismatch {
            expected: 4,
            got: 3
        }
    );
    let jacobian = model.jacobian(&rows(), JacobianWrt::Params).unwrap();
    // θ of the wrong length.
    assert_eq!(
        jacobian.circuits(&[0.1]).unwrap_err(),
        QmlError::Circuit(CircuitError::WrongNumberOfParams {
            expected: model.num_params(),
            got: 1
        })
    );
    // Results of the wrong length.
    let theta = vec![0.2; model.num_params()];
    let mut probabilities = exact_probabilities(&jacobian.circuits(&theta).unwrap());
    probabilities.pop();
    assert_eq!(
        jacobian
            .jacobian_from_probabilities(&probabilities)
            .unwrap_err(),
        QmlError::CountsLengthMismatch {
            expected: jacobian.num_circuits(),
            got: jacobian.num_circuits() - 1
        }
    );
}

/// No sample: no circuit and an empty Jacobian, never a panic.
#[test]
fn no_samples_is_an_empty_jacobian() {
    let model = encoded(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(raw(&[0]))
        .compile(4)
        .unwrap();
    let samples: Vec<Vec<f64>> = Vec::new();
    let jacobian = model.jacobian(&samples, JacobianWrt::Both).unwrap();
    assert_eq!(jacobian.num_circuits(), 0);
    let theta = vec![0.0; model.num_params()];
    assert!(jacobian.circuits(&theta).unwrap().is_empty());
    let empty = jacobian.jacobian_from_probabilities(&[]).unwrap();
    assert_eq!(empty.params(), Some(&[][..]));
    assert_eq!(empty.inputs(), Some(&[][..]));
}
