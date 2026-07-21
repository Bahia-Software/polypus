//! End-to-end native training: the executable proof of the crate's promise —
//! a full QML classifier trained in pure Rust, with no Python in the loop
//! (design doc §13.4).
//!
//! A small, clearly separable 2D dataset is encoded with an angle encoder and a
//! `real_amplitudes(1)` ansatz, read out with `⟨Z₀⟩`, and trained with the
//! `Hinge` loss via Differential Evolution. The scorer is a bespoke
//! [`EvaluationOracle`] that binds each candidate's parameters into circuits,
//! simulates them on the statevector backend, converts the sampled counts into
//! the C-3 bitstring format, and asks the [`QmlProblem`] for a fitness. After
//! optimization the trained parameters must classify the whole training set
//! perfectly.
//!
//! `polypus-sim` and `polypus-optimizers` enter only as dev-dependencies here —
//! the crate's public API never executes a circuit or runs an optimizer.

use std::collections::HashMap;

use polypus_optimizers::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, EvaluationOracle, Optimizer,
};
use polypus_qml::{
    Dataset, Decision, HardwareEfficientAnsatz, Layer, Loss, Observable, Pauli, PauliString,
    QmlProblem, QuantumModel, Readout, RotationAxis,
};
use polypus_sim::StatevectorSimulator;

/// Convert `polypus-sim`'s `HashMap<state_index, count>` into the C-3 bitstring
/// format `expectation_from_counts` expects. Same pattern as
/// `crates/polypus/src/infrastructure/native.rs`: standard binary formatting,
/// no bit reversal — the character at `width - 1 - k` is qubit `k`.
fn to_bitstring_counts(raw: HashMap<usize, u64>, width: usize) -> HashMap<String, u64> {
    raw.into_iter()
        .map(|(state, count)| (format!("{state:0width$b}"), count))
        .collect()
}

/// A test-only oracle: for each candidate `θ`, bind → simulate → count → score.
/// It **owns** the problem (the optimizer's `Box<dyn EvaluationOracle>` is
/// `'static`, so it cannot borrow one from the test's stack).
struct QmlTestOracle {
    problem: QmlProblem,
    shots: usize,
    seed: u64,
    simulator: StatevectorSimulator,
}

impl EvaluationOracle for QmlTestOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        candidates
            .iter()
            .map(|theta| {
                let circuits = self
                    .problem
                    .bind_batch(theta)
                    .expect("bind_batch on a valid θ length");
                let counts: Vec<HashMap<String, u64>> = circuits
                    .iter()
                    .enumerate()
                    .map(|(i, circuit)| {
                        // Per-sample seed offset: deterministic w.r.t. the outer
                        // seed, but decorrelates shot noise across samples.
                        let raw = self
                            .simulator
                            .run_and_sample(circuit, self.shots, self.seed + i as u64)
                            .expect("statevector simulation succeeds");
                        to_bitstring_counts(raw, circuit.num_qubits)
                    })
                    .collect();
                self.problem
                    .fitness_from_counts(&counts)
                    .expect("fitness on valid counts")
            })
            .collect()
    }
}

/// Two well-separated clouds in `[0, 3]²`, labelled `{−1, +1}`, then min–max
/// scaled to `[0, π]` for angle encoding.
fn separable_dataset() -> Dataset {
    let rows = vec![
        // Cloud A near (0.5, 0.5) → label −1.
        vec![0.5, 0.5],
        vec![0.6, 0.4],
        vec![0.4, 0.6],
        vec![0.5, 0.7],
        // Cloud B near (2.5, 2.5) → label +1.
        vec![2.5, 2.5],
        vec![2.6, 2.4],
        vec![2.4, 2.6],
        vec![2.5, 2.3],
    ];
    let labels = vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
    let mut ds = Dataset::from_rows(&rows, &labels).unwrap();
    ds.scale_features_to(0.0, std::f64::consts::PI);
    ds
}

#[test]
fn native_training_reaches_perfect_train_accuracy() {
    let train = separable_dataset();

    // Sanity: no feature column collapsed to a constant under scaling (which
    // would destroy separability). Both columns must span a real range.
    for (lo, hi) in train.feature_ranges() {
        assert!(hi - lo > 1e-9, "a feature column collapsed to a constant");
    }

    // Model: Ry angle encoder + real_amplitudes(1) + ⟨Z₀⟩ readout, Sign.
    let z0 = Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap();
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(Readout::new(vec![z0], Decision::Sign).unwrap())
        .compile(2)
        .unwrap();

    let problem = QmlProblem::new(model, train.clone(), Loss::Hinge).unwrap();

    // Capture the optimizer dimensions before the problem moves into the oracle.
    let dimensions = problem.num_params() as u32;
    let shots = 2048;
    let sample_seed = 1234;

    let oracle = QmlTestOracle {
        problem: problem.clone(),
        shots,
        seed: sample_seed,
        simulator: StatevectorSimulator::new(),
    };

    let outcome = AlgorithmDifferentialEvolution
        .optimize(AlgorithmDifferentialEvolutionArgs {
            oracle: Box::new(oracle),
            population_size: 30,
            generations: 150,
            dimensions,
            tolerance: 1e-6,
            seed: Some(7),
        })
        .expect("DE optimizes successfully");

    // Evaluate the trained parameters on the whole training set: sign of each
    // prediction must match the label — accuracy 1.0.
    let simulator = StatevectorSimulator::new();
    let circuits = problem
        .bind_batch(&outcome.best_params)
        .expect("bind trained θ");
    let mut correct = 0usize;
    for (i, circuit) in circuits.iter().enumerate() {
        let raw = simulator
            .run_and_sample(circuit, shots, sample_seed + i as u64)
            .expect("simulation succeeds");
        let counts = to_bitstring_counts(raw, circuit.num_qubits);
        let prediction = problem.predict_from_counts(&counts).expect("prediction");
        if prediction == train.labels()[i] {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / train.num_samples() as f64;
    assert_eq!(
        accuracy,
        1.0,
        "expected perfect train accuracy, got {accuracy} ({correct}/{})",
        train.num_samples()
    );
}
