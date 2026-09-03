//! Cross-cutting contract tests over a battery of compiled models.
//!
//! These assert the crate-level invariants that no single unit test covers,
//! against a catalogue that varies the encoder, rotations, entangler,
//! entanglement, the final rotation layer, and qubit counts (including the
//! 1-qubit case where entanglement is a no-op, and a stack of two IQP encoders
//! around an ansatz — the data re-uploading pattern):
//!
//! - **C-4** (terminal measurement): every emitted template is terminal.
//! - **C-2** (gate vocabulary / QASM round-trip): a bound circuit's QASM is a
//!   fixed point of export → import → export.
//! - **num_params consistency**: `num_params()` equals the largest `Param`
//!   index emitted, plus one.
//! - **C-10** (problem ↔ oracle): over [`QmlProblem`]s built from the same
//!   catalogue, `bind_batch` yields exactly `num_circuits()` circuits (each
//!   C-4-clean and C-2-valid), `fitness_from_counts` returns a finite `f64`,
//!   and `num_params()` matches the underlying `CompiledModel`.
//! - **C-10, readout revalidation**: `compile` re-runs the readout's own
//!   construction checks, so a readout mutated past `Readout::new` cannot reach
//!   inference.
//! - **C-10, no zero-sample problem**: a `Dataset` is non-empty at *both* of its
//!   construction points, so no public path — `train_test_split`'s floor
//!   rounding included — can reach a `QmlProblem` whose mean fitness would be
//!   `NaN`.

use polypus_circuit::{
    terminal_measurement_violation, GateInstruction, GateParam, ParameterizedCircuit,
};
use std::collections::HashMap;

use polypus_qml::{
    CompiledModel, Dataset, Decision, Entanglement, Entangler, HardwareEfficientAnsatz, IqpEncoder,
    Layer, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout, RotationAxis,
    ValidationError,
};

/// A minimal `⟨Z₀⟩` / `Sign` readout, valid for every catalogue model (all
/// have at least one active qubit at logical position 0).
fn z0_readout() -> Readout {
    Readout::new(
        vec![Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap()],
        Decision::Sign,
    )
    .unwrap()
}

/// The model catalogue. Each entry is a compiled model paired with a few valid
/// samples to template.
fn catalogue() -> Vec<(CompiledModel, Vec<Vec<f64>>)> {
    fn samples(num_features: usize) -> Vec<Vec<f64>> {
        // A handful of finite feature vectors of the right width.
        vec![
            (0..num_features).map(|j| 0.1 * (j as f64 + 1.0)).collect(),
            (0..num_features).map(|j| -0.3 * (j as f64 + 1.0)).collect(),
            vec![0.0; num_features],
        ]
    }

    vec![
        // 1. 2 qubits: Ry encoder + real_amplitudes(1) (Ry, Cx linear, final).
        (
            QuantumModel::new(2)
                .angle_encoder(RotationAxis::Ry)
                .layer(Layer::HardwareEfficient(
                    HardwareEfficientAnsatz::real_amplitudes(1),
                ))
                .readout(z0_readout())
                .compile(2)
                .unwrap(),
            samples(2),
        ),
        // 2. 3 qubits: Rz encoder (prepends H) + default HWE(2) ([Ry,Rz], Cx, linear, final).
        (
            QuantumModel::new(3)
                .angle_encoder(RotationAxis::Rz)
                .hardware_efficient(2)
                .readout(z0_readout())
                .compile(3)
                .unwrap(),
            samples(3),
        ),
        // 3. 4 qubits: Rx encoder + HWE {reps 1, [Ry], Cz, Full, no final}.
        (
            QuantumModel::new(4)
                .angle_encoder(RotationAxis::Rx)
                .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz {
                    reps: 1,
                    rotations: vec![RotationAxis::Ry],
                    entangler: Entangler::Cz,
                    entanglement: Entanglement::Full,
                    final_rotation_layer: false,
                }))
                .readout(z0_readout())
                .compile(4)
                .unwrap(),
            samples(4),
        ),
        // 4. 3 qubits, only 2 features (surplus qubit unencoded): HWE {reps 2,
        //    [Rx,Ry], Cx, Circular, final}.
        (
            QuantumModel::new(3)
                .angle_encoder(RotationAxis::Ry)
                .layer(Layer::HardwareEfficient(HardwareEfficientAnsatz {
                    reps: 2,
                    rotations: vec![RotationAxis::Rx, RotationAxis::Ry],
                    entangler: Entangler::Cx,
                    entanglement: Entanglement::Circular,
                    final_rotation_layer: true,
                }))
                .readout(z0_readout())
                .compile(2)
                .unwrap(),
            samples(2),
        ),
        // 5. 1 qubit: Ry encoder + default HWE(1) — entanglement is a no-op.
        (
            QuantumModel::new(1)
                .angle_encoder(RotationAxis::Ry)
                .hardware_efficient(1)
                .readout(z0_readout())
                .compile(1)
                .unwrap(),
            samples(1),
        ),
        // 6. 3 qubits: IQP encoder (Full) + real_amplitudes(1) — the only
        //    catalogue entry emitting `Rzz` with `Fixed` angles.
        (
            QuantumModel::new(3)
                .iqp_encoder()
                .layer(Layer::HardwareEfficient(
                    HardwareEfficientAnsatz::real_amplitudes(1),
                ))
                .readout(z0_readout())
                .compile(3)
                .unwrap(),
            samples(3),
        ),
        // 7. 4 qubits, 3 features (one surplus qubit the IQP encoder leaves
        //    entirely alone): two stacked IQP encoders around an ansatz — the
        //    data re-uploading pattern that replaces a `reps` field (§6.6), and
        //    with it the case of an encoder that is *not* the first layer.
        (
            QuantumModel::new(4)
                .layer(Layer::Iqp(IqpEncoder {
                    entanglement: Entanglement::Linear,
                }))
                .hardware_efficient(1)
                .layer(Layer::Iqp(IqpEncoder {
                    entanglement: Entanglement::Circular,
                }))
                .readout(z0_readout())
                .compile(3)
                .unwrap(),
            samples(3),
        ),
    ]
}

#[test]
fn every_template_is_terminal_c4() {
    for (model, samples) in catalogue() {
        for x in &samples {
            let template = model.template_for(x).unwrap();
            assert_eq!(
                terminal_measurement_violation(&template.gates),
                None,
                "template must satisfy C-4 (terminal measurement)"
            );
        }
    }
}

#[test]
fn bound_circuit_qasm_is_a_fixed_point_c2() {
    for (model, samples) in catalogue() {
        // Deterministic, finite parameter vector of the right length.
        let theta: Vec<f64> = (0..model.num_params())
            .map(|k| 0.05 * (k as f64 + 1.0))
            .collect();
        for x in &samples {
            let concrete = model.bind(x, &theta).unwrap();
            let qasm1 = concrete.to_qasm2();
            let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
            let qasm2 = imported.to_qasm2_with_params(&[]).unwrap();
            assert_eq!(qasm1, qasm2, "QASM export must be a fixed point (C-2)");
        }
    }
}

#[test]
fn num_params_matches_largest_param_index() {
    for (model, samples) in catalogue() {
        // The template is independent of x for parameter structure; use the
        // first sample.
        let template = model.template_for(&samples[0]).unwrap();
        let mut max_index: Option<usize> = None;
        for gate in &template.gates {
            let theta = match gate {
                GateInstruction::Rx { theta, .. }
                | GateInstruction::Ry { theta, .. }
                | GateInstruction::Rz { theta, .. } => Some(theta),
                _ => None,
            };
            // Only trainable Param rotations count; Fixed ones come from the
            // encoder.
            if let Some(GateParam::Param(i)) = theta {
                max_index = Some(max_index.map_or(*i, |m: usize| m.max(*i)));
            }
        }
        let expected = max_index.map_or(0, |m| m + 1);
        assert_eq!(
            model.num_params(),
            expected,
            "num_params() must equal the largest Param index + 1"
        );
    }
}

/// Build a `QmlProblem` from each catalogue model plus a small synthetic
/// dataset of the right width. `SquaredError` accepts any finite label, so the
/// (arbitrary) labels never trip the domain check — this exercises the C-10
/// surface, not the loss's label validation.
fn problems() -> Vec<QmlProblem> {
    catalogue()
        .into_iter()
        .map(|(model, samples)| {
            let labels: Vec<f64> = (0..samples.len())
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let train = Dataset::from_rows(&samples, &labels).unwrap();
            QmlProblem::new(model, train, Loss::SquaredError).unwrap()
        })
        .collect()
}

/// A deterministic, finite parameter vector of length `n`.
fn theta(n: usize) -> Vec<f64> {
    (0..n).map(|k| 0.05 * (k as f64 + 1.0)).collect()
}

#[test]
fn bind_batch_yields_num_circuits_in_order_c10() {
    // Rebuild the catalogue models in parallel to read back num_params (the
    // problem exposes it too, checked separately below).
    for problem in problems() {
        let circuits = problem.bind_batch(&theta(problem.num_params())).unwrap();
        assert_eq!(
            circuits.len(),
            problem.num_circuits(),
            "bind_batch must return exactly num_circuits() circuits"
        );
    }
}

#[test]
fn bind_batch_circuits_are_c4_clean_and_c2_valid_c10() {
    for problem in problems() {
        let circuits = problem.bind_batch(&theta(problem.num_params())).unwrap();
        for concrete in &circuits {
            // C-4: terminal measurement.
            assert_eq!(
                terminal_measurement_violation(&concrete.gates),
                None,
                "bind_batch circuit must satisfy C-4"
            );
            // C-2: QASM export is a fixed point.
            let qasm1 = concrete.to_qasm2();
            let imported = ParameterizedCircuit::from_qasm2(&qasm1).unwrap();
            let qasm2 = imported.to_qasm2_with_params(&[]).unwrap();
            assert_eq!(
                qasm1, qasm2,
                "bind_batch circuit QASM must be a fixed point (C-2)"
            );
        }
    }
}

#[test]
fn fitness_from_counts_is_finite_c10() {
    for problem in problems() {
        let circuits = problem.bind_batch(&theta(problem.num_params())).unwrap();
        // Synthetic counts (hand-built, no simulation): each circuit gets a
        // full-shot map over "0…0" and "0…01" of the circuit's register width.
        let width = circuits[0].num_qubits;
        let zeros = "0".repeat(width);
        let one = format!("{}1", "0".repeat(width - 1));
        let counts: Vec<HashMap<String, u64>> = circuits
            .iter()
            .map(|_| {
                let mut m = HashMap::new();
                m.insert(zeros.clone(), 512u64);
                m.insert(one.clone(), 512u64);
                m
            })
            .collect();
        let fitness = problem.fitness_from_counts(&counts).unwrap();
        assert!(
            fitness.is_finite(),
            "fitness must be finite (C-10), got {fitness}"
        );
    }
}

#[test]
fn num_params_matches_compiled_model_c10() {
    // The problem's num_params must equal the underlying compiled model's,
    // and be > 0 (NoTrainableParams is rejected at compile).
    for (model, samples) in catalogue() {
        let expected = model.num_params();
        let labels = vec![1.0; samples.len()];
        let train = Dataset::from_rows(&samples, &labels).unwrap();
        let problem = QmlProblem::new(model, train, Loss::SquaredError).unwrap();
        assert_eq!(problem.num_params(), expected);
        assert!(problem.num_params() > 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C-10 · `compile` revalidates the readout it is handed
// ─────────────────────────────────────────────────────────────────────────────
//
// `Readout::observables` and `Observable::terms` are public and mutable, so a
// readout can reach `compile` having never passed `Readout::new` /
// `Observable::new` — mutated in place (below), or deserialized straight from a
// save file (the `TrainedModel::load` path, covered by the unit tests in
// `model.rs`). `compile` must therefore re-run those construction checks and
// fail with the *same* typed `ValidationError`, instead of trusting its input
// and handing back a `CompiledModel` that indexes past `observables[0]` at the
// first `predict`, or returns `Ok(NaN)` from `fitness_from_counts` in violation
// of C-10(b) ("returns a finite f64 … never NaN").
//
// Each assertion below doubles as a no-panic assertion: a panic in `compile` or
// a silently-accepted model would fail the test just as loudly as a wrong error.

/// A 2-qubit, 2-feature model carrying `readout`, ready to compile.
fn model_with(readout: Readout) -> QuantumModel {
    QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .layer(Layer::HardwareEfficient(
            HardwareEfficientAnsatz::real_amplitudes(1),
        ))
        .readout(readout)
}

/// A unit-weight `⟨Z_position⟩` observable.
fn z_observable(position: usize) -> Observable {
    Observable::new(vec![(
        1.0,
        PauliString::new(vec![(position, Pauli::Z)]).unwrap(),
    )])
    .unwrap()
}

#[test]
fn compile_rejects_readout_mutated_to_an_incompatible_observable_count_c10() {
    // `Sign` reads `observables[0]`; empty the vector after construction.
    let mut readout = z0_readout();
    readout.observables.clear();
    assert_eq!(
        model_with(readout).compile(2).unwrap_err(),
        ValidationError::DecisionObservableMismatch {
            decision: Decision::Sign,
            num_observables: 0,
        }
    );

    // `Argmax` needs two observables; drop one after construction.
    let mut readout =
        Readout::new(vec![z_observable(0), z_observable(1)], Decision::Argmax).unwrap();
    readout.observables.truncate(1);
    assert_eq!(
        model_with(readout).compile(2).unwrap_err(),
        ValidationError::DecisionObservableMismatch {
            decision: Decision::Argmax,
            num_observables: 1,
        }
    );
}

#[test]
fn compile_rejects_readout_mutated_to_a_non_finite_coefficient_c10() {
    // A `NaN` appended as a second term: reported at its own index, not the
    // first — the same `term_index` `Observable::new` would have reported.
    let mut readout = z0_readout();
    readout.observables[0]
        .terms
        .push((f64::NAN, PauliString::new(vec![(1, Pauli::Z)]).unwrap()));
    assert_eq!(
        model_with(readout).compile(2).unwrap_err(),
        ValidationError::NonFiniteCoefficient { term_index: 1 }
    );

    // Infinities are non-finite too, and are caught wherever they sit.
    let mut readout = z0_readout();
    readout.observables[0].terms[0].0 = f64::INFINITY;
    assert_eq!(
        model_with(readout).compile(2).unwrap_err(),
        ValidationError::NonFiniteCoefficient { term_index: 0 }
    );

    // A non-first observable is checked too, not just `observables[0]`.
    let mut readout =
        Readout::new(vec![z_observable(0), z_observable(1)], Decision::Argmax).unwrap();
    readout.observables[1].terms[0].0 = f64::NEG_INFINITY;
    assert_eq!(
        model_with(readout).compile(2).unwrap_err(),
        ValidationError::NonFiniteCoefficient { term_index: 0 }
    );
}

#[test]
fn compile_still_accepts_an_untouched_readout_c10() {
    // The guard rejects only what is actually broken: the same readouts, left
    // as `Readout::new` built them, still compile.
    assert!(model_with(z0_readout()).compile(2).is_ok());
    let argmax = Readout::new(vec![z_observable(0), z_observable(1)], Decision::Argmax).unwrap();
    assert!(model_with(argmax).compile(2).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// C-10 · a zero-sample problem is unreachable, so the mean fitness is never NaN
// ─────────────────────────────────────────────────────────────────────────────
//
// `fitness_from_counts` averages the loss over the training samples, so a
// `QmlProblem` carrying zero of them would compute `-0.0 / 0.0` and return
// `Ok(NaN)` — exactly what C-10(b) ("a finite `f64` … never `NaN`") forbids. The
// guarantee rests on `Dataset` being non-empty *by construction*:
// `Dataset::from_rows` and the in-crate `Dataset::select` are the only two ways
// to build one and both reject an empty sample set, so no zero-sample `Dataset`
// — hence no zero-circuit `QmlProblem` — can exist.
//
// This walks the public path that used to produce one: a dataset small enough
// that `train_test_split`'s documented floor rounding empties a partition, which
// the `(0, 1)` fraction guard cannot see. The failure must land at the
// construction point, typed, rather than downstream as a `NaN` fitness.

#[test]
fn a_rounding_emptied_split_partition_cannot_reach_a_problem_c10() {
    let model = catalogue().swap_remove(0).0;
    // Five samples of the model's width, so `floor(5 * 0.1) == 0` leaves the
    // test partition empty while `0.1` sits well inside the open interval.
    let rows: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![0.1 * (i as f64 + 1.0); model.num_features()])
        .collect();
    let labels = vec![1.0, -1.0, 1.0, -1.0, 1.0];
    let full = Dataset::from_rows(&rows, &labels).unwrap();

    assert_eq!(
        full.train_test_split(0.1, 7),
        Err(ValidationError::EmptyDataset),
        "a split that rounds a partition down to nothing must fail, \
         not hand back a zero-sample Dataset"
    );

    // Every partition that *does* come back is a usable, non-degenerate problem:
    // at least one circuit, and a finite (never `NaN`) fitness over it.
    let (train, test) = full.train_test_split(0.4, 7).unwrap();
    assert_eq!((train.num_samples(), test.num_samples()), (3, 2));
    for partition in [train, test] {
        let problem = QmlProblem::new(model.clone(), partition, Loss::SquaredError).unwrap();
        assert!(
            problem.num_circuits() >= 1,
            "a QmlProblem always has at least one circuit (C-10)"
        );
        let circuits = problem.bind_batch(&theta(problem.num_params())).unwrap();
        let width = circuits[0].num_qubits;
        let zeros = "0".repeat(width);
        let counts: Vec<HashMap<String, u64>> = circuits
            .iter()
            .map(|_| HashMap::from([(zeros.clone(), 1024u64)]))
            .collect();
        let fitness = problem.fitness_from_counts(&counts).unwrap();
        assert!(
            fitness.is_finite(),
            "fitness must be finite (C-10), got {fitness}"
        );
    }
}
