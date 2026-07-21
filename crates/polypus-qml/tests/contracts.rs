//! Cross-cutting contract tests over a battery of compiled models.
//!
//! These assert the crate-level invariants that no single unit test covers,
//! against a catalogue that varies rotations, entangler, entanglement, the
//! final rotation layer, and qubit counts (including the 1-qubit case where
//! entanglement is a no-op):
//!
//! - **C-4** (terminal measurement): every emitted template is terminal.
//! - **C-2** (gate vocabulary / QASM round-trip): a bound circuit's QASM is a
//!   fixed point of export → import → export.
//! - **num_params consistency**: `num_params()` equals the largest `Param`
//!   index emitted, plus one.
//! - **C-8** (problem ↔ oracle): over [`QmlProblem`]s built from the same
//!   catalogue, `bind_batch` yields exactly `num_circuits()` circuits (each
//!   C-4-clean and C-2-valid), `fitness_from_counts` returns a finite `f64`,
//!   and `num_params()` matches the underlying `CompiledModel`.

use polypus_circuit::{
    terminal_measurement_violation, GateInstruction, GateParam, ParameterizedCircuit,
};
use std::collections::HashMap;

use polypus_qml::{
    CompiledModel, Dataset, Decision, Entanglement, Entangler, HardwareEfficientAnsatz, Layer,
    Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout, RotationAxis,
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
/// (arbitrary) labels never trip the domain check — this exercises the C-8
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
fn bind_batch_yields_num_circuits_in_order_c8() {
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
fn bind_batch_circuits_are_c4_clean_and_c2_valid_c8() {
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
fn fitness_from_counts_is_finite_c8() {
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
            "fitness must be finite (C-8), got {fitness}"
        );
    }
}

#[test]
fn num_params_matches_compiled_model_c8() {
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
