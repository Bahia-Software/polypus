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

use polypus_circuit::{
    terminal_measurement_violation, GateInstruction, GateParam, ParameterizedCircuit,
};
use polypus_qml::{
    CompiledModel, Decision, Entanglement, Entangler, HardwareEfficientAnsatz, Layer, Observable,
    Pauli, PauliString, QuantumModel, Readout, RotationAxis,
};

/// A minimal `⟨Z₀⟩` / `Sign` readout, valid for every catalogue model (all
/// have at least one active qubit at logical position 0).
fn z0_readout() -> Readout {
    Readout::new(
        vec![
            Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap(),
        ],
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
