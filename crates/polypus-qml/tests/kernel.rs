//! The quantum kernel (`FeatureMap`, `FidelityKernel`) against closed forms,
//! through the reference simulator (a dev-dependency): every encoder's kernel
//! is the formula it should be, the two routes — the `0…0` probability of the
//! compute-uncompute circuit and the inner product of the two states — agree,
//! the diagonal is one, and the feature map emits the gates a model's encoders
//! emit.

use std::collections::HashMap;

use polypus_circuit::ConcreteCircuit;
use polypus_qml::{
    AmplitudeEncoder, AngleEncoder, BasisEncoder, Decision, Entanglement, FeatureMap,
    FidelityKernel, IqpEncoder, Layer, Observable, Pauli, PauliString, QuantumModel, Readout,
    RotationAxis,
};
use polypus_sim::{Simulator, StatevectorSimulator, C64};

/// How far two exact routes may differ: each value is a sum of a few dozen
/// products of amplitudes that went through at most a few dozen gates, every
/// one rounding at the `f64` epsilon (`2.2e-16`), so their gap is a small
/// multiple of it. `1e-12` leaves four orders of magnitude of slack for that
/// and still catches any wrong sign, order or angle, which moves a kernel
/// value by far more.
const ROUNDING: f64 = 1e-12;

fn statevector(circuit: &ConcreteCircuit) -> Vec<C64> {
    StatevectorSimulator::new()
        .run(circuit)
        .unwrap()
        .amplitudes()
        .to_vec()
}

/// Every entry by the inner product of the two states, in pair order.
fn by_inner_products(kernel: &FidelityKernel) -> Vec<f64> {
    let rows: Vec<_> = kernel.row_states().iter().map(statevector).collect();
    let columns: Vec<_> = kernel.column_states().iter().map(statevector).collect();
    kernel
        .pairs(0..kernel.num_pairs())
        .unwrap()
        .into_iter()
        .map(|(i, j)| {
            let overlap: C64 = rows[i]
                .iter()
                .zip(&columns[j])
                .map(|(a, b)| a.conj() * b)
                .sum();
            overlap.norm_sqr()
        })
        .collect()
}

/// Every entry from the exact probabilities of its compute-uncompute circuit,
/// keyed as C-3 counts, through the kernel's own reader.
fn by_compute_uncompute(kernel: &FidelityKernel) -> Vec<f64> {
    let width = kernel.num_qubits();
    kernel
        .circuits(0..kernel.num_pairs())
        .unwrap()
        .unwrap()
        .iter()
        .map(|circuit| {
            let probabilities: HashMap<String, f64> = StatevectorSimulator::new()
                .run(circuit)
                .unwrap()
                .probabilities()
                .into_iter()
                .enumerate()
                .filter(|&(_, p)| p != 0.0)
                .map(|(state, p)| (format!("{state:0width$b}"), p))
                .collect();
            kernel.fidelity_from_probabilities(&probabilities).unwrap()
        })
        .collect()
}

/// The matrix by both routes, checked to agree to rounding; the compute-uncompute one.
fn exact_matrix(kernel: &FidelityKernel) -> Vec<f64> {
    let inner = by_inner_products(kernel);
    let compute = by_compute_uncompute(kernel);
    for (k, (a, b)) in inner.iter().zip(&compute).enumerate() {
        assert!((a - b).abs() < ROUNDING, "pair {k}: {a} != {b}");
    }
    kernel.matrix(&compute).unwrap()
}

fn assert_matrix(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < ROUNDING, "entry {k}: {g} != {w}");
    }
}

fn map(num_qubits: usize, layers: Vec<Layer>, num_features: usize) -> FeatureMap {
    FeatureMap::compile(num_qubits, layers, num_features).unwrap()
}

#[test]
fn an_angle_kernel_is_the_cosine_squared_of_half_the_difference() {
    let feature_map = map(
        1,
        vec![Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry))],
        1,
    );
    let x = [vec![0.3], vec![-1.2], vec![2.5]];
    let y = [vec![0.7], vec![3.0]];
    let kernel = FidelityKernel::new(&feature_map, &x, Some(&y)).unwrap();
    let want: Vec<f64> = x
        .iter()
        .flat_map(|a| y.iter().map(move |b| ((a[0] - b[0]) / 2.0).cos().powi(2)))
        .collect();
    assert_matrix(&exact_matrix(&kernel), &want);
}

#[test]
fn surplus_qubits_do_not_change_the_kernel() {
    // Two features on four qubits: the two untouched qubits stay |0⟩ in both
    // states, so the kernel is the product over the encoded features.
    let feature_map = map(
        4,
        vec![Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry))],
        2,
    );
    let x = [vec![0.3, 1.0], vec![-0.4, 0.2]];
    let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
    let k = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(u, v)| ((u - v) / 2.0).cos().powi(2))
            .product::<f64>()
    };
    let want = [
        k(&x[0], &x[0]),
        k(&x[0], &x[1]),
        k(&x[1], &x[0]),
        k(&x[1], &x[1]),
    ];
    assert_matrix(&exact_matrix(&kernel), &want);
}

#[test]
fn a_basis_kernel_is_one_for_equal_samples_and_zero_otherwise() {
    let feature_map = map(3, vec![Layer::Basis(BasisEncoder)], 3);
    let x = [
        vec![1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
    ];
    let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
    let want: Vec<f64> = x
        .iter()
        .flat_map(|a| x.iter().map(move |b| f64::from(u8::from(a == b))))
        .collect();
    assert_matrix(&exact_matrix(&kernel), &want);
}

#[test]
fn an_amplitude_kernel_is_the_squared_cosine_of_the_normalized_samples() {
    let feature_map = map(2, vec![Layer::AmplitudeEncoder(AmplitudeEncoder)], 3);
    let x = [
        vec![1.0, 2.0, 0.5],
        vec![0.2, 0.0, 3.0],
        vec![4.0, 1.0, 1.0],
    ];
    let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
    let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
    let want: Vec<f64> = x
        .iter()
        .flat_map(|a| {
            x.iter().map(move |b| {
                let dot: f64 = a.iter().zip(b).map(|(u, v)| u * v).sum();
                (dot / (norm(a) * norm(b))).powi(2)
            })
        })
        .collect();
    assert_matrix(&exact_matrix(&kernel), &want);
}

/// `|⟨φ(x)|φ(y)⟩|²` of the IQP map by its closed form: `H⊗n` then diagonal
/// phases, so `|φ(x)⟩ = 2^(-n/2) Σ_b e^(-i·f(b, x)/2) |b⟩` with
/// `f(b, x) = Σ_i x_i z_i + Σ_(i,j) x_i x_j z_i z_j` (`z = +1` for a `0` bit,
/// `−1` for a `1`) over the pattern's pairs.
fn iqp_closed_form(x: &[f64], y: &[f64], pairs: &[(usize, usize)]) -> f64 {
    let n = x.len();
    let f = |b: usize, v: &[f64]| {
        let z = |i: usize| if (b >> i) & 1 == 0 { 1.0 } else { -1.0 };
        (0..n).map(|i| v[i] * z(i)).sum::<f64>()
            + pairs
                .iter()
                .map(|&(i, j)| v[i] * v[j] * z(i) * z(j))
                .sum::<f64>()
    };
    let overlap: C64 = (0..1usize << n)
        .map(|b| C64::from_polar(1.0, (f(b, x) - f(b, y)) / 2.0))
        .sum::<C64>()
        / f64::from(1u32 << n);
    overlap.norm_sqr()
}

/// Qubits, the IQP pattern, and the pairs it entangles.
type IqpCase = (usize, Entanglement, Vec<(usize, usize)>);

#[test]
fn an_iqp_kernel_is_its_closed_form_for_every_entanglement() {
    let cases: [IqpCase; 6] = [
        (2, Entanglement::Linear, vec![(0, 1)]),
        (2, Entanglement::Full, vec![(0, 1)]),
        (3, Entanglement::Linear, vec![(0, 1), (1, 2)]),
        (3, Entanglement::Circular, vec![(0, 1), (1, 2), (2, 0)]),
        (3, Entanglement::Full, vec![(0, 1), (0, 2), (1, 2)]),
        (2, Entanglement::Circular, vec![(0, 1), (1, 0)]),
    ];
    for (n, entanglement, pairs) in cases {
        let feature_map = map(n, vec![Layer::Iqp(IqpEncoder { entanglement })], n);
        let x: Vec<Vec<f64>> = (0..3)
            .map(|s| {
                (0..n)
                    .map(|i| 0.4 * s as f64 - 0.7 * i as f64 + 0.2)
                    .collect()
            })
            .collect();
        let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
        let want: Vec<f64> = x
            .iter()
            .flat_map(|a| x.iter().map(|b| iqp_closed_form(a, b, &pairs)))
            .collect();
        let got = exact_matrix(&kernel);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < ROUNDING,
                "{n} qubits, {entanglement:?}, entry {k}: {g} != {w}"
            );
        }
    }
}

#[test]
fn stacked_encoders_compose() {
    // Two `Ry(x)` layers are `Ry(2x)`: `cos²(x − y)`.
    let ry = || Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry));
    let feature_map = map(1, vec![ry(), ry()], 1);
    let x = [vec![0.25], vec![1.4], vec![-0.9]];
    let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
    let want: Vec<f64> = x
        .iter()
        .flat_map(|a| x.iter().map(move |b| (a[0] - b[0]).cos().powi(2)))
        .collect();
    assert_matrix(&exact_matrix(&kernel), &want);
}

#[test]
fn the_diagonal_is_executed_and_one_to_rounding_and_the_matrix_symmetric() {
    let feature_map = map(
        3,
        vec![
            Layer::Iqp(IqpEncoder::new()),
            Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Rx)),
        ],
        3,
    );
    let x: Vec<Vec<f64>> = (0..4)
        .map(|s| vec![0.3 * s as f64, -0.5, 1.0 - 0.2 * s as f64])
        .collect();
    let kernel = FidelityKernel::new(&feature_map, &x, None).unwrap();
    let matrix = exact_matrix(&kernel);
    for i in 0..4 {
        assert!((matrix[i * 4 + i] - 1.0).abs() < ROUNDING);
        for j in 0..4 {
            assert_eq!(matrix[i * 4 + j].to_bits(), matrix[j * 4 + i].to_bits());
            assert!((0.0..=1.0).contains(&matrix[i * 4 + j]));
        }
    }
    // The diagonal's circuits are run: each is U(x_i) then U(x_i)†.
    let circuits = kernel.circuits(0..kernel.num_pairs()).unwrap().unwrap();
    assert_eq!(circuits.len(), 10);
    assert!(circuits[0].gates().len() > 1);
}

#[test]
fn the_feature_map_emits_the_gates_of_a_models_encoders() {
    let encoders = vec![
        Layer::Iqp(IqpEncoder {
            entanglement: Entanglement::Linear,
        }),
        Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Rz)),
    ];
    let readout = Readout::new(
        vec![Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap()],
        Decision::Raw,
    )
    .unwrap();
    let model = encoders
        .iter()
        .cloned()
        .fold(QuantumModel::new(3), QuantumModel::layer)
        .hardware_efficient(1)
        .readout(readout)
        .compile(3)
        .unwrap();
    let feature_map = map(3, encoders, 3);
    let x = [0.2, -1.3, 0.8];
    let template = model.template_for(&x).unwrap();
    let state = feature_map.circuit(&x).unwrap();
    assert_eq!(state.gates(), &template.gates()[..state.gates().len()]);
    assert_eq!(state.num_qubits(), 3);
}
