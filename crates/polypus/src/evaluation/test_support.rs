//! Shared fixtures for the QML-oracle unit tests.
//!
//! [`native_qml_oracle`](super::native_qml_oracle) and
//! [`exact_native_qml_oracle`](super::exact_native_qml_oracle) build the same two
//! tiny `QmlProblem`s to drive their oracles; this module holds the single
//! definition both `#[cfg(test)] mod tests` import, so the fixtures cannot drift
//! apart. It is `#[cfg(test)]`-gated (declared that way in `evaluation/mod.rs`),
//! so it compiles only under test.

use polypus_qml::{
    Dataset, Decision, Loss, Observable, Pauli, PauliString, QmlProblem, QuantumModel, Readout,
    RotationAxis,
};

/// A tiny, fully-Rust `QmlProblem`: a 2-qubit angle-encoder + hardware-efficient
/// ansatz reading `⟨Z₀⟩` with a `Sign` decision, trained with `Hinge` over two
/// well-separated samples. Its compiled model reserves 8 trainable parameters.
pub(crate) fn small_problem() -> QmlProblem {
    let readout = Readout::new(
        vec![Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap()],
        Decision::Sign,
    )
    .unwrap();
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(readout);
    let ds = Dataset::from_rows(&[vec![0.4, 0.5], vec![2.6, 2.7]], &[-1.0, 1.0]).unwrap();
    let compiled = model.compile(ds.num_features()).unwrap();
    QmlProblem::new(compiled, ds, Loss::Hinge).unwrap()
}

/// A categorical counterpart of [`small_problem`]: the same 2-qubit
/// angle-encoder + hardware-efficient ansatz, but reading **two** observables
/// (`⟨Z₀⟩`, `⟨Z₁⟩`) with an `Argmax` decision and trained with
/// `CategoricalCrossEntropy` over two class-{0,1} samples. Exercises the
/// multiclass branch of `try_gradient`. Its compiled model reserves the same
/// 8 trainable parameters.
pub(crate) fn categorical_problem() -> QmlProblem {
    let readout = Readout::new(
        vec![
            Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())]).unwrap(),
            Observable::new(vec![(1.0, PauliString::new(vec![(1, Pauli::Z)]).unwrap())]).unwrap(),
        ],
        Decision::Argmax,
    )
    .unwrap();
    let model = QuantumModel::new(2)
        .angle_encoder(RotationAxis::Ry)
        .hardware_efficient(1)
        .readout(readout);
    let ds = Dataset::from_rows(&[vec![0.4, 0.5], vec![2.6, 2.7]], &[0.0, 1.0]).unwrap();
    let compiled = model.compile(ds.num_features()).unwrap();
    QmlProblem::new(compiled, ds, Loss::CategoricalCrossEntropy).unwrap()
}
