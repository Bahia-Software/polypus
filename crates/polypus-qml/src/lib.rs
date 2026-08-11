//! # polypus-qml
//!
//! Pure-Rust quantum machine learning for Polypus. No PyO3, no Python, no
//! knowledge of execution backends.
//!
//! The crate builds quantum machine-learning workloads end to end on top of
//! Polypus: validated datasets, layered variational models (encoders, ansatz,
//! convolution, pooling), Pauli readout and losses. It *produces*
//! [`polypus_circuit`] circuits and *consumes* measurement counts; it never
//! executes circuits itself — backend selection, batching and distribution
//! stay in `crates/polypus`. Optimization likewise stays decoupled, reached
//! through the `polypus-optimizers` `EvaluationOracle` contract.
//!
//! ## Reproducibility
//!
//! All randomness in this crate takes an explicit `u64` seed and runs through
//! a crate-local `SplitMix64`; the pure crate never reads OS entropy. Resolving
//! `None → OS entropy` is the responsibility of the bindings layer. In this
//! phase the only randomness is the dataset split; parameter initialization
//! joins it later.
//!
//! ## Current scope
//!
//! End-to-end training in pure Rust is now possible. On top of [`Dataset`]
//! (validated construction, deterministic splits, feature scaling) and the
//! layered model builder ([`QuantumModel`] → [`CompiledModel`]) with its
//! [`Layer`] catalogue ([`AngleEncoder`], [`AmplitudeEncoder`], [`IqpEncoder`],
//! [`BasisEncoder`], [`HardwareEfficientAnsatz`], [`ConvLayer`], [`PoolLayer`]),
//! a model now carries a [`Readout`] — Pauli [`Observable`]s plus a [`Decision`] — and
//! a [`Loss`] closes the loop: [`QmlProblem`] bundles a compiled model, a
//! training set and a loss into the pair of operations an optimizer oracle
//! needs (bind parameters into circuits; turn measurement counts into a
//! fitness), producing a fully trainable model without any Python in the loop.
//!
//! The convolution ([`ConvLayer`]) and unitary-pooling ([`PoolLayer`]) layers
//! that build QCNNs are now available, as are the amplitude-encoding layer
//! ([`AmplitudeEncoder`], `O(2^k)` state preparation), the IQP /
//! `ZZFeatureMap` encoder ([`IqpEncoder`], a non-linear feature map) and the
//! computational-basis encoder ([`BasisEncoder`], one `X` per set bit of a
//! binary sample) — four feature encoders in all. Readout
//! observables may be measured in the `X`/`Y` bases as long as the whole
//! readout resolves to a single basis group (design doc §7.2).

mod dataset;
mod error;
mod layers;
mod loss;
mod model;
mod observables;
mod problem;
mod readout;
mod rng;

pub use dataset::Dataset;
pub use error::{QmlError, ValidationError};
pub use layers::{
    AmplitudeEncoder, AngleEncoder, BasisEncoder, ConvBlock, ConvLayer, Entanglement, Entangler,
    HardwareEfficientAnsatz, IqpEncoder, KeepRule, Layer, Pairing, PoolBlock, PoolLayer,
    RotationAxis,
};
pub use loss::Loss;
#[cfg(feature = "serde")]
pub use model::TrainedModel;
pub use model::{CompiledModel, QuantumModel};
pub use observables::{Observable, Pauli, PauliString};
pub use problem::QmlProblem;
pub use readout::{Decision, Readout};
