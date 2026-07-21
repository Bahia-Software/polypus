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
//! Data and the model core are in place: [`Dataset`] (validated construction,
//! deterministic splits, feature scaling), the layered model builder
//! ([`QuantumModel`] → [`CompiledModel`]) with its [`Layer`] catalogue
//! ([`AngleEncoder`], [`HardwareEfficientAnsatz`]), and the two error enums
//! ([`ValidationError`], [`QmlError`]). A compiled model turns a sample into a
//! [`polypus_circuit`] template with the data features fixed and the trainable
//! parameters left free.
//!
//! Readout (Pauli observables + decision), losses and the training problem —
//! and the convolution/pooling and amplitude-encoding layers — arrive in later
//! phases.

mod dataset;
mod error;
mod layers;
mod model;
mod rng;

pub use dataset::Dataset;
pub use error::{QmlError, ValidationError};
pub use layers::{
    AngleEncoder, Entanglement, Entangler, HardwareEfficientAnsatz, Layer, RotationAxis,
};
pub use model::{CompiledModel, QuantumModel};
