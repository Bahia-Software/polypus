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
//! ## Phase 1 scope
//!
//! This is the skeleton phase: `Dataset` (validated construction,
//! deterministic splits, feature scaling), [`ValidationError`], and the
//! internal RNG. Models, layers, readout, losses and the training problem
//! arrive in later phases.

mod error;
mod rng;

pub use error::ValidationError;
