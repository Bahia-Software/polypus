//! Native, GIL-free cost observables for variational optimization.
//!
//! A [`CostObservable`] maps a batch of measurement counts — one
//! `HashMap<bitstring, count>` per candidate — to one expectation value per
//! candidate, entirely in Rust and parallelized over candidates with rayon.
//! This replaces the previous per-bitstring Python callback that ran serially
//! under the GIL and was the dominant cost of every optimizer generation.
//!
//! Two flavours exist:
//!
//! - **Declarative, native** ([`QuboObservable`], [`IsingObservable`]): the user
//!   describes the cost as *data* (a QUBO / Ising form), and evaluation is pure
//!   Rust. This is a general primitive — MaxCut, TSP, portfolio, Hamming weight,
//!   and any diagonal (computational-basis) cost reduce to it — without
//!   hardcoding a specific problem.
//! - **Arbitrary callback** (implemented in the `polypus` PyO3 crate, not here):
//!   wraps a user Python function but calls it once per *unique* bitstring in a
//!   single GIL section, then aggregates in Rust.
//!
//! ## Relationship to `polypus-physics::PauliSum`
//!
//! This is the *diagonal, counts-based* cost analogue of the physics crate's
//! `PauliSum`. They are deliberately separate: `PauliSum`'s terms are
//! single-qubit Paulis (no `Z_i·Z_j` coupling) and are Trotter-oriented, so it
//! cannot express a QUBO/Ising cost. A future unification could offer an
//! `IsingObservable::from_pauli_sum` for the diagonal (Z-only) subset, but the
//! two are not coupled today.
//!
//! ## Bit-ordering convention
//!
//! Counts keys follow the simulator/Qiskit convention: a key of width `w` has
//! the most-significant bit on the **left**, so variable `i` is the character at
//! position `w - 1 - i` (equivalently `x_i = (state >> i) & 1`). Every evaluator
//! here honours that convention; see [`QuboObservable`] for details.

use std::collections::HashMap;

mod error;
mod ising;
mod qubo;

pub use error::ObservableError;
pub use ising::IsingObservable;
pub use qubo::QuboObservable;

/// Maps a batch of measurement counts (one map per candidate) to one expectation
/// value per candidate, in the same order.
///
/// Implementations are `Send + Sync` so a single `Arc<dyn CostObservable>` can be
/// shared across an oracle's worker threads (e.g. the QML oracle's
/// `spawn_blocking` tasks) and evaluated concurrently.
pub trait CostObservable: Send + Sync {
    /// Evaluate every candidate's counts. Returns one value per candidate, or the
    /// first error encountered.
    ///
    /// Contract: an empty per-candidate map (or one whose counts sum to zero)
    /// yields `0.0`, matching the historical Python aggregation.
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError>;
}
