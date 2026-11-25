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
//! - **Declarative, native** ([`QuboObservable`], [`IsingObservable`], and
//!   [`ParityObservable`] for `Z` strings of any weight): the user
//!   describes the cost as *data* (a QUBO / Ising form, or weighted `Z`
//!   strings), and evaluation is pure
//!   Rust. This is a general primitive — MaxCut, TSP, portfolio, Hamming weight,
//!   and any diagonal (computational-basis) cost reduce to it — without
//!   hardcoding a specific problem.
//! - **Arbitrary callback** (implemented in the `polypus` PyO3 crate, not here):
//!   wraps a user Python function but calls it once per *unique* bitstring in a
//!   single GIL section, then aggregates in Rust.
//!
//! ## Pauli sums (issue #530)
//!
//! A general Hamiltonian, such as `polypus-physics`' `PauliSum`, is a sum of
//! multi-qubit Pauli strings (`Z0 Z1`, `X0 X1 Y2 Y3`), not a diagonal cost.
//! Hardware measures every qubit in the `Z` basis, so a term with `X` or `Y`
//! factors is measured by rotating those qubits first (`H` for `X`; `S†` then
//! `H` for `Y`; see [`BasisRotation`]), after which the term reads as a
//! product of `Z`s on the same qubits. Terms that agree on the Pauli of every
//! qubit they share (*qubit-wise commuting*) share one rotation, hence one
//! circuit: [`group_qubit_wise`] partitions a sum into such groups, and each
//! group's [`ParityObservable`] turns that circuit's counts into the group's
//! contribution. The energy is [`Grouping::constant`] plus the sum of the
//! groups' values. Running one circuit per group is the oracle's job
//! (issue #424).
//!
//! Design decisions, recorded here as issue #530 asks:
//!
//! 1. **Own bitmask type, not `polypus-physics`' `PauliSum`.** A
//!    [`PauliMask`] stores a string as two `u64` masks (`X → (1, 0)`,
//!    `Z → (0, 1)`, `Y → (1, 1)` per qubit), so qubit-wise commutation and
//!    parity are a few bit operations. The price is a limit of
//!    [`MAX_QUBITS`] = 64 qubits. This crate does not depend on
//!    `polypus-physics`; converting a `PauliSum` term by term is the caller's
//!    job (the oracle of #424), and the physics crate is a test-only
//!    dependency here.
//! 2. **Not unified with `polypus-qml`'s Pauli types.** `polypus-qml`'s
//!    `PauliString` names *logical* positions of a model, resolved to physical
//!    qubits when the model is compiled; the types here act on physical
//!    qubits. What the two crates share is the algorithms: `polypus-qml`'s
//!    `DiagonalObservable` is the same parity reduction, and its readout
//!    grouping applies the same greedy first-fit idea to whole observables
//!    (rejecting more than one group for now), so it can call these once its
//!    indices are physical (issue #620).
//! 3. **Identity terms are a constant, not a group.** Their value is the same
//!    in every state, so no shots are spent on them: [`group_qubit_wise`] sums
//!    them into [`Grouping::constant`]. A [`ParityObservable`] still accepts a
//!    mask-0 term (always `+c`), as `polypus-qml`'s `DiagonalObservable` does.
//!
//! Qubit `i` is bit `i` of a mask, and the counts keys follow the convention
//! below.
//!
//! ## Bit-ordering convention
//!
//! Counts keys follow the simulator/Qiskit convention: a key of width `w` has
//! the most-significant bit on the **left**, so variable `i` is the character at
//! position `w - 1 - i` (equivalently `x_i = (state >> i) & 1`). Every evaluator
//! here honours that convention; see [`QuboObservable`] for details.

#![warn(missing_docs)]

use std::collections::HashMap;

mod bound;
mod error;
mod ising;
mod pauli;
mod qubo;

pub use bound::MAX_WEIGHT_BOUND;
pub use error::ObservableError;
pub use ising::IsingObservable;
pub use pauli::{
    group_qubit_wise, BasisRotation, Grouping, MeasurementGroup, ParityObservable, Pauli,
    PauliMask, MAX_QUBITS,
};
pub use qubo::QuboObservable;

/// Where a reducer can run: `Portable` reducers are pure Rust and can run
/// anywhere (including, in a future phase, at the data source before counts are
/// shipped back), while `Local` reducers must run in the caller's process because
/// they hold non-serializable state — e.g. a Python callback that needs the GIL.
///
/// This is a forward-looking hook for reduce-at-source (plan §7.1): a `Planner`
/// may only push reduction to a remote worker when the reducer is `Portable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducerLocality {
    /// Runs anywhere: pure Rust, serializable, no host-process dependency.
    Portable,
    /// Must run in the caller's process (holds the GIL / non-serializable state).
    Local,
}

/// Maps a batch of measurement counts (one map per candidate) to one expectation
/// value per candidate, in the same order.
///
/// Implementations are `Send + Sync` so a single `Arc<dyn CostObservable>` can be
/// held by an oracle that is shared across threads, and evaluated concurrently.
pub trait CostObservable: Send + Sync {
    /// Evaluate every candidate's counts. Returns one value per candidate, or the
    /// first error encountered.
    ///
    /// Contract: an empty per-candidate map (or one whose counts sum to zero)
    /// yields `0.0`, matching the historical Python aggregation.
    ///
    /// The result must not depend on the maps' iteration order, which `HashMap`
    /// randomises per map and per process: floating-point addition is not
    /// associative, so a sum taken in that order can change in the last bit from
    /// one run to the next. The reducers in this crate sum each candidate's terms
    /// in ascending bitstring order.
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError>;

    /// Where this reducer may run. Defaults to [`ReducerLocality::Portable`]
    /// (the native `Qubo`/`Ising`/`Parity` evaluators are pure Rust); the Python-callback
    /// observable overrides it to [`ReducerLocality::Local`].
    fn locality(&self) -> ReducerLocality {
        ReducerLocality::Portable
    }
}
