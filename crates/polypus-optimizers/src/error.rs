//! Error type for the optimizers.

use std::fmt;

/// Errors returned by the optimizers in this crate.
///
/// Every variant is a *precondition* or *contract* violation surfaced by
/// [`Optimizer::optimize`](crate::Optimizer::optimize): an invalid
/// configuration (population too small, empty bounds) detected before any RNG
/// draw or oracle call, or an [`EvaluationOracle`](crate::EvaluationOracle)
/// that breaks the "one fitness value per candidate" length contract mid-run.
/// Returning these instead of panicking is what lets the Python seam map a bad
/// input to a `PyValueError` instead of unwinding across the FFI boundary.
///
/// A single shared enum (rather than one type per algorithm) matches the
/// "one seam, one error type" shape of this crate: all three optimizers report
/// the same three failure kinds, so per-algorithm types would only duplicate
/// the same variants without tightening anything the seam cares about.
///
/// Mirrors the hand-written style of `CircuitError`/`SimError` (no `thiserror`).
/// `Eq` is intentionally omitted — [`OptimizerError::InvalidBounds`] carries
/// `f64` bounds, which are not `Eq`, exactly as this crate's
/// [`OptimizationOutcome`](crate::OptimizationOutcome) omits `Eq` for the same
/// reason.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    /// Differential Evolution needs at least four population members: each
    /// trial mutation samples three *distinct* other members of the
    /// population, which is impossible with fewer than four.
    PopulationTooSmall {
        /// The rejected `population_size`.
        got: usize,
        /// The smallest accepted `population_size` (4).
        min: usize,
    },
    /// PSO/QNG bounds `(lb, ub)` must be a non-empty interval (`lb < ub`);
    /// positions are drawn uniformly from `[lb, ub)`, which is empty (and
    /// panics inside the sampler) when `lb >= ub`. A non-finite bound (`NaN`)
    /// is rejected here too, since it can never satisfy `lb < ub`.
    InvalidBounds {
        /// The provided lower bound.
        lb: f64,
        /// The provided upper bound.
        ub: f64,
    },
    /// An [`EvaluationOracle::evaluate_batch`](crate::EvaluationOracle::evaluate_batch)
    /// returned a number of fitness values different from the number of
    /// candidates it was given. The optimizers index the result positionally,
    /// so a short (or long) return would otherwise panic with an out-of-bounds
    /// index deep inside the optimization loop.
    OracleLengthMismatch {
        /// Number of candidates submitted (the required length).
        expected: usize,
        /// Number of fitness values the oracle actually returned.
        got: usize,
    },
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::PopulationTooSmall { got, min } => write!(
                f,
                "population_size must be at least {min} (each trial mutation samples 3 distinct other members), got {got}"
            ),
            OptimizerError::InvalidBounds { lb, ub } => write!(
                f,
                "bounds must be a non-empty interval with lb < ub, got ({lb}, {ub})"
            ),
            OptimizerError::OracleLengthMismatch { expected, got } => write!(
                f,
                "evaluation oracle returned {got} fitness value(s) for {expected} candidate(s); it must return exactly one value per candidate, in order"
            ),
        }
    }
}

impl std::error::Error for OptimizerError {}
