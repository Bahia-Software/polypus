//! Error type for the optimizers.

use std::fmt;

/// Errors returned by the optimizers in this crate.
///
/// Every variant is a *precondition* or *contract* violation surfaced by
/// [`Optimizer::optimize`](crate::Optimizer::optimize): an invalid
/// configuration (population too small, empty bounds) detected before any RNG
/// draw or oracle call, an [`EvaluationOracle`](crate::EvaluationOracle)
/// that breaks the "one fitness value per candidate" length contract mid-run,
/// or an [`AskTell`](crate::AskTell) optimizer that fails or proposes a row
/// that is not a candidate.
/// Returning these instead of panicking is what lets the Python seam map a bad
/// input to a `PyValueError` instead of unwinding across the FFI boundary.
///
/// A single shared enum (rather than one type per algorithm) matches the
/// "one seam, one error type" shape of this crate: the optimizers report the
/// same few failure kinds, so per-algorithm types would only duplicate the
/// same variants without tightening anything the seam cares about.
///
/// Mirrors the hand-written style of `CircuitError`/`SimError` (no `thiserror`).
/// `Eq` is intentionally omitted — [`OptimizerError::InvalidBounds`] carries
/// `f64` bounds, which are not `Eq`, exactly as this crate's
/// [`OptimizationOutcome`](crate::OptimizationOutcome) omits `Eq` for the same
/// reason.
///
/// The enum is `#[non_exhaustive]`: new optimizers add variants without a
/// breaking release, so a `match` outside this crate needs a wildcard arm.
///
/// ```compile_fail,E0004
/// use polypus_optimizers::OptimizerError;
///
/// fn kind(err: &OptimizerError) -> &'static str {
///     match err {
///         OptimizerError::PopulationTooSmall { .. } => "population",
///         OptimizerError::InvalidBounds { .. } => "bounds",
///         OptimizerError::OracleLengthMismatch { .. } => "length",
///         OptimizerError::NonFiniteGradient { .. } => "gradient",
///         OptimizerError::InvalidConfig { .. } => "config",
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// index deep inside the optimization loop. Also raised when a
    /// [`VarianceOracle::variance_diagonal`](crate::VarianceOracle::variance_diagonal)
    /// returns a diagonal whose length is not the number of dimensions (QNG), or
    /// a [`GradientOracle::gradient_batch`](crate::GradientOracle::gradient_batch)
    /// a gradient of the wrong length (Adam).
    OracleLengthMismatch {
        /// Number of candidates submitted (the required length).
        expected: usize,
        /// Number of fitness values the oracle actually returned.
        got: usize,
    },
    /// A [`GradientOracle::gradient_batch`](crate::GradientOracle::gradient_batch)
    /// returned a component that is not finite (Adam). The update adds the
    /// gradient to the parameters, so a `NaN` or an infinity would leave them
    /// non-finite for the rest of the run.
    NonFiniteGradient {
        /// The first non-finite component's parameter index.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// A configuration value is outside its valid domain: QNG's
    /// `max_iters` (must be ≥ 1; `0` never evaluates the objective and would
    /// report `-inf`), `finite_difference_step` (must be finite and `> 0`),
    /// `tikhonov_reg` (must be finite and `>= 0`) or `learning_rate` (must be
    /// finite); or CMA-ES's `dimensions` (≥ 1), `generations` (≥ 1),
    /// `sigma0` (finite, in `(0, 2π]`), `population_size` (≥ 2 when given) or
    /// `restarts` (the largest population must fit in a `u32`); or Adam's
    /// `max_iters` (≥ 1), `dimensions` (≥ 1), `learning_rate` (finite, `> 0`),
    /// `beta1`/`beta2` (in `[0, 1)`) or `epsilon` (finite, `> 0`); or SPSA's
    /// `max_iters`, `dimensions`, `resamplings` and `eval_interval` (each
    /// ≥ 1), `perturbation` and `target_step` (finite, `> 0`), `alpha` (in
    /// `(0.5, 1]`) or `gamma` (`> 0`, with `alpha − gamma > 0.5`); or
    /// Rotosolve's `sweeps`, `dimensions`, `reevaluate_every` and `patience`
    /// (each ≥ 1) or `tolerance` (finite, `>= 0`); or a
    /// [`FiniteDifferenceGradient`](crate::FiniteDifferenceGradient)'s
    /// `finite_difference_step` (finite, `> 0`); or any optimizer's
    /// `initial_point` (`dimensions` finite values, within `bounds` for PSO);
    /// or [`AlgorithmExternal`](crate::AlgorithmExternal)'s `max_evaluations`
    /// and `dimensions` (each ≥ 1), or a `max_evaluations` smaller than the
    /// external optimizer's first batch, found after its first `ask` but
    /// before any oracle call.
    /// Detected before any RNG draw or oracle call, so a bad config maps to a
    /// `PyValueError` at the seam rather than producing a `NaN`/`inf` result.
    /// (Empty or non-finite `bounds` keep their own
    /// [`InvalidBounds`](OptimizerError::InvalidBounds) variant.)
    InvalidConfig {
        /// The offending parameter's name.
        parameter: &'static str,
        /// Why it was rejected (valid domain + the offending value).
        reason: String,
    },
    /// An [`AskTell`](crate::AskTell) optimizer asked for a row that is not a
    /// candidate: it does not have one value per dimension, or one of its
    /// values is not finite. [`AlgorithmExternal`](crate::AlgorithmExternal)
    /// checks every row of a batch before scoring it, so nothing of that batch
    /// was scored or told, and the run stops.
    InvalidCandidate {
        /// The batch, counted from 0 over the run's `ask` calls.
        batch: usize,
        /// The row within that batch, counted from 0.
        row: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// An [`AskTell`](crate::AskTell) optimizer failed: its `ask` or `tell`
    /// returned an error, or its first `ask` proposed nothing, so that no
    /// candidate was scored and there is no result to report.
    ExternalOptimizer {
        /// What happened, with the optimizer's own message.
        reason: String,
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
            OptimizerError::NonFiniteGradient { index, value } => write!(
                f,
                "gradient oracle returned {value} for parameter {index}; every component must be finite"
            ),
            OptimizerError::InvalidConfig { parameter, reason } => {
                write!(f, "invalid {parameter}: {reason}")
            }
            OptimizerError::InvalidCandidate { batch, row, reason } => write!(
                f,
                "invalid candidate in row {row} of batch {batch} from the external optimizer's ask() (both 0-based): {reason}"
            ),
            OptimizerError::ExternalOptimizer { reason } => {
                write!(f, "external optimizer: {reason}")
            }
        }
    }
}

impl std::error::Error for OptimizerError {}
