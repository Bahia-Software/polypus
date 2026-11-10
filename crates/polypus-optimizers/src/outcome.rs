//! Optimizer result type and the shared optimizer trait.

use std::fmt;

use crate::error::OptimizerError;
use crate::observer::{IterationObserver, Unobserved};

/// Why a run stopped: what ended the optimizer's loop.
///
/// [`OptimizationOutcome::converged`] is `true` exactly for the optimizer's
/// own stopping tests ([`FitnessTolerance`](Self::FitnessTolerance),
/// [`ParameterTolerance`](Self::ParameterTolerance),
/// [`Degenerate`](Self::Degenerate) and
/// [`OptimizerFinished`](Self::OptimizerFinished)). The optimizers in this
/// crate never report [`Walltime`](Self::Walltime),
/// [`Interrupted`](Self::Interrupted) or [`Failed`](Self::Failed): those are
/// for the layer that runs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StopReason {
    /// The best fitness improved by less than `tolerance` over the last
    /// `patience` iterations (DE, SHADE, L-SHADE, CMA-ES, Adam; Rotosolve over
    /// whole sweeps).
    FitnessTolerance,
    /// The population's spread fell below `tolerance` in every dimension (PSO).
    ParameterTolerance,
    /// The iteration budget (`generations`, `max_iters`, `sweeps`) ran out.
    IterationBudget,
    /// The evaluation budget ran out: L-SHADE's schedule, or `External`'s
    /// `max_evaluations` (spent, or too small for the next batch).
    EvaluationBudget,
    /// An [`IterationObserver`] stopped the run; this wins over any stopping
    /// test that fired on the same iteration.
    Callback,
    /// CMA-ES's search distribution degenerated (non-finite, ill-conditioned,
    /// collapsed or diverging) with no restart left.
    Degenerate,
    /// The external optimizer ended the run itself, with an empty `ask`.
    OptimizerFinished,
    /// The run's walltime ran out.
    Walltime,
    /// The run was interrupted.
    Interrupted,
    /// The run failed.
    Failed,
}

impl StopReason {
    /// The `snake_case` name of the reason, as the Python edge reports it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FitnessTolerance => "fitness_tolerance",
            Self::ParameterTolerance => "parameter_tolerance",
            Self::IterationBudget => "iteration_budget",
            Self::EvaluationBudget => "evaluation_budget",
            Self::Callback => "callback",
            Self::Degenerate => "degenerate",
            Self::OptimizerFinished => "optimizer_finished",
            Self::Walltime => "walltime",
            Self::Interrupted => "interrupted",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of an optimization run.
///
/// The optimizers return this native struct instead of a Python object; the
/// conversion to whatever a caller needs (e.g. a Python list of the best
/// parameters) is the caller's responsibility. Exposing fitness, iteration
/// count, and the convergence flag keeps the surface forward-compatible: new
/// callers can read them without changing any optimizer signature.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationOutcome {
    /// Best parameter vector found (the quantity every current caller uses).
    pub best_params: Vec<f64>,
    /// Fitness of [`OptimizationOutcome::best_params`] (higher is better).
    pub best_fitness: f64,
    /// Number of generations/iterations actually executed.
    ///
    /// Lower than the configured budget when an early-stopping criterion fired,
    /// or when an [`IterationObserver`] stopped the run.
    pub iterations_run: usize,
    /// Whether the algorithm's convergence criterion was satisfied.
    ///
    /// Optimizers without an early-stopping test (QNG, SPSA) always report
    /// `false` — they simply exhaust their iteration budget. A run an
    /// [`IterationObserver`] stopped reports `false` too, even when the
    /// criterion fired on the iteration it stopped at.
    pub converged: bool,
    /// Best fitness recorded at the end of each generation/iteration, in order.
    ///
    /// One entry per executed generation, so `fitness_history.len() ==
    /// iterations_run` and `fitness_history.last()` (when non-empty) equals
    /// [`best_fitness`](OptimizationOutcome::best_fitness). This exposes the
    /// optimizer's quality trajectory, which is what DE's fitness-stagnation
    /// early stop is computed from (contract C-5). Empty only for a zero-budget
    /// run (`generations`/`max_iters == 0`).
    pub fitness_history: Vec<f64>,
    /// What ended the run (see [`StopReason`]).
    pub stop_reason: StopReason,
}

/// Shared entry point for the variational optimizers in this crate.
///
/// Design note: the whole input (the [`EvaluationOracle`](crate::EvaluationOracle)
/// plus hyper-parameters) is bundled in the associated `Args` struct and taken
/// by value, mirroring the previous `run(args)` shape for minimal churn. QNG
/// additionally carries a [`VarianceOracle`](crate::VarianceOracle) in its
/// `Args`, so a uniform "everything in `Args`" contract is cleaner than
/// splitting the primary oracle out of the configuration.
///
/// The trait deliberately does **not** fix a Python return type: it yields a
/// native [`OptimizationOutcome`], leaving any interpreter conversion to the
/// caller.
///
/// `optimize` returns a [`Result`]: invalid configuration (e.g. a DE
/// population smaller than four, or empty PSO/QNG bounds) and an
/// [`EvaluationOracle`](crate::EvaluationOracle) that breaks its length
/// contract are reported as [`OptimizerError`] rather than by panicking, so a
/// caller across an FFI boundary can turn them into a proper exception.
pub trait Optimizer {
    /// Bundle of oracle(s) and hyper-parameters for this optimizer.
    type Args;

    /// Run the optimization loop and return the best solution found, or an
    /// [`OptimizerError`] if the arguments or the oracle violate a contract.
    ///
    /// The same as [`optimize_observed`](Optimizer::optimize_observed) with an
    /// observer that never stops the run: the same oracle calls, RNG draws and
    /// outcome.
    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        self.optimize_observed(args, &mut Unobserved)
    }

    /// Run the optimization loop as [`optimize`](Optimizer::optimize) does,
    /// showing `observer` the run at the end of every iteration; the run stops
    /// after the iteration on which it answers
    /// [`Break`](std::ops::ControlFlow::Break) (see [`IterationObserver`]).
    ///
    /// The observer only reads, so an observer that never stops the run leaves
    /// the outcome exactly as [`optimize`](Optimizer::optimize) returns it.
    fn optimize_observed(
        &self,
        args: Self::Args,
        observer: &mut dyn IterationObserver,
    ) -> Result<OptimizationOutcome, OptimizerError>;
}
