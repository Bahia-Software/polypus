//! Optimizer result type and the shared optimizer trait.

use crate::error::OptimizerError;
use crate::observer::{IterationObserver, Unobserved};

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
