//! Watching a run iteration by iteration, and stopping it.
//!
//! [`Optimizer::optimize_observed`](crate::Optimizer::optimize_observed) hands
//! an [`IterationObserver`] a read-only [`IterationReport`] at the end of every
//! iteration. The observer can only look and answer: continue, or stop the run
//! there. A stopped run returns the outcome it has reached, which meets every
//! postcondition of contract C-5, and the optimizer calls no oracle after it.

use std::ops::ControlFlow;

/// The state of a run at the end of one iteration: a generation of DE, PSO or
/// CMA-ES, an iteration of QNG, Adam or SPSA, a step of Rotosolve, an
/// ask/tell batch of [`AlgorithmExternal`](crate::AlgorithmExternal).
///
/// `best_fitness` and `best_params` are what the outcome would report if the
/// run stopped here: `best_fitness` is the entry this iteration leaves in
/// [`fitness_history`](crate::OptimizationOutcome::fitness_history), so it
/// never falls from one iteration to the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationReport<'a> {
    /// Iterations completed, this one included: 1 for the first.
    pub iteration: usize,
    /// The best fitness so far (higher is better).
    pub best_fitness: f64,
    /// The parameters [`best_fitness`](Self::best_fitness) belongs to.
    pub best_params: &'a [f64],
}

/// Watches a run: called once at the end of every iteration, after everything
/// the iteration measures. A stop it asks for wins over the optimizer's own
/// stopping test (and a CMA-ES restart) on that iteration.
///
/// [`ControlFlow::Continue`] lets the run go on; [`ControlFlow::Break`] stops
/// it after this iteration. A stopped run reports `iterations_run` equal to
/// [`iteration`](IterationReport::iteration), `converged = false` even when
/// the optimizer's own test fired on the same iteration, and the
/// [`best_fitness`](IterationReport::best_fitness) and
/// [`best_params`](IterationReport::best_params) it was shown. The optimizer
/// calls no oracle and no [`AskTell`](crate::AskTell) method after a stop:
/// Adam, SPSA and Rotosolve, which measure the point their last step reached
/// once the run ends, measure it before the last iteration's report, and a
/// stop on an earlier iteration leaves that step unmeasured.
///
/// The observer is not called after the optimizer returns, nor when the
/// iteration fails: an error returns before the report.
pub trait IterationObserver {
    /// Look at the end of an iteration, and say whether the run goes on.
    fn observe(&mut self, report: &IterationReport<'_>) -> ControlFlow<()>;
}

/// The observer [`Optimizer::optimize`](crate::Optimizer::optimize) runs
/// with: it watches nothing and never stops a run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unobserved;

impl IterationObserver for Unobserved {
    fn observe(&mut self, _report: &IterationReport<'_>) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A closure is an observer, so a caller can watch a run without a type of
/// its own.
impl<F: FnMut(&IterationReport<'_>) -> ControlFlow<()>> IterationObserver for F {
    fn observe(&mut self, report: &IterationReport<'_>) -> ControlFlow<()> {
        self(report)
    }
}
