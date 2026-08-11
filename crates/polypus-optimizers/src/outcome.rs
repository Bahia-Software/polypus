//! Optimizer result type and the shared optimizer trait.

use crate::error::OptimizerError;

/// Outcome of an optimization run.
///
/// The optimizers return this native struct instead of a Python object; the
/// conversion to whatever a caller needs (e.g. a Python list of the best
/// parameters) is the caller's responsibility. Exposing fitness, the
/// per-iteration fitness history, iteration count, and the convergence flag
/// keeps the surface forward-compatible: new callers can read them without
/// changing any optimizer signature.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationOutcome {
    /// Best parameter vector found (the quantity every current caller uses).
    pub best_params: Vec<f64>,
    /// Fitness of [`OptimizationOutcome::best_params`] (higher is better).
    pub best_fitness: f64,
    /// Best fitness found *so far*, one entry per generation/iteration actually
    /// executed — so `fitness_history.len() == iterations_run`, on the
    /// early-stopping paths included.
    ///
    /// **Monotonically non-decreasing** for every optimizer in this crate: each
    /// entry is the same incumbent-best value the algorithm already tracks
    /// internally (DE's/PSO's elitist champion, QNG's/Adam's `best_energy`), not
    /// the fitness of that iteration's current candidate — which gradient ascent
    /// over a rough landscape lets oscillate freely. The monotonicity is
    /// structural, a property of taking the running maximum, so it holds
    /// whatever the oracle returns (a shot estimate, a minibatch estimate); what
    /// varies with the oracle is how noisy the *meaning* of each entry is, never
    /// the shape of the sequence.
    ///
    /// The last entry is therefore always
    /// [`best_fitness`](OptimizationOutcome::best_fitness) — both are read from
    /// the same incumbent — and the vector is empty exactly when
    /// `iterations_run == 0`.
    pub fitness_history: Vec<f64>,
    /// Number of generations/iterations actually executed.
    ///
    /// Lower than the configured budget when an early-stopping criterion fired.
    pub iterations_run: usize,
    /// Whether the algorithm's convergence criterion was satisfied.
    ///
    /// Optimizers without an early-stopping test (e.g. QNG) always report
    /// `false` — they simply exhaust their iteration budget.
    pub converged: bool,
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
    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError>;
}
