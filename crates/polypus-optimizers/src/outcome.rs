//! Optimizer result type and the shared optimizer trait.

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
pub trait Optimizer {
    /// Bundle of oracle(s) and hyper-parameters for this optimizer.
    type Args;

    /// Run the optimization loop and return the best solution found.
    fn optimize(&self, args: Self::Args) -> OptimizationOutcome;
}
