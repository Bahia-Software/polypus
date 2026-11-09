//! A bridge to optimizers that live outside this crate and speak ask/tell:
//! the caller's optimizer proposes a batch of candidates ([`AskTell::ask`]),
//! the bridge scores the whole batch with one oracle call, and hands the
//! values back ([`AskTell::tell`]).
//!
//! Optimization libraries such as Optuna, Nevergrad and pycma expose this
//! interface. The Polypus Python bindings implement [`AskTell`] over a Python
//! object, so a library the user installs runs through Polypus's scheduler,
//! each batch spread over the QPUs like any optimizer's. This crate knows
//! nothing of Python: [`AskTell`] is a plain Rust trait, and any Rust optimizer
//! can implement it too.

use crate::error::OptimizerError;
use crate::objective::EvaluationOracle;
use crate::observer::{IterationObserver, IterationReport};
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::util::{check_oracle_len, BestMeasured};
use std::fmt::Write;

/// The error an [`AskTell`] implementation returns when its `ask` or `tell`
/// fails. [`AlgorithmExternal`] stops at the first one and reports it as
/// [`OptimizerError::ExternalOptimizer`], with its message.
pub type AskTellError = Box<dyn std::error::Error + Send + Sync>;

/// An optimizer driven from outside: it proposes batches of candidates and is
/// told their fitness.
///
/// The fitness follows [`EvaluationOracle`]'s convention: **higher is
/// better**. An implementation over a library that minimises negates the
/// values it is told before handing them on.
///
/// [`AlgorithmExternal`] calls `ask`, scores the batch, then calls `tell`
/// with that batch and its values, and repeats; it never calls `ask` twice
/// without a `tell` in between, except that the last `ask` of a run may go
/// untold (its batch would have exceeded the evaluation budget, or the run
/// failed). An implementation that is used for several runs can re-offer
/// such a batch on its next `ask`.
///
/// `Send + Sync`, so that whatever holds one (a training method's
/// configuration) can be shared between threads; `ask` and `tell` take
/// `&mut self`, so an implementation needs no locking of its own.
pub trait AskTell: Send + Sync {
    /// The next batch of candidates to score, or `None` (or an empty batch)
    /// when the optimizer is done.
    ///
    /// Each candidate must have one finite value per dimension;
    /// [`AlgorithmExternal`] checks every row before scoring the batch.
    fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError>;

    /// The fitness of every candidate of the batch [`ask`](AskTell::ask)
    /// returned last: `candidates` is that batch, in order, and `values` holds
    /// one value per candidate, as the oracle returned it (higher is better).
    fn tell(&mut self, candidates: &[Vec<f64>], values: &[f64]) -> Result<(), AskTellError>;
}

/// Runs an [`AskTell`] optimizer against an [`EvaluationOracle`]: every
/// `ask` becomes **one** `evaluate_batch` call, whose values go back through
/// `tell`, until the optimizer is done or the evaluation budget would be
/// exceeded.
///
/// # The budget
///
/// [`max_evaluations`](AlgorithmExternalArgs::max_evaluations) counts the
/// candidates scored, and is never exceeded. The run stops when it is spent,
/// without asking again, or as soon as a batch would exceed it: that batch is
/// neither scored nor told, and the optimizer keeps it as asked. A first batch
/// larger than the whole budget is an error, since nothing could be scored.
///
/// # What is measured (contract C-5)
///
/// `best_params` is the best candidate the oracle scored, offered in batch
/// order (the first one, then only a strictly better one, so a `NaN` never
/// wins over a number), and `best_fitness` the value the oracle returned for
/// it, whatever the external optimizer would report as its own best.
/// `fitness_history` has one entry per batch, the best measured up to it, so
/// the last is `best_fitness`; `iterations_run` counts the batches scored and
/// told. `converged` is `true` when the optimizer ended the run with an empty
/// `ask`, and `false` when the budget did, or an
/// [`IterationObserver`], which sees each batch once
/// it is told and stops the run without another `ask`.
///
/// # Errors
///
/// - Before any `ask`: `max_evaluations` and `dimensions` must be at least 1
///   ([`OptimizerError::InvalidConfig`]).
/// - A row that is not `dimensions` finite values:
///   [`OptimizerError::InvalidCandidate`], naming the row and the batch;
///   nothing of that batch is scored or told.
/// - An error from `ask` or `tell`, or a first `ask` that proposes nothing,
///   so that there is no result: [`OptimizerError::ExternalOptimizer`].
/// - An oracle that returns the wrong number of values:
///   [`OptimizerError::OracleLengthMismatch`], and that batch is not told.
///
/// # Reproducibility
///
/// The external optimizer owns its randomness: there is no seed here. With a
/// deterministic optimizer and oracle, a run is deterministic.
///
/// # Example
///
/// ```
/// use polypus_optimizers::{
///     AlgorithmExternal, AlgorithmExternalArgs, AskTell, AskTellError, EvaluationOracle,
///     Optimizer,
/// };
///
/// // −(x − 1)², maximised at x = 1.
/// struct Parabola;
/// impl EvaluationOracle for Parabola {
///     fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
///         candidates.iter().map(|c| -(c[0] - 1.0).powi(2)).collect()
///     }
/// }
///
/// // Halves a bracket around the best point, three candidates at a time.
/// struct Bracket {
///     centre: f64,
///     width: f64,
/// }
/// impl AskTell for Bracket {
///     fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
///         let (c, w) = (self.centre, self.width);
///         Ok((w > 1e-3).then(|| vec![vec![c - w], vec![c], vec![c + w]]))
///     }
///     fn tell(&mut self, candidates: &[Vec<f64>], values: &[f64]) -> Result<(), AskTellError> {
///         let best = (0..values.len())
///             .max_by(|&a, &b| values[a].total_cmp(&values[b]))
///             .ok_or("an empty batch was told")?;
///         self.centre = candidates[best][0];
///         self.width /= 2.0;
///         Ok(())
///     }
/// }
///
/// let outcome = AlgorithmExternal
///     .optimize(AlgorithmExternalArgs {
///         oracle: Box::new(Parabola),
///         optimizer: Box::new(Bracket { centre: 0.0, width: 2.0 }),
///         max_evaluations: 300,
///         dimensions: 1,
///     })
///     .expect("valid arguments optimize successfully");
///
/// // The bracket shrank below 1e-3, so the optimizer ended the run itself.
/// assert!(outcome.converged);
/// assert_eq!(outcome.iterations_run, 11);
/// assert!((outcome.best_params[0] - 1.0).abs() < 1e-3);
/// ```
pub struct AlgorithmExternal;

/// Arguments for [`AlgorithmExternal`].
///
/// # Preconditions
///
/// Checked before the first `ask`, each violation an
/// [`OptimizerError::InvalidConfig`] naming the parameter, in this order:
/// `max_evaluations >= 1`, `dimensions >= 1`.
pub struct AlgorithmExternalArgs {
    /// Maps parameter vectors to fitness values (higher is better).
    pub oracle: Box<dyn EvaluationOracle>,
    /// The optimizer that proposes the batches and is told their values. It
    /// chooses its own starting point and owns its randomness.
    pub optimizer: Box<dyn AskTell>,
    /// Candidates scored at most, over the whole run. A batch that would
    /// exceed it is neither scored nor told.
    pub max_evaluations: u64,
    /// Number of parameters: every candidate must have exactly this many
    /// values.
    pub dimensions: u32,
}

impl AlgorithmExternal {
    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("External")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Runs an optimizer that speaks ask/tell (an optimization library, or one of your \
             own): each batch it asks for is scored in one oracle call and its values are told \
             back, until it stops asking or the evaluation budget would be exceeded. The best \
             candidate is the best one scored.",
        )
    }

    /// Check the preconditions of [`AlgorithmExternalArgs`] that
    /// [`optimize`](Optimizer::optimize) checks before the first `ask`, in the
    /// same order, without running anything: the configuration listed on
    /// [`AlgorithmExternalArgs`]. There is no point to check: the optimizer
    /// chooses its own start. `train` and `qml.train` call it before any
    /// backend exists.
    pub fn check_preconditions(
        max_evaluations: u64,
        dimensions: u32,
    ) -> Result<(), OptimizerError> {
        validate(max_evaluations, dimensions)
    }

    fn run(
        args: AlgorithmExternalArgs,
        observer: &mut dyn IterationObserver,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmExternalArgs {
            oracle,
            mut optimizer,
            max_evaluations,
            dimensions,
        } = args;
        Self::check_preconditions(max_evaluations, dimensions)?;
        let dims = dimensions as usize;

        let mut best = BestMeasured::new();
        // Best measured fitness after each batch (see
        // `OptimizationOutcome::fitness_history`).
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut spent: u64 = 0;
        let mut converged = false;

        while spent < max_evaluations {
            let batch_index = fitness_history.len();
            // ── 1. Ask ───────────────────────────────────────────────────────
            let asked = optimizer.ask().map_err(|err| failed("ask", err.as_ref()))?;
            let batch = match asked {
                Some(batch) if !batch.is_empty() => batch,
                _ if batch_index == 0 => {
                    return Err(OptimizerError::ExternalOptimizer {
                        reason: "ask() proposed no candidates on its first call, so nothing \
                                 was evaluated; an optimizer passed to an earlier run keeps \
                                 its state and may have finished there"
                            .to_string(),
                    });
                }
                _ => {
                    log::debug!(
                        "Batch {batch_index}: the external optimizer proposed no candidates; \
                         stopping after {spent} evaluations"
                    );
                    converged = true;
                    break;
                }
            };
            // ── 2. Never exceed the budget ──────────────────────────────────
            let size = u64::try_from(batch.len()).unwrap_or(u64::MAX);
            if size > max_evaluations - spent {
                // An implementation may stop reading a batch once it is too
                // large (the Python one does), so the message gives the bound,
                // not the size.
                if batch_index == 0 {
                    return Err(OptimizerError::InvalidConfig {
                        parameter: "max_evaluations",
                        reason: format!(
                            "the external optimizer's first batch has more than \
                             max_evaluations ({max_evaluations}) candidates; nothing was \
                             evaluated"
                        ),
                    });
                }
                log::debug!(
                    "Batch {batch_index}: more candidates than the {} left of max_evaluations \
                     ({max_evaluations}); stopping without evaluating them",
                    max_evaluations - spent
                );
                break;
            }
            check_candidates(&batch, dims, batch_index)?;
            // ── 3. One oracle call for the whole batch ──────────────────────
            let values = oracle.evaluate_batch(&batch);
            check_oracle_len(batch.len(), values.len())?;
            for (candidate, &value) in batch.iter().zip(&values) {
                best.offer(candidate, value);
            }
            fitness_history.push(best.value());
            spent += size;
            log::debug!(
                "Batch {batch_index}: {size} candidates, {spent} of {max_evaluations} \
                 evaluations, best measured {:.6}",
                best.value()
            );
            // ── 4. Tell ──────────────────────────────────────────────────────
            optimizer
                .tell(&batch, &values)
                .map_err(|err| failed("tell", err.as_ref()))?;

            // ── 5. Report, once the optimizer knows the whole batch ─────────
            // A stop ends the run here, without another `ask`.
            let report = IterationReport {
                iteration: fitness_history.len(),
                best_fitness: best.value(),
                best_params: best.params(),
            };
            if observer.observe(&report).is_break() {
                break;
            }
        }

        // `max_evaluations >= 1`, and a run that scored nothing returned an
        // error above, so `best` holds a measured candidate.
        let (best_params, best_fitness) = best.into_parts();
        Ok(OptimizationOutcome {
            best_params,
            best_fitness,
            iterations_run: fitness_history.len(),
            converged,
            fitness_history,
        })
    }
}

impl Optimizer for AlgorithmExternal {
    type Args = AlgorithmExternalArgs;

    fn optimize_observed(
        &self,
        args: Self::Args,
        observer: &mut dyn IterationObserver,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        Self::run(args, observer)
    }
}

/// Reject a configuration the bridge cannot run, before the first `ask`.
fn validate(max_evaluations: u64, dimensions: u32) -> Result<(), OptimizerError> {
    if max_evaluations == 0 {
        return Err(OptimizerError::InvalidConfig {
            parameter: "max_evaluations",
            reason: "must be at least 1 (candidates evaluated at most)".to_string(),
        });
    }
    if dimensions == 0 {
        return Err(OptimizerError::InvalidConfig {
            parameter: "dimensions",
            reason: "must be at least 1 (there is nothing to optimise)".to_string(),
        });
    }
    Ok(())
}

/// Check that every row of batch `batch_index` is a candidate: `dims` values,
/// every one finite. The first row that is not is the error.
fn check_candidates(
    batch: &[Vec<f64>],
    dims: usize,
    batch_index: usize,
) -> Result<(), OptimizerError> {
    for (row, candidate) in batch.iter().enumerate() {
        let invalid = |reason: String| {
            Err(OptimizerError::InvalidCandidate {
                batch: batch_index,
                row,
                reason,
            })
        };
        if candidate.len() != dims {
            return invalid(format!(
                "it has {} values, expected one per dimension ({dims})",
                candidate.len()
            ));
        }
        if let Some((index, value)) = candidate.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return invalid(format!(
                "its value at index {index} is {value}; every value must be finite"
            ));
        }
    }
    Ok(())
}

/// The error for a failed `ask` or `tell`, with the optimizer's message.
/// Formatting never panics: a `Display` that fails gives a fixed text.
fn failed(call: &str, err: &(dyn std::error::Error + Send + Sync)) -> OptimizerError {
    let mut message = String::new();
    if write!(message, "{err}").is_err() {
        message = "<unformattable error>".to_string();
    }
    OptimizerError::ExternalOptimizer {
        reason: format!("{call}() failed: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    /// An error whose `Display` fails.
    #[derive(Debug)]
    struct Unprintable;
    impl fmt::Display for Unprintable {
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            Err(fmt::Error)
        }
    }
    impl std::error::Error for Unprintable {}

    #[test]
    fn a_failure_keeps_the_message_and_names_the_call() {
        let err: AskTellError = "the study is closed".into();
        assert_eq!(
            failed("tell", err.as_ref()),
            OptimizerError::ExternalOptimizer {
                reason: "tell() failed: the study is closed".to_string()
            }
        );
    }

    #[test]
    fn an_unprintable_failure_does_not_panic() {
        assert_eq!(
            failed("ask", &Unprintable),
            OptimizerError::ExternalOptimizer {
                reason: "ask() failed: <unformattable error>".to_string()
            }
        );
    }

    #[test]
    fn the_first_bad_row_is_named() {
        let batch = vec![vec![0.0, 1.0], vec![0.5], vec![f64::NAN, 0.0]];
        assert_eq!(
            check_candidates(&batch, 2, 4),
            Err(OptimizerError::InvalidCandidate {
                batch: 4,
                row: 1,
                reason: "it has 1 values, expected one per dimension (2)".to_string()
            })
        );
        assert_eq!(
            check_candidates(&batch[2..], 2, 0),
            Err(OptimizerError::InvalidCandidate {
                batch: 0,
                row: 0,
                reason: "its value at index 0 is NaN; every value must be finite".to_string()
            })
        );
        assert_eq!(check_candidates(&batch[..1], 2, 0), Ok(()));
    }
}
