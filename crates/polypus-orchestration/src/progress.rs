//! Watching a training run iteration by iteration: the [`TrainObserver`] a
//! [`TrainFlow`](crate::TrainFlow) carries, shown a [`TrainProgress`] at the end
//! of every iteration of its optimizer, and the evaluation count it reports.
//!
//! The optimizer's own [`IterationObserver`] sees the iteration, the best
//! fitness and the best parameters; this layer adds the evaluations spent,
//! counted by a decorator of the run's evaluation oracles, and stops the run
//! once its [`OracleErrorSlot`] holds a failure (the observer is not called
//! then). The `polypus` edge implements [`TrainObserver`] over the user's
//! Python callback, so this crate stays `pyo3`-free.

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polypus_optimizers::{EvaluationOracle, GradientOracle, IterationObserver, IterationReport};

use crate::dispatch::OracleErrorSlot;

/// A training run at the end of one iteration of its optimizer, as a
/// [`TrainObserver`] sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainProgress<'a> {
    /// Iterations completed, this one included: 1 for the first (see
    /// [`IterationReport::iteration`]).
    pub iteration: usize,
    /// The best fitness so far, the entry this iteration leaves in the
    /// outcome's `fitness_history`. Under a minibatch, a minibatch estimate,
    /// except for Adam, which tracks its best on every sample (contract C-5).
    pub best_fitness: f64,
    /// The parameters [`best_fitness`](Self::best_fitness) belongs to.
    pub best_params: &'a [f64],
    /// Candidates the optimizer has had scored as a fitness so far: every row
    /// of every `evaluate_batch` call to its evaluation oracles, on the
    /// minibatch and, for a minibatched Adam, on every sample, plus one for
    /// each `θ` an exact gradient scores in its own batch
    /// ([`GradientOracle::value_and_gradient`]). The gradient's other circuits
    /// (a parameter shift's shifted ones) and QNG's variance calls are not
    /// counted, nor is the final batch of `final_shots`.
    pub evaluations: u64,
}

/// Watches a training run, once at the end of every iteration of its
/// optimizer (see [`IterationObserver`]): [`ControlFlow::Break`] stops the run
/// there, and it returns what it has reached; a flow with `final_shots` still
/// sends its final batch.
///
/// To fail the run instead, an implementation records its error in the run's
/// [`OracleErrorSlot`] and answers [`ControlFlow::Break`]: the recorded error
/// is then the run's, and no final batch is sent. Once the slot holds a
/// failure, whoever recorded it, the run stops without calling the observer
/// again.
///
/// `Send`, so that the [`TrainFlow`](crate::TrainFlow) that carries one can be
/// moved to the thread that runs it.
pub trait TrainObserver: Send {
    /// Look at the end of an iteration, and say whether the run goes on.
    fn observe(&mut self, progress: &TrainProgress<'_>) -> ControlFlow<()>;
}

/// A closure is a training observer.
impl<F: FnMut(&TrainProgress<'_>) -> ControlFlow<()> + Send> TrainObserver for F {
    fn observe(&mut self, progress: &TrainProgress<'_>) -> ControlFlow<()> {
        self(progress)
    }
}

/// Counts the candidates handed to `evaluate_batch`, every row of every call,
/// into a counter it shares with a [`Watch`].
pub(crate) struct Counted {
    inner: Box<dyn EvaluationOracle>,
    evaluations: Arc<AtomicU64>,
}

impl Counted {
    pub(crate) fn new(inner: Box<dyn EvaluationOracle>, evaluations: &Arc<AtomicU64>) -> Self {
        Self {
            inner,
            evaluations: Arc::clone(evaluations),
        }
    }
}

impl EvaluationOracle for Counted {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let rows = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
        self.evaluations.fetch_add(rows, Ordering::Relaxed);
        self.inner.evaluate_batch(candidates)
    }

    /// Forwarded and not counted: no optimizer sends a batch with other shots.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.inner.evaluate_batch_with_shots(candidates, shots)
    }
}

/// Counts the `θ` an exact gradient scores in its own batch: one for every
/// [`value_and_gradient`](GradientOracle::value_and_gradient) that returns a
/// value, into the counter a [`Counted`] oracle shares. Everything else is
/// forwarded unchanged, [`scores_theta`](GradientOracle::scores_theta)
/// included, so the run and its walltime projection are the same.
pub(crate) struct CountedGradient {
    inner: Box<dyn GradientOracle>,
    evaluations: Arc<AtomicU64>,
}

impl CountedGradient {
    pub(crate) fn new(inner: Box<dyn GradientOracle>, evaluations: &Arc<AtomicU64>) -> Self {
        Self {
            inner,
            evaluations: Arc::clone(evaluations),
        }
    }
}

impl GradientOracle for CountedGradient {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.inner.gradient(theta, param_index)
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.inner.gradient_batch(theta, dims)
    }

    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        let measured = self.inner.value_and_gradient(theta, dims);
        if measured.is_some() {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
        }
        measured
    }

    fn scores_theta(&self) -> bool {
        self.inner.scores_theta()
    }
}

/// The optimizer's observer for a watched training run: it stops the run once
/// `errors` holds a failure, and otherwise shows the [`TrainObserver`] the
/// iteration with the evaluations counted so far.
pub(crate) struct Watch<'a> {
    pub(crate) observer: &'a mut dyn TrainObserver,
    pub(crate) evaluations: Arc<AtomicU64>,
    pub(crate) errors: &'a OracleErrorSlot,
}

impl IterationObserver for Watch<'_> {
    fn observe(&mut self, report: &IterationReport<'_>) -> ControlFlow<()> {
        // The run has failed (an oracle, a backend, Ctrl+C between waves, the
        // observer itself): its error is the run's, so it ends here, and the
        // observer is not shown an iteration scored with the oracle's
        // sentinel values.
        if self.errors.failed() {
            return ControlFlow::Break(());
        }
        self.observer.observe(&TrainProgress {
            iteration: report.iteration,
            best_fitness: report.best_fitness,
            best_params: report.best_params,
            evaluations: self.evaluations.load(Ordering::Relaxed),
        })
    }
}
