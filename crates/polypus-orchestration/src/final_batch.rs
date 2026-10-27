//! The final batch of a training run: once the optimizer has stopped, its best
//! candidates are re-evaluated in one more batch with more shots
//! (`final_shots`, contract C-5).
//!
//! Under shot noise an optimizer's `best_fitness` is the best of many noisy
//! estimates, so it overstates the real value of its `best_params` (the
//! winner's curse). A [`TrainFlow`](crate::TrainFlow) with `final_shots` hands
//! the optimizer an [`EliteRecorder`] around its evaluation oracle, which keeps
//! the best distinct candidates the optimizer scored, and after the run sends
//! `best_params` and the best of the others, up to
//! [`Method::final_candidates`], in one batch with `final_shots` shots per
//! circuit. The recorder lives here rather than in the optimizers, so
//! `OptimizationOutcome` does not change and it works for any optimizer.
//!
//! A final value is a fresh estimate, so with `best_params` alone it has no
//! bias. With several candidates the reported one is the highest of a few
//! noisy values, so it is still slightly optimistic, though far less than
//! `best_fitness`, the highest of the whole run.

use std::sync::{Arc, Mutex};

use polypus_optimizers::{EvaluationOracle, OptimizerError};

use crate::dispatch::{Method, OracleError, OracleErrorSlot};
use crate::flow::FinalEvaluation;

/// Candidates the final batch re-evaluates for a population method.
const POPULATION_FINAL_CANDIDATES: usize = 5;

impl Method {
    /// How many candidates the final batch of a run with `final_shots`
    /// re-evaluates, at most: the five best distinct candidates for a
    /// population method (DE with every strategy, PSO and CMA-ES), which scores
    /// many points per generation, and only `best_params` for a local method
    /// (QNG, Adam, SPSA and Rotosolve), whose other measured points are
    /// earlier, worse iterates or their neighbours. `best_params` is always
    /// one of them, and the batch is smaller when the run scored fewer
    /// distinct candidates.
    ///
    /// An external optimizer counts as a population method: the bridge cannot
    /// tell how its library searches, and most of the libraries it serves
    /// (Bayesian optimization, evolution strategies) score many distinct
    /// points per batch.
    pub fn final_candidates(&self) -> usize {
        match self {
            Method::De(_) | Method::Pso(_) | Method::Cmaes(_) | Method::External(..) => {
                POPULATION_FINAL_CANDIDATES
            }
            Method::Qng(..) | Method::Adam(_) | Method::Spsa(_) | Method::Rotosolve(_) => 1,
        }
    }
}

/// An [`EvaluationOracle`] decorator that keeps the best distinct candidates
/// its inner oracle scored, returning every value unchanged.
///
/// - Only a finite value enters: a `NaN` or an infinity never does (contract
///   C-5 asks for finite values).
/// - Candidates are distinct by their exact bits (`f64::to_bits`, so `-0.0`
///   and `0.0` differ). A candidate scored again keeps its best value.
/// - The ranking is by value, highest first; between equal values, the one
///   scored first ranks higher. Candidates are numbered in the order the
///   optimizer sent them (batch by batch, in batch order), and a candidate
///   scored again with a better value takes the number of that evaluation. The
///   optimizer sends its batches in an order its seed fixes, so the ranking is
///   deterministic.
///
/// It keeps at most `capacity` candidates, and they are exactly the
/// `capacity` best: a candidate left out had a best value below the worst one
/// kept, which only rises.
pub(crate) struct EliteRecorder {
    inner: Arc<dyn EvaluationOracle>,
    capacity: usize,
    elite: Mutex<Elite>,
}

#[derive(Default)]
struct Elite {
    /// Candidates offered so far, finite or not: the number of the next one.
    offered: u64,
    /// The best distinct candidates, best first.
    entries: Vec<Entry>,
}

struct Entry {
    params: Vec<f64>,
    value: f64,
    /// The number of the evaluation that gave `value`.
    order: u64,
}

impl Elite {
    fn offer(&mut self, params: &[f64], value: f64, capacity: usize) {
        let order = self.offered;
        self.offered += 1;
        if !value.is_finite() || capacity == 0 {
            return;
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| same_bits(&entry.params, params))
        {
            if value <= entry.value {
                return;
            }
            entry.value = value;
            entry.order = order;
        } else if self.entries.len() < capacity {
            self.entries.push(Entry {
                params: params.to_vec(),
                value,
                order,
            });
        } else {
            match self.entries.last_mut() {
                // A tie keeps the earlier candidate.
                Some(worst) if value > worst.value => {
                    *worst = Entry {
                        params: params.to_vec(),
                        value,
                        order,
                    }
                }
                _ => return,
            }
        }
        // Finite values compare totally; `-0.0` equals `0.0` here, as above.
        self.entries.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.order.cmp(&b.order))
        });
    }
}

/// Whether `a` and `b` are the same candidate, bit for bit.
fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

impl EliteRecorder {
    /// Record the best `capacity` candidates `inner` scores.
    pub(crate) fn new(inner: Arc<dyn EvaluationOracle>, capacity: usize) -> Self {
        Self {
            inner,
            capacity,
            elite: Mutex::new(Elite::default()),
        }
    }

    /// The candidates kept so far, best first.
    pub(crate) fn ranked(&self) -> Vec<Vec<f64>> {
        self.elite
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .iter()
            .map(|entry| entry.params.clone())
            .collect()
    }

    /// The final batch: `best_params` first, then the kept candidates best
    /// first, without repeating a candidate, up to `count` of them.
    pub(crate) fn final_candidates(&self, best_params: &[f64], count: usize) -> Vec<Vec<f64>> {
        let mut candidates = vec![best_params.to_vec()];
        for params in self.ranked() {
            if candidates.len() >= count {
                break;
            }
            if !candidates.iter().any(|kept| same_bits(kept, &params)) {
                candidates.push(params);
            }
        }
        candidates
    }
}

impl EvaluationOracle for EliteRecorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let values = self.inner.evaluate_batch(candidates);
        let mut elite = self.elite.lock().unwrap_or_else(|p| p.into_inner());
        for (params, &value) in candidates.iter().zip(&values) {
            elite.offer(params, value, self.capacity);
        }
        values
    }

    /// Forwarded and not recorded: no optimizer sends a batch with other
    /// shots, and values measured with other shots would not rank with the
    /// run's.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.inner.evaluate_batch_with_shots(candidates, shots)
    }
}

/// Send the final batch: `candidates` re-evaluated by `evaluation` with
/// `shots` shots per circuit, in one call.
///
/// The highest value wins; between equal values the earlier candidate, so
/// `best_params`, sent first, wins a tie. A `NaN` never wins over a number;
/// when every value is `NaN` (only a custom oracle can return one, since
/// Polypus's oracles reject non-finite values) the first candidate,
/// `best_params`, is reported with its `NaN` and a warning is logged.
///
/// A failure recorded in `errors` during the batch (a backend error, an
/// interrupt) is returned as the run's, as for any batch; an oracle that
/// returns the wrong number of values is `OracleLengthMismatch`, as when an
/// optimizer sees it.
///
/// `candidates` is never empty: [`EliteRecorder::final_candidates`] always
/// puts `best_params` first.
pub(crate) fn final_batch(
    evaluation: &dyn EvaluationOracle,
    candidates: Vec<Vec<f64>>,
    shots: u32,
    errors: &OracleErrorSlot,
    run_id: &str,
) -> Result<FinalEvaluation, OracleError> {
    debug_assert!(
        !candidates.is_empty(),
        "the final batch always holds best_params"
    );
    log::info!(
        "run {run_id}: re-evaluating {} candidate(s) with {shots} shots each",
        candidates.len()
    );
    let values = evaluation.evaluate_batch_with_shots(&candidates, shots);
    if let Some(err) = errors.take() {
        return Err(OracleError::Evaluation(err));
    }
    if values.len() != candidates.len() {
        return Err(OracleError::Config(OptimizerError::OracleLengthMismatch {
            expected: candidates.len(),
            got: values.len(),
        }));
    }
    let mut best = 0;
    for (index, &value) in values.iter().enumerate().skip(1) {
        let current = values[best];
        if value > current || (current.is_nan() && !value.is_nan()) {
            best = index;
        }
    }
    if values[best].is_nan() {
        log::warn!(
            "run {run_id}: no candidate of the final batch scored a number; \
             final_params is best_params and final_fitness is NaN"
        );
    }
    Ok(FinalEvaluation {
        params: candidates[best].clone(),
        fitness: values[best],
        candidates,
        values,
        shots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{
        AdamConfig, CmaesConfig, DeConfig, ExternalConfig, PsoConfig, QngConfig, RotosolveConfig,
        SpsaConfig,
    };
    use polypus_optimizers::{AdamRule, AskTell, AskTellError, DeStrategy, VarianceOracle};

    /// One call: its candidates and the shots it asked for (`None`: the
    /// oracle's own).
    type Call = (Vec<Vec<f64>>, Option<u32>);

    /// Scores a candidate by its first coordinate, or a batch with shots by
    /// `values` when set, and records every call with its shots.
    #[derive(Default)]
    struct Scripted {
        calls: Mutex<Vec<Call>>,
        values: Option<Vec<f64>>,
    }

    impl Scripted {
        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl EvaluationOracle for Scripted {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.calls.lock().unwrap().push((candidates.to_vec(), None));
            candidates.iter().map(|c| c[0]).collect()
        }

        fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
            self.calls
                .lock()
                .unwrap()
                .push((candidates.to_vec(), Some(shots)));
            match &self.values {
                Some(values) => values.clone(),
                None => candidates.iter().map(|c| c[0]).collect(),
            }
        }
    }

    fn recorder(capacity: usize) -> (EliteRecorder, Arc<Scripted>) {
        let inner = Arc::new(Scripted::default());
        let erased: Arc<dyn EvaluationOracle> = Arc::clone(&inner) as _;
        (EliteRecorder::new(erased, capacity), inner)
    }

    /// One-dimensional candidates scored by their value, `[value, tag]` when
    /// two must score the same.
    fn point(value: f64, tag: f64) -> Vec<f64> {
        vec![value, tag]
    }

    #[test]
    fn keeps_the_five_best_distinct_candidates_best_first() {
        let (elite, inner) = recorder(5);
        let batch: Vec<Vec<f64>> = [3.0, 9.0, 1.0, 7.0, 5.0, 8.0, 2.0]
            .iter()
            .map(|&v| point(v, 0.0))
            .collect();
        let values = elite.evaluate_batch(&batch);
        // Forwarded unchanged.
        assert_eq!(values, vec![3.0, 9.0, 1.0, 7.0, 5.0, 8.0, 2.0]);
        assert_eq!(inner.calls().len(), 1);
        elite.evaluate_batch(&[point(6.0, 0.0), point(0.5, 0.0)]);
        let firsts: Vec<f64> = elite.ranked().iter().map(|c| c[0]).collect();
        assert_eq!(firsts, vec![9.0, 8.0, 7.0, 6.0, 5.0]);
    }

    #[test]
    fn keeps_fewer_when_fewer_distinct_candidates_were_scored() {
        let (elite, _) = recorder(5);
        elite.evaluate_batch(&[point(1.0, 0.0), point(2.0, 0.0), point(1.0, 0.0)]);
        assert_eq!(elite.ranked(), vec![point(2.0, 0.0), point(1.0, 0.0)]);
    }

    #[test]
    fn a_non_finite_value_never_enters() {
        let (elite, _) = recorder(5);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            elite.evaluate_batch(&[point(value, 0.0)]);
        }
        assert!(elite.ranked().is_empty());
        // Only the finite one, however the others compare.
        elite.evaluate_batch(&[point(f64::NAN, 1.0), point(-3.0, 1.0)]);
        assert_eq!(elite.ranked(), vec![point(-3.0, 1.0)]);
    }

    /// The oracle a duplicate goes through scores it anew; it is counted once,
    /// with its best value, and an improved score takes its new place.
    #[test]
    fn a_duplicate_is_counted_once_with_its_best_value() {
        let scores = [1.0, 4.0, 2.0];
        struct ByTag(Vec<f64>, Mutex<usize>);
        impl EvaluationOracle for ByTag {
            fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
                let mut call = self.1.lock().unwrap();
                *call += 1;
                // Candidate [0, 0] scores 1, then 4, then 2 on later calls;
                // every other candidate scores its first coordinate.
                candidates
                    .iter()
                    .map(|c| {
                        if c == &[0.0, 0.0] {
                            self.0[*call - 1]
                        } else {
                            c[0]
                        }
                    })
                    .collect()
            }
        }
        let elite = EliteRecorder::new(Arc::new(ByTag(scores.to_vec(), Mutex::new(0))), 3);
        elite.evaluate_batch(&[point(0.0, 0.0), point(3.0, 0.0), point(2.5, 0.0)]);
        assert_eq!(
            elite.ranked(),
            vec![point(3.0, 0.0), point(2.5, 0.0), point(0.0, 0.0)]
        );
        // Scored 4: it moves up, and is not added twice.
        elite.evaluate_batch(&[point(0.0, 0.0)]);
        assert_eq!(
            elite.ranked(),
            vec![point(0.0, 0.0), point(3.0, 0.0), point(2.5, 0.0)]
        );
        // Scored 2 afterwards: it keeps its 4.
        elite.evaluate_batch(&[point(0.0, 0.0)]);
        assert_eq!(elite.ranked()[0], point(0.0, 0.0));
        assert_eq!(elite.ranked().len(), 3);
    }

    #[test]
    fn distinct_means_distinct_bits() {
        let (elite, _) = recorder(5);
        elite.evaluate_batch(&[vec![0.0, 1.0], vec![-0.0, 1.0], vec![0.0, 1.0]]);
        assert_eq!(elite.ranked().len(), 2);
        let ranked = elite.ranked();
        // Equal values: the earlier first.
        assert_eq!(ranked[0][0].to_bits(), 0.0f64.to_bits());
        assert_eq!(ranked[1][0].to_bits(), (-0.0f64).to_bits());
    }

    /// Equal values rank in the order they were scored, within a batch and
    /// across batches, and a full elite keeps the earlier of a tie.
    #[test]
    fn ties_keep_the_earlier_candidate() {
        let (elite, _) = recorder(3);
        elite.evaluate_batch(&[point(1.0, 1.0), point(2.0, 1.0), point(1.0, 2.0)]);
        elite.evaluate_batch(&[point(1.0, 3.0), point(2.0, 2.0)]);
        assert_eq!(
            elite.ranked(),
            vec![point(2.0, 1.0), point(2.0, 2.0), point(1.0, 1.0)]
        );
        // A tie with the worst kept: the earlier stays.
        elite.evaluate_batch(&[point(1.0, 4.0)]);
        assert_eq!(
            elite.ranked(),
            vec![point(2.0, 1.0), point(2.0, 2.0), point(1.0, 1.0)]
        );
    }

    #[test]
    fn a_batch_with_shots_is_forwarded_but_not_ranked() {
        let (elite, inner) = recorder(5);
        let values = elite.evaluate_batch_with_shots(&[point(9.0, 0.0)], 4096);
        assert_eq!(values, vec![9.0]);
        assert_eq!(inner.calls(), vec![(vec![point(9.0, 0.0)], Some(4096))]);
        assert!(elite.ranked().is_empty());
    }

    #[test]
    fn the_final_candidates_start_with_best_params_without_repeats() {
        let (elite, _) = recorder(5);
        let batch: Vec<Vec<f64>> = (0..8).map(|v| point(f64::from(v), 0.0)).collect();
        elite.evaluate_batch(&batch);
        // `best_params` is the best scored: first, once.
        let best = point(7.0, 0.0);
        let firsts = |c: Vec<Vec<f64>>| c.iter().map(|c| c[0]).collect::<Vec<f64>>();
        assert_eq!(
            firsts(elite.final_candidates(&best, 5)),
            vec![7.0, 6.0, 5.0, 4.0, 3.0]
        );
        // One that the elite missed is still sent, first, and the batch keeps
        // its size.
        let missed = point(-1.0, 0.0);
        assert_eq!(
            firsts(elite.final_candidates(&missed, 5)),
            vec![-1.0, 7.0, 6.0, 5.0, 4.0]
        );
        // A local method: `best_params` alone.
        assert_eq!(elite.final_candidates(&best, 1), vec![best.clone()]);
        // Fewer distinct candidates than the count: all of them.
        let (few, _) = recorder(5);
        few.evaluate_batch(&[point(1.0, 0.0), point(2.0, 0.0)]);
        assert_eq!(
            firsts(few.final_candidates(&point(2.0, 0.0), 5)),
            vec![2.0, 1.0]
        );
    }

    #[test]
    fn final_candidates_counts_five_for_population_and_external_methods_and_one_for_local_ones() {
        struct Unit;
        impl VarianceOracle for Unit {
            fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
                1.0
            }
        }
        struct Done;
        impl AskTell for Done {
            fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
                Ok(None)
            }
            fn tell(&mut self, _: &[Vec<f64>], _: &[f64]) -> Result<(), AskTellError> {
                Ok(())
            }
        }
        let bounds = (-1.0, 1.0);
        let de = |strategy| {
            Method::De(DeConfig {
                generations: 1,
                population_size: 4,
                tolerance: 0.0,
                patience: 1,
                strategy,
                mutation_factor: None,
                crossover_rate: None,
            })
        };
        let methods = [
            (de(DeStrategy::Rand1Bin), 5),
            (de(DeStrategy::Shade), 5),
            (de(DeStrategy::LShade), 5),
            (
                Method::Pso(PsoConfig {
                    generations: 1,
                    population_size: 4,
                    bounds,
                    inertia_weight: 0.5,
                    cognitive_weight: 1.0,
                    social_weight: 1.0,
                    tolerance: 0.0,
                }),
                5,
            ),
            (
                Method::Cmaes(CmaesConfig {
                    generations: 1,
                    population_size: None,
                    sigma0: 1.0,
                    tolerance: 0.0,
                    patience: 1,
                    restarts: 0,
                }),
                5,
            ),
            (
                Method::Qng(
                    QngConfig {
                        max_iters: 1,
                        learning_rate: 0.1,
                        finite_difference_step: 0.1,
                        bounds,
                        tikhonov_reg: 0.0,
                    },
                    Box::new(Unit),
                ),
                1,
            ),
            (
                Method::Adam(AdamConfig {
                    max_iters: 1,
                    learning_rate: 0.1,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    bounds,
                    tolerance: 0.0,
                    patience: 1,
                    finite_difference_step: 0.1,
                    rule: AdamRule::Adam,
                }),
                1,
            ),
            (
                Method::Spsa(SpsaConfig {
                    max_iters: 1,
                    resamplings: 1,
                    perturbation: 0.1,
                    target_step: 0.63,
                    alpha: 0.602,
                    gamma: 0.101,
                    eval_interval: 1,
                }),
                1,
            ),
            (
                Method::Rotosolve(RotosolveConfig {
                    sweeps: 1,
                    reevaluate_every: 1,
                    tolerance: 0.0,
                    patience: 1,
                }),
                1,
            ),
            (
                Method::External(ExternalConfig { max_evaluations: 1 }, Box::new(Done)),
                5,
            ),
        ];
        for (method, count) in methods {
            assert_eq!(method.final_candidates(), count);
        }
    }

    #[test]
    fn the_highest_final_value_wins_and_a_tie_keeps_the_earlier() {
        let candidates = vec![point(1.0, 0.0), point(2.0, 0.0), point(3.0, 0.0)];
        let errors = OracleErrorSlot::new();
        let run = |values: Vec<f64>| {
            let oracle = Scripted {
                values: Some(values),
                ..Scripted::default()
            };
            let result = final_batch(&oracle, candidates.clone(), 4096, &errors, "final-batch");
            let Ok(evaluation) = result else {
                panic!("a healthy final batch succeeds");
            };
            assert_eq!(oracle.calls(), vec![(candidates.clone(), Some(4096))]);
            evaluation
        };
        let evaluation = run(vec![0.5, 0.8, 0.6]);
        assert_eq!(evaluation.params, point(2.0, 0.0));
        assert_eq!(evaluation.fitness, 0.8);
        assert_eq!(evaluation.candidates, candidates);
        assert_eq!(evaluation.values, vec![0.5, 0.8, 0.6]);
        assert_eq!(evaluation.shots, 4096);
        // A tie: the earlier, so `best_params` when it is one of them.
        assert_eq!(run(vec![0.8, 0.8, 0.1]).params, point(1.0, 0.0));
        assert_eq!(run(vec![0.1, 0.8, 0.8]).params, point(2.0, 0.0));
        // A NaN never wins over a number.
        let evaluation = run(vec![f64::NAN, -5.0, f64::NAN]);
        assert_eq!(
            (evaluation.params, evaluation.fitness),
            (point(2.0, 0.0), -5.0)
        );
    }

    #[test]
    fn when_every_final_value_is_nan_best_params_is_reported_with_nan() {
        use crate::test_log::{capture_logs, logged_for};
        capture_logs();
        let run_id = "final-batch-all-nan";
        let oracle = Scripted {
            values: Some(vec![f64::NAN, f64::NAN]),
            ..Scripted::default()
        };
        let candidates = vec![point(1.0, 0.0), point(2.0, 0.0)];
        let Ok(evaluation) = final_batch(&oracle, candidates, 64, &OracleErrorSlot::new(), run_id)
        else {
            panic!("a NaN is a value, not an error");
        };
        assert_eq!(evaluation.params, point(1.0, 0.0));
        assert!(evaluation.fitness.is_nan());
        assert_eq!(logged_for(run_id).len(), 1);
    }

    #[test]
    fn a_wrong_number_of_final_values_is_a_length_mismatch() {
        let oracle = Scripted {
            values: Some(vec![1.0]),
            ..Scripted::default()
        };
        let candidates = vec![point(1.0, 0.0), point(2.0, 0.0)];
        let result = final_batch(&oracle, candidates, 64, &OracleErrorSlot::new(), "x");
        assert!(matches!(
            result,
            Err(OracleError::Config(OptimizerError::OracleLengthMismatch {
                expected: 2,
                got: 1
            }))
        ));
    }
}
