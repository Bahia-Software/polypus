//! Behavioural tests for [`AlgorithmExternal`], the bridge to ask/tell
//! optimizers.
//!
//! The external optimizer here is a seeded random search written in Rust: it
//! records every batch it proposes and every batch it is told, and the oracle
//! records every batch it scores, so a test can check that each `ask` became
//! exactly one oracle batch, that `tell` received that batch with the
//! oracle's values, and every postcondition of contract C-5 against what the
//! oracle saw. No Python is involved.

use polypus_optimizers::{
    AlgorithmExternal, AlgorithmExternalArgs, AskTell, AskTellError, EvaluationOracle,
    OptimizationOutcome, Optimizer, OptimizerError,
};
use std::sync::{Arc, Mutex};

/// SplitMix64: a small deterministic generator, so the tests depend on no
/// RNG crate's stream.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// What the random search was asked for and told.
#[derive(Default)]
struct SearchLog {
    /// Every `ask` call, including the ones that ended the run.
    asks: usize,
    /// Every non-empty batch it proposed, in order.
    proposed: Vec<Vec<Vec<f64>>>,
    /// Every `tell`: the candidates and the values.
    told: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

/// How the random search ends a run once it has proposed its batches.
#[derive(Clone, Copy)]
enum End {
    /// `Ok(None)`.
    Nothing,
    /// `Ok(Some(vec![]))`.
    Empty,
}

/// A random search in `[-2, 2)^dims`, `sizes[k % sizes.len()]` candidates in
/// batch `k`, which ends the run after `batches` batches when that is set.
/// `edit` may change batch `k` before it is proposed; `fail_ask` and
/// `fail_tell` make the `k`-th call of each fail.
struct RandomSearch {
    rng: SplitMix,
    dims: usize,
    sizes: Vec<usize>,
    batches: Option<usize>,
    end: End,
    edit: Edit,
    fail_ask: Option<usize>,
    fail_tell: Option<usize>,
    log: Arc<Mutex<SearchLog>>,
}

impl RandomSearch {
    fn new(seed: u64, dims: usize, sizes: &[usize]) -> Self {
        RandomSearch {
            rng: SplitMix(seed),
            dims,
            sizes: sizes.to_vec(),
            batches: None,
            end: End::Nothing,
            edit: |_, _| {},
            fail_ask: None,
            fail_tell: None,
            log: Arc::default(),
        }
    }

    fn ending_after(mut self, batches: usize, end: End) -> Self {
        self.batches = Some(batches);
        self.end = end;
        self
    }
}

impl AskTell for RandomSearch {
    fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, AskTellError> {
        let mut log = self.log.lock().unwrap();
        let k = log.asks;
        log.asks += 1;
        if self.fail_ask == Some(k) {
            return Err(format!("ask {k} refused").into());
        }
        if self.batches.is_some_and(|b| log.proposed.len() >= b) {
            return Ok(match self.end {
                End::Nothing => None,
                End::Empty => Some(Vec::new()),
            });
        }
        let size = self.sizes[log.proposed.len() % self.sizes.len()];
        let mut batch: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                (0..self.dims)
                    .map(|_| self.rng.uniform(-2.0, 2.0))
                    .collect()
            })
            .collect();
        (self.edit)(log.proposed.len(), &mut batch);
        log.proposed.push(batch.clone());
        Ok(Some(batch))
    }

    fn tell(&mut self, candidates: &[Vec<f64>], values: &[f64]) -> Result<(), AskTellError> {
        let mut log = self.log.lock().unwrap();
        if self.fail_tell == Some(log.told.len()) {
            return Err("the study is closed".into());
        }
        log.told.push((candidates.to_vec(), values.to_vec()));
        Ok(())
    }
}

/// Every batch an oracle scored, with the values it returned.
type Scored = Arc<Mutex<Vec<(Vec<Vec<f64>>, Vec<f64>)>>>;

/// Scores with `fitness`, after `edit` has had its say on the values of batch
/// `k`, and records every batch.
struct Recorder {
    fitness: fn(&[f64]) -> f64,
    edit: fn(usize, &mut Vec<f64>),
    scored: Scored,
}

impl Recorder {
    fn new(fitness: fn(&[f64]) -> f64) -> Self {
        Recorder {
            fitness,
            edit: |_, _| {},
            scored: Scored::default(),
        }
    }
}

impl EvaluationOracle for Recorder {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let mut scored = self.scored.lock().unwrap();
        let mut values: Vec<f64> = candidates.iter().map(|c| (self.fitness)(c)).collect();
        (self.edit)(scored.len(), &mut values);
        scored.push((candidates.to_vec(), values.clone()));
        values
    }
}

/// `−Σ (xᵢ − 0.5)²`, maximised at `xᵢ = 0.5`.
fn sphere(x: &[f64]) -> f64 {
    -x.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>()
}

/// Run the bridge and return its result with what the oracle scored and what
/// the optimizer saw.
fn run(
    oracle: Recorder,
    search: RandomSearch,
    max_evaluations: u64,
    dimensions: u32,
) -> (
    Result<OptimizationOutcome, OptimizerError>,
    Scored,
    Arc<Mutex<SearchLog>>,
) {
    let scored = Arc::clone(&oracle.scored);
    let log = Arc::clone(&search.log);
    let result = AlgorithmExternal.optimize(AlgorithmExternalArgs {
        oracle: Box::new(oracle),
        optimizer: Box::new(search),
        max_evaluations,
        dimensions,
    });
    (result, scored, log)
}

/// The C-5 postconditions of `outcome` against the batches the oracle scored:
/// `best_params` the first candidate with the highest value (a `NaN` never
/// above a number), `best_fitness` its value, and `fitness_history` the best
/// value after each batch.
fn assert_measured_best(outcome: &OptimizationOutcome, scored: &[(Vec<Vec<f64>>, Vec<f64>)]) {
    let mut best: Option<(&Vec<f64>, f64)> = None;
    let mut history = Vec::new();
    for (candidates, values) in scored {
        for (candidate, &value) in candidates.iter().zip(values) {
            let wins = match best {
                None => true,
                Some((_, b)) => value > b || (b.is_nan() && !value.is_nan()),
            };
            if wins {
                best = Some((candidate, value));
            }
        }
        history.push(best.expect("a batch is never empty").1);
    }
    let (params, fitness) = best.expect("something was scored");
    assert_eq!(&outcome.best_params, params);
    assert_eq!(outcome.best_fitness.to_bits(), fitness.to_bits());
    assert_eq!(outcome.iterations_run, scored.len());
    assert_eq!(outcome.fitness_history.len(), outcome.iterations_run);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&outcome.fitness_history), bits(&history));
    assert_eq!(
        outcome.fitness_history.last().map(|v| v.to_bits()),
        Some(outcome.best_fitness.to_bits())
    );
}

#[test]
fn each_ask_is_one_oracle_batch_and_its_values_are_told() {
    let search = RandomSearch::new(7, 2, &[3, 5, 1]).ending_after(6, End::Nothing);
    let (result, scored, log) = run(Recorder::new(sphere), search, 1000, 2);
    let outcome = result.expect("the run succeeds");
    let scored = scored.lock().unwrap();
    let log = log.lock().unwrap();
    // Six batches, then the empty ask that ended the run.
    assert_eq!(log.asks, 7);
    let sizes: Vec<usize> = scored.iter().map(|(c, _)| c.len()).collect();
    assert_eq!(sizes, [3, 5, 1, 3, 5, 1]);
    // Exactly the proposed batches were scored, and each was told with its
    // own values, in order.
    let scored_batches: Vec<Vec<Vec<f64>>> = scored.iter().map(|(c, _)| c.clone()).collect();
    assert_eq!(scored_batches, log.proposed);
    assert_eq!(log.told, *scored);
    assert!(outcome.converged, "the optimizer ended the run");
    assert_measured_best(&outcome, &scored);
}

#[test]
fn an_empty_batch_ends_the_run_like_none() {
    for end in [End::Nothing, End::Empty] {
        let search = RandomSearch::new(1, 3, &[4]).ending_after(2, end);
        let (result, scored, log) = run(Recorder::new(sphere), search, 100, 3);
        let outcome = result.expect("the run succeeds");
        assert!(outcome.converged);
        assert_eq!(outcome.iterations_run, 2);
        assert_eq!(scored.lock().unwrap().len(), 2);
        assert_eq!(log.lock().unwrap().asks, 3);
    }
}

#[test]
fn a_batch_that_would_exceed_the_budget_is_neither_scored_nor_told() {
    // Batches of 4 against 10 evaluations: two fit (8), the third would make
    // 12, so it is asked for but neither scored nor told.
    let search = RandomSearch::new(3, 2, &[4]);
    let (result, scored, log) = run(Recorder::new(sphere), search, 10, 2);
    let outcome = result.expect("the run succeeds");
    let log = log.lock().unwrap();
    assert_eq!(log.asks, 3);
    assert_eq!(log.proposed.len(), 3);
    assert_eq!(scored.lock().unwrap().len(), 2);
    assert_eq!(log.told.len(), 2);
    assert!(!outcome.converged, "the budget ended the run");
    assert_eq!(outcome.iterations_run, 2);
    assert!(!scored
        .lock()
        .unwrap()
        .iter()
        .any(|(c, _)| *c == log.proposed[2]));
}

#[test]
fn a_budget_spent_exactly_ends_the_run_without_asking_again() {
    let search = RandomSearch::new(3, 2, &[5]);
    let (result, scored, log) = run(Recorder::new(sphere), search, 10, 2);
    let outcome = result.expect("the run succeeds");
    assert_eq!(log.lock().unwrap().asks, 2);
    assert_eq!(scored.lock().unwrap().len(), 2);
    assert!(!outcome.converged);
    assert_eq!(outcome.iterations_run, 2);
}

#[test]
fn the_budget_is_never_exceeded_whatever_the_batches() {
    for seed in 0..25u64 {
        for budget in [1u64, 7, 12, 30] {
            let search = RandomSearch::new(seed, 2, &[1 + (seed as usize % 4), 3, 2]);
            let (result, scored, log) = run(Recorder::new(sphere), search, budget, 2);
            let scored = scored.lock().unwrap();
            let spent: usize = scored.iter().map(|(c, _)| c.len()).sum();
            assert!(spent as u64 <= budget, "seed {seed}, budget {budget}");
            let log = log.lock().unwrap();
            match result {
                Ok(outcome) => {
                    assert!(!outcome.converged);
                    // Stopped by the budget: spent exactly, or the next
                    // batch, asked but not scored, would have exceeded it.
                    if spent as u64 != budget {
                        let next = log.proposed[scored.len()].len();
                        assert!((spent + next) as u64 > budget, "seed {seed}");
                        assert_eq!(log.proposed.len(), scored.len() + 1);
                    } else {
                        assert_eq!(log.proposed.len(), scored.len());
                    }
                    assert_measured_best(&outcome, &scored);
                }
                // Only a first batch larger than the whole budget.
                Err(OptimizerError::InvalidConfig { parameter, .. }) => {
                    assert_eq!(parameter, "max_evaluations");
                    assert!(scored.is_empty());
                    assert!(log.proposed[0].len() as u64 > budget);
                }
                Err(other) => panic!("seed {seed}, budget {budget}: {other}"),
            }
        }
    }
}

#[test]
fn a_first_ask_that_proposes_nothing_is_an_error_with_no_oracle_call() {
    for end in [End::Nothing, End::Empty] {
        let search = RandomSearch::new(1, 2, &[4]).ending_after(0, end);
        let (result, scored, log) = run(Recorder::new(sphere), search, 100, 2);
        match result {
            Err(OptimizerError::ExternalOptimizer { reason }) => {
                assert!(
                    reason.contains("no candidates on its first call"),
                    "{reason}"
                )
            }
            other => panic!("expected ExternalOptimizer, got {other:?}"),
        }
        assert!(scored.lock().unwrap().is_empty());
        assert!(log.lock().unwrap().told.is_empty());
    }
}

#[test]
fn a_first_batch_larger_than_the_budget_is_rejected_before_any_oracle_call() {
    let search = RandomSearch::new(1, 2, &[8]);
    let (result, scored, log) = run(Recorder::new(sphere), search, 5, 2);
    let err = result.expect_err("nothing fits the budget");
    assert_eq!(
        err.to_string(),
        "invalid max_evaluations: the external optimizer's first batch has more than \
         max_evaluations (5) candidates; nothing was evaluated"
    );
    assert!(scored.lock().unwrap().is_empty());
    assert!(log.lock().unwrap().told.is_empty());
}

/// Changes batch `k` before the random search proposes it.
type Edit = fn(usize, &mut Vec<Vec<f64>>);

/// Edits that spoil row 2 of batch 1, and the reason each gets.
fn bad_rows() -> Vec<(Edit, &'static str)> {
    vec![
        (
            |k, batch| {
                if k == 1 {
                    batch[2].push(0.0);
                }
            },
            "it has 3 values, expected one per dimension (2)",
        ),
        (
            |k, batch| {
                if k == 1 {
                    batch[2].pop();
                }
            },
            "it has 1 values, expected one per dimension (2)",
        ),
        (
            |k, batch| {
                if k == 1 {
                    batch[2][1] = f64::NAN;
                }
            },
            "its value at index 1 is NaN; every value must be finite",
        ),
        (
            |k, batch| {
                if k == 1 {
                    batch[2][0] = f64::INFINITY;
                }
            },
            "its value at index 0 is inf; every value must be finite",
        ),
        (
            |k, batch| {
                if k == 1 {
                    batch[2][1] = f64::NEG_INFINITY;
                }
            },
            "its value at index 1 is -inf; every value must be finite",
        ),
    ]
}

#[test]
fn a_row_that_is_not_a_candidate_stops_the_run_before_its_batch_is_scored() {
    for (edit, reason) in bad_rows() {
        let mut search = RandomSearch::new(5, 2, &[4]);
        search.edit = edit;
        let (result, scored, log) = run(Recorder::new(sphere), search, 100, 2);
        assert_eq!(
            result,
            Err(OptimizerError::InvalidCandidate {
                batch: 1,
                row: 2,
                reason: reason.to_string()
            })
        );
        // Only the first batch was scored and told.
        assert_eq!(scored.lock().unwrap().len(), 1);
        assert_eq!(log.lock().unwrap().told.len(), 1);
    }
    let err = OptimizerError::InvalidCandidate {
        batch: 1,
        row: 2,
        reason: "it has 3 values, expected one per dimension (2)".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "invalid candidate in row 2 of batch 1 from the external optimizer's ask() (both \
         0-based): it has 3 values, expected one per dimension (2)"
    );
}

#[test]
fn an_ask_that_fails_ends_the_run_with_its_message() {
    let mut search = RandomSearch::new(2, 2, &[3]);
    search.fail_ask = Some(2);
    let (result, scored, log) = run(Recorder::new(sphere), search, 100, 2);
    let err = result.expect_err("ask failed");
    assert_eq!(
        err,
        OptimizerError::ExternalOptimizer {
            reason: "ask() failed: ask 2 refused".to_string()
        }
    );
    assert_eq!(
        err.to_string(),
        "external optimizer: ask() failed: ask 2 refused"
    );
    assert_eq!(scored.lock().unwrap().len(), 2);
    assert_eq!(log.lock().unwrap().told.len(), 2);
}

#[test]
fn a_tell_that_fails_ends_the_run_with_its_message() {
    let mut search = RandomSearch::new(2, 2, &[3]);
    search.fail_tell = Some(1);
    let (result, scored, log) = run(Recorder::new(sphere), search, 100, 2);
    assert_eq!(
        result,
        Err(OptimizerError::ExternalOptimizer {
            reason: "tell() failed: the study is closed".to_string()
        })
    );
    // The batch was scored, its tell failed, and nothing was asked after it.
    assert_eq!(scored.lock().unwrap().len(), 2);
    let log = log.lock().unwrap();
    assert_eq!(log.told.len(), 1);
    assert_eq!(log.asks, 2);
}

/// Returns one value too few on its second batch.
struct Short;
impl EvaluationOracle for Short {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        let n = if candidates.len() == 2 {
            1
        } else {
            candidates.len()
        };
        vec![0.0; n]
    }
}

#[test]
fn an_oracle_of_the_wrong_length_is_an_error_and_its_batch_is_not_told() {
    let search = RandomSearch::new(4, 2, &[3, 2]);
    let log = Arc::clone(&search.log);
    let result = AlgorithmExternal.optimize(AlgorithmExternalArgs {
        oracle: Box::new(Short),
        optimizer: Box::new(search),
        max_evaluations: 100,
        dimensions: 2,
    });
    assert_eq!(
        result,
        Err(OptimizerError::OracleLengthMismatch {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(log.lock().unwrap().told.len(), 1);
}

#[test]
fn an_invalid_configuration_is_rejected_before_any_ask() {
    for (max_evaluations, dimensions, parameter) in [
        (0, 2, "max_evaluations"),
        (0, 0, "max_evaluations"),
        (10, 0, "dimensions"),
    ] {
        let search = RandomSearch::new(1, 2, &[2]);
        let (result, scored, log) = run(Recorder::new(sphere), search, max_evaluations, dimensions);
        match result {
            Err(OptimizerError::InvalidConfig { parameter: got, .. }) => {
                assert_eq!(got, parameter)
            }
            other => panic!("expected InvalidConfig for {parameter}, got {other:?}"),
        }
        assert_eq!(log.lock().unwrap().asks, 0);
        assert!(scored.lock().unwrap().is_empty());
    }
}

/// C-5 over 25 seeds and several batch patterns: `best_params` is the best
/// candidate the oracle scored, `best_fitness` the value it returned for it,
/// `fitness_history` the best after each batch, and `tell` got exactly what
/// the oracle returned.
#[test]
fn best_params_fitness_invariant_holds_across_seeds() {
    for seed in 0..25u64 {
        for sizes in [&[6][..], &[1, 4, 9], &[5, 5, 2]] {
            let search = RandomSearch::new(seed, 3, sizes).ending_after(12, End::Empty);
            let (result, scored, log) = run(Recorder::new(sphere), search, 60, 3);
            let outcome = result.expect("the run succeeds");
            let scored = scored.lock().unwrap();
            assert_measured_best(&outcome, &scored);
            assert_eq!(log.lock().unwrap().told, *scored, "seed {seed}");
            let spent: usize = scored.iter().map(|(c, _)| c.len()).sum();
            assert!(spent <= 60);
            // The run ends on its own only when all twelve batches fit.
            let total: usize = (0..12).map(|k| sizes[k % sizes.len()]).sum();
            assert_eq!(outcome.converged, total < 60, "seed {seed}, {sizes:?}");
        }
    }
}

#[test]
fn a_nan_never_wins_over_a_number() {
    // The first value of every batch is NaN: never the best.
    let mut oracle = Recorder::new(sphere);
    oracle.edit = |_, values| values[0] = f64::NAN;
    let search = RandomSearch::new(9, 2, &[3]).ending_after(4, End::Nothing);
    let (result, scored, _) = run(oracle, search, 100, 2);
    let outcome = result.expect("the run succeeds");
    assert!(outcome.best_fitness.is_finite());
    assert!(outcome.fitness_history.iter().all(|v| v.is_finite()));
    assert_measured_best(&outcome, &scored.lock().unwrap());

    // Only NaN: the first candidate scored, with its NaN.
    let mut oracle = Recorder::new(sphere);
    oracle.edit = |_, values| values.iter_mut().for_each(|v| *v = f64::NAN);
    let search = RandomSearch::new(9, 2, &[3]).ending_after(2, End::Nothing);
    let (result, scored, _) = run(oracle, search, 100, 2);
    let outcome = result.expect("the run succeeds");
    assert!(outcome.best_fitness.is_nan());
    assert_eq!(outcome.best_params, scored.lock().unwrap()[0].0[0]);
}

#[test]
fn a_deterministic_optimizer_gives_a_deterministic_run() {
    let once = || {
        let search = RandomSearch::new(42, 4, &[7, 3]).ending_after(9, End::Nothing);
        run(Recorder::new(sphere), search, 40, 4)
            .0
            .expect("the run succeeds")
    };
    assert_eq!(once(), once());
}
