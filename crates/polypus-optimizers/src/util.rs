//! Internal helpers shared by the optimizers.
//!
//! These are implementation details of the crate's optimizers —
//! [`AlgorithmDifferentialEvolution`](crate::AlgorithmDifferentialEvolution) and
//! [`AlgorithmPSO`](crate::AlgorithmPSO) for the population-based convergence
//! test, [`AlgorithmQNG`](crate::AlgorithmQNG) and [`AlgorithmAdam`](crate::AlgorithmAdam)
//! for the gradient-norm one — not part of the crate's public API, so the module
//! is `pub(crate)`-scoped. Each helper is a small, single-purpose free function
//! that used to be inlined (and duplicated) inside the optimizer loops. The
//! behaviour is preserved exactly — same operation order, same comparators, same
//! diagnostics — so the determinism the tests pin (`tests/optimizers.rs`) is
//! unaffected.

use crate::error::OptimizerError;
use ndarray::{Array2, Axis};

/// NaN-safe argmax over a fitness slice.
///
/// Returns the index of the maximum value, treating incomparable pairs (any
/// comparison involving `NaN`) as [`Equal`](std::cmp::Ordering::Equal) so it
/// never panics on non-finite fitness. On ties the *last* maximal index wins
/// (the documented [`Iterator::max_by`] behaviour), and an empty slice returns
/// `0`. Used by DE and PSO to locate the current best candidate.
pub(crate) fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Convert each row of a population matrix into an owned `Vec<f64>` candidate,
/// the shape [`EvaluationOracle::evaluate_batch`](crate::EvaluationOracle::evaluate_batch)
/// expects. Used by DE and PSO to hand their `Array2` populations to the oracle
/// as a single batch.
pub(crate) fn rows_to_candidates(population: &Array2<f64>) -> Vec<Vec<f64>> {
    population.outer_iter().map(|r| r.to_vec()).collect()
}

/// Population-based early-stopping test shared by DE and PSO.
///
/// Returns `true` once **every** dimension's population standard deviation
/// drops below the absolute threshold `tolerance`, i.e. the population has
/// collapsed towards a point in every coordinate. Emits the same per-generation
/// `log::debug!` diagnostics the two loops previously inlined (now reporting the
/// worst per-dimension spread), including the "stopping early" line when
/// convergence fires. QNG has no population and does not use this.
///
/// The old criterion — summed `std < tolerance * mean` against the summed
/// per-dimension *means* — was dimensionally incoherent: the mean is a *signed*
/// sum, so for any search space symmetric about zero (e.g. PSO's default bounds
/// `(-π, π)`) a population collapsing around a near-zero point drives `mean → 0`
/// and the test reduces to `std < 0`, which (`std ≥ 0`) can never fire. Summing
/// across dimensions also let one tight dimension mask a wide one. The
/// per-dimension absolute comparison is scale-honest (a plain std threshold in
/// the parameters' own units) and independent of where the optimum sits.
pub(crate) fn population_converged(
    population: &Array2<f64>,
    tolerance: f64,
    generation: usize,
) -> bool {
    let std = population.std_axis(Axis(0), 0.0);
    let max_std = std.iter().copied().fold(0.0_f64, f64::max);
    log::debug!("Generation {generation}: max per-dimension std {max_std:.4}");
    // An empty std vector (zero dimensions) has nothing to collapse, so it is
    // never "converged"; otherwise the worst dimension must be below tolerance.
    let converged = !std.is_empty() && max_std < tolerance;
    if converged {
        log::debug!("Stopping early at generation {generation} due to convergence");
    }
    converged
}

/// Gradient-norm early-stopping test shared by QNG and Adam.
///
/// The single-point counterpart of [`population_converged`]: instead of a
/// population collapsing in every dimension, the convergence signal is the L2
/// norm `‖∇fitness(θ)‖` of the iteration's gradient dropping below the absolute
/// threshold `tolerance`. Returns `true` — the loop should stop with
/// `converged = true` — only once that has held for `patience` **consecutive**
/// iterations; a single sub-tolerance iteration is deliberately not enough,
/// since a minibatch gradient can cancel to exactly zero at a `θ` whose
/// full-dataset gradient is far from zero (see the minibatch note beside
/// C-5/C-7 in `docs/CONTRACTS.md`).
///
/// `below_tolerance_streak` is the caller's persistent counter, threaded by
/// `&mut` across iterations: it is incremented on each sub-tolerance iteration
/// and reset to `0` on any iteration that is not, so the required iterations
/// must be *consecutive*. The gradient norm is computed here rather than by the
/// caller so the whole rule — norm, streak bookkeeping and the `>= patience`
/// decision — lives in one place; `CONTRACTS.md` states QNG and Adam behave
/// identically precisely because it does. `patience = 0` behaves like `1`: the
/// streak is only ever tested right after being incremented, so it is always
/// `>= 1` at the comparison.
pub(crate) fn patience_converged(
    grad: &[f64],
    tolerance: f64,
    patience: usize,
    below_tolerance_streak: &mut usize,
) -> bool {
    let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    if grad_norm < tolerance {
        *below_tolerance_streak += 1;
        *below_tolerance_streak >= patience
    } else {
        *below_tolerance_streak = 0;
        false
    }
}

/// Fitness-stagnation early-stopping test used by Differential Evolution.
///
/// The optimizers *maximise* (higher fitness is better) and DE's best fitness
/// is monotonically non-decreasing, so `history` — the best fitness recorded at
/// the end of each generation — never falls. This returns `true` once the best
/// fitness has improved by **less than `tolerance`** over the last `patience`
/// generations, i.e. the search has stalled in *quality* rather than in
/// population spread.
///
/// Concretely, with `history` holding one entry per executed generation
/// (`history[g]` is the best fitness at the end of generation `g`, 0-indexed),
/// the test fires at the current generation `g` iff
///
/// ```text
/// g >= patience  &&  history[g] - history[g - patience] < tolerance
/// ```
///
/// The `g >= patience` guard (`history.len() > patience`) means the first
/// `patience` generations can never trigger a stop — a full `patience`-wide
/// window of history must exist behind the current generation first. This is
/// the criterion the C-5 contract prose describes, and it replaces the former
/// population-standard-deviation collapse test for DE: on landscapes like QAOA
/// the population would collapse (std → 0) within a handful of generations and
/// stop the run long before the fitness had actually plateaued.
///
/// `tolerance` is therefore a *minimum cumulative fitness improvement* in the
/// oracle's own fitness units, not a spread in parameter units.
pub(crate) fn fitness_stagnated(
    history: &[f64],
    tolerance: f64,
    patience: usize,
    generation: usize,
) -> bool {
    // Need a full `patience`-wide window behind the current generation: with one
    // entry per generation, `history.len() > patience` is exactly `generation >=
    // patience` (`generation == history.len() - 1`).
    if history.len() <= patience {
        return false;
    }
    let current = history[history.len() - 1];
    let past = history[history.len() - 1 - patience];
    let improvement = current - past;
    log::debug!(
        "Generation {generation}: best-fitness improvement over last {patience} generations = {improvement:.6}"
    );
    let stagnated = improvement < tolerance;
    if stagnated {
        log::debug!("Stopping early at generation {generation} due to fitness stagnation");
    }
    stagnated
}

/// Validate that an oracle returned exactly one fitness value per candidate.
///
/// Every optimizer — and every helper built on the oracle traits, such as
/// [`linear_parameter_shift_gradient`](crate::linear_parameter_shift_gradient),
/// whose per-parameter ±π/2 pair makes the expected length `2 * dims` — calls
/// [`EvaluationOracle::evaluate_batch`](crate::EvaluationOracle::evaluate_batch)
/// and then indexes the returned slice positionally; a short (or long) return
/// would otherwise panic with an out-of-bounds index deep inside the loop.
/// Checking the length immediately after each batch call — for *any* oracle,
/// Python-backed or not — turns that into the typed
/// [`OptimizerError::OracleLengthMismatch`] the FFI seam maps to a
/// `PyValueError` (or, where a [`GradientOracle`](crate::GradientOracle) that
/// delegates to that helper records the error instead of returning it, to the
/// bindings' evaluation exception).
pub(crate) fn check_oracle_len(expected: usize, got: usize) -> Result<(), OptimizerError> {
    if expected == got {
        Ok(())
    } else {
        Err(OptimizerError::OracleLengthMismatch { expected, got })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn argmax_single_element() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn argmax_finds_maximum() {
        assert_eq!(argmax(&[0.1, 5.0, 2.0, -3.0]), 1);
    }

    #[test]
    fn argmax_ties_return_last() {
        // `Iterator::max_by` keeps the last maximal element on ties; the inline
        // code being extracted relied on exactly this, so it is preserved.
        assert_eq!(argmax(&[3.0, 1.0, 3.0]), 2);
    }

    #[test]
    fn argmax_all_nan_does_not_panic() {
        // Mirrors the `NanOracle` test intent: the comparator maps NaN to
        // `Equal`, so no panic; the last index wins.
        assert_eq!(argmax(&[f64::NAN, f64::NAN, f64::NAN]), 2);
    }

    #[test]
    fn argmax_nan_mixed_does_not_panic() {
        // A NaN is neither greater nor less than a real number, so it never
        // "wins" a comparison and the scan simply keeps advancing. The point of
        // the assertion is that it returns an in-bounds index without panicking.
        let idx = argmax(&[1.0, f64::NAN, 2.0]);
        assert!(idx < 3);
    }

    #[test]
    fn rows_to_candidates_maps_each_row() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(
            rows_to_candidates(&m),
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
    }

    #[test]
    fn rows_to_candidates_empty_population() {
        // Zero rows → zero candidates.
        let m = Array2::<f64>::zeros((0, 3));
        assert!(rows_to_candidates(&m).is_empty());
    }

    #[test]
    fn rows_to_candidates_zero_dimensions() {
        // Rows exist but are empty (dimensions == 0), mirroring
        // `de_handles_zero_dimensions`.
        let m = Array2::<f64>::zeros((2, 0));
        assert_eq!(rows_to_candidates(&m), vec![Vec::<f64>::new(), Vec::new()]);
    }

    #[test]
    fn population_converged_true_when_collapsed() {
        // Every row identical → every per-dimension std == 0 < tolerance.
        let pop = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        assert!(population_converged(&pop, 0.5, 0));
    }

    #[test]
    fn population_converged_true_when_collapsed_at_zero() {
        // Regression for the symmetric-bounds bug: a population collapsed around
        // 0 (every per-dimension mean ≈ 0, as under PSO's default bounds
        // (-π, π)) must still be detected. The old `std < tolerance * mean`
        // reduced to `std < 0` here and could never fire; the per-dimension
        // absolute test does.
        let pop = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert!(population_converged(&pop, 0.01, 0));
    }

    #[test]
    fn population_converged_false_when_spread() {
        let pop = array![[0.0, 0.0], [10.0, 10.0]];
        assert!(!population_converged(&pop, 1e-9, 0));
    }

    #[test]
    fn population_converged_false_when_symmetric_and_spread() {
        // Symmetric about 0 (per-dimension mean ≈ 0) but widely spread: the old
        // criterion compared against `tolerance * 0 ≈ 0` and mislabelled such
        // populations; the new one correctly reports "not converged" because
        // each dimension's std (3.0) far exceeds the tolerance.
        let pop = array![[-3.0, -3.0], [3.0, 3.0]];
        assert!(!population_converged(&pop, 0.01, 0));
    }

    #[test]
    fn population_converged_requires_every_dimension_below_tolerance() {
        // One tight dimension (std 0) and one wide dimension (std 2.0):
        // convergence is per-dimension, so the wide dimension alone keeps the
        // whole population "not converged" — the old summed comparison could let
        // a tight dimension mask a wide one.
        let pop = array![[1.0, -2.0], [1.0, 2.0]];
        assert!(!population_converged(&pop, 0.5, 0));
    }

    #[test]
    fn population_converged_zero_dimensions_does_not_panic() {
        // No dimensions → empty std vector → never converged, no panic. Matches
        // the zero-dimension DE edge case (`de_handles_zero_dimensions`).
        let pop = Array2::<f64>::zeros((3, 0));
        assert!(!population_converged(&pop, 1e-9, 0));
    }

    #[test]
    fn patience_converged_streak_fires_exactly_at_patience() {
        // A gradient whose norm is below tolerance every call: the streak must
        // accumulate and fire on exactly the `patience`-th consecutive call —
        // not before, not after.
        let below = [0.0, 0.0];
        let mut streak = 0usize;
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 1
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 2
        assert!(patience_converged(&below, 1e-6, 3, &mut streak)); // streak 3 → fire
        assert_eq!(streak, 3);
    }

    #[test]
    fn patience_converged_above_tolerance_resets_streak_mid_run() {
        // An iteration above tolerance resets the streak to 0 even partway
        // through a run, so the `patience` consecutive iterations must restart.
        let below = [0.0, 0.0];
        let above = [1.0, 0.0];
        let mut streak = 0usize;
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 1
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 2
        assert!(!patience_converged(&above, 1e-6, 3, &mut streak)); // reset → 0
        assert_eq!(streak, 0);
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 1 again
        assert!(!patience_converged(&below, 1e-6, 3, &mut streak)); // streak 2
        assert!(patience_converged(&below, 1e-6, 3, &mut streak)); // streak 3 → fire
    }

    #[test]
    fn patience_converged_patience_one_fires_immediately() {
        // `patience = 1` reproduces the pre-`patience` single-iteration rule:
        // the first sub-tolerance iteration fires.
        let below = [0.0, 0.0];
        let mut streak = 0usize;
        assert!(patience_converged(&below, 1e-6, 1, &mut streak));
        assert_eq!(streak, 1);
    }

    #[test]
    fn patience_converged_streak_persists_across_calls() {
        // Passing the same `&mut usize` on successive calls (as the optimizer
        // loop does) must carry the streak forward rather than starting over.
        let below = [0.0];
        let mut streak = 0usize;
        assert!(!patience_converged(&below, 1e-6, 2, &mut streak));
        assert_eq!(streak, 1);
        assert!(patience_converged(&below, 1e-6, 2, &mut streak));
        assert_eq!(streak, 2);
    }

    #[test]
    fn patience_converged_norm_is_l2_over_all_components() {
        // The norm absorbed into the helper is the L2 norm over every gradient
        // component: a vector each of whose components is individually below
        // tolerance can still have a norm above it, so no stop.
        let grad = [0.4, 0.4, 0.4]; // ‖·‖ ≈ 0.69 > 0.5, though each |g| < 0.5
        let mut streak = 0usize;
        assert!(!patience_converged(&grad, 0.5, 1, &mut streak));
        assert_eq!(streak, 0);
    }

    #[test]
    fn fitness_stagnated_false_before_patience_window_filled() {
        // Fewer than `patience + 1` generations of history → no full window to
        // look back over, so it can never fire regardless of the values.
        let history = [0.0, 1.0, 2.0, 3.0];
        assert!(!fitness_stagnated(&history, 0.5, 4, 3));
        // Exactly `patience` entries is still one short of a full window.
        assert!(!fitness_stagnated(&history, 100.0, 4, 3));
    }

    #[test]
    fn fitness_stagnated_true_when_improvement_below_tolerance() {
        // len == patience + 1: window spans history[0]..history[4]. Improvement
        // 3.02 - 3.0 = 0.02 < tolerance 0.5 ⇒ stagnated.
        let history = [3.0, 3.005, 3.01, 3.015, 3.02];
        assert!(fitness_stagnated(&history, 0.5, 4, 4));
    }

    #[test]
    fn fitness_stagnated_false_when_still_improving() {
        // Improvement 4.0 - 0.0 = 4.0 over the window is well above tolerance.
        let history = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(!fitness_stagnated(&history, 0.5, 4, 4));
    }

    #[test]
    fn fitness_stagnated_compares_only_the_last_patience_generations() {
        // A big early gain then a flat tail: with patience 2 the window is the
        // last two steps (5.001 - 5.0 = 0.001 < tolerance), so it stagnates even
        // though the run improved a lot earlier.
        let history = [0.0, 5.0, 5.0005, 5.001];
        assert!(fitness_stagnated(&history, 0.01, 2, 3));
    }

    #[test]
    fn check_oracle_len_accepts_matching_length() {
        assert_eq!(check_oracle_len(5, 5), Ok(()));
    }

    #[test]
    fn check_oracle_len_rejects_mismatch() {
        assert_eq!(
            check_oracle_len(5, 4),
            Err(OptimizerError::OracleLengthMismatch {
                expected: 5,
                got: 4
            })
        );
    }
}
