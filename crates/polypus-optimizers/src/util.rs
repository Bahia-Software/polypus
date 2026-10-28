//! Internal helpers shared by the population-based optimizers.
//!
//! These are implementation details of [`AlgorithmDifferentialEvolution`](crate::AlgorithmDifferentialEvolution),
//! [`AlgorithmPSO`](crate::AlgorithmPSO) and (the oracle length check and the
//! stagnation rule) [`AlgorithmCmaEs`](crate::AlgorithmCmaEs), not part of the
//! crate's public API, so the module is `pub(crate)`-scoped. Every optimizer,
//! [`AlgorithmQNG`](crate::AlgorithmQNG) included, checks its optional starting
//! point with [`check_initial_point`], DE, CMA-ES and
//! [`AlgorithmRotosolve`](crate::AlgorithmRotosolve) share [`wrap_angle`], and
//! [`AlgorithmAdam`](crate::AlgorithmAdam),
//! [`AlgorithmSPSA`](crate::AlgorithmSPSA), Rotosolve and
//! [`AlgorithmExternal`](crate::AlgorithmExternal) keep their best point with
//! [`BestMeasured`].
//! Each helper is a small, single-purpose free function that used to be inlined
//! (and duplicated) inside the optimizer loops. The behaviour is preserved
//! exactly — same operation order, same comparators, same diagnostics — so the
//! determinism the tests pin (`tests/optimizers.rs`) is unaffected.

use crate::error::OptimizerError;
use ndarray::{Array2, Axis};
use std::f64::consts::{PI, TAU};

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
///
/// This is the logging form DE calls; the rule itself is [`stagnated`], which
/// CMA-ES applies to each of its restarts with its own log lines.
pub(crate) fn fitness_stagnated(
    history: &[f64],
    tolerance: f64,
    patience: usize,
    generation: usize,
) -> bool {
    if let Some(improvement) = recent_improvement(history, patience) {
        log::debug!(
            "Generation {generation}: best-fitness improvement over last {patience} generations = {improvement:.6}"
        );
    }
    let stagnated = stagnated(history, tolerance, patience);
    if stagnated {
        log::debug!("Stopping early at generation {generation} due to fitness stagnation");
    }
    stagnated
}

/// The stagnation rule of [`fitness_stagnated`], without its logging: `true`
/// iff `history` holds a full `patience`-wide window and the best fitness
/// improved by less than `tolerance` over it.
pub(crate) fn stagnated(history: &[f64], tolerance: f64, patience: usize) -> bool {
    recent_improvement(history, patience).is_some_and(|improvement| improvement < tolerance)
}

/// `patience` must be at least 1: the stagnation rule compares the best
/// fitness with `patience` `unit` earlier, and `0` would compare one with
/// itself, so a run with a positive `tolerance` would stop after its first.
/// DE and CMA-ES check it here; Adam and Rotosolve, with their own wording.
pub(crate) fn check_patience(patience: u32, unit: &str) -> Result<(), OptimizerError> {
    if patience == 0 {
        return Err(OptimizerError::InvalidConfig {
            parameter: "patience",
            reason: format!(
                "must be at least 1 ({unit} back to compare the best fitness with; 0 \
                 would compare one with itself)"
            ),
        });
    }
    Ok(())
}

/// `history[g] - history[g - patience]` for the last generation `g`, or `None`
/// while there is no full `patience`-wide window behind it: with one entry per
/// generation, `history.len() > patience` is exactly `g >= patience`
/// (`g == history.len() - 1`).
pub(crate) fn recent_improvement(history: &[f64], patience: usize) -> Option<f64> {
    if history.len() <= patience {
        return None;
    }
    let current = history[history.len() - 1];
    let past = history[history.len() - 1 - patience];
    Some(current - past)
}

/// Validate that an oracle returned exactly one fitness value per candidate.
///
/// Every optimizer calls [`EvaluationOracle::evaluate_batch`](crate::EvaluationOracle::evaluate_batch)
/// and then indexes the returned slice positionally; a short (or long) return
/// would otherwise panic with an out-of-bounds index deep inside the loop.
/// Checking the length immediately after each batch call — for *any* oracle,
/// Python-backed or not — turns that into the typed
/// [`OptimizerError::OracleLengthMismatch`] the FFI seam maps to a
/// `polypus.EvaluationError`.
pub(crate) fn check_oracle_len(expected: usize, got: usize) -> Result<(), OptimizerError> {
    if expected == got {
        Ok(())
    } else {
        Err(OptimizerError::OracleLengthMismatch { expected, got })
    }
}

/// Validate an optimizer's optional starting point against its `dims`
/// parameters: exactly `dims` values, every one finite, and, when `bounds`
/// is `Some((lb, ub))`, every one within `[lb, ub]` (PSO's, whose positions
/// are clamped to them).
///
/// Each optimizer calls this from its `check_preconditions`, after its other
/// configuration checks and before any RNG draw or oracle call, so a bad
/// point is reported as [`OptimizerError::InvalidConfig`] for
/// `initial_point` with nothing drawn or evaluated. `None` (no starting
/// point) always passes.
pub(crate) fn check_initial_point(
    point: Option<&[f64]>,
    dims: usize,
    bounds: Option<(f64, f64)>,
) -> Result<(), OptimizerError> {
    let Some(point) = point else {
        return Ok(());
    };
    let invalid = |reason: String| OptimizerError::InvalidConfig {
        parameter: "initial_point",
        reason,
    };
    if point.len() != dims {
        return Err(invalid(format!(
            "must have one value per dimension ({dims}), got {}",
            point.len()
        )));
    }
    if let Some((index, value)) = point.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(invalid(format!(
            "every value must be finite, got {value} at index {index}"
        )));
    }
    if let Some((lb, ub)) = bounds {
        if let Some((index, value)) = point
            .iter()
            .enumerate()
            .find(|(_, x)| !(lb..=ub).contains(*x))
        {
            return Err(invalid(format!(
                "every value must lie within bounds [{lb}, {ub}], got {value} at index {index}"
            )));
        }
    }
    Ok(())
}

/// The best candidate an oracle has measured, and the value it returned for
/// it, so that `best_fitness` is the oracle's value for `best_params`
/// (contract C-5).
///
/// The first measurement is always kept, so a run that only ever measures
/// `NaN` still reports a candidate it measured and the value it got. After it,
/// a candidate replaces the best only with a strictly greater value, so a tie
/// keeps the earlier one. A `NaN` never wins: it is greater than nothing, and
/// any number replaces a `NaN` kept from the first measurement. Empty, it
/// reports no parameters and `-∞`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct BestMeasured {
    best: Option<(Vec<f64>, f64)>,
}

impl BestMeasured {
    /// Nothing measured yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Offer `params`, which the oracle scored `value`; returns whether it is
    /// now the best.
    pub(crate) fn offer(&mut self, params: &[f64], value: f64) -> bool {
        let wins = match &self.best {
            None => true,
            Some((_, best)) => value > *best || (best.is_nan() && !value.is_nan()),
        };
        if wins {
            self.best = Some((params.to_vec(), value));
        }
        wins
    }

    /// The best value so far; `-∞` before any measurement.
    pub(crate) fn value(&self) -> f64 {
        self.best
            .as_ref()
            .map_or(f64::NEG_INFINITY, |(_, value)| *value)
    }

    /// The best parameters and their value; no parameters and `-∞` before any
    /// measurement.
    pub(crate) fn into_parts(self) -> (Vec<f64>, f64) {
        self.best.unwrap_or((Vec::new(), f64::NEG_INFINITY))
    }
}

/// `x mod 2π` in `[0, 2π)`. `rem_euclid` alone can return 2π itself for a
/// tiny negative `x` (the exact result rounds up); that is the same angle as 0.
pub(crate) fn wrap_angle(x: f64) -> f64 {
    let wrapped = x.rem_euclid(TAU);
    if wrapped >= TAU {
        0.0
    } else {
        wrapped
    }
}

/// The difference of two angles `x` as the shortest signed arc, in `(−π, π]`:
/// members at 0.05 and 6.23 rad are 0.1 rad apart, not 6.18.
pub(crate) fn wrap_difference(x: f64) -> f64 {
    let wrapped = wrap_angle(x);
    if wrapped > PI {
        wrapped - TAU
    } else {
        wrapped
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
    fn stagnated_matches_fitness_stagnated() {
        // The silent rule and DE's logging form agree on every window.
        let histories: [&[f64]; 4] = [
            &[0.0, 1.0, 2.0, 3.0],
            &[3.0, 3.005, 3.01, 3.015, 3.02],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[0.0, 5.0, 5.0005, 5.001],
        ];
        for history in histories {
            for patience in 0..6 {
                for tolerance in [0.0, 0.01, 0.5, 100.0] {
                    assert_eq!(
                        stagnated(history, tolerance, patience),
                        fitness_stagnated(history, tolerance, patience, history.len() - 1),
                        "{history:?} patience {patience} tolerance {tolerance}"
                    );
                }
            }
        }
    }

    #[test]
    fn recent_improvement_needs_a_full_window() {
        assert_eq!(recent_improvement(&[1.0, 2.0], 2), None);
        assert_eq!(recent_improvement(&[1.0, 2.0, 4.0], 2), Some(3.0));
        // patience 0 compares the last generation with itself.
        assert_eq!(recent_improvement(&[1.0], 0), Some(0.0));
        assert_eq!(recent_improvement(&[], 0), None);
    }

    #[test]
    fn wrap_difference_is_the_shortest_signed_arc() {
        assert_eq!(wrap_difference(0.0), 0.0);
        assert_eq!(wrap_difference(PI), PI);
        assert_eq!(wrap_difference(-PI), PI);
        assert_eq!(wrap_difference(0.5), 0.5);
        assert_eq!(wrap_difference(-0.5), -0.5);
        assert!((wrap_difference(6.23 - 0.05) - (6.23 - 0.05 - TAU)).abs() < 1e-15);
        assert!((wrap_difference(0.05 - 6.23) - (0.05 - 6.23 + TAU)).abs() < 1e-15);
        for x in [-10.0, -TAU, -4.0, -1.0, 1.0, 4.0, TAU, 10.0] {
            let d = wrap_difference(x);
            assert!(d > -PI && d <= PI, "{x} -> {d}");
        }
    }

    #[test]
    fn wrap_angle_stays_in_the_half_open_circle() {
        assert_eq!(wrap_angle(0.0), 0.0);
        assert_eq!(wrap_angle(-0.0), 0.0);
        assert_eq!(wrap_angle(TAU), 0.0);
        assert_eq!(wrap_angle(-TAU), 0.0);
        assert_eq!(wrap_angle(1.0 + TAU), (1.0 + TAU).rem_euclid(TAU));
        // rem_euclid rounds this up to 2π exactly; the wrap folds it to 0.
        assert_eq!((-1e-20f64).rem_euclid(TAU), TAU);
        assert_eq!(wrap_angle(-1e-20), 0.0);
        for x in [-1e9, -7.5, -1e-300, 3.0, 6.283_185_307_179_585, 1e12] {
            let w = wrap_angle(x);
            assert!((0.0..TAU).contains(&w), "{x} -> {w}");
        }
        assert!(wrap_angle(f64::NAN).is_nan());
    }

    #[test]
    fn check_initial_point_accepts_none_and_a_finite_point_of_the_right_length() {
        assert_eq!(check_initial_point(None, 3, None), Ok(()));
        assert_eq!(
            check_initial_point(Some(&[0.5, -7.0, 1e300]), 3, None),
            Ok(())
        );
        assert_eq!(check_initial_point(Some(&[]), 0, None), Ok(()));
    }

    #[test]
    fn check_initial_point_rejects_a_wrong_length() {
        for point in [&[0.5][..], &[0.5, 1.0, 1.5, 2.0][..], &[][..]] {
            assert_eq!(
                check_initial_point(Some(point), 3, None),
                Err(OptimizerError::InvalidConfig {
                    parameter: "initial_point",
                    reason: format!("must have one value per dimension (3), got {}", point.len()),
                })
            );
        }
    }

    #[test]
    fn check_initial_point_rejects_a_non_finite_value_naming_its_index() {
        for (value, shown) in [
            (f64::NAN, "NaN"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            let err = check_initial_point(Some(&[0.5, 1.0, value]), 3, None)
                .expect_err("a non-finite value is rejected");
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid initial_point: every value must be finite, got {shown} at index 2"
                )
            );
        }
    }

    #[test]
    fn best_measured_is_empty_until_the_first_offer() {
        let best = BestMeasured::new();
        assert_eq!(best.value(), f64::NEG_INFINITY);
        assert_eq!(best.into_parts(), (Vec::new(), f64::NEG_INFINITY));
    }

    #[test]
    fn best_measured_keeps_the_first_offer_then_only_strict_improvements() {
        let mut best = BestMeasured::new();
        assert!(best.offer(&[1.0], 0.5));
        assert!(best.offer(&[2.0], 0.75));
        // A tie keeps the earlier candidate; a worse one changes nothing.
        assert!(!best.offer(&[3.0], 0.75));
        assert!(!best.offer(&[4.0], -1.0));
        assert_eq!(best.value(), 0.75);
        assert_eq!(best.into_parts(), (vec![2.0], 0.75));
    }

    #[test]
    fn best_measured_never_lets_a_nan_win() {
        let mut best = BestMeasured::new();
        assert!(best.offer(&[1.0], f64::NEG_INFINITY));
        assert!(!best.offer(&[2.0], f64::NAN));
        assert_eq!(best.value(), f64::NEG_INFINITY);
        assert!(best.offer(&[3.0], 0.0));
        assert!(!best.offer(&[4.0], f64::NAN));
        assert_eq!(best.into_parts(), (vec![3.0], 0.0));
    }

    #[test]
    fn best_measured_keeps_a_first_nan_until_any_number_replaces_it() {
        let mut best = BestMeasured::new();
        // Only NaN measured: the first candidate and its NaN are what was
        // measured, so they are what is reported.
        assert!(best.offer(&[1.0], f64::NAN));
        assert!(!best.offer(&[2.0], f64::NAN));
        assert!(best.value().is_nan());
        let (params, value) = best.clone().into_parts();
        assert_eq!(params, vec![1.0]);
        assert!(value.is_nan());
        // Any number, even -inf, beats it.
        assert!(best.offer(&[3.0], f64::NEG_INFINITY));
        assert_eq!(best.into_parts(), (vec![3.0], f64::NEG_INFINITY));
    }

    #[test]
    fn check_initial_point_checks_the_bounds_only_when_given_and_after_the_rest() {
        let bounds = Some((-1.0, 1.0));
        // Both ends are inside: PSO clamps positions to the closed interval.
        assert_eq!(
            check_initial_point(Some(&[-1.0, 0.0, 1.0]), 3, bounds),
            Ok(())
        );
        let err = check_initial_point(Some(&[0.5, 1.5, -2.0]), 3, bounds)
            .expect_err("a value outside the bounds is rejected");
        assert_eq!(
            err.to_string(),
            "invalid initial_point: every value must lie within bounds [-1, 1], got 1.5 at index 1"
        );
        assert_eq!(
            check_initial_point(Some(&[0.5, 1.5, -2.0]), 3, None),
            Ok(())
        );
        // The length and the finiteness are reported first.
        let err = check_initial_point(Some(&[5.0]), 3, bounds).expect_err("wrong length");
        assert!(err.to_string().contains("one value per dimension"), "{err}");
        let err = check_initial_point(Some(&[5.0, f64::NAN, 0.0]), 3, bounds).expect_err("NaN");
        assert!(err.to_string().contains("must be finite"), "{err}");
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
