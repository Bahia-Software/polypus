//! Internal helpers shared by the population-based optimizers.
//!
//! These are implementation details of [`AlgorithmDifferentialEvolution`](crate::AlgorithmDifferentialEvolution)
//! and [`AlgorithmPSO`](crate::AlgorithmPSO), not part of the crate's public
//! API, so the module is `pub(crate)`-scoped. Each helper is a small,
//! single-purpose free function that used to be inlined (and duplicated) inside
//! the optimizer loops. The behaviour is preserved exactly — same operation
//! order, same comparators, same diagnostics — so the determinism the tests pin
//! (`tests/optimizers.rs`) is unaffected.

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

/// Validate that an oracle returned exactly one fitness value per candidate.
///
/// Every optimizer calls [`EvaluationOracle::evaluate_batch`](crate::EvaluationOracle::evaluate_batch)
/// and then indexes the returned slice positionally; a short (or long) return
/// would otherwise panic with an out-of-bounds index deep inside the loop.
/// Checking the length immediately after each batch call — for *any* oracle,
/// Python-backed or not — turns that into the typed
/// [`OptimizerError::OracleLengthMismatch`] the FFI seam maps to a
/// `PyValueError`.
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
