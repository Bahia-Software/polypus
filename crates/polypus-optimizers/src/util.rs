//! Internal helpers shared by the population-based optimizers.
//!
//! These are implementation details of [`AlgorithmDifferentialEvolution`](crate::AlgorithmDifferentialEvolution)
//! and [`AlgorithmPSO`](crate::AlgorithmPSO), not part of the crate's public
//! API, so the module is `pub(crate)`-scoped. Each helper is a small,
//! single-purpose free function that used to be inlined (and duplicated) inside
//! the optimizer loops. The behaviour is preserved exactly — same operation
//! order, same comparators, same diagnostics — so the determinism the tests pin
//! (`tests/optimizers.rs`) is unaffected.

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
/// Returns `true` when the summed per-dimension standard deviation drops below
/// `tolerance * mean` (where `mean` is the summed per-dimension means), i.e. the
/// population has collapsed towards a point. Emits the same per-generation
/// `log::debug!` diagnostics the two loops previously inlined, including the
/// "stopping early" line when convergence fires. QNG has no population and does
/// not use this.
pub(crate) fn population_converged(
    population: &Array2<f64>,
    tolerance: f64,
    generation: usize,
) -> bool {
    let mean = population
        .mean_axis(Axis(0))
        .expect("Failed to compute mean")
        .sum();
    let std: f64 = population.std_axis(Axis(0), 0.0).iter().sum();
    log::debug!("Generation {generation}: Mean: {mean:.4}, Std: {std:.4}");
    let converged = std < tolerance * mean;
    if converged {
        log::debug!("Stopping early at generation {generation} due to convergence");
    }
    converged
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
        // Every row identical → std == 0 < tolerance * mean (mean > 0).
        let pop = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        assert!(population_converged(&pop, 0.5, 0));
    }

    #[test]
    fn population_converged_false_when_spread() {
        let pop = array![[0.0, 0.0], [10.0, 10.0]];
        assert!(!population_converged(&pop, 1e-9, 0));
    }

    #[test]
    fn population_converged_zero_dimensions_does_not_panic() {
        // mean/std sums are 0, so `0.0 < tolerance * 0.0` is false: no early
        // stop, no panic. Matches the zero-dimension DE behaviour.
        let pop = Array2::<f64>::zeros((3, 0));
        assert!(!population_converged(&pop, 1e-9, 0));
    }
}
