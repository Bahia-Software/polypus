//! The coefficient bound shared by the native reducers, which keeps every
//! value and every count-weighted mean they compute finite.

use crate::ObservableError;

/// The largest accepted `Σ_k |c_k| · |scale|` of a native reducer:
/// `f64::MAX / 2⁶⁵`, about `4.9e288`.
///
/// [`ParityObservable`](crate::ParityObservable),
/// [`QuboObservable`](crate::QuboObservable) and so
/// [`IsingObservable`](crate::IsingObservable) (checked on its QUBO form)
/// reject a larger sum at construction, and
/// [`group_qubit_wise`](crate::group_qubit_wise) rejects one over all terms.
/// It bounds every value the reducers compute, so none overflows:
///
/// - The value of one bitstring is a sum of terms `c_k` times `±1` (parity)
///   or times `0` or `1` (QUBO, whose constant counts as a term), times
///   `scale`. Without rounding it is at most `Σ_k |c_k| · |scale|`.
/// - A count-weighted sum weighs those values by counts whose total is at most
///   `u64::MAX < 2⁶⁴` (a larger total is an error in every reducer), and an
///   exact expectation weighs them by probabilities of at most `1`, at most
///   `2⁶⁴` of them. Without rounding, either sum is below `bound · 2⁶⁴`.
/// - Rounding raises that by a factor of at most `(1 + 2⁻⁵³)` per operation
///   (or `1 / (1 − 2⁻⁵³)` for the check's own sum, which can come out low),
///   except where a result is subnormal: there the error is absolute, at most
///   `2⁻¹⁰⁷⁵`, and cannot bring anything near `f64::MAX`.
///   With `N` coefficients and `k` summed keys or probabilities, there are at
///   most `2N + 1` roundings in a value (a QUBO first adds up the repeated
///   entries of each variable), `N + 1` in the check, and `k + 2` in the
///   weighted sum (each count to `f64`, each product, each addition). The
///   total factor, about `exp(2⁻⁵³ · (3N + k + 4))`, stays below `2` while
///   `N` and `k` are both below `2⁵⁰`: at 8 bytes or more each, 8 PiB of
///   memory. The extra factor `2` in the bound absorbs it, so both sums stay
///   below `bound · 2⁶⁵ = f64::MAX`.
pub const MAX_WEIGHT_BOUND: f64 = f64::MAX / 36_893_488_147_419_103_232.0;

/// `Σ |c|` over `coefficients`, summed in order, times `|scale|`; an error
/// unless it is at most [`MAX_WEIGHT_BOUND`] (a non-finite or `NaN` input
/// fails too).
pub(crate) fn check_weight_bound(
    coefficients: impl Iterator<Item = f64>,
    scale: f64,
) -> Result<(), ObservableError> {
    check_weight_bound_of("Σ|c|·|scale|", "", coefficients, scale)
}

/// [`check_weight_bound`], naming the sum `sum` in its error and ending the
/// error with `note`: for an observable that checks coefficients the caller
/// did not write, such as [`IsingObservable`](crate::IsingObservable)'s
/// expanded QUBO (issue #650).
pub(crate) fn check_weight_bound_of(
    sum: &str,
    note: &str,
    coefficients: impl Iterator<Item = f64>,
    scale: f64,
) -> Result<(), ObservableError> {
    let bound = coefficients.fold(0.0, |sum, c| sum + c.abs()) * scale.abs();
    if bound <= MAX_WEIGHT_BOUND {
        Ok(())
    } else {
        Err(ObservableError::Invalid(format!(
            "{sum} = {bound:e} exceeds {MAX_WEIGHT_BOUND:e}: a value or a \
             weighted mean of counts could overflow{note}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_is_f64_max_over_two_to_the_65() {
        // Exact: scaling by a power of two only changes the exponent.
        assert_eq!(MAX_WEIGHT_BOUND * 2f64.powi(65), f64::MAX);
    }

    #[test]
    fn the_check_is_inclusive_and_counts_every_input() {
        let b = MAX_WEIGHT_BOUND;
        let above = f64::from_bits(b.to_bits() + 1);
        assert!(check_weight_bound([b].into_iter(), 1.0).is_ok());
        assert!(check_weight_bound([-b].into_iter(), -1.0).is_ok());
        assert!(check_weight_bound([above].into_iter(), 1.0).is_err());
        // Exact: b/2 + b/2 = b; the scale multiplies it.
        assert!(check_weight_bound([b / 2.0, -b / 2.0].into_iter(), 1.0).is_ok());
        assert!(check_weight_bound([b / 2.0, -b / 2.0].into_iter(), 2.0).is_err());
        assert!(check_weight_bound(std::iter::empty(), f64::MAX).is_ok());
    }

    #[test]
    fn the_message_names_the_sum_and_ends_with_the_note() {
        let plain = check_weight_bound([f64::MAX].into_iter(), 1.0);
        assert!(
            matches!(&plain, Err(ObservableError::Invalid(m))
                if m.starts_with("Σ|c|·|scale| = ") && m.ends_with("could overflow")),
            "{plain:?}"
        );
        let named = check_weight_bound_of("the sum", " (a note)", [f64::MAX].into_iter(), 1.0);
        assert!(
            matches!(&named, Err(ObservableError::Invalid(m))
                if m.starts_with("the sum = ") && m.ends_with("could overflow (a note)")),
            "{named:?}"
        );
    }

    #[test]
    fn non_finite_inputs_fail_with_the_bound_in_the_message() {
        for (coefficients, scale) in [
            (vec![f64::NAN], 1.0),
            (vec![f64::INFINITY], 1.0),
            (vec![1e308, 1e308], 1.0),
            (vec![1.0], f64::NAN),
        ] {
            let result = check_weight_bound(coefficients.into_iter(), scale);
            assert!(
                matches!(&result, Err(ObservableError::Invalid(m)) if m.contains("exceeds")),
                "{result:?}"
            );
        }
    }
}
