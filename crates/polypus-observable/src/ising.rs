//! Ising cost observable: a spin-form quadratic cost, converted to an equivalent
//! [`QuboObservable`] at construction so a single evaluator serves the hot path.

use std::collections::HashMap;

use crate::bound::check_weight_bound_of;
use crate::{CostObservable, ObservableError, QuboObservable};

/// An Ising cost `f(s) = Σ_i fields[i]·z_i + Σ_(i,j) J·z_i·z_j + constant`,
/// multiplied by `scale`, with spins `z_i = 1 - 2·x_i ∈ {+1, -1}` (so bit
/// `x_i = 0 → z_i = +1`, `x_i = 1 → z_i = -1`).
///
/// Sign convention and bit ordering are those of [`QuboObservable`] (optimizers
/// maximise; use `scale = -1.0` for a minimisation problem).
#[derive(Debug, Clone)]
pub struct IsingObservable {
    qubo: QuboObservable,
}

/// How an expansion error ends: an overflowing coefficient means the QUBO
/// cannot be stored, even when `scale` would keep `Σ|c|·|scale|` small.
const UNBUILDABLE: &str = ", so the expanded QUBO cannot be built";

impl IsingObservable {
    /// Build an Ising cost from sparse fields and couplings.
    ///
    /// `fields` is a list of `(index, h)`; `couplings` a list of `(i, j, J)`.
    /// The expansion `z_i = 1 - 2·x_i` gives, per term:
    /// - `h·z_i = h - 2h·x_i`
    /// - `J·z_i·z_j = J - 2J·x_i - 2J·x_j + 4J·x_i·x_j` (for `i ≠ j`)
    /// - `J·z_i·z_i = J` (a self-coupling folds entirely into the constant,
    ///   since `z_i² = 1`).
    ///
    /// Validation mirrors [`QuboObservable::new`] (index range, finiteness),
    /// and its bound on the sum of coefficients
    /// ([`MAX_WEIGHT_BOUND`](crate::MAX_WEIGHT_BOUND)) applies to the expanded
    /// QUBO, whose coefficients add up to as much as 9 times the Ising ones
    /// (a coupling `J` becomes `J`, `-2J`, `-2J` and `4J`).
    ///
    /// The errors speak of the caller's input (issue #650). The input's own
    /// errors come first: `num_vars`, `constant`, `scale`, then each field
    /// and coupling in order (index, finiteness). Then the expansion's: a
    /// finite field or coupling whose expanded coefficient (`-2h`, `4J`)
    /// overflows `f64` is named, then a constant that overflows once the
    /// fields and couplings are folded into it, and last a sum above the
    /// bound, reported as the expanded QUBO's.
    pub fn new(
        num_vars: usize,
        fields: Vec<(usize, f64)>,
        couplings: Vec<(usize, usize, f64)>,
        constant: f64,
        scale: f64,
    ) -> Result<Self, ObservableError> {
        if num_vars == 0 {
            return Err(ObservableError::Invalid(
                "num_vars must be >= 1".to_string(),
            ));
        }
        if !constant.is_finite() {
            return Err(ObservableError::Invalid(
                "constant must be finite".to_string(),
            ));
        }
        if !scale.is_finite() {
            return Err(ObservableError::Invalid("scale must be finite".to_string()));
        }

        let mut q_constant = constant;
        let mut q_linear: Vec<(usize, f64)> =
            Vec::with_capacity(fields.len() + 2 * couplings.len());
        let mut q_quadratic: Vec<(usize, usize, f64)> = Vec::with_capacity(couplings.len());
        // The first term whose expansion overflows, reported once every term
        // of the input has been validated.
        let mut overflow: Option<String> = None;

        for (i, h) in fields {
            if i >= num_vars {
                return Err(ObservableError::Invalid(format!(
                    "field index {i} out of range for {num_vars} variables"
                )));
            }
            if !h.is_finite() {
                return Err(ObservableError::Invalid(format!(
                    "field coefficient for index {i} is not finite"
                )));
            }
            let linear = -2.0 * h;
            if !linear.is_finite() && overflow.is_none() {
                overflow = Some(format!(
                    "field coefficient for index {i} = {h:e} is too large: its \
                     expanded QUBO coefficient -2h overflows f64{UNBUILDABLE}"
                ));
            }
            q_constant += h;
            q_linear.push((i, linear));
        }

        for (i, j, jc) in couplings {
            if i >= num_vars || j >= num_vars {
                return Err(ObservableError::Invalid(format!(
                    "coupling index ({i},{j}) out of range for {num_vars} variables"
                )));
            }
            if !jc.is_finite() {
                return Err(ObservableError::Invalid(format!(
                    "coupling coefficient for ({i},{j}) is not finite"
                )));
            }
            if i == j {
                // z_i² = 1: a self-coupling is a pure constant.
                q_constant += jc;
                continue;
            }
            // `4J` overflows whenever `-2J` does, so it is the one named.
            let quadratic = 4.0 * jc;
            if !quadratic.is_finite() && overflow.is_none() {
                overflow = Some(format!(
                    "coupling coefficient for ({i},{j}) = {jc:e} is too large: its \
                     expanded QUBO coefficient 4J overflows f64{UNBUILDABLE}"
                ));
            }
            q_constant += jc;
            q_linear.push((i, -2.0 * jc));
            q_linear.push((j, -2.0 * jc));
            q_quadratic.push((i, j, quadratic));
        }

        if let Some(term) = overflow {
            return Err(ObservableError::Invalid(term));
        }
        if !q_constant.is_finite() {
            // Every term is finite, so the running sum overflowed (it cannot
            // come back from `inf`).
            return Err(ObservableError::Invalid(format!(
                "the expanded QUBO's constant (constant + Σ fields + Σ \
                 couplings) overflows f64{UNBUILDABLE}"
            )));
        }
        // The check `QuboObservable::new` runs, over the same coefficients in
        // the same order, so the QUBO's own never fails after it.
        check_weight_bound_of(
            "the expanded QUBO's Σ|c|·|scale|",
            " (each field h adds up to 3·|h| to it, and each coupling J up to 9·|J|)",
            std::iter::once(q_constant)
                .chain(q_linear.iter().map(|&(_, c)| c))
                .chain(q_quadratic.iter().map(|&(_, _, w)| w)),
            scale,
        )?;

        let qubo = QuboObservable::new(num_vars, q_linear, q_quadratic, q_constant, scale)?;
        Ok(Self { qubo })
    }

    /// Number of variables the observable is defined over.
    pub fn num_vars(&self) -> usize {
        self.qubo.num_vars()
    }
}

impl CostObservable for IsingObservable {
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError> {
        self.qubo.expectation_batch(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_WEIGHT_BOUND;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, n)| (k.to_string(), *n)).collect()
    }

    #[test]
    fn single_field_spin_signs() {
        // f = h·z_0, h = 2. x0=0 ("0") -> z0=+1 -> +2 ; x0=1 ("1") -> z0=-1 -> -2.
        let obs = IsingObservable::new(1, vec![(0, 2.0)], vec![], 0.0, 1.0).unwrap();
        let ev = obs
            .expectation_batch(&[counts(&[("0", 1)]), counts(&[("1", 1)])])
            .unwrap();
        assert_eq!(ev, vec![2.0, -2.0]);
    }

    #[test]
    fn zz_coupling() {
        // f = J·z_0·z_1, J = 1. Aligned spins -> +1, anti-aligned -> -1.
        let obs = IsingObservable::new(2, vec![], vec![(0, 1, 1.0)], 0.0, 1.0).unwrap();
        let ev = obs
            .expectation_batch(&[
                counts(&[("00", 1)]), // z0=z1=+1 -> +1
                counts(&[("01", 1)]), // z0=-1, z1=+1 -> -1
                counts(&[("11", 1)]), // z0=z1=-1 -> +1
            ])
            .unwrap();
        assert_eq!(ev, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn self_coupling_is_constant() {
        // J·z_0·z_0 = J regardless of the bit.
        let obs = IsingObservable::new(1, vec![], vec![(0, 0, 3.0)], 0.0, 1.0).unwrap();
        let ev = obs
            .expectation_batch(&[counts(&[("0", 1)]), counts(&[("1", 1)])])
            .unwrap();
        assert_eq!(ev, vec![3.0, 3.0]);
    }

    #[test]
    fn matches_hand_derived_qubo() {
        // Ising: h0=1 on var 0, J=0.5 on (0,1), constant 0.25.
        // z_i = 1-2x_i. Hand expansion:
        //   constant_q = 0.25 + 1 + 0.5 = 1.75
        //   linear_q[0] = -2*1 + -2*0.5 = -3 ; linear_q[1] = -2*0.5 = -1
        //   quad_q(0,1) = 4*0.5 = 2
        let ising = IsingObservable::new(2, vec![(0, 1.0)], vec![(0, 1, 0.5)], 0.25, 1.0).unwrap();
        let qubo = QuboObservable::new(2, vec![(0, -3.0), (1, -1.0)], vec![(0, 1, 2.0)], 1.75, 1.0)
            .unwrap();
        let batch = [
            counts(&[("00", 3), ("11", 1)]),
            counts(&[("01", 2), ("10", 5)]),
        ];
        let a = ising.expectation_batch(&batch).unwrap();
        let b = qubo.expectation_batch(&batch).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12, "{x} != {y}");
        }
    }

    #[test]
    fn mean_does_not_depend_on_hashmap_iteration_order() {
        // "001" -> 0.8, "010" -> 0.6, "100" -> -0.4 up to rounding, so the sum
        // depends on the order its terms are added in. Four shots make the division
        // by the shot count exact, so that difference reaches the mean.
        let ising =
            IsingObservable::new(3, vec![(0, 0.1), (1, 0.2), (2, 0.7)], vec![], 0.0, 1.0).unwrap();
        let value = |key: &str| ising.expectation_batch(&[counts(&[(key, 1)])]).unwrap()[0];
        let (a, b, c) = (value("001"), value("010"), value("100"));
        let ascending = ((a + b) + c * 2.0) / 4.0;
        assert_ne!(ascending, ((b + c * 2.0) + a) / 4.0);
        // Each map gets its own `RandomState`, hence its own iteration order.
        let batch: Vec<_> = (0..128)
            .map(|_| counts(&[("001", 1), ("010", 1), ("100", 2)]))
            .collect();
        for ev in ising.expectation_batch(&batch).unwrap() {
            assert_eq!(ev, ascending, "summed in ascending bitstring order");
        }
    }

    #[test]
    fn num_vars_is_the_declared_width() {
        for n in [2, 3, 70] {
            let obs = IsingObservable::new(n, vec![], vec![], 0.0, 1.0).unwrap();
            assert_eq!(obs.num_vars(), n);
        }
    }

    #[test]
    fn a_count_total_above_u64_is_an_error_not_a_panic() {
        // f = z_0: "00" -> +1, "01" -> -1. Each count fits in a u64, but the
        // total does not.
        let obs = IsingObservable::new(2, vec![(0, 1.0)], vec![], 0.0, 1.0).unwrap();
        let result = obs.expectation_batch(&[counts(&[("00", u64::MAX), ("01", 1)])]);
        assert!(
            matches!(&result, Err(ObservableError::Invalid(m)) if m.contains("overflows u64")),
            "{result:?}"
        );
        // Exactly u64::MAX in total is still evaluated. Exact: u64::MAX - 1
        // converts to 2⁶⁴, the numerator 2⁶⁴ - 1 rounds to 2⁶⁴, and so does the
        // total, so the mean is 1.
        let ev = obs
            .expectation_batch(&[counts(&[("00", u64::MAX - 1), ("01", 1)])])
            .unwrap();
        assert_eq!(ev, vec![1.0]);
    }

    #[test]
    fn the_weight_bound_applies_to_the_expanded_qubo() {
        // h·z_0 expands to h - 2h·x_0, whose coefficients add up to 3|h|: about
        // 3e288 is within MAX_WEIGHT_BOUND (about 4.9e288), 6e288 is not.
        assert!(IsingObservable::new(1, vec![(0, 1e288)], vec![], 0.0, 1.0).is_ok());
        let result = IsingObservable::new(1, vec![(0, 2e288)], vec![], 0.0, 1.0);
        assert!(
            matches!(&result, Err(ObservableError::Invalid(m)) if m.contains("exceeds")),
            "{result:?}"
        );
        // Accepted before #625, and then evaluated to NaN or -inf: a field and
        // a coupling on variable 0 expand to two finite linear entries,
        // -1.6e308 and -8e307, whose sum is not.
        let result = IsingObservable::new(2, vec![(0, 8e307)], vec![(0, 1, 4e307)], 0.0, 1.0);
        assert!(
            matches!(&result, Err(ObservableError::Invalid(m)) if m.contains("exceeds")),
            "{result:?}"
        );
    }

    #[test]
    fn the_weight_bound_is_inclusive_on_the_expanded_qubo() {
        let b = MAX_WEIGHT_BOUND;
        // f = -b/2 + (b/2)·z_0 expands to the QUBO 0 - b·x_0 (exactly: the
        // constant is -b/2 + b/2), whose coefficients add up to exactly b.
        let obs = IsingObservable::new(1, vec![(0, b / 2.0)], vec![], -b / 2.0, 1.0).unwrap();
        // And it stays finite at the bound. Exact: -b times a count that
        // rounds to 2⁶⁴, divided by the same count.
        assert_eq!(
            obs.expectation_batch(&[counts(&[("1", u64::MAX)])])
                .unwrap(),
            vec![-b]
        );
        // The next field above b/2 exceeds it.
        let above = f64::from_bits((b / 2.0).to_bits() + 1);
        let result = IsingObservable::new(1, vec![(0, above)], vec![], -b / 2.0, 1.0);
        assert!(
            matches!(&result, Err(ObservableError::Invalid(m)) if m.contains("exceeds")),
            "{result:?}"
        );
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(IsingObservable::new(0, vec![], vec![], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![(2, 1.0)], vec![], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![], vec![(0, 2, 1.0)], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![(0, f64::NAN)], vec![], 0.0, 1.0).is_err());
    }

    /// The message of an `Invalid` error, or a panic naming what came back.
    fn message(result: Result<IsingObservable, ObservableError>) -> String {
        match result {
            Err(ObservableError::Invalid(m)) => m,
            other => panic!("expected ObservableError::Invalid, got {other:?}"),
        }
    }

    // Issue #650: the expansion's errors name the caller's term, or speak of
    // the expanded QUBO, instead of naming a QUBO coefficient.
    #[test]
    fn a_field_whose_expansion_overflows_is_named() {
        // -2h overflows while h and the folded constant stay finite: the
        // field is named, not the bound.
        let m = message(IsingObservable::new(2, vec![(1, 1e308)], vec![], 0.0, 1.0));
        assert_eq!(
            m,
            "field coefficient for index 1 = 1e308 is too large: its expanded QUBO \
             coefficient -2h overflows f64, so the expanded QUBO cannot be built"
        );
        // The largest field whose -2h is finite gets the bound's message.
        let m = message(IsingObservable::new(
            1,
            vec![(0, f64::MAX / 2.0)],
            vec![],
            0.0,
            1.0,
        ));
        assert!(m.starts_with("the expanded QUBO's Σ|c|·|scale| = "), "{m}");
    }

    #[test]
    fn a_coupling_whose_expansion_overflows_is_named() {
        // 4J overflows while -2J does not, and both do.
        for jc in [5e307, -1e308] {
            let m = message(IsingObservable::new(3, vec![], vec![(2, 0, jc)], 0.0, 1.0));
            assert_eq!(
                m,
                format!(
                    "coupling coefficient for (2,0) = {jc:e} is too large: its expanded \
                     QUBO coefficient 4J overflows f64, so the expanded QUBO cannot be built"
                )
            );
        }
        // The largest coupling whose 4J is finite gets the bound's message.
        let m = message(IsingObservable::new(
            2,
            vec![],
            vec![(0, 1, f64::MAX / 4.0)],
            0.0,
            1.0,
        ));
        assert!(m.starts_with("the expanded QUBO's Σ|c|·|scale| = "), "{m}");
        // A self-coupling only adds to the constant, so it never overflows by
        // itself.
        assert!(message(IsingObservable::new(
            1,
            vec![],
            vec![(0, 0, 1e308)],
            0.0,
            1.0
        ))
        .starts_with("the expanded QUBO's Σ|c|·|scale| = "));
    }

    #[test]
    fn a_constant_that_overflows_once_folded_is_named() {
        let expected = "the expanded QUBO's constant (constant + Σ fields + Σ couplings) \
                        overflows f64, so the expanded QUBO cannot be built";
        // Through a field (whose -2h is finite), a coupling (whose 4J is
        // finite) and a self-coupling, upwards and downwards.
        for (fields, couplings, constant) in [
            (vec![(0, 8e307)], vec![], 1.7e308),
            (vec![], vec![(0, 1, 4e307)], 1.7e308),
            (vec![], vec![(1, 1, 1e308)], 1.7e308),
            (vec![(0, -8e307)], vec![(1, 1, -1e308)], -1e308),
        ] {
            let m = message(IsingObservable::new(2, fields, couplings, constant, 1.0));
            assert_eq!(m, expected);
        }
    }

    #[test]
    fn an_overflow_is_not_reported_as_a_sum_above_the_bound() {
        // With |scale| < 1 (or 0) the true Σ|c|·|scale| of an overflowing
        // expansion can be small, so its error says only that the QUBO cannot
        // be built; it was rejected before #650 too.
        for scale in [1e-300, 0.0] {
            for m in [
                message(IsingObservable::new(
                    2,
                    vec![],
                    vec![(0, 1, 1e308)],
                    0.0,
                    scale,
                )),
                message(IsingObservable::new(
                    1,
                    vec![(0, 1e308)],
                    vec![],
                    0.0,
                    scale,
                )),
                message(IsingObservable::new(
                    1,
                    vec![(0, 8e307)],
                    vec![],
                    1.7e308,
                    scale,
                )),
            ] {
                assert!(m.ends_with("so the expanded QUBO cannot be built"), "{m}");
                assert!(!m.contains("exceeds"), "{m}");
            }
        }
    }

    #[test]
    fn the_bound_is_reported_as_the_expanded_qubo_sum() {
        // 3·|h| = 6e288 is above MAX_WEIGHT_BOUND (about 4.9e288) though
        // |h| = 2e288 is not.
        let m = message(IsingObservable::new(1, vec![(0, 2e288)], vec![], 0.0, 1.0));
        assert_eq!(
            m,
            format!(
                "the expanded QUBO's Σ|c|·|scale| = 6e288 exceeds {MAX_WEIGHT_BOUND:e}: a \
                 value or a weighted mean of counts could overflow (each field h adds up \
                 to 3·|h| to it, and each coupling J up to 9·|J|)"
            )
        );
    }

    #[test]
    fn the_input_errors_come_before_the_expansion_errors() {
        let overflowing = vec![(0, 1e308)];
        // A later field or coupling out of range, or not finite: the Ising
        // messages, not the QUBO's "linear …" ones.
        assert_eq!(
            message(IsingObservable::new(
                2,
                vec![(0, 1e308), (2, 1.0)],
                vec![],
                0.0,
                1.0
            )),
            "field index 2 out of range for 2 variables"
        );
        assert_eq!(
            message(IsingObservable::new(
                2,
                overflowing.clone(),
                vec![(0, 1, f64::NAN)],
                0.0,
                1.0
            )),
            "coupling coefficient for (0,1) is not finite"
        );
        // A scale that is not finite, which the bound check would otherwise
        // report as an exceeded sum.
        for scale in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                message(IsingObservable::new(
                    2,
                    overflowing.clone(),
                    vec![],
                    0.0,
                    scale
                )),
                "scale must be finite"
            );
            assert_eq!(
                message(IsingObservable::new(2, vec![], vec![], 0.0, scale)),
                "scale must be finite"
            );
        }
        // The first overflowing term is named: fields before couplings, each
        // in order.
        let m = message(IsingObservable::new(
            3,
            vec![(2, 1e308), (1, 1e308)],
            vec![(0, 1, 1e308)],
            0.0,
            1.0,
        ));
        assert_eq!(
            m,
            "field coefficient for index 2 = 1e308 is too large: its expanded QUBO \
             coefficient -2h overflows f64, so the expanded QUBO cannot be built"
        );
        // A term's overflow before the constant's.
        let m = message(IsingObservable::new(
            2,
            vec![(0, 8e307)],
            vec![(0, 1, 1e308)],
            1.7e308,
            1.0,
        ));
        assert_eq!(
            m,
            "coupling coefficient for (0,1) = 1e308 is too large: its expanded QUBO \
             coefficient 4J overflows f64, so the expanded QUBO cannot be built"
        );
    }
}
