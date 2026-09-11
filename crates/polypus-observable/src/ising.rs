//! Ising cost observable: a spin-form quadratic cost, converted to an equivalent
//! [`QuboObservable`] at construction so a single evaluator serves the hot path.

use std::collections::HashMap;

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
    /// Validation mirrors [`QuboObservable::new`] (index range, finiteness).
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

        let mut q_constant = constant;
        let mut q_linear: Vec<(usize, f64)> =
            Vec::with_capacity(fields.len() + 2 * couplings.len());
        let mut q_quadratic: Vec<(usize, usize, f64)> = Vec::with_capacity(couplings.len());

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
            q_constant += h;
            q_linear.push((i, -2.0 * h));
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
            q_constant += jc;
            q_linear.push((i, -2.0 * jc));
            q_linear.push((j, -2.0 * jc));
            q_quadratic.push((i, j, 4.0 * jc));
        }

        // Delegate finiteness of the derived coefficients + storage to QUBO.
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
    fn rejects_bad_construction() {
        assert!(IsingObservable::new(0, vec![], vec![], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![(2, 1.0)], vec![], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![], vec![(0, 2, 1.0)], 0.0, 1.0).is_err());
        assert!(IsingObservable::new(2, vec![(0, f64::NAN)], vec![], 0.0, 1.0).is_err());
    }
}
