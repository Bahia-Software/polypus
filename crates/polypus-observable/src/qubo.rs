//! QUBO cost observable: a quadratic pseudo-Boolean function evaluated over
//! measurement bitstrings, diagonal in the computational basis.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{CostObservable, ObservableError};

/// A QUBO cost `f(x) = Σ_i linear[i]·x_i + Σ_(i,j) w·x_i·x_j + constant`,
/// multiplied by `scale`, with `x_i ∈ {0, 1}`.
///
/// The observable is diagonal: every basis state (bitstring) has a definite
/// value, so the expectation over measurement counts is the count-weighted mean
/// `Σ_s counts[s]·f(s) / Σ_s counts[s]`.
///
/// # Sign convention
///
/// Optimizers **maximise** the returned fitness. A minimisation QUBO (find the
/// lowest-cost bitstring) should be built with `scale = -1.0` so its optimum
/// becomes the maximum of the fitness.
///
/// # Bit ordering
///
/// A counts key of width `w` has the most-significant bit on the **left**, so
/// variable `i` is the character at byte position `w - 1 - i` (equivalently
/// `x_i = (state >> i) & 1`). This matches the native simulator's read-out and
/// Qiskit. Keys longer than `num_vars` are accepted; the extra high-order (left)
/// bits are simply not referenced. A key shorter than `num_vars` is a
/// [`ObservableError::BitWidthMismatch`].
#[derive(Debug, Clone)]
pub struct QuboObservable {
    num_vars: usize,
    /// Dense linear coefficients, `linear[i]` for variable `i`.
    linear: Vec<f64>,
    /// Sparse quadratic terms `(i, j, weight)` with `i < j`.
    quadratic: Vec<(u32, u32, f64)>,
    constant: f64,
    scale: f64,
}

impl QuboObservable {
    /// Build a QUBO from sparse coefficients.
    ///
    /// `linear` is a list of `(index, coefficient)`; `quadratic` a list of
    /// `(i, j, weight)`. Repeated entries accumulate. Validation (returned as
    /// [`ObservableError::Invalid`]) rejects: `num_vars == 0`, any index
    /// `>= num_vars`, a quadratic term with `i == j`, or any non-finite
    /// coefficient / `constant` / `scale`.
    pub fn new(
        num_vars: usize,
        linear: Vec<(usize, f64)>,
        quadratic: Vec<(usize, usize, f64)>,
        constant: f64,
        scale: f64,
    ) -> Result<Self, ObservableError> {
        if num_vars == 0 {
            return Err(ObservableError::Invalid("num_vars must be >= 1".to_string()));
        }
        if !constant.is_finite() {
            return Err(ObservableError::Invalid("constant must be finite".to_string()));
        }
        if !scale.is_finite() {
            return Err(ObservableError::Invalid("scale must be finite".to_string()));
        }

        let mut linear_dense = vec![0.0f64; num_vars];
        for (i, c) in linear {
            if i >= num_vars {
                return Err(ObservableError::Invalid(format!(
                    "linear index {i} out of range for {num_vars} variables"
                )));
            }
            if !c.is_finite() {
                return Err(ObservableError::Invalid(format!(
                    "linear coefficient for index {i} is not finite"
                )));
            }
            linear_dense[i] += c;
        }

        let mut quad = Vec::with_capacity(quadratic.len());
        for (i, j, w) in quadratic {
            if i >= num_vars || j >= num_vars {
                return Err(ObservableError::Invalid(format!(
                    "quadratic index ({i},{j}) out of range for {num_vars} variables"
                )));
            }
            if i == j {
                return Err(ObservableError::Invalid(format!(
                    "quadratic term on a single variable ({i},{j}) is not allowed; \
                     fold it into the linear part (x_i^2 = x_i)"
                )));
            }
            if !w.is_finite() {
                return Err(ObservableError::Invalid(format!(
                    "quadratic coefficient for ({i},{j}) is not finite"
                )));
            }
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            quad.push((lo as u32, hi as u32, w));
        }

        Ok(Self {
            num_vars,
            linear: linear_dense,
            quadratic: quad,
            constant,
            scale,
        })
    }

    /// Build a QUBO from a dense square matrix `Q`, evaluating `f(x) = xᵀ·Q·x`.
    ///
    /// The diagonal `Q_ii` folds into the linear part (`x_i² = x_i`); the
    /// off-diagonal pair `Q_ij + Q_ji` becomes the quadratic weight of `(i, j)`.
    /// `matrix` must be square and non-empty.
    pub fn from_matrix(matrix: &[Vec<f64>], scale: f64) -> Result<Self, ObservableError> {
        let n = matrix.len();
        if n == 0 {
            return Err(ObservableError::Invalid("matrix must be non-empty".to_string()));
        }
        for (r, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(ObservableError::Invalid(format!(
                    "matrix must be square; row {r} has length {} (expected {n})",
                    row.len()
                )));
            }
        }
        let mut linear = Vec::new();
        let mut quadratic = Vec::new();
        for i in 0..n {
            if matrix[i][i] != 0.0 {
                linear.push((i, matrix[i][i]));
            }
            for j in (i + 1)..n {
                let w = matrix[i][j] + matrix[j][i];
                if w != 0.0 {
                    quadratic.push((i, j, w));
                }
            }
        }
        // `new` performs finiteness validation (a NaN entry survives the `!= 0.0`
        // filter and is rejected there).
        Self::new(n, linear, quadratic, 0.0, scale)
    }

    /// Number of variables the observable is defined over.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Evaluate `scale·f(x)` for one bitstring key.
    #[inline]
    fn eval_bits(&self, key: &str) -> Result<f64, ObservableError> {
        let bytes = key.as_bytes();
        let w = bytes.len();
        if w < self.num_vars {
            return Err(ObservableError::BitWidthMismatch {
                num_vars: self.num_vars,
                key_len: w,
            });
        }

        let mut acc = self.constant;
        // Linear part: reads and validates every referenced bit (positions
        // `0..num_vars`). Quadratic indices are all `< num_vars`, so their bytes
        // are validated here too.
        for i in 0..self.num_vars {
            let xi = match bytes[w - 1 - i] {
                b'0' => 0.0,
                b'1' => 1.0,
                _ => return Err(ObservableError::InvalidBitstring(key.to_string())),
            };
            acc += self.linear[i] * xi;
        }
        // Quadratic part: bytes already validated above, so a plain `== b'1'`
        // read avoids allocating a decoded-bit buffer per key.
        for &(i, j, wq) in &self.quadratic {
            let xi = (bytes[w - 1 - i as usize] == b'1') as u32 as f64;
            let xj = (bytes[w - 1 - j as usize] == b'1') as u32 as f64;
            acc += wq * xi * xj;
        }
        Ok(acc * self.scale)
    }

    /// Count-weighted mean of `f` over one candidate's counts. Empty counts (or
    /// counts summing to zero) yield `0.0`.
    fn expectation_one(&self, counts: &HashMap<String, u64>) -> Result<f64, ObservableError> {
        let mut num = 0.0f64;
        let mut den = 0u64;
        for (key, &n) in counts {
            num += self.eval_bits(key)? * n as f64;
            den += n;
        }
        Ok(if den == 0 { 0.0 } else { num / den as f64 })
    }
}

impl CostObservable for QuboObservable {
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError> {
        // Embarrassingly parallel over candidates; `collect` short-circuits on
        // the first error. The optimizer already runs under `allow_threads`, so
        // this whole path holds no GIL.
        counts.par_iter().map(|c| self.expectation_one(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, n)| (k.to_string(), *n)).collect()
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(QuboObservable::new(0, vec![], vec![], 0.0, 1.0).is_err());
        assert!(QuboObservable::new(2, vec![(2, 1.0)], vec![], 0.0, 1.0).is_err()); // index oob
        assert!(QuboObservable::new(2, vec![], vec![(0, 0, 1.0)], 0.0, 1.0).is_err()); // i==j
        assert!(QuboObservable::new(2, vec![], vec![(0, 2, 1.0)], 0.0, 1.0).is_err()); // index oob
        assert!(QuboObservable::new(2, vec![(0, f64::NAN)], vec![], 0.0, 1.0).is_err());
        assert!(QuboObservable::new(2, vec![], vec![], f64::INFINITY, 1.0).is_err());
        assert!(QuboObservable::new(2, vec![], vec![], 0.0, f64::NAN).is_err());
    }

    #[test]
    fn bit_order_matches_convention() {
        // Linear on variable 0 only: it must read the RIGHT-most character.
        let obs = QuboObservable::new(2, vec![(0, 1.0)], vec![], 0.0, 1.0).unwrap();
        let ev = obs.expectation_batch(&[counts(&[("01", 10)])]).unwrap();
        assert_eq!(ev[0], 1.0); // "01" -> x_0 = 1
        let ev = obs.expectation_batch(&[counts(&[("10", 10)])]).unwrap();
        assert_eq!(ev[0], 0.0); // "10" -> x_0 = 0
    }

    #[test]
    fn maxcut_two_nodes() {
        // Cut value of edge (0,1): x0 + x1 - 2 x0 x1  (=1 iff the bits differ).
        let obs =
            QuboObservable::new(2, vec![(0, 1.0), (1, 1.0)], vec![(0, 1, -2.0)], 0.0, 1.0).unwrap();
        let ev = obs
            .expectation_batch(&[
                counts(&[("01", 1)]),
                counts(&[("10", 1)]),
                counts(&[("11", 1)]),
                counts(&[("00", 1)]),
            ])
            .unwrap();
        assert_eq!(ev, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn weighted_mean_and_scale() {
        // f = x0 (linear), scale = -1 -> fitness = -E[x0].
        let obs = QuboObservable::new(1, vec![(0, 1.0)], vec![], 0.0, -1.0).unwrap();
        // 3 shots read "1", 1 shot reads "0" -> E[x0] = 3/4 -> fitness = -0.75.
        let ev = obs
            .expectation_batch(&[counts(&[("1", 3), ("0", 1)])])
            .unwrap();
        assert!((ev[0] - (-0.75)).abs() < 1e-12);
    }

    #[test]
    fn empty_and_zero_counts_are_zero() {
        let obs = QuboObservable::new(2, vec![(0, 5.0)], vec![], 1.0, 1.0).unwrap();
        assert_eq!(obs.expectation_batch(&[HashMap::new()]).unwrap(), vec![0.0]);
        assert_eq!(
            obs.expectation_batch(&[counts(&[("11", 0)])]).unwrap(),
            vec![0.0]
        );
        assert_eq!(obs.expectation_batch(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn short_and_invalid_keys_error() {
        let obs = QuboObservable::new(3, vec![(2, 1.0)], vec![], 0.0, 1.0).unwrap();
        assert!(matches!(
            obs.expectation_batch(&[counts(&[("01", 1)])]),
            Err(ObservableError::BitWidthMismatch { num_vars: 3, key_len: 2 })
        ));
        assert!(matches!(
            obs.expectation_batch(&[counts(&[("0x1", 1)])]),
            Err(ObservableError::InvalidBitstring(_))
        ));
    }

    #[test]
    fn wide_keys_ignore_unreferenced_high_bits() {
        // num_vars = 2 but keys are 5 wide: the 3 left bits are ignored.
        let obs = QuboObservable::new(2, vec![(0, 1.0), (1, 1.0)], vec![], 0.0, 1.0).unwrap();
        let ev = obs.expectation_batch(&[counts(&[("11101", 1)])]).unwrap();
        assert_eq!(ev[0], 1.0); // x0 = 1 (rightmost), x1 = 0 -> 1
    }

    #[test]
    fn from_matrix_matches_sparse() {
        // Q = [[1, 2], [0, 3]] -> f = 1·x0 + 3·x1 + 2·x0·x1.
        let m = QuboObservable::from_matrix(&[vec![1.0, 2.0], vec![0.0, 3.0]], 1.0).unwrap();
        let s = QuboObservable::new(2, vec![(0, 1.0), (1, 3.0)], vec![(0, 1, 2.0)], 0.0, 1.0).unwrap();
        let batch = [counts(&[("11", 1)]), counts(&[("01", 1)]), counts(&[("10", 1)])];
        assert_eq!(
            m.expectation_batch(&batch).unwrap(),
            s.expectation_batch(&batch).unwrap()
        );
    }

    #[test]
    fn handles_more_than_64_variables() {
        // A 70-bit key with a linear term on the highest index exercises the
        // byte-indexed path (a u64 parse would overflow).
        let obs = QuboObservable::new(70, vec![(69, 1.0)], vec![], 0.0, 1.0).unwrap();
        let mut key = String::from("1");
        key.push_str(&"0".repeat(69)); // bit 69 set, rest clear
        let ev = obs.expectation_batch(&[counts(&[(key.as_str(), 1)])]).unwrap();
        assert_eq!(ev[0], 1.0);
    }
}
