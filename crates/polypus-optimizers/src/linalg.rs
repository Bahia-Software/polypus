//! Dense symmetric eigendecomposition, for [`AlgorithmCmaEs`](crate::AlgorithmCmaEs).
//!
//! CMA-ES samples from `N(m, σ²C)` through the eigendecomposition
//! `C = B·diag(λ)·Bᵀ`, and whitens steps with `C^(-1/2) = B·diag(λ^(-1/2))·Bᵀ`.
//! The crate has no linear-algebra dependency (no `nalgebra`, `ndarray-linalg`
//! or `faer`) and the matrices are small (one row per circuit parameter, and
//! CMA-ES recomputes the decomposition only every few generations), so a
//! cyclic Jacobi solver is enough: deterministic, no `unsafe`, and a few dozen
//! lines.

use ndarray::{Array1, Array2};

/// Eigenvalues and orthonormal eigenvectors of a symmetric matrix.
#[derive(Debug, Clone)]
pub(crate) struct SymmetricEigen {
    /// The eigenvalues, in no particular order.
    pub(crate) values: Array1<f64>,
    /// Column `j` is a unit eigenvector for `values[j]`; the columns are
    /// orthonormal.
    pub(crate) vectors: Array2<f64>,
}

/// Sweeps after which [`symmetric_eigen`] gives up. Cyclic Jacobi converges
/// quadratically once the off-diagonal part is small, and a finite symmetric
/// matrix converges in well under 20 sweeps; the cap only bounds the loop.
const MAX_SWEEPS: usize = 64;

/// Eigendecomposition of the symmetric matrix `a` by the cyclic Jacobi method
/// (Golub and Van Loan, *Matrix Computations*, §8.5): sweep over every
/// off-diagonal pair `(p, q)`, row by row, and apply the plane rotation that
/// zeroes `a[p][q]`, until the off-diagonal Frobenius norm is at most
/// `f64::EPSILON` times the norm of the whole matrix.
///
/// The caller passes an exactly symmetric matrix; each rotation writes both
/// triangles with the same values, so it stays exactly symmetric throughout. Returns `None` when `a`
/// is not square, holds a non-finite entry, or (which a finite symmetric
/// matrix does not do in practice) has not converged after `MAX_SWEEPS`
/// sweeps.
pub(crate) fn symmetric_eigen(a: &Array2<f64>) -> Option<SymmetricEigen> {
    let n = a.nrows();
    if a.ncols() != n || a.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut a = a.clone();
    let mut v = Array2::<f64>::eye(n);
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _ in 0..MAX_SWEEPS {
        if off_diagonal_norm(&a) <= f64::EPSILON * norm {
            return Some(SymmetricEigen {
                values: a.diag().to_owned(),
                vectors: v,
            });
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[[p, q]] != 0.0 {
                    rotate(&mut a, &mut v, p, q);
                }
            }
        }
    }
    None
}

/// `sqrt(Σ_{i≠j} a[i][j]²)`.
fn off_diagonal_norm(a: &Array2<f64>) -> f64 {
    a.indexed_iter()
        .filter(|((i, j), _)| i != j)
        .map(|(_, x)| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Apply the Jacobi rotation `J` that zeroes `a[p][q]` (`p < q`):
/// `a ← Jᵀ·a·J` and `v ← v·J`, with `J` the identity except
/// `J[p][p] = J[q][q] = c`, `J[p][q] = s` and `J[q][p] = -s`.
///
/// `t = tan θ` is the smaller root of `t² + 2τt − 1 = 0`, `τ = (a_qq − a_pp) /
/// (2 a_pq)`, so `|θ| ≤ π/4` (Golub and Van Loan's `sym.schur2`); `hypot`
/// keeps `τ²` from overflowing when `a_pq` is tiny, where `t` simply becomes 0.
/// The two diagonal entries take the closed forms `a_pp − t·a_pq` and
/// `a_qq + t·a_pq`, more accurate than the general update, and the rotated
/// pair is stored as an exact zero.
fn rotate(a: &mut Array2<f64>, v: &mut Array2<f64>, p: usize, q: usize) {
    let n = a.nrows();
    let apq = a[[p, q]];
    let tau = (a[[q, q]] - a[[p, p]]) / (2.0 * apq);
    let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    for k in (0..n).filter(|&k| k != p && k != q) {
        let (akp, akq) = (a[[k, p]], a[[k, q]]);
        let new_kp = c * akp - s * akq;
        let new_kq = s * akp + c * akq;
        a[[k, p]] = new_kp;
        a[[p, k]] = new_kp;
        a[[k, q]] = new_kq;
        a[[q, k]] = new_kq;
    }
    a[[p, p]] -= t * apq;
    a[[q, q]] += t * apq;
    a[[p, q]] = 0.0;
    a[[q, p]] = 0.0;
    for k in 0..n {
        let (vkp, vkq) = (v[[k, p]], v[[k, q]]);
        v[[k, p]] = c * vkp - s * vkq;
        v[[k, q]] = s * vkp + c * vkq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// `B·diag(values)·Bᵀ`, the matrix a decomposition describes.
    fn reconstruct(eig: &SymmetricEigen) -> Array2<f64> {
        let scaled = &eig.vectors * &eig.values;
        scaled.dot(&eig.vectors.t())
    }

    fn sorted(values: &Array1<f64>) -> Vec<f64> {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        v
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    /// `‖BᵀB − I‖_max`: how far the eigenvectors are from orthonormal.
    fn orthonormality_error(b: &Array2<f64>) -> f64 {
        let n = b.nrows();
        max_abs_diff(&b.t().dot(b), &Array2::eye(n))
    }

    /// A random symmetric positive definite matrix `Q·diag(d)·Qᵀ` with known
    /// eigenvalues `d`, condition number `10^decades`.
    fn random_spd(n: usize, decades: f64, rng: &mut StdRng) -> (Array2<f64>, Vec<f64>) {
        // Orthonormalise a random matrix by modified Gram-Schmidt.
        let mut q = Array2::<f64>::from_shape_fn((n, n), |_| rng.random_range(-1.0..1.0));
        for j in 0..n {
            for k in 0..j {
                let dot = q.column(j).dot(&q.column(k));
                let prev = q.column(k).to_owned();
                q.column_mut(j).scaled_add(-dot, &prev);
            }
            let norm = q.column(j).dot(&q.column(j)).sqrt();
            q.column_mut(j).mapv_inplace(|x| x / norm);
        }
        let d: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(decades * i as f64 / (n.max(2) - 1) as f64))
            .collect();
        let m = (&q * &Array1::from(d.clone())).dot(&q.t());
        // Symmetrise exactly, as CMA-ES's covariance is.
        let m = (&m + &m.t()) / 2.0;
        (m, d)
    }

    #[test]
    fn diagonal_matrix_is_its_own_decomposition() {
        let a = array![[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.5]];
        let eig = symmetric_eigen(&a).expect("finite symmetric input");
        assert_eq!(eig.values.to_vec(), vec![3.0, -1.0, 0.5]);
        assert_eq!(eig.vectors, Array2::<f64>::eye(3));
    }

    #[test]
    fn two_by_two_matches_the_analytic_solution() {
        // [[a, b], [b, c]] has eigenvalues (a+c)/2 ± sqrt(((a-c)/2)² + b²).
        let (a, b, c) = (2.0, 1.5, -0.5);
        let m = array![[a, b], [b, c]];
        let eig = symmetric_eigen(&m).expect("finite symmetric input");
        let mid = (a + c) / 2.0;
        let rad = (((a - c) / 2.0) * ((a - c) / 2.0) + b * b).sqrt();
        let got = sorted(&eig.values);
        assert!((got[0] - (mid - rad)).abs() < 1e-15, "{got:?}");
        assert!((got[1] - (mid + rad)).abs() < 1e-15, "{got:?}");
        // Each column is an eigenvector: M·v = λ·v.
        for j in 0..2 {
            let col = eig.vectors.column(j);
            let residual = &m.dot(&col) - &(&col * eig.values[j]);
            assert!(residual.iter().all(|r| r.abs() < 1e-14), "{residual}");
        }
        assert!(orthonormality_error(&eig.vectors) < 1e-15);
    }

    #[test]
    fn two_by_two_with_equal_diagonal_rotates_by_forty_five_degrees() {
        // τ = 0: the eigenvectors are (1, ±1)/√2 and the eigenvalues 1 ± 2.
        let m = array![[1.0, 2.0], [2.0, 1.0]];
        let eig = symmetric_eigen(&m).expect("finite symmetric input");
        assert_eq!(sorted(&eig.values), vec![-1.0, 3.0]);
        let r = std::f64::consts::FRAC_1_SQRT_2;
        for x in eig.vectors.iter() {
            assert!((x.abs() - r).abs() < 1e-15, "{}", eig.vectors);
        }
    }

    #[test]
    fn random_spd_matrices_reconstruct_within_tolerance() {
        let mut rng = StdRng::seed_from_u64(7);
        for n in [1, 2, 3, 5, 8, 13, 30] {
            for decades in [0.0, 3.0, 8.0, 12.0] {
                let (m, d) = random_spd(n, decades, &mut rng);
                let eig = symmetric_eigen(&m).expect("finite symmetric input");
                let scale = d.iter().fold(0.0_f64, |a, &b| a.max(b));
                // C = B D² Bᵀ, with D² the eigenvalues.
                let err = max_abs_diff(&reconstruct(&eig), &m);
                assert!(err <= 1e-13 * scale, "n={n} 10^{decades}: error {err}");
                assert!(orthonormality_error(&eig.vectors) < 1e-13, "n={n}");
                // The eigenvalues are the ones the matrix was built from, to
                // absolute accuracy ε·‖C‖ (Jacobi's accuracy).
                let mut want = d.clone();
                want.sort_by(f64::total_cmp);
                for (g, w) in sorted(&eig.values).iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-13 * scale, "n={n}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn indefinite_and_singular_matrices_decompose() {
        // Jacobi does not need definiteness: a rank-one matrix u·uᵀ has the
        // eigenvalues ‖u‖² and 0 (n−1 times).
        let u = array![1.0, -2.0, 0.5, 3.0];
        let m = Array2::from_shape_fn((4, 4), |(i, j)| u[i] * u[j]);
        let eig = symmetric_eigen(&m).expect("finite symmetric input");
        let got = sorted(&eig.values);
        assert!((got[3] - u.dot(&u)).abs() < 1e-13, "{got:?}");
        assert!(got[..3].iter().all(|x| x.abs() < 1e-13), "{got:?}");
        assert!(max_abs_diff(&reconstruct(&eig), &m) < 1e-13);
    }

    #[test]
    fn empty_and_scalar_matrices() {
        let eig = symmetric_eigen(&Array2::<f64>::zeros((0, 0))).expect("empty input");
        assert!(eig.values.is_empty());
        let eig = symmetric_eigen(&array![[4.5]]).expect("finite input");
        assert_eq!(eig.values.to_vec(), vec![4.5]);
        assert_eq!(eig.vectors, array![[1.0]]);
    }

    #[test]
    fn non_finite_or_non_square_input_is_rejected() {
        assert!(symmetric_eigen(&array![[1.0, f64::NAN], [f64::NAN, 1.0]]).is_none());
        assert!(symmetric_eigen(&array![[f64::INFINITY, 0.0], [0.0, 1.0]]).is_none());
        assert!(symmetric_eigen(&Array2::<f64>::zeros((2, 3))).is_none());
    }

    #[test]
    fn tiny_off_diagonal_entries_do_not_overflow() {
        // The (0, 1) pair forces a sweep; in it, τ = (a_22 − a_00)/(2 a_02)
        // overflows to infinity for the subnormal a_02, and that rotation
        // degenerates to the identity instead of producing NaN.
        let m = array![[1.0, 0.5, 1e-310], [0.5, 2.0, 0.0], [1e-310, 0.0, 3.0]];
        let eig = symmetric_eigen(&m).expect("finite symmetric input");
        assert!(eig.values.iter().all(|x| x.is_finite()));
        assert!(eig.vectors.iter().all(|x| x.is_finite()));
        assert!(max_abs_diff(&reconstruct(&eig), &m) < 1e-15);
        assert!(orthonormality_error(&eig.vectors) < 1e-15);
    }
}
