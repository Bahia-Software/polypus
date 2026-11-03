//! Dense matrices and exact diagonalization of small [`PauliSum`]s.
//!
//! The basis convention is little-endian: bit `q` of the basis index is the
//! state of qubit `q`, so on 2 qubits `Z1 = diag(1, 1, −1, −1)`.
//!
//! # Limitations
//!
//! - **Real matrices only.** A Pauli string with an odd number of `Y` factors
//!   has purely imaginary entries, so [`PauliSum::to_dense_real`] rejects it
//!   with [`PhysicsError::UnsupportedOperation`] instead of building a wrong
//!   matrix.
//! - **Why this covers molecules.** With real orbitals the molecular
//!   integrals are real, so the Hamiltonian matrix in the occupation basis is
//!   real; Jordan–Wigner and parity mappings only relabel basis states and
//!   flip signs, so the qubit matrix stays real. A real Hermitian matrix has a
//!   zero coefficient on every string with an odd number of `Y`
//!   (`c_P = Tr(P H) / 2^n` would be both real and purely imaginary).
//!   Ordinary molecules without external fields (e.g. H₂, LiH, BeH₂ from
//!   STO-3G or FCIDUMP) always admit real orbitals.
//! - **Not covered.** Complex orbitals or integrals: magnetic fields,
//!   spin–orbit coupling (relativistic methods), periodic systems at
//!   `k ≠ 0`, or mappings that introduce complex phases. These need a
//!   complex matrix, which this module does not build.
//! - **Size.** At most [`MAX_DENSE_QUBITS`] qubits; the whole spectrum is
//!   computed from the full `2^n × 2^n` matrix, with no restriction to a
//!   particle-number sector.
//! - **Eigenvalues only.** No eigenvectors are computed.
//!
//! The eigensolver is the classic Householder tridiagonalization followed by
//! the implicit QL algorithm (Numerical Recipes §11.2–11.3), eigenvalues only.

use super::pauli::{PauliAxis, PauliSum};
use crate::error::PhysicsError;

/// Largest number of qubits for which [`PauliSum::to_dense_real`] builds a
/// matrix (`2^10 × 2^10` doubles, 8 MiB).
pub const MAX_DENSE_QUBITS: usize = 10;

/// Maximum QL iterations per eigenvalue before reporting non-convergence.
const MAX_QL_ITERATIONS: usize = 50;

/// A square real matrix stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    dim: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    fn zeros(dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; dim * dim],
        }
    }

    /// Number of rows (and columns).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Entry `(row, col)`, or `None` if out of range.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        (row < self.dim && col < self.dim).then(|| self.data[row * self.dim + col])
    }

    /// The entries, row-major.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// `true` if the matrix equals its transpose exactly.
    pub fn is_symmetric(&self) -> bool {
        let n = self.dim;
        (0..n).all(|r| (0..r).all(|c| self.data[r * n + c] == self.data[c * n + r]))
    }

    /// Eigenvalues of the matrix, sorted ascending.
    ///
    /// A matrix built by [`PauliSum::to_dense_real`] is always symmetric
    /// (real coefficients times Hermitian Pauli strings with real entries);
    /// symmetry is checked anyway, exactly, because the solver only reads the
    /// lower triangle and would silently return wrong eigenvalues otherwise.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if the matrix is not symmetric;
    /// [`PhysicsError::SimulationError`] if the QL iteration does not
    /// converge (not expected for finite symmetric input).
    pub fn symmetric_eigenvalues(&self) -> Result<Vec<f64>, PhysicsError> {
        if !self.is_symmetric() {
            return Err(PhysicsError::UnsupportedOperation {
                message: "eigenvalues requested for a non-symmetric matrix".into(),
            });
        }
        let n = self.dim;
        let mut a = self.data.clone();
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        tridiagonalize(&mut a, n, &mut d, &mut e);
        tridiagonal_ql(&mut d, &mut e)?;
        d.sort_by(f64::total_cmp);
        Ok(d)
    }
}

impl PauliSum {
    /// The sum as a dense real matrix in the little-endian basis.
    ///
    /// Each Pauli string maps a basis state `|b⟩` to `±|b ⊕ x⟩`, where `x`
    /// marks the qubits carrying `X` or `Y`; the sign is
    /// `i^{#Y} · (−1)^{popcount(b ∧ (y ∨ z))}`.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if the sum has more than
    /// [`MAX_DENSE_QUBITS`] qubits, or if a term has an odd number of `Y`
    /// (its matrix is not real).
    ///
    /// # Example
    /// ```
    /// use polypus_physics::hamiltonians::{PauliString, PauliSum, PauliTerm};
    ///
    /// let z1: PauliString = "Z1".parse().unwrap();
    /// let m = PauliSum::new(2, vec![PauliTerm::new(1.0, z1)])
    ///     .unwrap()
    ///     .to_dense_real()
    ///     .unwrap();
    /// let diagonal: Vec<f64> = (0..4).map(|i| m.get(i, i).unwrap()).collect();
    /// assert_eq!(diagonal, [1.0, 1.0, -1.0, -1.0]);
    /// ```
    pub fn to_dense_real(&self) -> Result<DenseMatrix, PhysicsError> {
        let n = self.num_qubits();
        if n > MAX_DENSE_QUBITS {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "dense matrix of {n} qubits exceeds the limit of {MAX_DENSE_QUBITS}"
                ),
            });
        }
        let dim = 1usize << n;
        let mut m = DenseMatrix::zeros(dim);
        for term in self.terms() {
            let (mut x_mask, mut sign_mask, mut n_y) = (0usize, 0usize, 0u32);
            for &(q, axis) in term.string.ops() {
                match axis {
                    PauliAxis::X => x_mask |= 1 << q,
                    PauliAxis::Y => {
                        x_mask |= 1 << q;
                        sign_mask |= 1 << q;
                        n_y += 1;
                    }
                    PauliAxis::Z => sign_mask |= 1 << q,
                }
            }
            if n_y % 2 == 1 {
                return Err(PhysicsError::UnsupportedOperation {
                    message: format!(
                        "{} has an odd number of Y factors: its matrix is not real",
                        term.string
                    ),
                });
            }
            // i^{#Y} for an even #Y is (−1)^{#Y/2}.
            let y_sign = if n_y % 4 == 0 { 1.0 } else { -1.0 };
            for b in 0..dim {
                let sign = if (b & sign_mask).count_ones() % 2 == 0 {
                    y_sign
                } else {
                    -y_sign
                };
                m.data[(b ^ x_mask) * dim + b] += sign * term.coefficient;
            }
        }
        Ok(m)
    }

    /// All eigenvalues of the sum, sorted ascending (with multiplicity).
    ///
    /// # Errors
    /// As [`PauliSum::to_dense_real`] and
    /// [`DenseMatrix::symmetric_eigenvalues`].
    pub fn eigenvalues(&self) -> Result<Vec<f64>, PhysicsError> {
        self.to_dense_real()?.symmetric_eigenvalues()
    }

    /// Lowest eigenvalue of the sum.
    ///
    /// # Errors
    /// As [`PauliSum::eigenvalues`].
    pub fn ground_energy(&self) -> Result<f64, PhysicsError> {
        let eigenvalues = self.eigenvalues()?;
        eigenvalues
            .first()
            .copied()
            .ok_or_else(|| PhysicsError::SimulationError {
                message: "empty spectrum".into(),
            })
    }
}

/// Householder reduction of the symmetric matrix `a` (row-major, `n × n`,
/// lower triangle read, overwritten) to tridiagonal form: diagonal in `d`,
/// sub-diagonal in `e[1..]` (`e[0] = 0`).
fn tridiagonalize(a: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    let idx = |r: usize, c: usize| r * n + c;
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| a[idx(i, k)].abs()).sum();
            if scale == 0.0 {
                e[i] = a[idx(i, l)];
            } else {
                for k in 0..=l {
                    a[idx(i, k)] /= scale;
                    h += a[idx(i, k)] * a[idx(i, k)];
                }
                let f = a[idx(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[idx(i, l)] = f - g;
                let mut f = 0.0;
                for j in 0..=l {
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += a[idx(j, k)] * a[idx(i, k)];
                    }
                    for k in (j + 1)..=l {
                        g += a[idx(k, j)] * a[idx(i, k)];
                    }
                    e[j] = g / h;
                    f += e[j] * a[idx(i, j)];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    let f = a[idx(i, j)];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[idx(j, k)] -= f * e[k] + g * a[idx(i, k)];
                    }
                }
            }
        } else {
            e[i] = a[idx(i, l)];
        }
        d[i] = h;
    }
    if n > 0 {
        e[0] = 0.0;
    }
    for i in 0..n {
        d[i] = a[idx(i, i)];
    }
}

/// Eigenvalues of the symmetric tridiagonal matrix with diagonal `d` and
/// sub-diagonal `e[1..]`, by the implicit QL algorithm. On return `d` holds
/// the eigenvalues (unsorted) and `e` is destroyed.
fn tridiagonal_ql(d: &mut [f64], e: &mut [f64]) -> Result<(), PhysicsError> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iterations = 0;
        loop {
            // Find a negligible off-diagonal element e[m] to split the matrix.
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > MAX_QL_ITERATIONS {
                return Err(PhysicsError::SimulationError {
                    message: format!(
                        "QL eigenvalue iteration did not converge after {MAX_QL_ITERATIONS} steps"
                    ),
                });
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut underflow = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    // Recover from underflow: deflate and restart this l.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonians::pauli::{PauliString, PauliTerm};
    use crate::hamiltonians::photon_polarization::photon_polarization_hamiltonian;

    fn sum(num_qubits: usize, terms: &[(f64, &str)]) -> PauliSum {
        PauliSum::new(
            num_qubits,
            terms
                .iter()
                .map(|&(c, s)| PauliTerm::new(c, s.parse::<PauliString>().unwrap()))
                .collect(),
        )
        .unwrap()
    }

    fn rows(m: &DenseMatrix) -> Vec<Vec<f64>> {
        m.as_slice().chunks(m.dim()).map(<[f64]>::to_vec).collect()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn z1_on_two_qubits_is_little_endian() {
        let m = sum(2, &[(1.0, "Z1")]).to_dense_real().unwrap();
        let diagonal: Vec<f64> = (0..4).map(|i| m.get(i, i).unwrap()).collect();
        assert_eq!(diagonal, [1.0, 1.0, -1.0, -1.0]);
        assert_eq!(m.get(4, 0), None);
    }

    #[test]
    fn single_qubit_paulis_have_their_textbook_matrices() {
        let x = sum(1, &[(1.0, "X0")]).to_dense_real().unwrap();
        let z = sum(1, &[(1.0, "Z0")]).to_dense_real().unwrap();
        let i = sum(1, &[(1.0, "I")]).to_dense_real().unwrap();
        assert_eq!(rows(&x), [[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(rows(&z), [[1.0, 0.0], [0.0, -1.0]]);
        assert_eq!(rows(&i), [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn xx_and_yy_have_their_textbook_matrices() {
        // XX flips both bits; YY = (iσ⁺…) gives −1 on |00⟩↔|11⟩ and +1 on |01⟩↔|10⟩.
        let xx = sum(2, &[(1.0, "X0 X1")]).to_dense_real().unwrap();
        let yy = sum(2, &[(1.0, "Y0 Y1")]).to_dense_real().unwrap();
        assert_eq!(
            rows(&xx),
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(
            rows(&yy),
            [
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn four_y_string_has_positive_sign() {
        // Y⊗4 maps |0000⟩ to i^4 |1111⟩ = +|1111⟩.
        let m = sum(4, &[(1.0, "Y0 Y1 Y2 Y3")]).to_dense_real().unwrap();
        assert_eq!(m.get(15, 0), Some(1.0));
    }

    #[test]
    fn z_plus_x_has_eigenvalues_plus_minus_sqrt_two() {
        let ev = sum(1, &[(1.0, "Z0"), (1.0, "X0")]).eigenvalues().unwrap();
        assert_close(&ev, &[-2f64.sqrt(), 2f64.sqrt()], 1e-14);
    }

    #[test]
    fn heisenberg_dimer_has_singlet_and_triplet() {
        let ev = sum(2, &[(1.0, "X0 X1"), (1.0, "Y0 Y1"), (1.0, "Z0 Z1")])
            .eigenvalues()
            .unwrap();
        assert_close(&ev, &[-3.0, 1.0, 1.0, 1.0], 1e-12);
    }

    #[test]
    fn photon_spectrum_is_plus_minus_norm() {
        let (omega, coupling) = (1.0, 0.1);
        let h = photon_polarization_hamiltonian(omega, coupling).unwrap();
        let r = f64::hypot(omega, coupling);
        assert_close(&h.eigenvalues().unwrap(), &[-r, r], 1e-14);
        assert!((h.ground_energy().unwrap() + r).abs() < 1e-14);
    }

    #[test]
    fn spectrum_moments_match_pauli_orthogonality() {
        // Tr H = 2^n c_I and Tr H² = 2^n Σ c_k², for a 4-qubit sum with every
        // kind of real term (identity, Z, ZZ, XX, YY, XXYY, XYYX, XZX).
        let terms = [
            (-0.1, "I"),
            (0.17, "Z0"),
            (-0.22, "Z2"),
            (0.12, "Z0 Z2"),
            (0.3, "X0 X1"),
            (-0.07, "Y1 Y3"),
            (-0.045, "X0 X1 Y2 Y3"),
            (0.045, "X0 Y1 Y2 X3"),
            (0.2, "X1 Z2 X3"),
        ];
        let h = sum(4, &terms);
        let ev = h.eigenvalues().unwrap();
        let dim = 16.0;
        let trace: f64 = ev.iter().sum();
        let trace_sq: f64 = ev.iter().map(|x| x * x).sum();
        let sum_sq: f64 = terms.iter().map(|(c, _)| c * c).sum();
        assert!((trace - dim * -0.1).abs() < 1e-12);
        assert!((trace_sq - dim * sum_sq).abs() < 1e-12);
        assert!(ev.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn dense_matrix_of_a_mixed_sum_is_symmetric() {
        let h = sum(
            4,
            &[
                (0.3, "X0 X1"),
                (-0.07, "Y1 Y3"),
                (-0.045, "X0 X1 Y2 Y3"),
                (0.045, "X0 Y1 Y2 X3"),
                (0.2, "X1 Z2 X3"),
                (0.12, "Z0 Z2"),
            ],
        );
        assert!(h.to_dense_real().unwrap().is_symmetric());
    }

    #[test]
    fn non_symmetric_matrix_is_rejected() {
        let m = DenseMatrix {
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!(!m.is_symmetric());
        assert!(matches!(
            m.symmetric_eigenvalues(),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn diagonal_sum_eigenvalues_are_its_diagonal() {
        // A sum of Z strings is diagonal: the solver must return the
        // (sorted) diagonal exactly, on a larger register.
        let h = sum(6, &[(0.5, "Z0"), (-0.25, "Z3 Z5"), (0.125, "Z1 Z2 Z4")]);
        let m = h.to_dense_real().unwrap();
        let mut diagonal: Vec<f64> = (0..m.dim()).map(|i| m.get(i, i).unwrap()).collect();
        diagonal.sort_by(f64::total_cmp);
        assert_close(&h.eigenvalues().unwrap(), &diagonal, 1e-12);
    }

    #[test]
    fn odd_number_of_y_is_rejected() {
        for s in ["Y0", "X0 Y1", "Y0 Y1 Y2"] {
            let r = sum(3, &[(1.0, s)]).to_dense_real();
            assert!(
                matches!(r, Err(PhysicsError::UnsupportedOperation { .. })),
                "{s}"
            );
        }
    }

    #[test]
    fn too_many_qubits_is_rejected() {
        let h = sum(MAX_DENSE_QUBITS + 1, &[(1.0, "Z0")]);
        assert!(matches!(
            h.to_dense_real(),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn zero_qubit_sum_is_a_scalar() {
        let h = sum(0, &[(-0.75, "I")]);
        assert_eq!(h.eigenvalues().unwrap(), [-0.75]);
    }
}
