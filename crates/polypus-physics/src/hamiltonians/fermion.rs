//! Fermionic operators and the electronic Hamiltonian built from spatial
//! integrals.
//!
//! # Conventions
//!
//! - **Modes** (spin-orbitals) are numbered `0..num_modes`. In a basis state,
//!   bit `p` of the index is the occupation of mode `p` (little-endian, as in
//!   [`super::exact`]).
//! - **Fermionic sign**: creating or annihilating mode `p` multiplies the
//!   state by `(−1)^k`, with `k` the number of occupied modes `< p`. This is
//!   the sign that Jordan–Wigner reproduces with its `Z_0 ⋯ Z_{p−1}` string.
//! - **Spin-orbital order** of [`FermionHamiltonian`]: by blocks, first every
//!   spin-up orbital, then every spin-down one. Spatial orbital `p` with spin
//!   up is mode `p`, with spin down is mode `p + K` (`K` spatial orbitals).
//! - **Two-electron integrals** in chemists' notation, `(pq|rs)`, as in
//!   FCIDUMP files.

use super::exact::{DenseMatrix, MAX_DENSE_QUBITS};
use crate::error::PhysicsError;

/// Largest asymmetry accepted between integrals that must be equal
/// (`h_pq` vs `h_qp`, and the 8 permutations of `(pq|rs)`). Within it the
/// integrals are symmetrized; beyond it construction fails.
pub const INTEGRAL_SYMMETRY_TOLERANCE: f64 = 1e-10;

/// The 8 index permutations under which a real-orbital two-electron
/// integral `(pq|rs)` is invariant: `p↔q`, `r↔s` and `(pq)↔(rs)`.
pub(crate) fn chemist_permutations(
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> [(usize, usize, usize, usize); 8] {
    [
        (p, q, r, s),
        (q, p, r, s),
        (p, q, s, r),
        (q, p, s, r),
        (r, s, p, q),
        (s, r, p, q),
        (r, s, q, p),
        (s, r, q, p),
    ]
}

/// Kind of a fermionic ladder operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ladder {
    /// Creation operator `a†_p`.
    Create,
    /// Annihilation operator `a_p`.
    Annihilate,
}

/// One term `c · o_1 o_2 ⋯ o_k` of a [`FermionOperator`].
///
/// `ops` lists the ladder operators in written order, left to right; acting
/// on a state, the rightmost one acts first. An empty `ops` is the identity.
#[derive(Debug, Clone, PartialEq)]
pub struct FermionTerm {
    /// Real coefficient.
    pub coefficient: f64,
    /// `(mode, kind)` pairs in written order.
    pub ops: Vec<(usize, Ladder)>,
}

impl FermionTerm {
    /// Builds a term.
    pub fn new(coefficient: f64, ops: Vec<(usize, Ladder)>) -> Self {
        Self { coefficient, ops }
    }
}

/// A sum of products of fermionic ladder operators on `num_modes` modes, with
/// real coefficients.
///
/// Invariants (checked by [`FermionOperator::new`]): every coefficient is
/// finite and every mode is `< num_modes`.
#[derive(Debug, Clone, PartialEq)]
pub struct FermionOperator {
    num_modes: usize,
    terms: Vec<FermionTerm>,
}

impl FermionOperator {
    /// Builds a validated operator.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidFermionOperator`] if a coefficient is not finite
    /// or a mode is `≥ num_modes`.
    pub fn new(num_modes: usize, terms: Vec<FermionTerm>) -> Result<Self, PhysicsError> {
        for term in &terms {
            if !term.coefficient.is_finite() {
                return Err(PhysicsError::InvalidFermionOperator {
                    message: format!("coefficient is not finite: {}", term.coefficient),
                });
            }
            if let Some(&(mode, _)) = term.ops.iter().find(|&&(mode, _)| mode >= num_modes) {
                return Err(PhysicsError::InvalidFermionOperator {
                    message: format!("mode {mode} out of range for {num_modes} modes"),
                });
            }
        }
        Ok(Self { num_modes, terms })
    }

    /// Number of modes (spin-orbitals).
    pub fn num_modes(&self) -> usize {
        self.num_modes
    }

    /// The terms, in the order given.
    pub fn terms(&self) -> &[FermionTerm] {
        &self.terms
    }

    /// The operator as a dense real matrix in the occupation-number basis
    /// (bit `p` of the index is the occupation of mode `p`).
    ///
    /// Each term acts on every basis state directly through bit operations
    /// and the fermionic sign; no qubit mapping is involved, so this matrix
    /// is an independent reference for Jordan–Wigner.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if there are more than
    /// [`MAX_DENSE_QUBITS`] modes.
    pub fn to_dense_real(&self) -> Result<DenseMatrix, PhysicsError> {
        let n = self.num_modes;
        if n > MAX_DENSE_QUBITS {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "dense matrix of {n} modes exceeds the limit of {MAX_DENSE_QUBITS}"
                ),
            });
        }
        let dim = 1usize << n;
        let mut m = DenseMatrix::zeros(dim);
        for term in &self.terms {
            for b in 0..dim {
                if let Some((out, sign)) = apply_term(&term.ops, b) {
                    m.add_to(out, b, sign * term.coefficient);
                }
            }
        }
        Ok(m)
    }
}

/// Applies a product of ladder operators (rightmost first) to basis state
/// `b`. Returns the resulting state and sign, or `None` if it vanishes.
fn apply_term(ops: &[(usize, Ladder)], b: usize) -> Option<(usize, f64)> {
    let mut state = b;
    let mut sign = 1.0;
    for &(mode, kind) in ops.iter().rev() {
        let bit = 1usize << mode;
        let occupied = state & bit != 0;
        match (kind, occupied) {
            (Ladder::Create, true) | (Ladder::Annihilate, false) => return None,
            (Ladder::Create, false) | (Ladder::Annihilate, true) => {
                if (state & (bit - 1)).count_ones() % 2 == 1 {
                    sign = -sign;
                }
                state ^= bit;
            }
        }
    }
    Some((state, sign))
}

/// The electronic Hamiltonian of a molecule in a basis of `K` real spatial
/// orbitals:
///
/// ```text
/// H = E_nuc + Σ_{pq,σ} h_pq a†_{pσ} a_{qσ}
///           + ½ Σ_{pqrs,στ} (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
/// ```
///
/// Integrals are given as flat row-major vectors: `one_body[p·K + q] = h_pq`
/// and `two_body[((p·K + q)·K + r)·K + s] = (pq|rs)`. They must have the
/// symmetries of real orbitals (`h_pq = h_qp`; `(pq|rs)` invariant under
/// `p↔q`, `r↔s` and `(pq)↔(rs)`) to within
/// [`INTEGRAL_SYMMETRY_TOLERANCE`]; the stored integrals are then symmetrized
/// exactly by averaging each group of equal entries, so the Hamiltonian is
/// exactly Hermitian.
#[derive(Debug, Clone, PartialEq)]
pub struct FermionHamiltonian {
    num_spatial: usize,
    nuclear_repulsion: f64,
    one_body: Vec<f64>,
    two_body: Vec<f64>,
}

impl FermionHamiltonian {
    /// Builds the Hamiltonian from `K = num_spatial` orbitals, the nuclear
    /// repulsion energy and the integrals (layout in the type docs).
    ///
    /// # Errors
    /// [`PhysicsError::InvalidIntegrals`] if `num_spatial` is 0, a vector has
    /// the wrong length, a value is not finite, or a symmetry is broken by
    /// more than [`INTEGRAL_SYMMETRY_TOLERANCE`].
    pub fn new(
        num_spatial: usize,
        nuclear_repulsion: f64,
        one_body: Vec<f64>,
        two_body: Vec<f64>,
    ) -> Result<Self, PhysicsError> {
        let invalid = |message: String| PhysicsError::InvalidIntegrals { message };
        let k = num_spatial;
        if k == 0 {
            return Err(invalid("at least one spatial orbital is required".into()));
        }
        if one_body.len() != k * k {
            return Err(invalid(format!(
                "one-body integrals: expected {} values for {k} orbitals, got {}",
                k * k,
                one_body.len()
            )));
        }
        if two_body.len() != k * k * k * k {
            return Err(invalid(format!(
                "two-body integrals: expected {} values for {k} orbitals, got {}",
                k * k * k * k,
                two_body.len()
            )));
        }
        if !nuclear_repulsion.is_finite() {
            return Err(invalid(format!(
                "nuclear repulsion is not finite: {nuclear_repulsion}"
            )));
        }
        if let Some(v) = one_body.iter().chain(&two_body).find(|v| !v.is_finite()) {
            return Err(invalid(format!("integral is not finite: {v}")));
        }

        let h = |p: usize, q: usize| one_body[p * k + q];
        let mut sym_one = vec![0.0; k * k];
        for p in 0..k {
            for q in 0..k {
                let (a, b) = (h(p, q), h(q, p));
                if (a - b).abs() > INTEGRAL_SYMMETRY_TOLERANCE {
                    return Err(invalid(format!(
                        "h[{p}][{q}] = {a} and h[{q}][{p}] = {b} differ by more than \
                         {INTEGRAL_SYMMETRY_TOLERANCE}"
                    )));
                }
                // IEEE addition is commutative, so h_pq and h_qp get the same bits.
                sym_one[p * k + q] = (a + b) / 2.0;
            }
        }

        let idx = |p: usize, q: usize, r: usize, s: usize| ((p * k + q) * k + r) * k + s;
        let mut sym_two = vec![0.0; k * k * k * k];
        for p in 0..k {
            for q in 0..k {
                for r in 0..k {
                    for s in 0..k {
                        // Summed in a canonical (sorted) order so every member
                        // of the group gets bit-identical averages.
                        let mut group = chemist_permutations(p, q, r, s);
                        group.sort_unstable();
                        let own = two_body[idx(p, q, r, s)];
                        let mut total = 0.0;
                        for &(a, b, c, d) in &group {
                            let v = two_body[idx(a, b, c, d)];
                            if (v - own).abs() > INTEGRAL_SYMMETRY_TOLERANCE {
                                return Err(invalid(format!(
                                    "({p}{q}|{r}{s}) = {own} and ({a}{b}|{c}{d}) = {v} differ \
                                     by more than {INTEGRAL_SYMMETRY_TOLERANCE}"
                                )));
                            }
                            total += v;
                        }
                        sym_two[idx(p, q, r, s)] = total / 8.0;
                    }
                }
            }
        }

        Ok(Self {
            num_spatial: k,
            nuclear_repulsion,
            one_body: sym_one,
            two_body: sym_two,
        })
    }

    /// Number of spatial orbitals `K`.
    pub fn num_spatial_orbitals(&self) -> usize {
        self.num_spatial
    }

    /// Number of spin-orbitals (modes), `2K`.
    pub fn num_spin_orbitals(&self) -> usize {
        2 * self.num_spatial
    }

    /// Nuclear repulsion energy `E_nuc`.
    pub fn nuclear_repulsion(&self) -> f64 {
        self.nuclear_repulsion
    }

    /// Symmetrized one-body integral `h_pq` (`p, q < K`).
    pub fn one_body(&self, p: usize, q: usize) -> f64 {
        self.one_body[p * self.num_spatial + q]
    }

    /// Symmetrized two-body integral `(pq|rs)` (all indices `< K`).
    pub fn two_body(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let k = self.num_spatial;
        self.two_body[((p * k + q) * k + r) * k + s]
    }

    /// Mode of spatial orbital `p` with spin `spin` (0 = up, 1 = down), in
    /// the block order: up orbitals first.
    pub fn spin_orbital(&self, p: usize, spin: usize) -> usize {
        p + spin * self.num_spatial
    }

    /// Expands the Hamiltonian over spin-orbitals (block order).
    ///
    /// Terms with a zero integral, and two-body terms that create or
    /// annihilate the same spin-orbital twice (identically zero), are left
    /// out.
    pub fn to_fermion_operator(&self) -> FermionOperator {
        use Ladder::{Annihilate, Create};
        let k = self.num_spatial;
        let mut terms = vec![FermionTerm::new(self.nuclear_repulsion, Vec::new())];
        for spin in 0..2 {
            for p in 0..k {
                for q in 0..k {
                    let c = self.one_body(p, q);
                    if c != 0.0 {
                        let (a, b) = (self.spin_orbital(p, spin), self.spin_orbital(q, spin));
                        terms.push(FermionTerm::new(c, vec![(a, Create), (b, Annihilate)]));
                    }
                }
            }
        }
        for sigma in 0..2 {
            for tau in 0..2 {
                for p in 0..k {
                    for q in 0..k {
                        for r in 0..k {
                            for s in 0..k {
                                let c = self.two_body(p, q, r, s);
                                let (ps, qs) =
                                    (self.spin_orbital(p, sigma), self.spin_orbital(q, sigma));
                                let (rs, ss) =
                                    (self.spin_orbital(r, tau), self.spin_orbital(s, tau));
                                if c == 0.0 || ps == rs || qs == ss {
                                    continue;
                                }
                                terms.push(FermionTerm::new(
                                    0.5 * c,
                                    vec![
                                        (ps, Create),
                                        (rs, Create),
                                        (ss, Annihilate),
                                        (qs, Annihilate),
                                    ],
                                ));
                            }
                        }
                    }
                }
            }
        }
        // Modes are < 2K and coefficients are finite integrals (or halves of
        // them), so the invariants of `FermionOperator::new` hold.
        FermionOperator {
            num_modes: self.num_spin_orbitals(),
            terms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, symmetric_integral_arrays};
    use Ladder::{Annihilate, Create};

    fn op(num_modes: usize, terms: Vec<(f64, Vec<(usize, Ladder)>)>) -> FermionOperator {
        FermionOperator::new(
            num_modes,
            terms
                .into_iter()
                .map(|(c, ops)| FermionTerm::new(c, ops))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn creation_sign_counts_occupied_lower_modes() {
        // a†_1 on |01⟩ (mode 0 occupied) gives −|11⟩; on |00⟩ gives +|10⟩.
        let m = op(2, vec![(1.0, vec![(1, Create)])])
            .to_dense_real()
            .unwrap();
        assert_eq!(m.get(3, 1), Some(-1.0));
        assert_eq!(m.get(2, 0), Some(1.0));
        // a†_0 never picks up a sign: there is no lower mode.
        let m = op(2, vec![(1.0, vec![(0, Create)])])
            .to_dense_real()
            .unwrap();
        assert_eq!(m.get(1, 0), Some(1.0));
        assert_eq!(m.get(3, 2), Some(1.0));
    }

    #[test]
    fn ladder_operators_anticommute_and_square_to_zero() {
        let a = op(3, vec![(1.0, vec![(0, Create), (2, Create)])])
            .to_dense_real()
            .unwrap();
        let b = op(3, vec![(1.0, vec![(2, Create), (0, Create)])])
            .to_dense_real()
            .unwrap();
        let negated: Vec<f64> = b.as_slice().iter().map(|x| -x).collect();
        assert_eq!(a.as_slice(), negated.as_slice());
        let twice = op(3, vec![(1.0, vec![(1, Create), (1, Create)])])
            .to_dense_real()
            .unwrap();
        assert!(twice.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn number_operator_counts_occupation() {
        let m = op(2, vec![(1.0, vec![(1, Create), (1, Annihilate)])])
            .to_dense_real()
            .unwrap();
        let diagonal: Vec<f64> = (0..4).map(|i| m.get(i, i).unwrap()).collect();
        assert_eq!(diagonal, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn operator_rejects_bad_modes_and_coefficients() {
        let out_of_range = FermionOperator::new(2, vec![FermionTerm::new(1.0, vec![(2, Create)])]);
        assert!(matches!(
            out_of_range,
            Err(PhysicsError::InvalidFermionOperator { .. })
        ));
        let nan = FermionOperator::new(2, vec![FermionTerm::new(f64::NAN, vec![])]);
        assert!(matches!(
            nan,
            Err(PhysicsError::InvalidFermionOperator { .. })
        ));
        let too_big = op(MAX_DENSE_QUBITS + 1, vec![(1.0, vec![])]).to_dense_real();
        assert!(matches!(
            too_big,
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn one_orbital_hubbard_atom_spectrum() {
        // H = E + h (n↑ + n↓) + U n↑ n↓ has eigenvalues {E, E+h, E+h, E+2h+U}.
        let (e, h, u) = (0.7, -1.2, 0.65);
        let ham = FermionHamiltonian::new(1, e, vec![h], vec![u]).unwrap();
        let ev = ham
            .to_fermion_operator()
            .to_dense_real()
            .unwrap()
            .symmetric_eigenvalues()
            .unwrap();
        let mut expected = vec![e, e + h, e + h, e + 2.0 * h + u];
        expected.sort_by(f64::total_cmp);
        assert_close(&ev, &expected, 1e-14);
    }

    #[test]
    fn closed_shell_and_double_excitation_matrix_elements() {
        // Block order with K = 2: modes 0 = σ↑, 1 = σ*↑, 2 = σ↓, 3 = σ*↓.
        // ⟨σσ|H|σσ⟩ = E_nuc + 2 h_00 + (00|00) and ⟨σ*σ*|H|σσ⟩ = (01|01).
        let (_, one, two) = symmetric_integral_arrays(2, 3);
        let ham = FermionHamiltonian::new(2, 0.7138, one, two).unwrap();
        let m = ham.to_fermion_operator().to_dense_real().unwrap();
        let hf = 0b0101;
        let doubly_excited = 0b1010;
        let expected_hf = 0.7138 + 2.0 * ham.one_body(0, 0) + ham.two_body(0, 0, 0, 0);
        assert!((m.get(hf, hf).unwrap() - expected_hf).abs() < 1e-14);
        assert!((m.get(doubly_excited, hf).unwrap() - ham.two_body(0, 1, 0, 1)).abs() < 1e-14);
    }

    #[test]
    fn hamiltonian_is_hermitian_and_conserves_particles_and_spin() {
        let (_, one, two) = symmetric_integral_arrays(2, 3);
        let ham = FermionHamiltonian::new(2, 0.7138, one, two).unwrap();
        let m = ham.to_fermion_operator().to_dense_real().unwrap();
        assert!(m.is_symmetric());
        let up = |b: usize| (b & 0b0011).count_ones();
        let down = |b: usize| (b & 0b1100).count_ones();
        for r in 0..16 {
            for c in 0..16 {
                if m.get(r, c).unwrap() != 0.0 {
                    assert_eq!((up(r), down(r)), (up(c), down(c)), "({r}, {c})");
                }
            }
        }
    }

    #[test]
    fn small_asymmetry_is_symmetrized_exactly() {
        let (_, one, two) = symmetric_integral_arrays(2, 3);
        let clean = FermionHamiltonian::new(2, 0.7, one.clone(), two.clone()).unwrap();
        let mut noisy_one = one;
        let mut noisy_two = two;
        noisy_one[1] += 3e-12; // h_01 only, not h_10
        noisy_two[7] -= 4e-12; // (01|11) only: index ((0·2+1)·2+1)·2+1
        let noisy = FermionHamiltonian::new(2, 0.7, noisy_one, noisy_two).unwrap();
        assert_eq!(
            noisy.one_body(0, 1).to_bits(),
            noisy.one_body(1, 0).to_bits()
        );
        let group = [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)];
        for &(p, q, r, s) in &group {
            assert_eq!(
                noisy.two_body(p, q, r, s).to_bits(),
                noisy.two_body(0, 1, 1, 1).to_bits()
            );
            // The average moves by at most the injected noise (here by
            // 2·4e-12/8, since (01|11) fills two of the 8 permutation slots).
            assert!((noisy.two_body(p, q, r, s) - clean.two_body(p, q, r, s)).abs() <= 4e-12);
        }
        assert!(noisy
            .to_fermion_operator()
            .to_dense_real()
            .unwrap()
            .is_symmetric());
    }

    #[test]
    fn broken_symmetry_beyond_tolerance_is_rejected() {
        let (_, one, two) = symmetric_integral_arrays(2, 3);
        let mut bad_one = one.clone();
        bad_one[1] += 1e-9;
        assert!(matches!(
            FermionHamiltonian::new(2, 0.7, bad_one, two.clone()),
            Err(PhysicsError::InvalidIntegrals { .. })
        ));
        let mut bad_two = two;
        bad_two[5] += 1e-9; // (01|01) only: index ((0·2+1)·2+0)·2+1
        assert!(matches!(
            FermionHamiltonian::new(2, 0.7, one, bad_two),
            Err(PhysicsError::InvalidIntegrals { .. })
        ));
    }

    #[test]
    fn malformed_integrals_are_rejected() {
        let bad = [
            FermionHamiltonian::new(0, 0.0, vec![], vec![]),
            FermionHamiltonian::new(1, 0.0, vec![1.0, 2.0], vec![0.5]),
            FermionHamiltonian::new(1, 0.0, vec![1.0], vec![0.5, 0.5]),
            FermionHamiltonian::new(1, f64::NAN, vec![1.0], vec![0.5]),
            FermionHamiltonian::new(1, 0.0, vec![f64::INFINITY], vec![0.5]),
        ];
        for r in bad {
            assert!(matches!(r, Err(PhysicsError::InvalidIntegrals { .. })));
        }
    }
}
