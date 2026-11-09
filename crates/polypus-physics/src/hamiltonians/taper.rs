//! Removing qubits whose `Z` value is fixed ("tapering").
//!
//! If no term of a [`PauliSum`] acts on qubit `q` with `X` or `Y`, the sum
//! commutes with `Z_q` and, inside an eigenspace of `Z_q`, every `Z_q` can be
//! replaced by its eigenvalue `±1`: qubit `q` disappears and the remaining
//! qubits are renumbered in order. [`PauliSum::taper`] does this for qubits
//! and values chosen by the caller; [`parity_reduction_qubits`] gives the
//! usual choice for a molecule in the parity encoding.
//!
//! # Limitations
//!
//! - **Parity, not particle number.** In the parity encoding with
//!   spin-orbitals in block order, qubit `K − 1` holds the parity of `N↑` and
//!   qubit `2K − 1` the parity of `N` (`K` spatial orbitals). Fixing them
//!   fixes those **parities**, not `N↑` and `N↓` themselves. The tapered sum
//!   therefore still contains every sector `(N↑, N↓)` with the same parities:
//!   for H₂ in STO-3G (`K = 2`) "odd `N↑`" can only be `N↑ = 1`, so the
//!   tapered spectrum is exactly the one-up/one-down sector; for larger
//!   molecules it is not (LiH in STO-3G, `K = 6`, `N↑ = N↓ = 2`, also keeps
//!   `N↑, N↓ ∈ {0, 4, 6}`), and its lowest eigenvalue may belong to another
//!   number of electrons.
//! - **Consequence and fix.** An ansatz that conserves the number of
//!   electrons stays in the right sector, so VQE is unaffected; but an exact
//!   reference energy from the tapered sum is only valid when every other
//!   sector with the same parities lies higher. The fix is to diagonalize in
//!   the `(N↑, N↓)` sector (planned in issue #535, together with the generic
//!   detection of `Z₂` symmetries).
//! - **Only fixed `Z` qubits.** Qubits are removed only when the caller
//!   names them and every term acts on them with `I` or `Z`; symmetries that
//!   need a Clifford rotation first are not detected (also issue #535).

use super::pauli::{PauliAxis, PauliString, PauliSum, PauliTerm};
use crate::error::PhysicsError;

impl PauliSum {
    /// Removes the qubits in `fixed`, replacing each `Z_q` by its value
    /// (`+1` or `−1`), and renumbers the remaining qubits in order. Returns
    /// the canonical form ([`PauliSum::simplified`]).
    ///
    /// See the module docs for the limitations.
    ///
    /// ```
    /// use polypus_physics::hamiltonians::{PauliString, PauliSum, PauliTerm};
    ///
    /// let h = PauliSum::new(
    ///     2,
    ///     vec![
    ///         PauliTerm::new(0.3, "Z0 Z1".parse::<PauliString>().unwrap()),
    ///         PauliTerm::new(0.5, "X0".parse::<PauliString>().unwrap()),
    ///     ],
    /// )
    /// .unwrap();
    /// // With Z1 = −1: 0.3·Z0·(−1) + 0.5·X0.
    /// assert_eq!(h.taper(&[(1, -1)]).unwrap().to_string(), "+0.5 X0\n-0.3 Z0");
    /// ```
    ///
    /// # Errors
    /// - [`PhysicsError::InvalidPauliSum`] if a qubit is out of range or
    ///   repeated, or a value is not `+1` or `−1`.
    /// - [`PhysicsError::UnsupportedOperation`] if a term acts with `X` or `Y`
    ///   on a qubit in `fixed`: that qubit has no fixed value.
    pub fn taper(&self, fixed: &[(usize, i8)]) -> Result<PauliSum, PhysicsError> {
        let n = self.num_qubits();
        let invalid = |message: String| PhysicsError::InvalidPauliSum { message };
        let mut value_of: Vec<Option<i8>> = vec![None; n];
        for &(q, v) in fixed {
            if q >= n {
                return Err(invalid(format!(
                    "cannot taper qubit {q}: the sum has {n} qubits"
                )));
            }
            if v != 1 && v != -1 {
                return Err(invalid(format!(
                    "value {v} for qubit {q} is not a Z eigenvalue (+1 or -1)"
                )));
            }
            if value_of[q].is_some() {
                return Err(invalid(format!("qubit {q} is tapered more than once")));
            }
            value_of[q] = Some(v);
        }
        // New index of a kept qubit: its index minus the removed qubits below.
        let mut removed_below = vec![0usize; n];
        let mut removed = 0;
        for q in 0..n {
            removed_below[q] = removed;
            if value_of[q].is_some() {
                removed += 1;
            }
        }

        let mut terms = Vec::with_capacity(self.terms().len());
        for term in self.terms() {
            let mut coefficient = term.coefficient;
            let mut ops = Vec::with_capacity(term.string.weight());
            for &(q, axis) in term.string.ops() {
                match (value_of[q], axis) {
                    (None, _) => ops.push((q - removed_below[q], axis)),
                    (Some(v), PauliAxis::Z) => {
                        if v < 0 {
                            coefficient = -coefficient;
                        }
                    }
                    (Some(_), _) => {
                        return Err(PhysicsError::UnsupportedOperation {
                            message: format!(
                                "{} acts with {axis:?} on qubit {q}, which has no fixed Z value",
                                term.string
                            ),
                        })
                    }
                }
            }
            terms.push(PauliTerm::new(coefficient, PauliString::new(ops)?));
        }
        PauliSum::new(n - removed, terms)?.simplified()
    }
}

/// The two qubits that the parity encoding fixes for a molecule with
/// `num_spatial` (`K`) spatial orbitals, `n_up` spin-up and `n_down`
/// spin-down electrons, with spin-orbitals in block order:
/// `[(K − 1, (−1)^{N↑}), (2K − 1, (−1)^{N↑ + N↓})]`.
///
/// They fix the **parities** of `N↑` and `N`, not their values: see the
/// module docs for when the tapered spectrum contains other sectors.
///
/// ```
/// use polypus_physics::hamiltonians::parity_reduction_qubits;
///
/// assert_eq!(parity_reduction_qubits(2, 1, 1).unwrap(), [(1, -1), (3, 1)]); // H2
/// assert_eq!(parity_reduction_qubits(6, 2, 2).unwrap(), [(5, 1), (11, 1)]); // LiH
/// ```
///
/// # Errors
/// [`PhysicsError::UnsupportedOperation`] if `num_spatial` is 0 or so large
/// that `2K` overflows `usize`, or an electron count exceeds `num_spatial`.
pub fn parity_reduction_qubits(
    num_spatial: usize,
    n_up: usize,
    n_down: usize,
) -> Result<Vec<(usize, i8)>, PhysicsError> {
    if num_spatial == 0 || n_up > num_spatial || n_down > num_spatial {
        return Err(PhysicsError::UnsupportedOperation {
            message: format!(
                "{n_up} spin-up and {n_down} spin-down electrons do not fit in \
                 {num_spatial} spatial orbitals"
            ),
        });
    }
    let Some(num_modes) = num_spatial.checked_mul(2) else {
        return Err(PhysicsError::UnsupportedOperation {
            message: format!("{num_spatial} spatial orbitals: 2K spin-orbitals overflow usize"),
        });
    };
    let sign = |count: usize| if count % 2 == 0 { 1 } else { -1 };
    // n_up, n_down ≤ K, so their sum cannot overflow once 2K does not.
    Ok(vec![
        (num_spatial - 1, sign(n_up)),
        (num_modes - 1, sign(n_up + n_down)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonians::exact::DenseMatrix;
    use crate::hamiltonians::mapping::QubitMapping;
    use crate::test_support::assert_close;
    use crate::test_support::symmetric_integrals;

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

    /// Eigenvalues of the Jordan–Wigner Hamiltonian restricted to the
    /// basis states whose `(N↑, N↓)` is in `sectors` (block order).
    fn sector_spectrum(k: usize, seed: u64, sectors: &[(u32, u32)]) -> Vec<f64> {
        let jw = symmetric_integrals(k, seed)
            .to_fermion_operator()
            .to_qubits(QubitMapping::JordanWigner)
            .unwrap()
            .to_dense_real()
            .unwrap();
        let up_mask = (1usize << k) - 1;
        let states: Vec<usize> = (0..1usize << (2 * k))
            .filter(|b| {
                let counts = ((b & up_mask).count_ones(), (b >> k).count_ones());
                sectors.contains(&counts)
            })
            .collect();
        let mut block = DenseMatrix::zeros(states.len());
        for (i, &r) in states.iter().enumerate() {
            for (j, &c) in states.iter().enumerate() {
                block.add_to(i, j, jw.get(r, c).unwrap());
            }
        }
        block.symmetric_eigenvalues().unwrap()
    }

    fn tapered_spectrum(k: usize, seed: u64, n_up: usize, n_down: usize) -> Vec<f64> {
        symmetric_integrals(k, seed)
            .to_fermion_operator()
            .to_qubits(QubitMapping::Parity)
            .unwrap()
            .taper(&parity_reduction_qubits(k, n_up, n_down).unwrap())
            .unwrap()
            .eigenvalues()
            .unwrap()
    }

    #[test]
    fn two_orbitals_tapered_spectrum_is_the_one_up_one_down_sector() {
        for seed in [11, 12, 13] {
            assert_close(
                &tapered_spectrum(2, seed, 1, 1),
                &sector_spectrum(2, seed, &[(1, 1)]),
                1e-10,
            );
        }
    }

    #[test]
    fn three_orbitals_keep_every_sector_with_the_same_parities() {
        // Documents the limitation: N↑ = N↓ = 1 with K = 3 also keeps the
        // sectors with 3 electrons of either spin (odd N↑, odd N↓).
        let tapered = tapered_spectrum(3, 21, 1, 1);
        assert_eq!(tapered.len(), 16);
        assert_close(
            &tapered,
            &sector_spectrum(3, 21, &[(1, 1), (1, 3), (3, 1), (3, 3)]),
            1e-10,
        );
    }

    #[test]
    fn kept_qubits_are_renumbered_in_order() {
        let h = sum(4, &[(1.0, "Z0 Z2 Z3"), (0.5, "X0 Y2"), (0.25, "Z1 X3")]);
        let tapered = h.taper(&[(1, -1)]).unwrap();
        assert_eq!(tapered.num_qubits(), 3);
        assert_eq!(tapered.to_string(), "+0.5 X0 Y1\n+1 Z0 Z1 Z2\n-0.25 X2");
    }

    #[test]
    fn substituted_terms_merge() {
        // Z1 → −1 turns 0.75·Z1 into a constant that joins 0.25·I.
        let h = sum(2, &[(0.25, "I"), (0.75, "Z1"), (0.5, "Z0")]);
        assert_eq!(h.taper(&[(1, -1)]).unwrap().to_string(), "-0.5 I\n+0.5 Z0");
    }

    #[test]
    fn x_or_y_on_a_fixed_qubit_is_rejected() {
        for s in ["X1", "Y1 Z0", "X0 X1"] {
            let h = sum(2, &[(1.0, s)]);
            assert!(
                matches!(
                    h.taper(&[(1, 1)]),
                    Err(PhysicsError::UnsupportedOperation { .. })
                ),
                "{s}"
            );
        }
    }

    #[test]
    fn malformed_requests_are_rejected() {
        let h = sum(2, &[(1.0, "Z0")]);
        for fixed in [
            vec![(2, 1)],
            vec![(0, 0)],
            vec![(0, 2)],
            vec![(1, 1), (1, -1)],
        ] {
            assert!(
                matches!(h.taper(&fixed), Err(PhysicsError::InvalidPauliSum { .. })),
                "{fixed:?}"
            );
        }
    }

    #[test]
    fn parity_reduction_qubits_follow_the_electron_counts() {
        assert_eq!(parity_reduction_qubits(2, 1, 1).unwrap(), [(1, -1), (3, 1)]);
        assert_eq!(parity_reduction_qubits(6, 2, 2).unwrap(), [(5, 1), (11, 1)]);
        assert_eq!(parity_reduction_qubits(3, 2, 1).unwrap(), [(2, 1), (5, -1)]);
        assert!(parity_reduction_qubits(0, 0, 0).is_err());
        assert!(parity_reduction_qubits(2, 3, 0).is_err());
        // 2K would overflow: an error, not a panic or a wrapped qubit index.
        assert!(parity_reduction_qubits(usize::MAX / 2 + 1, 0, 0).is_err());
        assert!(parity_reduction_qubits(usize::MAX / 2, 0, 0).is_ok());
    }
}
