//! Fermion-to-qubit mappings: Jordan–Wigner and parity.
//!
//! Each ladder operator becomes a sum of two Pauli strings with complex
//! coefficients; a term is the product of its operators, expanded with
//! [`PauliString::mul`]. The complex sum is accumulated and, since a
//! Hermitian operator has real Pauli coefficients, every imaginary part must
//! vanish to within [`HERMITICITY_TOLERANCE`].
//!
//! On `n` modes, with the sign convention of [`super::fermion`]:
//!
//! ```text
//! Jordan–Wigner:  a†_p = Z_0 ⋯ Z_{p−1} · (X_p − i Y_p) / 2
//! Parity:         a†_p = ½ · X_{n−1} ⋯ X_{p+1} · (Z_{p−1} X_p − i Y_p)
//! ```
//!
//! and `a_p` is the same with `+ i Y_p`. In the parity encoding qubit `j`
//! stores the parity of the occupations of modes `0..=j`
//! ([`QubitMapping::encode_occupation`]).

use std::collections::BTreeMap;

use super::fermion::{FermionOperator, Ladder};
use super::pauli::{PauliAxis, PauliString, PauliSum, PauliTerm};
use crate::error::PhysicsError;

/// Largest imaginary part accepted on a mapped Pauli coefficient. Above it
/// the operator is not Hermitian and [`FermionOperator::to_qubits`] fails;
/// below it the imaginary part is rounding noise and is dropped.
pub const HERMITICITY_TOLERANCE: f64 = 1e-12;

/// A complex number `(re, im)`.
type Complex = (f64, f64);

fn complex_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// How fermionic modes are encoded in qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QubitMapping {
    /// Qubit `p` holds the occupation of mode `p`.
    JordanWigner,
    /// Qubit `j` holds the parity of the occupations of modes `0..=j`.
    Parity,
}

impl QubitMapping {
    /// Basis index that encodes the occupation-number state `occupation`
    /// (bit `p` = occupation of mode `p`) on `num_modes` modes.
    ///
    /// ```
    /// use polypus_physics::hamiltonians::QubitMapping;
    ///
    /// // H2 Hartree–Fock state (modes 0 and 2 occupied), block order.
    /// assert_eq!(QubitMapping::JordanWigner.encode_occupation(0b0101, 4).unwrap(), 0b0101);
    /// assert_eq!(QubitMapping::Parity.encode_occupation(0b0101, 4).unwrap(), 0b0011);
    /// ```
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if `num_modes` exceeds the bits
    /// of a `usize` (a basis index cannot hold the state), or if
    /// `occupation` has a bit set at a mode `≥ num_modes`.
    pub fn encode_occupation(
        self,
        occupation: usize,
        num_modes: usize,
    ) -> Result<usize, PhysicsError> {
        let bits = usize::BITS as usize;
        if num_modes > bits {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!("{num_modes} modes do not fit in a {bits}-bit basis index"),
            });
        }
        if num_modes < bits && occupation >> num_modes != 0 {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "occupation {occupation:#b} has modes beyond the {num_modes} available"
                ),
            });
        }
        Ok(match self {
            QubitMapping::JordanWigner => occupation,
            QubitMapping::Parity => {
                let mut parity = 0;
                let mut encoded = 0;
                for j in 0..num_modes {
                    parity ^= (occupation >> j) & 1;
                    encoded |= parity << j;
                }
                encoded
            }
        })
    }

    /// The two `(coefficient, string)` pieces of the ladder operator on mode
    /// `p` among `n` modes.
    fn ladder(
        self,
        p: usize,
        kind: Ladder,
        n: usize,
    ) -> Result<[(Complex, PauliString); 2], PhysicsError> {
        let y_coefficient = match kind {
            Ladder::Create => (0.0, -0.5),
            Ladder::Annihilate => (0.0, 0.5),
        };
        let (x_ops, y_ops): (Vec<_>, Vec<_>) = match self {
            QubitMapping::JordanWigner => {
                let z_string = (0..p).map(|q| (q, PauliAxis::Z));
                (
                    z_string.clone().chain([(p, PauliAxis::X)]).collect(),
                    z_string.chain([(p, PauliAxis::Y)]).collect(),
                )
            }
            QubitMapping::Parity => {
                let x_string = ((p + 1)..n).map(|q| (q, PauliAxis::X));
                let z_below = p.checked_sub(1).map(|q| (q, PauliAxis::Z));
                (
                    x_string
                        .clone()
                        .chain(z_below)
                        .chain([(p, PauliAxis::X)])
                        .collect(),
                    x_string.chain([(p, PauliAxis::Y)]).collect(),
                )
            }
        };
        Ok([
            ((0.5, 0.0), PauliString::new(x_ops)?),
            (y_coefficient, PauliString::new(y_ops)?),
        ])
    }
}

impl FermionOperator {
    /// Maps the operator to qubits.
    ///
    /// Returns the [`PauliSum`] in canonical form
    /// ([`PauliSum::simplified`]): only exactly-zero coefficients are
    /// dropped, so rounding residues (e.g. `1e-17`) are kept; use
    /// [`PauliSum::chop`] to remove them.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if a mapped coefficient has an
    /// imaginary part above [`HERMITICITY_TOLERANCE`], i.e. the operator is
    /// not Hermitian and has no real Pauli sum.
    pub fn to_qubits(&self, mapping: QubitMapping) -> Result<PauliSum, PhysicsError> {
        let n = self.num_modes();
        // The 2n distinct ladder operators, built once: ladders[p][0] is a†_p,
        // ladders[p][1] is a_p.
        let ladders = (0..n)
            .map(|p| {
                Ok([
                    mapping.ladder(p, Ladder::Create, n)?,
                    mapping.ladder(p, Ladder::Annihilate, n)?,
                ])
            })
            .collect::<Result<Vec<_>, PhysicsError>>()?;
        let mut accumulated: BTreeMap<PauliString, Complex> = BTreeMap::new();
        for term in self.terms() {
            let mut partial = vec![((term.coefficient, 0.0), PauliString::identity())];
            for &(mode, kind) in &term.ops {
                let pieces = &ladders[mode][match kind {
                    Ladder::Create => 0,
                    Ladder::Annihilate => 1,
                }];
                let mut next = Vec::with_capacity(partial.len() * pieces.len());
                for (coefficient, string) in &partial {
                    for (piece_coefficient, piece) in pieces {
                        let (phase, product) = string.mul(piece);
                        let c = complex_mul(
                            complex_mul(*coefficient, *piece_coefficient),
                            phase.to_complex(),
                        );
                        next.push((c, product));
                    }
                }
                partial = next;
            }
            for (c, string) in partial {
                let entry = accumulated.entry(string).or_insert((0.0, 0.0));
                entry.0 += c.0;
                entry.1 += c.1;
            }
        }
        let mut terms = Vec::with_capacity(accumulated.len());
        for (string, (re, im)) in accumulated {
            if im.abs() > HERMITICITY_TOLERANCE {
                return Err(PhysicsError::UnsupportedOperation {
                    message: format!(
                        "operator is not Hermitian: {string} has imaginary coefficient {im}"
                    ),
                });
            }
            terms.push(PauliTerm::new(re, string));
        }
        PauliSum::new(n, terms)?.simplified()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonians::exact::DenseMatrix;
    use crate::hamiltonians::fermion::{FermionHamiltonian, FermionTerm};
    use crate::test_support::symmetric_integrals as integrals;
    use Ladder::{Annihilate, Create};

    fn max_abs_diff(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn coefficient(h: &PauliSum, s: &str) -> f64 {
        let s: PauliString = s.parse().unwrap();
        h.terms()
            .iter()
            .find(|t| t.string == s)
            .map_or(0.0, |t| t.coefficient)
    }

    #[test]
    fn jordan_wigner_matches_the_fermionic_matrix() {
        for (k, seed) in [(1, 1), (2, 2), (3, 3)] {
            let fermions = integrals(k, seed).to_fermion_operator();
            let reference = fermions.to_dense_real().unwrap();
            let qubits = fermions
                .to_qubits(QubitMapping::JordanWigner)
                .unwrap()
                .to_dense_real()
                .unwrap();
            assert!(max_abs_diff(&reference, &qubits) <= 1e-12, "K = {k}");
        }
    }

    #[test]
    fn parity_equals_jordan_wigner_relabelled_by_the_parity_encoding() {
        for (k, seed) in [(1, 4), (2, 5), (3, 6)] {
            let n = 2 * k;
            let fermions = integrals(k, seed).to_fermion_operator();
            let jw = fermions
                .to_qubits(QubitMapping::JordanWigner)
                .unwrap()
                .to_dense_real()
                .unwrap();
            let parity = fermions
                .to_qubits(QubitMapping::Parity)
                .unwrap()
                .to_dense_real()
                .unwrap();
            let enc = |b| QubitMapping::Parity.encode_occupation(b, n).unwrap();
            for r in 0..(1 << n) {
                for c in 0..(1 << n) {
                    let d = parity.get(enc(r), enc(c)).unwrap() - jw.get(r, c).unwrap();
                    assert!(d.abs() <= 1e-12, "K = {k}, ({r}, {c})");
                }
            }
        }
    }

    #[test]
    fn one_orbital_atom_maps_to_the_hand_derived_sums() {
        // H = E + h(n0 + n1) + U n0 n1 with n_p = (I − Z_p)/2 under JW:
        // (E + h + U/4) I − (h/2 + U/4)(Z0 + Z1) + (U/4) Z0 Z1.
        let (e, h, u) = (0.7, -1.2, 0.65);
        let fermions = FermionHamiltonian::new(1, e, vec![h], vec![u])
            .unwrap()
            .to_fermion_operator();
        let jw = fermions.to_qubits(QubitMapping::JordanWigner).unwrap();
        let field = -(h / 2.0 + u / 4.0);
        for (s, c) in [
            ("I", e + h + u / 4.0),
            ("Z0", field),
            ("Z1", field),
            ("Z0 Z1", u / 4.0),
        ] {
            assert!((coefficient(&jw, s) - c).abs() < 1e-15, "JW {s}");
        }
        assert_eq!(jw.terms().len(), 4);
        // Parity: Z0 → Z0, Z1 → Z0 Z1 (n1 = b0 ⊕ b1) and Z0 Z1 → Z1.
        let parity = fermions.to_qubits(QubitMapping::Parity).unwrap();
        for (s, c) in [
            ("I", e + h + u / 4.0),
            ("Z0", field),
            ("Z0 Z1", field),
            ("Z1", u / 4.0),
        ] {
            assert!((coefficient(&parity, s) - c).abs() < 1e-15, "parity {s}");
        }
        assert_eq!(parity.terms().len(), 4);
    }

    #[test]
    fn number_operator_in_each_encoding() {
        // n_2 on 4 modes: (I − Z2)/2 under JW, (I − Z1 Z2)/2 under parity.
        let n2 = FermionOperator::new(
            4,
            vec![FermionTerm::new(1.0, vec![(2, Create), (2, Annihilate)])],
        )
        .unwrap();
        let jw = n2.to_qubits(QubitMapping::JordanWigner).unwrap();
        assert_eq!(jw.to_string(), "+0.5 I\n-0.5 Z2");
        let parity = n2.to_qubits(QubitMapping::Parity).unwrap();
        assert_eq!(parity.to_string(), "+0.5 I\n-0.5 Z1 Z2");
    }

    #[test]
    fn pairing_terms_map_correctly_in_both_encodings() {
        // Hermitian but not number-conserving: a†a† + aa products exercise the
        // mappings' phases beyond the a†a pairs of molecular Hamiltonians.
        let op = FermionOperator::new(
            4,
            vec![
                FermionTerm::new(1.0, vec![(0, Create), (2, Create)]),
                FermionTerm::new(1.0, vec![(2, Annihilate), (0, Annihilate)]),
                FermionTerm::new(0.7, vec![(1, Create), (3, Create)]),
                FermionTerm::new(0.7, vec![(3, Annihilate), (1, Annihilate)]),
                FermionTerm::new(
                    -0.4,
                    vec![(0, Create), (1, Create), (3, Create), (2, Create)],
                ),
                FermionTerm::new(
                    -0.4,
                    vec![
                        (2, Annihilate),
                        (3, Annihilate),
                        (1, Annihilate),
                        (0, Annihilate),
                    ],
                ),
                FermionTerm::new(0.25, vec![(3, Create), (3, Annihilate)]),
            ],
        )
        .unwrap();
        let reference = op.to_dense_real().unwrap();
        let jw = op
            .to_qubits(QubitMapping::JordanWigner)
            .unwrap()
            .to_dense_real()
            .unwrap();
        assert!(max_abs_diff(&reference, &jw) <= 1e-12);
        let parity = op
            .to_qubits(QubitMapping::Parity)
            .unwrap()
            .to_dense_real()
            .unwrap();
        let enc = |b| QubitMapping::Parity.encode_occupation(b, 4).unwrap();
        for r in 0..16 {
            for c in 0..16 {
                let d = parity.get(enc(r), enc(c)).unwrap() - reference.get(r, c).unwrap();
                assert!(d.abs() <= 1e-12, "({r}, {c})");
            }
        }
    }

    #[test]
    fn non_hermitian_operator_is_rejected() {
        let hop = FermionOperator::new(
            2,
            vec![FermionTerm::new(1.0, vec![(0, Create), (1, Annihilate)])],
        )
        .unwrap();
        for mapping in [QubitMapping::JordanWigner, QubitMapping::Parity] {
            assert!(matches!(
                hop.to_qubits(mapping),
                Err(PhysicsError::UnsupportedOperation { .. })
            ));
        }
        // With its conjugate the hopping is Hermitian: (X0 X1 + Y0 Y1)/2 in JW.
        let hermitian = FermionOperator::new(
            2,
            vec![
                FermionTerm::new(1.0, vec![(0, Create), (1, Annihilate)]),
                FermionTerm::new(1.0, vec![(1, Create), (0, Annihilate)]),
            ],
        )
        .unwrap();
        assert_eq!(
            hermitian
                .to_qubits(QubitMapping::JordanWigner)
                .unwrap()
                .to_string(),
            "+0.5 X0 X1\n+0.5 Y0 Y1"
        );
    }

    #[test]
    fn parity_encoding_of_h2_determinants() {
        let enc = |b| QubitMapping::Parity.encode_occupation(b, 4).unwrap();
        assert_eq!(enc(0b0101), 0b0011); // Hartree–Fock
        assert_eq!(enc(0b1010), 0b0110); // both electrons in σ*
        assert_eq!(
            QubitMapping::JordanWigner
                .encode_occupation(0b1010, 4)
                .unwrap(),
            0b1010
        );
        // The encoding is a bijection on 4 modes.
        let mut seen: Vec<usize> = (0..16).map(enc).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn encode_occupation_rejects_states_that_do_not_fit() {
        for mapping in [QubitMapping::JordanWigner, QubitMapping::Parity] {
            // More modes than bits in a basis index.
            assert!(matches!(
                mapping.encode_occupation(1, usize::BITS as usize + 6),
                Err(PhysicsError::UnsupportedOperation { .. })
            ));
            // An occupied mode beyond num_modes.
            assert!(matches!(
                mapping.encode_occupation(0b1_0000, 4),
                Err(PhysicsError::UnsupportedOperation { .. })
            ));
        }
        // Exactly usize::BITS modes is the largest valid register.
        let all = usize::MAX;
        let bits = usize::BITS as usize;
        assert_eq!(
            QubitMapping::JordanWigner
                .encode_occupation(all, bits)
                .unwrap(),
            all
        );
        // Parity of a prefix of all-ones alternates 1, 0, 1, 0, …
        let alternating = (0..bits).fold(0usize, |acc, j| acc | ((1 - j % 2) << j));
        assert_eq!(
            QubitMapping::Parity.encode_occupation(all, bits).unwrap(),
            alternating
        );
    }
}
