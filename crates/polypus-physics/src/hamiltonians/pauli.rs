//! Multi-qubit Pauli algebra: [`PauliString`], [`PauliTerm`] and [`PauliSum`].
//!
//! A [`PauliString`] is a tensor product of single-qubit Paulis, stored as
//! `(qubit, axis)` pairs sorted by qubit with no repeated qubit; the empty
//! string is the identity. Its text form is `"X0 X1 Y2 Y3"`, and `"I"` for the
//! identity.
//!
//! [`PauliSum`] holds real coefficients: it represents Hermitian operators.
//! Complex intermediates (e.g. fermionic ladder operators) are handled by the
//! code that builds the sum, using the [`Phase`] returned by
//! [`PauliString::mul`].

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Mul;
use std::str::FromStr;

use crate::error::PhysicsError;

/// Axis of a single-qubit Pauli operator. The identity is not an axis: it is
/// the absence of an operator on that qubit.
///
/// The derived order (`X < Y < Z`) is part of the canonical term order of
/// [`PauliSum::simplified`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PauliAxis {
    /// Pauli-X (σ_x).
    X,
    /// Pauli-Y (σ_y).
    Y,
    /// Pauli-Z (σ_z).
    Z,
}

impl PauliAxis {
    fn letter(self) -> char {
        match self {
            PauliAxis::X => 'X',
            PauliAxis::Y => 'Y',
            PauliAxis::Z => 'Z',
        }
    }

    /// Product of two single-qubit Paulis: `self · other = phase · result`,
    /// with `result = None` for the identity.
    fn mul(self, other: PauliAxis) -> (Phase, Option<PauliAxis>) {
        use PauliAxis::{X, Y, Z};
        match (self, other) {
            (X, X) | (Y, Y) | (Z, Z) => (Phase::PlusOne, None),
            (X, Y) => (Phase::PlusI, Some(Z)),
            (Y, Z) => (Phase::PlusI, Some(X)),
            (Z, X) => (Phase::PlusI, Some(Y)),
            (Y, X) => (Phase::MinusI, Some(Z)),
            (Z, Y) => (Phase::MinusI, Some(X)),
            (X, Z) => (Phase::MinusI, Some(Y)),
        }
    }
}

/// A phase in `{+1, +i, −1, −i}`: the factor produced by multiplying Pauli
/// strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// `+1`.
    PlusOne,
    /// `+i`.
    PlusI,
    /// `−1`.
    MinusOne,
    /// `−i`.
    MinusI,
}

impl Phase {
    /// Power of `i` this phase represents (`i^k`, `k ∈ 0..4`).
    fn power(self) -> u8 {
        match self {
            Phase::PlusOne => 0,
            Phase::PlusI => 1,
            Phase::MinusOne => 2,
            Phase::MinusI => 3,
        }
    }

    /// Phases multiply by adding their powers of `i` (mod 4).
    fn compose(self, other: Phase) -> Phase {
        Phase::from_power(self.power() + other.power())
    }

    fn from_power(k: u8) -> Phase {
        match k % 4 {
            0 => Phase::PlusOne,
            1 => Phase::PlusI,
            2 => Phase::MinusOne,
            _ => Phase::MinusI,
        }
    }

    /// The phase as a complex number `(re, im)`.
    pub fn to_complex(self) -> (f64, f64) {
        match self {
            Phase::PlusOne => (1.0, 0.0),
            Phase::PlusI => (0.0, 1.0),
            Phase::MinusOne => (-1.0, 0.0),
            Phase::MinusI => (0.0, -1.0),
        }
    }
}

impl Mul for Phase {
    type Output = Phase;

    fn mul(self, other: Phase) -> Phase {
        self.compose(other)
    }
}

/// A tensor product of single-qubit Paulis, e.g. `X0 X1 Y2 Y3`.
///
/// Invariant: the `(qubit, axis)` pairs are sorted by qubit and no qubit
/// appears twice. Qubits not listed carry the identity; the empty string is
/// the identity operator.
///
/// The derived order (lexicographic over the pairs, so the identity is the
/// smallest) is the canonical term order of [`PauliSum::simplified`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PauliString {
    ops: Vec<(usize, PauliAxis)>,
}

impl PauliString {
    /// The identity operator.
    pub fn identity() -> Self {
        Self { ops: Vec::new() }
    }

    /// A single Pauli on one qubit.
    pub fn single(qubit: usize, axis: PauliAxis) -> Self {
        Self {
            ops: vec![(qubit, axis)],
        }
    }

    /// Builds a string from `(qubit, axis)` pairs in any order.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidPauliSum`] if a qubit appears more than once.
    pub fn new(ops: impl IntoIterator<Item = (usize, PauliAxis)>) -> Result<Self, PhysicsError> {
        let mut ops: Vec<_> = ops.into_iter().collect();
        ops.sort_by_key(|&(q, _)| q);
        if let Some(w) = ops.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(PhysicsError::InvalidPauliSum {
                message: format!("qubit {} appears more than once in a Pauli string", w[0].0),
            });
        }
        Ok(Self { ops })
    }

    /// The `(qubit, axis)` pairs, sorted by qubit.
    pub fn ops(&self) -> &[(usize, PauliAxis)] {
        &self.ops
    }

    /// `true` for the identity operator.
    pub fn is_identity(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of qubits acted on non-trivially.
    pub fn weight(&self) -> usize {
        self.ops.len()
    }

    /// Highest qubit acted on, or `None` for the identity.
    pub fn max_qubit(&self) -> Option<usize> {
        self.ops.last().map(|&(q, _)| q)
    }

    /// Product `self · other = phase · result`.
    ///
    /// ```
    /// use polypus_physics::hamiltonians::{PauliString, Phase};
    ///
    /// let x: PauliString = "X0".parse().unwrap();
    /// let y: PauliString = "Y0".parse().unwrap();
    /// assert_eq!(x.mul(&y), (Phase::PlusI, "Z0".parse().unwrap()));
    /// ```
    pub fn mul(&self, other: &PauliString) -> (Phase, PauliString) {
        let mut phase = Phase::PlusOne;
        let mut ops = Vec::with_capacity(self.ops.len() + other.ops.len());
        let (mut i, mut j) = (0, 0);
        while i < self.ops.len() && j < other.ops.len() {
            let (qa, a) = self.ops[i];
            let (qb, b) = other.ops[j];
            if qa < qb {
                ops.push((qa, a));
                i += 1;
            } else if qb < qa {
                ops.push((qb, b));
                j += 1;
            } else {
                let (p, axis) = a.mul(b);
                phase = phase * p;
                if let Some(axis) = axis {
                    ops.push((qa, axis));
                }
                i += 1;
                j += 1;
            }
        }
        ops.extend_from_slice(&self.ops[i..]);
        ops.extend_from_slice(&other.ops[j..]);
        (phase, PauliString { ops })
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ops.is_empty() {
            return write!(f, "I");
        }
        for (k, (q, axis)) in self.ops.iter().enumerate() {
            if k > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}{q}", axis.letter())?;
        }
        Ok(())
    }
}

impl FromStr for PauliString {
    type Err = PhysicsError;

    /// Parses `"X0 X1 Y2 Y3"` (tokens in any order) or `"I"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let invalid = |message: String| PhysicsError::InvalidPauliSum { message };
        let s = s.trim();
        if s == "I" {
            return Ok(Self::identity());
        }
        if s.is_empty() {
            return Err(invalid(
                "empty Pauli string (write \"I\" for the identity)".into(),
            ));
        }
        let mut ops = Vec::new();
        for token in s.split_whitespace() {
            let mut chars = token.chars();
            let axis = match chars.next() {
                Some('X') => PauliAxis::X,
                Some('Y') => PauliAxis::Y,
                Some('Z') => PauliAxis::Z,
                _ => return Err(invalid(format!("invalid Pauli token {token:?}"))),
            };
            // Canonical decimal only: no sign, no leading zeros ("X+1",
            // "Z007" are rejected), so the text form round-trips.
            let digits = chars.as_str();
            let canonical = !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit())
                && (digits == "0" || !digits.starts_with('0'));
            let qubit = canonical
                .then(|| digits.parse::<usize>().ok())
                .flatten()
                .ok_or_else(|| invalid(format!("invalid qubit index in token {token:?}")))?;
            ops.push((qubit, axis));
        }
        Self::new(ops)
    }
}

/// One term `c · P` of a [`PauliSum`].
#[derive(Debug, Clone, PartialEq)]
pub struct PauliTerm {
    /// Real coefficient, in the caller's unit convention (e.g. hartree, or eV
    /// with `ħ = 1`). A Trotter compiler multiplying by an evolution time `t`
    /// must use the same convention.
    pub coefficient: f64,
    /// The Pauli string.
    pub string: PauliString,
}

impl PauliTerm {
    /// Builds a term.
    pub fn new(coefficient: f64, string: PauliString) -> Self {
        Self {
            coefficient,
            string,
        }
    }
}

/// A Hermitian operator `H = Σ_k c_k · P_k` on `num_qubits` qubits, with real
/// coefficients.
///
/// Invariants (checked by [`PauliSum::new`]): every coefficient is finite and
/// every string acts only on qubits `< num_qubits`. Terms are kept in the
/// order given; [`PauliSum::simplified`] returns the canonical form.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliSum {
    num_qubits: usize,
    terms: Vec<PauliTerm>,
}

impl PauliSum {
    /// Builds a validated sum.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidPauliSum`] if a coefficient is not finite or a
    /// string acts on a qubit `≥ num_qubits`.
    pub fn new(num_qubits: usize, terms: Vec<PauliTerm>) -> Result<Self, PhysicsError> {
        for term in &terms {
            if !term.coefficient.is_finite() {
                return Err(PhysicsError::InvalidPauliSum {
                    message: format!(
                        "coefficient of {} is not finite: {}",
                        term.string, term.coefficient
                    ),
                });
            }
            if let Some(q) = term.string.max_qubit() {
                if q >= num_qubits {
                    return Err(PhysicsError::InvalidPauliSum {
                        message: format!(
                            "{} acts on qubit {q}, but the sum has {num_qubits} qubits",
                            term.string
                        ),
                    });
                }
            }
        }
        Ok(Self { num_qubits, terms })
    }

    /// Total number of qubits spanned by the sum.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The terms, in the order given (canonical after
    /// [`PauliSum::simplified`]).
    pub fn terms(&self) -> &[PauliTerm] {
        &self.terms
    }

    /// Sum of absolute values of all coefficients.
    ///
    /// Useful for estimating the operator norm `‖H‖` and hence the Trotter
    /// step size needed for a target error.
    pub fn coefficient_norm(&self) -> f64 {
        self.terms.iter().map(|t| t.coefficient.abs()).sum()
    }

    /// Canonical form: equal strings merged, exactly-zero coefficients
    /// dropped, terms sorted by string (identity first). The result does not
    /// depend on the input term order up to floating-point summation order of
    /// merged coefficients.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidPauliSum`] if merging overflows a coefficient
    /// to infinity (finite inputs near `f64::MAX`).
    pub fn simplified(self) -> Result<Self, PhysicsError> {
        let mut merged: BTreeMap<PauliString, f64> = BTreeMap::new();
        for term in self.terms {
            *merged.entry(term.string).or_insert(0.0) += term.coefficient;
        }
        if let Some((string, c)) = merged.iter().find(|(_, c)| !c.is_finite()) {
            return Err(PhysicsError::InvalidPauliSum {
                message: format!("merged coefficient of {string} overflows: {c}"),
            });
        }
        let terms = merged
            .into_iter()
            .filter(|&(_, c)| c != 0.0)
            .map(|(string, coefficient)| PauliTerm {
                coefficient,
                string,
            })
            .collect();
        Ok(Self {
            num_qubits: self.num_qubits,
            terms,
        })
    }

    /// Drops the terms with `|c| ≤ tol`, keeping the order of the rest.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidPauliSum`] if `tol` is negative or not finite
    /// (a NaN tolerance would otherwise drop every term).
    pub fn chop(self, tol: f64) -> Result<Self, PhysicsError> {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(PhysicsError::InvalidPauliSum {
                message: format!("chop tolerance must be finite and non-negative, got {tol}"),
            });
        }
        let terms = self
            .terms
            .into_iter()
            .filter(|t| t.coefficient.abs() > tol)
            .collect();
        Ok(Self {
            num_qubits: self.num_qubits,
            terms,
        })
    }
}

impl fmt::Display for PauliSum {
    /// One `coefficient string` pair per line, e.g. `+0.171201 Z0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, term) in self.terms.iter().enumerate() {
            if k > 0 {
                writeln!(f)?;
            }
            write!(f, "{:+} {}", term.coefficient, term.string)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ps(s: &str) -> PauliString {
        s.parse().unwrap()
    }

    #[test]
    fn single_qubit_products_follow_the_pauli_algebra() {
        let cases = [
            ("X0", "Y0", Phase::PlusI, "Z0"),
            ("Y0", "Z0", Phase::PlusI, "X0"),
            ("Z0", "X0", Phase::PlusI, "Y0"),
            ("Y0", "X0", Phase::MinusI, "Z0"),
            ("Z0", "Y0", Phase::MinusI, "X0"),
            ("X0", "Z0", Phase::MinusI, "Y0"),
            ("X0", "X0", Phase::PlusOne, "I"),
            ("Y0", "Y0", Phase::PlusOne, "I"),
            ("Z0", "Z0", Phase::PlusOne, "I"),
        ];
        for (a, b, phase, result) in cases {
            assert_eq!(ps(a).mul(&ps(b)), (phase, ps(result)), "{a} · {b}");
        }
    }

    #[test]
    fn multi_qubit_product_accumulates_phases_and_merges_supports() {
        // (X0 Y1)(Y0 X1 Z2) = (X·Y)⊗(Y·X)⊗Z = (iZ)⊗(−iZ)⊗Z = Z0 Z1 Z2.
        assert_eq!(
            ps("X0 Y1").mul(&ps("Y0 X1 Z2")),
            (Phase::PlusOne, ps("Z0 Z1 Z2"))
        );
        // Disjoint supports just concatenate in qubit order.
        assert_eq!(ps("Z3").mul(&ps("X1")), (Phase::PlusOne, ps("X1 Z3")));
        assert_eq!(ps("I").mul(&ps("Y2")), (Phase::PlusOne, ps("Y2")));
    }

    #[test]
    fn phase_multiplication_is_cyclic() {
        assert_eq!(Phase::PlusI * Phase::PlusI, Phase::MinusOne);
        assert_eq!(Phase::MinusI * Phase::PlusI, Phase::PlusOne);
        assert_eq!(Phase::MinusOne * Phase::MinusI, Phase::PlusI);
        assert_eq!(Phase::MinusI.to_complex(), (0.0, -1.0));
    }

    #[test]
    fn text_form_round_trips_and_is_canonical() {
        for s in ["I", "Z0", "X0 X1 Y2 Y3", "Z1 X10"] {
            assert_eq!(ps(s).to_string(), s);
        }
        assert_eq!(ps("Y2 X0").to_string(), "X0 Y2");
        assert_eq!(ps("  Z1  ").to_string(), "Z1");
    }

    #[test]
    fn malformed_text_is_rejected() {
        for s in ["", "I0", "X", "W1", "X-1", "X0 Z0", "x0", "I Z1"] {
            assert!(
                matches!(
                    s.parse::<PauliString>(),
                    Err(PhysicsError::InvalidPauliSum { .. })
                ),
                "{s:?} should be rejected"
            );
        }
    }

    #[test]
    fn non_canonical_qubit_indices_are_rejected() {
        for s in ["X+1", "Z007", "Y00", "X 1", "X1.0", "Z-0"] {
            assert!(
                s.parse::<PauliString>().is_err(),
                "{s:?} should be rejected"
            );
        }
        assert_eq!(ps("Z0 X10").to_string(), "Z0 X10");
    }

    #[test]
    fn simplified_rejects_overflowing_merges() {
        let h = PauliSum::new(
            1,
            vec![
                PauliTerm::new(f64::MAX, ps("Z0")),
                PauliTerm::new(f64::MAX, ps("Z0")),
            ],
        )
        .unwrap();
        assert!(matches!(
            h.simplified(),
            Err(PhysicsError::InvalidPauliSum { .. })
        ));
    }

    #[test]
    fn chop_rejects_invalid_tolerances() {
        let h = PauliSum::new(1, vec![PauliTerm::new(1.0, ps("Z0"))]).unwrap();
        for tol in [f64::NAN, -1e-10, f64::INFINITY] {
            assert!(
                matches!(
                    h.clone().chop(tol),
                    Err(PhysicsError::InvalidPauliSum { .. })
                ),
                "{tol}"
            );
        }
        assert_eq!(h.clone().chop(0.0).unwrap(), h);
    }

    #[test]
    fn repeated_qubit_is_rejected() {
        let err = PauliString::new([(1, PauliAxis::X), (1, PauliAxis::Z)]).unwrap_err();
        assert!(err.to_string().contains("qubit 1"));
    }

    #[test]
    fn sum_rejects_out_of_range_qubits_and_non_finite_coefficients() {
        let out_of_range = PauliSum::new(2, vec![PauliTerm::new(1.0, ps("Z2"))]);
        assert!(matches!(
            out_of_range,
            Err(PhysicsError::InvalidPauliSum { .. })
        ));
        for c in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let r = PauliSum::new(1, vec![PauliTerm::new(c, ps("Z0"))]);
            assert!(matches!(r, Err(PhysicsError::InvalidPauliSum { .. })));
        }
        assert!(PauliSum::new(0, vec![PauliTerm::new(-0.5, ps("I"))]).is_ok());
    }

    #[test]
    fn simplified_merges_drops_zeros_and_sorts() {
        let h = PauliSum::new(
            3,
            vec![
                PauliTerm::new(0.5, ps("Z1")),
                PauliTerm::new(0.25, ps("X0 X2")),
                PauliTerm::new(-0.5, ps("Z1")),
                PauliTerm::new(1.0, ps("I")),
                PauliTerm::new(0.75, ps("X0 X2")),
                PauliTerm::new(0.0, ps("Y2")),
                PauliTerm::new(2.0, ps("Z0")),
            ],
        )
        .unwrap()
        .simplified()
        .unwrap();
        let got: Vec<_> = h
            .terms()
            .iter()
            .map(|t| (t.string.to_string(), t.coefficient))
            .collect();
        assert_eq!(
            got,
            vec![
                ("I".to_string(), 1.0),
                ("X0 X2".to_string(), 1.0),
                ("Z0".to_string(), 2.0),
            ]
        );
    }

    #[test]
    fn simplified_is_independent_of_input_order() {
        let terms = vec![
            PauliTerm::new(0.1, ps("Z0 Z1")),
            PauliTerm::new(-0.2, ps("X0 X1")),
            PauliTerm::new(0.3, ps("I")),
            PauliTerm::new(0.4, ps("Y1")),
        ];
        let mut reversed = terms.clone();
        reversed.reverse();
        let a = PauliSum::new(2, terms).unwrap().simplified().unwrap();
        let b = PauliSum::new(2, reversed).unwrap().simplified().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn chop_drops_small_terms_and_keeps_order() {
        let h = PauliSum::new(
            1,
            vec![
                PauliTerm::new(1e-12, ps("X0")),
                PauliTerm::new(-0.5, ps("Z0")),
                PauliTerm::new(0.3, ps("I")),
            ],
        )
        .unwrap()
        .chop(1e-10)
        .unwrap();
        let strings: Vec<_> = h.terms().iter().map(|t| t.string.to_string()).collect();
        assert_eq!(strings, ["Z0", "I"]);
    }

    #[test]
    fn display_lists_one_term_per_line() {
        let h = PauliSum::new(
            2,
            vec![
                PauliTerm::new(0.5, ps("Z0")),
                PauliTerm::new(-0.25, ps("X0 X1")),
            ],
        )
        .unwrap();
        assert_eq!(h.to_string(), "+0.5 Z0\n-0.25 X0 X1");
    }
}
