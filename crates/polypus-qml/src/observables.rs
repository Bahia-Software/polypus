//! Pauli observables and expectation estimation from measurement counts.
//!
//! The readout of a QML model is a sum of weighted Pauli strings (design doc
//! §7). This module carries the *public, unresolved* types the user builds
//! ([`Pauli`], [`PauliString`], [`Observable`]) and the *internal, resolved*
//! counterparts ([`ResolvedPauliString`], [`ResolvedObservable`]) that
//! [`compile`](crate::QuantumModel::compile) produces once, with logical qubit
//! positions already mapped to physical indices.
//!
//! Why a bespoke type and not `polypus-physics`' `PauliSum`: that crate's
//! `PauliTerm` is a single Pauli on a single qubit and cannot express a
//! multi-qubit string such as `Z₀Z₁` (parity — the most common QCNN
//! observable). See design doc §7.1 / D4.
//!
//! ## Positions
//!
//! A [`PauliString`] addresses **logical positions** over the model's final
//! active qubits (design doc §6): position 0 is "the first qubit still alive".
//! `compile` resolves those to physical indices and stores the resolved form
//! in the [`CompiledModel`](crate::CompiledModel); the resolved types below
//! therefore hold physical indices and skip revalidation — their invariants
//! are guaranteed by construction.

use std::collections::HashMap;

use crate::error::{QmlError, ValidationError};

/// A single-qubit Pauli operator. v1 readout only supports `Z`; `X`/`Y` are
/// carried so the type is complete and so `compile` can reject them with a
/// clear [`ValidationError::UnsupportedPauli`] until the base-grouping phase
/// (design doc §7.2) lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    /// Pauli-X.
    X,
    /// Pauli-Y.
    Y,
    /// Pauli-Z.
    Z,
}

/// A tensor product of Pauli operators on distinct qubit positions, e.g.
/// `Z₀Z₁`. Positions are **logical** over the model's final active qubits.
///
/// The invariant "positions are unique and sorted" is guaranteed by the type:
/// [`PauliString::new`] rejects duplicates and sorts the factors, so no other
/// code has to re-check it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString(Vec<(usize, Pauli)>);

impl PauliString {
    /// Build a Pauli string from `(position, pauli)` factors.
    ///
    /// Rejects two factors on the same position with
    /// [`ValidationError::DuplicatePauliPosition`]; otherwise sorts the factors
    /// by position so the stored form is canonical.
    pub fn new(terms: Vec<(usize, Pauli)>) -> Result<Self, ValidationError> {
        let mut terms = terms;
        terms.sort_by_key(|&(position, _)| position);
        for window in terms.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(ValidationError::DuplicatePauliPosition {
                    position: window[0].0,
                });
            }
        }
        Ok(PauliString(terms))
    }

    /// The factors, sorted by position (crate-internal: `compile` reads them to
    /// resolve positions to physical indices).
    pub(crate) fn terms(&self) -> &[(usize, Pauli)] {
        &self.0
    }
}

/// A Hermitian observable `O = Σ_k c_k · P_k`: a real-weighted sum of Pauli
/// strings.
#[derive(Debug, Clone, PartialEq)]
pub struct Observable {
    /// The weighted Pauli-string terms, `(coefficient, string)`.
    ///
    /// Note: mutating this field after construction can reintroduce a
    /// non-finite coefficient without passing through [`Observable::new`]'s
    /// validation; prefer rebuilding the observable over mutating in place.
    pub terms: Vec<(f64, PauliString)>,
}

impl Observable {
    /// Build an observable from weighted Pauli strings.
    ///
    /// Rejects a non-finite coefficient (`NaN`/infinite) with
    /// [`ValidationError::NonFiniteCoefficient`], reporting the first offending
    /// term — mirroring C-2's uniform "no non-finite parameter" policy.
    pub fn new(terms: Vec<(f64, PauliString)>) -> Result<Self, ValidationError> {
        for (term_index, (coeff, _)) in terms.iter().enumerate() {
            if !coeff.is_finite() {
                return Err(ValidationError::NonFiniteCoefficient { term_index });
            }
        }
        Ok(Observable { terms })
    }
}

/// A Pauli string with **physical** qubit indices, produced by `compile`. No
/// validation on construction: the positions come from resolving an already
/// validated [`PauliString`] against the model's active qubits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedPauliString(Vec<(usize, Pauli)>);

impl ResolvedPauliString {
    /// Wrap already-resolved `(physical_index, pauli)` factors.
    pub(crate) fn new(terms: Vec<(usize, Pauli)>) -> Self {
        ResolvedPauliString(terms)
    }
}

/// An observable with physical indices, produced by `compile`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedObservable {
    terms: Vec<(f64, ResolvedPauliString)>,
}

impl ResolvedObservable {
    /// Wrap already-resolved weighted terms.
    pub(crate) fn new(terms: Vec<(f64, ResolvedPauliString)>) -> Self {
        ResolvedObservable { terms }
    }

    /// Estimate `⟨O⟩ = Σ_k c_k · ⟨P_k⟩` from measurement `counts`.
    pub(crate) fn expectation(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        let mut sum = 0.0;
        for (coeff, string) in &self.terms {
            sum += coeff * expectation_from_counts(counts, string)?;
        }
        Ok(sum)
    }
}

/// Estimate `⟨Z_S⟩` from measurement `counts` for a resolved Pauli string `S`
/// (which is `Z`-only in v1, so the estimator is the computational-basis parity
/// estimator `Σ_b counts(b)·(−1)^{parity of b over S} / shots`).
///
/// Respects the C-3 counts format: keys are Qiskit little-endian bitstrings, so
/// physical qubit `pos` is the character at index `width − 1 − pos`. The empty
/// string `S` (identity) has parity 0 for every basis state, so its expectation
/// is `1.0`.
///
/// # Errors
///
/// - [`QmlError::EmptyCounts`] if `counts` is empty or records zero total shots.
/// - [`QmlError::CountsWidthMismatch`] if the keys are not all the same width,
///   or if `string` references a position wider than the keys carry.
pub(crate) fn expectation_from_counts(
    counts: &HashMap<String, u64>,
    string: &ResolvedPauliString,
) -> Result<f64, QmlError> {
    // Width is derived from the first key; an empty map has none.
    let width = match counts.keys().next() {
        Some(key) => key.len(),
        None => return Err(QmlError::EmptyCounts),
    };
    // Every key must share that width.
    for key in counts.keys() {
        if key.len() != width {
            return Err(QmlError::CountsWidthMismatch {
                expected: width,
                got: key.len(),
            });
        }
    }
    // Every referenced position must fit inside the register.
    for &(position, _) in &string.0 {
        if position >= width {
            return Err(QmlError::CountsWidthMismatch {
                expected: position + 1,
                got: width,
            });
        }
    }

    let mut weighted = 0.0;
    let mut shots: u64 = 0;
    for (key, &count) in counts {
        let bytes = key.as_bytes();
        let mut parity = 0u32;
        for &(position, _) in &string.0 {
            // C-3: qubit `position` is the character at `width - 1 - position`.
            if bytes[width - 1 - position] == b'1' {
                parity ^= 1;
            }
        }
        let sign = if parity == 0 { 1.0 } else { -1.0 };
        weighted += sign * count as f64;
        shots += count;
    }
    if shots == 0 {
        return Err(QmlError::EmptyCounts);
    }
    Ok(weighted / shots as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn pauli_string_sorts_and_rejects_duplicates() {
        let s = PauliString::new(vec![(2, Pauli::Z), (0, Pauli::Z)]).unwrap();
        assert_eq!(s.terms(), &[(0, Pauli::Z), (2, Pauli::Z)]);

        let err = PauliString::new(vec![(1, Pauli::Z), (1, Pauli::X)]).unwrap_err();
        assert_eq!(err, ValidationError::DuplicatePauliPosition { position: 1 });
    }

    #[test]
    fn observable_rejects_non_finite_coefficient() {
        let s = PauliString::new(vec![(0, Pauli::Z)]).unwrap();
        let err = Observable::new(vec![(1.0, s.clone()), (f64::NAN, s.clone())]).unwrap_err();
        assert_eq!(err, ValidationError::NonFiniteCoefficient { term_index: 1 });

        let err = Observable::new(vec![(f64::INFINITY, s)]).unwrap_err();
        assert_eq!(err, ValidationError::NonFiniteCoefficient { term_index: 0 });
    }

    #[test]
    fn empty_counts_rejected() {
        let string = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        assert_eq!(
            expectation_from_counts(&HashMap::new(), &string),
            Err(QmlError::EmptyCounts)
        );
        // All-zero shots is treated the same: nothing to estimate from.
        assert_eq!(
            expectation_from_counts(&counts(&[("0", 0)]), &string),
            Err(QmlError::EmptyCounts)
        );
    }

    #[test]
    fn width_mismatch_across_keys() {
        let string = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        let err = expectation_from_counts(&counts(&[("00", 1), ("0", 1)]), &string).unwrap_err();
        assert!(matches!(err, QmlError::CountsWidthMismatch { .. }));
    }

    #[test]
    fn position_wider_than_register_rejected() {
        let string = ResolvedPauliString::new(vec![(3, Pauli::Z)]);
        let err = expectation_from_counts(&counts(&[("00", 1)]), &string).unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsWidthMismatch {
                expected: 4,
                got: 2,
            }
        );
    }

    #[test]
    fn single_z_parity_estimator() {
        // ⟨Z_0⟩ over one qubit: "0" → +1, "1" → −1.
        let z0 = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        // All |0⟩: +1.
        assert_eq!(expectation_from_counts(&counts(&[("0", 100)]), &z0), Ok(1.0));
        // All |1⟩: −1.
        assert_eq!(expectation_from_counts(&counts(&[("1", 100)]), &z0), Ok(-1.0));
        // Even split: 0.
        let e = expectation_from_counts(&counts(&[("0", 50), ("1", 50)]), &z0).unwrap();
        assert!((e - 0.0).abs() < 1e-12);
    }

    #[test]
    fn z_reads_little_endian_position() {
        // Width-2 keys, ⟨Z_1⟩: qubit 1 is the *left* character (index 0).
        let z1 = ResolvedPauliString::new(vec![(1, Pauli::Z)]);
        // "10" → qubit 1 is '1' → −1; "01" → qubit 1 is '0' → +1.
        assert_eq!(expectation_from_counts(&counts(&[("10", 10)]), &z1), Ok(-1.0));
        assert_eq!(expectation_from_counts(&counts(&[("01", 10)]), &z1), Ok(1.0));
    }

    #[test]
    fn zz_parity_estimator() {
        // ⟨Z_0 Z_1⟩: parity over both qubits. "00","11" → +1; "01","10" → −1.
        let zz = ResolvedPauliString::new(vec![(0, Pauli::Z), (1, Pauli::Z)]);
        let c = counts(&[("00", 25), ("11", 25), ("01", 25), ("10", 25)]);
        let e = expectation_from_counts(&c, &zz).unwrap();
        assert!(e.abs() < 1e-12);
        assert_eq!(
            expectation_from_counts(&counts(&[("00", 5), ("11", 5)]), &zz),
            Ok(1.0)
        );
    }

    #[test]
    fn identity_string_has_expectation_one() {
        let identity = ResolvedPauliString::new(vec![]);
        assert_eq!(
            expectation_from_counts(&counts(&[("01", 3), ("10", 7)]), &identity),
            Ok(1.0)
        );
    }

    #[test]
    fn resolved_observable_sums_weighted_terms() {
        // 0.5·Z_0 + 2.0·I over "0" (⟨Z_0⟩=1, ⟨I⟩=1) = 2.5.
        let obs = ResolvedObservable::new(vec![
            (0.5, ResolvedPauliString::new(vec![(0, Pauli::Z)])),
            (2.0, ResolvedPauliString::new(vec![])),
        ]);
        let e = obs.expectation(&counts(&[("0", 100)])).unwrap();
        assert!((e - 2.5).abs() < 1e-12);
    }
}
