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

/// A single-qubit Pauli operator. All three bases are measurable: `compile`
/// inserts the basis change before the terminal measurement (`H` for `X`; `Sdg`
/// then `H` for `Y`), provided the whole readout resolves to a single basis
/// group. A readout that would need more than one group (e.g. `Z` and `X` on the
/// same qubit) is still rejected — that is the multi-circuit case, not
/// implemented yet (design doc §7.2, [`ValidationError::ReadoutNeedsMultipleBasisGroups`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        let string = PauliString(terms);
        string.validate()?;
        Ok(string)
    }

    /// Re-check the invariant [`new`](Self::new) established: no two factors on
    /// the same position.
    ///
    /// `new` is not the only way to obtain a `PauliString`: the `serde`
    /// `Deserialize` derive builds one straight from the wire, bypassing `new`
    /// entirely. `compile` therefore re-runs this check on every string it
    /// resolves, so a hand-tampered save file cannot smuggle in `Z₀Z₀`
    /// (whose parity is counted twice and always reads `+1`) as a silently
    /// wrong answer.
    ///
    /// Uniqueness only — not sortedness. Sorting is canonicalisation, not a
    /// correctness invariant: every consumer of the factors (parity, basis
    /// resolution) is order-independent.
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
        let mut positions: Vec<usize> = self.0.iter().map(|&(position, _)| position).collect();
        positions.sort_unstable();
        for window in positions.windows(2) {
            if window[0] == window[1] {
                return Err(ValidationError::DuplicatePauliPosition {
                    position: window[0],
                });
            }
        }
        Ok(())
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Observable {
    /// The weighted Pauli-string terms, `(coefficient, string)`.
    ///
    /// Public and mutable, so a caller can reintroduce a non-finite coefficient
    /// without passing through [`Observable::new`]'s validation. That is caught
    /// rather than trusted: [`compile`](crate::QuantumModel::compile) re-runs
    /// [`Observable::validate`] on every observable it resolves, so a mutated
    /// (or deserialized) observable fails compilation with a typed
    /// [`ValidationError`] instead of reaching inference. Prefer rebuilding the
    /// observable over mutating in place all the same.
    pub terms: Vec<(f64, PauliString)>,
}

impl Observable {
    /// Build an observable from weighted Pauli strings.
    ///
    /// Rejects a non-finite coefficient (`NaN`/infinite) with
    /// [`ValidationError::NonFiniteCoefficient`], reporting the first offending
    /// term — mirroring C-2's uniform "no non-finite parameter" policy.
    pub fn new(terms: Vec<(f64, PauliString)>) -> Result<Self, ValidationError> {
        let observable = Observable { terms };
        observable.validate()?;
        Ok(observable)
    }

    /// Re-check the invariants [`new`](Self::new) established: every
    /// coefficient finite, and every Pauli string free of duplicate positions.
    ///
    /// `new` is not the only way to obtain an `Observable`: [`terms`](Self::terms)
    /// is public and mutable, and the `serde` `Deserialize` derive builds one
    /// straight from the wire. [`compile`](crate::QuantumModel::compile) calls
    /// this on every observable it resolves so those two routes cannot produce
    /// a `CompiledModel` that violates C-10(b) — a `NaN` coefficient would
    /// otherwise make `fitness_from_counts` return `Ok(NaN)` instead of the
    /// finite `f64` or typed error the contract promises.
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
        for (term_index, (coeff, string)) in self.terms.iter().enumerate() {
            if !coeff.is_finite() {
                return Err(ValidationError::NonFiniteCoefficient { term_index });
            }
            string.validate()?;
        }
        Ok(())
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

    /// Compute `⟨O⟩ = Σ_k c_k · ⟨P_k⟩` from **exact** basis-state
    /// `probabilities` (the exact-mode mirror of
    /// [`expectation`](Self::expectation)).
    pub(crate) fn expectation_from_probabilities(
        &self,
        probabilities: &HashMap<String, f64>,
    ) -> Result<f64, QmlError> {
        let mut sum = 0.0;
        for (coeff, string) in &self.terms {
            sum += coeff * expectation_from_probabilities(probabilities, string)?;
        }
        Ok(sum)
    }
}

/// A measurement-weight type: finite-shot `u64` counts or exact `f64`
/// basis-state probabilities. It is what lets the two readout paths — sampled
/// (`counts`) and exact (`probabilities`) — share one estimator instead of
/// carrying byte-for-byte duplicate logic that differs only in this type.
///
/// [`as_weight`](Self::as_weight) converts one bucket's weight to the `f64` the
/// parity accumulator sums. [`observable_expectation`](Self::observable_expectation)
/// dispatches an *observable*-level expectation to the matching concrete
/// estimator, so the higher layers ([`ResolvedReadout`](crate::readout::ResolvedReadout)
/// prediction, [`QmlProblem`](crate::QmlProblem) fitness/gradients) stay generic
/// over the weight type while each concrete path keeps its own `pub(crate)`
/// entry point ([`ResolvedObservable::expectation`] /
/// [`ResolvedObservable::expectation_from_probabilities`]) as the single place it
/// is spelled out.
pub(crate) trait BitstringWeight: Copy {
    /// This bucket's contribution to the running weight, as `f64`.
    fn as_weight(self) -> f64;

    /// Estimate `⟨O⟩` for `observable` from a weight map of this type, routing to
    /// the concrete estimator (counts vs probabilities) for the type.
    fn observable_expectation(
        observable: &ResolvedObservable,
        weights: &HashMap<String, Self>,
    ) -> Result<f64, QmlError>;
}

impl BitstringWeight for u64 {
    fn as_weight(self) -> f64 {
        self as f64
    }

    fn observable_expectation(
        observable: &ResolvedObservable,
        weights: &HashMap<String, Self>,
    ) -> Result<f64, QmlError> {
        observable.expectation(weights)
    }
}

impl BitstringWeight for f64 {
    fn as_weight(self) -> f64 {
        self
    }

    fn observable_expectation(
        observable: &ResolvedObservable,
        weights: &HashMap<String, Self>,
    ) -> Result<f64, QmlError> {
        observable.expectation_from_probabilities(weights)
    }
}

/// Estimate `⟨Z_S⟩` from a `weights` map for a resolved Pauli string `S` — the
/// single implementation shared by the finite-shot ([`expectation_from_counts`],
/// `u64`) and exact ([`expectation_from_probabilities`], `f64`) estimators, which
/// differ only in the weight type. `S` is `Z`-only in v1, so this is the
/// computational-basis parity estimator `Σ_b w(b)·(−1)^{parity of b over S} /
/// Σ_b w(b)`.
///
/// Respects the C-3 counts format: keys are Qiskit little-endian bitstrings, so
/// physical qubit `pos` is the character at index `width − 1 − pos`. The empty
/// string `S` (identity) has parity 0 for every basis state, so its expectation
/// is `1.0`. The divisor is the sum of the weights present (`shots` for counts,
/// `≈ 1.0` for a normalised statevector), summed explicitly rather than assumed.
///
/// The buckets are summed in a deterministic order — keys sorted (fixed-width
/// bitstrings, so lexicographic == basis order) — for **both** weight types.
/// This is required for `f64`, where floating-point addition is not associative
/// and a per-instance-randomised `HashMap` order would otherwise make two
/// identical calls differ by a ULP, breaking the byte-for-byte reproducibility
/// the exact `qml.train` path promises (C-7). For `u64` it changes nothing
/// numerically: every `sign · count` and every partial sum is an exactly
/// representable integer (`< 2^53` in practice), so integer addition in `f64` is
/// exact and order-independent — sorting there is a harmless extra allocation,
/// not a behaviour change. Ordering is therefore unconditional rather than
/// branched on the type.
///
/// # Errors
///
/// - [`QmlError::EmptyCounts`] if `weights` is empty or its weights sum to zero.
///   The variant is reused for the exact path too: the conceptual failure
///   ("nothing to estimate from") is the same, even though the message names
///   "counts".
/// - [`QmlError::CountsWidthMismatch`] if the keys are not all the same width,
///   or if `string` references a position wider than the keys carry.
fn expectation_from_weighted<W: BitstringWeight>(
    weights: &HashMap<String, W>,
    string: &ResolvedPauliString,
) -> Result<f64, QmlError> {
    if weights.is_empty() {
        return Err(QmlError::EmptyCounts);
    }
    // Fix the summation order up front (see the doc comment): deterministic for
    // the exact path's reproducibility, harmless for the exact-integer counts
    // path. Width is then derived from the deterministic first key.
    let mut entries: Vec<(&String, W)> = weights.iter().map(|(k, &w)| (k, w)).collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let width = entries[0].0.len();
    // Every key must share that width.
    for (key, _) in &entries {
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
    let mut total = 0.0;
    for (key, w) in entries {
        let bytes = key.as_bytes();
        let mut parity = 0u32;
        for &(position, _) in &string.0 {
            // C-3: qubit `position` is the character at `width - 1 - position`.
            if bytes[width - 1 - position] == b'1' {
                parity ^= 1;
            }
        }
        let weight = w.as_weight();
        let sign = if parity == 0 { 1.0 } else { -1.0 };
        weighted += sign * weight;
        total += weight;
    }
    if total == 0.0 {
        return Err(QmlError::EmptyCounts);
    }
    Ok(weighted / total)
}

/// Estimate `⟨Z_S⟩` from measurement `counts` for a resolved Pauli string `S`.
/// A thin `u64` wrapper over [`expectation_from_weighted`]; see it for the parity
/// estimator, the C-3 bit order, and the error contract.
pub(crate) fn expectation_from_counts(
    counts: &HashMap<String, u64>,
    string: &ResolvedPauliString,
) -> Result<f64, QmlError> {
    expectation_from_weighted(counts, string)
}

/// The exact-mode mirror of [`expectation_from_counts`]: estimate `⟨Z_S⟩` from
/// exact basis-state `probabilities` (`|amplitude|²` per bitstring) instead of
/// finite-shot counts. A thin `f64` wrapper over [`expectation_from_weighted`],
/// which sums both paths in the same deterministic basis order.
pub(crate) fn expectation_from_probabilities(
    probabilities: &HashMap<String, f64>,
    string: &ResolvedPauliString,
) -> Result<f64, QmlError> {
    expectation_from_weighted(probabilities, string)
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
        assert_eq!(
            expectation_from_counts(&counts(&[("0", 100)]), &z0),
            Ok(1.0)
        );
        // All |1⟩: −1.
        assert_eq!(
            expectation_from_counts(&counts(&[("1", 100)]), &z0),
            Ok(-1.0)
        );
        // Even split: 0.
        let e = expectation_from_counts(&counts(&[("0", 50), ("1", 50)]), &z0).unwrap();
        assert!((e - 0.0).abs() < 1e-12);
    }

    #[test]
    fn z_reads_little_endian_position() {
        // Width-2 keys, ⟨Z_1⟩: qubit 1 is the *left* character (index 0).
        let z1 = ResolvedPauliString::new(vec![(1, Pauli::Z)]);
        // "10" → qubit 1 is '1' → −1; "01" → qubit 1 is '0' → +1.
        assert_eq!(
            expectation_from_counts(&counts(&[("10", 10)]), &z1),
            Ok(-1.0)
        );
        assert_eq!(
            expectation_from_counts(&counts(&[("01", 10)]), &z1),
            Ok(1.0)
        );
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

    // ── Exact-mode mirror of the counts catalogue above ──────────────────────

    fn probs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn empty_probabilities_rejected() {
        let string = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        assert_eq!(
            expectation_from_probabilities(&HashMap::new(), &string),
            Err(QmlError::EmptyCounts)
        );
        // All-zero weights are treated the same: nothing to estimate from.
        assert_eq!(
            expectation_from_probabilities(&probs(&[("0", 0.0)]), &string),
            Err(QmlError::EmptyCounts)
        );
    }

    #[test]
    fn width_mismatch_across_probability_keys() {
        let string = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        let err = expectation_from_probabilities(&probs(&[("00", 0.5), ("0", 0.5)]), &string)
            .unwrap_err();
        assert!(matches!(err, QmlError::CountsWidthMismatch { .. }));
    }

    #[test]
    fn probability_position_wider_than_register_rejected() {
        let string = ResolvedPauliString::new(vec![(3, Pauli::Z)]);
        let err = expectation_from_probabilities(&probs(&[("00", 1.0)]), &string).unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsWidthMismatch {
                expected: 4,
                got: 2,
            }
        );
    }

    #[test]
    fn single_z_parity_estimator_from_probabilities() {
        // ⟨Z_0⟩ over one qubit: "0" → +1, "1" → −1.
        let z0 = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        // All weight on |0⟩: +1.
        assert_eq!(
            expectation_from_probabilities(&probs(&[("0", 1.0)]), &z0),
            Ok(1.0)
        );
        // All weight on |1⟩: −1.
        assert_eq!(
            expectation_from_probabilities(&probs(&[("1", 1.0)]), &z0),
            Ok(-1.0)
        );
        // Even split: 0.
        let e = expectation_from_probabilities(&probs(&[("0", 0.5), ("1", 0.5)]), &z0).unwrap();
        assert!(e.abs() < 1e-12);
    }

    #[test]
    fn zz_parity_estimator_from_probabilities() {
        // ⟨Z_0 Z_1⟩: parity over both qubits. "00","11" → +1; "01","10" → −1.
        let zz = ResolvedPauliString::new(vec![(0, Pauli::Z), (1, Pauli::Z)]);
        let p = probs(&[("00", 0.25), ("11", 0.25), ("01", 0.25), ("10", 0.25)]);
        let e = expectation_from_probabilities(&p, &zz).unwrap();
        assert!(e.abs() < 1e-12);
        assert_eq!(
            expectation_from_probabilities(&probs(&[("00", 0.5), ("11", 0.5)]), &zz),
            Ok(1.0)
        );
    }

    #[test]
    fn identity_string_has_probability_expectation_one() {
        let identity = ResolvedPauliString::new(vec![]);
        assert_eq!(
            expectation_from_probabilities(&probs(&[("01", 0.3), ("10", 0.7)]), &identity),
            Ok(1.0)
        );
    }

    #[test]
    fn probabilities_normalise_over_present_weights() {
        // Weights that do not sum to 1 are still normalised by their own total:
        // ⟨Z_0⟩ over {"0": 0.3, "1": 0.1} = (0.3 − 0.1)/0.4 = 0.5.
        let z0 = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        let e = expectation_from_probabilities(&probs(&[("0", 0.3), ("1", 0.1)]), &z0).unwrap();
        assert!((e - 0.5).abs() < 1e-12);
    }

    #[test]
    fn probability_expectation_sums_in_deterministic_basis_order() {
        // A many-outcome distribution with values chosen so naive HashMap-order
        // summation would round differently from run to run. The estimator must
        // equal the reference computed by summing in explicit basis order — the
        // property the exact path's byte-for-byte reproducibility rests on.
        let z0 = ResolvedPauliString::new(vec![(0, Pauli::Z)]);
        let width = 3usize;
        let raw: Vec<f64> = (0..8).map(|i| 0.03 + 0.017 * (i as f64).sin()).collect();
        let sum: f64 = raw.iter().sum();
        let map: HashMap<String, f64> = raw
            .iter()
            .enumerate()
            .map(|(state, &p)| (format!("{state:0width$b}"), p))
            .collect();

        // Reference: sum sign·p in ascending basis order (matching the sort).
        let mut weighted = 0.0;
        for (state, &p) in raw.iter().enumerate() {
            let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };
            weighted += sign * p;
        }
        let reference = weighted / sum;

        // Bit-for-bit equality (not a tolerance): the order must match exactly.
        assert_eq!(
            expectation_from_probabilities(&map, &z0).unwrap(),
            reference
        );
    }

    #[test]
    fn resolved_observable_sums_weighted_terms_from_probabilities() {
        // 0.5·Z_0 + 2.0·I over "0" (⟨Z_0⟩=1, ⟨I⟩=1) = 2.5.
        let obs = ResolvedObservable::new(vec![
            (0.5, ResolvedPauliString::new(vec![(0, Pauli::Z)])),
            (2.0, ResolvedPauliString::new(vec![])),
        ]);
        let e = obs
            .expectation_from_probabilities(&probs(&[("0", 1.0)]))
            .unwrap();
        assert!((e - 2.5).abs() < 1e-12);
    }
}
