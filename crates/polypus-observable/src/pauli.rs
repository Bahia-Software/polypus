//! Measuring a sum of multi-qubit Pauli strings from computational-basis
//! counts: qubit-wise-commuting grouping and a parity reducer (issue #530).
//!
//! Hardware measures every qubit in the `Z` basis. A term with `X` or `Y`
//! factors is measured by rotating those qubits first (`H` for `X`; `S†` then
//! `H` for `Y`), after which the term reads as a product of `Z`s on the same
//! qubits. Terms that agree on the Pauli of every qubit they share
//! (*qubit-wise commuting*) share one rotation, hence one circuit:
//! [`group_qubit_wise`] partitions a sum into such groups, and each group's
//! [`ParityObservable`] turns that circuit's counts into the group's
//! contribution. The energy is [`Grouping::constant`] plus the sum of the
//! groups' values.
//!
//! ## Design decisions (recorded here, as issue #530 asks)
//!
//! 1. **Own bitmask type, not `polypus-physics`' `PauliSum`.** A
//!    [`PauliMask`] stores a string as two `u64` masks (`X → (1, 0)`,
//!    `Z → (0, 1)`, `Y → (1, 1)` per qubit), so qubit-wise commutation and
//!    parity are a few bit operations. The price is a limit of
//!    [`MAX_QUBITS`] = 64 qubits. This crate does not depend on
//!    `polypus-physics`; converting a `PauliSum` term by term is the caller's
//!    job (the oracle of #424), and the physics crate is a test-only
//!    dependency here.
//! 2. **Not unified with `polypus-qml`'s Pauli types.** `polypus-qml`'s
//!    `PauliString` names *logical* positions of a model, resolved to physical
//!    qubits when the model is compiled; the types here act on physical
//!    qubits. What the two crates share is the algorithms: `polypus-qml`'s
//!    `DiagonalObservable` is the same parity reduction, and its readout
//!    grouping is the same greedy qubit-wise grouping, so it can call these
//!    once its indices are physical (follow-up issue).
//! 3. **Identity terms are a constant, not a group.** Their value is the same
//!    in every state, so no shots are spent on them: [`group_qubit_wise`] sums
//!    them into [`Grouping::constant`]. A [`ParityObservable`] still accepts a
//!    mask-0 term (always `+c`), as `polypus-qml`'s `DiagonalObservable` does.
//!
//! ## Bit ordering
//!
//! Qubit `i` is bit `i` of a mask and the character at position `w - 1 - i` of
//! a counts key of width `w`, the crate-wide convention (see the crate docs).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{CostObservable, ObservableError};

/// The largest number of qubits a [`PauliMask`] or [`ParityObservable`] can
/// address: one bit of a `u64` per qubit.
pub const MAX_QUBITS: usize = 64;

/// A single-qubit Pauli operator. The identity is not a variant: it is the
/// absence of an operator on that qubit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Pauli-X, measured after `H`.
    X,
    /// Pauli-Y, measured after `S†` then `H`.
    Y,
    /// Pauli-Z, measured as is.
    Z,
}

/// A tensor product of Pauli operators on distinct qubits (`X₀Y₂Z₃`), stored
/// as two bitmasks: bit `i` of `x` is set for `X` or `Y` on qubit `i`, bit `i`
/// of `z` for `Z` or `Y`. The empty string is the identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PauliMask {
    x: u64,
    z: u64,
}

impl PauliMask {
    /// Build a string from `(qubit, pauli)` factors, in any order.
    ///
    /// # Errors
    ///
    /// [`ObservableError::Invalid`] if a qubit is `>= MAX_QUBITS`, or if a
    /// qubit appears twice: `Z₀Z₀` would count that qubit's parity twice and
    /// always read `+1`, a silently wrong answer.
    pub fn new(ops: impl IntoIterator<Item = (usize, Pauli)>) -> Result<Self, ObservableError> {
        let mut mask = PauliMask::identity();
        for (qubit, pauli) in ops {
            if qubit >= MAX_QUBITS {
                return Err(ObservableError::Invalid(format!(
                    "Pauli on qubit {qubit} exceeds the limit of {MAX_QUBITS} qubits"
                )));
            }
            let bit = 1u64 << qubit;
            if mask.support() & bit != 0 {
                return Err(ObservableError::Invalid(format!(
                    "Pauli string names qubit {qubit} twice"
                )));
            }
            match pauli {
                Pauli::X => mask.x |= bit,
                Pauli::Y => {
                    mask.x |= bit;
                    mask.z |= bit;
                }
                Pauli::Z => mask.z |= bit,
            }
        }
        Ok(mask)
    }

    /// The identity string (no operator on any qubit).
    pub fn identity() -> Self {
        PauliMask { x: 0, z: 0 }
    }

    /// Whether this is the identity.
    pub fn is_identity(&self) -> bool {
        self.support() == 0
    }

    /// The qubits with a non-identity factor, as a bitmask. After the basis
    /// rotation this is also the string's `Z`-parity mask.
    pub fn support(&self) -> u64 {
        self.x | self.z
    }

    /// The `X`-or-`Y` bitmask.
    pub fn x_bits(&self) -> u64 {
        self.x
    }

    /// The `Z`-or-`Y` bitmask.
    pub fn z_bits(&self) -> u64 {
        self.z
    }

    /// The Pauli on `qubit`, or `None` for the identity (or a qubit
    /// `>= MAX_QUBITS`).
    pub fn pauli(&self, qubit: usize) -> Option<Pauli> {
        if qubit >= MAX_QUBITS {
            return None;
        }
        match ((self.x >> qubit) & 1, (self.z >> qubit) & 1) {
            (1, 0) => Some(Pauli::X),
            (1, 1) => Some(Pauli::Y),
            (0, 1) => Some(Pauli::Z),
            _ => None,
        }
    }

    /// Whether the two strings carry the same Pauli on every qubit they both
    /// act on, so one basis rotation measures both.
    pub fn qubit_wise_commutes(&self, other: &PauliMask) -> bool {
        let shared = self.support() & other.support();
        ((self.x ^ other.x) | (self.z ^ other.z)) & shared == 0
    }
}

/// A diagonal cost `f(b) = scale · Σ_k c_k · (−1)^{popcount(b ∧ mask_k)}`: a
/// weighted sum of `Z` strings of any weight, each evaluated by the parity of
/// its qubits' bits. A mask of `0` is the identity and contributes `+c_k`.
///
/// Sign convention as [`QuboObservable`](crate::QuboObservable): optimizers
/// maximise, so build an energy to minimise with `scale = -1.0`.
#[derive(Debug, Clone)]
pub struct ParityObservable {
    num_qubits: usize,
    terms: Vec<(f64, u64)>,
    scale: f64,
}

impl ParityObservable {
    /// Build the reducer from `(coefficient, mask)` terms, bit `i` of a mask
    /// selecting qubit `i`. Terms are evaluated in the given order.
    ///
    /// # Errors
    ///
    /// [`ObservableError::Invalid`] if `num_qubits` is `0` or above
    /// [`MAX_QUBITS`], a mask selects a qubit `>= num_qubits`, or a coefficient
    /// or `scale` is not finite.
    pub fn new(
        num_qubits: usize,
        terms: Vec<(f64, u64)>,
        scale: f64,
    ) -> Result<Self, ObservableError> {
        if num_qubits == 0 || num_qubits > MAX_QUBITS {
            return Err(ObservableError::Invalid(format!(
                "num_qubits must be in 1..={MAX_QUBITS}, got {num_qubits}"
            )));
        }
        if !scale.is_finite() {
            return Err(ObservableError::Invalid("scale must be finite".to_string()));
        }
        let allowed = low_bits(num_qubits);
        for (index, &(coefficient, mask)) in terms.iter().enumerate() {
            if !coefficient.is_finite() {
                return Err(ObservableError::Invalid(format!(
                    "coefficient of term {index} is not finite"
                )));
            }
            if mask & !allowed != 0 {
                return Err(ObservableError::Invalid(format!(
                    "term {index} has mask {mask:#b}, outside {num_qubits} qubits"
                )));
            }
        }
        Ok(ParityObservable {
            num_qubits,
            terms,
            scale,
        })
    }

    /// Number of qubits the observable is defined over.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The `(coefficient, mask)` terms, in evaluation order.
    pub fn terms(&self) -> &[(f64, u64)] {
        &self.terms
    }

    /// `scale · Σ_k c_k · (−1)^{popcount(basis_state ∧ mask_k)}` for one basis
    /// state; bits above `num_qubits` are ignored.
    pub fn value(&self, basis_state: u64) -> f64 {
        let sum = self.terms.iter().fold(0.0, |acc, &(coefficient, mask)| {
            if (basis_state & mask).count_ones() % 2 == 0 {
                acc + coefficient
            } else {
                acc - coefficient
            }
        });
        sum * self.scale
    }

    /// The exact expectation `Σ_b p_b · f(b)` over a probability vector indexed
    /// by basis state (bit `i` of the index is qubit `i`, as a statevector's
    /// amplitudes are), summed in index order. Not renormalised.
    ///
    /// # Errors
    ///
    /// [`ObservableError::Invalid`] if the length is not a power of two of at
    /// least `2^num_qubits`, or a probability is negative or not finite.
    pub fn expectation_from_probabilities(
        &self,
        probabilities: &[f64],
    ) -> Result<f64, ObservableError> {
        let len = probabilities.len();
        let width = len.trailing_zeros() as usize;
        if !len.is_power_of_two() || width < self.num_qubits {
            return Err(ObservableError::Invalid(format!(
                "{len} probabilities do not cover the 2^{} basis states of the observable",
                self.num_qubits
            )));
        }
        let mut sum = 0.0;
        for (index, &p) in probabilities.iter().enumerate() {
            if !p.is_finite() || p < 0.0 {
                return Err(ObservableError::Invalid(format!(
                    "probability {p} of basis state {index} is not a probability"
                )));
            }
            sum += p * self.value(index as u64);
        }
        Ok(sum)
    }

    /// The basis state a counts key encodes, over the low `num_qubits` bits.
    fn decode(&self, key: &str) -> Result<u64, ObservableError> {
        let bytes = key.as_bytes();
        let width = bytes.len();
        if width < self.num_qubits {
            return Err(ObservableError::BitWidthMismatch {
                num_vars: self.num_qubits,
                key_len: width,
            });
        }
        let mut state = 0u64;
        for qubit in 0..self.num_qubits {
            match bytes[width - 1 - qubit] {
                b'0' => {}
                b'1' => state |= 1u64 << qubit,
                _ => return Err(ObservableError::InvalidBitstring(key.to_string())),
            }
        }
        Ok(state)
    }

    /// Count-weighted mean over one candidate's counts, in ascending bitstring
    /// order so the result does not depend on `HashMap` iteration order.
    /// Empty counts (or counts summing to zero) yield `0.0`.
    fn expectation_one(&self, counts: &HashMap<String, u64>) -> Result<f64, ObservableError> {
        let mut outcomes: Vec<(&str, u64)> =
            counts.iter().map(|(key, &n)| (key.as_str(), n)).collect();
        outcomes.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut num = 0.0f64;
        let mut den = 0u64;
        for (key, n) in outcomes {
            num += self.value(self.decode(key)?) * n as f64;
            den += n;
        }
        Ok(if den == 0 { 0.0 } else { num / den as f64 })
    }
}

impl CostObservable for ParityObservable {
    fn expectation_batch(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, ObservableError> {
        counts.par_iter().map(|c| self.expectation_one(c)).collect()
    }
}

/// One circuit's worth of terms: qubit-wise commuting, so one basis rotation
/// measures them all.
#[derive(Debug, Clone)]
pub struct MeasurementGroup {
    basis: PauliMask,
    term_indices: Vec<usize>,
    observable: ParityObservable,
}

impl MeasurementGroup {
    /// The Pauli measured on every qubit the group touches: the union of its
    /// terms, consistent because they commute qubit-wise.
    pub fn basis(&self) -> PauliMask {
        self.basis
    }

    /// The rotations to append before measuring, as the `X`/`Y` qubits of the
    /// basis in ascending order: `H` for `X`, `S†` then `H` for `Y`. `Z`
    /// qubits need none.
    pub fn rotations(&self) -> Vec<(usize, Pauli)> {
        (0..MAX_QUBITS)
            .filter_map(|qubit| match self.basis.pauli(qubit) {
                Some(Pauli::Z) | None => None,
                Some(pauli) => Some((qubit, pauli)),
            })
            .collect()
    }

    /// Indices into the input of [`group_qubit_wise`] of the terms in this
    /// group, ascending.
    pub fn term_indices(&self) -> &[usize] {
        &self.term_indices
    }

    /// The reducer for this group's rotated counts (`scale = 1`): each term's
    /// support becomes its `Z`-parity mask.
    pub fn observable(&self) -> &ParityObservable {
        &self.observable
    }
}

/// A Pauli sum split into [`MeasurementGroup`]s plus the constant of its
/// identity terms.
#[derive(Debug, Clone)]
pub struct Grouping {
    groups: Vec<MeasurementGroup>,
    constant: f64,
}

impl Grouping {
    /// The groups, in order of creation.
    pub fn groups(&self) -> &[MeasurementGroup] {
        &self.groups
    }

    /// The sum of the identity terms' coefficients, in input order. Not part
    /// of any group: its value needs no measurement.
    pub fn constant(&self) -> f64 {
        self.constant
    }
}

/// Partition `terms` into qubit-wise commuting groups, greedily: in input
/// order, each non-identity term joins the first group it commutes with
/// qubit-wise, or opens a new one. Deterministic for a given input order.
/// Identity terms go to [`Grouping::constant`], not to a group.
///
/// # Errors
///
/// [`ObservableError::Invalid`] if `num_qubits > MAX_QUBITS`, a term acts on a
/// qubit `>= num_qubits`, or a coefficient is not finite.
pub fn group_qubit_wise(
    num_qubits: usize,
    terms: &[(f64, PauliMask)],
) -> Result<Grouping, ObservableError> {
    if num_qubits > MAX_QUBITS {
        return Err(ObservableError::Invalid(format!(
            "num_qubits {num_qubits} exceeds the limit of {MAX_QUBITS}"
        )));
    }
    let allowed = low_bits(num_qubits);
    let mut constant = 0.0;
    let mut open: Vec<OpenGroup> = Vec::new();
    for (index, &(coefficient, mask)) in terms.iter().enumerate() {
        if !coefficient.is_finite() {
            return Err(ObservableError::Invalid(format!(
                "coefficient of term {index} is not finite"
            )));
        }
        if mask.support() & !allowed != 0 {
            return Err(ObservableError::Invalid(format!(
                "term {index} acts outside {num_qubits} qubits"
            )));
        }
        if mask.is_identity() {
            constant += coefficient;
            continue;
        }
        let entry = (coefficient, mask.support());
        match open
            .iter_mut()
            .find(|group| group.basis.qubit_wise_commutes(&mask))
        {
            Some(group) => {
                group.basis.x |= mask.x;
                group.basis.z |= mask.z;
                group.term_indices.push(index);
                group.terms.push(entry);
            }
            None => open.push(OpenGroup {
                basis: mask,
                term_indices: vec![index],
                terms: vec![entry],
            }),
        }
    }
    let groups = open
        .into_iter()
        .map(|group| {
            Ok(MeasurementGroup {
                basis: group.basis,
                term_indices: group.term_indices,
                observable: ParityObservable::new(num_qubits, group.terms, 1.0)?,
            })
        })
        .collect::<Result<Vec<_>, ObservableError>>()?;
    Ok(Grouping { groups, constant })
}

/// A group while [`group_qubit_wise`] is still filling it.
struct OpenGroup {
    basis: PauliMask,
    term_indices: Vec<usize>,
    /// `(coefficient, parity mask)` of each term, for the group's reducer.
    terms: Vec<(f64, u64)>,
}

/// The mask of qubits `0..n`, for `n <= MAX_QUBITS`.
fn low_bits(n: usize) -> u64 {
    if n >= MAX_QUBITS {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, n)| (k.to_string(), *n)).collect()
    }

    fn mask(ops: &[(usize, Pauli)]) -> PauliMask {
        PauliMask::new(ops.iter().copied()).unwrap()
    }

    #[test]
    fn masks_encode_each_pauli() {
        let m = mask(&[(0, Pauli::X), (2, Pauli::Y), (3, Pauli::Z)]);
        assert_eq!(m.x_bits(), 0b0101);
        assert_eq!(m.z_bits(), 0b1100);
        assert_eq!(m.support(), 0b1101);
        assert_eq!(m.pauli(0), Some(Pauli::X));
        assert_eq!(m.pauli(1), None);
        assert_eq!(m.pauli(2), Some(Pauli::Y));
        assert_eq!(m.pauli(3), Some(Pauli::Z));
        assert_eq!(m.pauli(MAX_QUBITS), None);
        assert!(PauliMask::identity().is_identity());
        assert!(!m.is_identity());
    }

    #[test]
    fn invalid_strings_are_rejected() {
        assert!(PauliMask::new([(0, Pauli::Z), (0, Pauli::Z)]).is_err());
        assert!(PauliMask::new([(1, Pauli::X), (1, Pauli::Z)]).is_err());
        assert!(PauliMask::new([(MAX_QUBITS, Pauli::Z)]).is_err());
        assert!(PauliMask::new([(MAX_QUBITS - 1, Pauli::Y)]).is_ok());
    }

    #[test]
    fn qubit_wise_commutation() {
        let z0z1 = mask(&[(0, Pauli::Z), (1, Pauli::Z)]);
        let z1x2 = mask(&[(1, Pauli::Z), (2, Pauli::X)]);
        let x1 = mask(&[(1, Pauli::X)]);
        let y3 = mask(&[(3, Pauli::Y)]);
        assert!(z0z1.qubit_wise_commutes(&z1x2));
        assert!(z0z1.qubit_wise_commutes(&y3));
        assert!(!z0z1.qubit_wise_commutes(&x1));
        assert!(!z1x2.qubit_wise_commutes(&x1));
        assert!(x1.qubit_wise_commutes(&PauliMask::identity()));
        // Y against X or Z on the same qubit differs in one of the two masks.
        let y1 = mask(&[(1, Pauli::Y)]);
        assert!(!y1.qubit_wise_commutes(&x1));
        assert!(!y1.qubit_wise_commutes(&z0z1));
        assert!(y1.qubit_wise_commutes(&y1));
    }

    #[test]
    fn parity_value_and_bit_order() {
        // f = 2·Z0 + 3·Z1Z2 + 0.5·I. Key "011": qubit 0 = 1, qubit 1 = 1,
        // qubit 2 = 0 → Z0 = −1, Z1Z2 = −1 → −2 − 3 + 0.5.
        let obs =
            ParityObservable::new(3, vec![(2.0, 0b001), (3.0, 0b110), (0.5, 0)], 1.0).unwrap();
        let ev = obs
            .expectation_batch(&[counts(&[("011", 1)]), counts(&[("000", 1)])])
            .unwrap();
        assert_eq!(ev, vec![-4.5, 5.5]);
    }

    #[test]
    fn weighted_mean_scale_and_empty_counts() {
        let obs = ParityObservable::new(1, vec![(1.0, 0b1)], -2.0).unwrap();
        let ev = obs
            .expectation_batch(&[
                counts(&[("0", 3), ("1", 1)]),
                HashMap::new(),
                counts(&[("1", 0)]),
            ])
            .unwrap();
        // ⟨Z⟩ = (3 − 1)/4 = 0.5, scaled by −2.
        assert_eq!(ev, vec![-1.0, 0.0, 0.0]);
    }

    #[test]
    fn bad_keys_are_errors_and_high_bits_are_ignored() {
        let obs = ParityObservable::new(2, vec![(1.0, 0b11)], 1.0).unwrap();
        assert!(matches!(
            obs.expectation_batch(&[counts(&[("1", 1)])]),
            Err(ObservableError::BitWidthMismatch {
                num_vars: 2,
                key_len: 1
            })
        ));
        assert!(matches!(
            obs.expectation_batch(&[counts(&[("1x", 1)])]),
            Err(ObservableError::InvalidBitstring(_))
        ));
        // Bits beyond num_qubits are neither read nor validated, as in QUBO.
        assert_eq!(
            obs.expectation_batch(&[counts(&[("x101", 1)])]).unwrap(),
            vec![-1.0]
        );
    }

    #[test]
    fn invalid_reducers_are_rejected() {
        assert!(ParityObservable::new(0, vec![], 1.0).is_err());
        assert!(ParityObservable::new(MAX_QUBITS + 1, vec![], 1.0).is_err());
        assert!(ParityObservable::new(2, vec![(1.0, 0b100)], 1.0).is_err());
        assert!(ParityObservable::new(2, vec![(f64::NAN, 0b1)], 1.0).is_err());
        assert!(ParityObservable::new(2, vec![(1.0, 0b1)], f64::INFINITY).is_err());
        assert!(ParityObservable::new(MAX_QUBITS, vec![(1.0, u64::MAX)], 1.0).is_ok());
    }

    #[test]
    fn sixty_four_qubit_keys_decode_every_bit() {
        let obs =
            ParityObservable::new(MAX_QUBITS, vec![(1.0, 1u64 << 63), (1.0, 1)], 1.0).unwrap();
        let key = format!("1{}", "0".repeat(63));
        assert_eq!(
            obs.expectation_batch(&[counts(&[(&key, 1)])]).unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn probabilities_are_validated() {
        let obs = ParityObservable::new(2, vec![(1.0, 0b10)], 1.0).unwrap();
        assert_eq!(
            obs.expectation_from_probabilities(&[0.25, 0.25, 0.5, 0.0])
                .unwrap(),
            0.0
        );
        // Wider vectors are fine (higher qubits ignored); narrower or odd ones not.
        assert!(obs.expectation_from_probabilities(&[0.125; 8]).is_ok());
        assert!(obs.expectation_from_probabilities(&[0.5, 0.5]).is_err());
        assert!(obs.expectation_from_probabilities(&[0.2; 5]).is_err());
        assert!(obs.expectation_from_probabilities(&[]).is_err());
        assert!(obs
            .expectation_from_probabilities(&[1.0, -0.5, 0.5, 0.0])
            .is_err());
        assert!(obs
            .expectation_from_probabilities(&[f64::NAN, 0.0, 0.0, 0.0])
            .is_err());
    }

    #[test]
    fn grouping_is_greedy_and_puts_identity_in_the_constant() {
        let terms = [
            (1.0, mask(&[(0, Pauli::Z)])),
            (0.5, PauliMask::identity()),
            (2.0, mask(&[(0, Pauli::X), (1, Pauli::X)])),
            (3.0, mask(&[(1, Pauli::Z)])),
            (4.0, mask(&[(1, Pauli::X)])),
            (-0.25, PauliMask::identity()),
            (5.0, mask(&[(0, Pauli::Y)])),
        ];
        let grouping = group_qubit_wise(2, &terms).unwrap();
        assert_eq!(grouping.constant(), 0.25);
        let indices: Vec<&[usize]> = grouping.groups().iter().map(|g| g.term_indices()).collect();
        assert_eq!(indices, vec![&[0, 3][..], &[2, 4][..], &[6][..]]);
        let rotations: Vec<Vec<(usize, Pauli)>> =
            grouping.groups().iter().map(|g| g.rotations()).collect();
        assert_eq!(
            rotations,
            vec![
                vec![],
                vec![(0, Pauli::X), (1, Pauli::X)],
                vec![(0, Pauli::Y)]
            ]
        );
        assert_eq!(
            grouping.groups()[1].observable().terms(),
            &[(2.0, 0b11), (4.0, 0b10)]
        );
    }

    #[test]
    fn grouping_rejects_invalid_input() {
        let z2 = mask(&[(2, Pauli::Z)]);
        assert!(group_qubit_wise(2, &[(1.0, z2)]).is_err());
        assert!(group_qubit_wise(3, &[(f64::NAN, z2)]).is_err());
        assert!(group_qubit_wise(MAX_QUBITS + 1, &[]).is_err());
        // Only identity terms: no groups, even over zero qubits.
        let only_constant = group_qubit_wise(0, &[(1.5, PauliMask::identity())]).unwrap();
        assert!(only_constant.groups().is_empty());
        assert_eq!(only_constant.constant(), 1.5);
    }
}
