//! Finding the `Z₂` symmetries of a qubit Hamiltonian and removing one qubit
//! per symmetry ("tapering"), generically (issue #535).
//!
//! A `Z₂` symmetry is a Pauli string `τ` that commutes with every term, so
//! `τ` and the Hamiltonian share eigenvectors and each eigenvector has
//! `τ = ±1`. A Clifford rotation `U` turns `τ` into a single `Z_q`; in the
//! rotated Hamiltonian every term acts on qubit `q` with `I` or `Z`, `Z_q`
//! can be replaced by its value, and the qubit disappears with no
//! approximation (Bravyi, Gambetta, Mezzacapo and Temme, *Tapering off
//! qubits to simulate fermionic Hamiltonians*, 2017, who rotate to `X_q`
//! instead).
//!
//! # What is done here
//!
//! - **Z-type symmetries only.** [`PauliSum::z_symmetries`] finds the strings
//!   made of `Z` alone that commute with every term: `τ` with `Z`s on the
//!   qubits of `z` anticommutes with a term exactly when an odd number of
//!   them carry the term's `X` or `Y`, so the symmetries are the null space
//!   over GF(2) of the terms' `X`/`Y` masks. Z-type symmetries commute with
//!   each other. A molecule with real orbitals in the Jordan–Wigner or parity
//!   encoding has the spin parities and the point-group symmetries its
//!   orbitals carry, all of this type (H₂, LiH and BeH₂ in STO-3G with
//!   symmetry-adapted orbitals, as PySCF's `symmetry=True` gives, have 3, 4
//!   and 5); a symmetry that is not a Pauli string in the chosen orbitals is
//!   not found (with Löwdin orbitals, H₂'s reflection swaps two orbitals and
//!   is lost, so 2 qubits go instead of 3; mixed degenerate π orbitals lose
//!   the point-group ones the same way). Coefficients are compared exactly
//!   and every listed term counts: a rounding residue, or a duplicate term
//!   that cancels another, can only hide a symmetry, never invent one, so
//!   simplify and drop residues first ([`PauliSum::simplified`],
//!   [`PauliSum::without_residues`]).
//! - **To `Z_q`, by CNOTs.** The symmetries are put in reduced row-echelon
//!   form over GF(2): each has a pivot qubit `q_i` (its lowest qubit) that no
//!   other has. For each, a CNOT from every other qubit of its support onto
//!   `q_i` turns it into `Z_{q_i}` (`CNOT(c, t)` maps `Z_c Z_t` to `Z_t`).
//!   Every control is a non-pivot and every target a pivot, so all these
//!   CNOTs commute, `U = U†`, and the other symmetries are untouched. A
//!   string made of `Z` alone stays one, so the number operators of a
//!   molecule stay diagonal after tapering.
//! - **The values `±1`.** Each symmetry's value selects one block of the
//!   spectrum; all `2^k` choices together give the whole spectrum. The values
//!   come from a reference basis state, by default the Hartree–Fock
//!   determinant of the electron sector ([`Tapering::for_sector`]), or are
//!   given explicitly ([`Tapering::with_values`]).

use super::fermion::FermionHamiltonian;
use super::mapping::QubitMapping;
use super::pauli::{PauliAxis, PauliString, PauliSum, PauliTerm};
use super::sector::combinations;
use crate::error::PhysicsError;

/// Relative size below which [`PauliSum::without_residues`] drops a term:
/// `|c| ≤ 1e-12 · max |c|`, the maximum over the non-identity terms (the
/// identity carries the energy's zero, which should not change what is a
/// residue). Integrals computed numerically leave residues of about `1e-17`
/// on terms a symmetry makes exactly zero (H₂ from `chemistry::h2` has six),
/// and a single one hides that symmetry. The physical terms of the molecules
/// here are above `1e-5`.
pub const RESIDUE_TOLERANCE: f64 = 1e-12;

/// The X/Y mask and the Y/Z mask of a string: `X → (1, 0)`, `Y → (1, 1)`,
/// `Z → (0, 1)` per qubit. The caller has checked its qubits fit in a
/// `usize`.
fn masks(string: &PauliString) -> (usize, usize) {
    let (mut x, mut z) = (0usize, 0usize);
    for &(q, axis) in string.ops() {
        match axis {
            PauliAxis::X => x |= 1 << q,
            PauliAxis::Y => {
                x |= 1 << q;
                z |= 1 << q;
            }
            PauliAxis::Z => z |= 1 << q,
        }
    }
    (x, z)
}

/// The string with `X`/`Y` mask `x` and `Y`/`Z` mask `z`.
fn string_of(x: usize, z: usize) -> PauliString {
    let support = x | z;
    let ops = (0..usize::BITS as usize)
        .filter(|&q| support >> q & 1 == 1)
        .map(|q| {
            let axis = match (x >> q & 1, z >> q & 1) {
                (1, 1) => PauliAxis::Y,
                (1, _) => PauliAxis::X,
                _ => PauliAxis::Z,
            };
            (q, axis)
        });
    // Distinct qubits, so this cannot fail.
    PauliString::new(ops).unwrap_or_default()
}

fn too_wide(n: usize) -> Option<PhysicsError> {
    (n > usize::BITS as usize).then(|| PhysicsError::UnsupportedOperation {
        message: format!(
            "{n} qubits do not fit in a {}-bit mask: tapering supports at most {} qubits",
            usize::BITS,
            usize::BITS
        ),
    })
}

/// Reduced row-echelon form over GF(2) of `rows`, each with its pivot (the
/// lowest set bit) absent from every other row, sorted by pivot; `values`
/// follow the row operations (a row added to another multiplies its value
/// into it). Returns `None` if the rows are dependent (one reduces to 0).
fn reduce(rows: &[usize], values: &[i8]) -> Option<Vec<(usize, i8)>> {
    let mut reduced: Vec<(usize, i8)> = Vec::with_capacity(rows.len());
    for (&row, &value) in rows.iter().zip(values) {
        let (mut row, mut value) = (row, value);
        for &(other, other_value) in &reduced {
            if row >> other.trailing_zeros() & 1 == 1 {
                row ^= other;
                value *= other_value;
            }
        }
        if row == 0 {
            return None;
        }
        let pivot = row.trailing_zeros();
        for (other, other_value) in &mut reduced {
            if *other >> pivot & 1 == 1 {
                *other ^= row;
                *other_value *= value;
            }
        }
        reduced.push((row, value));
    }
    reduced.sort_by_key(|&(row, _)| row.trailing_zeros());
    Some(reduced)
}

impl PauliSum {
    /// The sum without its rounding residues: the terms with
    /// `|c| ≤ `[`RESIDUE_TOLERANCE`]` · max |c|`, the maximum taken over the
    /// non-identity terms, are dropped (an identity-only sum keeps every
    /// nonzero term). [`Tapering::for_sector`] does this before looking for
    /// symmetries; doing the same before tapering other operators keeps them
    /// consistent with it.
    ///
    /// Dropping a term changes every eigenvalue by at most its `|c|`, so the
    /// spectrum moves by at most the sum of the dropped `|c|`, each at most
    /// `1e-12` of the largest term. A dropped term that was not a residue
    /// can make a symmetry appear that the exact sum does not have, within
    /// that same bound.
    ///
    /// # Errors
    /// As [`PauliSum::chop`].
    pub fn without_residues(self) -> Result<PauliSum, PhysicsError> {
        let largest = self
            .terms()
            .iter()
            .filter(|t| !t.string.is_identity())
            .fold(0.0_f64, |m, t| m.max(t.coefficient.abs()));
        self.chop(RESIDUE_TOLERANCE * largest)
    }

    /// The `Z₂` symmetries of **Z type** (strings of `Z` alone) that commute
    /// with every term: a basis of them, in reduced row-echelon form over
    /// GF(2), each with a pivot (its lowest qubit) that no other has, sorted
    /// by pivot; the identity is not one. See the [module docs](self) for
    /// what is and is not found.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if the sum has more qubits than
    /// a `usize` has bits.
    pub fn z_symmetries(&self) -> Result<Vec<PauliString>, PhysicsError> {
        let n = self.num_qubits();
        if let Some(error) = too_wide(n) {
            return Err(error);
        }
        // Row-echelon form of the terms' X/Y masks: z must have an even
        // overlap with each of them.
        let mut constraints: Vec<usize> = Vec::new();
        for term in self.terms() {
            let mut x = masks(&term.string).0;
            for &c in &constraints {
                if x >> c.trailing_zeros() & 1 == 1 {
                    x ^= c;
                }
            }
            if x != 0 {
                let pivot = x.trailing_zeros();
                for c in &mut constraints {
                    if *c >> pivot & 1 == 1 {
                        *c ^= x;
                    }
                }
                constraints.push(x);
            }
        }
        let pivots: usize = constraints
            .iter()
            .fold(0, |m, c| m | (1 << c.trailing_zeros()));
        // One null-space vector per free qubit f: e_f plus the pivot of every
        // constraint containing f.
        let null_space: Vec<usize> = (0..n)
            .filter(|&f| pivots >> f & 1 == 0)
            .map(|f| {
                constraints
                    .iter()
                    .filter(|&&c| c >> f & 1 == 1)
                    .fold(1usize << f, |z, c| z | (1 << c.trailing_zeros()))
            })
            .collect();
        let reduced = reduce(&null_space, &vec![1; null_space.len()]).unwrap_or_default();
        Ok(reduced.into_iter().map(|(z, _)| string_of(0, z)).collect())
    }
}

/// The removal of `k` independent Z-type symmetries of an `n`-qubit
/// operator, with their values: the CNOT network that turns each into a
/// single `Z` on its pivot qubit, then [`PauliSum::taper`] of the pivots (see
/// the [module docs](self)).
#[derive(Debug, Clone, PartialEq)]
pub struct Tapering {
    num_qubits: usize,
    /// The symmetries' `Z` masks, reduced, sorted by pivot.
    symmetries: Vec<usize>,
    /// Their values, aligned with `symmetries`.
    values: Vec<i8>,
    /// `(control, target)` of every CNOT; they all commute.
    cnots: Vec<(usize, usize)>,
}

impl Tapering {
    /// A tapering of `num_qubits` qubits by `symmetries` (strings of `Z`
    /// alone, independent, in any order), with their values `±1` in the same
    /// order. The symmetries are reduced as [`PauliSum::z_symmetries`] returns
    /// them; the values follow (a symmetry replaced by a product of the given
    /// ones gets the product of their values).
    ///
    /// # Errors
    /// - [`PhysicsError::UnsupportedOperation`] if `num_qubits` exceeds the
    ///   bits of a `usize`.
    /// - [`PhysicsError::InvalidPauliSum`] if a symmetry is the identity,
    ///   holds an `X` or a `Y`, acts outside `num_qubits`, or depends on the
    ///   others; if `values` has another length or a value is not `±1`.
    pub fn with_values(
        num_qubits: usize,
        symmetries: &[PauliString],
        values: &[i8],
    ) -> Result<Self, PhysicsError> {
        if let Some(error) = too_wide(num_qubits) {
            return Err(error);
        }
        let invalid = |message: String| PhysicsError::InvalidPauliSum { message };
        if values.len() != symmetries.len() {
            return Err(invalid(format!(
                "{} values for {} symmetries",
                values.len(),
                symmetries.len()
            )));
        }
        if let Some(v) = values.iter().find(|&&v| v != 1 && v != -1) {
            return Err(invalid(format!("value {v} is not +1 or -1")));
        }
        let mut rows = Vec::with_capacity(symmetries.len());
        for symmetry in symmetries {
            if symmetry.is_identity() {
                return Err(invalid(
                    "the identity is not a symmetry to taper".to_string(),
                ));
            }
            if let Some(&(q, _)) = symmetry.ops().iter().find(|&&(q, _)| q >= num_qubits) {
                return Err(invalid(format!(
                    "symmetry {symmetry} acts on qubit {q}, outside {num_qubits} qubits"
                )));
            }
            let (x, z) = masks(symmetry);
            if x != 0 {
                return Err(invalid(format!(
                    "symmetry {symmetry} has an X or a Y: only strings of Z are supported"
                )));
            }
            rows.push(z);
        }
        let reduced = reduce(&rows, values)
            .ok_or_else(|| invalid("the symmetries are not independent".to_string()))?;
        let cnots = reduced
            .iter()
            .flat_map(|&(z, _)| {
                let pivot = z.trailing_zeros() as usize;
                (pivot + 1..num_qubits)
                    .filter(move |&r| z >> r & 1 == 1)
                    .map(move |r| (r, pivot))
            })
            .collect();
        Ok(Self {
            num_qubits,
            symmetries: reduced.iter().map(|&(z, _)| z).collect(),
            values: reduced.iter().map(|&(_, v)| v).collect(),
            cnots,
        })
    }

    /// As [`with_values`](Self::with_values), each value read from the
    /// computational basis state `reference_state`: `(−1)^k` for `k` of its
    /// qubits set inside the symmetry.
    ///
    /// # Errors
    /// As [`with_values`](Self::with_values);
    /// [`PhysicsError::UnsupportedOperation`] if `reference_state` is not one
    /// of the `2^n` basis states.
    pub fn new(
        num_qubits: usize,
        symmetries: &[PauliString],
        reference_state: usize,
    ) -> Result<Self, PhysicsError> {
        if let Some(error) = too_wide(num_qubits) {
            return Err(error);
        }
        if num_qubits < usize::BITS as usize && reference_state >> num_qubits != 0 {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "reference state {reference_state:#b} is outside the 2^{num_qubits} basis \
                     states"
                ),
            });
        }
        // `with_values` rejects a symmetry outside the qubits or with X/Y;
        // the placeholder value given to it here is never used.
        let values: Vec<i8> = symmetries
            .iter()
            .map(|s| {
                let ones = s
                    .ops()
                    .iter()
                    .filter(|&&(q, _)| q < num_qubits && reference_state >> q & 1 == 1)
                    .count();
                if ones % 2 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        Self::with_values(num_qubits, symmetries, &values)
    }

    /// Tapers the qubit Hamiltonian of `hamiltonian` in `mapping` by all its
    /// Z-type symmetries, with the values of the reference determinant that
    /// fills the first `n_up` spin-up and the first `n_down` spin-down
    /// orbitals: the Hartree–Fock determinant when the orbitals are sorted by
    /// energy (as PySCF writes them; Molpro groups them by symmetry, where
    /// the reference must be built with [`Tapering::new`]). Returns the
    /// tapering and the tapered Hamiltonian.
    ///
    /// The point-group values are those of the reference: the lowest state of
    /// the sector is in the tapered space when it has the reference's
    /// symmetry, as the ground states of H₂, LiH and BeH₂ do.
    ///
    /// Before the symmetries are looked for, the qubit Hamiltonian loses its
    /// rounding residues ([`PauliSum::without_residues`]), and the tapered
    /// Hamiltonian is that of the remaining terms; taper other operators
    /// from `to_qubits(mapping)?.without_residues()?` the same way.
    /// [`PauliSum::z_symmetries`] itself compares exactly.
    ///
    /// # Errors
    /// - [`PhysicsError::InvalidSector`] if `n_up` or `n_down` exceeds the
    ///   number of spatial orbitals.
    /// - As [`FermionOperator::to_qubits`](super::FermionOperator::to_qubits),
    ///   [`PauliSum::z_symmetries`], [`Tapering::new`] and
    ///   [`Tapering::apply`].
    pub fn for_sector(
        hamiltonian: &FermionHamiltonian,
        mapping: QubitMapping,
        n_up: usize,
        n_down: usize,
    ) -> Result<(Self, PauliSum), PhysicsError> {
        let k = hamiltonian.num_spatial_orbitals();
        if n_up > k || n_down > k {
            return Err(PhysicsError::InvalidSector {
                message: format!(
                    "{n_up} spin-up and {n_down} spin-down electrons do not fit in {k} spatial \
                     orbitals"
                ),
            });
        }
        let qubits = hamiltonian.to_fermion_operator().to_qubits(mapping)?;
        let qubits = qubits.without_residues()?;
        let symmetries = qubits.z_symmetries()?;
        let occupation = low_bits(n_up) | (low_bits(n_down) << k);
        let reference = mapping.encode_occupation(occupation, 2 * k)?;
        let tapering = Self::new(2 * k, &symmetries, reference)?;
        let tapered = tapering.apply(&qubits)?;
        Ok((tapering, tapered))
    }

    /// The number of qubits before tapering.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The number of qubits after tapering: one fewer per symmetry.
    pub fn num_tapered_qubits(&self) -> usize {
        self.num_qubits - self.symmetries.len()
    }

    /// The symmetries, reduced and sorted by pivot (aligned with
    /// [`values`](Self::values)).
    pub fn symmetries(&self) -> Vec<PauliString> {
        self.symmetries.iter().map(|&z| string_of(0, z)).collect()
    }

    /// The value `±1` of each symmetry, aligned with
    /// [`symmetries`](Self::symmetries).
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// The CNOT network `U = U†` that turns each symmetry into a single `Z`,
    /// as `(control, target)` pairs (they all commute): what a circuit, an
    /// ansatz or an excitation operator must be conjugated by to act in the
    /// tapered frame.
    pub fn cnots(&self) -> &[(usize, usize)] {
        &self.cnots
    }

    /// The qubits that tapering removes (the pivots), ascending.
    pub fn removed_qubits(&self) -> Vec<usize> {
        self.symmetries
            .iter()
            .map(|z| z.trailing_zeros() as usize)
            .collect()
    }

    /// Tapers `operator`: conjugates every term by the CNOT network, replaces
    /// each pivot's `Z` by its value and renumbers the remaining qubits in
    /// order; the result is [`PauliSum::simplified`]. The Hamiltonian, the
    /// number operators and any observable that commutes with the
    /// symmetries can be tapered this way, consistently.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if `operator` acts on another
    /// number of qubits, or a term anticommutes with a symmetry (naming
    /// both); as [`PauliSum::taper`] and [`PauliSum::simplified`].
    pub fn apply(&self, operator: &PauliSum) -> Result<PauliSum, PhysicsError> {
        let unsupported = |message: String| PhysicsError::UnsupportedOperation { message };
        if operator.num_qubits() != self.num_qubits {
            return Err(unsupported(format!(
                "an operator on {} qubits cannot be tapered by a tapering of {}",
                operator.num_qubits(),
                self.num_qubits
            )));
        }
        let mut terms = Vec::with_capacity(operator.terms().len());
        for term in operator.terms() {
            let (mut x, mut z) = masks(&term.string);
            if let Some(&symmetry) = self
                .symmetries
                .iter()
                .find(|&&s| (s & x).count_ones() % 2 == 1)
            {
                return Err(unsupported(format!(
                    "{} anticommutes with the symmetry {}: the operator cannot be tapered",
                    term.string,
                    string_of(0, symmetry)
                )));
            }
            // A Pauli string is i^{|x∧z|} X^x Z^z; a CNOT conjugates X^x Z^z
            // with no phase (x_t ^= x_c, z_c ^= z_t), so the string picks up
            // i^{|x∧z| − |x'∧z'|}, which is real: the number of Y changes by
            // an even amount.
            let before = (x & z).count_ones();
            for &(c, t) in &self.cnots {
                x ^= (x >> c & 1) << t;
                z ^= (z >> t & 1) << c;
            }
            let after = (x & z).count_ones();
            let power = (before + 4 - after % 4) % 4;
            let sign = match power {
                0 => 1.0,
                2 => -1.0,
                // Unreachable: CNOTs change the number of Y by an even amount.
                // Reported rather than unwrapped.
                _ => {
                    return Err(unsupported(format!(
                        "conjugating {} gave an imaginary phase",
                        term.string
                    )))
                }
            };
            terms.push(PauliTerm::new(sign * term.coefficient, string_of(x, z)));
        }
        let rotated = PauliSum::new(self.num_qubits, terms)?;
        let fixed: Vec<(usize, i8)> = self
            .removed_qubits()
            .into_iter()
            .zip(self.values.iter().copied())
            .collect();
        rotated.taper(&fixed)
    }

    /// The tapered-space index of the computational basis state `state`: the
    /// CNOTs applied (`b_t ^= b_c`), each pivot checked against its value
    /// (`Z = +1` on a `0` bit), the pivots dropped and the remaining bits
    /// packed in order. A basis state of the original space (e.g. a VQE's
    /// Hartree–Fock initial state) is in the tapered space only if every
    /// symmetry has its value on it.
    ///
    /// # Errors
    /// [`PhysicsError::UnsupportedOperation`] if `state` is outside the `2^n`
    /// basis states; [`PhysicsError::InvalidSector`] if a symmetry has the
    /// other value on it.
    pub fn tapered_basis_state(&self, state: usize) -> Result<usize, PhysicsError> {
        let n = self.num_qubits;
        if n < usize::BITS as usize && state >> n != 0 {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!("basis state {state:#b} is outside the 2^{n} basis states"),
            });
        }
        let mut b = state;
        for &(c, t) in &self.cnots {
            b ^= (b >> c & 1) << t;
        }
        let mut packed = 0usize;
        let mut position = 0;
        for q in 0..n {
            match self
                .symmetries
                .iter()
                .position(|z| z.trailing_zeros() as usize == q)
            {
                Some(i) => {
                    let value = if b >> q & 1 == 0 { 1 } else { -1 };
                    if value != self.values[i] {
                        return Err(PhysicsError::InvalidSector {
                            message: format!(
                                "basis state {state:#b} has {} = {value}, not {}",
                                string_of(0, self.symmetries[i]),
                                self.values[i]
                            ),
                        });
                    }
                }
                None => {
                    packed |= (b >> q & 1) << position;
                    position += 1;
                }
            }
        }
        Ok(packed)
    }

    /// The tapered-space indices, ascending, of the determinants of the
    /// electron sector `(n_up, n_down)` (spin-orbitals in block order,
    /// encoded with `mapping`) that are in the tapered space: a basis for
    /// [`PauliSum::restricted_eigenvalues`] of the tapered Hamiltonian. The
    /// determinants with other symmetry values are left out.
    ///
    /// # Errors
    /// [`PhysicsError::InvalidSector`] if the qubits are not `2K` spin-orbitals
    /// holding the sector; as
    /// [`QubitMapping::encode_occupation`].
    pub fn sector_basis(
        &self,
        mapping: QubitMapping,
        n_up: usize,
        n_down: usize,
    ) -> Result<Vec<usize>, PhysicsError> {
        let modes = self.num_qubits;
        let k = modes / 2;
        if modes % 2 != 0 || n_up > k || n_down > k {
            return Err(PhysicsError::InvalidSector {
                message: format!(
                    "{n_up} spin-up and {n_down} spin-down electrons in {modes} spin-orbitals \
                     (block order needs 2K, at most K of each spin)"
                ),
            });
        }
        // Every determinant is enumerated: refuse a sector so large that the
        // tapered block (about 2^-s of it, `s` the number of symmetries)
        // would exceed the dense limit anyway, before enumerating it.
        // `checked_shl` alone fails only for a shift of 64 or more, not when
        // bits fall off the top, so the factor `2^s` is multiplied with an
        // overflow check; past `usize::MAX` there is no cap.
        let cap = 1usize
            .checked_shl(self.symmetries.len() as u32)
            .and_then(|factor| super::MAX_SECTOR_DIMENSION.checked_mul(factor))
            .unwrap_or(usize::MAX);
        match super::sector_dimension(k, n_up, n_down) {
            Some(d) if d <= cap => {}
            dimension => {
                let count =
                    dimension.map_or_else(|| "more than usize::MAX".to_string(), |d| d.to_string());
                return Err(PhysicsError::UnsupportedOperation {
                    message: format!(
                        "the sector ({n_up}, {n_down}) of {k} spatial orbitals has {count} \
                         determinants: too many to enumerate for a dense tapered block"
                    ),
                });
            }
        }
        let ups = combinations(k, n_up);
        let mut basis = Vec::new();
        for down in combinations(k, n_down) {
            for &up in &ups {
                let qubits = mapping.encode_occupation(up | (down << k), modes)?;
                if let Ok(index) = self.tapered_basis_state(qubits) {
                    basis.push(index);
                }
            }
        }
        basis.sort_unstable();
        Ok(basis)
    }
}

/// `n` low bits set.
fn low_bits(n: usize) -> usize {
    if n >= usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << n) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::{H2Encoding, H2Molecule};
    use crate::hamiltonians::exact::DenseMatrix;
    use crate::hamiltonians::parity_reduction_qubits;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const MAPPINGS: [QubitMapping; 2] = [QubitMapping::JordanWigner, QubitMapping::Parity];

    fn h2() -> FermionHamiltonian {
        H2Molecule::new(0.7414, H2Encoding::JordanWigner)
            .expect("a valid length")
            .fermion_hamiltonian()
            .clone()
    }

    fn z(qubits: &[usize]) -> PauliString {
        PauliString::new(qubits.iter().map(|&q| (q, PauliAxis::Z))).expect("distinct qubits")
    }

    /// Every symmetry commutes with every term.
    fn assert_commutes(sum: &PauliSum, symmetries: &[PauliString]) {
        for symmetry in symmetries {
            let (_, zs) = masks(symmetry);
            for term in sum.terms() {
                let (x, _) = masks(&term.string);
                assert_eq!(
                    (zs & x).count_ones() % 2,
                    0,
                    "{symmetry} vs {}",
                    term.string
                );
            }
        }
    }

    /// A random real sum on 7 qubits that commutes with three overlapping
    /// Z-strings (so the reduction and CNOTs sharing qubits are exercised).
    fn planted_sum(seed: u64) -> (PauliSum, [usize; 3]) {
        let planted = [0b101_1010, 0b011_0011, 0b111_0100];
        let mut rng = StdRng::seed_from_u64(seed);
        let mut terms = Vec::new();
        while terms.len() < 60 {
            let (x, zm): (usize, usize) = (rng.random_range(0..128), rng.random_range(0..128));
            let real = (x & zm).count_ones() % 2 == 0;
            let commutes = planted.iter().all(|&p| (p & x).count_ones() % 2 == 0);
            if real && commutes {
                terms.push(PauliTerm::new(
                    rng.random_range(-1.0..1.0),
                    string_of(x, zm),
                ));
            }
        }
        (PauliSum::new(7, terms).expect("valid terms"), planted)
    }

    /// The eigenvalues of `sum` restricted to the basis states on which each
    /// symmetry has its value, from the dense matrix (independent of the
    /// tapering and of `restricted_eigenvalues`).
    fn block_spectrum(sum: &PauliSum, symmetries: &[PauliString], values: &[i8]) -> Vec<f64> {
        let full = sum.to_dense_real().expect("a small sum");
        let states: Vec<usize> = (0..1usize << sum.num_qubits())
            .filter(|&b| {
                symmetries.iter().zip(values).all(|(s, &v)| {
                    let parity = (masks(s).1 & b).count_ones() % 2;
                    (parity == 0) == (v == 1)
                })
            })
            .collect();
        let mut block = DenseMatrix::zeros(states.len());
        for (r, &a) in states.iter().enumerate() {
            for (c, &b) in states.iter().enumerate() {
                block.add_to(r, c, full.get(a, b).unwrap());
            }
        }
        block.symmetric_eigenvalues().expect("a symmetric block")
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-10 * scale, "{x} vs {y}");
        }
    }

    #[test]
    fn h2_has_three_z_symmetries_in_both_encodings() {
        for mapping in MAPPINGS {
            let raw = h2().to_fermion_operator().to_qubits(mapping).unwrap();
            // Six residues of ~1e-17 hide the reflection symmetry (exact
            // detection), and dropping them restores it.
            assert_eq!(raw.z_symmetries().unwrap().len(), 2, "{mapping:?}");
            let sum = raw.chop(1e-14).unwrap();
            let symmetries = sum.z_symmetries().unwrap();
            assert_eq!(symmetries.len(), 3, "{mapping:?}");
            assert_commutes(&sum, &symmetries);
            // Reduced: each pivot (lowest qubit) is in no other symmetry.
            let tapering = Tapering::with_values(4, &symmetries, &[1, 1, 1]).unwrap();
            assert_eq!(tapering.symmetries(), symmetries);
        }
    }

    /// Every value combination tapers to exactly its block of the spectrum:
    /// checked combination by combination, so a value with the wrong sign is
    /// caught (the union over all combinations would not catch it).
    #[test]
    fn each_value_combination_tapers_to_its_block() {
        let mut cases: Vec<PauliSum> = MAPPINGS
            .iter()
            .map(|&m| {
                let raw = h2().to_fermion_operator().to_qubits(m).unwrap();
                raw.chop(1e-14).unwrap()
            })
            .collect();
        let (planted, masks_planted) = planted_sum(5);
        cases.push(planted);
        for sum in &cases {
            let symmetries = sum.z_symmetries().unwrap();
            let k = symmetries.len();
            let n = sum.num_qubits();
            for combination in 0..1usize << k {
                let values: Vec<i8> = (0..k)
                    .map(|i| if combination >> i & 1 == 0 { 1 } else { -1 })
                    .collect();
                let tapering = Tapering::with_values(n, &symmetries, &values).unwrap();
                let tapered = tapering.apply(sum).unwrap();
                assert_eq!(tapered.num_qubits(), n - k);
                assert_close(
                    &tapered.eigenvalues().unwrap(),
                    &block_spectrum(sum, &symmetries, &values),
                );
            }
        }
        // The planted symmetries are among those found.
        let found = cases.last().unwrap().z_symmetries().unwrap();
        let rows: Vec<usize> = found.iter().map(|s| masks(s).1).collect();
        for p in masks_planted {
            let mut all = rows.clone();
            all.push(p);
            assert!(
                reduce(&all, &vec![1; all.len()]).is_none(),
                "{p:#b} is not in their span"
            );
        }
    }

    /// The order and the basis in which the symmetries are given do not
    /// matter: the same span gives the same tapering.
    #[test]
    fn the_order_and_basis_of_the_symmetries_do_not_matter() {
        let (sum, _) = planted_sum(8);
        let reference = 0b101_0011;
        let given = [
            vec![z(&[1, 3, 4, 6]), z(&[0, 1, 4, 5]), z(&[2, 4, 5, 6])],
            vec![z(&[2, 4, 5, 6]), z(&[1, 3, 4, 6]), z(&[0, 1, 4, 5])],
            // The same span with one product swapped in.
            vec![z(&[0, 3, 5, 6]), z(&[0, 1, 4, 5]), z(&[2, 4, 5, 6])],
        ];
        let taperings: Vec<Tapering> = given
            .iter()
            .map(|s| Tapering::new(7, s, reference).unwrap())
            .collect();
        for t in &taperings[1..] {
            assert_eq!(t, &taperings[0]);
            // Exact: the same tapering applied to the same sum.
            assert_eq!(t.apply(&sum).unwrap(), taperings[0].apply(&sum).unwrap());
        }
    }

    /// H₂ tapers to one qubit holding its FCI energy, with the values of the
    /// Hartree–Fock determinant, and the generic tapering removes at least
    /// the two qubits the parity reduction fixes, keeping a subset of its
    /// spectrum with the same ground energy.
    #[test]
    fn h2_tapers_to_one_qubit_with_its_fci_energy() {
        let fci = h2().sector_ground_energy(1, 1).unwrap();
        for mapping in MAPPINGS {
            let (tapering, tapered) = Tapering::for_sector(&h2(), mapping, 1, 1).unwrap();
            assert_eq!(tapered.num_qubits(), 1);
            let ground = tapered.ground_energy().unwrap();
            assert!(
                (ground - fci).abs() <= 1e-12,
                "{mapping:?}: {ground} vs {fci}"
            );
            let hartree_fock = mapping.encode_occupation(0b0101, 4).unwrap();
            assert!(tapering.tapered_basis_state(hartree_fock).is_ok());
        }
        let parity = h2()
            .to_fermion_operator()
            .to_qubits(QubitMapping::Parity)
            .unwrap()
            .chop(1e-14)
            .unwrap();
        let (tapering, tapered) = Tapering::for_sector(&h2(), QubitMapping::Parity, 1, 1).unwrap();
        let removed = tapering.removed_qubits();
        assert!(removed.contains(&1) && removed.contains(&3), "{removed:?}");
        let old = parity
            .taper(&parity_reduction_qubits(2, 1, 1).unwrap())
            .unwrap()
            .eigenvalues()
            .unwrap();
        for value in tapered.eigenvalues().unwrap() {
            assert!(
                old.iter().any(|o| (o - value).abs() <= 1e-12),
                "{value} in {old:?}"
            );
        }
        assert!((old[0] - tapered.ground_energy().unwrap()).abs() <= 1e-12);
        // The tapered parity Hamiltonian itself, with lowest-qubit pivots
        // (pinned to 1e-6 against an independent prototype).
        let terms: Vec<(String, f64)> = tapered
            .terms()
            .iter()
            .map(|t| (t.string.to_string(), t.coefficient))
            .collect();
        let expected = [("I", -0.328717), ("X0", 0.181289), ("Z0", -0.787967)];
        assert_eq!(terms.len(), expected.len(), "{terms:?}");
        for ((string, c), (want, w)) in terms.iter().zip(expected) {
            assert_eq!(string, want);
            assert!((c - w).abs() < 1e-6, "{string}: {c} vs {w}");
        }
    }

    #[test]
    fn a_basis_state_with_other_values_is_not_in_the_tapered_space() {
        let (tapering, _) = Tapering::for_sector(&h2(), QubitMapping::JordanWigner, 1, 1).unwrap();
        // Two spin-up electrons have the other spin parities.
        let error = tapering.tapered_basis_state(0b0011).unwrap_err();
        assert!(
            matches!(error, PhysicsError::InvalidSector { .. }),
            "{error:?}"
        );
        let error = tapering.tapered_basis_state(1 << 4).unwrap_err();
        assert!(
            matches!(error, PhysicsError::UnsupportedOperation { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn sums_with_no_symmetry_or_only_symmetries_are_handled() {
        // X0 + Z0: no Z-type symmetry; tapering by none only simplifies.
        let x_plus_z: PauliSum = PauliSum::new(
            1,
            vec![
                PauliTerm::new(1.0, "X0".parse().unwrap()),
                PauliTerm::new(0.5, "Z0".parse().unwrap()),
            ],
        )
        .unwrap();
        assert!(x_plus_z.z_symmetries().unwrap().is_empty());
        let none = Tapering::new(1, &[], 0).unwrap();
        assert_eq!(
            none.apply(&x_plus_z).unwrap(),
            x_plus_z.clone().simplified().unwrap()
        );
        // A constant on 2 qubits: both Z0 and Z1 are symmetries; no qubit is
        // left, and the constant remains (exact: nothing is multiplied).
        let constant =
            PauliSum::new(2, vec![PauliTerm::new(0.75, PauliString::identity())]).unwrap();
        let symmetries = constant.z_symmetries().unwrap();
        assert_eq!(symmetries, vec![z(&[0]), z(&[1])]);
        let tapered = Tapering::new(2, &symmetries, 0)
            .unwrap()
            .apply(&constant)
            .unwrap();
        assert_eq!(tapered.num_qubits(), 0);
        assert_eq!(tapered.ground_energy().unwrap(), 0.75);
        // An idle qubit is a symmetry of its own.
        let idle = PauliSum::new(2, vec![PauliTerm::new(1.0, "X0".parse().unwrap())]).unwrap();
        assert_eq!(idle.z_symmetries().unwrap(), vec![z(&[1])]);
    }

    #[test]
    fn invalid_symmetries_and_operators_are_rejected() {
        let invalid = |r: Result<Tapering, PhysicsError>| {
            matches!(r, Err(PhysicsError::InvalidPauliSum { .. }))
        };
        assert!(invalid(Tapering::new(2, &["X0".parse().unwrap()], 0)));
        assert!(invalid(Tapering::new(2, &[z(&[2])], 0)));
        // Far outside the qubits (beyond a usize mask): an error, not a
        // panic.
        assert!(invalid(Tapering::new(2, &[z(&[70])], 0)));
        assert!(invalid(Tapering::new(2, &[PauliString::identity()], 0)));
        assert!(invalid(Tapering::new(
            3,
            &[z(&[0, 1]), z(&[1, 2]), z(&[0, 2])],
            0
        )));
        assert!(invalid(Tapering::with_values(2, &[z(&[0])], &[1, 1])));
        assert!(invalid(Tapering::with_values(2, &[z(&[0])], &[0])));
        assert!(matches!(
            Tapering::new(2, &[z(&[0])], 0b100),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        assert!(matches!(
            Tapering::new(65, &[], 0),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        assert!(matches!(
            PauliSum::new(65, vec![]).unwrap().z_symmetries(),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        let tapering = Tapering::new(2, &[z(&[0, 1])], 0).unwrap();
        let x0 = PauliSum::new(2, vec![PauliTerm::new(1.0, "X0".parse().unwrap())]).unwrap();
        let error = tapering.apply(&x0).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("X0 anticommutes with the symmetry Z0 Z1"),
            "{error}"
        );
        let three = PauliSum::new(3, vec![]).unwrap();
        assert!(matches!(
            tapering.apply(&three),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        assert!(matches!(
            Tapering::new(3, &[z(&[0])], 0)
                .unwrap()
                .sector_basis(QubitMapping::JordanWigner, 1, 1),
            Err(PhysicsError::InvalidSector { .. })
        ));
    }

    #[test]
    fn restricted_eigenvalues_need_an_invariant_ascending_basis() {
        let sum = h2()
            .to_fermion_operator()
            .to_qubits(QubitMapping::JordanWigner)
            .unwrap();
        // The one-up, one-down determinants span an invariant subspace.
        let sector = [0b0101, 0b0110, 0b1001, 0b1010];
        let ground = sum.restricted_eigenvalues(&sector).unwrap()[0];
        assert!((ground - h2().sector_ground_energy(1, 1).unwrap()).abs() <= 1e-12);
        let unsupported = |basis: &[usize]| {
            matches!(
                sum.restricted_eigenvalues(basis),
                Err(PhysicsError::UnsupportedOperation { .. })
            )
        };
        assert!(unsupported(&[0b0110, 0b0101]));
        assert!(unsupported(&[0b0101, 0b0101]));
        assert!(unsupported(&[0b0101, 1 << 4]));
        assert!(unsupported(&[0b0101]), "not invariant");
        // A hopping (X0 X1 + Y0 Y1)/2 conserves the electrons although each
        // term alone does not: on {|01⟩, |10⟩} it is invariant (eigenvalues
        // ±1, to the solver's rounding), and on {|00⟩} too (each term sends
        // ±1/2 to |11⟩, cancelling exactly).
        let hopping = PauliSum::new(
            2,
            vec![
                PauliTerm::new(0.5, "X0 X1".parse().unwrap()),
                PauliTerm::new(0.5, "Y0 Y1".parse().unwrap()),
            ],
        )
        .unwrap();
        assert_close(
            &hopping.restricted_eigenvalues(&[0b01, 0b10]).unwrap(),
            &[-1.0, 1.0],
        );
        assert_eq!(hopping.restricted_eigenvalues(&[0b00]).unwrap(), vec![0.0]);
        let wide = PauliSum::new(12, vec![]).unwrap();
        let too_many: Vec<usize> = (0..=MAX_SECTOR).collect();
        assert!(matches!(
            wide.restricted_eigenvalues(&too_many),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        // Two columns leaking to the same outside state with opposite signs:
        // each amplitude must cancel on its own. X0 − X1 sends |00⟩ to |01⟩
        // (+1) and |11⟩ to |01⟩ (−1): not invariant.
        let leak = PauliSum::new(
            2,
            vec![
                PauliTerm::new(1.0, "X0".parse().unwrap()),
                PauliTerm::new(-1.0, "X1".parse().unwrap()),
            ],
        )
        .unwrap();
        assert!(matches!(
            leak.restricted_eigenvalues(&[0b00, 0b11]),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        // The identity never leaves the span, so it does not widen the
        // tolerance: a constant shift of 10⁶ (as a molecule's energy offset
        // may be) must not hide a leak of 10⁻⁷, which is the whole of the
        // non-identity part (with Σ|c| over all terms the tolerance would be
        // 10⁻⁶ and the leak would pass).
        let shifted_leak = PauliSum::new(
            1,
            vec![
                PauliTerm::new(1e6, PauliString::identity()),
                PauliTerm::new(1e-7, "X0".parse().unwrap()),
            ],
        )
        .unwrap();
        assert!(matches!(
            shifted_leak.restricted_eigenvalues(&[0]),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        // More qubits than a usize has bits: an error, not a panic.
        let wide = PauliSum::new(65, vec![PauliTerm::new(1.0, z(&[64]))]).unwrap();
        assert!(matches!(
            wide.restricted_eigenvalues(&[0]),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        let odd_y = PauliSum::new(1, vec![PauliTerm::new(1.0, "Y0".parse().unwrap())]).unwrap();
        assert!(matches!(
            odd_y.restricted_eigenvalues(&[0, 1]),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
    }

    const MAX_SECTOR: usize = crate::hamiltonians::MAX_SECTOR_DIMENSION;

    /// Open-shell sectors too: for each sector, the restricted spectra of
    /// every value combination, together, are the sector's spectrum (a spin
    /// swap in `sector_basis` or in the reference would break it).
    #[test]
    fn every_sector_splits_into_its_value_blocks() {
        for mapping in MAPPINGS {
            let sum = h2()
                .to_fermion_operator()
                .to_qubits(mapping)
                .unwrap()
                .without_residues()
                .unwrap();
            let symmetries = sum.z_symmetries().unwrap();
            let k = symmetries.len();
            for (n_up, n_down) in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)] {
                let mut together = Vec::new();
                for combination in 0..1usize << k {
                    let values: Vec<i8> = (0..k)
                        .map(|i| if combination >> i & 1 == 0 { 1 } else { -1 })
                        .collect();
                    let tapering = Tapering::with_values(4, &symmetries, &values).unwrap();
                    let tapered = tapering.apply(&sum).unwrap();
                    let basis = tapering.sector_basis(mapping, n_up, n_down).unwrap();
                    together.extend(tapered.restricted_eigenvalues(&basis).unwrap());
                }
                together.sort_by(f64::total_cmp);
                let sector = h2().sector_eigenvalues(n_up, n_down).unwrap();
                assert_close(&together, &sector);
            }
        }
    }

    /// The spins are not swapped: the spectra cannot tell (the Hamiltonian
    /// treats both spins alike, so the sectors (N↑, N↓) and (N↓, N↑) have
    /// the same spectrum), so the states are checked directly.
    #[test]
    fn the_sector_basis_and_the_reference_keep_their_spins() {
        for mapping in MAPPINGS {
            let sum = h2()
                .to_fermion_operator()
                .to_qubits(mapping)
                .unwrap()
                .without_residues()
                .unwrap();
            let symmetries = sum.z_symmetries().unwrap();
            // Values of one spin-up electron in orbital 0.
            let up = mapping.encode_occupation(0b0001, 4).unwrap();
            let tapering = Tapering::new(4, &symmetries, up).unwrap();
            let image = tapering.tapered_basis_state(up).unwrap();
            assert!(
                tapering
                    .sector_basis(mapping, 1, 0)
                    .unwrap()
                    .contains(&image),
                "{mapping:?}"
            );
            // A spin-down electron has the other spin parities.
            assert!(
                tapering.sector_basis(mapping, 0, 1).unwrap().is_empty(),
                "{mapping:?}"
            );
            // `for_sector` keeps its own reference, one spin-up electron.
            let (tapering, _) = Tapering::for_sector(&h2(), mapping, 1, 0).unwrap();
            assert!(tapering.tapered_basis_state(up).is_ok(), "{mapping:?}");
            let down = mapping.encode_occupation(0b0100, 4).unwrap();
            assert!(tapering.tapered_basis_state(down).is_err(), "{mapping:?}");
        }
    }

    /// Residues are measured against the largest non-identity term: a large
    /// constant (the energy's zero) does not raise the threshold.
    #[test]
    fn residues_are_relative_to_the_largest_non_identity_term() {
        let sum = PauliSum::new(
            2,
            vec![
                PauliTerm::new(1e6, PauliString::identity()),
                PauliTerm::new(1.0, z(&[0])),
                PauliTerm::new(1e-9, "X0 X1".parse().unwrap()),
                PauliTerm::new(1e-13, "Y0 Y1".parse().unwrap()),
            ],
        )
        .unwrap();
        let kept: Vec<String> = sum
            .without_residues()
            .unwrap()
            .terms()
            .iter()
            .map(|t| t.string.to_string())
            .collect();
        assert_eq!(kept, ["I", "Z0", "X0 X1"]);
    }

    /// A sector too large to enumerate is refused up front; the CNOTs are
    /// exposed for rotating circuits into the tapered frame.
    #[test]
    fn a_huge_sector_is_refused_and_the_cnots_are_exposed() {
        // K = 10, five electrons of each spin: C(10, 5)² = 63504
        // determinants, above the cap of 2500 · 2¹ for one symmetry, and
        // quick to enumerate if the cap were missing (so this fails, rather
        // than hangs, without it).
        let tapering = Tapering::new(20, &[z(&[0, 1])], 0).unwrap();
        let error = tapering
            .sector_basis(QubitMapping::JordanWigner, 5, 5)
            .unwrap_err();
        assert!(
            matches!(&error, PhysicsError::UnsupportedOperation { message } if message.contains("too many")),
            "{error:?}"
        );
        assert_eq!(tapering.cnots(), &[(1, 0)]);
    }

    #[test]
    fn the_sector_cap_does_not_wrap_with_many_symmetries() {
        // 63 symmetries: 2500 · 2⁶³ is above usize::MAX, so there is no cap,
        // but shifting 2500 left by 63 bits drops every bit and gives 0,
        // which would refuse this sector of 32 determinants. Z_1 … Z_63 at
        // +1 keep only the determinant with its electron on qubit 0, the one
        // qubit left after tapering.
        let symmetries: Vec<PauliString> = (1..64).map(|q| z(&[q])).collect();
        let tapering = Tapering::new(64, &symmetries, 0).unwrap();
        assert_eq!(tapering.num_tapered_qubits(), 1);
        assert_eq!(
            tapering
                .sector_basis(QubitMapping::JordanWigner, 1, 0)
                .unwrap(),
            vec![1]
        );
    }
}
