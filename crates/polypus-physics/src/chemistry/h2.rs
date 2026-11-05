//! The hydrogen molecule in the STO-3G basis.
//!
//! [`H2Molecule::new`] runs the whole chain for a bond length `R`:
//!
//! 1. STO-3G 1s integrals on each atom ([`super::sto3g`]);
//! 2. molecular orbitals fixed by symmetry,
//!    `σ = (φ_A + φ_B)/√(2(1+S))` and `σ* = (φ_A − φ_B)/√(2(1−S))`;
//! 3. one- and two-electron integrals over `σ, σ*`;
//! 4. the [`FermionHamiltonian`] (spin-orbitals in block order:
//!    `σ↑, σ*↑, σ↓, σ*↓`);
//! 5. Jordan–Wigner (4 qubits), or parity plus tapering of the two qubits
//!    fixed by `N↑ = N↓ = 1` (2 qubits);
//! 6. removal of rounding residues below [`H2_ROUNDING_TOLERANCE`].
//!
//! The molecular orbitals are exact by symmetry only for a homonuclear
//! diatomic; there is no iterative Hartree–Fock, so this does not extend to
//! other molecules (they enter through FCIDUMP, issue #535).

use super::sto3g::{
    electron_repulsion, kinetic, nuclear_attraction, overlap, ContractedGaussian, Nucleus,
};
use crate::constants::BOHR_RADIUS_ANGSTROM;
use crate::error::PhysicsError;
use crate::hamiltonians::{
    parity_reduction_qubits, FermionHamiltonian, PauliSum, PhysicsHamiltonian, QubitMapping,
};

/// Pauli coefficients with `|c| ≤ 1e-14` Ha are dropped as rounding noise.
///
/// Measured over `R ∈ [0.2, 10]` Å (steps of 0.05 Å): the largest noise
/// coefficient is `6.7e-16` (at 0.2 Å; noise scales with the integrals, which
/// grow at short range) and the smallest physical one is `6.9e-4`. The
/// tolerance leaves a margin of ~15 over the noise and ten orders of magnitude
/// below any physical term; a physical coefficient that crosses zero at some
/// `R` may be dropped there, changing the energy by less than `1e-14` Ha.
pub const H2_ROUNDING_TOLERANCE: f64 = 1e-14;

/// Below this value of `1 − S` the atoms are so close that `σ*` cannot be
/// normalized reliably (`R ≲ 1e-4` Å).
const MIN_ONE_MINUS_OVERLAP: f64 = 1e-10;

/// Qubit encoding of the H₂ Hamiltonian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum H2Encoding {
    /// Jordan–Wigner on the 4 spin-orbitals: 4 qubits, 15 Pauli terms.
    JordanWigner,
    /// Parity encoding with qubits 1 and 3 tapered (`Z1 = −1`, `Z3 = +1`):
    /// 2 qubits, 5 Pauli terms, cross term `X0 X1`.
    ParityTapered,
}

/// H₂ in STO-3G at a given bond length, in a given qubit encoding.
///
/// Everything is computed on construction, so [`PhysicsHamiltonian`] methods
/// cannot fail.
///
/// ```
/// use polypus_physics::chemistry::{H2Encoding, H2Molecule};
/// use polypus_physics::hamiltonians::PhysicsHamiltonian;
///
/// let h2 = H2Molecule::new(0.7414, H2Encoding::ParityTapered).unwrap();
/// assert_eq!(h2.num_qubits(), 2);
/// let energy = h2.to_pauli_sum().ground_energy().unwrap();
/// assert!((energy + 1.137270).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct H2Molecule {
    bond_length_bohr: f64,
    encoding: H2Encoding,
    fermion: FermionHamiltonian,
    pauli: PauliSum,
}

impl H2Molecule {
    /// Builds the Hamiltonian for a bond length in ångström.
    ///
    /// # Errors
    /// - [`PhysicsError::InvalidIntegrals`] if the bond length is not finite
    ///   and positive.
    /// - [`PhysicsError::UnsupportedOperation`] if the atoms are so close
    ///   that `1 − S < 1e-10` (`R ≲ 1e-4` Å).
    pub fn new(bond_length_angstrom: f64, encoding: H2Encoding) -> Result<Self, PhysicsError> {
        if !(bond_length_angstrom.is_finite() && bond_length_angstrom > 0.0) {
            return Err(PhysicsError::InvalidIntegrals {
                message: format!(
                    "H2 bond length must be finite and positive, got {bond_length_angstrom} Å"
                ),
            });
        }
        let r = bond_length_angstrom / BOHR_RADIUS_ANGSTROM;
        let fermion = molecular_hamiltonian(r)?;
        let operator = fermion.to_fermion_operator();
        let pauli = match encoding {
            H2Encoding::JordanWigner => operator.to_qubits(QubitMapping::JordanWigner)?,
            H2Encoding::ParityTapered => operator
                .to_qubits(QubitMapping::Parity)?
                .taper(&parity_reduction_qubits(2, 1, 1)?)?,
        }
        .chop(H2_ROUNDING_TOLERANCE)?;
        Ok(Self {
            bond_length_bohr: r,
            encoding,
            fermion,
            pauli,
        })
    }

    /// Bond length (Å).
    pub fn bond_length_angstrom(&self) -> f64 {
        self.bond_length_bohr * BOHR_RADIUS_ANGSTROM
    }

    /// Bond length (bohr).
    pub fn bond_length_bohr(&self) -> f64 {
        self.bond_length_bohr
    }

    /// The qubit encoding.
    pub fn encoding(&self) -> H2Encoding {
        self.encoding
    }

    /// The electronic Hamiltonian over `σ` (orbital 0) and `σ*` (orbital 1).
    pub fn fermion_hamiltonian(&self) -> &FermionHamiltonian {
        &self.fermion
    }
}

impl PhysicsHamiltonian for H2Molecule {
    fn num_qubits(&self) -> usize {
        self.pauli.num_qubits()
    }

    fn to_pauli_sum(&self) -> PauliSum {
        self.pauli.clone()
    }
}

/// Integrals over `σ, σ*` for atoms at `0` and `(0, 0, r)` (bohr).
fn molecular_hamiltonian(r: f64) -> Result<FermionHamiltonian, PhysicsError> {
    let (pa, pb) = ([0.0; 3], [0.0, 0.0, r]);
    let a = ContractedGaussian::sto3g_hydrogen_1s(pa)?;
    let b = ContractedGaussian::sto3g_hydrogen_1s(pb)?;
    let nuclei = [
        Nucleus {
            position: pa,
            charge: 1.0,
        },
        Nucleus {
            position: pb,
            charge: 1.0,
        },
    ];
    let ao = [&a, &b];

    let s = overlap(&a, &b);
    if 1.0 - s < MIN_ONE_MINUS_OVERLAP {
        return Err(PhysicsError::UnsupportedOperation {
            message: format!(
                "atoms too close (R = {r} bohr, 1 − S = {}): σ* cannot be normalized",
                1.0 - s
            ),
        });
    }
    // c[μ][p]: coefficient of atomic orbital μ in molecular orbital p.
    let (plus, minus) = (
        1.0 / (2.0 * (1.0 + s)).sqrt(),
        1.0 / (2.0 * (1.0 - s)).sqrt(),
    );
    let c = [[plus, minus], [plus, -minus]];

    let mut ao_one = [[0.0; 2]; 2];
    for (m, am) in ao.iter().enumerate() {
        for (n, an) in ao.iter().enumerate() {
            ao_one[m][n] = kinetic(am, an) + nuclear_attraction(am, an, &nuclei);
        }
    }
    let mut one_body = vec![0.0; 4];
    for p in 0..2 {
        for q in 0..2 {
            for m in 0..2 {
                for n in 0..2 {
                    one_body[p * 2 + q] += c[m][p] * c[n][q] * ao_one[m][n];
                }
            }
        }
    }

    let mut ao_two = [0.0; 16];
    for (i, ai) in ao.iter().enumerate() {
        for (j, aj) in ao.iter().enumerate() {
            for (k, ak) in ao.iter().enumerate() {
                for (l, al) in ao.iter().enumerate() {
                    ao_two[((i * 2 + j) * 2 + k) * 2 + l] = electron_repulsion(ai, aj, ak, al);
                }
            }
        }
    }
    let mut two_body = vec![0.0; 16];
    for p in 0..2 {
        for q in 0..2 {
            for rr in 0..2 {
                for ss in 0..2 {
                    let mut v = 0.0;
                    for i in 0..2 {
                        for j in 0..2 {
                            for k in 0..2 {
                                for l in 0..2 {
                                    v += c[i][p]
                                        * c[j][q]
                                        * c[k][rr]
                                        * c[l][ss]
                                        * ao_two[((i * 2 + j) * 2 + k) * 2 + l];
                                }
                            }
                        }
                    }
                    two_body[((p * 2 + q) * 2 + rr) * 2 + ss] = v;
                }
            }
        }
    }

    FermionHamiltonian::new(2, 1.0 / r, one_body, two_body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonians::PauliString;
    use std::collections::BTreeMap;

    const ENCODINGS: [H2Encoding; 2] = [H2Encoding::JordanWigner, H2Encoding::ParityTapered];

    fn ground(r: f64, encoding: H2Encoding) -> f64 {
        H2Molecule::new(r, encoding)
            .unwrap()
            .to_pauli_sum()
            .ground_energy()
            .unwrap()
    }

    #[test]
    fn fci_at_equilibrium_in_both_encodings() {
        for encoding in ENCODINGS {
            let e = ground(0.7414, encoding);
            assert!((e + 1.137270).abs() <= 1e-5, "{encoding:?}: {e}");
            // Pinned with normalized contracted orbitals (as PySCF/OpenFermion);
            // issue #529 quotes −1.1372701672, computed without renormalizing
            // the rounded STO-3G contraction (7.4e-9 higher).
            assert!((e + 1.137_270_174_7).abs() <= 1e-8, "{encoding:?}: {e}");
        }
    }

    #[test]
    fn dissociates_into_two_sto3g_hydrogen_atoms() {
        // At 5 Å the FCI energy is twice the STO-3G H-atom energy.
        let atom = -0.466_581_845_3;
        for encoding in ENCODINGS {
            let e = ground(5.0, encoding);
            assert!((e - 2.0 * atom).abs() <= 1e-6, "{encoding:?}: {e}");
        }
    }

    #[test]
    fn tapered_cross_term_is_x0_x1() {
        let h = H2Molecule::new(0.7414, H2Encoding::ParityTapered)
            .unwrap()
            .to_pauli_sum();
        let xx: PauliString = "X0 X1".parse().unwrap();
        let c = h
            .terms()
            .iter()
            .find(|t| t.string == xx)
            .map(|t| t.coefficient)
            .unwrap();
        // Issue #529 quotes 0.1812888024, from orbitals without the
        // renormalized contraction (5.8e-9 lower), like its energy.
        assert!((c - 0.181_288_808_2).abs() <= 1e-10, "{c}");
        let strings: Vec<String> = h.terms().iter().map(|t| t.string.to_string()).collect();
        assert_eq!(strings, ["I", "X0 X1", "Z0", "Z0 Z1", "Z1"]);
    }

    #[test]
    fn jordan_wigner_has_the_fifteen_expected_strings() {
        let h = H2Molecule::new(0.7414, H2Encoding::JordanWigner).unwrap();
        assert_eq!(h.num_qubits(), 4);
        let strings: Vec<String> = h
            .to_pauli_sum()
            .terms()
            .iter()
            .map(|t| t.string.to_string())
            .collect();
        assert_eq!(
            strings,
            [
                "I",
                "X0 X1 X2 X3",
                "X0 X1 Y2 Y3",
                "Y0 Y1 X2 X3",
                "Y0 Y1 Y2 Y3",
                "Z0",
                "Z0 Z1",
                "Z0 Z2",
                "Z0 Z3",
                "Z1",
                "Z1 Z2",
                "Z1 Z3",
                "Z2",
                "Z2 Z3",
                "Z3",
            ]
        );
    }

    #[test]
    fn construction_is_deterministic() {
        for encoding in ENCODINGS {
            let a = H2Molecule::new(1.3, encoding).unwrap().to_pauli_sum();
            let b = H2Molecule::new(1.3, encoding).unwrap().to_pauli_sum();
            assert_eq!(a, b);
            assert_eq!(a.to_string(), b.to_string());
        }
    }

    #[test]
    fn bond_length_round_trips_through_bohr() {
        let h2 = H2Molecule::new(0.7414, H2Encoding::JordanWigner).unwrap();
        assert!((h2.bond_length_angstrom() - 0.7414).abs() < 1e-15);
        assert!((h2.bond_length_bohr() - 0.7414 / BOHR_RADIUS_ANGSTROM).abs() < 1e-15);
        assert_eq!(h2.encoding(), H2Encoding::JordanWigner);
        // σ–σ* one-body coupling vanishes by symmetry (up to rounding).
        assert!(h2.fermion_hamiltonian().one_body(0, 1).abs() < 1e-15);
    }

    /// Rows of a CSV file in `reference-data/h2-sto3g/` (generated with
    /// PySCF + OpenFermion, see its README), header skipped.
    fn reference_rows(file: &str) -> Vec<Vec<String>> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../reference-data/h2-sto3g/"
        );
        let text = std::fs::read_to_string(format!("{path}{file}"))
            .unwrap_or_else(|e| panic!("read reference file {path}{file}: {e}"));
        text.lines()
            .skip(1)
            .filter(|line| !line.is_empty())
            .map(|line| line.split(',').map(str::to_string).collect())
            .collect()
    }

    #[test]
    fn jordan_wigner_matches_the_openfermion_reference() {
        // R (as written in the file) -> (Pauli string -> coefficient).
        let mut reference: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for row in reference_rows("jw_coefficients.csv") {
            reference
                .entry(row[0].clone())
                .or_default()
                .insert(row[1].clone(), row[2].parse().unwrap());
        }
        assert_eq!(reference.len(), 7, "bond lengths in the reference");
        for (r, expected) in &reference {
            let ours: BTreeMap<String, f64> =
                H2Molecule::new(r.parse().unwrap(), H2Encoding::JordanWigner)
                    .unwrap()
                    .to_pauli_sum()
                    .terms()
                    .iter()
                    .map(|t| (t.string.to_string(), t.coefficient))
                    .collect();
            let ours_strings: Vec<&String> = ours.keys().collect();
            let expected_strings: Vec<&String> = expected.keys().collect();
            assert_eq!(ours_strings, expected_strings, "R = {r} Å: Pauli strings");
            for (s, c) in expected {
                assert!(
                    (ours[s] - c).abs() <= 1e-6,
                    "R = {r} Å, {s}: {} vs {c}",
                    ours[s]
                );
            }
        }
    }

    #[test]
    fn fci_energies_match_the_pyscf_reference() {
        let rows = reference_rows("fci_energies.csv");
        assert_eq!(rows.len(), 7, "bond lengths in the reference");
        for row in rows {
            let r: f64 = row[0].parse().unwrap();
            let fci: f64 = row[2].parse().unwrap();
            for encoding in ENCODINGS {
                let e = ground(r, encoding);
                assert!(
                    (e - fci).abs() <= 1e-8,
                    "R = {r} Å, {encoding:?}: {e} vs {fci}"
                );
            }
        }
    }

    /// Closed-shell Hartree–Fock energy `2 h_σσ + (σσ|σσ) + E_nuc`.
    fn hartree_fock_energy(h2: &H2Molecule) -> f64 {
        let f = h2.fermion_hamiltonian();
        2.0 * f.one_body(0, 0) + f.two_body(0, 0, 0, 0) + f.nuclear_repulsion()
    }

    #[test]
    fn hartree_fock_energies_match_the_pyscf_reference() {
        // Pins the σ/σ* integrals themselves, not only the FCI eigenvalue.
        let rows = reference_rows("fci_energies.csv");
        assert_eq!(rows.len(), 7, "bond lengths in the reference");
        for row in rows {
            let r: f64 = row[0].parse().unwrap();
            let hf: f64 = row[1].parse().unwrap();
            let h2 = H2Molecule::new(r, H2Encoding::JordanWigner).unwrap();
            let e = hartree_fock_energy(&h2);
            assert!((e - hf).abs() <= 1e-8, "R = {r} Å: {e} vs {hf}");
            // The same energy is the qubit Hamiltonian's diagonal at the
            // Hartree–Fock state |0101⟩ (σ↑ and σ↓ occupied, block order).
            let diagonal = h2
                .to_pauli_sum()
                .to_dense_real()
                .unwrap()
                .get(0b0101, 0b0101)
                .unwrap();
            assert!((diagonal - e).abs() <= 1e-12, "R = {r} Å");
        }
    }

    #[test]
    fn hartree_fock_matches_szabo_ostlund_at_1_4_bohr() {
        // Szabo & Ostlund ch. 3: E₀ = −1.8310 (electronic), −1.1167 (total).
        let h2 = H2Molecule::new(1.4 * BOHR_RADIUS_ANGSTROM, H2Encoding::JordanWigner).unwrap();
        let total = hartree_fock_energy(&h2);
        let electronic = total - h2.fermion_hamiltonian().nuclear_repulsion();
        assert!((total + 1.1167).abs() <= 5e-5, "{total}");
        assert!((electronic + 1.8310).abs() <= 5e-5, "{electronic}");
    }

    #[test]
    fn invalid_bond_lengths_are_rejected() {
        for r in [0.0, -0.7, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    H2Molecule::new(r, H2Encoding::JordanWigner),
                    Err(PhysicsError::InvalidIntegrals { .. })
                ),
                "{r}"
            );
        }
        assert!(matches!(
            H2Molecule::new(1e-6, H2Encoding::JordanWigner),
            Err(PhysicsError::UnsupportedOperation { .. })
        ));
        assert!(H2Molecule::new(0.3, H2Encoding::JordanWigner).is_ok());
    }
}
