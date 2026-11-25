//! Acceptance tests of issue #535 (first part) against the PySCF reference
//! data in `reference-data/fcidump-sto3g/` (see its README): H₂ read from
//! FCIDUMP equals the built-in construction integral by integral, and the
//! sector diagonalization of LiH and BeH₂ reproduces PySCF's FCI.
//!
//! The files are read at runtime, never embedded: jobs that check the crate
//! without `reference-data/` must still compile it.

use polypus_physics::chemistry::{Fcidump, H2Encoding, H2Molecule};
use polypus_physics::hamiltonians::{
    FermionHamiltonian, PauliAxis, PauliSum, QubitMapping, Tapering,
};

/// A file of `reference-data/fcidump-sto3g/`.
fn reference(file: &str) -> String {
    format!(
        "{}/../../reference-data/fcidump-sto3g/{file}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `(hf_energy, fci_energy)` of `molecule` at `geometry` from
/// `fci_energies.csv`.
fn energies(molecule: &str, geometry: &str) -> (f64, f64) {
    let csv = std::fs::read_to_string(reference("fci_energies.csv")).expect("the reference CSV");
    let row = csv
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect::<Vec<_>>())
        .find(|fields| fields[0] == molecule && fields[1] == geometry)
        .unwrap_or_else(|| panic!("{molecule} at {geometry} Å is in the CSV"));
    (
        row[4].parse().expect("an HF energy"),
        row[5].parse().expect("an FCI energy"),
    )
}

fn read(molecule: &str, geometry: &str) -> Fcidump {
    Fcidump::read(reference(&format!("{molecule}_{geometry}.fcidump"))).expect("a valid FCIDUMP")
}

/// The energy of the closed-shell determinant filling the lowest `occupied`
/// spatial orbitals with both spins:
/// `E_core + 2 Σ_i h_ii + Σ_ij [2 (ii|jj) − (ij|ji)]`, `i, j < occupied`.
fn closed_shell_energy(h: &FermionHamiltonian, occupied: usize) -> f64 {
    let one = |p, q| h.one_body(p, q).expect("an orbital");
    let two = |p, q, r, s| h.two_body(p, q, r, s).expect("an orbital");
    let mut energy = h.nuclear_repulsion();
    for i in 0..occupied {
        energy += 2.0 * one(i, i);
        for j in 0..occupied {
            energy += 2.0 * two(i, i, j, j) - two(i, j, j, i);
        }
    }
    energy
}

/// Issue #535's first criterion: H₂ read from FCIDUMP equals the built-in
/// STO-3G construction within 1e-10, integral by integral, at every bond
/// length, and so does its FCI energy (also PySCF's).
#[test]
fn h2_from_fcidump_equals_the_built_in_construction() {
    for geometry in ["0.3", "0.5", "0.7414", "1.0", "1.5", "2.0", "2.5"] {
        let file = read("h2", geometry);
        assert_eq!((file.num_orbitals(), file.sector()), (2, (1, 1)));
        let read = file.hamiltonian();
        let r: f64 = geometry.parse().expect("a bond length");
        let built = H2Molecule::new(r, H2Encoding::JordanWigner).expect("a valid length");
        let built = built.fermion_hamiltonian();

        let mut largest: f64 = (read.nuclear_repulsion() - built.nuclear_repulsion()).abs();
        for p in 0..2 {
            for q in 0..2 {
                largest = largest
                    .max((read.one_body(p, q).unwrap() - built.one_body(p, q).unwrap()).abs());
                for r in 0..2 {
                    for s in 0..2 {
                        largest = largest.max(
                            (read.two_body(p, q, r, s).unwrap()
                                - built.two_body(p, q, r, s).unwrap())
                            .abs(),
                        );
                    }
                }
            }
        }
        assert!(
            largest <= 1e-10,
            "R = {geometry} Å: integrals differ by {largest}"
        );

        let (_, fci) = energies("h2", geometry);
        let from_file = read.sector_ground_energy(1, 1).expect("a valid sector");
        let built_in = built.sector_ground_energy(1, 1).expect("a valid sector");
        assert!(
            (from_file - fci).abs() <= 1e-10,
            "R = {geometry} Å: {from_file} vs {fci}"
        );
        assert!((from_file - built_in).abs() <= 1e-10, "R = {geometry} Å");
    }
}

/// Issue #535's second criterion (before tapering): the sector
/// diagonalization reproduces PySCF's FCI within 1e-6; the Hartree–Fock
/// determinant's energy checks the integrals on their own, within 1e-10.
fn check_sector_fci(molecule: &str, geometry: &str, n_electrons: usize, dimension: usize) {
    let file = read(molecule, geometry);
    let (n_up, n_down) = file.sector();
    assert_eq!((n_up + n_down, file.ms2()), (n_electrons, 0));
    assert_eq!(
        polypus_physics::hamiltonians::sector_dimension(file.num_orbitals(), n_up, n_down),
        Some(dimension)
    );
    let (hf, fci) = energies(molecule, geometry);
    let h = file.hamiltonian();
    let closed_shell = closed_shell_energy(h, n_up);
    assert!(
        (closed_shell - hf).abs() <= 1e-10,
        "{molecule} at {geometry} Å: HF {closed_shell} vs {hf}"
    );
    let energy = h
        .sector_ground_energy(n_up, n_down)
        .expect("a sector within the limit");
    assert!(
        (energy - fci).abs() <= 1e-6,
        "{molecule} at {geometry} Å: sector FCI {energy} vs PySCF {fci}"
    );
    // Beyond the issue's 1e-6: both sides diagonalize the same integrals
    // exactly (measured agreement about 1e-13), so an error of 1e-7 in the
    // solver or the reader must not pass.
    assert!(
        (energy - fci).abs() <= 1e-9,
        "{molecule} at {geometry} Å: sector FCI {energy} vs PySCF {fci}"
    );
}

#[test]
fn lih_near_equilibrium_matches_pyscf_fci() {
    check_sector_fci("lih", "1.595", 4, 225);
}

#[test]
fn lih_stretched_matches_pyscf_fci() {
    check_sector_fci("lih", "3.0", 4, 225);
}

#[test]
fn beh2_near_equilibrium_matches_pyscf_fci() {
    check_sector_fci("beh2", "1.326", 6, 1225);
}

#[test]
fn beh2_stretched_matches_pyscf_fci() {
    check_sector_fci("beh2", "2.6", 6, 1225);
}

/// The diagonal element `⟨b|H|b⟩` of a qubit sum: the terms with no `X` or
/// `Y`, each `c · (−1)^k` for `k` of its `Z` qubits set in `b`.
fn diagonal_element(sum: &PauliSum, b: usize) -> f64 {
    sum.terms()
        .iter()
        .filter(|t| t.string.ops().iter().all(|&(_, axis)| axis == PauliAxis::Z))
        .map(|t| {
            let ones = t
                .string
                .ops()
                .iter()
                .filter(|&&(q, _)| b >> q & 1 == 1)
                .count();
            if ones % 2 == 0 {
                t.coefficient
            } else {
                -t.coefficient
            }
        })
        .sum()
}

/// Issue #535's second criterion, after tapering: the Hamiltonian tapered
/// by all its Z-type symmetries (values of the Hartree–Fock determinant),
/// restricted to the electron sector, still has the PySCF FCI energy, in
/// both encodings; the Hartree–Fock determinant mapped to the tapered space
/// still has the Hartree–Fock energy.
fn check_tapered_fci(molecule: &str, geometry: &str, removed: usize, tapered_dimension: usize) {
    let file = read(molecule, geometry);
    let (n_up, n_down) = file.sector();
    let k = file.num_orbitals();
    let (hf, fci) = energies(molecule, geometry);
    for mapping in [QubitMapping::JordanWigner, QubitMapping::Parity] {
        let (tapering, tapered) =
            Tapering::for_sector(file.hamiltonian(), mapping, n_up, n_down).expect("a tapering");
        assert_eq!(
            tapering.removed_qubits().len(),
            removed,
            "{molecule} at {geometry} Å, {mapping:?}"
        );
        assert_eq!(tapered.num_qubits(), 2 * k - removed);
        if mapping == QubitMapping::Parity {
            // At least the two qubits the parity reduction fixes.
            let qubits = tapering.removed_qubits();
            assert!(
                qubits.contains(&(k - 1)) && qubits.contains(&(2 * k - 1)),
                "{qubits:?}"
            );
        }
        let basis = tapering
            .sector_basis(mapping, n_up, n_down)
            .expect("a sector");
        assert_eq!(
            basis.len(),
            tapered_dimension,
            "{molecule} at {geometry} Å, {mapping:?}"
        );
        let energy = tapered
            .restricted_eigenvalues(&basis)
            .expect("an invariant sector")[0];
        assert!(
            (energy - fci).abs() <= 1e-6,
            "{molecule} at {geometry} Å, {mapping:?}: tapered {energy} vs PySCF {fci}"
        );
        // Beyond the issue's 1e-6, as before tapering: measured about 1e-13.
        assert!(
            (energy - fci).abs() <= 1e-9,
            "{molecule} at {geometry} Å, {mapping:?}: tapered {energy} vs PySCF {fci}"
        );
        let occupation = ((1 << n_up) - 1) | (((1 << n_down) - 1) << k);
        let reference = mapping
            .encode_occupation(occupation, 2 * k)
            .expect("an occupation");
        let hartree_fock = tapering
            .tapered_basis_state(reference)
            .expect("the reference is in the tapered space");
        let closed_shell = diagonal_element(&tapered, hartree_fock);
        assert!(
            (closed_shell - hf).abs() <= 1e-10,
            "{molecule} at {geometry} Å, {mapping:?}: HF {closed_shell} vs {hf}"
        );
    }
}

#[test]
fn lih_near_equilibrium_keeps_its_fci_energy_after_tapering() {
    check_tapered_fci("lih", "1.595", 4, 69);
}

#[test]
fn lih_stretched_keeps_its_fci_energy_after_tapering() {
    check_tapered_fci("lih", "3.0", 4, 69);
}

#[test]
fn beh2_near_equilibrium_keeps_its_fci_energy_after_tapering() {
    check_tapered_fci("beh2", "1.326", 5, 169);
}

#[test]
fn beh2_stretched_keeps_its_fci_energy_after_tapering() {
    check_tapered_fci("beh2", "2.6", 5, 169);
}

/// Open-shell sectors of LiH after tapering: the restricted spectra of every
/// value combination, together, are the sector's spectrum before tapering
/// (a spin swap in the sector basis or the reference would break it).
#[test]
fn lih_open_shell_sectors_split_into_their_value_blocks() {
    let file = read("lih", "1.595");
    let h = file.hamiltonian();
    for mapping in [QubitMapping::JordanWigner, QubitMapping::Parity] {
        let sum = h
            .to_fermion_operator()
            .to_qubits(mapping)
            .expect("a qubit Hamiltonian")
            .without_residues()
            .expect("residues dropped");
        let symmetries = sum.z_symmetries().expect("symmetries");
        let k = symmetries.len();
        for (n_up, n_down) in [(2, 1), (1, 2), (3, 1)] {
            let mut together = Vec::new();
            for combination in 0..1usize << k {
                let values: Vec<i8> = (0..k)
                    .map(|i| if combination >> i & 1 == 0 { 1 } else { -1 })
                    .collect();
                let tapering =
                    Tapering::with_values(sum.num_qubits(), &symmetries, &values).expect("valid");
                let tapered = tapering.apply(&sum).expect("tapered");
                let basis = tapering
                    .sector_basis(mapping, n_up, n_down)
                    .expect("a sector");
                together.extend(tapered.restricted_eigenvalues(&basis).expect("invariant"));
            }
            together.sort_by(f64::total_cmp);
            let sector = h.sector_eigenvalues(n_up, n_down).expect("a sector");
            assert_eq!(
                together.len(),
                sector.len(),
                "({n_up}, {n_down}) {mapping:?}"
            );
            for (a, b) in together.iter().zip(&sector) {
                assert!(
                    (a - b).abs() <= 1e-10,
                    "({n_up}, {n_down}) {mapping:?}: {a} vs {b}"
                );
            }
        }
    }
}
