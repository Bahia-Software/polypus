//! Exact diagonalization of a [`FermionHamiltonian`] within one electron
//! sector `(N↑, N↓)` (issue #535).
//!
//! The electronic Hamiltonian conserves the number of spin-up and spin-down
//! electrons, so its matrix is block-diagonal: one block per sector `(N↑, N↓)`.
//! The energy of a molecule with given charge and spin projection
//! `M_S = (N↑ − N↓)/2` is the lowest eigenvalue of **its** block, which is
//! much smaller than the whole matrix
//! (`C(K, N↑) · C(K, N↓)` determinants against `4^K`; for BeH₂ in STO-3G,
//! `K = 7`, `N↑ = N↓ = 3`: 1225 against 16384). The lowest eigenvalue of the
//! whole matrix may belong to another number of electrons (an ion's neighbour,
//! or a sector the parity tapering of [`super::taper`] keeps).
//!
//! A sector fixes `N` and `M_S`, **not** the total spin `S²` nor the spatial
//! symmetry: its lowest eigenvalue is the lowest state with that `M_S` and
//! any total spin `S ≥ |M_S|`, in any irreducible representation. For
//! `N↑ = N↓` that is the singlet only when no triplet (or higher spin) lies
//! below it. The eigenvalues carry no label of `S`, but the neighbouring
//! sector tells them apart: a spin-`S` multiplet has one state for each
//! `M_S` from `−S` to `S`, so for `N↑ ≥ N↓` the roots of `(N↑, N↓)` with
//! `S > M_S` are exactly those that are also roots of `(N↑ + 1, N↓ − 1)`,
//! with multiplicity, and the rest have `S = M_S`. With
//! [`FermionHamiltonian::sector_eigenvalues`] of both sectors, the lowest
//! root of `(N, N)` left over is the lowest singlet. The spatial symmetry
//! cannot be told apart this way. The ground states of H₂, LiH and BeH₂ are
//! singlets.
//!
//! # Method
//!
//! A basis determinant is a bitmask over the `2K` spin-orbitals in the block
//! order of [`FermionHamiltonian`] (bit `p` is spatial orbital `p` spin-up,
//! bit `p + K` the same orbital spin-down). The sector's determinants are
//! enumerated in ascending order, every term of
//! [`FermionHamiltonian::to_fermion_operator`] is applied to each of them
//! with the fermionic sign, and the dense block is diagonalized with
//! [`DenseMatrix::symmetric_eigenvalues`]. The cost is `O(d³)` in time and
//! two `d × d` matrices in memory for a sector of dimension `d`, hence
//! [`MAX_SECTOR_DIMENSION`]; larger sectors need an iterative method
//! (Lanczos), not implemented.

use super::exact::DenseMatrix;
use super::fermion::{apply_term, FermionHamiltonian};
use crate::error::PhysicsError;

/// The largest sector [`FermionHamiltonian::sector_eigenvalues`]
/// diagonalizes: 2500 determinants, two 2500 × 2500 matrices (about 100 MB),
/// a few seconds in an optimized build. The dense solver grows as `d³`: 4096
/// determinants take minutes. The fermionic operator the matrix is built
/// from has up to `4K⁴` terms besides, and each is applied to every
/// determinant, so with many orbitals building the matrix costs more than
/// diagonalizing it: `terms × d` applications (with dense integrals at
/// `K = 32`, about 4·10⁶ terms, a few hundred MB, and ~10¹⁰ applications
/// for a sector of 2500 determinants, minutes). Building the block straight
/// from the integrals (Slater–Condon rules) would avoid it; not implemented.
/// Molecules of a few orbitals, as in this crate's tests, take milliseconds.
pub const MAX_SECTOR_DIMENSION: usize = 2500;

/// The number of determinants of the sector `(n_up, n_down)` in `num_spatial`
/// orbitals, `C(K, N↑) · C(K, N↓)`; `None` if it overflows `usize`. A sector
/// with more electrons of one spin than orbitals is empty (`Some(0)`).
pub fn sector_dimension(num_spatial: usize, n_up: usize, n_down: usize) -> Option<usize> {
    binomial(num_spatial, n_up)?.checked_mul(binomial(num_spatial, n_down)?)
}

/// `C(n, k)`, or `None` if it overflows `usize`; `0` for `k > n`.
fn binomial(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut value: u128 = 1;
    for i in 0..k {
        // Exact at every step: the running product is C(n, i + 1).
        value = value.checked_mul((n - i) as u128)? / (i as u128 + 1);
    }
    usize::try_from(value).ok()
}

/// Every `k`-bit mask with exactly `ones` bits set, ascending (Gosper's hack).
/// The caller guarantees `ones <= k` and `k < usize::BITS`.
pub(super) fn combinations(k: usize, ones: usize) -> Vec<usize> {
    if ones == 0 {
        return vec![0];
    }
    let limit = 1usize << k;
    let mut masks = Vec::new();
    let mut mask = (1usize << ones) - 1;
    while mask < limit {
        masks.push(mask);
        let lowest = mask & mask.wrapping_neg();
        let ripple = mask + lowest;
        mask = (((ripple ^ mask) >> 2) / lowest) | ripple;
    }
    masks
}

impl FermionHamiltonian {
    /// Every eigenvalue of the Hamiltonian restricted to the determinants
    /// with `n_up` spin-up and `n_down` spin-down electrons, ascending (see
    /// the [module docs](super::sector)). The lowest is the sector's FCI
    /// energy, [`sector_ground_energy`](Self::sector_ground_energy).
    ///
    /// # Errors
    /// - [`PhysicsError::InvalidSector`] if `n_up` or `n_down` exceeds the
    ///   number of spatial orbitals.
    /// - [`PhysicsError::UnsupportedOperation`] if the sector has more than
    ///   [`MAX_SECTOR_DIMENSION`] determinants, or the `2K` spin-orbitals do
    ///   not fit in a `usize` bitmask.
    /// - As [`DenseMatrix::symmetric_eigenvalues`].
    pub fn sector_eigenvalues(&self, n_up: usize, n_down: usize) -> Result<Vec<f64>, PhysicsError> {
        let k = self.num_spatial_orbitals();
        if n_up > k || n_down > k {
            return Err(PhysicsError::InvalidSector {
                message: format!(
                    "{n_up} spin-up and {n_down} spin-down electrons do not fit in {k} spatial \
                     orbitals"
                ),
            });
        }
        if 2 * k > usize::BITS as usize {
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "{k} spatial orbitals: the {} spin-orbitals do not fit in a {}-bit \
                     determinant",
                    2 * k,
                    usize::BITS
                ),
            });
        }
        let dimension = sector_dimension(k, n_up, n_down);
        let Some(dimension) = dimension.filter(|&d| d <= MAX_SECTOR_DIMENSION) else {
            let count =
                dimension.map_or_else(|| "more than usize::MAX".to_string(), |d| d.to_string());
            return Err(PhysicsError::UnsupportedOperation {
                message: format!(
                    "the sector ({n_up}, {n_down}) of {k} spatial orbitals has {count} \
                     determinants, above the dense limit of {MAX_SECTOR_DIMENSION}; it needs an \
                     iterative solver"
                ),
            });
        };

        // Ascending: spin-down masks (the high bits) in the outer loop.
        let ups = combinations(k, n_up);
        let basis: Vec<usize> = combinations(k, n_down)
            .into_iter()
            .flat_map(|down| ups.iter().map(move |&up| up | (down << k)))
            .collect();
        debug_assert_eq!(basis.len(), dimension);

        let mut matrix = DenseMatrix::zeros(dimension);
        for term in self.to_fermion_operator().terms() {
            for (column, &determinant) in basis.iter().enumerate() {
                let Some((image, sign)) = apply_term(&term.ops, determinant) else {
                    continue;
                };
                // Every term conserves N↑ and N↓, so the image is in the
                // sector; reported rather than unwrapped if it ever is not.
                let row =
                    basis
                        .binary_search(&image)
                        .map_err(|_| PhysicsError::SimulationError {
                            message: format!(
                                "a Hamiltonian term maps determinant {determinant:#b} out of the \
                             sector ({n_up}, {n_down})"
                            ),
                        })?;
                matrix.add_to(row, column, sign * term.coefficient);
            }
        }
        matrix.symmetric_eigenvalues()
    }

    /// The lowest eigenvalue of the sector `(n_up, n_down)`: the FCI energy
    /// of the lowest state with that many electrons of each spin, of any total
    /// spin `S ≥ |N↑ − N↓|/2` and any spatial symmetry (see the
    /// [module docs](super::sector)).
    ///
    /// # Errors
    /// As [`sector_eigenvalues`](Self::sector_eigenvalues).
    pub fn sector_ground_energy(&self, n_up: usize, n_down: usize) -> Result<f64, PhysicsError> {
        self.sector_eigenvalues(n_up, n_down)?
            .first()
            .copied()
            .ok_or_else(|| PhysicsError::SimulationError {
                message: "a sector has at least one determinant".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::{H2Encoding, H2Molecule};
    use crate::hamiltonians::PhysicsHamiltonian;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// A random `FermionHamiltonian` of `k` orbitals with the symmetries of
    /// real integrals (fixed seed).
    fn random_hamiltonian(k: usize, seed: u64) -> FermionHamiltonian {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut one = vec![0.0; k * k];
        for p in 0..k {
            for q in 0..=p {
                let v = rng.random_range(-1.0..1.0);
                one[p * k + q] = v;
                one[q * k + p] = v;
            }
        }
        let idx = |p: usize, q: usize, r: usize, s: usize| ((p * k + q) * k + r) * k + s;
        let mut two = vec![0.0; k * k * k * k];
        for p in 0..k {
            for q in 0..k {
                for r in 0..k {
                    for s in 0..k {
                        let v = rng.random_range(-0.5..0.5);
                        for (a, b, c, d) in
                            crate::hamiltonians::fermion::chemist_permutations(p, q, r, s)
                        {
                            two[idx(a, b, c, d)] = v;
                        }
                    }
                }
            }
        }
        FermionHamiltonian::new(k, 0.7, one, two).expect("symmetric integrals")
    }

    /// The sectors partition the Fock space: the spectra of all sectors,
    /// together, are the spectrum of the whole matrix.
    #[test]
    fn the_sectors_together_have_the_whole_spectrum() {
        let h2 = H2Molecule::new(0.7414, H2Encoding::JordanWigner).expect("a valid length");
        let cases = [
            h2.fermion_hamiltonian().clone(),
            random_hamiltonian(3, 3),
            random_hamiltonian(3, 7),
        ];
        for hamiltonian in &cases {
            let k = hamiltonian.num_spatial_orbitals();
            let mut sectors = Vec::new();
            for n_up in 0..=k {
                for n_down in 0..=k {
                    let values = hamiltonian
                        .sector_eigenvalues(n_up, n_down)
                        .expect("a valid sector");
                    assert_eq!(Some(values.len()), sector_dimension(k, n_up, n_down));
                    sectors.extend(values);
                }
            }
            sectors.sort_by(f64::total_cmp);
            let whole = hamiltonian
                .to_fermion_operator()
                .to_dense_real()
                .expect("a small operator")
                .symmetric_eigenvalues()
                .expect("a symmetric matrix");
            assert_eq!(sectors.len(), whole.len());
            let scale = whole.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for (a, b) in sectors.iter().zip(&whole) {
                assert!((a - b).abs() <= 1e-12 * scale, "{a} vs {b}");
            }
        }
    }

    /// H₂'s FCI energy is the lowest eigenvalue of the one-up, one-down
    /// sector, as the full qubit Hamiltonian gives it.
    #[test]
    fn h2_sector_energy_is_its_fci_energy() {
        let h2 = H2Molecule::new(0.7414, H2Encoding::JordanWigner).expect("a valid length");
        let sector = h2
            .fermion_hamiltonian()
            .sector_ground_energy(1, 1)
            .expect("a valid sector");
        let full = h2.to_pauli_sum().ground_energy().expect("a small sum");
        assert!((sector - full).abs() <= 1e-12, "{sector} vs {full}");
        assert_eq!(sector_dimension(2, 1, 1), Some(4));
    }

    /// The rule of the module docs: for `N↑ ≥ N↓`, the roots of
    /// `(N↑ + 1, N↓ − 1)` are roots of `(N↑, N↓)` too, with multiplicity (the
    /// states with `S > M_S`), since spin-free integrals conserve `S²`. For
    /// H₂, the one root of `(2, 0)` is the triplet of `(1, 1)`, and the ground
    /// state of `(1, 1)` is not among them: a singlet.
    #[test]
    fn the_next_sector_holds_the_higher_spin_roots() {
        let h2 = H2Molecule::new(0.7414, H2Encoding::JordanWigner).expect("a valid length");
        let cases = [
            (h2.fermion_hamiltonian().clone(), 1, 1),
            (random_hamiltonian(3, 3), 1, 1),
            (random_hamiltonian(3, 3), 2, 1),
            (random_hamiltonian(3, 7), 2, 2),
        ];
        for (hamiltonian, n_up, n_down) in &cases {
            let lower = hamiltonian.sector_eigenvalues(*n_up, *n_down).unwrap();
            let upper = hamiltonian
                .sector_eigenvalues(n_up + 1, n_down - 1)
                .unwrap();
            let scale = lower.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            // Each root of the upper sector takes a distinct equal root of the
            // lower one (both ascending, so a greedy match is enough).
            let mut unmatched = lower.clone();
            for root in &upper {
                let position = unmatched
                    .iter()
                    .position(|v| (v - root).abs() <= 1e-10 * scale)
                    .unwrap_or_else(|| panic!("({n_up}, {n_down}): {root} is missing"));
                unmatched.remove(position);
            }
            if *n_up == 1 && *n_down == 1 && hamiltonian.num_spatial_orbitals() == 2 {
                // H₂: the lowest root left over is the ground state, a singlet.
                assert_eq!(unmatched[0], lower[0]);
                assert!(upper[0] > lower[0] + 0.1);
            }
        }
    }

    /// With no electrons only the core energy remains.
    #[test]
    fn the_empty_sector_is_the_core_energy() {
        let hamiltonian = random_hamiltonian(3, 11);
        // Exact: a 1 × 1 matrix holding the constant term alone.
        assert_eq!(hamiltonian.sector_eigenvalues(0, 0).unwrap(), vec![0.7]);
    }

    #[test]
    fn a_sector_with_too_many_electrons_is_rejected() {
        let hamiltonian = random_hamiltonian(2, 1);
        for (n_up, n_down) in [(3, 0), (0, 3), (3, 3)] {
            let error = hamiltonian.sector_eigenvalues(n_up, n_down).unwrap_err();
            assert!(
                matches!(error, PhysicsError::InvalidSector { .. }),
                "{error:?}"
            );
        }
    }

    /// A sector above the limit is refused from its dimension alone, before
    /// anything is allocated.
    #[test]
    fn a_sector_above_the_limit_is_rejected() {
        // K = 16, N↑ = N↓ = 8: C(16, 8)² = 12870² ≈ 1.66e8 determinants.
        let k = 16;
        let hamiltonian =
            FermionHamiltonian::new(k, 0.0, vec![0.0; k * k], vec![0.0; k * k * k * k])
                .expect("zero integrals");
        let error = hamiltonian.sector_eigenvalues(8, 8).unwrap_err();
        assert!(
            matches!(&error, PhysicsError::UnsupportedOperation { message }
                if message.contains("iterative") && message.contains("165636900 determinants")),
            "{error:?}"
        );
        assert_eq!(sector_dimension(16, 8, 8), Some(12870 * 12870));
        // BeH₂ in STO-3G (K = 7, three electrons of each spin) fits.
        assert!(sector_dimension(7, 3, 3).is_some_and(|d| d <= MAX_SECTOR_DIMENSION));
    }

    /// More than 32 spatial orbitals do not fit in a 64-bit determinant, even
    /// for a tiny sector; 32 do.
    #[test]
    fn spin_orbitals_beyond_the_bitmask_are_rejected() {
        let k = usize::BITS as usize / 2 + 1;
        let hamiltonian =
            FermionHamiltonian::new(k, 0.0, vec![0.0; k * k], vec![0.0; k * k * k * k])
                .expect("zero integrals");
        let error = hamiltonian.sector_eigenvalues(1, 0).unwrap_err();
        assert!(
            matches!(&error, PhysicsError::UnsupportedOperation { message } if message.contains("bit")),
            "{error:?}"
        );
        let k = usize::BITS as usize / 2;
        let hamiltonian =
            FermionHamiltonian::new(k, 0.5, vec![0.0; k * k], vec![0.0; k * k * k * k])
                .expect("zero integrals");
        // Exact: zero integrals leave the core energy on the diagonal.
        assert_eq!(hamiltonian.sector_ground_energy(1, 0).unwrap(), 0.5);
    }

    #[test]
    fn dimensions_and_combinations_are_exact() {
        assert_eq!(binomial(7, 3), Some(35));
        assert_eq!(binomial(3, 5), Some(0));
        assert_eq!(binomial(200, 100), None);
        assert_eq!(
            combinations(4, 2),
            vec![0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
        );
        assert_eq!(combinations(3, 0), vec![0]);
        assert_eq!(combinations(3, 3), vec![0b111]);
    }
}
