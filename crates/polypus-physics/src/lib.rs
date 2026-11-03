//! # polypus-physics
//!
//! Pure-Rust physics layer for Polypus: classical Monte Carlo particle
//! transport, and quantum Hamiltonians, including molecular ones (H₂, STO-3G).
//! No Python, no PyO3, no external physics framework.
//!
//! ## Layers
//!
//! | Layer | Responsibility |
//! |---|---|
//! | [`particle`] | Species definitions and kinematic state |
//! | [`medium`] | Material properties |
//! | [`interactions`] | Cross-sections and kinematic sampling |
//! | [`monte_carlo`] | Classical MC transport engine |
//! | [`hamiltonians`] | Quantum Hamiltonians as Pauli sums: fermionic operators, qubit mappings, tapering, exact diagonalization |
//! | [`chemistry`] | Molecular integrals (STO-3G) and molecules (H₂) built on [`hamiltonians`] |
//!
//! ## ENDF-6 photon data
//!
//! [`medium::compound::CompoundMedium`] and
//! [`interactions::photon::mass_attenuation_coefficients`] read the EPDL2023
//! photon cross-sections (`ZA{Z:03}000.txt`, one file per element) at runtime
//! from a local directory. The data is not part of the crate and is never
//! downloaded. The directory is `POLYPUS_ENDF_DATA_DIR` if set, otherwise
//! `reference-data/endf-epdl-2023/` of the Polypus repository, which holds the
//! files and their provenance. Outside a checkout (e.g. a dependency from
//! crates.io), fetch that folder and point `POLYPUS_ENDF_DATA_DIR` at it:
//!
//! ```text
//! git clone --depth 1 --filter=blob:none --sparse \
//!     https://github.com/Bahia-Software/polypus.git polypus-data
//! git -C polypus-data sparse-checkout set reference-data/endf-epdl-2023
//! export POLYPUS_ENDF_DATA_DIR=$PWD/polypus-data/reference-data/endf-epdl-2023
//! ```
//!
//! The files are the unmodified IAEA evaluations, so saving
//! `https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/ZA{Z:03}000` from a
//! browser as `ZA{Z:03}000.txt` for the elements you use works too (the site
//! may refuse non-browser clients). The example below needs the data.
//!
//! ## Classical Monte Carlo — 1.25 MeV photon in water
//!
//! ```
//! use polypus_physics::{
//!     particle::photon::Photon,
//!     medium::compound::CompoundMedium,
//!     interactions::photon::PhotonInteractionModel,
//!     monte_carlo::{MonteCarloEngine, RunConfig},
//! };
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let water = CompoundMedium::new("H2O", 1000.0, 5000).unwrap();
//! let engine = MonteCarloEngine::new(
//!     Photon,
//!     water,
//!     PhotonInteractionModel,
//!     RunConfig { n_histories: 1_000, seed: 42, ..Default::default() },
//! );
//! let mut rng = StdRng::seed_from_u64(42);
//! let result  = engine.run(Photon::state_along_z(1.25), &mut rng).unwrap();
//! println!("Mean deposit: {:.4} MeV", result.mean_deposit_mev);
//! ```
//!
//! ## Quantum Hamiltonians
//!
//! Every Hamiltonian is a [`hamiltonians::PauliSum`] with real coefficients
//! over multi-qubit Pauli strings, the input that VQE, QPE or Trotter
//! compilation will consume. A molecule goes from integrals to qubits in
//! [`chemistry`]; small sums can be diagonalized exactly as a reference.
//!
//! ```
//! use polypus_physics::chemistry::{H2Encoding, H2Molecule};
//! use polypus_physics::hamiltonians::PhysicsHamiltonian;
//!
//! // H₂ at its equilibrium bond length, reduced to 2 qubits.
//! let h = H2Molecule::new(0.7414, H2Encoding::ParityTapered)
//!     .unwrap()
//!     .to_pauli_sum();
//! println!("{h}"); // -0.3399… I, +0.1812… X0 X1, +0.3939… Z0, …
//! assert!((h.ground_energy().unwrap() + 1.137270).abs() < 1e-5);
//! ```
//!
//! The single-photon polarization Hamiltonian `ω σ_z + Ω σ_x` is in
//! [`hamiltonians::photon_polarization`].

#![deny(clippy::all)]

pub mod chemistry;
pub mod constants;
pub mod error;
pub mod hamiltonians;
pub mod interactions;
pub mod medium;
pub mod monte_carlo;
pub mod particle;

#[cfg(feature = "plotters")]
pub(crate) mod fonts;

/// Test support shared by this crate's modules.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};

    /// A scratch directory under the system temp dir, removed on drop (also on
    /// panic). Its name holds the process id and a tag unique to the test, so
    /// neither parallel tests nor concurrent runs on a shared machine write to
    /// the same path.
    pub(crate) struct ScratchDir(PathBuf);

    impl ScratchDir {
        pub(crate) fn new(tag: &str) -> ScratchDir {
            let dir =
                std::env::temp_dir().join(format!("polypus-physics-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create the scratch directory");
            ScratchDir(dir)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            // Report, never panic: a panic while unwinding aborts the test run.
            if let Err(e) = std::fs::remove_dir_all(&self.0) {
                eprintln!("could not remove {}: {e}", self.0.display());
            }
        }
    }

    /// Deterministic molecular integrals for `k` spatial orbitals with every
    /// real-orbital symmetry and magnitudes of order 1 (a small LCG, no
    /// dependency).
    pub(crate) fn symmetric_integrals(
        k: usize,
        seed: u64,
    ) -> crate::hamiltonians::FermionHamiltonian {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut one = vec![0.0; k * k];
        for p in 0..k {
            for q in p..k {
                let v = next();
                one[p * k + q] = v;
                one[q * k + p] = v;
            }
        }
        let idx = |p: usize, q: usize, r: usize, s: usize| ((p * k + q) * k + r) * k + s;
        let mut two = vec![f64::NAN; k * k * k * k];
        for p in 0..k {
            for q in 0..k {
                for r in 0..k {
                    for s in 0..k {
                        if !two[idx(p, q, r, s)].is_nan() {
                            continue;
                        }
                        let v = next();
                        for (a, b, c, d) in [
                            (p, q, r, s),
                            (q, p, r, s),
                            (p, q, s, r),
                            (q, p, s, r),
                            (r, s, p, q),
                            (s, r, p, q),
                            (r, s, q, p),
                            (s, r, q, p),
                        ] {
                            two[idx(a, b, c, d)] = v;
                        }
                    }
                }
            }
        }
        crate::hamiltonians::FermionHamiltonian::new(k, next(), one, two)
            .expect("generated integrals are symmetric and finite")
    }
}
