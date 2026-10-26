//! # polypus-physics
//!
//! Pure-Rust particle-physics simulation layer for Polypus.
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
//! | [`hamiltonians`] | Quantum Hamiltonians as `PauliSum` (data only) |
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
//! ## Hamiltonian bridge (data contract for future VQE/QPE)
//!
//! ```
//! use polypus_physics::hamiltonians::photon_polarization::photon_polarization_hamiltonian;
//!
//! // Express H = 1.0·σ_z + 0.1·σ_x  (natural units, ħ=1).
//! let h = photon_polarization_hamiltonian(1.0, 0.1);
//! // h.to_string() or pass to polypus-circuit's Trotter compiler (future).
//! assert_eq!(h.coefficient_norm(), 1.1);
//! ```

#![deny(clippy::all)]

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
}
