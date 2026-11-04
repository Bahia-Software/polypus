//! Quantum chemistry: molecular integrals for electronic Hamiltonians.
//!
//! This layer is chemistry-specific; it produces integrals that the generic
//! [`crate::hamiltonians`] layer turns into fermionic and qubit operators.
//! Units are atomic: lengths in bohr, energies in hartree.
//!
//! ```
//! use polypus_physics::chemistry::{overlap, ContractedGaussian};
//!
//! // H₂ at 1.4 bohr: overlap of the two STO-3G 1s orbitals (Szabo & Ostlund).
//! let a = ContractedGaussian::sto3g_hydrogen_1s([0.0, 0.0, 0.0]).unwrap();
//! let b = ContractedGaussian::sto3g_hydrogen_1s([0.0, 0.0, 1.4]).unwrap();
//! assert!((overlap(&a, &b) - 0.6593).abs() < 5e-5);
//! ```

pub mod h2;
pub mod sto3g;

pub use h2::{H2Encoding, H2Molecule, H2_ROUNDING_TOLERANCE};
pub use sto3g::{
    electron_repulsion, kinetic, nuclear_attraction, overlap, ContractedGaussian, Nucleus,
};
