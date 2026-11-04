//! Quantum Hamiltonians as Pauli sums, and the generic tools to build and
//! check them.
//!
//! | Module | Content |
//! |---|---|
//! | [`pauli`] | Multi-qubit [`PauliString`], [`PauliTerm`], [`PauliSum`]: the common representation |
//! | [`exact`] | Dense matrix and exact eigenvalues of a small [`PauliSum`] (the reference for everything else) |
//! | [`fermion`] | [`FermionOperator`] and the electronic [`FermionHamiltonian`] from spatial integrals |
//! | [`mapping`] | Jordan–Wigner and parity mappings ([`QubitMapping`]) |
//! | [`taper`] | Removing qubits with a fixed `Z` value |
//! | [`photon_polarization`] | Single-photon polarization Hamiltonian |
//!
//! Nothing here is specific to chemistry: molecular integrals live in
//! [`crate::chemistry`]. The module contains no circuit-compilation or
//! measurement logic and has no dependency on `polypus-circuit`.

pub mod exact;
pub mod fermion;
pub mod mapping;
pub mod pauli;
pub mod photon_polarization;
pub mod taper;

pub use exact::{DenseMatrix, MAX_DENSE_QUBITS};
pub use fermion::{
    FermionHamiltonian, FermionOperator, FermionTerm, Ladder, INTEGRAL_SYMMETRY_TOLERANCE,
};
pub use mapping::{QubitMapping, HERMITICITY_TOLERANCE};
pub use pauli::{PauliAxis, PauliString, PauliSum, PauliTerm, Phase};
pub use taper::parity_reduction_qubits;

/// Trait for any physical system that can express its quantum Hamiltonian as a
/// sum of Pauli operators.
///
/// # Design note
///
/// Implementors only encode **what** the Hamiltonian is — which Pauli terms
/// and which physical coefficients. **How** that sum is compiled into a
/// quantum circuit (Trotterization, VQE ansatz, QPE encoding) is the
/// responsibility of `polypus-circuit` and the `polypus` runtime; this crate
/// depends on neither.
///
/// [`PauliSum`] is the shared representation for that work. Where it finally
/// lives (here or in `polypus-observable`, which will measure it) is decided
/// in issue #530; the physics constructors keep returning it either way.
pub trait PhysicsHamiltonian {
    /// Number of qubits required to represent this Hamiltonian.
    fn num_qubits(&self) -> usize;

    /// Express the Hamiltonian `H = Σ_k c_k · P_k` as a [`PauliSum`].
    fn to_pauli_sum(&self) -> PauliSum;
}
