//! Quantum Hamiltonians expressed as Pauli sums (data only).
//!
//! This module defines the *data contract* between `polypus-physics` (which
//! says **what** a Hamiltonian is) and `polypus-circuit` (which, in a future
//! task, will compile a [`PauliSum`] into a Trotterized circuit). It contains
//! no circuit-compilation logic and has no dependency on `polypus-circuit`.

pub mod photon_polarization;

pub use photon_polarization::{PauliAxis, PauliSum, PauliTerm};

/// Trait for any physical system that can express its quantum Hamiltonian as a
/// sum of Pauli operators.
///
/// # Design note
///
/// Implementors only encode **what** the Hamiltonian is — which Pauli terms
/// and which physical coefficients. **How** that sum is compiled into a
/// quantum circuit (Trotterization, VQE ansatz, QPE encoding) is the
/// responsibility of `polypus-circuit` (a future `trotter` / `pauli` module)
/// and the `polypus` runtime. This crate depends on neither.
///
/// When `polypus-circuit` gains its own `PauliSum` type these definitions will
/// move there; the physics constructors will simply return that type.
pub trait PhysicsHamiltonian {
    /// Number of qubits required to represent this Hamiltonian.
    fn num_qubits(&self) -> usize;

    /// Express the Hamiltonian `H = Σ_k c_k · P_k` as a [`PauliSum`].
    fn to_pauli_sum(&self) -> PauliSum;
}
