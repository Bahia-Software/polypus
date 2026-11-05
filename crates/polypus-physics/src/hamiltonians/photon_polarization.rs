//! Single-photon polarization Hamiltonian.

use super::pauli::{PauliAxis, PauliString, PauliSum, PauliTerm};
use crate::error::PhysicsError;

/// Single-photon polarization Hamiltonian:
///
/// ```text
/// H = ω · σ_z  +  Ω · σ_x
/// ```
///
/// Physical interpretation (1-qubit system):
/// - `|0⟩ ≡` horizontal polarization `|H⟩`
/// - `|1⟩ ≡` vertical polarization `|V⟩`
///
/// Parameters:
/// - `omega` (`ω`): free energy splitting between `|H⟩` and `|V⟩`. In SI,
///   `ω = E/ħ` (rad/s); in natural units (`ħ=1`), `ω = E` (eV).
/// - `coupling_omega` (`Ω`): transverse coupling amplitude (birefringence,
///   external driving field). Same units as `ω`.
///
/// The returned [`PauliSum`] can be passed directly to `polypus-circuit`'s
/// Trotter compiler (future) to obtain a time-evolution circuit for VQE or QPE
/// execution on the polypus distributed runtime.
///
/// # Errors
/// [`PhysicsError::InvalidPauliSum`] if `omega` or `coupling_omega` is not
/// finite.
///
/// # Example
/// ```
/// use polypus_physics::hamiltonians::photon_polarization::photon_polarization_hamiltonian;
///
/// // 1 eV photon (ħ=1 convention), weak coupling Ω = 0.1 eV.
/// let h = photon_polarization_hamiltonian(1.0, 0.1).unwrap();
/// assert_eq!(h.num_qubits(), 1);
/// assert_eq!(h.terms().len(), 2);
/// ```
pub fn photon_polarization_hamiltonian(
    omega: f64,
    coupling_omega: f64,
) -> Result<PauliSum, PhysicsError> {
    PauliSum::new(
        1,
        vec![
            PauliTerm::new(omega, PauliString::single(0, PauliAxis::Z)),
            PauliTerm::new(coupling_omega, PauliString::single(0, PauliAxis::X)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamiltonian_has_one_qubit() {
        assert_eq!(
            photon_polarization_hamiltonian(1.0, 0.1)
                .unwrap()
                .num_qubits(),
            1
        );
    }

    #[test]
    fn hamiltonian_has_two_terms() {
        assert_eq!(
            photon_polarization_hamiltonian(1.0, 0.1)
                .unwrap()
                .terms()
                .len(),
            2
        );
    }

    #[test]
    fn term_axes_and_coefficients() {
        let h = photon_polarization_hamiltonian(1.0, 0.1).unwrap();
        assert_eq!(h.terms()[0].string, PauliString::single(0, PauliAxis::Z));
        assert_eq!(h.terms()[0].coefficient, 1.0);
        assert_eq!(h.terms()[1].string, PauliString::single(0, PauliAxis::X));
        assert_eq!(h.terms()[1].coefficient, 0.1);
    }

    #[test]
    fn coefficient_norm_sums_absolute_values() {
        let h = photon_polarization_hamiltonian(1.0, 0.1).unwrap();
        assert!((h.coefficient_norm() - 1.1).abs() < 1e-12);
    }

    #[test]
    fn zero_coupling_is_diagonal() {
        let h = photon_polarization_hamiltonian(1.0, 0.0).unwrap();
        let non_zero: Vec<_> = h
            .terms()
            .iter()
            .filter(|t| t.coefficient.abs() > f64::EPSILON)
            .collect();
        assert_eq!(non_zero.len(), 1);
        assert_eq!(non_zero[0].string, PauliString::single(0, PauliAxis::Z));
    }

    #[test]
    fn non_finite_parameters_are_rejected() {
        assert!(matches!(
            photon_polarization_hamiltonian(f64::NAN, 0.1),
            Err(PhysicsError::InvalidPauliSum { .. })
        ));
        assert!(matches!(
            photon_polarization_hamiltonian(1.0, f64::INFINITY),
            Err(PhysicsError::InvalidPauliSum { .. })
        ));
    }
}
