//! Single-photon polarization Hamiltonian and the [`PauliSum`] data contract.

/// Axis of a single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliAxis {
    /// Pauli-X (σ_x).
    X,
    /// Pauli-Y (σ_y).
    Y,
    /// Pauli-Z (σ_z).
    Z,
}

/// One term in a Pauli-sum Hamiltonian: a coefficient times a single Pauli
/// operator acting on one qubit.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliTerm {
    /// Energy coefficient in the caller's unit convention (e.g. eV with `ħ=1`,
    /// or rad/s in SI). The same convention must be used when a Trotter
    /// compiler multiplies by an evolution time `t`.
    pub coefficient: f64,
    /// Index of the qubit this Pauli acts on (0-based).
    pub qubit: usize,
    /// Which Pauli operator: X, Y, or Z.
    pub axis: PauliAxis,
}

/// A Hamiltonian `H = Σ_k c_k · P_k` expressed as a list of [`PauliTerm`]s.
///
/// This is the data contract between `polypus-physics` (physics definition)
/// and `polypus-circuit` (future Trotterization). It deliberately contains no
/// circuit-compilation logic.
#[derive(Debug, Clone)]
pub struct PauliSum {
    /// Total number of qubits spanned by the Hamiltonian.
    pub num_qubits: usize,
    /// The Pauli terms. Order does not affect correctness but may affect
    /// Trotter error; physics constructors should group commuting terms.
    pub terms: Vec<PauliTerm>,
}

impl PauliSum {
    /// Sum of absolute values of all coefficients.
    ///
    /// Useful for estimating the operator norm `‖H‖` and hence the Trotter
    /// step size needed for a target error.
    pub fn coefficient_norm(&self) -> f64 {
        self.terms.iter().map(|t| t.coefficient.abs()).sum()
    }
}

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
/// # Example
/// ```
/// use polypus_physics::hamiltonians::photon_polarization::photon_polarization_hamiltonian;
///
/// // 1 eV photon (ħ=1 convention), weak coupling Ω = 0.1 eV.
/// let h = photon_polarization_hamiltonian(1.0, 0.1);
/// assert_eq!(h.num_qubits, 1);
/// assert_eq!(h.terms.len(), 2);
/// ```
pub fn photon_polarization_hamiltonian(omega: f64, coupling_omega: f64) -> PauliSum {
    PauliSum {
        num_qubits: 1,
        terms: vec![
            PauliTerm {
                coefficient: omega,
                qubit: 0,
                axis: PauliAxis::Z,
            },
            PauliTerm {
                coefficient: coupling_omega,
                qubit: 0,
                axis: PauliAxis::X,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamiltonian_has_one_qubit() {
        assert_eq!(photon_polarization_hamiltonian(1.0, 0.1).num_qubits, 1);
    }

    #[test]
    fn hamiltonian_has_two_terms() {
        assert_eq!(photon_polarization_hamiltonian(1.0, 0.1).terms.len(), 2);
    }

    #[test]
    fn term_axes_and_coefficients() {
        let h = photon_polarization_hamiltonian(1.0, 0.1);
        assert_eq!(h.terms[0].axis, PauliAxis::Z);
        assert_eq!(h.terms[0].coefficient, 1.0);
        assert_eq!(h.terms[1].axis, PauliAxis::X);
        assert_eq!(h.terms[1].coefficient, 0.1);
    }

    #[test]
    fn coefficient_norm_sums_absolute_values() {
        let h = photon_polarization_hamiltonian(1.0, 0.1);
        assert!((h.coefficient_norm() - 1.1).abs() < 1e-12);
    }

    #[test]
    fn zero_coupling_is_diagonal() {
        let h = photon_polarization_hamiltonian(1.0, 0.0);
        let non_zero: Vec<_> = h
            .terms
            .iter()
            .filter(|t| t.coefficient.abs() > f64::EPSILON)
            .collect();
        assert_eq!(non_zero.len(), 1);
        assert_eq!(non_zero[0].axis, PauliAxis::Z);
    }
}
