//! The exact gradient by **adjoint differentiation**, split like the
//! parameter-shift one into what to run and how to read the results back.
//!
//! The parameter-shift rule runs two circuits per trainable gate. A caller
//! that holds the exact statevector (the native simulator) can instead get the
//! derivative of an expectation with respect to every gate of a circuit at
//! once, by adjoint differentiation, in about three simulations
//! (`polypus_sim::StatevectorSimulator::adjoint_gradient`). This crate never
//! executes a circuit, so it hands such a caller one [`AdjointRun`] per
//! simulation — the circuit, the gates to differentiate and the diagonal
//! observable — and turns the derivatives that come back into `∂fitness/∂θ`
//! ([`QmlProblem::gradient_from_adjoint`](crate::QmlProblem::gradient_from_adjoint))
//! or into a [`Jacobian`](crate::Jacobian)
//! ([`JacobianBlock::jacobian_from_adjoint`](crate::JacobianBlock::jacobian_from_adjoint)).
//!
//! The circuits are the parameter shift's untied templates bound at the
//! unshifted values, so the gates differentiated are exactly the ones the shift
//! rule moves, with the same per-occurrence bookkeeping: an angle that several
//! gates share is the sum of their derivatives, and a gate that reads the
//! sample adds its derivative times each feature's chain-rule factor.

use polypus_circuit::ConcreteCircuit;

use crate::error::QmlError;

/// A readout observable as the diagonal operator it is on the measured state:
/// `O|b⟩ = weight(b)|b⟩`, a weighted sum of `Z` products. The model already
/// rotates `X`/`Y` factors into the measured basis before the terminal
/// measurement, so every term is a parity of basis-state bits.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalObservable {
    /// `(coefficient, mask)`: the term is `coefficient · (−1)^{|b ∧ mask|}`.
    terms: Vec<(f64, usize)>,
}

impl DiagonalObservable {
    /// Build from `(coefficient, physical qubits)` terms.
    ///
    /// A qubit at or past `usize::BITS` cannot be a bit of a basis-state index:
    /// [`QmlError::NoExactGradient`].
    pub(crate) fn from_terms<'a>(
        terms: impl IntoIterator<Item = (f64, &'a [usize])>,
    ) -> Result<Self, QmlError> {
        let terms = terms
            .into_iter()
            .map(|(coefficient, qubits)| {
                let mut mask = 0usize;
                for &qubit in qubits {
                    let bit = u32::try_from(qubit)
                        .ok()
                        .and_then(|q| 1usize.checked_shl(q))
                        .ok_or_else(|| QmlError::NoExactGradient {
                            reason: format!(
                                "the readout reads qubit {qubit}, past what a basis-state index \
                                 can address"
                            ),
                        })?;
                    mask |= bit;
                }
                Ok((coefficient, mask))
            })
            .collect::<Result<Vec<_>, QmlError>>()?;
        Ok(DiagonalObservable { terms })
    }

    /// `Σ_j coefficients[j] · observables[j]`, keeping every term (in order).
    pub(crate) fn combine(coefficients: &[f64], observables: &[DiagonalObservable]) -> Self {
        let terms = coefficients
            .iter()
            .zip(observables)
            .flat_map(|(&c, observable)| {
                observable
                    .terms
                    .iter()
                    .map(move |&(coefficient, mask)| (c * coefficient, mask))
            })
            .collect();
        DiagonalObservable { terms }
    }

    /// `Σ |c|` over the terms, summed in order: a bound on every eigenvalue,
    /// `|weight(b)| ≤ weight_bound()` for every basis state `b`, rounding
    /// included (each partial sum of [`weight`](Self::weight) is bounded by the
    /// matching partial sum here, rounding being monotonic). Finite exactly when
    /// no coefficient is non-finite and the sum does not overflow, so a finite
    /// bound means no eigenvalue is non-finite: a simulator that refuses to
    /// weigh a state by a non-finite value will take this observable.
    pub fn weight_bound(&self) -> f64 {
        self.terms
            .iter()
            .fold(0.0, |sum, &(coefficient, _)| sum + coefficient.abs())
    }

    /// The eigenvalue of basis state `basis_state` (qubit `q` is bit `q`, the
    /// index of a statevector's amplitudes): `Σ c · (−1)^{|b ∧ mask|}` over the
    /// terms, summed in order. Pure, so it can be called from several threads.
    pub fn weight(&self, basis_state: usize) -> f64 {
        self.terms.iter().fold(0.0, |sum, &(coefficient, mask)| {
            if (basis_state & mask).count_ones() % 2 == 0 {
                sum + coefficient
            } else {
                sum - coefficient
            }
        })
    }
}

/// One adjoint differentiation to run: the derivatives of
/// `⟨ψ|observable|ψ⟩`, with `|ψ⟩` the final state of `circuit`, with respect
/// to the angle of each gate at a position in `wrt`.
///
/// The circuit ends in its `measure_all` and is to be simulated **as written**
/// — not transpiled, nor otherwise rewritten — since `wrt` holds positions in
/// it. Every gate in `wrt` is a Pauli rotation (`Rx`, `Ry`, `Rz`, `Rxx`,
/// `Rzz`). The result handed back is one derivative per entry of `wrt`, in
/// `wrt`'s order.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjointRun {
    /// The untied template bound at the unshifted values.
    pub circuit: ConcreteCircuit,
    /// Positions in `circuit.gates()` of the gates to differentiate.
    pub wrt: Vec<usize>,
    /// The observable to differentiate.
    pub observable: DiagonalObservable,
}

/// Check `derivatives` against the runs they answer: one row per run, each as
/// long as its run's `wrt`. A wrong count is [`QmlError::CountsLengthMismatch`]
/// with the expected number of rows; a wrong row, the same error with the
/// expected row length.
pub(crate) fn check_rows(
    derivatives: &[Vec<f64>],
    runs: usize,
    row_length: usize,
) -> Result<(), QmlError> {
    if derivatives.len() != runs {
        return Err(QmlError::CountsLengthMismatch {
            expected: runs,
            got: derivatives.len(),
        });
    }
    if let Some(row) = derivatives.iter().find(|row| row.len() != row_length) {
        return Err(QmlError::CountsLengthMismatch {
            expected: row_length,
            got: row.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weight_is_the_signed_sum_of_the_terms() {
        let observable = DiagonalObservable::from_terms([
            (0.5, [0usize, 2].as_slice()),
            (-1.5, [1usize].as_slice()),
            (2.0, [].as_slice()),
        ])
        .unwrap();
        // b = 0b101: Z0Z2 parity even (+0.5), Z1 bit 0 (−1.5·+1), identity +2.
        assert_eq!(observable.weight(0b101), 0.5 - 1.5 + 2.0);
        // b = 0b011: Z0Z2 odd (−0.5), Z1 set (+1.5).
        assert_eq!(observable.weight(0b011), -0.5 + 1.5 + 2.0);
    }

    #[test]
    fn combining_scales_each_observable_by_its_coefficient() {
        let z0 = DiagonalObservable::from_terms([(1.0, [0usize].as_slice())]).unwrap();
        let z1 = DiagonalObservable::from_terms([(2.0, [1usize].as_slice())]).unwrap();
        let combined = DiagonalObservable::combine(&[3.0, -0.5], &[z0.clone(), z1.clone()]);
        for b in 0..4 {
            assert_eq!(combined.weight(b), 3.0 * z0.weight(b) - 0.5 * z1.weight(b));
        }
    }

    #[test]
    fn the_weight_bound_bounds_every_eigenvalue_and_flags_an_overflow() {
        let observable = DiagonalObservable::from_terms([
            (0.5, [0usize, 2].as_slice()),
            (-1.5, [1usize].as_slice()),
            (2.0, [].as_slice()),
        ])
        .unwrap();
        assert_eq!(observable.weight_bound(), 4.0);
        for b in 0..8 {
            assert!(observable.weight(b).abs() <= observable.weight_bound());
        }
        // Finite coefficients whose sum overflows: the bound says so, and so
        // does the eigenvalue it bounds.
        let z0 = DiagonalObservable::from_terms([(1.0, [0usize].as_slice())]).unwrap();
        let huge = DiagonalObservable::combine(&[1e308, 1e308], &[z0.clone(), z0.clone()]);
        assert!(!huge.weight_bound().is_finite());
        assert!(!huge.weight(0).is_finite());
        // A large but representable sum stays finite.
        let large = DiagonalObservable::combine(&[1e307, -1e307], &[z0.clone(), z0.clone()]);
        assert!(large.weight_bound().is_finite());
        // A non-finite coefficient makes the bound non-finite.
        let nan = DiagonalObservable::combine(&[f64::NAN], &[z0]);
        assert!(nan.weight_bound().is_nan());
    }

    #[test]
    fn a_qubit_past_the_index_width_is_a_typed_error() {
        let wide = [usize::BITS as usize];
        assert!(matches!(
            DiagonalObservable::from_terms([(1.0, wide.as_slice())]),
            Err(QmlError::NoExactGradient { .. })
        ));
    }

    #[test]
    fn rows_are_checked_against_the_runs() {
        assert_eq!(check_rows(&[vec![0.0; 3], vec![0.0; 3]], 2, 3), Ok(()));
        assert_eq!(
            check_rows(&[vec![0.0; 3]], 2, 3),
            Err(QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            check_rows(&[vec![0.0; 3], vec![0.0; 2]], 2, 3),
            Err(QmlError::CountsLengthMismatch {
                expected: 3,
                got: 2
            })
        );
    }
}
