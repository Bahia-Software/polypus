//! Readout: the observables a model measures and how their expectations become
//! a prediction (design doc §7).
//!
//! [`Readout`] is the public, unresolved type the user attaches to a
//! [`QuantumModel`](crate::QuantumModel); `compile` turns it into the internal
//! [`ResolvedReadout`] with physical qubit indices. A [`Decision`] maps the
//! observable expectations to a scalar prediction at inference time — it is a
//! separate step from the loss, which always operates on the raw `⟨O₀⟩`
//! (design doc §8).

use std::collections::HashMap;

use crate::error::{QmlError, ValidationError};
use crate::observables::{Observable, ResolvedObservable};

/// How observable expectations become a prediction at inference time.
///
/// `Decision` never participates in the loss (design doc §8): it is an
/// inference-time projection of the expectations, applied by
/// [`ResolvedReadout::predict`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Decision {
    /// Binary: `sign(⟨O₀⟩) → {−1, +1}`, with a tie at `0` resolving to `+1`.
    Sign,
    /// Binary with a threshold on `⟨O₀⟩`: `⟨O₀⟩ ≥ t → +1`, else `−1`.
    Threshold(f64),
    /// Multiclass: `argmax_k ⟨O_k⟩`, returned as the winning index. Ties go to
    /// the first (lowest) index.
    Argmax,
    /// Regression: `⟨O₀⟩` returned unchanged.
    Raw,
}

/// The observables a model reads out, plus the decision rule.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Readout {
    /// One or more observables. `Argmax` uses all of them (one per class); the
    /// binary/regression decisions use only `observables[0]`.
    pub observables: Vec<Observable>,
    /// How the expectations become a prediction.
    pub decision: Decision,
}

impl Readout {
    /// Build a readout, checking that the number of observables is compatible
    /// with the decision (design doc §8):
    ///
    /// - [`Decision::Argmax`] needs **at least two** observables.
    /// - [`Decision::Sign`] / [`Decision::Threshold`] / [`Decision::Raw`] need
    ///   **at least one** (they read `observables[0]`).
    ///
    /// A mismatch is [`ValidationError::DecisionObservableMismatch`]. (Whether
    /// a decision is trainable by the chosen loss is a separate check, made in
    /// [`QmlProblem::new`](crate::QmlProblem::new) where both coexist.)
    pub fn new(observables: Vec<Observable>, decision: Decision) -> Result<Self, ValidationError> {
        let min_observables = match decision {
            Decision::Argmax => 2,
            Decision::Sign | Decision::Threshold(_) | Decision::Raw => 1,
        };
        if observables.len() < min_observables {
            return Err(ValidationError::DecisionObservableMismatch {
                decision,
                num_observables: observables.len(),
            });
        }
        Ok(Readout {
            observables,
            decision,
        })
    }
}

/// A readout with physical qubit indices, produced by `compile`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedReadout {
    observables: Vec<ResolvedObservable>,
    decision: Decision,
}

impl ResolvedReadout {
    /// Wrap already-resolved observables and the decision.
    pub(crate) fn new(observables: Vec<ResolvedObservable>, decision: Decision) -> Self {
        ResolvedReadout {
            observables,
            decision,
        }
    }

    /// The decision rule.
    pub(crate) fn decision(&self) -> Decision {
        self.decision
    }

    /// The resolved observables (guaranteed non-empty by construction).
    pub(crate) fn observables(&self) -> &[ResolvedObservable] {
        &self.observables
    }

    /// Turn measurement `counts` into a prediction according to the decision
    /// (design doc §7.1, numeric contract in the phase-3 execution plan).
    ///
    /// `observables[0]` is always present: a [`Readout`] cannot be built with
    /// zero observables (see [`Readout::new`]).
    pub(crate) fn predict(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        match self.decision {
            Decision::Sign => {
                let e = self.observables[0].expectation(counts)?;
                Ok(if e >= 0.0 { 1.0 } else { -1.0 })
            }
            Decision::Threshold(t) => {
                let e = self.observables[0].expectation(counts)?;
                Ok(if e >= t { 1.0 } else { -1.0 })
            }
            Decision::Raw => self.observables[0].expectation(counts),
            Decision::Argmax => {
                let mut best_index = 0usize;
                let mut best_value = self.observables[0].expectation(counts)?;
                for (index, observable) in self.observables.iter().enumerate().skip(1) {
                    let value = observable.expectation(counts)?;
                    // Strict `>`: ties keep the first (lowest) index.
                    if value > best_value {
                        best_value = value;
                        best_index = index;
                    }
                }
                Ok(best_index as f64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observables::{Pauli, PauliString, ResolvedPauliString};

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn z_observable(position: usize) -> Observable {
        Observable::new(vec![(
            1.0,
            PauliString::new(vec![(position, Pauli::Z)]).unwrap(),
        )])
        .unwrap()
    }

    fn resolved_z(position: usize) -> ResolvedObservable {
        ResolvedObservable::new(vec![(
            1.0,
            ResolvedPauliString::new(vec![(position, Pauli::Z)]),
        )])
    }

    #[test]
    fn argmax_needs_two_observables() {
        let err = Readout::new(vec![z_observable(0)], Decision::Argmax).unwrap_err();
        assert_eq!(
            err,
            ValidationError::DecisionObservableMismatch {
                decision: Decision::Argmax,
                num_observables: 1,
            }
        );
        // Two observables is accepted.
        assert!(Readout::new(vec![z_observable(0), z_observable(1)], Decision::Argmax).is_ok());
    }

    #[test]
    fn binary_decisions_need_one_observable() {
        for decision in [Decision::Sign, Decision::Threshold(0.5), Decision::Raw] {
            let err = Readout::new(vec![], decision).unwrap_err();
            assert_eq!(
                err,
                ValidationError::DecisionObservableMismatch {
                    decision,
                    num_observables: 0,
                }
            );
            assert!(Readout::new(vec![z_observable(0)], decision).is_ok());
        }
    }

    #[test]
    fn sign_breaks_ties_positive() {
        let readout = ResolvedReadout::new(vec![resolved_z(0)], Decision::Sign);
        // ⟨Z_0⟩ = 0 (even split) → tie → +1.
        assert_eq!(readout.predict(&counts(&[("0", 5), ("1", 5)])), Ok(1.0));
        assert_eq!(readout.predict(&counts(&[("1", 5)])), Ok(-1.0));
    }

    #[test]
    fn threshold_compares_against_t() {
        let readout = ResolvedReadout::new(vec![resolved_z(0)], Decision::Threshold(0.5));
        // ⟨Z_0⟩ = 1.0 ≥ 0.5 → +1.
        assert_eq!(readout.predict(&counts(&[("0", 10)])), Ok(1.0));
        // ⟨Z_0⟩ = 0 < 0.5 → −1.
        assert_eq!(readout.predict(&counts(&[("0", 5), ("1", 5)])), Ok(-1.0));
    }

    #[test]
    fn raw_returns_expectation_unchanged() {
        let readout = ResolvedReadout::new(vec![resolved_z(0)], Decision::Raw);
        let p = readout.predict(&counts(&[("0", 75), ("1", 25)])).unwrap();
        assert!((p - 0.5).abs() < 1e-12);
    }

    #[test]
    fn argmax_returns_winning_index_and_breaks_ties_low() {
        // Two observables, width-2 counts. Z_0 over "01" (=+1 for qubit0? "01":
        // qubit0 is right char '1' → −1), Z_1 over "01" (qubit1 left '0' → +1).
        let readout = ResolvedReadout::new(vec![resolved_z(0), resolved_z(1)], Decision::Argmax);
        // "01": ⟨Z_0⟩ = −1, ⟨Z_1⟩ = +1 → argmax is index 1.
        assert_eq!(readout.predict(&counts(&[("01", 10)])), Ok(1.0));
        // Tie (both +1 over "00") → lowest index 0.
        assert_eq!(readout.predict(&counts(&[("00", 10)])), Ok(0.0));
    }
}
