//! [`QmlProblem`]: the crate's final product — a compiled model, a training
//! set and a loss, precompiled into the pair of operations a
//! `polypus-optimizers` oracle needs (design doc §9).
//!
//! `bind_batch` turns a parameter vector into one concrete circuit per sample;
//! `fitness_from_counts` turns the resulting measurement counts into a scalar
//! fitness (`−mean_loss`, since the optimizers maximise). Everything between —
//! backend, shots, distribution — is `crates/polypus`' concern, which keeps
//! this crate free of any execution dependency (design doc §1, D11). These two
//! operations plus `num_params`/`num_circuits` are the QML side of contract
//! C-8.

use std::collections::HashMap;

use polypus_circuit::{ConcreteCircuit, ParameterizedCircuit};

use crate::dataset::Dataset;
use crate::error::{QmlError, ValidationError};
use crate::loss::Loss;
use crate::model::CompiledModel;
use crate::readout::Decision;

/// A trainable QML problem: a compiled model, a training set and a loss, with
/// one circuit template precompiled per training sample (features fixed, `θ`
/// still free). Construction is the expensive step; binding parameters is cheap
/// and repeated once per optimizer candidate.
#[derive(Debug, Clone)]
pub struct QmlProblem {
    model: CompiledModel,
    train: Dataset,
    loss: Loss,
    /// One template per training sample, in sample order.
    templates: Vec<ParameterizedCircuit>,
}

impl QmlProblem {
    /// Build a problem, running the cross-checks that no single piece could do
    /// alone (design doc §9) and precompiling one template per sample:
    ///
    /// 1. `train.num_features() == model.num_features()`
    ///    ([`ValidationError::FeatureCountMismatch`]).
    /// 2. the readout decision is trainable by *some* v1 loss — i.e. not
    ///    [`Decision::Argmax`], which no v1 loss supports
    ///    ([`ValidationError::DecisionNotSupportedByLoss`], design doc §8).
    /// 3. every label lies in the loss's domain
    ///    ([`Loss::validate_label`] → [`ValidationError::LabelDomain`]).
    ///
    /// The per-sample [`template_for`](CompiledModel::template_for) can only
    /// fail on a feature-count mismatch, already ruled out by check 1, so the
    /// `?` here never actually fires — but it is typed via
    /// [`ValidationError::Template`] rather than assumed away with an `expect`.
    pub fn new(model: CompiledModel, train: Dataset, loss: Loss) -> Result<Self, ValidationError> {
        if train.num_features() != model.num_features() {
            return Err(ValidationError::FeatureCountMismatch {
                expected: model.num_features(),
                got: train.num_features(),
            });
        }

        let decision = model.resolved_readout().decision();
        if decision == Decision::Argmax {
            return Err(ValidationError::DecisionNotSupportedByLoss { decision });
        }

        for (sample, &label) in train.labels().iter().enumerate() {
            loss.validate_label(label, sample)?;
        }

        let mut templates = Vec::with_capacity(train.num_samples());
        for i in 0..train.num_samples() {
            // `?` converts QmlError → ValidationError via `From` (see error.rs).
            templates.push(model.template_for(train.sample(i))?);
        }

        Ok(QmlProblem {
            model,
            train,
            loss,
            templates,
        })
    }

    /// The number of trainable parameters — the `dimensions` an optimizer sees.
    pub fn num_params(&self) -> usize {
        self.model.num_params()
    }

    /// The number of circuits produced per candidate evaluation. In v1 this is
    /// the number of training samples (one basis group per sample).
    pub fn num_circuits(&self) -> usize {
        self.templates.len()
    }

    /// Bind `theta` into one [`ConcreteCircuit`] per training sample, in stable
    /// sample-major order (contract C-8). A wrong number of parameters surfaces
    /// as [`QmlError::Circuit`]`(`[`WrongNumberOfParams`]`)`.
    ///
    /// [`WrongNumberOfParams`]: polypus_circuit::CircuitError::WrongNumberOfParams
    pub fn bind_batch(&self, theta: &[f64]) -> Result<Vec<ConcreteCircuit>, QmlError> {
        let mut circuits = Vec::with_capacity(self.templates.len());
        for template in &self.templates {
            circuits.push(template.assign_parameters(theta)?);
        }
        Ok(circuits)
    }

    /// The per-sample raw expectation `⟨O₀⟩` (the readout's first observable)
    /// for each sample's measurement `counts`, in the same (stable, sample-major)
    /// order as [`bind_batch`](Self::bind_batch).
    ///
    /// This is the shared building block of both [`fitness_from_counts`] (which
    /// composes a [`Loss`] over these) and [`param_gradient`] (which shifts
    /// them): every loss, and the exact parameter-shift gradient, operates on
    /// these raw expectations, never on the [`Decision`] output (design doc §8).
    /// The `counts` length is checked exactly as `fitness_from_counts` does,
    /// yielding [`QmlError::CountsLengthMismatch`] on a mismatch.
    ///
    /// [`fitness_from_counts`]: Self::fitness_from_counts
    /// [`param_gradient`]: Self::param_gradient
    pub fn expectations_from_counts(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<f64>, QmlError> {
        if counts.len() != self.templates.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.templates.len(),
                got: counts.len(),
            });
        }
        let observable = &self.model.resolved_readout().observables()[0];
        counts
            .iter()
            .map(|sample_counts| observable.expectation(sample_counts))
            .collect()
    }

    /// Turn per-sample measurement `counts` (in the same order as
    /// [`bind_batch`](Self::bind_batch)) into a fitness `= −mean_loss`.
    ///
    /// A thin layer over [`expectations_from_counts`](Self::expectations_from_counts):
    /// it evaluates the [`Loss`] against each raw `⟨O₀⟩` and averages. The loss
    /// always operates on the raw expectation, never on the [`Decision`] output
    /// (design doc §8). Returns a finite `f64` for valid counts, or a typed
    /// [`QmlError`] — never `NaN` (contract C-8).
    pub fn fitness_from_counts(&self, counts: &[HashMap<String, u64>]) -> Result<f64, QmlError> {
        let expectations = self.expectations_from_counts(counts)?;
        let labels = self.train.labels();
        let total: f64 = expectations
            .iter()
            .zip(labels)
            .map(|(&expectation, &label)| self.loss.evaluate(expectation, label))
            .sum();
        // `counts.len() == templates.len() >= 1` (the dataset is non-empty), so
        // the mean is well defined and finite.
        Ok(-total / expectations.len() as f64)
    }

    /// The exact parameter-shift gradient of the fitness with respect to **one**
    /// trainable parameter (design doc §17).
    ///
    /// "Exact" here means an exact mathematical identity in the noiseless limit;
    /// every argument is built from finite-shot `counts`, so the returned value
    /// is an unbiased *estimator* of the true gradient component, not a
    /// noise-free number.
    ///
    /// The fitness composes a nonlinear [`Loss`] over the raw per-sample
    /// expectations, so the shift rule cannot be applied to the aggregate
    /// fitness directly — it needs the chain rule. Given the base expectations
    /// `⟨O_i⟩(θ)`, and the counts at `θ ± π/2·e_k` (the parameter shifted by
    /// `±π/2`), this returns
    ///
    /// `−(1/n) · Σᵢ Loss'(⟨O_i⟩, yᵢ) · (⟨O_i⟩(θ+) − ⟨O_i⟩(θ−)) / 2`
    ///
    /// where `(⟨O_i⟩(θ+) − ⟨O_i⟩(θ−))/2` is the parameter-shift derivative of
    /// the raw expectation (exact for the `±1`-eigenvalue generators this crate
    /// uses) and `Loss'` is [`Loss::gradient`]. The leading `−(1/n)` mirrors
    /// `fitness_from_counts`' `−mean_loss` so the sign convention agrees: higher
    /// fitness is better, and this is `∂fitness/∂θ_k` (the ascent direction).
    ///
    /// Length validation, in this order, each against
    /// [`num_circuits`](Self::num_circuits): `plus_counts`, then `minus_counts`,
    /// then `base_expectations` — the first mismatch returns
    /// [`QmlError::CountsLengthMismatch`], deterministically, even if several
    /// disagree at once.
    pub fn param_gradient(
        &self,
        base_expectations: &[f64],
        plus_counts: &[HashMap<String, u64>],
        minus_counts: &[HashMap<String, u64>],
    ) -> Result<f64, QmlError> {
        let n = self.num_circuits();
        if plus_counts.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: plus_counts.len(),
            });
        }
        if minus_counts.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: minus_counts.len(),
            });
        }
        if base_expectations.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: base_expectations.len(),
            });
        }

        // Both helper calls re-check their own length against `num_circuits()`,
        // so the checks above are redundant for `plus`/`minus` — but they keep
        // the documented, deterministic error ordering and guard `base` (which
        // never reaches the helper) before the parallel iteration below.
        let plus_expectations = self.expectations_from_counts(plus_counts)?;
        let minus_expectations = self.expectations_from_counts(minus_counts)?;
        let labels = self.train.labels();

        let total: f64 = (0..n)
            .map(|i| {
                self.loss.gradient(base_expectations[i], labels[i])
                    * (plus_expectations[i] - minus_expectations[i])
                    / 2.0
            })
            .sum();
        Ok(-total / n as f64)
    }

    /// Infer a prediction from one sample's `counts`, applying the readout's
    /// [`Decision`] (design doc §7.1).
    pub fn predict_from_counts(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        self.model.resolved_readout().predict(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QuantumModel;
    use crate::observables::{Observable, Pauli, PauliString};
    use crate::readout::{Decision, Readout};
    use crate::RotationAxis;

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn z0_readout(decision: Decision) -> Readout {
        Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            decision,
        )
        .unwrap()
    }

    fn two_qubit_model(decision: Decision) -> CompiledModel {
        QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(z0_readout(decision))
            .compile(2)
            .unwrap()
    }

    fn dataset(labels: &[f64]) -> Dataset {
        let rows: Vec<Vec<f64>> = labels.iter().map(|_| vec![0.1, 0.2]).collect();
        Dataset::from_rows(&rows, labels).unwrap()
    }

    #[test]
    fn new_rejects_feature_count_mismatch() {
        let model = two_qubit_model(Decision::Sign);
        // Dataset with 3 features, model expects 2.
        let ds = Dataset::from_rows(&[vec![0.1, 0.2, 0.3]], &[1.0]).unwrap();
        let err = QmlProblem::new(model, ds, Loss::SquaredError).unwrap_err();
        assert_eq!(
            err,
            ValidationError::FeatureCountMismatch {
                expected: 2,
                got: 3,
            }
        );
    }

    #[test]
    fn new_rejects_argmax_decision() {
        // Argmax needs >=2 observables; build one and confirm QmlProblem rejects
        // it as untrainable by any v1 loss.
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
                Observable::new(vec![(1.0, PauliString::new(vec![(1, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Argmax,
        )
        .unwrap();
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(2)
            .unwrap();
        let err = QmlProblem::new(model, dataset(&[1.0, -1.0]), Loss::Hinge).unwrap_err();
        assert_eq!(
            err,
            ValidationError::DecisionNotSupportedByLoss {
                decision: Decision::Argmax,
            }
        );
    }

    #[test]
    fn new_validates_label_domain() {
        let model = two_qubit_model(Decision::Sign);
        // Hinge needs {-1, 1}; a 0.0 label is out of domain.
        let err = QmlProblem::new(model, dataset(&[1.0, 0.0]), Loss::Hinge).unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelDomain {
                loss: Loss::Hinge,
                expected: "{-1.0, 1.0}",
                found_sample: 1,
            }
        );
    }

    #[test]
    fn num_params_and_circuits_track_model_and_dataset() {
        let model = two_qubit_model(Decision::Sign);
        let num_params = model.num_params();
        let ds = dataset(&[1.0, -1.0, 1.0]);
        let problem = QmlProblem::new(model, ds, Loss::Hinge).unwrap();
        assert_eq!(problem.num_params(), num_params);
        assert_eq!(problem.num_circuits(), 3);
    }

    #[test]
    fn bind_batch_produces_one_circuit_per_sample() {
        let model = two_qubit_model(Decision::Sign);
        let problem = QmlProblem::new(model, dataset(&[1.0, -1.0]), Loss::Hinge).unwrap();
        let theta: Vec<f64> = (0..problem.num_params()).map(|k| 0.1 * k as f64).collect();
        let circuits = problem.bind_batch(&theta).unwrap();
        assert_eq!(circuits.len(), problem.num_circuits());
    }

    #[test]
    fn fitness_from_counts_is_finite_and_length_checked() {
        let model = two_qubit_model(Decision::Sign);
        let problem = QmlProblem::new(model, dataset(&[1.0, -1.0]), Loss::Hinge).unwrap();
        // Two samples → two counts maps of width 2.
        let c = vec![counts(&[("00", 1024)]), counts(&[("01", 1024)])];
        let fitness = problem.fitness_from_counts(&c).unwrap();
        assert!(fitness.is_finite());

        // Wrong number of counts maps is a typed error.
        let err = problem.fitness_from_counts(&c[..1]).unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
    }

    /// A 1-qubit `⟨Z₀⟩` model reading `Loss::SquaredError`, used by the
    /// expectation/gradient tests where the bitstring width is 1 (`"0"`/`"1"`)
    /// so synthetic counts map to exact expectation values.
    fn one_qubit_z0_problem(labels: &[f64]) -> QmlProblem {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let model = QuantumModel::new(1)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(1)
            .unwrap();
        let rows: Vec<Vec<f64>> = labels.iter().map(|_| vec![0.3]).collect();
        let ds = Dataset::from_rows(&rows, labels).unwrap();
        QmlProblem::new(model, ds, Loss::SquaredError).unwrap()
    }

    #[test]
    fn expectations_from_counts_reads_raw_z0_and_checks_length() {
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        // "0":3,"1":1 → (3−1)/4 = +0.5 ; "0":1,"1":3 → −0.5.
        let c = vec![counts(&[("0", 3), ("1", 1)]), counts(&[("0", 1), ("1", 3)])];
        let exps = problem.expectations_from_counts(&c).unwrap();
        assert!((exps[0] - 0.5).abs() < 1e-12);
        assert!((exps[1] + 0.5).abs() < 1e-12);

        let err = problem.expectations_from_counts(&c[..1]).unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn fitness_from_counts_is_neg_mean_loss_over_expectations() {
        // The refactor makes `fitness_from_counts` a thin layer over
        // `expectations_from_counts`; confirm the observable behaviour is
        // unchanged by checking it equals `−mean(SquaredError(⟨O_i⟩, yᵢ))`
        // recomputed independently from the same raw expectations.
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        let c = vec![counts(&[("0", 3), ("1", 1)]), counts(&[("0", 1), ("1", 3)])];
        let exps = problem.expectations_from_counts(&c).unwrap();
        // SquaredError: (0.5−1)² = 0.25 ; (−0.5−(−1))² = 0.25 → mean 0.25.
        let expected = -(0.25 + 0.25) / 2.0;
        let fitness = problem.fitness_from_counts(&c).unwrap();
        assert!((fitness - expected).abs() < 1e-12, "fitness={fitness}");
        // And it must agree with a fully independent recomputation from `exps`.
        let manual: f64 = exps
            .iter()
            .zip([1.0, -1.0])
            .map(|(&e, y)| (e - y).powi(2))
            .sum::<f64>()
            / exps.len() as f64;
        assert!((fitness + manual).abs() < 1e-12);
    }

    #[test]
    fn param_gradient_matches_hand_computation() {
        // Two samples, ⟨Z₀⟩, SquaredError. Feed synthetic counts whose
        // (a−b)/(a+b) reproduce chosen exact expectations, then check
        // param_gradient equals the by-hand chain-rule value.
        //
        //   sample 0 (y=+1): base=0.5, plus=+1.0, minus=0.0
        //   sample 1 (y=−1): base=−0.5, plus=0.0, minus=+1.0
        //
        //   Loss'(pred,y)=2(pred−y); shiftᵢ=(plus−minus)/2
        //   s0: 2(0.5−1)=−1 ; shift=(1−0)/2=0.5     → −0.5
        //   s1: 2(−0.5+1)=+1 ; shift=(0−1)/2=−0.5    → −0.5
        //   grad = −(1/2)·(−0.5 + −0.5) = +0.5
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        let base = vec![0.5, -0.5];
        let plus = vec![counts(&[("0", 1)]), counts(&[("0", 1), ("1", 1)])];
        let minus = vec![counts(&[("0", 1), ("1", 1)]), counts(&[("0", 1)])];
        let grad = problem.param_gradient(&base, &plus, &minus).unwrap();
        assert!((grad - 0.5).abs() < 1e-12, "grad={grad}");
    }

    #[test]
    fn param_gradient_checks_lengths_in_documented_order() {
        let problem = one_qubit_z0_problem(&[1.0, -1.0]); // num_circuits() == 2
        let ok_pair = vec![counts(&[("0", 1)]), counts(&[("1", 1)])];
        let base_ok = vec![0.0, 0.0];

        // plus wrong → reported first (expected 2, got 1).
        let err = problem
            .param_gradient(&base_ok, &ok_pair[..1], &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
        // plus ok, minus wrong → reported next.
        let err = problem
            .param_gradient(&base_ok, &ok_pair, &ok_pair[..1])
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
        // plus/minus ok, base wrong → reported last.
        let err = problem
            .param_gradient(&base_ok[..1], &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn predict_from_counts_applies_decision() {
        let model = two_qubit_model(Decision::Sign);
        let problem = QmlProblem::new(model, dataset(&[1.0, -1.0]), Loss::Hinge).unwrap();
        // ⟨Z₀⟩ over "00" (width 2) = +1 → Sign → +1.
        assert_eq!(problem.predict_from_counts(&counts(&[("00", 10)])), Ok(1.0));
        // ⟨Z₀⟩ over "01" = −1 → Sign → −1.
        assert_eq!(
            problem.predict_from_counts(&counts(&[("01", 10)])),
            Ok(-1.0)
        );
    }
}
