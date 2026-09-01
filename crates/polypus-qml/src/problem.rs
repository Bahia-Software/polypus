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
use crate::loss::{categorical_cross_entropy, categorical_cross_entropy_gradient, Loss};
use crate::model::CompiledModel;
use crate::readout::Decision;
use crate::rng::{shuffle, SplitMix64};

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

        // Decision ↔ loss is a bidirectional pairing (design doc §17): the
        // multiclass `Argmax` decision pairs with (and only with) the multiclass
        // `CategoricalCrossEntropy` loss; every scalar loss pairs with (and only
        // with) a non-`Argmax` decision. `!=` on the two booleans catches both
        // mismatches (Argmax under a scalar loss, or categorical under a scalar
        // decision) in one check.
        let decision = model.resolved_readout().decision();
        let is_categorical = loss == Loss::CategoricalCrossEntropy;
        if (decision == Decision::Argmax) != is_categorical {
            return Err(ValidationError::DecisionNotSupportedByLoss { decision, loss });
        }

        for (sample, &label) in train.labels().iter().enumerate() {
            loss.validate_label(label, sample)?;
        }

        // For the categorical loss, `validate_label` has already confirmed every
        // label is a non-negative integer; the upper bound needs `num_classes`
        // (the observable count), which only this cross-check knows. Reject any
        // label naming a class `≥ num_classes`.
        if is_categorical {
            let num_classes = model.resolved_readout().observables().len();
            for (sample, &label) in train.labels().iter().enumerate() {
                if label as usize >= num_classes {
                    return Err(ValidationError::LabelClassOutOfRange {
                        sample,
                        label,
                        num_classes,
                    });
                }
            }
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

    /// The problem's loss. Exposed so a caller (e.g.
    /// [`NativeQmlOracle`](../../polypus/evaluation/index.html)) can decide
    /// whether to take the scalar or the categorical fitness/gradient path
    /// without re-deriving it. [`Loss`] is `Copy`, so this is a cheap read.
    pub fn loss(&self) -> Loss {
        self.loss
    }

    /// Deterministic minibatch selection (design doc §17): a pseudo-random subset
    /// of `batch_size` sample indices out of `num_circuits()`, one call = one
    /// minibatch. `call_index` must be a value that increments by exactly one per
    /// oracle call (the caller's job — see `NativeQmlOracle`); combined with
    /// `seed` it derives an independent, deterministic shuffle per call via
    /// `SplitMix64`. Assumes `0 < batch_size <= num_circuits()` — that precondition
    /// is validated once at construction time by the Python-facing boundary, not
    /// re-checked here on every call.
    pub fn minibatch_indices(&self, seed: u64, call_index: u64, batch_size: usize) -> Vec<usize> {
        // Mixing constant: SplitMix64's own golden-ratio increment
        // (0x9E3779B97F4A7C15), reused here to decorrelate `call_index` values
        // without an O(call_index) fast-forward.
        let mixed = seed ^ call_index.wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = SplitMix64::new(mixed);
        let mut indices: Vec<usize> = (0..self.num_circuits()).collect();
        shuffle(&mut indices, &mut rng);
        indices.truncate(batch_size);
        indices
    }

    /// Build a smaller `QmlProblem` from the samples at `indices` (design doc
    /// §17): the minibatch counterpart of using the full problem. Reuses the
    /// *already-compiled* templates at those indices — no recompilation, and no
    /// re-validation of the **per-sample** checks (the full problem was already
    /// validated by `new`, and a subset of an already-valid dataset can never
    /// violate a per-sample check that isn't already excluded). Cheaper than the
    /// full-problem clone `try_evaluate`/`try_gradient` already do today for the
    /// non-minibatch case.
    ///
    /// Emptiness is the one thing a subset *can* violate, precisely because it is
    /// not a per-sample property: an empty `indices` would build a problem with
    /// zero templates, whose `fitness_from_counts` would divide by zero and hand
    /// back `Ok(NaN)` against contract C-8. So this is the second entry point at
    /// which `Dataset`'s non-empty invariant is reasserted (`Dataset::from_rows`
    /// is the first), and it propagates that
    /// [`ValidationError::EmptyDataset`] instead of trusting its callers: this
    /// method is `pub`, and the fact that today's in-crate callers happen to feed
    /// it an already-validated `batch_size` is not something its contract may
    /// rest on.
    // `from_*` on `&self` trips `wrong_self_convention`, but this is genuinely a
    // "derive a smaller problem *from* this one's samples" operation, not a
    // constructor — the same API-naming exception the crate already takes for
    // `Infrastructure::from_str`. It weakens no correctness check.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_subset(&self, indices: &[usize]) -> Result<QmlProblem, ValidationError> {
        Ok(QmlProblem {
            model: self.model.clone(),
            train: self.train.select(indices)?,
            loss: self.loss,
            templates: indices.iter().map(|&i| self.templates[i].clone()).collect(),
        })
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

    /// The per-sample expectation vector `[⟨O₀⟩, …, ⟨O_{k−1}⟩]` over **all** the
    /// readout's observables, for each sample's measurement `counts`, in the same
    /// (stable, sample-major) order as [`bind_batch`](Self::bind_batch).
    ///
    /// The multiclass counterpart of
    /// [`expectations_from_counts`](Self::expectations_from_counts), which reads
    /// only `⟨O₀⟩`: `CategoricalCrossEntropy` scores the whole vector, one
    /// component per class. Applies the same `counts`-length check
    /// ([`QmlError::CountsLengthMismatch`]) and yields a `Vec<f64>` of length
    /// `k = observables().len()` per sample.
    pub fn expectations_per_class_from_counts(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Vec<Vec<f64>>, QmlError> {
        if counts.len() != self.templates.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.templates.len(),
                got: counts.len(),
            });
        }
        let observables = self.model.resolved_readout().observables();
        counts
            .iter()
            .map(|sample_counts| {
                observables
                    .iter()
                    .map(|observable| observable.expectation(sample_counts))
                    .collect::<Result<Vec<f64>, QmlError>>()
            })
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
        // The categorical loss scores the whole per-class expectation vector, so
        // it takes a wholly separate path (design doc §17): all-class
        // expectations + `categorical_cross_entropy` per sample, then the same
        // `−mean` aggregation. Branching here keeps every caller (e.g.
        // `NativeQmlOracle::evaluate_batch`) agnostic — they call
        // `fitness_from_counts` identically regardless of the loss.
        if self.loss == Loss::CategoricalCrossEntropy {
            let per_class = self.expectations_per_class_from_counts(counts)?;
            let labels = self.train.labels();
            let total: f64 = per_class
                .iter()
                .zip(labels)
                .map(|(expectations, &label)| {
                    categorical_cross_entropy(expectations, label as usize)
                })
                .sum();
            return Ok(-total / per_class.len() as f64);
        }

        let expectations = self.expectations_from_counts(counts)?;
        let labels = self.train.labels();
        let total: f64 = expectations
            .iter()
            .zip(labels)
            .map(|(&expectation, &label)| self.loss.evaluate(expectation, label))
            .sum::<Result<f64, QmlError>>()?;
        // `counts.len() == templates.len() >= 1`: a `Dataset` is never empty —
        // `Dataset::from_rows` and `Dataset::select` are the only two ways to
        // build one and both reject an empty sample set — so the mean is well
        // defined and finite, never `0.0 / 0.0` (contract C-8).
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
            .map(|i| -> Result<f64, QmlError> {
                Ok(self.loss.gradient(base_expectations[i], labels[i])?
                    * (plus_expectations[i] - minus_expectations[i])
                    / 2.0)
            })
            .sum::<Result<f64, QmlError>>()?;
        Ok(-total / n as f64)
    }

    /// The exact parameter-shift gradient of the **categorical** fitness with
    /// respect to one trainable parameter (design doc §17): the multiclass
    /// counterpart of [`param_gradient`](Self::param_gradient).
    ///
    /// The fitness composes `CategoricalCrossEntropy` over the per-sample
    /// *vector* of class expectations, so the chain rule runs over every class:
    ///
    /// `−(1/n) · Σᵢ Σⱼ CE'(z_i, yᵢ)[j] · (⟨O_j⟩(θ+) − ⟨O_j⟩(θ−)) / 2`
    ///
    /// where `z_i = base_expectations[i]` is sample `i`'s full class-expectation
    /// vector, `CE'` is [`categorical_cross_entropy_gradient`] (a per-class
    /// vector), and each `(⟨O_j⟩(θ+) − ⟨O_j⟩(θ−))/2` is the parameter-shift
    /// derivative of class `j`'s raw expectation. The leading `−(1/n)` matches
    /// [`fitness_from_counts`](Self::fitness_from_counts)'s `−mean_loss`, so the
    /// sign convention agrees (this is `∂fitness/∂θ_k`, the ascent direction).
    ///
    /// Length validation, in this order, each against
    /// [`num_circuits`](Self::num_circuits): `plus_counts`, then `minus_counts`,
    /// then `base_expectations` — the same deterministic ordering as
    /// [`param_gradient`](Self::param_gradient). Then, once those *outer*
    /// lengths agree, the *inner* width of every `base_expectations[i]` is
    /// checked against the readout's observable count, reporting the first
    /// offending sample as
    /// [`ClassCountMismatch`](crate::QmlError::ClassCountMismatch): the chain
    /// rule below zips `CE'` against the per-class shifts, so a vector of the
    /// wrong width would silently drop (or ignore) classes and return a
    /// plausible but wrong gradient.
    pub fn param_gradient_categorical(
        &self,
        base_expectations: &[Vec<f64>],
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
        let num_classes = self.model.resolved_readout().observables().len();
        for (i, base) in base_expectations.iter().enumerate() {
            if base.len() != num_classes {
                return Err(QmlError::ClassCountMismatch {
                    sample: i,
                    expected: num_classes,
                    got: base.len(),
                });
            }
        }

        let plus_per_class = self.expectations_per_class_from_counts(plus_counts)?;
        let minus_per_class = self.expectations_per_class_from_counts(minus_counts)?;
        let labels = self.train.labels();

        let total: f64 = (0..n)
            .map(|i| {
                let g =
                    categorical_cross_entropy_gradient(&base_expectations[i], labels[i] as usize);
                // Chain rule summed over every class j: CE'[j] · shift_j.
                g.iter()
                    .zip(plus_per_class[i].iter())
                    .zip(minus_per_class[i].iter())
                    .map(|((&g_j, &plus_j), &minus_j)| g_j * (plus_j - minus_j) / 2.0)
                    .sum::<f64>()
            })
            .sum();
        Ok(-total / n as f64)
    }

    /// Infer a prediction from one sample's `counts`, applying the readout's
    /// [`Decision`] (design doc §7.1).
    pub fn predict_from_counts(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        self.model.resolved_readout().predict(counts)
    }

    // ── Exact mode (design doc §17) ──────────────────────────────────────────
    //
    // Exact mirrors of the counts-based methods above, taking per-sample exact
    // basis-state `probabilities` (`|amplitude|²`) instead of finite-shot
    // counts. Each has identical structure, length validation and error
    // ordering to its counterpart — only the expectation estimator differs
    // (`expectation_from_probabilities` vs. `expectation_from_counts`). The
    // native exact `qml.train` path is their sole caller.

    /// Exact-mode mirror of
    /// [`expectations_from_counts`](Self::expectations_from_counts): the
    /// per-sample raw expectation `⟨O₀⟩` from exact `probabilities`.
    pub fn expectations_from_probabilities(
        &self,
        probabilities: &[HashMap<String, f64>],
    ) -> Result<Vec<f64>, QmlError> {
        if probabilities.len() != self.templates.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.templates.len(),
                got: probabilities.len(),
            });
        }
        let observable = &self.model.resolved_readout().observables()[0];
        probabilities
            .iter()
            .map(|sample_probs| observable.expectation_from_probabilities(sample_probs))
            .collect()
    }

    /// Exact-mode mirror of
    /// [`expectations_per_class_from_counts`](Self::expectations_per_class_from_counts):
    /// the per-sample per-class expectation vector from exact `probabilities`.
    pub fn expectations_per_class_from_probabilities(
        &self,
        probabilities: &[HashMap<String, f64>],
    ) -> Result<Vec<Vec<f64>>, QmlError> {
        if probabilities.len() != self.templates.len() {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.templates.len(),
                got: probabilities.len(),
            });
        }
        let observables = self.model.resolved_readout().observables();
        probabilities
            .iter()
            .map(|sample_probs| {
                observables
                    .iter()
                    .map(|observable| observable.expectation_from_probabilities(sample_probs))
                    .collect::<Result<Vec<f64>, QmlError>>()
            })
            .collect()
    }

    /// Exact-mode mirror of [`fitness_from_counts`](Self::fitness_from_counts):
    /// the fitness `= −mean_loss` from exact `probabilities`, branching on the
    /// loss exactly as the counts version does so the caller stays agnostic.
    pub fn fitness_from_probabilities(
        &self,
        probabilities: &[HashMap<String, f64>],
    ) -> Result<f64, QmlError> {
        if self.loss == Loss::CategoricalCrossEntropy {
            let per_class = self.expectations_per_class_from_probabilities(probabilities)?;
            let labels = self.train.labels();
            let total: f64 = per_class
                .iter()
                .zip(labels)
                .map(|(expectations, &label)| {
                    categorical_cross_entropy(expectations, label as usize)
                })
                .sum();
            return Ok(-total / per_class.len() as f64);
        }

        let expectations = self.expectations_from_probabilities(probabilities)?;
        let labels = self.train.labels();
        let total: f64 = expectations
            .iter()
            .zip(labels)
            .map(|(&expectation, &label)| self.loss.evaluate(expectation, label))
            .sum::<Result<f64, QmlError>>()?;
        Ok(-total / expectations.len() as f64)
    }

    /// Exact-mode mirror of [`param_gradient`](Self::param_gradient): the exact
    /// parameter-shift gradient component from exact `probabilities`. Same
    /// length-validation order (plus, minus, base) as its counterpart.
    ///
    /// Here "exact" is doubly so: the parameter-shift identity holds exactly in
    /// the noiseless limit, and the probabilities carry no shot noise, so the
    /// returned value is the true gradient component rather than an estimator.
    pub fn param_gradient_exact(
        &self,
        base_expectations: &[f64],
        plus_probs: &[HashMap<String, f64>],
        minus_probs: &[HashMap<String, f64>],
    ) -> Result<f64, QmlError> {
        let n = self.num_circuits();
        if plus_probs.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: plus_probs.len(),
            });
        }
        if minus_probs.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: minus_probs.len(),
            });
        }
        if base_expectations.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: base_expectations.len(),
            });
        }

        let plus_expectations = self.expectations_from_probabilities(plus_probs)?;
        let minus_expectations = self.expectations_from_probabilities(minus_probs)?;
        let labels = self.train.labels();

        let total: f64 = (0..n)
            .map(|i| -> Result<f64, QmlError> {
                Ok(self.loss.gradient(base_expectations[i], labels[i])?
                    * (plus_expectations[i] - minus_expectations[i])
                    / 2.0)
            })
            .sum::<Result<f64, QmlError>>()?;
        Ok(-total / n as f64)
    }

    /// Exact-mode mirror of
    /// [`param_gradient_categorical`](Self::param_gradient_categorical): the
    /// exact parameter-shift gradient component of the categorical fitness from
    /// exact `probabilities`. Same length-validation order as its counterpart:
    /// `plus_probs`, then `minus_probs`, then `base_expectations` against
    /// [`num_circuits`](Self::num_circuits), and finally the *inner* width of
    /// every `base_expectations[i]` against the readout's observable count,
    /// reporting the first offending sample as
    /// [`ClassCountMismatch`](crate::QmlError::ClassCountMismatch).
    pub fn param_gradient_categorical_exact(
        &self,
        base_expectations: &[Vec<f64>],
        plus_probs: &[HashMap<String, f64>],
        minus_probs: &[HashMap<String, f64>],
    ) -> Result<f64, QmlError> {
        let n = self.num_circuits();
        if plus_probs.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: plus_probs.len(),
            });
        }
        if minus_probs.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: minus_probs.len(),
            });
        }
        if base_expectations.len() != n {
            return Err(QmlError::CountsLengthMismatch {
                expected: n,
                got: base_expectations.len(),
            });
        }
        let num_classes = self.model.resolved_readout().observables().len();
        for (i, base) in base_expectations.iter().enumerate() {
            if base.len() != num_classes {
                return Err(QmlError::ClassCountMismatch {
                    sample: i,
                    expected: num_classes,
                    got: base.len(),
                });
            }
        }

        let plus_per_class = self.expectations_per_class_from_probabilities(plus_probs)?;
        let minus_per_class = self.expectations_per_class_from_probabilities(minus_probs)?;
        let labels = self.train.labels();

        let total: f64 = (0..n)
            .map(|i| {
                let g =
                    categorical_cross_entropy_gradient(&base_expectations[i], labels[i] as usize);
                g.iter()
                    .zip(plus_per_class[i].iter())
                    .zip(minus_per_class[i].iter())
                    .map(|((&g_j, &plus_j), &minus_j)| g_j * (plus_j - minus_j) / 2.0)
                    .sum::<f64>()
            })
            .sum();
        Ok(-total / n as f64)
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

    /// A 2-class `Argmax` readout (`⟨Z₀⟩`, `⟨Z₁⟩`) compiled over 2 features, the
    /// categorical counterpart of [`two_qubit_model`].
    fn argmax_two_class_model() -> CompiledModel {
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
        QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(2)
            .unwrap()
    }

    #[test]
    fn new_rejects_argmax_decision_with_scalar_loss() {
        // Argmax under a scalar loss (Hinge) is one direction of the bidirectional
        // pairing mismatch: rejected, carrying both the decision and the loss.
        let err = QmlProblem::new(argmax_two_class_model(), dataset(&[1.0, -1.0]), Loss::Hinge)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::DecisionNotSupportedByLoss {
                decision: Decision::Argmax,
                loss: Loss::Hinge,
            }
        );
    }

    #[test]
    fn new_rejects_categorical_loss_with_scalar_decision() {
        // The other direction: CategoricalCrossEntropy under a non-Argmax
        // decision (Sign) is equally rejected.
        let model = two_qubit_model(Decision::Sign);
        let err = QmlProblem::new(model, dataset(&[0.0, 1.0]), Loss::CategoricalCrossEntropy)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::DecisionNotSupportedByLoss {
                decision: Decision::Sign,
                loss: Loss::CategoricalCrossEntropy,
            }
        );
    }

    #[test]
    fn new_accepts_argmax_with_categorical_loss() {
        // The matching pairing (Argmax + CategoricalCrossEntropy) is accepted.
        let problem = QmlProblem::new(
            argmax_two_class_model(),
            dataset(&[0.0, 1.0]),
            Loss::CategoricalCrossEntropy,
        )
        .unwrap();
        assert_eq!(problem.loss(), Loss::CategoricalCrossEntropy);
    }

    #[test]
    fn new_rejects_label_class_out_of_range() {
        // Two observables → 2 classes {0, 1}; label 2 is a valid non-negative
        // integer (passes validate_label) but names a non-existent class.
        let err = QmlProblem::new(
            argmax_two_class_model(),
            dataset(&[0.0, 2.0]),
            Loss::CategoricalCrossEntropy,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ValidationError::LabelClassOutOfRange {
                sample: 1,
                label: 2.0,
                num_classes: 2,
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

    /// A 2-class categorical problem over `argmax_two_class_model`, labels
    /// `[0, 1]`, used by the categorical fitness/gradient hand-computation tests.
    fn categorical_two_class_problem() -> QmlProblem {
        QmlProblem::new(
            argmax_two_class_model(),
            dataset(&[0.0, 1.0]),
            Loss::CategoricalCrossEntropy,
        )
        .unwrap()
    }

    #[test]
    fn expectations_per_class_reads_all_observables_and_checks_length() {
        let problem = categorical_two_class_problem();
        // sample 0: "00":3,"10":1 → ⟨Z₀⟩=(3+1)/4=1.0 ; ⟨Z₁⟩=(3−1)/4=0.5
        // sample 1: "10":1,"11":3 → ⟨Z₀⟩=(1−3)/4=−0.5 ; ⟨Z₁⟩=(−1−3)/4=−1.0
        let c = vec![
            counts(&[("00", 3), ("10", 1)]),
            counts(&[("10", 1), ("11", 3)]),
        ];
        let per_class = problem.expectations_per_class_from_counts(&c).unwrap();
        assert_eq!(per_class.len(), 2);
        assert_eq!(per_class[0].len(), 2);
        assert!((per_class[0][0] - 1.0).abs() < 1e-12);
        assert!((per_class[0][1] - 0.5).abs() < 1e-12);
        assert!((per_class[1][0] + 0.5).abs() < 1e-12);
        assert!((per_class[1][1] + 1.0).abs() < 1e-12);

        let err = problem
            .expectations_per_class_from_counts(&c[..1])
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
    fn fitness_from_counts_categorical_matches_hand_computation() {
        let problem = categorical_two_class_problem();
        // sample 0 (y=0): z=[1.0, 0.5] ; sample 1 (y=1): z=[−0.5, −1.0]
        //   CE0 = −z_0 + ln(e^1 + e^0.5) = −1.0 + 1.474064 = 0.474064
        //   CE1 = −z_1 + ln(e^−0.5 + e^−1) = 1.0 + (−0.025916) = 0.974084
        //   fitness = −(CE0 + CE1)/2 = −0.724074
        let c = vec![
            counts(&[("00", 3), ("10", 1)]),
            counts(&[("10", 1), ("11", 3)]),
        ];
        let fitness = problem.fitness_from_counts(&c).unwrap();
        assert!(fitness.is_finite());
        assert!((fitness + 0.724074).abs() < 1e-3, "fitness={fitness}");
    }

    #[test]
    fn param_gradient_categorical_matches_hand_computation() {
        let problem = categorical_two_class_problem();
        // base per-class expectations: s0 (y=0) z=[1.0,0.5] ; s1 (y=1) z=[−0.5,−1.0]
        let base = vec![vec![1.0, 0.5], vec![-0.5, -1.0]];
        // plus  ("10") → z=[1.0, −1.0] ; minus ("01") → z=[−1.0, 1.0]
        //   shift_j = (plus − minus)/2 = [1.0, −1.0] for both samples
        let plus = vec![counts(&[("10", 1)]), counts(&[("10", 1)])];
        let minus = vec![counts(&[("01", 1)]), counts(&[("01", 1)])];
        // CE'(z0,0) = softmax([1.0,0.5]) − [1,0] = [−0.377541, 0.377541]
        //   Σⱼ CE'·shift = −0.377541·1.0 + 0.377541·(−1.0) = −0.755082
        // CE'(z1,1) = softmax([−0.5,−1.0]) − [0,1] = [0.622459, −0.622459]
        //   Σⱼ CE'·shift = 0.622459·1.0 + (−0.622459)·(−1.0) = 1.244918
        // grad = −(1/2)·(−0.755082 + 1.244918) = −0.244918
        let grad = problem
            .param_gradient_categorical(&base, &plus, &minus)
            .unwrap();
        assert!((grad + 0.244918).abs() < 1e-3, "grad={grad}");
    }

    #[test]
    fn param_gradient_categorical_checks_lengths_in_documented_order() {
        let problem = categorical_two_class_problem(); // num_circuits() == 2
        let ok_pair = vec![counts(&[("00", 1)]), counts(&[("11", 1)])];
        let base_ok = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        // plus wrong → reported first.
        let err = problem
            .param_gradient_categorical(&base_ok, &ok_pair[..1], &ok_pair)
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
            .param_gradient_categorical(&base_ok, &ok_pair, &ok_pair[..1])
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
            .param_gradient_categorical(&base_ok[..1], &ok_pair, &ok_pair)
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
    fn param_gradient_categorical_rejects_wrong_class_count() {
        let problem = categorical_two_class_problem(); // 2 circuits, 2 classes
        let ok_pair = vec![counts(&[("00", 1)]), counts(&[("11", 1)])];

        // Outer length right (2 samples), inner width short on sample 1: the
        // chain-rule `zip` would silently drop class 1 and return a plausible
        // but wrong gradient, so this must be a typed error naming sample 1.
        let base_short = vec![vec![0.0, 0.0], vec![0.0]];
        let err = problem
            .param_gradient_categorical(&base_short, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 1,
                expected: 2,
                got: 1,
            }
        );

        // Symmetric case: a *longer* inner vector must not pass "by accident"
        // just because the `zip` truncates in the other direction.
        let base_long = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0]];
        let err = problem
            .param_gradient_categorical(&base_long, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 0,
                expected: 2,
                got: 3,
            }
        );

        // The inner check runs *after* the outer ones: with plus (or minus, or
        // the outer base length) also wrong, the outer mismatch still wins.
        let err = problem
            .param_gradient_categorical(&base_short, &ok_pair[..1], &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
        let err = problem
            .param_gradient_categorical(&base_short, &ok_pair, &ok_pair[..1])
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );
        let err = problem
            .param_gradient_categorical(&base_short[..1], &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1,
            }
        );

        // First offender wins: both samples are wrong, sample 0 is reported.
        let base_both = vec![vec![0.0], vec![0.0, 0.0, 0.0]];
        let err = problem
            .param_gradient_categorical(&base_both, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 0,
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

    // ── Minibatching (design doc §17) ────────────────────────────────────────

    /// A 1-qubit `⟨Z₀⟩` / `SquaredError` problem whose samples carry **distinct**
    /// feature values, so every precompiled template differs. That is what makes
    /// `from_subset`'s template selection observable: a subset built from the
    /// wrong indices would bind different circuits.
    fn distinct_feature_problem(features: &[f64], labels: &[f64]) -> QmlProblem {
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
        let rows: Vec<Vec<f64>> = features.iter().map(|&f| vec![f]).collect();
        let ds = Dataset::from_rows(&rows, labels).unwrap();
        QmlProblem::new(model, ds, Loss::SquaredError).unwrap()
    }

    #[test]
    fn minibatch_indices_is_deterministic_for_seed_and_call_index() {
        let problem = distinct_feature_problem(&[0.1, 0.2, 0.3, 0.4, 0.5], &[1.0; 5]);
        // Same (seed, call_index) → byte-identical index vector.
        let a = problem.minibatch_indices(42, 3, 3);
        let b = problem.minibatch_indices(42, 3, 3);
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        // Every index is in range and the subset has no duplicates.
        assert!(a.iter().all(|&i| i < problem.num_circuits()));
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), a.len());
    }

    #[test]
    fn minibatch_indices_differs_across_call_index() {
        // Different `call_index` values (same seed) draw different minibatches
        // with overwhelming probability. Over a 20-sample dataset choosing 5,
        // at least one of the first few call indices must differ from call 0 —
        // asserting "not all identical" avoids a spurious failure if one pair
        // happens to coincide.
        let problem = distinct_feature_problem(
            &(0..20).map(|k| 0.1 * k as f64).collect::<Vec<_>>(),
            &[1.0; 20],
        );
        let base = problem.minibatch_indices(7, 0, 5);
        let differs = (1..8).any(|c| problem.minibatch_indices(7, c, 5) != base);
        assert!(differs, "every call_index produced the same minibatch");
    }

    #[test]
    fn from_subset_reproduces_the_full_problem_on_those_samples() {
        // Full problem over 4 distinct-feature samples with distinct labels.
        let features = [0.15, 0.25, 0.35, 0.45];
        let labels = [1.0, -1.0, 1.0, -1.0];
        let full = distinct_feature_problem(&features, &labels);

        // Carve out samples 2 and 0 (order preserved), the shape a minibatch has.
        let indices = [2usize, 0usize];
        let subset = full.from_subset(&indices).unwrap();
        assert_eq!(subset.num_circuits(), indices.len());

        // `bind_batch` on the subset binds exactly the full problem's templates
        // at those indices — proving the correct templates were selected, since
        // distinct features make every template distinct.
        let theta: Vec<f64> = (0..full.num_params())
            .map(|k| 0.1 + 0.07 * k as f64)
            .collect();
        let full_circuits = full.bind_batch(&theta).unwrap();
        let subset_circuits = subset.bind_batch(&theta).unwrap();
        assert_eq!(subset_circuits[0], full_circuits[2]);
        assert_eq!(subset_circuits[1], full_circuits[0]);

        // `fitness_from_counts` on the subset equals `−mean_loss` over exactly
        // those two samples (with their labels 1.0 and -1.0), computed by hand
        // from the same synthetic counts.
        //   sample 2 (y=+1): "0":3,"1":1 → ⟨Z₀⟩=+0.5, SquaredError=(0.5−1)²=0.25
        //   sample 0 (y=+1): "0":1,"1":3 → ⟨Z₀⟩=−0.5, SquaredError=(−0.5−1)²=2.25
        let c = vec![counts(&[("0", 3), ("1", 1)]), counts(&[("0", 1), ("1", 3)])];
        let fitness = subset.fitness_from_counts(&c).unwrap();
        assert!(
            (fitness + (0.25 + 2.25) / 2.0).abs() < 1e-12,
            "fitness={fitness}"
        );
    }

    #[test]
    fn from_subset_rejects_an_empty_index_set() {
        // An empty minibatch would build a 0-template problem whose
        // `fitness_from_counts` divides by zero and returns `Ok(NaN)`, breaking
        // C-8. `from_subset` reasserts `Dataset`'s non-empty invariant instead
        // of building that degenerate problem.
        let full = distinct_feature_problem(&[0.15, 0.25, 0.35], &[1.0, -1.0, 1.0]);
        assert_eq!(
            full.from_subset(&[]).unwrap_err(),
            ValidationError::EmptyDataset
        );
        // The full problem itself is untouched by the rejected call.
        assert_eq!(full.num_circuits(), 3);
    }

    // ── Exact mode (design doc §17) ──────────────────────────────────────────

    fn probs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    /// Regression tying the two estimators together: for a real problem and a
    /// fixed `θ`, `expectations_from_counts` at a very high shot count (fixed
    /// seed) converges to the exact `expectations_from_probabilities`. Same
    /// spirit as Phase 5's `counts_expectation_matches_exact_expectation`, but
    /// comparing the two `QmlProblem` estimators directly. The exact
    /// probabilities and the counts both come from the same statevector, so any
    /// gap is pure sampling noise, tamed by the shot count and the loose
    /// tolerance.
    #[test]
    fn counts_expectations_converge_to_exact_probability_expectations() {
        use polypus_sim::{Simulator, SplitMix64, StatevectorSimulator};

        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        let theta: Vec<f64> = (0..problem.num_params())
            .map(|k| 0.1 + 0.13 * k as f64)
            .collect();
        let circuits = problem.bind_batch(&theta).unwrap();

        let sim = StatevectorSimulator::new();
        let mut exact_probs = Vec::with_capacity(circuits.len());
        let mut sampled_counts = Vec::with_capacity(circuits.len());
        let mut rng = SplitMix64::new(0xC0FFEE_u64);
        for circuit in &circuits {
            let sv = sim.run(circuit).unwrap();
            let width = circuit.num_qubits;
            // Exact probabilities, keyed exactly as the native backend does.
            let probs_map: HashMap<String, f64> = sv
                .probabilities()
                .into_iter()
                .enumerate()
                .map(|(state, p)| (format!("{state:0width$b}"), p))
                .collect();
            exact_probs.push(probs_map);
            // High-shot counts drawn from the same state.
            let counts_map: HashMap<String, u64> = sv
                .sample(400_000, &mut rng)
                .into_iter()
                .map(|(state, c)| (format!("{state:0width$b}"), c))
                .collect();
            sampled_counts.push(counts_map);
        }

        let exact = problem
            .expectations_from_probabilities(&exact_probs)
            .unwrap();
        let estimated = problem.expectations_from_counts(&sampled_counts).unwrap();
        assert_eq!(exact.len(), estimated.len());
        for (e, s) in exact.iter().zip(&estimated) {
            assert!(
                (e - s).abs() < 0.01,
                "counts estimate {s} deviates from exact {e}"
            );
        }
    }

    #[test]
    fn expectations_from_probabilities_reads_raw_z0_and_checks_length() {
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        // "0":0.75,"1":0.25 → +0.5 ; "0":0.25,"1":0.75 → −0.5.
        let p = vec![
            probs(&[("0", 0.75), ("1", 0.25)]),
            probs(&[("0", 0.25), ("1", 0.75)]),
        ];
        let exps = problem.expectations_from_probabilities(&p).unwrap();
        assert!((exps[0] - 0.5).abs() < 1e-12);
        assert!((exps[1] + 0.5).abs() < 1e-12);

        let err = problem
            .expectations_from_probabilities(&p[..1])
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
    fn fitness_from_probabilities_is_neg_mean_loss() {
        // Mirror of `fitness_from_counts_is_neg_mean_loss_over_expectations`
        // with exact probabilities that reproduce the same ⟨O₀⟩ = ±0.5.
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        let p = vec![
            probs(&[("0", 0.75), ("1", 0.25)]),
            probs(&[("0", 0.25), ("1", 0.75)]),
        ];
        // SquaredError: (0.5−1)² = 0.25 ; (−0.5−(−1))² = 0.25 → mean 0.25.
        let expected = -(0.25 + 0.25) / 2.0;
        let fitness = problem.fitness_from_probabilities(&p).unwrap();
        assert!((fitness - expected).abs() < 1e-12, "fitness={fitness}");
    }

    #[test]
    fn param_gradient_exact_matches_hand_computation() {
        // Mirror of `param_gradient_matches_hand_computation` with exact probs:
        //   sample 0 (y=+1): base=0.5, plus=+1.0, minus=0.0
        //   sample 1 (y=−1): base=−0.5, plus=0.0, minus=+1.0  → grad = +0.5
        let problem = one_qubit_z0_problem(&[1.0, -1.0]);
        let base = vec![0.5, -0.5];
        let plus = vec![probs(&[("0", 1.0)]), probs(&[("0", 0.5), ("1", 0.5)])];
        let minus = vec![probs(&[("0", 0.5), ("1", 0.5)]), probs(&[("0", 1.0)])];
        let grad = problem.param_gradient_exact(&base, &plus, &minus).unwrap();
        assert!((grad - 0.5).abs() < 1e-12, "grad={grad}");
    }

    #[test]
    fn param_gradient_exact_checks_lengths_in_documented_order() {
        let problem = one_qubit_z0_problem(&[1.0, -1.0]); // num_circuits() == 2
        let ok_pair = vec![probs(&[("0", 1.0)]), probs(&[("1", 1.0)])];
        let base_ok = vec![0.0, 0.0];

        // plus wrong → reported first.
        let err = problem
            .param_gradient_exact(&base_ok, &ok_pair[..1], &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        // plus ok, minus wrong → reported next.
        let err = problem
            .param_gradient_exact(&base_ok, &ok_pair, &ok_pair[..1])
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        // plus/minus ok, base wrong → reported last.
        let err = problem
            .param_gradient_exact(&base_ok[..1], &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn fitness_from_probabilities_categorical_matches_hand_computation() {
        // Mirror of `fitness_from_counts_categorical_matches_hand_computation`
        // with exact probs reproducing z=[1.0,0.5] and z=[−0.5,−1.0].
        let problem = categorical_two_class_problem();
        let p = vec![
            probs(&[("00", 0.75), ("10", 0.25)]),
            probs(&[("10", 0.25), ("11", 0.75)]),
        ];
        let fitness = problem.fitness_from_probabilities(&p).unwrap();
        assert!(fitness.is_finite());
        assert!((fitness + 0.724074).abs() < 1e-3, "fitness={fitness}");
    }

    #[test]
    fn param_gradient_categorical_exact_matches_hand_computation() {
        // Mirror of `param_gradient_categorical_matches_hand_computation`.
        let problem = categorical_two_class_problem();
        let base = vec![vec![1.0, 0.5], vec![-0.5, -1.0]];
        // plus ("10") → z=[1.0,−1.0] ; minus ("01") → z=[−1.0,1.0].
        let plus = vec![probs(&[("10", 1.0)]), probs(&[("10", 1.0)])];
        let minus = vec![probs(&[("01", 1.0)]), probs(&[("01", 1.0)])];
        let grad = problem
            .param_gradient_categorical_exact(&base, &plus, &minus)
            .unwrap();
        assert!((grad + 0.244918).abs() < 1e-3, "grad={grad}");
    }

    #[test]
    fn param_gradient_categorical_exact_checks_lengths_in_documented_order() {
        let problem = categorical_two_class_problem(); // num_circuits() == 2
        let ok_pair = vec![probs(&[("00", 1.0)]), probs(&[("11", 1.0)])];
        let base_ok = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        let err = problem
            .param_gradient_categorical_exact(&base_ok, &ok_pair[..1], &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        let err = problem
            .param_gradient_categorical_exact(&base_ok, &ok_pair, &ok_pair[..1])
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        let err = problem
            .param_gradient_categorical_exact(&base_ok[..1], &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn param_gradient_categorical_exact_rejects_wrong_class_count() {
        // Mirror of `param_gradient_categorical_rejects_wrong_class_count`.
        let problem = categorical_two_class_problem(); // 2 circuits, 2 classes
        let ok_pair = vec![probs(&[("00", 1.0)]), probs(&[("11", 1.0)])];

        let base_short = vec![vec![0.0, 0.0], vec![0.0]];
        let err = problem
            .param_gradient_categorical_exact(&base_short, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 1,
                expected: 2,
                got: 1,
            }
        );

        // A longer inner vector fails too — the `zip` must not hide it.
        let base_long = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0]];
        let err = problem
            .param_gradient_categorical_exact(&base_long, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 0,
                expected: 2,
                got: 3,
            }
        );

        // The inner check runs after all three outer ones.
        let err = problem
            .param_gradient_categorical_exact(&base_short, &ok_pair[..1], &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        let err = problem
            .param_gradient_categorical_exact(&base_short, &ok_pair, &ok_pair[..1])
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );
        let err = problem
            .param_gradient_categorical_exact(&base_short[..1], &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::CountsLengthMismatch {
                expected: 2,
                got: 1
            }
        );

        // First offender wins.
        let base_both = vec![vec![0.0], vec![0.0, 0.0, 0.0]];
        let err = problem
            .param_gradient_categorical_exact(&base_both, &ok_pair, &ok_pair)
            .unwrap_err();
        assert_eq!(
            err,
            QmlError::ClassCountMismatch {
                sample: 0,
                expected: 2,
                got: 1,
            }
        );
    }
}
