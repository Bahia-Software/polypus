//! [`ExpectationJacobian`]: the loss-free Jacobian of a model's readout
//! expectations, with respect to its trainable parameters, its inputs, or both.
//!
//! [`QmlProblem`](crate::QmlProblem) differentiates a *fitness*: a loss over
//! labelled samples. This is the layer beneath it: for each sample `x` and each
//! readout observable `O_o`, the derivatives of `⟨O_o⟩(x, θ)` itself, so a
//! caller can compose any loss it likes (the chain rule is then its own). It
//! needs no labels and no loss, so it takes every readout, `Argmax` included.
//!
//! The mechanics are [`QmlProblem`](crate::QmlProblem)'s exact gradient, reused
//! rather than copied: every template is *untied* (each angle occurrence its own
//! free parameter) and each occurrence is shifted by `±π/2` on its own, exact for
//! the Pauli rotations `exp(−iθP/2)` this crate emits. Trainable gates add
//! `(⟨O⟩(+) − ⟨O⟩(−))/2` to the parameter they read, summed over the gates that
//! share it. The input derivative goes through the gates the encoders record as
//! reading the sample (their [`InputTrace`](crate::model::InputTrace)): a gate
//! whose angle is `a(x)` adds `∂a/∂x_f · (⟨O⟩(+) − ⟨O⟩(−))/2` to each feature
//! `f` it reads. An angle encoder's `R(x_f)` has factor `1`; an IQP encoder's
//! `Rz(x_i)` has factor `1` and its `Rzz(x_i·x_j)` factor `x_j` towards `x_i`
//! and `x_i` towards `x_j`, the gate being shifted once for both.
//!
//! **Cost.** `2 · shifts` circuits per sample, where `shifts` counts the gates
//! that read a trainable parameter (with `wrt` [`Params`](JacobianWrt::Params)),
//! the gates that read the sample (with [`Inputs`](JacobianWrt::Inputs)), or both
//! (with [`Both`](JacobianWrt::Both)). No unshifted circuit is needed: the
//! expectations themselves are the business of
//! [`CompiledModel::expectations_from_counts`].

use std::collections::HashMap;

use polypus_circuit::{CircuitError, ConcreteCircuit, ParameterizedCircuit};

use crate::error::QmlError;
use crate::model::{CompiledModel, InputTrace};
use crate::observables::BitstringWeight;
use crate::problem::{untie_traced, Slot};

/// What an [`ExpectationJacobian`] differentiates with respect to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianWrt {
    /// The trainable parameters `θ`: one column per parameter.
    Params,
    /// The sample's features `x`: one column per feature. Angle and IQP
    /// encoders only (see [`QmlError::NoInputGradient`]).
    Inputs,
    /// Both, from one set of shifted circuits.
    Both,
}

impl JacobianWrt {
    fn params(self) -> bool {
        matches!(self, JacobianWrt::Params | JacobianWrt::Both)
    }

    fn inputs(self) -> bool {
        matches!(self, JacobianWrt::Inputs | JacobianWrt::Both)
    }
}

/// The circuits and the reader of the Jacobian of every readout observable's
/// expectation over a set of samples, from
/// [`CompiledModel::jacobian`].
///
/// Built once per set of samples (that is where every check runs), it binds a
/// parameter vector into the shifted circuits and turns their results back into
/// a [`Jacobian`], for all the samples at once or block by block
/// ([`blocks`](Self::blocks)), so a caller can bound how many circuits are alive.
///
/// ## Circuit order
///
/// Sample-major: for each sample in order, for each shifted occurrence in gate
/// order, the circuit with that one occurrence shifted by `+π/2`, then the one
/// shifted by `−π/2`. Results must be handed back in the same order.
#[derive(Debug, Clone)]
pub struct ExpectationJacobian {
    model: CompiledModel,
    wrt: JacobianWrt,
    samples: Vec<UntiedSample>,
    /// The slots that are shifted, in gate order: the same for every sample.
    shifted: Vec<usize>,
}

/// One sample's untied template and what each of its slots reads.
#[derive(Debug, Clone)]
struct UntiedSample {
    template: ParameterizedCircuit,
    slots: Vec<Slot>,
}

/// A slot without its sample-dependent values: what must agree across samples.
#[derive(Debug, PartialEq)]
enum SlotShape {
    Param(usize),
    Input(Vec<usize>),
}

impl SlotShape {
    fn of(slot: &Slot) -> SlotShape {
        match slot {
            Slot::Param(k) => SlotShape::Param(*k),
            Slot::Input { features, .. } => {
                SlotShape::Input(features.iter().map(|&(f, _)| f).collect())
            }
        }
    }
}

impl ExpectationJacobian {
    /// See [`CompiledModel::jacobian`], which documents the checks.
    pub(crate) fn new<S: AsRef<[f64]>>(
        model: &CompiledModel,
        samples: &[S],
        wrt: JacobianWrt,
    ) -> Result<Self, QmlError> {
        let traced = samples
            .iter()
            .map(|sample| model.traced_template_for(sample.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_traced(model, traced, wrt)
    }

    /// The checks of [`new`](Self::new) that follow building the templates,
    /// and the untying, over one `(template, trace)` per sample, in order: an
    /// opaque encoder against `wrt`, then each template untied, then every
    /// sample's structure against sample 0's. Apart from `new`, only the tests
    /// call it, with templates a model would never emit.
    fn from_traced(
        model: &CompiledModel,
        traced: Vec<(ParameterizedCircuit, InputTrace)>,
        wrt: JacobianWrt,
    ) -> Result<Self, QmlError> {
        let mut untied = Vec::with_capacity(traced.len());
        let mut structure: Option<Vec<SlotShape>> = None;
        for (index, (template, trace)) in traced.into_iter().enumerate() {
            if wrt.inputs() {
                if let Some(&encoder) = trace.opaque.first() {
                    return Err(QmlError::NoInputGradient { encoder });
                }
            }
            let (template, slots) = untie_traced(&template, &trace.gates)?;
            let shape: Vec<SlotShape> = slots.iter().map(SlotShape::of).collect();
            match &structure {
                Some(first) if *first != shape => {
                    return Err(QmlError::NoExactGradient {
                        reason: format!(
                            "sample {index} reads its parameters or features through different \
                             gates than sample 0"
                        ),
                    });
                }
                Some(_) => {}
                None => structure = Some(shape),
            }
            untied.push(UntiedSample { template, slots });
        }
        let shifted = structure
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter(|(_, shape)| match shape {
                SlotShape::Param(_) => wrt.params(),
                SlotShape::Input(_) => wrt.inputs(),
            })
            .map(|(slot, _)| slot)
            .collect();
        Ok(ExpectationJacobian {
            model: model.clone(),
            wrt,
            samples: untied,
            shifted,
        })
    }

    /// What this Jacobian differentiates with respect to.
    pub fn wrt(&self) -> JacobianWrt {
        self.wrt
    }

    /// The number of samples.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// The number of readout observables: the Jacobian's second axis.
    pub fn num_observables(&self) -> usize {
        self.model.num_observables()
    }

    /// The number of trainable parameters: the parameter Jacobian's last axis.
    pub fn num_params(&self) -> usize {
        self.model.num_params()
    }

    /// The number of features: the input Jacobian's last axis.
    pub fn num_features(&self) -> usize {
        self.model.num_features()
    }

    /// The occurrences shifted per sample: the gates that read a trainable
    /// parameter (a parameter shared by `g` gates counts `g` times), the gates
    /// that read the sample, or both, as `wrt` says. The same for every sample.
    pub fn num_shifts(&self) -> usize {
        self.shifted.len()
    }

    /// The circuits per sample: `2 · num_shifts()`.
    pub fn circuits_per_sample(&self) -> usize {
        2 * self.shifted.len()
    }

    /// Every circuit: `num_samples() · circuits_per_sample()`.
    pub fn num_circuits(&self) -> usize {
        self.samples.len() * self.circuits_per_sample()
    }

    /// Every sample as one block.
    pub fn all(&self) -> JacobianBlock<'_> {
        JacobianBlock {
            jacobian: self,
            samples: &self.samples,
        }
    }

    /// The block of the samples at `samples`, or `None` when the range does not
    /// lie within `0..num_samples()`.
    pub fn block(&self, samples: std::ops::Range<usize>) -> Option<JacobianBlock<'_>> {
        Some(JacobianBlock {
            jacobian: self,
            samples: self.samples.get(samples)?,
        })
    }

    /// The samples in consecutive blocks of `samples_per_block` (the last one
    /// possibly shorter; `0` counts as `1`), in order. Concatenating the blocks'
    /// Jacobians sample-major gives [`all`](Self::all)'s, value for value.
    pub fn blocks(&self, samples_per_block: usize) -> impl Iterator<Item = JacobianBlock<'_>> {
        self.samples
            .chunks(samples_per_block.max(1))
            .map(move |samples| JacobianBlock {
                jacobian: self,
                samples,
            })
    }

    /// [`all`](Self::all)`().circuits(theta)`.
    pub fn circuits(&self, theta: &[f64]) -> Result<Vec<ConcreteCircuit>, QmlError> {
        self.all().circuits(theta)
    }

    /// [`all`](Self::all)`().jacobian_from_counts(counts)`.
    pub fn jacobian_from_counts(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Jacobian, QmlError> {
        self.all().jacobian_from_counts(counts)
    }

    /// [`all`](Self::all)`().jacobian_from_probabilities(probabilities)`.
    pub fn jacobian_from_probabilities(
        &self,
        probabilities: &[HashMap<String, f64>],
    ) -> Result<Jacobian, QmlError> {
        self.all().jacobian_from_probabilities(probabilities)
    }
}

/// Consecutive samples of an [`ExpectationJacobian`], from
/// [`blocks`](ExpectationJacobian::blocks) or
/// [`all`](ExpectationJacobian::all): their shifted circuits, and their
/// Jacobian from those circuits' results.
#[derive(Debug, Clone, Copy)]
pub struct JacobianBlock<'a> {
    jacobian: &'a ExpectationJacobian,
    samples: &'a [UntiedSample],
}

impl JacobianBlock<'_> {
    /// The number of samples in the block.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// The block's circuits: `num_samples() · circuits_per_sample()`.
    pub fn num_circuits(&self) -> usize {
        self.samples.len() * self.jacobian.circuits_per_sample()
    }

    /// The block's shifted circuits at `theta`, in the
    /// [order](ExpectationJacobian#circuit-order) the readers expect. A `theta`
    /// of the wrong length is
    /// [`QmlError::Circuit`]`(`[`CircuitError::WrongNumberOfParams`]`)`.
    pub fn circuits(&self, theta: &[f64]) -> Result<Vec<ConcreteCircuit>, QmlError> {
        let num_params = self.jacobian.num_params();
        if theta.len() != num_params {
            return Err(QmlError::Circuit(CircuitError::WrongNumberOfParams {
                expected: num_params,
                got: theta.len(),
            }));
        }
        let shift = std::f64::consts::FRAC_PI_2;
        let mut circuits = Vec::with_capacity(self.num_circuits());
        for sample in self.samples {
            // Each slot at its unshifted value: the parameter it reads, or the
            // angle the encoder computed from the sample.
            let mut values: Vec<f64> = sample
                .slots
                .iter()
                .map(|slot| match slot {
                    Slot::Param(k) => theta[*k],
                    Slot::Input { angle, .. } => *angle,
                })
                .collect();
            for &slot in &self.jacobian.shifted {
                let base = values[slot];
                for delta in [shift, -shift] {
                    values[slot] = base + delta;
                    circuits.push(sample.template.assign_parameters(&values)?);
                }
                values[slot] = base;
            }
        }
        Ok(circuits)
    }

    /// The block's [`Jacobian`] from the measurement `counts` of
    /// [`circuits`](Self::circuits), in that order: an unbiased estimator.
    /// A slice of the wrong length is [`QmlError::CountsLengthMismatch`].
    pub fn jacobian_from_counts(
        &self,
        counts: &[HashMap<String, u64>],
    ) -> Result<Jacobian, QmlError> {
        self.jacobian_from_weighted(counts)
    }

    /// The exact-mode mirror of [`jacobian_from_counts`](Self::jacobian_from_counts), from exact
    /// basis-state probabilities: the Jacobian itself, with no shot noise.
    pub fn jacobian_from_probabilities(
        &self,
        probabilities: &[HashMap<String, f64>],
    ) -> Result<Jacobian, QmlError> {
        self.jacobian_from_weighted(probabilities)
    }

    fn jacobian_from_weighted<W: BitstringWeight>(
        &self,
        weights: &[HashMap<String, W>],
    ) -> Result<Jacobian, QmlError> {
        let expected = self.num_circuits();
        let mismatch = QmlError::CountsLengthMismatch {
            expected,
            got: weights.len(),
        };
        if weights.len() != expected {
            return Err(mismatch);
        }
        let jacobian = self.jacobian;
        let (n, k) = (self.samples.len(), jacobian.num_observables());
        let (num_params, num_features) = (jacobian.num_params(), jacobian.num_features());
        let wrt = jacobian.wrt;
        let mut params = wrt.params().then(|| vec![0.0; n * k * num_params]);
        let mut inputs = wrt.inputs().then(|| vec![0.0; n * k * num_features]);
        let per_sample = jacobian.circuits_per_sample();
        for (s, sample) in self.samples.iter().enumerate() {
            let results = weights
                .get(s * per_sample..(s + 1) * per_sample)
                .ok_or_else(|| mismatch.clone())?;
            for (pair, &slot) in results.chunks_exact(2).zip(&jacobian.shifted) {
                let plus = jacobian.model.expectations_from_weighted(&pair[0])?;
                let minus = jacobian.model.expectations_from_weighted(&pair[1])?;
                for (o, (plus, minus)) in plus.iter().zip(&minus).enumerate() {
                    // The shift rule: the derivative of `⟨O_o⟩` with respect
                    // to this one gate's angle.
                    let derivative = (plus - minus) / 2.0;
                    match (&sample.slots[slot], &mut params, &mut inputs) {
                        (Slot::Param(p), Some(params), _) => {
                            params[(s * k + o) * num_params + p] += derivative;
                        }
                        (Slot::Input { features, .. }, _, Some(inputs)) => {
                            for &(f, factor) in features {
                                inputs[(s * k + o) * num_features + f] += factor * derivative;
                            }
                        }
                        // Only the slots `wrt` asks for are shifted.
                        _ => {}
                    }
                }
            }
        }
        Ok(Jacobian {
            num_samples: n,
            num_observables: k,
            num_params,
            num_features,
            params,
            inputs,
        })
    }
}

/// The Jacobian of every readout observable's expectation over some samples:
/// with respect to the parameters, the inputs, or both, as the
/// [`JacobianWrt`] it was built with says.
///
/// Each part is flat and row-major: `params` has shape
/// `(num_samples, num_observables, num_params)`, entry
/// `[s][o][p] = ∂⟨O_o⟩(x_s, θ)/∂θ_p`, and `inputs` has shape
/// `(num_samples, num_observables, num_features)`, entry
/// `[s][o][f] = ∂⟨O_o⟩(x_s, θ)/∂x_{s,f}`. A part `wrt` did not ask for is
/// `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct Jacobian {
    num_samples: usize,
    num_observables: usize,
    num_params: usize,
    num_features: usize,
    params: Option<Vec<f64>>,
    inputs: Option<Vec<f64>>,
}

impl Jacobian {
    /// The number of samples: both parts' first axis.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// The number of readout observables: both parts' second axis.
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// The number of trainable parameters: the parameter part's last axis.
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// The number of features: the input part's last axis.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// The Jacobian with respect to the parameters, flat and row-major, if
    /// asked for.
    pub fn params(&self) -> Option<&[f64]> {
        self.params.as_deref()
    }

    /// The Jacobian with respect to the inputs, flat and row-major, if asked
    /// for.
    pub fn inputs(&self) -> Option<&[f64]> {
        self.inputs.as_deref()
    }

    /// `∂⟨O_observable⟩(x_sample, θ)/∂θ_param`, or `None` when out of range or
    /// not asked for.
    pub fn param(&self, sample: usize, observable: usize, param: usize) -> Option<f64> {
        if sample >= self.num_samples
            || observable >= self.num_observables
            || param >= self.num_params
        {
            return None;
        }
        let index = (sample * self.num_observables + observable) * self.num_params + param;
        self.params.as_ref()?.get(index).copied()
    }

    /// `∂⟨O_observable⟩(x_sample, θ)/∂x_{sample,feature}`, or `None` when out of
    /// range or not asked for.
    pub fn input(&self, sample: usize, observable: usize, feature: usize) -> Option<f64> {
        if sample >= self.num_samples
            || observable >= self.num_observables
            || feature >= self.num_features
        {
            return None;
        }
        let index = (sample * self.num_observables + observable) * self.num_features + feature;
        self.inputs.as_ref()?.get(index).copied()
    }

    /// The two parts, `(params, inputs)`, each `None` if not asked for.
    pub fn into_parts(self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (self.params, self.inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QuantumModel;
    use crate::observables::{Observable, Pauli, PauliString};
    use crate::readout::{Decision, Readout};
    use crate::RotationAxis;
    use polypus_circuit::{GateInstruction, GateParam};

    /// A one-qubit model: only its readout and sizes matter here, the
    /// templates are built by hand.
    fn model() -> CompiledModel {
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Raw,
        )
        .unwrap();
        QuantumModel::new(1)
            .angle_encoder(RotationAxis::Ry)
            .layer(crate::Layer::HardwareEfficient(
                crate::HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(1)
            .unwrap()
    }

    /// A measured one-qubit template of `gates`, with an empty trace.
    fn traced(gates: Vec<GateInstruction>) -> (ParameterizedCircuit, InputTrace) {
        let mut template = ParameterizedCircuit::new(1);
        for gate in gates {
            template.try_push(gate).unwrap();
        }
        template.try_push(GateInstruction::MeasureAll).unwrap();
        (template, InputTrace::default())
    }

    fn ry(k: usize) -> GateInstruction {
        GateInstruction::Ry {
            qubit: 0,
            theta: GateParam::Param(k),
        }
    }

    fn reason(result: Result<ExpectationJacobian, QmlError>) -> String {
        match result {
            Err(QmlError::NoExactGradient { reason }) => reason,
            other => panic!("expected NoExactGradient, got {other:?}"),
        }
    }

    #[test]
    fn samples_with_another_structure_than_sample_0_are_rejected() {
        // Another parameter at the same gate, and one occurrence more: each a
        // typed error naming the sample, never a Jacobian over mismatched slots.
        for second in [vec![ry(1)], vec![ry(0), ry(0)]] {
            let samples = vec![traced(vec![ry(0)]), traced(second)];
            let reason = reason(ExpectationJacobian::from_traced(
                &model(),
                samples,
                JacobianWrt::Params,
            ));
            assert!(reason.contains("sample 1"), "{reason}");
            assert!(reason.contains("than sample 0"), "{reason}");
        }
        // The same structure is accepted.
        let samples = vec![traced(vec![ry(0)]), traced(vec![ry(0)])];
        assert!(ExpectationJacobian::from_traced(&model(), samples, JacobianWrt::Params).is_ok());
    }

    #[test]
    fn an_angle_gate_that_is_not_a_pauli_rotation_has_no_exact_jacobian() {
        let phase = GateInstruction::P {
            qubit: 0,
            lam: GateParam::Param(0),
        };
        for wrt in [JacobianWrt::Params, JacobianWrt::Inputs, JacobianWrt::Both] {
            let reason = reason(ExpectationJacobian::from_traced(
                &model(),
                vec![traced(vec![phase.clone()])],
                wrt,
            ));
            assert!(reason.contains("Pauli rotation"), "{wrt:?}: {reason}");
        }
    }

    #[test]
    fn an_opaque_encoder_refuses_the_input_jacobian_only() {
        let opaque = || {
            let (template, mut trace) = traced(vec![ry(0)]);
            trace.opaque("amplitude");
            (template, trace)
        };
        for wrt in [JacobianWrt::Inputs, JacobianWrt::Both] {
            assert_eq!(
                ExpectationJacobian::from_traced(&model(), vec![opaque()], wrt).unwrap_err(),
                QmlError::NoInputGradient {
                    encoder: "amplitude"
                },
                "{wrt:?}"
            );
        }
        let jacobian =
            ExpectationJacobian::from_traced(&model(), vec![opaque()], JacobianWrt::Params)
                .unwrap();
        assert_eq!(jacobian.num_shifts(), 1);
    }
}
