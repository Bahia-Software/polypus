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
//! The adjoint runs ([`JacobianBlock::adjoint_runs`]) differentiate the same
//! occurrences, and also the trainable angles of the gates the two-term shift
//! rule has no rule for (the controlled rotations, the phase gates, the `u`
//! family; issue #561). A model with one is built like any other; only its
//! shifted circuits, and reading their results, fail
//! ([`QmlError::NoExactGradient`]).
//!
//! **Cost.** `2 · shifts` circuits per sample, where `shifts` counts the gates
//! that read a trainable parameter (with `wrt` [`Params`](JacobianWrt::Params)),
//! the gates that read the sample (with [`Inputs`](JacobianWrt::Inputs)), or both
//! (with [`Both`](JacobianWrt::Both)). No unshifted circuit is needed: the
//! expectations themselves are the business of
//! [`CompiledModel::expectations_from_counts`].

use std::collections::HashMap;

use polypus_circuit::{AngleRef, CircuitError, ConcreteCircuit, ParameterizedCircuit};

use crate::adjoint::{check_rows, AdjointRun};
use crate::error::{AdjointOnlyGate, QmlError};
use crate::model::{CompiledModel, InputTrace};
use crate::observables::BitstringWeight;
use crate::problem::{untie_traced, Slot, UntiedTemplate};

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
    /// The slots that are shifted (or differentiated by the adjoint), in gate
    /// order: the same for every sample.
    shifted: Vec<usize>,
    /// Whether the parameter shift differentiates the model, or the first
    /// instruction only the adjoint differentiates.
    shift: Result<(), AdjointOnlyGate>,
}

/// One sample's untied template and what each of its slots reads.
#[derive(Debug, Clone)]
struct UntiedSample {
    template: ParameterizedCircuit,
    slots: Vec<Slot>,
    /// `positions[j]`: where slot `j`'s gate sits in `template`, and which of
    /// its angles.
    positions: Vec<AngleRef>,
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
        let mut shift = Ok(());
        for (index, (template, trace)) in traced.into_iter().enumerate() {
            if wrt.inputs() {
                if let Some(&encoder) = trace.opaque.first() {
                    return Err(QmlError::NoInputGradient { encoder });
                }
            }
            let UntiedTemplate {
                template,
                slots,
                positions,
                shift: sample_shift,
            } = untie_traced(&template, &trace.gates)?;
            if shift.is_ok() {
                shift = sample_shift;
            }
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
            untied.push(UntiedSample {
                template,
                slots,
                positions,
            });
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
            shift,
        })
    }

    /// A Jacobian over `templates` (one per sample, none reading the sample),
    /// for the tests of the crates downstream, which cannot build a model
    /// whose templates hold the gates no layer emits (those only the adjoint
    /// differentiates). The checks of [`new`](Self::new) that follow building
    /// the templates still run. Behind the `test-support` feature, which only
    /// `[dev-dependencies]` enable: no build of the library or of the wheel has
    /// it.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn from_templates_for_tests(
        model: &CompiledModel,
        templates: Vec<ParameterizedCircuit>,
        wrt: JacobianWrt,
    ) -> Result<Self, QmlError> {
        let traced = templates
            .into_iter()
            .map(|template| (template, InputTrace::default()))
            .collect();
        Self::from_traced(model, traced, wrt)
    }

    /// Whether the exact Jacobian can fall back to the shifted circuits when
    /// the adjoint's two statevectors of `num_qubits` qubits do not fit the
    /// memory budget: `Ok` when the parameter shift differentiates this model
    /// too, and otherwise [`QmlError::NoExactGradient`] saying so — the memory,
    /// and the instruction only the adjoint differentiates. A caller about to
    /// fall back asks this first, so the end user learns why exact mode failed
    /// rather than being told to use it.
    pub fn check_shift_fallback(&self, num_qubits: usize) -> Result<(), QmlError> {
        self.shift
            .as_ref()
            .map_err(|gate| gate.memory_error(num_qubits))?;
        Ok(())
    }

    /// The parameter shift's own check: the shifted circuits, and reading
    /// their results, need a model the shift differentiates.
    fn check_shift(&self) -> Result<(), QmlError> {
        self.shift.as_ref().map_err(AdjointOnlyGate::shift_error)?;
        Ok(())
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

    /// [`all`](Self::all)`().adjoint_runs(theta)`.
    pub fn adjoint_runs(&self, theta: &[f64]) -> Result<Vec<AdjointRun>, QmlError> {
        self.all().adjoint_runs(theta)
    }

    /// [`all`](Self::all)`().jacobian_from_adjoint(derivatives)`.
    pub fn jacobian_from_adjoint(&self, derivatives: &[Vec<f64>]) -> Result<Jacobian, QmlError> {
        self.all().jacobian_from_adjoint(derivatives)
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
    /// [order](ExpectationJacobian#circuit-order) the readers expect. A model
    /// only the adjoint differentiates has none: [`QmlError::NoExactGradient`].
    /// A `theta` of the wrong length is
    /// [`QmlError::Circuit`]`(`[`CircuitError::WrongNumberOfParams`]`)`.
    pub fn circuits(&self, theta: &[f64]) -> Result<Vec<ConcreteCircuit>, QmlError> {
        self.jacobian.check_shift()?;
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
    /// A slice of the wrong length is [`QmlError::CountsLengthMismatch`]; a
    /// model only the adjoint differentiates, [`QmlError::NoExactGradient`].
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
        self.jacobian.check_shift()?;
        let expected = self.num_circuits();
        let mismatch = QmlError::CountsLengthMismatch {
            expected,
            got: weights.len(),
        };
        if weights.len() != expected {
            return Err(mismatch);
        }
        let jacobian = self.jacobian;
        let mut out = self.zero_jacobian();
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
                    out.add(s, o, &sample.slots[slot], (plus - minus) / 2.0);
                }
            }
        }
        Ok(out)
    }

    /// The Jacobian of the block's samples, all zero, with the parts `wrt`
    /// asks for.
    fn zero_jacobian(&self) -> Jacobian {
        let jacobian = self.jacobian;
        let (n, k) = (self.samples.len(), jacobian.num_observables());
        let (num_params, num_features) = (jacobian.num_params(), jacobian.num_features());
        Jacobian {
            num_samples: n,
            num_observables: k,
            num_params,
            num_features,
            params: jacobian.wrt.params().then(|| vec![0.0; n * k * num_params]),
            inputs: jacobian
                .wrt
                .inputs()
                .then(|| vec![0.0; n * k * num_features]),
        }
    }

    /// The number of adjoint runs of the block:
    /// `num_samples() · num_observables()`.
    pub fn num_adjoint_runs(&self) -> usize {
        self.samples.len() * self.jacobian.num_observables()
    }

    /// The block's [adjoint runs](crate::AdjointRun) at `theta`: for each
    /// sample in order, one per readout observable in order, each
    /// differentiating the angles the shift rule of [`circuits`](Self::circuits)
    /// would move, in gate order — and, on a model only the adjoint
    /// differentiates, the trainable angles of those gates too. The exact-mode
    /// alternative to the shifted circuits, for a caller that holds the
    /// statevector; read the results with
    /// [`jacobian_from_adjoint`](Self::jacobian_from_adjoint). A `theta` of the
    /// wrong length is
    /// [`QmlError::Circuit`]`(`[`CircuitError::WrongNumberOfParams`]`)`.
    pub fn adjoint_runs(&self, theta: &[f64]) -> Result<Vec<AdjointRun>, QmlError> {
        let num_params = self.jacobian.num_params();
        if theta.len() != num_params {
            return Err(QmlError::Circuit(CircuitError::WrongNumberOfParams {
                expected: num_params,
                got: theta.len(),
            }));
        }
        let observables = self
            .jacobian
            .model
            .resolved_readout()
            .observables()
            .iter()
            .map(|observable| observable.diagonal())
            .collect::<Result<Vec<_>, QmlError>>()?;
        let mut runs = Vec::with_capacity(self.num_adjoint_runs());
        for sample in self.samples {
            let values: Vec<f64> = sample
                .slots
                .iter()
                .map(|slot| match slot {
                    Slot::Param(k) => theta[*k],
                    Slot::Input { angle, .. } => *angle,
                })
                .collect();
            let circuit = sample.template.assign_parameters(&values)?;
            let wrt: Vec<AngleRef> = self
                .jacobian
                .shifted
                .iter()
                .map(|&slot| sample.positions[slot])
                .collect();
            for observable in &observables {
                runs.push(AdjointRun {
                    circuit: circuit.clone(),
                    wrt: wrt.clone(),
                    observable: observable.clone(),
                });
            }
        }
        Ok(runs)
    }

    /// The block's [`Jacobian`] from the derivatives of the runs of
    /// [`adjoint_runs`](Self::adjoint_runs): one row per run, in order, one
    /// derivative per shifted gate. Each derivative goes where the shift rule's
    /// would ([`jacobian_from_probabilities`](Self::jacobian_from_probabilities)):
    /// to the parameter its gate reads, or to each feature it reads times that
    /// feature's chain-rule factor. The wrong number of rows, or a row of the
    /// wrong length, is [`QmlError::CountsLengthMismatch`].
    pub fn jacobian_from_adjoint(&self, derivatives: &[Vec<f64>]) -> Result<Jacobian, QmlError> {
        let jacobian = self.jacobian;
        let k = jacobian.num_observables();
        check_rows(derivatives, self.num_adjoint_runs(), jacobian.shifted.len())?;
        let mut out = self.zero_jacobian();
        for (s, sample) in self.samples.iter().enumerate() {
            for o in 0..k {
                let row = &derivatives[s * k + o];
                for (&derivative, &slot) in row.iter().zip(&jacobian.shifted) {
                    out.add(s, o, &sample.slots[slot], derivative);
                }
            }
        }
        Ok(out)
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
    /// Add the derivative of `⟨O_o⟩(x_s)` with respect to one gate's angle
    /// where it belongs: to the parameter the gate reads, or to each feature
    /// it reads times that feature's chain-rule factor. A slot of a part this
    /// Jacobian does not hold is skipped (only the slots `wrt` asks for are
    /// differentiated).
    fn add(&mut self, s: usize, o: usize, slot: &Slot, derivative: f64) {
        let k = self.num_observables;
        match (slot, &mut self.params, &mut self.inputs) {
            (Slot::Param(p), Some(params), _) => {
                params[(s * k + o) * self.num_params + p] += derivative;
            }
            (Slot::Input { features, .. }, _, Some(inputs)) => {
                for &(f, factor) in features {
                    inputs[(s * k + o) * self.num_features + f] += factor * derivative;
                }
            }
            _ => {}
        }
    }

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

    /// The exact probabilities of `circuit`, keyed as the native backend keys
    /// them.
    fn probabilities(circuit: &ConcreteCircuit) -> HashMap<String, f64> {
        use polypus_sim::{Simulator, StatevectorSimulator};
        let width = circuit.num_qubits();
        StatevectorSimulator::new()
            .run(circuit)
            .unwrap()
            .probabilities()
            .into_iter()
            .enumerate()
            .map(|(state, p)| (format!("{state:0width$b}"), p))
            .collect()
    }

    /// #561, decision D2 and acceptance criterion 5 for the Jacobian: a model
    /// with gates only the adjoint differentiates is built, whatever `wrt`;
    /// its shifted circuits and their readers fail with the shift's message,
    /// its memory fallback with its own; and its adjoint Jacobian is the
    /// derivative of the expectations (central differences), a parameter two
    /// angles share included.
    #[test]
    fn an_angle_gate_only_the_adjoint_differentiates_has_an_adjoint_jacobian_only() {
        const STEP: f64 = 1e-5;
        const TOLERANCE: f64 = 1e-9;
        let gates = || {
            vec![
                GateInstruction::H(0),
                GateInstruction::P {
                    qubit: 0,
                    lam: GateParam::Param(0),
                },
                ry(1),
                GateInstruction::U {
                    qubit: 0,
                    theta: GateParam::Param(1),
                    phi: GateParam::Param(0),
                    lam: GateParam::Fixed(0.4),
                },
                GateInstruction::Sx(0),
            ]
        };
        for wrt in [JacobianWrt::Params, JacobianWrt::Inputs, JacobianWrt::Both] {
            assert!(
                ExpectationJacobian::from_traced(&model(), vec![traced(gates())], wrt).is_ok(),
                "{wrt:?}"
            );
        }
        let model = model();
        let (template, trace) = traced(gates());
        let jacobian = ExpectationJacobian::from_traced(
            &model,
            vec![(template.clone(), trace)],
            JacobianWrt::Params,
        )
        .unwrap();
        let theta = [0.7, -0.3];
        assert_eq!(model.num_params(), theta.len());
        for error in [
            jacobian.circuits(&theta).unwrap_err(),
            jacobian.jacobian_from_counts(&[]).unwrap_err(),
            jacobian.jacobian_from_probabilities(&[]).unwrap_err(),
        ] {
            let reason = reason(Err(error));
            assert!(reason.contains("instruction 1 (P {"), "{reason}");
            assert!(reason.contains("exact=True"), "{reason}");
        }
        let memory = reason(Err(jacobian.check_shift_fallback(1).unwrap_err()));
        assert!(
            memory.contains("two statevectors of 1 qubit at"),
            "{memory}"
        );
        assert!(memory.contains("instruction 1 (P {"), "{memory}");

        let runs = jacobian.adjoint_runs(&theta).unwrap();
        let derivatives: Vec<Vec<f64>> = runs
            .iter()
            .map(|run| {
                polypus_sim::StatevectorSimulator::new()
                    .adjoint_gradient(&run.circuit, &run.wrt, |b| run.observable.weight(b), None)
                    .unwrap()
                    .derivatives
            })
            .collect();
        let exact = jacobian.jacobian_from_adjoint(&derivatives).unwrap();
        let expectation = |theta: &[f64]| {
            let circuit = template.assign_parameters(theta).unwrap();
            model
                .expectations_from_probabilities(&probabilities(&circuit))
                .unwrap()[0]
        };
        for p in 0..theta.len() {
            let mut plus = theta;
            plus[p] += STEP;
            let mut minus = theta;
            minus[p] -= STEP;
            let fd = (expectation(&plus) - expectation(&minus)) / (2.0 * STEP);
            let d = exact.param(0, 0, p).unwrap();
            assert!((d - fd).abs() < TOLERANCE, "θ[{p}]: adjoint {d} vs {fd}");
            assert!(d.abs() > 1e-4, "θ[{p}]: {d} is vacuous");
        }
        // A model of Pauli rotations falls back.
        let pauli = ExpectationJacobian::from_traced(
            &model,
            vec![traced(vec![ry(0)])],
            JacobianWrt::Params,
        )
        .unwrap();
        assert_eq!(pauli.check_shift_fallback(1), Ok(()));
    }

    /// What no exact gradient differentiates is still refused at construction,
    /// whatever `wrt`: the call of a declared gate.
    #[test]
    fn a_declared_gate_has_no_exact_jacobian() {
        let declared = ParameterizedCircuit::from_qasm2(
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
             gate g(t) a { rz(t) a; }\nqreg q[1];\ng(0.3) q[0];\n",
        )
        .unwrap()
        .gates()[0]
            .clone();
        for wrt in [JacobianWrt::Params, JacobianWrt::Inputs, JacobianWrt::Both] {
            let reason = reason(ExpectationJacobian::from_traced(
                &model(),
                vec![traced(vec![declared.clone()])],
                wrt,
            ));
            assert!(reason.contains("declared gate"), "{wrt:?}: {reason}");
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
