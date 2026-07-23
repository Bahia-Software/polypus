//! The layer-compilation core: the context threaded through a model's layers
//! during compilation, the per-layer allocation each layer records, and the
//! internal `LayerOps` trait that every layer implements.
//!
//! The public model builder ([`QuantumModel`]) and its compiled counterpart
//! ([`CompiledModel`]) also live here — see the second half of the file. This
//! first half is the machinery they drive.
//!
//! ## The two passes
//!
//! A layer behaves in two phases (design doc §5.3), and `LayerOps` names both:
//!
//! 1. [`plan`](LayerOps::plan) runs during `compile`, *without* data. It
//!    validates precondition (enough active qubits?), reserves its slice of the
//!    global `θ` index space by advancing [`LayerContext::param_cursor`], and
//!    records the [`LayerAllocation`] — the single source of truth for its `θ`
//!    range and the qubits it works on.
//! 2. [`emit`](LayerOps::emit) runs during `template_for(x)`. It appends gates
//!    to the circuit using the allocation recorded in `plan`, never
//!    recomputing it — so the two passes cannot diverge by construction.
//!
//! **Hard rule for every `emit` (design doc §5.3, ENGINEERING §9):** gates are
//! appended *exclusively* through
//! [`try_push`](polypus_circuit::ParameterizedCircuit::try_push), propagating
//! its `Result` with `?` into [`QmlError::Circuit`](crate::QmlError). The
//! fluent builder methods (`.ry(..)`, `.cx(..)`, …) `panic!` on any
//! `CircuitError`, which would turn an internal bookkeeping bug into an
//! unrecoverable crash instead of a typed error crossing the FFI boundary.

use std::ops::Range;

use polypus_circuit::{ConcreteCircuit, GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::{
    AngleEncoder, ConvBlock, ConvLayer, HardwareEfficientAnsatz, Layer, PoolBlock, PoolLayer,
    RotationAxis,
};
use crate::observables::{Pauli, ResolvedObservable, ResolvedPauliString};
use crate::readout::{Readout, ResolvedReadout};

/// The accumulator threaded through a model's layers, in order, during
/// compilation. Carries exactly the three things a layer's `plan` needs.
pub(crate) struct LayerContext {
    /// Live qubits, in logical order. Layers address logical positions of this
    /// list, not physical indices (design doc §6); pooling (a later phase)
    /// removes entries.
    pub(crate) active: Vec<usize>,
    /// The model's feature capacity, used by encoders to size their emission.
    pub(crate) num_features: usize,
    /// The next free global `θ` index. `plan` advances it by the number of
    /// parameters the layer reserves.
    pub(crate) param_cursor: usize,
    /// How many layers have already been planned before the current one. The
    /// [`AmplitudeEncoder`](crate::AmplitudeEncoder) reads this to enforce that
    /// it is the first layer (it prepares a state from `|0…0⟩`, so it cannot
    /// compose on top of earlier gates). `compile` bumps it after every
    /// [`plan`](LayerOps::plan).
    pub(crate) layers_planned: usize,
}

/// What a layer records in `plan` and consumes in `emit`: the half-open range
/// of global `θ` indices it owns, and a snapshot of the qubits it operates on.
#[derive(Debug, Clone)]
pub(crate) struct LayerAllocation {
    /// The layer's slice of the global parameter space, `start..end`. Empty
    /// (`start == end`) for layers that consume no `θ` (e.g. encoders).
    pub(crate) params: Range<usize>,
    /// The active qubits at this layer's position, captured during `plan`.
    pub(crate) active: Vec<usize>,
}

/// The two-pass behaviour every layer implements. Internal: the public surface
/// is the closed [`Layer`](crate::Layer) enum, which dispatches to the concrete
/// layer's implementation.
pub(crate) trait LayerOps {
    /// Validate and reserve resources during compilation (no data). See the
    /// module docs for the contract.
    fn plan(&self, ctx: &mut LayerContext) -> Result<LayerAllocation, ValidationError>;

    /// Append this layer's gates to `qc` using the recorded `alloc` and the
    /// sample `x`. Must use only `try_push` (see the module docs).
    fn emit(
        &self,
        qc: &mut ParameterizedCircuit,
        alloc: &LayerAllocation,
        x: &[f64],
    ) -> Result<(), QmlError>;
}

/// A quantum model under construction: an ordered list of layers over a fixed
/// number of qubits. The builder accepts any sequence; validation happens once,
/// in [`compile`](Self::compile), which yields the immutable [`CompiledModel`]
/// ("make illegal states unrepresentable", ENGINEERING §9).
///
/// The [`Readout`] (observables + decision) is attached with
/// [`readout`](Self::readout); [`compile`](Self::compile) requires one.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumModel {
    num_qubits: usize,
    layers: Vec<Layer>,
    readout: Option<Readout>,
}

impl QuantumModel {
    /// Start an empty model over `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        QuantumModel {
            num_qubits,
            layers: Vec::new(),
            readout: None,
        }
    }

    /// Append `layer` to the model.
    pub fn layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Attach the [`Readout`] that [`compile`](Self::compile) will resolve
    /// against the model's final active qubits. A model without one is rejected
    /// with [`ValidationError::MissingReadout`]. Calling this twice keeps the
    /// last readout.
    pub fn readout(mut self, readout: Readout) -> Self {
        self.readout = Some(readout);
        self
    }

    /// Sugar for `.layer(Layer::AngleEncoder(AngleEncoder::new(axis)))`.
    pub fn angle_encoder(self, axis: RotationAxis) -> Self {
        self.layer(Layer::AngleEncoder(AngleEncoder::new(axis)))
    }

    /// Sugar for a default [`HardwareEfficientAnsatz::new`] with `reps`
    /// repetitions.
    pub fn hardware_efficient(self, reps: usize) -> Self {
        self.layer(Layer::HardwareEfficient(HardwareEfficientAnsatz::new(reps)))
    }

    /// Sugar for `.layer(Layer::Conv(ConvLayer::new(block)))`, with the default
    /// [`Pairing::Alternating`](crate::Pairing).
    pub fn conv(self, block: ConvBlock) -> Self {
        self.layer(Layer::Conv(ConvLayer::new(block)))
    }

    /// Sugar for `.layer(Layer::Pool(PoolLayer::new(block)))`, with the default
    /// [`KeepRule::EvenPositions`](crate::KeepRule).
    pub fn pool(self, block: PoolBlock) -> Self {
        self.layer(Layer::Pool(PoolLayer::new(block)))
    }

    /// Validate the model against a dataset of `num_features` features and
    /// freeze it into a [`CompiledModel`].
    ///
    /// Checks the global invariants no single layer can see on its own, in
    /// order: at least one qubit ([`ValidationError::NoQubits`]), at least one
    /// layer ([`ValidationError::EmptyModel`]), a readout present
    /// ([`ValidationError::MissingReadout`]), then each layer's `plan` (which
    /// may raise e.g. [`ValidationError::NotEnoughQubits`]), that the model
    /// reserved at least one trainable parameter
    /// ([`ValidationError::NoTrainableParams`]), and finally that the readout
    /// resolves against the final active qubits (see [`resolve_readout`]).
    pub fn compile(self, num_features: usize) -> Result<CompiledModel, ValidationError> {
        if self.num_qubits < 1 {
            return Err(ValidationError::NoQubits);
        }
        if self.layers.is_empty() {
            return Err(ValidationError::EmptyModel);
        }
        let readout = match &self.readout {
            Some(readout) => readout,
            None => return Err(ValidationError::MissingReadout),
        };

        let mut ctx = LayerContext {
            active: (0..self.num_qubits).collect(),
            num_features,
            param_cursor: 0,
            layers_planned: 0,
        };
        let mut allocations = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            allocations.push(layer.plan(&mut ctx)?);
            // Track position so a later layer can tell it is *not* first (the
            // amplitude encoder rejects any position but the first).
            ctx.layers_planned += 1;
        }

        if ctx.param_cursor == 0 {
            return Err(ValidationError::NoTrainableParams);
        }
        let num_params = ctx.param_cursor;

        // The readout's logical positions are resolved against the *final*
        // active qubits (after any layer that removed some, e.g. a future
        // pooling), so this must run after the plan loop.
        let resolved_readout = resolve_readout(readout, &ctx.active)?;

        log::debug!(
            "compiled model: {} qubit(s), {} layer(s), {num_params} trainable parameter(s), {} feature(s), {} readout observable(s)",
            self.num_qubits,
            self.layers.len(),
            num_features,
            resolved_readout.observables().len(),
        );

        Ok(CompiledModel {
            spec: self,
            num_features,
            num_params,
            allocations,
            resolved_readout,
        })
    }
}

/// Resolve a [`Readout`]'s logical qubit positions to physical indices against
/// the model's final `active` qubits, validating as it goes:
///
/// - a position `>= active.len()` is [`ValidationError::ObservableQubitOutOfRange`];
/// - any Pauli other than `Z` is [`ValidationError::UnsupportedPauli`] (v1
///   readout is computational-basis only, design doc §7.2).
fn resolve_readout(
    readout: &Readout,
    active: &[usize],
) -> Result<ResolvedReadout, ValidationError> {
    let mut resolved_observables = Vec::with_capacity(readout.observables.len());
    for observable in &readout.observables {
        let mut resolved_terms = Vec::with_capacity(observable.terms.len());
        for (coeff, string) in &observable.terms {
            let mut resolved_positions = Vec::with_capacity(string.terms().len());
            for &(position, pauli) in string.terms() {
                if position >= active.len() {
                    return Err(ValidationError::ObservableQubitOutOfRange {
                        position,
                        num_active: active.len(),
                    });
                }
                if pauli != Pauli::Z {
                    return Err(ValidationError::UnsupportedPauli { pauli, position });
                }
                resolved_positions.push((active[position], pauli));
            }
            resolved_terms.push((*coeff, ResolvedPauliString::new(resolved_positions)));
        }
        resolved_observables.push(ResolvedObservable::new(resolved_terms));
    }
    Ok(ResolvedReadout::new(resolved_observables, readout.decision))
}

/// A validated, immutable model: it cannot be in an invalid state, so
/// [`template_for`](Self::template_for) and [`bind`](Self::bind) only fail on a
/// bad sample or bad parameter values, never on structural problems.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    spec: QuantumModel,
    num_features: usize,
    num_params: usize,
    /// One allocation per layer, in layer order (parallel to `spec.layers`).
    allocations: Vec<LayerAllocation>,
    /// The readout resolved to physical qubit indices (design doc §5.2).
    resolved_readout: ResolvedReadout,
}

impl CompiledModel {
    /// The number of trainable parameters — the `dimensions` an optimizer sees.
    /// Always `> 0` (guaranteed by [`compile`](QuantumModel::compile)).
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// The number of features per sample the model was compiled for.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// The resolved readout (crate-internal: consumed by
    /// [`QmlProblem`](crate::QmlProblem) to estimate expectations and predict).
    pub(crate) fn resolved_readout(&self) -> &ResolvedReadout {
        &self.resolved_readout
    }

    /// Build the parameterized circuit template for one sample `x`: features
    /// fixed, `θ` still free, terminated by a `measure_all`.
    ///
    /// Fails with [`QmlError::FeatureCountMismatch`] if `x.len()` differs from
    /// [`num_features`](Self::num_features).
    pub fn template_for(&self, x: &[f64]) -> Result<ParameterizedCircuit, QmlError> {
        if x.len() != self.num_features {
            return Err(QmlError::FeatureCountMismatch {
                expected: self.num_features,
                got: x.len(),
            });
        }

        let mut qc = ParameterizedCircuit::new(self.spec.num_qubits);
        for (layer, alloc) in self.spec.layers.iter().zip(self.allocations.iter()) {
            layer.emit(&mut qc, alloc, x)?;
        }
        // v1 readout is computational-basis only (design doc §7.2): no basis
        // change, just a terminal measurement.
        qc.try_push(GateInstruction::MeasureAll)?;
        Ok(qc)
    }

    /// Bind trainable parameters `theta` for sample `x`, producing a concrete
    /// circuit. Equivalent to `template_for(x)?.assign_parameters(theta)`; the
    /// wrong number of parameters surfaces as
    /// [`QmlError::Circuit`]`(`[`CircuitError::WrongNumberOfParams`]`)`.
    ///
    /// [`CircuitError::WrongNumberOfParams`]: polypus_circuit::CircuitError::WrongNumberOfParams
    pub fn bind(&self, x: &[f64], theta: &[f64]) -> Result<ConcreteCircuit, QmlError> {
        let template = self.template_for(x)?;
        Ok(template.assign_parameters(theta)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observables::{Observable, Pauli, PauliString};
    use crate::readout::{Decision, Readout};

    /// A minimal readout: `⟨Z₀⟩` with a `Sign` decision, reusable by every
    /// test that needs `compile` to reach past the `MissingReadout` check.
    fn z0_readout() -> Readout {
        Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap()
    }

    #[test]
    fn compile_rejects_zero_qubits() {
        let err = QuantumModel::new(0)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .compile(1)
            .unwrap_err();
        assert_eq!(err, ValidationError::NoQubits);
    }

    #[test]
    fn compile_rejects_empty_model() {
        let err = QuantumModel::new(2).compile(1).unwrap_err();
        assert_eq!(err, ValidationError::EmptyModel);
    }

    #[test]
    fn compile_rejects_model_without_trainable_params() {
        // An encoder alone reserves no θ.
        let err = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .readout(z0_readout())
            .compile(2)
            .unwrap_err();
        assert_eq!(err, ValidationError::NoTrainableParams);
    }

    #[test]
    fn compile_propagates_layer_validation() {
        // Encoder needs num_features <= active qubits.
        let err = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(z0_readout())
            .compile(3)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::NotEnoughQubits {
                needed: 3,
                active: 2,
            }
        );
    }

    #[test]
    fn template_for_rejects_wrong_feature_count() {
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(z0_readout())
            .compile(2)
            .unwrap();
        let err = model.template_for(&[0.1]).unwrap_err();
        assert_eq!(
            err,
            QmlError::FeatureCountMismatch {
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn compile_template_bind_end_to_end() {
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(z0_readout())
            .compile(2)
            .unwrap();

        // real_amplitudes(1) on 2 qubits: 2 × 1 × (1 rep + final) = 4 params.
        assert_eq!(model.num_params(), 4);
        assert_eq!(model.num_features(), 2);

        let template = model.template_for(&[0.1, 0.2]).unwrap();
        assert_eq!(template.num_params, 4);
        // Encoder rotations (Fixed), ansatz rotations (Param), one Cx, and the
        // terminal measure.
        assert_eq!(*template.gates.last().unwrap(), GateInstruction::MeasureAll);
        assert_eq!(
            polypus_circuit::terminal_measurement_violation(&template.gates),
            None
        );

        let concrete = model.bind(&[0.1, 0.2], &[0.3, 0.4, 0.5, 0.6]).unwrap();
        assert_eq!(concrete.num_qubits, 2);
        assert_eq!(*concrete.gates.last().unwrap(), GateInstruction::MeasureAll);

        // Wrong θ length surfaces as a wrapped CircuitError, not a panic.
        let err = model.bind(&[0.1, 0.2], &[0.3]).unwrap_err();
        assert!(matches!(
            err,
            QmlError::Circuit(polypus_circuit::CircuitError::WrongNumberOfParams {
                expected: 4,
                got: 1,
            })
        ));
    }

    #[test]
    fn compile_rejects_missing_readout() {
        // A model with qubits and a trainable layer but no readout is rejected
        // *before* the plan loop (MissingReadout precedes NoTrainableParams).
        let err = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .compile(2)
            .unwrap_err();
        assert_eq!(err, ValidationError::MissingReadout);
    }

    #[test]
    fn compile_rejects_observable_qubit_out_of_range() {
        // A 2-qubit model with a readout on logical position 2.
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(2, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let err = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(2)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::ObservableQubitOutOfRange {
                position: 2,
                num_active: 2,
            }
        );
    }

    #[test]
    fn compile_rejects_unsupported_pauli() {
        // v1 readout is Z-only; an X factor is rejected at compile time.
        let readout = Readout::new(
            vec![
                Observable::new(vec![(1.0, PauliString::new(vec![(0, Pauli::X)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Sign,
        )
        .unwrap();
        let err = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(readout)
            .compile(2)
            .unwrap_err();
        assert_eq!(
            err,
            ValidationError::UnsupportedPauli {
                pauli: Pauli::X,
                position: 0,
            }
        );
    }

    #[test]
    fn compile_resolves_readout_and_keeps_it() {
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(z0_readout())
            .compile(2)
            .unwrap();
        // The resolved readout survives and is cloneable together with the model.
        assert_eq!(model.resolved_readout().observables().len(), 1);
        let cloned = model.clone();
        assert_eq!(cloned.num_params(), model.num_params());
    }
}
