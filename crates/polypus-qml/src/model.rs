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

use std::collections::HashMap;
use std::ops::Range;

use polypus_circuit::{ConcreteCircuit, GateInstruction, ParameterizedCircuit};

use crate::error::{QmlError, ValidationError};
use crate::layers::{
    AmplitudeEncoder, AngleEncoder, ConvBlock, ConvLayer, HardwareEfficientAnsatz, Layer,
    PoolBlock, PoolLayer, RotationAxis,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Sugar for `.layer(Layer::AmplitudeEncoder(AmplitudeEncoder))`. The
    /// [`AmplitudeEncoder`] takes no configuration in v1 and must be the first
    /// layer of the model.
    pub fn amplitude_encoder(self) -> Self {
        self.layer(Layer::AmplitudeEncoder(AmplitudeEncoder))
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
        // pooling), so this must run after the plan loop. Resolution also yields
        // the single basis change the readout is measured under (design doc
        // §7.2) — empty for an all-`Z` readout.
        let (resolved_readout, basis_change) = resolve_readout(readout, &ctx.active)?;

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
            basis_change,
        })
    }
}

/// Resolve a [`Readout`]'s logical qubit positions to physical indices against
/// the model's final `active` qubits, and work out the single basis change the
/// whole readout is measured under (design doc §7.2).
///
/// Returns the [`ResolvedReadout`] plus the *basis change*: the `(physical
/// index, Pauli)` pairs — sorted ascending by physical index, only the `X`/`Y`
/// entries that actually need a gate — that
/// [`template_for`](CompiledModel::template_for) inserts before the terminal
/// measurement (`H` for `X`; `Sdg` then `H` for `Y`). An all-`Z` readout — the
/// only case before this phase — yields an empty basis change, so its circuit
/// is byte-identical to before.
///
/// Validation, in order:
///
/// - a position `>= active.len()` is [`ValidationError::ObservableQubitOutOfRange`];
/// - an observable that asks for two different Paulis on the same qubit across
///   its terms is [`ValidationError::ObservableHasIncompatibleBases`];
/// - a readout whose observables do not all fit one basis group is
///   [`ValidationError::ReadoutNeedsMultipleBasisGroups`] — the multi-circuit
///   case, not implemented yet.
///
/// ## Grouping rule (single group only)
///
/// A *basis group* is a `position → Pauli` assignment such that every term of
/// every observable in the group requires, on each qubit it touches, exactly
/// the Pauli the group assigns there (untouched qubits impose nothing — they
/// measure in `Z` with no gate). Each observable is first collapsed to its own
/// per-qubit basis (checking internal compatibility), then placed greedily into
/// the first existing compatible group, opening a new one only when it fits
/// none. This phase supports a readout that resolves to **exactly one** group;
/// two or more (e.g. one class in `Z`, another in `X` on the same qubit) is the
/// rejected multi-circuit case.
fn resolve_readout(
    readout: &Readout,
    active: &[usize],
) -> Result<(ResolvedReadout, Vec<(usize, Pauli)>), ValidationError> {
    let mut resolved_observables = Vec::with_capacity(readout.observables.len());
    // One per-qubit basis map per observable (logical positions), built in
    // declaration order so the greedy grouping below is deterministic.
    let mut observable_bases: Vec<HashMap<usize, Pauli>> =
        Vec::with_capacity(readout.observables.len());
    for observable in &readout.observables {
        let mut resolved_terms = Vec::with_capacity(observable.terms.len());
        let mut bases: HashMap<usize, Pauli> = HashMap::new();
        for (coeff, string) in &observable.terms {
            let mut resolved_positions = Vec::with_capacity(string.terms().len());
            for &(position, pauli) in string.terms() {
                if position >= active.len() {
                    return Err(ValidationError::ObservableQubitOutOfRange {
                        position,
                        num_active: active.len(),
                    });
                }
                // Every term of one observable must agree on the basis of each
                // qubit it touches: `Z₀ + X₀` cannot be one measurement.
                match bases.get(&position) {
                    Some(&existing) if existing != pauli => {
                        return Err(ValidationError::ObservableHasIncompatibleBases {
                            position,
                            first: existing,
                            second: pauli,
                        });
                    }
                    Some(_) => {}
                    None => {
                        bases.insert(position, pauli);
                    }
                }
                resolved_positions.push((active[position], pauli));
            }
            resolved_terms.push((*coeff, ResolvedPauliString::new(resolved_positions)));
        }
        resolved_observables.push(ResolvedObservable::new(resolved_terms));
        observable_bases.push(bases);
    }

    // Greedy first-fit grouping over the per-observable basis maps.
    let mut groups: Vec<HashMap<usize, Pauli>> = Vec::new();
    for bases in observable_bases {
        let fits = groups.iter_mut().find(|group| {
            bases
                .iter()
                .all(|(pos, pauli)| group.get(pos).is_none_or(|g| g == pauli))
        });
        match fits {
            Some(group) => group.extend(bases),
            None => groups.push(bases),
        }
    }
    if groups.len() > 1 {
        return Err(ValidationError::ReadoutNeedsMultipleBasisGroups {
            groups: groups.len(),
        });
    }

    // The single group's non-`Z` entries become the circuit's basis change,
    // resolved to physical indices and sorted for a deterministic emission.
    let mut basis_change: Vec<(usize, Pauli)> = groups
        .into_iter()
        .next()
        .unwrap_or_default()
        .into_iter()
        .filter(|&(_, pauli)| pauli != Pauli::Z)
        .map(|(position, pauli)| (active[position], pauli))
        .collect();
    basis_change.sort_unstable_by_key(|&(index, _)| index);

    Ok((
        ResolvedReadout::new(resolved_observables, readout.decision),
        basis_change,
    ))
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
    /// The pre-measurement basis change the readout is measured under: the
    /// `(physical index, Pauli)` pairs — sorted by index, only the `X`/`Y`
    /// entries that need a gate — that [`template_for`](Self::template_for)
    /// emits before `measure_all` (design doc §7.2). Empty for an all-`Z`
    /// readout. Derived in [`compile`](QuantumModel::compile) alongside
    /// `resolved_readout`, so serde (which stores only `{spec, num_features}`
    /// and recompiles) regenerates it for free.
    basis_change: Vec<(usize, Pauli)>,
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
        // Rotate each non-`Z` qubit into the computational basis before the
        // terminal measurement (design doc §7.2): `H` measures `X`; `Sdg` then
        // `H` measures `Y`. `basis_change` is sorted by physical index, so the
        // emission is deterministic; it is empty for an all-`Z` readout, leaving
        // the circuit byte-identical to a bare computational-basis measurement.
        for &(qubit, pauli) in &self.basis_change {
            match pauli {
                Pauli::X => qc.try_push(GateInstruction::H(qubit))?,
                Pauli::Y => {
                    qc.try_push(GateInstruction::Sdg(qubit))?;
                    qc.try_push(GateInstruction::H(qubit))?;
                }
                // Z entries are filtered out in `resolve_readout`: they need no
                // gate. Guarded here so a future change cannot silently drop one.
                Pauli::Z => {}
            }
        }
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

    /// Infer a prediction from one sample's `counts`, applying the readout's
    /// [`Decision`](crate::Decision) (design doc §7.1) — the inference-only
    /// counterpart of
    /// [`QmlProblem::predict_from_counts`](crate::QmlProblem::predict_from_counts),
    /// usable directly on a `CompiledModel` (e.g. one loaded via
    /// [`TrainedModel`]) without a `Dataset`/`Loss`.
    ///
    /// Not gated behind `serde`: inference from counts is useful whether or not
    /// the model was loaded from disk.
    pub fn predict_from_counts(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        self.resolved_readout().predict(counts)
    }

    /// Infer a prediction from one sample's exact basis-state `probabilities`
    /// (design doc §17) — the exact-mode mirror of
    /// [`predict_from_counts`](Self::predict_from_counts).
    ///
    /// Not gated behind `serde`: it is inference, independent of serialization.
    pub fn predict_from_probabilities(
        &self,
        probabilities: &HashMap<String, f64>,
    ) -> Result<f64, QmlError> {
        self.resolved_readout()
            .predict_from_probabilities(probabilities)
    }
}

/// Serialize a [`CompiledModel`] as just `{spec, num_features}` — never its
/// derived fields (`num_params`, `allocations`, `resolved_readout`), which
/// [`Deserialize`](CompiledModel) recomputes by re-running `compile`. Written by
/// hand rather than derived so that loading always repeats the full validation
/// of [`compile`](QuantumModel::compile): a corrupt or tampered file can never
/// produce an internally inconsistent model (design doc §17).
#[cfg(feature = "serde")]
impl serde::Serialize for CompiledModel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct Wire<'a> {
            spec: &'a QuantumModel,
            num_features: usize,
        }
        Wire {
            spec: &self.spec,
            num_features: self.num_features,
        }
        .serialize(serializer)
    }
}

/// Deserialize a [`CompiledModel`] by recompiling `{spec, num_features}`: the
/// spec and feature count are read from the wire and fed straight back through
/// [`compile`](QuantumModel::compile), so every structural invariant is
/// revalidated on load. A spec that no longer compiles surfaces as a
/// deserialization error (via [`serde::de::Error::custom`] over the
/// [`ValidationError`]'s `Display`), never as a panic or a silently-accepted
/// inconsistent model.
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledModel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            spec: QuantumModel,
            num_features: usize,
        }
        let wire = Wire::deserialize(deserializer)?;
        wire.spec
            .compile(wire.num_features)
            .map_err(serde::de::Error::custom)
    }
}

/// A trained model ready for inference: a compiled model plus its optimal
/// trainable parameters (design doc §17). Exists only under the `serde`
/// feature — saving/loading is its only purpose.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainedModel {
    /// The compiled model (round-tripped as `{spec, num_features}`).
    pub model: CompiledModel,
    /// The optimal trainable parameters, `θ`.
    pub theta: Vec<f64>,
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

    /// Compile a 2-qubit `angle_encoder + real_amplitudes(1)` model with the
    /// given readout. `real_amplitudes` emits only `Ry`/`Cx`, so any terminal
    /// `H`/`Sdg` in the circuit can only be a basis change.
    fn compile_with_readout(readout: Readout) -> Result<CompiledModel, ValidationError> {
        QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .layer(Layer::HardwareEfficient(
                HardwareEfficientAnsatz::real_amplitudes(1),
            ))
            .readout(readout)
            .compile(2)
    }

    fn readout_of(observables: Vec<Observable>, decision: Decision) -> Readout {
        Readout::new(observables, decision).unwrap()
    }

    fn single(position: usize, pauli: Pauli) -> Observable {
        Observable::new(vec![(
            1.0,
            PauliString::new(vec![(position, pauli)]).unwrap(),
        )])
        .unwrap()
    }

    #[test]
    fn z_readout_records_no_basis_change() {
        // The pre-phase case: an all-Z readout adds no gate, so the emitted
        // circuit is byte-identical to a bare computational-basis measurement.
        let model = compile_with_readout(z0_readout()).unwrap();
        assert!(model.basis_change.is_empty());
        let template = model.template_for(&[0.1, 0.2]).unwrap();
        // Nothing but the layer gates precedes the terminal measurement — in
        // particular no H/Sdg was inserted (real_amplitudes emits neither).
        assert_eq!(*template.gates.last().unwrap(), GateInstruction::MeasureAll);
        assert!(!template
            .gates
            .iter()
            .any(|g| matches!(g, GateInstruction::H(_) | GateInstruction::Sdg(_))));
    }

    #[test]
    fn x_readout_inserts_hadamard_before_measure() {
        let model =
            compile_with_readout(readout_of(vec![single(0, Pauli::X)], Decision::Sign)).unwrap();
        // active == [0, 1], so logical position 0 is physical qubit 0.
        assert_eq!(model.basis_change, vec![(0, Pauli::X)]);
        let g = model.template_for(&[0.1, 0.2]).unwrap().gates;
        let n = g.len();
        assert_eq!(g[n - 2], GateInstruction::H(0));
        assert_eq!(g[n - 1], GateInstruction::MeasureAll);
    }

    #[test]
    fn y_readout_inserts_sdg_then_hadamard_before_measure() {
        let model =
            compile_with_readout(readout_of(vec![single(1, Pauli::Y)], Decision::Sign)).unwrap();
        assert_eq!(model.basis_change, vec![(1, Pauli::Y)]);
        let g = model.template_for(&[0.1, 0.2]).unwrap().gates;
        let n = g.len();
        // Temporal order: Sdg first, then H, then the measurement.
        assert_eq!(g[n - 3], GateInstruction::Sdg(1));
        assert_eq!(g[n - 2], GateInstruction::H(1));
        assert_eq!(g[n - 1], GateInstruction::MeasureAll);
    }

    #[test]
    fn shared_non_z_basis_groups_into_one_circuit() {
        // Two Argmax classes both measured in X, on different qubits, share one
        // basis group — the rule is not tied to Sign/Threshold/Raw.
        let model = compile_with_readout(readout_of(
            vec![single(0, Pauli::X), single(1, Pauli::X)],
            Decision::Argmax,
        ))
        .unwrap();
        assert_eq!(model.basis_change, vec![(0, Pauli::X), (1, Pauli::X)]);
    }

    #[test]
    fn compile_rejects_observable_with_incompatible_bases() {
        // A single observable asking for both Z and X on qubit 0 cannot be one
        // measurement.
        let observable = Observable::new(vec![
            (0.5, PauliString::new(vec![(0, Pauli::Z)]).unwrap()),
            (0.5, PauliString::new(vec![(0, Pauli::X)]).unwrap()),
        ])
        .unwrap();
        let err = compile_with_readout(readout_of(vec![observable], Decision::Sign)).unwrap_err();
        assert_eq!(
            err,
            ValidationError::ObservableHasIncompatibleBases {
                position: 0,
                first: Pauli::Z,
                second: Pauli::X,
            }
        );
    }

    #[test]
    fn compile_rejects_readout_needing_two_basis_groups() {
        // One class in Z and another in X on the *same* qubit needs two distinct
        // measurement bases — the multi-circuit case, not implemented yet.
        let err = compile_with_readout(readout_of(
            vec![single(0, Pauli::Z), single(0, Pauli::X)],
            Decision::Argmax,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            ValidationError::ReadoutNeedsMultipleBasisGroups { groups: 2 }
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

    #[test]
    fn predict_from_counts_applies_decision() {
        // Mirrors `QmlProblem::predict_from_counts`'s test, but straight on a
        // `CompiledModel` — no `Dataset`/`Loss` involved.
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(z0_readout())
            .compile(2)
            .unwrap();
        // ⟨Z₀⟩ over "00" (width 2) = +1 → Sign → +1.
        assert_eq!(model.predict_from_counts(&counts(&[("00", 10)])), Ok(1.0));
        // ⟨Z₀⟩ over "01" = −1 → Sign → −1.
        assert_eq!(model.predict_from_counts(&counts(&[("01", 10)])), Ok(-1.0));
    }

    #[test]
    fn predict_from_probabilities_applies_decision() {
        // The exact-mode mirror of `predict_from_counts_applies_decision`, fed
        // exact basis-state probabilities instead of counts.
        let model = QuantumModel::new(2)
            .angle_encoder(RotationAxis::Ry)
            .hardware_efficient(1)
            .readout(z0_readout())
            .compile(2)
            .unwrap();
        // ⟨Z₀⟩ over "00" (width 2) = +1 → Sign → +1.
        assert_eq!(
            model.predict_from_probabilities(&probabilities(&[("00", 1.0)])),
            Ok(1.0)
        );
        // ⟨Z₀⟩ over "01" = −1 → Sign → −1.
        assert_eq!(
            model.predict_from_probabilities(&probabilities(&[("01", 1.0)])),
            Ok(-1.0)
        );
    }

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn probabilities(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }
}

/// Round-trip serialization tests (design doc §17). Gated behind `serde` — the
/// only feature under which `CompiledModel`/`TrainedModel` serialize — and using
/// `serde_json` as the concrete format, exactly as a real save/load would.
#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use crate::layers::{ConvBlock, PoolBlock};
    use crate::observables::{Observable, Pauli, PauliString};
    use crate::readout::{Decision, Readout};

    /// A model exercising the entire serializable type tree: angle encoder,
    /// convolution, pooling, a hardware-efficient ansatz, and a multi-observable
    /// readout whose first observable is a weighted two-term sum with a `Z₀Z₁`
    /// string (so coefficients ≠ 1 and multi-factor strings both round-trip).
    fn full_model() -> CompiledModel {
        let readout = Readout::new(
            vec![
                Observable::new(vec![
                    (0.5, PauliString::new(vec![(0, Pauli::Z)]).unwrap()),
                    (
                        1.5,
                        PauliString::new(vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap(),
                    ),
                ])
                .unwrap(),
                Observable::new(vec![(1.0, PauliString::new(vec![(1, Pauli::Z)]).unwrap())])
                    .unwrap(),
            ],
            Decision::Argmax,
        )
        .unwrap();
        // 4 qubits, 4 features; pooling halves the active set to 2, which the
        // two-position readout resolves against.
        QuantumModel::new(4)
            .angle_encoder(RotationAxis::Ry)
            .conv(ConvBlock::Basic)
            .pool(PoolBlock::Basic)
            .hardware_efficient(1)
            .readout(readout)
            .compile(4)
            .unwrap()
    }

    #[test]
    fn compiled_model_round_trips_and_binds_identically() {
        let model = full_model();
        let json = serde_json::to_string(&model).unwrap();
        let loaded: CompiledModel = serde_json::from_str(&json).unwrap();

        // Derived fields recomputed on load must match the original.
        assert_eq!(loaded.num_params(), model.num_params());
        assert_eq!(loaded.num_features(), model.num_features());

        // The reloaded model must produce byte-identical circuits, not merely
        // compile: compare the concrete `template_for`/`bind` outputs.
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(
            loaded.template_for(&x).unwrap(),
            model.template_for(&x).unwrap()
        );
        let theta: Vec<f64> = (0..model.num_params()).map(|i| 0.05 * i as f64).collect();
        assert_eq!(
            loaded.bind(&x, &theta).unwrap(),
            model.bind(&x, &theta).unwrap()
        );
    }

    #[test]
    fn deserialization_revalidates_via_recompile() {
        // Take a valid serialized model and corrupt its spec to an empty-layer
        // model, which `compile` rejects with `EmptyModel`. Deserialization must
        // surface that as a clean error, never a panic or an accepted model.
        let json = serde_json::to_string(&full_model()).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["spec"]["layers"] = serde_json::json!([]);
        let corrupted = serde_json::to_string(&value).unwrap();

        let err = serde_json::from_str::<CompiledModel>(&corrupted).unwrap_err();
        // The `ValidationError::EmptyModel` Display flows through `Error::custom`.
        assert!(
            err.to_string().contains("no layers") || err.to_string().contains("empty"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn serialization_omits_derived_fields() {
        let value: serde_json::Value = serde_json::to_value(full_model()).unwrap();
        let object = value
            .as_object()
            .expect("CompiledModel serializes as a map");
        // Exactly the two wire fields, none of the derived ones.
        assert!(object.contains_key("spec"));
        assert!(object.contains_key("num_features"));
        assert!(!object.contains_key("allocations"));
        assert!(!object.contains_key("num_params"));
        assert!(!object.contains_key("resolved_readout"));
        assert_eq!(object.len(), 2);
    }

    #[test]
    fn trained_model_round_trips_model_and_theta() {
        let model = full_model();
        let theta: Vec<f64> = (0..model.num_params())
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let trained = TrainedModel {
            model,
            theta: theta.clone(),
        };
        let json = serde_json::to_string(&trained).unwrap();
        let loaded: TrainedModel = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.theta, theta);
        // The reloaded model still binds identically with the reloaded θ.
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(
            loaded.model.bind(&x, &loaded.theta).unwrap(),
            trained.model.bind(&x, &theta).unwrap()
        );
    }
}
