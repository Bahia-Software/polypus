//! The Pauli-sum oracle: trains a cost given as a sum of multi-qubit Pauli
//! strings, such as a molecular Hamiltonian (issue #531; design in
//! `docs/adr/0009-pauli-sum-oracle.md`).
//!
//! Hardware measures every qubit in the `Z` basis, so a term with `X` or `Y`
//! factors is measured after rotating those qubits. The Hamiltonian is split
//! into qubit-wise-commuting groups ([`group_qubit_wise`], #530); each group is
//! one circuit, the template followed by the group's rotations and a full
//! measurement, and each group's counts reduce to its contribution through the
//! group's [`ParityObservable`](polypus_observable::ParityObservable). A
//! candidate's energy is the identity terms' constant plus the groups'
//! contributions; the oracle returns it with its sign flipped, `−E`, because
//! the optimizers maximise.
//!
//! # Cost and noise
//!
//! A candidate is `G` circuits, one per group, each measured with the run's
//! shots (or `final_shots` for the final batch): `G · shots` shots in all.
//! A batch of `N` candidates is run in windows of `W` whole candidates,
//! `W` from [`candidate_window_size`] (as the QML oracle sizes its own): each
//! window is **one** [`Planner::execute`] call with `W · G` tasks,
//! candidate-major; `execute` returns one counts map per task, in input
//! order, however many waves it splits the window into, and the oracle pairs
//! them with their candidate and group itself (ADR 0009, decision 6, which
//! also records why this differs from the issue's suggested one call per
//! group). A window's circuits are dropped before the next is bound, so at
//! most `W · G` are alive at once, however large the population or the
//! Hamiltonian (issue #667).
//!
//! The windows change no expected value: the tasks keep their order across
//! windows and each candidate's energy is summed in the same order. On the
//! native backend the sampled values do not change either: it seeds each
//! circuit by its position in the run (`circuit_seed`, contract C-7), not by
//! the call it is in, so a seeded run gives the same values bit for bit
//! whatever the window. A backend seeded per call (Local/Aer, CUNQA) gives
//! each window's call its own stream, so there the noise of a seeded run
//! depends on the window, and so on the host's parallelism, as it does for
//! the QML oracle.
//!
//! The groups of one candidate usually share a backend call. Their noise is
//! independent where the backend gives each circuit of a call its own
//! sampling stream, as the native backend does (`circuit_seed`, contract
//! C-7). A third-party backend is only bound to one stream per call, so there
//! two groups of a candidate may share random numbers.
//!
//! # Gradient
//!
//! When every parameter of the template is a single-frequency coordinate
//! ([`ParameterizedCircuit::check_single_frequency`]: the angle `±θ_k + c`
//! of exactly one `rx`, `ry`, `rz`, `rxx`, `rzz`, `p`, `u1`, `u`, `u2` or
//! `u3`), the energy is one sinusoid in each parameter, and the oracle's
//! [`GradientOracle`] gives the exact parameter shift (issue #536; ADR 0009,
//! decision 8): `∂(−E)/∂θ_k = [f(θ + π/2·e_k) − f(θ − π/2·e_k)] / 2`, the
//! shifted points measured as ordinary candidates, with the run's shots. An
//! Adam iteration is then one batch of `2P + 1` candidates, `θ` and its
//! shifts, `(2P + 1) · G` circuits. Otherwise the factory provides no
//! gradient ([`PauliSumOracleFactory::gradient_unavailable`] says why) and
//! Adam differentiates by finite differences.

use std::fmt;
use std::sync::Arc;

use polypus_circuit::{CircuitError, GateInstruction, ParameterizedCircuit, SingleFrequencyError};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, Counts, Planner, QuantumBackend, RunParams,
};
use polypus_observable::{group_qubit_wise, BasisRotation, Grouping, ObservableError, PauliMask};
use polypus_optimizers::GradientOracle;
use polypus_orchestration::{OracleFactory, Oracles, Resources};

use crate::{
    candidate_window_size, CostObservable, EvaluationError, EvaluationOracle, OracleErrorSlot,
};

/// Why a [`PauliSumOracleFactory`] could not be built.
///
/// Every check runs in [`PauliSumOracleFactory::new`], because
/// [`OracleFactory::build`] cannot fail. A failure while evaluating is an
/// [`EvaluationError`] instead, recorded in the run's [`OracleErrorSlot`].
#[derive(Debug)]
#[non_exhaustive]
pub enum PauliSumOracleError {
    /// The template already measures: instruction `index` is a `Measure` or a
    /// `MeasureAll`. The basis rotations are appended after the template, and
    /// measurements must be terminal (contract C-4).
    MeasuringTemplate {
        /// Position of the first measurement in the template's gates.
        index: usize,
    },
    /// The Hamiltonian has no term besides the identity and zero-coefficient
    /// terms (or no term at all): there is nothing to measure, and its value
    /// would not depend on the candidate.
    NothingToMeasure,
    /// The Hamiltonian is invalid for the template: a term acting outside the
    /// template's qubits, a template wider than 64 qubits, a non-finite
    /// coefficient, or coefficients whose `Σ|c|` exceeds
    /// [`MAX_WEIGHT_BOUND`](polypus_observable::MAX_WEIGHT_BOUND).
    Hamiltonian(ObservableError),
    /// Appending a group's rotations or the measurement to the template
    /// failed. The checks above make this unreachable (the rotated qubits are
    /// within the template, and the template does not measure); it is
    /// reported rather than unwrapped.
    Circuit(CircuitError),
}

impl fmt::Display for PauliSumOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliSumOracleError::MeasuringTemplate { index } => write!(
                f,
                "the template must not measure: instruction {index} is a measurement, and the \
                 oracle appends each group's basis rotations and the measurement itself"
            ),
            PauliSumOracleError::NothingToMeasure => write!(
                f,
                "the Hamiltonian has no term besides the identity and zero-coefficient terms, \
                 so there is nothing to measure and its value does not depend on the parameters"
            ),
            PauliSumOracleError::Hamiltonian(e) => write!(f, "invalid Hamiltonian: {e}"),
            PauliSumOracleError::Circuit(e) => {
                write!(f, "could not build a measurement circuit: {e}")
            }
        }
    }
}

impl std::error::Error for PauliSumOracleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PauliSumOracleError::Hamiltonian(e) => Some(e),
            PauliSumOracleError::Circuit(e) => Some(e),
            PauliSumOracleError::MeasuringTemplate { .. }
            | PauliSumOracleError::NothingToMeasure => None,
        }
    }
}

/// Builds a [`PauliSumOracle`] for a training flow: the template and the
/// Hamiltonian, checked and grouped once in [`new`](Self::new), then wired to
/// the run's backend, planner, config, cancellation and error slot in
/// [`build`](OracleFactory::build).
#[derive(Debug, Clone)]
pub struct PauliSumOracleFactory {
    /// One measurement template per group, in [`Grouping::groups`] order.
    templates: Vec<ParameterizedCircuit>,
    grouping: Grouping,
    /// Why the template has no exact gradient, or `None` when it has one
    /// (see [`gradient_unavailable`](Self::gradient_unavailable)).
    gradient_unavailable: Option<SingleFrequencyError>,
}

impl PauliSumOracleFactory {
    /// Check `template` and the Hamiltonian `terms` (`(coefficient, mask)`
    /// pairs, ungrouped), group the terms with [`group_qubit_wise`] over the
    /// template's width, and build each group's measurement template: the
    /// template, then the group's rotations in application order (`H` for
    /// `X`; `S†` then `H` for `Y`), then a measurement of every qubit.
    ///
    /// It also decides whether the oracle has an exact gradient (ADR 0009,
    /// decision 8): when the template passes
    /// [`ParameterizedCircuit::check_single_frequency`], the run's
    /// [`Oracles`] carry the parameter-shift gradient of
    /// [`PauliSumOracle`]'s [`GradientOracle`] implementation; otherwise
    /// they carry none, Adam differentiates by finite differences, the reason
    /// is logged once with `log::warn!`, and
    /// [`gradient_unavailable`](Self::gradient_unavailable) returns it. That
    /// is not an error: the optimizers other than Adam need no gradient.
    ///
    /// # Errors
    ///
    /// [`PauliSumOracleError::MeasuringTemplate`] if the template measures,
    /// [`PauliSumOracleError::Hamiltonian`] if [`group_qubit_wise`] rejects
    /// the terms, and [`PauliSumOracleError::NothingToMeasure`] if no term
    /// remains besides the identity and zero-coefficient terms (which
    /// [`group_qubit_wise`] leaves out of every group).
    /// [`PauliSumOracleError::Circuit`] is unreachable (see its docs) and
    /// reported rather than unwrapped.
    pub fn new(
        template: ParameterizedCircuit,
        terms: &[(f64, PauliMask)],
    ) -> Result<Self, PauliSumOracleError> {
        if let Some(index) = template.gates().iter().position(|gate| {
            matches!(
                gate,
                GateInstruction::Measure { .. } | GateInstruction::MeasureAll
            )
        }) {
            return Err(PauliSumOracleError::MeasuringTemplate { index });
        }
        let grouping = group_qubit_wise(template.num_qubits(), terms)
            .map_err(PauliSumOracleError::Hamiltonian)?;
        if grouping.groups().is_empty() {
            return Err(PauliSumOracleError::NothingToMeasure);
        }
        let templates = grouping
            .groups()
            .iter()
            .map(|group| {
                let mut circuit = template.clone();
                for (qubit, rotation) in group.rotations() {
                    match rotation {
                        BasisRotation::H => circuit.try_push(GateInstruction::H(qubit))?,
                        BasisRotation::SdgThenH => {
                            circuit.try_push(GateInstruction::Sdg(qubit))?;
                            circuit.try_push(GateInstruction::H(qubit))?;
                        }
                    }
                }
                circuit.try_push(GateInstruction::MeasureAll)?;
                Ok(circuit)
            })
            .collect::<Result<Vec<_>, CircuitError>>()
            .map_err(PauliSumOracleError::Circuit)?;
        let gradient_unavailable = template.check_single_frequency().err();
        if let Some(reason) = &gradient_unavailable {
            log::warn!(
                "Pauli-sum cost: no exact parameter-shift gradient ({reason}); Adam will use \
                 finite differences"
            );
        }
        Ok(Self {
            templates,
            grouping,
            gradient_unavailable,
        })
    }

    /// Why the oracle has no exact gradient: the
    /// [`SingleFrequencyError`] naming the parameter and the gate that make
    /// the template fail [`ParameterizedCircuit::check_single_frequency`]
    /// (a parameter in two angles, in a controlled rotation, `u0` or a
    /// declared gate, in an expression such as `2·θ`, or in no angle).
    /// `None` when the oracle provides the exact parameter-shift gradient
    /// (ADR 0009, decision 8).
    pub fn gradient_unavailable(&self) -> Option<&SingleFrequencyError> {
        self.gradient_unavailable.as_ref()
    }

    /// The Hamiltonian's grouping: the identity terms' constant and the
    /// groups, each measured by one circuit per candidate.
    pub fn grouping(&self) -> &Grouping {
        &self.grouping
    }

    /// The number of circuits per candidate, one per group.
    pub fn num_groups(&self) -> usize {
        self.templates.len()
    }
}

impl OracleFactory for PauliSumOracleFactory {
    /// One [`PauliSumOracle`] as the run's evaluation oracle and, when the
    /// template has one, its exact gradient: contract C-5 requires the value
    /// [`GradientOracle::value_and_gradient`] returns to come from the
    /// optimizer's own evaluation oracle.
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        let has_gradient = self.gradient_unavailable.is_none();
        let oracle = Arc::new(PauliSumOracle {
            templates: self.templates,
            grouping: self.grouping,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            cancel: cancel.clone(),
            errors: errors.clone(),
        });
        Oracles {
            evaluation: Box::new(Arc::clone(&oracle)),
            gradient: has_gradient.then(|| Box::new(oracle) as Box<dyn GradientOracle>),
            full_evaluation: None,
        }
    }
}

/// Scores each candidate with `−E`, the Hamiltonian's estimated energy with
/// its sign flipped (see the [module docs](self)). Built by a
/// [`PauliSumOracleFactory`].
pub struct PauliSumOracle {
    /// One measurement template per group, in [`Grouping::groups`] order.
    templates: Vec<ParameterizedCircuit>,
    grouping: Grouping,
    config: Arc<RunParams>,
    backend: Arc<dyn QuantumBackend>,
    planner: Arc<dyn Planner>,
    cancel: CancelToken,
    /// The first evaluation failure is recorded here and surfaced by the entry
    /// point, since [`EvaluationOracle::evaluate_batch`] returns no `Result`.
    errors: OracleErrorSlot,
}

impl EvaluationOracle for PauliSumOracle {
    /// Every group of every candidate is measured with the run's shots.
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.evaluate(candidates, self.config.shots)
    }

    /// Every group of every candidate is measured with `shots` shots
    /// (`final_shots` of a training run). Without this override, the trait's
    /// default would score the final batch with the training's shots.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.evaluate(candidates, shots)
    }
}

impl PauliSumOracle {
    /// Both trait methods: one value per candidate, or finite sentinels once
    /// a failure has been recorded (then nothing runs, so a traced run makes
    /// no call the untraced one would not, contract C-12).
    fn evaluate(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates, shots) {
            Ok(values) => values,
            Err(e) => {
                self.errors.record(Box::new(e), &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }

    /// The fallible core: [`try_evaluate_windowed`](Self::try_evaluate_windowed)
    /// with this host's and backend's window.
    fn try_evaluate(
        &self,
        candidates: &[Vec<f64>],
        shots: u32,
    ) -> Result<Vec<f64>, EvaluationError> {
        let window = candidate_window_size(
            self.backend.capabilities().max_concurrency,
            self.templates.len(),
        );
        self.try_evaluate_windowed(candidates, window, shots)
    }

    /// Score `candidates` in windows of at most `window` whole candidates
    /// (issue #667; ADR 0009, decision 6): bind a window's `W · G` circuits
    /// candidate-major, run them in one [`Planner::execute`] call, reduce
    /// each candidate's `G` counts maps with its groups' observables (each
    /// map first checked to hold exactly `shots` shots, [`check_shot_total`]),
    /// and drop the window's circuits before binding the next. At most
    /// `window · G` bound circuits are alive at once, and a failing window
    /// stops the batch before the next is bound.
    ///
    /// The window is an argument so that tests can fix it; the values do not
    /// depend on it (see the [module docs](self)).
    fn try_evaluate_windowed(
        &self,
        candidates: &[Vec<f64>],
        window: usize,
        shots: u32,
    ) -> Result<Vec<f64>, EvaluationError> {
        let num_groups = self.templates.len();
        let mut values = Vec::with_capacity(candidates.len());
        // `chunks` panics on 0; `candidate_window_size` is never 0, but the
        // loop stays total. An empty batch has no window, so the planner is
        // not called with zero tasks.
        for (number, window_candidates) in candidates.chunks(window.max(1)).enumerate() {
            let first = number * window.max(1);
            // Task `i · G + g` is the window's candidate `i`'s circuit for
            // group `g`.
            let bound: Vec<BoundCircuit> = window_candidates
                .iter()
                .flat_map(|params| {
                    self.templates.iter().map(move |template| {
                        template
                            .assign_parameters(params)
                            .map(BoundCircuit::Native)
                            .map_err(EvaluationError::Binding)
                    })
                })
                .collect::<Result<_, _>>()?;
            let tasks: Vec<CircuitTask> = bound
                .iter()
                .map(|circuit| CircuitTask { circuit, shots })
                .collect();

            // `execute`, not `evaluate`: it returns one counts map per task,
            // in input order, whatever waves it runs (ADR 0009, decision 6).
            let counts =
                self.planner
                    .execute(self.backend.as_ref(), &tasks, &self.config, &self.cancel)?;
            if counts.len() != tasks.len() {
                return Err(EvaluationError::WrongLength {
                    expected: tasks.len(),
                    got: counts.len(),
                });
            }

            for (offset, candidate_counts) in counts.chunks(num_groups).enumerate() {
                let candidate = first + offset;
                // The constant, then the groups in order: a fixed summation
                // order, so that a seeded run is reproducible bit for bit.
                let mut energy = self.grouping.constant();
                for (index, (group, group_counts)) in self
                    .grouping
                    .groups()
                    .iter()
                    .zip(candidate_counts)
                    .enumerate()
                {
                    check_shot_total(group_counts, shots, candidate, index)?;
                    let reduced = group
                        .observable()
                        .expectation_batch(std::slice::from_ref(group_counts))?;
                    match reduced.as_slice() {
                        [value] => energy += value,
                        _ => {
                            return Err(EvaluationError::WrongLength {
                                expected: 1,
                                got: reduced.len(),
                            })
                        }
                    }
                }
                values.push(-energy);
            }
            // `bound` and `counts` drop here, before the next window is bound.
        }

        // Contract C-5: one finite value per candidate. `MAX_WEIGHT_BOUND`
        // keeps every value finite, so this check is defensive.
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(values)
    }
}

/// The exact parameter-shift gradient of ADR 0009, decision 8, for a
/// template that passes `check_single_frequency` (the factory provides it as
/// the run's gradient only then). Every component is of the fitness `−E`,
/// uphill (contract C-5), and every shifted candidate is measured like any
/// other, with the run's shots. A failure is recorded in the error slot and
/// returns finite zeros, as [`evaluate_batch`](EvaluationOracle::evaluate_batch)
/// does.
impl GradientOracle for PauliSumOracle {
    /// `∂fitness/∂θ[param_index]` from the two candidates `θ ± π/2·e_k`
    /// alone. An index outside `θ` is `0.0`, with nothing run.
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        if param_index >= theta.len() {
            return 0.0;
        }
        self.measured(1, || {
            shift_gradient(theta, &[param_index], false, |candidates| {
                self.try_evaluate(candidates, self.config.shots)
            })
        })
        .1
        .first()
        .copied()
        .unwrap_or(0.0)
    }

    /// Every component from the `2P` shifted candidates in one batch,
    /// `[θ+π/2·e_0, θ−π/2·e_0, θ+π/2·e_1, …]`, `P = θ.len()`; a `dims` other
    /// than `P` is left to the optimizer's length check, as `QmlOracle`
    /// does.
    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        let components: Vec<usize> = (0..theta.len()).collect();
        self.measured(dims, || {
            shift_gradient(theta, &components, false, |candidates| {
                self.try_evaluate(candidates, self.config.shots)
            })
        })
        .1
    }

    /// `θ`'s fitness and every component from one batch of `2P + 1`
    /// candidates, `[θ, θ+π/2·e_0, θ−π/2·e_0, …]`, so an Adam iteration is
    /// one call.
    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        let components: Vec<usize> = (0..theta.len()).collect();
        let (value, gradient) = self.measured(dims, || {
            shift_gradient(theta, &components, true, |candidates| {
                self.try_evaluate(candidates, self.config.shots)
            })
        });
        Some((value.unwrap_or(0.0), gradient))
    }

    /// `true`: [`value_and_gradient`](Self::value_and_gradient) always
    /// scores `θ` in the gradient's own batch.
    fn scores_theta(&self) -> bool {
        true
    }
}

impl PauliSumOracle {
    /// The sentinel discipline of [`evaluate`](Self::evaluate) for a
    /// gradient: nothing runs once a failure is recorded, and a new failure
    /// is recorded; either way the result is a zero value and `dims` zeros.
    /// `value_and_gradient` then still returns `Some`, as `scores_theta`
    /// promises, so a caller that counts evaluations counts one that did not
    /// run; the run stops at its next check of the error slot, as it does
    /// after `evaluate`'s sentinels.
    fn measured(
        &self,
        dims: usize,
        run: impl FnOnce() -> Result<(Option<f64>, Vec<f64>), EvaluationError>,
    ) -> (Option<f64>, Vec<f64>) {
        if self.errors.failed() {
            return (Some(0.0), vec![0.0; dims]);
        }
        match run() {
            Ok(measured) => measured,
            Err(e) => {
                self.errors.record(Box::new(e), &self.config.id);
                (Some(0.0), vec![0.0; dims])
            }
        }
    }
}

/// The parameter shift of ADR 0009, decision 8: scores `θ` (when
/// `with_theta`) and `θ ± π/2·e_k` for each `k` of `components`, in the
/// order `[θ, θ+π/2·e_k0, θ−π/2·e_k0, θ+π/2·e_k1, …]`, through `evaluate`,
/// and returns `θ`'s value and the components `(f₊ − f₋)/2`, in the order
/// of `components`. The oracle passes its own shot-based evaluation; the
/// tests pass an exact one, so that the criterion of #536 is checked on the
/// code the oracle runs. Exact when every parameter is a single-frequency
/// coordinate (`check_single_frequency`): the fitness is then one sinusoid
/// `a + b·cos θ_k + c·sin θ_k` in each of them.
fn shift_gradient(
    theta: &[f64],
    components: &[usize],
    with_theta: bool,
    evaluate: impl FnOnce(&[Vec<f64>]) -> Result<Vec<f64>, EvaluationError>,
) -> Result<(Option<f64>, Vec<f64>), EvaluationError> {
    const SHIFT: f64 = std::f64::consts::FRAC_PI_2;
    let mut candidates = Vec::with_capacity(usize::from(with_theta) + 2 * components.len());
    if with_theta {
        candidates.push(theta.to_vec());
    }
    for &k in components {
        for sign in [1.0, -1.0] {
            let mut shifted = theta.to_vec();
            if let Some(angle) = shifted.get_mut(k) {
                *angle += sign * SHIFT;
            }
            candidates.push(shifted);
        }
    }
    let values = evaluate(&candidates)?;
    if values.len() != candidates.len() {
        return Err(EvaluationError::WrongLength {
            expected: candidates.len(),
            got: values.len(),
        });
    }
    let (value, shifted) = if with_theta {
        (Some(values[0]), &values[1..])
    } else {
        (None, &values[..])
    };
    let gradient = shifted
        .chunks_exact(2)
        .map(|pair| (pair[0] - pair[1]) / 2.0)
        .collect();
    Ok((value, gradient))
}

/// Contract C-3: a circuit's counts total the shots it asked for. A map with
/// fewer (empty, say) or more would still reduce, to a value that is wrong
/// with no sign of it (an empty map reduces to 0), so it fails the
/// evaluation instead. The built-in planners already reject such a wave
/// (`validate_run_results`); this holds the oracle to C-3 under any other
/// [`Planner`] too (a Rust caller's own), which the `execute` contract does
/// not oblige to check the counts.
fn check_shot_total(
    counts: &Counts,
    shots: u32,
    candidate: usize,
    group: usize,
) -> Result<(), EvaluationError> {
    let total = counts
        .values()
        .try_fold(0u64, |total, &n| total.checked_add(n));
    if total == Some(u64::from(shots)) {
        return Ok(());
    }
    let total = total.map_or_else(|| "more than u64::MAX".to_string(), |t| t.to_string());
    Err(EvaluationError::Observable(ObservableError::Invalid(
        format!(
            "group {group} of candidate {candidate}: its counts total {total} shots, not the \
             {shots} requested (contract C-3)"
        ),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::{ConcreteCircuit, GateParam};
    use polypus_infrastructure::{
        BackendCapabilities, BackendError, InfrastructureError, NativeStatevectorBackend, OptLevel,
        PlannerRequirements, SequentialPlanner,
    };
    use polypus_observable::{Pauli, MAX_WEIGHT_BOUND};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// The coefficients of the test Hamiltonian's five terms, one per group:
    /// far apart, so that pairing a counts map with the wrong group changes
    /// the value.
    const COEFFICIENTS: [f64; 5] = [1.0, 10.0, 100.0, 1000.0, 10_000.0];

    /// `X0X1, X0Y1, Y0X1, Y0Y1, Z0Z1`: no two commute qubit-wise, so each is
    /// its own group, in this order.
    fn terms() -> Vec<(f64, PauliMask)> {
        let strings = [
            [Pauli::X, Pauli::X],
            [Pauli::X, Pauli::Y],
            [Pauli::Y, Pauli::X],
            [Pauli::Y, Pauli::Y],
            [Pauli::Z, Pauli::Z],
        ];
        strings
            .iter()
            .zip(COEFFICIENTS)
            .map(|(ops, c)| {
                (
                    c,
                    PauliMask::new([(0, ops[0]), (1, ops[1])]).expect("a valid string"),
                )
            })
            .collect()
    }

    /// The rotations each group's circuit must end with, before `MeasureAll`:
    /// `H` for `X`, `S†` then `H` for `Y`, nothing for `Z`.
    fn rotations(group: usize) -> Vec<GateInstruction> {
        use GateInstruction::{Sdg, H};
        match group {
            0 => vec![H(0), H(1)],
            1 => vec![H(0), Sdg(1), H(1)],
            2 => vec![Sdg(0), H(0), H(1)],
            3 => vec![Sdg(0), H(0), Sdg(1), H(1)],
            _ => vec![],
        }
    }

    /// One parameter on qubit 0, so the mock can read the candidate back.
    fn template() -> ParameterizedCircuit {
        ParameterizedCircuit::new(2).ry(0, GateParam::Param(0))
    }

    fn config() -> Arc<RunParams> {
        Arc::new(RunParams {
            id: "pauli-sum-oracle-test".to_string(),
            shots: 16,
            seed: Some(7),
            opt_level: OptLevel::default(),
        })
    }

    /// Candidate `i` is the angle `i`, read back by [`ContentBackend`].
    fn candidates(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64]).collect()
    }

    /// The sign [`ContentBackend`] gives candidate `i`'s group `g`. No row
    /// reads the same reversed or shifted, and the rows differ, so pairing a
    /// counts map with the wrong group or candidate changes some sign.
    fn sign(candidate: usize, group: usize) -> f64 {
        const SIGNS: [[f64; 5]; 2] = [[1.0, 1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1.0, -1.0]];
        SIGNS[candidate][group]
    }

    /// The value the oracle must return for candidate `i`: `−Σ_g c_g s(i, g)`
    /// (the test Hamiltonian has no constant).
    fn expected(candidate: usize) -> f64 {
        -COEFFICIENTS
            .iter()
            .enumerate()
            .map(|(g, c)| c * sign(candidate, g))
            .sum::<f64>()
    }

    /// A backend whose counts come from each circuit's **content**: the bound
    /// `ry` angle names the candidate, the gates after it name the group, and
    /// every shot reads `00` (parity `+1`) or `01` (parity `−1`) as
    /// [`sign`] says. It records each call's circuits and shots.
    struct ContentBackend {
        max_concurrency: usize,
        calls: Mutex<Vec<(u32, Vec<ConcreteCircuit>)>>,
        seeds: Mutex<Vec<Option<u64>>>,
    }

    impl ContentBackend {
        fn new(max_concurrency: usize) -> Self {
            Self {
                max_concurrency,
                calls: Mutex::new(Vec::new()),
                seeds: Mutex::new(Vec::new()),
            }
        }

        fn seeds(&self) -> Vec<Option<u64>> {
            self.seeds.lock().unwrap_or_else(|p| p.into_inner()).clone()
        }

        fn calls(&self) -> Vec<(u32, Vec<ConcreteCircuit>)> {
            self.calls.lock().unwrap_or_else(|p| p.into_inner()).clone()
        }
    }

    /// The candidate and group a circuit belongs to, read from its gates.
    fn identify(circuit: &ConcreteCircuit) -> (usize, usize) {
        let gates = circuit.gates();
        let candidate = match &gates[0] {
            GateInstruction::Ry {
                qubit: 0,
                theta: GateParam::Fixed(angle),
            } => *angle as usize,
            other => panic!("the template starts with ry(0): {other:?}"),
        };
        let after = &gates[1..gates.len() - 1];
        let group = (0..COEFFICIENTS.len())
            .find(|&g| after == rotations(g).as_slice())
            .unwrap_or_else(|| panic!("unknown rotations {after:?}"));
        (candidate, group)
    }

    impl QuantumBackend for ContentBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &RunParams,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            let circuits: Vec<ConcreteCircuit> = qcs
                .iter()
                .map(|qc| match qc {
                    BoundCircuit::Native(c) => c.clone(),
                    _ => panic!("the oracle binds native circuits"),
                })
                .collect();
            let counts = circuits
                .iter()
                .map(|c| {
                    let (candidate, group) = identify(c);
                    let key = if sign(candidate, group) > 0.0 {
                        "00"
                    } else {
                        "01"
                    };
                    HashMap::from([(key.to_string(), u64::from(config.shots))])
                })
                .collect();
            self.calls
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((config.shots, circuits));
            self.seeds
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(config.seed);
            Ok(counts)
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                max_concurrency: self.max_concurrency,
                supports_shot_distribution: true,
            }
        }
    }

    fn oracle(backend: Arc<dyn QuantumBackend>, planner: Arc<dyn Planner>) -> PauliSumOracle {
        let oracle = oracle_for(template(), &terms(), backend, planner);
        assert_eq!(oracle.templates.len(), 5);
        oracle
    }

    /// The oracle for `terms` over `template`, with [`config`].
    fn oracle_for(
        template: ParameterizedCircuit,
        terms: &[(f64, PauliMask)],
        backend: Arc<dyn QuantumBackend>,
        planner: Arc<dyn Planner>,
    ) -> PauliSumOracle {
        let factory = PauliSumOracleFactory::new(template, terms).expect("a valid cost");
        PauliSumOracle {
            templates: factory.templates,
            grouping: factory.grouping,
            config: config(),
            backend,
            planner,
            cancel: CancelToken::default(),
            errors: OracleErrorSlot::new(),
        }
    }

    /// ADR 0009, decision 6: one window's `execute` call (the default window
    /// holds both candidates: at least 6 with a backend of 3), split into
    /// waves of 3, still gives every candidate its own value. 2 candidates of
    /// 5 groups are 10 tasks, so waves of 3, 3, 3, 1, and candidate 0's last
    /// group shares a wave with candidate 1's first.
    #[test]
    fn every_candidate_gets_its_own_groups_across_waves() {
        let backend = Arc::new(ContentBackend::new(3));
        let oracle = oracle(backend.clone(), Arc::new(SequentialPlanner));
        let values = oracle.evaluate_batch(&candidates(2));
        assert!(!oracle.errors.failed());
        // Exact: sums of ±1, ±10, …, ±10⁴, all integers.
        assert_eq!(values, vec![expected(0), expected(1)]);
        assert_ne!(expected(0), expected(1));
        let waves: Vec<usize> = backend.calls().iter().map(|(_, c)| c.len()).collect();
        assert_eq!(waves, vec![3, 3, 3, 1]);
    }

    /// Each task is the bound template, then its group's rotations in order
    /// (`S†` before `H`), then `MeasureAll`, candidate-major and in
    /// [`Grouping::groups`] order.
    #[test]
    fn each_circuit_is_the_template_its_rotations_and_a_full_measurement() {
        let backend = Arc::new(ContentBackend::new(64));
        let oracle = oracle(backend.clone(), Arc::new(SequentialPlanner));
        oracle.evaluate_batch(&candidates(2));
        let circuits: Vec<ConcreteCircuit> = backend
            .calls()
            .into_iter()
            .flat_map(|(_, circuits)| circuits)
            .collect();
        assert_eq!(circuits.len(), 10);
        for (task, circuit) in circuits.iter().enumerate() {
            let (candidate, group) = (task / 5, task % 5);
            let mut want = vec![GateInstruction::Ry {
                qubit: 0,
                theta: GateParam::Fixed(candidate as f64),
            }];
            want.extend(rotations(group));
            want.push(GateInstruction::MeasureAll);
            assert_eq!(circuit.gates(), want.as_slice(), "task {task}");
        }
    }

    /// `evaluate_batch` measures every circuit with the run's shots, and
    /// `evaluate_batch_with_shots` (`final_shots`) with its own, in every
    /// wave, and both send the run's seed, which the backend derives each
    /// call's from (contract C-5, C-7).
    #[test]
    fn a_batch_with_shots_puts_them_in_every_task() {
        let backend = Arc::new(ContentBackend::new(3));
        let oracle = oracle(backend.clone(), Arc::new(SequentialPlanner));
        let trained = oracle.evaluate_batch(&candidates(2));
        let final_values = oracle.evaluate_batch_with_shots(&candidates(2), 4096);
        assert!(!oracle.errors.failed());
        let shots: Vec<u32> = backend.calls().iter().map(|&(s, _)| s).collect();
        assert_eq!(shots, vec![16, 16, 16, 16, 4096, 4096, 4096, 4096]);
        assert_eq!(backend.seeds(), vec![Some(7); 8]);
        // Exact: the mock's counts do not depend on the shots, and every
        // value is a sum of integers.
        assert_eq!(trained, final_values);
    }

    /// A backend that reads every shot as all zeros, so that each group's
    /// value is the sum of its coefficients.
    struct ZerosBackend;

    impl QuantumBackend for ZerosBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            config: &RunParams,
        ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
            Ok(qcs
                .iter()
                .map(|qc| {
                    let width = match qc {
                        BoundCircuit::Native(c) => c.num_qubits(),
                        _ => panic!("the oracle binds native circuits"),
                    };
                    HashMap::from([("0".repeat(width), u64::from(config.shots))])
                })
                .collect())
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                max_concurrency: 64,
                supports_shot_distribution: true,
            }
        }
    }

    /// ADR 0009, decision 6: the constant first, then the groups in order.
    /// With a constant of 1 and two groups of 2⁻⁵³ each, that order gives
    /// exactly 1 (each addition rounds back to 1, ties to even), and the
    /// groups first would give 1 + 2⁻⁵² (the two add exactly to 2⁻⁵²).
    #[test]
    fn the_constant_is_added_first_then_the_groups_in_order() {
        let tiny = 2f64.powi(-53);
        let terms = [
            (1.0, PauliMask::identity()),
            (
                tiny,
                PauliMask::new([(0, Pauli::Z)]).expect("a valid string"),
            ),
            (
                tiny,
                PauliMask::new([(0, Pauli::X)]).expect("a valid string"),
            ),
        ];
        let template = ParameterizedCircuit::new(1).ry(0, GateParam::Param(0));
        let oracle = oracle_for(
            template,
            &terms,
            Arc::new(ZerosBackend),
            Arc::new(SequentialPlanner),
        );
        assert_eq!(oracle.templates.len(), 2);
        let value = oracle.evaluate_batch(&[vec![0.0]])[0];
        assert!(!oracle.errors.failed());
        assert_eq!(value, -1.0);
        assert_ne!(value, -(1.0 + 2f64.powi(-52)));
    }

    /// A candidate with the wrong number of parameters fails to bind: the
    /// error is recorded, the batch gets one finite sentinel per candidate,
    /// and nothing runs.
    #[test]
    fn a_candidate_that_does_not_bind_is_recorded_and_nothing_runs() {
        let planner = Arc::new(ShortPlanner {
            calls: AtomicUsize::new(0),
        });
        let oracle = oracle(Arc::new(ContentBackend::new(3)), planner.clone());
        assert_eq!(oracle.evaluate_batch(&[vec![0.0, 1.0]]), vec![0.0]);
        assert_eq!(planner.calls.load(Ordering::SeqCst), 0);
        let error = oracle
            .errors
            .take()
            .expect("an error is recorded")
            .downcast::<EvaluationError>()
            .expect("an evaluation error");
        assert!(matches!(*error, EvaluationError::Binding(_)), "{error:?}");
    }

    /// The counts map of task `task` asked for `shots` shots.
    type MapsOf = fn(usize, usize, u64) -> Counts;

    /// A planner whose counts maps come from `maps(call, task, shots)`:
    /// `call` counts its `execute` calls from 0, `task` the tasks of a call.
    struct MapsPlanner {
        maps: MapsOf,
        calls: AtomicUsize,
    }

    impl MapsPlanner {
        fn new(maps: MapsOf) -> Self {
            Self {
                maps,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Planner for MapsPlanner {
        fn requirements(&self) -> PlannerRequirements {
            PlannerRequirements {
                needs_shot_distribution: false,
                min_concurrency: 1,
            }
        }

        fn execute(
            &self,
            _backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            _params: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(tasks
                .iter()
                .enumerate()
                .map(|(task, t)| (self.maps)(call, task, u64::from(t.shots)))
                .collect())
        }
    }

    /// Contract C-3: every group's counts must total the shots. Task 7 is
    /// candidate 1's group 2 (5 groups per candidate); a map there that is
    /// empty, one short, one over or overflowing is recorded as an error
    /// naming it, with finite sentinels, instead of biasing the energy.
    #[test]
    fn counts_that_do_not_total_the_shots_are_recorded() {
        let cases: [(MapsOf, &str); 4] = [
            (
                |_, task, shots| match task {
                    7 => HashMap::new(),
                    _ => HashMap::from([("00".to_string(), shots)]),
                },
                "total 0 shots, not the 16",
            ),
            (
                |_, task, shots| match task {
                    7 => HashMap::from([("00".to_string(), shots - 1)]),
                    _ => HashMap::from([("00".to_string(), shots)]),
                },
                "total 15 shots, not the 16",
            ),
            (
                |_, task, shots| match task {
                    7 => HashMap::from([("00".to_string(), shots), ("01".to_string(), 1)]),
                    _ => HashMap::from([("00".to_string(), shots)]),
                },
                "total 17 shots, not the 16",
            ),
            (
                |_, task, shots| match task {
                    7 => {
                        HashMap::from([("00".to_string(), u64::MAX), ("01".to_string(), u64::MAX)])
                    }
                    _ => HashMap::from([("00".to_string(), shots)]),
                },
                "total more than u64::MAX shots",
            ),
        ];
        for (maps, message) in cases {
            let oracle = oracle(
                Arc::new(ContentBackend::new(3)),
                Arc::new(MapsPlanner::new(maps)),
            );
            assert_eq!(oracle.evaluate_batch(&candidates(2)), vec![0.0, 0.0]);
            let error = oracle
                .errors
                .take()
                .expect("an error is recorded")
                .downcast::<EvaluationError>()
                .expect("an evaluation error");
            match *error {
                EvaluationError::Observable(ObservableError::Invalid(ref text)) => {
                    assert!(text.contains("group 2 of candidate 1"), "{text}");
                    assert!(text.contains(message), "{text}");
                }
                ref other => panic!("{other:?}"),
            }
        }
        // Maps that total the shots, here split over two keys, pass.
        let split = oracle(
            Arc::new(ContentBackend::new(3)),
            Arc::new(MapsPlanner::new(|_, _, shots| {
                HashMap::from([("00".to_string(), shots - 4), ("01".to_string(), 4)])
            })),
        );
        assert_eq!(split.evaluate_batch(&candidates(2)).len(), 2);
        assert!(!split.errors.failed());
        // In windows of one candidate, the second call's group 2 is
        // candidate 1's: the message names the candidate in the batch, not
        // in its window.
        let windowed = oracle(
            Arc::new(ContentBackend::new(3)),
            Arc::new(MapsPlanner::new(|call, task, shots| match (call, task) {
                (1, 2) => HashMap::new(),
                _ => HashMap::from([("00".to_string(), shots)]),
            })),
        );
        match windowed.try_evaluate_windowed(&candidates(2), 1, 16) {
            Err(EvaluationError::Observable(ObservableError::Invalid(text))) => {
                assert!(text.contains("group 2 of candidate 1:"), "{text}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A planner that runs like [`SequentialPlanner`] and records how many
    /// tasks each `execute` call carried.
    #[derive(Default)]
    struct RecordingPlanner {
        tasks_per_call: Mutex<Vec<usize>>,
    }

    impl RecordingPlanner {
        fn tasks_per_call(&self) -> Vec<usize> {
            self.tasks_per_call
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }
    }

    impl Planner for RecordingPlanner {
        fn requirements(&self) -> PlannerRequirements {
            SequentialPlanner.requirements()
        }

        fn execute(
            &self,
            backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            params: &RunParams,
            cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            self.tasks_per_call
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(tasks.len());
            SequentialPlanner.execute(backend, tasks, params, cancel)
        }
    }

    /// Issue #667: the windows bound each call to `W · G` tasks and change
    /// no value. On the native backend, which seeds each circuit by its
    /// position in the run (C-7), five noisy candidates give the same values
    /// bit for bit in windows of 1, 2, 3 or all five, with a fresh backend
    /// of the same seed each time.
    #[test]
    fn windows_bound_each_call_and_change_no_value_bit_for_bit() {
        let candidates: Vec<Vec<f64>> = [0.3, 1.1, 2.0, -0.7, 2.9]
            .iter()
            .map(|&theta| vec![theta])
            .collect();
        let run = |seed: u64, window: usize| {
            let planner = Arc::new(RecordingPlanner::default());
            let oracle = oracle(
                Arc::new(NativeStatevectorBackend::new(seed)),
                planner.clone(),
            );
            let values = oracle
                .try_evaluate_windowed(&candidates, window, 16)
                .expect("a valid batch");
            (values, planner.tasks_per_call())
        };
        let (all, calls) = run(667, 5);
        assert_eq!(calls, vec![25]);
        // 16 shots per group: the values are noisy (another seed changes
        // them), so the equal values below come from equal noise.
        assert_ne!(run(668, 5).0, all, "the native backend samples");
        for (window, expected_calls) in [
            (1, vec![5, 5, 5, 5, 5]),
            (2, vec![10, 10, 5]),
            (3, vec![15, 10]),
        ] {
            let (values, calls) = run(667, window);
            assert_eq!(calls, expected_calls, "window {window}");
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "window {window}"
            );
        }
    }

    /// A failing window stops the batch: the next window is never bound or
    /// run, and the error is recorded as for an unwindowed batch.
    #[test]
    fn a_failing_window_stops_the_batch() {
        let planner = Arc::new(ShortPlanner {
            calls: AtomicUsize::new(0),
        });
        let oracle = oracle(Arc::new(ContentBackend::new(3)), planner.clone());
        assert!(matches!(
            oracle.try_evaluate_windowed(&candidates(2), 1, 16),
            Err(EvaluationError::WrongLength {
                expected: 5,
                got: 4
            })
        ));
        assert_eq!(planner.calls.load(Ordering::SeqCst), 1);
    }

    /// An empty batch is an empty result, and the planner is not called:
    /// there is nothing to run.
    #[test]
    fn an_empty_batch_is_an_empty_result_and_runs_nothing() {
        let planner = Arc::new(ShortPlanner {
            calls: AtomicUsize::new(0),
        });
        let oracle = oracle(Arc::new(ContentBackend::new(3)), planner.clone());
        assert!(oracle.evaluate_batch(&[]).is_empty());
        assert!(oracle.evaluate_batch_with_shots(&[], 4096).is_empty());
        assert_eq!(planner.calls.load(Ordering::SeqCst), 0);
        assert!(!oracle.errors.failed());
    }

    /// A planner that returns one counts map too few, and counts its calls.
    struct ShortPlanner {
        calls: AtomicUsize,
    }

    impl Planner for ShortPlanner {
        fn requirements(&self) -> PlannerRequirements {
            PlannerRequirements {
                needs_shot_distribution: false,
                min_concurrency: 1,
            }
        }

        fn execute(
            &self,
            _backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            _params: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![
                HashMap::from([("00".to_string(), 16)]);
                tasks.len().saturating_sub(1)
            ])
        }
    }

    /// A wrong number of counts maps is `WrongLength`, recorded with one
    /// finite sentinel per candidate; later batches then run nothing.
    #[test]
    fn a_short_result_is_recorded_and_later_batches_run_nothing() {
        let planner = Arc::new(ShortPlanner {
            calls: AtomicUsize::new(0),
        });
        let oracle = oracle(Arc::new(ContentBackend::new(3)), planner.clone());
        assert_eq!(oracle.evaluate_batch(&candidates(2)), vec![0.0, 0.0]);
        // A later batch runs nothing: the planner is not called again.
        assert_eq!(oracle.evaluate_batch(&candidates(2)), vec![0.0, 0.0]);
        assert_eq!(planner.calls.load(Ordering::SeqCst), 1);
        assert!(oracle.errors.failed());
        let recorded = oracle.errors.take().expect("an error is recorded");
        let error = recorded
            .downcast::<EvaluationError>()
            .expect("an evaluation error");
        assert!(
            matches!(
                *error,
                EvaluationError::WrongLength {
                    expected: 10,
                    got: 9
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn a_measuring_template_is_rejected() {
        for template in [
            ParameterizedCircuit::new(2)
                .ry(0, GateParam::Param(0))
                .measure(0, 0),
            ParameterizedCircuit::new(2)
                .ry(0, GateParam::Param(0))
                .measure_all(),
        ] {
            let error = PauliSumOracleFactory::new(template, &terms()).unwrap_err();
            assert!(
                matches!(error, PauliSumOracleError::MeasuringTemplate { index: 1 }),
                "{error:?}"
            );
            assert!(error.to_string().contains("must not measure"), "{error}");
        }
    }

    #[test]
    fn a_hamiltonian_without_a_non_identity_term_is_rejected() {
        let identity = PauliMask::identity();
        let z0 = PauliMask::new([(0, Pauli::Z)]).expect("a valid string");
        for terms in [
            vec![],
            vec![(0.5, identity)],
            vec![(0.5, identity), (-1.0, identity)],
            // Zero terms are in no group (#667), so nothing is left to measure.
            vec![(0.5, identity), (0.0, z0), (-0.0, z0)],
        ] {
            let error = PauliSumOracleFactory::new(template(), &terms).unwrap_err();
            assert!(
                matches!(error, PauliSumOracleError::NothingToMeasure),
                "{error:?}"
            );
        }
    }

    /// Every check of `group_qubit_wise` reaches the caller as
    /// `Hamiltonian`, with its message.
    #[test]
    fn an_invalid_hamiltonian_is_rejected() {
        let z = |qubit| PauliMask::new([(qubit, Pauli::Z)]).expect("a valid string");
        /// A template, a Hamiltonian, and a piece of the expected message.
        type Case = (ParameterizedCircuit, Vec<(f64, PauliMask)>, &'static str);
        let cases: Vec<Case> = vec![
            (template(), vec![(1.0, z(2))], "acts outside 2 qubits"),
            (template(), vec![(f64::NAN, z(0))], "not finite"),
            (template(), vec![(f64::INFINITY, z(0))], "not finite"),
            (
                template(),
                vec![(MAX_WEIGHT_BOUND, z(0)), (MAX_WEIGHT_BOUND, z(1))],
                "exceeds",
            ),
            (ParameterizedCircuit::new(65), vec![(1.0, z(0))], "64"),
        ];
        for (template, terms, message) in cases {
            let error = PauliSumOracleFactory::new(template, &terms).unwrap_err();
            assert!(
                matches!(error, PauliSumOracleError::Hamiltonian(_)),
                "{error:?}"
            );
            assert!(error.to_string().contains(message), "{error}");
            assert!(std::error::Error::source(&error).is_some());
        }
    }

    // ---- Exact parameter-shift gradient (#536, ADR 0009 decision 8) ----

    /// The fitness `−E` of each candidate, exactly: each group's measurement
    /// template bound and run on a statevector (its `measure_all` reads
    /// nothing), its probabilities reduced by the group's
    /// `ParityObservable`, plus the constant. The exact path of #534 will
    /// use the same pieces; here it stands in for the oracle's shot-based
    /// evaluation in [`shift_gradient`].
    fn exact_fitness(
        factory: &PauliSumOracleFactory,
        candidates: &[Vec<f64>],
    ) -> Result<Vec<f64>, EvaluationError> {
        candidates
            .iter()
            .map(|params| {
                let mut energy = factory.grouping.constant();
                for (group, template) in factory.grouping.groups().iter().zip(&factory.templates) {
                    let circuit = template
                        .assign_parameters(params)
                        .map_err(EvaluationError::Binding)?;
                    let mut state =
                        polypus_sim::Statevector::new(circuit.num_qubits()).expect("a small state");
                    for gate in circuit.gates() {
                        state.apply(gate).expect("a simulable gate");
                    }
                    let probabilities: Vec<f64> =
                        state.amplitudes().iter().map(|a| a.norm_sqr()).collect();
                    energy += group
                        .observable()
                        .expectation_from_probabilities(&probabilities)?;
                }
                Ok(-energy)
            })
            .collect()
    }

    /// The 4-qubit Jordan–Wigner H2 at 0.7414 Å as `(coefficient, mask)`
    /// terms, and its dense matrix (as in `tests/pauli_sum_h2.rs`).
    fn h2() -> (
        Vec<(f64, PauliMask)>,
        polypus_physics::hamiltonians::DenseMatrix,
    ) {
        use polypus_physics::chemistry::{H2Encoding, H2Molecule};
        use polypus_physics::hamiltonians::{PauliAxis, PhysicsHamiltonian};
        let sum = H2Molecule::new(0.7414, H2Encoding::JordanWigner)
            .expect("a valid bond length")
            .to_pauli_sum();
        let terms = sum
            .terms()
            .iter()
            .map(|term| {
                let ops = term.string.ops().iter().map(|&(qubit, axis)| {
                    let pauli = match axis {
                        PauliAxis::X => Pauli::X,
                        PauliAxis::Y => Pauli::Y,
                        PauliAxis::Z => Pauli::Z,
                    };
                    (qubit, pauli)
                });
                (
                    term.coefficient,
                    PauliMask::new(ops).expect("a valid string"),
                )
            })
            .collect();
        (terms, sum.to_dense_real().expect("a 4-qubit matrix"))
    }

    /// The H2 ansatz of ADR 0009, decision 7:
    /// `cos(θ/2)|1010⟩ + sin(θ/2)|0101⟩`.
    fn h2_ansatz() -> ParameterizedCircuit {
        ParameterizedCircuit::new(4)
            .ry(0, GateParam::Param(0))
            .cx(0, 2)
            .x(1)
            .cx(0, 1)
            .cx(1, 3)
    }

    /// #536's first criterion: on the exact path, the oracle's parameter
    /// shift gives `−E(θ)` and `−dE/dθ` of the H2 ansatz's analytic energy
    /// `E(θ) = (a + b)/2 + (a − b)/2·cos θ + h·sin θ` within 1e-10, at several
    /// `θ`, the optimum (where the gradient vanishes) included.
    #[test]
    fn the_shift_gradient_is_the_analytic_derivative_on_the_exact_path() {
        let (terms, dense) = h2();
        let get = |row, col| dense.get(row, col).expect("inside the matrix");
        let (a, b, h) = (
            get(0b1010, 0b1010),
            get(0b0101, 0b0101),
            get(0b0101, 0b1010),
        );
        let energy = |t: f64| (a + b) / 2.0 + (a - b) / 2.0 * t.cos() + h * t.sin();
        let derivative = |t: f64| -(a - b) / 2.0 * t.sin() + h * t.cos();
        let optimum = (-h).atan2(-(a - b) / 2.0);
        let factory = PauliSumOracleFactory::new(h2_ansatz(), &terms).expect("a valid cost");
        assert!(factory.gradient_unavailable().is_none());
        for theta in [-2.0, -0.3, 0.9, 2.5, optimum] {
            let (value, gradient) = shift_gradient(&[theta], &[0], true, |candidates| {
                exact_fitness(&factory, candidates)
            })
            .expect("an exact evaluation");
            let value = value.expect("θ is scored");
            assert!(
                (value + energy(theta)).abs() <= 1e-10,
                "θ = {theta}: value {value}, −E {}",
                -energy(theta)
            );
            assert_eq!(gradient.len(), 1);
            assert!(
                (gradient[0] + derivative(theta)).abs() <= 1e-10,
                "θ = {theta}: gradient {}, −dE/dθ {}",
                gradient[0],
                -derivative(theta)
            );
        }
        assert!(derivative(optimum).abs() <= 1e-12);
    }

    /// Component `k` is `−∂E/∂θ_k`: on `rx(θ₀)` ⊗ `ry(θ₁)` with
    /// `H = 0.5·Z₀ + 2·X₁ + 3·Z₀X₁`,
    /// `E = 0.5·cos θ₀ + 2·sin θ₁ + 3·cos θ₀·sin θ₁`, exactly, at random
    /// points; without `θ`, only the shifted candidates are scored.
    #[test]
    fn each_component_is_the_derivative_in_its_own_parameter() {
        let (factory, energy, gradient) = two_parameter_cost();
        let mut rng = polypus_circuit::rng::SplitMix64::new(536);
        for _ in 0..10 {
            let theta = [(rng.next_f64() - 0.5) * 8.0, (rng.next_f64() - 0.5) * 8.0];
            let (value, measured) = shift_gradient(&theta, &[0, 1], true, |candidates| {
                exact_fitness(&factory, candidates)
            })
            .expect("an exact evaluation");
            assert!((value.expect("θ is scored") + energy(theta)).abs() <= 1e-10);
            let want = gradient(theta);
            for k in 0..2 {
                assert!(
                    (measured[k] + want[k]).abs() <= 1e-10,
                    "θ = {theta:?}, component {k}: {} vs {}",
                    measured[k],
                    -want[k]
                );
            }
            // One component alone, without θ: the same value, two candidates.
            let mut asked = Vec::new();
            let (value, alone) = shift_gradient(&theta, &[1], false, |candidates| {
                asked = candidates.to_vec();
                exact_fitness(&factory, candidates)
            })
            .expect("an exact evaluation");
            assert_eq!(value, None);
            let half_pi = std::f64::consts::FRAC_PI_2;
            assert_eq!(
                asked,
                vec![
                    vec![theta[0], theta[1] + half_pi],
                    vec![theta[0], theta[1] - half_pi]
                ]
            );
            assert!((alone[0] + want[1]).abs() <= 1e-10);
        }
    }

    /// Every kind of angle `check_single_frequency` accepts beyond `rx` and
    /// `ry` has frequency 1, so the shift is exact for it too: `u3`'s `θ`,
    /// `u2`'s `φ`, `rzz`, `rz` and `p` of `−θ + 0.4`, `rxx` and `u1`, in a
    /// circuit whose terms read X and Y, against a five-point derivative of
    /// the exact fitness (error about 1e-12). A gate let in with another
    /// frequency would break the shift here.
    #[test]
    fn the_shift_is_exact_for_every_single_frequency_gate_kind() {
        use polypus_circuit::ParamExpr;
        let mut template = ParameterizedCircuit::new(2);
        let turned = template
            .add_expr(-ParamExpr::param(4) + 0.4)
            .expect("a valid expression");
        let template = template
            .u(0, GateParam::Param(0), 0.3, -0.2)
            .u2(1, GateParam::Param(1), 0.5)
            .rzz(0, 1, GateParam::Param(2))
            .rz(0, GateParam::Param(3))
            .p(1, turned)
            .rxx(0, 1, GateParam::Param(5))
            .u1(0, GateParam::Param(6))
            .h(1);
        let mask = |ops: &[(usize, Pauli)]| PauliMask::new(ops.iter().copied()).expect("valid");
        let terms = [
            (0.7, mask(&[(0, Pauli::X)])),
            (-1.3, mask(&[(1, Pauli::Y)])),
            (0.4, mask(&[(0, Pauli::X), (1, Pauli::X)])),
            (0.9, mask(&[(0, Pauli::Z), (1, Pauli::Y)])),
            (0.25, mask(&[(0, Pauli::Y), (1, Pauli::Z)])),
        ];
        let factory = PauliSumOracleFactory::new(template, &terms).expect("a valid cost");
        assert!(factory.gradient_unavailable().is_none());
        let fitness = |theta: &[f64]| exact_fitness(&factory, &[theta.to_vec()]).unwrap()[0];
        let mut rng = polypus_circuit::rng::SplitMix64::new(5360);
        let h = 1e-3;
        for _ in 0..5 {
            let theta: Vec<f64> = (0..7).map(|_| (rng.next_f64() - 0.5) * 6.0).collect();
            let components: Vec<usize> = (0..7).collect();
            let (_, shifted) = shift_gradient(&theta, &components, false, |candidates| {
                exact_fitness(&factory, candidates)
            })
            .expect("an exact evaluation");
            for k in 0..7 {
                let at = |step: f64| {
                    let mut point = theta.clone();
                    point[k] += step;
                    fitness(&point)
                };
                let reference =
                    (-at(2.0 * h) + 8.0 * at(h) - 8.0 * at(-h) + at(-2.0 * h)) / (12.0 * h);
                assert!(
                    (shifted[k] - reference).abs() <= 1e-9,
                    "parameter {k} at {theta:?}: shift {} vs five-point {reference}",
                    shifted[k]
                );
            }
            // Not trivially zero: the shift sees every parameter.
            assert!(shifted.iter().any(|g| g.abs() > 1e-3), "{shifted:?}");
        }
    }

    type Energy = fn([f64; 2]) -> f64;
    type Gradient = fn([f64; 2]) -> [f64; 2];

    /// The cost of [`each_component_is_the_derivative_in_its_own_parameter`]:
    /// its factory (one group: `Z₀`, `X₁` and `Z₀X₁` commute qubit-wise),
    /// `E(θ)` and `∇E(θ)`.
    fn two_parameter_cost() -> (PauliSumOracleFactory, Energy, Gradient) {
        let z0 = PauliMask::new([(0, Pauli::Z)]).expect("a valid string");
        let x1 = PauliMask::new([(1, Pauli::X)]).expect("a valid string");
        let z0x1 = PauliMask::new([(0, Pauli::Z), (1, Pauli::X)]).expect("a valid string");
        let template = ParameterizedCircuit::new(2)
            .rx(0, GateParam::Param(0))
            .ry(1, GateParam::Param(1));
        let factory = PauliSumOracleFactory::new(template, &[(0.5, z0), (2.0, x1), (3.0, z0x1)])
            .expect("a valid cost");
        assert_eq!(factory.num_groups(), 1);
        assert!(factory.gradient_unavailable().is_none());
        (
            factory,
            |t| 0.5 * t[0].cos() + 2.0 * t[1].sin() + 3.0 * t[0].cos() * t[1].sin(),
            |t| {
                [
                    -0.5 * t[0].sin() - 3.0 * t[0].sin() * t[1].sin(),
                    2.0 * t[1].cos() + 3.0 * t[0].cos() * t[1].cos(),
                ]
            },
        )
    }

    /// The two-parameter cost's oracle on the native backend, behind a
    /// planner that records the tasks of each call, with its gradient.
    fn two_parameter_oracles(planner: Arc<RecordingPlanner>, errors: &OracleErrorSlot) -> Oracles {
        let resources = Resources::new(
            Arc::new(NativeStatevectorBackend::new(536)),
            Some(planner),
            config(),
        )
        .expect("the planner suits the backend");
        two_parameter_cost()
            .0
            .build(&resources, &CancelToken::default(), errors)
    }

    /// With shots, each gradient method is one batch of the candidates
    /// ADR 0009 decision 8 lists: `value_and_gradient` scores `2P + 1`,
    /// `gradient_batch` `2P`, `gradient` 2 and an index outside `θ` none;
    /// `scores_theta` is `true`; and the evaluation and the gradient are one
    /// oracle (the value is the evaluation oracle's, C-5). The tasks are
    /// counted over every call of a method, since a batch is one call per
    /// window and the window depends on the host's parallelism (#667).
    #[test]
    fn each_gradient_call_is_one_batch_of_its_shifted_candidates() {
        let planner = Arc::new(RecordingPlanner::default());
        let errors = OracleErrorSlot::new();
        let oracles = two_parameter_oracles(planner.clone(), &errors);
        let gradient = oracles.gradient.expect("a single-frequency template");
        assert!(gradient.scores_theta());
        let theta = [0.4, -1.1];
        // The tasks the planner received since `before` calls, over all of them.
        let tasks_since =
            |before: usize| -> usize { planner.tasks_per_call()[before..].iter().sum() };
        let calls = || planner.tasks_per_call().len();
        // One group, so one task per candidate.
        let before = calls();
        let (value, components) = gradient.value_and_gradient(&theta, 2).expect("θ is scored");
        assert_eq!(components.len(), 2);
        assert_eq!(tasks_since(before), 5);
        let before = calls();
        assert_eq!(gradient.gradient_batch(&theta, 2).len(), 2);
        assert_eq!(tasks_since(before), 4);
        let before = calls();
        let _ = gradient.gradient(&theta, 1);
        assert_eq!(tasks_since(before), 2);
        let before = calls();
        assert_eq!(gradient.gradient(&theta, 2), 0.0);
        assert_eq!(calls(), before, "an index outside θ runs nothing");
        assert!(!errors.failed());
        // Finite and noisy: 16 shots of a cost of norm 5.5.
        assert!(value.is_finite() && value.abs() <= 5.5);
        assert!(components.iter().all(|c| c.is_finite() && c.abs() <= 5.5));
        // The same seed on a fresh backend: the evaluation oracle scores θ as
        // the gradient's first candidate did, bit for bit (C-7: the native
        // backend seeds each circuit by its position in the run).
        let fresh = two_parameter_oracles(
            Arc::new(RecordingPlanner::default()),
            &OracleErrorSlot::new(),
        );
        assert_eq!(
            fresh.evaluation.evaluate_batch(&[theta.to_vec()]),
            vec![value]
        );
    }

    /// A captured `WARN` line, for the one test that reads the log.
    struct Warnings(Mutex<Vec<String>>);

    impl log::Log for Warnings {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Warn
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                self.0
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(record.args().to_string());
            }
        }

        fn flush(&self) {}
    }

    static WARNINGS: Warnings = Warnings(Mutex::new(Vec::new()));

    /// A template that fails `check_single_frequency` gets no gradient: the
    /// factory says why and logs it, and the run's `Oracles` carry no
    /// gradient, so Adam takes the finite difference. The other optimizers
    /// still evaluate it.
    #[test]
    fn a_template_without_single_frequency_parameters_has_no_gradient() {
        // The only test of this binary that installs a logger.
        log::set_logger(&WARNINGS).expect("no other logger in this test binary");
        log::set_max_level(log::LevelFilter::Warn);
        let z0 = PauliMask::new([(0, Pauli::Z)]).expect("a valid string");
        // θ₀ in two angles.
        let repeated = ParameterizedCircuit::new(2)
            .ry(0, GateParam::Param(0))
            .ry(1, GateParam::Param(0));
        let factory = PauliSumOracleFactory::new(repeated, &[(1.0, z0)]).expect("a valid cost");
        let reason = factory
            .gradient_unavailable()
            .expect("a repeated parameter")
            .clone();
        assert!(
            matches!(reason, SingleFrequencyError::Repeated { param: 0, .. }),
            "{reason:?}"
        );
        let logged = WARNINGS.0.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert!(
            logged
                .iter()
                .any(|line| line.contains("no exact parameter-shift gradient")
                    && line.contains(&reason.to_string())),
            "{logged:?}"
        );
        let resources =
            Resources::new(Arc::new(NativeStatevectorBackend::new(536)), None, config())
                .expect("the default planner");
        let oracles = factory.build(&resources, &CancelToken::default(), &OracleErrorSlot::new());
        assert!(oracles.gradient.is_none());
        assert_eq!(oracles.evaluation.evaluate_batch(&[vec![0.3]]).len(), 1);
    }

    /// A failure while computing a gradient is recorded, with finite zeros;
    /// afterwards nothing runs, for the gradient or the evaluation.
    #[test]
    fn a_gradient_failure_is_recorded_and_later_calls_run_nothing() {
        let planner = Arc::new(RecordingPlanner::default());
        let errors = OracleErrorSlot::new();
        let oracles = two_parameter_oracles(planner.clone(), &errors);
        let gradient = oracles.gradient.expect("a single-frequency template");
        // Three parameters for a two-parameter template: binding fails.
        assert_eq!(
            gradient.value_and_gradient(&[0.1, 0.2, 0.3], 3),
            Some((0.0, vec![0.0; 3]))
        );
        assert!(errors.failed());
        assert_eq!(gradient.gradient_batch(&[0.1, 0.2], 2), vec![0.0; 2]);
        assert_eq!(gradient.gradient(&[0.1, 0.2], 0), 0.0);
        assert_eq!(
            oracles.evaluation.evaluate_batch(&[vec![0.1, 0.2]]),
            vec![0.0]
        );
        assert!(planner.tasks_per_call().is_empty());
        let error = errors
            .take()
            .expect("an error is recorded")
            .downcast::<EvaluationError>()
            .expect("an evaluation error");
        assert!(matches!(*error, EvaluationError::Binding(_)), "{error:?}");
    }
}
