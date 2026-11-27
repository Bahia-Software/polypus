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

use std::fmt;
use std::sync::Arc;

use polypus_circuit::{CircuitError, GateInstruction, ParameterizedCircuit};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, Counts, Planner, QuantumBackend, RunParams,
};
use polypus_observable::{group_qubit_wise, BasisRotation, Grouping, ObservableError, PauliMask};
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
}

impl PauliSumOracleFactory {
    /// Check `template` and the Hamiltonian `terms` (`(coefficient, mask)`
    /// pairs, ungrouped), group the terms with [`group_qubit_wise`] over the
    /// template's width, and build each group's measurement template: the
    /// template, then the group's rotations in application order (`H` for
    /// `X`; `S†` then `H` for `Y`), then a measurement of every qubit.
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
        Ok(Self {
            templates,
            grouping,
        })
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
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        Box::new(PauliSumOracle {
            templates: self.templates,
            grouping: self.grouping,
            config: Arc::clone(&resources.config),
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            cancel: cancel.clone(),
            errors: errors.clone(),
        })
        .into()
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
}
