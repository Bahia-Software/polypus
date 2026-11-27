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
//! A batch of `N` candidates is **one** [`Planner::execute`] call with `N · G`
//! tasks, candidate-major; `execute` returns one counts map per task, in
//! input order, however many waves it splits the batch into, and the oracle
//! pairs them with their candidate and group itself (ADR 0009, decision 6,
//! which also records why this differs from the issue's suggested one call
//! per group).
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
    BoundCircuit, CancelToken, CircuitTask, Planner, QuantumBackend, RunParams,
};
use polypus_observable::{group_qubit_wise, BasisRotation, Grouping, ObservableError, PauliMask};
use polypus_orchestration::{OracleFactory, Oracles, Resources};

use crate::{CostObservable, EvaluationError, EvaluationOracle, OracleErrorSlot};

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
    /// The Hamiltonian has no term besides the identity (or no term at all):
    /// there is nothing to measure, and its value would not depend on the
    /// candidate.
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
                "the Hamiltonian has no term besides the identity, so there is nothing to \
                 measure and its value does not depend on the parameters"
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
    /// remains besides the identity. [`PauliSumOracleError::Circuit`] is
    /// unreachable (see its docs) and reported rather than unwrapped.
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

    /// The fallible core: bind the `N · G` circuits candidate-major, run them
    /// in one [`Planner::execute`] call, and reduce each candidate's `G`
    /// counts maps with its groups' observables.
    fn try_evaluate(
        &self,
        candidates: &[Vec<f64>],
        shots: u32,
    ) -> Result<Vec<f64>, EvaluationError> {
        let num_groups = self.templates.len();
        // Task `i · G + g` is candidate `i`'s circuit for group `g`.
        let bound: Vec<BoundCircuit> = candidates
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

        // `execute`, not `evaluate`: it returns one counts map per task, in
        // input order, whatever waves it runs (ADR 0009, decision 6).
        let counts =
            self.planner
                .execute(self.backend.as_ref(), &tasks, &self.config, &self.cancel)?;
        if counts.len() != tasks.len() {
            return Err(EvaluationError::WrongLength {
                expected: tasks.len(),
                got: counts.len(),
            });
        }

        let mut values = Vec::with_capacity(candidates.len());
        for candidate_counts in counts.chunks(num_groups) {
            // The constant, then the groups in order: a fixed summation order,
            // so that a seeded run is reproducible bit for bit.
            let mut energy = self.grouping.constant();
            for (group, group_counts) in self.grouping.groups().iter().zip(candidate_counts) {
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

        // Contract C-5: one finite value per candidate. `MAX_WEIGHT_BOUND`
        // keeps every value finite, so this check is defensive.
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(EvaluationError::NonFinite { index, value });
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypus_circuit::{ConcreteCircuit, GateParam};
    use polypus_infrastructure::{
        BackendCapabilities, BackendError, Counts, InfrastructureError, OptLevel,
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

    /// ADR 0009, decision 6: one `execute` call of `N · G` tasks, split into
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

    #[test]
    fn an_empty_batch_is_an_empty_result() {
        let oracle = oracle(
            Arc::new(ContentBackend::new(3)),
            Arc::new(SequentialPlanner),
        );
        assert!(oracle.evaluate_batch(&[]).is_empty());
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
                tasks.len() - 1
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
        for terms in [
            vec![],
            vec![(0.5, identity)],
            vec![(0.5, identity), (-1.0, identity)],
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
