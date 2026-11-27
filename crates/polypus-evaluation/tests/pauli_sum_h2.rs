//! Acceptance tests of issue #531 on the native backend: the Pauli-sum oracle
//! (ADR 0009) on the 4-qubit Jordan–Wigner H2 of `polypus-physics`, at the
//! analytic optimum of a one-parameter ansatz and end to end with DE and
//! `final_shots`; its `final_shots` re-evaluation drawing new noise; the
//! basis rotation of `Y`, which H2 alone cannot pin; and the exact
//! parameter-shift gradient of #536 (ADR 0009, decision 8) with shots and
//! through Adam.
//!
//! Statistics follow the epic (#528): σ² = Σ_g Var_g / shots, with `Var_g`
//! the variance of group `g`'s per-shot value, read from the counts of the
//! evaluation it qualifies; seeds are fixed beforehand.

use std::f64::consts::FRAC_PI_2;
use std::sync::{Arc, Mutex};

use polypus_circuit::{GateInstruction, GateParam, ParameterizedCircuit};
use polypus_evaluation::{EvaluationOracle, OracleErrorSlot, PauliSumOracleFactory};
use polypus_infrastructure::{
    BoundCircuit, CancelToken, CircuitTask, Counts, InfrastructureError, NativeStatevectorBackend,
    OptLevel, Planner, PlannerRequirements, QuantumBackend, RunParams, SequentialPlanner,
};
use polypus_observable::{CostObservable, Grouping, MeasurementGroup, Pauli, PauliMask};
use polypus_optimizers::{AdamRule, DeStrategy};
use polypus_orchestration::{
    AdamConfig, DeConfig, Method, OracleFactory, Resources, Scheduler, TrainFlow,
};
use polypus_physics::chemistry::{H2Encoding, H2Molecule};
use polypus_physics::hamiltonians::{PauliAxis, PauliSum, PhysicsHamiltonian};
use polypus_sim::Statevector;

/// Shots per group in every evaluation below.
const SHOTS: u32 = 10_000;

/// The 4-qubit Jordan–Wigner H2 at equilibrium (0.7414 Å).
fn h2() -> PauliSum {
    H2Molecule::new(0.7414, H2Encoding::JordanWigner)
        .expect("a valid bond length")
        .to_pauli_sum()
}

/// A `PauliSum` term by term as `(coefficient, mask)`.
fn to_masks(sum: &PauliSum) -> Vec<(f64, PauliMask)> {
    sum.terms()
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
        .collect()
}

/// The ansatz of ADR 0009, decision 7: `cos(θ/2)|1010⟩ + sin(θ/2)|0101⟩`,
/// bit `i` for qubit `i`, in `polypus-physics`'s spin-orbital order
/// `σ↑, σ*↑, σ↓, σ*↓`; `θ = π` is the Hartree–Fock state `|0101⟩`.
fn ansatz() -> ParameterizedCircuit {
    ParameterizedCircuit::new(4)
        .ry(0, GateParam::Param(0))
        .cx(0, 2)
        .x(1)
        .cx(0, 1)
        .cx(1, 3)
}

/// H restricted to span{`|1010⟩`, `|0101⟩`}: `(H₁₀,₁₀, H₅,₅, H₅,₁₀)`.
fn subspace_matrix() -> (f64, f64, f64) {
    let dense = h2().to_dense_real().expect("a 4-qubit dense matrix");
    let get = |row, col| dense.get(row, col).expect("inside the matrix");
    (
        get(0b1010, 0b1010),
        get(0b0101, 0b0101),
        get(0b0101, 0b1010),
    )
}

/// The exact energy of the ansatz at `θ`:
/// `(a + b)/2 + (a − b)/2 · cos θ + h · sin θ`.
fn exact_energy(theta: f64) -> f64 {
    let (a, b, h) = subspace_matrix();
    (a + b) / 2.0 + (a - b) / 2.0 * theta.cos() + h * theta.sin()
}

/// The `θ` minimising [`exact_energy`].
fn optimal_theta() -> f64 {
    let (a, b, h) = subspace_matrix();
    (-h).atan2(-(a - b) / 2.0)
}

/// A planner that runs [`SequentialPlanner`] and keeps the counts of every
/// task, so that a test can compute σ from the evaluation it checks.
#[derive(Default)]
struct RecordingPlanner {
    counts: Mutex<Vec<Counts>>,
}

impl RecordingPlanner {
    fn take(&self) -> Vec<Counts> {
        std::mem::take(&mut *self.counts.lock().unwrap_or_else(|p| p.into_inner()))
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
        let counts = SequentialPlanner.execute(backend, tasks, params, cancel)?;
        self.counts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend(counts.iter().cloned());
        Ok(counts)
    }
}

fn config(id: &str) -> Arc<RunParams> {
    Arc::new(RunParams {
        id: id.to_string(),
        shots: SHOTS,
        seed: Some(531),
        opt_level: OptLevel::default(),
    })
}

/// The oracle for `terms` over `template`, on `backend` behind a recording
/// planner, with the run's shots [`SHOTS`].
fn oracle(
    template: ParameterizedCircuit,
    terms: &[(f64, PauliMask)],
    backend: Arc<dyn QuantumBackend>,
    planner: Arc<RecordingPlanner>,
) -> Box<dyn EvaluationOracle> {
    let resources = Resources::new(backend, Some(planner), config("pauli-sum-h2"))
        .expect("the planner suits the backend");
    PauliSumOracleFactory::new(template, terms)
        .expect("a valid cost")
        .build(&resources, &CancelToken::default(), &OracleErrorSlot::new())
        .evaluation
}

/// The mean and the variance of a group's per-shot value over `counts`
/// (keys most significant bit first, so bit `i` is qubit `i`).
fn group_stats(group: &MeasurementGroup, counts: &Counts) -> (f64, f64) {
    let total: u64 = counts.values().sum();
    let (mut sum, mut squares) = (0.0, 0.0);
    for (key, &n) in counts {
        let basis_state = u64::from_str_radix(key, 2).expect("a bitstring key");
        let value = group.observable().value(basis_state);
        sum += value * n as f64;
        squares += value * value * n as f64;
    }
    let mean = sum / total as f64;
    (mean, squares / total as f64 - mean * mean)
}

/// σ of one candidate's energy from its groups' counts: `Σ_g Var_g / shots`.
fn sigma(grouping: &Grouping, counts: &[Counts]) -> f64 {
    assert_eq!(counts.len(), grouping.groups().len());
    grouping
        .groups()
        .iter()
        .zip(counts)
        .map(|(group, c)| group_stats(group, c).1 / f64::from(SHOTS))
        .sum::<f64>()
        .sqrt()
}

/// The ansatz reaches the exact ground state: it prepares
/// `cos(θ/2)|1010⟩ + sin(θ/2)|0101⟩`, and the lowest energy in that span is
/// H2's FCI energy. Checked without shots, so the tests below measure the
/// oracle and not the ansatz.
#[test]
fn the_ansatz_spans_the_ground_state() {
    let theta = 0.7;
    let circuit = ansatz().assign_parameters(&[theta]).expect("one parameter");
    let mut state = Statevector::new(4).expect("4 qubits");
    for gate in circuit.gates() {
        state.apply(gate).expect("a simulable gate");
    }
    for (index, amplitude) in state.amplitudes().iter().enumerate() {
        let want = match index {
            0b1010 => (theta / 2.0).cos(),
            0b0101 => (theta / 2.0).sin(),
            _ => 0.0,
        };
        assert!(
            (amplitude.re - want).abs() < 1e-12 && amplitude.im.abs() < 1e-12,
            "amplitude {index}: {amplitude}"
        );
    }
    let ground = h2().ground_energy().expect("a ground energy");
    assert!(
        (exact_energy(optimal_theta()) - ground).abs() < 1e-10,
        "{} vs {ground}",
        exact_energy(optimal_theta())
    );
}

/// At the analytic optimum, `−value` is within 3σ of the exact energy (which
/// also pins the sign, ADR 0009 decision 3), and it is the constant plus the
/// groups' means, summed in that order, bit for bit (decision 6).
#[test]
fn at_the_optimum_the_oracle_is_within_three_sigma_of_the_exact_energy() {
    let terms = to_masks(&h2());
    let grouping = PauliSumOracleFactory::new(ansatz(), &terms)
        .expect("a valid cost")
        .grouping()
        .clone();
    assert_eq!(grouping.groups().len(), 5);
    let planner = Arc::new(RecordingPlanner::default());
    let backend = Arc::new(NativeStatevectorBackend::new(531));
    let oracle = oracle(ansatz(), &terms, backend, planner.clone());

    let value = oracle.evaluate_batch(&[vec![optimal_theta()]])[0];
    let counts = planner.take();
    let energy = -value;
    let exact = h2().ground_energy().expect("a ground energy");
    let sigma = sigma(&grouping, &counts);
    assert!(
        (energy - exact).abs() <= 3.0 * sigma,
        "E = {energy}, exact {exact}, σ = {sigma}"
    );

    let mut sum = grouping.constant();
    for (group, c) in grouping.groups().iter().zip(&counts) {
        sum += group
            .observable()
            .expectation_batch(std::slice::from_ref(c))
            .expect("valid counts")[0];
    }
    assert_eq!(energy, sum, "the constant, then the groups in order");
}

/// The `final_shots` path draws new noise: the same candidate, scored by
/// `evaluate_batch` and then by `evaluate_batch_with_shots` with the same
/// shots, gets two different values, both statistically right.
#[test]
fn a_final_shots_batch_does_not_repeat_the_training_noise() {
    let terms = to_masks(&h2());
    let grouping = PauliSumOracleFactory::new(ansatz(), &terms)
        .expect("a valid cost")
        .grouping()
        .clone();
    let planner = Arc::new(RecordingPlanner::default());
    let backend = Arc::new(NativeStatevectorBackend::new(531));
    let oracle = oracle(ansatz(), &terms, backend, planner.clone());
    let theta = vec![optimal_theta()];
    let exact = exact_energy(theta[0]);

    let trained = oracle.evaluate_batch(std::slice::from_ref(&theta))[0];
    let trained_sigma = sigma(&grouping, &planner.take());
    let again = oracle.evaluate_batch_with_shots(std::slice::from_ref(&theta), SHOTS)[0];
    let again_sigma = sigma(&grouping, &planner.take());

    assert_ne!(trained, again, "a new batch must draw new shots");
    assert!((-trained - exact).abs() <= 3.0 * trained_sigma);
    assert!((-again - exact).abs() <= 3.0 * again_sigma);
}

/// Issue #531's end-to-end criterion: train the unreduced 4-qubit H2 with DE
/// on the native backend with `final_shots`, then, as the epic's statistical
/// rules ask (the final batch keeps the best of up to five, so it is still
/// optimistic), evaluate `final_params` once more, on the same backend and so
/// with new noise: within 3σ of FCI.
#[test]
fn de_trains_h2_to_within_three_sigma_of_fci() {
    let terms = to_masks(&h2());
    let grouping = PauliSumOracleFactory::new(ansatz(), &terms)
        .expect("a valid cost")
        .grouping()
        .clone();
    let backend = Arc::new(NativeStatevectorBackend::new(531));
    let resources = Resources::new(backend.clone(), None, config("pauli-sum-h2-de"))
        .expect("the default planner");
    let flow = TrainFlow {
        factory: PauliSumOracleFactory::new(ansatz(), &terms).expect("a valid cost"),
        method: Method::De(DeConfig {
            generations: 20,
            population_size: 10,
            tolerance: 0.0,
            patience: 20,
            strategy: DeStrategy::Rand1Bin,
            mutation_factor: None,
            crossover_rate: None,
        }),
        dimensions: 1,
        seed: 531,
        initial_point: None,
        final_shots: Some(SHOTS),
        observer: None,
        errors: OracleErrorSlot::new(),
        trace: None,
    };
    // `OracleError` does not implement `Debug`, so match rather than `.expect`.
    let outcome = match Scheduler::ephemeral(resources).run(flow) {
        Ok(outcome) => outcome,
        Err(_) => panic!("the train flow must run"),
    };
    let final_evaluation = outcome
        .final_evaluation
        .expect("a final batch was asked for");
    // The final batch draws new noise (contract C-5): `best_params` is its
    // candidate 0, scored again, so its value differs from `best_fitness`.
    assert_eq!(
        final_evaluation.candidates[0],
        outcome.optimization.best_params
    );
    assert_ne!(
        final_evaluation.values[0],
        outcome.optimization.best_fitness
    );
    let final_params = final_evaluation.params;
    // Convergence apart from the oracle's accuracy, as the epic asks: the
    // exact energy at `final_params` is within 1 mHa of FCI.
    let fci = h2().ground_energy().expect("a ground energy");
    assert!(
        (exact_energy(final_params[0]) - fci).abs() < 1e-3,
        "DE did not converge: θ = {}, exact energy {}, FCI {fci}",
        final_params[0],
        exact_energy(final_params[0])
    );

    let planner = Arc::new(RecordingPlanner::default());
    let oracle = oracle(ansatz(), &terms, backend, planner.clone());
    let energy = -oracle.evaluate_batch_with_shots(std::slice::from_ref(&final_params), SHOTS)[0];
    let sigma = sigma(&grouping, &planner.take());
    assert!(
        (energy - fci).abs() <= 3.0 * sigma,
        "E = {energy}, FCI {fci}, σ = {sigma}; exact energy at final_params {} \
         (θ = {}), so convergence error {}",
        exact_energy(final_params[0]),
        final_params[0],
        exact_energy(final_params[0]) - fci
    );
}

/// The rotation of `Y` is `S†` then `H`: after `Rx(π/2)`, `|0⟩` is the `−1`
/// eigenstate of `Y`, so `⟨Y⟩ = −1` with no shot noise, and the oracle
/// returns `+1`. `S` instead of `S†` would give `−1`, and `H` then `S†`
/// would measure `X`, `0` up to shot noise. A term with an odd number of `Y`
/// tells these apart; every H2 term has an even number.
#[test]
fn the_y_rotation_measures_y() {
    let terms = [(
        1.0,
        PauliMask::new([(0, Pauli::Y)]).expect("a valid string"),
    )];
    let template = ParameterizedCircuit::new(1).rx(0, GateParam::Param(0));
    let planner = Arc::new(RecordingPlanner::default());
    let oracle = oracle(
        template.clone(),
        &terms,
        Arc::new(NativeStatevectorBackend::new(531)),
        planner,
    );
    assert_eq!(oracle.evaluate_batch(&[vec![FRAC_PI_2]]), vec![1.0]);

    // At another angle ⟨Y⟩ = −sin θ, within 3σ, σ² = (1 − sin²θ)/shots.
    let theta = 1.0_f64;
    let value = oracle.evaluate_batch(&[vec![theta]])[0];
    let sigma = ((1.0 - theta.sin().powi(2)) / f64::from(SHOTS)).sqrt();
    assert!(
        (-value - -theta.sin()).abs() <= 3.0 * sigma,
        "⟨Y⟩ = {}, want {}",
        -value,
        -theta.sin()
    );
}

/// #536 with shots: H2's parameter-shift gradient is within 3σ of the
/// analytic `−dE/dθ`, σ² = (σ₊² + σ₋²)/4 with each `σ±²` from its candidate's
/// group variances (ADR 0009, decision 8), at a point where the slope is
/// large and at the optimum, where it vanishes. Fixed seed.
#[test]
fn the_shot_gradient_is_within_three_sigma_of_the_analytic_derivative() {
    let terms = to_masks(&h2());
    let factory = PauliSumOracleFactory::new(ansatz(), &terms).expect("a valid cost");
    assert!(factory.gradient_unavailable().is_none());
    let grouping = factory.grouping().clone();
    let groups = grouping.groups().len();
    let planner = Arc::new(RecordingPlanner::default());
    let resources = Resources::new(
        Arc::new(NativeStatevectorBackend::new(536)),
        Some(planner.clone()),
        config("pauli-sum-h2-gradient"),
    )
    .expect("the planner suits the backend");
    let errors = OracleErrorSlot::new();
    let gradient = factory
        .build(&resources, &CancelToken::default(), &errors)
        .gradient
        .expect("the H2 ansatz is single-frequency");
    let (a, b, h) = subspace_matrix();
    let derivative = |t: f64| -(a - b) / 2.0 * t.sin() + h * t.cos();
    for theta in [1.0, optimal_theta()] {
        let measured = gradient.gradient_batch(&[theta], 1);
        assert!(!errors.failed());
        let counts = planner.take();
        assert_eq!(counts.len(), 2 * groups, "θ + π/2, then θ − π/2");
        let (plus, minus) = counts.split_at(groups);
        let sigma = (sigma(&grouping, plus).powi(2) + sigma(&grouping, minus).powi(2)).sqrt() / 2.0;
        assert!(
            (measured[0] + derivative(theta)).abs() <= 3.0 * sigma,
            "θ = {theta}: gradient {}, −dE/dθ {}, σ = {sigma}",
            measured[0],
            -derivative(theta)
        );
    }
}

/// A planner that runs [`SequentialPlanner`] and records the angle of the
/// first gate (the ansatz's `ry(θ)`) of every task, call by call.
#[derive(Default)]
struct AnglePlanner {
    calls: Mutex<Vec<Vec<f64>>>,
}

impl Planner for AnglePlanner {
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
        let angles = tasks
            .iter()
            .map(|task| match task.circuit {
                BoundCircuit::Native(circuit) => match circuit.gates().first() {
                    Some(GateInstruction::Ry {
                        theta: GateParam::Fixed(angle),
                        ..
                    }) => *angle,
                    other => panic!("the ansatz starts with ry(θ): {other:?}"),
                },
                _ => panic!("the oracle binds native circuits"),
            })
            .collect();
        self.calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(angles);
        SequentialPlanner.execute(backend, tasks, params, cancel)
    }
}

/// #536 through a training run: Adam takes the oracle's parameter shift, not
/// the finite difference (each iteration scores `θ` and `θ ± π/2`, five
/// groups each, not `θ ± h`), each iteration is one batch of three
/// candidates, and it converges: the exact energy at its best `θ` is within
/// 1 mHa of FCI (convergence apart from the oracle's accuracy, as the epic
/// asks). Fixed seed. The angles are read over every planner call, since a
/// batch is one call per window and the window depends on the host's
/// parallelism (#667).
#[test]
fn adam_trains_h2_with_the_parameter_shift_gradient() {
    let terms = to_masks(&h2());
    let planner = Arc::new(AnglePlanner::default());
    let resources = Resources::new(
        Arc::new(NativeStatevectorBackend::new(536)),
        Some(planner.clone()),
        config("pauli-sum-h2-adam"),
    )
    .expect("the planner suits the backend");
    let start = 2.0;
    let iterations = 60;
    let flow = TrainFlow {
        factory: PauliSumOracleFactory::new(ansatz(), &terms).expect("a valid cost"),
        method: Method::Adam(AdamConfig {
            max_iters: iterations,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (-std::f64::consts::PI, std::f64::consts::PI),
            tolerance: 0.0,
            patience: iterations,
            finite_difference_step: 1e-3,
            rule: AdamRule::Adam,
        }),
        dimensions: 1,
        seed: 536,
        initial_point: Some(vec![start]),
        final_shots: None,
        observer: None,
        errors: OracleErrorSlot::new(),
        trace: None,
    };
    let outcome = match Scheduler::ephemeral(resources).run(flow) {
        Ok(outcome) => outcome,
        Err(_) => panic!("the train flow must run"),
    };
    let angles: Vec<f64> = planner
        .calls
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .concat();
    // Each iteration's batch (2P + 1 = 3 candidates, five groups each), then
    // the final θ alone.
    let iterations = iterations as usize;
    assert_eq!(angles.len(), iterations * 15 + 5);
    let shifted = |theta: f64| -> Vec<f64> {
        [theta, theta + FRAC_PI_2, theta - FRAC_PI_2]
            .iter()
            .flat_map(|&angle| std::iter::repeat_n(angle, 5))
            .collect()
    };
    assert_eq!(angles[..15], shifted(start), "θ₀ and θ₀ ± π/2");
    for batch in angles[..iterations * 15].chunks(15) {
        assert_eq!(batch, shifted(batch[0]), "θ and θ ± π/2, five groups each");
    }
    let best = outcome.optimization.best_params[0];
    let fci = h2().ground_energy().expect("a ground energy");
    assert!(
        (exact_energy(best) - fci).abs() < 1e-3,
        "Adam did not converge: θ = {best}, exact energy {}, FCI {fci}",
        exact_energy(best)
    );
}
