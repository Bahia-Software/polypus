//! Optimizer benchmark on a molecular energy (issue #536): DE, PSO, CMA-ES,
//! SPSA, Rotosolve and Adam train the 4-qubit Jordan–Wigner H₂ (STO-3G) of
//! `polypus-physics` through `PauliSumOracle` (ADR 0009) on the native
//! backend, along the bond-length curve of `reference-data/h2-sto3g/`.
//!
//! ```text
//! cargo run --release -p polypus-evaluation --example h2_optimizers
//! cargo run --release -p polypus-evaluation --example h2_optimizers -- --quick
//! cargo run --release -p polypus-evaluation --example h2_optimizers -- --json out.json
//! ```
//!
//! (Like the crate's tests, it links libpython: run it with the conda
//! environment's `lib` on `LD_LIBRARY_PATH`.)
//!
//! What it does, following the epic's statistical rules (#528):
//!
//! - **Two ansätze.** `exact`: the one-parameter double excitation of ADR
//!   0009, decision 7, which reaches the FCI energy, so every optimizer
//!   should get there and a failure is the optimizer's or the noise's.
//!   `hea`: a hardware-efficient ansatz, three layers of `ry` on every qubit
//!   with a chain of `cz` between them (12 parameters, each in one `ry`, so
//!   Rotosolve applies and Adam has the exact parameter-shift gradient of
//!   #536); it need not reach FCI, so its own minimum is reported too.
//! - **Paired seeds.** Every optimizer runs every (ansatz, bond length) with
//!   the same seeds, fixed beforehand.
//! - **The same budget.** Each run may execute at most `--budget` training
//!   circuits (10 000 by default) of 10⁴ shots each, counted by the planner,
//!   so Adam's shifted candidates count too. A run is stopped at the end of
//!   the iteration after which the next one would exceed the budget.
//! - **The energy reported** is never the optimizer's best value (biased
//!   low): the run's `final_shots` batch picks `final_params`, which are then
//!   evaluated once more with another seed and 2.5·10⁵ shots per group
//!   (σ ≈ 0.5 mHa). Its error against FCI is the accuracy; the exact energy
//!   at `final_params` (no shots) gives the convergence apart from the noise.
//! - **Optimizer settings** are the defaults of the Python classes, except
//!   what sets the budget (iterations, generations, sweeps; DE's population
//!   is 10, as in the H₂ tests) and early stopping, which is off so that
//!   every run spends its budget. "Adam (finite differences)" is Adam
//!   without the oracle's exact gradient, to show what #536 brings. COBYLA
//!   needs the Python bindings of #532 and is not run.

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Write as _;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polypus_circuit::{GateParam, ParameterizedCircuit};
use polypus_evaluation::{OracleErrorSlot, PauliSumOracleFactory};
use polypus_infrastructure::{
    CancelToken, CircuitTask, Counts, InfrastructureError, NativeStatevectorBackend, OptLevel,
    Planner, PlannerRequirements, QuantumBackend, RunParams, SequentialPlanner,
};
use polypus_observable::{Pauli, PauliMask};
use polypus_optimizers::{AdamRule, DeStrategy};
use polypus_orchestration::{
    AdamConfig, CmaesConfig, DeConfig, Method, OracleFactory, Oracles, PsoConfig, Resources,
    RotosolveConfig, Scheduler, SpsaConfig, TrainFlow, TrainObserver, TrainProgress,
};
use polypus_physics::chemistry::{H2Encoding, H2Molecule};
use polypus_physics::hamiltonians::{DenseMatrix, PauliAxis, PauliSum, PhysicsHamiltonian};
use polypus_sim::Statevector;
use rayon::prelude::*;

/// Shots per circuit while training.
const SHOTS: u32 = 10_000;
/// Shots per group of the final batch and of the re-evaluation.
const FINAL_SHOTS: u32 = 250_000;
/// The seeds of the re-evaluation are the run's plus this offset.
const REEVALUATION_SEED_OFFSET: u64 = 1_000_000;
/// Chemical accuracy, in hartree.
const CHEMICAL_ACCURACY: f64 = 1.6e-3;
/// The 0.975 quantile of Student's t with `df` degrees of freedom: a table
/// up to 30, then the Cornish–Fisher expansion around the normal quantile
/// (within 0.001 there).
fn t_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if (1..=30).contains(&df) {
        return TABLE[df - 1];
    }
    let z: f64 = 1.959_964;
    let n = df as f64;
    z + (z.powi(3) + z) / (4.0 * n)
        + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * n * n)
}

/// The command-line settings.
struct Settings {
    seeds: u64,
    budget: u64,
    lengths: Vec<f64>,
    json: Option<String>,
}

fn settings() -> Settings {
    let mut settings = Settings {
        seeds: 10,
        budget: 10_000,
        lengths: Vec::new(),
        json: None,
    };
    let mut quick = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value"))
        };
        match arg.as_str() {
            "--seeds" => settings.seeds = value("--seeds").parse().expect("--seeds: an integer"),
            "--budget" => {
                settings.budget = value("--budget").parse().expect("--budget: an integer")
            }
            "--json" => settings.json = Some(value("--json")),
            "--quick" => quick = true,
            other => {
                panic!("unknown argument {other} (--seeds N, --budget N, --json FILE, --quick)")
            }
        }
    }
    assert!((1..=31).contains(&settings.seeds), "--seeds: 1 to 31");
    settings.lengths = reference_curve().into_iter().map(|(r, _)| r).collect();
    if quick {
        settings.seeds = 2;
        settings.budget = 2_000;
        settings.lengths = vec![0.7414, 2.0];
    }
    settings
}

/// `(R, FCI)` from `reference-data/h2-sto3g/fci_energies.csv` (PySCF),
/// which `polypus-physics`'s tests match within 1e-10.
fn reference_curve() -> Vec<(f64, f64)> {
    let path = format!(
        "{}/../../reference-data/h2-sto3g/fci_energies.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let csv = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    csv.lines()
        .skip(1)
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            (
                fields[0].parse().expect("a bond length"),
                fields[2].parse().expect("an FCI energy"),
            )
        })
        .collect()
}

fn hamiltonian(length: f64) -> PauliSum {
    H2Molecule::new(length, H2Encoding::JordanWigner)
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Ansatz {
    Exact,
    Hea,
}

impl Ansatz {
    fn name(self) -> &'static str {
        match self {
            Ansatz::Exact => "exact",
            Ansatz::Hea => "hea",
        }
    }

    fn circuit(self) -> ParameterizedCircuit {
        match self {
            // cos(θ/2)|1010⟩ + sin(θ/2)|0101⟩ (ADR 0009, decision 7).
            Ansatz::Exact => ParameterizedCircuit::new(4)
                .ry(0, GateParam::Param(0))
                .cx(0, 2)
                .x(1)
                .cx(0, 1)
                .cx(1, 3),
            Ansatz::Hea => {
                let mut circuit = ParameterizedCircuit::new(4);
                for layer in 0..3 {
                    if layer > 0 {
                        circuit = circuit.cz(0, 1).cz(1, 2).cz(2, 3);
                    }
                    for qubit in 0..4 {
                        circuit = circuit.ry(qubit, GateParam::Param(4 * layer + qubit));
                    }
                }
                circuit
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Optimizer {
    De,
    Pso,
    Cmaes,
    Spsa,
    Rotosolve,
    Adam,
    AdamFiniteDifference,
}

const OPTIMIZERS: [Optimizer; 7] = [
    Optimizer::De,
    Optimizer::Pso,
    Optimizer::Cmaes,
    Optimizer::Spsa,
    Optimizer::Rotosolve,
    Optimizer::Adam,
    Optimizer::AdamFiniteDifference,
];

/// Iteration limits no run reaches: the budget stops it first.
const UNREACHED: u32 = 1_000_000;

impl Optimizer {
    fn name(self) -> &'static str {
        match self {
            Optimizer::De => "DE (rand1bin, population 10)",
            Optimizer::Pso => "PSO",
            Optimizer::Cmaes => "CMA-ES",
            Optimizer::Spsa => "SPSA",
            Optimizer::Rotosolve => "Rotosolve",
            Optimizer::Adam => "Adam (parameter shift)",
            Optimizer::AdamFiniteDifference => "Adam (finite differences)",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Optimizer::De => "de",
            Optimizer::Pso => "pso",
            Optimizer::Cmaes => "cmaes",
            Optimizer::Spsa => "spsa",
            Optimizer::Rotosolve => "rotosolve",
            Optimizer::Adam => "adam",
            Optimizer::AdamFiniteDifference => "adam-fd",
        }
    }

    /// The Python classes' defaults, early stopping off, iterations unbounded.
    fn method(self) -> Method {
        let bounds = (-PI, PI);
        match self {
            Optimizer::De => Method::De(DeConfig {
                generations: UNREACHED,
                population_size: 10,
                tolerance: 0.0,
                patience: UNREACHED,
                strategy: DeStrategy::Rand1Bin,
                mutation_factor: None,
                crossover_rate: None,
            }),
            Optimizer::Pso => Method::Pso(PsoConfig {
                generations: UNREACHED,
                population_size: 50,
                bounds,
                inertia_weight: 0.5,
                cognitive_weight: 1.0,
                social_weight: 1.0,
                tolerance: 0.0,
            }),
            Optimizer::Cmaes => Method::Cmaes(CmaesConfig {
                generations: UNREACHED,
                population_size: None,
                sigma0: FRAC_PI_2,
                tolerance: 0.0,
                patience: UNREACHED,
                restarts: 0,
            }),
            Optimizer::Spsa => Method::Spsa(SpsaConfig {
                max_iters: UNREACHED,
                resamplings: 1,
                perturbation: 0.1,
                target_step: 0.63,
                alpha: 0.602,
                gamma: 0.101,
                stability_constant: 0.0,
                eval_interval: 20,
            }),
            Optimizer::Rotosolve => Method::Rotosolve(RotosolveConfig {
                sweeps: UNREACHED,
                reevaluate_every: 5,
                tolerance: 0.0,
                patience: UNREACHED,
            }),
            Optimizer::Adam | Optimizer::AdamFiniteDifference => Method::Adam(AdamConfig {
                max_iters: UNREACHED,
                learning_rate: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                bounds,
                tolerance: 0.0,
                patience: UNREACHED,
                finite_difference_step: 0.2,
                rule: AdamRule::Adam,
            }),
        }
    }
}

/// A planner that runs [`SequentialPlanner`] and counts the training
/// circuits (those measured with [`SHOTS`]) it runs.
#[derive(Default)]
struct CountingPlanner {
    circuits: Arc<AtomicU64>,
}

impl Planner for CountingPlanner {
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
        let training = tasks.iter().filter(|task| task.shots == SHOTS).count();
        self.circuits.fetch_add(training as u64, Ordering::SeqCst);
        SequentialPlanner.execute(backend, tasks, params, cancel)
    }
}

/// The oracle factory without its exact gradient, so that Adam takes the
/// finite difference.
struct WithoutGradient(PauliSumOracleFactory);

impl OracleFactory for WithoutGradient {
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        let mut oracles = self.0.build(resources, cancel, errors);
        oracles.gradient = None;
        oracles
    }
}

/// The exact energy `⟨ψ(θ)|H|ψ(θ)⟩` from the statevector, without shots.
fn exact_energy(circuit: &ParameterizedCircuit, dense: &DenseMatrix, theta: &[f64]) -> f64 {
    let bound = circuit
        .assign_parameters(theta)
        .expect("the ansatz's parameters");
    let mut state = Statevector::new(bound.num_qubits()).expect("4 qubits");
    for gate in bound.gates() {
        state.apply(gate).expect("a simulable gate");
    }
    let amplitudes = state.amplitudes();
    let mut energy = 0.0;
    for (i, a) in amplitudes.iter().enumerate() {
        for (j, b) in amplitudes.iter().enumerate() {
            let h = dense.get(i, j).expect("inside the matrix");
            if h != 0.0 {
                energy += h * (a.conj() * b).re;
            }
        }
    }
    energy
}

/// The lowest exact energy the ansatz reaches, found by exact
/// parameter-shift gradient descent (Adam's update, no shots) from many
/// seeded starts: an upper bound on the ansatz's minimum.
fn ansatz_minimum(ansatz: Ansatz, dense: &DenseMatrix) -> f64 {
    let circuit = ansatz.circuit();
    let dims = circuit.num_params();
    let mut rng = polypus_circuit::rng::SplitMix64::new(536);
    let starts: Vec<Vec<f64>> = (0..32)
        .map(|_| {
            (0..dims)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) * PI)
                .collect()
        })
        .collect();
    starts
        .par_iter()
        .map(|start| {
            let mut theta = start.clone();
            let (mut m, mut v) = (vec![0.0; dims], vec![0.0; dims]);
            let mut best = exact_energy(&circuit, dense, &theta);
            for step in 1..=1500 {
                for k in 0..dims {
                    let shifted = |delta: f64| {
                        let mut point = theta.clone();
                        point[k] += delta;
                        exact_energy(&circuit, dense, &point)
                    };
                    let g = (shifted(FRAC_PI_2) - shifted(-FRAC_PI_2)) / 2.0;
                    m[k] = 0.9 * m[k] + 0.1 * g;
                    v[k] = 0.999 * v[k] + 0.001 * g * g;
                    let m_hat = m[k] / (1.0 - 0.9f64.powi(step));
                    let v_hat = v[k] / (1.0 - 0.999f64.powi(step));
                    theta[k] -= 0.05 * m_hat / (v_hat.sqrt() + 1e-8);
                }
                best = best.min(exact_energy(&circuit, dense, &theta));
            }
            best
        })
        .reduce(|| f64::INFINITY, f64::min)
}

/// One (ansatz, bond length): the Hamiltonian, its FCI and the ansatz's
/// exact minimum.
struct Point {
    ansatz: Ansatz,
    length: f64,
    terms: Vec<(f64, PauliMask)>,
    dense: DenseMatrix,
    fci: f64,
    minimum: f64,
}

/// One training run's results.
struct Run {
    ansatz: Ansatz,
    length: f64,
    optimizer: Optimizer,
    seed: u64,
    /// The re-evaluated energy of `final_params` minus FCI, hartree.
    error: f64,
    /// The exact energy at `final_params` minus FCI, hartree.
    exact_error: f64,
    /// The exact energy at `final_params` minus the ansatz's minimum.
    ansatz_error: f64,
    circuits: u64,
}

fn run_config(seed: u64, id: &str) -> Arc<RunParams> {
    Arc::new(RunParams {
        id: id.to_string(),
        shots: SHOTS,
        seed: Some(seed),
        opt_level: OptLevel::default(),
    })
}

/// The training run of `optimizer` over `factory`, with `final_shots`.
fn training_flow<F: OracleFactory>(
    factory: F,
    optimizer: Optimizer,
    dimensions: u32,
    seed: u64,
    observer: Box<dyn TrainObserver>,
) -> TrainFlow<F> {
    TrainFlow {
        factory,
        method: optimizer.method(),
        dimensions,
        seed,
        initial_point: None,
        final_shots: Some(FINAL_SHOTS),
        observer: Some(observer),
        errors: OracleErrorSlot::new(),
        trace: None,
    }
}

fn train(point: &Point, optimizer: Optimizer, seed: u64, budget: u64) -> Run {
    let circuit = point.ansatz.circuit();
    let factory = PauliSumOracleFactory::new(circuit.clone(), &point.terms).expect("a valid cost");
    let id = format!(
        "h2-{}-{}-{}-{seed}",
        point.ansatz.name(),
        point.length,
        optimizer.key()
    );
    let planner = Arc::new(CountingPlanner::default());
    let circuits = Arc::clone(&planner.circuits);
    let resources = Resources::new(
        Arc::new(NativeStatevectorBackend::new(seed)),
        Some(planner),
        run_config(seed, &id),
    )
    .expect("the planner suits the backend");
    // Stop at the end of the iteration after which another one as costly as
    // the costliest so far would exceed the budget. The first iteration is
    // left out of that maximum: it also carries what ran before it (DE's and
    // PSO's first population, SPSA's calibration), which the later ones do
    // not repeat. Every later iteration costs at most as much as the
    // costliest seen (an occasional re-measurement included), so no run
    // exceeds the budget, which `train` checks.
    let observed = Arc::clone(&circuits);
    let (mut last, mut costliest, mut first) = (0u64, 0u64, true);
    let observer = move |_: &TrainProgress<'_>| {
        let spent = observed.load(Ordering::SeqCst);
        if !first {
            costliest = costliest.max(spent - last);
        }
        first = false;
        last = spent;
        if spent + costliest > budget {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    let observer: Box<dyn TrainObserver> = Box::new(observer);
    let scheduler = Scheduler::ephemeral(resources);
    let dimensions = u32::try_from(circuit.num_params()).expect("a few parameters");
    let outcome = match optimizer {
        Optimizer::AdamFiniteDifference => scheduler.run(training_flow(
            WithoutGradient(factory),
            optimizer,
            dimensions,
            seed,
            observer,
        )),
        _ => scheduler.run(training_flow(
            factory, optimizer, dimensions, seed, observer,
        )),
    };
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(_) => panic!("{id}: the train flow failed"),
    };
    let final_params = outcome
        .final_evaluation
        .expect("a final batch was asked for")
        .params;

    // A new evaluation, with another seed, of `final_params`.
    let reevaluation_seed = seed + REEVALUATION_SEED_OFFSET;
    let resources = Resources::new(
        Arc::new(NativeStatevectorBackend::new(reevaluation_seed)),
        None,
        run_config(reevaluation_seed, &format!("{id}-reevaluation")),
    )
    .expect("the default planner");
    let oracle = PauliSumOracleFactory::new(circuit.clone(), &point.terms)
        .expect("a valid cost")
        .build(&resources, &CancelToken::default(), &OracleErrorSlot::new())
        .evaluation;
    let energy =
        -oracle.evaluate_batch_with_shots(std::slice::from_ref(&final_params), FINAL_SHOTS)[0];
    let exact = exact_energy(&circuit, &point.dense, &final_params);
    Run {
        ansatz: point.ansatz,
        length: point.length,
        optimizer,
        seed,
        error: energy - point.fci,
        exact_error: exact - point.fci,
        ansatz_error: exact - point.minimum,
        circuits: {
            let spent = circuits.load(Ordering::SeqCst);
            assert!(
                spent <= budget,
                "{id}: {spent} circuits, over the budget of {budget}"
            );
            spent
        },
    }
}

/// The median of `values`.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The two-sided p-value of `k` successes in `n` fair coin flips (the
/// exact sign test, and McNemar's test on the discordant pairs).
fn binomial_two_sided(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let low = k.min(n - k);
    let mut tail = 0.0;
    let mut choose = 1.0f64;
    for i in 0..=low {
        if i > 0 {
            choose = choose * (n - i + 1) as f64 / i as f64;
        }
        tail += choose;
    }
    (2.0 * tail / 2f64.powi(n as i32)).min(1.0)
}

/// Mean and 95 % confidence half-width of `values`.
fn mean_ci(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::NAN);
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, t_975(n - 1) * (variance / n as f64).sqrt())
}

fn main() {
    let settings = settings();
    let reference: BTreeMap<String, f64> = reference_curve()
        .into_iter()
        .map(|(r, fci)| (format!("{r}"), fci))
        .collect();
    let started = std::time::Instant::now();

    let mut points = Vec::new();
    for ansatz in [Ansatz::Exact, Ansatz::Hea] {
        for &length in &settings.lengths {
            let sum = hamiltonian(length);
            let dense = sum.to_dense_real().expect("a 4-qubit matrix");
            let fci = reference[&format!("{length}")];
            let minimum = ansatz_minimum(ansatz, &dense);
            points.push(Point {
                ansatz,
                length,
                terms: to_masks(&sum),
                dense,
                fci,
                minimum,
            });
        }
    }

    let jobs: Vec<(usize, Optimizer, u64)> = (0..points.len())
        .flat_map(|p| {
            OPTIMIZERS
                .iter()
                .flat_map(move |&o| (0..settings.seeds).map(move |s| (p, o, s)))
        })
        .collect();
    let runs: Vec<Run> = jobs
        .par_iter()
        .map(|&(p, optimizer, seed)| train(&points[p], optimizer, seed, settings.budget))
        .collect();
    let elapsed = started.elapsed();

    print!("{}", report(&settings, &points, &runs, elapsed));
    if let Some(path) = &settings.json {
        std::fs::write(path, json(&settings, &points, &runs)).expect("the JSON file");
        println!("\nRaw runs written to {path}.");
    }
}

/// The Markdown report: per ansatz, the error of the re-evaluated energy
/// against FCI at each bond length, then a summary over the curve.
fn report(
    settings: &Settings,
    points: &[Point],
    runs: &[Run],
    elapsed: std::time::Duration,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "H₂ (STO-3G, Jordan–Wigner, 4 qubits): {} seeds, budget {} training circuits of {} \
         shots, re-evaluation with {} shots per group; {:.0} s.\n",
        settings.seeds,
        settings.budget,
        SHOTS,
        FINAL_SHOTS,
        elapsed.as_secs_f64()
    );
    let select = |ansatz: Ansatz, optimizer: Optimizer, length: Option<f64>| -> Vec<&Run> {
        runs.iter()
            .filter(|r| {
                r.ansatz == ansatz
                    && r.optimizer == optimizer
                    && length.is_none_or(|l| r.length == l)
            })
            .collect()
    };
    for ansatz in [Ansatz::Exact, Ansatz::Hea] {
        let _ = writeln!(
            out,
            "### Ansatz `{}`: error of the re-evaluated energy against FCI, mHa: median (runs \
             outside 1.6 mHa)\n",
            ansatz.name()
        );
        let _ = write!(out, "| Optimizer |");
        for length in &settings.lengths {
            let _ = write!(out, " {length} Å |");
        }
        let _ = writeln!(out);
        let _ = writeln!(out, "|---|{}", "---|".repeat(settings.lengths.len()));
        for optimizer in OPTIMIZERS {
            let _ = write!(out, "| {} |", optimizer.name());
            for &length in &settings.lengths {
                let errors: Vec<f64> = select(ansatz, optimizer, Some(length))
                    .iter()
                    .map(|r| r.error * 1e3)
                    .collect();
                let outside = errors
                    .iter()
                    .filter(|e| e.abs() > CHEMICAL_ACCURACY * 1e3)
                    .count();
                let _ = write!(
                    out,
                    " {:+.2} ({outside}/{}) |",
                    median(&errors),
                    errors.len()
                );
            }
            let _ = writeln!(out);
        }
        let _ = write!(out, "| ansatz minimum − FCI |");
        for &length in &settings.lengths {
            let point = points
                .iter()
                .find(|p| p.ansatz == ansatz && p.length == length)
                .expect("every point");
            let _ = write!(out, " {:+.3} |", (point.minimum - point.fci) * 1e3);
        }
        let _ = writeln!(out, "\n");
        let _ = writeln!(
            out,
            "Over the whole curve ({} runs per optimizer):\n",
            settings.lengths.len() as u64 * settings.seeds
        );
        let _ = writeln!(
            out,
            "| Optimizer | within 1.6 mHa | exact error vs FCI, mHa | exact error vs the ansatz's \
             minimum, mHa | training circuits |"
        );
        let _ = writeln!(out, "|---|---|---|---|---|");
        for optimizer in OPTIMIZERS {
            let selected = select(ansatz, optimizer, None);
            let within = selected
                .iter()
                .filter(|r| r.error.abs() <= CHEMICAL_ACCURACY)
                .count();
            let exact: Vec<f64> = selected.iter().map(|r| r.exact_error * 1e3).collect();
            let own: Vec<f64> = selected.iter().map(|r| r.ansatz_error * 1e3).collect();
            let circuits: Vec<f64> = selected.iter().map(|r| r.circuits as f64).collect();
            let (exact_mean, exact_ci) = mean_ci(&exact);
            let (own_mean, own_ci) = mean_ci(&own);
            let (circuits_mean, _) = mean_ci(&circuits);
            let max_circuits = selected.iter().map(|r| r.circuits).max().unwrap_or(0);
            let _ = writeln!(
                out,
                "| {} | {:.0} % | {exact_mean:.3} ± {exact_ci:.3} | {own_mean:.3} ± {own_ci:.3} | \
                 {circuits_mean:.0} (max {max_circuits}) |",
                optimizer.name(),
                100.0 * within as f64 / selected.len() as f64
            );
        }
        let _ = writeln!(
            out,
            "| COBYLA | — (needs the Python bindings, #532) | — | — | — |\n"
        );
        let _ = writeln!(out, "Paired comparisons (same bond length and seed):\n");
        for (a, b) in [
            (Optimizer::Adam, Optimizer::AdamFiniteDifference),
            (Optimizer::Rotosolve, Optimizer::Cmaes),
        ] {
            let (left, right) = (select(ansatz, a, None), select(ansatz, b, None));
            let pairs: Vec<(&Run, &Run)> = left
                .iter()
                .map(|l| {
                    let r = right
                        .iter()
                        .find(|r| r.length == l.length && r.seed == l.seed)
                        .expect("paired runs");
                    (*l, *r)
                })
                .collect();
            let differences: Vec<f64> = pairs
                .iter()
                .map(|(l, r)| (l.exact_error - r.exact_error) * 1e3)
                .collect();
            let lower = differences.iter().filter(|d| **d < 0.0).count();
            let untied = differences.iter().filter(|d| **d != 0.0).count();
            let (mean, ci) = mean_ci(&differences);
            let within = |r: &Run| r.error.abs() <= CHEMICAL_ACCURACY;
            let only_left = pairs
                .iter()
                .filter(|(l, r)| within(l) && !within(r))
                .count();
            let only_right = pairs
                .iter()
                .filter(|(l, r)| !within(l) && within(r))
                .count();
            let _ = writeln!(
                out,
                "- {} against {}: exact error lower in {lower} of {untied} pairs (sign test p = \
                 {:.3}), difference {mean:+.2} ± {ci:.2} mHa; within 1.6 mHa for only the first \
                 in {only_left} pairs, only the second in {only_right} (McNemar p = {:.3}).",
                a.name(),
                b.name(),
                binomial_two_sided(lower, untied),
                binomial_two_sided(only_left, only_left + only_right)
            );
        }
        let noise: Vec<f64> = runs
            .iter()
            .filter(|r| r.ansatz == ansatz)
            .map(|r| (r.error - r.exact_error) * 1e3)
            .collect();
        let mean = noise.iter().sum::<f64>() / noise.len() as f64;
        let sd = (noise.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (noise.len() - 1).max(1) as f64)
            .sqrt();
        let _ = writeln!(
            out,
            "- Re-evaluation noise (re-evaluated minus exact energy): {mean:+.3} mHa on average, \
             standard deviation {sd:.3} mHa.\n"
        );
    }
    out
}

/// Every run, as JSON lines in one array.
fn json(settings: &Settings, points: &[Point], runs: &[Run]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{{\"seeds\": {}, \"budget\": {}, \"shots\": {SHOTS}, \"final_shots\": {FINAL_SHOTS},",
        settings.seeds, settings.budget
    );
    let _ = writeln!(out, "\"points\": [");
    for (i, p) in points.iter().enumerate() {
        let comma = if i + 1 < points.len() { "," } else { "" };
        let _ = writeln!(
            out,
            "  {{\"ansatz\": \"{}\", \"length\": {}, \"fci\": {}, \"ansatz_minimum\": {}}}{comma}",
            p.ansatz.name(),
            p.length,
            p.fci,
            p.minimum
        );
    }
    let _ = writeln!(out, "],\n\"runs\": [");
    for (i, r) in runs.iter().enumerate() {
        let comma = if i + 1 < runs.len() { "," } else { "" };
        let _ = writeln!(
            out,
            "  {{\"ansatz\": \"{}\", \"length\": {}, \"optimizer\": \"{}\", \"seed\": {}, \
             \"error\": {}, \"exact_error\": {}, \"ansatz_error\": {}, \"circuits\": {}}}{comma}",
            r.ansatz.name(),
            r.length,
            r.optimizer.key(),
            r.seed,
            r.error,
            r.exact_error,
            r.ansatz_error,
            r.circuits
        );
    }
    let _ = writeln!(out, "]}}");
    out
}
