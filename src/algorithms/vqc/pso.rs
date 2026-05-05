use ndarray::{Array2, Axis};
use rand::{thread_rng, Rng};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, IntoPyDict};
use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

pub struct AlgorithmPSO;

/// Arguments required to run the Particle Swarm Optimization algorithm.
/// - base: Base algorithm arguments.
/// - population_size: Number of particles in the swarm.
/// - generations: Maximum number of generations.
/// - dimensions: Number of dimensions for each particle (e.g. 2*p_layers for QAOA).
/// - bounds: Lower and upper bounds for each dimension.
/// - inertia_weight: Controls the influence of the previous velocity.
/// - cognitive_weight: Controls the pull towards the particle's personal best.
/// - social_weight: Controls the pull towards the global best.
/// - expectation_function: Python function to compute the expectation value.
/// - tolerance: Tolerance for early convergence stopping.
pub struct AlgorithmPSOArgs {
    pub base: AlgorithmArgs,
    pub population_size: u32,
    pub generations: u32,
    pub dimensions: u32,
    pub bounds: (f64, f64),
    pub inertia_weight: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
    pub expectation_function: Py<PyAny>,
    pub tolerance: f64,
}

/// Assign `params` to the template circuit and run a batch of particles through the runner.
/// Returns a Vec<f64> of expectation values (one per particle in `particle_ids`).
fn evaluate_batch(
    base: &AlgorithmArgs,
    runner: &dyn QuantumRunner,
    expectation_function: &Py<PyAny>,
    positions: &Array2<f64>,
    particle_ids: std::ops::Range<usize>,
) -> Vec<f64> {
    let mut qcs_batch: Vec<Py<PyAny>> = Vec::new();
    let mut params_batch: Vec<Vec<f64>> = Vec::new();

    for particle_id in particle_ids.clone() {
        let params: Vec<f64> = positions.row(particle_id).to_vec();
        params_batch.push(params.clone());

        let qc_assigned = Python::with_gil(|py| {
            let qc_any = match base.qcs[0].clone_ref(py).into_pyobject(py) {
                Ok(obj) => obj,
                Err(e) => panic!("Error converting QC to PyObject: {e}"),
            };
            let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
            match qc_any.call_method("assign_parameters", (params,), Some(&kwargs)) {
                Ok(bound) => bound.unbind(),
                Err(e) => panic!("Error assigning parameters: {e}"),
            }
        });
        qcs_batch.push(qc_assigned);
    }

    let batch_args = AlgorithmArgs {
        id: base.id.clone(),
        qcs: qcs_batch,
        shots: base.shots,
        n_qpus: base.n_qpus,
        infrastructure: base.infrastructure.clone(),
        backend: base.backend.clone(),
        nodes: base.nodes,
        cores_per_qpu: base.cores_per_qpu,
    };

    let running_result = runner.run(&batch_args);

    Python::with_gil(|py| {
        let qaoa_utils = match PyModule::import(py, "polypus_python") {
            Ok(module) => module,
            Err(e) => panic!("Failed to import polypus_python: {e}"),
        };
        let expectations = qaoa_utils.call_method(
            "expectation_values",
            (running_result, expectation_function),
            None,
        );
        match expectations {
            Ok(result) => match result.extract::<Vec<f64>>() {
                Ok(vals) => vals,
                Err(e) => panic!("Failed to extract expectation values: {e}"),
            },
            Err(e) => panic!("Error computing expectation values: {e}"),
        }
    })
}

/// Evaluate all particles in `positions` in batches of `n_qpus`.
/// Returns a Vec<f64> of expectation values indexed by particle.
fn evaluate_population(
    base: &AlgorithmArgs,
    runner: &dyn QuantumRunner,
    expectation_function: &Py<PyAny>,
    positions: &Array2<f64>,
) -> Vec<f64> {
    let popsize = positions.nrows();
    let n_qpus = base.n_qpus as usize;
    let mut fitness = vec![0.0f64; popsize];

    for i in (0..popsize).step_by(n_qpus) {
        let end = (i + n_qpus).min(popsize);
        let batch_fitness = evaluate_batch(base, runner, expectation_function, positions, i..end);
        for (local_idx, particle_id) in (i..end).enumerate() {
            fitness[particle_id] = batch_fitness[local_idx];
        }
    }

    fitness
}

impl AlgorithmTrait for AlgorithmPSO {
    type Args = AlgorithmPSOArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmPSOArgs) -> Self::AlgorithmReturnType {
        let AlgorithmPSOArgs {
            base,
            population_size,
            generations,
            dimensions,
            bounds,
            inertia_weight,
            cognitive_weight,
            social_weight,
            expectation_function,
            tolerance,
        } = args;

        let popsize = population_size as usize;
        let max_generations = generations as usize;
        let dims = dimensions as usize;
        let (lb, ub) = bounds;
        let max_vel = (ub - lb) * 0.2;
        let vel_range = (ub - lb) * 0.1;

        let mut rng = thread_rng();

        // --- Initialise positions uniformly in [lb, ub) ---
        let mut positions = Array2::<f64>::zeros((popsize, dims));
        for mut row in positions.outer_iter_mut() {
            for elem in row.iter_mut() {
                *elem = rng.gen_range(lb..ub);
            }
        }

        // --- Initialise velocities uniformly in [-vel_range, vel_range) ---
        let mut velocities = Array2::<f64>::zeros((popsize, dims));
        for mut row in velocities.outer_iter_mut() {
            for elem in row.iter_mut() {
                *elem = rng.gen_range(-vel_range..vel_range);
            }
        }

        println!("Running PSO with infrastructure: {}", &base.infrastructure);
        let infra = Infrastructure::from_str(&base.infrastructure);
        let runner: Box<dyn QuantumRunner> = match infra {
            Infrastructure::Local => Box::new(LocalRunner),
            Infrastructure::Cunqa => Box::new(CunqaRunner::new(base.n_qpus, base.nodes, &base.id, base.cores_per_qpu)),
        };

        // --- Evaluate initial population to seed personal bests ---
        let initial_fitness =
            evaluate_population(&base, runner.as_ref(), &expectation_function, &positions);

        let mut personal_best_positions = positions.clone();
        let mut personal_best_fitness = initial_fitness;

        // --- Locate initial global best (highest expectation value) ---
        let mut global_best_idx = personal_best_fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let mut global_best_pos: Vec<f64> =
            personal_best_positions.row(global_best_idx).to_vec();

        // --- Main PSO loop ---
        for generation in 0..max_generations {
            let mut new_positions = Array2::<f64>::zeros((popsize, dims));
            let mut new_velocities = Array2::<f64>::zeros((popsize, dims));

            for i in 0..popsize {
                for d in 0..dims {
                    let r1: f64 = rng.gen();
                    let r2: f64 = rng.gen();

                    let new_vel = inertia_weight * velocities[[i, d]]
                        + cognitive_weight * r1 * (personal_best_positions[[i, d]] - positions[[i, d]])
                        + social_weight * r2 * (global_best_pos[d] - positions[[i, d]]);

                    let new_vel = new_vel.max(-max_vel).min(max_vel);
                    let new_pos = (positions[[i, d]] + new_vel).max(lb).min(ub);

                    new_velocities[[i, d]] = new_vel;
                    new_positions[[i, d]] = new_pos;
                }
            }

            // Evaluate the updated swarm
            let new_fitness =
                evaluate_population(&base, runner.as_ref(), &expectation_function, &new_positions);

            // Update personal bests
            for i in 0..popsize {
                if new_fitness[i] > personal_best_fitness[i] {
                    personal_best_fitness[i] = new_fitness[i];
                    personal_best_positions.row_mut(i).assign(&new_positions.row(i));
                }
            }

            // Recompute global best after all personal bests are updated
            global_best_idx = personal_best_fitness
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            global_best_pos = personal_best_positions.row(global_best_idx).to_vec();

            positions = new_positions;
            velocities = new_velocities;

            // --- Convergence check (mirrors DE approach) ---
            let mean = positions
                .mean_axis(Axis(0))
                .expect("Failed to compute mean");
            let mean_sum = mean.sum();
            let std = positions.std_axis(Axis(0), 0.0);
            let std_sum: f64 = std.iter().sum();

            println!(
                "Generation {}: Mean: {:?}, Std: {:?}",
                generation, mean_sum, std_sum
            );

            if std_sum < tolerance * mean_sum {
                println!(
                    "Stopping early at generation {} due to convergence",
                    generation
                );
                break;
            }
        }

        runner.close();

        Python::with_gil(|py| {
            match global_best_pos.into_pyobject(py) {
                Ok(obj) => obj.unbind(),
                Err(e) => panic!("Error converting best to result: {}", e),
            }
        })
    }

    fn name(&self) -> String {
        String::from("Particle Swarm Optimization")
    }

    fn description(&self) -> String {
        String::from(
            "This algorithm trains a QAOA circuit using Particle Swarm Optimization (PSO).",
        )
    }
}
