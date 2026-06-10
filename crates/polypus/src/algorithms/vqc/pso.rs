use ndarray::{Array2, Axis};
use rand::{thread_rng, Rng};
use pyo3::prelude::*;
use crate::algorithms::AlgorithmTrait;
use crate::evaluation::EvaluationOracle;

pub struct AlgorithmPSO;

/// Arguments for the Particle Swarm Optimization optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// all evaluation is delegated to `oracle`.
pub struct AlgorithmPSOArgs {
    /// Oracle that maps parameter vectors → fitness values.
    pub oracle: Box<dyn EvaluationOracle>,
    pub population_size: u32,
    pub generations: u32,
    pub dimensions: u32,
    pub bounds: (f64, f64),
    pub inertia_weight: f64,
    pub cognitive_weight: f64,
    pub social_weight: f64,
    pub tolerance: f64,
}

impl AlgorithmTrait for AlgorithmPSO {
    type Args = AlgorithmPSOArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmPSOArgs) -> PyObject {
        let AlgorithmPSOArgs {
            oracle,
            population_size,
            generations,
            dimensions,
            bounds,
            inertia_weight,
            cognitive_weight,
            social_weight,
            tolerance,
        } = args;

        let popsize = population_size as usize;
        let max_gen = generations as usize;
        let dims = dimensions as usize;
        let (lb, ub) = bounds;
        let max_vel = (ub - lb) * 0.2;
        let vel_range = (ub - lb) * 0.1;

        let mut rng = thread_rng();

        // Initialise positions uniformly in [lb, ub)
        let mut positions = Array2::<f64>::zeros((popsize, dims));
        for mut row in positions.outer_iter_mut() {
            for e in row.iter_mut() {
                *e = rng.gen_range(lb..ub);
            }
        }

        // Initialise velocities uniformly in [-vel_range, vel_range)
        let mut velocities = Array2::<f64>::zeros((popsize, dims));
        for mut row in velocities.outer_iter_mut() {
            for e in row.iter_mut() {
                *e = rng.gen_range(-vel_range..vel_range);
            }
        }

        // Evaluate initial population
        let init_candidates: Vec<Vec<f64>> = positions.outer_iter().map(|r| r.to_vec()).collect();
        let mut personal_best_fitness = oracle.evaluate_batch(&init_candidates);
        let mut personal_best_positions = positions.clone();

        let mut global_best_idx = personal_best_fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut global_best_pos = personal_best_positions.row(global_best_idx).to_vec();

        for generation in 0..max_gen {
            // Update velocities and positions
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

            // Evaluate new positions in a single oracle call
            let candidates: Vec<Vec<f64>> = new_positions.outer_iter().map(|r| r.to_vec()).collect();
            let new_fitness = oracle.evaluate_batch(&candidates);

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
                .map(|(i, _)| i)
                .unwrap_or(0);
            global_best_pos = personal_best_positions.row(global_best_idx).to_vec();

            positions = new_positions;
            velocities = new_velocities;

            let mean = positions
                .mean_axis(Axis(0))
                .expect("Failed to compute mean")
                .sum();
            let std: f64 = positions.std_axis(Axis(0), 0.0).iter().sum();
            println!("Generation {generation}: Mean: {mean:.4}, Std: {std:.4}");
            if std < tolerance * mean {
                println!("Stopping early at generation {generation} due to convergence");
                break;
            }
        }

        Python::with_gil(|py| {
            global_best_pos
                .into_pyobject(py)
                .expect("Error converting best to PyObject")
                .unbind()
        })
    }

    fn name(&self) -> String {
        String::from("Particle Swarm Optimization")
    }

    fn description(&self) -> String {
        String::from("Trains a variational quantum circuit using Particle Swarm Optimization (PSO).")
    }
}

