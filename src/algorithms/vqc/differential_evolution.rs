use ndarray::{Array1, Array2, Axis};
use rand::{thread_rng, seq::SliceRandom, Rng};
use pyo3::prelude::*;
use std::f64::consts::PI;
use crate::algorithms::AlgorithmTrait;
use crate::evaluation::EvaluationOracle;

pub struct AlgorithmDifferentialEvolution;

/// Arguments for the Differential Evolution optimizer.
///
/// The algorithm is completely decoupled from circuits and infrastructure:
/// all evaluation is delegated to \.
pub struct AlgorithmDifferentialEvolutionArgs {
    /// Oracle that maps parameter vectors to fitness values.
    pub oracle: Box<dyn EvaluationOracle>,
    pub population_size: u32,
    pub generations: u32,
    pub dimensions: u32,
    pub tolerance: f64,
}

impl AlgorithmTrait for AlgorithmDifferentialEvolution {
    type Args = AlgorithmDifferentialEvolutionArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmDifferentialEvolutionArgs) -> PyObject {
        let AlgorithmDifferentialEvolutionArgs {
            oracle,
            population_size,
            generations,
            dimensions,
            tolerance,
        } = args;

        let popsize = population_size as usize;
        let max_gen = generations as usize;
        let dims = dimensions as usize;
        let mut rng = thread_rng();

        // Initialise population with angles in [0, 2pi)
        let mut pop = Array2::<f64>::zeros((popsize, dims));
        for mut row in pop.outer_iter_mut() {
            for e in row.iter_mut() {
                *e = rng.gen_range(0.0..2.0 * PI);
            }
        }

        // Evaluate initial population (fixes the bug of starting fitness at 0.0)
        let init_candidates: Vec<Vec<f64>> = pop.outer_iter().map(|r| r.to_vec()).collect();
        let mut fitness = oracle.evaluate_batch(&init_candidates);
        let mut best_idx = fitness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut best = pop.row(best_idx).to_vec();

        for generation in 0..max_gen {
            // Build all trial vectors for this generation.
            let mut trials: Vec<Array1<f64>> = Vec::with_capacity(popsize);
            for i in 0..popsize {
                let ids: Vec<usize> = (0..popsize).filter(|&j| j != i).collect();
                let sel: Vec<usize> = ids.choose_multiple(&mut rng, 3).cloned().collect();
                let c1 = pop.row(sel[0]).to_owned();
                let c2 = pop.row(sel[1]).to_owned();
                let c3 = pop.row(sel[2]).to_owned();
                let mut mutant = &c1 + 0.8 * (&c2 - &c3);
                mutant.mapv_inplace(|x| x.rem_euclid(2.0 * PI));
                let cross: Array1<bool> = Array1::from_shape_fn(dims, |_| rng.gen_bool(0.7));
                let trial = cross
                    .iter()
                    .zip(mutant.iter())
                    .zip(pop.row(i).iter())
                    .map(|((cp, m), p)| if *cp { *m } else { *p })
                    .collect::<Array1<f64>>();
                trials.push(trial);
            }

            // Evaluate all trials in a single oracle call.
            let trial_candidates: Vec<Vec<f64>> = trials.iter().map(|t| t.to_vec()).collect();
            let trial_fitness = oracle.evaluate_batch(&trial_candidates);

            // Selection
            for i in 0..popsize {
                if trial_fitness[i] > fitness[i] {
                    fitness[i] = trial_fitness[i];
                    pop.row_mut(i).assign(&trials[i]);
                    if trial_fitness[i] > fitness[best_idx] {
                        best_idx = i;
                        best = trials[i].to_vec();
                    }
                }
            }

            let mean = pop.mean_axis(Axis(0)).expect("Failed to compute mean").sum();
            let std: f64 = pop.std_axis(Axis(0), 0.0).iter().sum();
            println!("Generation {generation}: Mean: {mean:.4}, Std: {std:.4}");
            if std < tolerance * mean {
                println!("Stopping early at generation {generation} due to convergence");
                break;
            }
        }

        Python::with_gil(|py| {
            best.into_pyobject(py)
                .expect("Error converting best to PyObject")
                .unbind()
        })
    }

    fn name(&self) -> String {
        String::from("Differential Evolution")
    }

    fn description(&self) -> String {
        String::from("Trains a variational quantum circuit using Differential Evolution.")
    }
}
