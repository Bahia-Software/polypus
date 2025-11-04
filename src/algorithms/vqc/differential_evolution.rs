use ndarray::{Array1, Axis};
use rand::{thread_rng, seq::SliceRandom, Rng};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, IntoPyDict};
use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

pub struct AlgorithmDifferentialEvolution;

/// Arguments required to run the Differential Evolution algorithm.
/// - base: Base algorithm arguments.
/// - population_size: Size of the population.
/// - generations: Number of generations to run.
/// - dimensions: Number of dimensions for each individual.
/// - expectation_function: Python function to compute the expectation value.
/// - tolerance: Tolerance for convergence.
pub struct AlgorithmDifferentialEvolutionArgs{
	pub base: AlgorithmArgs,
	pub population_size: u32,
	pub generations: u32,
	pub dimensions: u32,
	pub expectation_function: Py<PyAny>,
	pub tolerance: f64,
}
	
impl AlgorithmTrait for AlgorithmDifferentialEvolution {

    type Args = AlgorithmDifferentialEvolutionArgs;
    type AlgorithmReturnType = PyObject;

    fn run(&self, args: AlgorithmDifferentialEvolutionArgs) -> Self::AlgorithmReturnType {
		let AlgorithmDifferentialEvolutionArgs {base, population_size, generations, dimensions, expectation_function, tolerance } = args;
		let popsize = population_size as usize;
		let max_generations = generations as usize;
		let dimensions = dimensions as usize;
		let tolerance = tolerance;
		let expectation_function = expectation_function;
		let mut rng = thread_rng();
		let mut pop = ndarray::Array2::<f64>::zeros((popsize, dimensions));
		for mut row in pop.outer_iter_mut() {
			for elem in row.iter_mut() {
				*elem = rng.gen_range(0.0..1.0);
			}
		}
		let pop_denorm = pop.clone();

         // Initialize fitness vector
		let mut fitness: Vec<f64> = vec![0.0; popsize];

		// Best index and best individual
		let mut best_idx = 0;
		let mut best = pop_denorm.row(best_idx).to_vec().clone();

		println!("Running with infrastructure: {}", &base.infrastructure);
		let infra = Infrastructure::from_str(&base.infrastructure);
		let runner: Box<dyn QuantumRunner> = match infra {
			Infrastructure::Local => Box::new(LocalRunner),
			Infrastructure::Cunqa => Box::new(CunqaRunner::new(base.n_qpus, base.nodes, &base.id)),
		};
		for generation in 0..max_generations {
			for i in (0..popsize).step_by(base.n_qpus as usize) {
				let end = (i + base.n_qpus as usize).min(popsize);
				let batch_ids = i..end;

				// Create trial for each individual in the batch and assign parameters to the quantum circuits
				let mut qcs_individuals: Vec<Py<PyAny>> = Vec::new();
				let mut trials_batch: Vec<Array1<f64>> = Vec::new();
				let mut trials_denorms_batch: Vec<Vec<f64>> = Vec::new();
				for _trial_id in batch_ids.clone() {					
					let ids: Vec<usize> = (0..popsize).filter(|&idx| idx != i).collect();
					let selected_ids: Vec<usize> = ids.choose_multiple(&mut rng, 3).cloned().collect();
					let c1 = pop.row(selected_ids[0]).to_owned();
					let c2 = pop.row(selected_ids[1]).to_owned();
					let c3 = pop.row(selected_ids[2]).to_owned();
					let mut mutant = &c1 + 0.8 * (&c2 - &c3);
					mutant.mapv_inplace(|x| x.max(0.0).min(1.0));
					let cross_points: Array1<bool> = Array1::from_shape_fn(dimensions, |_| rng.gen_bool(0.7));
					let trial = cross_points.iter().zip(mutant.iter()).zip(pop.row(i).iter()).map(|((cp, m), p)| if *cp { *m } else { *p }).collect::<Array1<f64>>();
					trials_batch.push(trial.clone());
					let trial_denorm: Vec<f64> = trial.to_vec();
					trials_denorms_batch.push(trial_denorm.clone());
					let qc_assigned = Python::with_gil(|py| {
						let qc_any = match base.qcs[0].clone_ref(py).into_pyobject(py) {
							Ok(obj) => obj,
							Err(e) => {
								panic!("Error converting QC to PyObject: {e}");
							}
						};
						let params_py = trial_denorm.clone();
						let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
						let qc_assigned = qc_any.call_method("assign_parameters", (params_py,), Some(&kwargs));
						match qc_assigned {
							Ok(bound) => bound.unbind(),
							Err(e) => {
								panic!("Error assigning parameters: {e}");
							},
						}
					});
					qcs_individuals.push(qc_assigned);
				}

				// Run
				let args = AlgorithmArgs {
					id: base.id.clone(),
					qcs: qcs_individuals,
					shots: base.shots,
					n_qpus: base.n_qpus,
					infrastructure: base.infrastructure.clone(),
					backend: "AerSimulator".to_string(),
					nodes: base.nodes,
				};

				let running_result = runner.run(&args);
				let _expectation = Python::with_gil(|py| {
					let qaoa_utils = match PyModule::import(py, "polypus_python") {
						Ok(module) => module,
						Err(e) => {
							panic!("Failed to import polypus_python: {e}");
						}
					};
					let expectations = qaoa_utils.call_method("expectation_values", (running_result, &expectation_function), None);
					let expectation_values = match expectations{
						Ok(result) => result,
						Err(e) => {
							panic!("{e}, Error serializing qc");
						},
					};
					let expectation_values = match expectation_values.extract::<Vec<f64>>() {
						Ok(val) => val,
						Err(e) => {
							panic!("Failed to extract float from expectation_value: {e}");
						}
					};
					// Update fitness and population
					for (idx, trial_id) in batch_ids.clone().enumerate() {
						if expectation_values[idx] > fitness[trial_id] {
							fitness[trial_id] = expectation_values[idx];
							pop.row_mut(trial_id).assign(&trials_batch[idx]);
							if expectation_values[idx] > fitness[best_idx] {
								best_idx = trial_id;
								best = trials_denorms_batch[idx].clone();
							}
						}
					}
				});

			}
			let mean = pop.mean_axis(Axis(0)).expect("Failed to compute mean");
			let mean = mean.sum();
			let std = pop.std_axis(Axis(0), 0.0);
			let std: f64 = std.iter().sum();
			println!("Generation {}: Mean: {:?}, Std: {:?}", generation, mean, std);	
			let converged = std < (tolerance * mean);
			if converged {
				println!("Stopping early at generation {} due to convergence", generation);
				break; 
			} 
		}

		// Close
		runner.close();

		let result = Python::with_gil(|py| {
            match best.into_pyobject(py) {
               Ok(obj) => obj.unbind(),
               Err(e) => {
                //  error!("Error converting best to result: {}", e);
                 panic!("Error converting best to result: {}", e);
               }
            }
         });
   
		return result;		
	}
	
	fn name(&self) -> String {
		String::from("Differential Evolution")
	}
	
	fn description(&self) -> String {
		String::from("This algorithm is designed to train a QAOA circuit using differential evolution.")
	}
}