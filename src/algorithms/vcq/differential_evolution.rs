// // Differential Evolution algorithm implementation moved from lib.rs
use ndarray::{Array1, Axis};
use rand::{thread_rng, seq::SliceRandom, Rng};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyInt, PyModule, IntoPyDict};

use crate::algorithms::{AlgorithmTrait, AlgorithmArgs};
use crate::infrastructure::{Infrastructure, QuantumRunner, LocalRunner, CunqaRunner};

pub struct AlgorithmDifferentialEvolution;

pub struct AlgorithmDifferentialEvolutionArgs<'py> {
	pub base: AlgorithmArgs<'py>,
	pub population_size: Option<usize>,
	pub generations: Option<usize>,
	pub dimensions: Option<usize>,
	pub expectation_function: Option<Bound<'py, PyAny>>,
	pub tolerance: Option<f64>,
}
	
impl AlgorithmTrait for AlgorithmDifferentialEvolution {

    type Args<'py> = AlgorithmDifferentialEvolutionArgs<'py>;
    type AlgorithmReturnType = PyObject;

    fn run<'py>(&self, args: AlgorithmDifferentialEvolutionArgs<'py>) -> Self::AlgorithmReturnType {
		// Implementation migrated from lib.rs
		let AlgorithmDifferentialEvolutionArgs { base, population_size, generations, dimensions, expectation_function, tolerance } = args;
		let popsize = population_size.unwrap_or(20);
		let max_generations = generations.unwrap_or(100);
		let dimensions = dimensions.unwrap_or(2);
		let tolerance = tolerance.unwrap_or(0.01);
		let expectation_function = expectation_function.expect("expectation_function required");
		let mut rng = thread_rng();
		let mut pop = ndarray::Array2::<f64>::zeros((popsize, dimensions));
		let mut fitness = vec![f64::MIN; popsize];
		let mut best_idx = 0;
		let mut best = vec![0.0; dimensions];
		for _generation in 0..max_generations {
			for i in 0..popsize {
				let ids: Vec<usize> = (0..popsize).filter(|&idx| idx != i).collect();
				let selected_ids: Vec<usize> = ids.choose_multiple(&mut rng, 3).cloned().collect();
				let c1 = pop.row(selected_ids[0]).to_owned();
				let c2 = pop.row(selected_ids[1]).to_owned();
				let c3 = pop.row(selected_ids[2]).to_owned();
				let mut mutant = &c1 + 0.8 * (&c2 - &c3);
				mutant.mapv_inplace(|x| x.max(0.0).min(1.0));
				let cross_points: Array1<bool> = Array1::from_shape_fn(dimensions, |_| rng.gen_bool(0.7));
				let trial = cross_points.iter().zip(mutant.iter()).zip(pop.row(i).iter()).map(|((cp, m), p)| if *cp { *m } else { *p }).collect::<Array1<f64>>();
				let trial_denorm: Vec<f64> = trial.to_vec();
				let qc_assigned = Python::with_gil(|py| {
					let params_py = trial_denorm.clone();
					let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
					let qc_assigned = base.qc.call_method("assign_parameters", (params_py,), Some(&kwargs));
					match qc_assigned {
						Ok(bound) => bound.unbind(),
						Err(e) => {
							// error!("Error assigning parameters: {e}");
							panic!("Error assigning parameters: {e}");
						},
					}
				});
				let expectation = Python::with_gil(|py| {
					let shots_py = PyInt::new(py, base.shots.clone().unwrap_or(PyInt::new(py, 1)).extract::<u32>().unwrap_or(1));
					let qc_bound = qc_assigned.into_bound(py);
					let args = AlgorithmArgs {
						id: base.id.clone(),
						qc: qc_bound,
						shots: Some(shots_py),
						n_qpus: base.n_qpus,
						infrastructure: base.infrastructure.clone(),
                        backend: "AerSimulator".to_string(),
						nodes: base.nodes,
					};
					// let single_run_algo = crate::algorithms::AlgorithmSingleRun;
					// let result = single_run_algo.run(args);

                    let infra = Infrastructure::from_str(&args.infrastructure);
                    let runner: Box<dyn QuantumRunner> = match infra {
                        Infrastructure::Local => Box::new(LocalRunner),
                        Infrastructure::Cunqa => Box::new(CunqaRunner),
                    };
                    let result = runner.run(&args);


					let qaoa_utils = PyModule::import(py, "polypus_python").expect("Failed to import polypus_python");
					let expectation = qaoa_utils.call_method("expectation_value", (result, &expectation_function), None);
					match expectation {
						Ok(val) => val.extract::<f64>().unwrap_or(f64::MIN),
						Err(e) => {
							// error!("Error calling expectation_value: {e}");
							panic!("Error calling expectation_value: {e}");
						},
					}
				});
				if expectation > fitness[i] {
					fitness[i] = expectation;
					pop.row_mut(i).assign(&trial);
					if expectation > fitness[best_idx] {
						best_idx = i;
						best = trial_denorm.clone();
					}
				}
			}
			let mean = pop.mean_axis(Axis(0)).map(|a| a.sum()).unwrap_or(0.0);
			let std = pop.std_axis(Axis(0), 0.0).iter().sum::<f64>();
			if std < (tolerance * mean) {
				break;
			}
		}
		Python::with_gil(|py| {
			match best.into_pyobject(py) {
				Ok(obj) => obj.unbind(),
				Err(e) => {
					// error!("Error converting best to result: {}", e);
					panic!("Error converting best to result: {}", e);
				}
			}
		})
	}
	
	fn name(&self) -> String {
		String::from("Differential Evolution")
	}
	fn description(&self) -> String {
		String::from("This algorithm is designed to train a QAOA circuit using differential evolution.")
	}
}