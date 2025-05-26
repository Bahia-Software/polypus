use pyo3::prelude::*;
use pyo3::types::PyInt;
use pyo3::types::{PyDict, IntoPyDict};
use std::fmt;
use std::process::{Command, Stdio};
use serde_json::Value;
use std::collections::HashMap;
use ndarray::{Array1,Array2};
use rand::Rng;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand::distributions::Bernoulli;
use log::{debug, error};
use std::io::ErrorKind;

pub mod algorithms {
   use super::*; 

   pub struct AlgorithmArgs<'py> {
      pub qc: Bound<'py, PyAny>,
      pub shots: Option<Bound<'py, PyInt>>,
      pub n_qpus: Option<Bound<'py, PyInt>>,
   }

   pub struct AlgorithmDifferentialEvolutionArgs<'py> {
      pub base: AlgorithmArgs<'py>,
      pub population_size: Option<usize>,
      pub generations: Option<usize>,
      pub dimensions: Option<usize>,
      pub expectation_function: Option<Bound<'py, PyAny>>,
   }

   // Algorithm Trait Definition
   pub trait AlgorithmTrait {
      type Args<'py>;
      type AlgorithmReturnType: fmt::Display;

      fn run<'py>(&self, args: Self::Args<'py>) -> Self::AlgorithmReturnType;

      fn name(&self) -> String;

      fn description(&self) -> String;
   }

   fn get_num_cpus() -> usize {
      // Try to get the number of logical CPUs dynamically
      match std::thread::available_parallelism() {
         Ok(n) => n.get(),
         Err(e) => {
            error!("Failed to get number of CPUs: {}", e);
            32
         }
      }
   }

   // Single Run 
   pub struct AlgorithmSingleRun;

   impl AlgorithmTrait for AlgorithmSingleRun {
      type Args<'py> = AlgorithmArgs<'py>;
      type AlgorithmReturnType = PyObject;
      
      fn run<'py>(&self, args: AlgorithmArgs<'py>) -> Self::AlgorithmReturnType {
         // Call the run_qc method from the polypus_python module
         let qc = args.qc;
         let result = Python::with_gil(|py| -> PyObject {
            let module = PyModule::import(py, "polypus_python").unwrap();
            let running_result = module.call_method("run_qc", (qc,), None);

            match running_result{
               Ok(result) => result.unbind(),
                  Err(e) => {
                    error!("Error running run_qc: {e}");
                    panic!("{e}, Error running run_qc")
                  },
            }
         });
         result
      }

      fn name(&self) -> String {
         String::from("Single Run Algorithm")
      }

      fn description(&self) -> String {
         String::from("Base algorithm to run a quantum circuit")
      }
   }

   pub struct AlgorithmDistributeByShots;

   impl AlgorithmTrait for AlgorithmDistributeByShots {
      type Args<'py> = AlgorithmArgs<'py>;
      type AlgorithmReturnType = PyObject;

      fn run<'py>(&self, args: AlgorithmArgs<'py>) -> Self::AlgorithmReturnType {
         // Call the run_qc method from the polypus_python module
         let qc = args.qc;
          let n_shots = match args.shots.as_ref() {
            Some(val) => val,
            None => {
               error!("shots is required");
               panic!("shots is required");
            }
          };
         let n_shots_val = n_shots.extract::<i64>().unwrap_or(1);
          let n_qpus = match args.n_qpus.as_ref() {
            Some(val) => val,
            None => {
               error!("n_qpus is required");
               panic!("n_qpus is required");
            }
          };
         let n_qpus_val = n_qpus.extract::<i64>().unwrap_or(1);
         
         // Compute number of shots per QPU
         let shots_per_qpu = n_shots_val / n_qpus_val;

         // Dynamically get number of logical CPUs
         let n_qpus_val = n_qpus_val as usize;  
         let num_cores = get_num_cpus();  
         debug!("Number of QPUs: {}", n_qpus_val);
         let cores_per_worker = num_cores / n_qpus_val;

         // 1. Serialize the quantum circuit
         Python::with_gil(|py| {
            let module = PyModule::import(py, "polypus_python").unwrap();
            let running_result = module.call_method("serialize_quantum_circuit", (&qc,), None);
            match running_result {
               Ok(result) => result,
               Err(e) => {
                 error!("Error serializing qc: {e}");
                 panic!("{e}, Error serializing qc")
               },
            };
         });

         // 2. Create a number of processes based on the number of QPUs and run the quantum circuits
         let mut handles = Vec::new();
         for i in 0..n_qpus_val {
            let start_core = i * cores_per_worker;
            let mut end_core = start_core + cores_per_worker - 1;
            if i == n_qpus_val - 1 {
               end_core = num_cores - 1;
            }
            let core_range = format!("{}-{}", start_core, end_core);
            debug!("Starting process {i} for QPU {i} on cores {core_range}");

            let process = Command::new("taskset")
               .arg("-c")
               .arg(core_range)
               .arg("python")
               .arg("packages/polypus_python/polypus_python/run_worker.py")
               .arg("--shots")
               .arg(format!("{}", shots_per_qpu))
               .arg("--max_parallel_threads")
               .arg(format!("{}", cores_per_worker))
               .env("OMP_NUM_THREADS", format!("{}", cores_per_worker))
               .stdout(Stdio::piped())
               .spawn()
               .expect("Failed to start worker process");
            handles.push((i, process));
         }

         // 3. Collect outputs
         let mut parsed_outputs = Vec::new();
         for (i, process) in handles {
            let output = match process.wait_with_output() {
               Ok(output) => output,
               Err(e) => {
                 error!("Failed to wait on child process {}: {}", i, e);
                 panic!("Failed to wait on child process {}: {}", i, e);
               }
            };
            
            let stdout = String::from_utf8_lossy(&output.stdout);

            // Get JSON an convert to a dictionary
            if let Some(json_line) = stdout.lines().find(|line| line.trim_start().starts_with('{')) {
               match serde_json::from_str::<Value>(json_line) {
                     Ok(json) => {
                        parsed_outputs.push(json);
                     }
                     Err(e) => {
                        error!("Process {i} output is not valid JSON: {e}\nRaw output:\n{stdout}");
                        panic!("Process {i} output is not valid JSON: {e}\nRaw output:\n{stdout}");
                     }
               }
            } else {
                  error!("Process {i} output does not contain JSON:\n{stdout}");
                  panic!("Process {i} output does not contain JSON:\n{stdout}");
            }
         }

         // Sum the counts from all dictionaries
         let mut total_counts: HashMap<String, u64> = HashMap::new();
         for dict in &parsed_outputs {
            if let Some(obj) = dict.as_object() {
               for (key, value) in obj {
                  if let Some(count) = value.as_u64() {
                     *total_counts.entry(key.clone()).or_insert(0) += count;
                  }
               }
            }
         }

         // 4. Convert the dictionary to a PyObject
         let py_dict = Python::with_gil(|py| -> PyObject {
            let dict = PyDict::new(py);
            for (key, value) in total_counts {
               dict.set_item(key, value).unwrap();
            }
            // let py_dict = dict.into_pyobject(py);
            // py_dict
            dict.unbind().into()
         });

         // 5. Return the result
         return py_dict;
         
      }

      fn name(&self) -> String {
         String::from("Distribute By Shots Algorithm")
      }

      fn description(&self) -> String {
         String::from("Algorithm to distribute a quantum circuit by shots")
      }
   }


   pub struct AlgorithmDifferentialEvolution;

   impl AlgorithmTrait for AlgorithmDifferentialEvolution {
      type Args<'py> = AlgorithmDifferentialEvolutionArgs<'py>;
      type AlgorithmReturnType = PyObject;

      fn run<'py>(&self, args: AlgorithmDifferentialEvolutionArgs<'py>) -> Self::AlgorithmReturnType {

         // Process input arguments
         let qc = args.base.qc;
         let shots = match args.base.shots.as_ref() {
            Some(val) => val,
            None => {
               error!("shots is required");
               panic!("shots is required");
            }
          };
         let shots_val = shots.extract::<i64>().unwrap_or(1);
         let n_qpus = match args.base.n_qpus.as_ref() {
            Some(val) => val,
            None => {
               let err_msg = "n_qpus is required";
               error!("{}", err_msg);
               panic!("{}", err_msg);
            }
         };
         let n_qpus_val = n_qpus.extract::<i64>().unwrap_or(1);
         let expectation_function = args.expectation_function;

         // DE parameters
         let mut_factor = 0.8;
         let crossp = 0.7;
         let popsize = match args.population_size {
            Some(val) => val,
            None => {
               let err_msg = "population_size is required";
               error!("{}", err_msg);
               panic!("{}", err_msg);
            }
          };
         let its = match args.generations {
            Some(val) => val,
            None => {
               let err_msg = "generations is required";
               error!("{}", err_msg);
               panic!("{}", err_msg);
            }
          };
         let dimensions = match args.dimensions {
            Some(val) => val,
            None => {
               let err_msg = "dimensions is required";
               error!("{}", err_msg);
               panic!("{}", err_msg);
            }
         };

         // Initialize population
         let mut rng = rand::thread_rng();
         let mut pop = Array2::<f64>::zeros((popsize, dimensions));
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

         // Train
         let mut rng = thread_rng();
         for generation in 0..its {
            for i in 0..popsize {
               let ids: Vec<usize> = (0..popsize).filter(|&idx| idx != i).collect();
               let selected_ids: Vec<usize> = ids.choose_multiple(&mut rng, 3).cloned().collect();
               let c1 = pop.row(selected_ids[0]).to_owned();
               let c2 = pop.row(selected_ids[1]).to_owned();
               let c3 = pop.row(selected_ids[2]).to_owned();
               let mut mutant = &c1 + mut_factor * (&c2 - &c3);
               mutant.mapv_inplace(|x| x.max(0.0).min(1.0));
               let cross_points: Array1<bool> = Array1::from_shape_fn(dimensions,|_| rng.sample(Bernoulli::new(crossp).unwrap()));
               let trial = cross_points.iter().zip(mutant.iter()).zip(pop.row(i).iter()).map(|((cp, m), p)| if *cp { *m } else { *p }).collect::<Array1<f64>>();
               let trial_denorm = trial.clone();
               let trial_denorm: Vec<f64> = trial_denorm.to_vec();

               // Assign parameters to the quantum circuit
               let qc_assigned = Python::with_gil(|py| {
                  let params_py = trial_denorm.clone();
                  let kwargs = [("inplace", false)].into_py_dict(py).unwrap();
                  let qc_assigned = qc.call_method("assign_parameters", (params_py,), Some(&kwargs));
                  match qc_assigned {
                     Ok(bound) => bound.unbind(),
                     Err(e) => {
                        error!("Error assigning parameters: {e}");
                        panic!("Error assigning parameters: {e}");
                      },
                  }
               });
               
               // Compute expectation value
               let expectation = Python::with_gil(|py| {
                  let shots_py = PyInt::new(py, shots_val);
                  let n_qpus_py = PyInt::new(py, n_qpus_val);
                  let qc_bound = qc_assigned.into_bound(py); 

                  let args = algorithms::AlgorithmArgs {
                     qc: qc_bound,
                     shots: Some(shots_py),
                     n_qpus: Some(n_qpus_py),
                  };

                  let distribute_algo = algorithms::AlgorithmDistributeByShots;
                  let result = distribute_algo.run(args);

                  // Compute extpectation value
                    let qaoa_utils = match PyModule::import(py, "polypus_python") {
                      Ok(module) => module,
                      Err(e) => {
                        error!("Failed to import polypus_python: {e}");
                        panic!("Failed to import polypus_python: {e}");
                      }
                    };
                  let expectation = qaoa_utils.call_method("expectation_value", (result, &expectation_function), None);
                  let expectation_value = match expectation{
                     Ok(result) => result,
                      Err(e) => {
                        error!("Error calling expectation_value: {e}");
                        panic!("{e}, Error serializing qc");
                      },
                  };
                  let expectation_value = match expectation_value.extract::<f64>() {
                     Ok(val) => val,
                     Err(e) => {
                        error!("Failed to extract float from expectation_value: {e}");
                        panic!("Failed to extract float from expectation_value: {e}");
                     }
                  };

                  expectation_value
               });

               if expectation > fitness[i] {
                  fitness[i] = expectation;
                  pop.row_mut(i).assign(&trial);
                  if expectation > fitness[best_idx] {
                     best_idx = i;
                     best = trial_denorm.clone();
                  }
               }
               debug!("Generation: {}, Individual: {}, Params: {:?}, Fitness: {}", generation, i, trial_denorm, fitness[i]);
            };
         };
         let result = Python::with_gil(|py| {
            match best.into_pyobject(py) {
               Ok(obj) => obj.unbind(),
               Err(e) => {
                 error!("Error converting best to result: {}", e);
                 panic!("Error converting best to result: {}", e);
               }
            }
         });
   
         result
      }

      fn name(&self) -> String {
         String::from("Differential Evolution")
      }

      fn description(&self) -> String {
         String::from("This algorithm is designed to train a QAOA circuit using differential evolution.")
      }
   }

   #[pyfunction(signature = (qc, shots=None, n_qpus=None, expectation_function=None, generations=None, population_size=None, dimensions=None, method=None))]
   pub fn run_quantum_circuit<'py>(
      qc: Bound<'py, PyAny>,
      shots: Option<Bound<'py, PyInt>>,
      n_qpus: Option<Bound<'py, PyInt>>,
      expectation_function: Option<Bound<'py, PyAny>>,
      generations: Option<usize>,
      population_size: Option<usize>,
      dimensions: Option<usize>,
      method: Option<String>,
   ) -> PyObject {

      // Process input arguments
      let args = AlgorithmArgs {
         qc,
         shots,
         n_qpus,
      };

      // Import algorithm
      match method.as_deref() {
         Some("single_run") => {
            debug!("Using Single Run Algorithm");
            let algorithm = AlgorithmSingleRun;
            return algorithm.run(args);
         }
         Some("distribute_by_shots") => {
            debug!("Using Distribute By Shots Algorithm");
            let algorithm = AlgorithmDistributeByShots;
            return algorithm.run(args);
         }
         Some("differential_evolution") => {
            debug!("Using Differential Evolution Algorithm");
            let differential_evolution_args = AlgorithmDifferentialEvolutionArgs {
               base: args,
               population_size,
               generations,
               dimensions,
               expectation_function,
            };
            let algorithm = algorithms::AlgorithmDifferentialEvolution;
            return algorithm.run(differential_evolution_args);
         }
         other => {
            let msg = format!("Unknown method: {:?}", other);
            error!("{}", msg);
            panic!("{}", msg);
         }
      }
   }
}

#[pymodule]
fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {
   setup_logger("my_log_file.log").unwrap_or_else(|e| panic!("Failed to set up logger: {}", e));

   use algorithms::*;
   m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
   Ok(())
}

fn setup_logger(log_path: &str) -> Result<(), fern::InitError> {

   if let Err(e) = std::fs::remove_file(log_path) {
        if e.kind() != ErrorKind::NotFound {
            panic!("Failed to remove log file '{}': {}", log_path, e);
        }
    }

   fern::Dispatch::new()
      .format(|out, message, record| {
         out.finish(format_args!(
            "[{} {} {}] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            record.level(),
            record.target(),
            message
         ))
      })
      .level(log::LevelFilter::Debug)
      .chain(fern::log_file(log_path)?)
      .apply()?;
   Ok(())
}