
// Description: This module provides the main functionalities for running quantum circuits.
// Created: 05-26-2025
// Last Modified: 11-05-2025

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyDict, IntoPyDict};
use std::fmt;
use std::process::{Command, Stdio};
use serde_json::Value;
use std::collections::HashMap;
use ndarray::{Array1,Array2, Axis};
use rand::Rng;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand::distributions::Bernoulli;
use log::{debug, error, info};
use std::io::ErrorKind;
use std::{thread, time};
use std::time::Instant;
use std::fs;
use std::sync::Once;

pub mod algorithms {
   use super::*; 

   pub struct AlgorithmArgs<'py> {
      pub id: String,
      pub qc: Bound<'py, PyAny>,
      pub shots: Option<Bound<'py, PyInt>>,
      pub n_qpus: u32,
      pub infraestructure: String,
      pub nodes: u32
   }

   pub struct AlgorithmDifferentialEvolutionArgs<'py> {
      pub base: AlgorithmArgs<'py>,
      pub population_size: Option<usize>,
      pub generations: Option<usize>,
      pub dimensions: Option<usize>,
      pub expectation_function: Option<Bound<'py, PyAny>>,
      pub tolerance: Option<f64>,
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
            1
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
         let infraestructure = args.infraestructure;
         let shots = match args.shots.as_ref() {
               Some(val) => val,
               None => {
                  error!("shots is required");
                  panic!("shots is required");
               }
            };

         let shots = shots.extract::<u32>().unwrap_or(1);

         let shots_per_qpu = shots / args.n_qpus;

         if infraestructure == "local" {
            debug!("Running local infraestructure for shots={}", shots);
            let result = Python::with_gil(|py| -> PyObject {
            let module = PyModule::import(py, "polypus_python").unwrap();
            let running_result = module.call_method("run_qc", (args.id.clone(), qc,args.shots,args.n_qpus), None);

            match running_result{
               Ok(result) => result.unbind(),
               Err(e) => {
                  error!("Error running run_qc: {e}");
                  panic!("{e}, Error running run_qc")
               },
            }
         });
         result

         } else if infraestructure == "qmio" {
            debug!("Raising QPUs for qmio infrastructure id family: {}", args.id.clone());
            let _raise_qpus = Command::new("qraise")
               .arg("-n")
               .arg(format!("{}", args.n_qpus))
               .arg("-N")
               .arg(format!("{}", args.nodes))
               .arg("-t")
               .arg("10:00:00")
               .arg("--fam")
               .arg(format!("{}", args.id))
               .arg("--cloud")
               .spawn()
               .expect("Failed to start qraise command");
            thread::sleep(time::Duration::from_secs(30));
            debug!("qraise command executed");

            
            debug!("Running {} shots per qpu", shots_per_qpu);

            let result = Python::with_gil(|py| -> PyObject {
               let module = PyModule::import(py, "polypus_python").unwrap();
               let running_result = module.call_method("run_qc_in_qpu", (args.id.clone(), qc, shots,), None);
               debug!("CALLING PYTHON!");

               match running_result{
                  Ok(results) => results.unbind(),
                  Err(e) => {
                     error!("Error running run_qc: {e}");
                     panic!("{e}, Error running run_qc")
                  },
               }
            });

            result 


         } else {
            error!("Infraestructure {} not valid!", infraestructure);
            panic!("Infraestructure {} not valid!", infraestructure)
         }

         
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
         
         // Process input arguments
         let start_prepare_running = Instant::now();
         let id = args.id.clone();
         let qc = args.qc;
         let n_shots = match args.shots.as_ref() {
            Some(val) => val,
            None => {
               error!("shots is required");
               panic!("shots is required");
            }
          };
         let n_shots_val = n_shots.extract::<i64>().unwrap_or(1);
         // let n_qpus = match args.n_qpus.as_ref() {
         //    Some(val) => val,
         //    None => {
         //       error!("n_qpus is required");
         //       panic!("n_qpus is required");
         //    }
         // };
         let n_qpus_val: i64 = args.n_qpus.clone() as i64;
         // let n_qpus_val = n_qpus.extract::<i64>().unwrap_or(1);
         let infraestructure = args.infraestructure;
         
         // Compute number of shots per QPU
         let shots_per_qpu = n_shots_val / n_qpus_val;

         // Dynamically get number of logical CPUs
         let n_qpus_val = n_qpus_val as usize;  
         let num_cores = get_num_cpus();  
         let cores_per_worker = num_cores / n_qpus_val;

         // Running information
         debug!("Infraestructure: {}", infraestructure);
         debug!("Number of QPUs: {}", n_qpus_val);
         debug!("Number of cores {}", num_cores);
         let duration_prepare_running = start_prepare_running.elapsed(); 
         debug!("Prepare running took {:?}", duration_prepare_running);

         // 1. Serialize the quantum circuit
         let time_serialize = Instant::now();
         Python::with_gil(|py| {
            let module = PyModule::import(py, "polypus_python").unwrap();
            let running_result = module.call_method("serialize_quantum_circuit", (id.clone(), &qc,), None);
            match running_result {
               Ok(result) => result,
               Err(e) => {
                 error!("Error serializing qc: {e}");
                 panic!("{e}, Error serializing qc")
               },
            };
         });
         let duration_serialize = time_serialize.elapsed();
         debug!("Quantum circuit serialized in {:?}", duration_serialize);

         // 2. Create a number of processes based on the number of QPUs and run the quantum circuits
         let mut handles = Vec::new();
         for i in 0..n_qpus_val {
            let start_process = Instant::now();
            let start_core = i * cores_per_worker;
            let mut end_core = start_core + cores_per_worker - 1;
            if i == n_qpus_val - 1 {
               end_core = num_cores - 1;
            }
            let core_range = format!("{}-{}", start_core, end_core);
            debug!("Starting process {i} for QPU {i} on cores {core_range}");

            // 2.1 Select running method based on infraestructure
            if infraestructure == "local" {
               debug!("Running on local infrastructure");
               let process = Command::new("python")
                  .arg("packages/polypus_python/polypus_python/run_worker.py")
                  .arg("--shots")
                  .arg(format!("{}", shots_per_qpu))
                  .arg("--max_parallel_threads")
                  .arg(format!("{}", cores_per_worker))
                  .arg("--id")
                  .arg(format!("{}", id))
                  .stdout(Stdio::piped())
                  .spawn()
                  .expect("Failed to start worker process");
               handles.push((i, process));
            } else if infraestructure == "qmio" {
               debug!("Running on qmio infrastructure");
               let process = Command::new("python")
                  .arg("packages/polypus_python/polypus_python/run_worker_qpu.py")
                  .arg("--id")
                  .arg(format!("{}", i))
                  .arg("--shots")
                  .arg(format!("{}", shots_per_qpu))
                  .stdout(Stdio::piped())
                  .spawn()
                  .expect("Failed to start worker process");
               handles.push((i, process));
            } else {
               error!("Unknown infrastructure: {}", infraestructure);
               panic!("Unknown infrastructure: {}", infraestructure);
            }
            let duration_process = start_process.elapsed();
            debug!("Process {i} started in {:?}", duration_process);
         }

         // 3. Collect outputs
         let start_collecting_total = Instant::now();
         let mut parsed_outputs = Vec::new();
         for (i, process) in handles {
            let start_collecting = Instant::now();
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
            let duration_collecting = start_collecting.elapsed();
            debug!("Collected output from process {} in {:?}", i, duration_collecting);
         }
         let duration_collecting = start_collecting_total.elapsed();
         debug!("Collected outputs from all processes in {:?}", duration_collecting);

         // Sum the counts from all dictionaries to aggregate results
         let start_summing = Instant::now();
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
         let duration_summing = start_summing.elapsed();
         debug!("Summed counts from all dictionaries in {:?}", duration_summing);

         // 4. Convert the dictionary to a PyObject
         let start_converting = Instant::now();
         let py_dict = Python::with_gil(|py| -> PyObject {
            let dict = PyDict::new(py);
            for (key, value) in total_counts {
               dict.set_item(key, value).unwrap();
            }
            dict.unbind().into()
         });
         let duration_converting = start_converting.elapsed();
         debug!("Converted dictionary to PyObject in {:?}", duration_converting);

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
         // let n_qpus = match args.base.n_qpus.as_ref() {
         //    Some(val) => val,
         //    None => {
         //       let err_msg = "n_qpus is required";
         //       error!("{}", err_msg);
         //       panic!("{}", err_msg);
         //    }
         // };
         // let n_qpus_val = n_qpus.extract::<i64>().unwrap_or(1);
         let n_qpus_val: i64 = args.base.n_qpus.clone() as i64;
         let expectation_function = args.expectation_function;
         let infraestructure = args.base.infraestructure;

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
         let max_generations = match args.generations {
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
         let tolerance = match args.tolerance {
            Some(val) => val,
            None => {
               let err_msg = "tolerance is required";
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

         // If infraestructure is qmio, raise QPUs with CUNQA
         if infraestructure == "qmio" {
            
            debug!("Raising QPUs for qmio infrastructure id family: {}", args.base.id.clone());
            let n_qpus_val = n_qpus_val as usize;  
            let _raise_qpus = Command::new("qraise")
               .arg("-n")
               .arg(format!("{}", n_qpus_val))
               .arg("-N")
               .arg(format!("{}", args.base.nodes.clone()))
               .arg("-t")
               .arg("10:00:00")
               .arg("--fam")
               .arg(format!("{}", args.base.id.clone()))
               .arg("--cloud")
               .spawn()
               .expect("Failed to start qraise command");
            thread::sleep(time::Duration::from_secs(30));
            debug!("qraise command executed");

            let _shots_py = Python::with_gil(|py| {            
               PyInt::new(py, shots_val);
            });
            for generation in 0..max_generations {
               for i in (0..popsize).step_by(n_qpus_val) {
                  let end = (i + n_qpus_val).min(popsize);
                  let batch_ids = i..end;
                  debug!("Generation: {}, Processing batch of individuals: {:?}", generation, batch_ids);

                  // Create trial for each individual in the batch and assign parameters to the quantum circuits
                  let mut qcs: Vec<PyObject> = Vec::new();
                  let mut trials_batch: Vec<Array1<f64>> = Vec::new();
                  let mut trials_denorms_batch: Vec<Vec<f64>> = Vec::new();
                  for trial_id in batch_ids.clone() {
                  
                     debug!("Processing individual: {}", trial_id);
                     let ids: Vec<usize> = (0..popsize).filter(|&idx| idx != trial_id).collect();
                     let selected_ids: Vec<usize> = ids.choose_multiple(&mut rng, 3).cloned().collect();
                     let c1 = pop.row(selected_ids[0]).to_owned();
                     let c2 = pop.row(selected_ids[1]).to_owned();
                     let c3 = pop.row(selected_ids[2]).to_owned();
                     let mut mutant = &c1 + mut_factor * (&c2 - &c3);
                     mutant.mapv_inplace(|x| x.max(0.0).min(1.0));
                     let cross_points: Array1<bool> = Array1::from_shape_fn(dimensions,|_| rng.sample(Bernoulli::new(crossp).unwrap()));
                     let trial = cross_points.iter().zip(mutant.iter()).zip(pop.row(trial_id).iter()).map(|((cp, m), p)| if *cp { *m } else { *p }).collect::<Array1<f64>>();
                     trials_batch.push(trial.clone());
                     let trial_denorm = trial.clone();
                     let trial_denorm: Vec<f64> = trial_denorm.to_vec();
                     trials_denorms_batch.push(trial_denorm.clone());

                     // Assign parameters to the quantum circuit
                     let _start_assign_params = Instant::now();
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
                     qcs.push(qc_assigned);

                  }
                  debug!("qcs: {:?}", qcs);

                  // Compute expectation values for the batch
                  let results = Python::with_gil(|py| -> PyObject {
                     let module = PyModule::import(py, "polypus_python").unwrap();
                     let running_result = module.call_method("run_qcs_in_qpu", (args.base.id.clone(), qcs,shots_val,), None);
                     debug!("CALLING PYTHON!");

                     match running_result{
                        Ok(results) => results.unbind(),
                        Err(e) => {
                           error!("Error running run_qc: {e}");
                           panic!("{e}, Error running run_qc")
                        },
                     }
                  });

                  let _expectation = Python::with_gil(|py| {

                     let qaoa_utils = match PyModule::import(py, "polypus_python") {
                        Ok(module) => module,
                        Err(e) => {
                           error!("Failed to import polypus_python: {e}");
                           panic!("Failed to import polypus_python: {e}");
                        }
                     };
                     let expectations = qaoa_utils.call_method("expectation_values", (results, &expectation_function), None);

                     let expectation_values = match expectations{
                        Ok(result) => result,
                        Err(e) => {
                           error!("Error calling expectation_value: {e}");
                           panic!("{e}, Error serializing qc");
                        },
                     };
                     let expectation_values = match expectation_values.extract::<Vec<f64>>() {
                        Ok(val) => val,
                        Err(e) => {
                           error!("Failed to extract float from expectation_value: {e}");
                           panic!("Failed to extract float from expectation_value: {e}");
                        }
                     };

                     debug!("Expectation values: {:?}", expectation_values);

                     // Update fitness and population
                     for (idx, trial_id) in batch_ids.clone().enumerate() {
                        debug!("Current Expectation value for individual {}: {}", trial_id, fitness[trial_id]);
                        if expectation_values[idx] > fitness[trial_id] {
                           fitness[trial_id] = expectation_values[idx];
                           pop.row_mut(trial_id).assign(&trials_batch[idx]);
                           if expectation_values[idx] > fitness[best_idx] {
                              best_idx = trial_id;
                              best = trials_denorms_batch[idx].clone();
                           }
                        }
                        debug!("Updated fitness for individual {}: {}", trial_id, fitness[trial_id]);

                     }
                  });

               }

               info!("Generation {} completed. Best fitness: {}", generation, fitness[best_idx]);

               let mean = pop.mean_axis(Axis(0)).expect("Failed to compute mean");
               let mean = mean.sum();
               let std = pop.std_axis(Axis(0), 0.0);
               let std: f64 = std.iter().sum();
               debug!("Generation {}: Mean: {:?}, Std: {:?}", generation, mean, std);
               
               let converged = std < (tolerance * mean);

               if converged {
                  info!("Stopping early at generation {} due to convergence", generation);
                  break; 
               } 

            }

            let result = Python::with_gil(|py| {
            match best.into_pyobject(py) {
               Ok(obj) => obj.unbind(),
               Err(e) => {
                 error!("Error converting best to result: {}", e);
                 panic!("Error converting best to result: {}", e);
               }
            }
         });
   
         return result;

         } else {
            debug!("Running on local infrastructure, no need to raise QPUs");
         }

         // Train
         let mut rng = thread_rng();
         for generation in 0..max_generations {
            for i in 0..popsize {
               let start_individual = Instant::now();
               
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
               let duration_select_trial = start_individual.elapsed();
               debug!("Generation: {}, Individual: {}, Trial selected in {:?}", generation, i, duration_select_trial);

               // Assign parameters to the quantum circuit
               let start_assign_params = Instant::now();
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
               let duration_assign_params = start_assign_params.elapsed();
               debug!("Generation: {}, Individual: {}, Parameters assigned in {:?}", generation, i, duration_assign_params);
               
               // Compute expectation value
               let infraestructure_to_run = infraestructure.clone();
               let expectation = Python::with_gil(|py| {
                  let start_prepare_running = Instant::now();
                  let shots_py = PyInt::new(py, shots_val);
                  let _n_qpus_py = PyInt::new(py, n_qpus_val);
                  let qc_bound = qc_assigned.into_bound(py); 

                  // let args = algorithms::AlgorithmArgs {
                  //    id: args.base.id.clone(),
                  //    qc: qc_bound,
                  //    shots: Some(shots_py),
                  //    n_qpus: Some(n_qpus_py),
                  //    infraestructure: infraestructure_to_run,
                  //    nodes: args.base.nodes.clone(),
                  // };
                  let args = algorithms::AlgorithmArgs {
                     id: args.base.id.clone(),
                     qc: qc_bound,
                     shots: Some(shots_py),
                     n_qpus: args.base.n_qpus.clone(),
                     infraestructure: infraestructure_to_run,
                     nodes: args.base.nodes.clone(),
                  };
                  
                  // Run
                  let distribute_algo = algorithms::AlgorithmSingleRun;
                  let duration_prepare_running = start_prepare_running.elapsed();
                  debug!("Generation: {}, Individual: {}, Prepare running took {:?}", generation, i, duration_prepare_running);

                  let start_run = Instant::now();
                  let result = distribute_algo.run(args);
                  let duration_run = start_run.elapsed();
                  debug!("Generation: {}, Individual: {}, Running took {:?}", generation, i, duration_run);

                  // Compute extpectation value
                  let start_compute_expectation = Instant::now();
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
                  let duration_compute_expectation = start_compute_expectation.elapsed();
                  debug!("Generation: {}, Individual: {}, Expectation value computed in {:?}", generation, i, duration_compute_expectation);
                  expectation_value
               });

               // Update fitness and population
               let start_update_fitness = Instant::now();
               if expectation > fitness[i] {
                  fitness[i] = expectation;
                  pop.row_mut(i).assign(&trial);
                  if expectation > fitness[best_idx] {
                     best_idx = i;
                     best = trial_denorm.clone();
                  }
               }
               let duration_update_fitness = start_update_fitness.elapsed();
               debug!("Generation: {}, Individual: {}, Fitness updated in {:?}", generation, i, duration_update_fitness);
               debug!("Generation: {}, Individual: {}, Params: {:?}, Fitness: {}", generation, i, trial_denorm, fitness[i]);
               let duration_individual = start_individual.elapsed();
               debug!("Individual {} in generation {} took {:?} to evaluate", i, generation, duration_individual);
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

   #[pyfunction(signature=(qc, shots, infraestructure, n_qpus=1))]
   pub fn run_quantum_circuit<'py>(
      qc: Bound<'py, PyAny>,
      shots: Option<Bound<'py, PyInt>>,
      infraestructure: Option<String>,
      n_qpus: Option<u32>,
   ) -> PyObject {

      // n_qpus
      let n_qpus = match n_qpus {
         Some(val) => val,
         None => {
            1
         }
      };

      // Id
      let infraestructure = infraestructure.unwrap_or_else(|| "local".to_string());
      let id:String = format!("run_{}_{}", n_qpus, infraestructure);

      // Create logger
      setup_logger(&format!("logger_{}.log", id));

      // Process input arguments
      let args = AlgorithmArgs {
         id: id.clone(),
         qc,
         shots,
         n_qpus,
         infraestructure,
         nodes: 1, 
      };

      // Run
      let algorithm = AlgorithmSingleRun;
      return algorithm.run(args);
   }

   #[pyfunction]
   pub fn raise_qpus() {
      // This function is a placeholder for raising QPUs if needed.
      // It can be implemented later if required.
      debug!("raise_qpus function called, but not implemented yet.");
   }

   #[pyfunction(signature = (qc, shots=None, n_qpus=None, expectation_function=None, generations=None, population_size=None, dimensions=None, infraestructure=None, nodes=None, id=None, tolerance=0.01))]
   pub fn differential_evolution<'py>(
      qc: Bound<'py, PyAny>,
      shots: Option<Bound<'py, PyInt>>,
      n_qpus: Option<u32>,
      expectation_function: Option<Bound<'py, PyAny>>,
      generations: Option<usize>,
      population_size: Option<usize>,
      dimensions: Option<usize>,
      infraestructure: Option<String>, 
      nodes: Option<u32>,
      id: Option<String>,
      tolerance: Option<f64>,
   ) -> PyObject {

      // Create logger
      setup_logger(&format!("logger_{}.log", id.as_ref().unwrap()));

      // Process input arguments
      let args = AlgorithmArgs {
         id: id.unwrap_or_else(|| "default_id".to_string()), 
         qc,
         shots,
         n_qpus: n_qpus.expect("nqpus required"),
         infraestructure: infraestructure.unwrap_or_else(|| "local".to_string()),
         nodes: nodes.unwrap_or(1),
      };
      let differential_evolution_args = AlgorithmDifferentialEvolutionArgs {
         base: args,
         population_size,
         generations,
         dimensions,
         expectation_function,
         tolerance,
      };

      // Run
      let algorithm = algorithms::AlgorithmDifferentialEvolution;
      return algorithm.run(differential_evolution_args);
   }
}

#[pymodule]
fn polypus(m: &Bound<'_, PyModule>) -> PyResult<()> {

   use algorithms::*;
   m.add_function(wrap_pyfunction!(run_quantum_circuit, m)?)?;
   m.add_function(wrap_pyfunction!(raise_qpus, m)?)?;
   m.add_function(wrap_pyfunction!(differential_evolution, m)?)?;
   Ok(())
}


static INIT_LOGGER: Once = Once::new();

fn setup_logger(log_path: &str) {

   INIT_LOGGER.call_once(|| { 
      if let Err(e) = fs::remove_file(log_path) {
         if e.kind() != ErrorKind::NotFound {
            panic!("Failed to remove log file '{}': {}", log_path, e);
         }
      }

   let logger = fern::Dispatch::new()
      .format(|out, message, record| {
         out.finish(format_args!(
            "[{} {} {}] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.f"),
            record.level(),
            record.target(),
            message
         ))
      })
      .level(log::LevelFilter::Debug)
      .chain(fern::log_file(log_path).expect("Failed to open log file"));
   logger.apply().expect("Failed to initialize logger");
   });

}