// Polypus

//! # Polypus
//!
//! Polypus is a distributed quantum computing library designed to optimize the execution of quantum algorithms by distributing computation
//! across available hardware resources. The core is written in Rust; Python bindings are provided via PyO3.
//!
//! Polypus is agnostic to the underlying hardware infrastructure. Currently two backends are supported:
//! - **local**: Runs on local infrastructure using Qiskit AerSimulator.
//! - **cunqa**: Distributed QPU platform developed by CESGA (<https://github.com/CESGA-Quantum-Spain/cunqa>).
//!
//! ## Features
//! - Run a quantum circuit on one or multiple QPUs.
//! - Train variational quantum circuits using Differential Evolution, PSO or Quantum Natural Gradient.
//! - Unified `train()` API — swap optimizers by changing a single argument.
//!
//! ## Running a circuit
//! ```python
//! import polypus
//!
//! # Single QPU
//! result = polypus.run_quantum_circuit(qc, shots=1000, infrastructure="local")
//!
//! # Distribute shots across multiple QPUs
//! result = polypus.run_quantum_circuit(qc, shots=1000, infrastructure="local", n_qpus=10)
//! ```
//!
//! ## Training a variational circuit
//! Pass a method object as the second argument to `train()`. The optimizer-specific
//! parameters are encapsulated in the method object; execution parameters are shared.
//!
//! ```python
//! # Differential Evolution
//! result_params = polypus.train(
//!     qc, polypus.DE(generations=100, population_size=50, tolerance=0.01),
//!     shots=N_SHOTS, n_qpus=N_QPUS, dimensions=2*layers,
//!     expectation_function=bitstring_to_obj,
//!     infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
//! )
//!
//! # Particle Swarm Optimization
//! result_params = polypus.train(
//!     qc, polypus.PSO(generations=100, population_size=50, bounds=(0.0, 3.14)),
//!     shots=N_SHOTS, n_qpus=N_QPUS, dimensions=2*layers,
//!     expectation_function=bitstring_to_obj,
//!     infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
//! )
//!
//! # Quantum Natural Gradient
//! result_params = polypus.train(
//!     qc, polypus.QNG(variance_fn, max_iters=100, learning_rate=0.1),
//!     shots=N_SHOTS, n_qpus=N_QPUS, dimensions=2*layers,
//!     expectation_function=bitstring_to_obj,
//!     infrastructure="local", nodes=1, cores_per_qpu=2, id="run1"
//! )
//! ```
//!
//! # Authors
//! Diego Beltrán Fernández Prada (<diego.fernandez@bahiasoftware.es>), Víctor Sóñora Pombo, Sergio Figueiras Gómez, Miguel Boubeta Martínez, Galicia Supercomputing Center (CESGA)
//!
//! # License
//! European Union Public Licence (EUPL), version 1.2 or – as soon as they will be approved by the European Commission – subsequent versions of the EUPL.

mod bindings;
pub mod infrastructure;
pub mod algorithms;

pub use crate::algorithms::*;