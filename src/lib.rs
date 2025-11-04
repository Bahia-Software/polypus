// Polypus

//! # Polypus Version 0.2.0
//!
//! Polypus is a distributed quantum computing library designed to optimize the execution of quantum algorithms by distributing computation 
//! across available hardware resources.
//! 
//! Polypus is desgined to be agnostic to the underlying hardware infrastructure. Currently, it support two infraestructures:
//! - **local**: Runs on local infraestructure (e.g. a local server).
//! - **qmio**: Platform to simulate distributed quantum computing using multiple emulated QPUs. This infraestructure is desgined by CESGA (<https://github.com/CESGA-Quantum-Spain/cunqa>).
//!
//!
//! ## Features
//! - Run a quantum circuit.
//! - Run a quantum circuit using multiple QPUs.
//! - Train a QAOA using differential evolution.
//!
//! ## Examples
//! Polypus is directly imported from Python. 
//! ```python
//! import polypus
//! ```
//!
//! To run a quantum circuit, you can use the `run_quantum_circuit` function:
//! ```python
//! result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infraestructure="local")
//! ```
//!
//! We can distribute the quantum circuit shots using multiple QPUs:
//! ```python
//! result = polypus.run_quantum_circuit(qft_circuit, shots=1000, infraestructure="local", n_qpus=10)
//! ```
//! We can directly import an optimization algorithm. For example, if we define a QAOA circuit and specify how to compute its expectation value, 
//! we can optimize this circuit using differential evolution. Polypus takes care of optimizing the execution by distributing the shots across 
//! the available hardware resources.
//! ```python
//!polypus.differential_evolution(
//!   qc=qc, 
//!   shots=N_SHOTS, 
//!   n_qpus = N_QPUS, 
//!   expectation_function=bitstring_to_obj, 
//!   generations=MAX_GENERATIONS, 
//!   population_size=POPULATION_SIZE, 
//!   dimensions=2*layers, 
//!   infraestructure="qmio", 
//!   id=id,
//!   tolerance=TOL,
//!   nodes=N)
//! ```
//! # Authors 
//! Diego Beltran Fernandez Prada (<diego.fernandez@bahiasoftware.es>), Victor Soñora Pombo, Sergio Figueiras Gomez, Miguel Boubeta Martinez, Galicia Supercomputing Center (CESGA)
//! # License:
//! European Union Public Licence (EUPL), version 1.2 or – as soon they will be approved by the European Commission – subsequent versions of the EUPL.

mod bindings;
pub mod infrastructure;
pub mod algorithms;

pub use crate::algorithms::*;