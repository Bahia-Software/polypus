//! Error type for the statevector simulator.

use std::fmt;

/// Errors that can occur while preparing or running a simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    /// The circuit needs more qubits than the backend will allocate
    /// (see [`MAX_QUBITS`](crate::MAX_QUBITS)). A statevector requires `2^n`
    /// complex amplitudes, so this guards against doomed allocations and
    /// `1 << n` overflow.
    TooManyQubits {
        /// Qubits requested by the circuit.
        requested: usize,
        /// Largest supported qubit count.
        max: usize,
    },
    /// A gate still references a free parameter `Param(index)`. A
    /// [`ConcreteCircuit`](polypus_circuit::ConcreteCircuit) produced by
    /// `assign_parameters` is always fully bound; this can only happen for a
    /// manually assembled circuit. Bind all parameters before simulating.
    UnboundParameter {
        /// The offending parameter index.
        index: usize,
    },
    /// A gate angle resolved to a non-finite value (`NaN` or infinity), which
    /// would corrupt the statevector.
    NonFiniteAmplitude,
    /// A gate acts on a qubit that was already measured, violating the
    /// terminal-measurement model (contract C-4). This can only happen for a
    /// manually assembled circuit; the builder and QASM importer reject it at
    /// construction time.
    GateAfterMeasure {
        /// The qubit that was operated on after being measured.
        qubit: usize,
    },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::TooManyQubits { requested, max } => write!(
                f,
                "circuit requests {requested} qubits but the statevector backend supports at most {max} (a statevector needs 2^n complex amplitudes)"
            ),
            SimError::UnboundParameter { index } => write!(
                f,
                "circuit contains an unbound free parameter (index {index}); bind all parameters before simulating"
            ),
            SimError::NonFiniteAmplitude => {
                write!(f, "a gate angle resolved to a non-finite value (NaN or infinity)")
            }
            SimError::GateAfterMeasure { qubit } => write!(
                f,
                "gate acts on qubit {qubit} after it was measured; Polypus circuits use terminal measurement (contract C-4)"
            ),
        }
    }
}

impl std::error::Error for SimError {}
