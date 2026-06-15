//! Error types for circuit construction and parameter binding.

use std::fmt;

/// Errors that can occur when building a circuit, binding parameter values
/// to a [`ParameterizedCircuit`](crate::ParameterizedCircuit), or importing
/// OpenQASM 2.0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    /// The number of values passed to `assign_parameters` does not match the
    /// number of free parameters declared by the circuit.
    WrongNumberOfParams { expected: usize, got: usize },
    /// A gate references `Param(index)` but `index` is outside the range of
    /// the provided parameter values. This can only happen when the circuit
    /// was assembled manually (the fluent builder keeps `num_params` in sync).
    ParamIndexOutOfBounds { index: usize, num_params: usize },
    /// A gate addresses a qubit index `>= num_qubits`.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// A two-qubit gate was given the same qubit twice.
    IdenticalQubits { qubit: usize },
    /// The OpenQASM 2.0 source could not be parsed
    /// (see [`ParameterizedCircuit::from_qasm2`](crate::ParameterizedCircuit::from_qasm2)).
    Parse {
        /// 1-based source line where the error was detected.
        line: usize,
        /// Human-readable description of the problem.
        message: String,
    },
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitError::WrongNumberOfParams { expected, got } => write!(
                f,
                "wrong number of parameter values: circuit declares {expected} free parameter(s) but {got} value(s) were provided"
            ),
            CircuitError::ParamIndexOutOfBounds { index, num_params } => write!(
                f,
                "gate references parameter index {index}, but only {num_params} parameter value(s) are available"
            ),
            CircuitError::QubitOutOfRange { qubit, num_qubits } => write!(
                f,
                "qubit index {qubit} out of range for circuit with {num_qubits} qubits"
            ),
            CircuitError::IdenticalQubits { qubit } => write!(
                f,
                "two-qubit gate requires distinct qubits, got ({qubit}, {qubit})"
            ),
            CircuitError::Parse { line, message } => {
                write!(f, "QASM parse error at line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for CircuitError {}
