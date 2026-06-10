//! Error types for circuit construction and parameter binding.

use std::fmt;

/// Errors that can occur when binding parameter values to a
/// [`ParameterizedCircuit`](crate::ParameterizedCircuit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitError {
    /// The number of values passed to `assign_parameters` does not match the
    /// number of free parameters declared by the circuit.
    WrongNumberOfParams { expected: usize, got: usize },
    /// A gate references `Param(index)` but `index` is outside the range of
    /// the provided parameter values. This can only happen when the circuit
    /// was assembled manually (the fluent builder keeps `num_params` in sync).
    ParamIndexOutOfBounds { index: usize, num_params: usize },
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
        }
    }
}

impl std::error::Error for CircuitError {}
