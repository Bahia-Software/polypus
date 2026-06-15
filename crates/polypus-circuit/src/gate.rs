//! Core gate data types: [`GateParam`] and [`GateInstruction`].

use crate::error::CircuitError;

/// An angle argument of a rotation gate.
///
/// Either a concrete value ([`Fixed`](GateParam::Fixed)) or a reference to the
/// free parameter at a given index ([`Param`](GateParam::Param)), to be bound
/// later via [`ParameterizedCircuit::assign_parameters`](crate::ParameterizedCircuit::assign_parameters).
///
/// `GateParam` implements `From<f64>`, so builder methods accept plain floats:
///
/// ```
/// use polypus_circuit::{ParameterizedCircuit, Param};
///
/// let qc = ParameterizedCircuit::new(1)
///     .rx(0, 0.5)        // fixed angle
///     .rz(0, Param(0));  // free parameter #0
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateParam {
    /// A concrete angle value (radians).
    Fixed(f64),
    /// A reference to the free parameter at this index.
    Param(usize),
}

impl From<f64> for GateParam {
    fn from(value: f64) -> Self {
        GateParam::Fixed(value)
    }
}

impl GateParam {
    /// Resolve to a concrete value, looking up `Param` indices in `params`.
    pub(crate) fn resolve(&self, params: &[f64]) -> Result<f64, CircuitError> {
        match *self {
            GateParam::Fixed(v) => Ok(v),
            GateParam::Param(i) => params.get(i).copied().ok_or(
                CircuitError::ParamIndexOutOfBounds {
                    index: i,
                    num_params: params.len(),
                },
            ),
        }
    }
}

/// A single instruction in a quantum circuit.
///
/// Gate names follow the standard `qelib1.inc` vocabulary of OpenQASM 2.0
/// (`h`, `cx`, `rz`, `rzz`, `u3`, …), so every variant maps one-to-one onto a
/// QASM statement.
#[derive(Debug, Clone, PartialEq)]
pub enum GateInstruction {
    /// Hadamard gate.
    H(usize),
    /// Pauli-X gate.
    X(usize),
    /// Pauli-Y gate.
    Y(usize),
    /// Pauli-Z gate.
    Z(usize),
    /// Phase gate S (√Z).
    S(usize),
    /// T gate (√S).
    T(usize),
    /// Conjugate transpose of S.
    Sdg(usize),
    /// Conjugate transpose of T.
    Tdg(usize),
    /// Rotation around the X axis.
    Rx { qubit: usize, theta: GateParam },
    /// Rotation around the Y axis.
    Ry { qubit: usize, theta: GateParam },
    /// Rotation around the Z axis.
    Rz { qubit: usize, theta: GateParam },
    /// Controlled-NOT: control, target.
    Cx(usize, usize),
    /// Controlled-Z: control, target.
    Cz(usize, usize),
    /// Two-qubit ZZ-interaction rotation, exp(-i θ/2 Z⊗Z).
    Rzz { q0: usize, q1: usize, theta: GateParam },
    /// Two-qubit XX-interaction rotation, exp(-i θ/2 X⊗X).
    Rxx { q0: usize, q1: usize, theta: GateParam },
    /// Generic single-qubit gate `u3(theta, phi, lambda)`.
    U {
        qubit: usize,
        theta: GateParam,
        phi: GateParam,
        lam: GateParam,
    },
    /// Barrier. An empty vector means "all qubits" (`barrier q;`).
    Barrier(Vec<usize>),
    /// Measure one qubit into one classical bit.
    Measure { qubit: usize, cbit: usize },
    /// Measure every qubit `i` into classical bit `i` (`measure q -> c;`).
    MeasureAll,
}

impl GateInstruction {
    /// Largest classical-bit index used by this instruction, if any.
    /// `MeasureAll` is handled separately by the circuit (it needs `num_qubits`).
    pub(crate) fn max_cbit(&self) -> Option<usize> {
        match self {
            GateInstruction::Measure { cbit, .. } => Some(*cbit),
            _ => None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_eval_integration(){
        let param_fixed = GateParam::Fixed(1.5);
        let param_variable = GateParam::Param(0);
        let external_values = vec![3.0];

        let fixed_instruction = GateInstruction::Rx { qubit: 0, theta: param_fixed };
        let variable_instruction = GateInstruction::Rx { qubit: 0, theta: param_variable };

        let fixed_result = match fixed_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };

        let variable_result = match variable_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };

        assert_eq!(fixed_result.unwrap(), 1.5);
        assert_eq!(variable_result.unwrap(), 3.0);
    }

    #[test]
    fn test_param_multiple_index() {
        let params = vec![10.0, 20.0, 30.0];

        assert_eq!(GateParam::Param(0).resolve(&params).unwrap(), 10.0);
        assert_eq!(GateParam::Param(1).resolve(&params).unwrap(), 20.0);
        assert_eq!(GateParam::Param(2).resolve(&params).unwrap(), 30.0);
    }

    #[test]
    fn test_param_eval_out_of_bounds(){
        let param_variable = GateParam::Param(1);
        let external_values = vec![3.0];

        let variable_instruction = GateInstruction::Rx { qubit: 0, theta: param_variable };
        
        let variable_result = match variable_instruction {
            GateInstruction::Rx { theta, .. } => theta.resolve(&external_values),
            _ => panic!("Expected Rx"),
        };


        match variable_result {
            Err(CircuitError::ParamIndexOutOfBounds { index, num_params }) => {
                assert_eq!(index, 1);
                assert_eq!(num_params, 1);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_param_resolve_empty_params() {
        let param = GateParam::Param(0);
        let params = vec![];

        let result = param.resolve(&params);

        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_keeps_original_values() {
        let param = GateParam::Param(0);
        let params = vec![42.3];

        let _ = param.resolve(&params);

        assert_eq!(params, vec![42.3]);
    }

    #[test]
    fn test_resolve_special_values() {
        let params = vec![f64::INFINITY, f64::NAN];

        let inf = GateParam::Param(0).resolve(&params).unwrap();
        let nan = GateParam::Param(1).resolve(&params).unwrap();

        assert!(inf.is_infinite());
        assert!(nan.is_nan());
    }

    #[test]
    fn test_max_cbit_measure() {
        let instruction = GateInstruction::Measure { qubit: 0, cbit: 5 };

        let result = instruction.max_cbit();

        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_max_cbit_non_measure() {
        let instruction = GateInstruction::H(0);

        let result = instruction.max_cbit();

        assert_eq!(result, None);
    }

    #[test]
    fn test_max_cbit_measure_all() {
        let instruction = GateInstruction::MeasureAll;

        let result = instruction.max_cbit();

        assert_eq!(result, None);
    }
}