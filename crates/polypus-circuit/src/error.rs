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
    /// A gate parameter resolved to a non-finite value (`NaN` or infinity),
    /// which is not a valid rotation angle. Rejected both at construction (a
    /// `Fixed` angle) and at binding time (a caller-supplied value bound to a
    /// `Param`), matching the native simulator's reference behaviour
    /// (contract C-2).
    NonFiniteParam,
    /// A parameter expression divided by zero while it was evaluated (at
    /// binding time, or when exporting with parameter values). Division by
    /// zero is an error of its own, never an infinite angle.
    DivisionByZero,
    /// A gate refers to an expression ([`GateParam::Expr`](crate::GateParam::Expr))
    /// that is not stored in this circuit: an [`ExprId`](crate::ExprId) is
    /// only meaningful in the circuit that issued it
    /// ([`ParameterizedCircuit::add_expr`](crate::ParameterizedCircuit::add_expr))
    /// and its clones. Also reported by exports of a
    /// [`ConcreteCircuit`](crate::ConcreteCircuit), which holds no expressions.
    UnknownExpression,
    /// A parameter expression cannot be stored in a circuit: it is larger or
    /// nests deeper than a circuit allows (see
    /// [`ParameterizedCircuit::add_expr`](crate::ParameterizedCircuit::add_expr)).
    InvalidExpression {
        /// Human-readable description of the problem.
        reason: String,
    },
    /// A gate addresses a qubit index `>= num_qubits`.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// A two-qubit gate was given the same qubit twice.
    IdenticalQubits { qubit: usize },
    /// A unitary gate acts on a qubit that was already measured, violating the
    /// terminal-measurement model (contract C-4). See
    /// `docs/adr/0001-terminal-measurements.md`.
    QubitAlreadyMeasured { qubit: usize },
    /// The OpenQASM 2.0 source could not be parsed
    /// (see [`ParameterizedCircuit::from_qasm2`](crate::ParameterizedCircuit::from_qasm2)).
    Parse {
        /// 1-based source line where the error was detected.
        line: usize,
        /// Human-readable description of the problem.
        message: String,
    },
    /// A call of a declared gate was given a different number of angles or
    /// qubits than the declaration takes
    /// ([`CustomGate::with_arguments`](crate::CustomGate::with_arguments)).
    GateSignature {
        /// The declared gate.
        name: String,
        /// Angles the declaration takes, and angles given (saturating at
        /// `u32::MAX`; 32-bit counts keep the error, which binding returns for
        /// every angle, as small as it was).
        params: (u32, u32),
        /// Qubits the declaration takes, and qubits given.
        qubits: (u32, u32),
    },
    /// A declared gate cannot be written in the target dialect: its body uses
    /// something the dialect lacks (for example a barrier in an OpenQASM 3
    /// gate body, or `arcsin` in OpenQASM 2.0). Export fails rather than change
    /// the gate.
    GateNotExpressible {
        /// The declared gate.
        name: String,
        /// What the dialect cannot express. A boxed `str` keeps the error as
        /// small as it was (see `GateSignature`).
        reason: Box<str>,
    },
    /// An OpenQASM 3 export would exceed one of the importer's budgets (its
    /// source size, inputs, declarations, register size, instructions,
    /// expression nodes, or the expansion of declared-gate calls), so it could
    /// not be read back.
    ExportLimit {
        /// The importer's budget, by the name its errors give it
        /// (`MAX_INSTRUCTIONS`).
        limit: &'static str,
        /// Its value.
        max: usize,
    },
    /// The circuit calls two different gates declared under the same name
    /// (only possible when combining calls from separately imported programs),
    /// which one OpenQASM 2.0 program cannot declare.
    ConflictingGateDefinitions { name: String },
    /// QIR bitcode export requires an external assembler (`llvm-as`) that is
    /// not available on `PATH`.
    QirAssemblyToolNotFound { tool: String },
    /// The external assembler failed while converting QIR text (`.ll`) to
    /// LLVM bitcode (`.bc`).
    QirAssemblyFailed { tool: String, message: String },
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
            CircuitError::NonFiniteParam => write!(
                f,
                "gate parameter resolved to a non-finite value (NaN or infinity)"
            ),
            CircuitError::DivisionByZero => {
                write!(f, "division by zero in parameter expression")
            }
            CircuitError::UnknownExpression => write!(
                f,
                "gate refers to an expression that is not part of this circuit (an ExprId belongs to the circuit that returned it from add_expr)"
            ),
            CircuitError::InvalidExpression { reason } => {
                write!(f, "invalid parameter expression: {reason}")
            }
            CircuitError::QubitOutOfRange { qubit, num_qubits } => write!(
                f,
                "qubit index {qubit} out of range for circuit with {num_qubits} qubits"
            ),
            CircuitError::IdenticalQubits { qubit } => write!(
                f,
                "two-qubit gate requires distinct qubits, got ({qubit}, {qubit})"
            ),
            CircuitError::QubitAlreadyMeasured { qubit } => write!(
                f,
                "gate acts on qubit {qubit} after it was measured; Polypus circuits use terminal measurement (contract C-4)"
            ),
            CircuitError::Parse { line, message } => {
                write!(f, "QASM parse error at line {line}: {message}")
            }
            CircuitError::GateSignature {
                name,
                params,
                qubits,
            } => write!(
                f,
                "gate '{name}' takes {} angle(s) and {} qubit(s), got {} and {}",
                params.0, qubits.0, params.1, qubits.1
            ),
            CircuitError::GateNotExpressible { name, reason } => {
                write!(f, "declared gate '{name}' {reason}")
            }
            CircuitError::ExportLimit { limit, max } => write!(
                f,
                "the export would exceed the OpenQASM 3 importer's {limit} ({max}), so it could not be read back"
            ),
            CircuitError::ConflictingGateDefinitions { name } => write!(
                f,
                "the circuit calls two different gates declared as '{name}'; one OpenQASM 2.0 program cannot declare a gate twice"
            ),
            CircuitError::QirAssemblyToolNotFound { tool } => write!(
                f,
                "QIR bitcode export requires '{tool}', but it was not found on PATH"
            ),
            CircuitError::QirAssemblyFailed { tool, message } => write!(
                f,
                "QIR bitcode export failed while running '{tool}': {message}"
            ),
        }
    }
}

impl std::error::Error for CircuitError {}
