//! Error type for the statevector simulator.

use std::fmt;

/// Errors that can occur while preparing or running a simulation.
/// `#[non_exhaustive]`: new failure modes are added as variants.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    /// A gate angle is still an expression of free parameters
    /// ([`GateParam::Expr`](polypus_circuit::GateParam::Expr)). Only the
    /// circuit that holds the expression can evaluate it, and `assign_parameters`
    /// always does; like an unbound parameter, this can only happen for a
    /// manually assembled circuit.
    UnboundExpression,
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
    /// A gate references a qubit index outside the circuit's register
    /// (`qubit >= num_qubits`). The fluent builder rejects this at construction,
    /// as does [`ConcreteCircuit::try_new`](polypus_circuit::ConcreteCircuit::try_new),
    /// but a gate passed to [`Statevector::apply`](crate::Statevector::apply) can
    /// carry one. The
    /// simulator rejects it *before any kernel runs*: an out-of-range qubit index
    /// yields an out-of-bounds amplitude index (`1 << qubit`), which the kernels
    /// would otherwise only catch with a panic.
    QubitIndexOutOfRange {
        /// The offending qubit index.
        qubit: usize,
        /// The circuit's qubit count; valid indices are `0..num_qubits`.
        num_qubits: usize,
    },
    /// A measurement targets a classical bit index `>= max`
    /// ([`MAX_CLBITS`](crate::MAX_CLBITS)), which a sampled outcome's `usize`
    /// key cannot represent. Rejected before sampling: shifting the bit in would
    /// overflow (a panic with overflow checks, a silently wrong key without).
    ClbitOutOfRange {
        /// The offending classical bit index.
        cbit: usize,
        /// Valid indices are `0..max`.
        max: usize,
    },
    /// A unitary names the same qubit more than once (e.g. `Cx(0, 0)`). The
    /// fluent builder rejects this at construction, but a hand-assembled
    /// [`ConcreteCircuit`](polypus_circuit::ConcreteCircuit) can carry one. The
    /// multi-qubit kernels assume distinct operands, so it is rejected before
    /// any kernel runs.
    RepeatedQubit {
        /// The qubit named more than once.
        qubit: usize,
    },
    /// The instruction is newer than this simulator: [`GateInstruction`] is
    /// `#[non_exhaustive]`, so a gate added to `polypus-circuit` reaches here
    /// until the simulator implements it.
    ///
    /// [`GateInstruction`]: polypus_circuit::GateInstruction
    UnsupportedInstruction {
        /// The instruction's `Debug` rendering.
        instruction: String,
    },
    /// The caller asked for the simulation to stop before it finished, through
    /// the cancellation hook passed to
    /// [`Simulator::run_cancellable`](crate::Simulator::run_cancellable). The
    /// partially evolved statevector is discarded: a run that stopped part-way
    /// through the gate sequence is not a state any circuit describes.
    Cancelled,
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
            SimError::UnboundExpression => write!(
                f,
                "circuit contains an unbound parameter expression; bind all parameters before simulating"
            ),
            SimError::NonFiniteAmplitude => {
                write!(f, "a gate angle resolved to a non-finite value (NaN or infinity)")
            }
            SimError::GateAfterMeasure { qubit } => write!(
                f,
                "gate acts on qubit {qubit} after it was measured; Polypus circuits use terminal measurement (contract C-4)"
            ),
            SimError::QubitIndexOutOfRange { qubit, num_qubits } => write!(
                f,
                "gate references qubit {qubit} but the circuit has only {num_qubits} qubit(s) (valid indices are 0..{num_qubits})"
            ),
            SimError::ClbitOutOfRange { cbit, max } => write!(
                f,
                "measurement targets classical bit {cbit} but the statevector backend keys counts on at most {max} classical bits (valid indices are 0..{max})"
            ),
            SimError::RepeatedQubit { qubit } => write!(
                f,
                "gate names qubit {qubit} more than once; a gate's qubit operands must be distinct"
            ),
            SimError::UnsupportedInstruction { instruction } => write!(
                f,
                "the statevector simulator does not support the instruction {instruction}"
            ),
            SimError::Cancelled => {
                write!(f, "the simulation was cancelled before it completed")
            }
        }
    }
}

impl std::error::Error for SimError {}
