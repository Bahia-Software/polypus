//! Error type for native cost-observable evaluation.
//!
//! Kept **pure** (no PyO3): this crate must be usable from any Rust project.
//! The one concession to the FFI is [`ObservableError::External`], a type-erased
//! box that a non-native implementation (e.g. a Python-callback observable in the
//! `polypus` crate) uses to carry its own error back up through the pure trait.
//! The callback observable boxes its Python exception as
//! `polypus_infrastructure::DisplaySafePyErr`, the only carrier the `polypus`
//! crate downcasts to in order to re-raise the original exception verbatim.

use std::fmt;

/// A failure while turning measurement counts into an expectation value.
#[derive(Debug)]
pub enum ObservableError {
    /// A counts key was shorter than the observable's variable count, so a
    /// referenced variable index has no corresponding bit.
    BitWidthMismatch {
        /// Variables the observable references.
        num_vars: usize,
        /// Length of the offending counts key.
        key_len: usize,
    },
    /// A counts key contained a character other than `'0'` or `'1'`.
    InvalidBitstring(String),
    /// An invalid input: a construction-time invariant was violated (index out
    /// of range, `i == j` coupling, non-finite coefficient, coefficients whose
    /// `Σ|c|·|scale|` exceeds `MAX_WEIGHT_BOUND`, a Pauli string naming a qubit
    /// twice), or data passed for evaluation cannot be evaluated (a probability
    /// outside `[0, 1]`, counts whose total overflows `u64`). Produced by the
    /// constructors of every observable here, by `group_qubit_wise`, and by the
    /// evaluation of the native reducers. At the Python boundary a construction
    /// error is a `ValueError`, and an evaluation error a
    /// `polypus.EvaluationError`.
    Invalid(String),
    /// A foreign error from a non-native implementation, type-erased so this
    /// pure crate needn't depend on PyO3. The FFI crate downcasts it to
    /// `DisplaySafePyErr` to re-raise the original exception verbatim; a bare
    /// `PyErr` boxed here surfaces as `polypus.EvaluationError` with its message.
    ///
    /// The bound is `Send` (not `Send + Sync`): rayon's `collect` into a
    /// `Result` only requires the error type to be `Send`, and the callback path
    /// never produces this variant from inside a rayon worker.
    External(Box<dyn std::error::Error + Send + 'static>),
}

impl fmt::Display for ObservableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObservableError::BitWidthMismatch { num_vars, key_len } => write!(
                f,
                "measurement bitstring of length {key_len} is too short for an \
                 observable over {num_vars} variables"
            ),
            ObservableError::InvalidBitstring(key) => {
                write!(
                    f,
                    "measurement bitstring contains a non-binary character: {key:?}"
                )
            }
            ObservableError::Invalid(msg) => write!(f, "invalid observable: {msg}"),
            ObservableError::External(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ObservableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // `&(dyn Error + Send)` unsize-coerces to `&(dyn Error)`.
            ObservableError::External(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}
