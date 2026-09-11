//! Error type for native cost-observable evaluation.
//!
//! Kept **pure** (no PyO3): this crate must be usable from any Rust project.
//! The one concession to the FFI is [`ObservableError::External`], a type-erased
//! box that a non-native implementation (e.g. a Python-callback observable in the
//! `polypus` crate) uses to carry its own error — including a `PyErr` — back up
//! through the pure trait. The `polypus` crate downcasts it to re-raise the
//! original Python exception verbatim.

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
    /// A construction-time invariant was violated (index out of range, `i == j`
    /// coupling, non-finite coefficient). Produced by the `Qubo`/`Ising`
    /// constructors and surfaced as a `ValueError` at the Python boundary.
    Invalid(String),
    /// A foreign error from a non-native implementation, type-erased so this
    /// pure crate needn't depend on PyO3. The FFI crate downcasts it back to its
    /// concrete type (a `PyErr`) to re-raise the original exception verbatim.
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
