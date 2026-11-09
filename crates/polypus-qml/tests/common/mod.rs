//! Shared helpers for the `polypus-qml` integration tests.
//!
//! Each file under `tests/` compiles as its own binary, so code shared between
//! them lives in a `common/mod.rs` submodule (the `common/mod.rs` path, not a
//! bare `common.rs`, which would compile as a test binary of its own) that each
//! test file pulls in with `mod common;`.

use std::collections::HashMap;

/// Convert `polypus-sim`'s `HashMap<state_index, count>` into the C-3 bitstring
/// format `expectation_from_counts` expects. Same pattern as
/// `crates/polypus/src/infrastructure/native.rs`: standard binary formatting,
/// no bit reversal — the character at `width - 1 - k` is qubit `k`.
pub fn to_bitstring_counts(raw: HashMap<usize, u64>, width: usize) -> HashMap<String, u64> {
    raw.into_iter()
        .map(|(state, count)| (format!("{state:0width$b}"), count))
        .collect()
}
