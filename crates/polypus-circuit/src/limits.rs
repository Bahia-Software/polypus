//! The budgets both OpenQASM importers enforce on untrusted input.
//!
//! Each bounds the memory or work a source can cause, whatever its size, and
//! each is a [`CircuitError::Parse`] when exceeded, with the same message in
//! both dialects. The OpenQASM 3 exporter stays within them (see
//! [`crate::qasm3`]), so whatever it writes imports again.

use crate::error::CircuitError;
use crate::qasm_import::err;

/// Largest source accepted, in bytes. The OpenQASM 3 exporter writes at most
/// this much, so everything it writes can be imported again.
pub(crate) const MAX_SOURCE_BYTES: usize = 32 << 20;

/// Longest identifier, number or string literal, in bytes.
pub(crate) const MAX_TOKEN_BYTES: usize = 4096;

/// Most `gate` declarations.
pub(crate) const MAX_DECLARATIONS: usize = 10_000;

/// Most expression nodes (numbers, names, operators, calls) a program keeps:
/// every expression in OpenQASM 3, the bodies of gate declarations in
/// OpenQASM 2.0 (whose top-level angles are evaluated as they are read).
pub(crate) const MAX_PROGRAM_NODES: usize = 4_000_000;

/// Most instructions a program may produce, after register broadcasting.
pub(crate) const MAX_INSTRUCTIONS: usize = 4_000_000;

/// Most qubit operands all `barrier` statements may list together, after
/// register expansion: `barrier q;` over `n` qubits lists `n`.
pub(crate) const MAX_BARRIER_OPERANDS: usize = 4_000_000;

/// Upper bound on the *total* number of declared qubits (and, separately, of
/// declared classical bits). Guards against a hostile `qreg q[4000000000];`
/// materialising a multi-gigabyte index vector during argument expansion. One
/// million bits is far beyond any simulable circuit (the statevector backend
/// caps out around 30 qubits) yet cheap to reject.
pub(crate) const MAX_REGISTER_BITS: usize = 1_000_000;

/// Upper bound on the body statements the importer visits, over the whole
/// program, to validate calls of declared gates (each call's angles
/// are evaluated through its whole body, so a non-finite angle or a division
/// by zero is rejected at parse time like everywhere else, contract C-2).
/// Bounds the parse time of hostile input that calls a large declaration many
/// times; real programs stay orders of magnitude below it.
pub(crate) const MAX_VALIDATED_EXPANSION: usize = 20_000_000;

/// Reject a source longer than [`MAX_SOURCE_BYTES`] before reading it.
pub(crate) fn check_source(src: &str) -> Result<(), CircuitError> {
    if src.len() > MAX_SOURCE_BYTES {
        return Err(err(
            1,
            format!(
                "source of {} bytes exceeds MAX_SOURCE_BYTES ({MAX_SOURCE_BYTES} bytes)",
                src.len()
            ),
        ));
    }
    Ok(())
}

/// Reject a token of `bytes` bytes longer than [`MAX_TOKEN_BYTES`].
pub(crate) fn check_token(bytes: usize, line: usize) -> Result<(), CircuitError> {
    if bytes > MAX_TOKEN_BYTES {
        return Err(err(
            line,
            format!("token longer than MAX_TOKEN_BYTES ({MAX_TOKEN_BYTES} bytes)"),
        ));
    }
    Ok(())
}

/// Reject one more `gate` declaration after `declared`.
pub(crate) fn check_declarations(declared: usize, line: usize) -> Result<(), CircuitError> {
    if declared >= MAX_DECLARATIONS {
        return Err(err(
            line,
            format!("more than MAX_DECLARATIONS ({MAX_DECLARATIONS}) gate declarations"),
        ));
    }
    Ok(())
}

/// Reject `more` instructions after the `pushed` ones, when together they
/// exceed [`MAX_INSTRUCTIONS`]. Checked before a broadcast is expanded, so a
/// statement over the budget never materialises its instructions.
pub(crate) fn check_instructions(
    pushed: usize,
    more: usize,
    line: usize,
) -> Result<(), CircuitError> {
    if pushed.saturating_add(more) > MAX_INSTRUCTIONS {
        return Err(err(
            line,
            format!("the program has more than MAX_INSTRUCTIONS ({MAX_INSTRUCTIONS}) instructions"),
        ));
    }
    Ok(())
}

/// Spend `operands` barrier operands from the program's `spent` ones,
/// rejecting them beyond [`MAX_BARRIER_OPERANDS`].
pub(crate) fn spend_barrier_operands(
    spent: &mut usize,
    operands: usize,
    line: usize,
) -> Result<(), CircuitError> {
    *spent = spent.saturating_add(operands);
    if *spent > MAX_BARRIER_OPERANDS {
        return Err(err(
            line,
            format!(
                "the program's barriers list more than MAX_BARRIER_OPERANDS ({MAX_BARRIER_OPERANDS}) qubits"
            ),
        ));
    }
    Ok(())
}

/// Spend `nodes` expression nodes from the program's `spent` ones, rejecting
/// them beyond [`MAX_PROGRAM_NODES`].
pub(crate) fn spend_nodes(
    spent: &mut usize,
    nodes: usize,
    line: usize,
) -> Result<(), CircuitError> {
    *spent = spent.saturating_add(nodes);
    if *spent > MAX_PROGRAM_NODES {
        return Err(err(
            line,
            format!(
                "the program's expressions exceed MAX_PROGRAM_NODES ({MAX_PROGRAM_NODES} nodes)"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_budget_admits_its_maximum_and_rejects_one_more() {
        assert!(check_source(&" ".repeat(MAX_SOURCE_BYTES)).is_ok());
        assert!(check_source(&" ".repeat(MAX_SOURCE_BYTES + 1)).is_err());
        assert!(check_token(MAX_TOKEN_BYTES, 1).is_ok());
        assert!(check_token(MAX_TOKEN_BYTES + 1, 1).is_err());
        assert!(check_declarations(MAX_DECLARATIONS - 1, 1).is_ok());
        assert!(check_declarations(MAX_DECLARATIONS, 1).is_err());
        assert!(check_instructions(MAX_INSTRUCTIONS - 10, 10, 1).is_ok());
        assert!(check_instructions(MAX_INSTRUCTIONS - 10, 11, 1).is_err());
        assert!(check_instructions(usize::MAX, usize::MAX, 1).is_err());

        let mut spent = 0;
        assert!(spend_barrier_operands(&mut spent, MAX_BARRIER_OPERANDS, 1).is_ok());
        assert!(spend_barrier_operands(&mut spent, 1, 1).is_err());
        let mut spent = MAX_PROGRAM_NODES - 1;
        assert!(spend_nodes(&mut spent, 1, 1).is_ok());
        assert!(spend_nodes(&mut spent, 1, 1).is_err());
    }
}
