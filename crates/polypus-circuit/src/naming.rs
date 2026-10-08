//! Identifiers an export writes: which names a dialect accepts, and the
//! deterministic renaming of the others.
//!
//! A name that is valid in the target dialect, not reserved there, and not
//! yet used in its scope is written as it is. Any other name is replaced:
//! every character the dialect does not allow in an identifier becomes `_`;
//! a letter naming the kind (`g` for a gate, `p` for a parameter, `q` for a
//! qubit argument or register, `c` for a bit register) is prefixed if the
//! result cannot start an identifier; `_` is appended to a reserved word; and
//! `_1`, `_2`, … is appended until the name is unused. In OpenQASM 3, a name
//! longer than the importer's `MAX_TOKEN_BYTES` (4096 bytes; only an
//! OpenQASM 2.0 import has such names) is renamed too, cut to its first
//! 4075 bytes, so that any suffix still fits. The names that can be kept are
//! claimed before any is renamed, so a renamed name never takes one that was
//! valid and unique already, and no two names in a scope collide.
//!
//! Gate definitions are named in the order they are written, and each
//! definition's formal parameters, then its qubit arguments, avoid every
//! program-level name (gates, registers, inputs).

use crate::custom_gate::GateDefinition;
use crate::expr::Dialect;
use crate::qasm3_import::{
    is_ident_continue, is_ident_start, is_reserved, MAX_TOKEN_BYTES, STDGATES,
};
use crate::qasm_import::builtin_gate;
use std::collections::{HashMap, HashSet};

/// OpenQASM 2.0's keywords and the builtins of its expression grammar.
const QASM2_KEYWORDS: &[&str] = &[
    "OPENQASM", "include", "qreg", "creg", "gate", "opaque", "barrier", "measure", "reset", "if",
    "U", "CX", "pi", "sin", "cos", "tan", "exp", "ln", "sqrt",
];

fn starts(c: char, dialect: Dialect) -> bool {
    match dialect {
        // The OpenQASM 2.0 specification's `[a-z][A-Za-z0-9_]*`.
        Dialect::Qasm2 => c.is_ascii_lowercase(),
        Dialect::Qasm3 => is_ident_start(c),
    }
}

fn continues(c: char, dialect: Dialect) -> bool {
    match dialect {
        Dialect::Qasm2 => c.is_ascii_alphanumeric() || c == '_',
        Dialect::Qasm3 => is_ident_continue(c),
    }
}

/// Whether no declaration may take `name` in `dialect`: a keyword or builtin,
/// or a gate the target's standard library (`qelib1.inc`, `stdgates.inc`)
/// defines — which the exports always include.
fn reserved(name: &str, dialect: Dialect) -> bool {
    match dialect {
        Dialect::Qasm2 => QASM2_KEYWORDS.contains(&name) || builtin_gate(name).is_some(),
        Dialect::Qasm3 => is_reserved(name) || STDGATES.contains(&name),
    }
}

/// Longest name `fresh` builds before its suffix: `_` and up to 20 digits
/// then still fit in `MAX_TOKEN_BYTES`.
const MAX_BASE_BYTES: usize = MAX_TOKEN_BYTES - 21;

/// Whether `name` can be written as it is in `dialect`, given the names
/// already `taken` in its scope.
pub(crate) fn keepable(name: &str, dialect: Dialect, taken: &HashSet<String>) -> bool {
    let mut chars = name.chars();
    (dialect == Dialect::Qasm2 || name.len() <= MAX_TOKEN_BYTES)
        && chars.next().is_some_and(|c| starts(c, dialect))
        && chars.all(|c| continues(c, dialect))
        && !reserved(name, dialect)
        && !taken.contains(name)
}

/// The name `wanted` is replaced by (see the module documentation).
pub(crate) fn fresh(
    wanted: &str,
    letter: char,
    dialect: Dialect,
    taken: &HashSet<String>,
) -> String {
    let mut base: String = wanted
        .chars()
        .map(|c| if continues(c, dialect) { c } else { '_' })
        .collect();
    if !base.chars().next().is_some_and(|c| starts(c, dialect)) {
        base.insert(0, letter);
    }
    if dialect == Dialect::Qasm3 && base.len() > MAX_BASE_BYTES {
        let mut end = MAX_BASE_BYTES;
        while !base.is_char_boundary(end) {
            end -= 1;
        }
        base.truncate(end);
    }
    if reserved(&base, dialect) {
        base.push('_');
    }
    if !taken.contains(&base) {
        return base;
    }
    (1usize..)
        .map(|k| format!("{base}_{k}"))
        .find(|name| !taken.contains(name))
        .unwrap_or(base)
}

/// The final name of each of `wanted`, in one scope: kept where possible,
/// renamed otherwise, never clashing with `taken` (which receives them all)
/// or with each other.
pub(crate) fn assign(
    wanted: &[&str],
    letter: char,
    dialect: Dialect,
    taken: &mut HashSet<String>,
) -> Vec<String> {
    let mut kept: Vec<Option<String>> = vec![None; wanted.len()];
    for (slot, name) in kept.iter_mut().zip(wanted) {
        if keepable(name, dialect, taken) {
            taken.insert(name.to_string());
            *slot = Some(name.to_string());
        }
    }
    wanted
        .iter()
        .zip(kept)
        .map(|(name, kept)| {
            kept.unwrap_or_else(|| {
                let renamed = fresh(name, letter, dialect, taken);
                taken.insert(renamed.clone());
                renamed
            })
        })
        .collect()
}

/// The names an export writes for gate definitions, by address: each
/// definition's own, and those of its formal parameters and qubits.
#[derive(Default)]
pub(crate) struct DefinitionNames {
    gates: HashMap<*const GateDefinition, String>,
    formals: HashMap<*const GateDefinition, (Vec<String>, Vec<String>)>,
}

impl DefinitionNames {
    /// Name `definitions`, in order, avoiding `taken`, which receives the
    /// names.
    pub(crate) fn name_gates(
        &mut self,
        definitions: &[&GateDefinition],
        dialect: Dialect,
        taken: &mut HashSet<String>,
    ) {
        let wanted: Vec<&str> = definitions.iter().map(|d| d.name()).collect();
        let names = assign(&wanted, 'g', dialect, taken);
        for (&definition, name) in definitions.iter().zip(names) {
            self.gates.insert(definition, name);
        }
    }

    /// Name each definition's formal parameters, then its qubits, avoiding
    /// `taken` (left as it was: formals are local to their definition).
    pub(crate) fn name_formals(
        &mut self,
        definitions: &[&GateDefinition],
        dialect: Dialect,
        taken: &mut HashSet<String>,
    ) {
        for &definition in definitions {
            let params: Vec<&str> = definition
                .param_names()
                .iter()
                .map(String::as_str)
                .collect();
            let params = assign(&params, 'p', dialect, taken);
            let qubits: Vec<&str> = definition
                .qubit_names()
                .iter()
                .map(String::as_str)
                .collect();
            let qubits = assign(&qubits, 'q', dialect, taken);
            // `assign` added exactly these names, none of which was taken.
            for name in params.iter().chain(&qubits) {
                taken.remove(name);
            }
            self.formals.insert(definition, (params, qubits));
        }
    }

    /// The name `definition` is written with (its own, if it was not named
    /// here).
    pub(crate) fn gate<'a>(&'a self, definition: &'a GateDefinition) -> &'a str {
        self.gates
            .get(&(definition as *const GateDefinition))
            .map_or(definition.name(), String::as_str)
    }

    /// The names of `definition`'s formal parameters and qubits (its own, if
    /// they were not named here).
    pub(crate) fn formals<'a>(
        &'a self,
        definition: &'a GateDefinition,
    ) -> (&'a [String], &'a [String]) {
        self.formals
            .get(&(definition as *const GateDefinition))
            .map_or(
                (definition.param_names(), definition.qubit_names()),
                |(p, q)| (p.as_slice(), q.as_slice()),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names_are_kept_and_the_rest_renamed_without_collisions() {
        let mut taken: HashSet<String> = ["q".to_string()].into();
        let names = assign(
            &["a", "_gate_q_0", "1x", "h", "a", "q", "a_1"],
            'g',
            Dialect::Qasm3,
            &mut taken,
        );
        assert_eq!(names, ["a", "_gate_q_0", "g1x", "h_", "a_2", "q_1", "a_1"]);
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn overlong_names_are_cut_in_openqasm3_only() {
        let long = "é".repeat(3000);
        let mut taken = HashSet::new();
        let names = assign(&[&long, &long], 'g', Dialect::Qasm3, &mut taken);
        assert!(names[0].len() <= MAX_BASE_BYTES && long.starts_with(&names[0]));
        assert_eq!(names[1], format!("{}_1", names[0]));
        assert!(names.iter().all(|n| n.len() <= MAX_TOKEN_BYTES));
        let ascii = "a".repeat(MAX_TOKEN_BYTES + 1);
        assert!(!keepable(&ascii, Dialect::Qasm3, &HashSet::new()));
        assert!(keepable(&ascii, Dialect::Qasm2, &HashSet::new()));
        assert!(keepable(&ascii[1..], Dialect::Qasm3, &HashSet::new()));
    }

    #[test]
    fn openqasm2_names_start_with_a_lowercase_letter() {
        let mut taken = HashSet::new();
        let names = assign(
            &["_gate_q_0", "Theta", "θ", "p0", "rzz", "measure", "ok_1"],
            'p',
            Dialect::Qasm2,
            &mut taken,
        );
        assert_eq!(
            names,
            [
                "p_gate_q_0",
                "pTheta",
                "p_",
                "p0",
                "rzz_",
                "measure_",
                "ok_1"
            ]
        );
    }
}
