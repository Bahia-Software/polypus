//! OpenQASM 2.0 serialization.
//!
//! Emits the standard header (`OPENQASM 2.0; include "qelib1.inc";`), the
//! declarations of the gates the circuit calls, one `qreg`/`creg`
//! declaration pair, and one statement per instruction using the standard
//! `qelib1.inc` gate names. Angle values are written with 12 decimal places.
//!
//! A declaration imported from OpenQASM 2.0 is re-emitted as it was written.
//! One imported from OpenQASM 3 is printed from its definition, on one line
//! (`gate name(a,b) x,y { h x; cx x,y; }`), with `log` written as `ln` and
//! `**` as `^`; its names are renamed where OpenQASM 2.0 cannot take them
//! (see [`crate::naming`]), and one it cannot express (a body using
//! `arcsin`, `arccos` or `arctan`) is an error naming the gate.
//!
//! Note: `rzz`/`rxx` are part of Qiskit's `qelib1.inc` (and accepted by
//! `QuantumCircuit.from_qasm_str`). Strict parsers limited to the original
//! paper version of `qelib1.inc` may need Qiskit's
//! `qasm2.LEGACY_CUSTOM_INSTRUCTIONS` to recognise them.

use crate::custom_gate::{BodyOp, GateDefinition};
use crate::error::CircuitError;
use crate::expr::{print, Dialect, ExprArena};
use crate::gate::{GateInstruction, GateParam};
use crate::naming::DefinitionNames;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Format an angle with 12 decimal places (≥ 10 required for round-tripping
/// optimizer outputs without observable precision loss).
fn fmt_angle(value: f64) -> String {
    format!("{value:.12}")
}

/// The OpenQASM 2.0 declarations `roots` need: their definitions and,
/// transitively, those of the declared gates their bodies call — each once,
/// in source order (a body only calls gates declared before it, so this
/// order also puts every declaration before its first use).
///
/// # Errors
///
/// [`CircuitError::ConflictingGateDefinitions`] if two *different* definitions
/// share a name (only possible when combining calls from separately imported
/// programs): one OpenQASM 2.0 program cannot declare both.
fn declared_gates<'c>(
    roots: impl Iterator<Item = &'c GateDefinition>,
) -> Result<Vec<&'c GateDefinition>, CircuitError> {
    let mut found: HashMap<&str, &GateDefinition> = HashMap::new();
    let mut pending: Vec<&GateDefinition> = roots.collect();
    while let Some(definition) = pending.pop() {
        if let Some(&seen) = found.get(definition.name()) {
            if !std::ptr::eq(seen, definition) && seen != definition {
                return Err(CircuitError::ConflictingGateDefinitions {
                    name: definition.name().to_string(),
                });
            }
            continue;
        }
        found.insert(definition.name(), definition);
        pending.extend(definition.callees().map(|callee| &**callee));
    }
    let mut ordered: Vec<&GateDefinition> = found.into_values().collect();
    ordered.sort_by(|a, b| a.ordinal().cmp(&b.ordinal()).then(a.name().cmp(b.name())));
    Ok(ordered)
}

/// The declarations an export writes, and the names it calls them by.
struct Declarations<'c> {
    /// Imported from OpenQASM 2.0: re-emitted verbatim, under their names.
    verbatim: Vec<&'c GateDefinition>,
    /// Imported from OpenQASM 3: printed from their definitions, callees
    /// first, in order of first use.
    printed: Vec<&'c GateDefinition>,
    names: DefinitionNames,
}

impl<'c> Declarations<'c> {
    /// The declarations the calls in `gates` need. With none imported from
    /// OpenQASM 3 (a declaration only calls gates of its own program), this
    /// is exactly the verbatim path; otherwise the verbatim declarations keep
    /// their names and the printed ones avoid them, the registers `q` and
    /// `c`, and the reserved words and `qelib1.inc` gates.
    fn new(gates: &'c [GateInstruction]) -> Result<Self, CircuitError> {
        // One pass over the instructions, allocating nothing without calls.
        let (mut qasm2, mut qasm3) = (Vec::new(), Vec::new());
        for gate in gates {
            if let GateInstruction::Custom(call) = gate {
                let definition = call.definition();
                match definition.dialect() {
                    Dialect::Qasm2 => qasm2.push(definition),
                    Dialect::Qasm3 => qasm3.push(definition),
                }
            }
        }
        let verbatim = declared_gates(qasm2.into_iter())?;
        let printed = printed_order(qasm3.into_iter());
        let mut names = DefinitionNames::default();
        if !printed.is_empty() {
            let mut taken: HashSet<String> =
                verbatim.iter().map(|d| d.name().to_string()).collect();
            taken.extend(["q".to_string(), "c".to_string()]);
            names.name_gates(&printed, Dialect::Qasm2, &mut taken);
            names.name_formals(&printed, Dialect::Qasm2, &mut taken);
        }
        Ok(Declarations {
            verbatim,
            printed,
            names,
        })
    }

    fn write(&self, out: &mut String) -> Result<(), CircuitError> {
        for definition in &self.verbatim {
            out.push_str(definition.declaration());
            out.push('\n');
        }
        for definition in &self.printed {
            write_definition(out, definition, &self.names)?;
        }
        Ok(())
    }
}

/// `roots` and the definitions they call, transitively: each once, callees
/// first, in order of first use. Nesting is bounded by `MAX_GATE_NESTING`, so
/// the recursion is shallow.
fn printed_order<'c>(roots: impl Iterator<Item = &'c GateDefinition>) -> Vec<&'c GateDefinition> {
    fn visit<'c>(
        definition: &'c GateDefinition,
        seen: &mut HashSet<*const GateDefinition>,
        order: &mut Vec<&'c GateDefinition>,
    ) {
        if seen.insert(definition) {
            for callee in definition.callees() {
                visit(callee, seen, order);
            }
            order.push(definition);
        }
    }
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for root in roots {
        visit(root, &mut seen, &mut order);
    }
    order
}

/// `definition` as one OpenQASM 2.0 line: `gate name(a,b) x,y { h x; }`.
fn write_definition(
    out: &mut String,
    definition: &GateDefinition,
    names: &DefinitionNames,
) -> Result<(), CircuitError> {
    let unexpressible = |reason: String| CircuitError::GateNotExpressible {
        name: definition.name().to_string(),
        reason: reason.into(),
    };
    let (formal_params, formal_qubits) = names.formals(definition);
    let operands = |qubits: &[usize]| {
        qubits
            .iter()
            .map(|&q| formal_qubits.get(q).map(String::as_str))
            .collect::<Option<Vec<&str>>>()
            .map(|operands| operands.join(","))
            .ok_or_else(|| unexpressible("a statement names a qubit it does not declare".into()))
    };
    let _ = write!(out, "gate {}", names.gate(definition));
    if !formal_params.is_empty() {
        let _ = write!(out, "({})", formal_params.join(","));
    }
    let _ = write!(out, " {} {{", formal_qubits.join(","));
    for op in definition.body() {
        let (callee, exprs, qubits) = match op {
            BodyOp::Builtin {
                gate,
                params,
                qubits,
            } => (gate.exported_name(), params, qubits),
            BodyOp::Call {
                definition: callee,
                params,
                qubits,
            } => (names.gate(callee), params, qubits),
            BodyOp::Barrier(qubits) => {
                let _ = write!(out, " barrier {};", operands(qubits)?);
                continue;
            }
        };
        out.push(' ');
        out.push_str(callee);
        if !exprs.is_empty() {
            out.push('(');
            for (i, expr) in exprs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                print(expr.nodes(), Dialect::Qasm2, formal_params, out).map_err(|e| {
                    unexpressible(format!("cannot be written in OpenQASM 2.0: {e}"))
                })?;
            }
            out.push(')');
        }
        let _ = write!(out, " {};", operands(qubits)?);
    }
    out.push_str(" }\n");
    Ok(())
}

/// Serialize a gate sequence to a complete OpenQASM 2.0 program.
///
/// `params` supplies values for any unresolved [`GateParam::Param`], and
/// `exprs` holds the expressions of any [`GateParam::Expr`], evaluated with
/// them; pass an empty slice and arena for fully concrete circuits.
pub(crate) fn write_qasm2(
    num_qubits: usize,
    num_clbits: usize,
    gates: &[GateInstruction],
    params: &[f64],
    exprs: &ExprArena,
) -> Result<String, CircuitError> {
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    // The declarations of the gates the circuit calls (and of the gates those
    // call), before the registers — where Qiskit's exporter puts them too.
    let declarations = Declarations::new(gates)?;
    declarations.write(&mut out)?;
    if num_qubits > 0 {
        let _ = writeln!(out, "qreg q[{num_qubits}];");
    }
    if num_clbits > 0 {
        let _ = writeln!(out, "creg c[{num_clbits}];");
    }

    let mut stack = Vec::new();
    let mut angle = |p: &GateParam| -> Result<String, CircuitError> {
        Ok(fmt_angle(p.resolve(params, exprs, &mut stack)?))
    };
    // Calls of declared gates resolve their angles apart: `angle` holds `stack`.
    let mut call_stack = Vec::new();

    for gate in gates {
        match gate {
            GateInstruction::H(q) => {
                let _ = writeln!(out, "h q[{q}];");
            }
            GateInstruction::X(q) => {
                let _ = writeln!(out, "x q[{q}];");
            }
            GateInstruction::Y(q) => {
                let _ = writeln!(out, "y q[{q}];");
            }
            GateInstruction::Z(q) => {
                let _ = writeln!(out, "z q[{q}];");
            }
            GateInstruction::S(q) => {
                let _ = writeln!(out, "s q[{q}];");
            }
            GateInstruction::T(q) => {
                let _ = writeln!(out, "t q[{q}];");
            }
            GateInstruction::Sdg(q) => {
                let _ = writeln!(out, "sdg q[{q}];");
            }
            GateInstruction::Tdg(q) => {
                let _ = writeln!(out, "tdg q[{q}];");
            }
            GateInstruction::Id(q) => {
                let _ = writeln!(out, "id q[{q}];");
            }
            GateInstruction::Rx { qubit, theta } => {
                let _ = writeln!(out, "rx({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Ry { qubit, theta } => {
                let _ = writeln!(out, "ry({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Rz { qubit, theta } => {
                let _ = writeln!(out, "rz({}) q[{qubit}];", angle(theta)?);
            }
            GateInstruction::Cx(c, t) => {
                let _ = writeln!(out, "cx q[{c}],q[{t}];");
            }
            GateInstruction::Cz(c, t) => {
                let _ = writeln!(out, "cz q[{c}],q[{t}];");
            }
            GateInstruction::Swap(q0, q1) => {
                let _ = writeln!(out, "swap q[{q0}],q[{q1}];");
            }
            GateInstruction::Rzz { q0, q1, theta } => {
                let _ = writeln!(out, "rzz({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Rxx { q0, q1, theta } => {
                let _ = writeln!(out, "rxx({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Cp { q0, q1, theta } => {
                let _ = writeln!(out, "cp({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::U {
                qubit,
                theta,
                phi,
                lam,
            } => {
                let _ = writeln!(
                    out,
                    "u3({},{},{}) q[{qubit}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Sx(q) => {
                let _ = writeln!(out, "sx q[{q}];");
            }
            GateInstruction::Sxdg(q) => {
                let _ = writeln!(out, "sxdg q[{q}];");
            }
            GateInstruction::Cy(c, t) => {
                let _ = writeln!(out, "cy q[{c}],q[{t}];");
            }
            GateInstruction::Ch(c, t) => {
                let _ = writeln!(out, "ch q[{c}],q[{t}];");
            }
            GateInstruction::Csx(c, t) => {
                let _ = writeln!(out, "csx q[{c}],q[{t}];");
            }
            GateInstruction::Ccx(c0, c1, t) => {
                let _ = writeln!(out, "ccx q[{c0}],q[{c1}],q[{t}];");
            }
            GateInstruction::Cswap(c, t0, t1) => {
                let _ = writeln!(out, "cswap q[{c}],q[{t0}],q[{t1}];");
            }
            GateInstruction::Crx {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "crx({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Cry {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "cry({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Crz {
                control,
                target,
                theta,
            } => {
                let _ = writeln!(out, "crz({}) q[{control}],q[{target}];", angle(theta)?);
            }
            GateInstruction::Cu1 { q0, q1, theta } => {
                let _ = writeln!(out, "cu1({}) q[{q0}],q[{q1}];", angle(theta)?);
            }
            GateInstruction::Cu3 {
                control,
                target,
                theta,
                phi,
                lam,
            } => {
                let _ = writeln!(
                    out,
                    "cu3({},{},{}) q[{control}],q[{target}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Cu {
                control,
                target,
                theta,
                phi,
                lam,
                gamma,
            } => {
                let _ = writeln!(
                    out,
                    "cu({},{},{},{}) q[{control}],q[{target}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?,
                    angle(gamma)?
                );
            }
            GateInstruction::U0 { qubit, gamma } => {
                let _ = writeln!(out, "u0({}) q[{qubit}];", angle(gamma)?);
            }
            GateInstruction::P { qubit, lam } => {
                let _ = writeln!(out, "p({}) q[{qubit}];", angle(lam)?);
            }
            GateInstruction::U1 { qubit, lam } => {
                let _ = writeln!(out, "u1({}) q[{qubit}];", angle(lam)?);
            }
            GateInstruction::U2 { qubit, phi, lam } => {
                let _ = writeln!(out, "u2({},{}) q[{qubit}];", angle(phi)?, angle(lam)?);
            }
            GateInstruction::UGate {
                qubit,
                theta,
                phi,
                lam,
            } => {
                let _ = writeln!(
                    out,
                    "u({},{},{}) q[{qubit}];",
                    angle(theta)?,
                    angle(phi)?,
                    angle(lam)?
                );
            }
            GateInstruction::Rccx(a, b, c) => {
                let _ = writeln!(out, "rccx q[{a}],q[{b}],q[{c}];");
            }
            GateInstruction::Rc3x(a, b, c, d) => {
                let _ = writeln!(out, "rc3x q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C3x(a, b, c, d) => {
                let _ = writeln!(out, "c3x q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C3sqrtx(a, b, c, d) => {
                let _ = writeln!(out, "c3sqrtx q[{a}],q[{b}],q[{c}],q[{d}];");
            }
            GateInstruction::C4x(a, b, c, d, e) => {
                let _ = writeln!(out, "c4x q[{a}],q[{b}],q[{c}],q[{d}],q[{e}];");
            }
            GateInstruction::Custom(call) => {
                let operands: Vec<String> =
                    call.qubits().iter().map(|q| format!("q[{q}]")).collect();
                let values = call
                    .params()
                    .iter()
                    .map(|p| p.resolve(params, exprs, &mut call_stack))
                    .collect::<Result<Vec<f64>, _>>()?;
                // A call with free angles is checked through its body for the
                // values it is exported with.
                if call.has_free_angles() {
                    call.check_body(&values)?;
                }
                let name = declarations.names.gate(call.definition());
                if values.is_empty() {
                    let _ = writeln!(out, "{name} {};", operands.join(","));
                } else {
                    let angles: Vec<String> = values.iter().map(|&v| fmt_angle(v)).collect();
                    let _ = writeln!(out, "{name}({}) {};", angles.join(","), operands.join(","));
                }
            }
            GateInstruction::Barrier(qubits) => {
                if qubits.is_empty() {
                    out.push_str("barrier q;\n");
                } else {
                    let args: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
                    let _ = writeln!(out, "barrier {};", args.join(","));
                }
            }
            GateInstruction::Measure { qubit, cbit } => {
                let _ = writeln!(out, "measure q[{qubit}] -> c[{cbit}];");
            }
            GateInstruction::MeasureAll => {
                if num_clbits == num_qubits {
                    out.push_str("measure q -> c;\n");
                } else {
                    // Register sizes differ (mixed Measure/MeasureAll usage):
                    // `measure q -> c;` would be invalid, expand per qubit.
                    for q in 0..num_qubits {
                        let _ = writeln!(out, "measure q[{q}] -> c[{q}];");
                    }
                }
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::fmt_angle;

    #[test]
    fn angles_have_at_least_ten_decimals() {
        assert_eq!(fmt_angle(0.5), "0.500000000000");
        assert_eq!(fmt_angle(-1.0), "-1.000000000000");
        assert_eq!(fmt_angle(std::f64::consts::FRAC_PI_2), "1.570796326795");
    }
}
