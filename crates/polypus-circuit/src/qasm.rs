//! OpenQASM 2.0 serialization.
//!
//! Emits the standard header (`OPENQASM 2.0; include "qelib1.inc";`), the
//! declarations of the gates the circuit calls, one `qreg`/`creg`
//! declaration pair, and one statement per instruction using the standard
//! `qelib1.inc` gate names. Angle values are written as the shortest decimal
//! that reads back as the same `f64` ([`crate::expr::fmt_number`]), so an
//! export loses no precision: `pi/2` is `1.5707963267948966`, which a backend
//! reads as exactly the angle the circuit holds.
//!
//! A declaration imported from OpenQASM 2.0 is re-emitted as it was written,
//! unless it is named `q` or `c`, like the registers the export declares in
//! OpenQASM 2.0's single namespace: then it is printed and renamed as below,
//! and so is every declaration that calls it. One imported from OpenQASM 3
//! is printed from its definition, on one line
//! (`gate name(a,b) x,y { h x; cx x,y; }`), with `log` written as `ln` and
//! `**` as `^`; its names are renamed where OpenQASM 2.0 cannot take them
//! (see [`crate::naming`]), and one it cannot express (a body using
//! `arcsin`, `arccos` or `arctan`) is an error naming the gate.
//!
//! Note: `rzz`/`rxx` are part of Qiskit's `qelib1.inc` (and accepted by
//! `QuantumCircuit.from_qasm_str`). Strict parsers limited to the original
//! paper version of `qelib1.inc` may need Qiskit's
//! `qasm2.LEGACY_CUSTOM_INSTRUCTIONS` to recognise them.

use crate::custom_gate::{BodyOp, CustomGate, GateDefinition};
use crate::error::CircuitError;
use crate::expr::{fmt_number, print, Dialect, ExprArena};
use crate::gate::{GateInstruction, GateParam};
use crate::naming::{reserved, DefinitionNames};
use crate::qasm_import::normalized;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

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

/// The registers every OpenQASM 2.0 export declares (when non-empty).
/// OpenQASM 2.0 has one namespace, so no declaration may take these names.
const REGISTERS: [&str; 2] = ["q", "c"];

/// The declarations an export writes, and the names it calls them by.
struct Declarations<'c> {
    /// Imported from OpenQASM 2.0: re-emitted verbatim, under their names.
    verbatim: Vec<&'c GateDefinition>,
    /// Printed from their definitions, callees first: those imported from
    /// OpenQASM 2.0 that cannot keep their name, in source order, then those
    /// imported from OpenQASM 3, in order of first use.
    printed: Vec<&'c GateDefinition>,
    names: DefinitionNames,
}

impl<'c> Declarations<'c> {
    /// The declarations the calls in `gates` need. With none imported from
    /// OpenQASM 3 (a declaration only calls gates of its own program) and
    /// none named `q` or `c`, this is exactly the verbatim path. An OpenQASM
    /// 2.0 declaration named like a register (or a reserved word), and every
    /// one whose text calls such a gate, is printed instead, so it can be
    /// renamed. The verbatim declarations keep their names and the printed
    /// ones avoid them, the registers `q` and `c`, and the reserved words and
    /// `qelib1.inc` gates.
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
        // Source order puts every callee before its callers, so one pass
        // finds the callers of a renamed declaration. Names identify OpenQASM
        // 2.0 declarations here: `declared_gates` rejects two under one name.
        let mut verbatim = Vec::new();
        let mut renamed: Vec<&GateDefinition> = Vec::new();
        for definition in declared_gates(qasm2.into_iter())? {
            let name = definition.name();
            let clashes = REGISTERS.contains(&name) || reserved(name, Dialect::Qasm2);
            if clashes
                || definition
                    .callees()
                    .any(|callee| renamed.iter().any(|r| r.name() == callee.name()))
            {
                renamed.push(definition);
            } else {
                verbatim.push(definition);
            }
        }
        let mut printed = renamed.clone();
        printed.extend(printed_order(qasm3.into_iter()));
        let mut names = DefinitionNames::default();
        if !printed.is_empty() {
            let mut taken: HashSet<String> =
                verbatim.iter().map(|d| d.name().to_string()).collect();
            taken.extend(REGISTERS.map(String::from));
            names.name_gates(&printed, Dialect::Qasm2, &mut taken);
            names.name_formals(&printed, Dialect::Qasm2, &mut taken);
            names.alias_qasm2(&renamed);
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
///
/// A barrier that lists every qubit in order and the `measure q[k] -> c[k]` of
/// every qubit in order are written in their canonical form, `barrier q;` and
/// `measure q -> c;`, exactly as the importer reads them back (contract C-2),
/// so `to_qasm2(from_qasm2(to_qasm2(c)))` is `to_qasm2(c)` byte for byte.
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

    let angles = Angles { params, exprs };
    let mut stack = Vec::new();
    for gate in normalized(gates, num_qubits) {
        match statement(&gate) {
            Statement::Gate(name) => write_gate(&mut out, name, &gate, angles, &mut stack)?,
            Statement::Call(call) => write_call(&mut out, call, &declarations, angles, &mut stack)?,
            Statement::Barrier(qubits) => write_barrier(&mut out, qubits),
            Statement::Measure { qubit, cbit } => {
                let _ = writeln!(out, "measure q[{qubit}] -> c[{cbit}];");
            }
            Statement::MeasureAll => write_measure_all(&mut out, num_qubits, num_clbits),
        }
    }

    Ok(out)
}

/// The values an export resolves angles with: the parameters' and the
/// expressions'.
#[derive(Clone, Copy)]
struct Angles<'a> {
    params: &'a [f64],
    exprs: &'a ExprArena,
}

impl Angles<'_> {
    /// The value of `p`, with `stack` as the evaluation scratch space.
    fn resolve(&self, p: &GateParam, stack: &mut Vec<f64>) -> Result<f64, CircuitError> {
        p.resolve(self.params, self.exprs, stack)
    }
}

/// How an instruction is written.
enum Statement<'g> {
    /// A `qelib1.inc` gate of this name, its angles and its operands.
    Gate(&'static str),
    /// A call of a gate the program declares.
    Call(&'g CustomGate),
    /// `barrier` on these qubits; every qubit when empty.
    Barrier(&'g [usize]),
    /// `measure q[qubit] -> c[cbit]`.
    Measure { qubit: usize, cbit: usize },
    /// `measure q -> c`.
    MeasureAll,
}

/// How `gate` is written. Exhaustive over the vocabulary (no wildcard arm),
/// so a new instruction cannot compile until it says how it is exported.
fn statement(gate: &GateInstruction) -> Statement<'_> {
    use GateInstruction as G;
    Statement::Gate(match gate {
        G::H(_) => "h",
        G::X(_) => "x",
        G::Y(_) => "y",
        G::Z(_) => "z",
        G::S(_) => "s",
        G::T(_) => "t",
        G::Sdg(_) => "sdg",
        G::Tdg(_) => "tdg",
        G::Id(_) => "id",
        G::Sx(_) => "sx",
        G::Sxdg(_) => "sxdg",
        G::Rx { .. } => "rx",
        G::Ry { .. } => "ry",
        G::Rz { .. } => "rz",
        G::U { .. } => "u3",
        G::U0 { .. } => "u0",
        G::P { .. } => "p",
        G::U1 { .. } => "u1",
        G::U2 { .. } => "u2",
        G::UGate { .. } => "u",
        G::Cx(..) => "cx",
        G::Cz(..) => "cz",
        G::Swap(..) => "swap",
        G::Cy(..) => "cy",
        G::Ch(..) => "ch",
        G::Csx(..) => "csx",
        G::Rzz { .. } => "rzz",
        G::Rxx { .. } => "rxx",
        G::Cp { .. } => "cp",
        G::Crx { .. } => "crx",
        G::Cry { .. } => "cry",
        G::Crz { .. } => "crz",
        G::Cu1 { .. } => "cu1",
        G::Cu3 { .. } => "cu3",
        G::Cu { .. } => "cu",
        G::Ccx(..) => "ccx",
        G::Cswap(..) => "cswap",
        G::Rccx(..) => "rccx",
        G::Rc3x(..) => "rc3x",
        G::C3x(..) => "c3x",
        G::C3sqrtx(..) => "c3sqrtx",
        G::C4x(..) => "c4x",
        G::Custom(call) => return Statement::Call(call),
        G::Barrier(qubits) => return Statement::Barrier(qubits),
        G::Measure { qubit, cbit } => {
            return Statement::Measure {
                qubit: *qubit,
                cbit: *cbit,
            }
        }
        G::MeasureAll => return Statement::MeasureAll,
    })
}

/// `name(angles) operands;` for a `qelib1.inc` gate: its angles in QASM
/// argument order (resolved in that order, so the first unbound or
/// non-finite one is the error), without parentheses when it has none, and
/// its operands in operand order.
fn write_gate(
    out: &mut String,
    name: &str,
    gate: &GateInstruction,
    angles: Angles<'_>,
    stack: &mut Vec<f64>,
) -> Result<(), CircuitError> {
    let values = gate
        .params()
        .map(|p| Ok(fmt_number(angles.resolve(p, stack)?)))
        .collect::<Result<Vec<String>, CircuitError>>()?;
    let operands: Vec<String> = gate
        .acts_on()
        .qubits()
        .iter()
        .map(|q| format!("q[{q}]"))
        .collect();
    if values.is_empty() {
        let _ = writeln!(out, "{name} {};", operands.join(","));
    } else {
        let _ = writeln!(out, "{name}({}) {};", values.join(","), operands.join(","));
    }
    Ok(())
}

/// A call of a declared gate, under the name the export declares it by.
fn write_call(
    out: &mut String,
    call: &CustomGate,
    declarations: &Declarations<'_>,
    angles: Angles<'_>,
    stack: &mut Vec<f64>,
) -> Result<(), CircuitError> {
    let operands: Vec<String> = call.qubits().iter().map(|q| format!("q[{q}]")).collect();
    let values = call
        .params()
        .iter()
        .map(|p| angles.resolve(p, stack))
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
        let angles: Vec<String> = values.iter().map(|&v| fmt_number(v)).collect();
        let _ = writeln!(out, "{name}({}) {};", angles.join(","), operands.join(","));
    }
    Ok(())
}

/// `barrier q;` for every qubit (an empty list), else the qubits listed.
fn write_barrier(out: &mut String, qubits: &[usize]) {
    if qubits.is_empty() {
        out.push_str("barrier q;\n");
    } else {
        let args: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
        let _ = writeln!(out, "barrier {};", args.join(","));
    }
}

/// `measure q -> c;`, or one `measure` per qubit when the registers differ
/// in size and the register-wide form would be invalid.
fn write_measure_all(out: &mut String, num_qubits: usize, num_clbits: usize) {
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

#[cfg(test)]
mod tests {
    use crate::ParameterizedCircuit;

    /// The statements of every instruction the exporter writes, on five qubits,
    /// declared gates included: a program of them is exported as itself.
    const EVERY_STATEMENT: &str = "\
OPENQASM 2.0;
include \"qelib1.inc\";
gate g(t) a,b { rz(t/2) a; cx a,b; }
gate k a { x a; }
qreg q[5];
creg c[2];
h q[0];
x q[1];
y q[2];
z q[3];
s q[4];
t q[0];
sdg q[1];
tdg q[2];
id q[3];
sx q[4];
sxdg q[0];
rx(0.1) q[0];
ry(0.2) q[1];
rz(-0.3) q[2];
u3(0.1,0.2,0.3) q[3];
u0(0.5) q[4];
p(0.5) q[0];
u1(0.5) q[1];
u2(0.1,0.2) q[2];
u(0.1,0.2,0.3) q[3];
cx q[0],q[1];
cz q[1],q[2];
swap q[2],q[3];
cy q[3],q[4];
ch q[4],q[0];
csx q[0],q[2];
rzz(0.4) q[0],q[1];
rxx(0.5) q[1],q[2];
cp(0.6) q[2],q[3];
crx(0.1) q[0],q[1];
cry(0.2) q[1],q[2];
crz(0.3) q[2],q[3];
cu1(0.4) q[3],q[4];
cu3(0.1,0.2,0.3) q[4],q[0];
cu(0.1,0.2,0.3,0.4) q[0],q[1];
ccx q[2],q[0],q[1];
cswap q[3],q[1],q[2];
rccx q[2],q[3],q[4];
rc3x q[0],q[1],q[2],q[3];
c3x q[1],q[2],q[3],q[4];
c3sqrtx q[0],q[2],q[3],q[4];
c4x q[4],q[2],q[0],q[3],q[1];
g(0.5) q[0],q[1];
k q[2];
barrier q[0],q[2];
barrier q;
measure q[0] -> c[1];
";

    /// Every instruction is written as its statement, declared gates
    /// included, so the exporter's split by instruction family (#419) cannot
    /// change one character of it.
    #[test]
    fn every_instruction_is_exported_as_its_statement() {
        let imported = ParameterizedCircuit::from_qasm2(EVERY_STATEMENT).unwrap();
        assert_eq!(imported.to_qasm2_with_params(&[]).unwrap(), EVERY_STATEMENT);
    }

    /// Angles that are free parameters and expressions of them are written
    /// with the values they are exported with; `measure_all` is written
    /// register-wide when the registers match and qubit by qubit otherwise.
    #[test]
    fn parameter_and_expression_angles_are_exported_with_their_values() {
        use crate::{Param, ParamExpr};
        let mut qc = ParameterizedCircuit::new(2);
        let twice = qc.add_expr(2.0 * ParamExpr::param(1)).unwrap();
        let qc = qc
            .rx(0, Param(0))
            .ry(1, twice)
            .u(0, Param(1), twice, 0.3)
            .cu(0, 1, twice, Param(0), Param(1), -0.0)
            .measure_all();
        assert_eq!(
            qc.to_qasm2_with_params(&[0.5, 0.25]).unwrap(),
            "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
rx(0.5) q[0];
ry(0.5) q[1];
u3(0.25,0.5,0.3) q[0];
cu(0.5,0.5,0.25,-0.0) q[0],q[1];
measure q -> c;
"
        );
        assert_eq!(
            ParameterizedCircuit::new(2)
                .h(1)
                .measure(0, 2)
                .measure_all()
                .to_qasm2_with_params(&[])
                .unwrap(),
            "\
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[3];
h q[1];
measure q[0] -> c[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
"
        );
    }

    /// Every angle is exported as the shortest decimal that reads back as the
    /// same `f64`, in a form the importer reads: positional, scientific
    /// outside `1e-5 <= |v| < 1e16`, and negative zero with its sign.
    #[test]
    fn angles_are_exported_at_full_precision() {
        let cases = [
            (std::f64::consts::FRAC_PI_2, "1.5707963267948966"),
            (-std::f64::consts::PI, "-3.141592653589793"),
            (0.1 + 0.2, "0.30000000000000004"),
            (0.5, "0.5"),
            (-1.0, "-1.0"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.5e-7, "1.5e-7"),
            (1e16, "1.0e16"),
        ];
        for (value, text) in cases {
            let statement = format!("rz({text}) q[0];");
            let source =
                format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n{statement}\n");
            let imported = ParameterizedCircuit::from_qasm2(&source).unwrap();
            assert_eq!(
                imported.to_qasm2_with_params(&[]).unwrap(),
                source,
                "{value:e}"
            );
            let exported = ParameterizedCircuit::new(1)
                .rz(0, value)
                .to_qasm2_with_params(&[])
                .unwrap();
            assert_eq!(
                exported.lines().last(),
                Some(statement.as_str()),
                "{value:e}"
            );
            let back = ParameterizedCircuit::from_qasm2(&exported).unwrap();
            assert_eq!(back.to_qasm2_with_params(&[]).unwrap(), exported);
        }
    }
}
