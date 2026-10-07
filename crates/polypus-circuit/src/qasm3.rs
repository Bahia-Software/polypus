//! OpenQASM 3 serialization, in the profile [`crate::qasm3_import`] reads.
//!
//! The output is canonical, so that importing it and exporting again gives
//! the same bytes:
//!
//! ```text
//! OPENQASM 3.0;
//! include "stdgates.inc";
//! input float[64] <parameter>;      one per free parameter, in index order
//! gate <name>(<p>, …) <q>, … {       every gate definition the circuit needs,
//!   <statement>;                     callees first, in order of first use
//! }
//! qubit[<n>] q;                      unless n = 0
//! bit[<m>] c;                        unless m = 0
//! <statement>;                       one per instruction
//! ```
//!
//! Instructions keep their names; `u` is written as the builtin `U`. Polypus
//! instructions that `stdgates.inc` lacks (`rzz`, `rxx`, `sxdg`, `csx`, `cu1`,
//! `cu3`, `u0`, `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x`) are written with gate
//! definitions whose bodies reproduce their matrices exactly, with `U`, `u2`
//! and `u3` read as Qiskit's (the profile's convention). Declared gates are
//! printed from their definition, never copied from their source text.
//! Numbers are written as the shortest decimal that reads back as the same
//! `f64` ([`crate::expr::fmt_number`]), and expressions with the fewest
//! parentheses.

use crate::circuit::ParameterizedCircuit;
use crate::custom_gate::{BodyOp, GateDefinition};
use crate::custom_gate::{MAX_GATE_EXPANSION, MAX_GATE_NESTING};
use crate::error::CircuitError;
use crate::expr::{
    fmt_number, print, printed_len, Dialect, ExprArena, Unprintable, MAX_EXPR_NODES,
};
use crate::gate::{GateInstruction, GateParam};
use crate::naming::{fresh, DefinitionNames};
use crate::qasm3_import::{
    parse_qasm3, MAX_DECLARATIONS, MAX_INPUTS, MAX_INSTRUCTIONS, MAX_PROGRAM_NODES,
    MAX_SOURCE_BYTES,
};
use crate::qasm_import::{normalize, MAX_REGISTER_BITS, MAX_VALIDATED_EXPANSION};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::OnceLock;

// ─────────────────────────── Helper definitions ──────────────────────────

/// The Polypus instructions `stdgates.inc` lacks, in the order of
/// [`HELPER_PROGRAM`]'s calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Helper {
    Rzz,
    Rxx,
    Sxdg,
    Csx,
    Cu1,
    Cu3,
    U0,
    Rccx,
    Rc3x,
    C3x,
    C3sqrtx,
    C4x,
}

/// Their definitions, in canonical form, then one call of each. Every body
/// reproduces the instruction's matrix exactly (checked with full matrices in
/// `polypus-sim`'s OpenQASM 3 tests): `rzz` and `rxx` as `cx`-conjugated
/// `rz`, `sxdg` as `h·sdg·h`, `csx` as `h`-conjugated `cp(π/2)`, `cu1` as
/// `cp`, `cu3` as `cu` with no phase, `u0` as the identity, and the others as
/// their `qelib1.inc` definitions, which the simulator also uses.
const HELPER_PROGRAM: &str = "OPENQASM 3.0;
include \"stdgates.inc\";
gate rzz(p0) a, b {
  cx a, b;
  rz(p0) b;
  cx a, b;
}
gate rxx(p0) a, b {
  h a;
  h b;
  cx a, b;
  rz(p0) b;
  cx a, b;
  h a;
  h b;
}
gate sxdg a {
  h a;
  sdg a;
  h a;
}
gate csx a, b {
  h b;
  cp(pi/2.0) a, b;
  h b;
}
gate cu1(p0) a, b {
  cp(p0) a, b;
}
gate cu3(p0, p1, p2) a, b {
  cu(p0, p1, p2, 0.0) a, b;
}
gate u0(p0) a {
}
gate rccx a, b, c {
  h c;
  t c;
  cx b, c;
  tdg c;
  cx a, c;
  t c;
  cx b, c;
  tdg c;
  h c;
}
gate rc3x a, b, c, d {
  h d;
  t d;
  cx c, d;
  tdg d;
  h d;
  cx a, d;
  t d;
  cx b, d;
  tdg d;
  cx a, d;
  t d;
  cx b, d;
  tdg d;
  h d;
  t d;
  cx c, d;
  tdg d;
  h d;
}
gate c3x a, b, c, d {
  h d;
  p(pi/8.0) a;
  p(pi/8.0) b;
  p(pi/8.0) c;
  p(pi/8.0) d;
  cx a, b;
  p(-pi/8.0) b;
  cx a, b;
  cx b, c;
  p(-pi/8.0) c;
  cx a, c;
  p(pi/8.0) c;
  cx b, c;
  p(-pi/8.0) c;
  cx a, c;
  cx c, d;
  p(-pi/8.0) d;
  cx b, d;
  p(pi/8.0) d;
  cx c, d;
  p(-pi/8.0) d;
  cx a, d;
  p(pi/8.0) d;
  cx c, d;
  p(-pi/8.0) d;
  cx b, d;
  p(pi/8.0) d;
  cx c, d;
  p(-pi/8.0) d;
  cx a, d;
  h d;
}
gate c3sqrtx a, b, c, d {
  h d;
  cp(pi/8.0) a, d;
  h d;
  cx a, b;
  h d;
  cp(-pi/8.0) b, d;
  h d;
  cx a, b;
  h d;
  cp(pi/8.0) b, d;
  h d;
  cx b, c;
  h d;
  cp(-pi/8.0) c, d;
  h d;
  cx a, c;
  h d;
  cp(pi/8.0) c, d;
  h d;
  cx b, c;
  h d;
  cp(-pi/8.0) c, d;
  h d;
  cx a, c;
  h d;
  cp(pi/8.0) c, d;
  h d;
}
gate c4x a, b, c, d, e {
  h e;
  cp(pi/2.0) d, e;
  h e;
  c3x a, b, c, d;
  h e;
  cp(-pi/2.0) d, e;
  h e;
  c3x a, b, c, d;
  c3sqrtx a, b, c, e;
}
qubit[5] q;
rzz(0.0) q[0], q[1];
rxx(0.0) q[0], q[1];
sxdg q[0];
csx q[0], q[1];
cu1(0.0) q[0], q[1];
cu3(0.0, 0.0, 0.0) q[0], q[1];
u0(0.0) q[0];
rccx q[0], q[1], q[2];
rc3x q[0], q[1], q[2], q[3];
c3x q[0], q[1], q[2], q[3];
c3sqrtx q[0], q[1], q[2], q[3];
c4x q[0], q[1], q[2], q[3], q[4];
";

/// The definition the exporter writes for `helper`.
fn helper_definition(helper: Helper) -> Result<&'static GateDefinition, CircuitError> {
    static HELPERS: OnceLock<Result<ParameterizedCircuit, CircuitError>> = OnceLock::new();
    let program = HELPERS
        .get_or_init(|| parse_qasm3(HELPER_PROGRAM))
        .as_ref()
        .map_err(Clone::clone)?;
    match program.gates.get(helper as usize) {
        Some(GateInstruction::Custom(call)) => Ok(call.definition()),
        _ => Err(CircuitError::GateNotExpressible {
            name: format!("{helper:?}"),
            reason: "the exporter's definition of it is missing".into(),
        }),
    }
}

/// How OpenQASM 3 spells an instruction: a `stdgates.inc` gate (or `U`), or
/// a gate the exporter defines. `None` for calls of declared gates, barriers
/// and measurements, which have statements of their own.
enum Spelling {
    Std(&'static str),
    Defined(Helper),
}

fn spelling(gate: &GateInstruction) -> Option<Spelling> {
    use GateInstruction as G;
    use Spelling::{Defined, Std};
    Some(match gate {
        G::H(_) => Std("h"),
        G::X(_) => Std("x"),
        G::Y(_) => Std("y"),
        G::Z(_) => Std("z"),
        G::S(_) => Std("s"),
        G::T(_) => Std("t"),
        G::Sdg(_) => Std("sdg"),
        G::Tdg(_) => Std("tdg"),
        G::Id(_) => Std("id"),
        G::Rx { .. } => Std("rx"),
        G::Ry { .. } => Std("ry"),
        G::Rz { .. } => Std("rz"),
        G::Cx(..) => Std("cx"),
        G::Cz(..) => Std("cz"),
        G::Swap(..) => Std("swap"),
        G::Cp { .. } => Std("cp"),
        G::U { .. } => Std("u3"),
        G::Sx(_) => Std("sx"),
        G::Cy(..) => Std("cy"),
        G::Ch(..) => Std("ch"),
        G::Ccx(..) => Std("ccx"),
        G::Cswap(..) => Std("cswap"),
        G::Crx { .. } => Std("crx"),
        G::Cry { .. } => Std("cry"),
        G::Crz { .. } => Std("crz"),
        G::Cu { .. } => Std("cu"),
        G::P { .. } => Std("p"),
        G::U1 { .. } => Std("u1"),
        G::U2 { .. } => Std("u2"),
        // Qiskit's `u` is the builtin `U` under the profile's convention.
        G::UGate { .. } => Std("U"),
        G::Rzz { .. } => Defined(Helper::Rzz),
        G::Rxx { .. } => Defined(Helper::Rxx),
        G::Sxdg(_) => Defined(Helper::Sxdg),
        G::Csx(..) => Defined(Helper::Csx),
        G::Cu1 { .. } => Defined(Helper::Cu1),
        G::Cu3 { .. } => Defined(Helper::Cu3),
        G::U0 { .. } => Defined(Helper::U0),
        G::Rccx(..) => Defined(Helper::Rccx),
        G::Rc3x(..) => Defined(Helper::Rc3x),
        G::C3x(..) => Defined(Helper::C3x),
        G::C3sqrtx(..) => Defined(Helper::C3sqrtx),
        G::C4x(..) => Defined(Helper::C4x),
        G::Custom(_) | G::Barrier(_) | G::Measure { .. } | G::MeasureAll => return None,
    })
}

/// The instruction a built-in statement of a gate body stands for.
fn body_instruction(gate: &crate::qasm_import::BuiltinGate) -> GateInstruction {
    let params = vec![GateParam::Fixed(0.0); gate.params];
    let qubits: Vec<usize> = (0..gate.qubits).collect();
    (gate.build)(&params, &qubits)
}

// ──────────────────────────────── Names ──────────────────────────────────

/// Every name an export writes.
struct Names {
    params: Vec<String>,
    /// Gate definitions (helpers and declared gates) and their formals.
    gates: DefinitionNames,
    qreg: String,
    creg: String,
}

// ─────────────────────────────── Export ──────────────────────────────────

/// What reading a definition back costs the importer: the body statements a
/// full expansion of one call visits, and its nesting depth (as
/// `GateDefinition` counts them), with the Polypus instructions
/// `stdgates.inc` lacks counted as the calls of helpers the output makes them.
/// An OpenQASM 2.0 declaration that calls `c4x` grows by the helper's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Size {
    expansion: usize,
    depth: usize,
}

/// Each definition's [`Size`], by address.
type Sizes = HashMap<*const GateDefinition, Size>;

/// The gate definitions `gates` needs, callees first, in order of first use,
/// and the size of each: the declared gates they call and the helpers their
/// instructions (and those of the declared gates' bodies) need. Nesting is
/// bounded by `MAX_GATE_NESTING`, so the recursion is shallow.
///
/// # Errors
///
/// [`CircuitError::GateNotExpressible`] for a definition that, read back,
/// would nest deeper or expand further than the importer accepts.
fn definitions(gates: &[GateInstruction]) -> Result<(Vec<&GateDefinition>, Sizes), CircuitError> {
    fn visit<'c>(
        definition: &'c GateDefinition,
        sizes: &mut Sizes,
        order: &mut Vec<&'c GateDefinition>,
    ) -> Result<Size, CircuitError> {
        if let Some(&size) = sizes.get(&(definition as *const GateDefinition)) {
            return Ok(size);
        }
        let mut size = Size {
            expansion: 0,
            depth: 1,
        };
        for op in definition.body() {
            let callee = match op {
                BodyOp::Builtin { gate, .. } => match spelling(&body_instruction(gate)) {
                    Some(Spelling::Defined(helper)) => {
                        Some(visit(helper_definition(helper)?, sizes, order)?)
                    }
                    _ => None,
                },
                BodyOp::Call {
                    definition: callee, ..
                } => Some(visit(callee, sizes, order)?),
                BodyOp::Barrier(_) => None,
            };
            size.expansion = size.expansion.saturating_add(1);
            if let Some(callee) = callee {
                size.expansion = size.expansion.saturating_add(callee.expansion);
                size.depth = size.depth.max(callee.depth + 1);
            }
        }
        let beyond = |what: String| {
            CircuitError::GateNotExpressible {
            name: definition.name().to_string(),
            reason: format!(
                "cannot be written in OpenQASM 3 within the importer's limits: with the Polypus instructions stdgates.inc lacks written as gate calls, it {what}"
            )
            .into(),
        }
        };
        if size.depth > MAX_GATE_NESTING {
            return Err(beyond(format!(
                "nests gate calls more than {MAX_GATE_NESTING} levels deep"
            )));
        }
        if size.expansion > MAX_GATE_EXPANSION {
            return Err(beyond(format!(
                "expands to more than {MAX_GATE_EXPANSION} instructions"
            )));
        }
        sizes.insert(definition, size);
        order.push(definition);
        Ok(size)
    }
    let mut sizes = HashMap::new();
    let mut order = Vec::new();
    for gate in gates {
        match gate {
            GateInstruction::Custom(call) => {
                visit(call.definition(), &mut sizes, &mut order)?;
            }
            other => {
                if let Some(Spelling::Defined(helper)) = spelling(other) {
                    visit(helper_definition(helper)?, &mut sizes, &mut order)?;
                }
            }
        }
    }
    Ok((order, sizes))
}

/// The importer's program-wide budgets, spent as the output is written, so
/// that the exporter never writes a program [`parse_qasm3`] would reject.
#[derive(Default)]
struct Spent {
    instructions: usize,
    nodes: usize,
    expansion: usize,
}

fn over(limit: &'static str, max: usize) -> CircuitError {
    CircuitError::ExportLimit { limit, max }
}

impl Spent {
    /// `n` instructions, as the importer pushes them (`c = measure q;`
    /// pushes one measurement per qubit).
    fn instructions(&mut self, n: usize) -> Result<(), CircuitError> {
        self.instructions = self.instructions.saturating_add(n);
        if self.instructions > MAX_INSTRUCTIONS {
            return Err(over("MAX_INSTRUCTIONS", MAX_INSTRUCTIONS));
        }
        Ok(())
    }

    /// One expression the importer reads as `nodes` nodes.
    fn expression(&mut self, nodes: usize) -> Result<(), CircuitError> {
        if nodes > MAX_EXPR_NODES {
            return Err(over("MAX_EXPR_NODES", MAX_EXPR_NODES));
        }
        self.nodes = self.nodes.saturating_add(nodes);
        if self.nodes > MAX_PROGRAM_NODES {
            return Err(over("MAX_PROGRAM_NODES", MAX_PROGRAM_NODES));
        }
        Ok(())
    }

    /// A call of a declared gate whose full expansion visits `n` statements.
    fn call(&mut self, n: usize) -> Result<(), CircuitError> {
        self.expansion = self.expansion.saturating_add(n);
        if self.expansion > MAX_VALIDATED_EXPANSION {
            return Err(over("MAX_VALIDATED_EXPANSION", MAX_VALIDATED_EXPANSION));
        }
        Ok(())
    }
}

/// Serialize a circuit to a program in the OpenQASM 3 profile. With
/// `bound = Some(values)` the circuit is bound first and has no inputs.
pub(crate) fn write_qasm3(
    circuit: &ParameterizedCircuit,
    bound: Option<&[f64]>,
) -> Result<String, CircuitError> {
    let concrete;
    let no_exprs = ExprArena::default();
    let (gates, exprs, params) = match bound {
        Some(values) => {
            concrete = circuit.assign_parameters(values)?;
            (&concrete.gates, &no_exprs, Vec::new())
        }
        None => (&circuit.gates, &circuit.exprs, circuit.param_names()),
    };
    let num_qubits = circuit.num_qubits;
    let num_clbits = crate::circuit::num_clbits(num_qubits, gates);
    if params.len() > MAX_INPUTS {
        return Err(over("MAX_INPUTS", MAX_INPUTS));
    }
    if num_qubits.max(num_clbits) > MAX_REGISTER_BITS {
        return Err(over("MAX_REGISTER_BITS", MAX_REGISTER_BITS));
    }
    let (order, sizes) = definitions(gates)?;
    if order.len() > MAX_DECLARATIONS {
        return Err(over("MAX_DECLARATIONS", MAX_DECLARATIONS));
    }

    // Names: the parameters' are kept; gate names, then register names, then
    // each definition's formals avoid everything before them.
    let mut taken: HashSet<String> = params.iter().cloned().collect();
    let mut gates_named = DefinitionNames::default();
    gates_named.name_gates(&order, Dialect::Qasm3, &mut taken);
    let qreg = fresh("q", 'q', Dialect::Qasm3, &taken);
    taken.insert(qreg.clone());
    let creg = fresh("c", 'c', Dialect::Qasm3, &taken);
    taken.insert(creg.clone());
    gates_named.name_formals(&order, Dialect::Qasm3, &mut taken);
    let names = Names {
        params,
        gates: gates_named,
        qreg,
        creg,
    };

    let mut out = String::new();
    let mut spent = Spent::default();
    out.push_str("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    for name in &names.params {
        let _ = writeln!(out, "input float[64] {name};");
    }
    check_size(&out)?;
    for definition in &order {
        write_definition(&mut out, definition, &names, &mut spent)?;
        check_size(&out)?;
    }
    if num_qubits > 0 {
        let _ = writeln!(out, "qubit[{num_qubits}] {};", names.qreg);
    }
    if num_clbits > 0 {
        let _ = writeln!(out, "bit[{num_clbits}] {};", names.creg);
    }
    let expansion = |definition: &GateDefinition| {
        sizes
            .get(&(definition as *const GateDefinition))
            .map_or(0, |size| size.expansion)
    };
    for gate in normalize(gates, num_qubits) {
        spent.instructions(match gate {
            GateInstruction::MeasureAll => num_qubits,
            _ => 1,
        })?;
        match &gate {
            GateInstruction::Custom(call) => spent.call(expansion(call.definition()))?,
            other => {
                if let Some(Spelling::Defined(helper)) = spelling(other) {
                    spent.call(expansion(helper_definition(helper)?))?;
                }
            }
        }
        write_statement(
            &mut out, &gate, &names, exprs, num_qubits, num_clbits, &mut spent,
        )?;
        check_size(&out)?;
    }
    Ok(out)
}

fn check_size(out: &str) -> Result<(), CircuitError> {
    if out.len() > MAX_SOURCE_BYTES {
        Err(over("MAX_SOURCE_BYTES", MAX_SOURCE_BYTES))
    } else {
        Ok(())
    }
}

/// `gate name(params) qubits {` … `}`, one statement per line.
fn write_definition(
    out: &mut String,
    definition: &GateDefinition,
    names: &Names,
    spent: &mut Spent,
) -> Result<(), CircuitError> {
    let (formal_params, formal_qubits) = names.gates.formals(definition);
    let unexpressible = |reason: String| CircuitError::GateNotExpressible {
        name: definition.name().to_string(),
        reason: reason.into(),
    };
    let _ = write!(out, "gate {}", names.gates.gate(definition));
    if !formal_params.is_empty() {
        let _ = write!(out, "({})", formal_params.join(", "));
    }
    let _ = writeln!(out, " {} {{", formal_qubits.join(", "));
    for op in definition.body() {
        let (callee, exprs, operands) = match op {
            BodyOp::Builtin {
                gate,
                params,
                qubits,
            } => {
                let name = match spelling(&body_instruction(gate)) {
                    Some(Spelling::Std(name)) => name,
                    Some(Spelling::Defined(helper)) => names.gates.gate(helper_definition(helper)?),
                    None => return Err(unexpressible("an instruction without a gate spelling".into())),
                };
                (name, params, qubits)
            }
            BodyOp::Call {
                definition: callee,
                params,
                qubits,
            } => (names.gates.gate(callee), params, qubits),
            BodyOp::Barrier(_) => {
                return Err(unexpressible(
                    "cannot be written in OpenQASM 3: its body has a barrier, and an OpenQASM 3 gate body holds gate calls only".to_string(),
                ))
            }
        };
        out.push_str("  ");
        out.push_str(callee);
        if !exprs.is_empty() {
            out.push('(');
            for (i, expr) in exprs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                spent.expression(printed_len(expr.nodes()))?;
                print(expr.nodes(), Dialect::Qasm3, formal_params, out)
                    .map_err(|e| unexpressible(format!("cannot be written in OpenQASM 3: {e}")))?;
            }
            out.push(')');
        }
        let operands = operands
            .iter()
            .map(|&q| formal_qubits.get(q).map(String::as_str))
            .collect::<Option<Vec<&str>>>()
            .ok_or_else(|| unexpressible("a statement names a qubit it does not declare".into()))?;
        let _ = writeln!(out, " {};", operands.join(", "));
    }
    out.push_str("}\n");
    Ok(())
}

/// One instruction's statement.
fn write_statement(
    out: &mut String,
    gate: &GateInstruction,
    names: &Names,
    exprs: &ExprArena,
    num_qubits: usize,
    num_clbits: usize,
    spent: &mut Spent,
) -> Result<(), CircuitError> {
    let (q, c) = (&names.qreg, &names.creg);
    let operands = |qubits: &[usize]| -> String {
        qubits
            .iter()
            .map(|i| format!("{q}[{i}]"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let (name, params, qubits): (&str, Vec<&GateParam>, Vec<usize>) = match gate {
        GateInstruction::Barrier(qubits) => {
            if qubits.is_empty() {
                if num_qubits > 0 {
                    let _ = writeln!(out, "barrier {q};");
                } else {
                    out.push_str("barrier;\n");
                }
            } else {
                let _ = writeln!(out, "barrier {};", operands(qubits));
            }
            return Ok(());
        }
        GateInstruction::Measure { qubit, cbit } => {
            let _ = writeln!(out, "{c}[{cbit}] = measure {q}[{qubit}];");
            return Ok(());
        }
        GateInstruction::MeasureAll => {
            if num_clbits == num_qubits {
                let _ = writeln!(out, "{c} = measure {q};");
            } else {
                for i in 0..num_qubits {
                    let _ = writeln!(out, "{c}[{i}] = measure {q}[{i}];");
                }
            }
            return Ok(());
        }
        GateInstruction::Custom(call) => (
            names.gates.gate(call.definition()),
            call.params().iter().collect(),
            call.qubits().to_vec(),
        ),
        other => {
            let name = match spelling(other) {
                Some(Spelling::Std(name)) => name,
                Some(Spelling::Defined(helper)) => names.gates.gate(helper_definition(helper)?),
                None => return Ok(()),
            };
            (
                name,
                other.params().collect(),
                other.acts_on().qubits().to_vec(),
            )
        }
    };
    out.push_str(name);
    if !params.is_empty() {
        out.push('(');
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_angle(out, param, names, exprs, spent)?;
        }
        out.push(')');
    }
    let _ = writeln!(out, " {};", operands(&qubits));
    Ok(())
}

fn write_angle(
    out: &mut String,
    param: &GateParam,
    names: &Names,
    exprs: &ExprArena,
    spent: &mut Spent,
) -> Result<(), CircuitError> {
    match *param {
        GateParam::Fixed(v) if !v.is_finite() => Err(CircuitError::NonFiniteParam),
        GateParam::Fixed(v) => {
            // A negative number reads back as a negation: two nodes.
            spent.expression(1 + usize::from(v.is_sign_negative()))?;
            out.push_str(&fmt_number(v));
            Ok(())
        }
        GateParam::Param(i) => match names.params.get(i) {
            Some(name) => {
                spent.expression(1)?;
                out.push_str(name);
                Ok(())
            }
            None => Err(CircuitError::ParamIndexOutOfBounds {
                index: i,
                num_params: names.params.len(),
            }),
        },
        GateParam::Expr(id) => {
            let nodes = exprs.nodes(id).ok_or(CircuitError::UnknownExpression)?;
            spent.expression(printed_len(nodes))?;
            print(nodes, Dialect::Qasm3, &names.params, out).map_err(|e| match e {
                Unprintable::Malformed => CircuitError::ParamIndexOutOfBounds {
                    index: names.params.len(),
                    num_params: names.params.len(),
                },
                other => CircuitError::InvalidExpression {
                    reason: other.to_string(),
                },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_helper_is_defined() {
        for helper in [
            Helper::Rzz,
            Helper::Rxx,
            Helper::Sxdg,
            Helper::Csx,
            Helper::Cu1,
            Helper::Cu3,
            Helper::U0,
            Helper::Rccx,
            Helper::Rc3x,
            Helper::C3x,
            Helper::C3sqrtx,
            Helper::C4x,
        ] {
            let definition = helper_definition(helper).unwrap();
            assert_eq!(
                definition.name(),
                format!("{helper:?}").to_lowercase(),
                "{helper:?}"
            );
        }
    }
}
