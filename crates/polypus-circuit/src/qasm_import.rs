//! OpenQASM 2.0 import.
//!
//! Parses the subset of OpenQASM 2.0 emitted by this crate's exporter
//! ([`crate::qasm`]) plus the common `qelib1.inc` vocabulary produced by other
//! toolchains (notably Qiskit's `qasm2.dumps`):
//!
//! - Gates: `h x y z s t sdg tdg id sx sxdg rx ry rz p u1 u2 u3 u U cx CX cz
//!   cy ch csx swap ccx cswap rzz rxx cp crx cry crz cu1 cu3 cu`. Each is
//!   represented one-to-one and re-emitted under the same name — nothing is
//!   decomposed here — except that `p`/`u1`/`u2`/`u`/`U` are canonicalised to
//!   `u3` and `CX` to `cx`. `id` is an instruction of its own, never dropped.
//! - `gate` declarations (`gate name(params) qargs { body }`): each call of a
//!   declared gate is one [`GateInstruction::Custom`] instruction, and the
//!   exporter re-emits the declaration verbatim plus the call — the body is
//!   never expanded into the circuit (see [`crate::custom_gate`]).
//! - `barrier`, `measure` (including register broadcast `measure q -> c;`).
//! - Multiple `qreg`/`creg` declarations, flattened into one index space in
//!   declaration order.
//! - Register broadcasting on gate arguments per the QASM 2.0 spec, for gates
//!   of any arity.
//! - Angle expressions: numbers, `pi`, `+ - * / ^`, unary minus, and the
//!   spec's unary functions `sin cos tan exp ln sqrt` — constant at top level,
//!   over the formal parameters inside a gate body.
//!
//! Unsupported statements (`opaque`, `if`, `reset`) and gates that are neither
//! built in nor declared are rejected with a [`CircuitError::Parse`] carrying
//! the 1-based line number and naming the construct — never silently dropped.
//! A gate acting on an already-measured qubit is likewise rejected
//! (terminal-measurement model, contract C-4).
//!
//! This is an **untrusted input surface**, so a few resource limits guard
//! against denial-of-service inputs, each surfaced as a [`CircuitError::Parse`]:
//! - angle expressions may nest at most [`MAX_EXPR_DEPTH`] levels (no stack
//!   overflow from `((((…))))` or `----…-1`); long flat sums and products are
//!   evaluated iteratively;
//! - the total declared qubits, and separately classical bits, may not exceed
//!   [`MAX_REGISTER_BITS`] (no multi-gigabyte index vector from a hostile
//!   `qreg q[4000000000];`);
//! - a gate declaration may nest calls at most [`MAX_GATE_NESTING`] levels and
//!   expand to at most [`MAX_GATE_EXPANSION`] built-in instructions (no
//!   exponential blow-up from a few nested lines), and validating all calls
//!   instantiates at most [`MAX_VALIDATED_EXPANSION`] instructions in total.
//!
//! OpenQASM 2.0 has no free parameters, so imported circuits are always fully
//! concrete (`num_params == 0`).

use crate::circuit::ParameterizedCircuit;
use crate::custom_gate::{
    AddOp, BodyOp, CustomGate, DefinitionError, EvalError, Expr, Func, GateDefinition, MulOp,
    MAX_GATE_EXPANSION, MAX_GATE_NESTING,
};
use crate::error::CircuitError;
use crate::gate::{first_repeated_qubit, GateInstruction, GateParam};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

/// Maximum nesting depth of a constant angle expression. Bounds parser
/// recursion so untrusted input like `((((…))))` or `----…-1` cannot overflow
/// the stack. Legitimate `qelib1.inc` angle expressions (`pi/2`, `-pi/4`,
/// `(1+2)*pi`, …) nest only a handful of levels, so this is generous.
const MAX_EXPR_DEPTH: usize = 64;

/// Upper bound on the *total* number of declared qubits (and, separately, of
/// declared classical bits). Guards against a hostile `qreg q[4000000000];`
/// materialising a multi-gigabyte index vector during argument expansion. One
/// million bits is far beyond any simulable circuit (the statevector backend
/// caps out around 30 qubits) yet cheap to reject.
const MAX_REGISTER_BITS: usize = 1_000_000;

/// Shorthand for building a [`CircuitError::Parse`].
fn err(line: usize, message: impl Into<String>) -> CircuitError {
    CircuitError::Parse {
        line,
        message: message.into(),
    }
}

// ─────────────────────────────── Lexer ───────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Int(usize),
    Real(f64),
    Str(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Arrow,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    /// `{`, `}`, `=`: legal QASM 2.0 grammar (gate bodies, `if` conditions).
    /// Lexed so the parser can reject the *statement* with a useful message
    /// instead of the lexer choking on the character.
    LBrace,
    RBrace,
    Eq,
}

impl Tok {
    /// Human-readable token name for error messages.
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("'{s}'"),
            Tok::Int(v) => format!("'{v}'"),
            Tok::Real(v) => format!("'{v}'"),
            Tok::Str(_) => "string literal".into(),
            Tok::LParen => "'('".into(),
            Tok::RParen => "')'".into(),
            Tok::LBracket => "'['".into(),
            Tok::RBracket => "']'".into(),
            Tok::Comma => "','".into(),
            Tok::Semi => "';'".into(),
            Tok::Arrow => "'->'".into(),
            Tok::Plus => "'+'".into(),
            Tok::Minus => "'-'".into(),
            Tok::Star => "'*'".into(),
            Tok::Slash => "'/'".into(),
            Tok::Caret => "'^'".into(),
            Tok::LBrace => "'{'".into(),
            Tok::RBrace => "'}'".into(),
            Tok::Eq => "'='".into(),
        }
    }
}

/// A token stream: each token with its 1-based line, and (in a parallel
/// vector) the byte offset in the source where it starts, so the parser can
/// recover source text verbatim (the text of a `gate` declaration).
type Tokens = (Vec<(Tok, usize)>, Vec<usize>);

/// Tokenize `src`, tracking 1-based line numbers and token start offsets.
/// Comments (`// …`) and whitespace are skipped.
fn tokenize(src: &str) -> Result<Tokens, CircuitError> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut starts = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;
    // Byte offset of `chars[i]`, advanced incrementally as `i` grows.
    let (mut cursor_char, mut cursor_byte) = (0usize, 0usize);

    while i < chars.len() {
        while cursor_char < i {
            cursor_byte += chars[cursor_char].len_utf8();
            cursor_char += 1;
        }
        let (token_start, tokens_before) = (cursor_byte, toks.len());
        let c = chars[i];
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            _ if c.is_whitespace() => i += 1,
            '/' if chars.get(i + 1) == Some(&'/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '"' => {
                let start = line;
                i += 1;
                let mut s = String::new();
                loop {
                    match chars.get(i) {
                        Some('"') => {
                            i += 1;
                            break;
                        }
                        Some('\n') | None => return Err(err(start, "unterminated string literal")),
                        Some(&ch) => {
                            s.push(ch);
                            i += 1;
                        }
                    }
                }
                toks.push((Tok::Str(s), start));
            }
            '-' if chars.get(i + 1) == Some(&'>') => {
                toks.push((Tok::Arrow, line));
                i += 2;
            }
            '(' => {
                toks.push((Tok::LParen, line));
                i += 1;
            }
            ')' => {
                toks.push((Tok::RParen, line));
                i += 1;
            }
            '[' => {
                toks.push((Tok::LBracket, line));
                i += 1;
            }
            ']' => {
                toks.push((Tok::RBracket, line));
                i += 1;
            }
            ',' => {
                toks.push((Tok::Comma, line));
                i += 1;
            }
            ';' => {
                toks.push((Tok::Semi, line));
                i += 1;
            }
            '+' => {
                toks.push((Tok::Plus, line));
                i += 1;
            }
            '-' => {
                toks.push((Tok::Minus, line));
                i += 1;
            }
            '*' => {
                toks.push((Tok::Star, line));
                i += 1;
            }
            '/' => {
                toks.push((Tok::Slash, line));
                i += 1;
            }
            '^' => {
                toks.push((Tok::Caret, line));
                i += 1;
            }
            '{' => {
                toks.push((Tok::LBrace, line));
                i += 1;
            }
            '}' => {
                toks.push((Tok::RBrace, line));
                i += 1;
            }
            '=' => {
                toks.push((Tok::Eq, line));
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut is_real = false;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i < chars.len() && chars[i] == '.' {
                    is_real = true;
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    is_real = true;
                    i += 1;
                    if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                if is_real {
                    let v: f64 = text
                        .parse()
                        .map_err(|_| err(line, format!("invalid number '{text}'")))?;
                    toks.push((Tok::Real(v), line));
                } else if let Ok(v) = text.parse::<usize>() {
                    toks.push((Tok::Int(v), line));
                } else {
                    // Too large for a register size or index, but still a valid
                    // number in an expression (it is lexed as the nearest `f64`,
                    // as Qiskit does); where an index is required, the parser
                    // reports the real token.
                    let v: f64 = text
                        .parse()
                        .map_err(|_| err(line, format!("invalid integer '{text}'")))?;
                    toks.push((Tok::Real(v), line));
                }
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                toks.push((Tok::Ident(chars[start..i].iter().collect()), line));
            }
            other => return Err(err(line, format!("unexpected character '{other}'"))),
        }
        // Every arm pushes at most one token.
        if toks.len() > tokens_before {
            starts.push(token_start);
        }
    }
    Ok((toks, starts))
}

// ──────────────────────────── Gate vocabulary ────────────────────────────

/// Builds one instruction from its angle parameters and qubit operands, both
/// already checked against the gate's signature.
type BuildGate = fn(&[GateParam], &[usize]) -> GateInstruction;

/// A built-in gate: its OpenQASM 2.0 spelling and signature, and its IR
/// constructor.
pub(crate) struct BuiltinGate {
    /// Spelling in OpenQASM 2.0 source.
    pub(crate) name: &'static str,
    /// Number of angle parameters.
    pub(crate) params: usize,
    /// Number of qubit arguments.
    pub(crate) qubits: usize,
    /// The IR constructor.
    pub(crate) build: BuildGate,
}

// A row is identified by its (unique) spelling; comparing the constructor's
// function pointer would be meaningless.
impl PartialEq for BuiltinGate {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl std::fmt::Debug for BuiltinGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BuiltinGate({})", self.name)
    }
}

const fn builtin(
    name: &'static str,
    params: usize,
    qubits: usize,
    build: BuildGate,
) -> BuiltinGate {
    BuiltinGate {
        name,
        params,
        qubits,
        build,
    }
}

/// The importer's whole gate vocabulary (the `qelib1.inc` gates it accepts),
/// in one table. Supporting a new gate is one row here, plus the other places
/// contract C-2 lists (exporter, simulator, QIR, bindings); the unit tests
/// check every row against the exporter. A `static` (not a `const`), so the
/// lookup index can hand out `&'static` references into this one table.
static BUILTIN_GATES: &[BuiltinGate] = &[
    // ── 1-qubit, no parameters ──
    builtin("h", 0, 1, |_, q| GateInstruction::H(q[0])),
    builtin("x", 0, 1, |_, q| GateInstruction::X(q[0])),
    builtin("y", 0, 1, |_, q| GateInstruction::Y(q[0])),
    builtin("z", 0, 1, |_, q| GateInstruction::Z(q[0])),
    builtin("s", 0, 1, |_, q| GateInstruction::S(q[0])),
    builtin("t", 0, 1, |_, q| GateInstruction::T(q[0])),
    builtin("sdg", 0, 1, |_, q| GateInstruction::Sdg(q[0])),
    builtin("tdg", 0, 1, |_, q| GateInstruction::Tdg(q[0])),
    // Kept, not dropped: it counts towards the gate count and depth the source
    // program (and Qiskit) reports.
    builtin("id", 0, 1, |_, q| GateInstruction::Id(q[0])),
    // ── 1-qubit rotations ──
    builtin("rx", 1, 1, |p, q| GateInstruction::Rx {
        qubit: q[0],
        theta: p[0],
    }),
    builtin("ry", 1, 1, |p, q| GateInstruction::Ry {
        qubit: q[0],
        theta: p[0],
    }),
    builtin("rz", 1, 1, |p, q| GateInstruction::Rz {
        qubit: q[0],
        theta: p[0],
    }),
    // ── 1-qubit generic family, canonicalised to u3 ──
    builtin("p", 1, 1, u3_from_phase),
    builtin("u1", 1, 1, u3_from_phase),
    builtin("u2", 2, 1, |p, q| GateInstruction::U {
        qubit: q[0],
        theta: GateParam::Fixed(std::f64::consts::FRAC_PI_2),
        phi: p[0],
        lam: p[1],
    }),
    builtin("u3", 3, 1, u3),
    builtin("u", 3, 1, u3),
    builtin("U", 3, 1, u3),
    // ── 2-qubit gates ──
    builtin("cx", 0, 2, |_, q| GateInstruction::Cx(q[0], q[1])),
    builtin("CX", 0, 2, |_, q| GateInstruction::Cx(q[0], q[1])),
    builtin("cz", 0, 2, |_, q| GateInstruction::Cz(q[0], q[1])),
    builtin("swap", 0, 2, |_, q| GateInstruction::Swap(q[0], q[1])),
    builtin("rzz", 1, 2, |p, q| GateInstruction::Rzz {
        q0: q[0],
        q1: q[1],
        theta: p[0],
    }),
    builtin("rxx", 1, 2, |p, q| GateInstruction::Rxx {
        q0: q[0],
        q1: q[1],
        theta: p[0],
    }),
    builtin("cp", 1, 2, |p, q| GateInstruction::Cp {
        q0: q[0],
        q1: q[1],
        theta: p[0],
    }),
    // ── The rest of qelib1.inc: every one kept under its own spelling ──
    builtin("sx", 0, 1, |_, q| GateInstruction::Sx(q[0])),
    builtin("sxdg", 0, 1, |_, q| GateInstruction::Sxdg(q[0])),
    builtin("cy", 0, 2, |_, q| GateInstruction::Cy(q[0], q[1])),
    builtin("ch", 0, 2, |_, q| GateInstruction::Ch(q[0], q[1])),
    builtin("csx", 0, 2, |_, q| GateInstruction::Csx(q[0], q[1])),
    builtin("ccx", 0, 3, |_, q| GateInstruction::Ccx(q[0], q[1], q[2])),
    builtin("cswap", 0, 3, |_, q| {
        GateInstruction::Cswap(q[0], q[1], q[2])
    }),
    builtin("crx", 1, 2, |p, q| GateInstruction::Crx {
        control: q[0],
        target: q[1],
        theta: p[0],
    }),
    builtin("cry", 1, 2, |p, q| GateInstruction::Cry {
        control: q[0],
        target: q[1],
        theta: p[0],
    }),
    builtin("crz", 1, 2, |p, q| GateInstruction::Crz {
        control: q[0],
        target: q[1],
        theta: p[0],
    }),
    // The same operator as `cp`, but its own instruction: `cu1` stays `cu1`.
    builtin("cu1", 1, 2, |p, q| GateInstruction::Cu1 {
        q0: q[0],
        q1: q[1],
        theta: p[0],
    }),
    builtin("cu3", 3, 2, |p, q| GateInstruction::Cu3 {
        control: q[0],
        target: q[1],
        theta: p[0],
        phi: p[1],
        lam: p[2],
    }),
    builtin("cu", 4, 2, |p, q| GateInstruction::Cu {
        control: q[0],
        target: q[1],
        theta: p[0],
        phi: p[1],
        lam: p[2],
        gamma: p[3],
    }),
];

/// `p(λ)` / `u1(λ)` as the equal `u3(0, 0, λ)`.
fn u3_from_phase(p: &[GateParam], q: &[usize]) -> GateInstruction {
    GateInstruction::U {
        qubit: q[0],
        theta: GateParam::Fixed(0.0),
        phi: GateParam::Fixed(0.0),
        lam: p[0],
    }
}

/// `u3(θ, φ, λ)` and its synonyms `u` / `U`.
fn u3(p: &[GateParam], q: &[usize]) -> GateInstruction {
    GateInstruction::U {
        qubit: q[0],
        theta: p[0],
        phi: p[1],
        lam: p[2],
    }
}

/// Look up a built-in gate by its OpenQASM 2.0 spelling (O(1): the table is
/// indexed once, on first use).
pub(crate) fn builtin_gate(name: &str) -> Option<&'static BuiltinGate> {
    static INDEX: OnceLock<HashMap<&'static str, &'static BuiltinGate>> = OnceLock::new();
    INDEX
        .get_or_init(|| BUILTIN_GATES.iter().map(|g| (g.name, g)).collect())
        .get(name)
        .copied()
}

/// Check a gate application's parameter and argument counts against the
/// gate's `(parameters, qubits)` signature.
fn check_signature(
    name: &str,
    (want_params, want_qubits): (usize, usize),
    (got_params, got_qubits): (usize, usize),
    line: usize,
) -> Result<(), CircuitError> {
    if got_params != want_params {
        return Err(err(
            line,
            format!("gate '{name}' expects {want_params} parameter(s), found {got_params}"),
        ));
    }
    if got_qubits != want_qubits {
        return Err(err(
            line,
            format!("gate '{name}' expects {want_qubits} argument(s), found {got_qubits}"),
        ));
    }
    Ok(())
}

/// Qiskit's multi-controlled gate families, whose control count varies from
/// call to call; there is no fixed-signature built-in for them.
const MULTI_CONTROLLED: [&str; 17] = [
    "mcx",
    "mcx_gray",
    "mcx_recursive",
    "mcx_vchain",
    "mcphase",
    "mcp",
    "mcu1",
    "mcu2",
    "mcu3",
    "mcu",
    "mcrx",
    "mcry",
    "mcrz",
    "mcr",
    "mcsx",
    "mcy",
    "mcz",
];

/// The error for a gate that is neither built in nor declared, naming the
/// construct and the limitation behind it.
fn unsupported_gate(name: &str, line: usize) -> CircuitError {
    let reason = if MULTI_CONTROLLED.contains(&name) {
        "multi-controlled gates with an arbitrary number of controls are a known limitation; a program that declares it with a `gate` block (as Qiskit's exporter does) is supported"
    } else {
        "it is neither a qelib1.inc gate nor declared with a `gate` block"
    };
    err(line, format!("unsupported gate '{name}': {reason}"))
}

// ─────────────────────────────── Parser ──────────────────────────────────

/// A declared register, mapped into the flat global index space.
struct Reg {
    name: String,
    offset: usize,
    size: usize,
}

/// One resolved gate/measure argument: either a single (global) bit index or
/// a whole register expanded to its indices.
struct ArgIndices {
    indices: Vec<usize>,
    /// `true` when the argument was a bare register name (participates in
    /// broadcasting), `false` for `name[i]`.
    is_register: bool,
}

struct Parser<'src> {
    /// The source, for recovering a `gate` declaration's text verbatim.
    src: &'src str,
    toks: Vec<(Tok, usize)>,
    /// Byte offset in `src` where each token starts (parallel to `toks`).
    starts: Vec<usize>,
    pos: usize,
    qregs: Vec<Reg>,
    cregs: Vec<Reg>,
    num_qubits: usize,
    num_cbits: usize,
    gates: Vec<GateInstruction>,
    /// Qubits already measured, for the terminal-measurement check (C-4). The
    /// parser emits per-qubit `Measure`s (never `MeasureAll`, which `finish`
    /// synthesizes later), so tracking a qubit set is sufficient.
    measured: BTreeSet<usize>,
    /// Gates declared so far with `gate` blocks, by name.
    gate_defs: HashMap<String, Arc<GateDefinition>>,
    /// Formal parameter names in scope while parsing a gate body (empty at
    /// top level, where expressions are constant).
    param_scope: Vec<String>,
    /// Built-in instructions the importer may still instantiate to validate
    /// calls of declared gates (see [`MAX_VALIDATED_EXPANSION`]).
    expansion_budget: usize,
}

/// Upper bound on the built-in instructions the importer instantiates, over
/// the whole program, to validate calls of declared gates (each call's angles
/// are evaluated through its whole body, so a non-finite angle or a division
/// by zero is rejected at parse time like everywhere else, contract C-2).
/// Bounds the parse time of hostile input that calls a large declaration many
/// times; real programs stay orders of magnitude below it.
const MAX_VALIDATED_EXPANSION: usize = 20_000_000;

/// Parse a complete OpenQASM 2.0 program into a (fully concrete)
/// [`ParameterizedCircuit`]. Entry point used by
/// [`ParameterizedCircuit::from_qasm2`].
pub(crate) fn parse_qasm2(src: &str) -> Result<ParameterizedCircuit, CircuitError> {
    let (toks, starts) = tokenize(src)?;
    let mut p = Parser {
        src,
        toks,
        starts,
        pos: 0,
        qregs: Vec::new(),
        cregs: Vec::new(),
        num_qubits: 0,
        num_cbits: 0,
        gates: Vec::new(),
        measured: BTreeSet::new(),
        gate_defs: HashMap::new(),
        param_scope: Vec::new(),
        expansion_budget: MAX_VALIDATED_EXPANSION,
    };
    p.header()?;
    while !p.at_end() {
        p.statement()?;
    }
    Ok(p.finish())
}

impl Parser<'_> {
    // ── Token-stream helpers ─────────────────────────────────────────────

    fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }

    /// Line of the current (or last) token, for error reporting.
    fn line(&self) -> usize {
        self.toks
            .get(self.pos)
            .or_else(|| self.toks.last())
            .map_or(1, |(_, l)| *l)
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }

    fn next(&mut self, what: &str) -> Result<(Tok, usize), CircuitError> {
        let item = self.toks.get(self.pos).cloned().ok_or_else(|| {
            err(
                self.line(),
                format!("unexpected end of input, expected {what}"),
            )
        })?;
        self.pos += 1;
        Ok(item)
    }

    fn expect(&mut self, tok: Tok) -> Result<usize, CircuitError> {
        let (got, line) = self.next(&tok.describe())?;
        if got == tok {
            Ok(line)
        } else {
            Err(err(
                line,
                format!("expected {}, found {}", tok.describe(), got.describe()),
            ))
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<(String, usize), CircuitError> {
        match self.next(what)? {
            (Tok::Ident(s), line) => Ok((s, line)),
            (other, line) => Err(err(
                line,
                format!("expected {what}, found {}", other.describe()),
            )),
        }
    }

    // ── Statements ───────────────────────────────────────────────────────

    /// `OPENQASM 2.0;` — required first statement.
    fn header(&mut self) -> Result<(), CircuitError> {
        let (kw, line) = self.expect_ident("'OPENQASM'")?;
        if kw != "OPENQASM" {
            return Err(err(
                line,
                format!("expected 'OPENQASM 2.0;' header, found '{kw}'"),
            ));
        }
        let version_ok = match self.next("version number")? {
            (Tok::Real(v), _) => v == 2.0,
            (Tok::Int(v), _) => v == 2,
            _ => false,
        };
        if !version_ok {
            return Err(err(line, "only OpenQASM version 2.0 is supported"));
        }
        self.expect(Tok::Semi)?;
        Ok(())
    }

    fn statement(&mut self) -> Result<(), CircuitError> {
        let (name, line) = self.expect_ident("a statement")?;
        match name.as_str() {
            "include" => {
                match self.next("file name")? {
                    (Tok::Str(_), _) => {} // contents of qelib1.inc are built in
                    (other, l) => {
                        return Err(err(
                            l,
                            format!("expected file name, found {}", other.describe()),
                        ))
                    }
                }
                self.expect(Tok::Semi)?;
                Ok(())
            }
            "qreg" => self.register_decl(line, true),
            "creg" => self.register_decl(line, false),
            "barrier" => self.barrier_stmt(),
            "measure" => self.measure_stmt(),
            "gate" => self.gate_decl(line),
            "opaque" => self.opaque_decl(line),
            "if" => Err(err(
                line,
                "'if' statements are not supported: classical control (`if (creg==n) …`) makes a dynamic circuit, and Polypus circuits use terminal measurement (contract C-4, docs/adr/0001-terminal-measurements.md) — a known limitation",
            )),
            "reset" => Err(err(
                line,
                "'reset' is not supported: Polypus circuits use terminal measurement and no other non-unitary operation (contract C-4, docs/adr/0001-terminal-measurements.md) — a known limitation",
            )),
            _ => self.gate_stmt(name, line),
        }
    }

    /// `qreg name[n];` / `creg name[n];`
    fn register_decl(&mut self, line: usize, quantum: bool) -> Result<(), CircuitError> {
        let (name, _) = self.expect_ident("register name")?;
        self.expect(Tok::LBracket)?;
        let size = match self.next("register size")? {
            (Tok::Int(n), _) => n,
            (other, l) => {
                return Err(err(
                    l,
                    format!("expected register size, found {}", other.describe()),
                ))
            }
        };
        self.expect(Tok::RBracket)?;
        self.expect(Tok::Semi)?;

        if size == 0 {
            return Err(err(line, format!("register '{name}' has size 0")));
        }
        // Cap the running total. `checked_add` also rejects a `size` so large
        // it would overflow `usize`, before argument expansion ever tries to
        // materialise the index vector.
        let running_total = if quantum {
            self.num_qubits
        } else {
            self.num_cbits
        };
        let kind = if quantum { "qubit" } else { "classical bit" };
        if running_total
            .checked_add(size)
            .is_none_or(|t| t > MAX_REGISTER_BITS)
        {
            return Err(err(
                line,
                format!(
                    "total {kind} count would exceed MAX_REGISTER_BITS ({MAX_REGISTER_BITS}); register '{name}' has size {size}"
                ),
            ));
        }
        // QASM 2.0 identifiers share one namespace.
        if self.qregs.iter().chain(&self.cregs).any(|r| r.name == name) {
            return Err(err(line, format!("register '{name}' already declared")));
        }
        if quantum {
            self.qregs.push(Reg {
                name,
                offset: self.num_qubits,
                size,
            });
            self.num_qubits += size;
        } else {
            self.cregs.push(Reg {
                name,
                offset: self.num_cbits,
                size,
            });
            self.num_cbits += size;
        }
        Ok(())
    }

    /// `name` or `name[i]`, resolved into global indices against the quantum
    /// (`quantum = true`) or classical register table.
    fn argument(&mut self, quantum: bool) -> Result<ArgIndices, CircuitError> {
        let kind = if quantum { "quantum" } else { "classical" };
        let (name, line) = self.expect_ident(&format!("a {kind} register"))?;
        let regs = if quantum { &self.qregs } else { &self.cregs };
        let (offset, size) = regs
            .iter()
            .find(|r| r.name == name)
            .map(|r| (r.offset, r.size))
            .ok_or_else(|| err(line, format!("undeclared {kind} register '{name}'")))?;

        if self.peek() == Some(&Tok::LBracket) {
            self.expect(Tok::LBracket)?;
            let idx = match self.next("bit index")? {
                (Tok::Int(n), _) => n,
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected bit index, found {}", other.describe()),
                    ))
                }
            };
            self.expect(Tok::RBracket)?;
            if idx >= size {
                return Err(err(
                    line,
                    format!("index {idx} out of range for register '{name}' of size {size}"),
                ));
            }
            Ok(ArgIndices {
                indices: vec![offset + idx],
                is_register: false,
            })
        } else {
            Ok(ArgIndices {
                indices: (offset..offset + size).collect(),
                is_register: true,
            })
        }
    }

    /// Expand register-broadcast semantics (OpenQASM 2.0 §3.1), for a gate of
    /// any arity: every register argument must have the same length `n`, the
    /// gate is applied `n` times (the `k`-th application takes element `k` of
    /// every register argument), and single-bit arguments are repeated in every
    /// application. E.g. `ccx a,b,c[0];` over `qreg a[2]; qreg b[2];` expands to
    /// `ccx a[0],b[0],c[0]; ccx a[1],b[1],c[0];`.
    fn broadcast(args: &[ArgIndices], line: usize) -> Result<Vec<Vec<usize>>, CircuitError> {
        let span = args
            .iter()
            .filter(|a| a.is_register)
            .map(|a| a.indices.len())
            .max()
            .unwrap_or(1);
        for a in args {
            if a.is_register && a.indices.len() != span {
                return Err(err(
                    line,
                    format!(
                        "register size mismatch in broadcast: expected {span}, found {}",
                        a.indices.len()
                    ),
                ));
            }
        }
        Ok((0..span)
            .map(|k| {
                args.iter()
                    .map(|a| {
                        if a.is_register {
                            a.indices[k]
                        } else {
                            a.indices[0]
                        }
                    })
                    .collect()
            })
            .collect())
    }

    /// `barrier <args>;` — expanded to explicit indices, then normalised in
    /// [`finish`](Self::finish).
    fn barrier_stmt(&mut self) -> Result<(), CircuitError> {
        let line = self.line();
        let mut indices = Vec::new();
        loop {
            let arg = self.argument(true)?;
            indices.extend(arg.indices);
            match self.next("',' or ';'")? {
                (Tok::Comma, _) => continue,
                (Tok::Semi, _) => break,
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected ',' or ';', found {}", other.describe()),
                    ))
                }
            }
        }
        // Barriers are always allowed, even on measured qubits (C-4).
        self.push_validated(GateInstruction::Barrier(indices), line)
    }

    /// `measure q -> c;` (register or single-bit form).
    fn measure_stmt(&mut self) -> Result<(), CircuitError> {
        let line = self.line();
        let q = self.argument(true)?;
        self.expect(Tok::Arrow)?;
        let c = self.argument(false)?;
        self.expect(Tok::Semi)?;

        if q.indices.len() != c.indices.len() {
            return Err(err(
                line,
                format!(
                    "measure size mismatch: {} qubit(s) -> {} classical bit(s)",
                    q.indices.len(),
                    c.indices.len()
                ),
            ));
        }
        for (&qubit, &cbit) in q.indices.iter().zip(&c.indices) {
            self.push_validated(GateInstruction::Measure { qubit, cbit }, line)?;
        }
        Ok(())
    }

    /// An optional parenthesised parameter list, `(e1, e2, …)`: each expression
    /// with the line it starts on. Empty when there is no list (or `()`).
    fn param_exprs(&mut self) -> Result<Vec<(Expr, usize)>, CircuitError> {
        let mut exprs = Vec::new();
        if self.peek() != Some(&Tok::LParen) {
            return Ok(exprs);
        }
        self.expect(Tok::LParen)?;
        if self.peek() == Some(&Tok::RParen) {
            self.expect(Tok::RParen)?;
            return Ok(exprs);
        }
        loop {
            let line = self.line();
            exprs.push((self.expr(0)?, line));
            match self.next("',' or ')'")? {
                (Tok::Comma, _) => continue,
                (Tok::RParen, _) => return Ok(exprs),
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected ',' or ')', found {}", other.describe()),
                    ))
                }
            }
        }
    }

    /// Any gate application: `name[(params)] arg[,arg…];`
    fn gate_stmt(&mut self, name: String, line: usize) -> Result<(), CircuitError> {
        // Optional parameter list: constant expressions, evaluated here.
        let mut params = Vec::new();
        for (expr, line) in self.param_exprs()? {
            let value = expr.eval_angle(&[]).map_err(|e| match e {
                EvalError::DivisionByZero { line } => err(line, e.to_string()),
                EvalError::NonFinite => err(line, e.to_string()),
            })?;
            params.push(value);
        }

        // Argument list.
        let mut args = Vec::new();
        loop {
            args.push(self.argument(true)?);
            match self.next("',' or ';'")? {
                (Tok::Comma, _) => continue,
                (Tok::Semi, _) => break,
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected ',' or ';', found {}", other.describe()),
                    ))
                }
            }
        }

        self.apply_gate(&name, &params, &args, line)
    }

    /// Translate one gate application into (broadcast-expanded) instructions:
    /// look the name up among the declared gates, then in the built-in
    /// vocabulary ([`builtin_gate`]), check the parameter and argument counts,
    /// expand register arguments, and push one instruction per expansion.
    fn apply_gate(
        &mut self,
        name: &str,
        params: &[f64],
        args: &[ArgIndices],
        line: usize,
    ) -> Result<(), CircuitError> {
        if let Some(definition) = self.gate_defs.get(name).cloned() {
            return self.apply_declared(definition, params, args, line);
        }
        let Some(spec) = builtin_gate(name) else {
            return Err(unsupported_gate(name, line));
        };
        check_signature(
            name,
            (spec.params, spec.qubits),
            (params.len(), args.len()),
            line,
        )?;
        let params: Vec<GateParam> = params.iter().map(|&v| GateParam::Fixed(v)).collect();
        for qubits in Self::broadcast(args, line)? {
            Self::check_distinct(&qubits, line)?;
            self.push_validated((spec.build)(&params, &qubits), line)?;
        }
        Ok(())
    }

    /// A call of a declared gate: one [`GateInstruction::Custom`] per
    /// broadcast expansion, never the expanded body. The call's angles are
    /// first evaluated through the whole body (once: they do not depend on the
    /// qubits), so a non-finite angle or a division by zero inside the body is
    /// rejected here, at parse time, like any other angle (contract C-2).
    fn apply_declared(
        &mut self,
        definition: Arc<GateDefinition>,
        params: &[f64],
        args: &[ArgIndices],
        line: usize,
    ) -> Result<(), CircuitError> {
        let name = definition.name();
        check_signature(
            name,
            (definition.num_params(), definition.num_qubits()),
            (params.len(), args.len()),
            line,
        )?;
        self.expansion_budget = self
            .expansion_budget
            .checked_sub(definition.expansion_size())
            .ok_or_else(|| {
                err(
                    line,
                    format!(
                        "validating the calls of declared gates would instantiate more than {MAX_VALIDATED_EXPANSION} instructions"
                    ),
                )
            })?;
        definition
            .validate(params)
            .map_err(|e| err(line, format!("in gate '{name}': {e}")))?;

        let params: Vec<GateParam> = params.iter().map(|&v| GateParam::Fixed(v)).collect();
        for qubits in Self::broadcast(args, line)? {
            Self::check_distinct(&qubits, line)?;
            let call = CustomGate::new(Arc::clone(&definition), params.clone(), qubits);
            self.push_validated(GateInstruction::Custom(call), line)?;
        }
        Ok(())
    }

    // ── Gate declarations ────────────────────────────────────────────────

    /// `gate name(params) qargs { body }`: declare a gate. The body becomes a
    /// template over the formal arguments (see [`crate::custom_gate`]) and the
    /// declaration's text is kept verbatim, so the exporter can re-emit it.
    fn gate_decl(&mut self, line: usize) -> Result<(), CircuitError> {
        let start = self.starts[self.pos - 1];
        let (name, _) = self.expect_ident("gate name")?;
        if builtin_gate(&name).is_some() {
            return Err(err(
                line,
                format!(
                    "gate '{name}' is already defined by qelib1.inc, which Polypus always provides; it cannot be declared again"
                ),
            ));
        }
        if self.gate_defs.contains_key(&name) {
            return Err(err(line, format!("gate '{name}' is already declared")));
        }

        // Formal parameters: `(a, b, …)`, optional, possibly empty.
        let mut param_names: Vec<String> = Vec::new();
        if self.peek() == Some(&Tok::LParen) {
            self.expect(Tok::LParen)?;
            if self.peek() == Some(&Tok::RParen) {
                self.expect(Tok::RParen)?;
            } else {
                loop {
                    let (param, l) = self.expect_ident("a parameter name")?;
                    if param_names.contains(&param) {
                        return Err(err(
                            l,
                            format!("parameter '{param}' of gate '{name}' is declared twice"),
                        ));
                    }
                    param_names.push(param);
                    match self.next("',' or ')'")? {
                        (Tok::Comma, _) => continue,
                        (Tok::RParen, _) => break,
                        (other, l) => {
                            return Err(err(
                                l,
                                format!("expected ',' or ')', found {}", other.describe()),
                            ))
                        }
                    }
                }
            }
        }

        // Formal qubit arguments: at least one, then the body.
        let mut qubit_names: Vec<String> = Vec::new();
        loop {
            let (qubit, l) = self.expect_ident("a qubit argument name")?;
            if qubit_names.contains(&qubit) || param_names.contains(&qubit) {
                return Err(err(
                    l,
                    format!("argument '{qubit}' of gate '{name}' is declared twice"),
                ));
            }
            qubit_names.push(qubit);
            match self.next("',' or '{'")? {
                (Tok::Comma, _) => continue,
                (Tok::LBrace, _) => break,
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected ',' or '{{', found {}", other.describe()),
                    ))
                }
            }
        }

        self.param_scope = param_names.clone();
        let body = self.gate_body(&name, &qubit_names);
        self.param_scope.clear();
        let body = body?;

        // Just past the closing `}` (a one-byte token).
        let end = self.starts[self.pos - 1] + 1;
        let declaration = self.src[start..end].replace("\r\n", "\n");
        let ordinal = self.gate_defs.len();
        let definition = GateDefinition::new(
            name.clone(),
            param_names,
            qubit_names,
            body,
            declaration,
            ordinal,
        )
        .map_err(|e| match e {
            DefinitionError::TooDeep => err(
                line,
                format!("gate '{name}' nests gate calls more than {MAX_GATE_NESTING} levels deep"),
            ),
            DefinitionError::TooLarge => err(
                line,
                format!(
                    "gate '{name}' expands to more than {MAX_GATE_EXPANSION} built-in instructions"
                ),
            ),
        })?;
        self.gate_defs.insert(name, Arc::new(definition));
        Ok(())
    }

    /// The statements of a gate body, up to and including its closing `}`.
    fn gate_body(&mut self, gate: &str, qubits: &[String]) -> Result<Vec<BodyOp>, CircuitError> {
        let mut body = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::RBrace) => {
                    self.pos += 1;
                    return Ok(body);
                }
                None => {
                    return Err(err(
                        self.line(),
                        format!("unterminated body of gate '{gate}': expected '}}'"),
                    ))
                }
                _ => body.push(self.body_stmt(gate, qubits)?),
            }
        }
    }

    /// One statement of a gate body: a call of a built-in or an earlier
    /// declared gate, or a `barrier`, over the formal qubits.
    fn body_stmt(&mut self, gate: &str, qubits: &[String]) -> Result<BodyOp, CircuitError> {
        let (op, line) = self.expect_ident("a gate operation")?;
        match op.as_str() {
            "barrier" => return Ok(BodyOp::Barrier(self.body_args(gate, qubits)?)),
            "measure" | "reset" | "if" | "gate" | "opaque" | "qreg" | "creg" | "include" => {
                return Err(err(
                    line,
                    format!("'{op}' is not allowed inside the body of gate '{gate}'"),
                ))
            }
            _ => {}
        }
        if op == gate {
            return Err(err(
                line,
                format!("gate '{gate}' calls itself: recursive gate declarations are not allowed"),
            ));
        }

        /// What a body statement calls.
        enum Callee {
            Declared(Arc<GateDefinition>),
            Builtin(&'static BuiltinGate),
        }
        let callee = if let Some(definition) = self.gate_defs.get(&op) {
            Callee::Declared(Arc::clone(definition))
        } else if let Some(spec) = builtin_gate(&op) {
            Callee::Builtin(spec)
        } else {
            return Err(unsupported_gate(&op, line));
        };
        let params: Vec<Expr> = self.param_exprs()?.into_iter().map(|(e, _)| e).collect();
        let args = self.body_args(gate, qubits)?;
        let expected = match &callee {
            Callee::Declared(definition) => (definition.num_params(), definition.num_qubits()),
            Callee::Builtin(spec) => (spec.params, spec.qubits),
        };
        check_signature(&op, expected, (params.len(), args.len()), line)?;
        Self::check_distinct(&args, line)?;
        Ok(match callee {
            Callee::Declared(definition) => BodyOp::Call {
                definition,
                params,
                qubits: args,
            },
            Callee::Builtin(gate) => BodyOp::Builtin {
                gate,
                params,
                qubits: args,
            },
        })
    }

    /// `a, b, …;` inside a gate body: formal qubit arguments, by position.
    fn body_args(&mut self, gate: &str, qubits: &[String]) -> Result<Vec<usize>, CircuitError> {
        let mut args = Vec::new();
        loop {
            let (arg, line) = self.expect_ident("a qubit argument")?;
            let Some(index) = qubits.iter().position(|q| *q == arg) else {
                return Err(err(
                    line,
                    format!("'{arg}' is not a qubit argument of gate '{gate}'"),
                ));
            };
            if self.peek() == Some(&Tok::LBracket) {
                return Err(err(
                    line,
                    format!(
                        "argument '{arg}' of gate '{gate}' cannot be indexed: inside a gate body every argument is a single qubit"
                    ),
                ));
            }
            args.push(index);
            match self.next("',' or ';'")? {
                (Tok::Comma, _) => continue,
                (Tok::Semi, _) => return Ok(args),
                (other, l) => {
                    return Err(err(
                        l,
                        format!("expected ',' or ';', found {}", other.describe()),
                    ))
                }
            }
        }
    }

    /// `opaque name(params) qargs;` — an opaque gate has no definition: no
    /// backend could run it and the exporter could not re-emit it as a
    /// circuit, so it is rejected by name.
    fn opaque_decl(&self, line: usize) -> Result<(), CircuitError> {
        let name = match self.peek() {
            Some(Tok::Ident(name)) => name.as_str(),
            _ => "",
        };
        Err(err(
            line,
            format!(
                "'opaque' declarations are not supported: opaque gate '{name}' has no definition, so it can be neither simulated nor re-emitted as a circuit"
            ),
        ))
    }

    /// Reject a (broadcast-expanded) gate application that names the same qubit
    /// twice, for any arity: `cx q[1],q[1];`, or `ccx a,b,c;` where two of the
    /// arguments resolve to the same qubit after expansion.
    fn check_distinct(qubits: &[usize], line: usize) -> Result<(), CircuitError> {
        if first_repeated_qubit(qubits).is_none() {
            return Ok(());
        }
        let arity = match qubits.len() {
            2 => "two-qubit".to_string(),
            3 => "three-qubit".to_string(),
            n => format!("{n}-qubit"),
        };
        let list: Vec<String> = qubits.iter().map(usize::to_string).collect();
        Err(err(
            line,
            format!(
                "{arity} gate requires distinct qubits, got ({})",
                list.join(", ")
            ),
        ))
    }

    /// Append a fully-resolved instruction, enforcing the terminal-measurement
    /// model (contract C-4): a unitary gate acting on an already-measured qubit
    /// is rejected with the offending line, rather than silently accepted. This
    /// is the single push point for every statement handler.
    fn push_validated(&mut self, gate: GateInstruction, line: usize) -> Result<(), CircuitError> {
        // The first measured operand in operand order, for any arity.
        let violated = gate
            .acts_on()
            .qubits()
            .iter()
            .copied()
            .find(|q| self.measured.contains(q));
        if let Some(q) = violated {
            return Err(err(
                line,
                format!(
                    "gate acts on qubit {q} after it was measured; Polypus circuits use terminal measurement (contract C-4)"
                ),
            ));
        }
        if let GateInstruction::Measure { qubit, .. } = &gate {
            self.measured.insert(*qubit);
        }
        self.gates.push(gate);
        Ok(())
    }

    // ── Constant expressions ─────────────────────────────────────────────
    //
    // Grammar (QASM 2.0 spec):
    //   expr   := term (('+'|'-') term)*
    //   term   := factor (('*'|'/') factor)*
    //   factor := ('-'|'+') factor | power
    //   power  := primary ('^' factor)?          (right-associative)
    //   primary:= real | int | 'pi' | fn '(' expr ')' | '(' expr ')'
    //
    // `depth` bounds the recursion so untrusted input like `(((…)))` or
    // `----…-1` cannot overflow the stack; it is incremented only when
    // descending into a nested sub-expression (parenthesis, function argument,
    // unary operator, exponent).

    /// Guard against runaway recursion (DoS via deeply nested expressions).
    fn check_depth(&self, depth: usize) -> Result<(), CircuitError> {
        if depth > MAX_EXPR_DEPTH {
            Err(err(
                self.line(),
                format!("expression nested too deeply (max {MAX_EXPR_DEPTH})"),
            ))
        } else {
            Ok(())
        }
    }

    fn expr(&mut self, depth: usize) -> Result<Expr, CircuitError> {
        self.check_depth(depth)?;
        let first = self.term(depth)?;
        let mut rest = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => AddOp::Add,
                Some(Tok::Minus) => AddOp::Sub,
                _ => break,
            };
            self.pos += 1;
            rest.push((op, self.term(depth)?));
        }
        Ok(if rest.is_empty() {
            first
        } else {
            Expr::Sum(Box::new(first), rest)
        })
    }

    fn term(&mut self, depth: usize) -> Result<Expr, CircuitError> {
        self.check_depth(depth)?;
        let first = self.factor(depth)?;
        let mut rest = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => MulOp::Mul,
                Some(Tok::Slash) => MulOp::Div { line: self.line() },
                _ => break,
            };
            self.pos += 1;
            rest.push((op, self.factor(depth)?));
        }
        Ok(if rest.is_empty() {
            first
        } else {
            Expr::Product(Box::new(first), rest)
        })
    }

    fn factor(&mut self, depth: usize) -> Result<Expr, CircuitError> {
        self.check_depth(depth)?;
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(Expr::Neg(Box::new(self.factor(depth + 1)?)))
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.factor(depth + 1)
            }
            _ => self.power(depth),
        }
    }

    fn power(&mut self, depth: usize) -> Result<Expr, CircuitError> {
        self.check_depth(depth)?;
        let base = self.primary(depth)?;
        if self.peek() == Some(&Tok::Caret) {
            self.pos += 1;
            let exponent = self.factor(depth + 1)?;
            Ok(Expr::Pow(Box::new(base), Box::new(exponent)))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self, depth: usize) -> Result<Expr, CircuitError> {
        self.check_depth(depth)?;
        match self.next("an expression")? {
            (Tok::Real(v), _) => Ok(Expr::Num(v)),
            (Tok::Int(v), _) => Ok(Expr::Num(v as f64)),
            (Tok::LParen, _) => {
                let e = self.expr(depth + 1)?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            (Tok::Ident(name), line) => {
                if name == "pi" {
                    return Ok(Expr::Num(std::f64::consts::PI));
                }
                // Inside a gate body: one of the gate's formal parameters.
                if let Some(index) = self.param_scope.iter().position(|p| *p == name) {
                    return Ok(Expr::Param(index));
                }
                let Some(f) = Func::from_name(&name) else {
                    return Err(err(
                        line,
                        format!("unknown identifier '{name}' in expression"),
                    ));
                };
                self.expect(Tok::LParen)?;
                let e = self.expr(depth + 1)?;
                self.expect(Tok::RParen)?;
                Ok(Expr::Func(f, Box::new(e)))
            }
            (other, line) => Err(err(
                line,
                format!("expected an expression, found {}", other.describe()),
            )),
        }
    }

    // ── Final assembly ───────────────────────────────────────────────────

    /// Normalise the instruction stream and build the circuit:
    ///
    /// - A maximal run `measure q[0]->c[0]; … measure q[n-1]->c[n-1];`
    ///   covering every qubit collapses to [`GateInstruction::MeasureAll`]
    ///   (matches both this crate's `measure q -> c;` and Qiskit's expanded
    ///   per-qubit form).
    /// - A barrier listing every qubit in order becomes the whole-register
    ///   barrier.
    ///
    /// Both rewrites are semantically identity; they exist so that
    /// export → import → export is byte-stable.
    fn finish(self) -> ParameterizedCircuit {
        let n = self.num_qubits;
        let all: Vec<usize> = (0..n).collect();
        let mut gates = Vec::with_capacity(self.gates.len());
        let mut i = 0;
        while i < self.gates.len() {
            if n > 0 && i + n <= self.gates.len() {
                let full_measure_run = (0..n)
                    .all(|k| self.gates[i + k] == GateInstruction::Measure { qubit: k, cbit: k });
                if full_measure_run {
                    gates.push(GateInstruction::MeasureAll);
                    i += n;
                    continue;
                }
            }
            match &self.gates[i] {
                GateInstruction::Barrier(v) if *v == all => {
                    gates.push(GateInstruction::Barrier(Vec::new()))
                }
                other => gates.push(other.clone()),
            }
            i += 1;
        }

        // OpenQASM 2.0 has no free parameters: always fully concrete.
        ParameterizedCircuit {
            num_qubits: n,
            num_params: 0,
            gates,
            // Left un-derived on purpose: the parser's own `measured` set covers
            // the per-qubit `Measure`s it validated, but `finish` also synthesises
            // `MeasureAll`, so the builder's cache is rebuilt from `gates` on the
            // first push into the imported circuit.
            measured: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(indices: std::ops::Range<usize>) -> ArgIndices {
        ArgIndices {
            indices: indices.collect(),
            is_register: true,
        }
    }

    fn bit(index: usize) -> ArgIndices {
        ArgIndices {
            indices: vec![index],
            is_register: false,
        }
    }

    /// The spellings the importer canonicalises instead of preserving. Every
    /// other built-in must be re-emitted under exactly the name it was parsed
    /// from, or a benchmark file would reach Aer as a different program.
    fn canonical_spelling(name: &str) -> &str {
        match name {
            "p" | "u1" | "u2" | "u" | "U" => "u3",
            "CX" => "cx",
            other => other,
        }
    }

    #[test]
    fn builtin_vocabulary_has_unique_spellings() {
        let mut names: Vec<&str> = BUILTIN_GATES.iter().map(|g| g.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate spelling in BUILTIN_GATES");
        for gate in BUILTIN_GATES {
            assert!(std::ptr::eq(builtin_gate(gate.name).unwrap(), gate));
        }
        assert!(builtin_gate("ccz").is_none());
    }

    /// Table ↔ exporter agreement, row by row: the instruction a row builds is
    /// exported under the row's own spelling (modulo `canonical_spelling`),
    /// with the row's parameter and operand counts, in operand order.
    #[test]
    fn every_builtin_is_exported_under_its_own_name() {
        for gate in BUILTIN_GATES {
            let params: Vec<GateParam> = (1..=gate.params)
                .map(|i| GateParam::Fixed(0.125 * i as f64))
                .collect();
            // Descending operands, so an exporter that re-sorted them shows.
            let qubits: Vec<usize> = (0..gate.qubits).rev().collect();
            let instruction = (gate.build)(&params, &qubits);
            let qasm = crate::qasm::write_qasm2(gate.qubits, 0, &[instruction], &[]).unwrap();
            let statement = qasm.lines().last().unwrap();

            let operands: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
            let expected_params = match canonical_spelling(gate.name) {
                // The canonical u3 form carries the synthesised angles.
                "u3" => None,
                _ if gate.params == 0 => Some(String::new()),
                _ => Some(format!(
                    "({})",
                    params
                        .iter()
                        .map(|p| match p {
                            GateParam::Fixed(v) => format!("{v:.12}"),
                            GateParam::Param(_) => unreachable!(),
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                )),
            };
            let head = canonical_spelling(gate.name);
            assert!(
                statement.starts_with(head),
                "{}: exported as {statement:?}",
                gate.name
            );
            assert!(
                statement.ends_with(&format!(" {};", operands.join(","))),
                "{}: operands reordered in {statement:?}",
                gate.name
            );
            if let Some(expected) = expected_params {
                assert_eq!(
                    statement,
                    format!("{head}{expected} {};", operands.join(",")),
                    "{}",
                    gate.name
                );
            }
        }
    }

    #[test]
    fn broadcast_three_arguments_mixed_registers_and_bits() {
        // `ccx a,b,c[0];` with a = q[0..2], b = q[2..4], c[0] = q[4]: the same
        // expansion Qiskit's QASM 2 loader produces for this statement.
        let args = [reg(0..2), reg(2..4), bit(4)];
        assert_eq!(
            Parser::broadcast(&args, 1).unwrap(),
            vec![vec![0, 2, 4], vec![1, 3, 4]]
        );
        // A single register in any position drives the expansion.
        let args = [bit(0), reg(1..4), bit(5)];
        assert_eq!(
            Parser::broadcast(&args, 1).unwrap(),
            vec![vec![0, 1, 5], vec![0, 2, 5], vec![0, 3, 5]]
        );
        // All single bits: exactly one application.
        let args = [bit(2), bit(0), bit(1)];
        assert_eq!(Parser::broadcast(&args, 1).unwrap(), vec![vec![2, 0, 1]]);
    }

    #[test]
    fn broadcast_rejects_register_size_mismatch_at_any_arity() {
        let args = [reg(0..2), reg(2..5), bit(5)];
        match Parser::broadcast(&args, 7) {
            Err(CircuitError::Parse { line: 7, message }) => {
                assert!(message.contains("register size mismatch"), "{message}")
            }
            other => panic!("expected a size-mismatch error, got {other:?}"),
        }
    }

    #[test]
    fn check_distinct_rejects_repeats_at_any_arity() {
        assert!(Parser::check_distinct(&[0, 1], 1).is_ok());
        assert!(Parser::check_distinct(&[2, 0, 1], 1).is_ok());
        for (qubits, expected) in [
            (
                &[1, 1][..],
                "two-qubit gate requires distinct qubits, got (1, 1)",
            ),
            (
                &[0, 1, 0][..],
                "three-qubit gate requires distinct qubits, got (0, 1, 0)",
            ),
            (
                &[3, 2, 1, 2][..],
                "4-qubit gate requires distinct qubits, got (3, 2, 1, 2)",
            ),
        ] {
            match Parser::check_distinct(qubits, 9) {
                Err(CircuitError::Parse { line: 9, message }) => assert_eq!(message, expected),
                other => panic!("expected a distinct-qubits error, got {other:?}"),
            }
        }
    }
}
