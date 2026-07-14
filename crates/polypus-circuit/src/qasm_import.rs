//! OpenQASM 2.0 import.
//!
//! Parses the subset of OpenQASM 2.0 emitted by this crate's exporter
//! ([`crate::qasm`]) plus the common `qelib1.inc` vocabulary produced by other
//! toolchains (notably Qiskit's `qasm2.dumps`):
//!
//! - Gates: `h x y z s t sdg tdg id rx ry rz p u1 u2 u3 u U cx CX cz swap rzz
//!   rxx cp` (`p`/`u1`/`u2` are canonicalised to `u3`, `swap` to its standard
//!   3×`cx` decomposition, `id` is dropped — it is the identity).
//! - `barrier`, `measure` (including register broadcast `measure q -> c;`).
//! - Multiple `qreg`/`creg` declarations, flattened into one index space in
//!   declaration order.
//! - Register broadcasting on gate arguments per the QASM 2.0 spec.
//! - Constant angle expressions: numbers, `pi`, `+ - * / ^`, unary minus, and
//!   the spec's unary functions `sin cos tan exp ln sqrt`.
//!
//! Unsupported statements (`gate` definitions, `opaque`, `if`, `reset`) are
//! rejected with a [`CircuitError::Parse`] carrying the 1-based line number —
//! never silently dropped. A gate acting on an already-measured qubit is
//! likewise rejected (terminal-measurement model, contract C-4).
//!
//! This is an **untrusted input surface**, so a few resource limits guard
//! against denial-of-service inputs, each surfaced as a [`CircuitError::Parse`]:
//! - angle expressions may nest at most [`MAX_EXPR_DEPTH`] levels (no stack
//!   overflow from `((((…))))` or `----…-1`);
//! - the total declared qubits, and separately classical bits, may not exceed
//!   [`MAX_REGISTER_BITS`] (no multi-gigabyte index vector from a hostile
//!   `qreg q[4000000000];`).
//!
//! OpenQASM 2.0 has no free parameters, so imported circuits are always fully
//! concrete (`num_params == 0`).

use crate::circuit::ParameterizedCircuit;
use crate::error::CircuitError;
use crate::gate::{ActsOn, GateInstruction, GateParam};
use std::collections::BTreeSet;

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

/// Tokenize `src`, tracking 1-based line numbers. Comments (`// …`) and
/// whitespace are skipped.
fn tokenize(src: &str) -> Result<Vec<(Tok, usize)>, CircuitError> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;

    while i < chars.len() {
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
                } else {
                    let v: usize = text
                        .parse()
                        .map_err(|_| err(line, format!("invalid integer '{text}'")))?;
                    toks.push((Tok::Int(v), line));
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
    }
    Ok(toks)
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

struct Parser {
    toks: Vec<(Tok, usize)>,
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
}

/// Parse a complete OpenQASM 2.0 program into a (fully concrete)
/// [`ParameterizedCircuit`]. Entry point used by
/// [`ParameterizedCircuit::from_qasm2`].
pub(crate) fn parse_qasm2(src: &str) -> Result<ParameterizedCircuit, CircuitError> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks,
        pos: 0,
        qregs: Vec::new(),
        cregs: Vec::new(),
        num_qubits: 0,
        num_cbits: 0,
        gates: Vec::new(),
        measured: BTreeSet::new(),
    };
    p.header()?;
    while !p.at_end() {
        p.statement()?;
    }
    Ok(p.finish())
}

impl Parser {
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
            "gate" => Err(err(line, "custom gate definitions are not supported")),
            "opaque" => Err(err(line, "'opaque' declarations are not supported")),
            "if" => Err(err(line, "'if' statements are not supported")),
            "reset" => Err(err(line, "'reset' is not supported")),
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

    /// Expand register-broadcast semantics: every register argument must have
    /// the same length; single-bit arguments are repeated.
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

    /// Any gate application: `name[(params)] arg[,arg…];`
    fn gate_stmt(&mut self, name: String, line: usize) -> Result<(), CircuitError> {
        // Optional parameter list.
        let params = if self.peek() == Some(&Tok::LParen) {
            self.expect(Tok::LParen)?;
            let mut values = Vec::new();
            if self.peek() != Some(&Tok::RParen) {
                loop {
                    let line = self.line();
                    let value = self.expr(0)?;
                    if !value.is_finite() {
                        return Err(err(
                            line,
                            "parameter expression evaluated to a non-finite value (NaN or infinity)",
                        ));
                    }
                    values.push(value);
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
            } else {
                self.expect(Tok::RParen)?;
            }
            values
        } else {
            Vec::new()
        };

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

    /// Translate one (broadcast-expanded) gate application into instructions.
    ///
    /// This match is the single place to extend when adding gate support.
    fn apply_gate(
        &mut self,
        name: &str,
        params: &[f64],
        args: &[ArgIndices],
        line: usize,
    ) -> Result<(), CircuitError> {
        let arity = |n: usize| -> Result<(), CircuitError> {
            if args.len() != n {
                Err(err(
                    line,
                    format!(
                        "gate '{name}' expects {n} argument(s), found {}",
                        args.len()
                    ),
                ))
            } else {
                Ok(())
            }
        };
        let n_params = |n: usize| -> Result<(), CircuitError> {
            if params.len() != n {
                Err(err(
                    line,
                    format!(
                        "gate '{name}' expects {n} parameter(s), found {}",
                        params.len()
                    ),
                ))
            } else {
                Ok(())
            }
        };
        let fixed = |i: usize| GateParam::Fixed(params[i]);

        match name {
            // ── 1-qubit, no parameters ──
            "h" | "x" | "y" | "z" | "s" | "t" | "sdg" | "tdg" | "id" => {
                n_params(0)?;
                arity(1)?;
                for t in Self::broadcast(args, line)? {
                    let q = t[0];
                    let gate = match name {
                        "h" => GateInstruction::H(q),
                        "x" => GateInstruction::X(q),
                        "y" => GateInstruction::Y(q),
                        "z" => GateInstruction::Z(q),
                        "s" => GateInstruction::S(q),
                        "t" => GateInstruction::T(q),
                        "sdg" => GateInstruction::Sdg(q),
                        "tdg" => GateInstruction::Tdg(q),
                        // Identity: a no-op for any backend; dropped.
                        "id" => continue,
                        _ => unreachable!(),
                    };
                    self.push_validated(gate, line)?;
                }
                Ok(())
            }
            // ── 1-qubit rotations ──
            "rx" | "ry" | "rz" => {
                n_params(1)?;
                arity(1)?;
                for t in Self::broadcast(args, line)? {
                    let (qubit, theta) = (t[0], fixed(0));
                    let gate = match name {
                        "rx" => GateInstruction::Rx { qubit, theta },
                        "ry" => GateInstruction::Ry { qubit, theta },
                        _ => GateInstruction::Rz { qubit, theta },
                    };
                    self.push_validated(gate, line)?;
                }
                Ok(())
            }
            // ── 1-qubit generic family, canonicalised to u3 ──
            "p" | "u1" => {
                n_params(1)?;
                arity(1)?;
                for t in Self::broadcast(args, line)? {
                    self.push_validated(
                        GateInstruction::U {
                            qubit: t[0],
                            theta: GateParam::Fixed(0.0),
                            phi: GateParam::Fixed(0.0),
                            lam: fixed(0),
                        },
                        line,
                    )?;
                }
                Ok(())
            }
            "u2" => {
                n_params(2)?;
                arity(1)?;
                for t in Self::broadcast(args, line)? {
                    self.push_validated(
                        GateInstruction::U {
                            qubit: t[0],
                            theta: GateParam::Fixed(std::f64::consts::FRAC_PI_2),
                            phi: fixed(0),
                            lam: fixed(1),
                        },
                        line,
                    )?;
                }
                Ok(())
            }
            "u3" | "u" | "U" => {
                n_params(3)?;
                arity(1)?;
                for t in Self::broadcast(args, line)? {
                    self.push_validated(
                        GateInstruction::U {
                            qubit: t[0],
                            theta: fixed(0),
                            phi: fixed(1),
                            lam: fixed(2),
                        },
                        line,
                    )?;
                }
                Ok(())
            }
            // ── 2-qubit gates ──
            "cx" | "CX" | "cz" | "swap" => {
                n_params(0)?;
                arity(2)?;
                for t in Self::broadcast(args, line)? {
                    let (a, b) = (t[0], t[1]);
                    self.check_distinct(a, b, line)?;
                    match name {
                        "cz" => self.push_validated(GateInstruction::Cz(a, b), line)?,
                        // qelib1.inc: swap a,b = cx a,b; cx b,a; cx a,b
                        "swap" => {
                            self.push_validated(GateInstruction::Cx(a, b), line)?;
                            self.push_validated(GateInstruction::Cx(b, a), line)?;
                            self.push_validated(GateInstruction::Cx(a, b), line)?;
                        }
                        _ => self.push_validated(GateInstruction::Cx(a, b), line)?,
                    }
                }
                Ok(())
            }
            "rzz" | "rxx" | "cp" => {
                n_params(1)?;
                arity(2)?;
                for t in Self::broadcast(args, line)? {
                    let (q0, q1, theta) = (t[0], t[1], fixed(0));
                    self.check_distinct(q0, q1, line)?;
                    let gate = match name {
                        "rzz" => GateInstruction::Rzz { q0, q1, theta },
                        "rxx" => GateInstruction::Rxx { q0, q1, theta },
                        _ => GateInstruction::Cp { q0, q1, theta },
                    };
                    self.push_validated(gate, line)?;
                }
                Ok(())
            }
            _ => Err(err(line, format!("unsupported gate '{name}'"))),
        }
    }

    fn check_distinct(&self, q0: usize, q1: usize, line: usize) -> Result<(), CircuitError> {
        if q0 == q1 {
            Err(err(
                line,
                format!("two-qubit gate requires distinct qubits, got ({q0}, {q1})"),
            ))
        } else {
            Ok(())
        }
    }

    /// Append a fully-resolved instruction, enforcing the terminal-measurement
    /// model (contract C-4): a unitary gate acting on an already-measured qubit
    /// is rejected with the offending line, rather than silently accepted. This
    /// is the single push point for every statement handler.
    fn push_validated(&mut self, gate: GateInstruction, line: usize) -> Result<(), CircuitError> {
        let violated = match gate.acts_on() {
            ActsOn::One(q) if self.measured.contains(&q) => Some(q),
            ActsOn::Two(a, _) if self.measured.contains(&a) => Some(a),
            ActsOn::Two(_, b) if self.measured.contains(&b) => Some(b),
            _ => None,
        };
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

    fn expr(&mut self, depth: usize) -> Result<f64, CircuitError> {
        self.check_depth(depth)?;
        let mut v = self.term(depth)?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.pos += 1;
                    v += self.term(depth)?;
                }
                Some(Tok::Minus) => {
                    self.pos += 1;
                    v -= self.term(depth)?;
                }
                _ => return Ok(v),
            }
        }
    }

    fn term(&mut self, depth: usize) -> Result<f64, CircuitError> {
        self.check_depth(depth)?;
        let mut v = self.factor(depth)?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.pos += 1;
                    v *= self.factor(depth)?;
                }
                Some(Tok::Slash) => {
                    let line = self.line();
                    self.pos += 1;
                    let divisor = self.factor(depth)?;
                    if divisor == 0.0 {
                        return Err(err(line, "division by zero in parameter expression"));
                    }
                    v /= divisor;
                }
                _ => return Ok(v),
            }
        }
    }

    fn factor(&mut self, depth: usize) -> Result<f64, CircuitError> {
        self.check_depth(depth)?;
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(-self.factor(depth + 1)?)
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.factor(depth + 1)
            }
            _ => self.power(depth),
        }
    }

    fn power(&mut self, depth: usize) -> Result<f64, CircuitError> {
        self.check_depth(depth)?;
        let base = self.primary(depth)?;
        if self.peek() == Some(&Tok::Caret) {
            self.pos += 1;
            let exp = self.factor(depth + 1)?;
            Ok(base.powf(exp))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self, depth: usize) -> Result<f64, CircuitError> {
        self.check_depth(depth)?;
        match self.next("an expression")? {
            (Tok::Real(v), _) => Ok(v),
            (Tok::Int(v), _) => Ok(v as f64),
            (Tok::LParen, _) => {
                let v = self.expr(depth + 1)?;
                self.expect(Tok::RParen)?;
                Ok(v)
            }
            (Tok::Ident(name), line) => {
                if name == "pi" {
                    return Ok(std::f64::consts::PI);
                }
                let f: fn(f64) -> f64 = match name.as_str() {
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "exp" => f64::exp,
                    "ln" => f64::ln,
                    "sqrt" => f64::sqrt,
                    _ => {
                        return Err(err(
                            line,
                            format!("unknown identifier '{name}' in expression"),
                        ))
                    }
                };
                self.expect(Tok::LParen)?;
                let v = self.expr(depth + 1)?;
                self.expect(Tok::RParen)?;
                Ok(f(v))
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
        }
    }
}
