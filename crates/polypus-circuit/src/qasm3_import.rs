//! OpenQASM 3 import: a straight-line profile with Qiskit phase conventions.
//!
//! The importer reads the part of OpenQASM 3 that carries a parameterised,
//! terminal-measurement circuit, and rejects everything else with a
//! [`CircuitError::Parse`] that names the construct and its 1-based line.
//!
//! Accepted:
//!
//! - the version statement `OPENQASM 3;` or `OPENQASM 3.0;` (optional, first);
//! - `include "stdgates.inc";`, provided internally (no file is ever read);
//! - `qubit[n] name;`, `qubit name;`, `bit[n] name;`, `bit name;`, flattened in
//!   declaration order;
//! - `input float[64] name;` and `input float name;` (both binary64): the
//!   circuit's free parameters, in declaration order, under their names
//!   (never a `stdgates.inc` gate's name, included or not: the exports keep
//!   input names and always include `stdgates.inc`);
//! - calls of the `stdgates.inc` gates, of the builtin `U`, and of gates
//!   declared earlier with `gate` blocks (whose bodies hold gate calls only);
//! - `barrier`, and terminal measurement as `c[i] = measure q[j];`,
//!   `c = measure q;` or `measure q -> c;`;
//! - angle expressions over the inputs (or, in a gate body, the gate's
//!   parameters) and the constants `pi`/`π`, `tau`/`τ`, `euler`/`ℇ`: numbers,
//!   `+ - * /`, unary minus, `**`, and `sin cos tan arcsin arccos arctan exp
//!   log sqrt`;
//! - `//` and `/* */` comments.
//!
//! **Qiskit phase conventions.** `U`, `u2` and `u3` are read with Qiskit's
//! matrices, as Polypus's `u`, `u2` and `u3`; the specification's `U` differs
//! from Qiskit's by the global phase `e^{iθ/2}`, and its `u2`/`u3` by
//! `e^{−i(φ+λ)/2}`. Every other `stdgates.inc` gate follows the specification
//! exactly (`CX` as `standard_library.rst` defines it, an alias of `cx`).
//!
//! Spellings: `U` becomes `u`, `CX` becomes `cx`, `phase` becomes `p` and
//! `cphase` becomes `cp`; every other gate keeps its name. A declared gate is
//! always a declared gate ([`GateInstruction::Custom`]), even when it is
//! named like a Polypus instruction (`rzz`).
//!
//! This is an **untrusted input surface**. The budgets below, and those of
//! the OpenQASM 2.0 importer it shares (register size, expression depth,
//! declared-gate nesting and expansion), bound the work any input can cause;
//! each is a [`CircuitError::Parse`] when exceeded.

use crate::circuit::ParameterizedCircuit;
use crate::custom_gate::{
    BodyOp, CustomGate, DefinitionError, GateDefinition, MAX_GATE_EXPANSION, MAX_GATE_NESTING,
};
use crate::error::CircuitError;
use crate::expr::{
    evaluate, Classified, Constant, Dialect, EvalError, ExprArena, Formal, FormalExpr, Function,
    Input, Node, ParamExpr, MAX_EXPR_DEPTH, MAX_EXPR_NODES,
};
use crate::gate::{GateInstruction, GateParam};
use crate::qasm_import::{
    broadcast, builtin_gate, check_distinct, check_signature, check_terminal, err, normalize,
    normalize_line_endings, ArgIndices, BuiltinGate, MAX_REGISTER_BITS, MAX_VALIDATED_EXPANSION,
};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

/// Largest source accepted, in bytes. The OpenQASM 3 exporter writes at most
/// this much, so everything it writes can be imported again.
pub(crate) const MAX_SOURCE_BYTES: usize = 32 << 20;

/// Longest identifier, number or string literal, in bytes.
pub(crate) const MAX_TOKEN_BYTES: usize = 4096;

/// Most `input` declarations.
pub(crate) const MAX_INPUTS: usize = 100_000;

/// Most `gate` declarations.
pub(crate) const MAX_DECLARATIONS: usize = 10_000;

/// Most expression nodes (numbers, names, operators, calls) in a program.
pub(crate) const MAX_PROGRAM_NODES: usize = 4_000_000;

/// Most instructions a program may produce, after register broadcasting.
pub(crate) const MAX_INSTRUCTIONS: usize = 4_000_000;

// ─────────────────────────────── Vocabulary ──────────────────────────────

/// The gates `stdgates.inc` defines (the same list in OpenQASM 3.0 and 3.1).
pub(crate) const STDGATES: [&str; 32] = [
    "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cy", "cz",
    "cp", "crx", "cry", "crz", "ch", "swap", "ccx", "cswap", "cu", "CX", "phase", "cphase", "id",
    "u1", "u2", "u3",
];

/// The instruction a `stdgates.inc` gate is, or `None` for another name.
/// `CX`, `phase` and `cphase` are the only spellings that change (to `cx`,
/// `p`, `cp`); every other gate is the instruction of its own name.
pub(crate) fn stdgate(name: &str) -> Option<&'static BuiltinGate> {
    match name {
        "phase" => builtin_gate("p"),
        "cphase" => builtin_gate("cp"),
        _ if STDGATES.contains(&name) => builtin_gate(name),
        _ => None,
    }
}

/// OpenQASM 3's keywords: never identifiers.
const KEYWORDS: &[&str] = &[
    "OPENQASM",
    "include",
    "defcalgrammar",
    "def",
    "cal",
    "defcal",
    "gate",
    "extern",
    "box",
    "let",
    "break",
    "continue",
    "if",
    "else",
    "end",
    "return",
    "for",
    "while",
    "in",
    "switch",
    "case",
    "default",
    "pragma",
    "input",
    "output",
    "const",
    "readonly",
    "mutable",
    "qreg",
    "qubit",
    "creg",
    "bool",
    "bit",
    "int",
    "uint",
    "float",
    "angle",
    "complex",
    "array",
    "void",
    "duration",
    "stretch",
    "gphase",
    "inv",
    "pow",
    "ctrl",
    "negctrl",
    "durationof",
    "delay",
    "reset",
    "measure",
    "barrier",
    "true",
    "false",
    "im",
];

/// The built-in functions of OpenQASM 3 the profile does not accept.
const OTHER_FUNCTIONS: &[&str] = &[
    "ceiling", "floor", "mod", "popcount", "pow", "rotl", "rotr", "sizeof", "real", "imag",
];

/// A built-in constant.
fn constant(name: &str) -> Option<Constant> {
    Some(match name {
        "pi" | "π" => Constant::Pi,
        "tau" | "τ" => Constant::Tau,
        "euler" | "ℇ" => Constant::Euler,
        _ => return None,
    })
}

/// A function the profile accepts.
fn function(name: &str) -> Option<Function> {
    Some(match name {
        "sin" => Function::Sin,
        "cos" => Function::Cos,
        "tan" => Function::Tan,
        "arcsin" => Function::Arcsin,
        "arccos" => Function::Arccos,
        "arctan" => Function::Arctan,
        "exp" => Function::Exp,
        "log" => Function::Ln,
        "sqrt" => Function::Sqrt,
        _ => return None,
    })
}

/// Whether `name` is a keyword, a built-in constant or function, or a
/// builtin gate (`U`, `gphase`): a name no declaration may take.
pub(crate) fn is_reserved(name: &str) -> bool {
    KEYWORDS.contains(&name)
        || constant(name).is_some()
        || function(name).is_some()
        || OTHER_FUNCTIONS.contains(&name)
        || name == "U"
        || name == "gphase"
}

/// The first character of an identifier: `_`, an ASCII letter, or a non-ASCII
/// letter. The specification allows the Unicode letter categories
/// `Lu Ll Lt Lm Lo Nl`; the Unicode `Alphabetic` property used here contains
/// them all, plus a few marks and symbols (such as combining vowel signs),
/// which are accepted too and kept as written.
pub(crate) fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic() || (!c.is_ascii() && c.is_alphabetic())
}

/// A later character of an identifier: a first character, or an ASCII digit.
pub(crate) fn is_ident_continue(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

/// Whether `name` is an OpenQASM 3 identifier that is not reserved.
#[cfg(test)]
fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(is_ident_start) && chars.all(is_ident_continue) && !is_reserved(name)
}

// ──────────────────────────────── Lexer ──────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok<'s> {
    Ident(&'s str),
    Int(&'s str),
    Float(&'s str),
    /// A string literal's contents.
    Str(&'s str),
    /// `$0`.
    Hardware(&'s str),
    Punct(&'static str),
}

impl Tok<'_> {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) | Tok::Int(s) | Tok::Float(s) | Tok::Hardware(s) => format!("'{s}'"),
            Tok::Str(_) => "a string literal".into(),
            Tok::Punct(p) => format!("'{p}'"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Token<'s> {
    tok: Tok<'s>,
    line: usize,
    /// Byte offsets of the token in the source.
    start: usize,
    end: usize,
}

/// Punctuation, longest first, so `**=` wins over `**` and `**` over `*`.
const PUNCT: &[&str] = &[
    "**=", "<<=", ">>=", "**", "->", "++", "+=", "-=", "*=", "/=", "&=", "|=", "~=", "^=", "%=",
    "==", "!=", ">=", "<=", ">>", "<<", "&&", "||", "(", ")", "[", "]", "{", "}", ",", ";", ":",
    "=", "+", "-", "*", "/", "%", "|", "&", "^", "~", "!", "<", ">", "@", ".", "#",
];

struct Lexer<'s> {
    src: &'s str,
    pos: usize,
    line: usize,
}

/// The end of a run of digits (per `is_digit`) with single underscores
/// between them, starting at `from`. An underscore that is not between two
/// digits ends the run and is left for the caller to reject.
fn digit_run(bytes: &[u8], from: usize, is_digit: impl Fn(u8) -> bool) -> usize {
    let mut i = from;
    while i < bytes.len() {
        if is_digit(bytes[i]) {
            i += 1;
        } else if bytes[i] == b'_' && i > from && i + 1 < bytes.len() && is_digit(bytes[i + 1]) {
            i += 2;
        } else {
            break;
        }
    }
    i
}

impl<'s> Lexer<'s> {
    /// Skip whitespace and comments, counting lines.
    fn skip_trivia(&mut self) -> Result<(), CircuitError> {
        loop {
            let rest = &self.src[self.pos..];
            let Some(c) = rest.chars().next() else {
                return Ok(());
            };
            if c == '\n' {
                self.line += 1;
                self.pos += 1;
            } else if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else if rest.starts_with("//") {
                self.pos += rest.find('\n').unwrap_or(rest.len());
            } else if let Some(after) = rest.strip_prefix("/*") {
                let Some(end) = after.find("*/") else {
                    return Err(err(
                        self.line,
                        "unterminated block comment: '/*' without '*/'",
                    ));
                };
                let comment = &rest[..end + 4];
                self.line += comment.matches('\n').count();
                self.pos += comment.len();
            } else {
                return Ok(());
            }
        }
    }

    fn next(&mut self) -> Result<Option<Token<'s>>, CircuitError> {
        self.skip_trivia()?;
        let rest = &self.src[self.pos..];
        let Some(c) = rest.chars().next() else {
            return Ok(None);
        };
        let (start, line) = (self.pos, self.line);
        let tok = if is_ident_start(c) {
            let len = rest
                .char_indices()
                .find(|&(_, ch)| !is_ident_continue(ch))
                .map_or(rest.len(), |(i, _)| i);
            self.pos += len;
            Tok::Ident(&rest[..len])
        } else if c.is_ascii_digit()
            || (c == '.' && rest[1..].starts_with(|d: char| d.is_ascii_digit()))
        {
            self.number(rest, line)?
        } else if c == '"' || c == '\'' {
            let body = &rest[1..];
            let Some(len) = body.find([c, '\n', '\r', '\t']) else {
                return Err(err(line, "unterminated string literal"));
            };
            if !body[len..].starts_with(c) {
                return Err(err(line, "unterminated string literal"));
            }
            self.pos += len + 2;
            Tok::Str(&body[..len])
        } else if c == '$' {
            let len = rest[1..]
                .find(|ch: char| !ch.is_ascii_digit())
                .unwrap_or(rest.len() - 1);
            if len == 0 {
                return Err(err(line, "unexpected character '$'"));
            }
            self.pos += len + 1;
            Tok::Hardware(&rest[..len + 1])
        } else if let Some(&punct) = PUNCT.iter().find(|p| rest.starts_with(**p)) {
            self.pos += punct.len();
            Tok::Punct(punct)
        } else {
            return Err(err(line, format!("unexpected character '{c}'")));
        };
        let end = self.pos;
        if !matches!(tok, Tok::Punct(_)) && end - start > MAX_TOKEN_BYTES {
            return Err(err(
                line,
                format!("token longer than MAX_TOKEN_BYTES ({MAX_TOKEN_BYTES} bytes)"),
            ));
        }
        Ok(Some(Token {
            tok,
            line,
            start,
            end,
        }))
    }

    /// A numeric literal starting at `rest`: an integer (decimal, `0x`, `0o`
    /// or `0b`) or a float (with a `.`, an exponent, or both).
    fn number(&mut self, rest: &'s str, line: usize) -> Result<Tok<'s>, CircuitError> {
        let bytes = rest.as_bytes();
        let invalid = || err(line, "invalid number literal");
        if bytes.len() > 1
            && bytes[0] == b'0'
            && matches!(bytes[1], b'x' | b'X' | b'o' | b'b' | b'B')
        {
            let is_digit: fn(u8) -> bool = match bytes[1] {
                b'x' | b'X' => |b| b.is_ascii_hexdigit(),
                b'o' => |b| (b'0'..=b'7').contains(&b),
                _ => |b| b == b'0' || b == b'1',
            };
            let len = digit_run(bytes, 2, is_digit);
            if len == 2 {
                return Err(invalid());
            }
            self.pos += len;
            self.after_number(line)?;
            return Ok(Tok::Int(&rest[..len]));
        }
        let decimal = |b: u8| b.is_ascii_digit();
        let mut len = digit_run(bytes, 0, decimal);
        let mut real = false;
        if len < bytes.len() && bytes[len] == b'.' {
            real = true;
            len = digit_run(bytes, len + 1, decimal);
        }
        if len < bytes.len() && matches!(bytes[len], b'e' | b'E') {
            let mut exponent = len + 1;
            if exponent < bytes.len() && matches!(bytes[exponent], b'+' | b'-') {
                exponent += 1;
            }
            let end = digit_run(bytes, exponent, decimal);
            if end == exponent {
                return Err(invalid());
            }
            real = true;
            len = end;
        }
        self.pos += len;
        self.after_number(line)?;
        let text = &rest[..len];
        Ok(if real {
            Tok::Float(text)
        } else {
            Tok::Int(text)
        })
    }

    /// Reject what OpenQASM 3 lexes as one token with the number just read:
    /// imaginary literals (`2im`), durations (`100ns`, `2 s`), and letters
    /// glued to the digits.
    fn after_number(&self, line: usize) -> Result<(), CircuitError> {
        let rest = &self.src[self.pos..];
        let trimmed = rest.trim_start_matches([' ', '\t']);
        let word_len = trimmed
            .char_indices()
            .find(|&(_, c)| !is_ident_continue(c))
            .map_or(trimmed.len(), |(i, _)| i);
        match &trimmed[..word_len] {
            "im" => Err(err(
                line,
                "imaginary literals are not supported: angles are real numbers",
            )),
            unit @ ("dt" | "ns" | "us" | "µs" | "ms" | "s") => Err(err(
                line,
                format!(
                    "duration literals ('…{unit}') are not supported: the profile has no timing"
                ),
            )),
            _ if word_len > 0 && trimmed.len() == rest.len() => Err(err(
                line,
                "invalid number literal: letters directly after the digits",
            )),
            _ => Ok(()),
        }
    }
}

/// The value of an integer literal, as the nearest `f64`.
fn int_value(text: &str) -> Option<f64> {
    let digits: String = text.chars().filter(|&c| c != '_').collect();
    let (radix, body) = match digits.get(..2) {
        Some("0x" | "0X") => (16, &digits[2..]),
        Some("0o") => (8, &digits[2..]),
        Some("0b" | "0B") => (2, &digits[2..]),
        _ => (10, digits.as_str()),
    };
    let value = if radix == 10 {
        body.parse::<f64>().ok()?
    } else {
        u128::from_str_radix(body, radix).ok()? as f64
    };
    value.is_finite().then_some(value)
}

/// The value of a float literal.
fn float_value(text: &str) -> Option<f64> {
    let digits: String = text.chars().filter(|&c| c != '_').collect();
    digits.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// The value of an integer literal used as a size or an index.
fn index_value(text: &str) -> Option<usize> {
    let value = int_value(text)?;
    (value >= 0.0 && value.fract() == 0.0 && value < usize::MAX as f64).then_some(value as usize)
}

/// The token stream, with lookahead.
struct Tokens<'s> {
    lexer: Lexer<'s>,
    ahead: VecDeque<Token<'s>>,
    /// Line and end offset of the last token taken.
    last_line: usize,
    last_end: usize,
}

impl<'s> Tokens<'s> {
    fn fill(&mut self, n: usize) -> Result<(), CircuitError> {
        while self.ahead.len() < n {
            match self.lexer.next()? {
                Some(token) => self.ahead.push_back(token),
                None => break,
            }
        }
        Ok(())
    }

    fn peek(&mut self) -> Result<Option<Tok<'s>>, CircuitError> {
        self.fill(1)?;
        Ok(self.ahead.front().map(|t| t.tok))
    }

    /// The line of the next token, or of the last one at the end of input.
    fn line(&self) -> usize {
        self.ahead.front().map_or(self.last_line, |t| t.line)
    }

    fn next(&mut self, what: &str) -> Result<Token<'s>, CircuitError> {
        self.fill(1)?;
        match self.ahead.pop_front() {
            Some(token) => {
                self.last_line = token.line;
                self.last_end = token.end;
                Ok(token)
            }
            None => Err(err(
                self.last_line,
                format!("unexpected end of input, expected {what}"),
            )),
        }
    }

    fn expect(&mut self, punct: &'static str) -> Result<Token<'s>, CircuitError> {
        let token = self.next(&format!("'{punct}'"))?;
        if token.tok == Tok::Punct(punct) {
            Ok(token)
        } else {
            Err(err(
                token.line,
                format!("expected '{punct}', found {}", token.tok.describe()),
            ))
        }
    }

    /// Take `punct` if it is next.
    fn eat(&mut self, punct: &'static str) -> Result<bool, CircuitError> {
        if self.peek()? == Some(Tok::Punct(punct)) {
            self.next("")?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn ident(&mut self, what: &str) -> Result<(&'s str, usize), CircuitError> {
        let token = self.next(what)?;
        match token.tok {
            Tok::Ident(name) => Ok((name, token.line)),
            other => Err(err(
                token.line,
                format!("expected {what}, found {}", other.describe()),
            )),
        }
    }
}

// ───────────────────────────── Expressions ───────────────────────────────

/// What a name means where an expression is read.
enum Lookup<R> {
    Value(R),
    /// A name of something that is not a value: its description.
    NotAValue(&'static str),
    Unknown,
}

/// The type of a subexpression, as far as the integer-division rule needs:
/// integer if it is made of integer literals only, real otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Real,
}

impl Kind {
    fn and(self, other: Kind) -> Kind {
        if self == Kind::Int && other == Kind::Int {
            Kind::Int
        } else {
            Kind::Real
        }
    }
}

/// One expression being read: its nodes in postfix order and the line of
/// each division.
struct ExprParser<'t, 's, R> {
    tokens: &'t mut Tokens<'s>,
    nodes_used: &'t mut usize,
    lookup: &'t dyn Fn(&str) -> Lookup<R>,
    nodes: Vec<Node<R>>,
    div_lines: Vec<(usize, usize)>,
}

// Grammar, OpenQASM 3's precedence for the profile's operators:
//   expr   := term (('+'|'-') term)*
//   term   := factor (('*'|'/') factor)*
//   factor := '-' factor | power
//   power  := primary ('**' factor)?          (right-associative)
//   primary:= number | constant | name | fn '(' expr ')' | '(' expr ')'
// `depth` counts nesting as the OpenQASM 2.0 importer does (and as
// `expr::depth` measures what the exporter writes): one level per
// parenthesis, unary minus, function argument and exponent.
impl<R: Copy> ExprParser<'_, '_, R> {
    fn push(&mut self, node: Node<R>) -> Result<(), CircuitError> {
        *self.nodes_used += 1;
        if *self.nodes_used > MAX_PROGRAM_NODES {
            return Err(err(
                self.tokens.line(),
                format!("the program's expressions exceed MAX_PROGRAM_NODES ({MAX_PROGRAM_NODES} nodes)"),
            ));
        }
        if self.nodes.len() >= MAX_EXPR_NODES {
            return Err(err(
                self.tokens.line(),
                format!("expression has more than {MAX_EXPR_NODES} nodes"),
            ));
        }
        self.nodes.push(node);
        Ok(())
    }

    fn check_depth(&self, depth: usize) -> Result<(), CircuitError> {
        if depth > MAX_EXPR_DEPTH {
            Err(err(
                self.tokens.line(),
                format!("expression nested too deeply (max {MAX_EXPR_DEPTH})"),
            ))
        } else {
            Ok(())
        }
    }

    fn expr(&mut self, depth: usize) -> Result<Kind, CircuitError> {
        self.check_depth(depth)?;
        let mut kind = self.term(depth)?;
        loop {
            let node = match self.tokens.peek()? {
                Some(Tok::Punct("+")) => Node::Add,
                Some(Tok::Punct("-")) => Node::Sub,
                Some(Tok::Punct("^")) => {
                    return Err(err(
                        self.tokens.line(),
                        "the operator '^' is bitwise xor in OpenQASM 3, which the profile does not support; write '**' for a power",
                    ))
                }
                Some(Tok::Punct(
                    op @ ("<<" | ">>" | "&" | "|" | "&&" | "||" | "==" | "!=" | "<" | ">" | "<="
                    | ">=" | "++"),
                )) => {
                    return Err(err(
                        self.tokens.line(),
                        format!("the operator '{op}' is not supported: angle expressions use + - * / ** and unary minus"),
                    ))
                }
                _ => return Ok(kind),
            };
            self.tokens.next("")?;
            kind = kind.and(self.term(depth)?);
            self.push(node)?;
        }
    }

    fn term(&mut self, depth: usize) -> Result<Kind, CircuitError> {
        self.check_depth(depth)?;
        let mut kind = self.factor(depth)?;
        loop {
            let divide = match self.tokens.peek()? {
                Some(Tok::Punct("*")) => false,
                Some(Tok::Punct("/")) => true,
                Some(Tok::Punct("%")) => {
                    return Err(err(
                        self.tokens.line(),
                        "the operator '%' (mod) is not supported: angle expressions use + - * / ** and unary minus",
                    ))
                }
                _ => return Ok(kind),
            };
            let line = self.tokens.next("")?.line;
            let right = self.factor(depth)?;
            if divide {
                if kind == Kind::Int && right == Kind::Int {
                    return Err(err(
                        line,
                        "a division of two integer expressions is integer division in OpenQASM 3; write a floating-point literal (for example 1.0/2 instead of 1/2)",
                    ));
                }
                self.div_lines.push((self.nodes.len(), line));
                self.push(Node::Div)?;
                kind = Kind::Real;
            } else {
                self.push(Node::Mul)?;
                kind = kind.and(right);
            }
        }
    }

    fn factor(&mut self, depth: usize) -> Result<Kind, CircuitError> {
        self.check_depth(depth)?;
        match self.tokens.peek()? {
            Some(Tok::Punct("-")) => {
                self.tokens.next("")?;
                let kind = self.factor(depth + 1)?;
                self.push(Node::Neg)?;
                Ok(kind)
            }
            Some(Tok::Punct("+")) => Err(err(
                self.tokens.line(),
                "unary '+' is not OpenQASM 3: write the operand alone",
            )),
            Some(Tok::Punct(op @ ("~" | "!"))) => Err(err(
                self.tokens.line(),
                format!("the operator '{op}' is not supported: angle expressions use + - * / ** and unary minus"),
            )),
            _ => self.power(depth),
        }
    }

    fn power(&mut self, depth: usize) -> Result<Kind, CircuitError> {
        self.check_depth(depth)?;
        let base = self.primary(depth)?;
        if self.tokens.peek()? == Some(Tok::Punct("**")) {
            self.tokens.next("")?;
            let exponent = self.factor(depth + 1)?;
            self.push(Node::Pow)?;
            Ok(base.and(exponent))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self, depth: usize) -> Result<Kind, CircuitError> {
        self.check_depth(depth)?;
        let token = self.tokens.next("an expression")?;
        let line = token.line;
        let out_of_range = |text: &str| {
            err(
                line,
                format!("number literal '{text}' is out of range for binary64"),
            )
        };
        match token.tok {
            Tok::Int(text) => {
                let value = int_value(text).ok_or_else(|| out_of_range(text))?;
                self.push(Node::Num(value))?;
                Ok(Kind::Int)
            }
            Tok::Float(text) => {
                let value = float_value(text).ok_or_else(|| out_of_range(text))?;
                self.push(Node::Num(value))?;
                Ok(Kind::Real)
            }
            Tok::Punct("(") => {
                let kind = self.expr(depth + 1)?;
                self.tokens.expect(")")?;
                Ok(kind)
            }
            Tok::Ident(name) => self.name(name, line, depth),
            Tok::Str(_) => Err(err(
                line,
                "bitstring and string literals are not supported in angle expressions",
            )),
            Tok::Hardware(q) => Err(err(
                line,
                format!("physical qubits ('{q}') are not supported"),
            )),
            other => Err(err(
                line,
                format!("expected an expression, found {}", other.describe()),
            )),
        }
    }

    fn name(&mut self, name: &str, line: usize, depth: usize) -> Result<Kind, CircuitError> {
        if let Some(c) = constant(name) {
            self.push(Node::Const(c))?;
            return Ok(Kind::Real);
        }
        if let Some(f) = function(name) {
            self.tokens.expect("(")?;
            self.expr(depth + 1)?;
            if self.tokens.peek()? == Some(Tok::Punct(",")) {
                return Err(err(line, format!("'{name}' takes one argument")));
            }
            self.tokens.expect(")")?;
            self.push(Node::Call(f))?;
            return Ok(Kind::Real);
        }
        if OTHER_FUNCTIONS.contains(&name) {
            return Err(err(
                line,
                format!("the function '{name}' is not supported: the profile's functions are sin, cos, tan, arcsin, arccos, arctan, exp, log and sqrt"),
            ));
        }
        match name {
            "int" | "uint" | "float" | "angle" | "bool" | "bit" | "complex" | "duration" => {
                return Err(err(
                    line,
                    format!(
                        "casts ('{name}(…)') are not supported: angle expressions are real-valued"
                    ),
                ))
            }
            "true" | "false" => {
                return Err(err(
                    line,
                    "boolean literals are not supported in angle expressions",
                ))
            }
            "durationof" => {
                return Err(err(
                    line,
                    "'durationof' is not supported: the profile has no timing",
                ))
            }
            _ => {}
        }
        match (self.lookup)(name) {
            Lookup::Value(reference) => {
                if self.tokens.peek()? == Some(Tok::Punct("[")) {
                    return Err(err(
                        line,
                        format!("indexing '{name}' is not supported: inputs are scalar floats"),
                    ));
                }
                self.push(Node::Ref(reference))?;
                Ok(Kind::Real)
            }
            Lookup::NotAValue(what) => Err(err(
                line,
                format!("'{name}' is {what}, not a value an angle expression can use"),
            )),
            Lookup::Unknown => Err(err(
                line,
                format!("unknown identifier '{name}' in expression"),
            )),
        }
    }
}

// ──────────────────────────────── Parser ─────────────────────────────────

/// A global name.
#[derive(Debug, Clone, Copy)]
enum Symbol {
    Input(usize),
    Qubits(usize),
    Bits(usize),
    Gate,
}

impl Symbol {
    fn describe(self) -> &'static str {
        match self {
            Symbol::Input(_) => "an input",
            Symbol::Qubits(_) => "a qubit register",
            Symbol::Bits(_) => "a bit register",
            Symbol::Gate => "a gate",
        }
    }
}

/// A declared register, in the flat index space of its kind.
struct Reg {
    offset: usize,
    size: usize,
    /// Declared without a size (`qubit q;`): a single qubit, never indexed.
    scalar: bool,
}

struct Parser<'s> {
    src: &'s str,
    tokens: Tokens<'s>,
    nodes_used: usize,
    symbols: HashMap<String, Symbol>,
    stdgates: bool,
    inputs: Vec<String>,
    qregs: Vec<Reg>,
    cregs: Vec<Reg>,
    num_qubits: usize,
    num_cbits: usize,
    gate_defs: HashMap<String, Arc<GateDefinition>>,
    gates: Vec<GateInstruction>,
    /// Qubits already measured, for the terminal-measurement check (C-4).
    measured: BTreeSet<usize>,
    exprs: ExprArena,
    /// Body statements the calls of declared gates may still visit, at parse
    /// time (fixed angles) or at every binding (free angles).
    expansion_budget: usize,
}

/// Parse a complete program in the OpenQASM 3 profile. Entry point of
/// [`ParameterizedCircuit::from_qasm3`].
pub(crate) fn parse_qasm3(src: &str) -> Result<ParameterizedCircuit, CircuitError> {
    if src.len() > MAX_SOURCE_BYTES {
        return Err(err(
            1,
            format!(
                "source of {} bytes exceeds MAX_SOURCE_BYTES ({MAX_SOURCE_BYTES} bytes)",
                src.len()
            ),
        ));
    }
    let mut parser = Parser {
        src,
        tokens: Tokens {
            lexer: Lexer {
                src,
                pos: 0,
                line: 1,
            },
            ahead: VecDeque::new(),
            last_line: 1,
            last_end: 0,
        },
        nodes_used: 0,
        symbols: HashMap::new(),
        stdgates: false,
        inputs: Vec::new(),
        qregs: Vec::new(),
        cregs: Vec::new(),
        num_qubits: 0,
        num_cbits: 0,
        gate_defs: HashMap::new(),
        gates: Vec::new(),
        measured: BTreeSet::new(),
        exprs: ExprArena::default(),
        expansion_budget: MAX_VALIDATED_EXPANSION,
    };
    if parser.tokens.peek()? == Some(Tok::Ident("OPENQASM")) {
        parser.version()?;
    }
    while parser.tokens.peek()?.is_some() {
        parser.statement()?;
    }
    Ok(parser.finish())
}

/// Messages for the constructs the profile rejects by keyword.
fn rejected_keyword(word: &str) -> Option<String> {
    let control_flow = "the OpenQASM 3 profile is straight-line code, and Polypus circuits use terminal measurement (contract C-4, docs/adr/0001-terminal-measurements.md)";
    Some(match word {
        "if" | "else" => format!(
            "'if' statements are not supported: classical control makes a dynamic circuit; {control_flow}"
        ),
        "for" | "while" | "in" | "switch" | "case" | "default" | "break" | "continue" | "end"
        | "return" => format!("'{word}' is not supported: {control_flow}"),
        "reset" => "'reset' is not supported: Polypus circuits use terminal measurement and no other non-unitary operation (contract C-4, docs/adr/0001-terminal-measurements.md)".to_string(),
        "let" => "'let' is not supported: the profile has no aliases".to_string(),
        "const" => "'const' is not supported: the profile's only classical values are inputs and bits".to_string(),
        "output" => "'output' is not supported: the profile's only classical values are inputs and bits".to_string(),
        "def" => "'def' is not supported: the profile has no subroutines".to_string(),
        "extern" => "'extern' is not supported: the profile has no subroutines".to_string(),
        "defcal" | "cal" | "defcalgrammar" => format!("'{word}' is not supported: the profile has no pulse-level calibrations"),
        "box" | "delay" | "stretch" | "duration" | "durationof" => format!("'{word}' is not supported: the profile has no timing"),
        // Sound only because it is rejected, here and inside gate bodies: the
        // profile reads `U`, `u2` and `u3` with Qiskit's global phases, and
        // Polypus has no global-phase state, so a program that could observe
        // a global phase could not be represented.
        "gphase" => "'gphase' is not supported: Polypus has no global-phase state".to_string(),
        // Sound only because they are rejected, here and inside gate bodies:
        // `U`, `u2` and `u3` are read with Qiskit's matrices, which differ
        // from the specification's by a global phase. Under `ctrl @` (or
        // `negctrl @`, or `pow @` of a non-integer power) that global phase
        // becomes observable, so admitting modifiers would require tracking
        // global phases.
        "ctrl" | "negctrl" | "inv" | "pow" => format!("gate modifiers ('{word} @') are not supported"),
        "int" | "uint" | "float" | "angle" | "bool" | "complex" | "array" | "void"
        | "readonly" | "mutable" => format!(
            "classical type '{word}' is not supported: the profile's classical values are 'input float[64]' parameters and 'bit' registers"
        ),
        "qreg" | "creg" => format!("'{word}' is OpenQASM 2.0 syntax: declare 'qubit[n] name;' or 'bit[n] name;' (or import with from_qasm2)"),
        "opaque" => "'opaque' declarations are not supported: an opaque gate has no definition".to_string(),
        "pragma" => "pragmas are not supported".to_string(),
        "im" => "imaginary literals are not supported: angles are real numbers".to_string(),
        "true" | "false" => format!("'{word}' is not a statement"),
        "OPENQASM" => "the version statement ('OPENQASM 3;') must be the program's first statement".to_string(),
        _ => return None,
    })
}

impl<'s> Parser<'s> {
    /// `OPENQASM 3;` or `OPENQASM 3.0;`.
    fn version(&mut self) -> Result<(), CircuitError> {
        let line = self.tokens.next("'OPENQASM'")?.line;
        let version = self.tokens.next("a version number")?;
        match version.tok {
            Tok::Int("3") | Tok::Float("3.0") => {}
            Tok::Int("2") | Tok::Float("2.0") => {
                return Err(err(
                    line,
                    "only OpenQASM version 3 is supported here: import OpenQASM 2.0 with from_qasm2",
                ))
            }
            other => {
                return Err(err(
                    line,
                    format!(
                        "only OpenQASM version 3 is supported ('OPENQASM 3;' or 'OPENQASM 3.0;'), found {}",
                        other.describe()
                    ),
                ))
            }
        }
        self.tokens.expect(";")?;
        Ok(())
    }

    fn statement(&mut self) -> Result<(), CircuitError> {
        let token = self.tokens.next("a statement")?;
        let line = token.line;
        let word = match token.tok {
            Tok::Ident(word) => word,
            Tok::Punct("@") => {
                return Err(err(line, "annotations ('@…') are not supported"));
            }
            Tok::Punct("#") => {
                return Err(err(line, "pragmas and directives ('#…') are not supported"));
            }
            Tok::Punct("{") => {
                return Err(err(
                    line,
                    "blocks ('{ … }') are not supported: the profile is straight-line code",
                ));
            }
            Tok::Hardware(q) => {
                return Err(err(
                    line,
                    format!("physical qubits ('{q}') are not supported"),
                ));
            }
            other => {
                return Err(err(
                    line,
                    format!("expected a statement, found {}", other.describe()),
                ))
            }
        };
        match word {
            "include" => self.include(line),
            "qubit" => self.register_decl(line, true),
            "bit" => self.register_decl(line, false),
            "input" => self.input_decl(line),
            "gate" => self.gate_decl(token),
            "barrier" => self.barrier_stmt(line),
            "measure" => self.measure_arrow(line),
            _ => match rejected_keyword(word) {
                Some(message) => Err(err(line, message)),
                None => self.call_or_assignment(word, line),
            },
        }
    }

    /// `include "stdgates.inc";`. The file is provided internally; no other
    /// file can be included, and none is ever read.
    fn include(&mut self, line: usize) -> Result<(), CircuitError> {
        let token = self.tokens.next("a file name")?;
        let Tok::Str(path) = token.tok else {
            return Err(err(
                token.line,
                format!("expected a file name, found {}", token.tok.describe()),
            ));
        };
        self.tokens.expect(";")?;
        if path != "stdgates.inc" {
            return Err(err(
                line,
                format!("only \"stdgates.inc\" can be included, not \"{path}\": Polypus never reads files"),
            ));
        }
        if self.stdgates {
            return Err(err(line, "\"stdgates.inc\" is included twice"));
        }
        if let Some(name) = STDGATES
            .iter()
            .find(|name| self.symbols.contains_key(**name))
        {
            return Err(err(
                line,
                format!("'{name}' is already declared, so \"stdgates.inc\" cannot define it"),
            ));
        }
        self.stdgates = true;
        Ok(())
    }

    /// Check that a global declaration can take `name`.
    fn check_new_name(&self, name: &str, line: usize, what: &str) -> Result<(), CircuitError> {
        if is_reserved(name) {
            return Err(err(
                line,
                format!("'{name}' is reserved in OpenQASM 3 and cannot name {what}"),
            ));
        }
        if self.stdgates && STDGATES.contains(&name) {
            return Err(err(
                line,
                format!("'{name}' is already defined by stdgates.inc and cannot name {what}"),
            ));
        }
        if let Some(existing) = self.symbols.get(name) {
            return Err(err(
                line,
                format!("'{name}' is already declared as {}", existing.describe()),
            ));
        }
        Ok(())
    }

    /// `qubit[n] name;`, `qubit name;`, `bit[n] name;`, `bit name;`.
    fn register_decl(&mut self, line: usize, quantum: bool) -> Result<(), CircuitError> {
        let mut size = None;
        if self.tokens.eat("[")? {
            let token = self.tokens.next("a register size")?;
            size = Some(match token.tok {
                Tok::Int(text) => index_value(text).ok_or_else(|| {
                    err(
                        token.line,
                        format!("register size '{text}' is out of range"),
                    )
                })?,
                other => {
                    return Err(err(
                        token.line,
                        format!(
                            "expected an integer literal register size, found {}",
                            other.describe()
                        ),
                    ))
                }
            });
            self.tokens.expect("]")?;
        }
        let (name, _) = self.tokens.ident("a register name")?;
        let end = self.tokens.next("';'")?;
        match end.tok {
            Tok::Punct(";") => {}
            Tok::Punct("=") if !quantum => {
                let message = if self.tokens.peek()? == Some(Tok::Ident("measure")) {
                    "a declaration initialised by a measurement ('bit c = measure q;') is not supported: declare the bit, then assign it ('c = measure q;')"
                } else {
                    "initialising a bit is not supported: the profile's bits only hold measurements"
                };
                return Err(err(end.line, message));
            }
            other => {
                return Err(err(
                    end.line,
                    format!("expected ';', found {}", other.describe()),
                ))
            }
        }
        let kind = if quantum {
            "a qubit register"
        } else {
            "a bit register"
        };
        self.check_new_name(name, line, kind)?;
        let scalar = size.is_none();
        let size = size.unwrap_or(1);
        if size == 0 {
            return Err(err(line, format!("register '{name}' has size 0")));
        }
        let (total, noun) = if quantum {
            (self.num_qubits, "qubit")
        } else {
            (self.num_cbits, "classical bit")
        };
        if total
            .checked_add(size)
            .is_none_or(|t| t > MAX_REGISTER_BITS)
        {
            return Err(err(
                line,
                format!(
                    "total {noun} count would exceed MAX_REGISTER_BITS ({MAX_REGISTER_BITS}); register '{name}' has size {size}"
                ),
            ));
        }
        let reg = Reg {
            offset: total,
            size,
            scalar,
        };
        if quantum {
            self.symbols
                .insert(name.to_string(), Symbol::Qubits(self.qregs.len()));
            self.qregs.push(reg);
            self.num_qubits += size;
        } else {
            self.symbols
                .insert(name.to_string(), Symbol::Bits(self.cregs.len()));
            self.cregs.push(reg);
            self.num_cbits += size;
        }
        Ok(())
    }

    /// `input float[64] name;` or `input float name;`: the next free
    /// parameter.
    fn input_decl(&mut self, line: usize) -> Result<(), CircuitError> {
        let (ty, ty_line) = self.tokens.ident("a type")?;
        if ty != "float" {
            return Err(err(
                ty_line,
                format!("inputs of type '{ty}' are not supported: the profile's inputs are 'input float[64] name;'"),
            ));
        }
        if self.tokens.eat("[")? {
            let token = self.tokens.next("a float size")?;
            if token.tok != Tok::Int("64") {
                return Err(err(
                    token.line,
                    format!(
                        "only float[64] inputs are supported (Polypus angles are binary64), found float[{}]",
                        match token.tok {
                            Tok::Int(text) | Tok::Float(text) | Tok::Ident(text) => text,
                            _ => "…",
                        }
                    ),
                ));
            }
            self.tokens.expect("]")?;
        }
        let (name, _) = self.tokens.ident("an input name")?;
        let end = self.tokens.next("';'")?;
        if end.tok != Tok::Punct(";") {
            return Err(err(
                end.line,
                format!("expected ';', found {}", end.tok.describe()),
            ));
        }
        self.check_new_name(name, line, "an input")?;
        // Unlike a gate or register name, an input's name is kept by every
        // export, which always includes "stdgates.inc".
        if STDGATES.contains(&name) {
            return Err(err(
                line,
                format!("'{name}' is a stdgates.inc gate name, which an input cannot take even when \"stdgates.inc\" is not included: input names are kept, and the OpenQASM 3 export always includes it"),
            ));
        }
        if self.inputs.len() >= MAX_INPUTS {
            return Err(err(
                line,
                format!("more than MAX_INPUTS ({MAX_INPUTS}) input declarations"),
            ));
        }
        self.symbols
            .insert(name.to_string(), Symbol::Input(self.inputs.len()));
        self.inputs.push(name.to_string());
        Ok(())
    }

    // ── Operands ─────────────────────────────────────────────────────────

    /// `[i]` after a register name: a non-negative integer literal.
    fn index(&mut self, name: &str) -> Result<usize, CircuitError> {
        let token = self.tokens.next("an index")?;
        let index = match token.tok {
            Tok::Int(text) => index_value(text).ok_or_else(|| {
                err(
                    token.line,
                    format!("index '{text}' of '{name}' is out of range"),
                )
            })?,
            Tok::Punct("-") => {
                return Err(err(
                    token.line,
                    format!("negative indices of '{name}' are not supported"),
                ))
            }
            Tok::Punct("{") => {
                return Err(err(
                    token.line,
                    format!("index sets of '{name}' are not supported"),
                ))
            }
            other => {
                return Err(err(
                    token.line,
                    format!(
                        "only an integer literal can index '{name}', found {}",
                        other.describe()
                    ),
                ))
            }
        };
        let close = self.tokens.next("']'")?;
        match close.tok {
            Tok::Punct("]") => {}
            Tok::Punct(":") => {
                return Err(err(
                    close.line,
                    format!("slices of '{name}' are not supported"),
                ))
            }
            Tok::Punct(",") => {
                return Err(err(
                    close.line,
                    format!("index lists of '{name}' are not supported"),
                ))
            }
            other => {
                return Err(err(
                    close.line,
                    format!("expected ']', found {}", other.describe()),
                ))
            }
        }
        if self.tokens.peek()? == Some(Tok::Punct("[")) {
            return Err(err(
                self.tokens.line(),
                format!(
                    "'{name}' is a register of single qubits or bits: arrays are not supported"
                ),
            ));
        }
        Ok(index)
    }

    /// A register operand, `name` or `name[i]`, of qubits or of bits.
    fn operand(&mut self, quantum: bool) -> Result<ArgIndices, CircuitError> {
        let kind = if quantum { "quantum" } else { "classical" };
        let token = self.tokens.next(&format!("a {kind} register"))?;
        match token.tok {
            Tok::Ident(name) => self.operand_from(name, token.line, quantum),
            Tok::Hardware(q) => Err(err(
                token.line,
                format!("physical qubits ('{q}') are not supported"),
            )),
            other => Err(err(
                token.line,
                format!("expected a {kind} register, found {}", other.describe()),
            )),
        }
    }

    /// [`Self::operand`] whose name, on `line`, was already read.
    fn operand_from(
        &mut self,
        name: &str,
        line: usize,
        quantum: bool,
    ) -> Result<ArgIndices, CircuitError> {
        let kind = if quantum { "quantum" } else { "classical" };
        let reg = match (self.symbols.get(name), quantum) {
            (Some(&Symbol::Qubits(i)), true) => &self.qregs[i],
            (Some(&Symbol::Bits(i)), false) => &self.cregs[i],
            (Some(other), _) => {
                return Err(err(
                    line,
                    format!("'{name}' is {}, not a {kind} register", other.describe()),
                ))
            }
            (None, _) => return Err(err(line, format!("undeclared {kind} register '{name}'"))),
        };
        let (offset, size, scalar) = (reg.offset, reg.size, reg.scalar);
        let indices = if self.tokens.eat("[")? {
            if scalar {
                return Err(err(
                    line,
                    format!(
                        "'{name}' is a single {}, not a register: it cannot be indexed",
                        if quantum { "qubit" } else { "bit" }
                    ),
                ));
            }
            let index = self.index(name)?;
            if index >= size {
                return Err(err(
                    line,
                    format!("index {index} out of range for register '{name}' of size {size}"),
                ));
            }
            ArgIndices {
                indices: vec![offset + index],
                is_register: false,
            }
        } else {
            ArgIndices {
                indices: (offset..offset + size).collect(),
                is_register: !scalar,
            }
        };
        if self.tokens.peek()? == Some(Tok::Punct("++")) {
            return Err(err(
                self.tokens.line(),
                "register concatenation ('++') is not supported",
            ));
        }
        Ok(indices)
    }

    // ── Statements ───────────────────────────────────────────────────────

    /// A statement starting with a name: a gate call, or an assignment
    /// (which must be a measurement).
    fn call_or_assignment(&mut self, name: &'s str, line: usize) -> Result<(), CircuitError> {
        match self.tokens.peek()? {
            Some(Tok::Punct("=" | "[")) => self.assignment(name, line),
            Some(Tok::Punct(
                op @ ("+=" | "-=" | "*=" | "/=" | "&=" | "|=" | "~=" | "^=" | "<<=" | ">>=" | "%="
                | "**="),
            )) => Err(err(
                line,
                format!("assignments ('{op}') other than measurement are not supported: the profile has no classical computation"),
            )),
            Some(Tok::Punct("@")) => Err(err(line, "gate modifiers ('… @') are not supported")),
            _ => self.gate_call(name, line),
        }
    }

    /// `c = measure q;` or `c[i] = measure q[j];`.
    fn assignment(&mut self, target: &'s str, line: usize) -> Result<(), CircuitError> {
        let bits = match self.symbols.get(target) {
            Some(Symbol::Bits(_)) => self.operand_from(target, line, false)?,
            Some(other) => {
                return Err(err(
                    line,
                    format!("assignments other than measurement are not supported: '{target}' is {}, not a bit register", other.describe()),
                ))
            }
            None => {
                return Err(err(line, format!("undeclared classical register '{target}'")))
            }
        };
        let equals = self.tokens.next("'='")?;
        if equals.tok != Tok::Punct("=") {
            return Err(err(
                equals.line,
                format!("expected '=', found {}", equals.tok.describe()),
            ));
        }
        let rhs = self.tokens.next("'measure'")?;
        if rhs.tok != Tok::Ident("measure") {
            return Err(err(
                rhs.line,
                "assignments other than measurement are not supported: the profile has no classical computation ('c = measure q;')",
            ));
        }
        let qubits = self.operand(true)?;
        self.tokens.expect(";")?;
        self.measure(qubits, bits, line)
    }

    /// `measure q -> c;` (after `measure`).
    fn measure_arrow(&mut self, line: usize) -> Result<(), CircuitError> {
        let qubits = self.operand(true)?;
        let token = self.tokens.next("'->'")?;
        match token.tok {
            Tok::Punct("->") => {}
            Tok::Punct(";") => {
                return Err(err(
                    line,
                    "a measurement must be assigned to bits ('c = measure q;')",
                ))
            }
            other => {
                return Err(err(
                    token.line,
                    format!("expected '->', found {}", other.describe()),
                ))
            }
        }
        let bits = self.operand(false)?;
        self.tokens.expect(";")?;
        self.measure(qubits, bits, line)
    }

    fn measure(
        &mut self,
        qubits: ArgIndices,
        bits: ArgIndices,
        line: usize,
    ) -> Result<(), CircuitError> {
        if qubits.indices.len() != bits.indices.len() {
            return Err(err(
                line,
                format!(
                    "measure size mismatch: {} qubit(s) -> {} classical bit(s)",
                    qubits.indices.len(),
                    bits.indices.len()
                ),
            ));
        }
        for (&qubit, &cbit) in qubits.indices.iter().zip(&bits.indices) {
            self.push(GateInstruction::Measure { qubit, cbit }, line)?;
        }
        Ok(())
    }

    /// `barrier;` (every qubit) or `barrier <operands>;`.
    fn barrier_stmt(&mut self, line: usize) -> Result<(), CircuitError> {
        let mut indices = Vec::new();
        if !self.tokens.eat(";")? {
            loop {
                indices.extend(self.operand(true)?.indices);
                let token = self.tokens.next("',' or ';'")?;
                match token.tok {
                    Tok::Punct(",") => continue,
                    Tok::Punct(";") => break,
                    other => {
                        return Err(err(
                            token.line,
                            format!("expected ',' or ';', found {}", other.describe()),
                        ))
                    }
                }
            }
        }
        self.push(GateInstruction::Barrier(indices), line)
    }

    /// An angle argument of a gate call: a fixed value, a free parameter, or
    /// an expression stored in the circuit.
    fn argument(&mut self) -> Result<GateParam, CircuitError> {
        let line = self.tokens.line();
        let symbols = &self.symbols;
        let lookup = |name: &str| match symbols.get(name) {
            Some(&Symbol::Input(i)) => Lookup::Value(Input(i)),
            Some(other) => Lookup::NotAValue(other.describe()),
            None => Lookup::Unknown,
        };
        let mut parser = ExprParser {
            tokens: &mut self.tokens,
            nodes_used: &mut self.nodes_used,
            lookup: &lookup,
            nodes: Vec::new(),
            div_lines: Vec::new(),
        };
        parser.expr(0)?;
        let (nodes, div_lines) = (parser.nodes, parser.div_lines);
        let expr = ParamExpr::from_nodes(nodes);
        let classified = ExprArena::classify(&expr).map_err(|e| err(line, e.to_string()))?;
        Ok(match classified {
            Classified::Param(index) => GateParam::Param(index),
            Classified::Constant => {
                let value = evaluate(
                    expr.nodes(),
                    |Input(index)| Err(EvalError::Unbound { index }),
                    &mut Vec::new(),
                )
                .and_then(|v| {
                    if v.is_finite() {
                        Ok(v)
                    } else {
                        Err(EvalError::NonFinite)
                    }
                })
                .map_err(|e| {
                    let line = match e {
                        EvalError::DivisionByZero { at } => div_lines
                            .iter()
                            .find(|&&(position, _)| position == at)
                            .map_or(line, |&(_, line)| line),
                        _ => line,
                    };
                    err(line, e.to_string())
                })?;
                GateParam::Fixed(value)
            }
            Classified::Expr { max_input } => GateParam::Expr(
                self.exprs
                    .insert(&expr, max_input)
                    .map_err(|e| err(line, e.to_string()))?,
            ),
        })
    }

    /// `name[(args)] operands;` at top level.
    fn gate_call(&mut self, name: &'s str, line: usize) -> Result<(), CircuitError> {
        let mut params = Vec::new();
        if self.tokens.eat("(")? && !self.tokens.eat(")")? {
            loop {
                params.push(self.argument()?);
                let token = self.tokens.next("',' or ')'")?;
                match token.tok {
                    Tok::Punct(",") => continue,
                    Tok::Punct(")") => break,
                    other => {
                        return Err(err(
                            token.line,
                            format!("expected ',' or ')', found {}", other.describe()),
                        ))
                    }
                }
            }
        }
        let mut args = Vec::new();
        loop {
            args.push(self.operand(true)?);
            let token = self.tokens.next("',' or ';'")?;
            match token.tok {
                Tok::Punct(",") => continue,
                Tok::Punct(";") => break,
                other => {
                    return Err(err(
                        token.line,
                        format!("expected ',' or ';', found {}", other.describe()),
                    ))
                }
            }
        }

        if let Some(definition) = self.gate_defs.get(name).cloned() {
            return self.apply_declared(definition, params, &args, line);
        }
        let spec = self.builtin(name, line)?;
        check_signature(
            name,
            (spec.params, spec.qubits),
            (params.len(), args.len()),
            line,
        )?;
        for qubits in broadcast(&args, line)? {
            check_distinct(&qubits, line)?;
            self.push((spec.build)(&params, &qubits), line)?;
        }
        Ok(())
    }

    /// The builtin gate `name` names: `U`, or a `stdgates.inc` gate once it
    /// is included.
    fn builtin(&self, name: &str, line: usize) -> Result<&'static BuiltinGate, CircuitError> {
        let spec = if name == "U" {
            builtin_gate("U")
        } else if self.stdgates {
            stdgate(name)
        } else {
            None
        };
        spec.ok_or_else(|| {
            let reason = match self.symbols.get(name) {
                Some(other) => format!("'{name}' is {}, not a gate", other.describe()),
                None if STDGATES.contains(&name) => {
                    "it is a stdgates.inc gate, but \"stdgates.inc\" is not included".to_string()
                }
                None => {
                    "it is neither a stdgates.inc gate nor declared with a `gate` block".to_string()
                }
            };
            err(line, format!("unsupported gate '{name}': {reason}"))
        })
    }

    /// A call of a declared gate: one [`GateInstruction::Custom`] per
    /// broadcast expansion. Its expansion counts against the program's
    /// budget; a call with fixed angles is checked through its body now, one
    /// with free angles whenever they are bound.
    fn apply_declared(
        &mut self,
        definition: Arc<GateDefinition>,
        params: Vec<GateParam>,
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
        let expansions = broadcast(args, line)?;
        let cost = definition.expansion_size().saturating_mul(expansions.len());
        self.expansion_budget = self.expansion_budget.checked_sub(cost).ok_or_else(|| {
            err(
                line,
                format!(
                    "the calls of declared gates would expand more than {MAX_VALIDATED_EXPANSION} instructions (built-in gates and nested calls)"
                ),
            )
        })?;
        let fixed: Option<Vec<f64>> = params
            .iter()
            .map(|p| match p {
                GateParam::Fixed(v) => Some(*v),
                _ => None,
            })
            .collect();
        if let Some(values) = fixed {
            definition
                .validate(&values)
                .map_err(|e| err(line, format!("in gate '{name}': {e}")))?;
        }
        for qubits in expansions {
            check_distinct(&qubits, line)?;
            let call = CustomGate::new(Arc::clone(&definition), params.clone(), qubits);
            self.push(GateInstruction::Custom(call), line)?;
        }
        Ok(())
    }

    /// Append an instruction: within the instruction budget, and terminal
    /// (contract C-4).
    fn push(&mut self, gate: GateInstruction, line: usize) -> Result<(), CircuitError> {
        if self.gates.len() >= MAX_INSTRUCTIONS {
            return Err(err(
                line,
                format!(
                    "the program has more than MAX_INSTRUCTIONS ({MAX_INSTRUCTIONS}) instructions"
                ),
            ));
        }
        check_terminal(&self.measured, &gate, line)?;
        if let GateInstruction::Measure { qubit, .. } = &gate {
            self.measured.insert(*qubit);
        }
        self.gates.push(gate);
        Ok(())
    }

    // ── Gate declarations ────────────────────────────────────────────────

    /// A formal parameter or qubit name of gate `gate`.
    fn formal_name(
        &self,
        name: &str,
        line: usize,
        gate: &str,
        taken: &[String],
        what: &str,
    ) -> Result<(), CircuitError> {
        if is_reserved(name) {
            return Err(err(
                line,
                format!(
                    "'{name}' is reserved in OpenQASM 3 and cannot name a {what} of gate '{gate}'"
                ),
            ));
        }
        if taken.iter().any(|t| t == name) {
            return Err(err(
                line,
                format!("{what} '{name}' of gate '{gate}' is declared twice"),
            ));
        }
        Ok(())
    }

    /// `gate name(params) qargs { body }`.
    fn gate_decl(&mut self, keyword: Token<'s>) -> Result<(), CircuitError> {
        let line = keyword.line;
        let (name, _) = self.tokens.ident("a gate name")?;
        self.check_new_name(name, line, "a gate")?;
        if self.gate_defs.len() >= MAX_DECLARATIONS {
            return Err(err(
                line,
                format!("more than MAX_DECLARATIONS ({MAX_DECLARATIONS}) gate declarations"),
            ));
        }

        let mut params: Vec<String> = Vec::new();
        if self.tokens.eat("(")? && !self.tokens.eat(")")? {
            loop {
                let (param, l) = self.tokens.ident("a parameter name")?;
                self.formal_name(param, l, name, &params, "parameter")?;
                params.push(param.to_string());
                let token = self.tokens.next("',' or ')'")?;
                match token.tok {
                    Tok::Punct(",") => continue,
                    Tok::Punct(")") => break,
                    other => {
                        return Err(err(
                            token.line,
                            format!("expected ',' or ')', found {}", other.describe()),
                        ))
                    }
                }
            }
        }

        let mut qubits: Vec<String> = Vec::new();
        loop {
            let token = self.tokens.next("a qubit argument name")?;
            let qubit = match token.tok {
                Tok::Ident(qubit) => qubit,
                Tok::Punct("{") if qubits.is_empty() => {
                    return Err(err(
                        line,
                        format!("gate '{name}' must act on at least one qubit"),
                    ))
                }
                other => {
                    return Err(err(
                        token.line,
                        format!("expected a qubit argument name, found {}", other.describe()),
                    ))
                }
            };
            let mut taken = params.clone();
            taken.extend(qubits.iter().cloned());
            self.formal_name(qubit, token.line, name, &taken, "qubit argument")?;
            qubits.push(qubit.to_string());
            let token = self.tokens.next("',' or '{'")?;
            match token.tok {
                Tok::Punct(",") => continue,
                Tok::Punct("{") => break,
                other => {
                    return Err(err(
                        token.line,
                        format!("expected ',' or '{{', found {}", other.describe()),
                    ))
                }
            }
        }

        let body = self.gate_body(name, &params, &qubits)?;
        let declaration = normalize_line_endings(&self.src[keyword.start..self.tokens.last_end]);
        let ordinal = self.gate_defs.len();
        let definition = GateDefinition::new(
            name.to_string(),
            params,
            qubits,
            body,
            declaration,
            Dialect::Qasm3,
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
                    "gate '{name}' expands to more than {MAX_GATE_EXPANSION} instructions (built-in gates and nested calls)"
                ),
            ),
        })?;
        self.symbols.insert(name.to_string(), Symbol::Gate);
        self.gate_defs
            .insert(name.to_string(), Arc::new(definition));
        Ok(())
    }

    /// The statements of a gate body, up to and including its closing `}`:
    /// calls of `U`, of `stdgates.inc` gates and of earlier declarations, on
    /// the formal qubits.
    fn gate_body(
        &mut self,
        gate: &str,
        params: &[String],
        qubits: &[String],
    ) -> Result<Vec<BodyOp>, CircuitError> {
        let mut body = Vec::new();
        loop {
            let token = self
                .tokens
                .next(&format!("'}}' closing the body of gate '{gate}'"))?;
            let line = token.line;
            let op = match token.tok {
                Tok::Punct("}") => return Ok(body),
                Tok::Ident(op) => op,
                other => {
                    return Err(err(
                        line,
                        format!("{} is not allowed inside the body of gate '{gate}': a gate body holds gate calls only", other.describe()),
                    ))
                }
            };
            match op {
                // Rejected in bodies too: see `rejected_keyword`.
                "gphase" => {
                    return Err(err(
                        line,
                        format!("'gphase' is not supported (inside the body of gate '{gate}'): Polypus has no global-phase state"),
                    ))
                }
                "ctrl" | "negctrl" | "inv" | "pow" => {
                    return Err(err(
                        line,
                        format!("gate modifiers ('{op} @') are not supported (inside the body of gate '{gate}')"),
                    ))
                }
                _ if KEYWORDS.contains(&op) => {
                    return Err(err(
                        line,
                        format!("'{op}' is not allowed inside the body of gate '{gate}': a gate body holds gate calls only"),
                    ))
                }
                _ => {}
            }
            if op == gate {
                return Err(err(
                    line,
                    format!(
                        "gate '{gate}' calls itself: recursive gate declarations are not allowed"
                    ),
                ));
            }
            // A formal shadows a gate of its name inside the body, as Qiskit
            // reads it: the name is no longer a gate there.
            let formal = if params.iter().any(|p| p == op) {
                Some("parameter")
            } else if qubits.iter().any(|q| q == op) {
                Some("qubit argument")
            } else {
                None
            };
            if let Some(what) = formal {
                return Err(err(
                    line,
                    format!("'{op}' is a {what} of gate '{gate}', not a gate: inside the body, the {what} shadows the gate of that name"),
                ));
            }
            enum Callee {
                Declared(Arc<GateDefinition>),
                Builtin(&'static BuiltinGate),
            }
            let callee = match self.gate_defs.get(op) {
                Some(definition) => Callee::Declared(Arc::clone(definition)),
                None => Callee::Builtin(self.builtin(op, line)?),
            };

            let mut exprs = Vec::new();
            if self.tokens.eat("(")? && !self.tokens.eat(")")? {
                loop {
                    exprs.push(self.body_expr(gate, params, qubits)?);
                    let token = self.tokens.next("',' or ')'")?;
                    match token.tok {
                        Tok::Punct(",") => continue,
                        Tok::Punct(")") => break,
                        other => {
                            return Err(err(
                                token.line,
                                format!("expected ',' or ')', found {}", other.describe()),
                            ))
                        }
                    }
                }
            }
            let args = self.body_args(gate, qubits)?;
            let expected = match &callee {
                Callee::Declared(definition) => (definition.num_params(), definition.num_qubits()),
                Callee::Builtin(spec) => (spec.params, spec.qubits),
            };
            check_signature(op, expected, (exprs.len(), args.len()), line)?;
            check_distinct(&args, line)?;
            body.push(match callee {
                Callee::Declared(definition) => BodyOp::Call {
                    definition,
                    params: exprs,
                    qubits: args,
                },
                Callee::Builtin(gate) => BodyOp::Builtin {
                    gate,
                    params: exprs,
                    qubits: args,
                },
            });
        }
    }

    /// An angle expression inside the body of gate `gate`, over its formal
    /// parameters.
    fn body_expr(
        &mut self,
        gate: &str,
        params: &[String],
        qubits: &[String],
    ) -> Result<FormalExpr, CircuitError> {
        let symbols = &self.symbols;
        let lookup = |name: &str| {
            if let Some(i) = params.iter().position(|p| p == name) {
                Lookup::Value(Formal(i))
            } else if qubits.iter().any(|q| q == name) {
                Lookup::NotAValue("a qubit argument")
            } else {
                match symbols.get(name) {
                    Some(Symbol::Input(_)) => {
                        Lookup::NotAValue("a global input, which a gate body cannot use")
                    }
                    Some(other) => Lookup::NotAValue(other.describe()),
                    None => Lookup::Unknown,
                }
            }
        };
        let mut parser = ExprParser {
            tokens: &mut self.tokens,
            nodes_used: &mut self.nodes_used,
            lookup: &lookup,
            nodes: Vec::new(),
            div_lines: Vec::new(),
        };
        parser.expr(0).map_err(|e| match e {
            CircuitError::Parse { line, message } if message.starts_with("unknown identifier") => {
                CircuitError::Parse {
                    line,
                    message: format!("{message} (in the body of gate '{gate}')"),
                }
            }
            other => other,
        })?;
        Ok(FormalExpr::from_nodes(parser.nodes))
    }

    /// `a, b, …;` inside the body of gate `gate`: formal qubits, by position.
    fn body_args(&mut self, gate: &str, qubits: &[String]) -> Result<Vec<usize>, CircuitError> {
        let mut args = Vec::new();
        loop {
            let token = self.tokens.next("a qubit argument")?;
            let (arg, line) = match token.tok {
                Tok::Ident(arg) => (arg, token.line),
                Tok::Hardware(q) => {
                    return Err(err(
                        token.line,
                        format!("physical qubits ('{q}') are not supported"),
                    ))
                }
                other => {
                    return Err(err(
                        token.line,
                        format!("expected a qubit argument, found {}", other.describe()),
                    ))
                }
            };
            let Some(index) = qubits.iter().position(|q| q == arg) else {
                return Err(err(
                    line,
                    format!("'{arg}' is not a qubit argument of gate '{gate}'"),
                ));
            };
            if self.tokens.peek()? == Some(Tok::Punct("[")) {
                return Err(err(
                    line,
                    format!("argument '{arg}' of gate '{gate}' cannot be indexed: inside a gate body every argument is a single qubit"),
                ));
            }
            args.push(index);
            let token = self.tokens.next("',' or ';'")?;
            match token.tok {
                Tok::Punct(",") => continue,
                Tok::Punct(";") => return Ok(args),
                other => {
                    return Err(err(
                        token.line,
                        format!("expected ',' or ';', found {}", other.describe()),
                    ))
                }
            }
        }
    }

    /// Normalise the instruction stream and build the circuit: the inputs are
    /// its free parameters, in declaration order, under their names.
    fn finish(self) -> ParameterizedCircuit {
        let gates = normalize(&self.gates, self.num_qubits);
        ParameterizedCircuit {
            num_qubits: self.num_qubits,
            num_params: self.inputs.len(),
            gates,
            measured: Default::default(),
            exprs: self.exprs,
            param_names: self.inputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stdgates_name_is_an_instruction() {
        for name in STDGATES {
            assert!(stdgate(name).is_some(), "{name}");
        }
        assert!(stdgate("u").is_none());
        assert!(stdgate("rzz").is_none());
    }

    #[test]
    fn identifiers_follow_the_specification() {
        for name in ["q", "_gate_q_0", "θ", "_θ_0_", "x1", "π_2", "Q"] {
            assert!(is_valid_identifier(name), "{name}");
        }
        for name in [
            "1q", "", "pi", "π", "measure", "U", "gphase", "log", "mod", "a-b", "a b",
        ] {
            assert!(!is_valid_identifier(name), "{name}");
        }
    }

    #[test]
    fn numbers_read_as_the_nearest_binary64() {
        assert_eq!(int_value("1_000"), Some(1000.0));
        assert_eq!(int_value("0x1F"), Some(31.0));
        assert_eq!(int_value("0o17"), Some(15.0));
        assert_eq!(int_value("0b101"), Some(5.0));
        assert_eq!(float_value("1.5e-3"), Some(1.5e-3));
        assert_eq!(float_value(".5"), Some(0.5));
        assert_eq!(float_value("1."), Some(1.0));
        assert_eq!(float_value("1_0.2_5"), Some(10.25));
        assert_eq!(float_value("1e400"), None);
        assert_eq!(index_value("3"), Some(3));
    }
}
