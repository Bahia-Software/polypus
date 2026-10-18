//! Circuit drawings: text (Unicode box drawing or plain ASCII) and SVG.
//!
//! Both renderers share one layout. Each instruction becomes an op (what
//! is drawn on which wires), and the ops are packed greedily into columns: an
//! op goes in the first column after the last one that uses any wire between
//! its topmost and bottommost qubit, so the vertical line joining a control to
//! its target never crosses another gate. A measurement reaches down to the
//! classical register, which is drawn as one bundled wire below the qubits.
//!
//! Angles are drawn as the circuit holds them: a free parameter by its name
//! ([`ParameterizedCircuit::param_names`]), an expression as its OpenQASM 3
//! text, and a fixed value as a short multiple of π when it is one (`π/2`,
//! `-3π/4`) or to four decimals otherwise. A drawing is for reading, not for
//! reading back: use the OpenQASM exports to serialise a circuit.
//!
//! Nothing here can fail or panic: a drawing is produced for any circuit,
//! including one whose `gates` were assembled by hand with qubit indices
//! beyond `num_qubits` (the drawing grows to show them).

use crate::circuit::{ConcreteCircuit, ParameterizedCircuit};
use crate::expr::{print, Dialect, ExprArena};
use crate::gate::{GateInstruction, GateParam};
use std::f64::consts::PI;
use std::fmt::{self, Write};

/// How [`ParameterizedCircuit::draw_text`] draws a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextOptions {
    /// Draw with plain ASCII (`-`, `|`, `+`, `*`, `pi`) instead of Unicode
    /// box-drawing characters, for terminals and logs that cannot show them.
    pub ascii: bool,
    /// Wrap the drawing into pages at most this many characters wide (a
    /// single column wider than that gets a page of its own); `None` never
    /// wraps. A wrapped wire ends in `»` and resumes after `«`.
    pub fold: Option<usize>,
}

impl Default for TextOptions {
    /// Unicode, wrapped at 80 characters.
    fn default() -> Self {
        TextOptions {
            ascii: false,
            fold: Some(80),
        }
    }
}

impl ParameterizedCircuit {
    /// Draw the circuit as text, one wire per qubit and a bundled classical
    /// wire `c` when it measures.
    ///
    /// ```
    /// use polypus_circuit::{draw::TextOptions, Param, ParameterizedCircuit};
    ///
    /// let qc = ParameterizedCircuit::new(2)
    ///     .h(0)
    ///     .cx(0, 1)
    ///     .rz(1, Param(0))
    ///     .measure_all();
    /// let text = qc.draw_text(TextOptions::default());
    /// assert!(text.contains("theta_0"));
    /// println!("{text}");
    /// ```
    pub fn draw_text(&self, options: TextOptions) -> String {
        let names = self.param_names();
        let source = Source {
            num_qubits: self.num_qubits,
            gates: &self.gates,
            exprs: Some(&self.exprs),
            names: &names,
        };
        draw_text(&source, options)
    }

    /// Draw the circuit as a standalone SVG document (no scripts, no external
    /// resources), ready to save as a `.svg` file or embed in HTML.
    pub fn draw_svg(&self) -> String {
        let names = self.param_names();
        let source = Source {
            num_qubits: self.num_qubits,
            gates: &self.gates,
            exprs: Some(&self.exprs),
            names: &names,
        };
        draw_svg(&source)
    }
}

impl ConcreteCircuit {
    /// Draw the circuit as text (see [`ParameterizedCircuit::draw_text`]).
    pub fn draw_text(&self, options: TextOptions) -> String {
        draw_text(&self.source(), options)
    }

    /// Draw the circuit as SVG (see [`ParameterizedCircuit::draw_svg`]).
    pub fn draw_svg(&self) -> String {
        draw_svg(&self.source())
    }

    fn source(&self) -> Source<'_> {
        Source {
            num_qubits: self.num_qubits,
            gates: &self.gates,
            exprs: None,
            names: &[],
        }
    }
}

/// The text drawing with the default [`TextOptions`].
impl fmt::Display for ParameterizedCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.draw_text(TextOptions::default()))
    }
}

/// The text drawing with the default [`TextOptions`].
impl fmt::Display for ConcreteCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.draw_text(TextOptions::default()))
    }
}

/// What a drawing needs from either circuit type.
struct Source<'a> {
    num_qubits: usize,
    gates: &'a [GateInstruction],
    exprs: Option<&'a ExprArena>,
    names: &'a [String],
}

// ───────────────────────────────── Ops ────────────────────────────────────

/// The colour family of an op in the SVG drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A fixed single-qubit gate (Pauli, Clifford, T, √X, …).
    Fixed,
    /// A gate with angle arguments.
    Rotation,
    /// A multi-qubit box (`rzz`, `rxx`, a declared gate, …).
    Multi,
    /// A measurement or a barrier.
    Other,
}

/// What an op draws on its target wire(s).
#[derive(Debug, Clone, PartialEq)]
enum Target {
    /// A box labelled `label` from the topmost to the bottommost of `qubits`.
    /// When `numbered`, each operand's wire shows its position in the call
    /// (for gates whose operand order is not evident from the drawing).
    Box {
        label: String,
        qubits: Vec<usize>,
        numbered: bool,
    },
    /// A control dot (the target of `cz`, which is symmetric).
    Dot(usize),
    /// The two crosses of a swap.
    Swap(usize, usize),
    /// A meter on `qubit`, joined to the classical wire at bit `cbit`.
    Measure { qubit: usize, cbit: usize },
    /// A barrier across `qubits`.
    Barrier(Vec<usize>),
}

/// One drawn instruction: control dots plus a target.
#[derive(Debug, Clone, PartialEq)]
struct Op {
    controls: Vec<usize>,
    target: Target,
    kind: Kind,
}

impl Op {
    fn new(controls: &[usize], target: Target, kind: Kind) -> Self {
        Op {
            controls: controls.to_vec(),
            target,
            kind,
        }
    }

    fn boxed(controls: &[usize], label: String, qubit: usize, kind: Kind) -> Self {
        let target = Target::Box {
            label,
            qubits: vec![qubit],
            numbered: false,
        };
        Op::new(controls, target, kind)
    }

    /// The target's qubits.
    fn target_qubits(&self) -> Vec<usize> {
        match &self.target {
            Target::Box { qubits, .. } | Target::Barrier(qubits) => qubits.clone(),
            Target::Dot(q) | Target::Measure { qubit: q, .. } => vec![*q],
            Target::Swap(a, b) => vec![*a, *b],
        }
    }

    /// The topmost and bottommost qubit the op touches (controls included).
    fn qubit_span(&self) -> (usize, usize) {
        let qubits = self.target_qubits();
        let all = self.controls.iter().chain(&qubits);
        let lo = all.clone().copied().min().unwrap_or(0);
        let hi = all.copied().max().unwrap_or(0);
        (lo, hi)
    }

    /// The layout rows the op occupies: its qubit span and, for a
    /// measurement, every wire below it down to the classical one (`row
    /// num_qubits`).
    fn rows(&self, num_qubits: usize) -> (usize, usize) {
        let (lo, hi) = self.qubit_span();
        match self.target {
            Target::Measure { .. } => (lo, num_qubits),
            _ => (lo, hi),
        }
    }
}

/// Write `param` as it reads in a drawing.
fn angle(param: &GateParam, source: &Source, ascii: bool) -> String {
    match *param {
        GateParam::Fixed(v) => number(v, ascii),
        GateParam::Param(i) => match source.names.get(i) {
            Some(name) => name.clone(),
            None => format!("p{i}"),
        },
        GateParam::Expr(id) => {
            let mut text = String::new();
            match source.exprs.and_then(|exprs| exprs.nodes(id)) {
                Some(nodes) if print(nodes, Dialect::Qasm3, source.names, &mut text).is_ok() => {
                    text
                }
                _ => "?".to_string(),
            }
        }
    }
}

/// A fixed angle: a short multiple of π when it is one, otherwise four
/// decimals (scientific notation far from 1).
fn number(value: f64, ascii: bool) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let pi = if ascii { "pi" } else { "π" };
    // The first denominator that makes `value / π` a whole fraction is the
    // reduced one.
    for den in [1u32, 2, 3, 4, 6, 8, 12, 16] {
        let k = value * f64::from(den) / PI;
        let whole = k.round();
        if whole != 0.0 && whole.abs() <= 64.0 && (k - whole).abs() < 1e-9 {
            let sign = if whole < 0.0 { "-" } else { "" };
            let magnitude = whole.abs();
            let numerator = match (magnitude == 1.0, ascii) {
                (true, _) => pi.to_string(),
                (false, true) => format!("{magnitude}*{pi}"),
                (false, false) => format!("{magnitude}{pi}"),
            };
            return match den {
                1 => format!("{sign}{numerator}"),
                _ => format!("{sign}{numerator}/{den}"),
            };
        }
    }
    let magnitude = value.abs();
    if !(1e-3..1e5).contains(&magnitude) {
        return format!("{value:.2e}");
    }
    let text = format!("{value:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `name(a, b, …)` with the gate's angles, or `name` when it has none.
fn with_angles<'p>(
    name: &str,
    params: impl IntoIterator<Item = &'p GateParam>,
    source: &Source,
    ascii: bool,
) -> String {
    let args: Vec<String> = params
        .into_iter()
        .map(|p| angle(p, source, ascii))
        .collect();
    if args.is_empty() {
        name.to_string()
    } else {
        format!("{name}({})", args.join(", "))
    }
}

/// The ops `gate` draws as (`measure_all` draws one meter per qubit).
fn ops_of(gate: &GateInstruction, source: &Source, num_qubits: usize, ascii: bool) -> Vec<Op> {
    use GateInstruction as G;
    use Kind::{Fixed, Multi, Other, Rotation};
    let pick = |unicode: &'static str, plain: &'static str| if ascii { plain } else { unicode };
    let label = |name: &str| with_angles(name, gate.params(), source, ascii);
    let kind = if gate.params().next().is_some() {
        Rotation
    } else {
        Fixed
    };
    let multi = |name: &str, qubits: &[usize], numbered: bool| {
        let target = Target::Box {
            label: label(name),
            qubits: qubits.to_vec(),
            numbered,
        };
        Op::new(&[], target, Multi)
    };
    let op = match gate {
        G::H(q) => Op::boxed(&[], "H".into(), *q, Fixed),
        G::X(q) => Op::boxed(&[], "X".into(), *q, Fixed),
        G::Y(q) => Op::boxed(&[], "Y".into(), *q, Fixed),
        G::Z(q) => Op::boxed(&[], "Z".into(), *q, Fixed),
        G::S(q) => Op::boxed(&[], "S".into(), *q, Fixed),
        G::T(q) => Op::boxed(&[], "T".into(), *q, Fixed),
        G::Sdg(q) => Op::boxed(&[], pick("S†", "Sdg").into(), *q, Fixed),
        G::Tdg(q) => Op::boxed(&[], pick("T†", "Tdg").into(), *q, Fixed),
        G::Id(q) => Op::boxed(&[], "I".into(), *q, Fixed),
        G::Sx(q) => Op::boxed(&[], pick("√X", "SX").into(), *q, Fixed),
        G::Sxdg(q) => Op::boxed(&[], pick("√X†", "SXdg").into(), *q, Fixed),
        G::Rx { qubit, .. } => Op::boxed(&[], label("Rx"), *qubit, kind),
        G::Ry { qubit, .. } => Op::boxed(&[], label("Ry"), *qubit, kind),
        G::Rz { qubit, .. } => Op::boxed(&[], label("Rz"), *qubit, kind),
        G::P { qubit, .. } => Op::boxed(&[], label("P"), *qubit, kind),
        G::U1 { qubit, .. } => Op::boxed(&[], label("U1"), *qubit, kind),
        G::U2 { qubit, .. } => Op::boxed(&[], label("U2"), *qubit, kind),
        G::U { qubit, .. } => Op::boxed(&[], label("U3"), *qubit, kind),
        G::UGate { qubit, .. } => Op::boxed(&[], label("U"), *qubit, kind),
        G::U0 { qubit, .. } => Op::boxed(&[], label("U0"), *qubit, kind),
        G::Cx(c, t) => Op::boxed(&[*c], "X".into(), *t, Fixed),
        G::Cy(c, t) => Op::boxed(&[*c], "Y".into(), *t, Fixed),
        G::Ch(c, t) => Op::boxed(&[*c], "H".into(), *t, Fixed),
        G::Csx(c, t) => Op::boxed(&[*c], pick("√X", "SX").into(), *t, Fixed),
        G::Cz(c, t) => Op::new(&[*c], Target::Dot(*t), Fixed),
        G::Cp { q0, q1, .. } => Op::boxed(&[*q0], label("P"), *q1, kind),
        G::Cu1 { q0, q1, .. } => Op::boxed(&[*q0], label("U1"), *q1, kind),
        G::Crx {
            control, target, ..
        } => Op::boxed(&[*control], label("Rx"), *target, kind),
        G::Cry {
            control, target, ..
        } => Op::boxed(&[*control], label("Ry"), *target, kind),
        G::Crz {
            control, target, ..
        } => Op::boxed(&[*control], label("Rz"), *target, kind),
        G::Cu3 {
            control, target, ..
        } => Op::boxed(&[*control], label("U3"), *target, kind),
        G::Cu {
            control, target, ..
        } => Op::boxed(&[*control], label("U"), *target, kind),
        G::Ccx(c0, c1, t) => Op::boxed(&[*c0, *c1], "X".into(), *t, Fixed),
        G::C3x(c0, c1, c2, t) => Op::boxed(&[*c0, *c1, *c2], "X".into(), *t, Fixed),
        G::C3sqrtx(c0, c1, c2, t) => {
            Op::boxed(&[*c0, *c1, *c2], pick("√X", "SX").into(), *t, Fixed)
        }
        G::C4x(c0, c1, c2, c3, t) => Op::boxed(&[*c0, *c1, *c2, *c3], "X".into(), *t, Fixed),
        G::Swap(a, b) => Op::new(&[], Target::Swap(*a, *b), Fixed),
        G::Cswap(c, a, b) => Op::new(&[*c], Target::Swap(*a, *b), Fixed),
        G::Rzz { q0, q1, .. } => multi("Rzz", &[*q0, *q1], false),
        G::Rxx { q0, q1, .. } => multi("Rxx", &[*q0, *q1], false),
        G::Rccx(a, b, c) => multi("Rccx", &[*a, *b, *c], true),
        G::Rc3x(a, b, c, d) => multi("Rc3x", &[*a, *b, *c, *d], true),
        G::Custom(call) => match call.qubits() {
            [] => return Vec::new(),
            [q] => Op::boxed(&[], label(call.name()), *q, Multi),
            qubits => multi(call.name(), qubits, true),
        },
        G::Barrier(qubits) => {
            let qubits = if qubits.is_empty() {
                (0..num_qubits).collect()
            } else {
                qubits.clone()
            };
            if qubits.is_empty() {
                return Vec::new();
            }
            Op::new(&[], Target::Barrier(qubits), Other)
        }
        G::Measure { qubit, cbit } => Op::new(
            &[],
            Target::Measure {
                qubit: *qubit,
                cbit: *cbit,
            },
            Other,
        ),
        G::MeasureAll => {
            return (0..num_qubits)
                .map(|q| Op::new(&[], Target::Measure { qubit: q, cbit: q }, Other))
                .collect();
        }
    };
    vec![op]
}

// ──────────────────────────────── Layout ──────────────────────────────────

/// The ops of a circuit packed into columns.
struct Layout {
    /// Qubit wires drawn: the circuit's, or more if a gate names a qubit
    /// beyond them.
    num_qubits: usize,
    /// Whether a classical wire is drawn (the circuit measures).
    classical: bool,
    columns: Vec<Vec<Op>>,
}

fn layout(source: &Source, ascii: bool) -> Layout {
    let highest = source
        .gates
        .iter()
        .flat_map(|g| match g {
            GateInstruction::Barrier(qubits) => qubits.clone(),
            GateInstruction::Measure { qubit, .. } => vec![*qubit],
            other => other.acts_on().qubits().to_vec(),
        })
        .max();
    let num_qubits = source.num_qubits.max(highest.map_or(0, |q| q + 1));
    let ops: Vec<Op> = source
        .gates
        .iter()
        .flat_map(|g| ops_of(g, source, num_qubits, ascii))
        .collect();
    let classical = ops
        .iter()
        .any(|op| matches!(op.target, Target::Measure { .. }));
    // `next[row]`: the first column with that row free.
    let mut next = vec![0usize; num_qubits + 1];
    let mut columns: Vec<Vec<Op>> = Vec::new();
    for op in ops {
        let (lo, hi) = op.rows(num_qubits);
        let span = &mut next[lo..=hi];
        let column = span.iter().copied().max().unwrap_or(0);
        span.fill(column + 1);
        if columns.len() <= column {
            columns.resize_with(column + 1, Vec::new);
        }
        columns[column].push(op);
    }
    Layout {
        num_qubits,
        classical,
        columns,
    }
}

fn wire_labels(layout: &Layout) -> Vec<String> {
    let mut labels: Vec<String> = (0..layout.num_qubits).map(|q| format!("q{q}")).collect();
    if layout.classical {
        labels.push("c".to_string());
    }
    labels
}

/// The classical bits drawn: one past the highest measured.
fn classical_width(layout: &Layout) -> usize {
    layout
        .columns
        .iter()
        .flatten()
        .filter_map(|op| match op.target {
            Target::Measure { cbit, .. } => Some(cbit + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

// ───────────────────────────────── Text ───────────────────────────────────

/// A character grid; writes outside it are ignored.
struct Grid {
    cells: Vec<Vec<char>>,
}

impl Grid {
    fn new(rows: usize, width: usize) -> Self {
        Grid {
            cells: vec![vec![' '; width]; rows],
        }
    }

    fn put(&mut self, row: usize, col: usize, ch: char) {
        if let Some(cell) = self.cells.get_mut(row).and_then(|r| r.get_mut(col)) {
            *cell = ch;
        }
    }

    fn text(&mut self, row: usize, col: usize, text: &str) {
        for (i, ch) in text.chars().enumerate() {
            self.put(row, col + i, ch);
        }
    }

    fn lines(self) -> Vec<String> {
        self.cells
            .into_iter()
            .map(|row| row.into_iter().collect::<String>().trim_end().to_string())
            .collect()
    }
}

/// The wire (middle) text row of layout row `row` (a qubit, or the
/// classical wire for `row == num_qubits`).
fn wire_row(row: usize) -> usize {
    3 * row + 1
}

fn digits(n: usize) -> usize {
    n.to_string().len()
}

/// The width of the box `label` draws in: `┤ label ├`, widened on both sides
/// by the operand numbers when `numbered`.
fn box_width(label: &str, operands: usize, numbered: bool) -> usize {
    let numbers = if numbered {
        2 * digits(operands.saturating_sub(1))
    } else {
        0
    };
    label.chars().count() + 4 + numbers
}

/// The characters op needs across, before centring.
fn text_width(op: &Op) -> usize {
    match &op.target {
        Target::Box {
            label,
            qubits,
            numbered,
        } => box_width(label, qubits.len(), *numbered),
        Target::Measure { cbit, .. } => digits(*cbit).max(3),
        Target::Dot(_) | Target::Swap(..) | Target::Barrier(_) => 1,
    }
}

/// A column's width: its widest op, made odd so every op centres on one
/// character.
fn column_width(ops: &[Op]) -> usize {
    let width = ops.iter().map(text_width).max().unwrap_or(1);
    width | 1
}

/// Draw a box from qubit `lo` to `hi` centred in the column at `x0`, `width`
/// wide; returns its top and bottom text rows.
fn draw_box(
    grid: &mut Grid,
    x0: usize,
    width: usize,
    inner: usize,
    lo: usize,
    hi: usize,
) -> (usize, usize) {
    let mut box_w = inner;
    if (width - box_w) % 2 == 1 {
        box_w += 1;
    }
    let left = x0 + (width - box_w) / 2;
    let right = left + box_w - 1;
    let (top, bottom) = (3 * lo, 3 * hi + 2);
    for x in left..=right {
        grid.put(top, x, '─');
        grid.put(bottom, x, '─');
    }
    grid.put(top, left, '┌');
    grid.put(top, right, '┐');
    grid.put(bottom, left, '└');
    grid.put(bottom, right, '┘');
    for row in top + 1..bottom {
        for x in left + 1..right {
            grid.put(row, x, ' ');
        }
        let (l, r) = if row % 3 == 1 {
            ('┤', '├')
        } else {
            ('│', '│')
        };
        grid.put(row, left, l);
        grid.put(row, right, r);
    }
    (top, bottom)
}

/// Draw `op` in the column at `x0`, `width` characters wide.
fn draw_op(grid: &mut Grid, op: &Op, x0: usize, width: usize, num_qubits: usize) {
    let cx = x0 + width / 2;
    let mut boxed: Option<(usize, usize)> = None;
    let mut marks: Vec<usize> = Vec::new();
    match &op.target {
        Target::Box {
            label,
            qubits,
            numbered,
        } => {
            let lo = qubits.iter().copied().min().unwrap_or(0);
            let hi = qubits.iter().copied().max().unwrap_or(0);
            let inner = box_width(label, qubits.len(), *numbered);
            let (top, bottom) = draw_box(grid, x0, width, inner, lo, hi);
            let label_len = label.chars().count();
            let label_row = (top + bottom) / 2;
            grid.text(label_row, cx - label_len / 2, label);
            if *numbered {
                let left = x0 + (width - inner) / 2;
                for (i, q) in qubits.iter().enumerate() {
                    grid.text(wire_row(*q), left + 1, &i.to_string());
                }
            }
            boxed = Some((top, bottom));
        }
        Target::Dot(q) => {
            grid.put(wire_row(*q), cx, '■');
            marks.push(*q);
        }
        Target::Swap(a, b) => {
            grid.put(wire_row(*a), cx, '×');
            grid.put(wire_row(*b), cx, '×');
            marks.extend([*a, *b]);
        }
        Target::Barrier(qubits) => {
            for q in qubits {
                for row in 3 * q..3 * q + 3 {
                    grid.put(row, cx, '░');
                }
            }
            return;
        }
        Target::Measure { qubit, cbit } => {
            let (_, bottom) = draw_box(grid, x0, width, 3, *qubit, *qubit);
            grid.put(wire_row(*qubit), cx, 'M');
            grid.put(bottom, cx, '╥');
            for row in bottom + 1..wire_row(num_qubits) {
                let ch = if row % 3 == 1 { '╫' } else { '║' };
                grid.put(row, cx, ch);
            }
            grid.put(wire_row(num_qubits), cx, '╩');
            let label = cbit.to_string();
            grid.text(wire_row(num_qubits) + 1, cx - label.len() / 2, &label);
            return;
        }
    }
    for c in &op.controls {
        grid.put(wire_row(*c), cx, '■');
        marks.push(*c);
    }
    // The vertical line joining controls and target(s).
    let (lo, hi) = op.qubit_span();
    for row in wire_row(lo)..=wire_row(hi) {
        if let Some((top, bottom)) = boxed {
            if row == top {
                grid.put(row, cx, '┴');
                continue;
            }
            if row == bottom {
                grid.put(row, cx, '┬');
                continue;
            }
            if row > top && row < bottom {
                continue;
            }
        }
        if row % 3 == 1 {
            if !marks.contains(&((row - 1) / 3)) {
                grid.put(row, cx, '┼');
            }
        } else {
            grid.put(row, cx, '│');
        }
    }
}

/// One page of the drawing: the wire labels, then `columns`.
fn text_page(
    layout: &Layout,
    labels: &[String],
    columns: &[(usize, &[Op])],
    first: bool,
    last: bool,
) -> Vec<String> {
    let label_w = labels.iter().map(|l| l.len() + 2).max().unwrap_or(0);
    let rows = layout.num_qubits + usize::from(layout.classical);
    let body: usize = columns.iter().map(|(w, _)| w + 1).sum();
    let width = label_w + usize::from(!first) + body + 1 + usize::from(!last);
    let mut grid = Grid::new(3 * rows, width);
    for (row, label) in labels.iter().enumerate() {
        let text = format!("{label}: ");
        grid.text(wire_row(row), label_w - text.len(), &text);
    }
    let start = label_w + usize::from(!first);
    for row in 0..rows {
        let wire = if row == layout.num_qubits {
            '═'
        } else {
            '─'
        };
        for x in start..width - usize::from(!last) {
            grid.put(wire_row(row), x, wire);
        }
        if !first {
            grid.put(wire_row(row), label_w, '«');
        }
        if !last {
            grid.put(wire_row(row), width - 1, '»');
        }
    }
    let mut x = start + 1;
    for (column_w, ops) in columns {
        for op in *ops {
            draw_op(&mut grid, op, x, *column_w, layout.num_qubits);
        }
        x += column_w + 1;
    }
    grid.lines()
}

fn draw_text(source: &Source, options: TextOptions) -> String {
    let layout = layout(source, options.ascii);
    if layout.num_qubits == 0 {
        return String::new();
    }
    let mut labels = wire_labels(&layout);
    // The classical wire is labelled with its width, e.g. `c3`.
    if layout.classical {
        labels.pop();
        labels.push(format!("c{}", classical_width(&layout)));
    }
    let label_w = labels.iter().map(|l| l.len() + 2).max().unwrap_or(0);
    let columns: Vec<(usize, &[Op])> = layout
        .columns
        .iter()
        .map(|ops| (column_width(ops), ops.as_slice()))
        .collect();
    // Split the columns into pages no wider than `fold`.
    let mut pages: Vec<&[(usize, &[Op])]> = Vec::new();
    match options.fold {
        None => pages.push(&columns),
        Some(fold) => {
            let mut begin = 0;
            let mut used = label_w + 3; // `«`, the trailing wire and `»`
            for (i, (w, _)) in columns.iter().enumerate() {
                if i > begin && used + w + 1 > fold {
                    pages.push(&columns[begin..i]);
                    begin = i;
                    used = label_w + 3;
                }
                used += w + 1;
            }
            pages.push(&columns[begin..]);
        }
    }
    let count = pages.len();
    let mut lines: Vec<String> = Vec::new();
    for (i, page) in pages.into_iter().enumerate() {
        if i > 0 {
            lines.push(String::new());
        }
        lines.extend(text_page(&layout, &labels, page, i == 0, i + 1 == count));
    }
    let text = lines.join("\n");
    if options.ascii {
        text.chars().map(to_ascii).collect()
    } else {
        text
    }
}

/// The ASCII stand-in for a drawing character.
fn to_ascii(ch: char) -> char {
    match ch {
        '─' => '-',
        '═' => '=',
        '│' | '║' | '┤' | '├' => '|',
        '┌' | '┐' | '└' | '┘' | '┴' | '┬' | '┼' | '╥' | '╫' | '╩' => '+',
        '■' => '*',
        '×' => 'x',
        '░' => '#',
        '«' => '<',
        '»' => '>',
        other => other,
    }
}

// ───────────────────────────────── SVG ────────────────────────────────────

/// Height of one wire's lane.
const LANE: f64 = 48.0;
/// Height of a single-qubit gate box.
const GATE: f64 = 32.0;
/// Approximate advance of one character of the 13 px monospace font.
const CHAR: f64 = 7.8;
/// Space between columns, and the margins.
const GAP: f64 = 14.0;
const WIRE_COLOUR: &str = "#1f2937";

fn colour(kind: Kind) -> &'static str {
    match kind {
        Kind::Fixed => "#2563eb",
        Kind::Rotation => "#7c3aed",
        Kind::Multi => "#0f766e",
        Kind::Other => "#4b5563",
    }
}

fn svg_width(op: &Op) -> f64 {
    match &op.target {
        Target::Box {
            label,
            qubits,
            numbered,
        } => {
            let numbers = if *numbered {
                2.0 * digits(qubits.len().saturating_sub(1)) as f64 * CHAR
            } else {
                0.0
            };
            (label.chars().count() as f64 * CHAR + 16.0 + numbers).max(GATE)
        }
        Target::Measure { .. } => GATE,
        Target::Dot(_) | Target::Swap(..) | Target::Barrier(_) => 14.0,
    }
}

/// `text` with the XML special characters escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

fn draw_svg(source: &Source) -> String {
    let layout = layout(source, false);
    let labels = wire_labels(&layout);
    let rows = layout.num_qubits + usize::from(layout.classical);
    let label_w = labels.iter().map(|l| l.len()).max().unwrap_or(0) as f64 * CHAR + GAP;
    let widths: Vec<f64> = layout
        .columns
        .iter()
        .map(|ops| ops.iter().map(svg_width).fold(GATE, f64::max))
        .collect();
    let width = label_w + GAP + widths.iter().map(|w| w + GAP).sum::<f64>() + GAP;
    let height = rows as f64 * LANE + if layout.classical { 14.0 } else { 0.0 } + 2.0 * 6.0;
    let wire_y = |row: usize| 6.0 + LANE * row as f64 + LANE / 2.0;

    let mut out = String::new();
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.1}\" height=\"{height:.1}\" \
         viewBox=\"0 0 {width:.1} {height:.1}\" font-family=\"ui-monospace, SFMono-Regular, \
         Menlo, Consolas, monospace\" font-size=\"13\">"
    );
    let _ = write!(
        out,
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
    );
    // Wires and their labels.
    let wire_end = width - GAP;
    for (row, label) in labels.iter().enumerate() {
        let y = wire_y(row);
        let _ = write!(
            out,
            "<text x=\"{:.1}\" y=\"{y:.1}\" text-anchor=\"end\" dominant-baseline=\"central\" \
             fill=\"{WIRE_COLOUR}\">{}</text>",
            label_w - 6.0,
            escape(label)
        );
        if row == layout.num_qubits {
            for dy in [-1.5, 1.5] {
                let _ = write!(
                    out,
                    "<line x1=\"{label_w:.1}\" y1=\"{0:.1}\" x2=\"{wire_end:.1}\" y2=\"{0:.1}\" \
                     stroke=\"{WIRE_COLOUR}\"/>",
                    y + dy
                );
            }
            // The bundle's width over a slash, as in the usual notation.
            let _ = write!(
                out,
                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                 stroke=\"{WIRE_COLOUR}\"/><text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" \
                 fill=\"{WIRE_COLOUR}\">{}</text>",
                label_w + 4.0,
                y + 5.0,
                label_w + 10.0,
                y - 5.0,
                label_w + 11.0,
                y - 5.0,
                classical_width(&layout)
            );
        } else {
            let _ = write!(
                out,
                "<line x1=\"{label_w:.1}\" y1=\"{y:.1}\" x2=\"{wire_end:.1}\" y2=\"{y:.1}\" \
                 stroke=\"{WIRE_COLOUR}\"/>"
            );
        }
    }
    let mut x = label_w + GAP;
    for (ops, column_w) in layout.columns.iter().zip(&widths) {
        let cx = x + column_w / 2.0;
        for op in ops {
            svg_op(&mut out, op, cx, layout.num_qubits, &wire_y);
        }
        x += column_w + GAP;
    }
    out.push_str("</svg>");
    out
}

fn svg_op(out: &mut String, op: &Op, cx: f64, num_qubits: usize, wire_y: &dyn Fn(usize) -> f64) {
    let fill = colour(op.kind);
    let (lo, hi) = op.qubit_span();
    if lo != hi
        && !matches!(op.target, Target::Barrier(_) | Target::Box { .. } if op.controls.is_empty())
    {
        let _ = write!(
            out,
            "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"{fill}\" \
             stroke-width=\"2\"/>",
            wire_y(lo),
            wire_y(hi)
        );
    }
    for c in &op.controls {
        let _ = write!(
            out,
            "<circle cx=\"{cx:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"{fill}\"/>",
            wire_y(*c)
        );
    }
    match &op.target {
        Target::Box {
            label,
            qubits,
            numbered,
        } => {
            let top_q = qubits.iter().copied().min().unwrap_or(0);
            let bottom_q = qubits.iter().copied().max().unwrap_or(0);
            let w = svg_width(op);
            let top = wire_y(top_q) - GATE / 2.0;
            let h = wire_y(bottom_q) - wire_y(top_q) + GATE;
            let _ = write!(
                out,
                "<rect x=\"{:.1}\" y=\"{top:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"4\" \
                 fill=\"{fill}\"/><text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
                 dominant-baseline=\"central\" fill=\"#ffffff\">{}</text>",
                cx - w / 2.0,
                top + h / 2.0,
                escape(label)
            );
            if *numbered {
                for (i, q) in qubits.iter().enumerate() {
                    let _ = write!(
                        out,
                        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" \
                         dominant-baseline=\"central\" fill=\"#ffffff\">{i}</text>",
                        cx - w / 2.0 + 4.0,
                        wire_y(*q)
                    );
                }
            }
        }
        Target::Dot(q) => {
            let _ = write!(
                out,
                "<circle cx=\"{cx:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"{fill}\"/>",
                wire_y(*q)
            );
        }
        Target::Swap(a, b) => {
            for q in [a, b] {
                let y = wire_y(*q);
                let _ = write!(
                    out,
                    "<path d=\"M{:.1} {:.1}L{:.1} {:.1}M{:.1} {:.1}L{:.1} {:.1}\" \
                     stroke=\"{fill}\" stroke-width=\"2\"/>",
                    cx - 6.0,
                    y - 6.0,
                    cx + 6.0,
                    y + 6.0,
                    cx - 6.0,
                    y + 6.0,
                    cx + 6.0,
                    y - 6.0
                );
            }
        }
        Target::Barrier(qubits) => {
            for q in qubits {
                let _ = write!(
                    out,
                    "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"10\" height=\"{LANE:.1}\" \
                     fill=\"#9ca3af\" fill-opacity=\"0.35\"/>",
                    cx - 5.0,
                    wire_y(*q) - LANE / 2.0
                );
            }
        }
        Target::Measure { qubit, cbit } => {
            let y = wire_y(*qubit);
            let classical = wire_y(num_qubits);
            for dx in [-1.5, 1.5] {
                let _ = write!(
                    out,
                    "<line x1=\"{0:.1}\" y1=\"{y:.1}\" x2=\"{0:.1}\" y2=\"{1:.1}\" \
                     stroke=\"{fill}\"/>",
                    cx + dx,
                    classical - 6.0
                );
            }
            let _ = write!(
                out,
                "<path d=\"M{:.1} {:.1}L{:.1} {:.1}L{:.1} {:.1}Z\" fill=\"{fill}\"/>",
                cx - 5.0,
                classical - 7.0,
                cx + 5.0,
                classical - 7.0,
                cx,
                classical
            );
            let _ = write!(
                out,
                "<text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10\" \
                 fill=\"{WIRE_COLOUR}\">{cbit}</text>",
                classical + 14.0
            );
            // The meter: a box with a dial and a needle.
            let half = GATE / 2.0;
            let _ = write!(
                out,
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{GATE:.1}\" height=\"{GATE:.1}\" rx=\"4\" \
                 fill=\"{fill}\"/><path d=\"M{:.1} {:.1}A9 9 0 0 1 {:.1} {:.1}M{cx:.1} {:.1}L{:.1} \
                 {:.1}\" stroke=\"#ffffff\" stroke-width=\"1.5\" fill=\"none\"/>",
                cx - half,
                y - half,
                cx - 9.0,
                y + 5.0,
                cx + 9.0,
                y + 5.0,
                y + 5.0,
                cx + 6.0,
                y - 6.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Param, ParameterizedCircuit};

    fn unicode(qc: &ParameterizedCircuit) -> String {
        qc.draw_text(TextOptions {
            ascii: false,
            fold: None,
        })
    }

    #[test]
    fn bell_pair_draws_as_expected() {
        let qc = ParameterizedCircuit::new(2).h(0).cx(0, 1).measure_all();
        let text = unicode(&qc);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 9, "{text}");
        assert!(lines[1].starts_with("q0: ─┤ H ├───■"), "{text}");
        assert!(lines[4].contains("┤ X ├"), "{text}");
        assert_eq!(lines[3].matches('┴').count(), 1, "{text}");
        assert!(lines[7].starts_with("c2: ═"), "{text}");
        assert!(lines[7].matches('╩').count() == 2, "{text}");
    }

    #[test]
    fn control_line_reaches_the_target_box() {
        let qc = ParameterizedCircuit::new(3).cx(0, 2);
        let text = unicode(&qc);
        let lines: Vec<&str> = text.lines().collect();
        let col = lines[1].chars().position(|c| c == '■').unwrap();
        let at = |row: usize| lines[row].chars().nth(col).unwrap();
        assert_eq!(at(2), '│');
        assert_eq!(at(4), '┼', "{text}");
        assert_eq!(at(6), '┴', "{text}");
        assert_eq!(at(7), 'X', "{text}");
    }

    #[test]
    fn gates_on_disjoint_qubits_share_a_column() {
        let qc = ParameterizedCircuit::new(2).h(0).x(1);
        let text = unicode(&qc);
        let lines: Vec<&str> = text.lines().collect();
        let h = lines[1].find('H').unwrap();
        let x = lines[4].find('X').unwrap();
        assert_eq!(h, x, "{text}");
    }

    #[test]
    fn angles_show_parameter_names_and_multiples_of_pi() {
        let qc = ParameterizedCircuit::new(1)
            .rx(0, Param(0))
            .rz(0, PI / 2.0)
            .ry(0, -3.0 * PI / 4.0)
            .p(0, 0.123456);
        let text = unicode(&qc);
        for label in ["Rx(theta_0)", "Rz(π/2)", "Ry(-3π/4)", "P(0.1235)"] {
            assert!(text.contains(label), "{label} missing:\n{text}");
        }
        let ascii = qc.draw_text(TextOptions {
            ascii: true,
            fold: None,
        });
        assert!(ascii.contains("Rz(pi/2)") && ascii.contains("Ry(-3*pi/4)"));
        assert!(ascii.is_ascii(), "{ascii}");
    }

    #[test]
    fn expressions_are_drawn_as_written() {
        let mut qc = ParameterizedCircuit::new(1);
        let theta = qc
            .add_expr(crate::ParamExpr::param(0) * 2.0)
            .expect("valid expression");
        let qc = qc.rx(0, theta);
        assert!(unicode(&qc).contains("Rx(theta_0*2.0)"), "{}", unicode(&qc));
    }

    #[test]
    fn every_instruction_draws_without_panicking() {
        let qc = ParameterizedCircuit::new(5)
            .h(0)
            .sdg(1)
            .sx(2)
            .sxdg(3)
            .id(4)
            .u(0, 0.1, 0.2, 0.3)
            .u2(1, 0.1, 0.2)
            .u0(2, 1.0)
            .cz(0, 4)
            .swap(1, 3)
            .cswap(0, 2, 4)
            .rzz(0, 3, Param(1))
            .rxx(1, 2, 0.5)
            .cu(0, 1, 0.1, 0.2, 0.3, 0.4)
            .cu3(2, 3, 0.1, 0.2, 0.3)
            .rccx(0, 2, 4)
            .rc3x(0, 1, 2, 3)
            .c3sqrtx(4, 3, 2, 1)
            .c4x(0, 1, 2, 3, 4)
            .barrier_on(&[1, 3])
            .barrier()
            .measure(2, 7);
        for ascii in [false, true] {
            let text = qc.draw_text(TextOptions { ascii, fold: None });
            assert!(text.contains("Rccx"), "{text}");
            assert!(text.contains("c8"), "{text}");
        }
        let svg = qc.draw_svg();
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    }

    #[test]
    fn wide_circuits_fold_into_pages() {
        let mut qc = ParameterizedCircuit::new(1);
        for _ in 0..40 {
            qc = qc.h(0);
        }
        let text = qc.draw_text(TextOptions {
            ascii: false,
            fold: Some(40),
        });
        assert!(text.lines().all(|l| l.chars().count() <= 40), "{text}");
        assert!(text.contains('»') && text.contains('«'));
        assert_eq!(text.matches('H').count(), 40);
        let unfolded = unicode(&qc);
        assert!(!unfolded.contains('»'));
    }

    #[test]
    fn gates_beyond_num_qubits_grow_the_drawing() {
        let mut qc = ParameterizedCircuit::new(1);
        qc.gates.push(GateInstruction::Cx(0, 2));
        let text = unicode(&qc);
        assert!(text.contains("q2: "), "{text}");
    }

    #[test]
    fn empty_and_zero_qubit_circuits() {
        assert_eq!(unicode(&ParameterizedCircuit::new(0)), "");
        let text = unicode(&ParameterizedCircuit::new(2));
        assert_eq!(text.lines().filter(|l| l.contains('─')).count(), 2);
        assert!(ParameterizedCircuit::new(0).draw_svg().ends_with("</svg>"));
    }

    #[test]
    fn svg_escapes_labels() {
        let qc = ParameterizedCircuit::from_qasm3(
            "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ninput float[64] a;\nqubit[1] q;\nrx(a) q[0];\n",
        )
        .expect("valid program");
        let svg = qc.draw_svg();
        assert!(svg.contains(">Rx(a)<"), "{svg}");
        assert_eq!(escape("<a&b>\"'"), "&lt;a&amp;b&gt;&quot;&apos;");
    }

    #[test]
    fn display_and_concrete_circuits_draw() {
        let qc = ParameterizedCircuit::new(1).rx(0, Param(0));
        assert_eq!(qc.to_string(), qc.draw_text(TextOptions::default()));
        let bound = qc.assign_parameters(&[PI]).expect("binds");
        assert!(bound.to_string().contains("Rx(π)"), "{bound}");
        assert!(bound.draw_svg().contains("Rx(π)"));
    }
}
