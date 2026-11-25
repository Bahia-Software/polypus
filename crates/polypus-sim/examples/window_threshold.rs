//! Sequential against parallel gate kernels for windows of `k` runs that share
//! the rayon pool (issue #593).
//!
//! The native backend runs a batch as windows of `k` circuits in one
//! `par_iter`, every circuit of a window at the same gate-parallel threshold
//! (`simulate_window`, `run_exact_with_cap`, `run_adjoint_with_cap` in
//! `polypus-infrastructure`). Its rules pick that threshold from `k` and the
//! pool size, so its API cannot force "parallel kernels with `k` = threads".
//! This example works one level down: it builds the `k` simulators itself, at
//! an explicit threshold, and times each window both ways:
//!
//! - **seq**: threshold `usize::MAX`, every kernel sequential;
//! - **par**: threshold `0`, every kernel parallel.
//!
//! For every family × path × `k` × width it prints the median, minimum and
//! maximum wall-clock time of the window over `reps` runs of each arm, after
//! one untimed warm-up of each. The arms alternate (seq first on even runs,
//! par first on odd ones), so drift is shared. Every output of every run (the
//! amplitudes, counts, probabilities or derivatives) is compared bit for bit
//! with the warm-up's sequential one: the parallel kernels must give the same
//! bits (docs/ENGINEERING.md §4), so a difference means the two arms did not
//! compute the same thing and the program stops with exit code 1.
//!
//! Families (each task of a window gets its own angles, as the rows of a batch
//! do; `qft` has none):
//!
//! - `hea`: the model of `benchmarks/bench_adjoint.py`, an `ry` angle encoder,
//!   `--depth` hardware-efficient repetitions (`Ry` and `Rz` on every qubit,
//!   then a linear `Cx` chain) and a final `Ry`/`Rz` block;
//! - `qft`: the production template, `templates::qft(n)`;
//! - `trotter`: `bench_fusion.py`'s, `H` on every qubit, then
//!   `--trotter-depth` layers of `Rz` on every qubit and `Rzz` on a ring.
//!
//! Paths:
//!
//! - `fused`, `unfused`: [`Simulator::run`] with gate fusion on and off;
//! - `sample`: [`StatevectorSimulator::run_and_sample`], `--shots` shots;
//! - `exact`: `run` and then the probabilities, as the exact path reads them;
//! - `adjoint` (`hea` only): [`StatevectorSimulator::adjoint_gradient`] of
//!   `Z₀`, every rotation of the ansatz differentiated.
//!
//! With `k = 1` the run is timed twice: called from the main thread, outside
//! the pool (`caller = outside`, as `polypus.statevector` or the calibration
//! probe run), and from a one-element `par_iter`, on a pool worker
//! (`caller = inside`, as the backend's windows run). A window of `k >= 2` is
//! always a `par_iter`.
//!
//! Run it in release, through SLURM on the cluster (never on the login node),
//! on an exclusive node when the numbers matter. The pool size is
//! `RAYON_NUM_THREADS`:
//!
//! ```text
//! RAYON_NUM_THREADS=16 srun -n 1 -c 16 --exclusive --time=01:00:00 --mem=32G \
//!     cargo run --release -p polypus-sim --features parallel \
//!     --example window_threshold -- --families hea --paths adjoint
//! ```
//!
//! Options (all optional; lists are comma-separated):
//!
//! - `--families` (`hea,qft,trotter`), `--paths`
//!   (`fused,unfused,sample,exact,adjoint`), `--k` (`1,2,4,8,16`),
//!   `--qubits` (`10,12,14,16,18`), `--callers` for `k = 1`
//!   (`outside,inside`);
//! - `--reps` (7), `--depth` of `hea` (4), `--trotter-depth` (16),
//!   `--shots` (1024);
//! - `--format` `md` (a markdown table, the default) or `csv`.

use polypus_circuit::{templates, ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
use polypus_sim::{AdjointGradient, AngleRef, Simulator, StatevectorSimulator, MAX_QUBITS};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};

const USAGE: &str = "usage: window_threshold [--families hea,qft,trotter] \
[--paths fused,unfused,sample,exact,adjoint] [--k 1,2,4,8,16] [--qubits 10,12,14,16,18] \
[--callers outside,inside] [--reps 7] [--depth 4] [--trotter-depth 16] [--shots 1024] \
[--format md|csv]";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Hea,
    Qft,
    Trotter,
}

impl Family {
    fn parse(text: &str) -> Self {
        match text {
            "hea" => Family::Hea,
            "qft" => Family::Qft,
            "trotter" => Family::Trotter,
            other => usage_error(&format!("unknown family {other:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Family::Hea => "hea",
            Family::Qft => "qft",
            Family::Trotter => "trotter",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Fused,
    Unfused,
    Sample,
    Exact,
    Adjoint,
}

impl Path {
    fn parse(text: &str) -> Self {
        match text {
            "fused" => Path::Fused,
            "unfused" => Path::Unfused,
            "sample" => Path::Sample,
            "exact" => Path::Exact,
            "adjoint" => Path::Adjoint,
            other => usage_error(&format!("unknown path {other:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Path::Fused => "fused",
            Path::Unfused => "unfused",
            Path::Sample => "sample",
            Path::Exact => "exact",
            Path::Adjoint => "adjoint",
        }
    }
}

/// Where a window's runs are called from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Caller {
    /// The main thread, outside the pool (`k = 1` only).
    Outside,
    /// A `par_iter` over the window: every run on a pool worker.
    Inside,
}

impl Caller {
    fn parse(text: &str) -> Self {
        match text {
            "outside" => Caller::Outside,
            "inside" => Caller::Inside,
            other => usage_error(&format!("unknown caller {other:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Caller::Outside => "outside",
            Caller::Inside => "inside",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Markdown,
    Csv,
}

struct Options {
    families: Vec<Family>,
    paths: Vec<Path>,
    ks: Vec<usize>,
    qubits: Vec<usize>,
    callers: Vec<Caller>,
    reps: usize,
    depth: usize,
    trotter_depth: usize,
    shots: usize,
    format: Format,
}

fn usage_error(message: &str) -> ! {
    eprintln!("window_threshold: {message}\n{USAGE}");
    std::process::exit(2)
}

fn parse_count(flag: &str, text: &str) -> usize {
    text.trim().parse().unwrap_or_else(|_| {
        usage_error(&format!(
            "{flag} takes a non-negative integer, not {text:?}"
        ))
    })
}

fn parse_list<T>(text: &str, item: impl Fn(&str) -> T) -> Vec<T> {
    text.split(',').map(|s| item(s.trim())).collect()
}

fn parse_options() -> Options {
    let mut options = Options {
        families: vec![Family::Hea, Family::Qft, Family::Trotter],
        paths: vec![
            Path::Fused,
            Path::Unfused,
            Path::Sample,
            Path::Exact,
            Path::Adjoint,
        ],
        ks: vec![1, 2, 4, 8, 16],
        qubits: vec![10, 12, 14, 16, 18],
        callers: vec![Caller::Outside, Caller::Inside],
        reps: 7,
        depth: 4,
        trotter_depth: 16,
        shots: 1024,
        format: Format::Markdown,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{USAGE}");
            std::process::exit(0);
        }
        let value = args
            .next()
            .unwrap_or_else(|| usage_error(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--families" => options.families = parse_list(&value, Family::parse),
            "--paths" => options.paths = parse_list(&value, Path::parse),
            "--k" => options.ks = parse_list(&value, |s| parse_count("--k", s)),
            "--qubits" => options.qubits = parse_list(&value, |s| parse_count("--qubits", s)),
            "--callers" => options.callers = parse_list(&value, Caller::parse),
            "--reps" => options.reps = parse_count("--reps", &value),
            "--depth" => options.depth = parse_count("--depth", &value),
            "--trotter-depth" => options.trotter_depth = parse_count("--trotter-depth", &value),
            "--shots" => options.shots = parse_count("--shots", &value),
            "--format" => {
                options.format = match value.as_str() {
                    "md" => Format::Markdown,
                    "csv" => Format::Csv,
                    other => usage_error(&format!("unknown format {other:?}")),
                }
            }
            other => usage_error(&format!("unknown option {other:?}")),
        }
    }
    if options.ks.contains(&0) {
        usage_error("a window holds at least one run (--k 0)");
    }
    if options.reps == 0 {
        usage_error("a point needs at least one repetition (--reps 0)");
    }
    if options.qubits.iter().any(|&n| n < 2) {
        usage_error("every family needs at least 2 qubits");
    }
    options
}

/// One run of a window: its circuit and, for the adjoint, the angles to
/// differentiate.
struct Task {
    circuit: ConcreteCircuit,
    wrt: Vec<AngleRef>,
    seed: u64,
}

/// `bench_adjoint.py`'s model for row `row`: an `ry` angle encoder of
/// `x_q = 0.1 · ((row + q) mod 7)`, `depth` repetitions of (`Ry`, `Rz` on every
/// qubit, linear `Cx` chain) and the final `Ry`/`Rz` block, with the
/// benchmark's parameters `θ_j = 0.1 + 0.03 j`. Returns the circuit and the
/// positions of the ansatz's rotations (the encoder is data, not a weight).
fn hea(n: usize, depth: usize, row: usize) -> (ConcreteCircuit, Vec<usize>) {
    let mut gates: Vec<G> = (0..n)
        .map(|qubit| G::Ry {
            qubit,
            theta: Fixed(0.1 * ((row + qubit) % 7) as f64),
        })
        .collect();
    let mut positions = Vec::new();
    let mut next = 0usize;
    let mut rotation_block = |gates: &mut Vec<G>, positions: &mut Vec<usize>| {
        for ry in [true, false] {
            for qubit in 0..n {
                let theta = Fixed(0.1 + 0.03 * next as f64);
                next += 1;
                positions.push(gates.len());
                gates.push(if ry {
                    G::Ry { qubit, theta }
                } else {
                    G::Rz { qubit, theta }
                });
            }
        }
    };
    for _ in 0..depth {
        rotation_block(&mut gates, &mut positions);
        gates.extend((0..n - 1).map(|q| G::Cx(q, q + 1)));
    }
    rotation_block(&mut gates, &mut positions);
    let circuit = ConcreteCircuit::try_new(n, gates).expect("a valid circuit");
    (circuit, positions)
}

/// `bench_fusion.py`'s Trotter circuit, its `Rz` angles moved by `0.001 · row`
/// so the rows of a window differ.
fn trotter(n: usize, depth: usize, row: usize) -> ConcreteCircuit {
    let mut gates: Vec<G> = (0..n).map(G::H).collect();
    for layer in 0..depth {
        for qubit in 0..n {
            gates.push(G::Rz {
                qubit,
                theta: Fixed(0.1 + 0.01 * layer as f64 + 0.001 * row as f64),
            });
        }
        for q0 in 0..n {
            gates.push(G::Rzz {
                q0,
                q1: (q0 + 1) % n,
                theta: Fixed(0.2 - 0.005 * layer as f64),
            });
        }
    }
    ConcreteCircuit::try_new(n, gates).expect("a valid circuit")
}

fn tasks(family: Family, n: usize, k: usize, options: &Options) -> Vec<Task> {
    (0..k)
        .map(|row| {
            let (circuit, wrt) = match family {
                Family::Hea => {
                    let (circuit, positions) = hea(n, options.depth, row);
                    let wrt = positions.iter().map(|&p| AngleRef::new(p, 0)).collect();
                    (circuit, wrt)
                }
                Family::Qft => (
                    templates::qft(n)
                        .and_then(|qc| qc.assign_parameters(&[]))
                        .expect("a valid QFT"),
                    Vec::new(),
                ),
                Family::Trotter => (trotter(n, options.trotter_depth, row), Vec::new()),
            };
            Task {
                circuit,
                wrt,
                seed: 1000 + row as u64,
            }
        })
        .collect()
}

/// `Z₀`, the readout of `bench_adjoint.py`'s model.
fn z0(b: usize) -> f64 {
    if b & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// What one run returns, kept to compare the two arms bit for bit.
enum Output {
    Amplitudes(Vec<polypus_sim::C64>),
    Counts(HashMap<usize, u64>),
    Probabilities(Vec<f64>),
    Gradient(AdjointGradient),
}

fn same_f64(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

impl Output {
    fn same_bits(&self, other: &Output) -> bool {
        match (self, other) {
            (Output::Amplitudes(a), Output::Amplitudes(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(x, y)| {
                        x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()
                    })
            }
            (Output::Counts(a), Output::Counts(b)) => a == b,
            (Output::Probabilities(a), Output::Probabilities(b)) => same_f64(a, b),
            (Output::Gradient(a), Output::Gradient(b)) => {
                a.expectation.to_bits() == b.expectation.to_bits()
                    && same_f64(&a.derivatives, &b.derivatives)
            }
            _ => false,
        }
    }
}

fn run_task(task: &Task, path: Path, threshold: usize, shots: usize) -> Output {
    let simulator = StatevectorSimulator {
        max_qubits: MAX_QUBITS,
        parallel_threshold: threshold,
        fusion: path != Path::Unfused,
    };
    match path {
        Path::Fused | Path::Unfused => Output::Amplitudes(
            simulator
                .run(&task.circuit)
                .expect("a valid circuit")
                .into_amplitudes(),
        ),
        Path::Sample => Output::Counts(
            simulator
                .run_and_sample(&task.circuit, shots, task.seed)
                .expect("a valid circuit"),
        ),
        Path::Exact => Output::Probabilities(
            simulator
                .run(&task.circuit)
                .expect("a valid circuit")
                .probabilities(),
        ),
        Path::Adjoint => Output::Gradient(
            simulator
                .adjoint_gradient(&task.circuit, &task.wrt, z0, None)
                .expect("a differentiable circuit"),
        ),
    }
}

/// Run one window at `threshold` and return its wall-clock time and outputs.
fn time_window(
    window: &[Task],
    path: Path,
    threshold: usize,
    caller: Caller,
    shots: usize,
) -> (Duration, Vec<Output>) {
    let start = Instant::now();
    let outputs: Vec<Output> = match caller {
        Caller::Outside => window
            .iter()
            .map(|task| run_task(task, path, threshold, shots))
            .collect(),
        Caller::Inside => window
            .par_iter()
            .map(|task| run_task(task, path, threshold, shots))
            .collect(),
    };
    (start.elapsed(), outputs)
}

/// Median, minimum and maximum in milliseconds.
fn summary(times: &mut [Duration]) -> Stats {
    times.sort_unstable();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    (
        ms(times[times.len() / 2]),
        ms(times[0]),
        ms(times[times.len() - 1]),
    )
}

/// The label of one point, for the rows and the mismatch message.
struct Point {
    family: Family,
    path: Path,
    k: usize,
    caller: Caller,
    n: usize,
}

impl Point {
    fn describe(&self) -> String {
        format!(
            "family={} path={} k={} caller={} qubits={}",
            self.family.name(),
            self.path.name(),
            self.k,
            self.caller.name(),
            self.n
        )
    }
}

fn check(point: &Point, arm: &str, reference: &[Output], outputs: &[Output]) {
    for (row, (a, b)) in reference.iter().zip(outputs).enumerate() {
        if !a.same_bits(b) {
            eprintln!(
                "window_threshold: the {arm} arm's output for row {row} differs bit for bit \
                 from the sequential reference at {}; the two arms must compute the same thing \
                 (docs/ENGINEERING.md §4)",
                point.describe()
            );
            std::process::exit(1);
        }
    }
}

/// Median, minimum and maximum of one arm, in milliseconds.
type Stats = (f64, f64, f64);

/// Time both arms of one point and return `(seq, par)` summaries.
fn measure(point: &Point, window: &[Task], options: &Options) -> (Stats, Stats) {
    let (seq, par) = (usize::MAX, 0);
    let run = |threshold| time_window(window, point.path, threshold, point.caller, options.shots);
    let (_, reference) = run(seq);
    let (_, warm) = run(par);
    check(point, "par", &reference, &warm);
    let mut seq_times = Vec::with_capacity(options.reps);
    let mut par_times = Vec::with_capacity(options.reps);
    for rep in 0..options.reps {
        let order = if rep % 2 == 0 { [seq, par] } else { [par, seq] };
        for threshold in order {
            let (time, outputs) = run(threshold);
            if threshold == seq {
                check(point, "seq", &reference, &outputs);
                seq_times.push(time);
            } else {
                check(point, "par", &reference, &outputs);
                par_times.push(time);
            }
        }
    }
    (summary(&mut seq_times), summary(&mut par_times))
}

fn main() {
    let options = parse_options();
    let threads = rayon::current_num_threads();
    println!(
        "# window_threshold: threads={threads} (RAYON_NUM_THREADS={}), threshold \
         StatevectorSimulator::new() resolves (cache or default)={}, reps={}, hea depth={}, \
         trotter depth={}, shots={}; seq = threshold \
         usize::MAX, par = threshold 0; times in ms (median, min, max of the window)",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string()),
        StatevectorSimulator::new().parallel_threshold,
        options.reps,
        options.depth,
        options.trotter_depth,
        options.shots,
    );
    let columns = [
        "family", "path", "k", "caller", "qubits", "gates", "seq_med", "seq_min", "seq_max",
        "par_med", "par_min", "par_max", "seq/par",
    ];
    match options.format {
        Format::Csv => println!("{}", columns.join(",")),
        Format::Markdown => {
            println!("| {} |", columns.join(" | "));
            println!("|{}", "---|".repeat(columns.len()));
        }
    }
    let mut stdout = std::io::stdout();
    for &family in &options.families {
        for &path in &options.paths {
            if path == Path::Adjoint && family != Family::Hea {
                continue;
            }
            for &k in &options.ks {
                let callers: &[Caller] = if k == 1 {
                    &options.callers
                } else {
                    &[Caller::Inside]
                };
                for &caller in callers {
                    for &n in &options.qubits {
                        let window = tasks(family, n, k, &options);
                        let point = Point {
                            family,
                            path,
                            k,
                            caller,
                            n,
                        };
                        let ((sm, s0, s1), (pm, p0, p1)) = measure(&point, &window, &options);
                        let gates = window[0].circuit.gates().len();
                        let ratio = sm / pm;
                        match options.format {
                            Format::Csv => println!(
                                "{},{},{k},{},{n},{gates},{sm:.3},{s0:.3},{s1:.3},{pm:.3},{p0:.3},\
                                 {p1:.3},{ratio:.2}",
                                family.name(),
                                path.name(),
                                caller.name(),
                            ),
                            Format::Markdown => {
                                // Bold where the parallel kernels win on the median.
                                let (b0, b1) = if ratio > 1.0 { ("**", "**") } else { ("", "") };
                                println!(
                                    "| {} | {} | {k} | {} | {n} | {gates} | {sm:.3} | {s0:.3} | \
                                     {s1:.3} | {b0}{pm:.3}{b1} | {p0:.3} | {p1:.3} | {ratio:.2} |",
                                    family.name(),
                                    path.name(),
                                    caller.name(),
                                );
                            }
                        }
                        // Progress shows up in a SLURM log as it is measured.
                        let _ = stdout.flush();
                    }
                }
            }
        }
    }
}
