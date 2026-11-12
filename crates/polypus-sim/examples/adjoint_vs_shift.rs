//! Adjoint differentiation against the parameter-shift rule (issue #503).
//!
//! For a hardware-efficient circuit with `P` trainable rotations (`Ry`/`Rz`
//! layers with a `Cx` ladder after each layer) on `n` qubits, times
//!
//! - **adjoint**: one [`StatevectorSimulator::adjoint_gradient`] call, every
//!   rotation differentiated;
//! - **shift**: the `2P + 1` exact simulations of the shift rule (the base
//!   circuit and each rotation moved by `±π/2`), each read out as an
//!   expectation from its probabilities, as the exact QML path does;
//!
//! - **forward**: one exact simulation of the circuit and the read-out of its
//!   probabilities — what the exact QML gradient adds to the adjoint to read
//!   `θ`'s value from `θ`'s own run (C-11 (f), "V2" in issue #503);
//!
//! and prints the three, the speedup, the forward pass's cost relative to the
//! adjoint (`v2_overhead`), and the largest difference between the two
//! gradients (they are both exact, so it is rounding). Every arm is timed the
//! same way: the best of `k` runs. One circuit at a time on
//! one simulator, so the comparison is per gradient, not per batch: the native
//! backend runs the shift circuits of a batch side by side, which the
//! end-to-end benchmark (`benchmarks/bench_adjoint.py`) measures.
//!
//! Run it in release, through SLURM on the cluster (never on the login node):
//!
//! ```text
//! srun -n 1 -c 16 --time=01:00:00 --mem=32G \
//!     cargo run --release -p polypus-sim --features parallel \
//!     --example adjoint_vs_shift -- 12,16,18 8,16,32,64,128,256 3
//! ```
//!
//! Arguments (all optional): comma-separated qubit counts, comma-separated
//! rotation counts, and `k`, the runs per arm whose best is kept (default 3).
//! The shift arm is `2P + 1` simulations per run, so its largest points take
//! `k` times as long: pass `k = 1` to time every arm once. Without
//! `--features parallel` every arm runs the sequential kernels.

use polypus_circuit::{ConcreteCircuit, GateInstruction as G, GateParam::Fixed};
use polypus_sim::{AngleRef, Simulator, StatevectorSimulator};
use std::f64::consts::FRAC_PI_2;
use std::time::{Duration, Instant};

/// Runs per arm whose best is kept, unless the third argument says otherwise.
const DEFAULT_REPS: usize = 3;

/// The best wall-clock time of `reps` runs of `f` (at least one), and the
/// value of the last run.
fn best_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut best = Duration::MAX;
    let mut value = None;
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        value = Some(f());
        best = best.min(start.elapsed());
    }
    (best, value.expect("at least one run"))
}

fn parse_list(arg: Option<String>, default: &[usize]) -> Vec<usize> {
    match arg {
        Some(text) => text
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .expect("a comma-separated list of integers")
            })
            .collect(),
        None => default.to_vec(),
    }
}

/// `p` rotations, alternating `Ry` and `Rz` over the qubits in order, with a
/// `Cx` ladder after every `n` of them; returns the circuit and the positions
/// of its rotations.
fn hea(n: usize, p: usize) -> (ConcreteCircuit, Vec<usize>) {
    let mut gates: Vec<G> = (0..n).map(G::H).collect();
    let mut positions = Vec::with_capacity(p);
    for k in 0..p {
        let qubit = k % n;
        let theta = Fixed(0.1 + 0.37 * k as f64);
        positions.push(gates.len());
        gates.push(if (k / n) % 2 == 0 {
            G::Ry { qubit, theta }
        } else {
            G::Rz { qubit, theta }
        });
        if qubit == n - 1 {
            gates.extend((0..n - 1).map(|q| G::Cx(q, q + 1)));
        }
    }
    gates.push(G::MeasureAll);
    let circuit = ConcreteCircuit::try_new(n, gates).expect("a valid circuit");
    (circuit, positions)
}

/// `Z₀Z₁` plus `Z_{n−1}`: a two-term diagonal observable.
fn weight(n: usize) -> impl Fn(usize) -> f64 + Sync {
    move |b: usize| {
        let z01 = if (b & 0b11).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let zl = if (b >> (n - 1)) & 1 == 0 { 1.0 } else { -1.0 };
        z01 + 0.5 * zl
    }
}

fn expectation(
    sim: &StatevectorSimulator,
    circuit: &ConcreteCircuit,
    w: &impl Fn(usize) -> f64,
) -> f64 {
    let sv = sim.run(circuit).expect("a valid circuit");
    sv.probabilities()
        .iter()
        .enumerate()
        .map(|(b, p)| p * w(b))
        .sum()
}

/// `circuit` with the rotation at `position` moved by `delta`.
fn shifted(circuit: &ConcreteCircuit, position: usize, delta: f64) -> ConcreteCircuit {
    let mut gates = circuit.gates().to_vec();
    gates[position] = match &gates[position] {
        G::Ry {
            qubit,
            theta: Fixed(t),
        } => G::Ry {
            qubit: *qubit,
            theta: Fixed(t + delta),
        },
        G::Rz {
            qubit,
            theta: Fixed(t),
        } => G::Rz {
            qubit: *qubit,
            theta: Fixed(t + delta),
        },
        other => unreachable!("{other:?} is not a rotation of this circuit"),
    };
    ConcreteCircuit::try_new(circuit.num_qubits(), gates).expect("a valid circuit")
}

fn shift_gradient(
    sim: &StatevectorSimulator,
    circuit: &ConcreteCircuit,
    positions: &[usize],
    w: &impl Fn(usize) -> f64,
) -> Vec<f64> {
    // The base circuit too: the shift path of an exact fit simulates it.
    // `black_box`, so the unused value cannot let the simulation be elided.
    std::hint::black_box(expectation(sim, circuit, w));
    positions
        .iter()
        .map(|&p| {
            let plus = expectation(sim, &shifted(circuit, p, FRAC_PI_2), w);
            let minus = expectation(sim, &shifted(circuit, p, -FRAC_PI_2), w);
            (plus - minus) / 2.0
        })
        .collect()
}

fn seconds(d: Duration) -> f64 {
    d.as_secs_f64()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let qubits = parse_list(args.next(), &[12, 16, 18]);
    let rotations = parse_list(args.next(), &[8, 16, 32, 64, 128, 256]);
    let reps = args
        .next()
        .map(|k| k.trim().parse().expect("k, a positive integer"))
        .unwrap_or(DEFAULT_REPS);
    let sim = StatevectorSimulator::new();
    println!(
        "# parallel feature: {}, parallel threshold: {}, fusion (shift and forward arms): {}, \
         best of {reps}",
        cfg!(feature = "parallel"),
        sim.parallel_threshold,
        sim.fusion
    );
    println!(
        "{:>6} {:>6} {:>12} {:>12} {:>9} {:>11} {:>12} {:>10}",
        "qubits", "P", "adjoint_s", "shift_s", "speedup", "forward_s", "v2_overhead", "max_diff"
    );
    for &n in &qubits {
        let w = weight(n);
        for &p in &rotations {
            let (circuit, positions) = hea(n, p);
            let wrt: Vec<AngleRef> = positions.iter().map(|&p| AngleRef::new(p, 0)).collect();

            let (adjoint_time, adjoint) = best_of(reps, || {
                sim.adjoint_gradient(&circuit, &wrt, &w, None)
                    .expect("a differentiable circuit")
                    .derivatives
            });
            let (shift_time, shift) =
                best_of(reps, || shift_gradient(&sim, &circuit, &positions, &w));
            let (forward_time, forward) = best_of(reps, || expectation(&sim, &circuit, &w));
            std::hint::black_box(forward);

            let max_diff = adjoint
                .iter()
                .zip(&shift)
                .map(|(a, s)| (a - s).abs())
                .fold(0.0, f64::max);
            println!(
                "{:>6} {:>6} {:>12.4} {:>12.4} {:>8.1}x {:>11.4} {:>11.1}% {:>10.1e}",
                n,
                p,
                seconds(adjoint_time),
                seconds(shift_time),
                seconds(shift_time) / seconds(adjoint_time),
                seconds(forward_time),
                100.0 * seconds(forward_time) / seconds(adjoint_time),
                max_diff
            );
        }
    }
}
