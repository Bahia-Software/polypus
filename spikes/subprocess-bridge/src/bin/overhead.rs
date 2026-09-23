//! Point 2d: measure the REAL round-trip cost of the bridge protocol, and put
//! it next to a plausible QPU latency (hundreds of ms to seconds) so we can
//! judge whether the IPC is negligible in context.
//!
//! Run: `cargo run --release --bin overhead`

use std::time::Instant;

use subprocess_bridge_spike::{python_bin, worker_script, Circuit, Request, Worker};

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn summarize(label: &str, mut samples_us: Vec<f64>) {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_us.len();
    let mean = samples_us.iter().sum::<f64>() / n as f64;
    println!(
        "{label:<28} n={n:<5} mean={:>7.1}µs  p50={:>7.1}µs  p99={:>8.1}µs  max={:>8.1}µs",
        mean,
        pct(&samples_us, 0.50),
        pct(&samples_us, 0.99),
        *samples_us.last().unwrap(),
    );
}

fn bench<F: FnMut(&mut Worker)>(w: &mut Worker, iters: usize, mut f: F) -> Vec<f64> {
    // Warm up (interpreter/JIT/pipe buffers) before timing.
    for _ in 0..50 {
        f(w);
    }
    let mut out = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f(w);
        out.push(t.elapsed().as_secs_f64() * 1e6);
    }
    out
}

fn main() {
    let mut w = Worker::spawn(&python_bin(), &worker_script(), false).expect("spawn worker");
    let iters = 5000;

    // 1. Bare protocol round-trip: ping/pong, no payload of substance.
    let ping = bench(&mut w, iters, |w| {
        let r = w.call(&Request::ping("p")).expect("ping");
        debug_assert_eq!(r.op, "pong");
    });
    summarize("ping/pong roundtrip", ping.clone());

    // 2. A realistic "run" frame: a handful of small circuits, 1024 shots,
    //    zero simulated QPU time — so this is pure serialization + IPC.
    let circuits: Vec<Circuit> = (0..8)
        .map(|i| Circuit {
            qasm: format!("OPENQASM 2.0; qreg q[{}];", 2 + i % 3),
            n_qubits: 2 + (i % 3) as u32,
        })
        .collect();
    let run = bench(&mut w, iters, |w| {
        let r = w
            .call(&Request::run("r", circuits.clone(), 1024, 0))
            .expect("run");
        debug_assert_eq!(r.op, "result");
    });
    summarize("run(8 circuits, 0ms QPU)", run.clone());

    // Context: compare median IPC overhead to QPU latency floor/ceiling.
    let mut sorted = run.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_run_us = pct(&sorted, 0.50);
    println!("\n--- context ---");
    for (name, qpu_ms) in [("optimistic QPU 100ms", 100.0), ("typical QPU 1s", 1000.0)] {
        let frac = (median_run_us / 1000.0) / qpu_ms * 100.0;
        println!("run-frame IPC overhead is {:.3}% of a {} call", frac, name);
    }

    let _ = w.call(&Request::shutdown());
}
