//! Measure the bridge's IPC round-trip overhead for **large, realistic counts
//! payloads** (many shots × many circuits × many distinct bitstrings) — closing the
//! Fase-2 open risk that the ~15 µs overhead was measured only with a minimal
//! payload.
//!
//! ```bash
//! cargo run -p polypus-subprocess-backend --release --bin payload_overhead
//! # override: CIRCUITS=200 KEYS=4096 SHOTS=100000 ITERS=30 cargo run ... --bin payload_overhead
//! ```
//!
//! It reports per-call wall time (p50/p99) and the decoded payload size, so the
//! numbers can be compared against a realistic QPU latency (100 ms – seconds). This
//! is a diagnostic binary, not a test.

use std::time::{Duration, Instant};

use polypus_backend::{BoundCircuit, OptLevel, QuantumBackend, RunParams};
use polypus_subprocess_backend::{SubprocessBackend, SubprocessConfig};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn resolve_python() -> Option<String> {
    std::env::var("POLYPUS_BRIDGE_PYTHON")
        .into_iter()
        .chain(["python3".to_string(), "python".to_string()])
        .find(|cand| {
            std::process::Command::new(cand)
                .arg("-c")
                .arg("pass")
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
}

fn main() {
    let circuits_n = env_usize("CIRCUITS", 100);
    let keys = env_usize("KEYS", 1024);
    let key_width = env_usize("KEY_WIDTH", 24);
    let shots = env_usize("SHOTS", 100_000) as u32;
    let iters = env_usize("ITERS", 20);

    let python = match resolve_python() {
        Some(p) => p,
        None => {
            eprintln!("no Python interpreter found (set POLYPUS_BRIDGE_PYTHON); skipping");
            return;
        }
    };
    let worker_script =
        concat!(env!("CARGO_MANIFEST_DIR"), "/python/worker_template.py").to_string();

    let config = SubprocessConfig {
        command: vec![python, worker_script],
        cwd: None,
        env: vec![
            ("WORKER_SPREAD_KEYS".to_string(), keys.to_string()),
            ("WORKER_KEY_WIDTH".to_string(), key_width.to_string()),
        ],
        recv_timeout: Duration::from_secs(120),
        arm_pdeathsig: true,
    };
    let backend = SubprocessBackend::spawn(config).expect("worker spawns and handshakes");

    // A batch of QASM2 circuits — content is irrelevant here (the worker fabricates
    // the large counts payload); what we measure is the framing + serde round trip
    // for a payload of `circuits_n * keys` count entries.
    let circuits: Vec<BoundCircuit> = (0..circuits_n)
        .map(|_| BoundCircuit::Qasm2("OPENQASM 2.0;".to_string()))
        .collect();
    let params = RunParams {
        id: "payload-overhead".to_string(),
        shots,
        seed: Some(1),
        opt_level: OptLevel::default(),
    };

    // Warm up (spawn/JIT/allocator effects out of the sample).
    let warm = backend
        .run_circuits(&circuits, &params)
        .expect("warmup run");
    let entries: usize = warm.iter().map(|m| m.len()).sum();
    let approx_bytes: usize = warm
        .iter()
        .flat_map(|m| m.keys())
        .map(|k| k.len() + 8)
        .sum();

    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let out = backend.run_circuits(&circuits, &params).expect("timed run");
        samples.push(start.elapsed());
        std::hint::black_box(out);
    }
    samples.sort();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99 / 100).min(samples.len() - 1)];

    backend.close();

    println!("subprocess bridge — large-payload IPC round-trip");
    println!("  circuits/call : {circuits_n}");
    println!("  keys/circuit  : {keys} (width {key_width})");
    println!("  shots/circuit : {shots}");
    println!("  count entries : {entries}");
    println!(
        "  payload (approx decoded key bytes): {} KiB",
        approx_bytes / 1024
    );
    println!("  iterations    : {iters}");
    println!("  round-trip p50: {p50:?}");
    println!("  round-trip p99: {p99:?}");
    println!(
        "  vs a 100 ms QPU call: p50 is {:.3}% of one shot-batch's latency",
        p50.as_secs_f64() / 0.100 * 100.0
    );
}
