//! Point 2c helper: spawn a worker that will OUTLIVE this parent, print its
//! PID, then leave without cleaning it up. Two knobs:
//!
//!   argv[1] = "guard"   -> arm PR_SET_PDEATHSIG (our own orphan guard)
//!             "no-guard" -> do NOT arm it; rely on the environment (e.g.
//!                           SLURM's proctrack/cgroup) to reap the orphan
//!
//! We `mem::forget` the Worker so its Drop (which would kill the child) does
//! NOT run — this deliberately creates a real orphan so we can see who cleans
//! it up. The worker is told to "run" for 120s, so it stays alive on its own.
//!
//! Prints exactly one line to stdout: `WORKER_PID=<pid>`.

use std::mem;

use subprocess_bridge_spike::{python_bin, worker_script, Circuit, Request, Worker};

fn main() {
    let guard = std::env::args().nth(1).as_deref() == Some("guard");
    let mut w = Worker::spawn(&python_bin(), &worker_script(), guard).expect("spawn worker");
    let pid = w.pid();

    // Kick off a long "QPU" call so the worker is busy and will not exit on its
    // own for a while. We do not read the reply.
    w.send(&Request::run(
        "orphan",
        vec![Circuit {
            qasm: "x".into(),
            n_qubits: 1,
        }],
        1,
        120_000,
    ))
    .expect("send long run");

    println!("WORKER_PID={pid}");
    // Leak the handle: skip Drop so the parent does not kill the child. The
    // whole point is to leave a running orphan behind.
    mem::forget(w);
}
