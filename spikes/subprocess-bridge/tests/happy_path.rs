//! Point 1: the bridge round-trips "run these circuits with these shots" and
//! gets counts back from a real Python subprocess.

use subprocess_bridge_spike::{python_bin, worker_script, Circuit, Request, Worker};

#[test]
fn run_circuits_returns_counts() {
    let mut w = Worker::spawn(&python_bin(), &worker_script(), false).expect("spawn worker");

    let circuits = vec![
        Circuit {
            qasm: "OPENQASM 2.0; qreg q[2];".into(),
            n_qubits: 2,
        },
        Circuit {
            qasm: "OPENQASM 2.0; qreg q[3];".into(),
            n_qubits: 3,
        },
    ];
    let resp = w
        .call(&Request::run("req-1", circuits, 1024, 0))
        .expect("call ok");

    assert_eq!(resp.op, "result");
    assert_eq!(resp.id.as_deref(), Some("req-1"));
    let counts = resp.counts.expect("counts present");
    assert_eq!(counts.len(), 2, "one histogram per circuit");

    // First circuit is 2 qubits: keys "00" and "11", summing to shots.
    let c0 = &counts[0];
    assert_eq!(c0.get("00").and_then(|v| v.as_u64()), Some(512));
    assert_eq!(c0.get("11").and_then(|v| v.as_u64()), Some(512));

    // Worker is reusable for a second call on the same process.
    let resp2 = w
        .call(&Request::run(
            "req-2",
            vec![Circuit {
                qasm: "x".into(),
                n_qubits: 1,
            }],
            100,
            0,
        ))
        .expect("second call ok");
    let c = &resp2.counts.unwrap()[0];
    assert_eq!(c.get("0").and_then(|v| v.as_u64()), Some(50));
    assert_eq!(c.get("1").and_then(|v| v.as_u64()), Some(50));
}
