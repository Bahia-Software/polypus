//! The orphan guard's actual purpose: when the host *process* dies hard, the worker
//! dies with it. The host is this very test binary re-executed in a helper role; the
//! test SIGKILLs it and checks whether the worker it had spawned survives. This guards
//! against the spawner-thread design silently breaking `PR_SET_PDEATHSIG` for real
//! process death.
//!
//! The worker here is deliberately *not* the reference one: that one exits by itself on
//! stdin EOF (which a dead host also causes), so it would die with or without the kernel
//! guard and prove nothing. This worker handshakes and then ignores EOF, so only
//! `PR_SET_PDEATHSIG` can end it. The disarmed run is the control that shows it.

#![cfg(target_os = "linux")]

mod common;

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use polypus_subprocess_backend::protocol::PROTOCOL_VERSION;
use polypus_subprocess_backend::SubprocessBackend;

const HOST_ROLE_ENV: &str = "POLYPUS_PROCESS_DEATH_HOST_ARM";

/// Handshake, log `[worker <pid>] started` to stderr, then sleep ignoring stdin EOF.
fn stubborn_worker_script() -> String {
    format!(
        r#"
import json, os, struct, sys, time
header = sys.stdin.buffer.read(4)
sys.stdin.buffer.read(struct.unpack('<I', header)[0])
out = json.dumps({{"op": "ready", "protocol": {PROTOCOL_VERSION}}}).encode()
sys.stdout.buffer.write(struct.pack('<I', len(out)) + out)
sys.stdout.buffer.flush()
print(f"[worker {{os.getpid()}}] started", file=sys.stderr, flush=True)
time.sleep(600)
"#
    )
}

/// Helper role: build a backend over the stubborn worker, then stay alive until the
/// parent test kills us.
fn run_as_host(arm_pdeathsig: bool) {
    let python = common::resolve_python().expect("the parent test resolved a Python already");
    let mut config = common::config(&python, 30_000, &[]);
    config.command = vec![python, "-c".to_string(), stubborn_worker_script()];
    config.arm_pdeathsig = arm_pdeathsig;
    let _backend = SubprocessBackend::spawn(config).expect("worker spawns and handshakes");
    thread::sleep(Duration::from_secs(120));
}

/// `true` once `/proc/<pid>` is gone or the process is a zombie (dead, not yet reaped).
fn is_dead(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => true,
        // `pid (comm) S ...`: the state is the first field after the last ')'.
        Ok(stat) => stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .is_some_and(|state| state == "Z" || state == "X"),
    }
}

/// Run the scenario from the parent test: spawn the host (re-executing this binary as
/// test `test_name`), SIGKILL it once its worker is up, and report whether the worker
/// died within the grace period. A surviving worker is killed before returning.
/// `None` means the test was skipped (no Python).
fn worker_dies_with_killed_host(
    test_name: &str,
    arm_pdeathsig: bool,
    grace: Duration,
) -> Option<bool> {
    common::resolve_python()?;

    let mut host = Command::new(std::env::current_exe().expect("path of the test binary"))
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(HOST_ROLE_ENV, if arm_pdeathsig { "1" } else { "0" })
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("re-execute the test binary as the host");

    // Learn the worker's pid from its startup line on the host's stderr. The reader
    // keeps draining afterwards so the host can never block on a full pipe.
    let stderr = host.stderr.take().expect("stderr piped");
    let (tx, rx) = mpsc::channel::<u32>();
    thread::spawn(move || {
        let mut tx = Some(tx);
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let pid = line
                .strip_prefix("[worker ")
                .and_then(|rest| rest.split_once("] started"))
                .and_then(|(pid, _)| pid.parse::<u32>().ok());
            if let (Some(pid), Some(tx)) = (pid, tx.take()) {
                let _ = tx.send(pid);
            }
        }
    });
    let worker_pid = match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(pid) => pid,
        Err(e) => {
            let _ = host.kill();
            let _ = host.wait();
            panic!("the host never reported its worker's pid: {e}");
        }
    };
    assert!(
        !is_dead(worker_pid),
        "the worker should be alive before the kill"
    );

    // Hard-kill the host: no `Drop`, no `close()` — only the kernel guard can act.
    host.kill().expect("SIGKILL the host");
    host.wait().expect("reap the host");

    let deadline = Instant::now() + grace;
    while !is_dead(worker_pid) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(25));
    }
    let dead = is_dead(worker_pid);
    if !dead {
        // Do not leave a stray worker behind.
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(worker_pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    Some(dead)
}

fn host_arm_from_env() -> Option<bool> {
    std::env::var(HOST_ROLE_ENV).ok().map(|v| v == "1")
}

#[test]
fn worker_dies_when_the_host_process_is_killed() {
    if let Some(arm) = host_arm_from_env() {
        run_as_host(arm);
        return;
    }
    match worker_dies_with_killed_host(
        "worker_dies_when_the_host_process_is_killed",
        true,
        Duration::from_secs(10),
    ) {
        None => eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)"),
        Some(dead) => assert!(
            dead,
            "the worker outlived its SIGKILLed host: the orphan guard did not fire"
        ),
    }
}

/// Control for the test above: with the guard disarmed the same scenario leaves the
/// worker alive, so the armed test passing is down to `PR_SET_PDEATHSIG`, not to the
/// worker exiting on its own.
#[test]
fn control_worker_survives_a_killed_host_when_the_guard_is_disarmed() {
    if let Some(arm) = host_arm_from_env() {
        run_as_host(arm);
        return;
    }
    match worker_dies_with_killed_host(
        "control_worker_survives_a_killed_host_when_the_guard_is_disarmed",
        false,
        Duration::from_secs(1),
    ) {
        None => eprintln!("skipping: no Python interpreter found (set POLYPUS_BRIDGE_PYTHON)"),
        Some(dead) => assert!(
            !dead,
            "the stubborn worker died without the guard: this test no longer proves anything"
        ),
    }
}
