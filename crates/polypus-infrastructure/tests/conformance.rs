//! The conformance battery, run against Polypus's own four backends. This is the
//! Fase-5 acceptance evidence: the same battery a third party runs, applied to the
//! built-ins, so "your backend passes it" is anchored to "ours do too".
//!
//! ## What runs where — and what is a CI guard vs one-off evidence
//!
//! **Two tests are continuous CI guards.** `native_backend_conforms` is pure Rust, runs
//! on every build, and must stay fully conformant. `local_aer_backend_conformance` is
//! `#[ignore]`d because it needs Qiskit Aer and the `polypus_python` seal, which the
//! Rust job does not install; CI's Python job runs it once the wheel is installed, with
//! `POLYPUS_REQUIRE_AER=1` so that a missing Aer fails it instead of skipping it. The
//! CUNQA and QMIO tests are **`#[ignore]`d** and are *not* run by CI: they are the
//! record of a **one-off manual run** performed for this phase (their findings, below,
//! are frozen in `docs/backends.md`), not something re-verified on each build. Re-run
//! them by hand (commands on each test) when touching the relevant backend.
//!
//! - **Native** (pure Rust) — CI guard: runs the whole battery unconditionally and
//!   must be fully conformant.
//! - **Local (Aer)** — CI guard in the Python job, `#[ignore]`d elsewhere: needs Qiskit
//!   Aer + the `polypus_python` seal, and is fragile off the main thread (run with
//!   `OMP_NUM_THREADS=1`).
//! - **CUNQA** — `#[ignore]`d manual run: delegates 100% to the seal and needs a SLURM
//!   allocation even to construct; self-skips when that environment is absent.
//! - **QMIO** — `#[ignore]`d manual run (behind `--features qmio`): no simulator mode,
//!   so only its *error classification* runs, against an unreachable endpoint. It is in
//!   `qmio_conformance.rs`, alone in its binary, because it sets the `QMIO_*` tuning
//!   variables in the process environment.
//!
//! ## Findings (documented in `docs/backends.md` §"Conformance of the built-in backends")
//!
//! The behavioural checks pass for every runnable backend. The differences the phase
//! set out to surface show up in the error-classification checks:
//!
//! - **Native**: no `Unresponsive`/`Aborted`/clean-error path (a synchronous in-process
//!   simulation cannot stop responding, be signalled mid-call, or report a provider
//!   error on a valid circuit) — those three checks legitimately **skip**.
//! - **Local / CUNQA (seal-delegating)**: same absence of `Unresponsive`/`Aborted` —
//!   a dead in-process interpreter is a crash of *our* process, not a recoverable
//!   "backend stopped responding". A clean provider error *is* produced (an Aer
//!   failure → `External`). Finding **LOCAL-1** (an empty batch raised through the seal)
//!   is fixed: it returns `Ok(vec![])`. Finding **LOCAL-2** (`run_qcs` kwargs the seal
//!   never read: `backend` and `id`) is fixed: Rust no longer sends them (issue #230).
//! - **QMIO**: a `Foreign` circuit is rejected as `UnsupportedCircuit` (finding
//!   **QMIO-1**, fixed). Its wire timeout maps to `External` (definitive) rather than
//!   `Unresponsive` — a deliberate choice, since QMIO exhausts its own retries first,
//!   so the `Unresponsive` check is not applicable to it.

use std::sync::Arc;

use polypus_backend::QuantumBackend;
use polypus_backend_conformance::{Conformance, Fault, Status};
use polypus_infrastructure::{LocalBackend, NativeStatevectorBackend};

/// Set to `1` where the Aer battery must run (CI's Python job): without Aer it then
/// fails instead of skipping.
const REQUIRE_AER: &str = "POLYPUS_REQUIRE_AER";

/// Whether Qiskit Aer and the `polypus_python` seal are importable, so the Local
/// backend can actually run. Keeps a seal-less CI from failing on the Aer battery.
fn aer_available() -> bool {
    use pyo3::prelude::*;
    pyo3::Python::initialize();
    Python::attach(|py| py.import("qiskit_aer").is_ok() && py.import("polypus_python").is_ok())
}

/// Assert a named check has the expected status, with the report in the message.
fn assert_status(report: &polypus_backend_conformance::Report, name: &str, want_passed: bool) {
    let check = report
        .checks
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("check {name} not found in report:\n{report}"));
    let passed = matches!(check.status, Status::Passed);
    assert_eq!(
        passed, want_passed,
        "check {name} status {:?} (want passed={want_passed}):\n{report}",
        check.status
    );
}

fn assert_skipped(report: &polypus_backend_conformance::Report, name: &str) {
    let check = report.checks.iter().find(|c| c.name == name).unwrap();
    assert!(
        matches!(check.status, Status::Skipped(_)),
        "check {name} should skip, got {:?}:\n{report}",
        check.status
    );
}

/// The pure-Rust native statevector backend is **fully conformant**. It has no
/// `Unresponsive`/`Aborted`/clean-provider-error path (a synchronous CPU simulation
/// cannot enter them), so those three classification checks legitimately skip — and
/// nothing fails.
#[test]
fn native_backend_conforms() {
    let report = Conformance::new("native", || {
        Ok(Arc::new(NativeStatevectorBackend::new(0xABCD)) as Arc<dyn QuantumBackend>)
    })
    .shots(1000)
    .run();
    println!("\n{report}");
    report.assert_conformant();
    assert_status(&report, "foreign_circuit_rejected_as_unsupported", true);
    for name in [
        "unresponsive_backend_is_classified",
        "interrupted_call_is_aborted",
        "clean_provider_error_is_classified",
    ] {
        assert_skipped(&report, name);
    }
}

/// The Aer-backed local backend, run through the seal when Aer is present. Its
/// behavioural checks and its clean-error classification (a bogus `sim_method` makes
/// Aer raise → `External`) pass; it has no `Unresponsive`/`Aborted` path (skips).
///
/// The printed report is the evidence. Finding LOCAL-1 (an empty batch raised through
/// the seal) is fixed, so `empty_batch_returns_empty` passes too.
///
/// `#[ignore]`d: it needs Qiskit Aer and the installed `polypus_python` seal, and it
/// drives Aer's C++/OpenMP engine from a cargo-test worker thread, which is fragile
/// there (Aer segfaults when it fans experiments across cores off the main thread).
/// Run it explicitly, pinned to one OpenMP thread, as CI's Python job does:
/// ```text
/// LD_LIBRARY_PATH=$CONDA_PREFIX/lib OMP_NUM_THREADS=1 \
///   cargo test -p polypus-infrastructure --test conformance local_aer -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs Qiskit Aer and polypus_python; run with OMP_NUM_THREADS=1 (see docs/backends.md)"]
fn local_aer_backend_conformance() {
    if !aer_available() {
        assert!(
            std::env::var_os(REQUIRE_AER).is_none_or(|v| v != "1"),
            "qiskit_aer / polypus_python not importable, and {REQUIRE_AER}=1 requires them"
        );
        eprintln!("skipping: qiskit_aer / polypus_python not importable in this environment");
        return;
    }
    let report = Conformance::new("local-aer", || {
        Ok(Arc::new(LocalBackend::new(
            "statevector".to_string(),
            None,
        )) as Arc<dyn QuantumBackend>)
    })
    .shots(1000)
    // A clean, definitive provider failure: a bogus Aer simulation method makes the
    // seal raise an AerError, which crosses the contract as External. (No
    // Unresponsive/Aborted factory — the seal-delegating backend cannot produce those.)
    .fault(Fault::CleanError, || {
        Ok(Arc::new(LocalBackend::new(
            "not_a_real_sim_method".to_string(),
            None,
        )) as Arc<dyn QuantumBackend>)
    })
    .run();
    println!("\n{report}");

    // The behavioural core and the classification that Local *can* express must pass.
    for name in [
        "healthy_run_satisfies_result_contract",
        "batch_order_is_preserved",
        "sequential_planner_runs_a_multi_wave_batch",
        "shot_distribution_conserves_shots",
        "zero_shot_replica_contributes_nothing",
        "foreign_circuit_rejected_as_unsupported",
        "clean_provider_error_is_classified",
    ] {
        assert_status(&report, name, true);
    }
    // The seal-delegating backend has no Unresponsive/Aborted path.
    assert_skipped(&report, "unresponsive_backend_is_classified");
    assert_skipped(&report, "interrupted_call_is_aborted");
    // Finding LOCAL-1 (fixed): an empty batch short-circuits to `Ok(vec![])`.
    assert_status(&report, "empty_batch_returns_empty", true);
}

/// CUNQA needs a SLURM allocation *and* the seal even to construct, so it cannot run
/// in a plain CI/dev environment. This attempts construction and, on the expected
/// failure, skips explicitly rather than pretending to have exercised it.
///
/// `#[ignore]`d for the same reason as the Local diagnostic (it drives Aer through the
/// seal when a SLURM+CUNQA environment is present). Run explicitly there.
#[test]
#[ignore = "diagnostic: needs SLURM + the CUNQA seal; run explicitly (see docs/backends.md)"]
fn cunqa_backend_conformance_is_environment_gated() {
    use polypus_infrastructure::{BackendConfig, ExecutionConfig, Infrastructure, OptLevel};
    // CUNQA construction crosses the seal (Python), so the interpreter must be up.
    pyo3::Python::initialize();

    let cfg = ExecutionConfig {
        id: "conformance-cunqa".to_string(),
        shots: 128,
        n_qpus: 1,
        infrastructure: "cunqa".to_string(),
        backend_config: BackendConfig::Cunqa {
            sim_method: "statevector".to_string(),
            nodes: 1,
            cores_per_qpu: 1,
            walltime: polypus_infrastructure::DEFAULT_WALLTIME.to_string(),
        },
        opt_level: OptLevel::default(),
        seed: Some(1),
    };
    match Infrastructure::create_backend(&cfg) {
        Ok(backend) => {
            let backend = Arc::new(backend);
            let report = Conformance::new("cunqa", move || Ok(Arc::clone(&backend)))
                .shots(128)
                .run();
            println!("\n{report}");
            // Same seal-delegation profile as Local: behaviour passes, Unresponsive/
            // Aborted are N/A.
            assert_status(&report, "healthy_run_satisfies_result_contract", true);
            assert_skipped(&report, "unresponsive_backend_is_classified");
        }
        Err(e) => {
            eprintln!(
                "skipping CUNQA conformance: backend could not be constructed in this \
                 environment (expected without SLURM+seal): {e}"
            );
        }
    }
}
