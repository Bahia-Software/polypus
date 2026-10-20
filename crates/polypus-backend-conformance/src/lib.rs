//! # polypus-backend-conformance
//!
//! A **reusable conformance battery** for [`QuantumBackend`] implementations. Point
//! it at your backend and it exercises the documented execution semantics — error
//! classification, shot conservation, batching and waves, declared concurrency, and
//! the [`validate_run_results`](polypus_backend::validate_run_results) invariants —
//! returning a [`Report`] of every check as passed, failed, or skipped
//! (not-applicable).
//!
//! It depends only on `polypus-backend` (and `polypus-circuit`, to build the probe
//! circuits), so it is as **pyo3-free** as the contract it certifies: a third party
//! can add it as a dev-dependency next to `polypus-backend` and nothing else.
//!
//! ## What it checks
//!
//! **Behaviour** (needs only a healthy backend):
//! - results satisfy the contract validator (count, non-empty, shot conservation
//!   C-3, bitstring keys), over the native and OpenQASM 2.0 representations;
//! - batch order is preserved;
//! - an empty batch returns no maps;
//! - a batch larger than `max_concurrency` runs correctly in waves;
//! - shot distribution conserves the total, including a zero-shot replica (C-3);
//! - a backend declaring a finite `max_concurrency` runs a full wave of that size.
//!
//! **Error classification** (needs the backend driven into each fault):
//! - an unrecognised `Foreign` circuit → [`UnsupportedCircuit`](polypus_backend::BackendError::UnsupportedCircuit);
//! - a dead/hung backend → [`Unresponsive`](polypus_backend::BackendError::Unresponsive);
//! - an out-of-band cancelled call → [`Aborted`](polypus_backend::BackendError::Aborted);
//! - a clean provider failure → a definitive error (not `Unresponsive`, not `Aborted`).
//!
//! ## Fault injection is the author's job — honestly
//!
//! The battery cannot force a live QPU to hang, so the error-classification checks
//! run only against a backend **you** rig into that state, supplied through
//! [`Conformance::fault`]. A fault you do not supply is **skipped, not passed** — the
//! report always states exactly what was and was not exercised. This is deliberate:
//! a backend that genuinely cannot enter a fault (a synchronous in-process simulator
//! cannot "stop responding") should show that check as skipped, not fake a pass.
//!
//! ## Usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use polypus_backend::{BackendError, BoundCircuit, Counts, QuantumBackend, RunParams};
//! use polypus_backend_conformance::{Conformance, Fault};
//! # use std::collections::HashMap;
//! # struct MyBackend;
//! # impl MyBackend { fn new() -> Self { MyBackend } fn rigged_dead() -> Self { MyBackend } }
//! # impl QuantumBackend for MyBackend {
//! #   fn run_circuits(&self, qcs: &[BoundCircuit], p: &RunParams)
//! #     -> Result<Vec<Counts>, BackendError> {
//! #       Ok(qcs.iter().map(|_| HashMap::from([("0".to_string(), u64::from(p.shots))])).collect())
//! #   }
//! # }
//! let report = Conformance::new("my-backend", || Ok(Arc::new(MyBackend::new()) as Arc<dyn QuantumBackend>))
//!     .shots(1024)
//!     .fault(Fault::Unresponsive, || Ok(Arc::new(MyBackend::rigged_dead()) as Arc<dyn QuantumBackend>))
//!     .run();
//! println!("{report}");
//! report.assert_conformant(); // panics with the full report if any check failed
//! ```

use std::sync::Arc;
use std::time::Duration;

use polypus_backend::QuantumBackend;

mod checks;
pub mod circuits;
mod report;

pub use report::{Check, Report, Status};

/// A circuit representation the battery feeds a backend.
///
/// The behavioural result-contract check runs over every representation configured
/// via [`Conformance::representations`]; the default is both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// A native `polypus-circuit` circuit ([`BoundCircuit::Native`](polypus_backend::BoundCircuit::Native)).
    Native,
    /// An OpenQASM 2.0 program ([`BoundCircuit::Qasm2`](polypus_backend::BoundCircuit::Qasm2)).
    Qasm2,
}

/// A fault the battery drives a rigged backend into, to check its error
/// classification. Supply a factory for each fault your backend can reach via
/// [`Conformance::fault`]; the rest are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The backend has stopped responding (a dead or hung worker). Its next
    /// `run_circuits` must return
    /// [`BackendError::Unresponsive`](polypus_backend::BackendError::Unresponsive).
    Unresponsive,
    /// The backend's in-flight call will be cancelled out-of-band by the battery
    /// (which calls [`QuantumBackend::cancel`] from another thread). The rigged
    /// backend must **block long enough** to be cancelled and then return
    /// [`BackendError::Aborted`](polypus_backend::BackendError::Aborted). Size the
    /// block against [`Conformance::abort_grace`] (default 400 ms).
    Aborted,
    /// The backend will hit a clean, definitive provider failure. Its next
    /// `run_circuits` must return an error that is **neither** `Unresponsive`
    /// (retryable) **nor** `Aborted` (a cancellation) — for a provider that is
    /// [`BackendError::External`](polypus_backend::BackendError::External).
    CleanError,
}

/// A factory producing a backend instance for a check.
type Factory = Box<dyn Fn() -> Result<Arc<dyn QuantumBackend>, String>>;

/// The conformance battery: configure it against your backend, then [`run`](Self::run).
pub struct Conformance {
    backend_label: String,
    healthy: Factory,
    faults: Vec<(Fault, Factory)>,
    shots: u32,
    representations: Vec<Representation>,
    check_foreign_rejection: bool,
    abort_grace: Duration,
}

impl Conformance {
    /// Start a battery for a backend labelled `backend_label`, with `healthy` a
    /// factory that builds a **fresh, working** backend. The factory is called once
    /// per check group so a check cannot leave the backend in a state that breaks the
    /// next (e.g. a subprocess worker spent by a crash). It returns `Err(String)` if
    /// construction itself failed, which the report records rather than panicking.
    pub fn new(
        backend_label: impl Into<String>,
        healthy: impl Fn() -> Result<Arc<dyn QuantumBackend>, String> + 'static,
    ) -> Self {
        Conformance {
            backend_label: backend_label.into(),
            healthy: Box::new(healthy),
            faults: Vec::new(),
            shots: 1024,
            representations: vec![Representation::Native, Representation::Qasm2],
            check_foreign_rejection: true,
            abort_grace: Duration::from_millis(400),
        }
    }

    /// Shots per circuit for the battery's runs (default 1024). Must divide sensibly
    /// across three replicas for the shot-distribution checks; any value ≥ 3 is fine.
    pub fn shots(mut self, shots: u32) -> Self {
        self.shots = shots;
        self
    }

    /// Which circuit representations to feed (default: both `Native` and `Qasm2`).
    /// Restrict this if your backend accepts only one.
    pub fn representations(mut self, representations: &[Representation]) -> Self {
        self.representations = representations.to_vec();
        self
    }

    /// Whether to run the foreign-circuit rejection check (default `true`). Set
    /// `false` only for a backend that legitimately accepts arbitrary `Foreign`
    /// objects — unusual, since a `Foreign` cannot cross a process/wire boundary.
    pub fn check_foreign_rejection(mut self, yes: bool) -> Self {
        self.check_foreign_rejection = yes;
        self
    }

    /// How long to let an [`Aborted`](Fault::Aborted)-rigged call get in flight
    /// before cancelling it (default 400 ms). The rigged backend must block longer
    /// than this.
    pub fn abort_grace(mut self, grace: Duration) -> Self {
        self.abort_grace = grace;
        self
    }

    /// Register a factory that builds a backend rigged to hit `fault` on its next
    /// run. Without it, the matching error-classification check is skipped.
    pub fn fault(
        mut self,
        fault: Fault,
        factory: impl Fn() -> Result<Arc<dyn QuantumBackend>, String> + 'static,
    ) -> Self {
        self.faults.push((fault, Box::new(factory)));
        self
    }

    fn fault_factory(&self, fault: Fault) -> Option<&Factory> {
        self.faults
            .iter()
            .find(|(f, _)| *f == fault)
            .map(|(_, factory)| factory)
    }

    /// Run the whole battery and collect the [`Report`]. Never panics: a failure to
    /// even build the backend, or any contract violation, is recorded as a check
    /// result.
    pub fn run(&self) -> Report {
        let mut checks = Vec::new();
        self.push_behavioural_checks(&mut checks);
        self.push_classification_checks(&mut checks);
        Report {
            backend: self.backend_label.clone(),
            checks,
        }
    }

    /// Run **only** the error-classification checks (foreign rejection plus the
    /// supplied fault factories), skipping the behavioural group entirely.
    ///
    /// For a backend whose *healthy* instance needs real hardware — QMIO has no
    /// simulator mode — the behavioural checks cannot run in CI, but the error
    /// classification still can, against a mock or unreachable endpoint. This runs
    /// exactly that subset, so a hardware backend can still certify how it maps
    /// failures without a live device.
    pub fn run_error_classification(&self) -> Report {
        let mut checks = Vec::new();
        self.push_classification_checks(&mut checks);
        Report {
            backend: self.backend_label.clone(),
            checks,
        }
    }

    /// The behavioural group: one fresh healthy backend runs the result-contract,
    /// ordering, empty-batch, multi-wave, shot-distribution and concurrency checks.
    fn push_behavioural_checks(&self, checks: &mut Vec<Check>) {
        match (self.healthy)() {
            Ok(backend) => {
                checks.push(checks::result_contract(
                    backend.as_ref(),
                    self.shots,
                    &self.representations,
                ));
                checks.push(checks::order_preserved(backend.as_ref(), self.shots));
                checks.push(checks::empty_batch(backend.as_ref(), self.shots));
                checks.push(checks::multi_wave_batch(backend.as_ref(), self.shots));
                checks.push(checks::shot_distribution(backend.as_ref(), self.shots));
                checks.push(checks::zero_shot_replica(backend.as_ref(), self.shots));
                checks.push(checks::declared_concurrency(backend.as_ref(), self.shots));
            }
            Err(e) => checks.push(Check::failed(
                "healthy_backend_builds",
                "the healthy backend factory constructs a backend",
                format!("could not build the backend under test: {e}"),
            )),
        }
    }

    /// The error-classification group: foreign-circuit rejection plus the three
    /// fault checks (each run against a rigged backend, or skipped if none supplied).
    fn push_classification_checks(&self, checks: &mut Vec<Check>) {
        // Foreign-circuit rejection: run against a fresh healthy backend.
        if self.check_foreign_rejection {
            match (self.healthy)() {
                Ok(backend) => checks.push(checks::rejects_foreign(backend.as_ref(), self.shots)),
                Err(e) => checks.push(Check::failed(
                    "foreign_circuit_rejected_as_unsupported",
                    "an unrecognised Foreign circuit yields BackendError::UnsupportedCircuit",
                    format!("could not build the backend under test: {e}"),
                )),
            }
        }

        checks.push(self.run_fault_check(Fault::Unresponsive, |b| {
            checks::unresponsive(b.as_ref(), self.shots)
        }));
        checks.push(self.run_fault_check(Fault::CleanError, |b| {
            checks::clean_error(b.as_ref(), self.shots)
        }));
        // Aborted needs the Arc itself (it moves a clone into a worker thread).
        checks.push(match self.fault_factory(Fault::Aborted) {
            Some(factory) => match factory() {
                Ok(backend) => checks::aborted(backend, self.shots, self.abort_grace),
                Err(e) => Check::failed(
                    "interrupted_call_is_aborted",
                    "an in-flight call cancelled out-of-band yields BackendError::Aborted",
                    format!("could not build the rigged backend: {e}"),
                ),
            },
            None => Check::skipped(
                "interrupted_call_is_aborted",
                "an in-flight call cancelled out-of-band yields BackendError::Aborted",
                "no Fault::Aborted factory supplied (backend cannot be cancelled to Aborted)",
            ),
        });
    }

    /// Run a fault check that needs only a `&dyn QuantumBackend`, building the rigged
    /// backend or skipping if no factory was supplied.
    fn run_fault_check(
        &self,
        fault: Fault,
        check: impl Fn(Arc<dyn QuantumBackend>) -> Check,
    ) -> Check {
        let (name, desc) = fault_check_meta(fault);
        match self.fault_factory(fault) {
            Some(factory) => match factory() {
                Ok(backend) => check(backend),
                Err(e) => Check::failed(
                    name,
                    desc,
                    format!("could not build the rigged backend: {e}"),
                ),
            },
            None => Check::skipped(
                name,
                desc,
                "no fault factory supplied for this classification (declared not applicable)",
            ),
        }
    }
}

/// The `(name, description)` a fault's check reports under when skipped or failed at
/// the build step, kept in sync with the check functions.
fn fault_check_meta(fault: Fault) -> (&'static str, &'static str) {
    match fault {
        Fault::Unresponsive => (
            "unresponsive_backend_is_classified",
            "a dead/hung backend yields BackendError::Unresponsive (retryable)",
        ),
        Fault::Aborted => (
            "interrupted_call_is_aborted",
            "an in-flight call cancelled out-of-band yields BackendError::Aborted",
        ),
        Fault::CleanError => (
            "clean_provider_error_is_classified",
            "a clean provider failure is a definitive error, not Unresponsive and not Aborted",
        ),
    }
}
