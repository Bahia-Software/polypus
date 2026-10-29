//! # polypus-orchestration
//!
//! **How a flow is orchestrated (policy)** — the pure, `pyo3`-free middle of the
//! Polypus workspace. It depends on `polypus-infrastructure` (the backends and
//! the `Planner` — *how a circuit executes*) and `polypus-optimizers` (the
//! variational optimizers), and it is depended on by `polypus-evaluation` and the
//! `polypus` edge.
//!
//! Two responsibilities live here:
//!
//! - **[`dispatch_optimizer`]** turns a chosen [`Method`] plus an
//!   [`EvaluationOracle`](polypus_optimizers::EvaluationOracle) into an
//!   optimization outcome. An oracle failure is recorded — **type-erased** — in an
//!   [`OracleErrorSlot`] and surfaced ahead of the optimizer's own config error.
//!   [`dispatch_optimizer_observed`] also shows a [`TrainObserver`] every
//!   iteration, which can stop the run.
//! - **[`Resources`] + [`Scheduler`] + [`Flow`]** bind a backend to its planner
//!   (validating the pairing in [`Resources::new`]) and run a flow over them.
//!   **Every flow lives here** — [`RunCircuitFlow`] and training ([`TrainFlow`])
//!   alike; a training flow's `pyo3`-touching oracle is built by an
//!   [`OracleFactory`] implemented in `polypus-evaluation`, so the *order* stays
//!   with the scheduler while its domain detail does not.
//!
//! The crate is deliberately **Python-free**: the GIL is released by the caller
//! (`polypus`, via `py.detach`) and a real oracle failure reaches here as a
//! [`BoxedError`] that the edge downcasts and re-raises with its original class
//! (plan §10.1; the edge's one exception, for failures of the circuit SDK behind
//! the Python backends, is contract C-1's), so nothing here names `pyo3` or
//! `PyErr`.

pub mod dispatch;
mod final_batch;
pub mod flow;
pub mod progress;
pub mod scheduler;
#[cfg(test)]
mod test_log;
mod walltime;

pub use dispatch::{
    dispatch_optimizer, dispatch_optimizer_observed, AdamConfig, BoxedError, CmaesConfig, DeConfig,
    ExternalConfig, Method, OracleError, OracleErrorSlot, Oracles, PsoConfig, QngConfig,
    RotosolveConfig, SpsaConfig,
};
pub use flow::{FinalEvaluation, Flow, OracleFactory, RunCircuitFlow, TrainFlow, TrainOutcome};
pub use progress::{TrainObserver, TrainProgress};
pub use scheduler::{Resources, Scheduler};
