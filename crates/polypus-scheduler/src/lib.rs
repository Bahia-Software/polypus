//! # polypus-scheduler
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
//! - **[`Resources`] + [`Scheduler`] + [`Flow`]** bind a backend to its planner
//!   (validating the pairing in [`Resources::new`]) and run a flow over them.
//!   [`RunCircuitFlow`] — which builds no oracle — lives here; the training flows
//!   that build oracles live in `polypus-evaluation`.
//!
//! The crate is deliberately **Python-free**: the GIL is released by the caller
//! (`polypus`, via `py.allow_threads`) and a real oracle failure reaches here as a
//! [`BoxedError`] that the edge downcasts and re-raises verbatim (plan §10.1), so
//! nothing here names `pyo3` or `PyErr`.

pub mod dispatch;
pub mod flow;
pub mod resources;

pub use dispatch::{
    dispatch_optimizer, BoxedError, DeConfig, Method, OracleError, OracleErrorSlot, PsoConfig,
    QngConfig,
};
pub use flow::{Flow, RunCircuitFlow};
pub use resources::{Resources, Scheduler};
