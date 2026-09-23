# Writing a Polypus backend

A **backend** is what actually executes quantum circuits: the local statevector
simulator, Qiskit Aer, CUNQA, the CESGA QMIO QPU — or *your* hardware. Every
algorithm in Polypus (circuit runs, VQC/QML training) talks to a backend only
through one small, stable trait, so adding a new one never touches algorithm code.

This document is the contract for backend authors. It grows one section per
delivery phase; this first section covers **implementing the trait against the
`polypus-backend` crate** (the pure-Rust path, and the one third parties should
target).

> Status: the pyo3-free `polypus-backend` crate exists as of the backend-extraction
> work (issue #191). The runtime *registry* that lets a third-party backend be used
> without recompiling Polypus, and the subprocess bridge for Python SDKs, are
> separate later phases and will get their own sections here.

## The one crate you depend on: `polypus-backend`

`polypus-backend` is deliberately tiny and **carries no PyO3, no Qiskit and no
Python runtime**. Its whole dependency tree is `polypus-circuit` (the native
circuit IR) and `polypus-observable` (the cost-observable seam). That is the
point: you can write and ship a Polypus backend without pulling the interpreter or
any of Polypus's execution machinery into your build.

```toml
[dependencies]
polypus-backend = "0.7"   # inherits the Polypus workspace version
polypus-circuit = "0.7"   # only if you construct native circuits yourself
```

Everything you implement or name lives in this crate:

| Item | Role |
|------|------|
| `QuantumBackend` | the trait you implement |
| `BoundCircuit` | a fully bound circuit you receive (`Native`, `Qasm2`, `Foreign`) |
| `RunParams` | the per-call parameters (`id`, `shots`, `seed`, `opt_level`) |
| `Counts` | `HashMap<String, u64>` — one bitstring→count map per circuit |
| `BackendError` | how you report failure |
| `BackendCapabilities` | what a `Planner` may assume about you |
| `Transpiler` / `OptLevel` / `TranspileOptions` | optional GIL-free circuit rewriting |
| `Planner`, `SequentialPlanner`, `ShotDistributingPlanner` | how batches are run over you |

## The trait

```rust
pub trait QuantumBackend: Send + Sync {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<HashMap<String, u64>>, BackendError>;

    // Everything below has a default — override only what you need.
    fn run_shots_distributed(&self, qc: &BoundCircuit, shot_batches: &[u32], params: &RunParams)
        -> Result<Vec<HashMap<String, u64>>, BackendError> { /* replicate + forward */ }
    fn close(&self) {}
    fn cancel(&self) {}
    fn capabilities(&self) -> BackendCapabilities { /* unbounded, shot-dist supported */ }
    fn capabilities_for(&self, tasks: &[CircuitTask<'_>])
        -> Result<BackendCapabilities, InfrastructureError> { /* = capabilities() */ }
    fn default_planner(&self) -> Arc<dyn Planner> { /* SequentialPlanner */ }
}
```

**The only method you must write is `run_circuits`.** It receives a slice of bound
circuits and the per-call `RunParams`, and returns one measurement-count map per
circuit, in the same order. That is enough to be usable by every algorithm.

A complete, dependency-free example lives in
[`crates/polypus-backend/tests/third_party_backend.rs`](../crates/polypus-backend/tests/third_party_backend.rs);
it implements `run_circuits` in a dozen lines and drives it through the shipped
planners.

### Circuits you receive: `BoundCircuit`

```rust
pub enum BoundCircuit {
    Native(ConcreteCircuit),          // polypus-circuit's native IR — no GIL
    Qasm2(String),                    // an OpenQASM 2.0 program
    Foreign(Box<dyn ForeignCircuit>), // an opaque provider-native object
}
```

- **`Native`** and **`Qasm2`** are the provider-agnostic representations. A
  pure-Rust backend consumes these directly — `ConcreteCircuit` exposes the gates,
  and `Qasm2` is text you can parse or forward. Both are GIL-free.
- **`Foreign`** is an escape hatch for a provider-native object (Polypus uses it to
  carry a Qiskit `QuantumCircuit` to Aer/CUNQA without `polypus-backend` ever
  knowing what Qiskit is). If you don't ship such objects, you never produce a
  `Foreign`; if you receive one you don't recognise, reject it with
  `BackendError::UnsupportedCircuit`. `BoundCircuit::native_qubit_width()` reads a
  circuit's width GIL-free for `Native`/`Qasm2` and returns `None` for `Foreign`.

### Parameters you read: `RunParams`

```rust
pub struct RunParams { pub id: String, pub shots: u32, pub seed: Option<u64>, pub opt_level: OptLevel }
```

These are the *only* fields any backend reads per call. Provider-specific
construction inputs (a noise model, a node count) belong in your own constructor,
not here — this keeps the per-call surface minimal and pyo3-free. `seed` is
`Some` whenever a seed-consuming backend runs (contract C-7); honour it if you
sample shots yourself, and fall back to your own default on `None`.

### Reporting failure: `BackendError`

Never panic across the boundary — return a `BackendError`:

- `UnsupportedCircuit(String)` — a representation you cannot run.
- `InvalidResults(String)` — your own results violated the contract (the shipped
  `validate_run_results` checks this for you: one non-empty map per circuit, counts
  summing to the requested shots, bitstring keys).
- `NativeCircuit(String)` / `Conversion(String)` — a parse/build/convert failure.
- `Unresponsive(String)` — **your backend stopped responding mid-call** (crashed,
  was killed, or hung past a deadline). Distinct from a clean error so a caller may
  choose to retry with a fresh worker. This is the variant a subprocess/remote
  backend maps a dead or wedged worker to.
- `External(Box<dyn Error + Send + Sync>)` — **your own error type, type-erased.**
  This is how a provider error crosses the contract without `polypus-backend` naming
  it. Box whatever `std::error::Error + Send + Sync` you like; the Polypus FFI edge
  recovers it (Polypus's own Python backends box a `PyErr` here and re-raise it
  verbatim). A Rust host embedding Polypus downcasts it back to your type.

### Capabilities and wave sizing

A `Planner` runs your circuits in *waves* and asks `capabilities()` how many it may
run concurrently (`usize::MAX` = the whole batch at once; CUNQA reports `n_qpus`;
QMIO reports `1`). Override `capabilities()` only if you have a real cap. If your
cap depends on the *specific batch* (e.g. a statevector memory budget scaled by the
widest circuit), override `capabilities_for` instead and leave `capabilities()`
alone — it is a frozen seam paired against planners up front.

### Cancellation

Two mechanisms, by design:

1. **Between waves (cooperative).** A wave (one `run_circuits` call) is *atomic*.
   The `Planner` checks a `CancelToken` and runs a host-interrupt guard *between*
   waves, so a Ctrl+C or a cooperative cancel takes effect at the next wave
   boundary. You get this for free.
2. **Within a call (out-of-band).** If a single `run_circuits` call can block for a
   long time on an external resource — a subprocess worker parked in `recv()`, a
   cloud session — override `fn cancel(&self)`. It is designed to be invoked **from
   another thread** while your call is blocked, and should signal that resource so
   the blocked call unblocks and returns (typically `BackendError::Unresponsive`).
   Polling a token from inside such a call cannot work — the thread is blocked and
   never reaches a poll point — which is why this is a separate, asynchronous hook.
   The default is a no-op, correct for any backend whose calls terminate on their
   own (the four built-in backends keep the default).

   > **Not wired yet.** As of this phase **no `Planner` calls `cancel()`** — the
   > method exists on the trait so the contract is stable, but the watcher thread
   > that invokes it when a `CancelToken` flips mid-wave is added in the
   > subprocess-bridge phase (Fase 4), which is the first backend that needs it.
   > Overriding `cancel()` today is therefore harmless but has no effect until that
   > wiring lands; implement it now if you like, and it will start being called then.

## Stability commitment

`polypus-backend` is a **public contract**: the whole reason it exists is that a
third party can build against it. We treat it accordingly.

- **Versioning.** `polypus-backend` inherits the unified Polypus workspace version
  (`version.workspace = true`, currently **0.7.x**, pre-1.0). It is not versioned
  independently: a given Polypus release and its backend contract move together, so
  `polypus-backend = "0.7"` and the matching `polypus` are always compatible.
- **Pre-1.0 semantics (SemVer).** While pre-1.0, a **minor** bump (0.7 → 0.8) may
  make breaking changes to the trait or its types; **patch** bumps (0.7.1 → 0.7.2)
  will not. Pin a minor range (`polypus-backend = "0.7"`) and expect to review the
  changelog when the minor moves. We will keep additive changes additive: new trait
  methods ship with defaults (as `cancel` did) rather than as breaking additions.
- **crates.io.** `polypus-backend` is **not yet published to crates.io.** The
  contract is still settling as the registry and conformance phases (a runtime
  backend registry, a published conformance suite) land; publishing before then
  would freeze a surface we still intend to extend. Until it is published, depend on
  it by path or git. This document will be updated when it goes to the registry.

When the contract reaches 1.0, this section will state the post-1.0 SemVer guarantee
(breaking changes only on a major bump).
