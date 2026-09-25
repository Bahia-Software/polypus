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
> work (issue #191). The runtime **registry** (use a backend by name without editing
> Polypus) and the **subprocess bridge** for Python SDKs landed next (issue #194) and
> have their own sections below: [Registering a backend from Rust](#registering-a-backend-from-rust)
> and [The Python subprocess bridge](#the-python-subprocess-bridge). Finally, two
> copy-me [reference templates](#reference-templates) and a
> [conformance battery](#testing-your-backend-the-conformance-battery) (issue #199) let
> you write a backend and verify it against this contract.

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
- `Aborted(String)` — **your call was aborted by an external signal** (a terminal
  Ctrl+C, say) and should be treated as a *cancellation*. If your backend can detect
  that its in-flight call was interrupted, return this and you get the same
  consistent classification as Polypus's own backends: the FFI edge raises
  `KeyboardInterrupt`, exactly like a between-wave cooperative cancel. Distinct from
  `Unresponsive` (a failure) — this is the expected end of an interrupted run.
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

Three things interact, by design:

1. **Between waves (cooperative).** A wave (one `run_circuits` call) is *atomic*.
   The `Planner` checks a `CancelToken` and runs a host-interrupt guard *between*
   waves, so a Ctrl+C or a cooperative cancel takes effect at the next wave
   boundary. You get this for free.
2. **A call aborted by a signal → `BackendError::Aborted`.** If your backend can
   tell that its in-flight call was interrupted by an external signal — the clearest
   case is a subprocess worker that catches a terminal Ctrl+C delivered to the
   process group and stops — return
   [`BackendError::Aborted(msg)`](#reporting-failure-backenderror). It is classified
   as a *cancellation whatever its source*: the FFI edge raises `KeyboardInterrupt`
   from it, exactly as it does for a between-wave cooperative cancel. This is what
   makes an interrupt observed *inside* a blocked call surface to Python identically
   to one observed between waves.
3. **Programmatic abort of a blocked call — `fn cancel(&self)`.** If a single
   `run_circuits` call can block for a long time on an external resource (a
   subprocess worker parked in `recv()`, a cloud session), override `fn cancel(&self)`
   *and* `fn wants_cancel_watcher(&self) -> bool { true }`. `cancel` is invoked **from
   another thread** while your call is blocked, and should signal that resource so the
   blocked call unblocks and returns (typically `Aborted`). Polling a token from
   inside the blocked call cannot work — the thread never reaches a poll point — which
   is why this is a separate, asynchronous hook. Both default to no-op/`false`, so the
   four built-in backends spawn no watcher thread and are unchanged.

   > **What triggers `cancel()` today.** The watcher lives in
   > `Scheduler::run_cancellable` and fires when the run's `CancelToken` is
   > **explicitly cancelled** — i.e. someone holding the token calls `.cancel()` from
   > another thread. That is a **programmatic** path, available to a Rust embedder.
   > **Polypus does not currently expose a cancellation handle from Python**, so a
   > `pip install polypus` user cannot trigger this path; it is groundwork for a
   > future Python cancel API, not a user-facing feature of this phase.
   >
   > The watcher deliberately does **not** try to detect a raw Ctrl+C itself: the
   > host-interrupt guard is CPython's `PyErr_CheckSignals`, which is a guaranteed
   > no-op on any non-main thread, and the watcher runs on a spawned thread. A
   > terminal Ctrl+C instead reaches a subprocess backend directly through the OS
   > process group (mechanism 2 above) and comes back as `Aborted` →
   > `KeyboardInterrupt`.

## Registering a backend from Rust

Implementing `QuantumBackend` makes your backend *usable*; the **runtime registry**
makes Polypus able to build it **by name**, so a backend that lives entirely outside
this repository can be selected exactly like a built-in — without editing any
`match` in Polypus.

This is the story for a **Rust embedder**: someone who compiles their own binary
that links Polypus. You register your backend once at startup:

```rust
use std::sync::Arc;
use polypus_backend::{register_backend, BackendBuildContext, BackendError, QuantumBackend};

// Your backend implements QuantumBackend (see above).
register_backend("acme-qpu", |ctx: &BackendBuildContext| {
    let endpoint = ctx.option("endpoint").unwrap_or("tcp://localhost:9000");
    Ok(Arc::new(AcmeBackend::connect(endpoint)?) as Arc<dyn QuantumBackend>)
});
```

After that, any name-driven construction path builds it — including Polypus's own
`Infrastructure::create_backend`, which routes a name it does not recognise to the
registry. From Python that means `infrastructure="acme-qpu"` (see the `options`
kwarg below) just works.

**What the factory receives.** A `BackendBuildContext` carries the pyo3-free,
provider-agnostic construction inputs — `id`, `shots`, `n_qpus`, `seed`,
`opt_level` — plus an `options: HashMap<String, String>` bag for whatever
provider-specific configuration you need (an endpoint, credentials env names, a
device id). Strings keep the registry free of any provider-type coupling; read what
you need with `ctx.option("key")` and ignore the rest. An unknown name yields
`BackendError::UnknownInfrastructure` (a clean `ValueError` at the FFI edge), never a
panic or a silent fallthrough.

**Last registration wins.** Re-registering a name replaces the previous factory, so
an embedder can deliberately override a built-in (e.g. swap Polypus's `"subprocess"`
bridge for their own).

> **This avoids editing our repository, but not recompiling *yours*.** Rust has no
> runtime plugin loading here: your backend is compiled into your binary. The
> registry removes the need to add a `match` arm to Polypus; it does not remove the
> `cargo build` of the binary that embeds it. That is the deliberate scope of the
> Rust story.

### Why an explicit call (and not `inventory`/`linkme`)

We chose an explicit `register_backend(...)` call over link-time static collection
(the `inventory` / `linkme` crates, where a backend self-registers via a `static`
that a collector gathers with no explicit call). The trade-offs we weighed:

| | Explicit `register_backend` (chosen) | `inventory`/`linkme` |
|---|---|---|
| Startup wiring | one call before use (an ordering requirement) | none — linking is enough |
| Failure mode | a missed call → clear `UnknownInfrastructure` | a static stripped by `--gc-sections`/LTO → "compiles but not found", hard to debug |
| Magic / portability | none; plain Rust, all platforms | linker-section machinery, platform quirks, an extra dependency |
| `#[cfg(feature)]` backends | composes cleanly | interacts awkwardly |

For "I compile my own binary", an explicit call in `main`/setup is idiomatic and has
no surprising failure modes, so it was the better default. (If you *want*
auto-registration, you can still layer `inventory` on top in your own crate and have
its collected entries call `register_backend` — Polypus does not need to know.)

## The Python subprocess bridge

The registry's Rust story needs a Rust backend. The **majority case is Python**: a
provider ships a Python SDK, and the user did `pip install polypus`. The
`polypus-subprocess-backend` crate is the answer — a built-in backend, registered
under the name **`"subprocess"`**, that runs the provider's SDK **in its own
subprocess** and talks to it over a small, versioned protocol. The interpreter runs
in the child, never embedded in ours: the bridge crate is **pyo3-free** (enforced in
CI). Use it when a provider only offers a Python SDK with no documented wire
protocol; if the provider *does* expose a documented network protocol, a pure-Rust
backend against that protocol (like QMIO) is preferable.

### Using it from Python

```python
import polypus

polypus.run_quantum_circuit(
    qc,
    shots=1024,
    infrastructure="subprocess",
    options={
        # Required: the worker command (argv, split on whitespace).
        "command": "python3 /path/to/my_worker.py",
        # Optional:
        "recv_timeout_ms": "600000",  # read timeout; default 300000 (5 min)
        "arm_pdeathsig": "true",  # orphan guard (Linux); default true
        "cwd": "/path/to/workdir",  # worker working directory
    },
)
```

The `options` dict is the same one every entry point (`run_quantum_circuit`,
`train`, `qml.train`) now accepts, and is how any registered backend receives its
configuration.

### Writing the worker

Copy [`crates/polypus-subprocess-backend/python/worker_template.py`](../crates/polypus-subprocess-backend/python/worker_template.py)
and replace one function, `execute_circuits(circuits, shots, seed)`, with a call
into your SDK. Everything else — the framing, the handshake, the SIGINT-driven abort
— is the stable contract and should be left as-is.

**Protocol, version 1.** Each message is one frame:

```text
frame = u32 little-endian length  ||  UTF-8 JSON payload
```

stdout carries protocol frames **only** (log to stderr — mixing corrupts the
stream). On spawn the bridge sends `{"op":"hello","protocol":1}` and the worker must
answer `{"op":"ready","protocol":1}`; a version mismatch is refused. Then, per run:

| direction | frame |
|---|---|
| bridge → worker | `{"op":"run","id","circuits":[{"qasm","n_qubits"},…],"shots","seed"?}` |
| worker → bridge | `{"op":"result","id","counts":[{bitstring:count,…},…]}` |
| worker → bridge | `{"op":"aborted","id","reason"}` — an in-flight run was cancelled |
| worker → bridge | `{"op":"error","id","message"}` — a clean, in-band failure |
| bridge → worker | `{"op":"shutdown"}` |

Circuits arrive as **OpenQASM 2.0** text (a provider-native `Foreign` object cannot
cross a process boundary and is rejected before it reaches the worker). Return one
counts map per circuit, in order.

### Liveness: how a hung worker is detected

Every read is bounded by `recv_timeout` (option `recv_timeout_ms`, default 5 min). A
worker that is *dead* is detected immediately as EOF; a worker that is *alive but
wedged* — a deadlocked SDK, a mute QPU that never replies — trips the timeout. **Both
map to `BackendError::Unresponsive`** (a `polypus.BackendError` at the edge), never
an infinite block.

> **Sizing the timeout.** It is a single knob serving two ends: it must exceed the
> longest a legitimate call can take, yet a genuine hang is only detected once it
> elapses. Set it from your workload — a fast simulator can use seconds; a real QPU
> queue may need many minutes. (We deliberately chose a plain read timeout over a
> heartbeat: for an opaque SDK call, a worker-side heartbeat thread keeps beating
> even while the SDK blocks on a mute QPU, so it would *not* detect that failure,
> whereas the timeout does.)

### Orphans, and the `PR_SET_PDEATHSIG` caveat

The worker is spawned with `PR_SET_PDEATHSIG` armed, so the kernel kills it if the
Rust process dies — no orphaned workers. **This is Linux-only** (`prctl(2)`). It is
sufficient for CESGA (all Linux). On a non-Linux host the guard is simply not in
effect; the portable fallback is the bridge's own teardown — `close()` and the
`Worker`'s `Drop` kill and reap the child — which covers a clean exit but not a hard
crash of the parent. If you run this off Linux and need crash-proof orphan cleanup,
wrap the run in your platform's job-object / process-group equivalent.

### Running under SLURM

The worker is an **ordinary child process inside your job step**: it shares the
job's cgroup, so it shares the step's `--mem` and CPU allocation with the Rust
process. There is no separate reservation. Therefore:

- **Request resources for both.** Size `--mem` and `--cpus-per-task` to cover the
  Rust process *and* the worker's SDK (Qiskit/CUDA/etc. can be memory-hungry). A
  worker OOM-killed by the cgroup surfaces as `Unresponsive` (EOF), so under-sizing
  looks like a flaky backend.
- **Always pass them explicitly.** Submit with explicit `--cpus-per-task` and
  `--mem`; do not rely on defaults. When the job ends, `proctrack/cgroup` reaps the
  worker with the step, so no orphan leaks even without the `PR_SET_PDEATHSIG` guard.

### IPC overhead

The Fase-2 spike measured the round trip at ~15 µs with a *minimal* payload. This
phase measured a deliberately **large, realistic** payload —
`cargo run -p polypus-subprocess-backend --release --bin payload_overhead` — with
100 circuits × 1024 distinct bitstrings × 100 000 shots (≈102 400 count entries,
≈3.2 MiB decoded): **p50 ≈ 42 ms** end-to-end (dominated by the Python worker
building and serialising the dict, not the Rust framing). That is well under a real
QPU's hundreds-of-ms-to-seconds latency, and realistic counts (sparse, far fewer
distinct bitstrings) are far smaller. JSON framing is not the bottleneck.

### Cancellation

Two paths reach the worker, and both end the run as `KeyboardInterrupt`:

- **Terminal Ctrl+C (the everyday case).** A SIGINT from the terminal is delivered
  to the whole foreground **process group**, so it reaches the worker directly,
  independently of any `CancelToken`. The worker's handler aborts the in-flight run
  and replies `aborted`; the bridge maps that to `BackendError::Aborted`, which the
  FFI edge raises as `KeyboardInterrupt`. This works for a `pip install polypus`
  user today.
- **Programmatic cancel (Rust embedder / future work).** `SubprocessBackend` sets
  `wants_cancel_watcher() == true`, so a cancellable run spawns the planner watcher
  (see [Cancellation](#cancellation) above). If someone holding the run's
  `CancelToken` calls `.cancel()` from another thread, the watcher calls `cancel()`,
  which sends the worker `SIGINT` via `nix::sys::signal::kill` (no `unsafe`); the
  worker aborts, replies `aborted`, and stays alive and reusable, and the run ends as
  `Cancelled` (→ `KeyboardInterrupt`). **Polypus does not yet expose a cancellation
  handle from Python**, so this path is only reachable from Rust today.

## Reference templates

Two copy-me starting points ship in `examples/`, each a **standalone crate outside
the workspace** (their own `[workspace]`), so they depend on `polypus-backend` exactly
as your crate would. Both are certified by the conformance battery (below).

- **[`examples/rust-backend-template`](../examples/rust-backend-template)** — the
  *native / wire-protocol* path (option D, the one QMIO uses). One `src/lib.rs` of
  ~150 lines, pyo3-free: it serialises each circuit, calls one `transport` function
  (the single thing you replace with your device's protocol), classifies failures,
  self-checks the results, and registers itself. Run its battery with
  `cd examples/rust-backend-template && cargo test`.
- **[`examples/python-backend-template`](../examples/python-backend-template)** — the
  *subprocess bridge* path, for a provider that only ships a Python SDK. Its
  [`worker.py`](../examples/python-backend-template/worker.py) is
  `worker_template.py` with the one `execute_circuits` function filled in; the framing
  and handshake are copied verbatim (they are the frozen protocol). Run its battery —
  which spawns the worker through the real bridge — with
  `cd examples/python-backend-template && cargo test` (needs a `python3`).

## Testing your backend: the conformance battery

`polypus-backend-conformance` is a reusable test battery that certifies a
`QuantumBackend` against the semantics documented above. It depends only on
`polypus-backend` (and `polypus-circuit`, to build its probe circuits), so it is as
pyo3-free as the contract — add it as a dev-dependency and nothing else:

```toml
[dev-dependencies]
polypus-backend-conformance = "0.7"
```

```rust
use std::sync::Arc;
use polypus_backend::QuantumBackend;
use polypus_backend_conformance::{Conformance, Fault};

let report = Conformance::new("my-backend", || Ok(Arc::new(MyBackend::new()) as Arc<dyn QuantumBackend>))
    .shots(1024)
    // Supply a factory for each fault your backend can be driven into:
    .fault(Fault::Unresponsive, || Ok(Arc::new(MyBackend::rigged_dead()) as _))
    .fault(Fault::CleanError,   || Ok(Arc::new(MyBackend::rigged_error()) as _))
    .fault(Fault::Aborted,      || Ok(Arc::new(MyBackend::rigged_slow()) as _))
    .run();

println!("{report}");
report.assert_conformant(); // panics with the full report if any check FAILED
```

### What it checks

**Behaviour** (needs only a healthy backend): results satisfy
[`validate_run_results`](#reporting-failure-backenderror) over both the native and
OpenQASM 2.0 representations (count, non-empty, shot conservation C-3, bitstring
keys); batch order is preserved; an empty batch returns no maps; a batch larger than
`max_concurrency` runs correctly in waves; shot distribution conserves the total,
including a zero-shot replica; a backend declaring a *finite* `max_concurrency` runs a
full wave of that size.

**Error classification** (needs the backend driven into each fault): an unrecognised
`Foreign` circuit → `UnsupportedCircuit`; a dead/hung backend → `Unresponsive`; an
out-of-band cancelled call → `Aborted`; a clean provider failure → a definitive error
(neither `Unresponsive` nor `Aborted`).

### Skips are honest, not passes

The battery cannot force a live QPU to hang, so each error-classification check runs
**only** against a backend *you* rig into that state via `.fault(...)`. A fault you do
not supply is **skipped, not passed**, and the report says so. A skip does not break
conformance (`assert_conformant` only fails on a `FAIL`): a backend that genuinely
cannot enter a fault — a synchronous in-process simulator cannot "stop responding" —
should show that check skipped, never fake a pass. So "passes the battery" always
comes with an exact statement of what was and was not exercised.

### A backend whose healthy instance needs hardware

If your backend has no simulator mode (like QMIO), its behavioural checks cannot run
without a device, but its error classification still can — against a mock or
unreachable endpoint. Use `Conformance::run_error_classification()`, which runs only
the foreign-rejection and fault checks and skips the behavioural group.

## Conformance of the built-in backends

We ran the battery against Polypus's own four backends
([`crates/polypus-infrastructure/tests/conformance.rs`](../crates/polypus-infrastructure/tests/conformance.rs)),
so "your backend passes it" is anchored to "ours do too". Following the phase's rule,
**every failure is documented here before any fix**. The behavioural checks pass for
every runnable backend; the interesting results are all in error classification, and
they trace exactly the **recoverable-vs-fatal** split between the Rust-native backends
(which fail *in Rust* with a typed variant) and the seal-delegating backends (which
have no failure variants of their own).

> **These results are a one-off manual run, not a continuous CI guard.** Only the
> pure-Rust `native_backend_conforms` test runs on every build; the Local, CUNQA and
> QMIO tests are `#[ignore]`d (they need Aer / SLURM / the qmio feature, and Aer is
> fragile off the main thread). The table below is the record of running them by hand
> for this phase — re-run them (commands are on each test) when touching a backend;
> do not read them as verified on each build.

| Backend | How it ran | Behaviour | Unresponsive | Aborted | Clean error | Foreign→Unsupported |
|---|---|---|---|---|---|---|
| **Native** | full battery | ✅ pass | n/a (skip) | n/a (skip) | n/a (skip) | ✅ pass |
| **Local (Aer)** | full battery (Aer present) | ✅ pass¹ | n/a (skip) | n/a (skip) | ✅ `External` | ✅ pass |
| **CUNQA** | env-gated (needs SLURM) | not run here | n/a | n/a | (as Local) | (as Local) |
| **QMIO** | error-classification only (unreachable endpoint) | needs hardware | n/a (see below) | n/a | ✅ `External` | ❌ **QMIO-1** |

¹ except the empty-batch edge case — finding **LOCAL-1** below.

### Native — fully conformant

The pure-Rust statevector backend passes every behavioural check and rejects a
`Foreign` circuit with `UnsupportedCircuit`. It has **no** `Unresponsive`, `Aborted`,
or clean-provider-error path: a synchronous CPU simulation cannot stop responding, be
signalled mid-call, or report a provider failure on a valid circuit. Those three
classification checks are therefore **skipped** (not applicable), and nothing fails.
This is the shape a self-contained in-process backend should have.

### Local (Aer) and CUNQA — seal-delegating

Both hand the whole call to the in-process `polypus_python` seal (contract C-1). Two
consequences the battery makes concrete:

- **No `Unresponsive`/`Aborted` path.** Unlike a subprocess or wire backend, these run
  the provider *inside our own process*. A dead interpreter is not a recoverable
  "backend stopped responding" — it is a crash of the host process — and there is no
  in-flight call to signal, so neither variant is reachable. The battery skips both.
  This is the recoverable-vs-fatal distinction the phase set out to surface: a
  crash/hang is *retryable* only when the backend is isolated (subprocess/wire); an
  in-process seal cannot offer that, by construction.
- **Clean provider error → `External`.** An Aer failure (the battery rigs one with a
  bogus `sim_method`) surfaces as a Python exception, boxed into `External` and
  re-raised verbatim at the FFI edge. This check passes.

Findings, **documented not fixed** (they live in `local.rs` / the seal — outside this
phase's module scope):

- **LOCAL-1 — an empty batch raises.** `LocalBackend::run_circuits(&[])` fails with
  `ValueError: not enough values to unpack (expected 2, got 0)` from the seal, instead
  of returning `Ok(vec![])` as the native backend does. It is **latent**: the
  `Planner` special-cases an empty batch and never calls a backend with one, so no
  Polypus path hits it today. Proposed fix: short-circuit `if qcs.is_empty() { return
  Ok(Vec::new()); }` at the top of `LocalBackend::run_circuits` (and the CUNQA one).
  Of the three findings this is the **cheapest to close**: unlike LOCAL-2 (seal /
  contract territory) and QMIO-1 (changes a Python exception class), the fix is a
  self-contained guard clause entirely inside `local.rs`/`cunqa.rs` with **no
  contract implications**. It is left documented-not-fixed purely to respect this
  phase's module boundary, not because it is hard.
- **LOCAL-2 — the `backend` kwarg is ignored (a C-1 contract violation).**
  `LocalBackend` sends a `backend` kwarg to the seal's `run_qcs`, but the seal's local
  path never reads it (it hard-codes `AerSimulator`). This is not cosmetic: contract
  C-1 states that every kwarg the Rust side sends **must be consumed**, and silently
  ignoring one is "a contract violation" — the same class of break as the historical
  `cores_per_qpu` one. It is now recorded in the C-1 known-breaks list in
  [`docs/CONTRACTS.md`](CONTRACTS.md) (audit LOCAL-2, open). Proposed fix: consume it
  in the seal or stop sending it from `local.rs`.

CUNQA was not runnable here (no SLURM, and the `cunqa` Python module is absent), so its
row is by analysis: it shares the seal-delegation profile above, including LOCAL-1's
empty-batch behaviour.

### QMIO — native wire backend

QMIO speaks pickle-over-ZeroMQ from Rust, so it *does* fail in Rust with a typed
`QmioError`. Run against an unreachable endpoint:

- **Clean error → `External`.** A wire failure (connect/timeout) is boxed into
  `External` (a `QmioError`), which the edge re-raises as the typed `polypus.QmioError`.
  This check passes.
- **`Unresponsive` is not applicable — by design.** A QMIO timeout *is* "wedged past a
  deadline", which the contract lists under `Unresponsive`. But QMIO exhausts its **own**
  internal retries (`QMIO_MAX_RETRIES`) before returning, so by the time it fails the
  outcome is *definitive*, not retryable — `External` is the honest classification, and
  the battery's `Unresponsive` check is simply not supplied for QMIO (skip). This is a
  deliberate reading of recoverable-vs-fatal, not a defect.
- **QMIO-1 — a `Foreign` circuit surfaces as `External`, not `UnsupportedCircuit`.**
  `QmioBackend::run_circuits` maps *every* `QmioError` — including
  `QmioError::UnsupportedCircuit` for a `Foreign` circuit — into `External`, so a caller
  matching on `BackendError::UnsupportedCircuit` misses it (it arrives as a
  `polypus.QmioError` at the edge, not the structural unsupported-circuit error). The
  foreign-rejection check **fails** for QMIO. Proposed fix: in `run_circuits`, map
  `QmioError::UnsupportedCircuit` to `BackendError::UnsupportedCircuit` before the
  catch-all `External`.

### Status of the open findings

| Finding | Severity | Reachable today | Fix touches | Deferred because |
|---|---|---|---|---|
| LOCAL-1 (empty batch) | low (latent) | no (planner shields it) | `local.rs`, `cunqa.rs` | outside module scope; cheapest to close (self-contained guard clause, no contract impact) |
| LOCAL-2 (ignored `backend` kwarg) | **C-1 contract violation** | n/a | the `polypus_python` seal or `local.rs` | outside scope; seal is C-1 |
| QMIO-1 (Foreign→External) | low | only on a misuse (Qiskit circuit to QMIO) | `qmio.rs` | outside scope; **changes the Python exception class**, so needs a contract call |

None of these code changes has been made. The only edit to a contract document is a
**record** of LOCAL-2 in the C-1 known-breaks list of
[`docs/CONTRACTS.md`](CONTRACTS.md) — documenting an existing violation where the
contract already says it belongs, not altering the contract or fixing the break. Note
that fixing QMIO-1 would change the Python-visible exception for that case (from
`polypus.QmioError` to the structural unsupported-circuit error), which is a
contract-visible decision and
should be taken as one.

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
  methods ship with defaults (as `cancel` and `wants_cancel_watcher` did) rather than
  as breaking additions.
- **Scope of the contract.** The stability commitment covers the `polypus-backend`
  surface: the `QuantumBackend` trait, its types, and the registry API
  (`register_backend`, `BackendBuildContext`, `BackendFactory`). The **subprocess
  bridge protocol** (version 1, above) is a second stable contract: the frame shapes
  are versioned by the `protocol` field, and a breaking change bumps that version
  (workers declare the version they speak in their `ready` handshake). The
  `polypus-subprocess-backend` crate inherits the same workspace version. The
  **conformance battery's API** (`Conformance`, `Fault`, `Report`, `Check`, `Status`
  in `polypus-backend-conformance`) is likewise a surface third parties build tests
  against, and follows the same pre-1.0 SemVer discipline.
- **crates.io.** `polypus-backend`, `polypus-subprocess-backend` and
  `polypus-backend-conformance` are **not yet published to crates.io.** The conformance
  battery has now landed (this phase), so the contract's shape is exercised end-to-end;
  publishing still waits on a deliberate decision to freeze the surface. Until it is
  published, depend on these crates by path or git. This document will be updated when
  they go to the registry.

When the contract reaches 1.0, this section will state the post-1.0 SemVer guarantee
(breaking changes only on a major bump).
