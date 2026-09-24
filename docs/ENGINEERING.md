# Polypus Engineering Guide — invariants and style

This document records the engineering rules that live *inside* the modules:
workspace boundaries, GIL discipline, numerical/quantum correctness,
memory-safety (`unsafe`) policy, feature hygiene, security decisions, and Rust
style. It complements:

- [`docs/CONTRACTS.md`](CONTRACTS.md) — the agreements *between* layers (seams).
- [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) — process: setup, checks, commits, PRs.

Like `docs/CONTRACTS.md`, this file is deliberately public: it is equally useful to
external contributors and to AI coding assistants working on a clone of this
repo. If a rule here and the code disagree, treat that as a bug in one of the
two and open an issue — do not silently pick a side.

## 1. Architecture at a glance

Polypus is an open-source distributed quantum computing library: a Rust core
with PyO3 Python bindings. The Cargo workspace has eleven crates:

| Crate | Role | PyO3? |
|---|---|---|
| `polypus-circuit` | Circuit IR + OpenQASM 2.0 import/export + QIR export | No |
| `polypus-sim` | Statevector simulator (GIL-free; optional rayon via the `parallel` feature) | No |
| `polypus-physics` | Particle physics: classical Monte Carlo transport + Hamiltonians as Pauli sums | No |
| `polypus-optimizers` | Variational optimizers (DE, PSO, QNG) behind evaluation oracles | No |
| `polypus-observable` | Cost observables (Qubo / Ising) reducing measurement counts to a cost; pure math | No |
| `polypus-backend` | The **pyo3-free backend contract** a third party implements: the `QuantumBackend` trait, `BoundCircuit`, `RunParams`, `BackendError`/`InfrastructureError`, the `Planner`, the transpiler seam, the memory budget and the runtime **backend registry** (`register_backend`, `BackendBuildContext`). Zero PyO3 in its dependency tree | No |
| `polypus-subprocess-backend` | A pyo3-free `QuantumBackend` bridging to a provider's Python SDK running in its **own subprocess** over a versioned length-prefixed JSON protocol; registered as the built-in `"subprocess"` backend. The interpreter runs in the child, never embedded here | No |
| `polypus-infrastructure` | The concrete execution backends (`local`/Aer, `cunqa`, `qmio`, `native`), the `Infrastructure` factory (which routes unknown names through the registry and registers the built-in `subprocess`/`qmio` backends), the Qiskit boundary (`QiskitCircuit`, `to_py_object`) and the construction-time `ExecutionConfig`; re-exports the `polypus-backend` contract | GIL only |
| `polypus-orchestration` | Flow orchestration (policy): `Resources`, the monomorphic `Scheduler`, the `Flow` trait + **all** flows (`RunCircuitFlow`, `TrainFlow`) with the `OracleFactory` seam, `dispatch_optimizer` and the type-erased `OracleErrorSlot` | No |
| `polypus-evaluation` | Candidate evaluation: the oracles (`VqcOracle`, `QmlOracle`) and the `OracleFactory` implementations that build them (`VqcOracleFactory`, `QmlOracleFactory`), plus `PyVarianceOracle`, `PyCallbackObservable`, the supervised QML objectives (`SupervisedObjective`: `PyLabelledCost`, `PySampleCost`; see `docs/adr/0002-qml-supervised-labels.md`), `CircuitSource` and `EvaluationError` | GIL only |
| `polypus-logger` | `log::Log` sink shared by the workspace; installed only by the app layer | No |
| `polypus` | The library + Python extension module; the FFI edge — `#[pyclass]`es, kwarg parsing, and error→`PyErr` conversion | **Yes** |

Interoperability: **Qiskit ≥ 2.0** and **qiskit-aer ≥ 0.17** (pinned in
`packages/polypus_python/pyproject.toml`), **CUNQA** (distributed QPUs over
SLURM) and **QMIO** (CESGA's real QPU). Formats: OpenQASM 2.0 and QIR (LLVM IR
`.ll`, plus bitcode `.bc` assembled by the **external** `llvm-as` tool — that
boundary stays out-of-process and explicit; see
`crates/polypus-circuit/src/qir.rs`).

## 2. Workspace boundaries

- `polypus-circuit`, `polypus-sim`, `polypus-physics`, `polypus-optimizers`,
  `polypus-observable`, `polypus-backend`, `polypus-subprocess-backend` and
  `polypus-logger` are **pure Rust: they must not depend on `pyo3` or Python**. Do
  not introduce `Py<...>`, `PyAny`, `Python`, the GIL, or Python types into them.
  `polypus-backend` is the strictest of these: it is the public backend contract, so
  its **whole dependency tree must stay PyO3-free** (a third party implements a
  backend against it without pulling the interpreter) — a Python/provider failure
  crosses it type-erased in `BackendError::External`, and the concrete backends live
  one layer up in `polypus-infrastructure`. `polypus-subprocess-backend` is pure Rust
  too and stays PyO3-free **by construction**: it reaches a provider's Python SDK by
  running it in a separate process, not by embedding the interpreter (its own tests
  spawn a `python3` worker as a subprocess, which is not a link-time dependency).
- `polypus-infrastructure` may depend on `pyo3` for the GIL and for carrying a
  `PyErr` verbatim (its Aer/CUNQA/QMIO seams call into Python), but it defines
  **no** `#[pyclass]` and **no** `From<_> for PyErr`: turning a `BackendError`
  into a typed `polypus.*` exception is the edge's job
  (`polypus::exceptions::backend_error_to_pyerr`).
- `polypus-orchestration` is **`pyo3`-free at the source level**: it must not name
  `pyo3`, `Python`, `PyErr` or the GIL (its `Cargo.toml` has no `pyo3`). It still
  links libpython *transitively* through `polypus-infrastructure`, so its tests
  run in the same job as `polypus`, not the pure-Rust group. A real oracle
  failure reaches it **type-erased** as a `Box<dyn Error + Send>` in the
  `OracleErrorSlot`; the `polypus` edge downcasts it back to the concrete
  `EvaluationError` to re-raise (plan §10.1). It owns **every** `Flow`, including
  the training flow (`TrainFlow`): a training flow's `pyo3`-touching oracle is
  built through an `OracleFactory` implemented in `polypus-evaluation`, so the
  *order* lives here with the scheduler while only its assembly stays behind the GIL.
- `polypus-evaluation` may depend on `pyo3` (Qiskit binding under the GIL, Python
  callbacks) like `polypus-infrastructure`, but likewise defines **no**
  `#[pyclass]` and **no** `From<_> for PyErr`: turning an `EvaluationError` into a
  typed `polypus.*` exception is the edge's job
  (`polypus::exceptions::evaluation_error_to_pyerr`).
- Only the `polypus` crate may contain `#[pyclass]` / `#[pymethods]` /
  `#[pyfunction]`, own the exception hierarchy, and convert errors to `PyErr`.
- The optimizers are decoupled from circuits and Python via the
  `EvaluationOracle` / `VarianceOracle` traits (contract C-5). A new optimizer
  is implemented against those oracles; it must **not** call Python or know
  about the backend.
- Logging: library crates emit through the `log` **facade** only and never
  depend on `polypus-logger`; only the top-level `polypus` crate installs the
  sink.

## 3. GIL discipline (`polypus` crate only)

- Parameter binding and heavy computation are **GIL-free**; this is a core
  performance property of the project and must not regress.
- **Never** hold the GIL while waiting (`block_on`, join) on workers that
  themselves need to acquire it: release it first with
  `Python::with_gil(|py| py.allow_threads(...))`. Ignoring this deadlocks
  (documented in `crates/polypus-evaluation/src/qml_oracle.rs`).
- **Releasing the GIL does not by itself process signals.** `allow_threads`
  lets other Python threads run, but a pending SIGINT (Ctrl+C) is turned into a
  `KeyboardInterrupt` only when the **main thread** runs Python bytecode or when
  `PyErr_CheckSignals` is called explicitly. Long-running Rust work is otherwise
  deaf to Ctrl+C until it returns. So a GIL-free loop that must stay
  interruptible has to call `py.check_signals()` at a safe boundary: the
  optimizer entry points (`train` / `qml.train`) release the GIL around the
  whole `optimize()` call, and the `Planner` runs a **host-interrupt check**
  between execution waves (inside `execute`) — the one place that boundary now
  lives, so both the VQC and QML oracles stay interruptible without their own
  signal loop.
  - **The `Planner` itself no longer names `py.check_signals()` — it moved to the
    edge.** The planner and all its types now live in the pyo3-free
    `polypus-backend` crate, so between waves it calls a small injected guard, the
    `Interrupt` trait, carried (optionally) on the `CancelToken`
    (`CancelToken::poll_interrupt`). The one implementation that actually calls
    `Python::with_gil(|py| py.check_signals())` is `SignalInterrupt` in
    `crates/polypus/src/bindings/mod.rs` (the edge); `train`, `qml.train` and
    `run_quantum_circuit` build a guarded token with it and drive the run through
    `Scheduler::run_cancellable`. A pure-Rust caller uses a guard-less token
    (`Scheduler::run`) and the check is a no-op — which is why `run` is documented
    as **not** interruptible and is never used on a Python-facing path.
  The resulting `PyErr` — a `KeyboardInterrupt`, or an error raised by the user
  `expectation_function` / variance callback — is recorded in the shared
  `OracleErrorSlot` and re-raised to Python by the entry point as the
  **original** exception. The interrupt travels type-erased through the pyo3-free
  layers, boxed in `BackendError::External`, and the edge downcasts it back and
  re-raises it verbatim (`external_to_pyerr`); a callback exception rides in
  `EvaluationError::Python` / `ObservableError::External` the same way. It is
  never swallowed into a panic by an `.expect()` (that would surface as an opaque
  `PanicException`; see §9 and `OracleErrorSlot` in `polypus-orchestration`).
- **How many waves that boundary produces is backend- *and* batch-dependent
  (issue #147).** `SequentialPlanner::execute` sizes each wave from
  `backend.capabilities_for(tasks).max_concurrency`, a **batch-aware** companion
  to the frozen, argument-free `QuantumBackend::capabilities()`. The native and
  local backends cap their real concurrency by a statevector memory budget scaled
  by the batch's *widest* circuit (`mem_budget::max_statevector_concurrency`);
  they override `capabilities_for` to expose that cap **only when the whole batch
  cannot be held in the budget at once** — the high-qubit regime — so a high-qubit
  batch is split into several waves and the between-wave interrupt check (the
  edge's `SignalInterrupt`, above) runs **once per wave**: a whole training
  generation is no longer one uninterruptible call. When
  the batch fits (the common low-qubit case) they report an unbounded wave, so the
  whole population still reaches the backend as a **single** call (Aer
  parallelises the experiments internally; splitting would only add per-call
  overhead — pinned by `tests/python/test_qml_concurrency.py`). The "does it fit"
  test (`wave_concurrency`) is made against the **pure** memory limit — how many
  statevectors the budget holds, with the thread count removed — *not* the
  thread-capped cap: a thread-based gate (`cap >= cores`) degenerates on a
  single-core allocation (a common SLURM/container case, or `RAYON_NUM_THREADS=1`),
  where the cap always equals the core count regardless of qubit count and would
  wrongly report one uninterruptible wave for a memory-heavy batch (issue #147's
  single-thread case). Backends whose cap is static (CUNQA at `n_qpus`, QMIO at 1)
  do not override it and inherit the default, which delegates to `capabilities()`
  ignoring the batch, so their behaviour is unchanged.
  - **`capabilities_for` is fallible, so a Ctrl+C during width-reading is not
    swallowed.** It runs at the top of every `execute` — for training, once per
    generation on the optimizer thread, with the GIL released. `LocalBackend` sizes
    waves from every circuit's width, including a `Qiskit` circuit's `num_qubits`,
    which it reads via `getattr` under the GIL exactly as `run_circuits` does — so a
    high-qubit Qiskit population (a Qiskit-templated ansatz on `backend="aer"`, the
    common QML case) is still wave-split for interruptibility. That `getattr` runs
    Python bytecode, which CPython can abort with a `KeyboardInterrupt` for a
    pending Ctrl+C; mapping *that* to "width unknown" (an `.ok()` that discards it)
    clears the signal before the planner's between-wave interrupt check can see
    it, which made `qml.train` unresponsive to Ctrl+C on constrained runners
    (issue #147 follow-up). So `capabilities_for` returns
    `Result<BackendCapabilities, InfrastructureError>`: a `KeyboardInterrupt` from
    the read is propagated verbatim (boxed in `BackendError::External` inside
    `InfrastructureError::Backend`, re-raised by the edge), and only a genuine
    non-interrupt failure (a missing attribute,
    which a real `QuantumCircuit` never has) falls back to "width unknown". The
    default impl and the native backend read widths GIL-free and never error (they
    return `Ok`); the call site in `execute` propagates with `?`, the same way it
    already handles `run_circuits`' errors.
  - **Why a new method rather than changing `capabilities()`.** The
    argument-free `capabilities()` is called *without* a batch by
    `Resources::new` (to validate the planner/backend pairing up front) and is
    overridden by test mocks; its signature and meaning are load-bearing outside
    this crate, so the fix is strictly additive — `capabilities_for(&[CircuitTask])
    -> Result<BackendCapabilities, InfrastructureError>`, taking the planner's
    borrowed task slice (each `CircuitTask` already holds a `&BoundCircuit`) so
    sizing a wave clones no circuit. The override reuses the *exact* cap arithmetic each backend's
    `run_circuits` already applies internally, so the wave size the planner picks
    matches the memory bound the backend would enforce anyway — and collapses to a
    single wave whenever the batch fits in the budget.
- The same discipline applies to `run_quantum_circuit`: it releases the GIL
  around the whole `scheduler.run_cancellable(flow, &token)` call, where `token`
  carries the edge's `SignalInterrupt` guard; the `Planner` runs that guard between
  execution waves (in `execute`), and the counts are converted to a Python object
  back at the edge — GIL re-acquired — only after the run returns. A pending Ctrl+C surfaces there as a `KeyboardInterrupt`
  propagated verbatim through the function's `Result`, never swallowed or retyped.
- `statevector` follows the same rule at a smaller scale: it releases the GIL
  around the `StatevectorSimulator::run_cancellable` call (parameter binding
  stays on the GIL side — it is O(gates) and allocates nothing of size `2^n`),
  and calls `py.check_signals()` both *mid-run* and the moment the GIL is
  reacquired, **before** handing the amplitudes to NumPy — the latter being the
  same "reacquire-then-check-before-building-the-result" boundary
  `run_quantum_circuit` uses.
- The mid-run half is worth spelling out, because it is how a pure-Rust crate
  stays interruptible without learning about Python (§2). `polypus-sim`'s gate
  loop takes an `Option<&mut dyn FnMut() -> bool>` and polls it periodically;
  `true` abandons the run with `SimError::Cancelled`. It knows nothing about
  *why* — a signal, a deadline, a cancel button all look the same to it. Only
  `statevector` fills that hook with `Python::with_gil(|py| py.check_signals())`
  (re-entrant from inside `allow_threads`, the same guarantee `cunqa.rs`'s
  `Drop` relies on — see §9). Two rules make it work:
  - **Throttle inside the pure crate, not at the Python boundary.** The hook is
    called at most once per ~25ms of wall clock (with the clock itself read once
    per ~64k amplitude updates), so its frequency is decoupled from gate cost —
    a cheap 1-qubit gate and a 25-qubit gate differ by orders of magnitude — and
    a circuit that finishes quickly never calls it at all. Interrupt latency is
    then bounded by a constant instead of by the run's own duration, which
    matters because `polypus_sim::MAX_QUBITS` bounds a run's *memory*, not its
    wall-clock time: cost scales with gates × `2^n` and nothing bounds the gate
    count. The interval is also stretched to a multiple of the hook's *measured*
    duration, because the simulator cannot know what a hook costs: this one is
    ~1µs when nothing else wants the GIL and milliseconds when another Python
    thread holds it (the interpreter only yields on its switch interval), and
    without that adaptation the latter case measurably slowed a contended run
    down. Backed by `benchmarks/bench_statevector.py`.
  - **Recover the real exception at the Python-facing boundary.** `SimError`
    cannot carry a `PyErr` (§2), so the hook stashes the `PyErr` and
    `statevector` re-raises *that* verbatim when the run comes back
    `Cancelled` — never `PyValueError::new_err(e.to_string())`, which would
    downgrade a `KeyboardInterrupt` into a bogus `ValueError`. Same problem, and
    the same answer, as `OracleErrorSlot` in
    `crates/polypus-orchestration/src/dispatch.rs`; no shared slot is needed here
    because `statevector` is single-shot and the hook runs on the calling
    thread. What is *not* interruptible is `Statevector::new`'s `2^n`
    allocation — one `vec![]` with nowhere to put a checkpoint — which is why
    the post-run check above stays: near the qubit ceiling that allocation is
    the whole run.
- Preserve concurrent execution: candidates that bind/evaluate truly in
  parallel. If you add a path that runs circuits from worker threads, keep
  this guarantee.
- The Rust test suite is **Python-runtime-free by design** — it proves
  GIL-freedom. Python is needed only at build/link time for PyO3; Qiskit and
  Aer are required only by the Python test suite (`tests/python/`). CI encodes
  this split (`.github/workflows/ci.yml`); don't add a Rust test that needs a
  live Python runtime.

## 4. Numerical and quantum correctness

- The statevector must remain **normalized**; gate matrices must be
  **unitary**. Every new gate/kernel needs matrix- and state-level tests (see
  `crates/polypus-sim/tests/gate_matrices.rs` and
  `crates/polypus-sim/tests/canonical_states.rs`).
- **Parallel == sequential:** kernels under the `parallel` feature (rayon)
  must produce **bit-identical** results to the sequential path. Every new
  parallel kernel is tested against its sequential version.
- **Qubit ceiling:** a dense statevector needs `2^n` complex amplitudes
  (`16 · 2^n` bytes), so `polypus-sim` refuses circuits above
  `polypus_sim::MAX_QUBITS` (30 ≈ 16 GiB) as the **first** thing
  `StatevectorSimulator::run` does — before any allocation, and low enough that
  `1 << n` cannot overflow. At the seam this surfaces as a `ValueError` naming
  the requested and the supported count (`polypus.statevector`;
  `tests/python/test_statevector.py`). Below the ceiling, those amplitudes cross
  the seam as a **NumPy array of `dtype=complex128`** — one contiguous buffer
  moved out of the `Statevector` (`into_amplitudes`, no copy) and wrapped by
  rust-numpy — rather than as a `list` of boxed Python `complex` objects. That
  removes a `2^n`-sized conversion which, at these sizes, cost more than the
  simulation: a gateless 30-qubit call went from 29.7s to 5.0s, and a 26-qubit
  one from 1.9s to 0.4s (one-off measurement on a 32-core dev box, not a tracked
  `benchmarks/` script; deep circuits gain proportionally less because the gates
  dominate — 30 qubits with a Hadamard layer: 50.4s → 25.6s). It does not make
  near-ceiling statevectors cheap, though: what remains is `Statevector::new`'s
  `2^n` allocation, which is why even a **gateless** 30-qubit call still costs
  ~5s. The ceiling is a memory bound and says nothing about wall-clock.
  `polypus.Circuit` itself stays unbounded on purpose: it is backend-agnostic
  IR, and CUNQA/QMIO/Aer have their own, different capacities — the ceiling is
  enforced where it applies, not in the IR.
- **Reproducibility:** the RNG is seedable (`rng.rs` in `polypus-sim` and in
  `polypus-optimizers`). Results must be deterministic given a seed. Don't
  introduce nondeterminism: iteration order over a `HashMap` affecting
  results, unseeded RNG, parallelism that changes reduction order, etc.
- In numeric tests, compare with a tolerance (epsilon), not exact float
  equality — **except** the QASM round-trip, which is byte-identical
  (contract C-2).
- Use `f64` and `num-complex::Complex64` consistently with the rest of the
  simulator.

## 5. `unsafe` policy

- `unsafe` is allowed **only** in the `polypus-sim` kernels
  (`crates/polypus-sim/src/kernels.rs`), where raw-pointer access to the
  statevector is performance-critical, and in the one **signed-off exception**
  below.
- **Every** `unsafe` block keeps its `// SAFETY: ...` comment justifying the
  invariants (indices `< 2^n = data.len()`; the amplitude pair does not
  alias). Do not weaken those invariants.
- Adding `unsafe` anywhere else requires a benchmark that justifies it, the
  same `// SAFETY:` documentation, and explicit maintainer sign-off in the PR.
- **Signed-off exception — `polypus-subprocess-backend`'s `pre_exec`.** The
  subprocess bridge holds exactly one `unsafe` block, in `Worker::spawn`
  (`crates/polypus-subprocess-backend/src/lib.rs`): `Command::pre_exec`, used to
  arm the `PR_SET_PDEATHSIG` orphan guard on the worker child so it cannot outlive
  a hard kill of the host process (a real HPC requirement — an orphaned worker
  would hold a SLURM node/QPU). This one is **not** a performance optimisation, so
  the "benchmark that justifies it" rule does not apply; it is unavoidable because
  `Command::pre_exec` is an `unsafe fn` in std with no safe equivalent, and
  `PR_SET_PDEATHSIG` resets across `fork` so it can only be armed in the child,
  between fork and exec — which is exactly what `pre_exec` is for. The block's body
  is itself safe: it calls `nix`'s audited `set_pdeathsig` wrapper (a single
  async-signal-safe `prctl(2)`), not hand-written FFI, and carries the mandated
  `// SAFETY:` comment. Everything else the bridge needs from POSIX — notably
  `kill` for out-of-band cancellation — goes through `nix`'s safe wrappers with no
  `unsafe`. Do not add a second `unsafe` block to this crate without the same
  sign-off.
- Let the compiler prove `Send`/`Sync`; never force them with `unsafe impl`.

## 6. Feature hygiene

- The **default** build stays lean: it must **not** pull ZeroMQ or pickle —
  those belong to the opt-in `qmio` feature. `polypus-sim` is sequential by
  default; rayon lives behind its `parallel` feature (the `polypus` crate
  enables `parallel` deliberately for the HPC target; a runtime threshold —
  `DEFAULT_PARALLEL_THRESHOLD` — keeps small circuits on the sequential path).
- Every heavy or optional dependency goes behind a feature.
  `extension-module` controls linking of the PyO3 extension. Declare features
  additively and don't enable them implicitly.
- The only default features are the compile-time log-level toggles
  (`info-logs` et al.), which forward to `polypus-logger`.

## 7. Interoperability notes (beyond contract C-2)

Contract C-2 owns the gate vocabulary and the byte-identical round-trip.
Additionally:

- Exported OpenQASM 2.0 uses standard `qelib1.inc` names (plus the verbatim
  `gate` declarations of any declared gate it calls) and must remain accepted
  by Qiskit (`QuantumCircuit.from_qasm_str`). Aer runs it directly as long as
  every instruction is in Aer's basis; `ch`, `u0`, `rccx`, `rc3x`, `c3x`,
  `c3sqrtx`, `c4x` and every declared gate are not, so such a circuit must be
  transpiled first (`qiskit.transpile(qc, AerSimulator())`) — Aer never unrolls
  them by itself.
- `from_qasm2` accepts both Polypus output and `qiskit.qasm2.dumps` output,
  `gate` declarations included. Canonicalizations performed on import: the
  builtins `U` → `u` and `CX` → `cx` (Qiskit's own names for them), multiple
  `qreg`/`creg` declarations flattened into one index space, constant
  parameter expressions (e.g. `pi/2`) evaluated. Nothing is decomposed or
  re-spelled otherwise (C-2): `p`, `u1`, `u2`, `u`, `u3` all stay as written.
- Parse errors carry the **1-based line number** (`CircuitError::Parse` on
  the Rust side, `ValueError` once across the Python boundary).

## 8. Security

- Validate and sanitize all inputs; never trust external data. Apply least
  privilege and secure-by-default values. Never embed secrets, credentials or
  tokens in code — use environment variables or a secrets manager.
- **Untrusted peers:** the QMIO backend speaks the pickle-over-ZMQ protocol,
  but it uses `serde-pickle` — which does **not** execute
  `__reduce__`/`GLOBAL` — instead of Python's `pickle.loads`, *specifically to
  avoid RCE* from a malicious peer. Any code that deserializes data coming
  from a QPU, backend or network peer must treat that data as **untrusted**
  and must not use a deserializer capable of executing code. Do not introduce
  dependencies that reintroduce a deserialization-RCE surface.
- The `zeromq` crate was chosen because it is **pure Rust** (no C `libzmq`
  build dependency, at the cost of an async tokio API). Do not replace it
  with a C binding without an explicit trade-off discussion with the
  maintainers.

## 9. Rust style

- The workspace uses **edition 2021**; keep code compatible with it. All code
  passes `cargo fmt` and `cargo clippy --workspace --all-targets -- -D
  warnings` (see `CONTRIBUTING.md` for the full check list and the lint-level
  policy).

**Error handling.** Use `Result<T, E>` and `?`. No `unwrap()` / `expect()` /
`panic!` in production paths (acceptable in tests, examples, or truly
unrecoverable invariants, always with a clear message). Each crate defines its
own error type (`error.rs`); reuse and extend it instead of introducing ad-hoc
errors or bare strings. Never silence or discard errors (`let _ = ...`).
Errors crossing the FFI boundary become `PyErr` via `Result` — never a panic.
In the `polypus` crate this means every path reachable from a `#[pyfunction]` /
`#[pymethods]` returns a typed error (`BackendError`, `EvaluationError`, or the
crate's `QmioError`) mapped to the `polypus::exceptions` Python hierarchy — even
where the failure is "unlikely". A Python exception raised by the
`polypus_python` seam is carried verbatim and re-raised with its original type,
so contract C-1's `ValueError`/`TypeError` failure modes are preserved.

**`Drop` must be panic-free.** No `Drop` impl may panic under any circumstance:
a panic while another panic is already unwinding aborts the whole process,
defeating the RAII guarantee (for `CunqaBackend` this would leak the SLURM
allocation). A `Drop` that does fallible cleanup (releasing QPUs, closing a
socket) **logs the failure with `log::error!` and continues** — it never
propagates the error. Because a per-instance flag is worthless once the instance
is gone, the failure is *also* recorded in process-wide state that a higher
layer can inspect (the `backend_cleanup_failures()` counter, exposed to Python).
Acquiring the GIL inside a `Drop` is allowed only when it is explicitly safe
against re-entrancy (PyO3 0.24's `Python::with_gil` is re-entrant) and cannot
propagate a panic through the in-progress unwind; keep the fallible operation
behind a `Result`-returning helper so the `Drop` body only logs and counts.

**Ownership and types.** Prefer borrowing over cloning; avoid unnecessary
`.clone()`. In signatures accept `&str` over `&String` and `&[T]` over
`&Vec<T>`. Leverage the type system: newtypes, enums for state,
`Option`/`Result` instead of sentinels — make illegal states unrepresentable.
Derive the usual traits (`Debug`, `Clone`, `PartialEq`, …) where sensible and
use exhaustive `match`.

**Concurrency and async.** The async runtime is **Tokio** (`rt-multi-thread`).
Never block the executor: wrap blocking operations in `spawn_blocking`. Don't
hold a lock (`Mutex`/`RwLock`) across an `.await`. GIL rules: §3.

**Performance.** Don't optimize prematurely; justify any
performance/readability trade-off. Any claimed performance improvement is
backed by the scripts in `benchmarks/` (`run_benchmarks.py`,
`bench_native_vs_qiskit.py`, `bench_batching.py`). Reserve with
`Vec::with_capacity` when the size is known; use slices and `Cow<str>` to
avoid copies; prefer iterators/combinators over manual loops when clearer.

**Design.** Prefer decoupled, stateless designs. Watch structures that don't
scale with qubit count or number of QPUs; don't load everything into memory.
Apply DRY and single responsibility; don't over-engineer — the simplest
solution that meets the requirements is preferred. Add `log` statements where
they aid debugging and observability (through the facade — see §2).

**API and documentation.** Follow the Rust API Guidelines. Control visibility
deliberately (`pub`/`pub(crate)`); don't expose more than necessary. Document
public items with `///` including examples that work as doctests — CI builds
the rustdoc with `-D warnings`, so broken intra-doc links fail the build.

**Dependencies.** Pin shared external dependencies once in
`[workspace.dependencies]` and reuse them. Adding a dependency is a
maintainer decision: propose it explicitly in the PR description, never slip
it into an unrelated diff. `cargo deny check` gates licenses and advisories.

## 10. Pre-PR checklist

- [ ] Does the affected crate respect its PyO3/Python boundary? (§2)
- [ ] If it touches `polypus`: is the GIL released before waiting on workers?
      Is binding/computation still GIL-free? (§3)
- [ ] If it touches the simulator: statevector normalized, gates unitary, and
      parallel == sequential with tests? (§4)
- [ ] Deterministic given a seed? (§4)
- [ ] Any new `unsafe`? Only in `sim/kernels.rs` (with `// SAFETY:` and a
      benchmark) or the signed-off `pre_exec` exception in
      `polypus-subprocess-backend`; anything else needs the same sign-off. (§5)
- [ ] If it touches circuits: byte-identical round-trip and Qiskit
      compatibility preserved? (§7, contract C-2)
- [ ] Heavy dependencies behind an opt-in feature? Default build lean? (§6)
- [ ] Network/backend data treated as untrusted, no execution-capable
      deserializers? (§8)
- [ ] `fmt` + `clippy -D warnings` + tests green; performance claims backed
      by a benchmark? (§9, `docs/CONTRIBUTING.md`)
- [ ] Explicit errors (no `unwrap`/`panic` in production, including
      `#[pyfunction]`-reachable paths), no `Drop` that can panic, edge cases
      covered, public items documented? (§9)
- [ ] Does the change touch a seam listed in `docs/CONTRACTS.md`? Then the same PR
      updates the contract and its enforcing test.
- [ ] If a requirement was ambiguous: the PR description states which
      interpretation was chosen and why.
