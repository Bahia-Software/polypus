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
with PyO3 Python bindings. The Cargo workspace has six crates:

| Crate | Role | PyO3? |
|---|---|---|
| `polypus-circuit` | Circuit IR + OpenQASM 2.0 import/export + QIR export | No |
| `polypus-sim` | Statevector simulator (GIL-free; optional rayon via the `parallel` feature) | No |
| `polypus-physics` | Particle physics: classical Monte Carlo transport + Hamiltonians as Pauli sums | No |
| `polypus-optimizers` | Variational optimizers (DE, PSO, QNG) behind evaluation oracles | No |
| `polypus-logger` | `log::Log` sink shared by the workspace; installed only by the app layer | No |
| `polypus` | The library + Python extension module; orchestration and infrastructures (`local`, `cunqa`, `qmio`, `native`) | **Yes** |

Interoperability: **Qiskit ≥ 2.0** and **qiskit-aer ≥ 0.17** (pinned in
`packages/polypus_python/pyproject.toml`), **CUNQA** (distributed QPUs over
SLURM) and **QMIO** (CESGA's real QPU). Formats: OpenQASM 2.0 and QIR (LLVM IR
`.ll`, plus bitcode `.bc` assembled by the **external** `llvm-as` tool — that
boundary stays out-of-process and explicit; see
`crates/polypus-circuit/src/qir.rs`).

## 2. Workspace boundaries

- `polypus-circuit`, `polypus-sim`, `polypus-physics`, `polypus-optimizers`
  and `polypus-logger` are **pure Rust: they must not depend on `pyo3` or
  Python**. Do not introduce `Py<...>`, `PyAny`, `Python`, the GIL, or Python
  types into them.
- Only the `polypus` crate may depend on `pyo3` and contain `#[pyclass]` /
  `#[pymethods]` / `#[pyfunction]`.
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
  (documented in `crates/polypus/src/evaluation/qml_oracle.rs`).
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
  statevector is performance-critical.
- **Every** `unsafe` block keeps its `// SAFETY: ...` comment justifying the
  invariants (indices `< 2^n = data.len()`; the amplitude pair does not
  alias). Do not weaken those invariants.
- Adding `unsafe` anywhere else requires a benchmark that justifies it, the
  same `// SAFETY:` documentation, and explicit maintainer sign-off in the PR.
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

- Exported OpenQASM 2.0 uses standard `qelib1.inc` names and must remain
  accepted by Qiskit (`QuantumCircuit.from_qasm_str`) and Aer.
- `from_qasm2` accepts both Polypus output and `qiskit.qasm2.dumps` output.
  Canonicalizations performed on import: `u`/`p`/`u1`/`u2` → `u3`, `swap` →
  its standard 3×`cx` decomposition, multiple `qreg`/`creg` declarations
  flattened into one index space, constant parameter expressions (e.g.
  `pi/2`) evaluated.
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
- [ ] Any new `unsafe`? Only in `sim/kernels.rs`, with `// SAFETY:` and a
      benchmark that justifies it. (§5)
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
