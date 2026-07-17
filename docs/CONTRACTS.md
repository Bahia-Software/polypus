# Polypus Inter-Layer Contracts

This document is the **single source of truth for the agreements between Polypus
layers**. The 2026-07 technical audit showed that every critical defect lived at
a seam between layers — not inside a module — so these contracts get their own
document, their own code owners (see `CODEOWNERS`), and their own tests.

Rules of the road:

1. **Changing a contract requires changing this file in the same PR**, plus the
   contract test that enforces it. A PR that alters a signature, kwarg, format
   or gate set listed here without touching `docs/CONTRACTS.md` must be
   rejected in review.
2. Each contract lists its **enforcing test**. "No test yet" is a temporary
   state, not an accepted one — the status table below tracks which are real.
3. This file is deliberately public: it is equally useful to external
   contributors and to AI coding assistants working on a clone of this repo.

## Status at a glance

| Contract | Seam | Enforcing test | Status | Known break (audit) |
|---|---|---|---|---|
| C-1 | Rust → Python execution | `tests/python/test_seam_contract.py` | ✅ present | `disconnect` reads `slurm_job_id`, not `family` (C1) |
| C-2 | Gate vocabulary symmetry | `polypus-circuit` + `polypus-sim` `tests/contracts.rs` | ✅ present | — |
| C-3 | Measurement counts format | shot-conservation + last-write-wins | ✅ present | shots dropped on uneven distribution (C6) |
| C-4 | Terminal measurement placement | `polypus-circuit` + `polypus-sim` `tests/contracts.rs` | ✅ present | — |
| C-5 | Optimizer ↔ oracle | invariant test, multi-seed + `tests/python/test_oracle_contract.py` | ✅ present | DE `best_fitness` mismatch (C4) |
| C-6 | Version coherence | `hygiene.yml` version step | ✅ present | tag/Cargo diverged at 0.6.0 |
| C-7 | Seeding & run manifest | `tests/python/test_seed_reproducibility.py` + bindings/native Rust tests | ✅ present | repeated runs byte-identical / `train` seed hardcoded `None` (#34) |

⏳ contracts are specified but not yet mechanically enforced; treat them as
review-enforced until the test lands. Each known break has a public issue
labelled `audit-2026-07`.
<!-- TODO(maintainers): create those issues and link them here; flip ⏳ to ✅
     as each test merges. -->

---

## C-1 · Rust → Python execution seam (`polypus_python`)

The Rust orchestration layer (`crates/polypus/src/infrastructure/*.rs`) calls
exactly three functions of the `polypus_python` package. Their names, argument
names and return shapes are frozen by this contract; the two sides must never
be changed independently.

### `connect_to_infrastructure(infrastructure: str, **kwargs)`

| `infrastructure` | kwargs (exact names) | returns |
|---|---|---|
| `"local"` | — | the string `"local"` |
| `"cunqa"` | `n: int` (QPUs), `t: str` (SLURM walltime `"HH:MM:SS"`), `n_nodes: int`, `family_name: str`, `cores_per_qpu: int` | CUNQA *family* handle (opaque object) |

Every kwarg the Rust side sends **must be consumed** by the Python side;
silently ignoring one (as happened with `cores_per_qpu`) is a contract
violation.

### `run_qcs(infrastructure: str, **kwargs)`

| backend | kwargs (exact names) |
|---|---|
| local | `id: str`, `backend: str`, `qcs: list`, `shots: int`, `sim_method: str`, `noise_model` (optional), `seed: int` (optional, C-7) |
| cunqa | `family_id: str`, `backend: str`, `qcs: list`, `shots: int`, `sim_method: str`, `seed: int` (optional, C-7) |

`qcs` elements are either Qiskit `QuantumCircuit` objects or OpenQASM 2.0
strings; the Python side parses strings (`QuantumCircuit.from_qasm_str`).
Returns `list[dict[str, int]]` — **one dict per circuit, in submission
order** (see C-3 for the dict format).

### `disconnect_from_infrastructure(infrastructure: str, **kwargs)`

For `"cunqa"`: single kwarg **`family`** (the handle returned by
`connect_to_infrastructure`). *Known break (audit C1): the current
implementation reads `slurm_job_id`. `family` is the canonical name; fix the
Python side, not the Rust side.*

### `expectation_values(counts: list[dict], fn) -> list[float]`

Returns exactly `len(counts)` finite floats, in order.

### Failure modes (all four functions)

- An unknown `infrastructure` value raises `ValueError` — never falls through
  to a default.
- An **unexpected** kwarg raises `TypeError` (the mirror of the
  "must-be-consumed" rule above: neither side silently drops or invents args).
- A missing **required** kwarg raises `TypeError`.

A failure **must never** cross the seam as a `pyo3_runtime.PanicException` or a
process abort (it used to: the Rust side `unwrap()`/`panic!`-ed on these calls).
The Rust orchestration layer now returns a typed `Result` on every path:

- A Python exception raised *by the seam function itself* (the three failure
  modes above, or any runtime error inside `run_qcs`) is **re-raised verbatim**,
  preserving its original type — so the `ValueError`/`TypeError` guarantees
  above hold unchanged.
- A failure originating *in the Rust layer* (backend construction, a native
  circuit that will not parse/simulate, the QMIO network path, a data
  conversion) raises a class from the `polypus` exception hierarchy:
  `PolypusError` (base) → `BackendError` → {`CunqaError`, `QmioError`,
  `NativeCircuitError`}, and `PolypusError` → `EvaluationError`. Catching
  `polypus.PolypusError` catches them all.

`disconnect_from_infrastructure` runs from `CunqaBackend`'s `Drop`, which **must
never panic**: a release failure is logged (`log::error!`) and recorded in the
process-wide `polypus.backend_cleanup_failures()` counter rather than raised
(see ENGINEERING.md §9). This is independent of the known break below, which is
about *which kwarg* the Python side reads.

**Enforcing test:** `tests/python/test_seam_contract.py` — runs in CI without
SLURM by monkeypatching the `polypus_python` seam (`run_qcs`) to force a
failure, asserting it surfaces as a typed Python exception (never a
`PanicException`) with the C-1 type preserved.

---

## C-2 · Gate vocabulary symmetry

The circuit vocabulary is:

```
h  x  y  z  s  t  sdg  tdg  rx  ry  rz  cx  cz  rzz  rxx  cp  u3(u/p/u1/u2 canonicalised)
barrier  measure  measure_all
```

**Invariant:** the four consumers/producers of this vocabulary — the OpenQASM
2.0 exporter (`qasm.rs`), the OpenQASM importer (`qasm_import.rs`), the native
simulator (`polypus-sim`) and the QIR exporter (`qir.rs`) — must all support
the **full set**, with **identical unitary semantics** (the native simulator is
the reference; QIR decompositions may differ only by a global phase).

Corollaries:

- **Canonical QASM form.** The exporter emits a single canonical form (fixed
  gate spelling, parameter formatting and declaration order), and the importer
  normalises to it. The round-trip guarantee is therefore:
  `to_qasm2(from_qasm2(to_qasm2(c)))` is **byte-identical** to `to_qasm2(c)`
  — i.e. output is a fixed point, without assuming arbitrary hand-written input
  is preserved byte-for-byte. Semantically, `from_qasm2(to_qasm2(c))` always
  reproduces the same instruction sequence and parameters as `c`.
- Adding a gate is a **five-place change** plus a row in the equivalence test —
  the OpenQASM exporter (`qasm.rs`), the importer (`qasm_import.rs`), the native
  simulator (`polypus-sim`), the QIR exporter (`qir.rs`) and the Python bindings
  (`crates/polypus/src/bindings/circuit.rs`, which expose the builder method).
  A PR adding it in fewer places must be rejected.
- **Non-finite parameters are rejected uniformly.** A `NaN` or infinite angle
  is never a valid parameter value: circuit construction and parameter binding
  reject it (`CircuitError::NonFiniteParam`), the OpenQASM importer rejects it
  at parse time (`CircuitError::Parse`), the QASM and QIR exporters refuse to
  serialise it, and the simulator rejects it (`SimError::NonFiniteAmplitude`).
  No producer may emit, and no consumer may accept, a non-finite parameter.

**Enforcing test:** parametric round-trip test over the whole vocabulary in
`crates/polypus-circuit/tests/contracts.rs`, plus the QIR-vs-simulator
unitary-equivalence test in `crates/polypus-sim/tests/contracts.rs`.

---

## C-3 · Measurement counts format

- Keys are **bitstrings** of width `num_clbits` (or `num_qubits` when the
  circuit has no measurements — full-register read-out convention).
- Bit order is **Qiskit little-endian**: qubit 0 is the least-significant
  (rightmost) character.
- `sum(counts.values()) == shots` requested for that circuit. When shots are
  distributed across `n` QPUs, the **total is conserved**: the remainder
  `shots % n` is spread over the first QPUs, never dropped.
- If several `measure` instructions write the same classical bit, the **last
  measurement wins** (OpenQASM 2.0 register semantics).

The per-circuit dict format above is unchanged by C-7: `run_quantum_circuit`
now returns that payload as the `counts` attribute of a `RunResult` wrapper
(`list[dict]` for a single-QPU run, a merged `dict` for `n_qpus > 1`), so
callers read `result.counts` rather than the bare value. The dict shape,
bit order and shot-conservation rule are exactly as specified here.

**Enforcing test:** shot-conservation assertion in the orchestration tests
(`crates/polypus/tests/running_quantum_circuits_local.rs`, plus the Python
public-API case in `tests/python/test_local_run.py`; audit C6) and
last-write-wins case in `polypus-sim` tests (to be added).

---

## C-4 · Measurement placement (terminal measurements)

Polypus circuits are straight-line programs with **terminal measurement**: no
gate may act on a qubit after that qubit has been measured, and each classical
bit is written at most once *(exception: the C-3 last-write-wins rule exists
only to define behaviour for hand-assembled circuits)*.

Backends and exporters **must reject** circuits that violate this with an
explicit error — never silently reorder, deduplicate or no-op the measurement.
The single shared check is `polypus_circuit::terminal_measurement_violation`,
enforced in the builder, the importer, the simulator and the QIR exporter.
Rationale and alternatives considered: see `docs/adr/0001-terminal-measurements.md`.

**Enforcing test:** rejection tests in
`crates/polypus-circuit/tests/contracts.rs` (builder, importer, QIR exporter)
and `crates/polypus-sim/tests/contracts.rs` (simulator).

---

## C-5 · Optimizer ↔ oracle contract (`polypus-optimizers`)

- `EvaluationOracle::evaluate_batch(candidates)` returns **exactly
  `candidates.len()` finite `f64` values**, in order; higher is better.
  Python-backed oracles must validate length before returning across the FFI.
- Preconditions validated with an error (not a panic): DE `population_size >= 4`;
  PSO/QNG `bounds.0 < bounds.1`; `dimensions >= 1`.
- Postcondition of every optimizer: `best_fitness` is the oracle's value **for
  the returned `best_params`** (audit C4).

**Enforcing test:** invariant test with multiple seeds in
`crates/polypus-optimizers/tests/` (the pure-Rust optimizer↔oracle invariant),
**plus** `tests/python/test_oracle_contract.py` for the Python-backed length and
finiteness guarantee: the Rust invariant test never exercises the
`polypus_python.expectation_values` return path, so it covers only one half of
the contract. The Python test forces a short return list and a non-finite value
through `polypus.train` and asserts each surfaces as a typed
`polypus.EvaluationError` (naming both lengths, or the offending index/value),
never a `pyo3_runtime.PanicException` and never a silently-poisoned result.

---

## C-6 · Version coherence

The workspace `Cargo.toml` version is the **single source of truth**. A release
tag `vX.Y.Z` must match it exactly, and the Python package version is derived
from it at build time. The release workflow refuses to publish when they
diverge.

*Historical note: tag and Cargo.toml diverged at 0.6.0 (see CHANGELOG). The
coherence check is enforced from 0.7.0 onwards; aligning the workspace version
is the first release action.*

**Enforcing check:** version-coherence step in `.github/workflows/hygiene.yml`.

---

## C-7 · Seeding & run manifest (Python entry points)

This contract governs the outer Rust↔Python boundary of the three public entry
points `polypus.run_quantum_circuit`, `polypus.train` and `polypus.qml.train`:
their `seed` kwarg and their return shape. (It is distinct from C-1, which
freezes the *internal* `run_qcs` seam to the `polypus_python` package.)

### The `seed` kwarg

- **`run_quantum_circuit(..., seed: int | None = None)`** seeds shot sampling
  across every *simulated* backend:
  - With `infrastructure="local"` (`backend="polypus"`, the native
    statevector simulator, or `backend="aer"`, Qiskit Aer): an explicit `seed`
    is used directly — the native backend seeds its own RNG in-process, and
    Aer receives it as the `seed` kwarg forwarded across the C-1 seam
    (`crates/polypus/src/infrastructure/local.rs`, which passes it to Aer's
    `seed_simulator` option in `polypus_python`'s `local.py`) — and
    reproduces the counts byte-for-byte across calls, verified against a real
    Aer install. `seed=None` draws a fresh seed from OS entropy, so repeated
    calls produce **independent** noise (never the run-`id`-derived
    repetition that motivated this contract). The run `id` is decoupled from
    the RNG — it is only a logging/temp-file/SLURM label.
  - With `infrastructure="cunqa"`: the same `seed` kwarg is forwarded across
    the same seam (`crates/polypus/src/infrastructure/cunqa.rs` mirrors
    `local.rs`, `polypus_python`'s `cunqa.py` mirrors `local.py`), with a
    per-QPU offset so distributed shots aren't identical copies. **This path
    is unverified** — the `cunqa` package isn't installed anywhere this can be
    tested, and reading CUNQA's actual source (CESGA-Quantum-Spain/cunqa) at
    the version README.md pins (`>= 2.3`) turned up API mismatches predating
    this contract (wrong module path, wrong kwarg names, a `.run()` method
    that may not exist on `QPU` objects at that version) — see the CUNQA
    integration follow-up.
  - With physical hardware (`infrastructure="qmio"`): passing an explicit
    `seed` raises `ValueError`. Real quantum processors rely on physical
    processes and cannot be deterministically seeded, so silently accepting
    a seed would give false confidence in reproducibility. `seed=None`
    behaves exactly as before.
- **`train(..., seed=None)` / `qml.train(..., seed=None)`** seed the optimizer's
  RNG and are **always accepted**, for every backend, because the seed's primary
  job (making population init / mutation deterministic, inside the pure-Rust
  `polypus-optimizers`) is independent of which backend evaluates the oracle.
  Precedence: the explicit `seed` kwarg wins; otherwise the `seed` field pinned
  on the `DE`/`PSO`/`QNG` instance; otherwise a fresh OS-entropy value. On the
  native, Aer, and CUNQA backends the resolved seed *also* seeds shot sampling,
  so a `train()` run on any of them is reproducible end-to-end; `qml.train` is
  Qiskit/Aer-only (native rejected), and since Aer shot noise is now seeded
  too, its reproducibility guarantee covers both the optimizer trajectory and
  Aer's sampling.

### The run manifest (return shapes)

- `run_quantum_circuit` returns a **`RunResult`** exposing:
  `counts` (the C-3 payload — `list[dict]` for one QPU, merged `dict` for
  `n_qpus > 1`), `id` (str), `seed` (`int | None`; the effective seed used, or
  `None` only for the `qmio` infrastructure), `backend` (str), `infrastructure`
  (str).
- `train` / `qml.train` return a **`TrainResult`** exposing the full
  optimization outcome — `best_params` (`list[float]`), `best_fitness` (float),
  `iterations_run` (int), `converged` (bool) — plus `seed` (int, the effective
  seed used) and `id` (str, the effective run id: the caller-supplied `id`
  prefix suffixed with a UUID v4 for uniqueness — a label for logging /
  SLURM / temp-file identification only, never for correlating runs by content;
  see #75). This replaces the former bare `list[float]`, which discarded
  fitness, iteration count and the convergence flag.

The effective `seed` on both result types is what lets a caller log a run and
replay it exactly.

**Enforcing test:** `tests/python/test_seed_reproducibility.py` (public-API
end-to-end: native and Aer reproducibility, entropy variation, the `qmio`
rejection, and the returned manifest/outcome fields), plus the Rust tests in
`crates/polypus/src/bindings/mod.rs` (native seed round-trip through
`run_quantum_circuit`, the `qmio` rejection path, and the seed-resolution
precedence / optimizer determinism) and `crates/polypus/src/infrastructure/native.rs`
(same-seed reproduces / omitted-seed differs at the backend level). CUNQA's
`seed` forwarding follows the same shape as Aer's on the Rust side
(`crates/polypus/src/infrastructure/cunqa.rs` mirrors `local.rs`) but has no
dedicated automated test and no verified-working status: per `ENGINEERING.md`
§3 the Rust suite is deliberately Python-runtime-free, so this seam can only
be tested from `tests/python/`, and the `cunqa` package isn't installed
anywhere in this project's CI or dev sandboxes (unlike Aer) — so, unlike the
Aer path, it has never actually been run. Treat CUNQA's `seed` support as
unverified until the CUNQA integration follow-up confirms it against a real
install.
