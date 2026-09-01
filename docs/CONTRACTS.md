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
| C-5 | Optimizer ↔ oracle | invariant test, multi-seed | ✅ present | DE `best_fitness` mismatch (C4) |
| C-6 | Version coherence | `hygiene.yml` version step | ✅ present | tag/Cargo diverged at 0.6.0 |
| C-7 | Seeding & run manifest | `tests/python/test_seed_reproducibility.py` + bindings/native Rust tests | ✅ present | repeated runs byte-identical / `train` seed hardcoded `None` (#34) |
| C-8 | QML problem ↔ oracle | `polypus-qml` `tests/contracts.rs` + `NativeQmlOracle` test in `polypus` | ✅ present | — |

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

**Out of scope — exact mode.** The native `qml.train` exact path
(`exact=True`, native Model+Dataset only — design doc §17) does **not** produce
`counts`: it reads exact basis-state probabilities (`|amplitude|²`) straight
off the statevector via `NativeStatevectorBackend::run_circuits_exact`, an
inherent method that never touches the `QuantumBackend` trait. C-3's guarantees
(`sum(counts.values()) == shots`, shot-conservation across QPUs) simply do not
apply there — there is no shot budget and nothing is sampled. The guarantee
that *does* hold in its place is **byte-for-byte determinism given the same
circuit, with no seed required** (there is no sampling RNG to seed): the
expectation summation is performed in a fixed basis order so it is independent
of `HashMap` iteration order, delivering the C-7 reproducibility promise
without a seed. This mode is exclusive to the pure-statevector backend;
"exact" has no physical meaning for a noisy Aer backend or real hardware
(QMIO/CUNQA), which is why it is native-only and rejected elsewhere.

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
  PSO/QNG `bounds.0 < bounds.1`; `dimensions >= 1`; Adam/QNG `max_iters >= 1`.
  DE/PSO need no `generations >= 1` check: they evaluate the initial population
  before their generation loop, so `generations == 0` still yields a real
  `best_fitness`. Adam/QNG are single-point with no such pre-loop evaluation, so
  `max_iters == 0` would return the unevaluated random initial `θ` with the
  `-inf` sentinel, violating the postcondition below.
- Postcondition of every optimizer: `best_fitness` is the oracle's value **for
  the returned `best_params`** (audit C4).
- **Additive:** `OptimizationOutcome` carries a `fitness_history: Vec<f64>`
  beside `best_fitness` — the convergence curve of the run. No existing field,
  guarantee or signature changes; every optimizer simply reports the incumbent
  best it already tracked internally. Its guarantees:
  - `fitness_history.len() == iterations_run`, on the early-stopping paths
    included (DE's/PSO's population collapse, QNG's/Adam's `patience` streak) —
    the entry is recorded before the `break`, never after it. It is empty
    exactly when `iterations_run == 0`.
  - `*fitness_history.last() == best_fitness`: both are read from the same
    incumbent-best variable, not recomputed.
  - **Monotonically non-decreasing for all four optimizers.** Each entry is the
    running best — DE's post-selection champion `fitness[argmax]`, PSO's global
    best over the personal bests, QNG's/Adam's `best_energy` *after* its
    `if energy > best_energy` update — and **not** the fitness of that
    iteration's current candidate, which gradient ascent lets oscillate freely.
    The monotonicity is structural (it is a running maximum), so it holds
    whatever the oracle returns: a shot estimate, or a minibatch estimate. What
    a noisy oracle changes is how much each entry *means*, never the shape of
    the sequence.
- `GradientOracle::gradient_batch(theta, dims)` (QNG and Adam) returns the fitness
  gradient `∂fitness/∂θ`, **exactly `dims` values**, in order, same ascent-sign
  convention as `EvaluationOracle` (higher fitness is better; the value points
  uphill). QNG length-checks it like any oracle output. "Exact" is the
  parameter-shift identity in the **noiseless limit**: evaluated over finite
  `shots` it is an unbiased *estimator* of the true gradient, not a noise-free
  value. Exactness is the caller's guarantee (the crate cannot see circuits or
  gate generators), exactly as the QFIM-diagonal `VarianceOracle` is. The free
  function `linear_parameter_shift_gradient` builds it for any oracle whose
  fitness is linear in the shifted expectations (raw expectation or unweighted
  mean, no nonlinear loss on top); an oracle that composes a nonlinear loss over
  per-sample expectations needs the chain rule instead (see
  `polypus_qml::QmlProblem::param_gradient`). That helper returns
  `Result<Vec<f64>, OptimizerError>`: it submits `2 * dims` shifted candidates
  and reads each parameter's ±π/2 pair positionally out of the result, so it
  length-checks the batch against `2 * dims` — as every optimizer checks its own
  batch calls — rather than trusting the oracle it was handed, which is any
  implementation of a public trait. A `GradientOracle` that delegates to it
  therefore has a failure to handle that its own trait method cannot return: the
  two `polypus` oracles that delegate (`QmlOracle`, `VqcOracle`) record it in
  their shared `OracleErrorSlot` and return finite sentinels, the same discipline
  `evaluate_batch` already follows, so the entry point surfaces it after
  `optimize` returns.
- **Breaking change:** `AlgorithmQNGArgs` no longer accepts
  `finite_difference_step` (and the Python `QNG` pyclass no longer exposes it).
  QNG now consumes the exact parameter-shift gradient above instead of a
  central finite-difference stencil — a deliberate contract change with no
  compatibility shim.
- **Breaking change:** `AlgorithmQNGArgs`/`AlgorithmAdamArgs` require a
  `patience: usize` field, and the Python `QNG`/`Adam` pyclasses expose it as a
  kwarg defaulting to **`3`**. Gradient-norm early stopping now needs `patience`
  **consecutive** iterations with `‖∇fitness(θ)‖₂ < tolerance` before reporting
  `converged = true`, where it previously needed one. The counter increments on
  each sub-tolerance iteration and **resets to `0`** on any iteration that is
  not, so three sub-tolerance iterations scattered among larger ones never stop
  a run at `patience = 3`. `patience = 1` reproduces the previous
  single-iteration rule exactly (pinned by its own test); `0` behaves as `1`,
  since the streak is only tested after an iteration that was itself below the
  tolerance. Runs whose gradient norm decreases smoothly are unaffected beyond
  spending the extra iterations it takes to make the streak consecutive.
  A deliberate behaviour change for **every** caller — minibatched or not,
  because the optimizer cannot tell the two apart — with no compatibility shim.
  Motivation and limits: the minibatch note below.

**Enforcing test:** invariant test with multiple seeds in
`crates/polypus-optimizers/tests/`.

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
  so a `train()` run on any of them is reproducible end-to-end.

  `qml.train` accepts **both** a Qiskit feature map and a native
  `polypus.qml.Model`, dispatching on the type of its first argument: a `Model`
  takes the native (pure-Rust) path, anything else the original Qiskit/Aer path.
  The **native path is reproducible byte-for-byte on any simulated backend**
  (`polypus`, `aer`, or `cunqa`) given the same seed: the `NativeQmlOracle`
  builds circuits deterministically from the `QmlProblem` (C-8) and evaluates
  candidates **concurrently** (one Tokio `spawn_blocking` task per candidate).
  Reproducibility no longer depends on the order in which candidates are
  evaluated, because `NativeStatevectorBackend` derives each circuit's shot-
  sampling seed **solely from that circuit's own content** (an FNV-1a hash of its
  OpenQASM 2.0 text) plus its position within a batch, added to the resolved base
  seed — never from any mutable state shared between calls. Concurrent candidates
  therefore can never race for seed assignment. On Aer the same resolved seed is
  forwarded across the C-1 seam to `seed_simulator`, which carries no shared state
  of its own. The **Qiskit path is
  unchanged**, and with Aer's shot noise seeded too its reproducibility covers
  both the optimizer trajectory and Aer's sampling. `qmio` still rejects an
  explicit `seed` exactly as `run_quantum_circuit` does (real hardware cannot be
  seeded) — unchanged by this phase. CUNQA's `seed` for the native path shares
  Aer's shape on the Rust side but is **unverified**, for the same reason CUNQA
  is flagged unverified elsewhere in this contract: the `cunqa` package isn't
  installed anywhere this can be tested.

  **Minibatching (`qml.train(..., batch_size=N)`, native path only, design doc
  §17)** does not weaken this reproducibility. Each oracle call
  (`evaluate_batch` or `gradient_batch`) draws its own minibatch from `seed`
  combined with a per-oracle call counter that advances by exactly one per call
  — the same counter is shared by both, not one each, so a given iteration's
  fitness-tracking call and its gradient call consume two different counter
  values and may see *different* minibatches (accepted); two independent
  counters would instead put both calls of the same iteration on the same
  value and make them see the *same* minibatch every time. Because the
  derivation is `seed` + a deterministic counter, two runs with the same
  seed and `batch_size` draw the identical sequence of minibatches and reproduce
  byte-for-byte. Within a single `gradient_batch` call, all `dims` parameters and
  both `θ±π/2` shifts share **one** minibatch — a correctness requirement of
  parameter-shift, not a reproducibility nicety. The reported `best_fitness`
  is **not** a minibatch estimate: after the optimizer loop ends it is recomputed
  **once** against the full dataset (via the oracle's inherent `evaluate_full`),
  so the C-5 guarantee that `best_fitness` is the fitness of `best_params` holds
  against the whole training set, exactly as for a non-minibatch run — the
  minibatch is only the cheap per-iteration heuristic that steers the search.

  **`fitness_history` is the one field that recompute deliberately leaves
  alone**, so under `batch_size` it is the only place C-5's
  `fitness_history[-1] == best_fitness` does not hold. The curve keeps the
  per-iteration *minibatch* estimates the optimizer actually steered by, which is
  what a convergence curve is for; splicing the full-dataset number onto its last
  point would mix two scales and — since a minibatch estimate is typically the
  rosier of the two — could make the sequence decrease, trading C-5's
  monotonicity guarantee for a cosmetic endpoint. Length and monotonicity hold
  under `batch_size` exactly as everywhere else; read `best_fitness` for the
  honest full-dataset number, as this contract already says.

  **Minibatching interacts badly with gradient-norm early stopping — evaluated,
  reproduces, mitigated by `patience` (not eliminated).** `AlgorithmQNG` and
  `AlgorithmAdam` set `converged` from that iteration's `‖∇fitness(θ)‖₂` against
  `tolerance`; under `batch_size` that is a *minibatch* gradient, so a minibatch
  whose samples cancel each other can stop the run at an arbitrary point. This is
  measured rather than suspected. It is **not** driven by shot noise: for a hinge
  loss, two samples with identical features and opposite labels have their loss
  derivatives cancel term by term, so a minibatch of exactly that pair has
  gradient **exactly** zero while the full-dataset norm is O(0.1–0.5) — two
  orders of magnitude above the `tolerance = 0.01` that `polypus.Adam`/
  `polypus.QNG` default to. Consequences, all reproducible from the public API
  (`test_qml_native.py::TestNativeQmlTrainMinibatch`, plus the norms themselves
  in `exact_native_qml_oracle.rs`), stated for the pre-`patience` rule that
  `patience = 1` still selects:

  - the run stops on iteration 1 with `converged = True`, having barely moved
    from its random initialization, where the same configuration without
    `batch_size` uses all its iterations and reaches the optimum;
  - lowering `tolerance` is **not** a mitigation — the norm is exactly zero, so
    it falls below any threshold, `1e-12` included;
  - QNG and Adam behave identically, since the rule lives in the convergence
    check rather than in one optimizer;
  - the trigger is *cancellation within one minibatch*, not minibatching as
    such: on a separable dataset with no contradictory samples, no seed in a
    ten-seed sweep stopped early.

  **C-5 was never violated.** `best_fitness` remains the full-dataset fitness of
  the returned `best_params` thanks to the final recompute above, so the reported
  number stayed honest throughout; what a spurious stop costs is the optimization
  itself and the meaning of the `converged` flag.

  **The fix is the `patience = 3` default** documented under C-5 above: a stop
  now needs three *consecutive* sub-tolerance iterations, so one cancelling
  minibatch no longer ends the run. It is applied uniformly to every QNG/Adam
  run rather than only under `batch_size`, because the optimizer has no way to
  tell the two apart — `GradientOracle::gradient_batch` deliberately hides
  whether its result came from a minibatch or the full dataset.

  **`patience` reduces the probability; it does not eliminate the failure.**
  Nothing stops the same cancelling minibatch from being redrawn `patience`
  times in a row, and that is demonstrated, not conceded: on the six-sample
  dataset with one contradictory pair, `batch_size=2`, `max_iters=60`, spurious
  stops fall from **194/200 seeds at `patience=1` to 39/200 at `2` and 4/200 at
  `3`** — a ~50× reduction with four survivors. `seed=140` is one, stopping on
  iteration 8 at fitness ≈ −0.94 where the run should reach ≈ −0.34, and
  `test_patience_is_a_probabilistic_mitigation_not_a_guarantee` pins it. So
  `converged = True` from a minibatched gradient-optimizer run is *much* more
  trustworthy than before but still not a guarantee: `best_fitness` remains the
  number to read, and a small `batch_size` over contradictory labels remains the
  configuration to avoid.

### The run manifest (return shapes)

- `run_quantum_circuit` returns a **`RunResult`** exposing:
  `counts` (the C-3 payload — `list[dict]` for one QPU, merged `dict` for
  `n_qpus > 1`), `id` (str), `seed` (`int | None`; the effective seed used, or
  `None` only for the `qmio` infrastructure), `backend` (str), `infrastructure`
  (str).
- `train` / `qml.train` return a **`TrainResult`** exposing the full
  optimization outcome — `best_params` (`list[float]`), `best_fitness` (float),
  `fitness_history` (`list[float]`, the convergence curve: one
  best-fitness-so-far per iteration run, with C-5's length / last-entry /
  monotonicity guarantees), `iterations_run` (int), `converged` (bool) — plus
  `seed` (int, the effective seed used) and `id` (str, the effective run id: the caller-supplied `id`
  prefix suffixed with a UUID v4 for uniqueness — a label for logging /
  SLURM / temp-file identification only, never for correlating runs by content;
  see #75). This replaces the former bare `list[float]`, which discarded
  fitness, iteration count and the convergence flag.
- **`qml.train`'s return type follows the path it dispatched to** (design doc
  §17), the same way its accepted kwargs already do — `x_train` /
  `expectation_function` on the Qiskit path versus `loss` / `batch_size` on the
  native one. This is an extension of that existing asymmetry, not a new
  inconsistency:
  - a native `polypus.qml.Model` (+ `Dataset`) returns a
    **`polypus.qml.QmlTrainResult`**: the seven `TrainResult` fields above, with
    identical names, types and meanings, plus **`trained_model`** — a
    ready-to-use `polypus.qml.TrainedModel` (the model compiled against the
    dataset's feature count and bound to `best_params`), built eagerly at the end
    of training. So `qml.train(...).trained_model.predict(x_new, ...)` is the
    whole train→predict flow, where before a caller had to rebuild
    `TrainedModel(model, dataset, result.best_params)` by hand, passing back in
    the two objects the `train` call already had.
  - a Qiskit feature map returns a plain **`TrainResult`**, unchanged: that path
    has no `Model`/`Dataset` to wrap.
  - `QmlTrainResult` is **not** a subclass of `TrainResult` — two independent
    pyclasses, so `isinstance(result, TrainResult)` is `False` on the native
    path. Read the fields, not the type. `polypus.train` (the generic entry
    point) is untouched and always returns `TrainResult`.
  - **`Model.train(dataset, ...)`** is the method spelling of that native path
    and therefore returns the same `QmlTrainResult`. It is the *same* run:
    identical arguments and seed produce a byte-identical result either way,
    since the method delegates to the same native implementation. Its kwargs are
    `qml.train`'s minus `ansatz` (the dataset takes that positional slot) and the
    two Qiskit-path ones, and `method` is a required positional argument there.
    `qml.train` remains the only entry point for a Qiskit feature map, which has
    no `Model` to hang a method off.

The effective `seed` on every one of these result types is what lets a caller log
a run and replay it exactly.

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
install. The per-path return shapes of `qml.train` are pinned by
`tests/python/test_qml_native.py` section 13
(`TestQmlTrainResultNativePath`, `TestQmlTrainResultTrainedModelEquivalence`,
`TestQiskitPathStillReturnsTrainResult`): the native path's type and fields, the
bit-for-bit equality between `result.trained_model`'s predictions and a
hand-built `TrainedModel`'s, and the Qiskit path still returning `TrainResult`.
`fitness_history` is covered on every one of those paths — the generic `train`
(`test_vqc.py::TestTrainFitnessHistory`, DE and Adam), the native path and its
`batch_size` exception (`test_qml_native.py` section 13), the Qiskit path
(`TestQiskitPathStillReturnsTrainResult`), `Model.train`'s field-by-field
agreement (section 14) and same-seed reproducibility of the whole curve
(`test_seed_reproducibility.py`) — with the per-optimizer guarantees themselves
pinned in `crates/polypus-optimizers/tests/optimizers.rs`.

---

## C-8 · QML problem ↔ oracle (`polypus-qml`)

This contract governs the seam between a [`QmlProblem`] (the trainable object
`polypus-qml` produces) and the evaluation oracle that scores it. Like C-5, it
is a **Rust-to-Rust** contract — no Python is involved — and it is what lets
`crates/polypus`' native QML oracle drive the pure-Rust optimizers without
either side knowing the other's internals.

A `QmlProblem` bundles a compiled model, a training set and a loss, and exposes
exactly the pair of operations an oracle needs: *bind these parameters into
circuits* and *here are the counts, give me the fitness*. Everything between —
backend, shots, batching, distribution — is `crates/polypus`' concern. The four
guarantees:

- **(a) `bind_batch` shape and order.** `bind_batch(θ)` returns exactly
  `num_circuits()` `ConcreteCircuit`s in a **stable sample-major order** (one
  circuit per training sample; full multi-circuit X/Y base grouping — one
  circuit per base group — is still future, see the readout-basis note below).
  Counts handed back to `fitness_from_counts` must be in the **same order** —
  misaligned counts are a corrupt result, so the order is specified and tested,
  never assumed.
- **(b) Finite fitness or typed error.** `fitness_from_counts` returns a finite
  `f64` (`= −mean_loss`, since the optimizers maximise) or a typed `QmlError` —
  **never `NaN`**. `BinaryCrossEntropy` clamps its probability away from the
  `log` singularities, and non-finite predictions are already impossible
  upstream (the dataset and circuits reject them). Length-mismatched counts are
  a typed `QmlError::CountsLengthMismatch`, not a panic.
- **(c) Emitted circuits obey C-4 and C-2.** Every circuit `bind_batch`
  produces satisfies C-4 (terminal measurement) and uses only the C-2 gate
  vocabulary — guaranteed by construction, since the model emits exclusively
  through `polypus-circuit`'s builder.
- **(d) `num_params()` is stable and positive.** `num_params()` is fixed at
  `compile` time, always `> 0` (a model with no trainable parameters is
  rejected with `ValidationError::NoTrainableParams`), and is the `dimensions`
  the optimizer consumes under C-5. `polypus.qml.Model.num_params()` reads that
  same number from Python before any training happens — it compiles a clone of
  the builder, so it inherits this guarantee verbatim, positivity included: an
  ansatz-free model raises `NoTrainableParams` rather than reporting `0`.

**Readout observables from Python (`Model.readout`).** Each element of
`observables` is **one of three** forms:

- a bare list of `(pauli, position)` factors — one Pauli string with implicit
  coefficient `1.0`;
- a `polypus.qml.PauliTerm`, built by the free functions
  `polypus.qml.Z/X/Y(position)` and multiplied with `@`: `Z(0)` and
  `Z(0) @ Z(1)`. This is the *exact* equivalent of the bare form — one term,
  coefficient `1.0` — so `Z(0)` builds the identical observable as `[("z", 0)]`
  and `Z(0) @ Z(1)` the identical one as `[("z", 0), ("z", 1)]`, guaranteed
  structurally: both go through one shared helper on the bindings side. A
  `PauliTerm` carries **no** coefficient and is **never** a sum — it has no `+`,
  `*` or `__rmul__`;
- a `polypus.qml.Observable`, the weighted sum `O = Σ cᵢ·Pᵢ` the Rust
  `Observable` has always supported (`Observable([(0.5, [("z", 0)]), (1.5,
  [("z", 0), ("z", 1)])])` is `0.5·Z₀ + 1.5·Z₀Z₁`) — the only form that can
  carry coefficients or several terms. Each term of an `Observable` is exactly
  the same `(pauli, position)` list the bare form is, so
  `Observable([(1.0, [("z", 0)])])` builds the identical observable as
  `[("z", 0)]`.

The two dedicated types are **additive**, not replacements: the bare form's
meaning is unchanged, and all three spellings may be mixed inside one `readout`
call (a multiclass `"argmax"` may spell one class bare, one as a `PauliTerm` and
one weighted). They are distinguished by **type**, in order of decreasing
specificity — `Observable`, then `PauliTerm`, then anything else extracted as the
bare form — never by guessing at the shape of the tuples, so no form can be
silently reinterpreted as another. A non-finite coefficient is a `ValueError`
(`ValidationError::NonFiniteCoefficient`, reporting the offending term index), as
are an unknown Pauli and a position repeated inside one term; an element that is
none of the three forms is a `TypeError` naming all three.

A repeated position is rejected **eagerly** in the `PauliTerm` form: `Z(0) @ Z(0)`
raises at the `@`, before any `readout` call, where the bare form's equivalent
`[("z", 0), ("z", 0)]` can only be caught once `readout` parses it. Same error,
earlier.

**String-typed kwargs and their named constants.** Nine kwargs on the Python QML
surface take a string from a closed set, each parsed strictly (an unrecognised
value is a `ValueError` listing the options):

| kwarg | accepted values | namespace |
| --- | --- | --- |
| `Model.angle_encoder(axis=)`, `Model.hardware_efficient(rotations=[…])` | `"rx"`, `"ry"`, `"rz"` | `polypus.qml.Axis` |
| `Model.readout(decision=)` | `"sign"`, `"threshold"`, `"argmax"`, `"raw"` | `polypus.qml.Decision` |
| `qml.train(loss=)`, `Model.train(loss=)` | `"squared_error"`, `"binary_cross_entropy"`, `"hinge"`, `"categorical_cross_entropy"` | `polypus.qml.Loss` |
| `Model.hardware_efficient(entanglement=)`, `Model.iqp_encoder(entanglement=)` | `"linear"`, `"circular"`, `"full"` | `polypus.qml.Entanglement` |
| `Model.hardware_efficient(entangler=)` | `"cx"`, `"cz"` | `polypus.qml.Entangler` |
| `Model.conv(block=)` | `"basic"`, `"cartan"` | `polypus.qml.ConvBlock` |
| `Model.conv(pairing=)` | `"even_pairs"`, `"odd_pairs"`, `"alternating"` | `polypus.qml.Pairing` |
| `Model.pool(block=)` | `"basic"` | `polypus.qml.PoolBlock` |
| `Model.pool(keep=)` | `"even_positions"`, `"odd_positions"` | `polypus.qml.KeepRule` |

The namespaces in the last column are **additive and nothing more**: each
constant *is* the plain string (`polypus.qml.Axis.RY == "ry"`), so the two
spellings are interchangeable everywhere and the plain string remains the only
accepted *type*. They exist so the vocabulary is discoverable —
`dir(polypus.qml.Loss)`, editor completion, a docstring — rather than only
recoverable from a `ValueError` after guessing wrong. No parser, no signature and
no default changes because of them.

`ConvBlock` and `PoolBlock` are deliberately separate namespaces even though both
spell a block `"basic"` today: they stand for two independent Rust enums, and
`Model.pool` does not accept a conv block (`pool(block=ConvBlock.CARTAN)` is the
same `ValueError` as `pool(block="cartan")`).

The constants are ordinary class attributes on `#[pyclass]` types with no
constructor, so `polypus.qml.Axis()` is a `TypeError` — a namespace, never a
value. They are **not** immutable: PyO3 builds mutable heap types, so
`polypus.qml.Axis.RY = …` succeeds and corrupts the vocabulary process-wide. That
is a foot-gun, not a supported operation.

**Readout measurement basis (single group).** A readout may measure `X`/`Y`
Paulis, not just `Z`: `compile` inserts the basis change (`H` for `X`; `Sdg`
then `H` for `Y`) before the terminal measurement. This is supported **only when
the whole readout resolves to exactly one basis group** — a qubit→Pauli
assignment every observable's every term agrees with on the qubits it touches
(untouched qubits measure in `Z`, no gate). Any all-`Z` readout, or one whose
observables share a compatible non-`Z` basis (e.g. a multiclass `Argmax` with
every class in `X`), stays a single circuit per sample and keeps `num_circuits`
unchanged. Two conflicting bases **inside one observable** (e.g. `Z₀ + X₀`) is a
typed `ValidationError::ObservableHasIncompatibleBases`; a readout that would
need **more than one group** (e.g. one class in `Z`, another in `X` on the same
qubit) is a typed `ValidationError::ReadoutNeedsMultipleBasisGroups { groups }`
— the multi-circuit case is not implemented yet (design doc §7.2), and is
rejected rather than mismeasured.

**Enforcing test:** `crates/polypus-qml/tests/contracts.rs` covers the QML side —
`bind_batch` length/order, C-4/C-2 on every bound circuit, finite fitness, and
`num_params()` matching the compiled model — and
`crates/polypus/src/evaluation/native_qml_oracle.rs` covers the **oracle half**:
the `NativeQmlOracle` that wires this problem to the C-5 optimizers returns one
finite fitness per candidate, yields the finite `0.0` sentinel while recording
the real failure in its shared `OracleErrorSlot` when a candidate cannot be
evaluated, and is byte-reproducible for a fixed seed. The public-API path is
exercised end-to-end from Python in `tests/python/test_qml_native.py`.

### Model save format (design doc §17)

A trained model is persisted as JSON — `polypus.qml.TrainedModel(model,
dataset, theta).save(path)` / `.load(path)`, with the file I/O and the concrete
JSON format living in `crates/polypus` (the pure `polypus-qml` crate only gains
an optional `serde` feature deriving `Serialize`/`Deserialize` on the model
tree). A `CompiledModel` serializes to **only** `{spec, num_features}`: its
derived fields (`num_params`, per-layer allocations, resolved readout) are never
written. Loading recompiles from those two fields via `QuantumModel::compile`,
so the derived state is regenerated, never trusted from the file.

The safety guarantee this buys: a corrupt or hand-tampered file can never
produce an internally inconsistent model. Loading re-runs the full `compile`
validation, so a spec that no longer compiles surfaces as a clean
deserialization error (a `ValueError` at the Python boundary) instead of a panic
or a silently-accepted broken model. `serde_json`'s `float_roundtrip` feature is
enabled wherever this JSON is parsed, so a saved `theta` reloads bit-for-bit and
the loaded model's predictions reproduce under C-7.

**End-to-end inference — `predict(X, ...)`.** `TrainedModel.predict(X, ...)` is
the one-call inference path: given a batch of **new** samples `X`
(`List[List[float]]`, one row per sample), it binds each to the trained `theta`,
runs it on a backend, and applies the model's readout decision, returning one
prediction per sample in `X`'s order. Seed resolution and backend construction
match `run_quantum_circuit` (direct `seed.unwrap_or_else(random_seed)`; `qmio`
rejects an explicit seed), and it reuses the **exact same** `exact=True` guard as
`qml.train` — exact mode requires `infrastructure="local"` and the native
`backend="polypus"`, and any other combination is rejected rather than silently
ignored. `predict_from_counts` remains the lower-level entry for a caller who
obtained counts themselves.

[`QmlProblem`]: ../crates/polypus-qml/src/problem.rs
