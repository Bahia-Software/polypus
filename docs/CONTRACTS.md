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
| C-1 | Rust → Python execution | `tests/python/test_seam_contract.py` | ✅ present | `disconnect` now forwards `family` to `qdrop` (C1 fixed); local `run_qcs` ignores the `backend` kwarg (LOCAL-2, open — see below) |
| C-2 | Gate vocabulary symmetry | `polypus-circuit` + `polypus-sim` `tests/contracts.rs` | ✅ present | — |
| C-3 | Measurement counts format | shot-conservation + key order + last-write-wins | ✅ present | shots dropped on uneven distribution (C6); the native `polypus` backend OR-ed repeated writes to one classical bit instead of letting the last win (#205, fixed) |
| C-4 | Terminal measurement placement | `polypus-circuit` + `polypus-sim` `tests/contracts.rs` | ✅ present | — |
| C-5 | Optimizer ↔ oracle | invariant test, multi-seed + `tests/python/test_oracle_contract.py` | ✅ present | DE `best_fitness` mismatch (C4) |
| C-6 | Version coherence | release-workflow check (planned; see §C-6) | ⚠️ planned (0.7.0) | tag/Cargo diverged at 0.6.0 |
| C-7 | Seeding & run manifest | `tests/python/test_seed_reproducibility.py` (+ `test_qml_predict.py`) + bindings/native Rust tests | ✅ present | repeated runs byte-identical / `train` seed hardcoded `None` (#34) |
| C-8 | qml.train row/dimension/label symmetry | `tests/python/test_qml_train_validation.py` (+ `test_qml_supervised.py`, `test_qml_predict.py`) | ✅ present | silent row truncation / late Qiskit error (#79) |
| C-9 | `id` charset (train/qml.train) | `tests/python/test_id_validation.py` | ✅ present | unvalidated `id` reached SLURM `family_name` / temp files / log streams (#89) |
| C-10 | OpenQASM 3 profile | `polypus-circuit` `tests/qasm3.rs` + `polypus-sim` `tests/qasm3_semantics.rs` + `tests/python/test_qasm3*.py` + `fuzz/fuzz_targets/from_qasm3.rs` | ✅ present | — |

⏳ contracts are specified but not yet mechanically enforced; treat them as
review-enforced until the test lands. Each known break has a public issue
labelled `audit-2026-07`.
<!-- TODO(maintainers): create those issues and link them here; flip ⏳ to ✅
     as each test merges. -->

---

## C-1 · Rust → Python execution seam (`polypus_python`)

The Rust infrastructure layer (`crates/polypus-infrastructure/src/*.rs`) calls
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

`family_name` is the run id (`RunParams::id`, derived from `ExecutionConfig::id`
at backend construction), whose charset is constrained by C-9.

### `run_qcs(infrastructure: str, **kwargs)`

| backend | kwargs (exact names) |
|---|---|
| local | `id: str`, `backend: str`, `qcs: list`, `shots: int`, `sim_method: str`, `max_parallel_experiments: int`, `noise_model` (optional), `seed: int` (optional, C-7) |
| cunqa | `family_id: str`, `backend: str`, `qcs: list`, `shots: int`, `sim_method: str`, `seed: int` (optional, C-7) |

`id` / `family_id` are the run id (`RunParams::id`, derived from
`ExecutionConfig::id`), whose charset is constrained by C-9. `qcs` elements are
either Qiskit `QuantumCircuit` objects or OpenQASM 2.0 strings; the Python side
parses strings (`QuantumCircuit.from_qasm_str`).
Returns `list[dict[str, int]]` — **one dict per circuit, in submission
order** (see C-3 for the dict format).

*(Known break, audit LOCAL-2, open: the `local` path's `run_qcs` receives the
`backend` kwarg the Rust side sends but never consumes it — the Python side
hard-codes `AerSimulator`. This is the same "must-be-consumed" violation as
`cores_per_qpu` above. Surfaced by the Fase-5 conformance run (see
`docs/backends.md` §"Conformance of the built-in backends"); recorded here and not
yet fixed, because the fix lives in the `polypus_python` seal, outside that phase's
module scope. Its resolution is either to consume the kwarg on the Python side or to
stop sending it from `local.rs`.)*

### `disconnect_from_infrastructure(infrastructure: str, **kwargs)`

For `"cunqa"`: single kwarg **`family`** (the handle returned by
`connect_to_infrastructure`), forwarded to CUNQA's `qdrop`. *(Historical break,
audit C1, now fixed: the Python side used to read `slurm_job_id` — a key the
Rust side never sends — so a `KeyError` fired before `qdrop` ran and the QPU
allocation leaked. The Python side was corrected to read `family`; the Rust
side was already canonical.)*

### `expectation_values(counts, fn)` — *no longer part of this seam*

*Historical: this was once the fourth seam function — the Rust layer handed the
counts back to Python to reduce them to expectation values. Expectation
computation is now **native**: `polypus-observable` evaluates declarative
QUBO/Ising costs, and `polypus-evaluation`'s `PyCallbackObservable` calls a user
callback once per unique bitstring in a single GIL section and aggregates in
Rust. The Rust layer no longer calls `expectation_values` across the seam, so it
is not frozen by this contract. The function still lives in
`polypus_python/qaoa_utils.py` for Python-side use.*

### Failure modes (all three functions)

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
h  x  y  z  s  t  sdg  tdg  id  u0  sx  sxdg  rx  ry  rz  p  u1  u2  u  u3  (U → u)
cx  (CX → cx)  cz  cy  ch  csx  swap  rzz  rxx  cp  cu1  crx  cry  crz  cu3  cu
ccx  cswap  rccx  rc3x  c3x  c3sqrtx  c4x
calls of gates declared with `gate` blocks
barrier  measure  measure_all
```

This is all of Qiskit's `qelib1.inc`. Gates outside it (`ryy`, `rzx`, `ecr`,
`iswap`, `xx_plus_yy`, `mcx`, …) reach Polypus the way Qiskit's exporter writes
them — declared with `gate` blocks — and are handled as declared gates. The
OpenQASM 3 profile (C-10) reads and writes the same vocabulary: `stdgates.inc`
names a subset of it, and the rest is written through gate definitions.

**One instruction per statement, re-emitted under the same name.** The
importer never decomposes: a `ccx` statement is one `Ccx` instruction and is
exported as `ccx` again, with its operands in the same order, so a benchmark
file reaches a backend (e.g. Aer, through the exporter) as the same program —
same gate count, same depth, same instruction names. The only spelling changes
are the language builtins `U` → `u` and `CX` → `cx`, which Qiskit names `u`
and `cx` itself, so no Qiskit consumer can tell them apart; the OpenQASM 3
profile also reads `phase` as `p` and `cphase` as `cp`, and writes `u` as the
builtin `U` (C-10). An instruction `stdgates.inc` lacks is written in OpenQASM
3 as a call of a gate the output defines with the same matrix, under the
instruction's name, and reads back as that declared gate (a declared gate is
never recognised as a built-in). Decomposition is allowed at exactly two
*lowering* boundaries, both confined to their module and invisible to the
OpenQASM exporters: the native simulator, for gates without a dedicated
kernel (`ccx`, `cswap`, `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x` — through their
exact `qelib1.inc` definitions, `GateInstruction::lowering` — and calls of
declared gates), and the QIR exporter, for gates without a base-profile
intrinsic.

`cu1` and `cp` are the same operator but distinct instructions: each keeps its
own spelling through import and export (neither is normalised into the other).
Likewise `p`/`u1` (one operator), and `u`/`u3` (one operator) with `u2`: every
spelling is its own instruction, re-emitted as written.

`id` is an instruction like any other, never a no-op to drop: it is imported,
exported and round-tripped one-to-one, so the gate count and depth of an
imported circuit match the source program (and what Qiskit computes for it).
Only the QIR lowering drops it (there is no identity intrinsic); the simulator
applies it as the identity. As a unitary it is subject to C-4.

**Declared gates.** A `gate` declaration, in OpenQASM 2.0 or in the
OpenQASM 3 profile, is kept as a definition (a template over its formal
arguments, plus its source text) and each call of it is *one* instruction,
`GateInstruction::Custom`, a unitary on all its qubits for C-4. A call's angles may be free parameters or expressions
of them (`CustomGate::with_arguments`); such a call is checked through its whole
body, nested declarations included, whenever its parameters are bound —
binding and the exports that take parameter values report `NonFiniteParam` or
`DivisionByZero` there — just as a call with fixed angles is checked when it is
created. The OpenQASM 2.0 exporter re-emits an OpenQASM 2.0 declaration
verbatim (only line endings normalised: every run of carriage returns before a
line feed dropped, so CRLF becomes LF) plus the call — never the expanded body —
so a backend that parses the export builds the same program as from the
original file; every other export prints a definition from its body (C-10).
Canonical OpenQASM 2.0 form: the declarations the circuit reaches (directly or
through other declarations) are emitted right after the include — first those
imported from OpenQASM 2.0, in source order, then those imported from
OpenQASM 3, printed, callees first in order of first use; unreachable
declarations are not re-emitted. Expansion into built-in
instructions is a lowering step of the simulator and the QIR exporter only.
Redeclaring a gate, or declaring one with a `qelib1.inc` name (always provided),
is rejected; so is recursion (a body may only call earlier declarations), and
so is naming a gate, parameter or argument `pi`, `sin`, `cos`, `tan`, `exp`,
`ln` or `sqrt` (keywords of the expression grammar, not identifiers).

**Invariant:** the consumers/producers of this vocabulary — the OpenQASM 2.0
exporter (`qasm.rs`) and importer (`qasm_import.rs`), the OpenQASM 3 exporter
(`qasm3.rs`) and importer (`qasm3_import.rs`, whose vocabulary is
`stdgates.inc`'s), the native simulator (`polypus-sim`) and the QIR exporter
(`qir.rs`) — must all support the **full set**, with **identical unitary
semantics** (the native simulator is the reference; QIR decompositions may
differ only by a global phase; the OpenQASM 3 definitions may not differ at
all, under the profile's Qiskit phase conventions).

Corollaries:

- **Canonical QASM form.** The exporter emits a single canonical form (fixed
  gate spelling, parameter formatting and declaration order), and the importer
  normalises to it. The round-trip guarantee is therefore:
  `to_qasm2(from_qasm2(to_qasm2(c)))` is **byte-identical** to `to_qasm2(c)`
  — i.e. output is a fixed point, without assuming arbitrary hand-written input
  is preserved byte-for-byte. Semantically, `from_qasm2(to_qasm2(c))` always
  reproduces the same instruction sequence and parameters as `c`. Conversely,
  a single gate statement already in canonical form (a `q` register, 12-decimal
  angles, canonical spelling) is re-emitted byte-identically:
  `to_qasm2(from_qasm2(s)) == s`. The OpenQASM 3 export has the same
  fixed-point guarantee, `to_qasm3(from_qasm3(to_qasm3(c))) == to_qasm3(c)`,
  for its own canonical form (C-10).
- Adding a gate is a **six-place change** plus a row in the equivalence test —
  the OpenQASM 2.0 exporter (`qasm.rs`), the OpenQASM 2.0 importer
  (`qasm_import.rs`), the OpenQASM 3 exporter (`qasm3.rs`: the gate's
  `stdgates.inc` name, or a definition in its `HELPER_PROGRAM` whose matrix
  `crates/polypus-sim/tests/qasm3_semantics.rs` checks), the native simulator
  (`polypus-sim`), the QIR exporter (`qir.rs`) and the Python bindings
  (`crates/polypus/src/bindings/circuit.rs`, which expose the builder method);
  the OpenQASM 3 importer too, if `stdgates.inc` names the gate. A PR adding it
  in fewer places must be rejected.
- **Non-finite parameters are rejected uniformly.** A `NaN` or infinite angle
  is never a valid parameter value: circuit construction and parameter binding
  reject it (`CircuitError::NonFiniteParam`), the OpenQASM importers reject it
  at parse time (`CircuitError::Parse`), the QASM and QIR exporters refuse to
  serialise it, and the simulator rejects it (`SimError::NonFiniteAmplitude`).
  No producer may emit, and no consumer may accept, a non-finite parameter.
- **Angle expressions exist only before binding.** An angle may be an
  expression of the circuit's free parameters (`GateParam::Expr`, stored in
  its circuit by `ParameterizedCircuit::add_expr`; a bare parameter stays
  `Param` and a constant expression becomes `Fixed`). `assign_parameters`
  evaluates every expression, so the OpenQASM 2.0 and QIR exporters see them
  only through `to_qasm2_with_params` / `to_qir_with_params`, which evaluate
  them with the given values, and the simulator only receives bound circuits
  (an expression left in a hand-assembled `ConcreteCircuit` is
  `SimError::UnboundExpression`). Evaluation performs exactly the written
  operations, in source order, without reassociation. The rule above applies
  to what an expression takes in and to its final value: a non-finite number
  in the expression, a non-finite value bound to a parameter it uses, or a
  non-finite result is `NonFiniteParam`, and a division by zero is
  `CircuitError::DivisionByZero`. An infinite intermediate whose result is
  finite (`1/exp(x)` for a large `x`) is accepted. An expression id means
  something only in the circuit that issued it; any other circuit rejects it
  (`CircuitError::UnknownExpression`).

**Enforcing test:** parametric round-trip tests over the whole vocabulary in
`crates/polypus-circuit/tests/contracts.rs` (export → import → export per gate
and for a circuit using every instruction kind — an exhaustive match fails the
build when a new variant is not covered — plus canonical statement → import →
export byte identity per gate, `cu1`/`cp` spelling preservation, and every
angled gate exported with expressions as with their values); the
table ↔ exporter spelling check in `qasm_import.rs`'s unit tests; the
QIR-vs-simulator unitary-equivalence test in
`crates/polypus-sim/tests/contracts.rs`, which parses the QIR actually emitted
for every gate and compares it with the native gate up to global phase; and,
for expressions, `crates/polypus-circuit/tests/expressions.rs` (binding
semantics, the non-finite and division-by-zero rules, ids foreign to a
circuit, bounds), the calls-with-free-parameters tests in
`crates/polypus-circuit/tests/gate_declarations.rs`, and
`unbound_expression_is_rejected` in
`crates/polypus-sim/tests/gate_matrices.rs`. For OpenQASM 3: the `c2_*qasm3*`
tests of `crates/polypus-circuit/tests/contracts.rs` (the whole vocabulary,
gate by gate, `stdgates.inc` statements byte-identical, the fuzz corpus), and
`crates/polypus-sim/tests/qasm3_semantics.rs` (every definition the export
writes, against its instruction, as full matrices).

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
- Every counts dict Polypus hands to Python lists its keys in **ascending
  bitstring order** (`00`, `01`, `10`, `11`): `RunResult.counts` from
  `run_quantum_circuit` and `qml.predict`, and the dict a `SampleCost` callback
  receives. A float reduction over `.items()` therefore adds its terms in the
  same order in every call and process. Dicts travelling the other way
  (`run_qcs`, C-1) may list their keys in any order.

The per-circuit dict format above is unchanged by C-7: `run_quantum_circuit`
now returns that payload as the `counts` attribute of a `RunResult` wrapper
(`list[dict]` for a single-QPU run, a merged `dict` for `n_qpus > 1`), so
callers read `result.counts` rather than the bare value. The dict shape,
bit order, key order and shot-conservation rule are exactly as specified here.

**Enforcing test:** shot-conservation assertion in the orchestration tests
(`crates/polypus/tests/running_quantum_circuits_local.rs`, plus the Python
public-API case in `tests/python/test_local_run.py`; audit C6); key order in
`tests/python/test_local_run.py` (`run_quantum_circuit`, one and several QPUs),
`tests/python/test_qml_predict.py` and `tests/python/test_qml_supervised.py`
(the `SampleCost` dict); flat bitstring keys for circuits with several
classical registers (Aer vs native parity) in
`tests/python/test_local_run_multi_register.py` (issue #206); last-write-wins
in the `c3_*` tests of `crates/polypus-sim/tests/contracts.rs` (simulator
semantics, incl. `MeasureAll` ordering against explicit `Measure`s) and
`TestLastMeasurementWins` in `tests/python/test_backend_selection.py` (native
vs. Aer, byte-identical counts; issue #205).

---

## C-4 · Measurement placement (terminal measurements)

Polypus circuits are straight-line programs with **terminal measurement**: no
gate may act on a qubit after that qubit has been measured, and each classical
bit is written at most once *(exception: the C-3 last-write-wins rule exists
only to define behaviour for hand-assembled circuits)*.

Backends and exporters **must reject** circuits that violate this with an
explicit error — never silently reorder, deduplicate or no-op the measurement.
The rule is enforced at four points, which must all reject identically (same
offending qubit, same error) even though they see the circuit differently:

- **Builder** (`ParameterizedCircuit::try_push`/`push`): incremental,
  push-time. Each push is checked against the per-circuit record of qubits
  already measured, which the push itself then updates — the prefix is known to
  be valid, so only the new instruction can offend. Rescanning here would make
  building a circuit quadratic in its gate count (issue #109).
- **QASM importer**: incremental, parse-time, against the parser's own
  `measured` set (which also carries the offending line number).
- **Simulator** (`polypus-sim`) and **QIR exporter**: a single full-sequence
  scan via `polypus_circuit::terminal_measurement_violation`, since both are
  handed a complete instruction list — possibly hand-assembled, i.e. never
  validated by the builder or the importer.

`terminal_measurement_violation` is the reference definition of the rule: a
unitary on a measured qubit is a violation, `Barrier` is always allowed, and
re-measuring an already-measured qubit is allowed.
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
- **DE early stopping — fitness stagnation.** DE stops early on *best-fitness
  stagnation*, not on population-spread collapse. With `fitness_history[g]` the
  best fitness at the end of generation `g` (0-indexed; DE maximises, so this
  series is monotonically non-decreasing), the run stops at the first generation
  `g` such that

  ```text
  g >= patience   and   fitness_history[g] - fitness_history[g - patience] < tolerance
  ```

  i.e. the best fitness improved by less than `tolerance` over the last
  `patience` generations. Consequences:
  - `tolerance` is a **minimum cumulative fitness improvement in the oracle's
    own fitness units** — *not* a population standard deviation in parameter
    units. (This replaces the former "max per-dimension population std <
    tolerance" collapse test, which on QAOA-like landscapes fired within ~5
    generations, long before the fitness had plateaued.)
  - `patience` (generations; DE binding default **20**) is the look-back window.
    No early stop can fire before generation `patience`, so a larger value makes
    the optimizer more patient. Exposed as `polypus.DE(..., patience=20)`.
  - The search dynamics are unchanged (DE/rand/1, `F = 0.8`, `CR = 0.7`, angles
    initialised in `[0, 2π)`); only the stopping rule differs. PSO still uses the
    per-dimension std-collapse test; QNG has no early stop.
- **Quality trajectory.** Every optimizer reports `fitness_history: Vec<f64>` —
  the best fitness at the end of each executed generation/iteration, so
  `fitness_history.len() == iterations_run` and `fitness_history.last()`
  equals `best_fitness`. It is surfaced on the Python `TrainResult`.

**Enforcing test:** invariant test with multiple seeds in
`crates/polypus-optimizers/tests/` (the pure-Rust optimizer↔oracle invariant),
plus the DE fitness-stagnation tests there
(`de_early_stops_on_fitness_stagnation`, `de_large_patience_never_stops_early`)
and the `fitness_stagnated` unit tests in `crates/polypus-optimizers/src/util.rs`,
**plus** `tests/python/test_oracle_contract.py` for the Python-backed length and
finiteness guarantee: the Rust invariant test never exercises the Python-callback
return path, so it covers only one half of the contract. The Python test forces a
short return list and a non-finite value through `polypus.train` and asserts each
surfaces as a typed `polypus.EvaluationError` (naming both lengths, or the
offending index/value), never a `pyo3_runtime.PanicException` and never a
silently-poisoned result.

---

## C-6 · Version coherence

The workspace `Cargo.toml` version is the **single source of truth**. A release
tag `vX.Y.Z` must match it exactly, and the Python package version is derived
from it at build time. The release workflow refuses to publish when they
diverge.

*Historical note: tag and Cargo.toml diverged at 0.6.0 (see CHANGELOG). The
coherence check is enforced from 0.7.0 onwards; aligning the workspace version
is the first release action.*

**Enforcing check:** *not yet present.* There is no `hygiene.yml` (nor a
version-coherence step in any current workflow); coherence is maintained by
convention until the release-workflow gate lands with 0.7.0 (see the historical
note above).

---

## C-7 · Seeding & run manifest (Python entry points)

This contract governs the outer Rust↔Python boundary of the four public entry
points `polypus.run_quantum_circuit`, `polypus.train`, `polypus.qml.train` and
`polypus.qml.predict`: their `seed` kwarg and their return shape. (It is distinct from C-1, which
freezes the *internal* `run_qcs` seam to the `polypus_python` package.)

### The `seed` kwarg

- **`run_quantum_circuit(..., seed: int | None = None)`** seeds shot sampling
  across every *simulated* backend:
  - With `infrastructure="local"` (`backend="polypus"`, the native
    statevector simulator, or `backend="aer"`, Qiskit Aer): an explicit `seed`
    is used directly — the native backend seeds its own RNG in-process, and
    Aer receives it as the `seed` kwarg forwarded across the C-1 seam
    (`crates/polypus-infrastructure/src/local.rs`, which passes it to Aer's
    `seed_simulator` option in `polypus_python`'s `local.py`) — and
    reproduces the counts byte-for-byte across calls, verified against a real
    Aer install. `seed=None` draws a fresh seed from OS entropy, so repeated
    calls produce **independent** noise (never the run-`id`-derived
    repetition that motivated this contract). The run `id` is decoupled from
    the RNG — it is only a logging/temp-file/SLURM label.
  - With `infrastructure="cunqa"`: the same `seed` kwarg is forwarded across
    the same seam (`crates/polypus-infrastructure/src/cunqa.rs` mirrors
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
- **`qml.predict(..., seed=None)`** seeds shot sampling like
  `run_quantum_circuit`: an explicit seed reproduces the counts, `None` draws one
  from OS entropy, and the effective value is reported. `qmio` is rejected
  outright, since a QML model is a Qiskit circuit.

### The run manifest (return shapes)

- `run_quantum_circuit` returns a **`RunResult`** exposing:
  `counts` (the C-3 payload — `list[dict]` for one QPU, merged `dict` for
  `n_qpus > 1`), `id` (str), `seed` (`int | None`; the effective seed used, or
  `None` only for the `qmio` infrastructure), `backend` (str), `infrastructure`
  (str).
- `qml.predict` returns the same **`RunResult`**, but `counts` is always a
  `list[dict]`, one C-3 dict per row of `x` in row order: `n_qpus` spreads the
  rows, never one row's shots. `seed` is always an `int`; `id` is generated
  internally (`predict_<n_qpus>_<infrastructure>_<uuid>`).
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
rejection, and the returned manifest/outcome fields), `tests/python/test_qml_predict.py`
(the `qml.predict` manifest, seed replay and per-row counts), plus the Rust tests in
`crates/polypus/src/bindings/mod.rs` (native seed round-trip through
`run_quantum_circuit`, the `qmio` rejection path, and the seed-resolution
precedence / optimizer determinism) and `crates/polypus-infrastructure/src/native.rs`
(same-seed reproduces / omitted-seed differs at the backend level). CUNQA's
`seed` forwarding follows the same shape as Aer's on the Rust side
(`crates/polypus-infrastructure/src/cunqa.rs` mirrors `local.rs`) but has no
dedicated automated test and no verified-working status: per `ENGINEERING.md`
§3 the Rust suite is deliberately Python-runtime-free, so this seam can only
be tested from `tests/python/`, and the `cunqa` package isn't installed
anywhere in this project's CI or dev sandboxes (unlike Aer) — so, unlike the
Aer path, it has never actually been run. Treat CUNQA's `seed` support as
unverified until the CUNQA integration follow-up confirms it against a real
install.

---

## C-8 · qml.train row/dimension/label symmetry (Python entry point)

`polypus.qml.train` composes a Qiskit `feature_map` with an `ansatz`, pre-binds
each row of `x_train` to the feature-map parameters, and hands the resulting
circuits to the optimizer, which searches a `dimensions`-wide vector and binds
it to the ansatz's free parameters. Two shape agreements must hold, and both are
validated **upfront** with a clear `ValueError` — before any circuit is composed
or executed — rather than surfacing as a silent truncation or a cryptic Qiskit
binding error deep inside the oracle. A third agreement governs the optional
labels.

- **Row width.** Every row of `x_train` must have **exactly
  `len(feature_map.parameters)`** elements. A longer row would silently drop the
  extra features (the pre-binding zip stops at the shorter iterator); a shorter
  row would leave feature-map parameters unbound and fail later as a cryptic
  Qiskit error inside the oracle. Either case is a `ValueError` reporting the
  offending **0-based** row index and both lengths (row features vs.
  `len(feature_map.parameters)`), consistent with how `x_train` is indexed as an
  array/list on the Python side.
- **Dimensions.** `dimensions` must be **exactly `len(ansatz.parameters)`**. This
  mirrors `train`, which validates `dimensions` against the circuit's free
  parameter count (`circuit_source.num_params()`); that symmetry was documented
  only implicitly (in `train`'s docstring) and was missing entirely from
  `qml.train`, which is precisely why it now earns an explicit contract. A
  mismatch is a `ValueError` naming both `dimensions` and the ansatz's free
  parameter count.

- **Labels.** `y_train` (keyword-only, optional) holds **exactly one label per
  `x_train` row**, in row order, because the oracle pairs circuits with labels by
  position; a mismatch is a `ValueError` naming both counts. Each label is a
  single finite number: a `str`, another non-number or a nested row (one-hot,
  column vector) is a `TypeError`, and `NaN`/`inf` a `ValueError`, each naming the
  0-based index. All-integer labels reach the objective as `int`, otherwise all as
  `float`. With labels, `expectation_function` must be a callable
  `(bitstring, label) -> float`, a `polypus.CachedCost(callable)` or a
  `polypus.SampleCost((counts, label) -> float)`; a `Qubo`/`Ising` is a
  `TypeError`, and so is a `SampleCost` without labels. These checks run before
  any backend is created. `y_train=None` is the unsupervised path, unchanged.

**Inference (`polypus.qml.predict`).** The same agreements hold, so a model runs
the circuit it was trained as: each row of `x` has exactly
`len(feature_map.parameters)` values (the row-width `ValueError`, naming `x`),
and `params` has exactly `len(ansatz.parameters)` finite values. An empty `x` is
rejected too, all before anything runs.

A row whose length cannot be read (e.g. a generator with no `__len__`) is a
legitimate type error and propagates as-is; it is not masked into the messages
above. `y_train` itself is only iterated, so a generator is fine there.

**Enforcing test:** `tests/python/test_qml_train_validation.py` (every rejection,
with nothing executed), `tests/python/test_qml_supervised.py` (each sample's
counts meet its own label; the objective calling convention) and
`tests/python/test_qml_predict.py` (the inference agreements).

---

## C-9 · `id` charset validation (`train` / `qml.train` Python entry points)

The `id` kwarg of `polypus.train` and `polypus.qml.train` is a caller-supplied
*prefix*: the entry point appends a UUID v4 to it and the result becomes
`ExecutionConfig::id` (and the `RunParams::id` derived from it), which names the
run's temp files and log streams and —
on `infrastructure="cunqa"` — travels to SLURM as the C-1 kwargs `family_name`
(`connect_to_infrastructure`) and `family_id` (`run_qcs`), and from there
verbatim into `qraise`. The crate cannot see how `qraise`/SLURM interpolate that
name, so the string is constrained at the boundary instead of trusted:

- **Charset.** Every character must be an ASCII letter, an ASCII digit, `.`,
  `_` or `-` (i.e. `^[A-Za-z0-9._-]+$`). Whitespace, path separators (`/`,
  `../`) and shell metacharacters (`;`, `|`, `` ` ``, `$`, `&`, newline) are
  rejected. The `ValueError` names the **offending character** so the caller
  can see which one failed.
- **Non-empty.** An empty `id` is rejected: it would degrade the effective id to
  a bare `_<uuid>` and carries no debugging value.
- **Length.** At most **64 characters**, measured on the caller-supplied prefix
  **before** the UUID suffix, keeping the effective id inside the limits SLURM
  job names and filesystem path components impose.

This is defense-in-depth, not a fix for a known exploit — the July 2026
technical audit (#89) found the value unvalidated anywhere in the crate, which
`docs/ENGINEERING.md` §8 ("validate and sanitize all inputs; never trust
external data") does not allow for a string that leaves the process.

**When it is checked.** Upfront, alongside the other kwarg guards
(`validate_shots_and_qpus`, `validate_cunqa_allocation`) as the entry point's
first statements — before any seam call, any backend creation and, crucially,
before `unique_id` appends the UUID, so a rejected `id` never yields a
partially-valid effective id. Validation is unconditional: unlike
`nodes`/`cores_per_qpu` it is not gated on `infrastructure == "cunqa"`, because
the temp-file and log-stream naming applies to every infrastructure.

`run_quantum_circuit` is not covered: it generates its own `id` internally and
takes no such kwarg.

**Enforcing test:** `tests/python/test_id_validation.py`.

---

## C-10 · OpenQASM 3 profile with Qiskit phase conventions

`ParameterizedCircuit::from_qasm3` / `to_qasm3` / `to_qasm3_with_params`
(Python: `Circuit.from_qasm3`, `Circuit.to_qasm3(params=None)`,
`Circuit.param_names`) read and write the **OpenQASM 3 profile with Qiskit
phase conventions** — never "a subset of OpenQASM 3": the straight-line part of
OpenQASM 3.0 that carries a parameterised, terminal-measurement circuit. Its
`stdgates.inc` is the one of tag `spec/v3.1.0`
(`c717508162a0eac892fa32134716fe77a284e835`), whose gate list is that of
`spec/v3.0.0` (`51c36946c687c8b17000962f6ce735ca1d1c9b3b`) with `pow(1/2)`
written `pow(0.5)`.

**Accepted.** `OPENQASM 3;` or `OPENQASM 3.0;` (optional, first); `include
"stdgates.inc";`, provided internally — no file is ever read; `qubit[n]`,
`qubit`, `bit[n]`, `bit` declarations, flattened in declaration order;
`input float[64] name;` and `input float name;` (both binary64); calls of `U`,
of the `stdgates.inc` gates and of earlier `gate` declarations (bodies of gate
calls only); `barrier`; measurements `c[i] = measure q[j];`, `c = measure q;`,
`measure q -> c;` (C-4 applies unchanged); angle expressions over the inputs
(in a body, over the gate's parameters): numbers, `pi`/`π`, `tau`/`τ`,
`euler`/`ℇ`, `+ - * /`, unary minus, `**` (right-associative, binding tighter
than unary minus), and `sin cos tan arcsin arccos arctan exp log sqrt`;
Unicode identifiers; `//` and `/* */` comments.

**Rejected**, each as `CircuitError::Parse` naming the construct and its
1-based line (a construct both dialects reject is worded as the OpenQASM 2.0
importer words it, so `benchmarks/qasm_coverage.py` classifies either):
other versions and includes;
`if`/`else`, `for`, `while`, `switch`; `reset`; classical types other than
`bit` and `input float`; `let`; assignments other than measurement;
`bit c = measure q;`; `const`; `output`; casts; `def`; `extern`; the modifiers
`ctrl @`, `negctrl @`, `inv @`, `pow @` and `gphase`, in bodies too; `delay`,
`stretch`, `duration`, `box`; `defcal`, `cal`, `defcalgrammar`; arrays,
slices, index sets, concatenation; physical qubits (`$0`); annotations and
pragmas; the functions `mod`, `popcount`, `rotl`, `rotr`, `floor`, `ceiling`,
`pow` (and `sizeof`, `real`, `imag`); a division whose two operands are both
integer expressions (`1/2`: integer division in OpenQASM 3 — write `1.0/2`);
an input named like a `stdgates.inc` gate, included or not (input names are
kept, and the export always includes it); in a body, a call of a gate its own
parameter or qubit argument shadows.

**Phase semantics.** `U`, `u2` and `u3` are read and written with Qiskit's
matrices — Polypus's `u`, `u2`, `u3` — a declared deviation from the
specification by exactly e^{−iθ/2} (`U`) and e^{+i(φ+λ)/2} (`u2`, `u3`).
Amplitudes may differ from a specification-normative reader by these global
factors; probabilities, counts and expectation values do not. This is sound
only because modifiers and `gphase` are rejected (under them a global phase
becomes observable); admitting either requires global-phase tracking. Every
other `stdgates.inc` gate follows the specification exactly. `CX` is read as
`cx`, as `standard_library.rst` describes it ("an alias for `cx`"), although
`stdgates.inc` defines it as `ctrl @ U(π, 0, π)`, which under the
specification's `U` is controlled-(iX). That Polypus reads Qiskit's
`qasm3.dumps` output as Qiskit does is tested for Qiskit 2.5.2 with
`qiskit-qasm3-import` 0.6.0, not guaranteed for other versions.

**Spellings.** On import `U` → `u`, `CX` → `cx`, `phase` → `p`, `cphase` →
`cp`; on export `u` → `U`; no other name changes. A declared gate is never
recognised as a built-in by its name.

**Parameters.** Inputs are the free parameters, in declaration order, unused
ones included. Their names are stored in the circuit (`param_names`), are part
of its equality and survive clone, export and import, Unicode included;
Qiskit's name mangling (`θ[0]` → `_θ_0_`) is not reversed. A parameter without
a name is `theta_<index>`, or the first free `theta_<index>_<k>` if a named
parameter or a declared gate the circuit calls takes that name.

**Numbers.** Angles are evaluated in binary64, exactly the operations written,
in source order. A constant angle is evaluated at import: a non-finite result
or a division by zero is a `Parse` error (on the division's line); an angle of
the inputs is evaluated at binding (`NonFiniteParam`, `DivisionByZero`). Only
the final value must be finite: `1.0/exp(1000.0)` is `0.0`.

**Canonical form.** What `to_qasm3` writes, and why its output is a fixed
point: `to_qasm3(from_qasm3(to_qasm3(c)))` is byte-identical to `to_qasm3(c)`
for every circuit `to_qasm3` accepts.

```text
OPENQASM 3.0;
include "stdgates.inc";
input float[64] <name>;         one per parameter, in index order
gate <name>(<p>, …) <q>, … {    each definition the circuit needs, once:
  <statement>;                    callees first, in order of first use
}
qubit[<n>] q;                   unless n = 0
bit[<m>] c;                     unless m = 0
<statement>;                    one per instruction
```

- Statements: `name(a, b) q[i], q[j];`; `barrier q;` for every qubit
  (`barrier;` with none), `barrier q[i], …;` otherwise; `c = measure q;` for a
  full measurement when the registers have one size, `c[i] = measure q[j];`
  otherwise. Inside a definition the same, over its formal names, indented two
  spaces.
- Numbers: the shortest decimal that reads back as the same binary64, always
  with a decimal point — positional for 1e-5 ≤ |v| < 1e16 (`0.00001`, `1.0`),
  scientific otherwise (`1.5e-7`, `1.0e16`); negative zero is `-0.0`.
- Expressions: `+` and `-` between spaces, `*`, `/`, `**` without; functions
  by name (`log` for the natural logarithm), constants as `pi`, `tau`,
  `euler`; the fewest parentheses that keep the expression: `**` binds
  tightest (right-associative; its exponent may be a negation), then unary
  minus, then `* /`, then `+ -` (both left-associative). A negative number is
  written with its minus and counts as a negation.
- Instructions `stdgates.inc` lacks are calls of these definitions (exact
  matrices, checked in `polypus-sim`): `rzz(θ)` = `cx; rz(θ); cx`, `rxx(θ)` =
  `h ⊗ h; cx; rz(θ); cx; h ⊗ h`, `sxdg` = `h; sdg; h`, `csx` =
  `h; cp(π/2); h` on the target, `cu1` = `cp`, `cu3(θ, φ, λ)` =
  `cu(θ, φ, λ, 0)`, `u0` = the identity (empty body), and `rccx`, `rc3x`,
  `c3x`, `c3sqrtx`, `c4x` as their `qelib1.inc` definitions (`qasm3.rs`'s
  `HELPER_PROGRAM` holds the exact text). Declared gates are printed from
  their definitions, never copied.
- Names: an input keeps its name. A gate, register or formal name is kept if it
  is a valid OpenQASM 3 identifier of at most 4096 bytes, not a keyword,
  constant, function, `U`, `gphase` or `stdgates.inc` gate, and not taken in its
  scope (inputs, then gates in order, then registers; a definition's formals
  avoid all of those). Otherwise it is renamed: characters an identifier cannot
  hold become `_`; `g` (gate), `p` (parameter), `q` (qubit argument or qubit
  register) or `c` (bit register) is prefixed if it cannot start one; `_` is
  appended to a reserved word; it is cut to 4075 bytes; `_1`, `_2`, … is
  appended until it is free. Names that can be kept are claimed before any is
  renamed, so the scheme never collides. The registers are `q` and `c`, renamed
  the same way when taken.
- Budgets: the export fails with `CircuitError::ExportLimit` rather than write a
  program the importer would reject (the budgets below, counted as the importer
  counts them), and with `CircuitError::GateNotExpressible`, naming the gate,
  for a definition it cannot write: an OpenQASM 2.0 body with a barrier, or one
  that would nest or expand beyond the importer's limits once its `c4x`-like
  calls become definition calls.

**OpenQASM 2.0 export of OpenQASM 3 declarations.** `to_qasm2` needs every
parameter bound. A declaration imported from OpenQASM 3 is printed from its
definition on one line (`gate name(a,b) x,y { h x; cx x,y; }`), with `log` as
`ln`, `**` as `^`, `tau` and `euler` as their values, numbers as above, and
names renamed by the same scheme for OpenQASM 2.0's `[a-z][A-Za-z0-9_]*`
(avoiding its keywords, the `qelib1.inc` gates, `q`, `c` and the names of
OpenQASM 2.0 declarations, which keep theirs). A body using `arcsin`, `arccos`
or `arctan` is `GateNotExpressible`, naming the gate (`ConcreteCircuit::to_qasm2`
panics there; `try_to_qasm2` returns the error). OpenQASM 2.0 declarations are
re-emitted verbatim as before (C-2).

**Budgets** (each a `Parse` error on import; the export stays within them):

| Budget | Value |
|---|---|
| Source size (`MAX_SOURCE_BYTES`) | 32 MiB |
| Identifier, number or string (`MAX_TOKEN_BYTES`) | 4096 bytes |
| Inputs (`MAX_INPUTS`) | 100 000 |
| `gate` declarations (`MAX_DECLARATIONS`) | 10 000 |
| Expression nodes, one expression (`MAX_EXPR_NODES`) | 1 000 000 |
| Expression nodes, whole program (`MAX_PROGRAM_NODES`) | 4 000 000 |
| Expression nesting as written (`MAX_EXPR_DEPTH`) | 64 |
| Qubits, and bits (`MAX_REGISTER_BITS`) | 1 000 000 each |
| Instructions after broadcasting (`MAX_INSTRUCTIONS`) | 4 000 000 |
| Declared-gate nesting (`MAX_GATE_NESTING`) | 64 |
| Statements one declaration expands to (`MAX_GATE_EXPANSION`) | 1 000 000 |
| Statements all calls of declared gates expand to (`MAX_VALIDATED_EXPANSION`) | 20 000 000 |

**Enforcing test:** `crates/polypus-circuit/tests/qasm3.rs` (every accepted
and rejected construct with its line and message, the budgets on both sides,
the canonical layout, numbers and parentheses, renaming and collisions — a
native and a declared `rzz` included —, cross-dialect export),
`crates/polypus-circuit/tests/contracts.rs` (`c2_*qasm3*`: the vocabulary's
fixed point), `crates/polypus-sim/tests/qasm3_semantics.rs` (full matrices:
every `stdgates.inc` gate and `U` against the specification, `U`/`u2`/`u3` by
the declared factors, `CX`, every exported definition, cross-dialect),
`tests/python/test_qasm3.py` and `tests/python/test_qasm3_interop.py` (Qiskit
2.5.2 in both directions: pinned fixtures and a generated set, explicit
parameter mapping, statevectors with the predicted phase only, counts and bit
order), and the `from_qasm3` fuzz target in CI (import, export, re-import,
fixed point, binding).
