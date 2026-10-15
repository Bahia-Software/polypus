# ADR 0006 — The circuit carries its declared classical width

- **Status:** Proposed (2026-10)
- **Contracts:** [C-2 · Gate vocabulary symmetry](../CONTRACTS.md#c-2--gate-vocabulary-symmetry), [C-3 · Measurement counts format](../CONTRACTS.md#c-3--measurement-counts-format), [C-10 · OpenQASM 3 profile](../CONTRACTS.md#c-10--openqasm-3-profile-with-qiskit-phase-conventions)
- **Supersedes:** —
- **Issue / PRs:** #279 (this record, tests and contract text); #311 (the implementation); follows #251

## Context

The circuit IR (`polypus-circuit`) knew one classical register, implicit: as
wide as the measurements need (`max(cbit) + 1`, `num_qubits` with
`measure_all`). The OpenQASM importers flattened every `creg` (or `bit`
register) into it and then forgot how wide the program had declared it.

Where that width matters is the key width of the counts (C-3): Qiskit keys a
measured circuit at the classical bits it declares, so `creg c[3]; measure q[0]
-> c[1];` gives `"010"`, not `"10"`. #251 made the native backend agree with Aer
for an OpenQASM **string**: the importer stored the declared total in a
`pub(crate)` field of `ParameterizedCircuit`, ignored by `==` and read through a
hidden getter, and the native backend carried it next to the circuit
(`NativeSource`) to format the counts.

That left the other paths inconsistent:

- `Circuit.from_qasm2(src)` / `from_qasm3(src)` dropped the width at binding,
  so the imported circuit ran at `max(cbit) + 1` on both backends — a different
  key width from the same program run as text (C-3's "Known limit").
- Every export wrote the implicit register: `to_qasm2(from_qasm2(src))` lost
  unwritten trailing bits, and a program without measurements lost its `creg`
  altogether.

## Decision

As implemented by #311:

1. **`ConcreteCircuit` carries the declared width in a private field**
   (`Option<usize>`, `None` for nothing declared), and is built with
   `ConcreteCircuit::new(num_qubits, gates)`, which declares none. A struct
   literal no longer compiles; a future field can be added without breaking
   callers. `ParameterizedCircuit` keeps its `pub(crate)` field;
   `assign_parameters` copies it, and `clone` and the builder methods keep it.
2. **`clbits_width()`** (both types) is the effective width:
   `max(declared, num_clbits())`, or `num_clbits()` when nothing was declared.
   **`num_clbits()` keeps its meaning**, the implicit register the
   measurements need (0 without measurements), so C-3's "no measurements ⇒
   `num_qubits`-wide keys" still reads it. The hidden `declared_clbits()`
   getter stays, on both types.
3. **`PartialEq`** (both types) compares `clbits_width()`: a declared width no
   larger than the measurements' is the same as declaring none; a larger one,
   measured or not, is a different circuit, because it exports differently.
4. **The OpenQASM 2.0 and 3 exports write `clbits_width()`** (`creg c[n];`,
   `bit[n] c;`), also without measurements. A builder circuit declares none and
   exports byte for byte as before.
5. **QIR is unchanged**: it records only the measured results, and
   `required_num_results` is `num_clbits()`.
6. **The native backend reads the declared width from the circuit before any
   transpiler runs** (`NativeSource`, now filled from the `Native` circuit
   too), so a transpiler that rebuilds the circuit with `ConcreteCircuit::new`
   cannot change the keys.
7. **Python's `Circuit.num_clbits`** keeps returning `num_clbits()`, the
   implicit register.

What is modelled is the **total** width — what changes the length of the
counts' keys. Named registers and their partition are not: `creg a[2]; creg
b[1];` is still flattened into one `creg c[3]`, as it already was at the level
of counts (C-3, #206).

## Consequences

- The same program gives the same keys on Aer and on the native backend
  whether it is run as text or as `Circuit.from_qasm2(src)` /
  `from_qasm3(src)`, one QPU or several; C-3 loses its "Known limit".
- `to_qasm2(from_qasm2(s))` and `to_qasm3(from_qasm3(s))` keep the declared
  width; the export stays a fixed point. A full measurement into a register
  wider than `num_qubits` is written bit by bit, since `measure q -> c;` needs
  equal sizes; the importer collapses it back to `MeasureAll`.
- On the author's local, untracked fuzz corpus (2026-10-14: 855 `from_qasm2`
  entries, 76 of which import), the export change of #279's first
  implementation — equivalent to this one for OpenQASM 2.0 — altered 17
  exports, all of them by gaining the `creg` of a program without
  measurements; no existing width changed. The repository only tracks the seed
  files of that corpus, so these figures cannot be reproduced from a clone.
- CUNQA receives the exported program as Aer does, so an imported circuit that
  declares a `creg` now reaches it with that `creg` even without measurements.
  That path stays unverified (#218), with or without measurements.
- Breaking for Rust code: `ConcreteCircuit` literals no longer compile (use
  `new`), and `==` distinguishes a wider declared register. Python is
  unaffected except for the counts and exports of imported circuits.

## Alternatives considered

1. **A public `declared_clbits: usize` field, `num_clbits()` as the effective
   width plus a new `measured_clbits()` for the implicit one, `NativeSource`
   retired, QIR's `required_num_results` and Python's `Circuit.num_clbits` as
   the effective width.** This was #279's first implementation. It kept
   literals constructible (`declared_clbits: 0`), made `Circuit.num_clbits`
   match Qiskit's `QuantumCircuit.num_clbits` for imported programs, and made
   QMIO's `qir`/`qir_bitcode` formats declare the width its `openqasm` format
   does. It was set aside because #311, the design above, was merged first:
   replacing it would have meant a second, incompatible API change to the same
   types within days, without a release in between.
2. **`Option<usize>` versus a plain `usize`.** With equality and exports by the
   effective width, `None` and `Some(0)` behave identically (and the importers
   reject empty registers, so `Some(0)` never arises from them); #311 keeps
   `Option`, which the private field makes an internal detail.

## Open questions (follow-ups)

- **QIR / QMIO parity.** For an imported program whose `creg` is wider than
  the bits it writes, QMIO's `openqasm` format submits the declared width but
  its `qir`/`qir_bitcode` formats declare `required_num_results =
  num_clbits()`, the measured register. Whether QIR should report
  `clbits_width()` instead (results declared but never recorded) is to be
  decided separately.
- **Python `Circuit.num_clbits`.** It reports the implicit register, while
  Qiskit's `QuantumCircuit.num_clbits` reports the declared width for the same
  program; whether to expose `clbits_width` (or change `num_clbits`) in Python
  is to be decided separately.
- Letting a builder user declare the width (e.g. `Circuit(3, num_clbits=5)`)
  is new public API no issue asks for yet.

## Reopening criteria

- Named classical registers, or their partition, become observable (counts keyed
  per register, mid-circuit classical control): the total width is no longer
  enough and the registers themselves must be modelled.
- One of the open questions above is decided in a way that changes the meaning
  of `num_clbits()` or `clbits_width()`.
