# ADR 0001 — Terminal measurement model

- **Status:** Accepted (2026-07)
- **Contract:** [C-4 · Measurement placement](../CONTRACTS.md#c-4--measurement-placement-terminal-measurements)
- **Supersedes:** —

## Context

A Polypus circuit is a straight-line sequence of `GateInstruction`s shared by
four layers: the OpenQASM 2.0 exporter/importer, the native statevector
simulator (`polypus-sim`) and the QIR Base Profile exporter (`qir.rs`).

Historically the layers disagreed about what a *measurement* means mid-circuit:

- the simulator treated `Measure`/`MeasureAll` as **no-ops**, taking statistics
  from the final statevector via sampling;
- the QIR exporter **deferred and reordered** every measurement to the end of
  the program (the Base Profile requires "all quantum operations first, all
  measurements last");
- the OpenQASM exporter emitted measurements in place.

For a *terminal* circuit (nothing acts on a qubit after it is measured) all
three agree. But nothing rejected a **non-terminal** circuit — e.g.
`measure q[0] -> c[0]; x q[0];`. Each layer then silently did something
different: the simulator ignored the measurement, the QIR exporter moved it
after the `x` (changing the meaning), and the exporter produced OpenQASM that
the importer would later re-order on the round-trip. The 2026-07 audit flagged
this as defect **C5**: a semantic disagreement hiding at a layer seam.

## Decision

Polypus circuits use **terminal measurement**: no gate may act on a qubit after
that qubit has been measured. Every layer **rejects** a violation with an
explicit error — it must never silently reorder, deduplicate or no-op it.

Concretely:

| Situation | Behaviour |
|---|---|
| A **unitary** gate on an already-measured qubit | **Error** (`CircuitError::QubitAlreadyMeasured` / `SimError::GateAfterMeasure`) |
| A `Barrier` on a measured qubit | **Allowed** — a scheduling hint that touches no state |
| **Re-measuring** an already-measured qubit | **Allowed** — idempotent in a terminal model |
| Two `measure`s writing the **same classical bit** | **Allowed**, last-write-wins (see contract C-3) — *not* an error |

The check is a single shared reference,
`polypus_circuit::terminal_measurement_violation`, enforced at four points so
the four layers reject identically:

1. the builder (`ParameterizedCircuit::try_push`, hence the fluent API and the
   Python bindings);
2. the OpenQASM importer (`from_qasm2`, with the offending source line);
3. the native simulator (`StatevectorSimulator::run`, defense in depth for
   hand-assembled `ConcreteCircuit`s);
4. the QIR exporter (`write_qir`, which would otherwise reorder).

The classical-bit "last-write-wins" rule (C-3) is deliberately *not* promoted to
an error: it defines behaviour for hand-assembled circuits and keeps the
counts-format contract self-consistent.

## Consequences

- The four layers can never again disagree about a mid-circuit measurement,
  because the illegal shape simply does not exist by the time it reaches them.
- Hand-assembled circuits (constructed via the public `pub` fields rather than
  the builder) are validated at the simulator and QIR boundaries, so the guard
  cannot be bypassed by skipping `try_push`.
- The infallible `ConcreteCircuit::to_qir()` panics on a violating
  hand-assembled circuit (like it already does for an unbound parameter); the
  fallible `ParameterizedCircuit::to_qir_with_params` returns the error.
- We intentionally do **not** support mid-circuit measurement, `reset`, or
  classically-controlled gates (feedforward). Reasons:
  - **Simulator cost.** Mid-circuit measurement forces a collapse-and-branch or
    per-shot re-simulation: the honest implementation is `O(shots · 2^n)`
    instead of one `O(2^n)` statevector evolution plus cheap sampling.
  - **QIR profile.** The Base Profile (our target) mandates terminal
    measurement; mid-circuit measurement and feedforward require the **Adaptive
    Profile**, which few providers accept today.
  - **Language surface.** Classical control flow (`if (c==1) ...`, `reset`)
    belongs to OpenQASM 3; the importer explicitly rejects `if`/`reset` rather
    than partially supporting them.

## Alternatives considered

1. **Keep silently reordering (status quo).** Rejected: it is the exact
   layer-seam disagreement the audit found — three layers, three meanings.
2. **Support mid-circuit measurement everywhere.** Rejected for now on the cost
   and profile grounds above; it is a large, cross-cutting feature, not a bug
   fix.
3. **Reject only in some layers.** Rejected: a circuit legal for the simulator
   but illegal for QIR (or vice versa) reintroduces a seam disagreement.

## Reopening criteria

Revisit this decision when a real, integrated backend needs **feedforward**
(classically-controlled gates or `reset`) — e.g. a hardware target exposing
mid-circuit measurement through the QIR Adaptive Profile. That is a deliberate
feature with its own ADR, a new counts/measurement contract, and simulator
support, not an incremental relaxation of this one.
