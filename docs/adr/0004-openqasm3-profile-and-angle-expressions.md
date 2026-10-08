# ADR 0004 — OpenQASM 3 profile with Qiskit phase conventions, and angle expressions

- **Status:** Accepted (2026-10)
- **Contracts:** [C-2 · Gate vocabulary symmetry](../CONTRACTS.md#c-2--gate-vocabulary-symmetry),
  [C-10 · OpenQASM 3 profile with Qiskit phase conventions](../CONTRACTS.md#c-10--openqasm-3-profile-with-qiskit-phase-conventions)
- **Supersedes:** —
- **Issue / PRs:** #236; #241 (angle expressions), #282 (OpenQASM 3)

## Context

Public benchmark suites (QED-C, MQT Bench, SupermarQ, qml-benchmarks) ship
parameterised circuits Polypus could not load:

1. `GateParam` was `Fixed(f64) | Param(usize)`, so an angle could not be an
   expression of free parameters. Suite ansätze need them: QED-C MaxCut writes
   `rzz(-gamma)` and `rx(2*beta)`, Qiskit's `zz_feature_map` writes
   `p((-pi + x0)*(-pi + x1)*2)`.
2. OpenQASM 2.0 cannot carry unbound parameters (Qiskit raises
   `QASM2ExportError`). OpenQASM 3 can, with `input float[64] name;`.

OpenQASM 3 and Qiskit, its main producer, disagree on the matrices of `U`,
`u2` and `u3` by global phases, and the specification is not consistent with
itself on `CX` (*Evidence*).

## Decision

C-10 specifies the profile and C-2 the expressions; this section records why.

1. **A straight-line profile.** `from_qasm3`/`to_qasm3` read and write the part
   of OpenQASM 3 that carries a parameterised, terminal-measurement circuit
   ([ADR 0001](0001-terminal-measurements.md)): `input float` parameters,
   `stdgates.inc`, `gate` declarations, angle expressions, `barrier`,
   measurements. Everything else (control flow, `reset`, classical
   computation, `def`, modifiers, `gphase`, timing, pulse) is a `Parse` error
   naming the construct and its line. The importer is our own streaming
   parser, with the hardening and fuzzing of the OpenQASM 2.0 one.
2. **Qiskit phase conventions.** `U`, `u2` and `u3` use Qiskit's matrices
   (Polypus's `u`, `u2`, `u3`), off the specification by exactly e^{−iθ/2}
   and e^{+i(φ+λ)/2}. The API, README and C-10 always call it "OpenQASM 3
   profile with Qiskit phase conventions", never "a subset of OpenQASM 3".
   Amplitudes differ by a global factor only; probabilities, counts and
   expectation values do not. This holds only while modifiers and `gphase` are
   rejected, because under them a global phase becomes observable.
3. **Spellings.** Import `U` → `u`, `CX` → `cx`, `phase` → `p`, `cphase` →
   `cp`; export `u` → `U`; nothing else. `CX` follows `standard_library.rst`
   ("an alias for cx") and Qiskit, not the `stdgates.inc` body.
4. **Declared gates are never built-ins by name.** A declared `rzz` stays a
   declared gate. Definitions are printed for the target dialect (`ln` ↔
   `log`, `^` ↔ `**`) and renamed deterministically on collision; a body the
   target cannot express (`arcsin` in OpenQASM 2.0, a `barrier` in
   OpenQASM 3) fails with `GateNotExpressible`. OpenQASM 2.0 declarations keep
   their byte-identical re-emission.
5. **The whole vocabulary exports.** The twelve instructions `stdgates.inc`
   lacks (`rzz`, `rxx`, `sxdg`, `csx`, `cu1`, `cu3`, `u0`, `rccx`, `rc3x`,
   `c3x`, `c3sqrtx`, `c4x`) are written as calls of definitions with the same
   matrices, parsed from one canonical text by the importer itself.
6. **Expressions live in an arena owned by the circuit.** `GateParam::Expr`
   holds an `ExprId` into a flat postfix arena, so `GateParam` stays `Copy` and
   16 bytes. Gate-body and circuit expressions are distinct types (`Formal`,
   `Input` references). Construction is validated; evaluation is one loop and
   never panics or recurses. An `ExprId` carries a content checksum, so an id
   from another circuit is rejected (`UnknownExpression`) without a global
   counter.
7. **Numbers.** Exactly the written operations, in source order, in binary64;
   no reassociation or folding. Only the final value must be finite. A division
   between two integer expressions (`1/2`) is rejected: it is 0 in OpenQASM 3
   and 0.5 in Qiskit and Python, so either reading silently changes some
   programs (`spec/v3.0.0`'s own `pow(1/2)` is one). Export writes the
   shortest decimal that reads back as the same value.
8. **Parameter identity.** Inputs are the parameters in declaration order;
   their names are stored (`theta_<i>` by default), part of equality, and
   exposed as `Circuit.param_names`. Qiskit's name mangling is not reversed.
9. **The export is a fixed point and stays importable.**
   `to_qasm3(from_qasm3(to_qasm3(c))) == to_qasm3(c)` byte for byte. Because
   a canonical export can be larger than the importer reads back (a broadcast
   measurement, a `c4x` becoming a 98-statement definition), the exporter spends
   the importer's budgets as it writes and fails with `ExportLimit` rather than
   write text `from_qasm3` would reject.
10. **Backends never panic on an inexpressible circuit.** The backends that
    submit OpenQASM 2.0 (Aer, CUNQA, QMIO, the subprocess bridge, and the Rust
    backend template) use `try_to_qasm2` and report `UnsupportedCircuit`
    (`polypus.NativeCircuitError`); the native backend runs the circuit.

## Evidence

The specification repository was read at `spec/v3.0.0`
(`51c36946c687c8b17000962f6ce735ca1d1c9b3b`), `spec/v3.1.0`
(`c717508162a0eac892fa32134716fe77a284e835`) and `main`
(`7fbf9e9eb3692a1288c014d6efd43523701886c6`), with Qiskit 2.5.2,
qiskit-qasm3-import 0.6.0, openqasm3 1.0.1 and antlr4-python3-runtime 4.13.2.
`stdgates.inc` is identical in `v3.1.0` and `main`; `v3.0.0` differs in its
header and in `pow(1/2)`. The profile uses `v3.1.0`'s.

Each `stdgates.inc` gate was evaluated with the specification's own semantics
(`U` from `gates.rst`, `gphase`, `ctrl @`, `inv @`, principal-branch `pow @`)
and compared, as a full matrix at six random draws, with Qiskit's gate of the
same name, by [`0004-phase-table.py`](0004-phase-table.py)
(`python 0004-phase-table.py <spec>/examples/stdgates.inc`). The result
is the same at all three tags:

| Result | Gates |
|---|---|
| Exactly equal (residual ≤ 8e-16) | `p x y z h s sdg t tdg sx rx ry rz cx cy cz cp crx cry crz ch swap ccx cswap cu phase cphase id u1` |
| Global phase e^{iθ/2} (spec = factor × Qiskit) | `U` |
| Global phase e^{−i(φ+λ)/2} | `u2` `u3` |
| Not a global phase | `CX` |

Qiskit behaviours, each pinned by a test in `tests/python/test_qasm3_interop.py`:

- `qasm3.dumps` writes `QuantumCircuit.u` as bare `U`, and `u2`/`u3` by name,
  without phase compensation.
- It drops the circuit's `global_phase` and the global phase of every
  definition it writes (`sxdg` becomes `s; h; s`, e^{iπ/4}·SXdg). The tests
  compare against Qiskit's own reading of the same text, and against the
  original circuit up to the phase `dumps` drops, computed from Qiskit's
  definitions.
- For QAOA, `efficient_su2`, `real_amplitudes`, `zz_feature_map` and one
  circuit per vocabulary gate, its output holds no `gphase`, no modifier and
  no division of two integers, so every text imports.
- `u3 = p(φ)·ry(θ)·p(λ)` exactly, so a specification-normative export without
  `gphase` remains possible.

Binding cost (#241): bare-parameter binding stayed within 5 % of the base
commit (`benchmarks/bench_native_vs_qiskit.py`, median 0.95×), after a
1.7× regression from a 4-byte-aligned `ExprId` was fixed with
`#[repr(align(8))]`. Tests pin `GateParam` at 16 bytes and `CircuitError` at
48 bytes.

Upstream inconsistencies, recorded and not resolved:

1. `stdgates.inc` defines `CX` as `ctrl @ U(π, 0, π)`, which under the
   specification's `U` is controlled-(iX), a relative phase away from CNOT;
   `standard_library.rst` calls `CX` "a convenience alias for cx" and
   `gates.rst` builds it as CNOT. No upstream issue found (2026-09-28).
2. `standard_library.rst` omits the half angle in the `rx`, `ry`, `rz`, `crx`,
   `cry` mappings, gives a non-diagonal `rz`, and maps `sdg`'s |1⟩ to i|1⟩.
3. The `gates.rst` footnote example `U_old(0, ϕ, θ) … gphase(-0/2)` is garbled.
4. `spec/v3.0.0` writes `pow(1/2)`, which is `pow(0)` under its own integer
   division; `v3.1.0` writes `pow(0.5)`.
5. Qiskit's `UGate` docstring claims the specification's matrix; numerically
   it is Qiskit's.

## Consequences

- Circuits from Qiskit and the benchmark suites load with the semantics Qiskit
  gives them, and Polypus's output reads back in Qiskit as the same circuit.
- A specification-normative reader sees `U`, `u2`, `u3` off by the declared
  global factors. Statevector comparisons across tools must apply them.
- The OpenQASM 2.0 path writes the same bytes for every circuit without
  OpenQASM 3 declarations; binding cost stays within 5 %.
- `ConcreteCircuit::to_qasm2` still panics where `try_to_qasm2` errs; new
  callers should use the fallible form.
- From Python, expressions come only from OpenQASM 3: `2 * polypus.Param(0)`
  is not supported yet (#239).
- The OpenQASM 3 exporter identifies definitions by address, so value-equal
  definitions from separately imported programs are written twice; the output
  is still a fixed point.
- Running an imported circuit on Aer is limited by Aer's basis, not by the
  profile (#252).

## Alternatives considered

1. **Specification-normative `U`/`u2`/`u3`.** Needs global-phase state in the
   IR, changes only statevector amplitudes, and diverges from Qiskit's output,
   the main input. Deferred; mandatory if modifiers or `gphase` are admitted.
2. **`CX` as its `stdgates.inc` body** (controlled-(iX)). Contradicts the
   library's prose and every Qiskit round trip.
3. **A third-party parser** (`oq3_semantics` 0.7.0, last release 2024-10).
   Full OpenQASM 3, but pre-1.0, a large dependency for a small profile, and
   it would bypass the hardened, fuzzed importer.
4. **A bespoke JSON circuit format.** No parser, but non-standard, and it still
   needs expressions in the IR.
5. **Re-parameterising instead of expressions** (`φ = 2β`). Changes the
   landscape's scale and initial parameters, and cannot express data encodings.
6. **Expressions as boxed trees inside `GateParam`.** Loses `Copy` and grows
   `GateParam`, whose layout binding cost is sensitive to: a layout change
   alone made binding 1.7× slower (*Evidence*).
7. **Recognising declared gates by name** (a declared `rzz` as the native one).
   Wrong whenever the body differs; collisions are renamed instead.

## Reopening criteria

- Admitting modifiers, `gphase` or global-phase tracking: switch to
  specification-normative `U`/`u2`/`u3`, in a new ADR.
- Upstream changes: `stdgates.inc` aligning `CX` with CNOT, or Qiskit changing
  its `U` convention or its export of phases.
- Admitting control flow or `reset`: that reopens ADR 0001 first.
