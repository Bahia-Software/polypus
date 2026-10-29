# Circuits and interchange

`polypus.Circuit` is the Python face of [`polypus-circuit`](../../crates/polypus-circuit), a pure-Rust circuit crate with no Python dependency. It is accepted by `run_quantum_circuit` and `train` exactly like a Qiskit `QuantumCircuit`. Qiskit circuits stay fully supported; the native circuit is the faster path, because parameter binding happens in Rust without the GIL.

## Building circuits

Gate methods modify the circuit and return it, so they can be chained or called in a loop. `polypus.Param(i)` is the `i`-th variational parameter; a parameter may be used by several gates.

```python
import polypus

bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()

qaoa = polypus.Circuit(4)
for q in range(4):
    qaoa.h(q)
for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
    qaoa.rzz(i, j, polypus.Param(0))
for q in range(4):
    qaoa.rx(q, polypus.Param(1))
qaoa.measure_all()
```

The same builder exists in Rust:

```rust
use polypus_circuit::{ParameterizedCircuit, Param};

let qc = ParameterizedCircuit::new(4)
    .h(0).h(1).h(2).h(3)
    .rzz(0, 1, Param(0)).rzz(1, 2, Param(0)).rzz(2, 3, Param(0)).rzz(3, 0, Param(0))
    .rx(0, Param(1)).rx(1, Param(1)).rx(2, Param(1)).rx(3, Param(1))
    .measure_all();
```

### Angle expressions

An angle can be an expression of the parameters: combine `polypus.Param`, numbers and the constants `polypus.pi`, `polypus.tau` and `polypus.euler` with `+ - * / **` and unary minus, and apply `polypus.sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `exp`, `log` (natural) and `sqrt`. The result, a `polypus.ParamExpr`, is accepted by every angle argument.

```python
import polypus
from polypus import Param

gamma, beta = Param(0), Param(1)
qaoa = polypus.Circuit(2).h(0).h(1).rzz(0, 1, 2 * gamma).rx(0, 2 * beta).rx(1, 2 * beta)
qaoa.measure_all()
qaoa.to_qasm3()  # rzz(2.0*theta_0) q[0], q[1]; ...
```

Expressions follow the same rules as those imported from OpenQASM 3 (contract C-2): a gate method rejects a `NaN` or infinite number, a constant division by zero, and an expression nested more than 64 levels deep with `ValueError`; binding evaluates the expression exactly as written, in binary64 and without reassociation, and raises `ValueError` for a division by zero or a non-finite angle. An expression without parameters (`polypus.pi / 2`) is evaluated when the gate is added. `repr` shows the expression in OpenQASM 3 syntax, with `Param(i)` for each parameter; two expressions compare equal when they are built the same way (`a + b != b + a`). The constants are expressions, not floats: use `math.pi` where a number is needed.

## Templates

`polypus.circuits.templates` builds ready-made circuits in Rust, without the GIL. They are ordinary `polypus.Circuit`s with no free parameters and no measurements: keep chaining gates onto them, measure them, or export them.

`qft(num_qubits, inverse=False, swaps=True)` is the quantum Fourier transform in Qiskit's convention, with the qubit-reversal swaps unless `swaps=False`; `inverse=True` builds its exact adjoint. It takes up to 2827 qubits, the largest QFT the OpenQASM importers read back.

`shor_order_finding(a, modulus, counting_qubits=None)` is the quantum part of Shor's algorithm: measuring its counting register estimates the order `r` of `a` modulo `modulus`, the smallest `r > 0` with `a**r % modulus == 1`. It is Beauregard's circuit with the counting register kept whole, so it runs on any backend. With `n = modulus.bit_length()` and `t = counting_qubits` (`2n` by default), qubits `0..t` are the counting register, then come `x` (`n` qubits, set to 1), `b` (`n + 1`) and an ancilla. Read as an integer, qubit 0 the least-significant bit, the counting register peaks at the values closest to `s * 2**t / r` for `s` in `range(r)`.

```python
import polypus
from polypus.circuits.templates import shor_order_finding

qc = shor_order_finding(7, 15)  # t = 8 counting qubits, 18 qubits in all
for k in range(8):
    qc.measure(k, k)
result = polypus.run_quantum_circuit(
    qc, shots=1024, infrastructure="local", backend="polypus"
)
result.merged_counts  # peaks at 0, 64, 128 and 192: 7 has order 4 modulo 15
```

`a` must be between 1 and `modulus - 1` and share no factor with `modulus` (a shared factor already divides it), and `counting_qubits` between 1 and `2n`; otherwise `ValueError`. Moduli go up to 24 bits: 98 qubits and 3.7 million gates, the largest circuit the OpenQASM importers read back. The native simulator holds up to 30 qubits, moduli of up to 7 bits with the default counting register. `to_qasm3` exports moduli of up to 16 bits; from 17 bits on, the program text would pass the importer's 32 MiB and it raises `ValueError`.

In Rust the templates are in `polypus_circuit::templates`, and the arithmetic blocks `shor_order_finding` is built from are public in `templates::arithmetic`: Draper's `phase_adder` and Beauregard's `modular_phase_adder`, `controlled_modular_multiplier` and `controlled_modular_multiplication`.

```rust
use polypus_circuit::templates::{qft_with_options, shor_order_finding};

let order_finding = shor_order_finding(7, 15, None)?; // 18 qubits, 8561 gates
let inverse_qft = qft_with_options(4, true, true)?;
```

## Drawing

A circuit draws itself, natively and with no plotting library: `print(qc)` shows a text drawing, and a notebook displays an SVG one.

```python
qc = polypus.Circuit(2).h(0).cx(0, 1).rz(1, Param(0)).measure_all()
print(qc)
```

```text
     ┌───┐                       ┌─┐
q0: ─┤ H ├───■───────────────────┤M├─────
     └───┘   │                   └╥┘
           ┌─┴─┐ ┌─────────────┐  ║  ┌─┐
q1: ───────┤ X ├─┤ Rz(theta_0) ├──╫──┤M├─
           └───┘ └─────────────┘  ║  └╥┘
                                  ║   ║
c2: ══════════════════════════════╩═══╩══
                                  0   1
```

`qc.draw()` returns that text; `qc.draw(ascii=True)` draws it with plain ASCII for terminals and logs without Unicode, and `fold` sets the width it wraps at (80 characters by default, `fold=None` never wraps). `qc.draw("svg")` returns a standalone SVG document to save or embed. Angles show as the circuit holds them: a parameter by its name (`param_names`), an expression as written, and a fixed value as a multiple of π when it is one (`π/2`). In Rust, the same drawings are `draw_text(TextOptions { .. })` and `draw_svg()` on `ParameterizedCircuit` and `ConcreteCircuit`, and `Display` prints the default text drawing.

## Export

| Format | Python | Rust |
|---|---|---|
| OpenQASM 2.0 | `qc.to_qasm2(values)` | `to_qasm2_with_params(&values)` |
| OpenQASM 3 | `qc.to_qasm3()`, `qc.to_qasm3(values)` | `to_qasm3`, `to_qasm3_with_params` |
| QIR text (`.ll`) | `qc.to_qir(values)` | `to_qir_with_params(&values)` |
| QIR bitcode (`.bc`) | `qc.to_qir_bitcode(values)` | `to_qir_bitcode_with_params(&values)` |

`values` binds the parameters and can be omitted for a circuit without them. The OpenQASM 2.0 output uses the standard `qelib1.inc` gate names and is accepted by Qiskit (`QuantumCircuit.from_qasm_str`) and Aer. QIR bitcode needs `llvm-as` in `PATH`, since the textual module is assembled externally; it is stopped after 120 s.

## OpenQASM 2.0 import

`Circuit.from_qasm2` is the inverse of `to_qasm2` — it accepts the QASM this library exports **and** Qiskit's `qasm2.dumps` output, `gate` declarations included (every instruction keeps its own spelling, one to one; only the builtins `U`/`CX` become `u`/`cx`; multiple registers are flattened; constant expressions like `pi/2` are evaluated). Parse errors raise `ValueError` with the offending line number.

```python
import polypus
from qiskit import qasm2

qc = polypus.Circuit.from_qasm2(qasm2.dumps(qiskit_circuit))  # interop
qc = polypus.Circuit.from_qasm2(open("ansatz.qasm").read())  # persistence
qc.rz(1, 0.5).measure_all()  # imported circuits are regular builders
```

For any circuit this library produces, exporting, importing and exporting again gives byte-identical text (covered by tests). The same API exists in Rust as `ParameterizedCircuit::from_qasm2`.

## OpenQASM 3 import and export

`Circuit.from_qasm3` and `Circuit.to_qasm3` read and write the **OpenQASM 3 profile with Qiskit phase conventions**: the straight-line part of OpenQASM 3 that carries a parameterised, terminal-measurement circuit. Unlike OpenQASM 2.0, it keeps free parameters: each `input float[64]` is a parameter, in declaration order, under its name (`Circuit.param_names`), and angles may be expressions of them (`rzz(-gamma)`, `p((-pi + x0)*(-pi + x1)*2)`).

```python
import polypus
from qiskit import qasm3

qc = polypus.Circuit.from_qasm3(qasm3.dumps(qiskit_circuit))  # parameters kept
qc.param_names  # the input names Qiskit wrote, in parameter order
text = qc.to_qasm3()  # inputs and expressions; qc.to_qasm3(values) binds first
```

Accepted: `include "stdgates.inc";` (provided internally; no file is read), `qubit` and `bit` registers, `input float[64]` parameters, calls of `U`, of the `stdgates.inc` gates and of gates declared with `gate` blocks, `barrier`, and measurements assigned to bits (`c[i] = measure q[j];`, `c = measure q;`, `measure q -> c;`). Angles use numbers, `pi`/`tau`/`euler`, `+ - * / **`, unary minus and `sin cos tan arcsin arccos arctan exp log sqrt`, evaluated in binary64 exactly as written. Everything else — control flow, `reset`, other classical types and computation, subroutines, gate modifiers, `gphase`, timing, pulses, arrays, physical qubits — raises `ValueError` naming the construct and its line, and so does `1/2`, which OpenQASM 3 types as integer division (write `1.0/2`).

**Phase convention.** `U`, `u2` and `u3` are read and written with Qiskit's matrices (Polypus's `u`, `u2` and `u3`), which differ from the OpenQASM 3 specification's by the global phases e^{-iθ/2} (`U`) and e^{i(φ+λ)/2} (`u2`, `u3`). Statevector amplitudes may therefore differ from those of a reader that follows the specification by these factors; probabilities, counts and expectation values do not. Every other `stdgates.inc` gate follows the specification. This is sound only because the profile rejects gate modifiers and `gphase`, under which a global phase becomes observable. That Polypus reads Qiskit's `qasm3.dumps` output as Qiskit does is tested for Qiskit 2.5.2 and the circuits of the test suite, not guaranteed in general.

The export is canonical: `to_qasm3(from_qasm3(to_qasm3(c)))` is byte-identical to `to_qasm3(c)` (the form is specified in contract C-10 of [`docs/CONTRACTS.md`](../CONTRACTS.md)). `u` is written as `U`; the instructions `stdgates.inc` lacks (`rzz`, `rxx`, `sxdg`, `csx`, `cu1`, `cu3`, `u0`, `rccx`, `rc3x`, `c3x`, `c3sqrtx`, `c4x`) are written with gate definitions of the same matrices; declared gates are printed from their definitions, renamed where a name would be invalid or would clash. On import, `CX`, `phase` and `cphase` become `cx`, `p` and `cp`, and a declared gate stays a declared gate even when it is named like a built-in (`rzz`). `to_qasm2` still needs every parameter bound, and it fails for a gate declared in OpenQASM 3 whose body OpenQASM 2.0 cannot express (`arcsin`, `arccos`, `arctan`). The same API exists in Rust as `ParameterizedCircuit::from_qasm3`, `to_qasm3` and `to_qasm3_with_params`.
