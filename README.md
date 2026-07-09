<p align="center">
  <img src="docs/Logo.png" alt="Polypus logo" width="350">
</p>

<h1 align="center">Polypus</h1>

<p align="center">
  <strong>A distributed quantum computing library — Rust core, Python bindings, one API across simulators and real QPUs.</strong>
</p>

<p align="center">
  <a href="https://github.com/Bahia-Software/polypus/actions/workflows/ci.yml"><img src="https://github.com/Bahia-Software/polypus/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-EUPL--1.2-blue.svg" alt="License: EUPL-1.2"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+">
  <a href="https://bahia-software.github.io/polypus/"><img src="https://img.shields.io/badge/docs-rustdoc-blue.svg" alt="Documentation"></a>
  <img src="https://img.shields.io/badge/status-active%20development-yellow.svg" alt="Status: active development">
</p>

## Table of Contents

- [What is Polypus?](#what-is-polypus)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Installation](#installation)
  - [CUNQA](#cunqa)
- [Usage](#usage)
  - [Running a Quantum Circuit](#running-a-quantum-circuit)
  - [Distributing Shots Across Multiple QPUs](#distributing-shots-across-multiple-qpus)
  - [Training Variational Circuits](#training-variational-circuits)
    - [Differential Evolution](#differential-evolution)
    - [Particle Swarm Optimization](#particle-swarm-optimization)
    - [Quantum Natural Gradient](#quantum-natural-gradient)
- [Rust-Native Circuits](#rust-native-circuits)
  - [From Python](#from-python)
  - [From Rust](#from-rust)
  - [QASM 2.0 Import](#qasm-20-import)
  - [Performance Notes](#performance-notes)
- [Project Architecture](#project-architecture)
- [Documentation](#documentation)
- [Citing Polypus](#citing-polypus)
- [Credits](#credits)
- [Dependencies](#dependencies)
- [License](#license)

## What is Polypus?

Polypus is an open-source **distributed quantum computing library**: it runs quantum circuits and trains variational quantum algorithms (VQE, QAOA, QML, …) across one or many QPUs — simulated or real — without changing your circuit code. The core is written in **Rust** for performance and correctness; **Python bindings** (via PyO3) make it a drop-in accelerator for existing Qiskit workflows.

It is built for researchers and engineers who need to:
- scale shot execution and population-based training across multiple QPUs instead of one,
- swap between a local simulator, CESGA's CUNQA platform, or CESGA's QMIO real QPU without touching algorithm code,
- keep a Python-friendly API while getting Rust-level performance on the hot paths (parameter binding, batched simulation).

> **Status:** Polypus is under active development (current version: `0.4.0`) and is not yet published on PyPI. Install from source — see [Installation](#installation).

## Key Features

- **Multi-QPU execution** — run any Qiskit `QuantumCircuit`, a native `polypus.Circuit`, or an OpenQASM 2.0 string, and automatically split shots across `n_qpus` to cut wall-clock time.
- **Unified variational training** — Differential Evolution, Particle Swarm Optimization, and Quantum Natural Gradient behind a single `polypus.train()` API; switch optimizers by changing one argument, populations are distributed across QPUs automatically.
- **Backend-agnostic** — local simulation via Qiskit Aer, CESGA's [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) distributed QPU platform, and CESGA's QMIO real quantum processor, the last reached through a pure-Rust, GIL-free ZeroMQ client.
- **Native circuit engine** (`polypus-circuit`) — optional pure-Rust circuit representation with OpenQASM 2.0 and QIR export; parameter binding is ~3x faster than Qiskit's `assign_parameters` and GIL-free, so concurrent evaluation threads bind candidates truly in parallel.
- **A real Cargo workspace, not a monolith** — circuits, simulator, physics layer, optimizers, and logger are independent, individually testable crates; the pure-Rust ones have no Python dependency and are usable from any Rust project.

## Quickstart

```bash
git clone https://github.com/Bahia-Software/polypus.git
cd polypus
bash install.sh --no-tests
```

```python
import polypus

bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
counts = polypus.run_quantum_circuit(bell, shots=1000, infrastructure="local")
print(counts)  # e.g. {'00': 512, '11': 488}
```

See [Usage](#usage) for Qiskit circuits, multi-QPU distribution, and variational training.

## Installation

The recommended way to install Polypus is using the provided script:

```bash
bash install.sh
```

This will interactively guide you through installing dependencies, building the wheel, and optionally running the test suite. For non-interactive environments (CI/CD):

```bash
bash install.sh --yes        # all defaults, run all tests
bash install.sh --no-tests   # skip tests
```

<details>
<summary>Manual installation</summary>

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Build and install the `polypus_python` package:

```bash
python -m build packages/polypus_python/
pip install packages/polypus_python/
```

Build the Rust extension:

```bash
maturin develop --release --features extension-module
```

</details>

### CUNQA

To install CUNQA please refer to the [CUNQA GitHub repository](https://github.com/CESGA-Quantum-Spain/cunqa).

## Usage

Check the [`examples/`](examples/) directory for complete, runnable scripts.

### Running a Quantum Circuit

Pass the circuit, the number of shots, the infrastructure and the number of QPUs:
```python
result = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure=INFRASTRUCTURE, n_qpus=1)
```

### Distributing Shots Across Multiple QPUs

Set `n_qpus > 1` to split the shots across available QPUs and reduce execution time:
```python
result = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure=INFRASTRUCTURE, n_qpus=10)
```

### Training Variational Circuits

Polypus optimizes variational quantum circuits via `polypus.train()`. The optimizer is selected by passing a method object as the second argument. Polypus distributes the population individuals across the available QPUs automatically.

The common parameters for all methods are:

| Parameter | Description |
|---|---|
| `qc` | Parameterized quantum circuit |
| `shots` | Number of shots per circuit execution |
| `n_qpus` | Number of QPUs to use |
| `dimensions` | Number of variational parameters |
| `expectation_function` | Python callable that maps a bitstring to a cost value |
| `infrastructure` | `"local"` or `"cunqa"` |
| `nodes` | Number of nodes (CUNQA only) |
| `cores_per_qpu` | Cores per QPU (CUNQA only) |
| `id` | Experiment identifier for logging |

If CUNQA is not available, set `infrastructure="local"`.

#### Differential Evolution

```python
result_params = polypus.train(
    qc,
    polypus.DE(generations=MAX_GENERATIONS, population_size=POPULATION_SIZE, tolerance=TOL),
    shots=N_SHOTS,
    n_qpus=N_QPUS,
    dimensions=2 * layers,
    expectation_function=bitstring_to_obj,
    infrastructure=infrastructure,
    nodes=NUM_NODES,
    cores_per_qpu=CORES_PER_QPU,
    id=id
)
```

#### Particle Swarm Optimization

```python
result_params = polypus.train(
    qc,
    polypus.PSO(generations=MAX_GENERATIONS, population_size=POPULATION_SIZE,
                bounds=(0.0, np.pi), tolerance=TOL),
    shots=N_SHOTS,
    n_qpus=N_QPUS,
    dimensions=2 * layers,
    expectation_function=bitstring_to_obj,
    infrastructure=infrastructure,
    nodes=NUM_NODES,
    cores_per_qpu=CORES_PER_QPU,
    id=id
)
```

#### Quantum Natural Gradient

`QNG` requires a `variance_function` callable that estimates the diagonal quantum Fisher information matrix element for each parameter. Default hyperparameters: `learning_rate=0.1`, `finite_difference_step=0.1`, `tikhonov_reg=0.05`.

```python
result_params = polypus.train(
    qc,
    polypus.QNG(variance_fn, max_iters=MAX_GENERATIONS, bounds=(0.0, np.pi),
                learning_rate=0.1, finite_difference_step=0.1, tikhonov_reg=0.05),
    shots=N_SHOTS,
    n_qpus=N_QPUS,
    dimensions=2 * layers,
    expectation_function=bitstring_to_obj,
    infrastructure=infrastructure,
    nodes=NUM_NODES,
    cores_per_qpu=CORES_PER_QPU,
    id=id
)
```

## Rust-Native Circuits

The repository is a Cargo workspace; [`polypus-circuit`](crates/polypus-circuit) is its pure-Rust quantum circuit crate — no PyO3/Python dependency, so circuits can be built and serialized without the GIL. `crates/polypus` (the Python extension) re-exports its API as `polypus.Circuit`. Qiskit circuits remain fully supported; the Rust API is an additional, high-performance path.

### From Python

`polypus.Circuit` is accepted by `run_quantum_circuit` and `train` exactly like a Qiskit `QuantumCircuit` (an OpenQASM 2.0 string also works for `run_quantum_circuit`):

```python
import polypus

# Fully bound circuit → run directly
bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
counts = polypus.run_quantum_circuit(bell, shots=1000, infrastructure="local")

# Parameterized ansatz → train (binding happens in Rust, GIL-free)
qaoa = (polypus.Circuit(4)
        .h(0).h(1).h(2).h(3)
        .rzz(0, 1, polypus.Param(0)).rzz(1, 2, polypus.Param(0))
        .rzz(2, 3, polypus.Param(0)).rzz(3, 0, polypus.Param(0))
        .rx(0, polypus.Param(1)).rx(1, polypus.Param(1))
        .rx(2, polypus.Param(1)).rx(3, polypus.Param(1))
        .measure_all())
params = polypus.train(qaoa, polypus.DE(generations=100, population_size=50),
                       shots=1024, n_qpus=1, dimensions=2,
                       expectation_function=my_cost, infrastructure="local",
                       nodes=1, cores_per_qpu=1, id="qaoa")
```

### From Rust

```rust
use polypus_circuit::{ParameterizedCircuit, Param};

let qc = ParameterizedCircuit::new(4)
    .h(0).h(1).h(2).h(3)
    .rzz(0, 1, Param(0)).rzz(1, 2, Param(0)).rzz(2, 3, Param(0)).rzz(3, 0, Param(0))
    .rx(0, Param(1)).rx(1, Param(1)).rx(2, Param(1)).rx(3, Param(1))
    .measure_all();

let qasm: String = qc.to_qasm2_with_params(&[0.4, 0.8])?; // OpenQASM 2.0
let qir_ll: String = qc.to_qir_with_params(&[0.4, 0.8])?; // QIR LLVM IR text (.ll)
let qir_bc: Vec<u8> = qc.to_qir_bitcode_with_params(&[0.4, 0.8])?; // QIR bitcode (.bc)
```

The generated OpenQASM 2.0 uses standard `qelib1.inc` gate names and is accepted by Qiskit (`QuantumCircuit.from_qasm_str`) and Aer.

`to_qir_bitcode_with_params` requires `llvm-as` available in `PATH` because Polypus assembles the textual QIR module into LLVM bitcode externally.

From Python, both outputs are also available:

```python
qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
qir_text = qc.to_qir()            # str (.ll)
qir_bitcode = qc.to_qir_bitcode() # bytes (.bc)
```

### QASM 2.0 Import

`Circuit.from_qasm2` is the inverse of `to_qasm2` — it accepts the QASM this library exports **and** Qiskit's `qasm2.dumps` output (`u`/`p`/`u1`/`u2` are canonicalised to `u3`, `swap` to its `cx` decomposition; multiple registers are flattened; constant expressions like `pi/2` are evaluated). Parse errors raise `ValueError` with the offending line number.

```python
import polypus
from qiskit import qasm2

qc = polypus.Circuit.from_qasm2(qasm2.dumps(qiskit_circuit))  # interop
qc = polypus.Circuit.from_qasm2(open("ansatz.qasm").read())   # persistence
qc.rz(1, 0.5).measure_all()        # imported circuits are regular builders
```

Round-trip guarantee (verified by tests): for any circuit produced by this library, export → import → export is byte-identical. The same API exists in Rust as `ParameterizedCircuit::from_qasm2`.

### Performance Notes

- **Parameter binding**: ~3x faster than Qiskit's `assign_parameters` and, crucially, **GIL-free** — concurrent evaluation threads bind candidates truly in parallel (see `benchmarks/bench_native_vs_qiskit.py`).
- **Batched simulation**: the local backend submits each evaluation batch (e.g. a whole DE population) in a *single* `AerSimulator.run` call with `max_parallel_experiments=0`, so Aer's C++ engine runs the experiments in parallel across cores with the GIL released. Measured ~1.4–2.1x end-to-end training speedup vs per-circuit submission, growing with circuit size (see `benchmarks/bench_batching.py`). Distributed backends cap each call at `n_qpus` via `QuantumBackend::max_batch_size`.
- Native circuits shine brightest with backends that consume OpenQASM directly (e.g. CUNQA), where the Qiskit re-parse disappears entirely.

## Project Architecture

Polypus is a Cargo workspace of focused crates plus one Python package:

| Path | Language | Role |
|---|---|---|
| [`crates/polypus`](crates/polypus) | Rust + PyO3 | Main library and Python extension module: backends, training loop, Python bindings |
| [`crates/polypus-circuit`](crates/polypus-circuit) | Pure Rust | Circuit representation, OpenQASM 2.0 / QIR export — no PyO3, usable from any Rust project |
| [`crates/polypus-sim`](crates/polypus-sim) | Pure Rust | Statevector simulator consuming `polypus-circuit`'s `ConcreteCircuit` directly (no OpenQASM round-trip) |
| [`crates/polypus-optimizers`](crates/polypus-optimizers) | Pure Rust | DE, PSO and QNG optimizers, decoupled from circuits and Python via `EvaluationOracle`/`VarianceOracle` |
| [`crates/polypus-physics`](crates/polypus-physics) | Pure Rust | Classical Monte Carlo transport and quantum Hamiltonians expressed as Pauli sums |
| [`crates/polypus-logger`](crates/polypus-logger) | Pure Rust | Shared `log::Log` sink for the whole workspace |
| [`packages/polypus_python`](packages/polypus_python) | Python | Python-side infrastructure glue (backend connectivity, worker processes) used by the extension module |

Only `crates/polypus` links against Python; every other Rust crate is dependency-free with respect to PyO3 and can be used standalone from any Rust project.

## Documentation

- API reference (generated with `cargo doc`, covers the Rust crates and the PyO3 bindings): **https://bahia-software.github.io/polypus/**
- Runnable, end-to-end scripts: [`examples/`](examples/)
- Performance benchmarks and how to reproduce them: [`benchmarks/`](benchmarks/)

## Citing Polypus

If Polypus is useful in your research, please cite it:

```bibtex
@software{polypus,
  author  = {Fernández Prada, Diego Beltrán and Sóñora Pombo, Víctor and Figueiras Gómez, Sergio and Boubeta Martínez, Miguel and {Galicia Supercomputing Center (CESGA)}},
  title   = {{Polypus: A Distributed Quantum Computing Library}},
  year    = {2026},
  url     = {https://github.com/Bahia-Software/polypus},
  version = {0.4.0}
}
```

## Credits
- Diego Beltrán Fernández Prada
- Víctor Sóñora Pombo
- Sergio Figueiras Gómez
- Miguel Boubeta Martínez
- Galicia Supercomputing Center (CESGA)

## Dependencies

Polypus relies on the following Python packages:

| Package | Version |
|---|---|
| `qiskit` | ≥ 2.0 |
| `qiskit-aer` | ≥ 0.17 |
| `numpy` | ≥ 2.0 |
| `scipy` | ≥ 1.13 |
| `matplotlib` | ≥ 3.9 |
| `networkx` | ≥ 3.2 |
| `cunqa` | ≥ 2.3 (optional) |

See [`requirements-dev.txt`](requirements-dev.txt) and [`requirements-examples.txt`](requirements-examples.txt) for the full pinned dependency lists.

## License

Polypus is licensed under the **European Union Public Licence, version 1.2 (EUPL-1.2)**. See [`LICENSE`](LICENSE) for the full text.
