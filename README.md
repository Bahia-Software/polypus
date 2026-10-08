<p align="center">
  <img src="https://raw.githubusercontent.com/Bahia-Software/polypus/main/assets/logo.png" alt="Polypus" width="260">
</p>

<h3 align="center">Run and train quantum circuits across many QPUs, simulated or real, with one API.</h3>

<p align="center">
  <a href="https://pypi.org/project/polypus-quantum/"><img src="https://img.shields.io/pypi/v/polypus-quantum.svg?label=PyPI" alt="PyPI"></a>
  <a href="https://crates.io/crates/polypus"><img src="https://img.shields.io/crates/v/polypus.svg?label=crates.io" alt="crates.io"></a>
  <a href="https://github.com/Bahia-Software/polypus/actions/workflows/ci.yml"><img src="https://github.com/Bahia-Software/polypus/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/polypus-quantum/"><img src="https://img.shields.io/pypi/pyversions/polypus-quantum.svg" alt="Python versions"></a>
  <a href="https://github.com/Bahia-Software/polypus/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-EUPL--1.2-blue.svg" alt="License: EUPL-1.2"></a>
  <a href="https://doi.org/10.5281/zenodo.22913065"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22913065.svg" alt="DOI"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="https://github.com/Bahia-Software/polypus/tree/main/docs/guide">User guide</a> |
  <a href="https://github.com/Bahia-Software/polypus-tutorials">Tutorials</a> |
  <a href="https://bahia-software.github.io/polypus/">API reference</a> |
  <a href="#citing-polypus">Cite</a>
</p>

---

Polypus is a quantum computing library with a **Rust core** and **Python bindings**. It splits shots and optimizer populations across `n_qpus` and runs them on a local simulator, on CESGA's [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) distributed QPUs or on CESGA's QMIO quantum processor. Changing the target means changing one argument, not the algorithm code.

<p align="center">
  <img src="https://raw.githubusercontent.com/Bahia-Software/polypus/main/assets/overview.svg" alt="Circuits enter the Python API; the Rust core plans the work and runs it on local Aer, the native simulator, CUNQA or QMIO" width="820">
</p>

## Highlights

| Feature | |
|---|---|
| **Multi-QPU execution** | Shots and populations are split across `n_qpus`; results come back per QPU and merged. |
| **Variational training** | Differential Evolution, Particle Swarm and Quantum Natural Gradient behind one `train()` call, plus supervised QML with `qml.train` and `qml.predict`. |
| **Native cost observables** | QUBO and Ising costs are evaluated in Rust, without a Python call per bitstring. |
| **Rust circuit engine** | `polypus.Circuit` imports and exports OpenQASM 2 and 3 and exports QIR. Parameter binding runs without the GIL, about 3x faster than Qiskit's `assign_parameters`. |
| **Reproducible runs** | Every result reports the seed it used; pass it back to replay the run. |
| **Safe on shared machines** | Simulations are sized against a memory budget, and a circuit that cannot fit is refused before allocation instead of being killed by the OOM killer. |

## Installation

```bash
pip install polypus-quantum
```

The package imports as `polypus`. Wheels are published for Linux, macOS and Windows, Python 3.9 and later.

| To use | Install |
|---|---|
| Local simulators (Aer and native) | `pip install polypus-quantum` |
| CUNQA | Polypus, plus [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) 2.3 or later on the cluster |
| QMIO | Polypus built from source with `--features qmio` |
| The Rust crates | `cargo add polypus` |

<details>
<summary>Building from source</summary>

```bash
git clone https://github.com/Bahia-Software/polypus.git
cd polypus
bash install.sh            # interactive; --yes takes the defaults, --no-tests skips the test suite
```

Or by hand:

```bash
pip install -r requirements-dev.txt
maturin build --release --out dist
pip install dist/polypus_quantum-*.whl
```

Install the wheel rather than using `maturin develop`, which leaves out the bundled `polypus_python` helpers.
</details>

> The name `polypus` on PyPI belongs to an unrelated project, so the distribution is `polypus-quantum`. The import name and the Rust crate are `polypus`.

## Quickstart

**Run a circuit on 4 QPUs**

```python
import polypus

bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
result = polypus.run_quantum_circuit(bell, shots=1000, infrastructure="local", n_qpus=4)

result.counts  # one counts dict per QPU
result.merged_counts  # e.g. {'00': 487, '11': 513}
```

Qiskit `QuantumCircuit` objects and OpenQASM 2.0 strings are accepted the same way.

**Train a QAOA circuit for MaxCut**

```python
import polypus

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
gamma, beta = polypus.Param(0), polypus.Param(1)

qaoa = polypus.Circuit(4)
for q in range(4):
    qaoa.h(q)
for i, j in edges:
    qaoa.rzz(i, j, gamma)
for q in range(4):
    qaoa.rx(q, beta)
qaoa.measure_all()

# Cut size as a QUBO, evaluated natively. Optimizers maximise.
cut = polypus.Qubo(
    4,
    linear=[(q, 2.0) for q in range(4)],
    quadratic=[(i, j, -2.0) for i, j in edges],
)

result = polypus.train(
    qaoa,
    polypus.DE(generations=100, population_size=50, seed=42),
    shots=1024,
    n_qpus=1,
    dimensions=2,
    expectation_function=cut,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="maxcut",
)
result.best_params, result.best_fitness
```

To try another optimizer, replace `polypus.DE` with `polypus.PSO` or `polypus.QNG`; to run on CUNQA, set `infrastructure="cunqa"`. The rest of the code stays the same. The [tutorials](https://github.com/Bahia-Software/polypus-tutorials) go from a first circuit to QML step by step, and [`examples/`](https://github.com/Bahia-Software/polypus/tree/main/examples) has complete scripts.

## Backends

| `infrastructure` | `backend` | Runs on | Seeded | Notes |
|---|---|---|---|---|
| `"local"` | `"aer"` (default) | Qiskit Aer | Yes | Full Qiskit circuit support |
| `"local"` | `"polypus"` | Native Rust statevector | Yes | Terminal measurements only ([ADR 0001](https://github.com/Bahia-Software/polypus/blob/main/docs/adr/0001-terminal-measurements.md)) |
| `"cunqa"` | | CUNQA QPUs over SLURM | Yes | `nodes` and `cores_per_qpu` size the allocation |
| `"qmio"` | | QMIO quantum processor | No | Pure-Rust ZeroMQ client; build with `--features qmio` |

`polypus.backend_compatibility(qc)` lists, for each backend, why it would reject a circuit, without running it. Third-party backends plug in through a Rust trait or a Python worker; see [Writing a backend](https://github.com/Bahia-Software/polypus/blob/main/docs/backends.md).

## User guide

| Topic | Covers |
|---|---|
| [Running circuits](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/running.md) | `RunResult`, multi-QPU runs, seeds, choosing a local backend, errors |
| [Variational training](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/training.md) | `train` parameters, `TrainResult`, cost functions, DE, PSO and QNG |
| [Quantum machine learning](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/qml.md) | `qml.train` with labels, `SampleCost`, `qml.predict` |
| [Circuits and interchange](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/circuits.md) | `polypus.Circuit`, the Rust builder, OpenQASM 2 and 3, QIR |
| [Memory budget](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/memory.md) | `POLYPUS_MEM_BUDGET`, cgroup detection, `InsufficientMemoryError` |
| [Performance](https://github.com/Bahia-Software/polypus/blob/main/docs/guide/performance.md) | Batched simulation, GIL-free binding, gate-parallel auto-calibration |

The [API reference](https://bahia-software.github.io/polypus/) is generated from the Rust crates and the Python bindings.

## Architecture

Polypus is a Cargo workspace. Only the three crates marked PyO3 link against Python; the others can be used from any Rust project.

<p align="center">
  <img src="https://raw.githubusercontent.com/Bahia-Software/polypus/main/assets/architecture.svg" alt="Crate graph: polypus depends on polypus-evaluation, then polypus-orchestration, which uses polypus-infrastructure and polypus-optimizers; polypus-infrastructure uses polypus-backend, polypus-sim and polypus-observable; polypus-sim uses polypus-circuit" width="760">
</p>

<details>
<summary>All crates</summary>

| Crate | Role |
|---|---|
| [`polypus`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus) | Python extension, exception hierarchy, re-exports |
| [`polypus-circuit`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-circuit) | Circuit representation, OpenQASM 2 and 3, QIR |
| [`polypus-sim`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-sim) | Statevector simulator |
| [`polypus-optimizers`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-optimizers) | DE, PSO and QNG |
| [`polypus-observable`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-observable) | QUBO and Ising cost observables |
| [`polypus-physics`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-physics) | Monte Carlo transport and Pauli-sum Hamiltonians |
| [`polypus-backend`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-backend) | The `QuantumBackend` trait that backends implement |
| [`polypus-backend-conformance`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-backend-conformance) | Conformance tests for backends |
| [`polypus-subprocess-backend`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-subprocess-backend) | Bridge for backends written in Python |
| [`polypus-infrastructure`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-infrastructure) | Built-in backends and the `Planner` that sizes circuit waves |
| [`polypus-orchestration`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-orchestration) | `Flow`, `Scheduler` and `Resources`; optimizer dispatch |
| [`polypus-evaluation`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-evaluation) | Oracles and objective assembly |
| [`polypus-logger`](https://github.com/Bahia-Software/polypus/tree/main/crates/polypus-logger) | Shared log sink |
| [`polypus_python`](https://github.com/Bahia-Software/polypus/tree/main/polypus_python) | Python-side backend glue, bundled in the wheel |

</details>

Design notes: [contracts](https://github.com/Bahia-Software/polypus/blob/main/docs/CONTRACTS.md), [engineering rules](https://github.com/Bahia-Software/polypus/blob/main/docs/ENGINEERING.md), [architecture decisions](https://github.com/Bahia-Software/polypus/tree/main/docs/adr). To contribute, see [CONTRIBUTING](https://github.com/Bahia-Software/polypus/blob/main/docs/CONTRIBUTING.md).

## Citing Polypus

If Polypus is useful in your research, please cite it:

```bibtex
@software{polypus,
  author  = {Fernández Prada, Diego Beltrán and Sóñora Pombo, Víctor and Figueiras Gómez, Sergio and Boubeta Martínez, Miguel and Pérez González, Kevin and Sendón Caamaño, Uxía and {Galicia Supercomputing Center (CESGA)}},
  title   = {{Polypus: A Distributed Quantum Computing Library}},
  year    = {2026},
  url     = {https://github.com/Bahia-Software/polypus},
  doi     = {10.5281/zenodo.22913065},
  version = {0.7.2}
}
```

The same metadata is in [`CITATION.cff`](https://github.com/Bahia-Software/polypus/blob/main/CITATION.cff), which GitHub shows as *Cite this repository*.

## License

Polypus is developed by Bahía Software with the Galicia Supercomputing Center (CESGA) and is licensed under the [European Union Public Licence 1.2](https://github.com/Bahia-Software/polypus/blob/main/LICENSE).
