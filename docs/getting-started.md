# Getting started

## Install

```bash
pip install polypus-quantum
```

The package imports as `polypus`. Wheels are published for CPython 3.10 to 3.13 on Linux, macOS on Apple Silicon and Windows.

| To use | Install |
|---|---|
| Local simulators (Aer and native) | `pip install polypus-quantum` |
| A native QML model as a PyTorch layer (`polypus.torch`) | `pip install "polypus-quantum[torch]"` |
| Optuna, Nevergrad or pycma through `polypus.External` | `pip install "polypus-quantum[optuna]"` (`[nevergrad]`, `[cma]`) |
| CUNQA | Polypus, plus [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) `>=2.4,<4` on the cluster. **CUNQA 2.3 is not supported** (it names the `qraise` argument `cores`, not `cores_per_qpu`) |
| QMIO | Polypus built from source with the `qmio` feature: `bash install.sh --qmio` |
| The Rust crates | `cargo add polypus-circuit`, or any crate in the workspace; all but three are pure Rust |

## A first circuit

```python
import polypus

bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
result = polypus.run_quantum_circuit(
    bell, shots=1000, infrastructure="local", backend="polypus", n_qpus=4
)

result.counts  # one counts dict per QPU
result.merged_counts  # e.g. {'00': 487, '11': 513}
```

`backend="polypus"` is the native Rust simulator, which does not load Qiskit Aer. Aer is the default `backend`.

## A first training run

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
    backend="polypus",
    nodes=1,
    cores_per_qpu=1,
    id="maxcut",
)
result.best_params, result.best_fitness
```

## Where next

- [Running circuits](guide/running.md) and [circuits and interchange](guide/circuits.md)
- [Variational training](guide/training.md): choosing an optimizer, callbacks, starting points
- [Quantum machine learning](guide/qml.md)
- [Tutorials](https://github.com/Bahia-Software/polypus-tutorials), from a first circuit to a native classifier
- [Rust API reference](https://bahia-software.github.io/polypus/rust/polypus/)
