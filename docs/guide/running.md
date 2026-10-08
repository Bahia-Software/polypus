# Running circuits

`polypus.run_quantum_circuit` runs one circuit and returns its measurement counts. It accepts a `polypus.Circuit`, a Qiskit `QuantumCircuit` or an OpenQASM 2.0 string.

```python
import polypus

bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
result = polypus.run_quantum_circuit(bell, shots=1000, infrastructure="local", seed=7)
result.merged_counts  # e.g. {'00': 507, '11': 493}
```

## The result

`run_quantum_circuit` returns a `RunResult` with the same shape whatever `n_qpus` is:

| Field | Content |
|---|---|
| `counts` | `list[dict[str, int]]`, one dict per QPU in the order the shots were split. A QPU left with no shots (when `shots < n_qpus`) has `{}`. |
| `merged_counts` | `dict[str, int]` summing all of them: the circuit's counts over every shot. |
| `seed` | The effective seed. `None` only for `"qmio"`, which is real hardware. |
| `id`, `backend`, `infrastructure` | The run manifest, for logging and replay. |

Pass `seed=...` for reproducible shot noise on every simulated backend (native, Aer and CUNQA's simulated QPUs). `"qmio"` rejects an explicit seed.

## Distributing shots across QPUs

With `n_qpus > 1` the shots are split across QPUs and run concurrently:

```python
result = polypus.run_quantum_circuit(
    qc, shots=10_000, infrastructure="cunqa", n_qpus=10
)
len(result.counts)  # 10, one counts dict per QPU
result.merged_counts  # the total over all 10 000 shots
```

With `infrastructure="cunqa"`, two keyword arguments size the SLURM allocation. Both must be `>= 1`; a `0` raises `ValueError`. `"local"` and `"qmio"` ignore them.

| Parameter | Default | Description |
|---|---|---|
| `nodes` | `1` | SLURM nodes to request |
| `cores_per_qpu` | `2` | CPU cores per QPU |

```python
result = polypus.run_quantum_circuit(
    qc, shots=10_000, infrastructure="cunqa", n_qpus=10, nodes=2, cores_per_qpu=4
)
```

## Choosing a local backend

With `infrastructure="local"`, `backend="aer"` (the default) runs Qiskit Aer and `backend="polypus"` runs the native Rust statevector simulator.

- The native backend runs terminal-measurement circuits only: it rejects `reset`, a gate after a measurement and `if` (see [ADR 0001](../adr/0001-terminal-measurements.md)). It cannot run a Qiskit `QuantumCircuit`.
- Aer runs all of those, but rejects gates outside its basis (such as `ch` or a declared `gate`) unless the circuit is transpiled first.

`polypus.backend_compatibility` returns, per backend, the reasons it would reject a circuit. An empty list means the circuit runs. The check is structural: it runs nothing and does not check resources such as memory.

```python
report = polypus.backend_compatibility(qc)
# e.g. {"aer": [], "polypus": ["native backend could not parse OpenQASM 2.0: ... 'reset' is not supported ..."]}
backend = "polypus" if not report["polypus"] else "aer"
result = polypus.run_quantum_circuit(
    qc, shots=1000, infrastructure="local", backend=backend
)
```

A circuit without measurements is read out on all its qubits by both backends.

## Errors

All Polypus exceptions derive from `polypus.PolypusError`.

- A Qiskit or Aer failure during a run raises `polypus.BackendError`, with the original exception as its `__cause__`.
- A Qiskit failure while `train`, `qml.train` or `qml.predict` prepares or binds your circuits (for example an ansatz wider than the feature map) raises `polypus.EvaluationError` the same way.
- An exception raised by your own `expectation_function` reaches you unchanged.
- A circuit too large for the memory budget raises `polypus.InsufficientMemoryError` before anything is allocated. See [Memory budget](memory.md).

[Writing a backend](../backends.md) documents the backend contract in detail.
