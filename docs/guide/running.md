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

Pass `seed=...` for reproducible shot noise on every simulated backend (native, Aer and CUNQA's simulated QPUs). `"qmio"` rejects an explicit seed, and also a `noise_model` or a `sim_method` other than `"automatic"`: both apply only to Aer.

## Distributing shots across QPUs

With `n_qpus > 1` the shots are split across QPUs and run concurrently:

```python
result = polypus.run_quantum_circuit(
    qc, shots=10_000, infrastructure="cunqa", n_qpus=10
)
len(result.counts)  # 10, one counts dict per QPU
result.merged_counts  # the total over all 10 000 shots
```

`infrastructure="cunqa"` needs [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) `>=2.4,<4` on the cluster. **CUNQA 2.3 is not supported**: it names the `qraise` argument `cores`, not `cores_per_qpu`, so the reservation fails with a `TypeError`.

CUNQA's QPUs run Aer, so with `infrastructure="cunqa"` the `backend` must be `"aer"` (the default) or `"AerSimulator"`, and `noise_model` is not accepted; either one raises `ValueError` before any QPU is requested.

With `infrastructure="cunqa"`, three keyword arguments size the SLURM allocation. `nodes` and `cores_per_qpu` must be `>= 1`; a `0` raises `ValueError`. `"local"` and `"qmio"` ignore all three.

| Parameter | Default | Description |
|---|---|---|
| `nodes` | `1` | SLURM nodes to request |
| `cores_per_qpu` | `2` | CPU cores per QPU |
| `walltime` | `"10:00:00"` | SLURM walltime of the QPUs, as `HH:MM:SS` |

`walltime` accepts only `HH:MM:SS`: hours are one or more digits (so `"48:00:00"` is valid), minutes and seconds are two digits below 60, and the total must be above zero. SLURM's `D-HH:MM:SS` form is rejected. Anything else raises `ValueError` before any QPU is requested. The QPUs are released when the walltime runs out, so set it above the expected duration of a long optimisation. Training logs a warning after the first generation when, at that pace, the run would outlast the walltime (see [Allocation walltime](../backends.md#allocation-walltime)).

```python
result = polypus.run_quantum_circuit(
    qc,
    shots=10_000,
    infrastructure="cunqa",
    n_qpus=10,
    nodes=2,
    cores_per_qpu=4,
    walltime="48:00:00",
)
```

## Choosing a local backend

With `infrastructure="local"`, `backend="aer"` (the default) runs Qiskit Aer and `backend="polypus"` runs the native Rust statevector simulator.

- The native backend runs terminal-measurement circuits only: it rejects `reset`, a gate after a measurement and `if` (see [ADR 0001](../adr/0001-terminal-measurements.md)). It cannot run a Qiskit `QuantumCircuit`.
- Aer runs all of those, but rejects gates outside its basis (such as `ch` or a declared `gate`) unless the circuit is transpiled first.
- The native backend is a noiseless statevector simulator: it rejects a `noise_model` and a `sim_method` other than `"automatic"` with `ValueError`.

`polypus.backend_compatibility` returns, per backend (`aer`, `polypus`, `cunqa`, `qmio`), the reasons it would reject a circuit. An empty list means the circuit runs. It runs and connects to nothing. By default it is structural; with `resources=True` it also checks the native backend's qubit ceiling and memory budget.

```python
report = polypus.backend_compatibility(qc)
# e.g. {"aer": [], "polypus": ["native backend could not parse OpenQASM 2.0: ... 'reset' is not supported ..."], ...}
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
- A circuit the native backend cannot run raises `polypus.NativeCircuitError`. That includes a measured circuit whose classical register is wider than 64 bits (for example `creg c[100];`, or `measure(0, 70)`); use `backend="aer"` for it.
- Invalid circuit construction (an out-of-range qubit or classical bit, a non-finite angle) raises `ValueError`.

Polypus is written not to panic. If a Rust panic does happen, it is a bug: it reaches Python as `pyo3_runtime.PanicException`, which derives from `BaseException`, so neither `except Exception` nor `except polypus.PolypusError` catches it. Please report it.

## Troubleshooting

### `import qiskit_aer` fails with "cannot allocate memory in static TLS block"

```text
ImportError: .../qiskit_aer.libs/libgomp-<hash>.so.1.0.0: cannot allocate memory in static TLS block
```

Aer bundles its own OpenMP runtime, `libgomp`, which needs room in glibc's static TLS reserve when Python loads it. If the libraries loaded before it have filled the reserve, the import fails, and so does every run on `backend="aer"`, the default. Polypus is not involved: `import qiskit_aer` alone fails the same way. Qiskit 2.6.0 is a known cause, for instance on the `python:3.10-slim` to `python:3.12-slim` images (Debian 13, glibc 2.41); Qiskit 2.6.1 fixes it ([Qiskit/qiskit#17213](https://github.com/Qiskit/qiskit/issues/17213)):

```bash
pip install --upgrade "qiskit>=2.6.1"
```

Otherwise, run on the native simulator, `backend="polypus"`, which does not load Aer, or enlarge the reserve before Python starts:

```bash
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
```

The dynamic loader reads the variable when the process starts, so setting it in `os.environ` from Python has no effect. In a Dockerfile, set it with `ENV`.

[Writing a backend](../backends.md) documents the backend contract in detail.
