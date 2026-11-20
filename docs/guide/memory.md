# Memory budget

A dense `n`-qubit statevector needs `16 · 2^n` bytes (1 GiB at 26 qubits, 16 GiB at 30). The local simulators (native `backend="polypus"`, Aer, and `polypus.statevector`) size their work against one **memory budget**, which they use in two ways:

- **Throttling**: how many circuits of a batch are simulated at once is capped so that their statevectors fit in the budget. This never changes the counts, only the speed.
- **Controlled refusal**: a circuit that cannot fit even on its own is refused **before anything is allocated**, with `polypus.InsufficientMemoryError`, instead of being started and killed by the Linux out-of-memory killer (no Python exception, no partial results). A result an exact computation keeps beside its statevectors counts too: the exact quantum kernel (`qml.kernel_matrix(..., exact=True)`) holds its `rows × columns` matrix (8 bytes an entry) while it simulates, on every route: with room for two statevectors beside it the kernel takes inner products, with room for one it computes each entry from one statevector, and a matrix that leaves no room even for one is refused the same way, with the same exception. The native backend and Aer with `sim_method="statevector"` refuse on the `16 · 2^n` model; for the other Aer methods (`automatic`, `density_matrix`, …) the budget is passed to Aer as `max_memory_mb`, and Aer checks it against the method it actually picks. A large Clifford circuit that Aer runs on its stabilizer method therefore keeps working.

## Setting the budget

`POLYPUS_MEM_BUDGET` takes a positive byte count with an optional `K`/`M`/`G`/`T` suffix, **base 1024** in every spelling and case-insensitive (`32G`, `32GiB`, `32GB` and `32g` are all 32 GiB; `512M`; `1048576`), the same convention as SLURM's `--mem`:

```bash
export POLYPUS_MEM_BUDGET=32G   # e.g. the job's --mem, minus what else runs in it
```

An invalid value (`abc`, `1.5G`, `0`) is not silently dropped: it is reported once per process as a warning in the Polypus log (visible once a logger is installed with `polypus.init_logger`), and the default below is used instead.

## The default

When the variable is unset or invalid, the budget is the smaller of the RAM available when the process starts (`MemAvailable`) and the process's cgroup memory limit (cgroup v2 `memory.max` of its cgroup and every ancestor, or cgroup v1 `memory.limit_in_bytes`), minus a safety reserve of 10 % of that limit, at least 512 MiB and at most half of it. When nothing can be detected (macOS, Windows), the budget is a fixed 16 GiB. Because that is a guess, it only throttles: no circuit is refused for lack of memory against it. What still stops a run there is the qubit ceiling below: 30 qubits.

> [!WARNING]
> **On a shared machine with no cgroup limit** (a login node, a workstation several people use), the default is whatever RAM happened to be free when your process started, so a single run may take most of it. Set `POLYPUS_MEM_BUDGET` to your fair share there.

## The qubit ceiling

The native simulator (`backend="polypus"` and `polypus.statevector`) holds up to **40 qubits** (16 TiB), and below that the budget decides (issue #428):

| Budget | 31 to 40 qubits |
|---|---|
| Set with `POLYPUS_MEM_BUDGET`, or detected (Linux) | runs if its statevector fits, `polypus.InsufficientMemoryError` before allocating otherwise |
| None detected (macOS, Windows) | refused as above 40: `ValueError` from `polypus.statevector`, `polypus.NativeCircuitError` from a run, the message naming `POLYPUS_MEM_BUDGET` |

So on a 64 GiB node the budget lets 31 and 32 qubits run, and refuses 33 and above; on macOS, setting `POLYPUS_MEM_BUDGET=64G` does the same. Above 40 qubits the ceiling's error is unconditional. If the system still refuses the memory after the budget let a circuit through (the budget was set above the memory really free), the run fails with a message saying so instead of crashing the process; on Linux the system may also grant the memory and kill the process later, so set the budget to what is truly available.

## Several processes in one cgroup

Processes in one cgroup (the MPI ranks of one SLURM job step, say) each see the *whole* cgroup limit, so each would budget for all of it. Set `POLYPUS_MEM_BUDGET` per process, e.g. the job's memory divided by the ranks per node.

## When a circuit is refused

The `InsufficientMemoryError` message gives the qubit count, the memory required, the budget and where it came from. It is a `polypus.BackendError`, so `except polypus.PolypusError` catches it. If more memory really is available (the detection is conservative, or you accept the risk), force the run with a larger explicit budget, which always takes precedence:

```python
import polypus

qc = polypus.Circuit(30).h(0).measure_all()
try:
    polypus.run_quantum_circuit(
        qc, shots=100, infrastructure="local", backend="polypus"
    )
except polypus.InsufficientMemoryError as exc:
    print(exc)  # ... set POLYPUS_MEM_BUDGET (e.g. POLYPUS_MEM_BUDGET=64G) to override
```

## Limits of the detection

It reads `memory.max` / `memory.limit_in_bytes` only: a cgroup v2 `memory.high` soft limit and the cgroup's swap allowance are ignored. It does not subtract `memory.current`, which includes reclaimable page cache. And `MemAvailable` is read once, when the process first needs the budget, so memory freed or taken by other processes afterwards is not seen; an explicit `POLYPUS_MEM_BUDGET` is re-read on every run.
