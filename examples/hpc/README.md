# Examples for CESGA's systems

These scripts need services that run only on CESGA's systems. CI type-checks
them against the stubs but does not run them.

| Script | Needs | What it does |
|---|---|---|
| `basic_qmio.py` | QMIO | Runs a one-qubit circuit on the QMIO QPU and prints its counts and timing. |
| `teleport_polypus.py` | QMIO | Teleports a qubit along QMIO's qubits 5-4-3 and compares P(Bob=1) with theory. |
| `teleport_qiskit.py` | QMIO, `qmiotools` | Runs the same teleportation circuit through Qiskit's `QmioBackend` and through Polypus, and compares their times. |
| `qft.py` | `qmiotools` | Transpiles the inverse QFT on 3 to 17 qubits for `QmioBackend` and prints the depth. |
| `polypus_cunqa_basic.py` | CUNQA | Runs a 5-qubit QFT locally and on CUNQA, on one QPU and with the shots split over ten. |
| `single_run.py` | CUNQA | Times one circuit run locally, on one CUNQA QPU and with the shots split over ten. |
| `experiment_max_cut.sh` | CUNQA | Runs `../max_cut_qaoa.py` with `polypus_cunqa` and `polypus_local` on 4 to 6 qubits. |

- **QMIO**: Polypus built with the `qmio` feature (`bash install.sh --qmio`),
  run where the QPU is reachable, with `ZMQ_SERVER` set to its endpoint. See
  [QMIO](../../docs/backends.md#qmio--native-wire-backend) in the backend guide.
- **`qmiotools`**: CESGA's Qiskit integration for QMIO
  (`qmiotools.integrations.qiskitqmio.QmioBackend`).
- **CUNQA**: [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) `>=2.4,<4`
  on the cluster, run from a SLURM job; see
  [Distributing shots across QPUs](../../docs/guide/running.md#distributing-shots-across-qpus).

Run them from the repository root, e.g. `python examples/hpc/single_run.py`:
like the other examples, they write their logs under `logs/` and their results
in the working directory.
