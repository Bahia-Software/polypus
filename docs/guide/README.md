# User guide

- [Running circuits](running.md): `RunResult`, multi-QPU runs, seeds, choosing a local backend, errors, troubleshooting
- [Variational training](training.md): `train` parameters, `TrainResult`, cost functions, DE, PSO, QNG, CMA-ES, Adam, SPSA, Rotosolve and external optimizers (Optuna, Nevergrad, pycma)
- [Quantum machine learning](qml.md): `qml.train` with Qiskit circuits, labels, `SampleCost` and `qml.predict`; native models with `qml.Model`: layers, readout, losses, `fit`, prediction, saving
- [Circuits and interchange](circuits.md): `polypus.Circuit`, the Rust builder, OpenQASM 2 and 3, QIR
- [Memory budget](memory.md): `POLYPUS_MEM_BUDGET`, cgroup detection, `InsufficientMemoryError`
- [Performance](performance.md): batched simulation, GIL-free binding, gate-parallel auto-calibration

For writing a new backend, see [Writing a backend](../backends.md).
