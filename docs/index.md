---
title: Polypus
---

# Polypus

**Run and train quantum circuits across many QPUs, simulated or real, with one API.**

A Rust core with Python bindings. Changing the target, a local simulator, CESGA's CUNQA or QMIO, means changing one argument, not the algorithm code.

![Circuits enter the Python API; the Rust core plans the work and runs it on local Aer, the native simulator, CUNQA or QMIO](assets/overview.svg)

<div class="grid cards" markdown>

- **Run**

    Split shots across `n_qpus` and merge the counts.

    ```python
    import polypus

    bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    result = polypus.run_quantum_circuit(
        bell, shots=1000, infrastructure="local", backend="polypus", n_qpus=4
    )
    ```

    [:octicons-arrow-right-24: Running circuits](guide/running.md)

- **Train**

    Variational training behind one `train()` call: DE, PSO, CMA-ES, Adam and more. A QAOA cost for MaxCut, evaluated natively:

    ```python
    cut = polypus.Qubo(
        4,
        linear=[(q, 2.0) for q in range(4)],
        quadratic=[(i, j, -2.0) for i, j in edges],
    )
    ```

    [:octicons-arrow-right-24: Variational training](guide/training.md)

- **QML**

    Train your own Qiskit circuits with `qml.train`, or build native models with `qml.Model`: encoders, ansatz layers and Pauli readouts, with exact gradients.

    [:octicons-arrow-right-24: Quantum machine learning](guide/qml.md)

</div>

[Getting started](getting-started.md){ .md-button .md-button--primary }
[Tutorials](https://github.com/Bahia-Software/polypus-tutorials){ .md-button }
[Reference](https://bahia-software.github.io/polypus/rust/polypus/){ .md-button }
