---
hide:
  - navigation
  - toc
---

<div class="pp-hero" markdown>

<img src="assets/favicon.png" class="pp-hero__mark" alt="">

# Run and train quantum circuits across many QPUs

<p class="pp-hero__lead">One Python API over a Rust core. Move from a laptop simulator to CESGA's CUNQA or QMIO by changing one argument, not the algorithm code.</p>

```bash
pip install polypus-quantum
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/Bahia-Software/polypus){ .md-button }

</div>

<div class="pp-figure" markdown>

![Circuits enter the Python API; the Rust core plans the work and runs it on local Aer, the native simulator, CUNQA or QMIO](assets/overview.svg)

</div>

<div class="grid cards pp-cards" markdown>

-   :material-chip:{ .lg .middle } **Run on many QPUs**

    ---

    Split shots across `n_qpus`, run on Aer, the native simulator, CUNQA or QMIO, and get the counts back per QPU and merged.

    [:octicons-arrow-right-24: Running circuits](guide/running.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Train variational circuits**

    ---

    DE, PSO, QNG, CMA-ES, Adam, SPSA and Rotosolve behind one `train()` call, with native QUBO and Ising costs.

    [:octicons-arrow-right-24: Variational training](guide/training.md)

-   :material-brain:{ .lg .middle } **Quantum machine learning**

    ---

    Train your own Qiskit circuits with `qml.train`, or build native models with `qml.Model`: encoders, ansatz layers and Pauli readouts, with exact gradients.

    [:octicons-arrow-right-24: Quantum machine learning](guide/qml.md)

-   :material-vector-polyline:{ .lg .middle } **Circuits and interchange**

    ---

    Build, compose and draw circuits natively; import and export OpenQASM 2 and 3, and export QIR.

    [:octicons-arrow-right-24: Circuits and interchange](guide/circuits.md)

</div>

## Quickstart

=== "Run"

    ```python
    import polypus

    bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    result = polypus.run_quantum_circuit(
        bell, shots=1000, infrastructure="local", backend="polypus", n_qpus=4
    )

    result.counts  # one counts dict per QPU
    result.merged_counts  # e.g. {'00': 487, '11': 513}
    ```

=== "Train"

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

=== "Draw"

    ```python
    import polypus

    bell = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    print(bell)
    ```

    ```text
         ┌───┐       ┌─┐
    q0: ─┤ H ├───■───┤M├─────
         └───┘   │   └╥┘
               ┌─┴─┐  ║  ┌─┐
    q1: ───────┤ X ├──╫──┤M├─
               └───┘  ║  └╥┘
                      ║   ║
    c2: ══════════════╩═══╩══
                      0   1
    ```

=== "Shor"

    ```python
    from polypus.shor import factor

    result = factor(15, a=7, infrastructure="local", backend="polypus", seed=1)
    result.factors, result.method, result.order  # ((3, 5), 'quantum', 4)
    result.peak_share, result.uniform_peak_share  # (0.734375, 0.03515625)
    ```

[Getting started](getting-started.md){ .md-button .md-button--primary }
[Tutorials](https://github.com/Bahia-Software/polypus-tutorials){ .md-button }
[Rust API](https://bahia-software.github.io/polypus/rust/polypus/){ .md-button }
