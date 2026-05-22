<p align="center">
  <img src="docs/Logo.png" alt="Logo" width="350">
</p>


# Polypus
Polypus is an open-source distributed quantum computing library designed to optimize the execution of quantum algorithms by distributing computation across available hardware resources. The core of the library is written in Rust, while Python bindings are provided to make it more accessible to a broader range of users.

## How to use Polypus?
*Polypus is a library currently under development* 

**Instalation**

First, build the polypus-python package:

```python
python -m build packages/polypus_python/
```

Next, install the package:

```python
pip install packages/polypus_python/
```

Finally, build Polypus

```bash
maturin develop --release
```

**CUNQA**

To instal cunqa please refer to cunqa github.

**Examples**

Check the *examples* section for complete running examples.

**Run a Quantum Circuit**

Pass the circuit, the number of shots, the infrastructure and the number of QPUs:
```python
result = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure=INFRASTRUCTURE, n_qpus=1)
```

**Distribute shots across multiple QPUs**

Set `n_qpus > 1` to split the shots across available QPUs and reduce execution time:
```python
result = polypus.run_quantum_circuit(qc, shots=NUM_SHOTS, infrastructure=INFRASTRUCTURE, n_qpus=10)
```

## Training variational circuits

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

### Differential Evolution

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

### Particle Swarm Optimization

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

### Quantum Natural Gradient

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

## Credits
- Diego Beltrán Fernández Prada
- Víctor Sóñora Pombo 
- Sergio Figueiras Gómez
- Miguel Boubeta Martínez
- Galicia Supercomputing Center (CESGA)

## Dependencies
Polypus relies on the following Python Packages:
- matplotlib==3.10.3
- networkx==3.2.1
- numpy==2.2.6
- qiskit==2.0.1
- qiskit_aer==0.17.0
- cunqa==2.3.0

## License
Polypus is **Licensed under the EUPL**. Check the *License.txt* file for more details.