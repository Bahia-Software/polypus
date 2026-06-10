<p align="center">
  <img src="docs/Logo.png" alt="Logo" width="350">
</p>


# Polypus
Polypus is an open-source distributed quantum computing library designed to optimize the execution of quantum algorithms by distributing computation across available hardware resources. The core of the library is written in Rust, while Python bindings are provided to make it more accessible to a broader range of users.

## How to use Polypus?
*Polypus is a library currently under development*

### Installation

The recommended way to install Polypus is using the provided script:

```bash
bash install.sh
```

This will interactively guide you through installing dependencies, building the wheel, and optionally running the test suite. For non-interactive environments (CI/CD):

```bash
bash install.sh --yes        # all defaults, run all tests
bash install.sh --no-tests   # skip tests
```

<details>
<summary>Manual installation</summary>

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Build and install the `polypus_python` package:

```bash
python -m build packages/polypus_python/
pip install packages/polypus_python/
```

Build the Rust extension:

```bash
maturin develop --release --features extension-module
```

</details>

**CUNQA**

To install CUNQA please refer to the [CUNQA GitHub repository](https://github.com/CESGA-Quantum-Spain/cunqa).

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

## Rust-native circuits (`polypus-circuit`)

The repository is a Cargo workspace with two crates:

- `crates/polypus-circuit` — pure-Rust quantum circuit representation with OpenQASM 2.0 export. No PyO3/Python dependency, so circuits can be built and serialized without the GIL.
- `crates/polypus` — the Polypus library and Python extension module (re-exports the circuit API as `polypus::circuit`).

Qiskit circuits remain fully supported; the Rust API is an additional, high-performance path.

```rust
use polypus_circuit::{ParameterizedCircuit, Param};

let qc = ParameterizedCircuit::new(4)
    .h(0).h(1).h(2).h(3)
    .rzz(0, 1, Param(0)).rzz(1, 2, Param(0)).rzz(2, 3, Param(0)).rzz(3, 0, Param(0))
    .rx(0, Param(1)).rx(1, Param(1)).rx(2, Param(1)).rx(3, Param(1))
    .measure_all();

let qasm: String = qc.to_qasm2_with_params(&[0.4, 0.8])?; // OpenQASM 2.0
```

The generated OpenQASM 2.0 uses standard `qelib1.inc` gate names and is accepted by Qiskit (`QuantumCircuit.from_qasm_str`) and Aer.

## Credits
- Diego Beltrán Fernández Prada
- Víctor Sóñora Pombo 
- Sergio Figueiras Gómez
- Miguel Boubeta Martínez
- Galicia Supercomputing Center (CESGA)

## Dependencies

Polypus relies on the following Python packages:

| Package | Version |
|---|---|
| `qiskit` | ≥ 2.0 |
| `qiskit-aer` | ≥ 0.17 |
| `numpy` | ≥ 2.0 |
| `scipy` | ≥ 1.13 |
| `matplotlib` | ≥ 3.9 |
| `networkx` | ≥ 3.2 |
| `cunqa` | ≥ 2.3 (optional) |

See `requirements-dev.txt` and `requirements-examples.txt` for the full pinned dependency lists.

## License
Polypus is **Licensed under the EUPL**. Check the *License.txt* file for more details.