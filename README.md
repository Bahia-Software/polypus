<p align="center">
  <img src="docs/Logo.png" alt="Logo" style="max-height: 200px; width: auto;">
</p>


# Polypus
Polypus is a distributed quantum computing library designed to optimize the execution of quantum algorithms by distributing computation across available hardware resources. The core of the library is written in Rust, while Python bindings are provided to make it more accessible to a broader range of users.

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

**Examples**
Check the *examples* section for complete running examples.

**Distribute By Shots**

In this version, all available functions can be accessed directly by importing them from the Polypus module. For example, a quantum circuit can be executed by distributing its shots across the available QPUs. Polypus handles the aggregation of partial results and returns the combined output. Although Polypus has its core written in Rust, the Python bindings allow the library to be imported and used just like any standard Python package.

```python
import polypus
result = polypus.run_quantum_circuit(qft_circuit, shots=N_SHOTS, n_qpus = N_QPUS)
```

**Differential Evolution**

We can also directly import an optimization algorithm. For example, if we define a QAOA circuit and specify how to compute its expectation value, we can optimize this circuit using differential evolution. Polypus takes care of optimizing the execution by distributing the shots across the available hardware resources.

```python
import polypus
result = polypus.run_quantum_circuit(qc, shots=N_SHOTS, n_qpus = N_QPUS, expectation_function=expectation_fun,  generations=MAX_GENERATIONS, population_size=POPULATION_SIZE, dimensions=DIMENSIONS)
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

## License
Polypus is **Licensed under the EUPL**. Check the *License.txt* file for more details.