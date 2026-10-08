# Variational training

`polypus.train` optimizes the parameters of a variational circuit. The optimizer is the second argument; the population of each generation is distributed across the available QPUs.

```python
result = polypus.train(
    qc,                          # parameterized polypus.Circuit or Qiskit QuantumCircuit
    polypus.DE(generations=100, population_size=50),
    shots=1024,
    n_qpus=1,
    dimensions=2,                # number of variational parameters
    expectation_function=cost,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="qaoa",
    seed=42,
)
```

Optimizers **maximise** the fitness. For a minimisation problem, return the negated cost.

## Parameters

| Parameter | Description |
|---|---|
| `qc` | Parameterized circuit: a `polypus.Circuit` with `polypus.Param(i)` angles, or a Qiskit `QuantumCircuit` |
| `method` | `polypus.DE`, `polypus.PSO` or `polypus.QNG` |
| `shots` | Shots per circuit execution |
| `n_qpus` | Number of QPUs the population is spread over |
| `dimensions` | Number of variational parameters |
| `expectation_function` | The fitness: a `bitstring -> float` callable, a `polypus.CachedCost`, or a native `polypus.Qubo` / `polypus.Ising` |
| `infrastructure` | `"local"`, `"cunqa"` or `"qmio"` |
| `nodes`, `cores_per_qpu` | SLURM allocation, used by `"cunqa"` only (must be `>= 1` there) |
| `id` | Run name prefix: `[A-Za-z0-9._-]`, at most 64 characters. A UUID is appended for uniqueness. |
| `backend` | `"aer"` (default) or `"polypus"` for `infrastructure="local"` |
| `seed` | Optional RNG seed. Without one, a seed is drawn from OS entropy and reported in the result. |

Ctrl+C stops the optimization promptly and raises `KeyboardInterrupt`. An exception raised by `expectation_function` propagates as itself.

## The result

`train` and `qml.train` return a `TrainResult`:

| Field | Content |
|---|---|
| `best_params` | `list[float]`, the optimized parameters |
| `best_fitness` | Fitness at those parameters |
| `iterations_run` | Iterations actually run, early stopping included |
| `converged` | Whether the convergence criterion was met |
| `seed` | The effective seed; pass it back as `seed=...` to reproduce the run |
| `id` | The effective run id |

The seed can also be pinned on the optimizer (`polypus.DE(..., seed=42)`). The `seed` argument of `train` takes precedence. On the native backend the same seed also drives shot sampling, so the whole run reproduces exactly.

## Cost functions

**Python callable.** `expectation_function(bitstring) -> float` maps one measured bitstring to its cost; Polypus weights it by the counts. Each distinct bitstring is evaluated once per batch.

**Cached callable.** `polypus.CachedCost(fn)` memoises `fn` across the whole optimization, not only within a batch. This pays off near convergence, when the population keeps producing the same bitstrings. `fn` must be pure.

**Native observables.** `polypus.Qubo` and `polypus.Ising` describe the cost as data and are evaluated in Rust, without calling Python per bitstring.

```python
# f(x) = sum_i linear_i x_i + sum_ij w_ij x_i x_j + constant, x_i in {0, 1}
cost = polypus.Qubo(4, linear=[(0, 1.0), (1, -2.0)], quadratic=[(0, 1, 3.0)], constant=0.5)
cost = polypus.Qubo.from_matrix(Q)  # f(x) = x^T Q x

# f(z) = sum_i h_i z_i + sum_ij J_ij z_i z_j + constant, z_i = 1 - 2 x_i
cost = polypus.Ising(4, fields=[(0, 0.5)], couplings=[(0, 1, -1.0), (1, 2, -1.0)])
```

Both take `scale=-1.0` to turn a minimisation into a maximisation. Bit order matches Qiskit's read-out: the right-most bit is variable 0.

## Optimizers

### Differential Evolution

```python
polypus.DE(generations=100, population_size=50, tolerance=0.01, patience=20, seed=None)
```

The run stops early when the best fitness improves by less than `tolerance` over the last `patience` generations.

### Particle Swarm Optimization

```python
polypus.PSO(
    generations=100,
    population_size=50,
    bounds=(-math.pi, math.pi),
    inertia_weight=0.5,
    cognitive_weight=1.0,
    social_weight=1.0,
    tolerance=0.01,
    seed=None,
)
```

### Quantum Natural Gradient

```python
polypus.QNG(
    variance_function,
    max_iters=100,
    bounds=(-math.pi, math.pi),
    learning_rate=0.1,
    finite_difference_step=0.1,
    tikhonov_reg=0.05,
    seed=None,
)
```

`variance_function(theta, a) -> float` estimates the diagonal element of the quantum Fisher information matrix for parameter `a` at the point `theta`. [`examples/max_cut_qaoa.py`](../../examples/max_cut_qaoa.py) implements one for QAOA.
