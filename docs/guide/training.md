# Variational training

`polypus.train` optimizes the parameters of a variational circuit. The optimizer is the second argument; the population of each generation is distributed across the available QPUs.

```python
result = polypus.train(
    qc,  # parameterized polypus.Circuit or Qiskit QuantumCircuit
    polypus.DE(generations=100, population_size=50),
    shots=1024,
    n_qpus=1,
    dimensions=2,  # number of variational parameters
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
| `method` | `polypus.DE`, `polypus.PSO`, `polypus.QNG`, `polypus.CMAES` or `polypus.Adam` |
| `shots` | Shots per circuit execution |
| `n_qpus` | Number of QPUs the population is spread over |
| `dimensions` | Number of variational parameters |
| `expectation_function` | The fitness: a `bitstring -> float` callable, a `polypus.CachedCost`, or a native `polypus.Qubo` / `polypus.Ising` |
| `infrastructure` | `"local"`, `"cunqa"` or `"qmio"` |
| `nodes`, `cores_per_qpu` | SLURM allocation, used by `"cunqa"` only (must be `>= 1` there) |
| `walltime` | Optional SLURM walltime of the QPUs as `HH:MM:SS` (default `"10:00:00"`), used by `"cunqa"` only |
| `id` | Run name prefix: `[A-Za-z0-9._-]`, at most 64 characters. A UUID is appended for uniqueness. |
| `backend` | `"aer"` (default) or `"polypus"` for `infrastructure="local"` |
| `seed` | Optional RNG seed. Without one, a seed is drawn from OS entropy and reported in the result. |
| `initial_point` | Optional point to start the search from, such as an earlier run's `best_params`: `dimensions` finite numbers. See [Starting point](#starting-point). |

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
cost = polypus.Qubo(
    4, linear=[(0, 1.0), (1, -2.0)], quadratic=[(0, 1, 3.0)], constant=0.5
)
cost = polypus.Qubo.from_matrix(Q)  # f(x) = x^T Q x

# f(z) = sum_i h_i z_i + sum_ij J_ij z_i z_j + constant, z_i = 1 - 2 x_i
cost = polypus.Ising(4, fields=[(0, 0.5)], couplings=[(0, 1, -1.0), (1, 2, -1.0)])
```

Both take `scale=-1.0` to turn a minimisation into a maximisation. Bit order matches Qiskit's read-out: the right-most bit is variable 0.

## Optimizers

### Differential Evolution

```python
polypus.DE(
    generations=100,
    population_size=50,
    tolerance=0.01,
    patience=20,
    seed=None,
    strategy="rand1bin",
)
```

The run stops early when the best fitness improves by less than `tolerance` over the last `patience` generations.

`strategy` picks how each generation builds its trial vectors:

| `strategy` | Algorithm |
|---|---|
| `"rand1bin"` (default) | Classic DE/rand/1/bin with fixed `F = 0.8` and `CR = 0.7` |
| `"shade"` | SHADE (Tanabe and Fukunaga, 2013): current-to-pbest/1 mutation with an archive of replaced parents. Each individual draws its own `F` and `CR` from a memory of the values that recently improved the population, so there is nothing to tune |
| `"lshade"` | L-SHADE (Tanabe and Fukunaga, 2014): SHADE with a population that shrinks linearly towards 4 |

Any other name raises `ValueError`. `population_size` must be at least 4 with every strategy, and every strategy evaluates one batch per generation, after the initial population's.

All three have the same budget of `population_size × (generations + 1)` evaluations: the initial population plus `generations` full generations. `"rand1bin"` and `"shade"` keep their population, so they run `generations` generations. For `"lshade"`, `generations` sets that budget rather than the number of generations: its population shrinks as the budget is spent, reaching 4 at the end, so it runs more, smaller generations until the next one would not fit. With the defaults that is 275 generations instead of 100, the last ones of 4 candidates, using 5048 of the 5050 evaluations. The early stop still counts generations, so late in an L-SHADE run `patience` generations hold fewer evaluations.

More, smaller batches mean more sequential rounds: with several QPUs, an L-SHADE batch smaller than `n_qpus` leaves some of them idle, and the run takes more rounds than one with a fixed population. With many QPUs, prefer `"shade"`, whose population stays fixed; [Choosing an optimizer](#choosing-an-optimizer) measures the trade-off. A CUNQA walltime warning counts L-SHADE's real number of generations.

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

### CMA-ES

```python
polypus.CMAES(
    generations=100,
    population_size=None,
    sigma0=math.pi / 2,
    tolerance=0.01,
    patience=20,
    restarts=0,
    seed=None,
)
```

The covariance matrix adaptation evolution strategy (Hansen's tutorial, arXiv:1604.00772) samples each generation from a normal distribution and moves its mean, step size and covariance towards the best candidates. The mean and the samples are not limited to `[0, 2π)`, so the search can cross the seam between `2π` and `0`; each candidate is wrapped into `[0, 2π)` before it is evaluated, and `best_params` is the best wrapped candidate evaluated.

| Parameter | Description |
|---|---|
| `generations` | Total generations, over the first run and every restart. Each generation is one batch; there is no separate initial batch. At least 1 |
| `population_size` | Candidates per generation in the first run, at least 2. `None` uses Hansen's default `4 + floor(3 ln d)` for `d = dimensions` (9 for 6 parameters) |
| `sigma0` | Initial step size in radians, in `(0, 2π]`. The default `π/2` suits angles spread over the whole circle; larger values make runs stall more often. During the run the spread of each angle stays within `max(sigma0, π/2)` |
| `tolerance`, `patience` | The run ends when its best fitness improves by less than `tolerance` over its last `patience` generations, as with DE |
| `restarts` | IPOP restarts allowed. When a run ends, the next one starts from a new random mean with twice the population |

A run also ends when its distribution degenerates: its spread collapses below `1e-12·sigma0` or grows beyond `1e4·sigma0`, its covariance becomes ill-conditioned (condition number above `1e14`), or a value is no longer finite. Either way the next restart starts if one is left; otherwise the optimisation stops with `converged=True`. If the generations run out first, `converged` is `False`. `iterations_run` counts the generations of all runs. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**Batch size and QPUs.** Each generation is one batch of `population_size` circuits, spread over the `n_qpus` QPUs. The default population is small, so a CMA-ES run takes many sequential batches and leaves QPUs idle when they outnumber it. Raising `population_size` to fill them lowered its mean ratio in the benchmark below, and `restarts` did not help either; see [Choosing an optimizer](#choosing-an-optimizer). All runs share `generations`, so restarts trade generations of one run for runs with larger batches.

With `infrastructure="cunqa"`, the walltime warning projects the first generation over `generations`. With `restarts`, it assumes the worst case, a restart after every generation until none are left. Because the population doubles at each restart, the projection can be well above the usual run time.

**Reproducibility.** A seed reproduces a CMA-ES run exactly on the same machine and build. Across operating systems or CPU architectures it may not: CMA-ES's own steps use exponentials, logarithms and cosines from the platform's math library, whose last bit can differ, and a one-bit difference early on can lead a run to a different optimum. DE's steps use only basic arithmetic, so a DE run over a backend whose results do not depend on the platform is reproducible everywhere; the native simulator's `sin`/`cos` add the same caveat to every optimizer.

### Adam

```python
polypus.Adam(
    max_iters=100,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    bounds=(-math.pi, math.pi),
    tolerance=0.01,
    patience=20,
    finite_difference_step=0.2,
    rule="adam",
    seed=None,
)
```

A gradient method: it climbs the fitness along its gradient, estimated by central differences. Each iteration sends one batch of `2 × dimensions + 1` circuits, spread over the `n_qpus` QPUs: the current parameters `θ`, whose fitness is the iteration's, and `θ ± finite_difference_step` along each parameter, which give the gradient (61 circuits for 30 parameters). It then steps by its `rule`:

| `rule` | Step |
|---|---|
| `"adam"` (default) | Adam (Kingma and Ba, arXiv:1412.6980): bias-corrected running averages of the gradient and of its square scale each parameter's step to about `learning_rate` radians, whatever the scale of the cost, so one configuration suits problems whose costs differ in scale |
| `"momentum"` | Polyak's heavy ball (1964): a velocity, `beta1 × velocity + learning_rate × gradient`, is added to the parameters. The step grows with the gradient, so the `learning_rate` that works depends on the scale of the cost: a rate tuned on one problem can make another, with a larger cost, diverge. `beta2` and `epsilon` are not used |

Any other `rule` raises `ValueError`.

| Parameter | Description |
|---|---|
| `max_iters` | Iterations, each one batch of `2 × dimensions + 1` circuits. At least 1 |
| `learning_rate` | Step size, finite and `> 0`. The default `0.1` is sized for rotation angles in radians rather than network weights, so it is larger than deep learning's usual `0.001`. With `"momentum"`, tune it to the scale of the cost |
| `beta1`, `beta2` | Decay of the running averages of the gradient and of its square, in `[0, 1)`. With `"momentum"`, `beta1` is the momentum and `beta2` is unused |
| `epsilon` | Added to the denominator for numerical stability, finite and `> 0`. Unused by `"momentum"` |
| `bounds` | Range the initial parameters are drawn from, finite with `lower < upper`. The run itself is not confined to it, and the parameters are not wrapped into `[0, 2π)` |
| `tolerance`, `patience` | The run ends when its best fitness improves by less than `tolerance` over its last `patience` iterations, as with DE |
| `finite_difference_step` | Step of the central difference, in radians, finite and `> 0` |
| `rule` | `"adam"` or `"momentum"`, above |

**The finite-difference step matters.** Shot noise enters the gradient through the difference of two fitness values divided by `2 × finite_difference_step`, so a small step amplifies it. Jones, Mineh and Montanaro (arXiv:2411.13742) found that a step of 0.4 works and 0.01 ruins the gradient. Prefer the default `0.2` to a small step, which needs many more shots for the same accuracy.

`best_params` is the measured point with the best fitness and `fitness_history` the best fitness measured up to each iteration. The step after the last iteration is not measured. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**A local method.** Adam and momentum climb to an optimum near where they start, so on a landscape with many local optima one run can end in a poor one. Start them from a global search's result, such as an L-SHADE run's `best_params`, with `initial_point` (see [Starting point](#starting-point)), or run them a few times with different seeds and keep the best. In [Choosing an optimizer](#choosing-an-optimizer), starting from L-SHADE was the more reliable of the two.

**Reproducibility.** With either rule Adam's steps use only basic arithmetic and square roots, so, as with DE and unlike CMA-ES, a seeded run over a backend whose results do not depend on the platform reproduces exactly everywhere.

## Starting point

By default every optimizer starts from random parameters. `initial_point` makes it start from a point you choose, typically the `best_params` of an earlier run. It takes `dimensions` finite numbers as a list, tuple or NumPy array; another length, a `NaN` or an infinity raises `ValueError` before anything runs. `qml.train` takes it too, as a keyword argument.

| Optimizer | What it does with `initial_point` |
|---|---|
| `DE`, every `strategy` | Makes it, wrapped into `[0, 2π)`, the first member of the initial population, so it is the first candidate evaluated. The other members are drawn as without it |
| `PSO` | Makes it the position of the first particle, the first candidate evaluated; the other particles are drawn as without it. Every value must lie within `bounds`, since positions are clamped to them, otherwise `ValueError` before anything runs |
| `QNG` | Starts from it: the first batch estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |
| `CMAES` | Makes it, wrapped into `[0, 2π)`, the mean of the first run: the first generation is sampled around it with step size `sigma0` |
| `Adam` | Starts from it and evaluates it first: the first batch is the point, then the finite-difference stencil that estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |

Without `initial_point`, every optimizer starts as before and a seeded run reproduces exactly. With one, a run reproduces with the same seed and the same point; the result does not echo the point, so keep it with the seed.

CMA-ES keeps its `sigma0`. With the default `π/2`, two standard deviations already span the whole circle, so the first generation explores around the point rather than refining it; to refine around a point, pass a smaller `sigma0` too, such as `0.3`. IPOP restarts start from a new random mean, not from the point. CMA-ES samples around the point without evaluating the point itself, so its `best_fitness` can come out below the one the point had in the earlier run.

**Explore, then refine.** A global search followed by a local refinement from its best point was the most reliable pattern in [Choosing an optimizer](#choosing-an-optimizer): L-SHADE with half of the evaluation budget, then Adam from its `best_params` with the other half. A complete example, QAOA with p = 1 for MaxCut on a 4-node ring:

```python
import polypus

edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
cost = polypus.Qubo(  # the cut: x_a + x_b - 2 x_a x_b per edge
    4, linear=[(i, 2.0) for i in range(4)], quadratic=[(a, b, -2.0) for a, b in edges]
)
qc = polypus.Circuit(4)
for q in range(4):
    qc.h(q)
for a, b in edges:
    qc.rzz(a, b, polypus.Param(0))  # γ
for q in range(4):
    qc.rx(q, 2 * polypus.Param(1))  # β
qc = qc.measure_all()

dimensions = 2
budget = 1500  # circuit evaluations for both phases
common = dict(
    shots=256,
    n_qpus=1,
    dimensions=dimensions,
    expectation_function=cost,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="maxcut",
    backend="polypus",
)
# Explore: L-SHADE spends population_size × (generations + 1), half the budget.
explore = polypus.train(
    qc,
    polypus.DE(
        strategy="lshade",
        population_size=30,
        generations=budget // 2 // 30 - 1,
        tolerance=0.0,
    ),
    **common,
    seed=1,
)
# Refine: Adam from L-SHADE's best point, 2 × dimensions + 1 circuits an iteration.
refine = polypus.train(
    qc,
    polypus.Adam(max_iters=budget // 2 // (2 * dimensions + 1), tolerance=0.0),
    **common,
    seed=1,
    initial_point=explore.best_params,
)
print(explore.best_fitness, refine.best_fitness)
```

`tolerance=0.0` turns the early stop off, so each phase spends its share of the budget, as in the benchmark. CMA-ES with a small `sigma0` can do the refining instead, as above.

**Restarts.** Without a global search, run Adam from a few random points, each with a share of the budget, and keep the run with the best `best_fitness`:

```python
runs = [
    polypus.train(
        qc,
        polypus.Adam(max_iters=budget // 3 // (2 * dimensions + 1), tolerance=0.0),
        **common,
        seed=seed,
    )
    for seed in (1, 2, 3)
]
best = max(runs, key=lambda run: run.best_fitness)
```

In the benchmark, three runs raised the worst instance at p = 5, where single runs sometimes ended in a poor optimum, but their mean there was within noise of one run's. With 256 shots at p = 3 and on the HEA they did worse, a third of the budget being too short for each run, and starting from L-SHADE had the higher mean in every scenario.

## Choosing an optimizer

[`benchmarks/bench_optimizers.py`](../../benchmarks/bench_optimizers.py) trains two circuits for MaxCut on random 3-regular graphs of 10 nodes, with each optimizer at the same evaluation budget: QAOA with depth p = 3 or 5 (6 or 10 parameters), and a hardware-efficient ansatz (HEA) of three layers of RY on every qubit with a linear chain of CZ between them (Qiskit's `TwoLocal` with `ry`, `cz`, `entanglement="linear"` and `reps=2`), 30 parameters. The 30 instances are paired: each is one graph, run by every optimizer with the same seed. The score is the exact approximation ratio of the returned `best_params`, the expected cut from the statevector divided by the maximum cut, so shot noise in `best_fitness` does not flatter it. DE uses a population of 30, and early stopping is off except for CMA-ES with restarts, whose restarts need it. Adam uses its defaults on both circuits. Momentum uses a learning rate tuned for each circuit on separate instances, 0.03 for QAOA and 0.1 for the HEA, with a `finite_difference_step` of 0.4. "L-SHADE, then Adam" gives half the budget to L-SHADE and the rest to Adam, started from L-SHADE's `best_params`; "Adam ×3" gives a third to each of three Adam runs with different seeds and keeps the one with the best `best_fitness`. The circuits run on Polypus's native simulator, through a meter that counts every evaluation and stops each optimizer at the budget.

Paired difference in approximation ratio from the classic DE (`rand1bin`), with its standard error, and the share of instances where the optimizer did better; the best mean of each scenario is in bold:

| Optimizer | p = 3, 256 shots, 1500 evals | p = 3, 2^20 shots, 1500 evals | p = 5, 256 shots, 3000 evals | p = 3, 256 shots, 500 evals | HEA, 256 shots, 5000 evals |
|---|---|---|---|---|---|
| `DE()` (ratio) | 0.862 | 0.860 | 0.871 | 0.843 | 0.783 |
| `DE(strategy="shade")` | +0.001 ± 0.003 (50 %) | +0.006 ± 0.003 (67 %) | +0.015 ± 0.003 (80 %) | +0.006 ± 0.006 (57 %) | +0.106 ± 0.007 (100 %) |
| `DE(strategy="lshade")` | +0.017 ± 0.003 (90 %) | +0.030 ± 0.004 (97 %) | +0.037 ± 0.004 (93 %) | +0.008 ± 0.005 (65 %) | +0.153 ± 0.007 (100 %) |
| `CMAES()` | +0.020 ± 0.004 (77 %) | +0.037 ± 0.004 (97 %) | +0.038 ± 0.006 (90 %) | +0.005 ± 0.006 (53 %) | +0.170 ± 0.017 (90 %) |
| `CMAES(population_size=30)` | +0.011 ± 0.005 (60 %) | +0.010 ± 0.004 (67 %) | +0.010 ± 0.006 (53 %) | −0.000 ± 0.004 (43 %) | +0.133 ± 0.019 (80 %) |
| `CMAES(restarts=3)` | +0.004 ± 0.004 (60 %) | +0.009 ± 0.006 (60 %) | −0.002 ± 0.004 (43 %) | +0.001 ± 0.005 (53 %) | +0.015 ± 0.014 (53 %) |
| `Adam()` | +0.030 ± 0.004 (90 %) | +0.032 ± 0.004 (93 %) | +0.050 ± 0.011 (93 %) | +0.035 ± 0.004 (97 %) | +0.197 ± 0.009 (100 %) |
| `Adam(rule="momentum")`, tuned rate | +0.033 ± 0.003 (97 %) | +0.038 ± 0.004 (87 %) | +0.066 ± 0.004 (100 %) | +0.037 ± 0.004 (100 %) | +0.201 ± 0.009 (100 %) |
| L-SHADE, then `Adam()` | **+0.043 ± 0.004 (97 %)** | **+0.039 ± 0.004 (97 %)** | **+0.071 ± 0.004 (100 %)** | **+0.044 ± 0.005 (97 %)** | **+0.214 ± 0.007 (100 %)** |
| `Adam()` ×3 | +0.024 ± 0.004 (83 %) | +0.036 ± 0.004 (97 %) | +0.066 ± 0.004 (100 %) | +0.023 ± 0.005 (77 %) | +0.171 ± 0.009 (100 %) |

The lowest ratio of the 30 instances, where a local method's occasional poor optimum shows:

| Optimizer | p = 3, 256 shots, 1500 evals | p = 3, 2^20 shots, 1500 evals | p = 5, 256 shots, 3000 evals | p = 3, 256 shots, 500 evals | HEA, 256 shots, 5000 evals |
|---|---|---|---|---|---|
| `DE()` | 0.793 | 0.758 | 0.822 | 0.760 | 0.675 |
| `DE(strategy="shade")` | 0.801 | 0.806 | 0.815 | 0.797 | 0.843 |
| `DE(strategy="lshade")` | 0.802 | 0.831 | 0.830 | 0.813 | 0.834 |
| `CMAES()` | 0.820 | 0.836 | 0.807 | 0.739 | 0.740 |
| `CMAES(population_size=30)` | 0.788 | 0.792 | 0.822 | 0.788 | 0.737 |
| `CMAES(restarts=3)` | 0.754 | 0.790 | 0.818 | 0.746 | 0.694 |
| `Adam()` | 0.817 | 0.784 | 0.661 | 0.792 | 0.844 |
| `Adam(rule="momentum")`, tuned rate | 0.823 | 0.824 | 0.861 | 0.813 | 0.845 |
| L-SHADE, then `Adam()` | 0.852 | 0.850 | 0.918 | 0.790 | 0.991 |
| `Adam()` ×3 | 0.786 | 0.822 | 0.907 | 0.784 | 0.829 |

Sequential batches each optimizer sent, the rounds that cost time when the QPUs outnumber a batch, and the winner's curse: how far the reported `best_fitness` overstated the exact value of `best_params`, as a share of the maximum cut, with 256 shots. An Adam batch holds `2 × dimensions + 1` circuits: 13, 21 and 61 here.

| Optimizer | Batches, 1500 evals | Batches, 3000 evals | Batches, HEA | Bias, p = 3 | Bias, p = 5 | Bias, HEA |
|---|---|---|---|---|---|---|
| `DE()` | 50 | 100 | 166 | +0.003 | +0.004 | +0.006 |
| `DE(strategy="shade")` | 50 | 100 | 166 | +0.007 | +0.008 | +0.011 |
| `DE(strategy="lshade")` | 115 | 232 | 385 | +0.009 | +0.011 | +0.014 |
| `CMAES()` | 166 | 300 | 357 | +0.019 | +0.015 | +0.004 |
| `CMAES(population_size=30)` | 50 | 100 | 166 | +0.008 | +0.008 | +0.004 |
| `CMAES(restarts=3)` | 90 | 115 | 116 | +0.008 | +0.002 | +0.007 |
| `Adam()` | 115 | 142 | 81 | +0.015 | +0.012 | +0.002 |
| `Adam(rule="momentum")`, tuned rate | 115 | 142 | 81 | +0.013 | +0.011 | +0.002 |
| L-SHADE, then `Adam()` | 114 | 186 | 233 | +0.013 | +0.011 | +0.002 |
| `Adam()` ×3 | 115 | 142 | 81 | +0.013 | +0.008 | +0.004 |

What the data says:

- **Explore with L-SHADE, then refine with Adam.** Half the budget to `DE(strategy="lshade")` and the other half to `Adam()` from its `best_params` ([Starting point](#starting-point)) gave the best mean ratio in all five scenarios, 500 evaluations included, and beat the classic DE on 97 to 100 % of the instances. Its lowest ratio was the highest of all the optimizers in four scenarios (0.918 with p = 5, 0.991 on the HEA), and its Adam phase improved on L-SHADE's half-budget result on every instance. Its lead over Adam alone was 0.007 to 0.022, more than two standard errors with 256 shots at p = 3 and 1500 evaluations (+0.013 ± 0.004) and on the HEA (+0.017 ± 0.007), and about two or fewer elsewhere. On the HEA the two were even on most instances, and the lead came from the few where Adam alone ended low.
- **Adam alone, with its defaults, is a strong single run with one configuration for both circuits.** It beat L-SHADE on average in every scenario: clearly with 256 shots at p = 3 (+0.013 ± 0.004 at 1500 evaluations, +0.027 ± 0.004 at 500) and on the HEA (+0.044 ± 0.007), within noise with 2^20 shots (+0.003 ± 0.004) and at p = 5 (+0.012 ± 0.011). As a local method it occasionally ends in a poor optimum: its lowest ratio was 0.661 with p = 5 and 0.784 with 2^20 shots, against 0.830 and 0.831 for L-SHADE. Its batches grow with the parameters, so with 30 it sent the fewest of all (81, against 385 for L-SHADE).
- **Use momentum only with a learning rate tuned to the problem.** With rates tuned on separate instances it matched Adam within noise (0.002 to 0.016 higher), and its lowest ratio at p = 5 was 0.861, against Adam's 0.661. Its rate depends on the scale of the cost (see [Adam](#adam)), so these figures are its best case; Adam needed no tuning.
- **Don't restart Adam by default.** Three runs on a third of the budget each raised the worst instance at p = 5 (lowest ratio 0.907, against 0.661 for one run), but their mean was within noise of one run's there (+0.016 ± 0.011) and with 2^20 shots (+0.004 ± 0.003). With 256 shots at p = 3 they were behind one run (−0.006 ± 0.003 at 1500 evaluations, −0.011 ± 0.003 at 500), and on the HEA too (−0.026 ± 0.008), where a third of the budget was too short. Starting from L-SHADE had the higher mean in every scenario.
- **CMA-ES with its default population is competitive with L-SHADE.** Its paired difference from L-SHADE was within two standard errors in every scenario: about even with 256 shots (+0.003 ± 0.005 at p = 3, +0.001 ± 0.006 at p = 5), slightly ahead with 2^20 shots (+0.007 ± 0.004) and on the HEA (+0.017 ± 0.016). It was less consistent on the HEA, beating the classic DE on 90 % of the instances where L-SHADE beat it on all, with a lowest ratio of 0.740 against 0.834. On QAOA it sent the most sequential batches and, at 1500 and 3000 evaluations, had the largest winner's curse.
- **Don't enlarge CMA-ES's population to fill QPUs, and don't count on IPOP restarts.** With a population of 30, CMA-ES's mean ratio was below its default population's in every scenario, by 0.005 to 0.037, though still about 0.01 above the classic DE's at 1500 and 3000 evaluations. IPOP restarts stayed within 0.01 of the classic DE on QAOA and gained +0.015 ± 0.014 on the HEA, against +0.170 without them.
- **With many QPUs, when sequential rounds dominate the run time,** `strategy="shade"` keeps the classic DE's batch size and number of rounds and still gained 1.5 points with 10 parameters and 10.6 with 30, though almost nothing with 6 at 256 shots. L-SHADE needs about 2.3 times the rounds of the classic DE for its larger gain.
- **With 500 evaluations only the Adam-based optimizers were clearly better than the classic DE,** by 0.023 to 0.044; the DE strategies and CMA-ES gained 0.008 at most.
- **The reported `best_fitness` is optimistic under shot noise,** because it is the best of many noisy estimates: by up to 1.9 % of the maximum cut with 256 shots, for CMA-ES at p = 3. Adam's `best_params` is the best point it measured, so Adam has the same bias: 1.0 to 1.5 % on QAOA with 256 shots, a little more than L-SHADE's and more than the classic DE's, and 0.2 % on the HEA. Re-evaluate `best_params` with more shots when the value itself matters.

These results cover MaxCut only, on 10 nodes, with QAOA at depth 3 to 5 and one 30-parameter hardware-efficient ansatz, on a noiseless simulator with shot sampling. Other problems, more parameters or hardware noise may rank the optimizers differently, and momentum's learning rates were tuned for these two circuits. CMA-ES's rows may differ on another platform or build; see its reproducibility note under [CMA-ES](#cma-es). Run `python benchmarks/bench_optimizers.py --full`, or a scenario of your own with `--scenario [hea:]LAYERS:SHOTS:BUDGET`, to check. The tables come from that command at commit `83a0bbd` (Polypus 0.7.2) on Linux x86_64 with an Intel Xeon at 2.10 GHz (4 CPUs). The 500-evaluation scenario gives a population of 30 only 480 evaluations, 16 batches of 30. The printed table gives the differences from the classic DE; the lowest ratios and the differences from L-SHADE or Adam quoted above come from the per-instance results that `--json FILE` writes.
