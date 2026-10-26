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

**Batch size and QPUs.** Each generation is one batch of `population_size` circuits, spread over the `n_qpus` QPUs. The default population is small, so a CMA-ES run takes many sequential batches and leaves QPUs idle when they outnumber it. Raising `population_size` to fill them did not pay off in the benchmark below, and neither did `restarts`; see [Choosing an optimizer](#choosing-an-optimizer). All runs share `generations`, so restarts trade generations of one run for runs with larger batches.

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

**A local method.** Adam and momentum climb to an optimum near where they start, so on a landscape with many local optima one run can end in a poor one. Start them from a global search's result, such as an L-SHADE run's `best_params`, with `initial_point` (see [Starting point](#starting-point)), or run them a few times with different seeds and keep the best.

**Reproducibility.** With either rule Adam's steps use only basic arithmetic and square roots, so, as with DE and unlike CMA-ES, a seeded run over a backend whose results do not depend on the platform reproduces exactly everywhere.

## Starting point

By default every optimizer starts from random parameters. `initial_point` makes it start from a point you choose, typically the `best_params` of an earlier run. It takes `dimensions` finite numbers as a list, tuple or NumPy array; another length, a `NaN` or an infinity raises `ValueError` before anything runs. `qml.train` takes it too, as a keyword argument.

| Optimizer | What it does with `initial_point` |
|---|---|
| `DE`, every `strategy` | Makes it, wrapped into `[0, 2π)`, the first member of the initial population, so it is the first candidate evaluated. The other members are drawn as without it |
| `PSO` | Makes it the position of the first particle, the first candidate evaluated; the other particles are drawn as without it. Every value must lie within `bounds`, since positions are clamped to them, otherwise `ValueError` before anything runs |
| `QNG` | Starts from it: the first batch estimates the gradient around it, without evaluating the point itself. It may lie outside `bounds`, which only limit the random start |
| `CMAES` | Makes it, wrapped into `[0, 2π)`, the mean of the first run: the first generation is sampled around it with step size `sigma0` |
| `Adam` | Starts from it and evaluates it first: the first batch is the point, then the finite-difference stencil that estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |

Without `initial_point`, every optimizer starts as before and a seeded run reproduces exactly. With one, a run reproduces with the same seed and the same point; the result does not echo the point, so keep it with the seed.

CMA-ES keeps its `sigma0`. With the default `π/2`, two standard deviations already span the whole circle, so the first generation explores around the point rather than refining it; to refine around a point, pass a smaller `sigma0` too, such as `0.3`. IPOP restarts start from a new random mean, not from the point.

CMA-ES and QNG never evaluate the point itself. CMA-ES samples around it; QNG estimates the gradient around it, and the first value it keeps is that of the point after one step. Their `best_fitness` can therefore come out below the one the point had in the earlier run.

PSO's default `bounds` are `(-π, π)`, while DE and CMA-ES return points in `[0, 2π)`, so starting PSO from their `best_params` raises `ValueError` as soon as a value passes π. Pass bounds that contain the point, such as `polypus.PSO(bounds=(0, 2 * math.pi))`, or subtract 2π from every value above π first: the angles are the same, as DE and CMA-ES assume when they wrap.

A broad search followed by a local refinement from its best point:

```python
common = dict(
    shots=1024,
    n_qpus=1,
    dimensions=2,
    expectation_function=cost,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="qaoa",
)
explore = polypus.train(
    qc, polypus.DE(strategy="lshade", generations=100, population_size=50), **common
)
refine = polypus.train(
    qc,
    polypus.CMAES(generations=50, sigma0=0.3),
    **common,
    initial_point=explore.best_params,
)
```

## Choosing an optimizer

[`benchmarks/bench_optimizers.py`](../../benchmarks/bench_optimizers.py) trains QAOA for MaxCut on random 3-regular graphs of 10 nodes with each optimizer at the same evaluation budget. The 30 instances are paired: each is one graph, run by every optimizer with the same seed. The score is the exact approximation ratio of the returned `best_params`, the expected cut from the statevector divided by the maximum cut, so shot noise in `best_fitness` does not flatter it. DE uses a population of 30, and early stopping is off except for CMA-ES with restarts, whose restarts need it. The circuits run on Polypus's native simulator, through a meter that counts every evaluation and stops each optimizer at the budget.

Paired difference in approximation ratio from the classic DE (`rand1bin`), with its standard error, and the share of instances where the optimizer did better:

| Optimizer | p = 3, 256 shots, 1500 evals | p = 3, 2^20 shots, 1500 evals | p = 5, 256 shots, 3000 evals | p = 3, 256 shots, 500 evals |
|---|---|---|---|---|
| `DE()` (ratio) | 0.862 | 0.860 | 0.871 | 0.843 |
| `DE(strategy="shade")` | +0.001 ± 0.003 (50 %) | +0.006 ± 0.003 (67 %) | +0.015 ± 0.003 (80 %) | +0.006 ± 0.006 (57 %) |
| `DE(strategy="lshade")` | **+0.017 ± 0.003 (90 %)** | **+0.030 ± 0.004 (97 %)** | **+0.037 ± 0.004 (93 %)** | +0.008 ± 0.005 (65 %) |
| `CMAES()` | +0.012 ± 0.005 (67 %) | +0.020 ± 0.006 (70 %) | +0.016 ± 0.007 (63 %) | +0.002 ± 0.005 (60 %) |
| `CMAES(population_size=30)` | −0.001 ± 0.004 (53 %) | −0.000 ± 0.004 (43 %) | +0.005 ± 0.005 (57 %) | +0.000 ± 0.004 (47 %) |
| `CMAES(restarts=3)` | −0.005 ± 0.005 (30 %) | +0.003 ± 0.004 (57 %) | −0.003 ± 0.004 (57 %) | −0.005 ± 0.005 (53 %) |

Sequential batches each optimizer sent, the rounds that cost time when the QPUs outnumber a batch, and the winner's curse: how far the reported `best_fitness` overstated the exact value of `best_params`, as a share of the maximum cut, with 256 shots:

| Optimizer | Batches, 1500 evals | Batches, 3000 evals | Bias, p = 3 | Bias, p = 5 |
|---|---|---|---|---|
| `DE()` | 50 | 100 | +0.003 | +0.004 |
| `DE(strategy="shade")` | 50 | 100 | +0.007 | +0.008 |
| `DE(strategy="lshade")` | 115 | 232 | +0.009 | +0.011 |
| `CMAES()` | 166 | 300 | +0.013 | +0.009 |
| `CMAES(population_size=30)` | 50 | 100 | +0.006 | +0.007 |
| `CMAES(restarts=3)` | 87 | 114 | +0.006 | +0.004 |

What the data says:

- **With a budget of 1500 evaluations or more, use `polypus.DE(strategy="lshade")`.** It had the best ratio in every scenario, beat the classic DE on 90 to 97 % of the instances, and its lead was largest with 10 parameters (p = 5).
- **CMA-ES with its default population helps less than L-SHADE and is less consistent.** It gained 1.2 to 2.0 points over the classic DE but won only 63 to 70 % of the instances, and it took the most sequential batches, about three times the classic DE's.
- **Don't enlarge CMA-ES's population to fill QPUs.** With a population of 30 it was no better than the classic DE: at a fixed budget, larger batches leave fewer generations to adapt. IPOP restarts did not help either at these budgets.
- **With many QPUs, when sequential rounds dominate the run time,** `strategy="shade"` keeps the classic DE's batch size and number of rounds and still gained 1.5 points with 10 parameters, though almost nothing with 6 at 256 shots. L-SHADE needs about 2.3 times the rounds of the classic DE for its larger gain.
- **With 500 evaluations no optimizer was clearly better than the classic DE.**
- **The reported `best_fitness` is optimistic under shot noise,** and more so for the adaptive optimizers, which pick their best from more noisy estimates: by up to 1.3 % of the maximum cut with 256 shots. Re-evaluate `best_params` with more shots when the value itself matters.

These results cover QAOA MaxCut only, at 10 nodes and depth 3 to 5, on a noiseless simulator with shot sampling. Other problems, more parameters or hardware noise may rank the optimizers differently. Run `python benchmarks/bench_optimizers.py --full`, or a scenario of your own with `--scenario LAYERS:SHOTS:BUDGET`, to check. The figures above come from that command at commit `1ea087b` (Polypus 0.7.2) on Linux x86_64 with an Intel Xeon at 2.10 GHz (4 CPUs). The 500-evaluation scenario gives a population of 30 only 480 evaluations, 16 batches of 30.
