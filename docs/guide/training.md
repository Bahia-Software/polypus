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
| `method` | `polypus.DE`, `polypus.PSO`, `polypus.QNG`, `polypus.CMAES`, `polypus.Adam`, `polypus.SPSA` or `polypus.Rotosolve` |
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
| `final_shots` | Optional shots per circuit for one last batch that re-evaluates the best candidates after the optimizer stops: an integer `>= 1`. See [Final re-evaluation](#final-re-evaluation). |

Ctrl+C stops the optimization promptly and raises `KeyboardInterrupt`. An exception raised by `expectation_function` propagates as itself.

## The result

`train` and `qml.train` return a `TrainResult`:

| Field | Content |
|---|---|
| `best_params` | `list[float]`, the optimized parameters |
| `best_fitness` | Fitness at those parameters |
| `iterations_run` | Iterations actually run, early stopping included |
| `converged` | Whether the convergence criterion was met |
| `final_params` | With `final_shots`, the re-evaluated candidate with the best value; otherwise `None` |
| `final_fitness` | With `final_shots`, its value with `final_shots` shots; otherwise `None` |
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

### SPSA

```python
polypus.SPSA(
    max_iters=300,
    resamplings=1,
    perturbation=0.1,
    target_step=0.63,
    alpha=0.602,
    gamma=0.101,
    eval_interval=20,
    seed=None,
)
```

Simultaneous perturbation stochastic approximation (Spall, 1992) is a gradient method whose cost per step does not grow with the number of parameters. It moves every parameter at once by `±c_k`, with random signs `Δ`, and estimates the gradient from the difference of the two fitness values, `(f(θ + c_k·Δ) − f(θ − c_k·Δ)) / (2·c_k) · Δ`: two circuits per estimate, where Adam's finite difference takes `2 × dimensions`. It is a common choice on hardware: in Bonet-Monroig et al.'s comparison of four optimizers on small chemistry and materials problems (arXiv:2111.13454), SPSA did best with default settings, and CMA-ES matched or beat it only once its hyperparameters were tuned for each problem.

A run has three parts:

1. **Calibration.** One batch of 50 circuits around the starting point θ₀, 25 random perturbations in both directions, measures how large the gradient estimates are there, and SPSA sets its gain so that the first step moves each parameter by about `target_step` radians, following Spall's guideline (1998), as Qiskit's `SPSA.calibrate` does. Each pair gives `|f₊ − f₋| / (2·perturbation)`, the size of every component of its estimate, and the gain is `target_step` over their mean. If every pair has the same fitness (a flat landscape around θ₀), the gain is `target_step` and a warning is logged.
2. **Iterations.** Iteration `k` sends one batch of `2 × resamplings` circuits, `θ ± c_k·Δ` for `resamplings` random perturbations `Δ` of ±1, averages their estimates and steps uphill by `a_k` times the average. Spall's gains (1998) shrink the steps and the perturbations as the run goes on: `a_k = a / k^alpha` and `c_k = perturbation / k^gamma`, with his practical exponents as defaults.
3. **Measuring θ.** SPSA never evaluates its own parameters to take a step, so θ is added to the iteration's batch on the first iteration, every `eval_interval` iterations after it and on the last, as Jones, Mineh and Montanaro do (arXiv:2411.13742). `best_params` is the measured θ with the best fitness and `fitness_history` the best measured up to each iteration. The step after the last iteration is not measured.

| Parameter | Description |
|---|---|
| `max_iters` | Iterations, each one batch of `2 × resamplings` circuits, plus one when θ is measured. At least 1 |
| `resamplings` | Perturbations averaged per iteration, at least 1. Each adds two circuits to the iteration's batch |
| `perturbation` | `c`, the size of the first perturbations in radians, finite and `> 0` |
| `target_step` | Size of the first step on each parameter, in radians, finite and `> 0`. The default `0.63` is about `2π/10` |
| `alpha`, `gamma` | Decay exponents of the steps and of the perturbations: `alpha` in `(0.5, 1]`, `gamma > 0` with `alpha − gamma > 0.5`, Spall's conditions for the iterates to converge |
| `eval_interval` | Iterations between measurements of θ, at least 1 |

θ₀ is drawn uniformly from `[0, 2π)`, since SPSA takes no `bounds`, or is `initial_point`; the parameters are not wrapped afterwards. The run always makes `max_iters` iterations and reports `converged=False`, as QNG does: there is no early stop. It sends `1 + max_iters` batches and `50 + 2 × resamplings × max_iters` circuits plus one per measurement of θ, 666 with the defaults. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**When to prefer SPSA: many parameters, few QPUs.** An Adam iteration is one batch of `2 × dimensions + 1` circuits, 61 for 30 parameters and 401 for 200, which a single QPU runs one after another. An SPSA iteration is 2 circuits whatever the number of parameters. Its estimates are noisier than a finite difference, so it takes many cheap steps where Adam takes few expensive ones.

**`resamplings` fills the QPUs.** With one perturbation per iteration, each batch is 2 circuits, so a run on many QPUs leaves most of them idle and takes as many sequential rounds as iterations. `resamplings=K` makes each batch `2K` circuits, spread over the QPUs, and averages `K` estimates, so each step is less noisy and the same budget takes `K` times fewer rounds. Set `2 × resamplings` close to `n_qpus`. Fewer, averaged steps can cost quality when the budget is small: see [Choosing an optimizer](#choosing-an-optimizer).

**Tune `target_step` and `perturbation`.** The calibration sets the scale of the steps, but the result still depends on both. A `target_step` too large jumps between optima, one too small crawls. A small `perturbation` lets shot noise swamp the difference of the two fitness values, which is divided by `2 × c_k`, as with Adam's finite-difference step. Try a few values on instances of your problem kept apart from the ones you report: in [Choosing an optimizer](#choosing-an-optimizer), one perturbation per iteration on QAOA used `perturbation=0.2` and `target_step=1.2`, tuned that way.

**SPSA needs budget.** The calibration takes 50 circuits before the first step, and the steps that follow are small and noisy, so with a small budget SPSA can end below a global search: with 500 evaluations in [Choosing an optimizer](#choosing-an-optimizer), eight perturbations per iteration were no better than the classic DE. Like Adam, it is a local method: [Starting point](#starting-point) shows how to start it from a global search's result.

**Reproducibility.** As with CMA-ES, a seed reproduces an SPSA run exactly on the same machine and build, but may not across operating systems or CPU architectures: the gains `a / k^alpha` and `perturbation / k^gamma` are powers from the platform's math library, whose last bit can differ.

**References.**

- J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation", IEEE Transactions on Automatic Control 37(3), 332–341 (1992).
- J. C. Spall, "Implementation of the simultaneous perturbation algorithm for stochastic optimization", IEEE Transactions on Aerospace and Electronic Systems 34(3), 817–823 (1998).
- B. D. M. Jones, L. Mineh and A. Montanaro, "Benchmarking a wide range of optimisers for solving the Fermi-Hubbard model using the variational quantum eigensolver", Quantum Science and Technology 10(4), 045032 (2025), arXiv:2411.13742.
- X. Bonet-Monroig, H. Wang, D. Vermetten, B. Senjean, C. Moussa, T. Bäck, V. Dunjko and T. E. O'Brien, "Performance comparison of optimization methods on variational quantum algorithms", Physical Review A 107, 032407 (2023), arXiv:2111.13454.

### Rotosolve

```python
polypus.Rotosolve(
    sweeps=10,
    reevaluate_every=5,
    tolerance=0.01,
    patience=2,
    seed=None,
)
```

Rotosolve (Ostaszewski, Grant and Benedetti, 2021) solves for one parameter at a time, exactly, in circuits of single-Pauli rotations. When a parameter θ is the angle of one rotation `exp(−iθP/2)` generated by a Pauli operator `P`, and the fitness averages a cost per bitstring over the shots, the fitness with every other parameter fixed is a sinusoid of θ, `a + A·cos θ + B·sin θ`. Three values determine it, so a step evaluates the circuit with θ moved by `+π/2` and by `−π/2`, computes from the three values where the sinusoid has its maximum, and moves θ there. A sweep does this for every parameter in turn. Qiskit's `NFT` optimizer and PennyLane's `RotosolveOptimizer` with one frequency take steps of the same kind.

**Which circuits qualify.** Every trainable parameter must be the angle of exactly one `rx`, `ry`, `rz`, `rxx`, `rzz`, `p` or `u1` gate, and in a Qiskit circuit also `ryy` or `rzx`, as `θ` or `−θ` plus a constant, such as `-polypus.Param(3) + polypus.pi / 2`. Hardware-efficient ansätze of RY and RZ layers with CNOT or CZ entanglers qualify, Qiskit's `real_amplitudes`, `efficient_su2` and `RealAmplitudes` included. These do not:

- a parameter in more than one gate, as in QAOA, where each γ is the angle of every cost gate;
- a coefficient other than ±1, such as `rz(2θ)` or a Pauli evolution `exp(−iθZ)`: the fitness then has two periods in `[0, 2π)`, and the step lands in the wrong place;
- a controlled rotation (`crx`, `cry`, `crz`, `cp`, `cu1`), `u`, `u2`, `u3` or `r`; in a `polypus.Circuit`, any custom gate, with or without a body; in a Qiskit circuit, a custom gate without a definition, or one whose definition Qiskit cannot build, such as a `HamiltonianGate` whose time is a parameter;
- a parameter in no gate, or inside a control-flow block.

`train` checks the circuit before anything runs, a `polypus.Circuit` or a Qiskit `QuantumCircuit` (whose composite gates, such as the one `RealAmplitudes` builds, are read through their definitions), and raises `ValueError` naming the first parameter and gate that do not qualify:

```text
polypus.Rotosolve cannot train this circuit: parameter 1 ('theta_1') is an angle of 'crx'
(instruction 1), which is not a rotation generated by a single Pauli operator. Rotosolve
needs every trainable parameter to be the angle of exactly one rx, ry, rz, rxx, rzz, p or
u1 gate (also ryy or rzx in a Qiskit circuit), as ±θ + c, ...
```

In `qml.train` only the ansatz is trained: the feature map's parameters are bound to each sample before anything runs, so they may appear in any gate. The other optimizers do not check the circuit.

**Which costs.** The fitness must average a cost per bitstring over the shots: a `bitstring -> float` callable, a `polypus.CachedCost`, a `polypus.Qubo` or a `polypus.Ising`, and in `qml.train` the mean of those over the samples, with or without labels. Polypus cannot see what a callable computes, so it must return the same value for the same bitstring (and label). A `polypus.SampleCost` scores a sample's whole distribution, a log-likelihood for instance, which is not such an average; `qml.train` rejects it with `ValueError` before anything runs.

**Two circuits per step.** The step also needs the fitness at θ itself. Nakanishi, Fujii and Todo (2020) noted that after a step it is already known: it is the maximum of the sinusoid, which the step computed. So each step sends one batch of two circuits, θ ± π/2, and uses the value the previous step predicted. Under shot noise that prediction carries the noise of the three values it came from, and from step to step the error would build up, so the current parameters are measured again, as a third circuit in the step's batch, on the first step and every `reevaluate_every` steps. `reevaluate_every=1` measures them on every step, as Ostaszewski, Grant and Benedetti's Rotosolve does.

| Parameter | Description |
|---|---|
| `sweeps` | Sweeps at most, each one step, one batch, per parameter. At least 1 |
| `reevaluate_every` | Steps between measurements of the current parameters, at least 1 |
| `tolerance`, `patience` | The run ends after `patience` consecutive sweeps (at least 1) that raise the best measured fitness by less than `tolerance` (finite, `>= 0`); `tolerance=0` never stops early |

θ₀ is drawn uniformly from `[0, 2π)`, or is `initial_point`, and measured on the first step; a parameter is wrapped into `[0, 2π)` when a step moves it. `best_params` is the best of the points measured, the current parameters and their neighbours at ±π/2, never a predicted value, and `fitness_history` has one entry per step, the best measured up to it; `iterations_run` counts steps. A run sends at most `sweeps × dimensions` batches, each of 2 or 3 circuits: 660 circuits for 30 parameters with the defaults. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**When to prefer Rotosolve: a circuit that qualifies, and few QPUs.** There is no learning rate, step size or perturbation to tune: each step goes to the maximum along its parameter. Each batch is 2 or 3 circuits whatever the number of parameters, so, as with SPSA, a run suits few QPUs and leaves many idle, and it takes one sequential round per step. Under shot noise each step lands on the maximum of a sinusoid through three noisy values, off by an angle that shrinks with more shots. It is a local method too, climbing one parameter at a time to an optimum near where it starts: [Starting point](#starting-point) shows how to start it from a global search's result.

**Reproducibility.** The seed only draws θ₀; the parameters are visited in order. As with CMA-ES, a seeded run reproduces exactly on the same machine and build, but may not across operating systems or CPU architectures: each step calls `atan2` and `hypot` from the platform's math library, whose last bit can differ.

**References.**

- M. Ostaszewski, E. Grant and M. Benedetti, "Structure optimization for parameterized quantum circuits", Quantum 5, 391 (2021), doi:10.22331/q-2021-01-28-391, arXiv:1905.09692.
- K. M. Nakanishi, K. Fujii and S. Todo, "Sequential minimal optimization for quantum-classical hybrid algorithms", Physical Review Research 2, 043158 (2020), doi:10.1103/PhysRevResearch.2.043158, arXiv:1903.12166.

## Starting point

By default every optimizer starts from random parameters. `initial_point` makes it start from a point you choose, typically the `best_params` of an earlier run. It takes `dimensions` finite numbers as a list, tuple or NumPy array; another length, a `NaN` or an infinity raises `ValueError` before anything runs. `qml.train` takes it too, as a keyword argument.

| Optimizer | What it does with `initial_point` |
|---|---|
| `DE`, every `strategy` | Makes it, wrapped into `[0, 2π)`, the first member of the initial population, so it is the first candidate evaluated. The other members are drawn as without it |
| `PSO` | Makes it the position of the first particle, the first candidate evaluated; the other particles are drawn as without it. Every value must lie within `bounds`, since positions are clamped to them, otherwise `ValueError` before anything runs |
| `QNG` | Starts from it: the first batch estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |
| `CMAES` | Makes it, wrapped into `[0, 2π)`, the mean of the first run: the first generation is sampled around it with step size `sigma0` |
| `Adam` | Starts from it and evaluates it first: the first batch is the point, then the finite-difference stencil that estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |
| `SPSA` | Starts from it: the calibration batch is centred on it, and the first iteration measures it first. It is taken as given, not wrapped into `[0, 2π)` |
| `Rotosolve` | Starts from it and measures it first: the first batch is the point, then its two neighbours at `±π/2` along the first parameter. It is taken as given; a parameter is wrapped into `[0, 2π)` when a step moves it |

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

## Final re-evaluation

Under shot noise, `best_fitness` overstates the value of `best_params`. An optimizer keeps the best of many noisy estimates, and the best estimate is more often one that came out high by chance: the winner's curse. With 256 shots it reached 1.9 % of the maximum cut in the [benchmark](#choosing-an-optimizer), with local optimizers as well as population-based ones. `final_shots` reduces it: there, with `final_shots=4096`, the bias of `final_fitness` was at most 0.16 % on average. Once the optimizer has stopped, its best candidates are evaluated once more, in one batch with `final_shots` shots per circuit, and the result goes to two fields of its own:

```python
result = polypus.train(
    qc,
    polypus.DE(strategy="lshade", generations=100, population_size=30),
    shots=256,
    n_qpus=1,
    dimensions=dimensions,
    expectation_function=cost,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    id="maxcut",
    final_shots=4096,
)
result.final_params  # the re-evaluated candidate with the best value
result.final_fitness  # its value with 4096 shots
```

| Optimizer | Candidates re-evaluated |
|---|---|
| `DE` (every `strategy`), `PSO`, `CMAES` | `best_params` and the four best other distinct candidates the run scored |
| `QNG`, `Adam`, `SPSA`, `Rotosolve` | `best_params` |

`final_params` is the candidate with the highest re-evaluated value, `best_params` on a tie. Re-evaluating the best few candidates of a population, as Novák et al. recommend, lets the run pick one whose value with many shots beats `best_params`, which may have come first only by its noise. `final_fitness` is then the highest of a few fresh estimates, so it is still slightly optimistic, though far less than `best_fitness`, the highest of the whole run; when `best_params` is re-evaluated alone, it has no bias. `best_params`, `best_fitness` and `fitness_history` remain what the optimizer found: a seeded run sends the same batches with and without `final_shots`, and then the final batch. Without it, `final_params` and `final_fitness` are `None`.

`final_shots` must be an integer `>= 1`; anything else, such as `0`, `4096.0` or `True`, raises `ValueError` before anything runs. `qml.train` takes it as a keyword argument and runs every training circuit of each candidate with those shots. The batch holds at most five candidates, so it costs up to `5 × final_shots` shots per training circuit. It is sent once the optimizer has stopped, whether by its budget or its early stop, but not after an error or Ctrl+C. With `infrastructure="cunqa"`, the walltime warning does not count it, so leave time for it in `walltime`; if the QPUs are released first, the batch fails like any other and `train` raises.

**References.**

- V. Novák, S. Illésová, T. Bezděk, I. Zelinka and M. Beseda, "Reliable Optimization Under Noise in Quantum Variational Algorithms", arXiv:2511.08289 (2025).

## Choosing an optimizer

[`benchmarks/bench_optimizers.py`](../../benchmarks/bench_optimizers.py) trains two circuits for MaxCut on random 3-regular graphs of 10 nodes, with each optimizer at the same evaluation budget: QAOA with depth p = 3 or 5 (6 or 10 parameters), and a hardware-efficient ansatz (HEA) of three layers of RY on every qubit with a linear chain of CZ between them (Qiskit's `TwoLocal` with `ry`, `cz`, `entanglement="linear"` and `reps=2`), 30 parameters. The 30 instances are paired: each is one graph, run by every optimizer with the same seed. The score is the exact approximation ratio of the returned `best_params`, the expected cut from the statevector divided by the maximum cut, so shot noise in `best_fitness` does not flatter it. DE uses a population of 30, and early stopping is off except for CMA-ES with restarts, whose restarts need it. Adam uses its defaults on both circuits. Momentum uses a learning rate tuned for each circuit on separate instances, 0.03 for QAOA and 0.1 for the HEA, with a `finite_difference_step` of 0.4. "L-SHADE, then Adam" gives half the budget to L-SHADE and the rest to Adam, started from L-SHADE's `best_params`; "Adam ×3" gives a third to each of three Adam runs with different seeds and keeps the one with the best `best_fitness`. SPSA runs with one perturbation per iteration and with eight (`resamplings=8`), with settings tuned on separate instances: `perturbation=0.2` and `target_step=1.2` with one perturbation on QAOA, the defaults (0.1 and 0.63) otherwise; each makes as many iterations as fit the budget after its calibration. The circuits run on Polypus's native simulator, through a meter that counts every evaluation and stops each optimizer at the budget.

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
| `SPSA()`, tuned steps | +0.031 ± 0.005 (83 %) | +0.030 ± 0.009 (90 %) | +0.038 ± 0.011 (83 %) | +0.011 ± 0.009 (63 %) | +0.194 ± 0.011 (100 %) |
| `SPSA(resamplings=8)` | +0.010 ± 0.009 (70 %) | +0.003 ± 0.010 (67 %) | +0.048 ± 0.005 (97 %) | −0.022 ± 0.016 (57 %) | +0.190 ± 0.011 (100 %) |

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
| `SPSA()`, tuned steps | 0.782 | 0.671 | 0.666 | 0.701 | 0.845 |
| `SPSA(resamplings=8)` | 0.667 | 0.671 | 0.869 | 0.580 | 0.840 |

Sequential batches each optimizer sent, the rounds that cost time when the QPUs outnumber a batch, and the winner's curse: how far the reported `best_fitness` overstated the exact value of `best_params`, as a share of the maximum cut, with 256 shots. An Adam batch holds `2 × dimensions + 1` circuits: 13, 21 and 61 here. An SPSA batch holds 2 or 16 circuits, plus the parameters every 20 iterations, after a calibration batch of 50.

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
| `SPSA()`, tuned steps | 707 | 1439 | 2415 | +0.010 | +0.008 | +0.001 |
| `SPSA(resamplings=8)` | 91 | 184 | 309 | +0.006 | +0.006 | +0.001 |

What the data says:

- **Explore with L-SHADE, then refine with Adam.** Half the budget to `DE(strategy="lshade")` and the other half to `Adam()` from its `best_params` ([Starting point](#starting-point)) gave the best mean ratio in all five scenarios, 500 evaluations included, and beat the classic DE on 97 to 100 % of the instances. Its lowest ratio was the highest of all the optimizers in four scenarios (0.918 with p = 5, 0.991 on the HEA), and its Adam phase improved on L-SHADE's half-budget result on every instance. Its lead over Adam alone was 0.007 to 0.022, more than two standard errors with 256 shots at p = 3 and 1500 evaluations (+0.013 ± 0.004) and on the HEA (+0.017 ± 0.007), and about two or fewer elsewhere. On the HEA the two were even on most instances, and the lead came from the few where Adam alone ended low.
- **Adam alone, with its defaults, is a strong single run with one configuration for both circuits.** It beat L-SHADE on average in every scenario: clearly with 256 shots at p = 3 (+0.013 ± 0.004 at 1500 evaluations, +0.027 ± 0.004 at 500) and on the HEA (+0.044 ± 0.007), within noise with 2^20 shots (+0.003 ± 0.004) and at p = 5 (+0.012 ± 0.011). As a local method it occasionally ends in a poor optimum: its lowest ratio was 0.661 with p = 5 and 0.784 with 2^20 shots, against 0.830 and 0.831 for L-SHADE. Its batches grow with the parameters, so with 30 it sent the fewest of all (81, against 385 for L-SHADE).
- **Use momentum only with a learning rate tuned to the problem.** With rates tuned on separate instances it matched Adam within noise (0.002 to 0.016 higher), and its lowest ratio at p = 5 was 0.861, against Adam's 0.661. Its rate depends on the scale of the cost (see [Adam](#adam)), so these figures are its best case; Adam needed no tuning.
- **Don't restart Adam by default.** Three runs on a third of the budget each raised the worst instance at p = 5 (lowest ratio 0.907, against 0.661 for one run), but their mean was within noise of one run's there (+0.016 ± 0.011) and with 2^20 shots (+0.004 ± 0.003). With 256 shots at p = 3 they were behind one run (−0.006 ± 0.003 at 1500 evaluations, −0.011 ± 0.003 at 500), and on the HEA too (−0.026 ± 0.008), where a third of the budget was too short. Starting from L-SHADE had the higher mean in every scenario.
- **SPSA with one perturbation per iteration was within noise of Adam once its steps were tuned.** With 1500 evaluations or more its paired difference from Adam was within noise in every scenario (+0.001 ± 0.006 at p = 3 with 256 shots, −0.003 ± 0.008 with 2^20, −0.011 ± 0.015 at p = 5, −0.003 ± 0.011 on the HEA), and it beat the classic DE on 83 to 100 % of the instances. Like Adam it sometimes ended in a poor optimum, with a lowest ratio of 0.671 with 2^20 shots and 0.666 at p = 5. On the 30-parameter HEA both SPSA variants ended level with Adam (0.977 and 0.973, against 0.980) and about 0.02 below L-SHADE then Adam. Every optimizer here has the same evaluations and is scored at the end, so the benchmark does not show whether SPSA's cheaper steps reach a good ratio sooner with many parameters.
- **SPSA needs budget.** With 500 evaluations, after its 50-circuit calibration, one perturbation per iteration gained only +0.011 ± 0.009 over the classic DE, 0.024 ± 0.009 below Adam, and eight perturbations per iteration, 27 iterations in all, were no better than it (−0.022 ± 0.016, within noise), with a lowest ratio of 0.580.
- **`resamplings=8` cuts SPSA's sequential batches about eightfold, at a cost in quality on the smaller problem.** It sent 91 batches at 1500 evaluations instead of 707 (115 for Adam), each of 16 or 17 circuits, a single round on 17 QPUs. At p = 3 its mean was below one perturbation's (−0.021 ± 0.009 with 256 shots, −0.027 ± 0.011 with 2^20), and one instance stayed at 0.667. At p = 5 with 3000 evaluations its mean was within noise of one perturbation's (+0.048 ± 0.005 over the classic DE, against +0.038 ± 0.011), and its lowest ratio was 0.869, where one perturbation had two instances below 0.8 (0.666 and 0.762). On the HEA they were level.
- **CMA-ES with its default population is competitive with L-SHADE.** Its paired difference from L-SHADE was within two standard errors in every scenario: about even with 256 shots (+0.003 ± 0.005 at p = 3, +0.001 ± 0.006 at p = 5), slightly ahead with 2^20 shots (+0.007 ± 0.004) and on the HEA (+0.017 ± 0.016). It was less consistent on the HEA, beating the classic DE on 90 % of the instances where L-SHADE beat it on all, with a lowest ratio of 0.740 against 0.834. On QAOA it sent the most sequential batches of the population-based optimizers and, at 1500 and 3000 evaluations, had the largest winner's curse.
- **Don't enlarge CMA-ES's population to fill QPUs, and don't count on IPOP restarts.** With a population of 30, CMA-ES's mean ratio was below its default population's in every scenario, by 0.005 to 0.037, though still about 0.01 above the classic DE's at 1500 and 3000 evaluations. IPOP restarts stayed within 0.01 of the classic DE on QAOA and gained +0.015 ± 0.014 on the HEA, against +0.170 without them.
- **With many QPUs, when sequential rounds dominate the run time,** `strategy="shade"` keeps the classic DE's batch size and number of rounds and still gained 1.5 points with 10 parameters and 10.6 with 30, though almost nothing with 6 at 256 shots. L-SHADE needs about 2.3 times the rounds of the classic DE for its larger gain.
- **With 500 evaluations only the Adam-based optimizers were clearly better than the classic DE,** by 0.023 to 0.044; the DE strategies and CMA-ES gained 0.008 at most.
- **The reported `best_fitness` is optimistic under shot noise,** because it is the best of many noisy estimates: by up to 1.9 % of the maximum cut with 256 shots, for CMA-ES at p = 3. Adam's `best_params` is the best point it measured, so Adam has the same bias: 1.0 to 1.5 % on QAOA with 256 shots, a little more than L-SHADE's and more than the classic DE's, and 0.2 % on the HEA. SPSA measures its parameters only every 20 iterations, so it picks from fewer estimates: 0.3 to 1.0 % on QAOA and 0.1 % on the HEA. Pass `final_shots` when the value itself matters ([Final re-evaluation](#final-re-evaluation)): with `final_shots=4096`, 16 times the training shots, `final_fitness` was within 0.16 % of the exact value of `final_params` on average for every optimizer in every scenario, and within 0.6 % in every run. For Adam and SPSA it is a fresh estimate of `best_params`, without bias: its mean error was under 0.07 % in every scenario. For DE and CMA-ES it is the highest of five noisy values, so it stays slightly optimistic, by up to 0.16 % (CMA-ES at p = 3 with 1500 evaluations). Their `final_params` was another candidate than `best_params` in 6 to 24 runs of 30, and a better one in 154 of those 179 runs (worse by 0.003 at most otherwise); their mean ratio rose by 0.001 to 0.005 in every scenario, by less than two standard errors in 4 of the 12 cases.

These results cover MaxCut only, on 10 nodes, with QAOA at depth 3 to 5 and one 30-parameter hardware-efficient ansatz, on a noiseless simulator with shot sampling. Other problems, more parameters or hardware noise may rank the optimizers differently, and momentum's learning rates and SPSA's steps were tuned for these two circuits. CMA-ES's and SPSA's rows may differ on another platform or build; see the reproducibility notes under [CMA-ES](#cma-es) and [SPSA](#spsa). Run `python benchmarks/bench_optimizers.py --full`, or a scenario of your own with `--scenario [hea:]LAYERS:SHOTS:BUDGET`, to check. The tables come from that command at commit `83a0bbd` (Polypus 0.7.2) on Linux x86_64 with an Intel Xeon at 2.10 GHz (4 CPUs), and the SPSA rows from `--full --only rand1bin spsa spsa-k8` at commit `7f3cd67` on the same platform, whose `rand1bin` runs were identical to the first run's. The `final_shots` figures come from `--full --scenario 3:256:1500 5:256:3000 3:256:500 hea:2:256:5000 --only rand1bin lshade cmaes adam spsa --final-shots 4096` at commit `2fb45a6` on the same platform, whose ratios, biases and batches were identical to those runs'. The 500-evaluation scenario gives a population of 30 only 480 evaluations, 16 batches of 30. The printed table gives the differences from the classic DE; the lowest ratios and the differences from L-SHADE or Adam quoted above come from the per-instance results that `--json FILE` writes.
