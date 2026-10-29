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
| `method` | `polypus.DE`, `polypus.PSO`, `polypus.QNG`, `polypus.CMAES`, `polypus.Adam`, `polypus.SPSA`, `polypus.Rotosolve`, or `polypus.External` for an optimizer from another library or of your own |
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

Ctrl+C stops the optimization promptly and raises `KeyboardInterrupt`. An exception raised by `expectation_function`, or by the `ask()` or `tell()` of an [external optimizer](#external-optimizers), propagates as itself.

## The result

`train`, `qml.train` and `qml.Model.fit` return a `TrainResult`:

| Field | Content |
|---|---|
| `best_params` | `list[float]`, the optimized parameters |
| `best_fitness` | Fitness at those parameters |
| `iterations_run` | Iterations actually run, early stopping included |
| `fitness_history` | `list[float]`, the convergence curve: the best fitness so far after each iteration run, so it never decreases and has `iterations_run` entries. It ends at `best_fitness`, except when a gradient-free optimizer trains on minibatches: its curve holds the minibatch estimates it steered by, while `best_fitness` is re-scored on the whole training set |
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
    mutation_factor=None,
    crossover_rate=None,
)
```

The run stops early when the best fitness improves by less than `tolerance` over the last `patience` generations. `patience` is at least 1: `train` raises `ValueError` for 0 before anything runs.

`strategy` picks how each generation builds its trial vectors:

| `strategy` | Algorithm |
|---|---|
| `"rand1bin"` (default) | Classic DE/rand/1/bin with a fixed mutation factor `F` and crossover rate `CR` |
| `"shade"` | SHADE (Tanabe and Fukunaga, 2013): current-to-pbest/1 mutation with an archive of replaced parents. Each individual draws its own `F` and `CR` from a memory of the values that recently improved the population, so there is nothing to tune |
| `"lshade"` | L-SHADE (Tanabe and Fukunaga, 2014): SHADE with a population that shrinks linearly towards 4 |

Any other name raises `ValueError`. `population_size` must be at least 4 with every strategy, and every strategy evaluates one batch per generation, after the initial population's.

For `"rand1bin"`, `mutation_factor` is `F`, which scales the difference vector, in `(0, 2]` (default 0.8), and `crossover_rate` is `CR`, the probability that a coordinate of a trial comes from the mutant, in `[0, 1]` (default 0.7). One coordinate always comes from the mutant, so no trial is a plain copy of its parent. SHADE and L-SHADE adapt both, so passing either with them raises `ValueError`, as does a value out of range.

Every strategy treats the parameters as angles: a difference between two members is the shorter way around the circle, in `(−π, π]`, so members at 0.05 and 6.23 rad are 0.1 rad apart, and an optimum near 0 is found as easily as one at π.

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
| `tolerance`, `patience` | The run ends when its best fitness improves by less than `tolerance` over its last `patience` generations, as with DE. `patience` is at least 1 |
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
| `"momentum"` | Polyak's heavy ball (1964): a velocity, `beta1 × velocity + learning_rate × gradient`, is added to the parameters. The step grows with the gradient, so the `learning_rate` that works depends on the scale of the cost: a rate tuned on one problem can make another, with a larger cost, diverge. With the defaults (`learning_rate=0.1`, `beta1=0.9`) a steady gradient `g` moves the parameters by `learning_rate / (1 − beta1) × g = 1.0 × g` per iteration, so on a cost of hundreds, such as a QUBO's energies, steps of hundreds of radians diverge: lower `learning_rate` in proportion to the cost, or keep `"adam"`. `beta2` and `epsilon` are not used |

Any other `rule` raises `ValueError`.

| Parameter | Description |
|---|---|
| `max_iters` | Iterations, each one batch of `2 × dimensions + 1` circuits and one step; the final parameters cost one circuit more. At least 1 |
| `learning_rate` | Step size, finite and `> 0`. The default `0.1` is sized for rotation angles in radians rather than network weights, so it is larger than deep learning's usual `0.001`. With `"momentum"`, tune it to the scale of the cost |
| `beta1`, `beta2` | Decay of the running averages of the gradient and of its square, in `[0, 1)`. With `"momentum"`, `beta1` is the momentum and `beta2` is unused |
| `epsilon` | Added to the denominator for numerical stability, finite and `> 0`. Unused by `"momentum"` |
| `bounds` | Range the initial parameters are drawn from, finite with `lower < upper`. The run itself is not confined to it, and the parameters are not wrapped into `[0, 2π)` |
| `tolerance`, `patience` | The run ends when its best fitness improves by less than `tolerance` over its last `patience` iterations, as with DE. `patience` is at least 1 |
| `finite_difference_step` | Step of the central difference, in radians, finite and `> 0` |
| `rule` | `"adam"` or `"momentum"`, above |

**The finite-difference step matters.** Shot noise enters the gradient through the difference of two fitness values divided by `2 × finite_difference_step`, so a small step amplifies it. Jones, Mineh and Montanaro (arXiv:2411.13742) found that a step of 0.4 works and 0.01 ruins the gradient. Prefer the default `0.2` to a small step, which needs many more shots for the same accuracy.

`best_params` is the `θ` with the best measured fitness, among the points the run stepped to: the `θ ± finite_difference_step` around them only give the gradient, and never count, even when one of them measured higher. `fitness_history` is the best fitness measured up to each iteration. After the last step, whether `max_iters` ran out or the run stagnated, one more circuit measures the parameters it reached, which count in the last entry, so no gradient is spent on a step that is never measured. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

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
    stability_constant=0.0,
)
```

Simultaneous perturbation stochastic approximation (Spall, 1992) is a gradient method whose cost per step does not grow with the number of parameters. It moves every parameter at once by `±c_k`, with random signs `Δ`, and estimates the gradient from the difference of the two fitness values, `(f(θ + c_k·Δ) − f(θ − c_k·Δ)) / (2·c_k) · Δ`: two circuits per estimate, where Adam's finite difference takes `2 × dimensions`. It is a common choice on hardware: in Bonet-Monroig et al.'s comparison of four optimizers on small chemistry and materials problems (arXiv:2111.13454), SPSA did best with default settings, and CMA-ES matched or beat it only once its hyperparameters were tuned for each problem.

A run has three parts:

1. **Calibration.** One batch of 50 circuits around the starting point θ₀, 25 random perturbations in both directions, measures how large the gradient estimates are there, and SPSA sets its gain so that, with `resamplings=1`, the first step moves each parameter by about `target_step` radians, following Spall's guideline (1998), as Qiskit's `SPSA.calibrate` does. With more perturbations per iteration the average of their estimates is smaller on average than one estimate, so the first step is usually shorter. Each pair gives `|f₊ − f₋| / (2·perturbation)`, the size of every component of its estimate, and the gain is `target_step` over their mean. If every pair has the same fitness (a flat landscape around θ₀), the gain is `target_step` and a warning is logged.
2. **Iterations.** Iteration `k` sends one batch of `2 × resamplings` circuits, `θ ± c_k·Δ` for `resamplings` random perturbations `Δ` of ±1, averages their estimates and steps uphill by `a_k` times the average. Spall's gains (1998) shrink the steps and the perturbations as the run goes on: `a_k = a / (k + A)^alpha` and `c_k = perturbation / k^gamma`, with his practical exponents as defaults and `A` the `stability_constant`, 0 by default. The calibration keeps the first step at `target_step` whatever `A`, so a larger `A` only makes the steps shrink more slowly at first: with `A = 30`, the tenth gain is still 86 % of the first, where with `A = 0` it is 25 %. Spall suggests an `A` of about a tenth of `max_iters` or less, for a run whose early steps would otherwise shrink before it has moved far. Qiskit's `SPSA.calibrate` does not scale `a` with `A`, so there a larger `A` makes the first step shorter.
3. **Measuring θ.** SPSA never evaluates its own parameters to take a step, so θ is added to the iteration's batch on the first iteration and every `eval_interval` iterations after it, as Jones, Mineh and Montanaro do (arXiv:2411.13742), and after the last step one more circuit measures the parameters it reached, so the last iteration's circuits are not wasted on a step that is never measured. `best_params` is the measured θ with the best fitness and `fitness_history` the best measured up to each iteration, the final parameters counted in the last entry.

| Parameter | Description |
|---|---|
| `max_iters` | Iterations, each one batch of `2 × resamplings` circuits, plus one when θ is measured; the final parameters cost one circuit more. At least 1 |
| `resamplings` | Perturbations averaged per iteration, from 1 to 2147483647, so that the batch fits in 32 bits. Each adds two circuits to the iteration's batch |
| `perturbation` | `c`, the size of the first perturbations in radians, finite and `> 0` |
| `target_step` | Size of the first step on each parameter with `resamplings=1`, in radians, finite and `> 0`; with more perturbations the first step is usually shorter. The default `0.63` is about `2π/10` |
| `alpha`, `gamma` | Decay exponents of the steps and of the perturbations: `alpha` in `(0.5, 1]`, `gamma > 0` with `alpha − gamma > 0.5`, Spall's conditions for the iterates to converge |
| `stability_constant` | Spall's `A`, finite and `>= 0`: a larger value slows the decay of the early steps without changing the first. It comes after `seed` in the positional order |
| `eval_interval` | Iterations between measurements of θ, at least 1 |

θ₀ is drawn uniformly from `[0, 2π)`, since SPSA takes no `bounds`, or is `initial_point`; the parameters are not wrapped afterwards. The run always makes `max_iters` iterations and reports `converged=False`, as QNG does: there is no early stop. It sends `2 + max_iters` batches and `50 + 2 × resamplings × max_iters` circuits plus one per measurement of θ, the final one included, 666 with the defaults. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**When to prefer SPSA: many parameters, few QPUs.** An Adam iteration is one batch of `2 × dimensions + 1` circuits, 61 for 30 parameters and 401 for 200, which a single QPU runs one after another. An SPSA iteration is 2 circuits whatever the number of parameters. Its estimates are noisier than a finite difference, so it takes many cheap steps where Adam takes few expensive ones.

**`resamplings` fills the QPUs.** With one perturbation per iteration, each batch is 2 circuits, so a run on many QPUs leaves most of them idle and takes as many sequential rounds as iterations. `resamplings=K` makes each batch `2K` circuits, spread over the QPUs, and averages `K` estimates, so each step is less noisy and the same budget takes `K` times fewer rounds. Set `2 × resamplings` close to `n_qpus`. The rounds saved can cost quality: in [Choosing an optimizer](#choosing-an-optimizer), eight perturbations per iteration, with the same settings as one, were below one perturbation with 256 shots at p = 3 and within noise of it elsewhere, never above it. Their steps were also shorter, since the calibration sets the gain from single estimates, so that comparison mixes the number of perturbations with the step size.

**Tune `target_step` and `perturbation`.** The calibration sets the scale of the steps, but the result still depends on both. A `target_step` too large jumps between optima, one too small crawls. A small `perturbation` lets shot noise swamp the difference of the two fitness values, which is divided by `2 × c_k`, as with Adam's finite-difference step. Try a few values on instances of your problem kept apart from the ones you report: in [Choosing an optimizer](#choosing-an-optimizer), SPSA on QAOA used `perturbation=0.2` and `target_step=1.2`, tuned that way with one perturbation per iteration.

**SPSA needs budget.** The calibration takes 50 circuits before the first step, and the steps that follow are small and noisy, so with a small budget SPSA can end below a global search: with 500 evaluations in [Choosing an optimizer](#choosing-an-optimizer), one perturbation per iteration was within noise of the classic DE and eight were below it. Like Adam, it is a local method: [Starting point](#starting-point) shows how to start it from a global search's result.

**Reproducibility.** As with CMA-ES, a seed reproduces an SPSA run exactly on the same machine and build, but may not across operating systems or CPU architectures: the gains `a / (k + A)^alpha` and `perturbation / k^gamma` are powers from the platform's math library, whose last bit can differ.

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

**Which circuits qualify.** Every trainable parameter must be exactly one angle of an `rx`, `ry`, `rz`, `rxx`, `rzz`, `p`, `u1`, `u`, `u2` or `u3` gate, and in a Qiskit circuit also `ryy`, `rzx` or the angle θ of an `r`, as `θ` or `−θ` plus a constant, such as `-polypus.Param(3) + polypus.pi / 2`. `u3(θ, φ, λ)` is `rz(φ)·ry(θ)·rz(λ)` up to a global phase, so each of its three angles qualifies, each with its own parameter. Hardware-efficient ansätze of RY and RZ layers with CNOT or CZ entanglers qualify, Qiskit's `real_amplitudes`, `efficient_su2` and `RealAmplitudes` included. These do not:

- a parameter in more than one gate, as in QAOA, where each γ is the angle of every cost gate;
- a coefficient other than ±1, such as `rz(2θ)` or a Pauli evolution `exp(−iθZ)`: the fitness then has two periods in `[0, 2π)`, and the step lands in the wrong place;
- a controlled rotation (`crx`, `cry`, `crz`, `cp`, `cu1`, `cu3`); the angle φ of an `r(θ, φ)`, which turns the axis of the rotation and so is in it twice; in a `polypus.Circuit`, any custom gate, with or without a body; in a Qiskit circuit, a custom gate without a definition, or one whose definition Qiskit cannot build, such as a `HamiltonianGate` whose time is a parameter;
- a parameter in no gate, or inside a control-flow block.

`train` checks the circuit before anything runs, a `polypus.Circuit` or a Qiskit `QuantumCircuit` (whose composite gates, such as the one `RealAmplitudes` builds, are read through their definitions), and raises `ValueError` naming the first parameter and gate that do not qualify:

```text
polypus.Rotosolve cannot train this circuit: parameter 1 ('theta_1') is an angle of 'crx'
(instruction 1), which is not a rotation generated by a single Pauli operator. Rotosolve
needs every trainable parameter to be exactly one angle of an rx, ry, rz, rxx, rzz, p, u1,
u, u2 or u3 gate (also ryy, rzx, or the θ of an r, in a Qiskit circuit), as ±θ + c, ...
```

In `qml.train` only the ansatz is trained: the feature map's parameters are bound to each sample before anything runs, so they may appear in any gate. The other optimizers do not check the circuit.

**Which costs.** The fitness must average a cost per bitstring over the shots: a `bitstring -> float` callable, a `polypus.CachedCost`, a `polypus.Qubo` or a `polypus.Ising`, and in `qml.train` the mean of those over the samples, with or without labels. Polypus cannot see what a callable computes, so it must return the same value for the same bitstring (and label). A `polypus.SampleCost` scores a sample's whole distribution, a log-likelihood for instance, which is not such an average; `qml.train` rejects it with `ValueError` before anything runs.

**Two circuits per step.** The step also needs the fitness at θ itself. Nakanishi, Fujii and Todo (2020) noted that after a step it is already known: it is the maximum of the sinusoid, which the step computed. So each step sends one batch of two circuits, θ ± π/2, and uses the value the previous step predicted. Under shot noise that prediction carries the noise of the three values it came from, and from step to step the error would build up, so the current parameters are measured again, as a third circuit in the step's batch, on the first step and every `reevaluate_every` steps. `reevaluate_every=1` measures them on every step, as Ostaszewski, Grant and Benedetti's Rotosolve does.

| Parameter | Description |
|---|---|
| `sweeps` | Sweeps at most, each one step, one batch, per parameter. At least 1 |
| `reevaluate_every` | Steps between measurements of the current parameters, at least 1 |
| `tolerance`, `patience` | The run ends after `patience` consecutive sweeps (at least 1) that raise the best measured fitness by less than `tolerance` (finite, `>= 0`); `tolerance=0` never stops early |

θ₀ is drawn uniformly from `[0, 2π)`, or is `initial_point`, and measured on the first step; a parameter is wrapped into `[0, 2π)` when a step moves it. `best_params` is the best of the points measured, the current parameters and their neighbours at ±π/2, never a predicted value. The neighbours are measured as they are, not wrapped, so `best_params` can lie up to π/2 outside `[0, 2π)`, or anywhere an `initial_point` outside it puts the parameters not yet moved. `fitness_history` has one entry per step, the best measured up to it; `iterations_run` counts steps. After the last step, whether the run used its sweeps or stopped early, the parameters it reached are measured too, in one more batch of one circuit, whose value joins the last entry of `fitness_history`. A run sends at most `sweeps × dimensions` batches of 2 or 3 circuits and that last one: 661 circuits for 30 parameters with the defaults. Invalid settings raise `ValueError` when `train` starts, before any circuit runs.

**When to prefer Rotosolve: a circuit that qualifies, and few QPUs.** There is no learning rate, step size or perturbation to tune: each step goes to the maximum along its parameter. Each batch is 2 or 3 circuits whatever the number of parameters, so, as with SPSA, a run suits few QPUs and leaves many idle, and it takes one sequential round per step. Under shot noise each step lands on the maximum of a sinusoid through three noisy values, off by an angle that shrinks with more shots. It is a local method too, climbing one parameter at a time to an optimum near where it starts: [Starting point](#starting-point) shows how to start it from a global search's result. In [Choosing an optimizer](#choosing-an-optimizer), on a 30-parameter hardware-efficient ansatz, it ended level with Adam and SPSA on average, and 0.021 ± 0.011 below L-SHADE then Adam, about two standard errors, because 5 of its 30 runs stopped in a poorer optimum.

**Reproducibility.** The seed only draws θ₀; the parameters are visited in order. As with CMA-ES, a seeded run reproduces exactly on the same machine and build, but may not across operating systems or CPU architectures: each step calls `atan2` and `hypot` from the platform's math library, whose last bit can differ.

**References.**

- M. Ostaszewski, E. Grant and M. Benedetti, "Structure optimization for parameterized quantum circuits", Quantum 5, 391 (2021), doi:10.22331/q-2021-01-28-391, arXiv:1905.09692.
- K. M. Nakanishi, K. Fujii and S. Todo, "Sequential minimal optimization for quantum-classical hybrid algorithms", Physical Review Research 2, 043158 (2020), doi:10.1103/PhysRevResearch.2.043158, arXiv:1903.12166.

### External optimizers

```python
polypus.External(optimizer, max_evaluations)
```

`polypus.External` runs an optimizer that is not part of Polypus through the same scheduler as the others: a library you install, such as Optuna, Nevergrad or pycma, or an object of your own. The optimizer only has to work in ask/tell form: `ask()` proposes a batch of candidates, Polypus evaluates the batch, and `tell(candidates, values)` hands back their values. Each `ask()` is one batch, spread over the `n_qpus` QPUs like a DE generation, so an optimizer that proposes several candidates at a time keeps several QPUs busy.

**The protocol.** `optimizer` is any object with two methods:

- `ask()` returns the next batch: an iterable of rows, such as a list of lists or a 2-D NumPy array, each row one finite number per circuit parameter. `None` or an empty batch means the optimizer is done.
- `tell(candidates, values)` receives the rows of that batch, as a `list[list[float]]`, and their fitness, as a `list[float]` in the same order. Its return value is ignored.

The values are fitness to **maximise**, as for every Polypus optimizer. Optuna minimises unless the study is created with `direction="maximize"`, and Nevergrad and pycma always minimise, so their values must be negated: the adapters in `polypus.ask_tell`, below, do it for you. `polypus.ask_tell.AskTell` is the protocol as a `typing.Protocol`, for type checkers.

**The budget.** `max_evaluations`, an integer `>= 1`, is the most candidates a run evaluates. It is required, since some libraries never say they are done. The run stops:

- when `ask()` returns `None` or an empty batch, and then `converged` is `True`;
- when the budget is spent, without asking again, or before a batch that would take the run past `max_evaluations`: that batch is neither evaluated nor told, so the budget is never exceeded, and `converged` is `False`.

A first `ask()` that proposes nothing, and a first batch larger than `max_evaluations`, raise `ValueError`: nothing was evaluated, so there is no result. With an optimizer that asks for a fixed number of candidates per batch, make `max_evaluations` a multiple of it, or the remainder is not spent.

**What is checked.** Every row must be one finite real number per parameter. A row of another length, a `NaN`, an infinity or a value that is not a real number (a `bool`, including a NumPy bool array's, a complex number, or an integer too large for a float) raises `ValueError` naming the batch and the row (both counted from 0), before that batch is evaluated, and the run stops. Polypus reads a batch only up to one row past the budget left, so an endless iterator ends the run like any batch that does not fit, and a row only up to one value past the number of parameters, so an endless row raises the `ValueError` of a row too long. An exception raised in `ask()` or `tell()`, `KeyboardInterrupt` included, stops the run and reaches the caller of `train` as itself, as one raised by `expectation_function` does; Ctrl+C during a batch stops the run before the optimizer's next call. `External` raises `TypeError` for an `optimizer` without callable `ask` and `tell`, `ValueError` for a `max_evaluations` below 1, and `TypeError` for one that is not an integer (`100.0` or `True`), when it is built or the attribute is set.

| Parameter | Description |
|---|---|
| `optimizer` | The ask/tell object: an adapter from `polypus.ask_tell`, or your own |
| `max_evaluations` | Candidates evaluated at most over a run, an integer `>= 1` |

**The result.** `best_params` and `best_fitness` are the best candidate Polypus measured and the value it measured, whatever the library would recommend as its own best. `fitness_history` has one entry per batch, the best value measured up to it, and `iterations_run` counts the batches evaluated and told.

**Seed and starting point.** `External` has no `seed`: the library owns its randomness, so seed it where you build it, as the examples below do. The `seed` of `train` still fixes the shot sampling and is reported in `result.seed`, so a deterministic optimizer on a seeded native run reproduces it exactly; whether a library is deterministic once seeded is up to the library. The start is the library's too: `initial_point` with `External` raises `ValueError` before anything runs. Give the start to the library instead: `from_optuna` and `from_nevergrad` take an `initial_point`, and pycma's start is the `x0` of `CMAEvolutionStrategy`.

**Threads and reuse.** As in every training run, the GIL is released while the run goes on. `ask()` and `tell()` run on the thread that called `train`, which holds the GIL only during each call, so other Python threads run while a batch is evaluated. `External` keeps your object, not a copy: a second run with it continues from the library's state (an Optuna study keeps its trials, a pycma strategy its generation), and `max_evaluations` counts each run's evaluations on their own. A pycma strategy that has stopped proposes nothing, so a second run with it raises `ValueError`. Do not share one optimizer between two runs at the same time.

With `final_shots`, the five best distinct candidates are re-evaluated, as for DE, PSO and CMA-ES, since Polypus cannot tell whether the library searches globally or locally. With `infrastructure="cunqa"`, the walltime warning projects the run from its first batch, assuming every batch has its size, and times that batch until the second starts, so the time the library itself takes in `tell()` and `ask()` between them counts too (see [Allocation walltime](../backends.md#allocation-walltime)). A model-based library takes longer as it sees more values, so its projection can still fall short.

None of the three libraries is a dependency of Polypus: install the one you use from PyPI, `optuna`, `nevergrad` or `cma` (pycma's name there), or install Polypus with it as an extra, `pip install "polypus-quantum[optuna]"` (`[nevergrad]`, `[cma]`), which asks for a version from the one these adapters are tested with (Optuna 5.0.0, Nevergrad 1.0.12, pycma 4.5.0) up to the next major. `polypus.ask_tell` does not import them; its adapters only call the public ask/tell methods of the object you built. Each adapter keeps a batch it handed out until it is told, and offers it again on the next `ask()`, so a batch the budget left untold is the first of the next run. Its `tell` checks that the candidates are the rows of that batch, in its order, with one value each, and raises `ValueError` otherwise, before the library is told anything: Polypus always tells the rows it was given, so only a `tell` of your own can get them wrong, and its values would reach other candidates. The examples below train a circuit `qc` with 6 parameters on 4 QPUs and share these arguments:

```python
import math

import numpy as np
import polypus
from polypus.ask_tell import from_nevergrad, from_optuna, from_pycma

common = dict(
    shots=1024,
    n_qpus=4,
    dimensions=6,
    expectation_function=cost,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    seed=42,  # fixes the shot sampling; each library has its own seed
)
```

**Optuna (TPE).** `from_optuna(study, dimensions, batch_size, bounds=(0, 2π), initial_point=None)` asks the study for `batch_size` trials per batch (`study.ask()`) and suggests one float per parameter in each, named `x0`, `x1`, … (`trial.suggest_float` within `bounds`), then tells each trial its value (`study.tell`), negated when the study minimises. A study with several objectives raises `ValueError`. Optuna's `TPESampler` is a form of Bayesian optimization: it builds a model of the values the trials returned and draws the next trials from it, after `n_startup_trials` random ones (10 by default in Optuna 5.0.0, the version checked here). With `constant_liar=True`, the default in Optuna 5.0 and passed explicitly below for older versions, it penalizes the trials still waiting for their values, so it avoids proposing the trials of one batch close to each other. Optuna never says it is done, so `max_evaluations` ends the run: make it a multiple of `batch_size`, since the trials of an untold remainder stay in the study as `RUNNING` trials, which TPE with `constant_liar=True` keeps penalising, until a later run with the same adapter tells them. To end a study instead, call the adapter's `close()`: it tells the study that those trials failed (`TrialState.FAIL`), which the samplers ignore, and does nothing when every batch was told. The adapter can go on after it, with new trials. `from_optuna` returns an `OptunaAskTell`, the protocol with `close()`. `initial_point` is enqueued as the study's first trial (`study.enqueue_trial`).

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=7, constant_liar=True),
)
adapter = from_optuna(study, dimensions=6, batch_size=4)
result = polypus.train(
    qc, polypus.External(adapter, max_evaluations=200), id="tpe", **common
)
len(study.trials)  # 200, each with the value Polypus measured
adapter.close()  # no trial left RUNNING when 4 does not divide the budget
```

**Nevergrad.** `from_nevergrad(optimizer, batch_size=None, initial_point=None)` asks the optimizer for `batch_size` candidates per batch, by default its `num_workers`, and tells each `-value`. When the optimizer has a `budget`, the run ends once that many candidates have been asked, the last batch cut to fit; without one, `max_evaluations` ends it. `initial_point` is suggested to the optimizer (`optimizer.suggest`), so it is the first candidate. Any optimizer of `ng.optimizers` works the same way; this example uses `TBPSA`, which Nevergrad's documentation (version 1.0.12, the one checked here) describes as suited to noisy objectives, and seeds the parametrization's `random_state`:

```python
import nevergrad as ng

parametrization = ng.p.Array(shape=(6,)).set_bounds(0, 2 * math.pi)
parametrization.random_state = np.random.RandomState(7)
optimizer = ng.optimizers.TBPSA(
    parametrization=parametrization, budget=200, num_workers=4
)
result = polypus.train(
    qc,
    polypus.External(from_nevergrad(optimizer), max_evaluations=200),
    id="nevergrad",
    **common,
)
```

**pycma.** `from_pycma(es)` asks a `cma.CMAEvolutionStrategy` for one generation per batch (`es.ask()`, `popsize` candidates) and tells it `-value` (`es.tell`). The run ends when `es.stop()` reports one of the strategy's termination criteria, such as `maxfevals`, or at `max_evaluations`. The start is the strategy's `x0` and its step size `sigma0`:

```python
import cma

es = cma.CMAEvolutionStrategy(
    6 * [math.pi], 0.5, {"seed": 7, "popsize": 8, "maxfevals": 200}
)
result = polypus.train(
    qc,
    polypus.External(from_pycma(es), max_evaluations=200),
    id="pycma",
    **common,
)
```

**An optimizer of your own** needs only the two methods. This one samples around the best point it has been told, and halves its step after a batch that found nothing better:

```python
class LocalSearch:
    def __init__(self, dimensions, size=4, step=0.5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.best = self.rng.uniform(0, 2 * math.pi, dimensions)
        self.best_value = -math.inf
        self.size = size
        self.step = step

    def ask(self):
        if self.step < 0.01:
            return None  # done: the run stops with converged=True
        noise = self.rng.normal(size=(self.size, len(self.best)))
        return self.best + self.step * noise  # a 2-D NumPy array, one row per candidate

    def tell(self, candidates, values):
        i = int(np.argmax(values))
        if values[i] > self.best_value:
            self.best, self.best_value = np.array(candidates[i]), values[i]
        else:
            self.step /= 2


result = polypus.train(
    qc,
    polypus.External(LocalSearch(dimensions=6), max_evaluations=400),
    id="local-search",
    **common,
)
```

**When to prefer an external optimizer: a method Polypus does not have.** Bayesian optimization, such as Optuna's TPE, spends computation choosing each candidate from a model of the values measured so far, to need fewer evaluations; that suits a small budget or circuits that are expensive to run. Tibaldi, Vodola, Tignone and Ercolessi (2023) apply Bayesian optimization with a Gaussian-process model to QAOA's parameters; TPE is a different model. Neither `polypus.External` nor any library is benchmarked: [Choosing an optimizer](#choosing-an-optimizer) has no row for them, so measure on your own problem before choosing one. Between two batches the QPUs wait for the optimizer: for `ask()` and `tell()`, each a Python call, and for the library's own computation, which in a model-based method grows with the candidates it has seen. With fast circuits that can take most of the run. A batch of `n_qpus` candidates, or a multiple of it, fills every QPU in whole rounds: set `batch_size` (Optuna), `num_workers` (Nevergrad) or `popsize` (pycma) accordingly.

**References.**

- S. Tibaldi, D. Vodola, E. Tignone and E. Ercolessi, "Bayesian Optimization for QAOA", IEEE Transactions on Quantum Engineering 4, 1–11 (2023), doi:10.1109/TQE.2023.3325167, arXiv:2209.03824.

## Starting point

By default every optimizer starts from random parameters. `initial_point` makes it start from a point you choose, typically the `best_params` of an earlier run. It takes `dimensions` finite numbers as a list, tuple or NumPy array; another length, a `NaN` or an infinity raises `ValueError` before anything runs. `qml.train` takes it too, as a keyword argument.

| Optimizer | What it does with `initial_point` |
|---|---|
| `DE`, every `strategy` | Makes it, wrapped into `[0, 2π)`, the first member of the initial population, so it is the first candidate evaluated. The other members are drawn as without it |
| `PSO` | Makes it the position of the first particle, the first candidate evaluated; the other particles are drawn as without it. Every value must lie within `bounds`, since positions are clamped to them, otherwise `ValueError` before anything runs |
| `QNG` | Starts from it: the first batch estimates the gradient around it, without evaluating the point itself. It may lie outside `bounds`, which only limit the random start |
| `CMAES` | Makes it, wrapped into `[0, 2π)`, the mean of the first run: the first generation is sampled around it with step size `sigma0` |
| `Adam` | Starts from it and evaluates it first: the first batch is the point, then the finite-difference stencil that estimates the gradient around it. It may lie outside `bounds`, which only limit the random start |
| `SPSA` | Starts from it: the calibration batch is centred on it, and the first iteration measures it first. It is taken as given, not wrapped into `[0, 2π)` |
| `Rotosolve` | Starts from it and measures it first: the first batch is the point, then its two neighbours at `±π/2` along the first parameter. It is taken as given; a parameter is wrapped into `[0, 2π)` when a step moves it |
| `External` | Refuses it with `ValueError` before anything runs: the library chooses its start. Pass the point to `from_optuna` or `from_nevergrad` as their `initial_point`, or as the `x0` of pycma's `CMAEvolutionStrategy` (see [External optimizers](#external-optimizers)) |

Without `initial_point`, every optimizer starts as before and a seeded run reproduces exactly. With one, a run reproduces with the same seed and the same point; the result does not echo the point, so keep it with the seed.

CMA-ES keeps its `sigma0`. With the default `π/2`, two standard deviations already span the whole circle, so the first generation explores around the point rather than refining it; to refine around a point, pass a smaller `sigma0` too, such as `0.3`. IPOP restarts start from a new random mean, not from the point.

CMA-ES and QNG never evaluate the point itself. CMA-ES samples around it; QNG estimates the gradient around it, and the first value it keeps is that of the point after one step. Their `best_fitness` can therefore come out below the one the point had in the earlier run.

PSO's default `bounds` are `(-π, π)`, while DE and CMA-ES return points in `[0, 2π)`, so starting PSO from their `best_params` raises `ValueError` as soon as a value passes π. Adam and SPSA never wrap their parameters, and Rotosolve measures its neighbours unwrapped, so their `best_params` can lie outside both ranges. Pass bounds that contain the point, such as `polypus.PSO(bounds=(0, 2 * math.pi))`, or move every value into `[-π, π)` first by adding or subtracting a multiple of 2π: the angles are the same, as DE and CMA-ES assume when they wrap.

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
# Refine: Adam from L-SHADE's best point, 2 × dimensions + 1 circuits an iteration
# and one for the point it ends at.
refine = polypus.train(
    qc,
    polypus.Adam(max_iters=(budget // 2 - 1) // (2 * dimensions + 1), tolerance=0.0),
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
        polypus.Adam(
            max_iters=(budget // 3 - 1) // (2 * dimensions + 1), tolerance=0.0
        ),
        **common,
        seed=seed,
    )
    for seed in (1, 2, 3)
]
best = max(runs, key=lambda run: run.best_fitness)
```

In the benchmark, three runs raised the worst instance at p = 5, where single runs sometimes ended in a poor optimum, but their mean there was within noise of one run's. With 500 evaluations and on the HEA they did worse, likely because a third of the budget was too short for each run, and starting from L-SHADE had the higher mean in every scenario.

## Final re-evaluation

Under shot noise, `best_fitness` overstates the value of `best_params`. An optimizer keeps the best of many noisy estimates, and the best estimate is more often one that came out high by chance: the winner's curse. With 256 shots it reached 1.9 % of the maximum cut in the [benchmark](#choosing-an-optimizer), with local optimizers as well as population-based ones. `final_shots` reduces it: there, with `final_shots=4096`, the bias of `final_fitness` was at most 0.15 % on average. Once the optimizer has stopped, its best candidates are evaluated once more, in one batch with `final_shots` shots per circuit, and the result goes to two fields of its own:

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
| `DE` (every `strategy`), `PSO`, `CMAES`, `External` | `best_params` and the four best other distinct candidates the run scored |
| `QNG`, `Adam`, `SPSA`, `Rotosolve` | `best_params` |

`final_params` is the candidate with the highest re-evaluated value, `best_params` on a tie. Re-evaluating the best few candidates of a population, as Novák et al. recommend, lets the run pick one whose value with many shots beats `best_params`, which may have come first only by its noise. `final_fitness` is then the highest of a few fresh estimates, so it is still slightly optimistic, though far less than `best_fitness`, the highest of the whole run; when `best_params` is re-evaluated alone, it has no bias. `best_params`, `best_fitness` and `fitness_history` remain what the optimizer found: a seeded run sends the same batches with and without `final_shots`, and then the final batch. Without it, `final_params` and `final_fitness` are `None`.

`final_shots` must be an integer `>= 1`, checked before anything runs: `0` raises `ValueError`, and a value that is not an integer, such as `4096.0` or `True`, raises `TypeError`, as it does for `shots`. `qml.train` takes it as a keyword argument and runs every training circuit of each candidate with those shots. The batch holds at most five candidates, so it costs up to `5 × final_shots` shots per training circuit. It is sent once the optimizer has stopped, whether by its budget or its early stop, but not after an error or Ctrl+C. It is sampled with a seed of its own, derived from the run's, so its values are independent of the training's and a seeded run still reproduces them. With `infrastructure="cunqa"`, the walltime warning does not count it, so leave time for it in `walltime`; if the QPUs are released first, the batch fails like any other and `train` raises. The run logs `best_params` and `best_fitness` before the batch (at `INFO`) and again as a warning if it fails, so the optimizer's result can still be read from the log.

**References.**

- V. Novák, S. Illésová, T. Bezděk, I. Zelinka and M. Beseda, "Reliable Optimization Under Noise in Quantum Variational Algorithms", arXiv:2511.08289 (2025).

## Choosing an optimizer

[`benchmarks/bench_optimizers.py`](../../benchmarks/bench_optimizers.py) trains two circuits for MaxCut on random 3-regular graphs of 10 nodes, with each optimizer at the same evaluation budget: QAOA with depth p = 3 or 5 (6 or 10 parameters), and a hardware-efficient ansatz (HEA) of three layers of RY on every qubit with a linear chain of CZ between them (Qiskit's `TwoLocal` with `ry`, `cz`, `entanglement="linear"` and `reps=2`), 30 parameters. The 30 instances are paired: each is one graph, run by every optimizer with the same seed. The score is the exact approximation ratio of the returned `best_params`, the expected cut from the statevector divided by the maximum cut, so shot noise in `best_fitness` does not flatter it. DE uses a population of 30, and early stopping is off except for CMA-ES with restarts, whose restarts need it. Adam uses its defaults on both circuits, which the review that proposed it (#395) chose on separate instances. Momentum uses a learning rate tuned for each circuit on separate instances, 0.03 for QAOA and 0.1 for the HEA, with a `finite_difference_step` of 0.4. "L-SHADE, then Adam" gives half the budget to L-SHADE and the rest to Adam, started from L-SHADE's `best_params`; "Adam ×3" gives a third to each of three Adam runs with different seeds and keeps the one with the best `best_fitness`. SPSA runs with one perturbation per iteration and with eight (`resamplings=8`), both with the settings tuned with one perturbation on separate instances, so their settings differ only in the perturbations, though with eight the averaged estimate is smaller on average and the steps usually shorter: `perturbation=0.2` and `target_step=1.2` on QAOA, the defaults (0.1 and 0.63) on the HEA; each makes as many iterations as fit the budget after its calibration. Rotosolve runs with `sweeps=75`, the most that fit the budget (75 sweeps of 30 steps and its final point, 4951 evaluations), and `tolerance=0`, where its defaults are 10 sweeps and 0.01, and re-measures the parameters every fifth step, its default `reevaluate_every=5`; it runs on the HEA only: QAOA's γ and β are each the angle of many gates, so Rotosolve does not apply to it (see [Rotosolve](#rotosolve)), and its QAOA cells read "—". The circuits run on Polypus's native simulator, through a meter that counts every evaluation and stops each optimizer at the budget.

Paired difference in approximation ratio from the classic DE (`rand1bin`), with its standard error, and the share of instances where the optimizer did better; the best mean of each scenario is in bold:

| Optimizer | p = 3, 256 shots, 1500 evals | p = 3, 2^20 shots, 1500 evals | p = 5, 256 shots, 3000 evals | p = 3, 256 shots, 500 evals | HEA, 256 shots, 5000 evals |
|---|---|---|---|---|---|
| `DE()` (ratio) | 0.862 | 0.860 | 0.871 | 0.843 | 0.783 |
| `DE(strategy="shade")` | +0.001 ± 0.003 (50 %) | +0.006 ± 0.003 (67 %) | +0.015 ± 0.003 (80 %) | +0.006 ± 0.006 (57 %) | +0.106 ± 0.007 (100 %) |
| `DE(strategy="lshade")` | +0.017 ± 0.003 (90 %) | +0.030 ± 0.004 (97 %) | +0.037 ± 0.004 (93 %) | +0.008 ± 0.005 (65 %) | +0.153 ± 0.007 (100 %) |
| `CMAES()` | +0.020 ± 0.004 (77 %) | +0.037 ± 0.004 (97 %) | +0.038 ± 0.006 (90 %) | +0.005 ± 0.006 (53 %) | +0.170 ± 0.017 (90 %) |
| `CMAES(population_size=30)` | +0.011 ± 0.005 (60 %) | +0.010 ± 0.004 (67 %) | +0.010 ± 0.006 (53 %) | −0.000 ± 0.004 (43 %) | +0.133 ± 0.019 (80 %) |
| `CMAES(restarts=3)` | +0.004 ± 0.004 (60 %) | +0.009 ± 0.006 (60 %) | −0.002 ± 0.004 (43 %) | +0.001 ± 0.005 (53 %) | +0.015 ± 0.014 (53 %) |
| `Adam()` | +0.031 ± 0.004 (90 %) | +0.032 ± 0.004 (93 %) | +0.050 ± 0.011 (93 %) | +0.036 ± 0.003 (100 %) | +0.197 ± 0.009 (100 %) |
| `Adam(rule="momentum")`, tuned rate | +0.033 ± 0.003 (97 %) | +0.038 ± 0.004 (87 %) | +0.066 ± 0.004 (100 %) | +0.039 ± 0.004 (100 %) | +0.201 ± 0.009 (100 %) |
| L-SHADE, then `Adam()` | **+0.043 ± 0.004 (97 %)** | **+0.039 ± 0.004 (97 %)** | **+0.071 ± 0.004 (100 %)** | **+0.045 ± 0.005 (97 %)** | **+0.214 ± 0.007 (100 %)** |
| `Adam()` ×3 | +0.030 ± 0.004 (87 %) | +0.036 ± 0.004 (97 %) | +0.064 ± 0.004 (100 %) | +0.025 ± 0.005 (87 %) | +0.177 ± 0.008 (100 %) |
| `SPSA()`, tuned steps | +0.032 ± 0.005 (87 %) | +0.030 ± 0.009 (90 %) | +0.039 ± 0.011 (83 %) | +0.009 ± 0.009 (63 %) | +0.194 ± 0.011 (100 %) |
| `SPSA(resamplings=8)`, the same settings | +0.009 ± 0.008 (77 %) | +0.008 ± 0.012 (77 %) | +0.032 ± 0.011 (93 %) | −0.035 ± 0.017 (43 %) | +0.190 ± 0.011 (100 %) |
| `Rotosolve()`, HEA only | — | — | — | — | +0.192 ± 0.010 (100 %) |

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
| L-SHADE, then `Adam()` | 0.852 | 0.850 | 0.918 | 0.799 | 0.991 |
| `Adam()` ×3 | 0.818 | 0.823 | 0.898 | 0.776 | 0.835 |
| `SPSA()`, tuned steps | 0.782 | 0.671 | 0.666 | 0.701 | 0.845 |
| `SPSA(resamplings=8)`, the same settings | 0.689 | 0.647 | 0.668 | 0.563 | 0.841 |
| `Rotosolve()`, HEA only | — | — | — | — | 0.732 |

Sequential batches each optimizer sent, the rounds that cost time when the QPUs outnumber a batch, and the winner's curse: how far the reported `best_fitness` overstated the exact value of `best_params`, as a share of the maximum cut, with 256 shots. An Adam batch holds `2 × dimensions + 1` circuits, 13, 21 and 61 here, and its last holds the one circuit of the point it ends at. An SPSA batch holds 2 or 16 circuits, plus the parameters every 20 iterations, after a calibration batch of 50 and before the one circuit of its final parameters. A Rotosolve batch holds 2 circuits, plus the parameters every fifth step, and its last the one circuit of its final point.

| Optimizer | Batches, 1500 evals | Batches, 3000 evals | Batches, HEA | Bias, p = 3 | Bias, p = 5 | Bias, HEA |
|---|---|---|---|---|---|---|
| `DE()` | 50 | 100 | 166 | +0.003 | +0.004 | +0.006 |
| `DE(strategy="shade")` | 50 | 100 | 166 | +0.007 | +0.008 | +0.011 |
| `DE(strategy="lshade")` | 115 | 232 | 385 | +0.009 | +0.011 | +0.014 |
| `CMAES()` | 166 | 300 | 357 | +0.019 | +0.015 | +0.004 |
| `CMAES(population_size=30)` | 50 | 100 | 166 | +0.008 | +0.008 | +0.004 |
| `CMAES(restarts=3)` | 90 | 115 | 116 | +0.008 | +0.002 | +0.007 |
| `Adam()` | 116 | 143 | 82 | +0.015 | +0.012 | +0.002 |
| `Adam(rule="momentum")`, tuned rate | 116 | 143 | 82 | +0.013 | +0.011 | +0.002 |
| L-SHADE, then `Adam()` | 115 | 187 | 234 | +0.013 | +0.011 | +0.002 |
| `Adam()` ×3 | 118 | 145 | 84 | +0.011 | +0.009 | +0.004 |
| `SPSA()`, tuned steps | 708 | 1440 | 2416 | +0.010 | +0.007 | +0.001 |
| `SPSA(resamplings=8)`, the same settings | 92 | 185 | 310 | +0.004 | +0.005 | +0.001 |
| `Rotosolve()`, HEA only | — | — | 2251 | — | — | +0.002 |

What the data says:

- **Explore with L-SHADE, then refine with Adam.** Half the budget to `DE(strategy="lshade")` and the other half to `Adam()` from its `best_params` ([Starting point](#starting-point)) gave the best mean ratio in all five scenarios, 500 evaluations included, and beat the classic DE on 97 to 100 % of the instances. Its lowest ratio was the highest of all the optimizers in four scenarios (0.918 with p = 5, 0.991 on the HEA), and its Adam phase improved on L-SHADE's half-budget result on every instance. Its lead over Adam alone was 0.007 to 0.021, more than two standard errors with 256 shots at p = 3 and 1500 evaluations (+0.012 ± 0.004) and on the HEA (+0.016 ± 0.007), and about two or fewer elsewhere. On the HEA the two were even on most instances, and the lead came from the few where Adam alone ended low.
- **Adam alone, with its defaults, is a strong single run with one configuration for both circuits.** It beat L-SHADE on average in every scenario: clearly with 256 shots at p = 3 (+0.014 ± 0.004 at 1500 evaluations, +0.028 ± 0.004 at 500) and on the HEA (+0.044 ± 0.007), within noise with 2^20 shots (+0.003 ± 0.004) and at p = 5 (+0.013 ± 0.011). As a local method it occasionally ends in a poor optimum: its lowest ratio was 0.661 with p = 5 and 0.784 with 2^20 shots, against 0.830 and 0.831 for L-SHADE. Its batches grow with the parameters, so with 30 it sent the fewest of all (82, against 385 for L-SHADE).
- **Use momentum only with a learning rate tuned to the problem.** With rates tuned on separate instances it matched Adam within noise (0.003 to 0.016 higher), and its lowest ratio at p = 5 was 0.861, against Adam's 0.661. Its rate depends on the scale of the cost (see [Adam](#adam)), so these figures are its best case. Adam ran with one configuration, its defaults, which the review that proposed it (#395) chose on separate instances of these problem families; DE and CMA-ES ran with their defaults, untuned.
- **Don't restart Adam by default.** Three runs on a third of the budget each raised the worst instance at p = 5 (lowest ratio 0.898, against 0.661 for one run), but their mean was within noise of one run's there (+0.014 ± 0.011), with 2^20 shots (+0.004 ± 0.003) and with 256 shots at p = 3 and 1500 evaluations (−0.001 ± 0.003). They were behind one run at 500 evaluations (−0.010 ± 0.003) and on the HEA (−0.020 ± 0.007), likely because a third of the budget was too short for each run. Starting from L-SHADE had the higher mean in every scenario.
- **SPSA with one perturbation per iteration was within noise of Adam once its steps were tuned.** With 1500 evaluations or more its paired difference from Adam was within noise in every scenario (+0.001 ± 0.005 at p = 3 with 256 shots, −0.003 ± 0.008 with 2^20, −0.011 ± 0.015 at p = 5, −0.003 ± 0.011 on the HEA), and it beat the classic DE on 83 to 100 % of the instances. Like Adam it sometimes ended in a poor optimum, with a lowest ratio of 0.671 with 2^20 shots and 0.666 at p = 5. On the 30-parameter HEA both SPSA variants ended level with Adam (0.977 and 0.973, against 0.981) and about 0.02 below L-SHADE then Adam. Every optimizer here has the same evaluations and is scored at the end, so the benchmark does not show whether SPSA's cheaper steps reach a good ratio sooner with many parameters.
- **SPSA needs budget.** With 500 evaluations, after its 50-circuit calibration, one perturbation per iteration gained only +0.009 ± 0.009 over the classic DE, 0.027 ± 0.009 below Adam, and eight perturbations per iteration, 27 iterations in all, ended below the classic DE (−0.035 ± 0.017, about two standard errors), with a lowest ratio of 0.563; as with `resamplings=8` below, its steps were also shorter.
- **`resamplings=8` cuts SPSA's sequential batches about eightfold, at a cost in quality.** With the same settings as one perturbation per iteration, it sent 92 batches at 1500 evaluations instead of 708 (116 for Adam), each of 16 or 17 circuits, a single round on 17 QPUs. Its mean was below one perturbation's with 256 shots at p = 3 (−0.022 ± 0.006 at 1500 evaluations, ahead on 27 % of the instances; −0.043 ± 0.012 at 500) and within noise of it elsewhere (−0.021 ± 0.014 with 2^20 shots, −0.007 ± 0.016 at p = 5, −0.005 ± 0.012 on the HEA), never above it. It also took shorter steps on average, since the average of eight estimates is smaller on average than one estimate, so the comparison mixes eight times fewer steps with shorter ones, and does not show what averaging alone costs or gains. Its lowest ratio at p = 3 was 0.689 with 256 shots and 0.647 with 2^20. Use it when sequential rounds, not evaluations, limit the run.
- **Rotosolve, with no step size to tune, ended level with Adam and SPSA on the HEA.** Its mean ratio was 0.976, within noise of Adam's (−0.005 ± 0.013), momentum's (−0.009 ± 0.013) and SPSA's (−0.002 ± 0.013), above L-SHADE's (+0.039 ± 0.012, ahead on 26 of 30 instances), and it beat the classic DE on every instance. It was 0.021 ± 0.011 below L-SHADE then Adam, about two standard errors: on 25 instances it ended within 0.01 of it, at 0.995 or more, and on the other 5 it stopped in a poorer optimum, between 0.732 and 0.916. Adam and SPSA each ended below 0.99 on 6 instances; Rotosolve's lowest ratio, 0.732, was below theirs (0.844 and 0.845). Its batches of 2 or 3 circuits took 2251 sequential rounds, about as many as SPSA with one perturbation (2416) and against 82 for Adam. It does not apply to QAOA.
- **CMA-ES with its default population is competitive with L-SHADE.** Its paired difference from L-SHADE was within two standard errors in every scenario: about even with 256 shots (+0.003 ± 0.005 at p = 3, +0.001 ± 0.006 at p = 5), slightly ahead with 2^20 shots (+0.007 ± 0.004) and on the HEA (+0.017 ± 0.016). It was less consistent on the HEA, beating the classic DE on 90 % of the instances where L-SHADE beat it on all, with a lowest ratio of 0.740 against 0.834. On QAOA it sent the most sequential batches of the population-based optimizers and, at 1500 and 3000 evaluations, had the largest winner's curse.
- **Don't enlarge CMA-ES's population to fill QPUs, and don't count on IPOP restarts.** With a population of 30, CMA-ES's mean ratio was below its default population's in every scenario, by 0.005 to 0.037, though still about 0.01 above the classic DE's at 1500 and 3000 evaluations. IPOP restarts stayed within 0.01 of the classic DE on QAOA and gained +0.015 ± 0.014 on the HEA, against +0.170 without them.
- **With many QPUs, when sequential rounds dominate the run time,** `strategy="shade"` keeps the classic DE's batch size and number of rounds and still gained 1.5 points with 10 parameters and 10.6 with 30, though almost nothing with 6 at 256 shots. L-SHADE needs about 2.3 times the rounds of the classic DE for its larger gain.
- **With 500 evaluations only the Adam-based optimizers were clearly better than the classic DE,** by 0.025 to 0.045; the DE strategies and CMA-ES gained 0.008 at most.
- **The reported `best_fitness` is optimistic under shot noise,** because it is the best of many noisy estimates: by up to 1.9 % of the maximum cut with 256 shots, for CMA-ES at p = 3. Adam's `best_params` is the best point it measured, so Adam has the same bias: 1.0 to 1.5 % on QAOA with 256 shots, a little more than L-SHADE's and more than the classic DE's, and 0.2 % on the HEA. SPSA measures its parameters only every 20 iterations, so it picks from fewer estimates: 0.4 to 1.0 % on QAOA and 0.1 % on the HEA. Rotosolve's `best_fitness` is the best value it measured, never a predicted one, so it has the same bias: 0.2 % on the HEA. Pass `final_shots` when the value itself matters ([Final re-evaluation](#final-re-evaluation)): with `final_shots=4096`, 16 times the training shots, `final_fitness` was within 0.15 % of the exact value of `final_params` on average for every optimizer in every scenario, and within 0.6 % in every run. For Adam and SPSA it is a fresh estimate of `best_params`, without bias: its mean error was 0.07 % at most in every scenario. For DE and CMA-ES it is the highest of five noisy values, so it stays slightly optimistic, by up to 0.15 % (CMA-ES at p = 3 with 1500 evaluations). Their `final_params` was another candidate than `best_params` in 6 to 24 runs of 30, and a better one in 154 of those 179 runs (worse by 0.0033 at most otherwise); their mean ratio rose in every scenario, by +0.0010 ± 0.0006 (the classic DE at p = 3 with 1500 evaluations) to +0.0050 ± 0.0010 (L-SHADE on the HEA).

These results cover MaxCut only, on 10 nodes, with QAOA at depth 3 to 5 and one 30-parameter hardware-efficient ansatz, on a noiseless simulator with shot sampling. Other problems, more parameters or hardware noise may rank the optimizers differently, and momentum's learning rates and SPSA's steps were tuned for these two circuits. CMA-ES's, SPSA's and Rotosolve's rows may differ on another platform or build; see the reproducibility notes under [CMA-ES](#cma-es), [SPSA](#spsa) and [Rotosolve](#rotosolve). Run `python benchmarks/bench_optimizers.py --full`, or a scenario of your own with `--scenario [hea:]LAYERS:SHOTS:BUDGET`, to check. The tables come from that command at commit `d08f4e3`, after Polypus 0.7.2, which has no Adam, on Linux x86_64 with an Intel Xeon at 2.10 GHz (4 CPUs), and the SPSA rows from `--full --only rand1bin lshade adam spsa spsa-k8` at commit `0973cf8` on the same platform, whose `rand1bin`, `lshade` and `adam` runs were identical to the first run's. The Rotosolve rows come from `--scenario hea:2:256:5000 --instances 30 --only rand1bin lshade adam momentum lshade-adam spsa rotosolve` at commit `d07bd73` on the same platform, whose runs of the other optimizers were identical to those runs'. The `final_shots` figures come from `--full --scenario 3:256:1500 5:256:3000 3:256:500 hea:2:256:5000 --only rand1bin lshade cmaes adam spsa --final-shots 4096` at commit `4a06b46` on the same platform, whose ratios, biases and batches were identical to those runs'. The 500-evaluation scenario gives a population of 30 only 480 evaluations, 16 batches of 30. All these runs predate the DE changes of #410 (one coordinate always from the mutant in `rand1bin`, differences measured around the circle in every strategy), so the DE rows and every paired difference, all measured against `DE()`, will move when they are rerun. The command prints every figure quoted here: for each scenario, each optimizer's ratio, lowest ratio, paired difference from the classic DE, bias and batches, its paired differences from L-SHADE, from Adam and from SPSA, and how much the Adam stage of "L-SHADE, then Adam" changed its L-SHADE stage's ratio; with `--final-shots`, also the ratio of `final_params`, the bias of `final_fitness` on average and at its largest, the paired change of ratio from `best_params` to `final_params`, and how many runs returned another candidate, how many of those it was better in, and the worst change among them. `--json FILE` also writes the per-instance results, from which Rotosolve's differences from momentum and from L-SHADE then Adam, and its counts of instances, were computed.
