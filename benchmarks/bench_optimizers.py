#!/usr/bin/env python3
"""
Optimizer benchmark: Polypus's DE strategies, CMA-ES, Adam and SPSA on MaxCut.

Each optimizer trains the same circuits through ``polypus.train`` with the
same evaluation budget, and is scored by the *exact* approximation ratio of the
parameters it returns: the expected cut of ``best_params`` from the native
statevector, divided by the maximum cut. Instances are paired: instance ``s``
is one random 3-regular graph, the same for every optimizer, and every
optimizer runs it with seed ``s``.

Two circuits train on each graph, both measured on the same cut observable:

  qaoa          QAOA with p layers: |+>^n, then p times exp(-iγC) and RX(2β)
                on every qubit, 2p parameters
  hea           a hardware-efficient ansatz with ``reps`` repetitions: |0>^n,
                then reps + 1 layers of RY on every qubit with a linear chain
                of CZ between consecutive layers (Qiskit's TwoLocal ry/cz with
                entanglement="linear"),
                n (reps + 1) parameters, each in exactly one RY; 30 with
                10 nodes and reps = 2

Optimizers compared (``--de-population`` sets ``P``, 30 by default; ``d`` is
the number of parameters):

  rand1bin      polypus.DE(population_size=P), the classic DE (the baseline)
  shade         polypus.DE(population_size=P, strategy="shade")
  lshade        polypus.DE(population_size=P, strategy="lshade")
  cmaes         polypus.CMAES(), Hansen's default population 4 + floor(3 ln d)
  cmaes-P       polypus.CMAES(population_size=P), DE's batch size
  cmaes-ipop    polypus.CMAES(restarts=R), IPOP restarts doubling the
                population (``--restarts``, 3 by default)
  adam          polypus.Adam() with its defaults: rule "adam", learning rate
                0.1, finite-difference step 0.2
  momentum      polypus.Adam(rule="momentum"), learning rate 0.03 on QAOA and
                0.1 on the HEA, finite-difference step 0.4 on both
  lshade-adam   L-SHADE on half the budget, then polypus.Adam() from its
                ``best_params`` (``initial_point``) with the evaluations left
  adam-x3       three polypus.Adam() runs with different seeds sharing the
                budget; the one with the best reported ``best_fitness`` wins
  spsa          polypus.SPSA(), one perturbation per iteration: perturbation
                0.2 and target step 1.2 on QAOA, 0.1 and 0.63 (the defaults)
                on the HEA
  spsa-k8       polypus.SPSA(resamplings=8), eight perturbations per
                iteration, with spsa's perturbation and target step on each
                problem, so the two differ only in the perturbations

Momentum's learning rate depends on the scale of the cost, so its rate and
step are tuned per problem: they are the values the review that proposed Adam
(#395) chose on reserved instances 100-109, never on the test instances
0-29. Adam uses one configuration, its defaults, for both problems. SPSA's
perturbation and target step are likewise the values the review that
proposed SPSA (#396) chose on instances 100-109.

Early stopping is off (``tolerance=0``) for all but ``cmaes-ipop``, so they
spend the whole budget; ``cmaes-ipop`` keeps the default ``tolerance=0.01`` and
``patience=20``, because its restarts start when a run stagnates.

Budgets. DE runs ``budget / P - 1`` generations after its initial population
(L-SHADE spends the same budget in more, smaller generations), CMA-ES
without restarts ``budget / λ`` generations, and Adam ``(budget − 1) / (2d + 1)``
iterations, each one batch of the current point and the central difference
around it, then one evaluation of the point its last step reached. SPSA runs
the most iterations that fit after its 50-circuit calibration: each is one
batch of ``2K`` circuits for ``K`` perturbations, plus the current point on the
first iteration and every 20th after it, and one evaluation measures its final
point (``spsa_iterations``). Every optimizer's budget is checked before the
first run, so a budget too small for one of them (``adam-x3``'s third must hold
an Adam iteration and its final point) is an error before anything runs. IPOP's
batch sizes depend on when its restarts happen, so no
generation count fixes its budget in advance. Instead every batch goes through
a meter that replaces the local seam
(``polypus_python.run_qcs``): it simulates each circuit on Polypus's native
statevector simulator (``polypus.statevector``), samples its shots with a
seeded NumPy RNG and counts the evaluations. A batch that would take an
optimizer past its budget is not simulated: its circuits all measure the empty
cut, so they never become the optimizer's best, and the run is scored on what
it found within the budget. All optimizers go through the same meter, so they
see the same shot noise model and are measured the same way. The fitness is
Polypus's native ``Qubo`` observable of the cut.

The chained optimizers make several ``polypus.train`` calls through one meter,
so the calls share the budget and their evaluations and batches add up.
``lshade-adam`` gives L-SHADE ``(budget / 2) / P - 1`` generations, DE's rule
for half the budget, and Adam the evaluations L-SHADE left; the run is scored
on Adam's result. ``adam-x3`` gives each run an equal share of the evaluations
still left (a third of the budget for the first), with seeds ``s``,
``s + 10^6`` and ``s + 2·10^6``.

Reported per optimizer, over the instances:

  ratio        mean exact approximation ratio of ``best_params``
  lowest       the lowest exact approximation ratio of the instances
  Δ vs DE      mean paired difference from ``rand1bin``, ± its standard error
  wins         share of instances where it beats ``rand1bin`` (ties count half)
  bias         mean (reported ``best_fitness`` − exact value) / max cut: the
               winner's curse of picking the best of many noisy estimates
  evals        mean evaluations (circuits) spent, within the budget
  batches      mean batches sent within the budget: the sequential rounds,
               which is what costs time when the QPUs outnumber a batch
  time         mean wall-clock seconds per run on this machine

A second table gives the paired differences from ``lshade``, from ``adam`` and
from ``spsa`` when they ran, and a last line how ``lshade-adam``'s Adam stage
changed the ratio of its L-SHADE stage.

``--final-shots N`` passes ``final_shots=N`` to every ``polypus.train``
call: once each optimizer has stopped, its best candidates (``best_params``
and, for DE, PSO and CMA-ES, four more) are re-evaluated in one batch with N
shots per circuit. The meter lets that batch through outside the budget,
with N shots and a sampling stream of its own, so every batch before it, and
so the ``best_params`` rows, are those of the run without it; it counts the
batch apart. N must differ from the scenario's shots, which is how the meter
tells the final batch. Two more columns are then reported:

  final ratio  mean exact approximation ratio of ``final_params``
  final bias   mean (``final_fitness`` − exact value of ``final_params``) /
               max cut

and every run in the JSON gets ``final_shots``, ``final_fitness``,
``final_ratio``, ``final_bias``, ``final_evaluations`` and ``final_batches``
(the final batch's circuits and calls) and ``final_is_best``
(``final_params == best_params``). A chained run reports the final values of
the result it is scored on and counts every stage's final batch. A last table
then compares ``final_params`` with ``best_params`` per optimizer: the mean
paired change in ratio, ± its standard error, the runs whose
``final_params`` is another candidate, how many of those it is better in,
the worst change, and the largest |final bias| of a run. The final values
are printed to four decimals, since a re-evaluation with many shots leaves
little bias to show.

Usage
-----
  python benchmarks/bench_optimizers.py                  # QAOA p=3, 256 shots, 1500 evals, 10 instances
  python benchmarks/bench_optimizers.py --full           # the five scenarios of the guide, 30 instances
  python benchmarks/bench_optimizers.py --quick          # seconds: a smoke run on 6 nodes
  python benchmarks/bench_optimizers.py --scenario 5:256:3000 --instances 20
  python benchmarks/bench_optimizers.py --scenario 3:exact:1500   # 2^20 shots
  python benchmarks/bench_optimizers.py --scenario hea:2:256:5000 # the 30-parameter HEA
  python benchmarks/bench_optimizers.py --only rand1bin adam lshade-adam --json results.json
  python benchmarks/bench_optimizers.py --only lshade cmaes adam --final-shots 4096

Every run is written to ``--json``; those of ``lshade-adam`` and ``adam-x3``
also list their ``stages``, the evaluations, batches, reported fitness and
exact ratio of each ``polypus.train`` call.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import polypus
    import polypus_python
except ImportError:
    print("[bench-optimizers] ERROR: polypus is not installed (maturin develop).")
    sys.exit(1)

REPO = Path(__file__).resolve().parents[1]
# The shots of the "exact" scenarios: the sampling error of the cut is then
# below 0.002 edges.
EXACT_SHOTS = 2**20
OPTIMIZERS = [
    "rand1bin",
    "shade",
    "lshade",
    "cmaes",
    "cmaes-P",
    "cmaes-ipop",
    "adam",
    "momentum",
    "lshade-adam",
    "adam-x3",
    "spsa",
    "spsa-k8",
]
BASELINE = "rand1bin"
PROBLEMS = ("qaoa", "hea")
# Momentum's (learning rate, finite-difference step) per problem, as tuned on
# the reserved instances 100-109 in the review that proposed Adam (#395).
MOMENTUM = {"qaoa": (0.03, 0.4), "hea": (0.1, 0.4)}
# adam-x3's runs use seeds s, s + RESTART_SEED_STEP, s + 2 RESTART_SEED_STEP.
RESTART_SEED_STEP = 1_000_000
# SPSA's perturbations per iteration, and its (perturbation, target step) per
# problem, as tuned on the reserved instances 100-109 in the review that
# proposed SPSA (#396) with one perturbation; spsa-k8 takes the same, so the
# two rows differ only in the perturbations.
SPSA_RESAMPLINGS = {"spsa": 1, "spsa-k8": 8}
SPSA = {"qaoa": (0.2, 1.2), "hea": (0.1, 0.63)}
# The circuits of SPSA's calibration batch, before its first iteration.
SPSA_CALIBRATION = 50


# ── Problem ──────────────────────────────────────────────────────────────────


def regular3_graph(n, rng):
    """A uniformly random simple 3-regular graph on ``n`` nodes (pairing
    model with rejection), as a sorted list of edges ``(a, b)``, ``a < b``."""
    if n % 2 or n < 4:
        raise ValueError("a 3-regular graph needs an even number of nodes >= 4")
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        edges = set()
        for a, b in stubs.reshape(-1, 2):
            a, b = int(min(a, b)), int(max(a, b))
            if a == b or (a, b) in edges:
                break
            edges.add((a, b))
        else:
            return sorted(edges)


class Instance:
    """MaxCut on one graph: the trainable circuit, its native cut observable,
    and the cut of every basis state for exact scoring. Qubit ``i`` is node
    ``i`` and variable ``i`` of the observable (the right-most bit of a
    bitstring).

    ``problem="qaoa"``: QAOA with ``layers`` = p, parameters
    ``[γ_0..γ_{p-1}, β_0..β_{p-1}]``. ``problem="hea"``: the hardware-efficient
    ansatz with ``layers`` = reps, parameter ``layer·n + q`` the RY angle of
    qubit ``q`` in RY layer ``layer`` (0 to reps)."""

    @staticmethod
    def parameters(problem, nodes, layers):
        """The circuit's number of parameters, without building it."""
        return 2 * layers if problem == "qaoa" else nodes * (layers + 1)

    def __init__(self, nodes, layers, seed, problem="qaoa"):
        if problem not in PROBLEMS:
            raise ValueError(f"unknown problem {problem!r}")
        self.nodes = nodes
        self.layers = layers
        self.problem = problem
        self.edges = regular3_graph(nodes, np.random.default_rng(1000 + seed))
        index = np.arange(2**nodes)
        bits = (index[:, None] >> np.arange(nodes)) & 1
        self.cuts = sum((bits[:, a] != bits[:, b]).astype(float) for a, b in self.edges)
        self.max_cut = float(self.cuts.max())
        degree = [0] * nodes
        for a, b in self.edges:
            degree[a] += 1
            degree[b] += 1
        # cut(x) = Σ_(a,b) x_a + x_b − 2 x_a x_b
        self.observable = polypus.Qubo(
            nodes,
            linear=[(i, float(d)) for i, d in enumerate(degree)],
            quadratic=[(a, b, -2.0) for a, b in self.edges],
        )
        qc = polypus.Circuit(nodes)
        self.dimensions = self.parameters(problem, nodes, layers)
        if problem == "qaoa":
            for q in range(nodes):
                qc.h(q)
            for k in range(layers):
                gamma, beta = polypus.Param(k), polypus.Param(layers + k)
                for a, b in self.edges:
                    qc.rzz(a, b, gamma)
                for q in range(nodes):
                    qc.rx(q, 2 * beta)
        else:
            for layer in range(layers + 1):
                if layer:
                    for q in range(nodes - 1):
                        qc.cz(q, q + 1)
                for q in range(nodes):
                    qc.ry(q, polypus.Param(layer * nodes + q))
        self.circuit = qc.measure_all()

    def expected_cut(self, params):
        state = polypus.statevector(self.circuit, list(params))
        return float(np.abs(state) ** 2 @ self.cuts)


# ── Evaluation meter (the local seam) ────────────────────────────────────────


class Meter:
    """Replacement for ``polypus_python.run_qcs``: simulates each circuit of
    a batch natively and samples the shots each call asks for, while the run
    is within its evaluation budget; past it, a batch measures the empty cut.

    With ``final_shots``, a call with those shots is a final batch: it is
    simulated outside the budget, counted apart, and sampled from a stream of
    its own, so the optimizer's batches sample exactly as without it."""

    def __init__(self, nodes, budget, seed, final_shots=None):
        self.nodes = nodes
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.final_shots = final_shots
        self.final_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.evaluations = 0
        self.batches = 0
        self.refused = 0
        self.final_evaluations = 0
        self.final_batches = 0
        self.keys = [format(k, f"0{nodes}b") for k in range(2**nodes)]

    def __call__(self, infrastructure, **kwargs):
        qcs, shots = kwargs["qcs"], kwargs["shots"]
        if self.final_shots is not None and shots == self.final_shots:
            self.final_evaluations += len(qcs)
            self.final_batches += 1
            return self.sample(qcs, shots, self.final_rng)
        if self.evaluations + len(qcs) > self.budget:
            self.refused += len(qcs)
            return [{"0" * self.nodes: shots} for _ in qcs]
        self.evaluations += len(qcs)
        self.batches += 1
        return self.sample(qcs, shots, self.rng)

    def sample(self, qcs, shots, rng):
        results = []
        for qasm in qcs:
            state = polypus.statevector(polypus.Circuit.from_qasm2(qasm))
            probabilities = np.abs(state) ** 2
            counts = rng.multinomial(shots, probabilities / probabilities.sum())
            results.append({self.keys[k]: int(c) for k, c in enumerate(counts) if c})
        return results


# ── Optimizers ───────────────────────────────────────────────────────────────


def default_lambda(dimensions):
    return 4 + math.floor(3 * math.log(dimensions))


def adam_iterations(dimensions, budget):
    """Adam's iterations within ``budget``: one batch of 2d + 1 each, then one
    evaluation of the final point."""
    iterations = (budget - 1) // (2 * dimensions + 1)
    if iterations < 1:
        raise ValueError(
            f"a budget of {budget} evaluations does not fit one Adam iteration "
            f"of {2 * dimensions + 1} and its final point"
        )
    return iterations


def spsa_evaluations(max_iters, resamplings, eval_interval=20):
    """The circuits an SPSA run sends: the calibration, then ``2K`` per
    iteration, plus the current point on iterations 1, 1 + eval_interval, …,
    and the final point after the last step."""
    measured = (max_iters - 1) // eval_interval + 1
    return SPSA_CALIBRATION + 2 * resamplings * max_iters + measured + 1


def spsa_iterations(budget, resamplings):
    """SPSA's iterations within ``budget``: the most whose run fits."""
    iterations = (budget - SPSA_CALIBRATION) // (2 * resamplings)
    while iterations >= 1 and spsa_evaluations(iterations, resamplings) > budget:
        iterations -= 1
    if iterations < 1:
        raise ValueError(
            f"a budget of {budget} evaluations does not fit SPSA's calibration, "
            f"one iteration of {2 * resamplings + 1} and its final point"
        )
    return iterations


def check_budget(name, dimensions, budget, population):
    """Raise ``ValueError`` when ``budget`` cannot run ``name``: checked for
    every optimizer before the first run, so that a small budget does not stop
    the benchmark half-way, its results unwritten. A chained optimizer's later
    stages get at least the share checked here: L-SHADE spends at most half
    the budget, and each Adam run at most its share."""
    try:
        if name in ("rand1bin", "shade", "lshade", "cmaes-P"):
            if budget < population:
                raise ValueError(
                    f"a budget of {budget} evaluations does not fit one "
                    f"population of {population}"
                )
        elif name in ("cmaes", "cmaes-ipop"):
            if budget < default_lambda(dimensions):
                raise ValueError(
                    f"a budget of {budget} evaluations does not fit one "
                    f"generation of {default_lambda(dimensions)}"
                )
        elif name in ("adam", "momentum"):
            adam_iterations(dimensions, budget)
        elif name == "lshade-adam":
            check_budget("lshade", dimensions, budget // 2, population)
            adam_iterations(dimensions, budget - budget // 2)
        elif name == "adam-x3":
            adam_iterations(dimensions, budget // 3)
        elif name in SPSA_RESAMPLINGS:
            spsa_iterations(budget, SPSA_RESAMPLINGS[name])
        else:
            raise ValueError("unknown optimizer")
    except ValueError as err:
        raise ValueError(f"{name}: {err}") from None


def make_method(name, dimensions, budget, population, restarts, problem="qaoa"):
    """The optimizer ``name`` configured for ``budget`` evaluations, the
    single-run ones only (``lshade-adam`` and ``adam-x3`` chain them)."""
    if name in ("rand1bin", "shade", "lshade"):
        return polypus.DE(
            generations=budget // population - 1,
            population_size=population,
            tolerance=0.0,
            strategy=name,
        )
    if name == "cmaes":
        return polypus.CMAES(
            generations=budget // default_lambda(dimensions), tolerance=0.0
        )
    if name == "cmaes-P":
        return polypus.CMAES(
            generations=budget // population, population_size=population, tolerance=0.0
        )
    if name == "cmaes-ipop":
        # Enough generations to spend the budget at the first population; the
        # meter stops the evaluations at the budget.
        return polypus.CMAES(
            generations=budget // default_lambda(dimensions), restarts=restarts
        )
    if name == "adam":
        return polypus.Adam(
            max_iters=adam_iterations(dimensions, budget), tolerance=0.0
        )
    if name == "momentum":
        learning_rate, step = MOMENTUM[problem]
        return polypus.Adam(
            max_iters=adam_iterations(dimensions, budget),
            learning_rate=learning_rate,
            tolerance=0.0,
            finite_difference_step=step,
            rule="momentum",
        )
    if name in SPSA_RESAMPLINGS:
        resamplings = SPSA_RESAMPLINGS[name]
        perturbation, target_step = SPSA[problem]
        return polypus.SPSA(
            max_iters=spsa_iterations(budget, resamplings),
            resamplings=resamplings,
            perturbation=perturbation,
            target_step=target_step,
        )
    raise ValueError(f"unknown optimizer {name!r}")


def train(instance, method, shots, seed, initial_point=None, final_shots=None):
    """One ``polypus.train`` run of ``method`` on ``instance``, through the
    local seam."""
    return polypus.train(
        instance.circuit,
        method,
        shots=shots,
        n_qpus=1,
        dimensions=instance.dimensions,
        expectation_function=instance.observable,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="bench_optimizers",
        backend="aer",
        seed=seed,
        initial_point=initial_point,
        final_shots=final_shots,
    )


def train_stage(instance, name, method, shots, seed, meter, stages, **kwargs):
    """``train``, recording in ``stages`` what the run spent and found."""
    evaluations, batches = meter.evaluations, meter.batches
    result = train(instance, method, shots, seed, **kwargs)
    stage = {
        "optimizer": name,
        "seed": seed,
        "evaluations": meter.evaluations - evaluations,
        "batches": meter.batches - batches,
        "iterations_run": result.iterations_run,
        "best_fitness": result.best_fitness,
        "ratio": instance.expected_cut(result.best_params) / instance.max_cut,
    }
    if result.final_params is not None:
        stage["final_fitness"] = result.final_fitness
        stage["final_ratio"] = (
            instance.expected_cut(result.final_params) / instance.max_cut
        )
    stages.append(stage)
    return result


def lshade_then_adam(
    instance, shots, budget, population, restarts, seed, meter, final_shots=None
):
    """L-SHADE explores with half the budget, then Adam refines its best
    point with the evaluations left. The run is Adam's."""
    stages = []
    d = instance.dimensions
    explore = train_stage(
        instance,
        "lshade",
        make_method("lshade", d, budget // 2, population, restarts),
        shots,
        seed,
        meter,
        stages,
        final_shots=final_shots,
    )
    refine = train_stage(
        instance,
        "adam",
        make_method("adam", d, budget - meter.evaluations, population, restarts),
        shots,
        seed,
        meter,
        stages,
        initial_point=explore.best_params,
        final_shots=final_shots,
    )
    return refine, stages


def adam_restarts(
    instance,
    shots,
    budget,
    population,
    restarts,
    seed,
    meter,
    final_shots=None,
    runs=3,
):
    """``runs`` Adam runs with different seeds, each with an equal share of
    the evaluations still left; the best reported ``best_fitness`` wins.

    The first run uses the instance's seed and the same meter seed as the
    ``adam`` row, so it repeats the start of that run on a shorter budget and
    the two rows stay paired."""
    stages, best = [], None
    for k in range(runs):
        share = (budget - meter.evaluations) // (runs - k)
        result = train_stage(
            instance,
            "adam",
            make_method("adam", instance.dimensions, share, population, restarts),
            shots,
            seed + RESTART_SEED_STEP * k,
            meter,
            stages,
            final_shots=final_shots,
        )
        if best is None or result.best_fitness > best.best_fitness:
            best = result
    return best, stages


CHAINED = {"lshade-adam": lshade_then_adam, "adam-x3": adam_restarts}


def run_one(
    instance, name, shots, budget, population, restarts, seed, final_shots=None
):
    if final_shots is not None and final_shots == shots:
        raise ValueError(
            f"final_shots ({final_shots}) must differ from the scenario's shots, "
            "which is how the meter tells the final batch"
        )
    meter = Meter(instance.nodes, budget, seed=10_000 + seed, final_shots=final_shots)
    polypus_python.run_qcs = meter
    start = time.perf_counter()
    if name in CHAINED:
        result, stages = CHAINED[name](
            instance,
            shots,
            budget,
            population,
            restarts,
            seed,
            meter,
            final_shots=final_shots,
        )
        iterations_run = sum(stage["iterations_run"] for stage in stages)
    else:
        method = make_method(
            name, instance.dimensions, budget, population, restarts, instance.problem
        )
        result = train(instance, method, shots, seed, final_shots=final_shots)
        stages = None
        iterations_run = result.iterations_run
    elapsed = time.perf_counter() - start
    if name != "cmaes-ipop" and meter.refused:
        raise AssertionError(f"{name} went past its budget: {meter.refused} refused")
    exact = instance.expected_cut(result.best_params)
    row = {
        "optimizer": name,
        "instance": seed,
        "ratio": exact / instance.max_cut,
        "bias": (result.best_fitness - exact) / instance.max_cut,
        "evaluations": meter.evaluations,
        "batches": meter.batches,
        "iterations_run": iterations_run,
        "seconds": elapsed,
    }
    if final_shots is not None:
        final_exact = instance.expected_cut(result.final_params)
        row.update(
            {
                "final_shots": final_shots,
                "final_fitness": result.final_fitness,
                "final_ratio": final_exact / instance.max_cut,
                "final_bias": (result.final_fitness - final_exact) / instance.max_cut,
                "final_evaluations": meter.final_evaluations,
                "final_batches": meter.final_batches,
                "final_is_best": result.final_params == result.best_params,
            }
        )
    if stages is not None:
        row["stages"] = stages
    return row


# ── Report ───────────────────────────────────────────────────────────────────


def paired(runs, reference):
    """The mean paired difference in ratio of ``runs`` from ``reference``
    (both by instance), ± its standard error, and the share of instances where
    ``runs`` did better (ties count half)."""
    diffs = [runs[s]["ratio"] - reference[s]["ratio"] for s in runs]
    share = sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs)
    return mean_and_error(diffs), f"{100 * share / len(diffs):.0f} %"


def mean_and_error(diffs, digits=3):
    """The mean of ``diffs`` ± its standard error, to ``digits`` decimals."""
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return f"{statistics.fmean(diffs):+.{digits}f} ± {se:.{digits}f}"


def summarise(rows, optimizers):
    by = {
        name: {r["instance"]: r for r in rows if r["optimizer"] == name}
        for name in optimizers
    }
    base = by.get(BASELINE)
    final = any("final_ratio" in r for r in rows)
    lines = [
        "| Optimizer | ratio | lowest | Δ vs rand1bin | wins | bias | evals | batches "
        "| time (s) |" + (" final ratio | final bias |" if final else ""),
        "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if final else ""),
    ]
    for name in optimizers:
        runs = list(by[name].values())
        mean = lambda key: statistics.fmean(r[key] for r in runs)  # noqa: E731
        lowest = min(r["ratio"] for r in runs)
        if base is None or name == BASELINE:
            delta, wins = "—", "—"
        else:
            delta, wins = paired(by[name], base)
        line = (
            f"| {name} | {mean('ratio'):.3f} | {lowest:.3f} | {delta} | {wins} "
            f"| {mean('bias'):+.3f} | {mean('evaluations'):.0f} "
            f"| {mean('batches'):.0f} | {mean('seconds'):.2f} |"
        )
        if final:
            line += f" {mean('final_ratio'):.3f} | {mean('final_bias'):+.4f} |"
        lines.append(line)
    references = [name for name in ("lshade", "adam", "spsa") if name in by]
    if references:
        lines += [
            "",
            "| Optimizer | "
            + " | ".join(f"Δ vs {name} | wins" for name in references)
            + " |",
            "|---|" + "---|---|" * len(references),
        ]
        for name in optimizers:
            cells = []
            for reference in references:
                cells += (
                    ["—", "—"]
                    if name == reference
                    else list(paired(by[name], by[reference]))
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if "lshade-adam" in by:
        gains = [
            run["stages"][1]["ratio"] - run["stages"][0]["ratio"]
            for run in by["lshade-adam"].values()
        ]
        lines += [
            "",
            f"lshade-adam: its Adam stage changed the ratio of its L-SHADE stage by "
            f"{statistics.fmean(gains):+.3f} on average ({min(gains):+.3f} at worst) "
            f"and raised it on {sum(g > 0 for g in gains)} of {len(gains)} instances.",
        ]
    if final:
        lines += [
            "",
            "| Optimizer | final − best ratio | other candidate | better | worst "
            "| largest final bias |",
            "|---|---|---|---|---|---|",
        ]
        for name in optimizers:
            runs = list(by[name].values())
            moved = [
                r["final_ratio"] - r["ratio"] for r in runs if not r["final_is_best"]
            ]
            changes = [r["final_ratio"] - r["ratio"] for r in runs]
            change = mean_and_error(changes, digits=4)
            worst = f"{min(moved):+.4f}" if moved else "—"
            largest = max(abs(r["final_bias"]) for r in runs)
            lines.append(
                f"| {name} | {change} | {len(moved)} of {len(runs)} "
                f"| {sum(m > 0 for m in moved)} | {worst} | {largest:.4f} |"
            )
    return "\n".join(lines)


def positive_int(text):
    """An integer >= 1, for ``--final-shots``."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {text!r}")
    return value


def parse_scenario(text):
    """``[PROBLEM:]LAYERS:SHOTS:BUDGET`` as ``(problem, layers, shots,
    budget)``; the problem is QAOA unless it says ``hea``."""
    fields = text.split(":")
    problem = fields.pop(0) if fields and fields[0] in PROBLEMS else "qaoa"
    try:
        layers, shots, budget = fields
        shots = EXACT_SHOTS if shots == "exact" else int(shots)
        return problem, int(layers), shots, int(budget)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected [qaoa:|hea:]LAYERS:SHOTS:BUDGET (SHOTS may be 'exact'), "
            f"got {text!r}"
        ) from None


def describe(problem, nodes, layers):
    """The scenario's circuit, for the report headings."""
    if problem == "qaoa":
        return f"QAOA p = {layers} ({2 * layers} parameters), n = {nodes}"
    return f"HEA reps = {layers} ({nodes * (layers + 1)} parameters), n = {nodes}"


def environment():
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "polypus": polypus.__version__,
        "commit": commit,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare Polypus's DE strategies, CMA-ES, Adam and SPSA on MaxCut.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        nargs="+",
        type=parse_scenario,
        default=None,
        metavar="[PROBLEM:]LAYERS:SHOTS:BUDGET",
        help="the circuit, 'qaoa' (the default, LAYERS = p) or 'hea' (LAYERS = "
        "reps), shots per circuit ('exact' for 2^20) and the evaluation budget "
        "(default: 3:256:1500).",
    )
    p.add_argument("--nodes", type=int, default=10, help="graph nodes (default: 10)")
    p.add_argument(
        "--instances", type=int, default=10, help="paired instances (default: 10)"
    )
    p.add_argument(
        "--first-instance", type=int, default=0, help="seed of the first instance"
    )
    p.add_argument(
        "--de-population",
        type=int,
        default=30,
        metavar="P",
        help="DE population, also cmaes-P's (default: 30)",
    )
    p.add_argument(
        "--restarts", type=int, default=3, help="cmaes-ipop's restarts (default: 3)"
    )
    p.add_argument(
        "--only", nargs="+", choices=OPTIMIZERS, default=None, help="optimizers to run"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="the guide's run: scenarios 3:256:1500 3:exact:1500 5:256:3000 "
        "3:256:500 hea:2:256:5000 with 30 instances",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: 6 nodes, scenario 1:64:120, 2 instances, P = 8",
    )
    p.add_argument(
        "--final-shots",
        type=positive_int,
        default=None,
        metavar="N",
        help="re-evaluate each run's best candidates with N shots once it stops "
        "(polypus.train's final_shots), outside the budget (default: off)",
    )
    p.add_argument("--json", type=Path, default=None, help="write every run here")
    args = p.parse_args(argv)
    if args.full:
        args.scenario = args.scenario or [
            ("qaoa", 3, 256, 1500),
            ("qaoa", 3, EXACT_SHOTS, 1500),
            ("qaoa", 5, 256, 3000),
            ("qaoa", 3, 256, 500),
            ("hea", 2, 256, 5000),
        ]
        args.instances = 30
    if args.quick:
        args.scenario = args.scenario or [("qaoa", 1, 64, 120)]
        args.nodes, args.instances, args.de_population = 6, 2, 8
    args.scenario = args.scenario or [("qaoa", 3, 256, 1500)]
    return args


def main(argv=None):
    args = parse_args(argv)
    optimizers = args.only or OPTIMIZERS
    env = environment()
    print("# Polypus optimizer benchmark: MaxCut on random 3-regular graphs\n")
    for key, value in env.items():
        print(f"- {key}: {value}")
    seeds = range(args.first_instance, args.first_instance + args.instances)
    problems = []
    for problem, layers, _, budget in args.scenario:
        dimensions = Instance.parameters(problem, args.nodes, layers)
        for name in optimizers:
            try:
                check_budget(name, dimensions, budget, args.de_population)
            except ValueError as err:
                problems.append(f"{problem}:{layers}, {budget} evaluations: {err}")
    if problems:
        print("[bench-optimizers] ERROR: " + "; ".join(problems), file=sys.stderr)
        sys.exit(2)
    report = {"environment": env, "scenarios": []}
    for problem, layers, shots, budget in args.scenario:
        shots_label = "2^20" if shots == EXACT_SHOTS else str(shots)
        print(
            f"\n## {describe(problem, args.nodes, layers)}, {shots_label} shots, "
            f"{budget} evaluations, {args.instances} instances\n",
            flush=True,
        )
        instances = [Instance(args.nodes, layers, s, problem) for s in seeds]
        rows = []
        for name in optimizers:
            for instance, seed in zip(instances, seeds):
                rows.append(
                    run_one(
                        instance,
                        name,
                        shots,
                        budget,
                        args.de_population,
                        args.restarts,
                        seed,
                        final_shots=args.final_shots,
                    )
                )
        print(summarise(rows, optimizers), flush=True)
        report["scenarios"].append(
            {
                "problem": problem,
                "nodes": args.nodes,
                "layers": layers,
                "parameters": instances[0].dimensions,
                "shots": shots,
                "budget": budget,
                "de_population": args.de_population,
                "restarts": args.restarts,
                "final_shots": args.final_shots,
                "runs": rows,
            }
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    return report


if __name__ == "__main__":
    main()
