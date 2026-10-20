#!/usr/bin/env python3
"""
Optimizer benchmark: Polypus's DE strategies and CMA-ES on QAOA MaxCut.

Each optimizer trains the same QAOA circuits through ``polypus.train`` with the
same evaluation budget, and is scored by the *exact* approximation ratio of the
parameters it returns: the expected cut of ``best_params`` from the native
statevector, divided by the maximum cut. Instances are paired: instance ``s``
is one random 3-regular graph, the same for every optimizer, and every
optimizer runs it with seed ``s``.

Optimizers compared (``--de-population`` sets ``P``, 30 by default):

  rand1bin      polypus.DE(population_size=P), the classic DE (the baseline)
  shade         polypus.DE(population_size=P, strategy="shade")
  lshade        polypus.DE(population_size=P, strategy="lshade")
  cmaes         polypus.CMAES(), Hansen's default population 4 + floor(3 ln d)
  cmaes-P       polypus.CMAES(population_size=P), DE's batch size
  cmaes-ipop    polypus.CMAES(restarts=R), IPOP restarts doubling the
                population (``--restarts``, 3 by default)

Early stopping is off (``tolerance=0``) for all but ``cmaes-ipop``, so they
spend the whole budget; ``cmaes-ipop`` keeps the default ``tolerance=0.01`` and
``patience=20``, because its restarts start when a run stagnates.

Budgets. DE runs ``budget / P - 1`` generations after its initial population
(L-SHADE spends the same budget in more, smaller generations), and CMA-ES
without restarts ``budget / λ`` generations. IPOP's batch sizes depend on when
its restarts happen, so no generation count fixes its budget in advance.
Instead every batch goes through a meter that replaces the local seam
(``polypus_python.run_qcs``): it simulates each circuit on Polypus's native
statevector simulator (``polypus.statevector``), samples its shots with a
seeded NumPy RNG and counts the evaluations. A batch that would take an
optimizer past its budget is not simulated: its circuits all measure the empty
cut, so they never become the optimizer's best, and the run is scored on what
it found within the budget. All optimizers go through the same meter, so they
see the same shot noise model and are measured the same way. The fitness is
Polypus's native ``Qubo`` observable of the cut.

Reported per optimizer, over the instances:

  ratio        mean exact approximation ratio of ``best_params``
  Δ vs DE      mean paired difference from ``rand1bin``, ± its standard error
  wins         share of instances where it beats ``rand1bin`` (ties count half)
  bias         mean (reported ``best_fitness`` − exact value) / max cut: the
               winner's curse of picking the best of many noisy estimates
  evals        mean evaluations (circuits) spent, within the budget
  batches      mean batches sent within the budget: the sequential rounds,
               which is what costs time when the QPUs outnumber a batch
  time         mean wall-clock seconds per run on this machine

Usage
-----
  python benchmarks/bench_optimizers.py                  # p=3, 256 shots, 1500 evals, 10 instances
  python benchmarks/bench_optimizers.py --full           # the four scenarios of the guide, 30 instances
  python benchmarks/bench_optimizers.py --quick          # seconds: a smoke run on 6 nodes
  python benchmarks/bench_optimizers.py --scenario 5:256:3000 --instances 20
  python benchmarks/bench_optimizers.py --scenario 3:exact:1500   # 2^20 shots
  python benchmarks/bench_optimizers.py --only rand1bin cmaes --json results.json
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
OPTIMIZERS = ["rand1bin", "shade", "lshade", "cmaes", "cmaes-P", "cmaes-ipop"]
BASELINE = "rand1bin"


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
    """QAOA for MaxCut on one graph: the trainable circuit (parameters
    ``[γ_0..γ_{p-1}, β_0..β_{p-1}]``), its native cut observable, and the cut
    of every basis state for exact scoring. Qubit ``i`` is node ``i`` and
    variable ``i`` of the observable (the right-most bit of a bitstring)."""

    def __init__(self, nodes, layers, seed):
        self.nodes = nodes
        self.layers = layers
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
        for q in range(nodes):
            qc.h(q)
        for k in range(layers):
            gamma, beta = polypus.Param(k), polypus.Param(layers + k)
            for a, b in self.edges:
                qc.rzz(a, b, gamma)
            for q in range(nodes):
                qc.rx(q, 2 * beta)
        self.circuit = qc.measure_all()

    def expected_cut(self, params):
        state = polypus.statevector(self.circuit, list(params))
        return float(np.abs(state) ** 2 @ self.cuts)


# ── Evaluation meter (the local seam) ────────────────────────────────────────


class Meter:
    """Replacement for ``polypus_python.run_qcs``: simulates each circuit of
    a batch natively and samples its shots, while the run is within its
    evaluation budget; past it, a batch measures the empty cut."""

    def __init__(self, nodes, budget, seed):
        self.nodes = nodes
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.batches = 0
        self.refused = 0
        self.keys = [format(k, f"0{nodes}b") for k in range(2**nodes)]

    def __call__(self, infrastructure, **kwargs):
        qcs, shots = kwargs["qcs"], kwargs["shots"]
        if self.evaluations + len(qcs) > self.budget:
            self.refused += len(qcs)
            return [{"0" * self.nodes: shots} for _ in qcs]
        self.evaluations += len(qcs)
        self.batches += 1
        results = []
        for qasm in qcs:
            state = polypus.statevector(polypus.Circuit.from_qasm2(qasm))
            probabilities = np.abs(state) ** 2
            counts = self.rng.multinomial(shots, probabilities / probabilities.sum())
            results.append({self.keys[k]: int(c) for k, c in enumerate(counts) if c})
        return results


# ── Optimizers ───────────────────────────────────────────────────────────────


def default_lambda(dimensions):
    return 4 + math.floor(3 * math.log(dimensions))


def make_method(name, dimensions, budget, population, restarts):
    """The optimizer ``name`` configured for ``budget`` evaluations."""
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
    raise ValueError(f"unknown optimizer {name!r}")


def run_one(instance, name, shots, budget, population, restarts, seed):
    meter = Meter(instance.nodes, budget, seed=10_000 + seed)
    polypus_python.run_qcs = meter
    dimensions = 2 * instance.layers
    start = time.perf_counter()
    result = polypus.train(
        instance.circuit,
        make_method(name, dimensions, budget, population, restarts),
        shots=shots,
        n_qpus=1,
        dimensions=dimensions,
        expectation_function=instance.observable,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="bench_optimizers",
        backend="aer",
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    if name != "cmaes-ipop" and meter.refused:
        raise AssertionError(f"{name} went past its budget: {meter.refused} refused")
    exact = instance.expected_cut(result.best_params)
    return {
        "optimizer": name,
        "instance": seed,
        "ratio": exact / instance.max_cut,
        "bias": (result.best_fitness - exact) / instance.max_cut,
        "evaluations": meter.evaluations,
        "batches": meter.batches,
        "iterations_run": result.iterations_run,
        "seconds": elapsed,
    }


# ── Report ───────────────────────────────────────────────────────────────────


def summarise(rows, optimizers):
    by = {
        name: {r["instance"]: r for r in rows if r["optimizer"] == name}
        for name in optimizers
    }
    base = by.get(BASELINE)
    lines = [
        "| Optimizer | ratio | Δ vs rand1bin | wins | bias | evals | batches | time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in optimizers:
        runs = list(by[name].values())
        mean = lambda key: statistics.fmean(r[key] for r in runs)  # noqa: E731
        if base is None or name == BASELINE:
            delta, wins = "—", "—"
        else:
            diffs = [by[name][s]["ratio"] - base[s]["ratio"] for s in by[name]]
            se = (
                statistics.stdev(diffs) / math.sqrt(len(diffs))
                if len(diffs) > 1
                else 0.0
            )
            delta = f"{statistics.fmean(diffs):+.3f} ± {se:.3f}"
            share = sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs)
            wins = f"{100 * share / len(diffs):.0f} %"
        lines.append(
            f"| {name} | {mean('ratio'):.3f} | {delta} | {wins} | {mean('bias'):+.3f} "
            f"| {mean('evaluations'):.0f} | {mean('batches'):.0f} | {mean('seconds'):.2f} |"
        )
    return "\n".join(lines)


def parse_scenario(text):
    try:
        layers, shots, budget = text.split(":")
        shots = EXACT_SHOTS if shots == "exact" else int(shots)
        return int(layers), shots, int(budget)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected LAYERS:SHOTS:BUDGET (SHOTS may be 'exact'), got {text!r}"
        ) from None


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
        description="Compare Polypus's DE strategies and CMA-ES on QAOA MaxCut.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        nargs="+",
        type=parse_scenario,
        default=None,
        metavar="LAYERS:SHOTS:BUDGET",
        help="QAOA layers p, shots per circuit ('exact' for 2^20) and the "
        "evaluation budget (default: 3:256:1500).",
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
        "3:256:500 with 30 instances",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: 6 nodes, scenario 1:64:120, 2 instances, P = 8",
    )
    p.add_argument("--json", type=Path, default=None, help="write every run here")
    args = p.parse_args(argv)
    if args.full:
        args.scenario = args.scenario or [
            (3, 256, 1500),
            (3, EXACT_SHOTS, 1500),
            (5, 256, 3000),
            (3, 256, 500),
        ]
        args.instances = 30
    if args.quick:
        args.scenario = args.scenario or [(1, 64, 120)]
        args.nodes, args.instances, args.de_population = 6, 2, 8
    args.scenario = args.scenario or [(3, 256, 1500)]
    return args


def main(argv=None):
    args = parse_args(argv)
    optimizers = args.only or OPTIMIZERS
    env = environment()
    print("# Polypus optimizer benchmark: QAOA MaxCut on random 3-regular graphs\n")
    for key, value in env.items():
        print(f"- {key}: {value}")
    seeds = range(args.first_instance, args.first_instance + args.instances)
    report = {"environment": env, "scenarios": []}
    for layers, shots, budget in args.scenario:
        shots_label = "2^20" if shots == EXACT_SHOTS else str(shots)
        print(
            f"\n## n = {args.nodes}, p = {layers}, {shots_label} shots, "
            f"{budget} evaluations, {args.instances} instances\n",
            flush=True,
        )
        instances = [Instance(args.nodes, layers, s) for s in seeds]
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
                    )
                )
        print(summarise(rows, optimizers), flush=True)
        report["scenarios"].append(
            {
                "nodes": args.nodes,
                "layers": layers,
                "shots": shots,
                "budget": budget,
                "de_population": args.de_population,
                "restarts": args.restarts,
                "runs": rows,
            }
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    return report


if __name__ == "__main__":
    main()
