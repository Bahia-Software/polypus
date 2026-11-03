#!/usr/bin/env python3
"""
Call trace benchmark: what writing a run's call trace (``trace=``, contract
C-12) costs in time, and how large the traces grow.

Overhead (the default). SPSA trains ``bench_optimizers.py``'s HEA scenario
(``hea:2:256:5000`` in its ``--full`` run): MaxCut on a random 3-regular
graph of 10 nodes with its hardware-efficient ansatz, ``reps = 2``, three
layers of RY on every qubit with a linear chain of CZ between consecutive
layers, 30 parameters. SPSA takes ``bench_optimizers.py``'s settings for the
HEA (one perturbation per iteration, perturbation 0.1, target step 0.63) and
makes ``--iterations`` iterations, by default the 2414 that fit the
scenario's 5000 evaluations, with 256 shots per circuit, on Polypus's native
backend with one QPU. A pair is the same run, with the same seed, once
untraced and once traced to a new file. A first pair warms up and is
discarded; the ``--pairs`` pairs after it alternate their order, the even
ones untraced first and the odd ones traced first, so that a drift in the
machine's speed weighs on both sides. A pair's overhead is its traced run's
wall-clock time (``time.perf_counter``) over its untraced run's, minus 1.
The report gives each pair's seconds and overhead, their median, the 95%
bootstrap interval of the median (percentile method, 10,000 resamples), its
verdict and the size of each trace.

The criterion: the upper end of the interval below 1%. The verdict is
``pass`` when it is; ``fail`` when the lower end is at 1% or above, so that
the overhead is 1% or more; and ``inconclusive`` when the interval holds 1%,
too wide to tell whether the overhead is below it. The number of pairs is
fixed before the run, from the spread of a pair's overhead: on a shared
virtual machine with 4 CPUs its standard deviation (``SPREAD``) was about
8%, over 70 pairs. The 95% interval of the median of n pairs then reaches
about 2.5·8%/√n on either side of it, and the default ``--pairs``
(``pairs_needed(SPREAD)``, 790) puts its upper end below 1% four times in
five when tracing costs nothing. A quieter machine needs fewer pairs: fix
them before the run, from the spread measured there.

Size (``--size``). One traced run of each optimizer of
``bench_optimizers.py`` whose budget fixes its run in advance (all but
``cmaes-ipop``, whose restarts depend on when it stagnates, and the chained
``lshade-adam`` and ``adam-x3``), on the same HEA, with 10,000 evaluations,
a population of 30 for DE and ``cmaes-P`` (``bench_optimizers.py``'s
default), and 256 shots. The criteria: the largest trace at most 2 MB
(2·10^6 bytes) and the mean at most 0.5 MB.

Tracing must not change what a run computes: every run of the overhead
benchmark, traced or not, must return the same result bit for bit
(``identical``), and every trace must end with the ``end`` line of a run that
returned, not truncated (``complete``). Otherwise the script exits with
status 1, whatever the times and sizes.

The runs use one thread: run the script with ``RAYON_NUM_THREADS=1``, or it
exits with status 2. With more threads the native backend spreads a batch's
circuits over the cores, and the timings then depend on how many cores other
work leaves free. ``POLYPUS_CALL_TRACE`` must be unset (status 2 otherwise),
since it would trace the untraced runs too. ``POLYPUS_NO_AUTOCALIBRATE=1``
skips the calibration of the native simulator's gate-parallel threshold that
the first native run of a process would do. The traces go to a temporary
directory (``TMPDIR`` chooses where), deleted at the end.

Run the overhead benchmark on an idle machine: builds, tests or other
benchmarks running at the same time make the timings noisy and widen the
interval. A run that other work disturbed does not count: discard it before
reading its verdict, and run it again on an idle machine. Each run is judged
on its own, with its pairs fixed before it starts: an inconclusive run calls
for a quieter machine or more pairs in a new run, never for pooling the
pairs of several runs or for running again until one passes. Do not change
the criterion.

Usage
-----
  RAYON_NUM_THREADS=1 POLYPUS_NO_AUTOCALIBRATE=1 python benchmarks/bench_trace_overhead.py --check --json trace-overhead.json   # 790 pairs: about 25 minutes
  RAYON_NUM_THREADS=1 POLYPUS_NO_AUTOCALIBRATE=1 python benchmarks/bench_trace_overhead.py --size --check --json trace-size.json
  RAYON_NUM_THREADS=1 python benchmarks/bench_trace_overhead.py --pairs 1 --iterations 20   # seconds: a smoke run

``--json`` writes the report, ``environment`` and then ``overhead`` or
``size``. ``--check`` makes the script exit with status 1 when a criterion
is not met (for the overhead, the verdict ``fail``), and with status 3 when
the overhead's verdict is ``inconclusive``.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    import polypus
except ImportError:
    print(
        "[bench-trace-overhead] ERROR: polypus is not installed (maturin develop).",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_bench_optimizers():
    """``bench_optimizers.py``, next to this script: the circuits, the
    optimizer settings and the budgets measured here are its own."""
    path = Path(__file__).resolve().with_name("bench_optimizers.py")
    spec = importlib.util.spec_from_file_location("bench_optimizers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench_optimizers = _load_bench_optimizers()

# bench_optimizers.py's HEA scenario (hea:2:256:5000): the graph's nodes (its
# --nodes default), the HEA's repetitions (30 parameters) and the shots per
# circuit.
NODES = 10
REPS = 2
SHOTS = 256
# The evaluation budgets: the scenario's for the overhead benchmark, and a
# long run's for the size benchmark.
OVERHEAD_BUDGET = 5000
SIZE_BUDGET = 10_000
# The population of DE and cmaes-P in the size benchmark (bench_optimizers.py's
# --de-population default).
SIZE_POPULATION = 30
# The optimizers of the size benchmark: bench_optimizers.py's whose budget
# fixes their run in advance.
SIZE_OPTIMIZERS = (
    "rand1bin",
    "shade",
    "lshade",
    "cmaes",
    "cmaes-P",
    "adam",
    "momentum",
    "spsa",
    "spsa-k8",
    "rotosolve",
)
# The overhead criterion: the upper end of the interval of the median below it.
MAX_OVERHEAD = 0.01
# The default --pairs is pairs_needed(SPREAD): SPREAD is the standard deviation
# of a pair's overhead, measured over 70 pairs on a shared virtual machine with
# 4 CPUs, and POWER the chance that the default gives the criterion to hold
# when tracing costs nothing.
SPREAD = 0.08
POWER = 0.8
# The exit status of --check when the overhead's verdict is inconclusive.
INCONCLUSIVE = 3
# The size criteria, in bytes: the largest trace and the mean trace.
MAX_TRACE_BYTES = 2_000_000
MAX_MEAN_TRACE_BYTES = 500_000


# ── Statistics ───────────────────────────────────────────────────────────────


def overheads(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """Each pair's overhead from its ``(untraced, traced)`` seconds: the
    traced run's time over the untraced run's, minus 1."""
    return [on / off - 1.0 for off, on in pairs]


def median(values: Sequence[float]) -> float:
    """The median of ``values``."""
    return float(statistics.median(values))


def bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """The percentile bootstrap interval of the median of ``values`` at
    confidence ``level``: the medians of ``resamples`` resamples of
    ``values`` with replacement, drawn with ``random.Random(seed)``, sorted,
    and the ``floor(a·B)``-th and ``(ceil((1 − a)·B) − 1)``-th of them
    (counting from 0), for ``B`` resamples and ``a = (1 − level) / 2``."""
    if resamples < 1 or not 0.0 < level < 1.0:
        raise ValueError(
            f"expected resamples >= 1 and 0 < level < 1, got {resamples} and {level}"
        )
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    tail = (1.0 - level) / 2.0
    # (1 - level) / 2 is not exact in binary floating point: for level 0.9 it
    # is 0.04999999999999999, so tail * 1000 is 49.999999999999986, and floor
    # would take the 49th median instead of the 50th. The products are rounded
    # to 9 decimals first, which puts such a near miss back on its integer.
    low = math.floor(round(tail * resamples, 9))
    high = math.ceil(round((1.0 - tail) * resamples, 9)) - 1
    return float(medians[low]), float(medians[high])


def verdict(low: float, high: float, threshold: float = MAX_OVERHEAD) -> str:
    """The verdict on the interval ``[low, high]`` of the median overhead
    against ``threshold``: ``"pass"`` when its upper end is below it,
    ``"fail"`` when its lower end is at or above it, and ``"inconclusive"``
    when the interval holds it, too wide to tell."""
    if high < threshold:
        return "pass"
    if low >= threshold:
        return "fail"
    return "inconclusive"


def pairs_needed(
    spread: float,
    *,
    margin: float = MAX_OVERHEAD,
    power: float = POWER,
    level: float = 0.95,
) -> int:
    """The pairs that put the upper end of the ``level`` interval of the
    median overhead below ``margin`` with probability ``power`` when tracing
    costs nothing and a pair's overhead has the standard deviation
    ``spread``: the least ``n`` with ``(z(a) + z(power))·s/√n <= margin``,
    for ``a = (1 + level) / 2``, ``z`` the quantiles of the standard normal
    distribution and ``s = √(π/2)·spread``, as ``s/√n`` is about the
    standard error of the median of ``n`` normal values."""
    if spread < 0.0 or margin <= 0.0 or not 0.0 < power < 1.0 or not 0.0 < level < 1.0:
        raise ValueError(
            "expected spread >= 0, margin > 0, 0 < power < 1 and 0 < level < 1, "
            f"got {spread}, {margin}, {power} and {level}"
        )
    z = statistics.NormalDist().inv_cdf
    quantiles = max(0.0, z((1.0 + level) / 2.0) + z(power))
    root = quantiles * math.sqrt(math.pi / 2.0) * spread / margin
    # Rounded to 9 decimals, as in bootstrap_interval, so that a rounding
    # error cannot take a whole square one pair up.
    return max(1, math.ceil(round(root * root, 9)))


def size_summary(sizes: Mapping[str, int]) -> dict:
    """The largest (``worst``) and the mean of the trace sizes ``sizes``, in
    bytes, and whether they meet the size criteria (``pass``)."""
    worst = max(sizes.values())
    mean = statistics.fmean(sizes.values())
    return {
        "worst": worst,
        "mean": mean,
        "pass": worst <= MAX_TRACE_BYTES and mean <= MAX_MEAN_TRACE_BYTES,
    }


# ── Runs ─────────────────────────────────────────────────────────────────────


def spsa(iterations):
    """SPSA as ``bench_optimizers.py`` runs ``spsa`` on the HEA, for
    ``iterations`` iterations."""
    perturbation, target_step = bench_optimizers.SPSA["hea"]
    return polypus.SPSA(
        max_iters=iterations,
        resamplings=1,
        perturbation=perturbation,
        target_step=target_step,
    )


def train(instance, method, seed, trace=None):
    """One ``polypus.train`` run of ``method`` on ``instance``, on the native
    backend with one QPU, traced to the new file ``trace`` unless it is
    ``None``."""
    return polypus.train(
        instance.circuit,
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=instance.dimensions,
        expectation_function=instance.observable,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="bench_trace_overhead",
        backend="polypus",
        seed=seed,
        trace=trace,
    )


def _hex(values):
    return None if values is None else tuple(value.hex() for value in values)


def result_bits(result):
    """Every field of a ``TrainResult``, each float as ``float.hex``, so that
    two results compare equal only when they are equal bit for bit. The ``id``
    loses the UUID ``train`` appends to every run's."""
    return (
        _hex(result.best_params),
        result.best_fitness.hex(),
        result.iterations_run,
        result.converged,
        _hex(result.fitness_history),
        _hex(result.final_params),
        None if result.final_fitness is None else result.final_fitness.hex(),
        result.seed,
        result.id.rsplit("_", 1)[0],
    )


def inspect_trace(path):
    """The size in bytes and the number of lines of the trace at ``path``,
    and whether it is complete: its last line the ``end`` line of a run that
    returned (``"status": "ok"``), not truncated. A trace whose file is
    missing has 0 bytes and 0 lines, and is not complete."""
    try:
        data = path.read_bytes()
    except OSError:
        return 0, 0, False
    lines = data.splitlines()
    try:
        end = json.loads(lines[-1]) if data.endswith(b"\n") else None
    except ValueError:
        end = None
    complete = (
        isinstance(end, dict)
        and end.get("ev") == "end"
        and end.get("status") == "ok"
        and end.get("truncated") is False
    )
    return len(data), len(lines), complete


def overhead_run(*, pairs: int, iterations: int, seed: int, workdir: Path) -> dict:
    """The overhead benchmark: a warm-up pair, then ``pairs`` pairs of the
    same SPSA run of ``iterations`` iterations, untraced and traced to a file
    in ``workdir``, all with ``seed``. The report gives the warm-up pair's
    and each pair's ``[untraced, traced]`` seconds, the pairs' ``overheads``,
    their standard deviation ``sd`` (``None`` for one pair), their ``median``
    and its 95% bootstrap interval ``ci95``, each pair's ``trace_bytes``,
    whether every run returned the same result (``identical``) and every
    trace is complete (``complete``), and the criterion's ``verdict``."""
    instance = bench_optimizers.Instance(NODES, REPS, seed, "hea")
    results = set()

    def pair(name, traced_first):
        """One pair of runs: its ``[untraced, traced]`` seconds, and its
        trace's size, lines and completeness."""
        seconds = {}
        path = workdir / f"{name}.jsonl"
        for traced in (True, False) if traced_first else (False, True):
            method = spsa(iterations)
            start = time.perf_counter()
            result = train(instance, method, seed, trace=path if traced else None)
            seconds[traced] = time.perf_counter() - start
            results.add(result_bits(result))
        return [seconds[False], seconds[True]], inspect_trace(path)

    print(
        f"\n## SPSA, {iterations} iterations, "
        f"{bench_optimizers.describe('hea', NODES, REPS)}, "
        f"{SHOTS} shots, seed {seed}\n",
        flush=True,
    )
    warmup, (_, _, complete) = pair("warmup", traced_first=False)
    seconds, trace_bytes = [], []
    print("| pair | first | untraced (s) | traced (s) | overhead | trace (bytes) |")
    print("|---|---|---|---|---|---|")
    for index in range(pairs):
        traced_first = index % 2 == 1
        times, (size, _, ok) = pair(f"pair-{index}", traced_first)
        complete = complete and ok
        seconds.append(times)
        trace_bytes.append(size)
        (overhead,) = overheads([times])
        print(
            f"| {index} | {'traced' if traced_first else 'untraced'} "
            f"| {times[0]:.4f} | {times[1]:.4f} | {overhead:+.2%} | {size} |",
            flush=True,
        )
    values = overheads(seconds)
    low, high = bootstrap_interval(values, seed=seed)
    report = {
        "nodes": NODES,
        "reps": REPS,
        "parameters": instance.dimensions,
        "shots": SHOTS,
        "iterations": iterations,
        "pairs": pairs,
        "seed": seed,
        "warmup": warmup,
        "seconds": seconds,
        "overheads": values,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": median(values),
        "ci95": [low, high],
        "trace_bytes": trace_bytes,
        "identical": len(results) == 1,
        "complete": complete,
        "verdict": verdict(low, high),
    }
    against = {
        "pass": "below",
        "fail": "at or above",
        "inconclusive": "holding",
    }[report["verdict"]]
    print(
        f"\nmedian overhead {report['median']:+.2%}, 95% bootstrap interval "
        f"[{low:+.2%}, {high:+.2%}], {against} {MAX_OVERHEAD:.0%}: "
        f"{report['verdict']}",
        flush=True,
    )
    return report


def size_run(*, seed: int, workdir: Path) -> dict:
    """The size benchmark: one run of each of ``SIZE_OPTIMIZERS`` with
    ``seed``, traced to a file in ``workdir``. The report gives each trace's
    size in bytes (``sizes``) and lines (``lines``), whether every trace is
    complete (``complete``), and ``size_summary`` of the sizes."""
    instance = bench_optimizers.Instance(NODES, REPS, seed, "hea")
    print(
        f"\n## One traced run per optimizer, "
        f"{bench_optimizers.describe('hea', NODES, REPS)}, "
        f"{SIZE_BUDGET} evaluations, population {SIZE_POPULATION}, "
        f"{SHOTS} shots, seed {seed}\n",
        flush=True,
    )
    print("| optimizer | trace (bytes) | lines |")
    print("|---|---|---|")
    sizes, lines = {}, {}
    complete = True
    for name in SIZE_OPTIMIZERS:
        method = bench_optimizers.make_method(
            name, instance.dimensions, SIZE_BUDGET, SIZE_POPULATION, 0, problem="hea"
        )
        path = workdir / f"{name}.jsonl"
        train(instance, method, seed, trace=path)
        sizes[name], lines[name], ok = inspect_trace(path)
        complete = complete and ok
        print(f"| {name} | {sizes[name]} | {lines[name]} |", flush=True)
    summary = size_summary(sizes)
    largest = max(sizes, key=sizes.get)
    print(
        f"\nlargest {summary['worst']} bytes ({largest}), mean "
        f"{summary['mean']:.0f} bytes: "
        + ("within" if summary["pass"] else "NOT within")
        + " the limits",
        flush=True,
    )
    return {
        "nodes": NODES,
        "reps": REPS,
        "parameters": instance.dimensions,
        "budget": SIZE_BUDGET,
        "population": SIZE_POPULATION,
        "shots": SHOTS,
        "seed": seed,
        "sizes": sizes,
        "lines": lines,
        "complete": complete,
        **summary,
    }


# ── Command line ─────────────────────────────────────────────────────────────


def parse_args(argv=None):
    iterations = bench_optimizers.spsa_iterations(OVERHEAD_BUDGET, 1)
    pairs = pairs_needed(SPREAD)
    p = argparse.ArgumentParser(
        description="Measure what Polypus's call trace costs in time, or "
        "(--size) how large the traces grow.",
    )
    p.add_argument(
        "--pairs",
        type=bench_optimizers.positive_int,
        default=pairs,
        help=f"timed pairs of runs after the warm-up pair (default: {pairs}, "
        f"fixed from a pair's standard deviation of {SPREAD * 100:g}%%)",
    )
    p.add_argument(
        "--iterations",
        type=bench_optimizers.positive_int,
        default=iterations,
        help=f"SPSA iterations of each run (default: {iterations}, the most "
        f"that fit {OVERHEAD_BUDGET} evaluations)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed of the graph, of every run and of the bootstrap (default: 1)",
    )
    p.add_argument(
        "--size",
        action="store_true",
        help="measure the size of the traces instead of the overhead",
    )
    p.add_argument("--json", type=Path, default=None, help="write the report here")
    p.add_argument(
        "--check",
        action="store_true",
        help="exit with status 1 when the criterion is not met, and with "
        f"status {INCONCLUSIVE} when the overhead's interval is too wide to tell",
    )
    args = p.parse_args(argv)
    if args.json is not None and not args.json.parent.is_dir():
        p.error(f"--json: {args.json.parent} is not a directory")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark ``argv`` asks for and return the exit status."""
    args = parse_args(argv)
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        print(
            "[bench-trace-overhead] ERROR: run with RAYON_NUM_THREADS=1: with "
            "more threads the native backend spreads a batch's circuits over "
            "the cores, and the timings depend on how many cores other work "
            "leaves free",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("POLYPUS_CALL_TRACE"):
        print(
            "[bench-trace-overhead] ERROR: unset POLYPUS_CALL_TRACE, which "
            "would trace the untraced runs too",
            file=sys.stderr,
        )
        return 2
    env = bench_optimizers.environment()
    title = "trace size" if args.size else "overhead"
    print(f"# Polypus call trace benchmark: {title}\n")
    for name, value in env.items():
        print(f"- {name}: {value}")
    with tempfile.TemporaryDirectory(prefix="bench-trace-") as workdir:
        if args.size:
            key = "size"
            result = size_run(seed=args.seed, workdir=Path(workdir))
        else:
            key = "overhead"
            result = overhead_run(
                pairs=args.pairs,
                iterations=args.iterations,
                seed=args.seed,
                workdir=Path(workdir),
            )
    if args.json is not None:
        report = {"environment": env, key: result}
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    failures = []
    if key == "overhead" and not result["identical"]:
        failures.append("the runs did not all return the same result, bit for bit")
    if not result["complete"]:
        failures.append("a trace does not end with the end line of a run that returned")
    if args.check and key == "size" and not result["pass"]:
        failures.append("the trace size criterion is not met")
    if args.check and key == "overhead" and result["verdict"] == "fail":
        failures.append(
            f"the overhead criterion is not met: the overhead is {MAX_OVERHEAD:.0%} "
            "or more"
        )
    for failure in failures:
        print(f"[bench-trace-overhead] FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    if args.check and key == "overhead" and result["verdict"] == "inconclusive":
        advice = ""
        if result["sd"]:
            advice = (
                f"; at this run's spread (a standard deviation of "
                f"{result['sd']:.1%}), a new run needs about "
                f"{pairs_needed(result['sd'])} pairs, fixed before it starts"
            )
        print(
            f"[bench-trace-overhead] INCONCLUSIVE: the interval holds "
            f"{MAX_OVERHEAD:.0%}, too wide to tell whether the overhead is "
            f"below it{advice}",
            file=sys.stderr,
        )
        return INCONCLUSIVE
    return 0


if __name__ == "__main__":
    sys.exit(main())
