"""Core library for the MaxCut–QAOA flagship experiment.

This module is the *engine* shared by the three CLI stages
(:mod:`run_maxcut`, :mod:`sweep_maxcut`, :mod:`report_maxcut`). It is pure and
importable — no argument parsing, no ``print``, no plotting — so the run stage
can be exercised directly from ``tests/python``.

Design contract for reproducibility (see the repo's C-7 seeding contract): a
single integer seed must make an entire QAOA optimization byte-reproducible.
That only holds if *every* source of shot randomness is seeded — the training
oracle (seeded through ``polypus.train(..., seed=...)``), the QNG variance
callback, the final evaluation sampling, and the scipy baseline's Aer calls.
Each site below threads the same effective seed.

Qubit/bitstring convention: Qiskit ``measure_all`` yields a string whose
leftmost character is the *highest* qubit index. To index by graph node ``i``
the string is reversed exactly **once** (``bitstring[::-1]``). The whole module
uses this single-reversal convention consistently, both for the optimization
objective and for the reported approximation ratio.
"""

from __future__ import annotations

import csv
import functools
import itertools
import logging
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import networkx as nx
import numpy as np

import polypus
from polypus_python.qaoa_utils import build_qaoa_circuit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

logger = logging.getLogger("maxcut")

#: Package directory and default (gitignored) output locations, shared by all
#: three CLI stages so they agree on where data and reports live.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PACKAGE_DIR, "output")
DEFAULT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, "maxcut_results.csv")
DEFAULT_REPORT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "report")


# ─────────────────────────────────────────────────────────────────────────────
# Errors — raised loudly; never swallowed. A failed run must not write a row.
# ─────────────────────────────────────────────────────────────────────────────


class ExperimentError(Exception):
    """Base class for experiment failures."""


class InputValidationError(ExperimentError):
    """A run configuration or generated problem instance is invalid."""


class OutputValidationError(ExperimentError):
    """A run produced results that violate a correctness invariant."""


# ─────────────────────────────────────────────────────────────────────────────
# Method vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# method string  ->  (optimizer, infrastructure)
METHOD_SPECS = {
    "scipy": ("scipy", "local"),
    "polypus_local": ("de", "local"),
    "polypus_local_pso": ("pso", "local"),
    "polypus_local_qng": ("qng", "local"),
    # CUNQA variants stay invokable manually (no SLURM cluster here, so they are
    # never part of the default sweep); do not remove — see the task scope note.
    "polypus_cunqa": ("de", "cunqa"),
    "polypus_cunqa_pso": ("pso", "cunqa"),
    "polypus_cunqa_qng": ("qng", "cunqa"),
}

ALL_METHODS = tuple(METHOD_SPECS)

#: Methods swept by default: the three local Polypus optimizers plus the pure
#: scipy baseline ("no Polypus") used as the classical reference.
DEFAULT_SWEEP_METHODS = (
    "scipy",
    "polypus_local",
    "polypus_local_pso",
    "polypus_local_qng",
)

_OPTIMIZER_LABEL = {"de": "DE", "pso": "PSO", "qng": "QNG", "scipy": "scipy_DE"}


# ─────────────────────────────────────────────────────────────────────────────
# CSV schema — one row per (n_qubits, method, repeat). Bumped if columns change;
# the reader rejects rows whose schema_version it does not understand.
# ─────────────────────────────────────────────────────────────────────────────

CSV_SCHEMA_VERSION = 1

CSV_FIELDS = (
    "schema_version",
    "timestamp",
    "git_commit",
    "polypus_version",
    "n_qubits",
    "method",
    "optimizer",
    "infrastructure",
    "backend",
    "seed",
    "base_seed",
    "repeat_index",
    "graph_seed",
    "graph_prob",
    "n_edges",
    "layers",
    "dimensions",
    "population_size",
    "max_generations",
    "n_shots",
    "n_qpus",
    "nodes",
    "cores_per_qpu",
    "best_solution_bruteforce",
    "best_cut",
    "best_bitstring",
    "approximation_ratio",
    "best_ratio",
    "best_fitness",
    "iterations_run",
    "converged",
    "time_seconds",
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration & result records
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Everything needed to run and reproduce ONE QAOA optimization."""

    n_qubits: int
    method: str
    seed: int
    repeat_index: int = 0
    base_seed: Optional[int] = None  # provenance; defaults to `seed` if unset
    n_shots: int = 10_000
    n_qpus: int = 1
    nodes: int = 1
    cores_per_qpu: int = 1
    # Problem instance: the graph depends only on (n_qubits, graph_seed, prob),
    # so a given qubit count always poses the *same* MaxCut instance across
    # methods and repeats — variance then comes purely from the optimizer seed.
    graph_seed: int = 7
    graph_prob: float = 0.6
    # Circuit / optimizer sizing (kept identical to the historical experiment).
    layers_factor: int = 2
    population_factor: int = 4
    generations_factor: int = 5
    tolerance: float = 1e-5
    # scipy baseline: number of worker processes for differential_evolution.
    scipy_workers: int = 1
    # QNG hyper-parameters.
    qng_learning_rate: float = 0.1
    qng_step: float = 0.1
    qng_tikhonov: float = 0.05

    def __post_init__(self) -> None:
        if self.base_seed is None:
            self.base_seed = self.seed

    @property
    def optimizer(self) -> str:
        return METHOD_SPECS[self.method][0]

    @property
    def infrastructure(self) -> str:
        return METHOD_SPECS[self.method][1]


@dataclass
class RunRecord:
    """One CSV row — the full, self-describing result of a single run."""

    schema_version: int
    timestamp: str
    git_commit: str
    polypus_version: str
    n_qubits: int
    method: str
    optimizer: str
    infrastructure: str
    backend: str
    seed: int
    base_seed: int
    repeat_index: int
    graph_seed: int
    graph_prob: float
    n_edges: int
    layers: int
    dimensions: int
    population_size: int
    max_generations: int
    n_shots: int
    n_qpus: int
    nodes: int
    cores_per_qpu: int
    best_solution_bruteforce: int
    best_cut: int
    best_bitstring: str
    approximation_ratio: float
    best_ratio: float
    best_fitness: float
    iterations_run: int
    converged: bool
    time_seconds: float

    def to_row(self) -> dict:
        row = asdict(self)
        assert set(row) == set(CSV_FIELDS), "RunRecord fields drifted from CSV_FIELDS"
        return row


# ─────────────────────────────────────────────────────────────────────────────
# Provenance helpers
# ─────────────────────────────────────────────────────────────────────────────


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def polypus_version() -> str:
    version = getattr(polypus, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as _v

        return _v("polypus")
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Graph / objective (all top-level so the scipy multiprocessing pool can pickle
# the objective; no reliance on module-global state).
# ─────────────────────────────────────────────────────────────────────────────


def maxcut_cost_layer(graph):
    def layer_fn(qc, layer, gamma):
        for i, j in graph.edges():
            qc.cx(i, j)
            qc.rz(-gamma, j)
            qc.cx(i, j)

    return layer_fn


def standard_mixer_layer(n_qubits):
    def layer_fn(qc, layer, beta):
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    return layer_fn


def bitstring_cut(bitstring: str, edges) -> int:
    """Cut value of a measured bitstring under the single-reversal convention.

    ``edges`` is bound via ``functools.partial`` when handed to the optimizer,
    which keeps the objective a top-level picklable callable.
    """
    b = bitstring[::-1]
    return sum(1 for i, j in edges if b[i] != b[j])


def maxcut_bruteforce(graph) -> int:
    edges = list(graph.edges())
    best = 0
    for bits in itertools.product([0, 1], repeat=graph.number_of_nodes()):
        cut = sum(1 for i, j in edges if bits[i] != bits[j])
        best = max(best, cut)
    return best


def mean_energy(counts: dict, edges) -> float:
    """Expected cut under the measured distribution (single-reversal, matching
    the optimization objective)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(bitstring_cut(bs, edges) * freq for bs, freq in counts.items()) / total


# ─────────────────────────────────────────────────────────────────────────────
# QNG-specific circuit + seeded variance callback
# ─────────────────────────────────────────────────────────────────────────────


def build_qaoa_circuit_qng(graph, layers: int) -> QuantumCircuit:
    """QAOA circuit with a single alternating ParameterVector (gamma/beta) as
    required by the QNG variance function (which builds partial circuits up to a
    given parameter index)."""
    n_qubits = graph.number_of_nodes()
    edges = list(graph.edges())
    theta = ParameterVector("θ", 2 * layers)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for layer in range(layers):
        gamma = theta[2 * layer]
        beta = theta[2 * layer + 1]
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(-gamma, j)
            qc.cx(i, j)
        for qubit in range(n_qubits):
            qc.rx(2 * beta, qubit)
    qc.measure_all()
    return qc


def make_variance_function(graph, n_qubits: int, n_shots: int, seed: int) -> Callable:
    """Return ``variance_fn(theta, a) -> float`` estimating the diagonal QFIM
    element for parameter ``a``.

    Aer sampling here is seeded with the run's effective seed (``seed_simulator``)
    so QNG becomes reproducible: without this the callback would inject fresh
    randomness that ``polypus.train``'s seed cannot reach.
    """
    edges = list(graph.edges())
    backend = AerSimulator()

    def variance_fn(theta, a):
        if a == 0:
            return 0.0

        theta_a = ParameterVector("θ", a)
        qc_inter = QuantumCircuit(n_qubits)
        qc_inter.h(range(n_qubits))
        for k in range(a):
            if k % 2 == 0:  # gamma (ZZ cost layer)
                for i, j in edges:
                    qc_inter.cx(i, j)
                    qc_inter.rz(-theta_a[k], j)
                    qc_inter.cx(i, j)
            else:  # beta (X mixer layer)
                for qubit in range(n_qubits):
                    qc_inter.rx(2 * theta_a[k], qubit)

        if a % 2 == 0:  # next param is gamma -> ZZ Hamiltonian -> Z basis
            qc_inter.measure_all()
        else:  # next param is beta -> X Hamiltonian -> X basis
            qc_inter.h(range(n_qubits))
            qc_inter.measure_all()

        qc_bound = qc_inter.assign_parameters(list(theta[:a]))
        counts = backend.run(qc_bound, shots=n_shots, seed_simulator=seed).result().get_counts()
        shots_total = sum(counts.values())

        exp_h = 0.0
        exp_h2 = 0.0
        for bitstring, count in counts.items():
            prob = count / shots_total
            bits = bitstring[::-1]
            if a % 2 == 0:
                h_val = sum(
                    (1 if bits[i] == "0" else -1) * (1 if bits[j] == "0" else -1)
                    for i, j in edges
                )
            else:
                h_val = sum(1 if bits[i] == "0" else -1 for i in range(n_qubits))
            exp_h += h_val * prob
            exp_h2 += (h_val ** 2) * prob
        return exp_h2 - exp_h ** 2

    return variance_fn


# ─────────────────────────────────────────────────────────────────────────────
# Problem instance
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Problem:
    graph: "nx.Graph"
    edges: list
    n_qubits: int
    layers: int
    dimensions: int
    population_size: int
    max_generations: int
    best_solution: int


def build_problem(cfg: RunConfig) -> Problem:
    """Generate and validate the MaxCut instance for ``cfg``."""
    if cfg.n_qubits < 2:
        raise InputValidationError(
            f"--qubits must be >= 2 (MaxCut needs at least one edge); got {cfg.n_qubits}"
        )
    graph = nx.gnp_random_graph(cfg.n_qubits, cfg.graph_prob, seed=cfg.graph_seed)
    n_qubits = graph.number_of_nodes()
    edges = list(graph.edges())
    if len(edges) == 0:
        raise InputValidationError(
            f"generated graph for n_qubits={n_qubits} (graph_seed={cfg.graph_seed}, "
            f"prob={cfg.graph_prob}) has no edges; MaxCut is undefined. Choose a "
            "different --graph-seed or a higher --graph-prob."
        )
    layers = (n_qubits // 2) * cfg.layers_factor
    dimensions = 2 * layers
    population_size = dimensions * cfg.population_factor
    max_generations = n_qubits * cfg.generations_factor
    best_solution = maxcut_bruteforce(graph)
    return Problem(
        graph=graph,
        edges=edges,
        n_qubits=n_qubits,
        layers=layers,
        dimensions=dimensions,
        population_size=population_size,
        max_generations=max_generations,
        best_solution=best_solution,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation (seeded) + output validation
# ─────────────────────────────────────────────────────────────────────────────

_RATIO_EPS = 1e-9


def _final_evaluation(qc: QuantumCircuit, params, edges, n_shots: int, seed: int, best_solution: int):
    """Bind params, sample once (seeded), return (counts, best_bitstring,
    best_cut, approximation_ratio, best_ratio)."""
    bound = qc.assign_parameters(list(params))
    counts = AerSimulator().run(bound, shots=n_shots, seed_simulator=seed).result().get_counts()
    best_bitstring = max(counts, key=counts.get)
    best_cut = bitstring_cut(best_bitstring, edges)
    approximation_ratio = mean_energy(counts, edges) / best_solution
    best_ratio = best_cut / best_solution
    return counts, best_bitstring, best_cut, approximation_ratio, best_ratio


def _validate_outputs(counts: dict, n_shots: int, n_qubits: int,
                      best_bitstring: str, approximation_ratio: float) -> None:
    total = sum(counts.values())
    if total != n_shots:
        raise OutputValidationError(
            f"shot conservation violated: sum(counts)={total} != n_shots={n_shots}"
        )
    if len(best_bitstring) != n_qubits:
        raise OutputValidationError(
            f"best_bitstring length {len(best_bitstring)} != n_qubits {n_qubits}"
        )
    if math.isnan(approximation_ratio) or math.isinf(approximation_ratio):
        raise OutputValidationError(
            f"approximation_ratio is not finite: {approximation_ratio}"
        )
    if approximation_ratio < -_RATIO_EPS or approximation_ratio > 1.0 + _RATIO_EPS:
        raise OutputValidationError(
            f"approximation_ratio {approximation_ratio} outside valid range [0, 1]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-method training
# ─────────────────────────────────────────────────────────────────────────────


def _logging_objective(edges, method: str, min_interval_s: float = 2.0):
    """Objective for the Polypus optimizers that logs a heartbeat at most once
    per ``min_interval_s`` seconds (time-throttled so the line count stays
    bounded regardless of the per-bitstring evaluation rate). This closure is
    NOT pickled (Polypus calls it in-process), so mutable state is safe here —
    unlike the scipy path, whose objective must stay picklable."""
    state = {"n": 0, "last": time.time()}

    def obj(bitstring: str) -> float:
        state["n"] += 1
        now = time.time()
        if now - state["last"] >= min_interval_s:
            state["last"] = now
            logger.info("%s: training… %d oracle evaluations", method, state["n"])
        return bitstring_cut(bitstring, edges)

    return obj


def _train_polypus(cfg: RunConfig, prob: Problem):
    """DE / PSO / QNG through polypus.train. Returns (result_params, TrainResult,
    eval_circuit) where eval_circuit is what the final evaluation must bind."""
    optimizer = cfg.optimizer
    objective = _logging_objective(prob.edges, cfg.method)
    common = dict(
        shots=cfg.n_shots,
        n_qpus=cfg.n_qpus,
        dimensions=prob.dimensions,
        expectation_function=objective,
        infrastructure=cfg.infrastructure,
        nodes=cfg.nodes,
        cores_per_qpu=cfg.cores_per_qpu,
        id=_run_id(cfg),
        seed=cfg.seed,
    )

    if optimizer in ("de", "pso"):
        qc = build_qaoa_circuit(
            prob.graph,
            prob.layers,
            [maxcut_cost_layer(prob.graph) for _ in range(prob.layers)],
            [standard_mixer_layer(prob.n_qubits) for _ in range(prob.layers)],
        )
        if optimizer == "de":
            method = polypus.DE(
                generations=prob.max_generations,
                population_size=prob.population_size,
                tolerance=cfg.tolerance,
            )
        else:
            method = polypus.PSO(
                generations=prob.max_generations,
                population_size=prob.population_size,
                bounds=(0.0, np.pi),
                tolerance=cfg.tolerance,
            )
        result = polypus.train(qc, method, **common)
        return result.best_params, result, qc

    # QNG: alternating-parameter circuit + seeded variance callback.
    qc = build_qaoa_circuit_qng(prob.graph, prob.layers)
    variance_fn = make_variance_function(prob.graph, prob.n_qubits, cfg.n_shots, cfg.seed)
    method = polypus.QNG(
        variance_fn,
        max_iters=prob.max_generations,
        bounds=(0.0, np.pi),
        learning_rate=cfg.qng_learning_rate,
        finite_difference_step=cfg.qng_step,
        tikhonov_reg=cfg.qng_tikhonov,
    )
    result = polypus.train(qc, method, **common)
    return result.best_params, result, qc


def _scipy_de_kwargs(seed: int) -> dict:
    """scipy renamed the RNG kwarg (`seed` -> `rng`) around 1.15. Prefer `rng`
    when available to avoid a deprecation warning, else fall back to `seed`."""
    import inspect

    from scipy.optimize import differential_evolution

    params = inspect.signature(differential_evolution).parameters
    if "rng" in params:
        return {"rng": seed}
    return {"seed": seed}


@dataclass
class _ScipyResult:
    best_params: list
    best_fitness: float
    iterations_run: int
    converged: bool
    seed: int


def _train_scipy(cfg: RunConfig, prob: Problem):
    """Pure-scipy differential_evolution baseline ("no Polypus"). Aer sampling
    is seeded (seed_simulator) and DE's RNG is seeded, so the run is
    reproducible; with scipy_workers>1 the picklable top-level objective is
    dispatched across a multiprocessing pool."""
    from multiprocessing import Pool

    from scipy.optimize import differential_evolution

    qc = build_qaoa_circuit(
        prob.graph,
        prob.layers,
        [maxcut_cost_layer(prob.graph) for _ in range(prob.layers)],
        [standard_mixer_layer(prob.n_qubits) for _ in range(prob.layers)],
    )
    bounds = [(0.0, np.pi)] * prob.dimensions
    objective = functools.partial(
        _scipy_objective, qc=qc, edges=prob.edges, n_shots=cfg.n_shots, seed=cfg.seed
    )

    def callback(*args, **kwargs):
        state["gen"] += 1
        logger.info("scipy DE: generation %d", state["gen"])

    state = {"gen": 0}
    de_kwargs = dict(
        maxiter=prob.max_generations,
        popsize=cfg.population_factor,
        polish=False,
        tol=cfg.tolerance,
        callback=callback,
        **_scipy_de_kwargs(cfg.seed),
    )

    pool = None
    try:
        if cfg.scipy_workers > 1:
            pool = Pool(cfg.scipy_workers)
            de_kwargs["workers"] = pool.map
        res = differential_evolution(objective, bounds, **de_kwargs)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    result = _ScipyResult(
        best_params=list(res.x),
        best_fitness=float(-res.fun),
        iterations_run=int(res.nit),
        converged=bool(res.success),
        seed=cfg.seed,
    )
    return result.best_params, result, qc


def _scipy_objective(params, qc, edges, n_shots, seed) -> float:
    """Top-level (picklable) scipy objective: negative expected cut, seeded."""
    bound = qc.assign_parameters(list(params))
    counts = AerSimulator().run(bound, shots=n_shots, seed_simulator=seed).result().get_counts()
    return -mean_energy(counts, edges)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def _run_id(cfg: RunConfig) -> str:
    return f"maxcut_{cfg.n_qubits}q_{cfg.method}_seed{cfg.seed}_r{cfg.repeat_index}"


def run_single(cfg: RunConfig, artifacts: Optional[dict] = None) -> RunRecord:
    """Run and validate ONE QAOA optimization; return a populated RunRecord.

    Raises :class:`InputValidationError` for a bad config/instance and
    :class:`OutputValidationError` if the result violates an invariant — in both
    cases the caller must NOT persist a row.

    If ``artifacts`` is a dict it is populated with ``problem``, ``params``,
    ``counts`` and ``best_bitstring`` so an optional diagnostic plot can reuse
    them instead of re-training.
    """
    if cfg.method not in METHOD_SPECS:
        raise InputValidationError(
            f"unknown method {cfg.method!r}; expected one of {sorted(METHOD_SPECS)}"
        )
    prob = build_problem(cfg)
    logger.info(
        "run start: %s | n_qubits=%d edges=%d layers=%d dims=%d pop=%d gens=%d "
        "shots=%d seed=%d repeat=%d | optimal_cut=%d",
        cfg.method, prob.n_qubits, len(prob.edges),
        prob.layers, prob.dimensions, prob.population_size, prob.max_generations,
        cfg.n_shots, cfg.seed, cfg.repeat_index, prob.best_solution,
    )

    tic = time.time()
    if cfg.optimizer == "scipy":
        params, result, eval_qc = _train_scipy(cfg, prob)
    else:
        params, result, eval_qc = _train_polypus(cfg, prob)
    elapsed = time.time() - tic

    counts, best_bitstring, best_cut, approximation_ratio, best_ratio = _final_evaluation(
        eval_qc, params, prob.edges, cfg.n_shots, cfg.seed, prob.best_solution
    )
    _validate_outputs(counts, cfg.n_shots, prob.n_qubits, best_bitstring, approximation_ratio)

    if artifacts is not None:
        artifacts.update(
            problem=prob, params=params, counts=counts, best_bitstring=best_bitstring,
            best_cut=best_cut, approximation_ratio=approximation_ratio,
        )

    logger.info(
        "run done: %s | ratio=%.5f best_cut=%d/%d best_bitstring=%s time=%.2fs",
        cfg.method, approximation_ratio, best_cut, prob.best_solution, best_bitstring, elapsed,
    )

    return RunRecord(
        schema_version=CSV_SCHEMA_VERSION,
        timestamp=_utc_now(),
        git_commit=git_commit(),
        polypus_version=polypus_version(),
        n_qubits=prob.n_qubits,
        method=cfg.method,
        optimizer=_OPTIMIZER_LABEL[cfg.optimizer],
        infrastructure=cfg.infrastructure,
        backend="aer",
        seed=cfg.seed,
        base_seed=cfg.base_seed,
        repeat_index=cfg.repeat_index,
        graph_seed=cfg.graph_seed,
        graph_prob=cfg.graph_prob,
        n_edges=len(prob.edges),
        layers=prob.layers,
        dimensions=prob.dimensions,
        population_size=prob.population_size,
        max_generations=prob.max_generations,
        n_shots=cfg.n_shots,
        n_qpus=cfg.n_qpus,
        nodes=cfg.nodes,
        cores_per_qpu=cfg.cores_per_qpu,
        best_solution_bruteforce=prob.best_solution,
        best_cut=best_cut,
        best_bitstring=best_bitstring,
        approximation_ratio=approximation_ratio,
        best_ratio=best_ratio,
        best_fitness=float(result.best_fitness),
        iterations_run=int(result.iterations_run),
        converged=bool(result.converged),
        time_seconds=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_record(csv_path: str, record: RunRecord) -> None:
    """Append one row, writing the header if the file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(record.to_row())


def _coerce(field_name: str, value: str):
    """Parse a CSV string cell into its typed value; raise on garbage."""
    int_fields = {
        "schema_version", "n_qubits", "seed", "base_seed", "repeat_index",
        "graph_seed", "n_edges", "layers", "dimensions", "population_size",
        "max_generations", "n_shots", "n_qpus", "nodes", "cores_per_qpu",
        "best_solution_bruteforce", "best_cut", "iterations_run",
    }
    float_fields = {
        "graph_prob", "approximation_ratio", "best_ratio", "best_fitness",
        "time_seconds",
    }
    bool_fields = {"converged"}
    if field_name in int_fields:
        return int(value)
    if field_name in float_fields:
        return float(value)
    if field_name in bool_fields:
        return value.strip().lower() in ("true", "1", "yes")
    return value


def read_records(csv_paths) -> list[dict]:
    """Read and validate one or more accumulated CSVs into typed dict rows.

    Fails loudly (``ExperimentError``) on a missing file, a wrong header, an
    unknown ``schema_version``, or a row that cannot be parsed — corrupt rows are
    never silently dropped.
    """
    if isinstance(csv_paths, (str, os.PathLike)):
        csv_paths = [csv_paths]
    rows: list[dict] = []
    for path in csv_paths:
        if not os.path.isfile(path):
            raise ExperimentError(f"CSV not found: {path}")
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            if header is None or set(header) != set(CSV_FIELDS):
                missing = set(CSV_FIELDS) - set(header or [])
                extra = set(header or []) - set(CSV_FIELDS)
                raise ExperimentError(
                    f"{path}: unexpected CSV header. missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )
            for lineno, raw in enumerate(reader, start=2):
                try:
                    parsed = {k: _coerce(k, raw[k]) for k in CSV_FIELDS}
                except (ValueError, KeyError, TypeError) as exc:
                    raise ExperimentError(
                        f"{path}:{lineno}: corrupt row ({exc}); refusing to drop it"
                    ) from exc
                if parsed["schema_version"] != CSV_SCHEMA_VERSION:
                    raise ExperimentError(
                        f"{path}:{lineno}: schema_version={parsed['schema_version']} "
                        f"!= supported {CSV_SCHEMA_VERSION}"
                    )
                rows.append(parsed)
    if not rows:
        raise ExperimentError(f"no data rows found in: {list(csv_paths)}")
    return rows
