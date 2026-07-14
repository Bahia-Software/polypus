#!/usr/bin/env python
"""Run ONE MaxCut–QAOA optimization and append a single row to the results CSV.

This is the *run* stage: it executes exactly one (qubits, method, seed)
combination and records it. It never plots by default and never aggregates —
the :mod:`report_maxcut` stage does that, independently, from the accumulated
CSV. The :mod:`sweep_maxcut` orchestrator calls this once per combination.

Reproducibility: pass ``--seed``. The same seed reproduces the run
byte-for-byte (same ``approximation_ratio`` and ``best_bitstring``) because
every Aer sampling site is seeded. If ``--seed`` is omitted a fresh seed is
drawn from OS entropy and logged so the run can still be replayed.

Examples::

    python examples/max_cut/run_maxcut.py --qubits 4 --method polypus_local --seed 42
    python examples/max_cut/run_maxcut.py --qubits 5 --method scipy --seed 7 --scipy-workers 4
"""

import argparse
import logging
import os
import secrets
import sys

# When run as a script, this file's directory is sys.path[0], so the sibling
# core module imports directly; the same works from tests that add it to path.
import maxcut_lib as mc

import polypus


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one MaxCut-QAOA optimization and append a CSV row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qubits", type=int, required=True, help="Number of qubits / graph nodes (>= 2).")
    p.add_argument("--method", required=True, choices=mc.ALL_METHODS,
                   help="Optimizer + infrastructure. CUNQA variants are for manual use only.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optimizer + sampling seed. Omit to draw (and log) a fresh entropy seed.")
    p.add_argument("--repeat-index", type=int, default=0, help="Repetition index (CSV provenance).")
    p.add_argument("--base-seed", type=int, default=None,
                   help="Sweep base seed this run derives from (CSV provenance).")
    p.add_argument("--shots", type=int, default=10_000, help="Shots per circuit evaluation.")
    p.add_argument("--csv", default=mc.DEFAULT_CSV, help="Results CSV to append to.")
    p.add_argument("--n-qpus", type=int, default=1, help="Number of (virtual) QPUs.")
    p.add_argument("--nodes", type=int, default=1, help="CUNQA node count (manual CUNQA runs only).")
    p.add_argument("--cores-per-qpu", type=int, default=1, help="CUNQA cores per QPU (manual only).")
    p.add_argument("--scipy-workers", type=int, default=1,
                   help="scipy baseline worker processes (>1 uses a multiprocessing pool).")
    p.add_argument("--graph-seed", type=int, default=7, help="Seed for the MaxCut problem graph.")
    p.add_argument("--graph-prob", type=float, default=0.6, help="Edge probability of the graph.")
    p.add_argument("--layers-factor", type=int, default=2, help="QAOA depth = (n//2) * this.")
    p.add_argument("--population-factor", type=int, default=4, help="Population = dimensions * this.")
    p.add_argument("--generations-factor", type=int, default=5, help="Generations = n_qubits * this.")
    p.add_argument("--tolerance", type=float, default=1e-5, help="Optimizer convergence tolerance.")
    p.add_argument("--qng-learning-rate", type=float, default=0.1)
    p.add_argument("--qng-step", type=float, default=0.1)
    p.add_argument("--qng-tikhonov", type=float, default=0.05)
    p.add_argument("--plot", action="store_true",
                   help="Also write a per-run 3-panel figure (off by default; not the report).")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"], help="Log verbosity.")
    return p


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Route Polypus' internal Rust log records to the console at the same level.
    try:
        polypus.init_logger(level=level, console=True)
    except Exception as exc:  # a second init in-process is a no-op; log, do not hide.
        logging.getLogger("maxcut").debug("polypus.init_logger skipped: %s", exc)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    log = logging.getLogger("maxcut")

    seed = args.seed
    if seed is None:
        seed = secrets.randbits(32)
        log.warning("no --seed given; generated seed=%d (record it to reproduce this run)", seed)

    cfg = mc.RunConfig(
        n_qubits=args.qubits,
        method=args.method,
        seed=seed,
        repeat_index=args.repeat_index,
        base_seed=args.base_seed if args.base_seed is not None else seed,
        n_shots=args.shots,
        n_qpus=args.n_qpus,
        nodes=args.nodes,
        cores_per_qpu=args.cores_per_qpu,
        graph_seed=args.graph_seed,
        graph_prob=args.graph_prob,
        layers_factor=args.layers_factor,
        population_factor=args.population_factor,
        generations_factor=args.generations_factor,
        tolerance=args.tolerance,
        scipy_workers=args.scipy_workers,
        qng_learning_rate=args.qng_learning_rate,
        qng_step=args.qng_step,
        qng_tikhonov=args.qng_tikhonov,
    )

    artifacts: dict = {} if args.plot else None
    try:
        record = mc.run_single(cfg, artifacts=artifacts)
    except mc.ExperimentError as exc:
        # Expected, explained failure (bad input / violated output invariant):
        # do NOT write a row; surface it via a clear message and non-zero exit.
        log.error("run aborted (%s): %s", type(exc).__name__, exc)
        return 1

    mc.append_record(args.csv, record)
    log.info(
        "appended row to %s: method=%s n_qubits=%d ratio=%.5f time=%.2fs",
        args.csv, record.method, record.n_qubits, record.approximation_ratio, record.time_seconds,
    )

    if args.plot:
        _plot_single(cfg, record, artifacts, args.csv)

    return 0


def _plot_single(cfg: mc.RunConfig, record: mc.RunRecord, artifacts: dict, csv_path: str) -> None:
    """Optional single-run diagnostic figure (graph + best cut + counts), reusing
    the already-computed artifacts (no re-training)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    prob = artifacts["problem"]
    counts = artifacts["counts"]
    best_bitstring = artifacts["best_bitstring"]

    pos = nx.spring_layout(prob.graph, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    nx.draw(prob.graph, pos, ax=axes[0], with_labels=True, node_color="lightgray",
            edge_color="black", node_size=800)
    axes[0].set_title("Graph")
    b = best_bitstring[::-1]
    colors = ["#d1495b" if b[i] == "0" else "#2e86ab" for i in range(prob.n_qubits)]
    nx.draw(prob.graph, pos, ax=axes[1], with_labels=True, node_color=colors,
            node_size=800, font_color="white")
    cut_edges = [(i, j) for i, j in prob.graph.edges() if b[i] != b[j]]
    nx.draw_networkx_edges(prob.graph, pos, ax=axes[1], edgelist=cut_edges, width=4, edge_color="#e9c46a")
    axes[1].set_title(
        f"{record.optimizer} cut={record.best_cut}/{prob.best_solution}  t={record.time_seconds:.2f}s"
    )
    top = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:16])
    axes[2].bar(list(top.keys()), list(top.values()), color="#2e86ab")
    axes[2].set_title("Top counts")
    axes[2].tick_params(axis="x", rotation=90)
    fig.suptitle(f"MaxCut {record.method}  ratio={record.approximation_ratio:.4f}  seed={cfg.seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "single_run_plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{mc._run_id(cfg)}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logging.getLogger("maxcut").info("wrote per-run figure: %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
