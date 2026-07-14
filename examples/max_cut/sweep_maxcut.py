#!/usr/bin/env python
"""Orchestrate the MaxCut–QAOA sweep: qubits × methods × repeats.

Replaces the old ``experiment_max_cut.sh``. Unlike bash it validates arguments,
runs each combination as an isolated subprocess (a crash in one run cannot abort
the sweep), keeps going when a combination fails, and prints a clear summary of
what failed and why — exiting non-zero if anything did.

The whole sweep is reproducible from a single base ``--seed``: repetition ``r``
of every (qubits, method) uses ``seed = base_seed + r``. A run manifest records
the base seed, git commit, Polypus version and timestamp so a report can cite
exactly where its data came from.

Examples::

    # Preview the plan (run count + rough time), without executing anything:
    python examples/max_cut/sweep_maxcut.py --qubits 4 5 6 7 --repeats 5 --dry-run

    # Small, fast sweep:
    python examples/max_cut/sweep_maxcut.py --qubits 4 5 --repeats 2 --shots 2000
"""

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import maxcut_lib as mc

log = logging.getLogger("maxcut.sweep")

RUN_SCRIPT = os.path.join(mc.PACKAGE_DIR, "run_maxcut.py")


@dataclass(frozen=True)
class Combo:
    n_qubits: int
    method: str
    repeat_index: int
    seed: int

    @property
    def label(self) -> str:
        return f"q{self.n_qubits}_{self.method}_r{self.repeat_index}_seed{self.seed}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep MaxCut-QAOA over qubits × methods × repeats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qubits", type=int, nargs="+", default=[4, 5, 6, 7],
                   help="Qubit counts to sweep (each >= 2).")
    p.add_argument("--methods", nargs="+", default=list(mc.DEFAULT_SWEEP_METHODS),
                   choices=mc.ALL_METHODS,
                   help="Methods to sweep. Default: the 3 local Polypus optimizers + scipy baseline.")
    p.add_argument("--repeats", type=int, default=10, help="Repetitions per (qubits, method).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; repetition r uses seed = base_seed + r.")
    p.add_argument("--shots", type=int, default=10_000, help="Shots per circuit evaluation.")
    p.add_argument("--csv", default=mc.DEFAULT_CSV, help="Results CSV (rows are appended).")
    p.add_argument("--scipy-workers", type=int, default=1, help="scipy baseline worker processes.")
    p.add_argument("--n-qpus", type=int, default=1, help="Number of (virtual) QPUs per run.")
    p.add_argument("--graph-seed", type=int, default=7,
                   help="Seed for the problem graph (same for every run → same instance per qubit count).")
    p.add_argument("--graph-prob", type=float, default=0.6, help="Edge probability of the problem graph.")
    p.add_argument("--fresh", action="store_true",
                   help="Delete the CSV before starting (otherwise rows accumulate).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (run count + rough time estimate) and exit without running.")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for run subprocesses.")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return p


def _plan(args) -> list[Combo]:
    combos: list[Combo] = []
    for q in args.qubits:
        for method in args.methods:
            for r in range(args.repeats):
                combos.append(Combo(n_qubits=q, method=method, repeat_index=r, seed=args.seed + r))
    return combos


def _rough_seconds(method: str, n_qubits: int, shots: int) -> float:
    """Order-of-magnitude per-run time estimate for the plan preview only.

    Anchored on local timings (~1 s at 4 qubits / 2000 shots); scales linearly
    with shots and roughly doubles per extra qubit (statevector cost). This is a
    rough guide, not a benchmark — the printed total is labelled approximate.
    """
    base = {"scipy": 1.2, "de": 0.6, "pso": 0.6, "qng": 0.9}[mc.METHOD_SPECS[method][0]]
    return base * (shots / 2000.0) * (2.0 ** (n_qubits - 4))


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_plan(args, combos: list[Combo]) -> None:
    est = sum(_rough_seconds(c.method, c.n_qubits, args.shots) for c in combos)
    print("=" * 70)
    print("MaxCut-QAOA sweep plan")
    print("=" * 70)
    print(f"  qubits          : {args.qubits}")
    print(f"  methods         : {args.methods}")
    print(f"  repeats         : {args.repeats}  (seeds {args.seed}..{args.seed + args.repeats - 1})")
    print(f"  shots           : {args.shots}")
    print(f"  total runs      : {len(combos)}")
    print(f"  rough est. time : ~{_fmt_duration(est)} (approximate; scipy_workers>1 will speed up)")
    print(f"  csv             : {args.csv}")
    print("=" * 70)


def _write_manifest(args, combos: list[Combo]) -> str:
    manifest = {
        "created": mc._utc_now(),
        "base_seed": args.seed,
        "git_commit": mc.git_commit(),
        "polypus_version": mc.polypus_version(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "qubits": args.qubits,
        "methods": args.methods,
        "repeats": args.repeats,
        "shots": args.shots,
        "n_qpus": args.n_qpus,
        "scipy_workers": args.scipy_workers,
        "csv": os.path.abspath(args.csv),
        "total_runs": len(combos),
        "combos": [
            {"n_qubits": c.n_qubits, "method": c.method,
             "repeat_index": c.repeat_index, "seed": c.seed}
            for c in combos
        ],
    }
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"maxcut_manifest_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _run_combo(args, combo: Combo, log_dir: str) -> tuple[int, str]:
    """Execute one combination as a subprocess. Returns (returncode, log_path)."""
    argv = [
        args.python, RUN_SCRIPT,
        "--qubits", str(combo.n_qubits),
        "--method", combo.method,
        "--seed", str(combo.seed),
        "--base-seed", str(args.seed),
        "--repeat-index", str(combo.repeat_index),
        "--shots", str(args.shots),
        "--csv", args.csv,
        "--n-qpus", str(args.n_qpus),
        "--scipy-workers", str(args.scipy_workers),
        "--graph-seed", str(args.graph_seed),
        "--graph-prob", str(args.graph_prob),
        "--log-level", args.log_level,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    log_path = os.path.join(log_dir, f"{combo.label}.log")
    with open(log_path, "w") as fh:
        fh.write(f"$ {' '.join(argv)}\n\n=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}")
    return proc.returncode, log_path


def _tail(path: str, n: int = 12) -> str:
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(log unavailable)"


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bad = [q for q in args.qubits if q < 2]
    if bad:
        print(f"error: --qubits must all be >= 2; got invalid {bad}", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print(f"error: --repeats must be >= 1; got {args.repeats}", file=sys.stderr)
        return 2

    combos = _plan(args)
    _print_plan(args, combos)
    if args.dry_run:
        return 0

    if args.fresh and os.path.isfile(args.csv):
        os.remove(args.csv)
        log.info("removed existing CSV (--fresh): %s", args.csv)

    manifest_path = _write_manifest(args, combos)
    log.info("wrote run manifest: %s", manifest_path)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log.info("per-run logs: %s  (tail -f a run's .log for live training progress)", log_dir)

    failures: list[tuple[Combo, str]] = []
    started = time.time()
    for i, combo in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] running {combo.label} …", flush=True)
        tic = time.time()
        rc, log_path = _run_combo(args, combo, log_dir)
        dt = time.time() - tic
        if rc == 0:
            print(f"[{i}/{len(combos)}] OK   {combo.label}  ({dt:.1f}s)", flush=True)
        else:
            print(f"[{i}/{len(combos)}] FAIL {combo.label}  (rc={rc}, {dt:.1f}s) — see {log_path}",
                  flush=True)
            failures.append((combo, log_path))

    total_dt = time.time() - started
    _print_summary(combos, failures, total_dt, manifest_path)
    return 1 if failures else 0


def _print_summary(combos, failures, total_dt, manifest_path) -> None:
    print("=" * 70)
    print("Sweep summary")
    print("=" * 70)
    print(f"  total     : {len(combos)}")
    print(f"  succeeded : {len(combos) - len(failures)}")
    print(f"  failed    : {len(failures)}")
    print(f"  wall time : {_fmt_duration(total_dt)}")
    print(f"  manifest  : {manifest_path}")
    if failures:
        print("-" * 70)
        print("FAILURES (reason = tail of the run log):")
        for combo, log_path in failures:
            print(f"\n  • {combo.label}")
            for line in _tail(log_path).splitlines():
                print(f"      {line}")
    print("=" * 70)
    if failures:
        print(f"{len(failures)} combination(s) failed — exit code 1.")
    else:
        print("All combinations succeeded.")


if __name__ == "__main__":
    raise SystemExit(main())
