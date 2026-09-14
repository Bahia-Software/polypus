"""
Benchmark: gate fusion (issues #131 and #132) on the native statevector.

Why it exists: fusion is a *performance-only* change — `fusion=False` and
`fusion=True` compute the same state (verified elsewhere), fusion only reduces
how many full-buffer passes a gate sequence costs. docs/ENGINEERING.md §9 asks
that any such claim be backed by a benchmark in this directory, and the fused
machinery in `crates/polypus-sim/src/simulator.rs` is substantial, so the
speedup is also what justifies that complexity. This script measures it.

Method — the fair, confound-free comparison. Both arms come from the *same*
installed build: `polypus.statevector(qc, fusion=False)` is exactly the
pre-fusion gate-by-gate path, `fusion=True` is the fused one. Same binary, same
circuit, same node, one flag flipped — no rebuild between arms (which would drag
in compiler/inlining differences). The speedup reported is `t_unfused /
t_fused`.

Each (family, n) point runs in its **own subprocess** so a previous, differently
sized point cannot contaminate the next through allocator arena reuse (the same
isolation `bench_qft_memory.py` uses). Inside a point: a shared untimed warm-up,
then `--reps` timed reps per arm, interleaved (which arm goes first alternates
each rep) so slow drift hits both arms equally; the reported time is the
**median** rep.

Families, each chosen to exercise one fusion mechanism:

  * ``qft`` — the production QFT template (``polypus.circuits.templates.qft``), a
    real-world circuit whose phase columns are runs of diagonal ``cp`` gates: the
    diagonal-run fusion of #131.
  * ``trotter`` — a transverse-field-Ising-style cost layer: one Hadamard layer
    (so amplitudes are non-trivial), then ``--depth`` layers of ``rz`` on every
    qubit and ``rzz`` on a neighbour ring. A deep, purely diagonal stretch —
    #131's target (a QAOA cost layer / Trotterized ZZ·RZ evolution) — where the
    per-gate-pass vs one-fused-pass difference is largest.
  * ``hea`` — a hardware-efficient ansatz: ``--depth`` layers of ``rx``+``ry`` on
    every qubit then a ``cx`` brickwork. The rotations fuse with the ``cx`` they
    share a qubit with into one composed 2-qubit matrix: the connected-component
    fusion of #132.

Fairness on SLURM (see bench_fusion.sbatch): run the whole sweep inside one
allocation so both arms of every point are on the identical node/CPUs, pin the
thread count (``RAYON_NUM_THREADS``) so the parallel kernels use the same width
in both arms, and prefer ``--exclusive`` so nothing else on the node perturbs
the timings. This script reads no resource knobs itself; it only records
``RAYON_NUM_THREADS`` into the output so a result file is self-describing.

Usage:
    RAYON_NUM_THREADS=16 python benchmarks/bench_fusion.py \
        --qubits 16 18 20 22 --out benchmarks/fusion_results

    # quick local smoke test:
    python benchmarks/bench_fusion.py --qubits 12 14 --reps 3 --smoke

Needs the extension built in **release** (perf claims are about release):
    CONDA_PREFIX=$CONDA_PREFIX maturin develop --release --features extension-module
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Per-family depth defaults, sized so a point finishes in seconds at the top of
# the default qubit range while still stacking enough passes that fusion's saved
# traffic is visible (fusion saves passes in proportion to depth). `qft` has no
# depth knob — its size is fixed by n.
_DEFAULT_DEPTH = {"trotter": 16, "hea": 8}

FAMILIES = ("qft", "trotter", "hea")


def build_circuit(family: str, n: int, depth: int):
    """Build one benchmark circuit with the public `polypus` API."""
    import polypus
    from polypus.circuits import templates

    if family == "qft":
        # The production template, as end users call it: O(n^2) diagonal `cp`
        # gates in phase-column runs, the #131 diagonal-run target.
        return templates.qft(n, inverse=False, swaps=True)

    if family == "trotter":
        qc = polypus.Circuit(n)
        for q in range(n):
            qc = qc.h(q)
        for layer in range(depth):
            for q in range(n):
                qc = qc.rz(q, 0.1 + 0.01 * layer)
            for q in range(n):
                qc = qc.rzz(q, (q + 1) % n, 0.2 - 0.005 * layer)
        return qc

    if family == "hea":
        qc = polypus.Circuit(n)
        for layer in range(depth):
            for q in range(n):
                qc = qc.rx(q, 0.3 + 0.01 * layer).ry(q, -0.2 + 0.02 * q)
            for q in range(layer % 2, n - 1, 2):
                qc = qc.cx(q, q + 1)
        return qc

    raise ValueError(f"unknown family {family!r}")


def run_point(family: str, n: int, depth: int, reps: int) -> dict:
    """Time both arms of one (family, n) point in this process. Interleaves the
    arms so any drift is shared, then reports each arm's median."""
    import polypus

    qc = build_circuit(family, n, depth)
    polypus.statevector(qc, fusion=False)  # shared warm-up, not timed
    polypus.statevector(qc, fusion=True)

    unfused, fused = [], []
    for i in range(reps):
        # Alternate which arm is timed first so a warming CPU never
        # systematically favours one arm.
        order = (False, True) if i % 2 == 0 else (True, False)
        for use_fusion in order:
            t0 = time.perf_counter()
            polypus.statevector(qc, fusion=use_fusion)
            dt = time.perf_counter() - t0
            (fused if use_fusion else unfused).append(dt)

    t_unfused = statistics.median(unfused)
    t_fused = statistics.median(fused)
    return {
        "family": family,
        "n": n,
        "depth": depth,
        "reps": reps,
        "t_unfused_s": t_unfused,
        "t_fused_s": t_fused,
        "speedup": (t_unfused / t_fused) if t_fused > 0 else float("nan"),
    }


def _worker() -> int:
    """Subprocess entry: run one point and print its result as JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    args = ap.parse_args(sys.argv[2:])
    print(json.dumps(run_point(args.family, args.n, args.depth, args.reps)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[16, 18, 20, 22],
        help="qubit counts to sweep (default: 16 18 20 22)",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES),
        choices=FAMILIES,
        help="circuit families to run (default: all)",
    )
    ap.add_argument("--reps", type=int, default=5, help="timed reps per arm (median)")
    ap.add_argument(
        "--depth",
        type=int,
        default=None,
        help="layers for trotter/hea (default: per-family, 16/8; ignored by qft)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write CSV + JSON here (prefix)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the script works")
    args = ap.parse_args()

    if args.smoke:
        args.reps = min(args.reps, 3)

    threads = os.environ.get("RAYON_NUM_THREADS", "<unset: rayon uses all cores>")
    print(f"# gate-fusion benchmark  (RAYON_NUM_THREADS={threads})")
    print(
        f"# {'family':8} {'n':>3} {'depth':>5} {'t_unfused_s':>12} "
        f"{'t_fused_s':>10} {'speedup':>8}"
    )

    rows = []
    for family in args.families:
        depth = args.depth if args.depth is not None else _DEFAULT_DEPTH.get(family, 1)
        for n in args.qubits:
            # Each point in its own subprocess: a fresh allocator per point, so a
            # bigger earlier point cannot skew a smaller later one.
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--family",
                family,
                "--n",
                str(n),
                "--depth",
                str(depth),
                "--reps",
                str(args.reps),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"! {family} n={n} FAILED:\n{proc.stderr.strip()}", file=sys.stderr)
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(
                f"  {row['family']:8} {row['n']:>3} {row['depth']:>5} "
                f"{row['t_unfused_s']:>12.4f} {row['t_fused_s']:>10.4f} "
                f"{row['speedup']:>7.2f}x"
            )

    if args.out is not None:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json_path = out.with_suffix(".json")
        csv_path = out.with_suffix(".csv")
        json_path.write_text(json.dumps({"rayon_num_threads": threads, "rows": rows}, indent=2))
        with csv_path.open("w") as f:
            f.write("family,n,depth,reps,t_unfused_s,t_fused_s,speedup\n")
            for r in rows:
                f.write(
                    f"{r['family']},{r['n']},{r['depth']},{r['reps']},"
                    f"{r['t_unfused_s']:.6f},{r['t_fused_s']:.6f},{r['speedup']:.4f}\n"
                )
        print(f"# wrote {csv_path} and {json_path}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker())
    raise SystemExit(main())
