"""
Benchmark: the controlled 1-qubit kernel (issue #216) on the native statevector.

Why it exists: `apply_controlled_1q` in `crates/polypus-sim/src/kernels.rs` —
the kernel behind `cx`, `cy`, `ch`, `csx`, `crx`, `cry`, `crz`, `cu1`, `cu3`,
`cu` and, by decomposition, the multi-controlled gates — used to walk all
`2^(n-1)` target pairs and throw away the half whose control bit is clear. It
now enumerates the `2^(n-2)` control-set pairs directly. That is a
*performance-only* change (the state is bit-for-bit the same, which the kernel's
unit tests check), and docs/ENGINEERING.md §9 asks that the speedup claim be
backed by a benchmark in this directory. This script measures it.

Method — a before/after comparison of two builds. The change *replaces* the
kernel, so unlike `bench_fusion.py` there is no flag that runs both versions
from one binary: the script is run once against a `--release` build of `main`
(`--label baseline`) and once against a `--release` build of the branch
(`--label patched`), on the same machine with the same thread count, and the
two result files are compared. The rebuild in between is an unavoidable
confound (inlining and code layout can move with it), so read small
differences — a few percent — as noise, not signal.

Every circuit is simulated with ``polypus.statevector(qc, fusion=False)``. With
fusion on (the default), a ``cx`` next to 1-qubit gates on the same pair is
composed into one 4×4 matrix and applied by the dense 2-qubit kernel instead —
it would never reach `apply_controlled_1q`. Fusion off sends every controlled
gate through this kernel and only this kernel.

Each (family, n) point runs in its **own subprocess** so a previous, differently
sized point cannot contaminate the next through allocator arena reuse (the
isolation `bench_fusion.py` uses). Inside a point: one untimed warm-up, then
`--reps` timed reps; the reported time is the **median** rep.

Circuits: a Hadamard layer (so amplitudes are non-trivial), then ``--depth``
layers of a controlled-gate ladder ``(q, q+1)`` for ``q = 0 … n-2``. Families:

  * ``cx`` — the most frequent two-qubit gate;
  * ``ch`` — a dense real 2×2 (every entry non-zero, unlike X);
  * ``crx`` — a parametrised rotation (complex 2×2), one angle per layer.

The default qubit counts straddle the parallel threshold (`n` at or above it
takes the rayon path; it is calibrated per machine and printed with the
results), so both the sequential and the parallel branch of the kernel are
measured. The script reads no resource knobs itself; it only records
``RAYON_NUM_THREADS`` and the threshold in the output so a result file is
self-describing. Pin ``RAYON_NUM_THREADS`` to the same value for both runs.

The threshold is obtained through ``polypus.calibrate_parallel_threshold(
force=False)``, the only public way to read it. When the calibration cache has
no valid entry for this machine and thread count, that call **calibrates and
writes** ``~/.cache/polypus`` — even under ``POLYPUS_NO_AUTOCALIBRATE=1``. The
printed threshold says where it came from (``from cache`` / ``calibrated now``);
if the cache could not be written, the workers do not see the measured value
(each one resolves the threshold from the cache and falls back to the built-in
default), and the header flags it so the seq/par split is not mislabelled.

Reported times are **end to end**: one ``polypus.statevector`` call, including
the Hadamard layer, the ``2^n`` allocation and the conversion to NumPy, not just
the controlled gates. ``ns_per_gate_amp`` is that end-to-end median divided by
``controlled gates · 2^n`` — a size-normalised figure for comparing points and
builds, not the kernel's isolated per-gate cost.

Usage:
    # on a release build of main:
    RAYON_NUM_THREADS=8 python benchmarks/bench_controlled.py \
        --label baseline --out /tmp/controlled_baseline
    # on a release build of the branch:
    RAYON_NUM_THREADS=8 python benchmarks/bench_controlled.py \
        --label patched --out /tmp/controlled_patched
    diff -y /tmp/controlled_baseline.csv /tmp/controlled_patched.csv

    # quick local smoke test:
    python benchmarks/bench_controlled.py --smoke

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

FAMILIES = ("cx", "ch", "crx")


def build_circuit(family: str, n: int, depth: int):
    """Build one benchmark circuit with the public `polypus` API."""
    import polypus

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")

    qc = polypus.Circuit(n)
    for q in range(n):
        qc = qc.h(q)
    for layer in range(depth):
        for q in range(n - 1):
            if family == "cx":
                qc = qc.cx(q, q + 1)
            elif family == "ch":
                qc = qc.ch(q, q + 1)
            else:
                qc = qc.crx(q, q + 1, 0.3 + 0.01 * layer)
    return qc


def run_point(family: str, n: int, depth: int, reps: int) -> dict:
    """Time one (family, n) point in this process and report its median."""
    import polypus

    qc = build_circuit(family, n, depth)
    polypus.statevector(qc, fusion=False)  # warm-up, not timed

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        polypus.statevector(qc, fusion=False)
        times.append(time.perf_counter() - t0)

    gates = depth * (n - 1)
    t = statistics.median(times)
    return {
        "family": family,
        "n": n,
        "depth": depth,
        "reps": reps,
        "controlled_gates": gates,
        "t_median_s": t,
        "ns_per_gate_amp": t / (gates * (1 << n)) * 1e9,
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


def _threshold() -> tuple[int, str]:
    """The gate-parallel threshold for this machine and where it came from.

    `calibrate_parallel_threshold` is the only public way to read it, and it
    calibrates (and writes the cache) when there is no valid entry; see the
    module docstring. Its flags tell the three outcomes apart, and the last one
    matters: a measured value that was not cached is *not* what the workers
    resolve, so the header must not present it as their threshold."""
    import polypus

    info = polypus.calibrate_parallel_threshold(force=False)
    if info["reused_cache"]:
        source = "from cache"
    elif info["cache_written"]:
        source = "calibrated now"
    else:
        source = "measured but NOT cached: workers use the built-in default"
    return info["threshold"], source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="", help="tag printed with the results")
    ap.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[10, 14, 18, 20],
        help="qubit counts to sweep (default: 10 14 18 20)",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES),
        choices=FAMILIES,
        help="controlled-gate families to run (default: all)",
    )
    ap.add_argument("--reps", type=int, default=7, help="timed reps (median)")
    ap.add_argument(
        "--depth", type=int, default=10, help="ladder layers per circuit (default: 10)"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write CSV + JSON here (prefix)"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny run to check the script works"
    )
    args = ap.parse_args()

    if args.smoke:
        args.qubits = [8, 12]
        args.reps = min(args.reps, 3)
        args.depth = min(args.depth, 2)

    threads = os.environ.get("RAYON_NUM_THREADS", "<unset: rayon uses all cores>")
    threshold, threshold_source = _threshold()
    tag = f" [{args.label}]" if args.label else ""
    print(
        f"# controlled 1-qubit kernel benchmark{tag}  "
        f"(RAYON_NUM_THREADS={threads}, "
        f"parallel threshold={threshold} ({threshold_source}))"
    )
    print(
        f"# {'family':6} {'n':>3} {'depth':>5} {'gates':>6} "
        f"{'t_median_s':>11} {'ns/gate/amp':>12}"
    )

    rows = []
    for family in args.families:
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
                str(args.depth),
                "--reps",
                str(args.reps),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(
                    f"! {family} n={n} FAILED:\n{proc.stderr.strip()}", file=sys.stderr
                )
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(
                f"  {row['family']:6} {row['n']:>3} {row['depth']:>5} "
                f"{row['controlled_gates']:>6} {row['t_median_s']:>11.5f} "
                f"{row['ns_per_gate_amp']:>12.3f}"
            )

    if args.out is not None:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json_path = out.with_suffix(".json")
        csv_path = out.with_suffix(".csv")
        json_path.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "rayon_num_threads": threads,
                    "parallel_threshold": threshold,
                    "parallel_threshold_source": threshold_source,
                    "rows": rows,
                },
                indent=2,
            )
        )
        with csv_path.open("w") as f:
            f.write("family,n,depth,reps,controlled_gates,t_median_s,ns_per_gate_amp\n")
            for r in rows:
                f.write(
                    f"{r['family']},{r['n']},{r['depth']},{r['reps']},"
                    f"{r['controlled_gates']},{r['t_median_s']:.6f},"
                    f"{r['ns_per_gate_amp']:.4f}\n"
                )
        print(f"# wrote {csv_path} and {json_path}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker())
    raise SystemExit(main())
