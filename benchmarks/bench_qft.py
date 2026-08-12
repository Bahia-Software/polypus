"""
Benchmark: n-qubit Quantum Fourier Transform, Qiskit Aer (statevector method)
vs polypus's native Rust statevector simulator (`polypus.statevector`).

Both circuits are built gate-for-gate from the same textbook QFT construction
(H + controlled-phase, no swap network) so the comparison is apples-to-apples:
same qubit count, same gate count, same gate types.

Each engine is timed in its own subprocess, to avoid any cross-run
interference (e.g. thread-pool/affinity state left behind by one library
affecting the next call in the same process).

Usage:
    python benchmarks/bench_qft.py [n_qubits] [--repeat N] [--outfile PATH]
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polypus
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

ENGINES = ("qiskit_aer", "polypus")
LABELS = {
    "qiskit_aer": "qiskit (Aer statevector)",
    "polypus": "polypus (native Rust)   ",
}


def load_average() -> str:
    try:
        with open("/proc/loadavg") as f:
            one, five, fifteen = f.read().split()[:3]
        return f"{one} {five} {fifteen} (1/5/15 min)"
    except OSError:
        return "unavailable"


def qiskit_qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / 2 ** (k - j), k, j)
    return qc


def native_qft(n: int) -> "polypus.Circuit":
    qc = polypus.Circuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(k, j, math.pi / 2 ** (k - j))
    return qc


def _run_isolated_engine(engine: str, n: int, repeat: int) -> None:
    """Internal mode: time exactly one engine in this fresh process, print
    a JSON list of per-round seconds to stdout, and exit."""
    if engine == "qiskit_aer":
        qk = qiskit_qft(n)
        qk.save_statevector()
        aer = AerSimulator(method="statevector")
        fn = lambda: aer.run(qk).result()
    elif engine == "polypus":
        nv = native_qft(n)
        fn = lambda: polypus.statevector(nv)
    else:
        raise ValueError(engine)

    fn()  # warmup
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(json.dumps(times))


def bench_isolated(engine: str, n: int, repeat: int) -> list:
    """Run `engine`'s timing loop in a brand-new subprocess so it can't be
    affected by (or accidentally taint) any other engine's run in-process."""
    out = subprocess.run(
        [sys.executable, __file__, "--engine", engine, str(n), "--repeat", str(repeat)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n_qubits", nargs="?", type=int, default=20)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument(
        "--outfile",
        type=Path,
        default=Path("benchmarks/bench_qft_results.md"),
        help="Markdown file to append the results to "
        "(default: benchmarks/bench_qft_results.md)",
    )
    p.add_argument(
        "--engine",
        choices=ENGINES,
        default=None,
        help=argparse.SUPPRESS,  # internal: run a single engine in isolation
    )
    args = p.parse_args()
    n, repeat = args.n_qubits, args.repeat

    if args.engine is not None:
        _run_isolated_engine(args.engine, n, repeat)
        return

    qk = qiskit_qft(n)
    nv = native_qft(n)

    # ── Correctness check: same amplitudes before trusting the timing ────────
    qk_ref = qk.copy()
    qk_ref.save_statevector()
    aer_ref = AerSimulator(method="statevector")
    sv_aer = np.asarray(aer_ref.run(qk_ref).result().get_statevector().data)
    sv_native = np.asarray(polypus.statevector(nv))
    max_err = np.max(np.abs(sv_aer - sv_native))
    print(f"QFT · {n} qubits · {len(qk.data)} gates")
    print(f"  max amplitude error (aer vs native): {max_err:.2e}")
    print(f"  load average: {load_average()}")
    assert max_err < 1e-9, "statevectors disagree — not a fair benchmark"

    # ── Timing: each engine in its own subprocess (see module docstring) ────
    stats = {}
    for engine in ENGINES:
        xs = bench_isolated(engine, n, repeat)
        stats[engine] = (min(xs), statistics.median(xs), max(xs))

    print(f"\n  {repeat} rounds per engine, isolated subprocess, ms (min / median / max):")
    for name, (lo, med, hi) in stats.items():
        print(
            f"  {LABELS[name]} : {lo * 1e3:8.3f} / {med * 1e3:8.3f} / {hi * 1e3:8.3f}"
        )

    med_native = stats["polypus"][1]
    print(f"\n  speedup vs Aer statevector (median): {stats['qiskit_aer'][1] / med_native:6.2f}x")

    # ── Save results ───────────────────────────────────────────────────────
    rows = "\n".join(
        f"| {LABELS[name].strip()} "
        f"| {lo * 1e3:.3f} | {med * 1e3:.3f} | {hi * 1e3:.3f} |"
        for name, (lo, med, hi) in stats.items()
    )
    report = (
        f"## QFT benchmark — {n} qubits ({len(qk.data)} gates)\n"
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_ · "
        f"{repeat} rounds per engine, each in its own subprocess · "
        f"load average: {load_average()}\n\n"
        f"max amplitude error (aer vs native): {max_err:.2e}\n\n"
        "| engine | min (ms) | median (ms) | max (ms) |\n"
        "|---|---|---|---|\n"
        f"{rows}\n\n"
        f"speedup vs Aer statevector (median): {stats['qiskit_aer'][1] / med_native:.2f}x\n\n"
    )
    with args.outfile.open("a") as f:
        f.write(report)
    print(f"\n  Saved: {args.outfile}")


if __name__ == "__main__":
    main()
