"""
Benchmark: same Aer engine underneath, three call paths, isolating polypus's
wrapper overhead from the simulator engine itself.

`polypus.run_quantum_circuit(..., backend="aer")` does not implement its own
simulator: for `infrastructure="local"` it instantiates `qiskit_aer.AerSimulator`
and calls `.run(...).result()` exactly like plain Qiskit does (see
`packages/polypus_python/polypus_python/local.py`). So this compares:

  1. aer_raw            - qiskit QuantumCircuit -> AerSimulator directly, no polypus
  2. aer_via_polypus_qk  - the same qiskit QuantumCircuit, through
                           polypus.run_quantum_circuit(..., backend="aer")
  3. aer_via_polypus_nv  - a polypus.Circuit (native Rust builder), through
                           polypus.run_quantum_circuit(..., backend="aer")
                           (crosses the FFI boundary as QASM2 text, parsed back
                           into a QuantumCircuit before Aer ever sees it)

All three are shots-based (measure_all + counts), since that's the only mode
`backend="aer"` supports through polypus.run_quantum_circuit — there is no
statevector-only path through that API. Each engine runs in its own
subprocess (see bench_qft.py for why: cross-run interference has been
observed between engines sharing a process).

Usage:
    python benchmarks/bench_qft_aer_backend.py [n_qubits] [--shots N] [--repeat N]
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

import polypus
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

ENGINES = ("aer_raw", "aer_via_polypus_qk", "aer_via_polypus_nv")
LABELS = {
    "aer_raw": "qiskit -> Aer directly (no polypus)",
    "aer_via_polypus_qk": "qiskit circuit -> polypus.run_quantum_circuit(backend=aer)",
    "aer_via_polypus_nv": "native circuit -> polypus.run_quantum_circuit(backend=aer)",
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
    qc.measure_all()
    return qc


def native_qft(n: int) -> "polypus.Circuit":
    qc = polypus.Circuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(k, j, math.pi / 2 ** (k - j))
    qc.measure_all()
    return qc


def _run_isolated_engine(engine: str, n: int, shots: int, repeat: int) -> None:
    """Internal mode: time exactly one engine in this fresh process, print
    a JSON list of per-round seconds to stdout, and exit."""
    if engine == "aer_raw":
        qk = qiskit_qft(n)
        sim = AerSimulator(method="automatic")
        fn = lambda: sim.run(qk, shots=shots, seed_simulator=42).result().get_counts()
    elif engine == "aer_via_polypus_qk":
        qk = qiskit_qft(n)
        fn = lambda: polypus.run_quantum_circuit(
            qk, shots=shots, infrastructure="local", backend="aer", seed=42
        )
    elif engine == "aer_via_polypus_nv":
        nv = native_qft(n)
        fn = lambda: polypus.run_quantum_circuit(
            nv, shots=shots, infrastructure="local", backend="aer", seed=42
        )
    else:
        raise ValueError(engine)

    fn()  # warmup
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(json.dumps(times))


def bench_isolated(engine: str, n: int, shots: int, repeat: int) -> list:
    out = subprocess.run(
        [
            sys.executable,
            __file__,
            "--engine",
            engine,
            str(n),
            "--shots",
            str(shots),
            "--repeat",
            str(repeat),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n_qubits", nargs="?", type=int, default=20)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument(
        "--outfile",
        type=Path,
        default=Path("benchmarks/bench_qft_results.md"),
    )
    p.add_argument("--engine", choices=ENGINES, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    n, shots, repeat = args.n_qubits, args.shots, args.repeat

    if args.engine is not None:
        _run_isolated_engine(args.engine, n, shots, repeat)
        return

    qk = qiskit_qft(n)
    print(f"QFT · {n} qubits · {len(qk.data) - 1} gates (excl. measure_all) · shots={shots}")
    print(f"  load average: {load_average()}")

    stats = {}
    for engine in ENGINES:
        xs = bench_isolated(engine, n, shots, repeat)
        stats[engine] = (min(xs), statistics.median(xs), max(xs))

    print(f"\n  {repeat} rounds per engine, isolated subprocess, ms (min / median / max):")
    for name, (lo, med, hi) in stats.items():
        print(f"  {LABELS[name]:55s} : {lo*1e3:8.3f} / {med*1e3:8.3f} / {hi*1e3:8.3f}")

    base = stats["aer_raw"][1]
    print("\n  overhead vs raw qiskit+Aer (median):")
    for name in ("aer_via_polypus_qk", "aer_via_polypus_nv"):
        print(f"  {LABELS[name]:55s} : {stats[name][1] / base:6.2f}x")

    rows = "\n".join(
        f"| {LABELS[name]} | {lo*1e3:.3f} | {med*1e3:.3f} | {hi*1e3:.3f} |"
        for name, (lo, med, hi) in stats.items()
    )
    report = (
        f"## QFT via Aer backend — {n} qubits, shots={shots}\n"
        f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_ · "
        f"{repeat} rounds per engine, each in its own subprocess · "
        f"load average: {load_average()}\n\n"
        "Isolates polypus's wrapper overhead: all three paths run the identical "
        "qiskit-aer engine underneath.\n\n"
        "| path | min (ms) | median (ms) | max (ms) |\n"
        "|---|---|---|---|\n"
        f"{rows}\n\n"
        f"overhead vs raw qiskit+Aer (median) — qiskit circuit via polypus: "
        f"{stats['aer_via_polypus_qk'][1] / base:.2f}x, "
        f"native circuit via polypus: {stats['aer_via_polypus_nv'][1] / base:.2f}x\n\n"
    )
    with args.outfile.open("a") as f:
        f.write(report)
    print(f"\n  Saved: {args.outfile}")


if __name__ == "__main__":
    main()
