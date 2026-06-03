#!/usr/bin/env python3
"""
Polypus system benchmark.

Measures circuit execution time and peak RAM for a GHZ circuit across
different n_qpus / shots combinations, using the local AerSimulator backend.

Usage
-----
  python benchmarks/run_benchmarks.py                         # defaults
  python benchmarks/run_benchmarks.py --quick                 # fast sweep
  python benchmarks/run_benchmarks.py --qpus 1 2 4 8
  python benchmarks/run_benchmarks.py --shots 500 2000
  python benchmarks/run_benchmarks.py --qubits 6
  python benchmarks/run_benchmarks.py --output results.csv
"""

import argparse
import csv
import statistics
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

# ── Try importing polypus ─────────────────────────────────────────────────────
try:
    import polypus
except ImportError:
    print("[polypus-bench] ERROR: polypus is not installed.")
    print("  Run 'bash install.sh' first.")
    sys.exit(1)

try:
    from qiskit import QuantumCircuit
except ImportError:
    print("[polypus-bench] ERROR: qiskit is not installed.")
    sys.exit(1)


# ── Circuit factory ───────────────────────────────────────────────────────────
def make_ghz(n_qubits: int) -> QuantumCircuit:
    """Return an n-qubit GHZ circuit with measurements."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


# ── Single timed + traced run ─────────────────────────────────────────────────
def _run_once(qc: QuantumCircuit, shots: int, n_qpus: int) -> tuple[float, float]:
    """Return (elapsed_seconds, peak_ram_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    polypus.run_quantum_circuit(qc, shots=shots, infrastructure="local", n_qpus=n_qpus)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024 / 1024  # bytes → MB


# ── Benchmark sweep ───────────────────────────────────────────────────────────
def run_sweep(
    qpus_list: list[int],
    shots_list: list[int],
    n_qubits: int,
    repeats: int,
) -> list[dict]:
    qc = make_ghz(n_qubits)
    results = []

    total = len(qpus_list) * len(shots_list)
    done = 0
    for shots in shots_list:
        for n_qpus in qpus_list:
            done += 1
            print(f"  [{done}/{total}] n_qpus={n_qpus:>2}  shots={shots:<6}", end="", flush=True)

            times, rams = [], []
            for _ in range(repeats):
                t, r = _run_once(qc, shots, n_qpus)
                times.append(t)
                rams.append(r)

            mean_t = statistics.mean(times)
            mean_r = statistics.mean(rams)
            throughput = shots / mean_t

            results.append({
                "n_qubits":    n_qubits,
                "n_qpus":      n_qpus,
                "shots":       shots,
                "time_mean_s": round(mean_t, 4),
                "time_std_s":  round(statistics.stdev(times) if repeats > 1 else 0.0, 4),
                "shots_per_s": round(throughput, 1),
                "ram_peak_mb": round(mean_r, 2),
            })

            print(f"  →  {mean_t:.3f}s  |  {throughput:,.0f} shots/s  |  {mean_r:.1f} MB")

    return results


# ── Speedup column (relative to n_qpus=1 for each shots value) ───────────────
def add_speedup(results: list[dict]) -> list[dict]:
    baseline: dict[int, float] = {}
    for row in results:
        if row["n_qpus"] == 1:
            baseline[row["shots"]] = row["time_mean_s"]
    for row in results:
        base = baseline.get(row["shots"])
        row["speedup"] = round(base / row["time_mean_s"], 2) if base else None
    return results


# ── Pretty table ──────────────────────────────────────────────────────────────
def print_table(results: list[dict], n_qubits: int) -> None:
    sep = "─" * 74
    header = (
        f"  {'n_qpus':>6}  {'shots':>7}  {'time (s)':>10}  "
        f"{'shots/s':>10}  {'RAM (MB)':>9}  {'speedup':>7}"
    )
    print(f"\n  GHZ circuit · {n_qubits} qubits · infrastructure=local")
    print(f"  {sep}")
    print(header)
    print(f"  {sep}")

    last_shots = None
    for row in results:
        if last_shots is not None and row["shots"] != last_shots:
            print(f"  {'·' * 72}")
        last_shots = row["shots"]

        speedup = f"{row['speedup']:.2f}×" if row.get("speedup") else "  —  "
        t_str = f"{row['time_mean_s']:.4f} ± {row['time_std_s']:.4f}"
        print(
            f"  {row['n_qpus']:>6}  {row['shots']:>7}  {t_str:>18}  "
            f"{row['shots_per_s']:>10,.0f}  {row['ram_peak_mb']:>9.1f}  {speedup:>7}"
        )

    print(f"  {sep}\n")


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(results: list[dict], path: Path) -> None:
    fieldnames = ["n_qubits", "n_qpus", "shots", "time_mean_s", "time_std_s",
                  "shots_per_s", "ram_peak_mb", "speedup"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Polypus system benchmark — time and RAM vs n_qpus / shots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--qpus", nargs="+", type=int, default=None,
        metavar="N",
        help="List of n_qpus values to test (default: 1 2 4)",
    )
    p.add_argument(
        "--shots", nargs="+", type=int, default=None,
        metavar="N",
        help="List of shot counts to test (default: 500 2000)",
    )
    p.add_argument(
        "--qubits", type=int, default=5,
        metavar="N",
        help="Number of qubits in the GHZ circuit (default: 5)",
    )
    p.add_argument(
        "--repeats", type=int, default=3,
        metavar="N",
        help="Repetitions per data point for averaging (default: 3)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Fast sweep: n_qpus=[1,2], shots=[500], repeats=1",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        metavar="FILE",
        help="Save results to a CSV file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        qpus   = [1, 2]
        shots  = [500]
        repeats = 1
    else:
        qpus   = args.qpus   or [1, 2, 4]
        shots  = args.shots  or [500, 2000]
        repeats = args.repeats

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[polypus-bench] Starting benchmark  —  {timestamp}")
    print(f"  Circuit : GHZ ({args.qubits} qubits)")
    print(f"  n_qpus  : {qpus}")
    print(f"  shots   : {shots}")
    print(f"  repeats : {repeats}\n")

    results = run_sweep(qpus, shots, args.qubits, repeats)
    results = add_speedup(results)

    print_table(results, args.qubits)

    if args.output:
        save_csv(results, args.output)
    else:
        default_out = Path("benchmarks") / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_csv(results, default_out)


if __name__ == "__main__":
    main()
