#!/usr/bin/env python3
"""
Polypus system benchmark.

Measures circuit execution time and peak RAM for a GHZ circuit across
different n_qpus / shots combinations, using the local AerSimulator backend
with `matrix_product_state` simulation by default (per-shot independent
simulation, enabling real QPU parallelism).

Results are saved to a timestamped folder under benchmarks/:
  benchmarks/bench_YYYYMMDD_HHMMSS/
    results.csv
    time_vs_qpus.png
    throughput_vs_qpus.png
    speedup_vs_qpus.png
    ram_vs_qpus.png

Usage
-----
  python benchmarks/run_benchmarks.py                         # defaults
  python benchmarks/run_benchmarks.py --quick                 # fast sweep
  python benchmarks/run_benchmarks.py --qpus 1 2 4 8
  python benchmarks/run_benchmarks.py --shots 500 2000
  python benchmarks/run_benchmarks.py --qubits 6
  python benchmarks/run_benchmarks.py --outdir my_run/
"""

import argparse
import csv
import statistics
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe in any environment
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.optimize import differential_evolution as scipy_de
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
def _run_once(qc: QuantumCircuit, shots: int, n_qpus: int, sim_method: str) -> tuple[float, float]:
    """Return (elapsed_seconds, peak_ram_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    polypus.run_quantum_circuit(qc, shots=shots, infrastructure="local", n_qpus=n_qpus, sim_method=sim_method)
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
    sim_method: str,
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
                t, r = _run_once(qc, shots, n_qpus, sim_method)
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
    sep = "─" * 65
    header = (
        f"  {'n_qpus':>6}  {'shots':>7}  {'time (s)':>10}  "
        f"{'shots/s':>10}  {'RAM (MB)':>9}"
    )
    print(f"\n  GHZ circuit · {n_qubits} qubits · infrastructure=local")
    print(f"  {sep}")
    print(header)
    print(f"  {sep}")

    last_shots = None
    for row in results:
        if last_shots is not None and row["shots"] != last_shots:
            print(f"  {'·' * 63}")
        last_shots = row["shots"]

        t_str = f"{row['time_mean_s']:.4f} ± {row['time_std_s']:.4f}"
        print(
            f"  {row['n_qpus']:>6}  {row['shots']:>7}  {t_str:>18}  "
            f"{row['shots_per_s']:>10,.0f}  {row['ram_peak_mb']:>9.1f}"
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
    print(f"  Saved: {path.name}")


# ── Plots ─────────────────────────────────────────────────────────────────────
def save_plots(results: list[dict], n_qubits: int, out_dir: Path) -> None:
    if not HAS_MATPLOTLIB:
        print("  [warn] matplotlib not installed — skipping plots.")
        return

    # Group data by shots value
    shots_values = sorted({r["shots"] for r in results})
    qpus_per_shots = {
        s: sorted([r for r in results if r["shots"] == s], key=lambda r: r["n_qpus"])
        for s in shots_values
    }
    all_qpus = sorted({r["n_qpus"] for r in results})

    metrics = [
        ("time_mean_s", "Execution time (s)",   "lower is better"),
        ("shots_per_s", "Throughput (shots/s)", "higher is better"),
        ("ram_peak_mb", "Peak RAM (MB)",         "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Polypus benchmark — GHZ {n_qubits}q · infrastructure=local",
        fontsize=13, fontweight="bold",
    )

    for ax, (field, ylabel, note) in zip(axes, metrics):
        for s in shots_values:
            rows = qpus_per_shots[s]
            xs = [r["n_qpus"] for r in rows if r[field] is not None]
            ys = [r[field]    for r in rows if r[field] is not None]
            ax.plot(xs, ys, marker="o", label=f"{s:,} shots")

        ax.set_xlabel("n_qpus")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({note})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    dest = out_dir / "benchmark.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dest.name}")


# ── Parameterized VQC (optimizer comparison) ─────────────────────────────────
def make_parameterized_vqc(n_qubits: int) -> tuple[QuantumCircuit, int]:
    from qiskit.circuit import ParameterVector
    n_params = n_qubits * 2
    params = ParameterVector("θ", n_params)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i * 2], i)
        qc.rz(params[i * 2 + 1], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc, n_params


def _hamming_cost_bitstring(bitstring: str) -> float:
    """Polypus expectation_function: fraction of 1s in a bitstring."""
    return float(bitstring.count("1")) / len(bitstring)


def _hamming_cost_counts(result) -> float:
    """SciPy objective: expected fraction of 1s from run_quantum_circuit output."""
    # run_quantum_circuit returns a RunResult wrapping the counts payload
    # (contract C-7): a list for a single QPU, a merged dict for n_qpus > 1.
    payload = result.counts
    counts = payload[0] if isinstance(payload, list) else payload
    total = sum(counts.values())
    return sum((bs.count("1") / len(bs)) * c for bs, c in counts.items()) / total


def run_optimizer_comparison(
    qpus_list: list[int],
    shots: int,
    n_qubits: int,
    generations: int,
    population_size: int,
    sim_method: str,
) -> dict:
    """Time Polypus DE/PSO/QNG vs SciPy DE on a small hardware-efficient VQC."""
    qc, n_params = make_parameterized_vqc(n_qubits)
    bounds_tuple = (-3.14159, 3.14159)
    bounds_list  = [bounds_tuple] * n_params
    comp: dict   = {"scipy_de": None, "de": {}, "pso": {}, "qng": {}}

    def _variance_fn(theta: list, a: int) -> float:
        """Constant diagonal QFI element — sufficient for benchmarking QNG plumbing."""
        return 0.5

    # ── SciPy DE (baseline: single-QPU, sequential) ───────────────────────────
    if HAS_SCIPY:
        def scipy_cost(params):
            bound_qc = qc.assign_parameters(dict(zip(qc.parameters, params)))
            result = polypus.run_quantum_circuit(
                bound_qc, shots=shots, infrastructure="local", n_qpus=1,
                sim_method=sim_method,
            )
            return _hamming_cost_counts(result)

        popsize_mult = max(1, population_size // n_params)
        print(f"  [scipy DE ] generations={generations}  pop_mult={popsize_mult}  shots={shots}",
              flush=True)
        t0 = time.perf_counter()
        scipy_de(scipy_cost, bounds_list, maxiter=generations, popsize=popsize_mult,
                 seed=42, workers=1)
        comp["scipy_de"] = round(time.perf_counter() - t0, 4)
        print(f"             → {comp['scipy_de']:.3f}s")
    else:
        print("  [scipy DE ] not installed — skipping.")

    # ── Polypus optimizers vs n_qpus ─────────────────────────────────────────
    optimizers = [
        ("de",  lambda: polypus.DE(generations=generations,
                                    population_size=population_size,
                                    tolerance=1e-10)),
        ("pso", lambda: polypus.PSO(generations=generations,
                                    population_size=population_size,
                                    bounds=bounds_tuple,
                                    tolerance=1e-10)),
        ("qng", lambda: polypus.QNG(_variance_fn,
                                    max_iters=generations,
                                    bounds=bounds_tuple,
                                    learning_rate=0.1)),
    ]

    total = len(qpus_list)
    for method_key, make_method in optimizers:
        print(f"\n  ── Polypus {method_key.upper()} ──")
        for i, n_qpus in enumerate(qpus_list, 1):
            print(
                f"  [{i}/{total}] n_qpus={n_qpus:>2}  generations={generations}"
                f"  population_size={population_size}  shots={shots}",
                flush=True,
            )
            t0 = time.perf_counter()
            polypus.train(
                qc,
                make_method(),
                shots=shots,
                n_qpus=n_qpus,
                dimensions=n_params,
                expectation_function=_hamming_cost_bitstring,
                infrastructure="local",
                nodes=1,
                cores_per_qpu=1,
                id=f"bench_{method_key}_q{n_qpus}",
                sim_method=sim_method,
            )
            comp[method_key][n_qpus] = round(time.perf_counter() - t0, 4)
            print(f"           → {comp[method_key][n_qpus]:.3f}s")

    return comp


def save_comparison_plot(
    comp: dict, n_qubits: int, generations: int, population_size: int, out_dir: Path
) -> None:
    if not HAS_MATPLOTLIB:
        print("  [warn] matplotlib not installed — skipping plots.")
        return

    qpus       = sorted({q for m in ("de", "pso", "qng") for q in comp[m]})
    scipy_time = comp["scipy_de"]

    methods = [
        ("de",  "Polypus DE",  "#1f77b4"),
        ("pso", "Polypus PSO", "#2ca02c"),
        ("qng", "Polypus QNG", "#9467bd"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"Polypus optimizers vs SciPy DE  —  VQC {n_qubits}q"
        f" · {generations} iterations · pop={population_size}",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: absolute time ────────────────────────────────────────────
    ax = axes[0]
    for key, label, color in methods:
        times = [comp[key].get(q) for q in qpus]
        ax.plot(qpus, times, marker="o", label=label, color=color)
    if scipy_time is not None:
        ax.axhline(scipy_time, linestyle="--", color="#ff7f0e", linewidth=1.5,
                   label=f"SciPy DE  ({scipy_time:.2f}s)")
    ax.set_xlabel("n_qpus")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Optimization time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: speedup vs SciPy DE ──────────────────────────────────────
    ax = axes[1]
    if scipy_time is not None:
        for key, label, color in methods:
            speedups = [scipy_time / comp[key][q] for q in qpus if comp[key].get(q)]
            ax.plot(qpus[:len(speedups)], speedups, marker="o", label=label, color=color)
        ax.axhline(1.0, linestyle="--", color="#ff7f0e", linewidth=1.0,
                   label="SciPy DE baseline")
        ax.set_ylabel("Speedup vs SciPy DE")
    else:
        for key, label, color in methods:
            times = [comp[key].get(q) for q in qpus]
            base  = times[0] or 1
            ax.plot(qpus, [base / t for t in times], marker="o", label=label, color=color)
        ax.axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
        ax.set_ylabel("Speedup vs n_qpus=1")
    ax.set_xlabel("n_qpus")
    ax.set_title("Speedup vs SciPy DE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: scaling efficiency (speedup relative to own n_qpus=1) ─────
    ax = axes[2]
    for key, label, color in methods:
        times = [comp[key].get(q) for q in qpus]
        base  = times[0] or 1
        ax.plot(qpus, [base / t for t in times], marker="o", label=label, color=color)
    ax.plot(qpus, qpus, linestyle="--", color="grey", linewidth=0.8, label="ideal")
    ax.set_xlabel("n_qpus")
    ax.set_ylabel("Speedup vs own n_qpus=1")
    ax.set_title("Scaling efficiency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    dest = out_dir / "optimizer_comparison.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dest.name}")


def save_qubit_heatmap(
    multi_comp: dict, generations: int, population_size: int, out_dir: Path
) -> None:
    """3-panel heatmap: one panel per optimizer, rows=n_qubits, cols=n_qpus, cells=speedup vs SciPy DE."""
    if not HAS_MATPLOTLIB:
        print("  [warn] matplotlib not installed — skipping plots.")
        return

    import numpy as np

    qubits_list = sorted(multi_comp.keys())
    qpus = sorted({q for n_q in qubits_list
                   for m in ("de", "pso", "qng")
                   for q in multi_comp[n_q][m]})
    methods = [
        ("de",  "Polypus DE"),
        ("pso", "Polypus PSO"),
        ("qng", "Polypus QNG"),
    ]

    # Shared colour scale across all panels
    vmax = 0.0
    for n_q in qubits_list:
        scipy_t = multi_comp[n_q].get("scipy_de")
        if scipy_t:
            for key, _ in methods:
                for q in qpus:
                    t = multi_comp[n_q][key].get(q)
                    if t:
                        vmax = max(vmax, scipy_t / t)
    vmax = vmax or 1.0

    n_rows = len(qubits_list)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(5 * len(methods), max(3, 1.2 + 1.1 * n_rows)),
    )
    fig.suptitle(
        f"Speedup vs SciPy DE  ·  {generations} iterations  ·  pop={population_size}",
        fontsize=11, fontweight="bold",
    )

    for ax, (key, label) in zip(axes, methods):
        data = np.full((n_rows, len(qpus)), np.nan)
        for i, n_q in enumerate(qubits_list):
            scipy_t = multi_comp[n_q].get("scipy_de")
            for j, q in enumerate(qpus):
                t = multi_comp[n_q][key].get(q)
                if t and scipy_t:
                    data[i, j] = scipy_t / t

        im = ax.imshow(
            data, aspect="auto", cmap="YlOrRd",
            origin="lower", vmin=0, vmax=vmax,
        )
        ax.set_xticks(range(len(qpus)))
        ax.set_xticklabels([str(q) for q in qpus])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([str(q) for q in qubits_list])
        ax.set_xlabel("n_qpus")
        ax.set_ylabel("n_qubits")
        ax.set_title(label, fontweight="bold")

        for i in range(n_rows):
            for j in range(len(qpus)):
                v = data[i, j]
                if not np.isnan(v):
                    txt_color = "black" if (v / vmax) > 0.55 else "white"
                    ax.text(j, i, f"{v:.1f}×",
                            ha="center", va="center",
                            fontsize=9, fontweight="bold", color=txt_color)

        plt.colorbar(im, ax=ax, label="Speedup vs SciPy DE", shrink=0.8)

    fig.tight_layout()
    dest = out_dir / "optimizer_comparison.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dest.name}")


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
        "--qubits", type=int, default=25,
        metavar="N",
        help="Number of qubits in the GHZ circuit (default: 25)",
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
        "--outdir", type=Path, default=None,
        metavar="DIR",
        help="Output folder for CSV and plots (default: benchmarks/bench_TIMESTAMP/)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots (CSV only)",
    )
    p.add_argument(
        "--no-compare", action="store_true",
        help="Skip Polypus DE · PSO · QNG vs SciPy DE optimizer comparison",
    )
    p.add_argument(
        "--sim-method", type=str, default="automatic",
        metavar="METHOD",
        help="Aer simulation method (default: automatic). "
             "Options: automatic, statevector, matrix_product_state, "
             "density_matrix, stabilizer, extended_stabilizer. "
             "Use matrix_product_state when testing with noise models.",
    )
    p.add_argument(
        "--cmp-generations", type=int, default=None, metavar="N",
        help="Generations for optimizer comparison (default: 20, quick: 5)",
    )
    p.add_argument(
        "--cmp-popsize", type=int, default=None, metavar="N",
        help="Population size for optimizer comparison (default: 8, quick: 4)",
    )
    p.add_argument(
        "--cmp-shots", type=int, default=None, metavar="N",
        help="Shots for optimizer comparison (default: 1000, quick: 300)",
    )
    p.add_argument(
        "--cmp-qubits", nargs="+", type=int, default=None, metavar="N",
        help="Qubit counts to sweep in optimizer comparison "
             "(default: [4, 8] in quick mode, [4, 8, 12] otherwise).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        qpus   = [1, 2, 4 ,8, 16]
        shots  = [10000]
        repeats = 3
    else:
        qpus   = args.qpus   or [1, 2, 4 ,8, 16]
        shots  = args.shots  or [10000]
        repeats = args.repeats

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[polypus-bench] Starting benchmark  —  {timestamp}")
    print(f"  Circuit    : GHZ ({args.qubits} qubits)")
    print(f"  sim_method : {args.sim_method}")
    print(f"  n_qpus     : {qpus}")
    print(f"  shots      : {shots}")
    print(f"  repeats    : {repeats}\n")

    results = run_sweep(qpus, shots, args.qubits, repeats, args.sim_method)
    results = add_speedup(results)

    print_table(results, args.qubits)

    # ── Output folder ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.outdir or Path("benchmarks") / f"bench_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_csv(results, out_dir / "results.csv")

    if not args.no_plots:
        save_plots(results, args.qubits, out_dir)

    # ── Optimizer comparison: Polypus DE · PSO · QNG vs SciPy DE ──────────────
    if not args.no_compare:
        cmp_gen   = args.cmp_generations or (5    if args.quick else 20)
        cmp_pop   = args.cmp_popsize     or (4    if args.quick else 8)
        cmp_shots = args.cmp_shots       or (300  if args.quick else 1000)

        cmp_qubits_list = args.cmp_qubits or ([4, 8] if args.quick else [4, 8, 12])

        print(f"\n[polypus-bench] Optimizer comparison — Polypus DE · PSO · QNG vs SciPy DE")
        print(f"  Qubit sweep : {cmp_qubits_list}")
        print(f"  generations : {cmp_gen}   population_size : {cmp_pop}   shots : {cmp_shots}")

        multi_comp: dict = {}
        for n_q in cmp_qubits_list:
            header = f"\n  {'─' * 10} {n_q} qubits · {n_q * 2} params {'─' * 10}"
            print(header)
            multi_comp[n_q] = run_optimizer_comparison(qpus, cmp_shots, n_q, cmp_gen, cmp_pop, args.sim_method)

        if not args.no_plots:
            if len(cmp_qubits_list) == 1:
                n_q = cmp_qubits_list[0]
                save_comparison_plot(multi_comp[n_q], n_q, cmp_gen, cmp_pop, out_dir)
            else:
                save_qubit_heatmap(multi_comp, cmp_gen, cmp_pop, out_dir)

    print(f"\n  Output folder: {out_dir}/")


if __name__ == "__main__":
    main()
