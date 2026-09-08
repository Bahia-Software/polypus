"""
Benchmark + cross-verification: the **real** Polypus QFT template
(`polypus.circuits.templates.qft`) against Qiskit, swept across qubit counts.

Unlike the older `bench_qft.py`, which rebuilt the QFT gate-by-gate on both
sides (H + controlled-phase, *no* swap network), this script exercises the
production template as end users call it, and compares it to Qiskit's
supported synthesiser `qiskit.synthesis.qft.synth_qft_full` — whose signature
`(num_qubits, do_swaps=…, inverse=…)` maps 1:1 onto
`polypus.circuits.templates.qft(num_qubits, inverse, swaps)`, so the two sides
build the *same* circuit (big-endian, trailing qubit-reversal swaps) with no
hand reimplementation on either side. (The deprecated
`qiskit.circuit.library.QFT` class, removed in Qiskit 3.0, is deliberately not
used.)

Two independent things are measured:

  1. Correctness (`verify`): full-statevector agreement between
     `polypus.statevector(...)` and Aer's `statevector` method for all four
     `inverse × swaps` combinations, to `max amplitude error < 1e-9`; plus two
     Qiskit-independent analytic checks (`QFT|0…0⟩` is the uniform
     superposition, and `QFT† · QFT = I` over every computational-basis input,
     assembled by splicing the template's own QASM so nothing is hand-rolled).

  2. Timing (`sweep`): wall-clock for the native Rust statevector
     (`polypus.statevector`) vs Qiskit Aer (`statevector` method), and
     optionally Terra's pure `Statevector`, over a range of qubit counts.
     Each engine is timed in its **own subprocess**: a single in-process call
     to Terra's `Statevector` leaves later Aer runs ~20x slower in that same
     process (an OpenBLAS/OpenMP thread-affinity interaction, not machine
     noise), so interleaving engines in one process would understate Aer.
     min/median/max over `--repeat` rounds and `/proc/loadavg` are recorded
     because the host is shared and multi-threaded wall-clock varies under load.

Results land in a timestamped folder (matching `run_benchmarks.py`):
  benchmarks/bench_YYYYMMDD_HHMMSS/
    results.csv                    timing per engine × n_qubits
    correctness.csv                max amplitude error vs Aer per n_qubits
    qft_time_vs_qubits.png
    qft_speedup_vs_qubits.png
    qft_correctness_vs_qubits.png

Usage:
  python benchmarks/bench_qft_sweep.py                 # verify + full sweep
  python benchmarks/bench_qft_sweep.py --quick         # small, fast sweep
  python benchmarks/bench_qft_sweep.py --qubits 4 8 16 24 26
  python benchmarks/bench_qft_sweep.py --no-terra      # skip Terra reference
  python benchmarks/bench_qft_sweep.py --verify-only    # correctness, no timing
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Engine registry ───────────────────────────────────────────────────────────
# label is padded so the console table lines up.
ENGINES = {
    "polypus": "polypus (native Rust)",
    "qiskit_aer": "qiskit (Aer statevector)",
    "qiskit_terra": "qiskit (Terra Statevector)",
}
# Short tags for the inline speedup line.
SHORT = {"polypus": "native", "qiskit_aer": "Aer", "qiskit_terra": "Terra"}

# Correctness tolerance: the max amplitude error (polypus vs Aer) we require and
# draw as the reference line on the correctness plot. Shared by `verify()`, the
# per-n correctness curve, and the plot so the three never drift apart.
TOL = 1e-9


def load_average() -> tuple[str, str, str]:
    try:
        with open("/proc/loadavg") as f:
            one, five, fifteen = f.read().split()[:3]
        return one, five, fifteen
    except OSError:
        return ("nan", "nan", "nan")


# ── Circuit builders: the real public APIs, no hand reimplementation ──────────
def polypus_qft(n: int, inverse: bool, swaps: bool):
    import polypus

    return polypus.circuits.templates.qft(n, inverse, swaps)


def qiskit_qft(n: int, inverse: bool, swaps: bool):
    from qiskit.synthesis.qft import synth_qft_full

    return synth_qft_full(n, do_swaps=swaps, inverse=inverse)


def _aer_statevector(qc) -> np.ndarray:
    """Amplitudes from Aer's `statevector` method for a Qiskit circuit."""
    from qiskit_aer import AerSimulator

    qc = qc.copy()
    qc.save_statevector()
    aer = AerSimulator(method="statevector")
    return np.asarray(aer.run(qc).result().get_statevector().data)


def max_error_vs_aer(n: int, inverse: bool, swaps: bool) -> float:
    """Max amplitude error between the polypus template and Aer for one config.

    A single statevector on each side — O(1) circuits, cheap enough to sample at
    every point of the timing sweep (unlike `verify`'s O(2^n) identity check)."""
    import polypus

    p_sv = np.asarray(polypus.statevector(polypus_qft(n, inverse, swaps)))
    a_sv = _aer_statevector(qiskit_qft(n, inverse, swaps))
    return float(np.max(np.abs(p_sv - a_sv)))


# ══════════════════════════════════════════════════════════════════════════════
# Correctness
# ══════════════════════════════════════════════════════════════════════════════
def _qasm_body(qasm: str) -> list[str]:
    """Gate lines of an OpenQASM 2.0 program, dropping header/declarations."""
    skip = ("OPENQASM", "include", "qreg", "creg")
    return [
        ln
        for ln in qasm.splitlines()
        if ln.strip() and not ln.startswith(skip)
    ]


def _qasm_header(qasm: str) -> list[str]:
    keep = ("OPENQASM", "include", "qreg")
    return [ln for ln in qasm.splitlines() if ln.startswith(keep)]


def verify(n: int, tol: float = TOL) -> bool:
    """Cross-validate the template against Qiskit and against analytic identities.

    Returns True iff every check passes. Raises AssertionError on a real
    statevector disagreement (a failed benchmark is worse than a slow one)."""
    import polypus

    print(f"── Correctness · {n} qubits · tolerance {tol:.0e} ──")

    # (1) Full-statevector agreement with Aer, all four inverse × swaps combos.
    ok = True
    for inverse in (False, True):
        for swaps in (False, True):
            err = max_error_vs_aer(n, inverse, swaps)
            tag = f"inverse={str(inverse):5} swaps={str(swaps):5}"
            print(f"  polypus vs Aer  {tag}: max amplitude error {err:.2e}")
            assert err < tol, (
                f"statevectors disagree for {tag} ({err:.2e} ≥ {tol:.0e}) — "
                f"the template is wrong, refusing to benchmark it"
            )

    # (2) Analytic, Qiskit-independent: QFT|0…0⟩ is the uniform superposition.
    sv0 = np.asarray(polypus.statevector(polypus_qft(n, False, True)))
    unif = 1.0 / np.sqrt(2**n)
    dev = float(np.max(np.abs(np.abs(sv0) - unif)))
    print(f"  QFT|0…0⟩ uniform: max |amplitude| deviation {dev:.2e}")
    ok = ok and dev < tol

    # (3) Analytic, Qiskit-independent: QFT† · QFT = I over every basis state.
    #     Assembled by splicing the template's *own* QASM (forward then inverse)
    #     behind an X-prepared basis state — nothing is reimplemented by hand.
    fwd = polypus_qft(n, False, True).to_qasm2()
    inv = polypus_qft(n, True, True).to_qasm2()
    header, fbody, ibody = _qasm_header(fwd), _qasm_body(fwd), _qasm_body(inv)
    worst = 0.0
    for k in range(2**n):
        xprep = [f"x q[{i}];" for i in range(n) if (k >> i) & 1]
        circ = polypus.Circuit.from_qasm2("\n".join(header + xprep + fbody + ibody))
        sv = np.asarray(polypus.statevector(circ))
        expected = np.zeros(2**n, dtype=complex)
        expected[k] = 1.0
        worst = max(worst, float(np.max(np.abs(sv - expected))))
    print(f"  QFT†·QFT = I over all {2**n} basis states: max error {worst:.2e}")
    ok = ok and worst < tol

    print(f"  → {'PASS' if ok else 'FAIL'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Timing (each engine isolated in its own subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def _timed_call(engine: str, n: int, inverse: bool, swaps: bool):
    """Return a zero-arg closure that runs one QFT on `engine`."""
    if engine == "polypus":
        import polypus

        circ = polypus_qft(n, inverse, swaps)
        return lambda: polypus.statevector(circ)
    if engine == "qiskit_aer":
        from qiskit_aer import AerSimulator

        qc = qiskit_qft(n, inverse, swaps)
        qc.save_statevector()
        aer = AerSimulator(method="statevector")
        return lambda: aer.run(qc).result()
    if engine == "qiskit_terra":
        from qiskit.quantum_info import Statevector

        qc = qiskit_qft(n, inverse, swaps)
        return lambda: Statevector(qc)
    raise ValueError(engine)


def _run_isolated_engine(engine: str, n: int, inverse: bool, swaps: bool, repeat: int):
    """Internal mode: time one engine in this fresh process; emit JSON seconds."""
    fn = _timed_call(engine, n, inverse, swaps)
    fn()  # warmup: thread-pool spin-up, first allocation
    fn()  # second warmup: native auto-parallel pool can pay a one-off cost
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(json.dumps(times))


def bench_isolated(engine, n, inverse, swaps, repeat) -> list[float]:
    """Run one engine's timing loop in a brand-new subprocess (see docstring)."""
    cmd = [
        sys.executable, __file__, "--engine", engine, "--qubits", str(n),
        "--repeat", str(repeat),
    ]
    if inverse:
        cmd.append("--inverse")
    if not swaps:
        cmd.append("--no-swaps")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Output: CSV + plots
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "n_qubits", "gates", "engine", "min_s", "median_s", "max_s",
        "repeat", "loadavg_1", "loadavg_5", "loadavg_15",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")


def save_correctness_csv(correctness: dict[int, float], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_qubits", "max_amp_error"])
        for n in sorted(correctness):
            w.writerow([n, f"{correctness[n]:.3e}"])
    print(f"  Saved: {path.name}")


def save_plots(
    rows: list[dict],
    engines: list[str],
    out_dir: Path,
    correctness: dict[int, float] | None = None,
    durable: int | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive: safe in any environment
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not installed — skipping plots.")
        return

    qubits = sorted({r["n_qubits"] for r in rows})
    med = {
        e: {r["n_qubits"]: r["median_s"] for r in rows if r["engine"] == e}
        for e in engines
    }
    colors = {
        "polypus": "#1f77b4",
        "qiskit_aer": "#ff7f0e",
        "qiskit_terra": "#2ca02c",
    }

    # ── Panel 1: median time vs n_qubits (log-y) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in engines:
        xs = [q for q in qubits if q in med[e]]
        ys = [med[e][q] for q in xs]
        ax.plot(xs, ys, marker="o", label=ENGINES[e], color=colors.get(e))
    ax.set_yscale("log")
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("median wall-clock time (s, log scale)")
    ax.set_title("QFT statevector — time vs qubit count")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    dest = out_dir / "qft_time_vs_qubits.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dest.name}")

    # ── Panel 2: speedup of polypus vs each Qiskit engine ─────────────────────
    # Log-y: the run spans ~470× down to ~0.25×, and the crossover region
    # (0.2×–0.6×) is the whole point — on a linear axis it collapses onto the
    # x-axis and becomes unreadable.
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in engines:
        if e == "polypus":
            continue
        xs = [q for q in qubits if q in med[e] and q in med["polypus"]]
        ys = [med[e][q] / med["polypus"][q] for q in xs]
        ax.plot(xs, ys, marker="o", label=f"vs {ENGINES[e]}", color=colors.get(e))
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1.0, label="parity (1×)")
    if durable is not None:
        ax.axvline(durable, linestyle="--", color="#d62728", linewidth=1.2)
        ax.annotate(
            f"durable crossover\nn={durable}",
            xy=(durable, 1.0),
            xytext=(6, 8),
            textcoords="offset points",
            color="#d62728",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_yscale("log")
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("polypus speedup (× faster; <1 = slower, log scale)")
    ax.set_title("QFT statevector — polypus speedup vs Qiskit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    dest = out_dir / "qft_speedup_vs_qubits.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dest.name}")

    # ── Panel 3: correctness — max amplitude error vs Aer, per n_qubits ────────
    if correctness:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = sorted(correctness)
        ys = [correctness[q] for q in xs]
        ax.plot(xs, ys, marker="o", color="#1f77b4", label="max amplitude error")
        ax.axhline(
            TOL, linestyle="--", color="#d62728", linewidth=1.2,
            label=f"tolerance ({TOL:.0e})",
        )
        ax.set_yscale("log")
        ax.set_xlabel("n_qubits")
        ax.set_ylabel("max |amplitude| error vs Aer (log scale)")
        ax.set_title("QFT statevector — polypus vs Aer agreement")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        dest = out_dir / "qft_correctness_vs_qubits.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print(f"  Saved: {dest.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep the real Polypus QFT template vs Qiskit across qubit counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--qubits", nargs="+", type=int, default=None, metavar="N",
        help="Explicit qubit counts. Default sweeps 4..26 with extra "
        "resolution near the reported ~26q native/Aer crossover.",
    )
    p.add_argument("--repeat", type=int, default=5, metavar="N",
                   help="Timed rounds per engine per point (default: 5).")
    p.add_argument("--quick", action="store_true",
                   help="Fast sweep: qubits=[4,8,12,16,20], repeat=3.")
    p.add_argument("--no-terra", action="store_true",
                   help="Skip Terra's pure Statevector reference engine.")
    p.add_argument("--terra-max", type=int, default=22, metavar="N",
                   help="Cap Terra to n<=N (it is ~4x slower per qubit; default: 22).")
    p.add_argument("--verify-qubits", type=int, default=8, metavar="N",
                   help="Qubit count for the correctness pass (default: 8).")
    p.add_argument("--verify-only", action="store_true",
                   help="Run only the correctness checks, no timing sweep.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the one-off O(2^n) verify() pass. The cheap per-n "
                   "correctness curve (error vs Aer) is still collected — it is "
                   "independent of this flag.")
    p.add_argument("--inverse", action="store_true",
                   help="Time the inverse transform QFT† (default: forward).")
    p.add_argument("--no-swaps", action="store_true",
                   help="Time without the qubit-reversal swap network.")
    p.add_argument("--outdir", type=Path, default=None, metavar="DIR",
                   help="Output folder (default: benchmarks/bench_TIMESTAMP/).")
    # Internal: run a single engine in isolation (see bench_isolated).
    p.add_argument("--engine", choices=list(ENGINES), default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    swaps = not args.no_swaps

    # ── Internal isolated-engine mode ─────────────────────────────────────────
    if args.engine is not None:
        n = args.qubits[0] if args.qubits else 4
        _run_isolated_engine(args.engine, n, args.inverse, swaps, args.repeat)
        return

    # ── Correctness ───────────────────────────────────────────────────────────
    if not args.no_verify:
        verify(args.verify_qubits)
        if args.verify_only:
            return

    # ── Which engines / which qubit counts ────────────────────────────────────
    engines = ["polypus", "qiskit_aer"]
    if not args.no_terra:
        engines.append("qiskit_terra")

    if args.quick:
        qubits = [4, 8, 12, 16, 20]
    elif args.qubits:
        qubits = sorted(set(args.qubits))
    else:
        # Extra resolution from 22 up to bracket the historical ~26q crossover.
        qubits = [4, 8, 12, 16, 18, 20, 22, 23, 24, 25, 26]

    label = ("QFT†" if args.inverse else "QFT") + ("" if swaps else " (no swaps)")
    print(f"── Timing sweep · {label} · engines: {', '.join(engines)} ──")
    print(f"  qubits: {qubits}")
    print(f"  {args.repeat} rounds per engine, each in its own subprocess\n")

    rows: list[dict] = []
    med_by_engine: dict[str, dict[int, float]] = {e: {} for e in engines}
    # Per-n correctness (max amplitude error vs Aer) sampled once per n — a
    # single statevector per side, independent of the timing engines. Collected
    # for the timed configuration (args.inverse × swaps) so the correctness plot
    # tracks exactly what was benchmarked. This is deliberately kept separate
    # from `--no-verify` (see below): it is cheap and its whole purpose is to
    # make the polypus/Aer agreement visible across the sweep.
    correctness: dict[int, float] = {}
    for n in qubits:
        gates = qiskit_qft(n, args.inverse, swaps).size()
        la1, la5, la15 = load_average()
        print(f"  n={n:>2} ({gates:>4} gates) load={la1}/{la5}/{la15}")
        correctness[n] = max_error_vs_aer(n, args.inverse, swaps)
        print(f"    max amplitude error vs Aer: {correctness[n]:.2e}")
        for e in engines:
            if e == "qiskit_terra" and n > args.terra_max:
                print(f"    {ENGINES[e]:26}: skipped (n>{args.terra_max})")
                continue
            xs = bench_isolated(e, n, args.inverse, swaps, args.repeat)
            lo, md, hi = min(xs), statistics.median(xs), max(xs)
            med_by_engine[e][n] = md
            rows.append({
                "n_qubits": n, "gates": gates, "engine": e,
                "min_s": round(lo, 6), "median_s": round(md, 6),
                "max_s": round(hi, 6), "repeat": args.repeat,
                "loadavg_1": la1, "loadavg_5": la5, "loadavg_15": la15,
            })
            print(
                f"    {ENGINES[e]:26}: {lo * 1e3:9.3f} / {md * 1e3:9.3f} / "
                f"{hi * 1e3:9.3f} ms (min/median/max)"
            )
        # Speedup snapshot for this n.
        if n in med_by_engine["polypus"]:
            base = med_by_engine["polypus"][n]
            parts = [
                f"vs {SHORT[e]} {med_by_engine[e][n] / base:5.2f}x"
                for e in engines if e != "polypus" and n in med_by_engine[e]
            ]
            print(f"    polypus speedup: {' · '.join(parts)}")

    # ── Crossover summary (the interesting finding) ───────────────────────────
    # "Durable" crossover = smallest n at (and beyond) which native stays
    # slower than Aer, distinguished from transient dips (e.g. the one-off
    # thread-pool spin-up near the native auto-parallel threshold).
    print("\n── polypus vs Aer statevector ──")
    shared = [
        n for n in qubits
        if n in med_by_engine["polypus"] and n in med_by_engine["qiskit_aer"]
    ]
    speedup = {n: med_by_engine["qiskit_aer"][n] / med_by_engine["polypus"][n]
               for n in shared}
    durable = None
    for i, n in enumerate(shared):
        if all(speedup[m] < 1.0 for m in shared[i:]):
            durable = n
            break
    transient = [n for n in shared if speedup[n] < 1.0 and (durable is None or n < durable)]
    for n in shared:
        flag = ""
        if n == durable:
            flag = "  ← durable crossover: native slower than Aer from here on"
        elif n in transient:
            flag = "  (transient dip)"
        print(f"  n={n:>2}: {speedup[n]:6.2f}x{flag}")
    if durable is not None:
        print(f"\n  Native statevector falls durably below Aer at n={durable} "
              f"(and stays below through n={shared[-1]}).")
    else:
        print("\n  Native statevector stayed faster than Aer across the whole sweep.")
    if transient:
        print(f"  Transient sub-1x dip(s) at n={transient} — likely one-off "
              f"thread-pool spin-up at the native auto-parallel threshold, not "
              f"the memory-bound crossover.")

    # ── Persist ───────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.outdir or Path("benchmarks") / f"bench_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, out_dir / "results.csv")
    save_correctness_csv(correctness, out_dir / "correctness.csv")
    save_plots(rows, engines, out_dir, correctness=correctness, durable=durable)
    print(f"\n  Output folder: {out_dir}/")


if __name__ == "__main__":
    main()
