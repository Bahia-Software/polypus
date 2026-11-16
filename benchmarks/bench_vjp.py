"""
Benchmark: the vector-Jacobian product in one adjoint run per row (issue #559).

Why it exists: the backward pass of ``polypus.torch.QuantumLayer`` used to ask
``qml.jacobian`` for the whole Jacobian and contract it with the incoming
gradient; with ``exact=True`` that is one adjoint differentiation per row **and
per observable**. ``qml.vjp`` folds the cotangent into one observable,
``Σ_k v_k · O_k``, and takes one adjoint run per row whatever the number of
observables ``K``. The result is the same to rounding (verified by the tests),
so this is a performance change, and docs/ENGINEERING.md §9 wants the speedup
measured here.

Method. Unlike bench_adjoint.py, **one build** is enough: ``qml.vjp`` and
``qml.jacobian`` live side by side. One discarded point first warms the
process up. For each point — a hardware-efficient model
on ``n`` qubits (an ``ry`` angle encoder, ``--depth`` layers) read out by an
``argmax`` over ``K`` observables (``Z_k``, then ``Z_k Z_{k+1}`` products past
``n``) — and each ``wrt``, after one untimed warm-up of each arm, it times the
median of ``--repeats`` runs of:

  * ``jacobian`` — ``qml.jacobian(model, x, params, wrt=..., exact=True)`` over
    ``--rows`` rows, then its contraction with the cotangent in NumPy
    (``einsum("bo,bok->k")`` / ``einsum("bo,bof->bf")``), what the layer did;
  * ``vjp`` — ``qml.vjp(model, x, params, cotangent, wrt=..., exact=True)``.

and prints ``speedup = t_jacobian / t_vjp``: about ``K`` once the adjoint runs
dominate. ``--compare A.json B.json`` prints two result files side by side (two
nodes, two thread counts, two builds of the branch), with each one's speedup
and ``t_vjp`` of A over B.

Usage:
    RAYON_NUM_THREADS=16 python benchmarks/bench_vjp.py --label vjp \
        --out benchmarks/vjp_results
    python benchmarks/bench_vjp.py --compare A.json B.json

    # quick smoke test
    python benchmarks/bench_vjp.py --label smoke --qubits 4 --observables 1 2 --repeats 1

Needs the extension built in **release** (``maturin build --release``); run it
through SLURM on the cluster (bench_vjp.sbatch), never on a login node.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

CONTRACTION = {"params": "bo,bok->k", "inputs": "bo,bof->bf"}


def observables(n: int, k: int):
    """``K`` distinct Pauli products on ``n`` qubits: ``Z_i`` for the first
    ``n``, then ``Z_i Z_{i+1+j}`` (wrapping) for the next rounds."""
    from polypus import qml

    out = []
    for o in range(k):
        i, round_ = o % n, o // n
        term = qml.Z(i)
        if round_:
            term = term @ qml.Z((i + round_) % n)
        out.append(term)
    return out


def build_model(n: int, depth: int, k: int):
    from polypus import qml

    return qml.Model(
        qml.Encoder(n).angle_encoder("ry"),
        qml.Ansatz(n).hardware_efficient(depth),
        qml.Readout(observables(n, k), decision="argmax" if k > 1 else "raw"),
    )


def rows(n: int, count: int):
    return [[0.1 * ((i + j) % 7) for j in range(n)] for i in range(count)]


def median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure(n: int, depth: int, k: int, wrt: str, num_rows: int, repeats: int) -> dict:
    import numpy as np
    from polypus import qml

    model = build_model(n, depth, k)
    params = [0.1 + 0.03 * p for p in range(model.num_params)]
    x = rows(n, num_rows)
    v = np.array(
        [[1.0 - 0.07 * (r * k + o) for o in range(k)] for r in range(num_rows)]
    )

    def contracted_jacobian():
        jacobian = qml.jacobian(model, x, params, wrt=wrt, exact=True)
        if wrt == "both":
            return (
                np.einsum(CONTRACTION["params"], v, jacobian[0]),
                np.einsum(CONTRACTION["inputs"], v, jacobian[1]),
            )
        return np.einsum(CONTRACTION[wrt], v, jacobian)

    def vjp():
        return qml.vjp(model, x, params, v, wrt=wrt, exact=True)

    # Warm-up, and the two must agree (exact either way).
    a, b = contracted_jacobian(), vjp()
    for got, expected in zip(*((a, b) if wrt == "both" else ((a,), (b,)))):
        worst = float(np.max(np.abs(got - expected)))
        assert worst < 1e-8, f"vjp and the contracted Jacobian differ by {worst}"
    t_jacobian = median_time(contracted_jacobian, repeats)
    t_vjp = median_time(vjp, repeats)
    return {
        "qubits": n,
        "depth": depth,
        "params": model.num_params,
        "observables": k,
        "wrt": wrt,
        "rows": num_rows,
        "jacobian_s": t_jacobian,
        "vjp_s": t_vjp,
        "speedup": t_jacobian / t_vjp,
    }


HEADER = (
    f"{'qubits':>6} {'depth':>5} {'P':>5} {'K':>3} {'wrt':>6} {'rows':>4} "
    f"{'jacobian_s':>11} {'vjp_s':>9} {'speedup':>8}"
)


def row(point: dict) -> str:
    return (
        f"{point['qubits']:>6} {point['depth']:>5} {point['params']:>5} "
        f"{point['observables']:>3} {point['wrt']:>6} {point['rows']:>4} "
        f"{point['jacobian_s']:>11.4f} {point['vjp_s']:>9.4f} {point['speedup']:>7.1f}x"
    )


def compare(a_path: str, b_path: str) -> None:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())

    def key(point):
        return (
            point["qubits"],
            point["depth"],
            point["observables"],
            point["wrt"],
            point["rows"],
        )

    b_points = {key(p): p for p in b["points"]}
    print(f"# A: {a['label']}  B: {b['label']}")
    print(
        f"{'qubits':>6} {'depth':>5} {'P':>5} {'K':>3} {'wrt':>6} {'rows':>4} "
        f"{'speedup_A':>10} {'speedup_B':>10} {'vjp_A':>9} {'vjp_B':>9} {'A/B':>6}"
    )
    for p in a["points"]:
        q = b_points.get(key(p))
        if q is None:
            continue
        print(
            f"{p['qubits']:>6} {p['depth']:>5} {p['params']:>5} {p['observables']:>3} "
            f"{p['wrt']:>6} {p['rows']:>4} {p['speedup']:>9.1f}x {q['speedup']:>9.1f}x "
            f"{p['vjp_s']:>9.4f} {q['vjp_s']:>9.4f} {p['vjp_s'] / q['vjp_s']:>5.2f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", help="name of this run (vjp, node-a, ...)")
    parser.add_argument("--qubits", type=int, nargs="+", default=[10, 12])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--observables", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--wrt",
        nargs="+",
        default=["params", "both"],
        choices=["params", "inputs", "both"],
    )
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="benchmarks/vjp_results")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.label:
        parser.error("--label is required unless --compare is given")

    import polypus

    points = []
    print(
        f"# label={args.label} polypus={polypus.__version__} "
        f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', '')}"
    )
    # One discarded point warms the process up (rayon's thread pool, the
    # extension's first calls), so the first measured point is not inflated.
    measure(args.qubits[0], args.depth, args.observables[0], args.wrt[0], args.rows, 1)
    print(HEADER)
    for n in args.qubits:
        for k in args.observables:
            for wrt in args.wrt:
                point = measure(n, args.depth, k, wrt, args.rows, args.repeats)
                points.append(point)
                print(row(point), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "label": args.label,
        "version": polypus.__version__,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "points": points,
    }
    (out / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print(f"# wrote {out / (args.label + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
