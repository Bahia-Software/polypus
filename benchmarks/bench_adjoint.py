"""
Benchmark: exact QML gradients by adjoint differentiation (issue #503).

Why it exists: with ``exact=True`` on the native backend, ``qml.jacobian``,
``polypus.torch.QuantumLayer``'s backward pass and Adam's gradient in
``Model.fit`` used to run the parameter-shift rule — ``2·P`` circuits per row
for ``P`` trainable gates — and now take one adjoint differentiation per row
(and observable), about three simulations long whatever ``P``. That is a
performance-only change (the result is the same to rounding, verified by the
tests), so docs/ENGINEERING.md §9 wants the speedup measured here.

Method. There is no public switch between the two (by design, no environment
variable or hidden flag), so the comparison is between two **builds**: this
script runs once with the wheel of the branch and once with a wheel built from
``main`` before the change (``abd0387``), each time under its own ``--label``,
and writes ``<out>/<label>.json``. ``--compare BEFORE.json AFTER.json`` then
prints the speedup ``t_before / t_after`` per point. Run both arms in the same
allocation, on the same node and thread count (see bench_adjoint.sbatch).

Each point is a hardware-efficient model on ``n`` qubits — an ``ry`` angle
encoder, ``--depth`` hardware-efficient layers, a ``Z₀`` readout — so ``P``
grows with ``n · depth``. For each point, after one untimed warm-up, it times
the median of ``--repeats`` runs of:

  * ``jacobian`` — ``qml.jacobian(model, x, params, exact=True)`` over
    ``--rows`` rows (``wrt="params"``);
  * ``fit_epoch`` — ``model.fit(x, y, method=polypus.Adam(max_iters=1), ...,
    exact=True)``: one Adam iteration over every row, the value and the
    gradient of a training epoch.

Usage:
    # with the branch's wheel installed
    RAYON_NUM_THREADS=16 python benchmarks/bench_adjoint.py --label adjoint \
        --out benchmarks/adjoint_results
    # with main's wheel installed (pip install --force-reinstall --no-deps ...)
    RAYON_NUM_THREADS=16 python benchmarks/bench_adjoint.py --label shift \
        --out benchmarks/adjoint_results
    python benchmarks/bench_adjoint.py --compare \
        benchmarks/adjoint_results/shift.json benchmarks/adjoint_results/adjoint.json

    # quick smoke test
    python benchmarks/bench_adjoint.py --label smoke --qubits 4 --depths 1 --repeats 1

Needs the extension built in **release** (``maturin build --release``); run it
through SLURM on the cluster (bench_adjoint.sbatch), never on a login node.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def build_model(n: int, depth: int):
    from polypus import qml

    return qml.Model(
        qml.Encoder(n).angle_encoder("ry"),
        qml.Ansatz(n).hardware_efficient(depth),
        qml.Readout([qml.Z(0)]),
    )


def rows(n: int, count: int):
    return [[0.1 * ((i + j) % 7) for j in range(n)] for i in range(count)]


def labels(count: int):
    return [0.5 if i % 2 == 0 else -0.5 for i in range(count)]


def median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure(n: int, depth: int, num_rows: int, repeats: int) -> dict:
    import polypus
    from polypus import qml

    model = build_model(n, depth)
    params = [0.1 + 0.03 * k for k in range(model.num_params)]
    x, y = rows(n, num_rows), labels(num_rows)

    def jacobian():
        qml.jacobian(model, x, params, exact=True, seed=1)

    def fit_epoch():
        model.fit(
            x,
            y,
            method=polypus.Adam(max_iters=1, patience=100),
            loss="squared_error",
            infrastructure="local",
            backend="polypus",
            exact=True,
            seed=1,
            initial_point=params,
        )

    jacobian()
    fit_epoch()
    return {
        "qubits": n,
        "depth": depth,
        "params": model.num_params,
        "rows": num_rows,
        "jacobian_s": median_time(jacobian, repeats),
        "fit_epoch_s": median_time(fit_epoch, repeats),
    }


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())

    def key(point):
        return (point["qubits"], point["depth"], point["rows"])

    after_points = {key(p): p for p in after["points"]}
    print(f"# before: {before['label']}  after: {after['label']}")
    print(
        f"{'qubits':>6} {'depth':>5} {'P':>5} {'rows':>4} "
        f"{'jac_before':>11} {'jac_after':>10} {'speedup':>8} "
        f"{'fit_before':>11} {'fit_after':>10} {'speedup':>8}"
    )
    for p in before["points"]:
        q = after_points.get(key(p))
        if q is None:
            continue
        print(
            f"{p['qubits']:>6} {p['depth']:>5} {p['params']:>5} {p['rows']:>4} "
            f"{p['jacobian_s']:>11.4f} {q['jacobian_s']:>10.4f} "
            f"{p['jacobian_s'] / q['jacobian_s']:>7.1f}x "
            f"{p['fit_epoch_s']:>11.4f} {q['fit_epoch_s']:>10.4f} "
            f"{p['fit_epoch_s'] / q['fit_epoch_s']:>7.1f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--label", help="name of this build's arm (adjoint, shift, ...)"
    )
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/adjoint_results")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
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
    print(
        f"{'qubits':>6} {'depth':>5} {'P':>5} {'rows':>4} {'jacobian_s':>11} {'fit_epoch_s':>12}"
    )
    for n in args.qubits:
        for depth in args.depths:
            point = measure(n, depth, args.rows, args.repeats)
            points.append(point)
            print(
                f"{n:>6} {depth:>5} {point['params']:>5} {args.rows:>4} "
                f"{point['jacobian_s']:>11.4f} {point['fit_epoch_s']:>12.4f}",
                flush=True,
            )
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
