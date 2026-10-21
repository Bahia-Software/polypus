"""
Routing study for the Euro-Q-Exa benchmark (#391): QAOA circuits transpiled to
EQE1's native gates (``r``, ``cz``) and coupling map.

For each (instance, p) the circuit is transpiled ``K`` times with different
transpiler seeds (Qiskit, optimization level 3, ALAP scheduling against
``device.build_target``), and the run with the highest estimated success
probability (ESP) is kept for every later condition. The target carries the
calibrated errors, so layout selection already avoids the weak qubits and edges;
the seeds explore routing variability, which is reported too.

ESP is the product of ``1 - error`` over every gate and measurement of the
circuit (the calibrated errors in the target): a cheap proxy for how much of the
signal survives, not a prediction of the approximation ratio.

Usage (writes ``instances.json``, ``routing.csv`` with every seed, and
``routing_summary.md``, the table for LRZ, into ``--out``):
    python benchmarks/euro_q_exa/routing.py \
        --calibration calibration_data_Euro-Q-Exa_20261019_active_qubits.json \
        --coupling-map coupling_map.txt --out benchmarks/euro_q_exa/results
"""

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import device as eqe
import instances as inst
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target

DEFAULT_SEEDS = 10
DEPTHS = (1, 2, 3)
_NOT_GATES = ("delay", "barrier")


@dataclass(frozen=True)
class RouteStats:
    """Cost of one transpiled circuit. ``depth`` and ``depth_2q`` ignore delays
    and barriers; ``duration_ns`` is the scheduled length."""

    seed: int
    cz: int
    one_qubit: int
    depth: int
    depth_2q: int
    physical_qubits: int
    duration_ns: int
    esp: float


@dataclass(frozen=True)
class Routed:
    """The kept circuit of a routing study and the stats of every seed."""

    circuit: QuantumCircuit
    best: RouteStats
    per_seed: tuple[RouteStats, ...]


def esp(circuit: QuantumCircuit, target: Target) -> float:
    """Product of ``1 - error`` over the circuit's gates and measurements, with
    the errors the target holds for their qubits (delays and barriers skipped)."""
    log_esp = 0.0
    for item in circuit.data:
        name = item.operation.name
        if name in _NOT_GATES:
            continue
        qargs = tuple(circuit.find_bit(q).index for q in item.qubits)
        error = target[name][qargs].error
        log_esp += math.log1p(-error)
    return math.exp(log_esp)


def stats(circuit: QuantumCircuit, target: Target, seed: int) -> RouteStats:
    ops = circuit.count_ops()
    used = {
        q
        for item in circuit.data
        if item.operation.name not in _NOT_GATES
        for q in item.qubits
    }
    return RouteStats(
        seed=seed,
        cz=ops.get(eqe.TWO_QUBIT_GATE, 0),
        one_qubit=ops.get(eqe.ONE_QUBIT_GATE, 0),
        depth=circuit.depth(lambda x: x.operation.name not in _NOT_GATES),
        depth_2q=circuit.depth(lambda x: x.operation.num_qubits == 2),
        physical_qubits=len(used),
        duration_ns=int(round(circuit.estimate_duration(target, unit="dt"))),
        esp=esp(circuit, target),
    )


def route(
    circuit: QuantumCircuit, target: Target, seeds=range(DEFAULT_SEEDS)
) -> Routed:
    """Transpile ``circuit`` once per seed and keep the highest-ESP result
    (ties: the lower seed). The kept circuit is scheduled (ALAP) and still
    parameterized."""
    runs = []
    for seed in seeds:
        routed = transpile(
            circuit,
            target=target,
            optimization_level=3,
            scheduling_method="alap",
            seed_transpiler=seed,
        )
        runs.append((routed, stats(routed, target, seed)))
    circuit_best, best = max(runs, key=lambda r: (r[1].esp, -r[1].seed))
    return Routed(circuit_best, best, tuple(s for _, s in runs))


def _summary_table(rows) -> str:
    """Markdown table: one row per (family, n, p), means over the instances of
    the kept circuits, plus mean ± sd of the CZ count over every seed of every instance."""
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["n"], r["p"]), []).append(r)
    lines = [
        "| Family | n | p | CZ (min without routing) | CZ, kept | CZ, all seeds and instances | Depth | "
        "2q depth | Physical qubits | Duration (us) | ESP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (family, n, p), group in sorted(groups.items()):
        kept = [r for r in group if r["kept"]]
        seeds_cz = [r["cz"] for r in group]

        def mean(key):
            return float(np.mean([r[key] for r in kept]))

        lines.append(
            f"| {family} | {n} | {p} | {mean('min_cz'):.0f} | {mean('cz'):.0f} | "
            f"{np.mean(seeds_cz):.0f} ± {np.std(seeds_cz):.0f} | {mean('depth'):.0f} | "
            f"{mean('depth_2q'):.0f} | {mean('physical_qubits'):.1f} | "
            f"{mean('duration_ns') / 1e3:.1f} | {mean('esp'):.3f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--coupling-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--p", type=int, nargs="+", default=list(DEPTHS))
    args = parser.parse_args()

    device = eqe.build_device(
        eqe.load_calibration(args.calibration), eqe.load_coupling_map(args.coupling_map)
    )
    target = eqe.build_target(device)
    instances = inst.make_instances(device.edges)
    args.out.mkdir(parents=True, exist_ok=True)
    inst.save_instances(args.out / "instances.json", instances)

    rows = []
    for instance in instances:
        for p in args.p:
            routed = route(inst.qaoa_circuit(instance, p), target, range(args.seeds))
            for s in routed.per_seed:
                rows.append(
                    {
                        "instance": instance.name,
                        "family": instance.family,
                        "n": instance.n,
                        "p": p,
                        "min_cz": 2 * instance.num_edges * p,
                        "kept": s.seed == routed.best.seed,
                        **asdict(s),
                    }
                )
            b = routed.best
            print(
                f"{instance.name} p={p}: CZ {b.cz} (min {2 * instance.num_edges * p}), "
                f"2q depth {b.depth_2q}, {b.physical_qubits} qubits, ESP {b.esp:.3f}",
                flush=True,
            )

    with open(args.out / "routing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = _summary_table(rows)
    cal = device.calibration
    (args.out / "routing_summary.md").write_text(
        f"Routing on {cal.backend_name}, calibration {cal.timestamp}; "
        f"{args.seeds} transpiler seeds, the highest-ESP one kept.\n\n{table}\n"
    )
    print("\n" + table)


if __name__ == "__main__":
    main()
