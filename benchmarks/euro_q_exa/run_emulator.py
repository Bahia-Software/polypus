"""
Emulation sweep for the Euro-Q-Exa benchmark (#391, phase F3b).

For each instance and depth p: route the QAOA circuit once (``routing.route``,
highest-ESP of the transpiler seeds), train it without noise on polypus's native
backend (DE, the cut as a native ``polypus.Ising``) once per DE seed, then run
the trained parameters under every condition:

  A    ideal, logical circuit (no routing, no noise)
  B    ideal, routed circuit (must match A: routing does not change the state)
  C    routed, EQE gate + readout noise                     <- main result
  D    C with readout mitigation (computed from C's counts, no extra run)
  I    C plus idle noise (relaxation while qubits wait, T2 echo)
  E    C with slow gates (r 60 ns, cz 100 ns)               } sensitivity: best DE
  I_R  I with the Ramsey T2 while waiting (pessimistic)     } seed, --sensitivity-p
  P0   uniformly random cuts (p = 0), analytic
  FA   published fixed angles for 3-regular graphs, under A, C and I

Noisy runs compact the routed circuit to the qubits it uses and go through
``polypus.run_quantum_circuit`` with the EQE noise model (Aer, statevector).
Each (instance, p) is written to ``<out>/sweep/<instance>_p<p>.json.gz`` with
its rows and counts; an existing file is skipped, so an interrupted sweep
resumes where it stopped.

Readout mitigation (D): with independent symmetric readout errors e on each
qubit, the measured <Z_i Z_j> is the ideal one times (1 - 2 e_i)(1 - 2 e_j), so
dividing by that factor recovers it exactly in expectation. It corrects the
approximation ratio (a sum of <Z_i Z_j>), not the probability of sampling the
optimum.

Usage:
    python benchmarks/euro_q_exa/run_emulator.py \
        --calibration calibration_data_Euro-Q-Exa_20261019_active_qubits.json \
        --coupling-map coupling_map.txt --out benchmarks/euro_q_exa/results \
        --sizes 16 --p 1 2 3
"""

import argparse
import gzip
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import device as eqe
import instances as inst
import numpy as np
import polypus
import routing

# Wurtz & Lykov, "The fixed angle conjecture for QAOA on regular MaxCut
# graphs", arXiv:2107.00677, Table 1 (D = 3), in this module's convention:
# exp(-i γ C) with C = sum (1 - ZZ)/2, then exp(-i β sum X). Guaranteed
# ratios on 3-regular graphs: 0.6925, 0.7559, 0.7924.
FIXED_ANGLES = {
    1: {"gamma": [0.616], "beta": [0.393]},
    2: {"gamma": [0.488, 0.898], "beta": [0.555, 0.293]},
    3: {"gamma": [0.422, 0.798, 0.937], "beta": [0.609, 0.459, 0.235]},
}
DE_SEEDS = (1, 2, 3)
# Training configuration, chosen in F3b on n = 16 (exact ideal ratio, 3
# instances x 3 seeds): the default DE (rand1bin, population 20, 50 generations,
# 1000 shots, early stop) stopped at generation 21-30 and fell short of the
# fixed angles (0.815 / 0.826 vs 0.840 / 0.882 at p = 2 / 3); SHADE with these
# values and no early stop reaches 0.839 / 0.874, ~20 s per run at n = 16.
DE_STRATEGY = "shade"
DE_POPULATION = 40
DE_GENERATIONS = 300
DE_TRAIN_SHOTS = 4000
SLOW_GATES = eqe.NoiseAssumptions(r_duration_ns=60, cz_duration_ns=100)
RAMSEY_IDLE = eqe.NoiseAssumptions(idle_t2="ramsey")


def fixed_angle_params(p: int) -> list[float]:
    """Fixed angles in the circuits' parameter order: betas, then gammas."""
    angles = FIXED_ANGLES[p]
    return list(angles["beta"]) + list(angles["gamma"])


def measured_qubits(circuit) -> list[int]:
    """Device qubit read into each classical bit of a routed circuit: entry k
    is where logical qubit k (clbit k of ``measure_all``) ends up."""
    where = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            where[circuit.find_bit(item.clbits[0]).index] = circuit.find_bit(
                item.qubits[0]
            ).index
    return [where[k] for k in range(len(where))]


def zz_expectations(counts, n: int, edges) -> np.ndarray:
    """<Z_i Z_j> per edge from counts (qubit 0 rightmost)."""
    keys = list(counts)
    weights = np.array([counts[k] for k in keys], dtype=float)
    bits = np.array([[int(c) for c in k[::-1]] for k in keys], dtype=np.int8)
    z = 1 - 2 * bits
    return np.array(
        [(weights * z[:, i] * z[:, j]).sum() / weights.sum() for i, j, _ in edges]
    )


def mitigated_ratio(counts, instance: inst.Instance, readout_error) -> float:
    """Approximation ratio with each <Z_i Z_j> divided by (1-2e_i)(1-2e_j);
    ``readout_error[k]`` is the error of the qubit logical qubit k is read on."""
    zz = zz_expectations(counts, instance.n, instance.edges)
    e = np.asarray(readout_error)
    cut = sum(
        w * (1 - zz[k] / ((1 - 2 * e[i]) * (1 - 2 * e[j]))) / 2
        for k, (i, j, w) in enumerate(instance.edges)
    )
    return float(cut / instance.optimum)


def _seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def _run(circuit, shots, seed, device=None, assumptions=None):
    """Counts and seconds for a bound circuit: noiseless when ``device`` is
    None, else compacted and run under the EQE noise model."""
    small, qubits = eqe.compact(circuit)
    kwargs = {"sim_method": "statevector"}
    if device is not None:
        kwargs["noise_model"] = eqe.build_noise_model(
            device, assumptions, qubits=qubits
        )[0]
    start = time.perf_counter()
    result = polypus.run_quantum_circuit(
        small, shots=shots, infrastructure="local", seed=seed, **kwargs
    )
    return result.counts[0], time.perf_counter() - start


def _row(instance, p, label, condition, counts, seconds, shots, **extra):
    best = max(inst.cut_value(k, instance.edges) for k in counts)
    return {
        "instance": instance.name,
        "family": instance.family,
        "n": instance.n,
        "p": p,
        "run": label,
        "condition": condition,
        "ar": inst.approximation_ratio(counts, instance),
        "p_opt": inst.optimum_probability(counts, instance),
        "best_cut_ratio": best / instance.optimum,
        "shots": shots,
        "seconds": seconds,
        **extra,
    }


def evaluate(instance, p, device, target, args, log=print):
    """All rows and counts of one (instance, p)."""
    logical = inst.qaoa_circuit(instance, p)
    routed = routing.route(logical, target, range(args.routing_seeds))
    circuit = routed.circuit
    readout_error = [1 - device.readout_fidelity[q] for q in measured_qubits(circuit)]
    shots = args.eval_shots
    rows, counts_out = [], {}
    default = eqe.NoiseAssumptions()

    def record(label, condition, counts, seconds, **extra):
        rows.append(
            _row(instance, p, label, condition, counts, seconds, shots, **extra)
        )
        counts_out[f"{label}/{condition}"] = counts

    def run_conditions(label, values, conditions, **extra):
        bound = circuit.assign_parameters(values)
        for condition in conditions:
            seed = _seed(instance.name, p, label, condition)
            if condition == "A":
                counts, s = _run(logical.assign_parameters(values), shots, seed)
            elif condition == "B":
                counts, s = _run(bound, shots, seed)
            elif condition == "C":
                counts, s = _run(bound, shots, seed, device, default)
                record(label, "C", counts, s, **extra)
                rows.append(
                    {
                        **rows[-1],
                        "condition": "D",
                        "ar": mitigated_ratio(counts, instance, readout_error),
                        "p_opt": None,
                        "best_cut_ratio": None,
                        "seconds": 0.0,
                    }
                )
                ar_c, ar_d = rows[-2]["ar"], rows[-1]["ar"]
                log(
                    f"  {instance.name} p={p} {label} C: AR {ar_c:.3f} ({s:.0f} s), D: {ar_d:.3f}"
                )
                continue
            elif condition == "E":
                counts, s = _run(bound, shots, seed, device, SLOW_GATES)
            elif condition == "I":
                idle = eqe.add_idle_noise(bound, device, default)
                counts, s = _run(idle, shots, seed, device, default)
            elif condition == "I_R":
                idle = eqe.add_idle_noise(bound, device, RAMSEY_IDLE)
                counts, s = _run(idle, shots, seed, device, RAMSEY_IDLE)
            else:
                raise ValueError(condition)
            record(label, condition, counts, s, **extra)
            log(
                f"  {instance.name} p={p} {label} {condition}: AR {rows[-1]['ar']:.3f} ({s:.0f} s)"
            )

    trained = []
    for seed in args.de_seeds:
        start = time.perf_counter()
        result = polypus.train(
            inst.native_qaoa_circuit(instance, p),
            polypus.DE(
                generations=args.generations,
                population_size=args.population,
                tolerance=args.tolerance,
                patience=args.patience or args.generations,
                seed=seed,
                strategy=args.strategy,
            ),
            shots=args.train_shots,
            n_qpus=1,
            dimensions=2 * p,
            expectation_function=inst.ising_cost(instance),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id=f"eqe-{instance.name}-p{p}",
            backend="polypus",
            seed=seed,
        )
        extra = {
            "train_ar": result.best_fitness / instance.optimum,
            "train_iterations": result.iterations_run,
            "train_seconds": time.perf_counter() - start,
            "params": list(result.best_params),
        }
        trained.append((result.best_fitness, seed, extra))
        run_conditions(f"de{seed}", result.best_params, ("A", "B", "C", "I"), **extra)

    if p in args.sensitivity_p:
        _, seed, extra = max(trained, key=lambda t: (t[0], -t[1]))
        run_conditions(f"de{seed}", extra["params"], ("E", "I_R"), **extra)

    if instance.family == "3reg" and p in FIXED_ANGLES:
        params = fixed_angle_params(p)
        run_conditions("fa", params, ("A", "C", "I"), params=params)

    rows.append(
        {
            "instance": instance.name,
            "family": instance.family,
            "n": instance.n,
            "p": p,
            "run": "p0",
            "condition": "P0",
            "ar": instance.random_cut / instance.optimum,
        }
    )
    return {"rows": rows, "counts": counts_out, "routing": asdict(routed.best)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--coupling-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(inst.SIZES))
    parser.add_argument("--p", type=int, nargs="+", default=list(routing.DEPTHS))
    parser.add_argument("--families", nargs="+", default=list(inst.FAMILIES))
    parser.add_argument("--de-seeds", type=int, nargs="+", default=list(DE_SEEDS))
    parser.add_argument("--strategy", default=DE_STRATEGY)
    parser.add_argument("--population", type=int, default=DE_POPULATION)
    parser.add_argument("--generations", type=int, default=DE_GENERATIONS)
    # DE stops early when the best fitness improves by less than `tolerance`
    # over `patience` generations; the defaults here never stop early.
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument(
        "--patience", type=int, default=None, help="default: --generations"
    )
    parser.add_argument("--train-shots", type=int, default=DE_TRAIN_SHOTS)
    parser.add_argument("--eval-shots", type=int, default=4000)
    parser.add_argument("--routing-seeds", type=int, default=routing.DEFAULT_SEEDS)
    parser.add_argument("--sensitivity-p", type=int, nargs="*", default=[1])
    args = parser.parse_args()

    device = eqe.build_device(
        eqe.load_calibration(args.calibration), eqe.load_coupling_map(args.coupling_map)
    )
    target = eqe.build_target(device)
    sweep = args.out / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    for instance in inst.make_instances(device.edges, args.sizes):
        if instance.family not in args.families:
            continue
        for p in args.p:
            path = sweep / f"{instance.name}_p{p}.json.gz"
            if path.exists():
                print(f"skip {path.name} (done)", flush=True)
                continue
            start = time.perf_counter()
            unit = evaluate(
                instance, p, device, target, args, log=lambda m: print(m, flush=True)
            )
            unit["meta"] = {
                "calibration": device.calibration.timestamp,
                "polypus": polypus.__version__,
                "args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "seconds": time.perf_counter() - start,
            }
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as f:
                json.dump(unit, f)
            tmp.rename(path)
            print(
                f"done {path.name} in {unit['meta']['seconds'] / 60:.1f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()
