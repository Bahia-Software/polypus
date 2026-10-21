"""
Cost pilot for the Euro-Q-Exa benchmark (#391, phase F3a).

Measures, on one 3-regular instance per size and each depth p, what the full
sweep (F3b) will cost on this machine, and projects it:

  * noiseless training: a short ``polypus.train`` run (DE, the cut as a native
    ``polypus.Ising``) on the logical circuit, on the native backend
    (``backend="polypus"``, with ``instances.native_qaoa_circuit``) and on Aer
    (the Qiskit circuit), timed per evaluation;
  * noisy evaluation: the routed circuit at the trained parameters, compacted
    and run through ``polypus.run_quantum_circuit`` with the EQE noise model,
    with gate and readout noise only and with idle noise added, timed per shot.

Wall-clock times depend on the machine and its load; the projections are for
the configuration given on the command line (``--full-*``).

Usage:
    python benchmarks/euro_q_exa/cost_pilot.py \
        --calibration calibration_data_Euro-Q-Exa_20261019_active_qubits.json \
        --coupling-map coupling_map.txt --out benchmarks/euro_q_exa/results
"""

import argparse
import json
import os
import time
from pathlib import Path

import device as eqe
import instances as inst
import polypus
import routing


def _train(circuit, cost, dimensions, backend, shots, population, generations):
    method = polypus.DE(generations=generations, population_size=population, seed=1)
    start = time.perf_counter()
    result = polypus.train(
        circuit,
        method,
        shots=shots,
        n_qpus=1,
        dimensions=dimensions,
        expectation_function=cost,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="eqe-cost-pilot",
        backend=backend,
        seed=1,
    )
    elapsed = time.perf_counter() - start
    evaluations = population * (result.iterations_run + 1)
    return result, elapsed / evaluations


def _noisy_run(circuit, device, assumptions, shots):
    small, qubits = eqe.compact(circuit)
    model, _ = eqe.build_noise_model(device, assumptions, qubits=qubits)
    start = time.perf_counter()
    result = polypus.run_quantum_circuit(
        small,
        shots=shots,
        infrastructure="local",
        noise_model=model,
        sim_method="statevector",
        seed=1,
    )
    return result.counts[0], (time.perf_counter() - start) / shots


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--coupling-map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(inst.SIZES))
    parser.add_argument("--p", type=int, nargs="+", default=list(routing.DEPTHS))
    parser.add_argument("--shots", type=int, default=1000, help="shots in the pilot")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument(
        "--generations", type=int, default=2, help="pilot DE generations"
    )
    parser.add_argument("--full-generations", type=int, default=50)
    parser.add_argument("--full-train-shots", type=int, default=1000)
    parser.add_argument("--full-eval-shots", type=int, default=4000)
    parser.add_argument("--routing-seeds", type=int, default=3)
    args = parser.parse_args()

    device = eqe.build_device(
        eqe.load_calibration(args.calibration), eqe.load_coupling_map(args.coupling_map)
    )
    assumptions = eqe.NoiseAssumptions()
    target = eqe.build_target(device, assumptions)
    full_evaluations = args.population * (args.full_generations + 1)

    rows = []
    for n in args.sizes:
        instance = inst.regular_instance(n, 0)
        cost = inst.ising_cost(instance)
        for p in args.p:
            logical = inst.qaoa_circuit(instance, p)
            native = inst.native_qaoa_circuit(instance, p)
            row = {"n": n, "p": p}
            for backend, circuit in (("polypus", native), ("aer", logical)):
                result, per_eval = _train(
                    circuit,
                    cost,
                    2 * p,
                    backend,
                    args.shots,
                    args.population,
                    args.generations,
                )
                row[f"train_{backend}_s_per_eval"] = per_eval
                if backend == "polypus":
                    # Same parameter order as the Qiskit circuit (instances.py).
                    params = result.best_params
                    row["pilot_fitness_ar"] = result.best_fitness / instance.optimum

            routed = routing.route(logical, target, range(args.routing_seeds)).circuit
            bound = routed.assign_parameters(params)
            counts, row["noisy_gates_s_per_shot"] = _noisy_run(
                bound, device, assumptions, args.shots
            )
            row["noisy_gates_ar"] = inst.approximation_ratio(counts, instance)
            idle = eqe.add_idle_noise(bound, device, assumptions)
            counts, row["noisy_idle_s_per_shot"] = _noisy_run(
                idle, device, assumptions, args.shots
            )
            row["noisy_idle_ar"] = inst.approximation_ratio(counts, instance)

            row["proj_train_noiseless_h"] = (
                row["train_polypus_s_per_eval"] * full_evaluations / 3600
            )
            row["proj_train_noisy_gates_h"] = (
                row["noisy_gates_s_per_shot"]
                * args.full_train_shots
                * full_evaluations
                / 3600
            )
            row["proj_eval_noisy_idle_min"] = (
                row["noisy_idle_s_per_shot"] * args.full_eval_shots / 60
            )
            rows.append(row)
            print(json.dumps(row), flush=True)

    meta = {
        "cpus": len(os.sched_getaffinity(0)),
        "polypus": polypus.__version__,
        "population": args.population,
        "full_generations": args.full_generations,
        "full_train_shots": args.full_train_shots,
        "full_eval_shots": args.full_eval_shots,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost_pilot.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
