"""Tests for the native MaxCut cost and the cost pilot
(`instances.ising_cost`, `benchmarks/euro_q_exa/cost_pilot.py`). They need the
polypus extension (the wheel built from this branch)."""

import json
import sys

import instances as inst
import numpy as np
import pytest
from synthetic import write_files

polypus = pytest.importorskip("polypus")


def test_ising_cost_is_the_cut_and_training_maximizes_it():
    # One edge: the optimum is 1 and p = 1 reaches it exactly, so DE on the
    # native backend must climb close to 1. A wrong sign in the Ising mapping
    # (or in the QAOA layer) would drive the fitness to 0 instead.
    edge = inst._finish("edge", "3reg", 2, 0, [(0, 1, 1.0)])
    result = polypus.train(
        inst.native_qaoa_circuit(edge, 1),
        polypus.DE(generations=15, population_size=8, seed=3),
        shots=400,
        n_qpus=1,
        dimensions=2,
        expectation_function=inst.ising_cost(edge),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="test-ising-edge",
        backend="polypus",
        seed=3,
    )
    assert 0.9 < result.best_fitness <= 1.0


def test_ising_cost_handles_signed_weights():
    # Two disjoint edges, -1 and +1: the best cut takes only the +1 edge.
    pair = inst._finish("signs", "lattice", 4, 0, [(0, 1, -1.0), (2, 3, 1.0)])
    result = polypus.train(
        inst.native_qaoa_circuit(pair, 1),
        polypus.DE(generations=15, population_size=8, seed=4),
        shots=400,
        n_qpus=1,
        dimensions=2,
        expectation_function=inst.ising_cost(pair),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="test-ising-signs",
        backend="polypus",
        seed=4,
    )
    assert 0.9 < result.best_fitness <= pair.optimum


@pytest.mark.parametrize("p", [1, 2])
def test_native_circuit_matches_the_qiskit_one(p):
    # Same state for the same parameter vector: trained parameters transfer.
    from qiskit.quantum_info import Statevector

    instance = inst._finish(
        "signs", "lattice", 4, 0, [(0, 1, -1.0), (1, 2, 1.0), (2, 3, 1.0)]
    )
    values = [0.3, -0.7, 1.1, 0.4][: 2 * p]
    native = polypus.statevector(
        inst.native_qaoa_circuit(instance, p, measure=False), values
    )
    qiskit_circuit = inst.qaoa_circuit(instance, p).remove_final_measurements(
        inplace=False
    )
    expected = Statevector(qiskit_circuit.assign_parameters(values)).data
    overlap = abs(np.vdot(expected, np.asarray(native)))
    assert overlap == pytest.approx(1.0, abs=1e-10)


def test_cost_pilot_runs_end_to_end(tmp_path, monkeypatch):
    import cost_pilot

    cal, cmap = write_files(tmp_path)
    out = tmp_path / "out"
    argv = [
        "cost_pilot.py",
        "--calibration", str(cal),
        "--coupling-map", str(cmap),
        "--out", str(out),
        "--sizes", "4",
        "--p", "1",
        "--shots", "50",
        "--population", "4",
        "--generations", "1",
        "--routing-seeds", "1",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    cost_pilot.main()
    report = json.loads((out / "cost_pilot.json").read_text())
    (row,) = report["rows"]
    assert (row["n"], row["p"]) == (4, 1)
    for key in ("pilot_fitness_ar", "noisy_gates_ar", "noisy_idle_ar"):
        assert 0 < row[key] <= 1
    for key in (
        "train_polypus_s_per_eval",
        "noisy_gates_s_per_shot",
        "noisy_idle_s_per_shot",
    ):
        assert row[key] > 0
    assert report["meta"]["full_eval_shots"] == 4000
