"""Tests for the MaxCut instances and QAOA circuits
(`benchmarks/euro_q_exa/instances.py`)."""

import math

import instances as inst
import networkx as nx
import numpy as np
import polypus
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector
from synthetic import COUPLING

TRIANGLE = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]


# ---- optimum and cut values ----------------------------------------------------


@pytest.mark.parametrize(
    ("n", "edges", "optimum"),
    [
        (3, TRIANGLE, 2.0),
        (4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)], 4.0),
        # A negative edge: cut {1} vs {0, 2} takes both +1 edges, leaves the -1.
        (3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, -1.0)], 2.0),
    ],
)
def test_brute_force_optimum(n, edges, optimum):
    assert inst.maxcut_optimum(n, edges) == optimum


def test_cut_value_reads_qubit_zero_rightmost():
    edges = [(0, 1, 1.0), (1, 2, 5.0)]
    # "001": qubit 0 is 1, qubits 1 and 2 are 0 -> only edge (0, 1) is cut.
    assert inst.cut_value("001", edges) == 1.0
    # "100": qubit 2 is 1 -> only edge (1, 2) is cut.
    assert inst.cut_value("100", edges) == 5.0


def test_approximation_ratio_and_optimum_probability_from_counts():
    instance = inst._finish("t", "3reg", 3, 0, TRIANGLE)
    counts = {"001": 3, "000": 1}  # cut 2 (optimal) three times, cut 0 once
    assert inst.approximation_ratio(counts, instance) == pytest.approx(1.5 / 2.0)
    assert inst.optimum_probability(counts, instance) == pytest.approx(0.75)
    assert instance.random_cut == 1.5


# ---- generated instances -------------------------------------------------------


def test_regular_instances_are_reproducible_three_regular_graphs():
    a = inst.regular_instance(16, 1)
    assert a == inst.regular_instance(16, 1)
    assert a != inst.regular_instance(16, 2)
    graph = nx.Graph([(i, j) for i, j, _ in a.edges])
    assert graph.number_of_nodes() == 16
    assert all(d == 3 for _, d in graph.degree())
    assert all(w == 1.0 for _, _, w in a.edges)
    assert a.optimum == inst.maxcut_optimum(16, a.edges)


def test_lattice_instance_is_a_connected_signed_device_subgraph():
    a = inst.lattice_instance(4, 0, COUPLING)
    assert a == inst.lattice_instance(4, 0, COUPLING)
    device_edges = {tuple(sorted(e)) for e in COUPLING}
    q = a.device_qubits
    assert len(set(q)) == 4
    # Every instance edge is a device edge, and every device edge among the
    # chosen qubits is in the instance.
    mapped = {tuple(sorted((q[i], q[j]))) for i, j, _ in a.edges}
    assert mapped == {e for e in device_edges if set(e) <= set(q)}
    assert nx.is_connected(nx.Graph([(i, j) for i, j, _ in a.edges]))
    assert {w for _, _, w in a.edges} <= {-1.0, 1.0}


def test_make_instances_covers_both_families():
    # The synthetic device has 5 qubits: smaller sizes than the benchmark's.
    instances = inst.make_instances(COUPLING, sizes=(4,))
    assert len(instances) == 2 * inst.INSTANCES_PER_SIZE
    assert {(i.family, i.n) for i in instances} == {("3reg", 4), ("lattice", 4)}
    assert len({i.name for i in instances}) == len(instances)


def test_instances_round_trip_through_json(tmp_path):
    instances = [inst.regular_instance(16, 0), inst.lattice_instance(4, 1, COUPLING)]
    path = tmp_path / "instances.json"
    inst.save_instances(path, instances)
    assert inst.load_instances(path) == instances


# ---- QAOA circuit --------------------------------------------------------------


def _expected_cut(instance, circuit, values):
    bound = circuit.remove_final_measurements(inplace=False).assign_parameters(values)
    terms = [("I" * instance.n, instance.random_cut)]
    for i, j, w in instance.edges:
        label = ["I"] * instance.n
        label[instance.n - 1 - i] = label[instance.n - 1 - j] = "Z"
        terms.append(("".join(label), -w / 2))
    return Statevector(bound).expectation_value(SparsePauliOp.from_list(terms)).real


def test_qaoa_parameters_are_betas_then_gammas():
    circuit = inst.qaoa_circuit(inst.regular_instance(16, 0), 2)
    assert [p.name for p in circuit.parameters] == ["β[0]", "β[1]", "γ[0]", "γ[1]"]
    assert circuit.num_qubits == 16
    assert circuit.count_ops()["measure"] == 16


def test_qaoa_sign_maximizes_the_cut():
    # p = 1 cuts a single edge with certainty at β = π/8, γ = π/2; with the
    # opposite sign of γ it would find the minimum cut instead.
    edge = inst._finish("edge", "3reg", 2, 0, [(0, 1, 1.0)])
    circuit = inst.qaoa_circuit(edge, 1)
    assert _expected_cut(edge, circuit, [math.pi / 8, math.pi / 2]) == pytest.approx(
        1.0
    )
    assert _expected_cut(edge, circuit, [math.pi / 8, -math.pi / 2]) == pytest.approx(
        0.0
    )


def test_qaoa_weights_enter_the_cost_layer():
    # Two disjoint edges, -1 and +1: at the angles that cut a +1 edge with
    # certainty, the -1 edge is left uncut, which is the optimum.
    pair = inst._finish("signs", "lattice", 4, 0, [(0, 1, -1.0), (2, 3, 1.0)])
    circuit = inst.qaoa_circuit(pair, 1)
    value = _expected_cut(pair, circuit, [math.pi / 8, math.pi / 2])
    assert pair.optimum == 1.0
    assert value == pytest.approx(1.0)


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
