"""Tests for the routing study (`benchmarks/euro_q_exa/routing.py`) on the
synthetic device of `synthetic.py` (compact coupling 0-1, 0-2, 1-3, 2-3, 3-4:
a square with a tail, no triangle)."""

import math

import device as eqe
import instances as inst
import pytest
import routing
from qiskit import QuantumCircuit
from synthetic import COUPLING

SEEDS = range(3)


@pytest.fixture
def target(device):
    return eqe.build_target(device)


def _device_edges():
    return {tuple(sorted(e)) for e in COUPLING}


def test_esp_multiplies_the_calibrated_successes(device, target):
    qc = QuantumCircuit(device.num_qubits, 1)
    qc.r(0.1, 0.2, 0)
    qc.cz(1, 0)
    qc.barrier()
    qc.delay(100, 0, unit="dt")
    qc.measure(0, 0)
    expected = (
        device.one_qubit_fidelity[0]
        * device.cz_fidelity[(0, 1)]
        * device.readout_fidelity[0]
    )
    assert routing.esp(qc, target) == pytest.approx(expected)


def test_routed_circuit_is_native_and_on_the_coupling_map(target):
    instance = inst.lattice_instance(4, 0, COUPLING)
    routed = routing.route(inst.qaoa_circuit(instance, 1), target, SEEDS)
    circuit = routed.circuit
    assert set(circuit.count_ops()) <= {"r", "cz", "measure", "delay", "barrier"}
    for item in circuit.data:
        if item.operation.name == "cz":
            pair = tuple(sorted(circuit.find_bit(q).index for q in item.qubits))
            assert pair in _device_edges()
    assert circuit.count_ops()["measure"] == instance.n
    assert [p.name for p in circuit.parameters] == ["β[0]", "γ[0]"]


def test_device_subgraph_needs_no_swaps(target):
    instance = inst.lattice_instance(4, 0, COUPLING)
    for p in (1, 2):
        routed = routing.route(inst.qaoa_circuit(instance, p), target, SEEDS)
        assert routed.best.cz == 2 * instance.num_edges * p
        assert routed.best.physical_qubits == instance.n


def test_triangle_needs_swaps_on_a_triangle_free_device(target):
    triangle = inst._finish(
        "tri", "3reg", 3, 0, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    )
    routed = routing.route(inst.qaoa_circuit(triangle, 1), target, SEEDS)
    assert routed.best.cz > 2 * triangle.num_edges


def test_kept_run_is_the_highest_esp_seed(target):
    instance = inst.lattice_instance(5, 1, COUPLING)
    routed = routing.route(inst.qaoa_circuit(instance, 2), target, SEEDS)
    assert [s.seed for s in routed.per_seed] == list(SEEDS)
    assert routed.best.esp == max(s.esp for s in routed.per_seed)
    assert routed.best == routing.stats(routed.circuit, target, routed.best.seed)


def test_stats_describe_the_schedule(device, target):
    instance = inst.lattice_instance(4, 0, COUPLING)
    routed = routing.route(inst.qaoa_circuit(instance, 1), target, SEEDS)
    s = routed.best
    assert 0 < s.depth_2q <= s.cz
    assert s.depth_2q < s.depth
    # The schedule holds at least the CZ layers plus the measurement.
    assert s.duration_ns >= s.depth_2q * 60 + 1000
    assert 0 < s.esp < 1
    assert math.isclose(s.esp, routing.esp(routed.circuit, target))


def test_summary_table_averages_the_kept_runs():
    base = {
        "family": "3reg",
        "n": 16,
        "p": 1,
        "min_cz": 48,
        "one_qubit": 0,
        "depth": 10,
    }
    rows = [
        {**base, "instance": "a", "kept": True, "cz": 70, "depth_2q": 30,
         "physical_qubits": 18, "duration_ns": 5000, "esp": 0.2},
        {**base, "instance": "a", "kept": False, "cz": 90, "depth_2q": 40,
         "physical_qubits": 18, "duration_ns": 6000, "esp": 0.1},
        {**base, "instance": "b", "kept": True, "cz": 80, "depth_2q": 34,
         "physical_qubits": 17, "duration_ns": 5400, "esp": 0.3},
    ]  # fmt: skip
    table = routing._summary_table(rows)
    line = table.splitlines()[-1]
    assert line == "| 3reg | 16 | 1 | 48 | 75 | 80 ± 8 | 10 | 32 | 17.5 | 5.2 | 0.250 |"
