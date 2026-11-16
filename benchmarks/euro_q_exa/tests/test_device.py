"""Tests for the Euro-Q-Exa device model (`benchmarks/euro_q_exa/device.py`),
on the synthetic calibration of `synthetic.py`."""

import math

import device as eqe
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveDensityMatrix
from qiskit_aer.noise import thermal_relaxation_error
from synthetic import COMPACT, COUPLING, CZ_ORIGINAL, NAN, snapshot, write_files

# ---- labels ------------------------------------------------------------------


def test_compact_labels_skip_the_out_of_service_qubit(device):
    assert device.to_original == (0, 1, 3, 4, 5)
    assert device.num_qubits == 5
    np.testing.assert_allclose(device.t1_us, [40.0, 50.0, 45.0, 30.0, 10.0])
    np.testing.assert_allclose(device.readout_fidelity, [0.96, 0.95, 0.98, 0.90, 0.93])


def test_cz_fidelities_follow_the_label_mapping(device):
    expected = {}
    for key, fidelity in CZ_ORIGINAL.items():
        a, b = (COMPACT[int(x)] for x in key.split(","))
        expected[(min(a, b), max(a, b))] = fidelity
    assert device.cz_fidelity == expected


@pytest.mark.parametrize("prefix", [True, False])
def test_coupling_map_reads_with_or_without_label(tmp_path, prefix):
    _, cmap = write_files(tmp_path, prefix=prefix)
    assert eqe.load_coupling_map(cmap) == [tuple(p) for p in COUPLING]


def test_more_qubits_in_the_map_than_active_is_refused(tmp_path):
    cal, cmap = write_files(tmp_path, coupling=COUPLING + [[4, 5]])
    with pytest.raises(ValueError, match="5 active calibrated qubits"):
        eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))


def test_coupling_edge_without_calibration_is_refused(tmp_path):
    # Compact (0, 3) = original (0, 4): not a calibrated edge.
    cal, cmap = write_files(tmp_path, coupling=COUPLING + [[0, 3]])
    with pytest.raises(ValueError, match="no calibrated CZ"):
        eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))


def test_calibrated_edge_missing_from_the_map_is_refused(tmp_path):
    # Swap compact edge (4, 3) = original (5, 4) for (4, 2) = original (5, 3):
    # original (4, 5) is still calibrated but no longer in the map.
    raw = snapshot(two_q_fidelity={**CZ_ORIGINAL, "3,5": 0.98})
    cal, cmap = write_files(tmp_path, raw=raw, coupling=COUPLING[:-1] + [[4, 2]])
    with pytest.raises(
        ValueError, match=r"missing from the coupling map: \[\(4, 5\)\]"
    ):
        eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))


def test_missing_readout_on_an_active_qubit_is_refused(tmp_path):
    raw = snapshot(readout_fidelity=[0.96, 0.95, 0.97, NAN, 0.90, 0.93])
    cal, cmap = write_files(tmp_path, raw=raw)
    with pytest.raises(
        ValueError, match=r"readout fidelity missing for compact qubits \[2\]"
    ):
        eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))


# ---- target ------------------------------------------------------------------


def test_target_holds_the_native_gates_with_calibrated_errors(device):
    target = eqe.build_target(device)
    assert set(target.operation_names) == {"r", "cz", "measure", "delay"}
    # CZ is symmetric: both orientations of each of the 5 edges.
    assert len(target["cz"]) == 10
    for (a, b), fidelity in device.cz_fidelity.items():
        assert target["cz"][(a, b)].error == pytest.approx(1 - fidelity)
        assert target["cz"][(b, a)].error == pytest.approx(1 - fidelity)
    for c in range(device.num_qubits):
        assert target["r"][(c,)].error == pytest.approx(
            1 - device.one_qubit_fidelity[c]
        )
        assert target["measure"][(c,)].error == pytest.approx(
            1 - device.readout_fidelity[c]
        )


def test_target_carries_the_assumed_durations(device):
    target = eqe.build_target(device, eqe.NoiseAssumptions(60, 100, 1500))
    assert target.dt == eqe.DT_S
    assert target["r"][(0,)].duration == pytest.approx(60e-9)
    assert target["cz"][(0, 1)].duration == pytest.approx(100e-9)
    assert target["measure"][(0,)].duration == pytest.approx(1500e-9)


# ---- noise model -------------------------------------------------------------

ASSUMPTIONS = [
    eqe.NoiseAssumptions(),
    eqe.NoiseAssumptions(r_duration_ns=60, cz_duration_ns=100),
]


@pytest.mark.parametrize("assumptions", ASSUMPTIONS)
def test_model_reproduces_the_calibrated_fidelities(device, assumptions):
    max_r, max_cz, report = eqe.fidelity_deviation(device, assumptions)
    assert report.relaxation_limited_r == ()
    assert report.relaxation_limited_cz == ()
    assert max_r < 1e-10
    assert max_cz < 1e-10


def test_both_cz_orientations_match_the_calibration(device):
    _, cz_errors, _ = eqe.gate_errors(device)
    assert len(cz_errors) == 10
    for (q0, q1), error in cz_errors.items():
        fidelity = device.cz_fidelity[(min(q0, q1), max(q0, q1))]
        assert average_gate_fidelity(error) == pytest.approx(fidelity, abs=1e-10)


def test_t2_above_t1_is_capped_and_fidelity_still_matches(device):
    # Original qubit 5 (compact 4): T1 10 us, T2 echo 30 us > T1.
    r_errors, _, _ = eqe.gate_errors(device)
    assert average_gate_fidelity(r_errors[4]) == pytest.approx(0.993, abs=1e-10)


def test_gate_worse_from_relaxation_alone_is_reported(tmp_path):
    # A near-perfect calibrated CZ on qubits with T1 = 10 us / 30 us: relaxation
    # over 60 ns alone already exceeds the calibrated infidelity.
    raw = snapshot(two_q_fidelity={**CZ_ORIGINAL, "4,5": 0.9999999})
    cal, cmap = write_files(tmp_path, raw=raw)
    device = eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))
    _, cz_errors, report = eqe.gate_errors(device)
    assert report.relaxation_limited_cz == ((3, 4),)
    assert average_gate_fidelity(cz_errors[(3, 4)]) < 0.9999999


@pytest.mark.parametrize(
    "kwargs",
    [
        {"idle_t2": "hahn"},
        {"r_duration_ns": 0},
        {"cz_duration_ns": -1},
        {"measure_duration_ns": 1.5},
    ],
)
def test_invalid_assumptions_are_refused(kwargs):
    with pytest.raises(ValueError):
        eqe.NoiseAssumptions(**kwargs)


def test_readout_error_is_symmetric_from_the_fidelity(device):
    for c, error in eqe.readout_errors(device).items():
        e = 1 - device.readout_fidelity[c]
        np.testing.assert_allclose(error.probabilities, [[1 - e, e], [e, 1 - e]])


def test_noise_model_runs_on_native_circuits(device):
    model, report = eqe.build_noise_model(device)
    assert set(model.basis_gates) >= {"r", "cz"}
    assert report.relaxation_limited_cz == ()
    # |00> through r(0) and a CZ is unchanged without noise; with this model the
    # flips seen are readout (~4-5 %) plus small gate errors on each qubit.
    qc = QuantumCircuit(2, 2)
    qc.r(0.0, 0.0, 0)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])
    shots = 20_000
    counts = (
        AerSimulator(noise_model=model)
        .run(qc, shots=shots, seed_simulator=1234)
        .result()
        .get_counts()
    )
    flipped_q0 = sum(n for key, n in counts.items() if key[-1] == "1") / shots
    expected = 1 - device.readout_fidelity[0]
    assert math.isclose(flipped_q0, expected, abs_tol=0.01)


# ---- idle noise --------------------------------------------------------------
# Compact qubits used below: 0 = original 0 (T1 40, T2 echo 35, Ramsey 20 us),
# 3 = original 4 (T1 30, echo 25, no Ramsey), 4 = original 5 (T1 10, echo 30
# capped at T1 = 10, Ramsey 12 us capped at 10 too).

WAIT_NS = 1000


def _idle_density_matrix(device, prepare, qubit, assumptions):
    """One-qubit density matrix after `prepare` then a WAIT_NS delay on `qubit`."""
    qc = QuantumCircuit(device.num_qubits)
    prepare(qc, qubit)
    qc.delay(WAIT_NS, qubit, unit="dt")
    qc.append(SaveDensityMatrix(1), [qubit])
    noisy = eqe.add_idle_noise(qc, device, assumptions)
    result = AerSimulator(method="density_matrix").run(noisy).result()
    return np.asarray(result.data(0)["density_matrix"])


@pytest.mark.parametrize(
    ("qubit", "t1_us"),
    [(0, 40.0), (3, 30.0), (4, 10.0)],
)
def test_idle_excited_state_decays_with_t1(device, qubit, t1_us):
    rho = _idle_density_matrix(
        device, lambda qc, q: qc.x(q), qubit, eqe.NoiseAssumptions()
    )
    assert rho[1, 1].real == pytest.approx(
        math.exp(-WAIT_NS / (t1_us * 1e3)), abs=1e-12
    )


@pytest.mark.parametrize(
    ("idle_t2", "qubit", "t2_us"),
    [
        ("echo", 0, 35.0),
        ("ramsey", 0, 20.0),
        # No Ramsey T2 in the snapshot: the echo one is kept.
        ("ramsey", 3, 25.0),
        # T2 above T1 is capped at T1 = 10 us.
        ("echo", 4, 10.0),
        ("ramsey", 4, 10.0),
    ],
)
def test_idle_coherence_decays_with_the_chosen_t2(device, idle_t2, qubit, t2_us):
    assumptions = eqe.NoiseAssumptions(idle_t2=idle_t2)
    rho = _idle_density_matrix(device, lambda qc, q: qc.h(q), qubit, assumptions)
    expected = 0.5 * math.exp(-WAIT_NS / (t2_us * 1e3))
    assert abs(rho[0, 1]) == pytest.approx(expected, abs=1e-12)


def test_idle_t2_reports_the_echo_fallback(device):
    _, fallback = eqe.idle_t2_us(device, eqe.NoiseAssumptions(idle_t2="ramsey"))
    assert fallback == (3,)
    _, fallback = eqe.idle_t2_us(device)
    assert fallback == ()


def test_scheduled_circuit_gets_idle_noise_only_on_used_qubits(device):
    # Compact qubits 0-1-3 form a path; qubits 2 and 4 stay unused.
    gamma = Parameter("gamma")
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])
    qc.rzz(gamma, 0, 1)
    qc.rzz(gamma, 1, 2)
    qc.rx(0.3, 0)
    qc.measure([0, 1, 2], [0, 1, 2])
    assumptions = eqe.NoiseAssumptions()
    scheduled = transpile(
        qc,
        target=eqe.build_target(device, assumptions),
        initial_layout=[0, 1, 3],
        scheduling_method="alap",
        optimization_level=1,
        seed_transpiler=7,
    )
    noisy = eqe.add_idle_noise(scheduled, device, assumptions)

    used = {0, 1, 3}
    touched = {noisy.find_bit(q).index for inst in noisy.data for q in inst.qubits}
    assert touched == used
    ops = noisy.count_ops()
    assert ops["delay"] > 0
    assert ops["quantum_channel"] == ops["delay"]
    assert noisy.parameters == scheduled.parameters
    assert noisy.layout == scheduled.layout

    model, _ = eqe.build_noise_model(device, assumptions)
    counts = (
        AerSimulator(noise_model=model)
        .run(noisy.assign_parameters([0.8]), shots=200, seed_simulator=3)
        .result()
        .get_counts()
    )
    assert sum(counts.values()) == 200
    assert all(len(key) == 3 for key in counts)


def test_idle_noise_needs_a_circuit_on_the_device(device):
    with pytest.raises(ValueError, match="transpile it against build_target"):
        eqe.add_idle_noise(QuantumCircuit(2), device)


# ---- compaction --------------------------------------------------------------


def _scheduled_path_circuit(device, assumptions, measure):
    """QAOA-like layer on compact qubits 0-1-3 (2 and 4 unused), scheduled."""
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])
    qc.rzz(0.7, 0, 1)
    qc.rzz(0.7, 1, 2)
    qc.rx(0.3, [0, 1, 2])
    if measure:
        qc.measure([0, 1, 2], [0, 1, 2])
    return transpile(
        qc,
        target=eqe.build_target(device, assumptions),
        initial_layout=[0, 1, 3],
        scheduling_method="alap",
        optimization_level=1,
        seed_transpiler=7,
    )


@pytest.mark.parametrize("idle_t2", [None, "echo", "ramsey"])
def test_compacted_circuit_matches_the_full_width_one(device, idle_t2):
    assumptions = eqe.NoiseAssumptions(idle_t2=idle_t2 or "echo")
    scheduled = _scheduled_path_circuit(device, assumptions, measure=False)
    full = eqe.add_idle_noise(scheduled, device, assumptions) if idle_t2 else scheduled
    small, qubits = eqe.compact(full)
    assert qubits == (0, 1, 3)
    assert small.num_qubits == 3

    full_model, _ = eqe.build_noise_model(device, assumptions)
    small_model, _ = eqe.build_noise_model(device, assumptions, qubits=qubits)
    full.append(SaveDensityMatrix(3), [0, 1, 3])
    small.append(SaveDensityMatrix(3), [0, 1, 2])
    sim = AerSimulator(method="density_matrix")
    rho_full = sim.run(full, noise_model=full_model).result().data(0)["density_matrix"]
    rho_small = (
        sim.run(small, noise_model=small_model).result().data(0)["density_matrix"]
    )
    np.testing.assert_allclose(np.asarray(rho_small), np.asarray(rho_full), atol=1e-12)


def test_compacted_circuit_keeps_classical_bits_and_parameters(device):
    gamma = Parameter("gamma")
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.rzz(gamma, 0, 1)
    qc.measure([0, 1], [1, 0])
    scheduled = transpile(
        qc,
        target=eqe.build_target(device),
        initial_layout=[3, 4],
        scheduling_method="alap",
        seed_transpiler=1,
    )
    small, qubits = eqe.compact(scheduled)
    assert qubits == (3, 4)
    assert small.cregs == scheduled.cregs
    assert small.parameters == scheduled.parameters
    measured = [
        (small.find_bit(i.qubits[0]).index, small.find_bit(i.clbits[0]).index)
        for i in small.data
        if i.operation.name == "measure"
    ]
    # Logical qubit 0 sits on device qubit 3 = position 0, read into clbit 1.
    assert sorted(measured) == [(0, 1), (1, 0)]


def test_compacted_noise_model_reads_out_with_the_mapped_qubits(device):
    # Device qubits 3 and 4 (readout F 0.90 and 0.93) at positions 0 and 1.
    model, _ = eqe.build_noise_model(device, qubits=(3, 4))
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    shots = 40_000
    counts = (
        AerSimulator(noise_model=model)
        .run(qc, shots=shots, seed_simulator=5)
        .result()
        .get_counts()
    )
    flips = [
        sum(n for k, n in counts.items() if k[-1 - pos] == "1") / shots
        for pos in (0, 1)
    ]
    assert flips[0] == pytest.approx(0.10, abs=0.006)
    assert flips[1] == pytest.approx(0.07, abs=0.006)


def _instruction_names(error):
    return {inst.operation.name for circuit in error.circuits for inst in circuit.data}


def test_no_relaxation_channel_is_kraus(device):
    # Aer 0.17.2 fails on models holding Kraus relaxation (see device._cap_t2).
    # Original qubit 5 (compact 4) has T2 echo and Ramsey > T1, so the cap is
    # exercised by its r gate, its CZ edge (3, 4) and its idle noise.
    r_errors, cz_errors, _ = eqe.gate_errors(device)
    for error in [*r_errors.values(), *cz_errors.values()]:
        assert "kraus" not in _instruction_names(error)
    for idle_t2 in ("echo", "ramsey"):
        # add_idle_noise builds its relaxation from idle_t2_us.
        t2, _ = eqe.idle_t2_us(device, eqe.NoiseAssumptions(idle_t2=idle_t2))
        assert np.all(t2 <= device.t1_us)
        idle = thermal_relaxation_error(device.t1_us[4] * 1e3, t2[4] * 1e3, WAIT_NS)
        assert "kraus" not in _instruction_names(idle)
