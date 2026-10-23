"""Tests for the emulation sweep (`benchmarks/euro_q_exa/run_emulator.py`)."""

import gzip
import json
import sys

import instances as inst
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from synthetic import write_files

polypus = pytest.importorskip("polypus")
import run_emulator as sweep  # noqa: E402  (imports polypus)

# ---- fixed angles ----------------------------------------------------------------


def _exact_ratio(instance, circuit, values):
    bound = circuit.remove_final_measurements(inplace=False).assign_parameters(values)
    terms = [("I" * instance.n, instance.random_cut)]
    for i, j, w in instance.edges:
        label = ["I"] * instance.n
        label[instance.n - 1 - i] = label[instance.n - 1 - j] = "Z"
        terms.append(("".join(label), -w / 2))
    cut = Statevector(bound).expectation_value(SparsePauliOp.from_list(terms)).real
    return cut / instance.optimum


@pytest.mark.parametrize(("p", "guarantee"), [(1, 0.6925), (2, 0.7559), (3, 0.7924)])
def test_fixed_angles_reach_their_published_guarantee(p, guarantee):
    # Wurtz & Lykov's angles guarantee these ratios on every 3-regular graph;
    # reaching them on a benchmark instance checks the angles and this
    # module's conventions (parameter order, signs) end to end.
    instance = inst.regular_instance(16, 0)
    ratio = _exact_ratio(
        instance, inst.qaoa_circuit(instance, p), sweep.fixed_angle_params(p)
    )
    assert ratio >= guarantee - 1e-3


def test_fixed_angle_params_are_betas_then_gammas():
    assert sweep.fixed_angle_params(2) == [0.555, 0.293, 0.488, 0.898]


# ---- read-out bookkeeping and mitigation -----------------------------------------


def test_measured_qubits_follow_the_clbits():
    qc = QuantumCircuit(5, 2)
    qc.measure(3, 0)
    qc.measure(1, 1)
    assert sweep.measured_qubits(qc) == [3, 1]


def test_mitigation_undoes_independent_readout_flips():
    # A perfect cut of one edge, {"01", "10"} half and half, read with flip
    # probabilities 0.1 (qubit 0) and 0.2 (qubit 1): the exact expected counts.
    edge = inst._finish("edge", "3reg", 2, 0, [(0, 1, 1.0)])
    counts = {"01": 3700, "10": 3700, "00": 1300, "11": 1300}
    zz = sweep.zz_expectations(counts, 2, edge.edges)
    np.testing.assert_allclose(zz, [-0.48])
    assert inst.approximation_ratio(counts, edge) == pytest.approx(0.74)
    assert sweep.mitigated_ratio(counts, edge, [0.1, 0.2]) == pytest.approx(1.0)


def test_mitigation_without_readout_error_changes_nothing():
    instance = inst._finish("t", "3reg", 3, 0, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    counts = {"001": 5, "011": 2, "000": 3}
    assert sweep.mitigated_ratio(counts, instance, [0, 0, 0]) == pytest.approx(
        inst.approximation_ratio(counts, instance)
    )


# ---- end to end ------------------------------------------------------------------


def _argv(cal, cmap, out):
    return [
        "run_emulator.py",
        "--calibration", str(cal),
        "--coupling-map", str(cmap),
        "--out", str(out),
        "--sizes", "4",
        "--p", "1",
        "--de-seeds", "1", "2",
        "--population", "4",
        "--generations", "3",
        "--train-shots", "200",
        "--eval-shots", "3000",
        "--routing-seeds", "2",
    ]  # fmt: skip


def test_sweep_writes_every_condition_and_resumes(tmp_path, monkeypatch, capsys):
    cal, cmap = write_files(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", _argv(cal, cmap, out))
    sweep.main()

    files = sorted((out / "sweep").glob("*.json.gz"))
    assert [f.name for f in files] == [
        "3reg-n4-i0_p1.json.gz",
        "3reg-n4-i1_p1.json.gz",
        "3reg-n4-i2_p1.json.gz",
        "lattice-n4-i0_p1.json.gz",
        "lattice-n4-i1_p1.json.gz",
        "lattice-n4-i2_p1.json.gz",
    ]
    with gzip.open(files[0], "rt") as f:
        unit = json.load(f)
    runs = {(r["run"], r["condition"]) for r in unit["rows"]}
    for seed in ("de1", "de2"):
        assert {(seed, c) for c in ("A", "B", "C", "D", "I")} <= runs
    # Sensitivity runs only for the best DE seed.
    assert len({r for r in runs if r[1] in ("E", "I_R")}) == 2
    assert {("fa", c) for c in ("A", "C", "I")} <= runs
    assert ("p0", "P0") in runs
    by = {(r["run"], r["condition"]): r for r in unit["rows"]}
    # No early stop by default: DE runs every generation asked for.
    assert by[("de1", "A")]["train_iterations"] == 3
    # Routing does not change the state: B matches A up to shot noise.
    assert by[("de1", "B")]["ar"] == pytest.approx(by[("de1", "A")]["ar"], abs=0.05)
    for row in unit["rows"]:
        assert 0 < row["ar"] <= 1 + 1e-9
    assert set(unit["counts"]) >= {"de1/C", "fa/I"}

    # The lattice family has no fixed angles.
    with gzip.open(files[3], "rt") as f:
        lattice = json.load(f)
    assert not any(r["run"] == "fa" for r in lattice["rows"])

    # A second run skips every finished unit.
    capsys.readouterr()
    mtimes = [f.stat().st_mtime_ns for f in files]
    sweep.main()
    assert capsys.readouterr().out.count("skip") == 6
    assert [f.stat().st_mtime_ns for f in files] == mtimes
