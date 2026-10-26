"""
Smoke test for ``benchmarks/bench_optimizers.py``, the optimizer benchmark
behind the training guide's results table: its quick mode runs every
optimizer end to end, within budget, and its exact scoring agrees with the
fitness the optimizers see.

The runs go through a subprocess because the script replaces
``polypus_python.run_qcs`` with its evaluation meter; the test that runs it
in-process restores the seam with ``monkeypatch``.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polypus
import polypus_python
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "bench_optimizers.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bench_optimizers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, *args):
    out = tmp_path / "runs.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--json", str(out), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(out.read_text()), completed.stdout


def test_quick_mode_runs_every_optimizer_within_budget(tmp_path):
    script = _load_script()
    report, stdout = _run(tmp_path)
    (scenario,) = report["scenarios"]
    assert (scenario["layers"], scenario["shots"], scenario["budget"]) == (1, 64, 120)
    runs = scenario["runs"]
    assert {r["optimizer"] for r in runs} == set(script.OPTIMIZERS)
    assert len(runs) == 2 * len(script.OPTIMIZERS)
    for run in runs:
        assert 0 < run["evaluations"] <= scenario["budget"], run
        assert 0 < run["batches"] <= run["evaluations"], run
        assert 0.0 < run["ratio"] <= 1.0, run
    # The table prints a row per optimizer, the baseline without a delta.
    for name in script.OPTIMIZERS:
        assert f"| {name} |" in stdout
    assert "| rand1bin | " in stdout and "| — | — |" in stdout
    # The chained optimizers spend one budget over their stages.
    for run in runs:
        stages = run.get("stages")
        if run["optimizer"] == "lshade-adam":
            assert [s["optimizer"] for s in stages] == ["lshade", "adam"], run
            assert stages[0]["evaluations"] <= scenario["budget"] // 2, run
            assert run["ratio"] == stages[1]["ratio"], run
        elif run["optimizer"] == "adam-x3":
            assert [s["optimizer"] for s in stages] == ["adam"] * 3, run
            assert len({s["seed"] for s in stages}) == 3, run
            best = max(stages, key=lambda s: s["best_fitness"])
            assert run["ratio"] == best["ratio"], run
        else:
            assert stages is None, run
            continue
        assert sum(s["evaluations"] for s in stages) == run["evaluations"], run
        assert sum(s["batches"] for s in stages) == run["batches"], run


def test_exact_shots_score_what_the_optimizer_saw(tmp_path):
    """With 2^20 shots the reported fitness is the exact expected cut, so the
    winner's-curse bias vanishes: the native Qubo, the meter's bitstrings and
    the statevector scoring all number the nodes the same way."""
    report, _ = _run(tmp_path, "--scenario", "1:exact:120", "--only", "cmaes")
    for run in report["scenarios"][0]["runs"]:
        assert abs(run["bias"]) < 1e-3, run


def test_quick_hea_scenario_runs_every_optimizer_within_budget(tmp_path):
    script = _load_script()
    report, stdout = _run(tmp_path, "--scenario", "hea:2:64:200")
    (scenario,) = report["scenarios"]
    assert scenario["problem"] == "hea"
    assert scenario["parameters"] == 6 * 3
    runs = scenario["runs"]
    assert {r["optimizer"] for r in runs} == set(script.OPTIMIZERS)
    for run in runs:
        assert 0 < run["evaluations"] <= scenario["budget"], run
        assert 0.0 < run["ratio"] <= 1.0, run
    assert "## HEA reps = 2 (18 parameters), n = 6," in stdout


def test_lshade_then_adam_refines_from_the_lshade_point(monkeypatch):
    """The hybrid's Adam stage starts from L-SHADE's best point, gets the
    evaluations L-SHADE left, and the run is scored on Adam's result."""
    script = _load_script()
    # run_one installs its meter as the local seam; restore it afterwards.
    monkeypatch.setattr(polypus_python, "run_qcs", polypus_python.run_qcs)
    train, calls = script.train, []

    def spy(instance, method, shots, seed, initial_point=None):
        result = train(instance, method, shots, seed, initial_point)
        calls.append((method, initial_point, result))
        return result

    monkeypatch.setattr(script, "train", spy)
    instance = script.Instance(nodes=6, layers=1, seed=3)
    run = script.run_one(instance, "lshade-adam", 64, 120, 8, 3, seed=3)
    (lshade, start, explore), (adam, initial_point, refine) = calls
    assert isinstance(lshade, polypus.DE) and lshade.strategy == "lshade"
    assert lshade.population_size * (lshade.generations + 1) <= 120 // 2
    assert start is None
    assert isinstance(adam, polypus.Adam) and adam.rule == "adam"
    assert initial_point == explore.best_params
    lshade_spent = run["stages"][0]["evaluations"]
    assert adam.max_iters == (120 - lshade_spent) // (2 * instance.dimensions + 1)
    assert run["evaluations"] <= 120
    exact = instance.expected_cut(refine.best_params) / instance.max_cut
    assert run["ratio"] == pytest.approx(exact)


def test_adam_x3_makes_three_runs_that_share_the_budget(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(polypus_python, "run_qcs", polypus_python.run_qcs)
    train, calls = script.train, []

    def spy(instance, method, shots, seed, initial_point=None):
        result = train(instance, method, shots, seed, initial_point)
        calls.append((method, seed, result))
        return result

    monkeypatch.setattr(script, "train", spy)
    instance = script.Instance(nodes=6, layers=1, seed=5)
    run = script.run_one(instance, "adam-x3", 64, 120, 8, 3, seed=5)
    assert len(calls) == 3
    assert all(isinstance(m, polypus.Adam) and m.rule == "adam" for m, _, _ in calls)
    assert len({seed for _, seed, _ in calls}) == 3
    batch = 2 * instance.dimensions + 1
    assert run["evaluations"] == sum(m.max_iters * batch for m, _, _ in calls)
    assert run["evaluations"] <= 120
    best = max((r for _, _, r in calls), key=lambda r: r.best_fitness)
    exact = instance.expected_cut(best.best_params) / instance.max_cut
    assert run["ratio"] == pytest.approx(exact)


def test_hea_has_one_ry_per_parameter_in_three_layers():
    """reps = 2: three RY layers of n parameters, each angle in exactly one
    RY, a CZ chain between consecutive layers, and the same qubit order as
    the cut table."""
    script = _load_script()
    nodes = 6
    instance = script.Instance(nodes=nodes, layers=2, seed=4, problem="hea")
    assert instance.dimensions == nodes * 3
    assert instance.circuit.num_params == nodes * 3
    # Angle k is the RY of qubit k mod n in layer k div n.
    qasm = instance.circuit.to_qasm2([float(k) for k in range(nodes * 3)])
    for k in range(nodes * 3):
        assert qasm.count(f"ry({float(k)}) q[{k % nodes}];") == 1
    assert qasm.count("ry(") == nodes * 3
    assert qasm.count("cz ") == 2 * (nodes - 1)
    # |0>^n cuts nothing; RY(π) on the first layer prepares a basis state.
    assert instance.expected_cut([0.0] * nodes * 3) == pytest.approx(0.0)
    bits = [1, 0, 1, 1, 0, 0]
    angles = [np.pi * b for b in bits] + [0.0] * nodes * 2
    index = sum(b << q for q, b in enumerate(bits))
    assert instance.expected_cut(angles) == pytest.approx(instance.cuts[index])
    assert instance.cuts[index] > 0
    # The same graph as QAOA's instance of the same seed.
    assert instance.edges == script.Instance(nodes=nodes, layers=1, seed=4).edges


def test_scenarios_name_their_problem():
    script = _load_script()
    assert script.parse_scenario("3:256:1500") == ("qaoa", 3, 256, 1500)
    assert script.parse_scenario("hea:2:256:5000") == ("hea", 2, 256, 5000)
    assert script.parse_scenario("qaoa:3:exact:1500") == (
        "qaoa",
        3,
        script.EXACT_SHOTS,
        1500,
    )
    full = script.parse_args(["--full"]).scenario
    assert ("hea", 2, 256, 5000) in full and len(full) == 5


def test_graphs_are_3_regular_and_the_cut_table_matches_them():
    script = _load_script()
    instance = script.Instance(nodes=8, layers=1, seed=4)
    degree = np.bincount(np.array(instance.edges).ravel(), minlength=8)
    assert list(degree) == [3] * 8
    assert len(set(instance.edges)) == 12
    # The uniform superposition (all angles 0) cuts half the edges on average.
    assert instance.expected_cut([0.0, 0.0]) == pytest.approx(6.0)
    assert instance.max_cut <= 12


def test_the_budget_fixes_every_generation_count():
    script = _load_script()
    de = script.make_method("lshade", 6, 1500, 30, 3)
    assert (de.generations, de.population_size, de.strategy) == (49, 30, "lshade")
    cmaes = script.make_method("cmaes", 6, 1500, 30, 3)
    assert cmaes.generations == 1500 // 9 and cmaes.population_size is None
    ipop = script.make_method("cmaes-ipop", 6, 1500, 30, 3)
    assert ipop.restarts == 3 and ipop.tolerance == pytest.approx(0.01)
    # Adam: one batch of 2d + 1 per iteration, the class defaults.
    adam = script.make_method("adam", 6, 1500, 30, 3)
    assert adam.max_iters == 1500 // 13 and adam.rule == "adam"
    assert (adam.learning_rate, adam.finite_difference_step) == (0.1, 0.2)
    assert adam.tolerance == 0.0
    assert script.make_method("adam", 30, 5000, 30, 3).max_iters == 5000 // 61
    # Momentum's rate is tuned per problem.
    qaoa = script.make_method("momentum", 6, 1500, 30, 3, "qaoa")
    hea = script.make_method("momentum", 30, 5000, 30, 3, "hea")
    assert qaoa.rule == hea.rule == "momentum"
    assert (qaoa.learning_rate, qaoa.finite_difference_step) == (0.03, 0.4)
    assert (hea.learning_rate, hea.finite_difference_step) == (0.1, 0.4)
    with pytest.raises(ValueError, match="does not fit one Adam batch"):
        script.make_method("adam", 30, 60, 30, 3)
    # SPSA: the most iterations whose calibration, 2K circuits each and the
    # measurements of θ fit, with the settings tuned per problem.
    spsa = script.make_method("spsa", 6, 1500, 30, 3, "qaoa")
    assert (spsa.max_iters, spsa.resamplings) == (706, 1)
    assert (spsa.perturbation, spsa.target_step) == (0.2, 1.2)
    assert (spsa.alpha, spsa.gamma, spsa.eval_interval) == (0.602, 0.101, 20)
    k8 = script.make_method("spsa-k8", 6, 1500, 30, 3, "qaoa")
    assert (k8.max_iters, k8.resamplings) == (90, 8)
    # The same steps as spsa, so the two differ only in K.
    assert (k8.perturbation, k8.target_step) == (0.2, 1.2)
    for name in ("spsa", "spsa-k8"):
        hea = script.make_method(name, 30, 5000, 30, 3, "hea")
        assert (hea.perturbation, hea.target_step) == (0.1, 0.63)
    # 50 + 2·706 + 36 measurements and the final point; one iteration more
    # would not fit.
    assert script.spsa_evaluations(706, 1) == 1499
    assert script.spsa_evaluations(707, 1) == 1501
    assert script.spsa_evaluations(90, 8) == 1496
    assert script.spsa_evaluations(91, 8) > 1500
    with pytest.raises(ValueError, match="does not fit SPSA's calibration"):
        script.make_method("spsa-k8", 6, 66, 30, 3)


@pytest.mark.parametrize("name", ["spsa", "spsa-k8"])
def test_spsa_spends_what_its_iterations_count(monkeypatch, name):
    """The meter sees exactly the circuits ``spsa_evaluations`` counts, in
    one batch for the calibration, one per iteration and one for the final
    point."""
    script = _load_script()
    monkeypatch.setattr(polypus_python, "run_qcs", polypus_python.run_qcs)
    train, calls = script.train, []

    def spy(instance, method, shots, seed, initial_point=None):
        calls.append(method)
        return train(instance, method, shots, seed, initial_point)

    monkeypatch.setattr(script, "train", spy)
    instance = script.Instance(nodes=6, layers=1, seed=2)
    run = script.run_one(instance, name, 64, 120, 8, 3, seed=2)
    (method,) = calls
    assert isinstance(method, polypus.SPSA)
    assert run["evaluations"] == script.spsa_evaluations(
        method.max_iters, method.resamplings
    )
    assert run["evaluations"] <= 120
    assert run["batches"] == method.max_iters + 2
    assert run["iterations_run"] == method.max_iters
