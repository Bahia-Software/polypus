"""
Smoke test for ``benchmarks/bench_optimizers.py``, the optimizer benchmark
behind the training guide's results table: its quick mode runs every
optimizer end to end, within budget, and its exact scoring agrees with the
fitness the optimizers see.

The runs go through a subprocess because the script replaces
``polypus_python.run_qcs`` with its evaluation meter.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_exact_shots_score_what_the_optimizer_saw(tmp_path):
    """With 2^20 shots the reported fitness is the exact expected cut, so the
    winner's-curse bias vanishes: the native Qubo, the meter's bitstrings and
    the statevector scoring all number the nodes the same way."""
    report, _ = _run(tmp_path, "--scenario", "1:exact:120", "--only", "cmaes")
    for run in report["scenarios"][0]["runs"]:
        assert abs(run["bias"]) < 1e-3, run


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
