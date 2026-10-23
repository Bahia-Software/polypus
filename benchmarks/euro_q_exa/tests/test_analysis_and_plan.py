"""Tests for the sweep analysis (`analyze.py`) and the QPU plan (`qpu_plan.py`)."""

import gzip
import json

import analyze
import pytest
import qpu_plan


def _row(instance, run, condition, ar, p_opt=None, family="3reg", n=16, p=1):
    return {
        "instance": instance,
        "family": family,
        "n": n,
        "p": p,
        "run": run,
        "condition": condition,
        "ar": ar,
        "p_opt": p_opt,
    }


def _write_unit(directory, name, rows):
    with gzip.open(directory / f"{name}.json.gz", "wt") as f:
        json.dump({"rows": rows, "counts": {}, "routing": {}}, f)


@pytest.fixture
def sweep(tmp_path):
    # Two instances, two DE seeds each, fixed angles and P0.
    for inst, base in (("a", 0.0), ("b", 0.1)):
        _write_unit(
            tmp_path,
            f"{inst}_p1",
            [
                _row(inst, "de1", "A", 0.80 + base, 0.02),
                _row(inst, "de2", "A", 0.70 + base, 0.04),
                _row(inst, "de1", "C", 0.70 + base, 0.01),
                _row(inst, "de2", "C", 0.60 + base, 0.03),
                _row(inst, "fa", "A", 0.90 + base),
                _row(inst, "fa", "C", 0.80 + base),
                _row(inst, "p0", "P0", 0.50 + base),
            ],
        )
    return tmp_path


def _by_condition(summary):
    return {r["condition"]: r for r in summary}


def test_summary_averages_runs_and_instances(sweep):
    rows = analyze.load_rows(sweep)
    assert len(rows) == 14
    s = _by_condition(analyze.summarize(rows))
    assert s["A"]["ar_mean"] == pytest.approx(0.80)  # (0.8 + 0.7 + 0.9 + 0.8) / 4
    assert s["A"]["count"] == 4
    assert s["C"]["ar_mean"] == pytest.approx(0.70)
    assert s["C"]["ar_sd"] == pytest.approx(0.0816497, rel=1e-5)
    assert s["C"]["p_opt_mean"] == pytest.approx(0.02)
    assert s["FA_A"]["ar_mean"] == pytest.approx(0.95)
    assert s["FA_C"]["count"] == 2
    assert s["P0"]["ar_mean"] == pytest.approx(0.55)


def test_kept_share_uses_the_group_means(sweep):
    s = _by_condition(analyze.summarize(analyze.load_rows(sweep)))
    assert s["C"]["kept"] == pytest.approx((0.70 - 0.55) / (0.80 - 0.55))
    assert s["FA_C"]["kept"] == pytest.approx((0.85 - 0.55) / (0.95 - 0.55))
    assert s["A"]["kept"] is None


def test_summary_table_has_one_row_per_group(sweep):
    table = analyze.summary_table(analyze.summarize(analyze.load_rows(sweep)))
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[-1].startswith("| 3reg | 16 | 1 | 0.550 | 0.800 | 0.700 ± 0.082 | - |")
    assert lines[-1].endswith("| 60% |")


def test_base_plan_counts_circuits_and_shots():
    blocks = {b.name: b for b in qpu_plan.base_plan()}
    # 2 families x 2 sizes x 2 depths x 3 instances = 24 units, 12 of them 3-regular.
    assert blocks["Trained parameters"].circuits == 72
    assert blocks["Fixed angles (3-regular)"].circuits == 12
    assert blocks["p = 0 baseline"].circuits == 24
    assert blocks["Readout calibration"].circuits == 48
    assert blocks["Repeat on a second day (drift)"].circuits == 108
    total = sum(b.shots for b in blocks.values())
    assert total == 264 * 4000


def test_base_plan_without_control_or_repeat():
    blocks = qpu_plan.base_plan(include_control=False, repeat=False)
    assert sum(b.circuits for b in blocks) == 36 + 12 + 12 + 24


def test_refinement_runs_one_job_per_generation():
    (block,) = qpu_plan.refinement_plan(population=10, generations=20, shots=1000)
    assert block.jobs == 2 * 21
    assert block.circuits == 420
    assert block.shots == 420_000


def test_hours_formula():
    blocks = [qpu_plan.Block("x", circuits=100, shots=360_000, jobs=6)]
    exec_h, reservation_h = qpu_plan.hours(
        blocks, t_shot_ms=10, t_overhead_s=36, t_latency_s=600
    )
    assert exec_h == pytest.approx(1.0 + 1.0)  # 3600 s of shots + 3600 s of overhead
    assert reservation_h == pytest.approx(exec_h + 1.0)


def test_jobs_split_at_the_per_job_limit():
    assert qpu_plan._jobs(0, 200) == 0
    assert qpu_plan._jobs(200, 200) == 1
    assert qpu_plan._jobs(201, 200) == 2


def test_normalized_ratio_uses_each_instance_random_baseline(sweep):
    # C: instance a (P0 0.5) has 0.70 and 0.60 -> 0.4 and 0.2; instance b
    # (P0 0.6) has 0.80 and 0.70 -> 0.5 and 0.25.
    s = _by_condition(analyze.summarize(analyze.load_rows(sweep)))
    assert s["C"]["norm_mean"] == pytest.approx((0.4 + 0.2 + 0.5 + 0.25) / 4)
    assert s["P0"]["norm_mean"] == pytest.approx(0.0)
    table = analyze.normalized_table(analyze.summarize(analyze.load_rows(sweep)))
    assert "| 3reg | 16 | 1 |" in table.splitlines()[-1]
