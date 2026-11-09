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


# The plan as submitted to LRZ: (number, session, circuits, shots, jobs).
SUBMITTED_PLAN = [
    (1, 1, 21, 84_000, 4),
    (2, 1, 450, 1_800_000, 150),
    (3, 1, 60, 240_000, 15),
    (4, 2, 500, 500_000, 50),
    (5, 2, 140, 560_000, 50),
    (6, 2, 18, 288_000, 6),
]
# Minutes per block for t_shot = 0.3, 1, 3 ms: reservation, then execution only.
SUBMITTED_MINUTES = {
    1: (1.4, 2.4, 5.2, 0.4, 1.4, 4.2),
    2: (41.5, 62.5, 122.5, 9.0, 30.0, 90.0),
    3: (4.7, 7.5, 15.5, 1.2, 4.0, 12.0),
    4: (19.2, 25.0, 41.7, 2.5, 8.3, 25.0),
    5: (13.5, 20.0, 38.7, 2.8, 9.3, 28.0),
    6: (2.7, 6.1, 15.7, 1.4, 4.8, 14.4),
}


def test_plan_matches_the_submitted_blocks():
    blocks = qpu_plan.plan()
    assert [
        (b.number, b.session, b.circuits, b.shots, b.jobs) for b in blocks
    ] == SUBMITTED_PLAN
    assert sum(b.circuits for b in blocks) == 1189
    assert sum(b.shots for b in blocks) == 3_472_000
    assert sum(b.jobs for b in blocks) == 275


def test_minutes_per_block_match_the_submitted_table():
    for b in qpu_plan.plan():
        per = [qpu_plan.seconds([b], t) for t in qpu_plan.T_SHOT_MS]
        minutes = [r / 60 for r, _ in per] + [e / 60 for _, e in per]
        assert [round(m, 1) for m in minutes] == list(SUBMITTED_MINUTES[b.number]), (
            b.name
        )


@pytest.mark.parametrize(
    ("session", "reservation_h", "exec_h"),
    [
        (1, (0.79, 1.21, 2.39), (0.18, 0.59, 1.77)),
        (2, (0.59, 0.85, 1.60), (0.11, 0.37, 1.12)),
        (None, (1.38, 2.06, 3.99), (0.29, 0.96, 2.89)),
    ],
)
def test_hours_per_session_match_the_submitted_table(session, reservation_h, exec_h):
    blocks = qpu_plan.plan()
    if session is not None:
        blocks = qpu_plan.session_blocks(blocks, session)
    per = [qpu_plan.seconds(blocks, t) for t in qpu_plan.T_SHOT_MS]
    assert tuple(round(r / 3600, 2) for r, _ in per) == reservation_h
    assert tuple(round(e / 3600, 2) for _, e in per) == exec_h


def test_worst_case_leaves_an_hour_within_the_budget():
    reservation_s, _ = qpu_plan.seconds(qpu_plan.plan(), max(qpu_plan.T_SHOT_MS))
    assert qpu_plan.BUDGET_H - reservation_s / 3600 >= 1.0


def test_seconds_formula():
    blocks = [qpu_plan.Block(0, "x", 1, circuits=100, shots=360_000, jobs=6)]
    reservation_s, exec_s = qpu_plan.seconds(
        blocks, t_shot_ms=10, t_overhead_s=36, t_latency_s=600
    )
    assert exec_s == pytest.approx(3600)  # shots only
    assert reservation_s == pytest.approx(3600 + 3600 + 3600)  # + overhead + latency


def test_drift_rejects_arm_b_sizes_outside_the_drift_sizes(monkeypatch):
    monkeypatch.setattr(qpu_plan, "ARM_B_SIZES", (4, 6))
    with pytest.raises(ValueError, match="arm B"):
        qpu_plan.drift()


def test_main_writes_markdown_and_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["qpu_plan.py", "--out", str(tmp_path)])
    qpu_plan.main()
    text = (tmp_path / "qpu_plan.md").read_text()
    assert "| Total | | 1,189 | 3,472,000 | 275 |" in text
    assert "| Total | 1.38 h | 2.06 h | 3.99 h | 0.29 h | 0.96 h | 2.89 h |" in text
    report = json.loads((tmp_path / "qpu_plan.json").read_text())
    assert len(report["blocks"]) == 6
    assert report["hours"]["total"]["reservation"][2] == pytest.approx(3.99, abs=5e-3)
    assert set(report["hours"]) == {"total", "session_1", "session_2"}
    assert capsys.readouterr().out == text + "\n"


def test_normalized_ratio_uses_each_instance_random_baseline(sweep):
    # C: instance a (P0 0.5) has 0.70 and 0.60 -> 0.4 and 0.2; instance b
    # (P0 0.6) has 0.80 and 0.70 -> 0.5 and 0.25.
    s = _by_condition(analyze.summarize(analyze.load_rows(sweep)))
    assert s["C"]["norm_mean"] == pytest.approx((0.4 + 0.2 + 0.5 + 0.25) / 4)
    assert s["P0"]["norm_mean"] == pytest.approx(0.0)
    table = analyze.normalized_table(analyze.summarize(analyze.load_rows(sweep)))
    assert "| 3reg | 16 | 1 |" in table.splitlines()[-1]
