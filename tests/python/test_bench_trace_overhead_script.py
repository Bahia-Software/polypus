"""
Tests for ``benchmarks/bench_trace_overhead.py``, the benchmark of what the
call trace (contract C-12) costs: its statistics (the overhead of each pair of
runs, the bootstrap interval of their median and its verdict, the pairs a
spread calls for, the summary of the trace sizes), its exit status, its
refusal to time runs that may use several threads, and a quick run end to end
that writes its report.
"""

import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "bench_trace_overhead.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bench_trace_overhead", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_script()


def _child_env(**env):
    """The environment of a run of the script: this one without any
    ``POLYPUS_CALL_TRACE*`` variable, which would change what the script does
    (it refuses to run under ``POLYPUS_CALL_TRACE``, and
    ``POLYPUS_CALL_TRACE_MAX_MB`` caps its traces), and with each of ``env``
    set, or removed if it is ``None``."""
    child = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("POLYPUS_CALL_TRACE")
    }
    for key, value in env.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    return child


def test_overheads_are_relative_to_the_untraced_run():
    assert bench.overheads([(2.0, 2.02), (1.0, 0.99)]) == pytest.approx([0.01, -0.01])


def test_bootstrap_interval_is_reproducible_and_brackets_the_median():
    values = [0.001 * i for i in range(1, 11)]
    low, high = bench.bootstrap_interval(values, seed=1)
    assert (low, high) == bench.bootstrap_interval(values, seed=1)
    assert 0.001 <= low < bench.median(values) < high <= 0.010
    assert bench.bootstrap_interval([0.003] * 10) == (0.003, 0.003)


def test_bootstrap_interval_takes_the_documented_order_statistics(monkeypatch):
    """The ends of the interval are the floor(a·B)-th and the
    (ceil((1 − a)·B) − 1)-th of the B sorted resampled medians, counting from
    0, for a = (1 − level) / 2: the 250th and the 9749th by default; the 50th
    and the 949th for B = 1000 at level 0.9, where a·B falls just below 50 in
    binary floating point; and the 50th and the 964th for B = 1015 at level
    0.9, where neither a·B nor (1 − a)·B is whole. Resampled medians tie, and
    a tie would hide an index off by one, so each resample's median is
    replaced by the number of resamples drawn before it: no two are equal, and
    each end of the interval is its order statistic's index."""

    def numbered(**kwargs):
        drawn = itertools.count()
        monkeypatch.setattr(bench.statistics, "median", lambda resample: next(drawn))
        return bench.bootstrap_interval([1.0, 2.0], **kwargs)

    assert numbered() == (250, 9749)
    assert numbered(resamples=1000, level=0.9) == (50, 949)
    assert numbered(resamples=1015, level=0.9) == (50, 964)


def test_the_verdict_tells_a_pass_from_a_fail_and_from_too_wide_to_tell():
    """``pass`` needs the upper end of the interval below the threshold and
    ``fail`` its lower end at or above it; an interval that holds the
    threshold, an end on it included, is ``inconclusive``."""
    assert bench.verdict(-0.05, 0.0099) == "pass"
    assert bench.verdict(-0.05, 0.01) == "inconclusive"
    assert bench.verdict(-0.05, 0.05) == "inconclusive"
    assert bench.verdict(0.0099, 0.05) == "inconclusive"
    assert bench.verdict(0.01, 0.05) == "fail"
    assert bench.verdict(0.02, 0.04, threshold=0.05) == "pass"


def test_pairs_needed_follows_the_standard_error_of_the_median():
    """The least n with (z(0.975) + z(power))·√(π/2)·σ/√n <= 1%: at σ = 6.4%,
    (2.8016·1.2533·6.4)² = 504.995, so 505 pairs give power 0.8, and 248 put
    the interval's half-width at 1% (power 0.5: 247.16). The default --pairs,
    at σ = 8%, is 790 (789.06). No spread needs one pair, and so does a power
    so low that the two quantiles cancel out."""
    assert bench.pairs_needed(0.064) == 505
    assert bench.pairs_needed(0.064, power=0.5) == 248
    assert bench.pairs_needed(0.0) == 1
    assert bench.pairs_needed(0.08, power=0.01) == 1
    assert bench.parse_args([]).pairs == bench.pairs_needed(bench.SPREAD) == 790
    assert bench.parse_args(["--pairs", "12"]).pairs == 12
    for bad in (
        {"spread": -0.01},
        {"spread": 0.08, "margin": 0.0},
        {"spread": 0.08, "power": 1.0},
        {"spread": 0.08, "level": 1.0},
    ):
        with pytest.raises(ValueError):
            bench.pairs_needed(**bad)


def test_size_summary_checks_the_size_limits():
    summary = bench.size_summary({"spsa": 1_300_000, "adam": 100_000})
    assert summary == {"worst": 1_300_000, "mean": 700_000.0, "pass": False}
    sizes = {"spsa": 1_300_000, "a": 100_000, "b": 100_000}
    assert bench.size_summary(sizes)["pass"] is True
    assert bench.size_summary({"spsa": 2_000_001})["pass"] is False


def test_the_exit_status_follows_the_report(monkeypatch, tmp_path, capsys):
    """``main`` exits with 1 when the runs differ or a trace is incomplete,
    and, with ``--check``, when a criterion is not met: the overhead's verdict
    ``fail``, or a trace size over its limits. With ``--check`` and nothing
    failed, an ``inconclusive`` overhead exits with 3 and names the pairs the
    run's spread calls for. ``main`` exits with 2 when ``POLYPUS_CALL_TRACE``
    would trace the untraced runs too."""
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.delenv("POLYPUS_CALL_TRACE", raising=False)
    calls = []

    def run_reporting(**report):
        def run(**kwargs):
            calls.append(kwargs)
            return {"identical": True, "complete": True, **report}

        return run

    def overhead_reporting(verdict, **report):
        return run_reporting(verdict=verdict, sd=0.05, **report)

    monkeypatch.setattr(bench, "overhead_run", overhead_reporting("pass"))
    assert bench.main(["--check"]) == 0
    monkeypatch.setattr(bench, "overhead_run", overhead_reporting("fail"))
    assert bench.main([]) == 0
    assert bench.main(["--check"]) == 1
    monkeypatch.setattr(bench, "overhead_run", overhead_reporting("inconclusive"))
    assert bench.main([]) == 0
    capsys.readouterr()
    assert bench.main(["--check"]) == bench.INCONCLUSIVE == 3
    err = capsys.readouterr().err
    assert "INCONCLUSIVE" in err and f"{bench.pairs_needed(0.05)} pairs" in err
    monkeypatch.setattr(
        bench, "overhead_run", overhead_reporting("inconclusive", identical=False)
    )
    assert bench.main([]) == 1
    assert bench.main(["--check"]) == 1
    monkeypatch.setattr(
        bench, "overhead_run", overhead_reporting("pass", complete=False)
    )
    assert bench.main([]) == 1
    monkeypatch.setattr(bench, "size_run", run_reporting(**{"pass": False}))
    assert bench.main(["--size"]) == 0
    assert bench.main(["--size", "--check"]) == 1
    monkeypatch.setattr(bench, "size_run", run_reporting(**{"pass": True}))
    assert bench.main(["--size", "--check"]) == 0
    monkeypatch.setenv("POLYPUS_CALL_TRACE", str(tmp_path))
    assert bench.main([]) == 2
    assert len(calls) == 11


def test_the_help_names_the_defaults_and_the_exit_statuses(capsys):
    """argparse expands ``%`` in a help text, so a percentage in one must be
    written ``%%``, or ``--help`` raises instead of printing."""
    with pytest.raises(SystemExit) as done:
        bench.parse_args(["--help"])
    assert done.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "default: 790, fixed from a pair's standard deviation of 8%" in text
    assert "status 3 when the overhead's interval is too wide to tell" in text


def test_the_script_refuses_without_one_rayon_thread():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--pairs", "1"],
        capture_output=True,
        text=True,
        env=_child_env(RAYON_NUM_THREADS=None),
        timeout=120,
    )
    assert done.returncode == 2
    assert "RAYON_NUM_THREADS=1" in done.stderr


@pytest.mark.integration
def test_a_quick_run_writes_its_report(monkeypatch, tmp_path):
    # A call trace variable in the environment of the tests must not reach
    # the script, which refuses to run under POLYPUS_CALL_TRACE.
    monkeypatch.setenv("POLYPUS_CALL_TRACE", str(tmp_path / "traces"))
    out = tmp_path / "report.json"
    env = _child_env(
        RAYON_NUM_THREADS="1",
        POLYPUS_NO_AUTOCALIBRATE="1",
        XDG_CACHE_HOME=str(tmp_path),
    )
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pairs",
            "1",
            "--iterations",
            "20",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(out.read_text(encoding="utf-8"))["overhead"]
    assert report["identical"] is True and report["iterations"] == 20
    assert len(report["overheads"]) == 1 and len(report["ci95"]) == 2
    assert report["sd"] is None
    assert report["verdict"] == bench.verdict(*report["ci95"])
