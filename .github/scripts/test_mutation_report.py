"""Tests for .github/scripts/mutation_report.py."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("mutation_report.py")
_spec = importlib.util.spec_from_file_location("mutation_report", SCRIPT)
mutation_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mutation_report)

RUN_URL = "https://github.com/o/r/actions/runs/1"


def outcomes(caught, missed, timeout=0, unviable=0):
    """A minimal `outcomes.json` of cargo-mutants 27: the counts it keeps at the
    top level (the per-scenario `outcomes` list is not read). Its
    `total_mutants` is the number tested so far, not the number planned."""
    tested = caught + missed + timeout + unviable
    return {
        "outcomes": [],
        "total_mutants": tested,
        "caught": caught,
        "missed": missed,
        "timeout": timeout,
        "unviable": unviable,
        "success": 0,
        "cargo_mutants_version": "27.1.0",
    }


def write(directory, counts=None, exit_code=None, missed=(), planned=None):
    directory.mkdir(parents=True, exist_ok=True)
    if planned is not None:
        # `mutants.json` lists every mutant the run will test; only its length
        # is read.
        mutants = [{"name": f"src/lib.rs:{i}:1: replace x"} for i in range(planned)]
        (directory / "mutants.json").write_text(json.dumps(mutants), encoding="utf-8")
    if counts is not None:
        (directory / "outcomes.json").write_text(json.dumps(counts), encoding="utf-8")
    if exit_code is not None:
        (directory / "exit-code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    (directory / "missed.txt").write_text(
        "".join(f"{m}\n" for m in missed), encoding="utf-8"
    )
    return directory


def test_a_completed_run_is_scored_on_caught_and_missed(tmp_path):
    for code in (0, 2, 3):
        result = mutation_report.load(
            write(tmp_path / str(code), outcomes(60, 9, 1, 3), code)
        )
        assert result["ran"], code
        assert (
            result["caught"],
            result["missed"],
            result["timeout"],
            result["unviable"],
        ) == (
            60,
            9,
            1,
            3,
        )
    assert mutation_report.score(60, 9) == "87.0 %"
    assert mutation_report.score(0, 0) == "–"


def test_a_run_without_an_exit_code_did_not_finish(tmp_path):
    # outcomes.json is written as the run goes, so a timed-out step leaves one.
    result = mutation_report.load(write(tmp_path, outcomes(10, 1), planned=500))
    assert not result["ran"]
    assert "did not finish" in result["reason"]


def test_a_tool_failure_is_not_a_run(tmp_path):
    result = mutation_report.load(write(tmp_path / "a", outcomes(0, 0), 4))
    assert not result["ran"]
    assert "exit code 4" in result["reason"]
    result = mutation_report.load(write(tmp_path / "b", outcomes(0, 0), 70))
    assert not result["ran"]
    assert result["reason"] == "cargo-mutants failed (exit code 70)"


def test_a_missing_or_broken_outcomes_file_is_no_results(tmp_path):
    assert "no results" in mutation_report.load(tmp_path / "absent")["reason"]
    broken = write(tmp_path / "broken", exit_code=2)
    (broken / "outcomes.json").write_text("{", encoding="utf-8")
    assert "no results" in mutation_report.load(broken)["reason"]


def test_summary_lists_the_counts_and_the_missed_mutants(tmp_path):
    missed = ["src/qubo.rs:144:34: replace + with - in QuboObservable::from_matrix"]
    directory = write(tmp_path, outcomes(60, 1, 0, 3), 2, missed, planned=64)
    text = mutation_report.summary("polypus-observable", directory)
    lines = text.splitlines()
    assert lines[0] == "### Mutation testing: `polypus-observable`"
    assert "| `polypus-observable` | ran | 60 | 1 | 0 | 3 | 98.4 % |" in lines
    assert "64 of 64 mutants tested." in lines
    assert not any("**total**" in line for line in lines), "one crate, no total"
    assert missed[0] in lines


def test_without_the_mutant_list_the_planned_count_is_unknown(tmp_path):
    directory = write(tmp_path, outcomes(5, 1), 2)
    assert mutation_report.planned(directory) is None
    assert "6 mutants tested." in mutation_report.summary("c", directory).splitlines()
    (directory / "mutants.json").write_text("{", encoding="utf-8")
    assert mutation_report.planned(directory) is None


def test_summary_caps_the_missed_list(tmp_path):
    missed = [
        f"src/a.rs:{i}:1: replace x" for i in range(mutation_report.MAX_LISTED + 5)
    ]
    text = mutation_report.summary(
        "c", write(tmp_path, outcomes(1, len(missed)), 2, missed)
    )
    assert missed[mutation_report.MAX_LISTED - 1] in text
    assert missed[mutation_report.MAX_LISTED] not in text
    assert "…and 5 more" in text


def test_summary_of_a_partial_run_shows_how_far_it_got(tmp_path):
    text = mutation_report.summary(
        "polypus-sim", write(tmp_path, outcomes(10, 1), planned=500)
    )
    assert "**not run**: cargo-mutants did not finish" in text
    assert "11 of 500 mutants tested." in text


def artifact(root, crate, shard=0, shards=1):
    return root / mutation_report.artifact_name(crate, shard, shards)


def test_artifacts_are_named_after_the_crate_and_the_shard():
    assert (
        mutation_report.artifact_name("polypus-sim", 1, 3) == "mutants-polypus-sim-1of3"
    )
    assert mutation_report.artifact_name("c", 0, 1) == "mutants-c-0of1"


def test_comment_keeps_every_crate_and_totals_those_that_ran(tmp_path):
    write(artifact(tmp_path, "polypus-circuit"), outcomes(90, 10), 2)
    write(artifact(tmp_path, "polypus-sim"), outcomes(10, 1), planned=500)  # timed out
    write(artifact(tmp_path, "polypus-observable"), outcomes(69, 0, 0, 3), 0)
    # polypus-physics: no artifact at all.
    crates = [
        ("polypus-circuit", 1),
        ("polypus-sim", 1),
        ("polypus-physics", 1),
        ("polypus-observable", 1),
    ]
    text = mutation_report.comment(crates, tmp_path, RUN_URL)
    lines = text.splitlines()
    assert lines[0] == f"Weekly `cargo mutants` run: {RUN_URL}"
    rows = [line for line in lines if line.startswith("| `") or line.startswith("| **")]
    assert rows == [
        "| `polypus-circuit` | ran | 90 | 10 | 0 | 0 | 90.0 % |",
        "| `polypus-sim` | **not run**: cargo-mutants did not finish (the step timed out or "
        "was cancelled) | – | – | – | – | – |",
        "| `polypus-physics` | **not run**: no results (cargo-mutants did not start, or the "
        "artifact is missing) | – | – | – | – | – |",
        "| `polypus-observable` | ran | 69 | 0 | 0 | 3 | 100.0 % |",
        "| **total** | 2 of 4 crates | 159 | 10 | 0 | 3 | 94.1 % |",
    ]


def test_a_crate_in_shards_adds_them_up_when_all_ran(tmp_path):
    write(artifact(tmp_path, "polypus-sim", 0, 3), outcomes(10, 2, 1, 0), 2, planned=13)
    write(artifact(tmp_path, "polypus-sim", 1, 3), outcomes(11, 1, 0, 1), 3, planned=13)
    write(artifact(tmp_path, "polypus-sim", 2, 3), outcomes(12, 0, 0, 0), 0, planned=12)
    merged = mutation_report.merge(
        [
            mutation_report.load(artifact(tmp_path, "polypus-sim", k, 3))
            for k in range(3)
        ]
    )
    assert merged["ran"] and merged["planned"] == 38
    text = mutation_report.comment([("polypus-sim", 3)], tmp_path, RUN_URL)
    assert "| `polypus-sim` | ran | 33 | 3 | 1 | 1 | 91.7 % |" in text.splitlines()


def test_a_crate_with_a_missing_shard_is_not_run_and_says_which(tmp_path):
    write(artifact(tmp_path, "polypus-sim", 0, 3), outcomes(10, 2), 2)
    write(
        artifact(tmp_path, "polypus-sim", 1, 3), outcomes(3, 0), planned=20
    )  # timed out
    # shard 2/3: no artifact.
    text = mutation_report.comment([("polypus-sim", 3)], tmp_path, RUN_URL)
    assert (
        "| `polypus-sim` | **not run**: shard 1/3: cargo-mutants did not finish (the step "
        "timed out or was cancelled); shard 2/3: no results (cargo-mutants did not start, "
        "or the artifact is missing) | – | – | – | – | – |"
    ) in text.splitlines()
    assert "**total**" not in text


def test_comment_with_nothing_run_has_no_total(tmp_path):
    text = mutation_report.comment([("polypus-sim", 1)], tmp_path, RUN_URL)
    assert "**not run**" in text
    assert "**total**" not in text


def test_crate_and_shard_arguments_are_checked():
    assert mutation_report.parse_crate("polypus-sim:3") == ("polypus-sim", 3)
    assert mutation_report.parse_crate("polypus-backend") == ("polypus-backend", 1)
    assert mutation_report.parse_shard("1/3") == (1, 3)
    for bad in ["polypus-sim:0", "polypus-sim:x", ":2"]:
        with pytest.raises(argparse.ArgumentTypeError):
            mutation_report.parse_crate(bad)
    for bad in ["3/3", "-1/2", "1", "a/b"]:
        with pytest.raises(argparse.ArgumentTypeError):
            mutation_report.parse_shard(bad)


def test_main_dispatches_both_commands(tmp_path, capsys):
    write(artifact(tmp_path, "c", 1, 2), outcomes(3, 1), 2)
    write(artifact(tmp_path, "c", 0, 2), outcomes(4, 0), 0)
    shard = str(artifact(tmp_path, "c", 1, 2))
    assert mutation_report.main(["summary", "c", shard, "--shard", "1/2"]) == 0
    assert "### Mutation testing: `c, shard 1/2`" in capsys.readouterr().out
    args = ["comment", "--artifacts", str(tmp_path), "--run-url", RUN_URL, "c:2"]
    assert mutation_report.main(args) == 0
    assert "| `c` | ran | 7 | 1 | 0 | 0 | 87.5 % |" in capsys.readouterr().out
