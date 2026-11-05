"""Tests for .github/scripts/coverage_summary.py."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("coverage_summary.py")
_spec = importlib.util.spec_from_file_location("coverage_summary", SCRIPT)
coverage_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coverage_summary)

# A minimal `cargo llvm-cov report --lcov`: two files of one crate, one of
# another, one outside `crates/`, absolute paths as on a runner. FN/FNDA are
# ignored; the first record has LF/LH, the others only DA records (the
# fallback).
LCOV = """\
SF:/home/runner/work/polypus/polypus/crates/polypus-sim/src/lib.rs
FN:3,polypus_sim::f
FNDA:1,polypus_sim::f
DA:1,4
DA:2,0
DA:3,1
LF:3
LH:2
end_of_record
SF:/home/runner/work/polypus/polypus/crates/polypus-sim/src/kernels.rs
DA:10,3000000000
DA:11,0
end_of_record
SF:/home/runner/work/polypus/polypus/crates/polypus/src/bindings/mod.rs
DA:5,0
DA:6,2
end_of_record
SF:/home/runner/work/polypus/polypus/examples/rust-backend-template/src/lib.rs
DA:1,1
end_of_record
"""


def test_crate_of_takes_the_directory_under_crates():
    assert (
        coverage_summary.crate_of("/w/p/crates/polypus-sim/src/lib.rs") == "polypus-sim"
    )
    assert coverage_summary.crate_of("crates/polypus/src/bindings/mod.rs") == "polypus"
    assert (
        coverage_summary.crate_of(r"C:\w\crates\polypus-circuit\src\gate.rs")
        == "polypus-circuit"
    )
    # A checkout under a directory called `crates`: the innermost one wins.
    assert (
        coverage_summary.crate_of("/crates/polypus/crates/polypus-sim/src/a.rs")
        == "polypus-sim"
    )
    assert coverage_summary.crate_of("/w/p/examples/x/src/lib.rs") == "other"


def test_lines_are_lf_and_lh_or_else_the_da_records():
    totals = coverage_summary.per_crate(coverage_summary.parse_lcov(LCOV))
    assert totals == {
        "polypus-sim": (5, 3),
        "polypus": (2, 1),
        "other": (1, 1),
    }


# The shape of a real record of `cargo llvm-cov report --lcov` (cargo-llvm-cov
# 0.9.1, `polypus-observable/src/qubo.rs`: 264 DA records, 254 hit; LF 269,
# LH 259), cut down: LF/LH declare more lines than the DA records list, and
# `cargo llvm-cov report` prints LF/LH.
REAL_SHAPE = """\
SF:/home/runner/work/polypus/polypus/crates/polypus-observable/src/qubo.rs
FN:55,_RNvMNtCs1_18polypus_observable4quboNtB2_13QuboObservable3new
FNDA:12,_RNvMNtCs1_18polypus_observable4quboNtB2_13QuboObservable3new
FNF:1
FNH:1
DA:55,12
DA:56,12
DA:57,0
BRF:0
BRH:0
LF:5
LH:3
end_of_record
"""


def test_lf_and_lh_win_over_fewer_da_records():
    files = coverage_summary.parse_lcov(REAL_SHAPE)
    path = "/home/runner/work/polypus/polypus/crates/polypus-observable/src/qubo.rs"
    assert files == {path: (5, 3)}
    assert coverage_summary.per_crate(files) == {"polypus-observable": (5, 3)}


def test_a_line_recorded_twice_counts_once_with_its_largest_hits():
    lcov = "SF:crates/a/src/lib.rs\nDA:1,0\nend_of_record\nSF:crates/a/src/lib.rs\nDA:1,7\nDA:2,0\nend_of_record\n"
    files = coverage_summary.parse_lcov(lcov)
    assert files == {"crates/a/src/lib.rs": (2, 1)}
    assert coverage_summary.per_crate(files) == {"a": (2, 1)}


def test_a_file_recorded_twice_keeps_its_largest_lf_and_lh():
    lcov = (
        "SF:crates/a/src/lib.rs\nDA:1,1\nLF:4\nLH:1\nend_of_record\n"
        "SF:crates/a/src/lib.rs\nDA:1,1\nLF:4\nLH:3\nend_of_record\n"
    )
    assert coverage_summary.parse_lcov(lcov) == {"crates/a/src/lib.rs": (4, 3)}


def test_a_malformed_record_is_an_error():
    for body in ["DA:1", "DA:x,1", "LF:many"]:
        with pytest.raises(ValueError, match="malformed"):
            coverage_summary.parse_lcov(
                f"SF:crates/a/src/lib.rs\n{body}\nend_of_record\n"
            )


def test_markdown_lists_crates_alphabetically_other_last_then_the_total():
    totals = coverage_summary.per_crate(coverage_summary.parse_lcov(LCOV))
    text = coverage_summary.markdown(totals, "Rust coverage")
    assert text.splitlines() == [
        "### Rust coverage",
        "",
        "| Crate | Lines | Covered | Line coverage |",
        "|---|---:|---:|---:|",
        "| `polypus` | 2 | 1 | 50.0 % |",
        "| `polypus-sim` | 5 | 3 | 60.0 % |",
        "| `other` | 1 | 1 | 100.0 % |",
        "| **total** | 8 | 5 | 62.5 % |",
    ]


def test_markdown_of_an_empty_report_says_so():
    text = coverage_summary.markdown({}, "Rust coverage")
    assert "no line records" in text
    assert "|" not in text


def test_main_prints_the_table(tmp_path, capsys):
    report = tmp_path / "lcov.info"
    report.write_text(LCOV, encoding="utf-8")
    assert coverage_summary.main([str(report), "--title", "Python suite"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### Python suite\n")
    assert "| **total** | 8 | 5 | 62.5 % |" in out


def test_main_reports_a_missing_file(tmp_path, capsys):
    assert coverage_summary.main([str(tmp_path / "absent.info")]) == 2
    assert "coverage_summary:" in capsys.readouterr().err
