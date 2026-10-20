"""
Smoke test for ``benchmarks/qasm_coverage.py``, the OpenQASM coverage report
over benchmark suites: it runs end to end on the real benchmark fixtures, and
it names the construct behind each import failure.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "qasm_coverage.py"
FIXTURES = Path(__file__).parent / "data" / "qasm_benchmarks"


def _load_script():
    spec = importlib.util.spec_from_file_location("qasm_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_on_the_benchmark_fixtures(tmp_path):
    out_json = tmp_path / "coverage.json"
    out_md = tmp_path / "coverage.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(FIXTURES),
            "--jobs",
            "5",
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    records = json.loads(out_json.read_text())
    assert len(records) == len(list(FIXTURES.rglob("*.qasm")))
    for record in records:
        assert record["qiskit"] == "ok", record
        assert record["polypus"] == "ok", record
        assert record["emit"] == "ok", record
        assert record["fidelity"] == "identical", record
    table = out_md.read_text()
    assert "| qasm_benchmarks |" in table
    assert "## Fidelity failures\n\n_none_" in table


def test_failures_are_classified_by_construct():
    script = _load_script()
    classify = script.classify_polypus_error
    assert (
        classify("QASM parse error at line 3: unsupported gate 'foo': x")
        == "gate `foo`"
    )
    assert (
        classify("QASM parse error at line 3: 'reset' is not supported: …") == "`reset`"
    )
    assert (
        classify("QASM parse error at line 4: 'if' statements are not supported: …")
        == "`if` (classical control)"
    )
    assert (
        classify(
            "QASM parse error at line 5: gate acts on qubit 0 after it was measured; …"
        )
        == "gate after measurement (C-4)"
    )
    assert classify("QASM parse error at line 9: something new").startswith("other: ")


def test_aer_is_compared_only_where_it_runs_the_original(monkeypatch):
    """A program Aer cannot run in its original form is not compared (not a
    Polypus failure); a failure on the re-emitted program alone is an error."""
    script = _load_script()
    grover = str(FIXTURES / "mqtbench" / "grover_n3.qasm")

    def aer_rejects_every_program(qasm, seed, shots):
        raise RuntimeError("Aer cannot run this program")

    monkeypatch.setattr(script, "_aer_counts", aer_rejects_every_program)
    assert script.process_file(grover, True, 16, 1, 32)["aer"] == "unsupported"

    runs = []

    def aer_rejects_the_reemission(qasm, seed, shots):
        runs.append(qasm)
        if len(runs) == 2:
            raise RuntimeError("unknown instruction")
        return {"000": shots}

    monkeypatch.setattr(script, "_aer_counts", aer_rejects_the_reemission)
    assert script.process_file(grover, True, 16, 1, 32)["aer"] == "error"
    assert runs[0] == Path(grover).read_text(encoding="utf-8")
