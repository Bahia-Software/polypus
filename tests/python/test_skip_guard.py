"""``POLYPUS_ALLOWED_SKIPS`` (``skip_guard.py``): with it set, as in CI, a run
that skips a test not on the list fails, so a test that backs a contract cannot
stop running unnoticed (#407)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from skip_guard import unexpected_skips, unused_entries

_HERE = Path(__file__).parent

_SUITE = """
import pytest


def test_runs():
    pass


@pytest.mark.skip(reason="needs a QPU")
def test_needs_a_qpu():
    pass


class TestOptIn:
    @pytest.mark.skip(reason="opt-in")
    def test_opt_in(self):
        pass


@pytest.mark.parametrize("n", [1, 2])
def test_param(n):
    pytest.skip("parametrized")
"""


class TestUnexpectedSkips:
    def test_files_and_classes_on_the_list_are_allowed(self):
        nodeids = [
            "tests/python/test_cunqa_real.py::test_real",
            "tests/python/test_determinism_baseline.py::TestSameBuildStrict::test_x",
            "tests/python/test_qasm3_interop.py",
        ]
        allowed = [
            "test_cunqa_real.py",
            "test_determinism_baseline.py::TestSameBuildStrict",
            "test_qasm3_interop.py",
        ]
        assert unexpected_skips(nodeids, allowed) == []

    def test_anything_else_is_unexpected(self):
        nodeids = [
            "tests/python/test_determinism_baseline.py::TestPortableExact::test_x",
            "tests/python/test_native_circuit.py::test_qir",
            "tests/python/test_cunqa_real_extra.py::test_x",
        ]
        allowed = [
            "test_cunqa_real.py",
            "test_determinism_baseline.py::TestSameBuildStrict",
        ]
        assert unexpected_skips(nodeids, allowed) == nodeids

    def test_a_test_covers_its_parametrized_cases(self):
        nodeids = [
            "tests/python/test_e.py::test_p[1]",
            "tests/python/test_e.py::test_pq[1]",
        ]
        assert unexpected_skips(nodeids, ["test_e.py::test_p"]) == nodeids[1:]

    def test_entries_that_cover_no_skip_are_reported(self):
        nodeids = ["tests/python/test_cunqa_real.py::test_real"]
        allowed = ["test_cunqa_real.py", "test_gone.py"]
        assert unused_entries(nodeids, allowed) == ["test_gone.py"]


def _run(tmp_path, allowed):
    (tmp_path / "test_suite.py").write_text(textwrap.dedent(_SUITE))
    env = {k: v for k, v in os.environ.items() if k != "POLYPUS_ALLOWED_SKIPS"}
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    if allowed is not None:
        env["POLYPUS_ALLOWED_SKIPS"] = allowed
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "skip_guard", "-p", "no:cacheprovider"]
        + ["--rootdir", str(tmp_path), str(tmp_path / "test_suite.py")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(
    "allowed, code",
    [
        # Unset: skips are reported, as without the guard.
        (None, 0),
        (
            "test_suite.py::test_needs_a_qpu test_suite.py::TestOptIn"
            " test_suite.py::test_param",
            0,
        ),
        ("test_suite.py", 0),
        ("test_suite.py::TestOptIn test_suite.py::test_param", 1),
        ("", 1),
    ],
)
def test_a_skip_not_on_the_list_fails_the_run(tmp_path, allowed, code):
    result = _run(tmp_path, allowed)
    assert result.returncode == code, result.stdout + result.stderr
    if code:
        assert "skipped but not in POLYPUS_ALLOWED_SKIPS" in result.stdout
        assert "test_suite.py::test_needs_a_qpu" in result.stdout
