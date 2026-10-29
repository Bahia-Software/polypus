"""Fail a run that skips a test CI expects to run (#407).

``POLYPUS_ALLOWED_SKIPS`` lists, separated by whitespace, the tests a run may
skip: a file (``test_cunqa_real.py``) or a class or test in it
(``test_determinism_baseline.py::TestSameBuildStrict``; a test covers its
parametrized cases). When it is set, as in every CI leg, any other skip fails
the run and is listed at the end, so a test that backs a contract cannot stop
running unnoticed (a dependency missing from a leg, a version pin gone stale).
An entry that covered no skip is reported too, without failing the run. Unset,
skips are only reported (``-rs``). ``conftest.py`` loads these hooks; outside
``tests/python``, ``pytest -p skip_guard`` does.
"""

import os
from pathlib import PurePosixPath

import pytest

ALLOWED_SKIPS = "POLYPUS_ALLOWED_SKIPS"

_skipped = []


def _covers(entry, nodeid):
    path, sep, rest = nodeid.partition("::")
    name = PurePosixPath(path).name + sep + rest
    return name == entry or name.startswith((entry + "::", entry + "["))


def unexpected_skips(nodeids, allowed):
    """The ``nodeids`` (as pytest reports them) that ``allowed`` does not cover,
    in order."""
    return [n for n in nodeids if not any(_covers(a, n) for a in allowed)]


def unused_entries(nodeids, allowed):
    """The entries of ``allowed`` that cover none of ``nodeids``."""
    return [a for a in allowed if not any(_covers(a, n) for n in nodeids)]


def _allowed():
    allowed = os.environ.get(ALLOWED_SKIPS)
    return None if allowed is None else allowed.split()


def pytest_sessionstart(session):
    _skipped.clear()


def pytest_collectreport(report):
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _skipped.append(report.nodeid)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    allowed = _allowed()
    if allowed is None:
        return
    skipped = list(dict.fromkeys(_skipped))
    unexpected = unexpected_skips(skipped, allowed)
    if unexpected:
        terminalreporter.write_sep(
            "=", f"skipped but not in {ALLOWED_SKIPS}", red=True, bold=True
        )
        for nodeid in unexpected:
            terminalreporter.write_line(nodeid)
    unused = unused_entries(skipped, allowed)
    if unused:
        terminalreporter.write_sep("=", f"{ALLOWED_SKIPS} entries no skip matched")
        for entry in unused:
            terminalreporter.write_line(entry)


def pytest_sessionfinish(session, exitstatus):
    allowed = _allowed()
    if (
        allowed is not None
        and exitstatus == pytest.ExitCode.OK
        and unexpected_skips(_skipped, allowed)
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
