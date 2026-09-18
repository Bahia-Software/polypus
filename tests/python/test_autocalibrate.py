"""
Auto-calibration of the gate-parallel threshold at ``import polypus`` (issue
#176).

A ``pip install polypus-quantum`` of a published wheel has no post-install hook,
so it can never run ``install.sh`` step 6b, and the CI machine that *builds* the
wheel is not the one that *runs* it — a build-time measurement would calibrate the
wrong hardware. The custom ``polypus/__init__.py`` therefore calls
``calibrate_parallel_threshold(force=False)`` once on import: it reuses a valid
on-disk cache in ~0s and only measures on a cold/changed cache (<1s), so the cost
is paid once per machine, on the *real* runtime hardware.

Contract exercised here:
  * fresh import calibrates and writes the cache;
  * a second import against that cache reuses it with no measurement;
  * ``POLYPUS_NO_AUTOCALIBRATE=1`` skips calibration entirely;
  * a read-only cache dir does not raise and does not break import;
  * calibration failing for *any* reason never turns ``import polypus`` into a
    hard error.

Import runs the calibration exactly once per process (before any test-body code),
so the on-disk-behaviour tests each use their own child interpreter with a private
``XDG_CACHE_HOME``, mirroring ``test_calibration_warning.py``. The pure fail-safe
logic of the ``__init__`` helper is additionally unit-tested in-process, since it
does not depend on a fresh interpreter.
"""

import os
import subprocess
import sys

import pytest

polypus = pytest.importorskip("polypus")

_CACHE_REL = ("polypus", "parallel_threshold.json")


def _run_child(code: str, cache_home, *, optout=False) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `cache_home` as XDG_CACHE_HOME.

    Unlike the warning tests, auto-calibration is left *enabled* by default here
    (that is the behaviour under test); pass ``optout=True`` to set
    ``POLYPUS_NO_AUTOCALIBRATE=1``.
    """
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_home)
    if optout:
        env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    else:
        env.pop("POLYPUS_NO_AUTOCALIBRATE", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def test_fresh_import_calibrates_and_writes_cache(tmp_path):
    # Cold cache dir + auto-calibration on: importing polypus must measure once
    # and persist the result, so the cache file exists afterwards.
    code = """
import os, polypus
cache = os.path.join(os.environ["XDG_CACHE_HOME"], "polypus", "parallel_threshold.json")
print("EXISTS", os.path.isfile(cache))
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "EXISTS True" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert (tmp_path / _CACHE_REL[0] / _CACHE_REL[1]).is_file()


def test_second_import_reuses_cache_without_measuring(tmp_path):
    # First import warms the cache for this machine's thread count.
    warm = _run_child("import polypus\nprint('WARMED')", tmp_path)
    assert warm.returncode == 0, warm.stderr
    assert "WARMED" in warm.stdout
    cache_file = tmp_path / _CACHE_REL[0] / _CACHE_REL[1]
    assert cache_file.is_file()

    # Second import records the cache mtime *before* importing polypus, then imports
    # (which auto-calibrates force=False). A valid cache is reused as-is and never
    # rewritten, so the mtime is unchanged — proof no measurement happened.
    code = """
import os
cache = os.path.join(os.environ["XDG_CACHE_HOME"], "polypus", "parallel_threshold.json")
before = os.path.getmtime(cache)
import polypus
after = os.path.getmtime(cache)
print("UNCHANGED", before == after)
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "UNCHANGED True" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_optout_env_skips_autocalibration(tmp_path):
    # POLYPUS_NO_AUTOCALIBRATE=1 on a cold cache: import must not measure or write,
    # so no cache file is created.
    code = """
import os, polypus
cache = os.path.join(os.environ["XDG_CACHE_HOME"], "polypus", "parallel_threshold.json")
print("EXISTS", os.path.isfile(cache))
"""
    r = _run_child(code, tmp_path, optout=True)
    assert r.returncode == 0, r.stderr
    assert "EXISTS False" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert not (tmp_path / _CACHE_REL[0] / _CACHE_REL[1]).exists()


def test_readonly_cache_dir_does_not_break_import(tmp_path):
    # Point the cache dir resolution at a path whose parent is a regular file, so
    # the polypus cache dir cannot be created: import must still succeed (the Rust
    # side returns cache_written=False rather than raising), never breaking import.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(blocker)
    env.pop("HOME", None)  # no fallback cache dir either
    env.pop("POLYPUS_NO_AUTOCALIBRATE", None)
    r = subprocess.run(
        [sys.executable, "-c", "import polypus\nprint('IMPORTED')"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "IMPORTED" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_helper_swallows_calibration_errors(monkeypatch):
    # The try/except in __init__ must make import robust to *any* calibration
    # failure. Replace the extension the helper calls into with one that raises,
    # then invoke the helper directly: it must not propagate the error.
    monkeypatch.delenv("POLYPUS_NO_AUTOCALIBRATE", raising=False)

    class _Boom:
        @staticmethod
        def calibrate_parallel_threshold(force=False):
            raise RuntimeError("kaboom")

    # The helper reads the module-global name ``polypus`` (the compiled extension,
    # bound in the package namespace by ``from .polypus import *``); shadow it.
    monkeypatch.setattr(polypus, "polypus", _Boom)
    polypus._autocalibrate_parallel_threshold()  # must not raise


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_helper_respects_optout_env(monkeypatch, value):
    # With the opt-out set, the helper must not call into the extension at all.
    monkeypatch.setenv("POLYPUS_NO_AUTOCALIBRATE", value)
    called = []

    class _Spy:
        @staticmethod
        def calibrate_parallel_threshold(force=False):
            called.append(force)

    monkeypatch.setattr(polypus, "polypus", _Spy)
    polypus._autocalibrate_parallel_threshold()
    assert called == [], "opt-out must skip the calibration call entirely"


def test_helper_calls_calibration_with_force_false(monkeypatch):
    # The non-opt-out path must call calibrate_parallel_threshold(force=False):
    # reuse a valid cache, only measure when cold.
    monkeypatch.delenv("POLYPUS_NO_AUTOCALIBRATE", raising=False)
    called = []

    class _Spy:
        @staticmethod
        def calibrate_parallel_threshold(force=False):
            called.append(force)

    monkeypatch.setattr(polypus, "polypus", _Spy)
    polypus._autocalibrate_parallel_threshold()
    assert called == [False]
