"""
Default-visible fallback warning for the gate-parallel threshold (issue #127
follow-up, revised for auto-calibration in issue #176).

Since issue #176 ``import polypus`` auto-calibrates the machine once, so on the
happy path the native statevector backend resolves to a *calibrated* threshold
and nothing warns. The only remaining default-visible ``UserWarning`` is the
``HardwareChanged`` case; the ``NotCalibrated`` case (empty/unreadable cache) was
demoted to a logger-only ``info!`` so it never nags ``pip install`` users whose
cache dir happens to be read-only (containers/CI) — the result is numerically
identical regardless.

Every child here sets ``POLYPUS_NO_AUTOCALIBRATE=1`` so import does **not**
silently calibrate the temp cache: that lets each scenario pin an exact cache
state (empty / old-schema / foreign-thread-count / freshly calibrated) and assert
what the *fallback* path does, which is the point of this file. The end-to-end
"import calibrates so nothing warns" behaviour is covered in
``test_autocalibrate.py``.

Why subprocesses: ``resolve_threshold()`` memoises its answer in a process-life
``OnceLock``, and the bindings guard the warning with a process-global
``AtomicBool``. Both are one-shot *per process*, so each scenario needs its own
process to observe a fresh resolution. This mirrors the subprocess isolation in
``test_interrupt.py``.

Each child gets its own ``XDG_CACHE_HOME`` so it never reads or writes the real
user cache; none of them call ``init_logger``, proving no warning is visible *by
default*.
"""

import os
import subprocess
import sys

import pytest

polypus = pytest.importorskip("polypus")

# Substring present in the fallback UserWarning and in the demoted info! log — the
# actionable call to action. Absence from stderr proves nothing warned by default.
CAL_MARKER = "calibrate_parallel_threshold"

# A 2-qubit Bell circuit as OpenQASM 2.0, usable by both the native and the Aer
# backends of run_quantum_circuit (passed as a string → BoundCircuit::Qasm2).
_BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""


def _run_child(code: str, cache_home) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `cache_home` as XDG_CACHE_HOME.

    Inherits the parent environment (so the editable `polypus` install and any
    LD_LIBRARY_PATH are visible) but never installs a logger, so any warning the
    child prints proves the warning is visible with no setup. Auto-calibration at
    import is disabled (POLYPUS_NO_AUTOCALIBRATE=1) so each test controls the
    exact on-disk cache state; the fallback path is what these tests assert.
    """
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_home)
    env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _seed_old_schema_cache(cache_home) -> None:
    """Write a cache in the pre-map ``schema:1`` layout (a flat
    ``{threshold, num_threads}``).

    Since commit ea6562a (``feat(sim): key the gate-parallel calibration cache by
    thread count``) the on-disk cache is a ``schema:2`` thread-count → threshold
    map, and ``read_cache_from`` treats any other schema as **absent** — no
    migration, worst case one recalibration. So this file does not model a
    "different hardware" cache (``FallbackReason::HardwareChanged`` is unreachable
    under thread-count keying); it models a cache the current build cannot read,
    which resolves to the ``NotCalibrated`` fallback, exactly as an empty cache
    dir does."""
    d = cache_home / "polypus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "parallel_threshold.json").write_text(
        '{"schema":1,"threshold":14,"num_threads":999999}'
    )


def _seed_foreign_thread_count_cache(cache_home) -> None:
    """Write a current-schema (``schema:2``) cache holding a calibrated entry for a
    thread count that cannot match this machine (``999999``), so the map is
    populated but has no entry for the current thread count.

    This is the routine "size-uncalibrated" case a shared SLURM node hits when a
    job runs at a ``--cpus-per-task`` allotment not yet calibrated: the machine is
    calibrated (for other thread counts), just not for this one. It resolves to
    ``Decision::SizeUncalibrated`` — the default threshold with an ``info!`` and,
    deliberately, **no** Python ``UserWarning``."""
    d = cache_home / "polypus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "parallel_threshold.json").write_text('{"schema":2,"entries":{"999999":14}}')


def test_statevector_does_not_warn_when_uncalibrated(tmp_path):
    # Empty cache dir → NotCalibrated fallback. Since #176 this is demoted to a
    # logger-only info!, so with no logger installed nothing reaches stderr.
    code = """
import polypus
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_statevector_does_not_warn_when_cache_is_old_schema(tmp_path):
    # A cache written in the old flat ``schema:1`` layout is treated as *absent*
    # by a ``schema:2`` build (commit ea6562a — no migration), so this is the
    # NotCalibrated fallback, exactly like an empty cache dir: demoted to info!,
    # no default-visible warning.
    _seed_old_schema_cache(tmp_path)
    code = """
import polypus
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_statevector_does_not_warn_when_only_other_thread_counts_calibrated(tmp_path):
    # A current-schema cache calibrated for other thread counts but not this one is
    # the routine size-uncalibrated case on a shared node whose jobs get different
    # CPU allotments: it uses the default threshold silently (info! only), and the
    # bindings raise NO default-visible UserWarning.
    _seed_foreign_thread_count_cache(tmp_path)
    code = """
import polypus
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_no_warning_when_calibrated(tmp_path):
    # Calibrating first writes a cache valid for this machine, so the subsequent
    # statevector run resolves to the cached threshold and must not warn.
    code = """
import polypus
polypus.calibrate_parallel_threshold(force=True)
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_run_quantum_circuit_native_does_not_warn_when_uncalibrated(tmp_path):
    # The native ("polypus") backend consults the threshold. Uncalibrated →
    # NotCalibrated → demoted to info!, so no default-visible warning.
    code = f'''
import polypus
qasm = """{_BELL_QASM}"""
polypus.run_quantum_circuit(qasm, shots=64, infrastructure="local", backend="polypus")
print("DONE")
'''
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_run_quantum_circuit_aer_never_warns(tmp_path):
    # backend="aer" never touches the native statevector path, so even with no
    # cache at all it must not emit the calibration warning.
    code = f'''
import polypus
qasm = """{_BELL_QASM}"""
polypus.run_quantum_circuit(qasm, shots=64, infrastructure="local", backend="aer")
print("DONE")
'''
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER not in r.stderr, r.stderr


def test_uncalibrated_native_runs_are_quiet_across_repeated_calls(tmp_path):
    # Two native runs on an uncalibrated machine: neither emits a default-visible
    # warning now that NotCalibrated is a logger-only info!. (Before #176 the first
    # call raised exactly one UserWarning; the one-shot AtomicBool guard now only
    # governs the still-reachable HardwareChanged case, which thread-count keying
    # makes unreachable from Python.)
    code = """
import warnings
import polypus
qc = polypus.Circuit(2).h(0).cx(0, 1)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    polypus.statevector(qc)
    polypus.statevector(qc)
hits = [w for w in caught if "calibrate_parallel_threshold" in str(w.message)]
print("HITS", len(hits))
"""
    r = _run_child(code, tmp_path)  # empty cache → NotCalibrated, now demoted
    assert r.returncode == 0, r.stderr
    assert "HITS 0" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
