"""
Default-visible fallback warning for the gate-parallel threshold (issue #127
follow-up).

When a process runs the native statevector backend with an *uncalibrated* (or
stale) threshold, ``polypus-sim`` logs a ``warn!``/``info!`` — but that is
silent unless the user called ``polypus.init_logger()``, which most do not. The
bindings therefore *also* raise a Python ``UserWarning``, printed to stderr by
default with no setup, naming ``polypus.calibrate_parallel_threshold()`` so the
user can fix it.

Why subprocesses: ``resolve_threshold()`` memoises its answer in a process-life
``OnceLock``, and the bindings guard the warning with a process-global
``AtomicBool``. Both are one-shot *per process*, so each scenario (uncalibrated
/ old-schema cache / size-uncalibrated / calibrated / aer-vs-native) needs its
**own** process to observe a fresh resolution — sharing one pytest process would
let whichever test ran first fix the answer for all the others. This mirrors the
subprocess isolation in ``test_interrupt.py``. The "at most once per process" case is the
one exception that fits in a single process, but it too runs in a child so its
count is not disturbed by other tests that already ran the native path.

Each child gets its own ``XDG_CACHE_HOME`` so it never reads or writes the real
user cache; none of them call ``init_logger``, proving the warning is visible
*by default*.
"""

import os
import subprocess
import sys

import pytest

polypus = pytest.importorskip("polypus")

# Substring present in *both* fallback messages — the actionable call to action.
CAL_MARKER = "calibrate_parallel_threshold"

# A 2-qubit Bell circuit as OpenQASM 2.0, usable by both the native and the Aer
# backends of run_quantum_circuit (passed as a string → BoundCircuit::Qasm2).
_BELL_QASM = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
'''


def _run_child(code: str, cache_home) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `cache_home` as XDG_CACHE_HOME.

    Inherits the parent environment (so the editable `polypus` install and any
    LD_LIBRARY_PATH are visible) but never installs a logger, so any warning the
    child prints proves the warning is visible with no setup.
    """
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_home)
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
    which resolves to the serious ``NotCalibrated`` fallback, exactly as an empty
    cache dir does."""
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


def test_statevector_warns_when_uncalibrated(tmp_path):
    # Empty cache dir → NotCalibrated fallback → default-visible UserWarning.
    code = """
import polypus
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER in r.stderr, r.stderr
    assert "has not been calibrated" in r.stderr, r.stderr


def test_statevector_warns_when_cache_is_old_schema(tmp_path):
    # A cache written in the old flat ``schema:1`` layout is treated as *absent*
    # by a ``schema:2`` build (commit ea6562a — no migration), so this is the
    # serious NotCalibrated fallback, exactly like an empty cache dir. It is NOT
    # the HardwareChanged case: thread-count keying makes that reason unreachable,
    # so the warning must be the "never calibrated" one and must NOT mention
    # force=True (which belongs only to the unreachable hardware-changed message).
    _seed_old_schema_cache(tmp_path)
    code = """
import polypus
polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))
print("DONE")
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER in r.stderr, r.stderr
    assert "has not been calibrated" in r.stderr, r.stderr
    assert "force=True" not in r.stderr, r.stderr


def test_statevector_does_not_warn_when_only_other_thread_counts_calibrated(tmp_path):
    # A current-schema cache calibrated for other thread counts but not this one is
    # the routine size-uncalibrated case on a shared node whose jobs get different
    # CPU allotments: it uses the default threshold silently (info! only), and the
    # bindings must raise NO default-visible UserWarning. There was no Python-level
    # test for this designed-for-SLURM behaviour before; only the Rust unit test
    # ``decide_uses_the_entry_for_the_current_thread_count`` covered it.
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


def test_run_quantum_circuit_native_warns_when_uncalibrated(tmp_path):
    # The native ("polypus") backend consults the threshold, so it warns.
    code = f'''
import polypus
qasm = """{_BELL_QASM}"""
polypus.run_quantum_circuit(qasm, shots=64, infrastructure="local", backend="polypus")
print("DONE")
'''
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DONE" in r.stdout
    assert CAL_MARKER in r.stderr, r.stderr


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


def test_warns_at_most_once_per_process(tmp_path):
    # Two native runs, one warning: the explicit AtomicBool guard fires once even
    # with Python's own duplicate filter disabled ("always"), so a second call is
    # provably suppressed by us, not by Python's dedup.
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
    r = _run_child(code, tmp_path)  # empty cache → NotCalibrated, so it does warn
    assert r.returncode == 0, r.stderr
    assert "HITS 1" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
