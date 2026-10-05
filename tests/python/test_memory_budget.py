"""
Memory budget at the seam (issue #215, contract C-1 ``InsufficientMemoryError``).

A circuit whose ``16 * 2**n``-byte statevector does not fit in a *known* memory
budget is refused **before** anything is allocated with a typed
``polypus.InsufficientMemoryError`` (a ``polypus.BackendError``, so catchable as
``polypus.PolypusError``) instead of being started and killed by the Linux
out-of-memory killer, which gave no Python exception and no partial results.

``POLYPUS_MEM_BUDGET`` is set in a **fresh child interpreter** rather than with
``monkeypatch.setenv`` in this process: the Rust side reads it with ``getenv``
while its own worker threads may be running, and mutating the environment of a
live multithreaded process is not safe. The child also keeps the budget from
leaking into the rest of the suite.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _run_child(code: str, budget, tmp_path) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ``POLYPUS_MEM_BUDGET=budget``
    (unset when ``budget`` is ``None``). Auto-calibration is disabled and the
    cache redirected so the child does no unrelated work or disk writes."""
    env = os.environ.copy()
    env.pop("POLYPUS_MEM_BUDGET", None)
    if budget is not None:
        env["POLYPUS_MEM_BUDGET"] = budget
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"child failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_insufficient_memory_error_is_in_the_polypus_hierarchy():
    import polypus

    assert issubclass(polypus.InsufficientMemoryError, polypus.BackendError)
    assert issubclass(polypus.InsufficientMemoryError, polypus.PolypusError)
    assert "POLYPUS_MEM_BUDGET" in (polypus.InsufficientMemoryError.__doc__ or "")


# Every refusal is checked for the same actionable message: the qubit count, the
# required size (16 MiB for 20 qubits), the budget (1 MiB), where the budget came
# from, and how to override it.
_REFUSED = """
import polypus

def check(label, call):
    try:
        call()
    except polypus.PolypusError as exc:  # C-1: catchable at the base class
        assert type(exc) is polypus.InsufficientMemoryError, (label, type(exc))
        message = str(exc)
        for needle in ("20-qubit", "16.00 MiB", "1.00 MiB",
                       "set by POLYPUS_MEM_BUDGET", "POLYPUS_MEM_BUDGET=64G"):
            assert needle in message, (label, needle, message)
    else:
        raise AssertionError(f"{label}: expected InsufficientMemoryError")
    print(f"refused: {label}")

qc = polypus.Circuit(20).h(0).cx(0, 1).measure_all()
check("native", lambda: polypus.run_quantum_circuit(
    qc, shots=16, infrastructure="local", backend="polypus", seed=1))
check("statevector", lambda: polypus.statevector(polypus.Circuit(20).h(0)))
check("aer statevector", lambda: polypus.run_quantum_circuit(
    qc, shots=16, infrastructure="local", backend="aer",
    sim_method="statevector", seed=1))
"""


def test_a_circuit_over_an_explicit_budget_is_refused_with_a_typed_error(tmp_path):
    result = _run_child(_REFUSED, "1M", tmp_path)
    _assert_ok(result)
    for label in ("native", "statevector", "aer statevector"):
        assert f"refused: {label}" in result.stdout


def test_the_qubit_ceiling_value_error_still_comes_first(tmp_path):
    # Above `MAX_QUBITS` the ceiling's ValueError is unchanged, even under a budget
    # that would also refuse the circuit.
    code = """
    import polypus
    try:
        polypus.statevector(polypus.Circuit(31))
    except polypus.InsufficientMemoryError:
        raise AssertionError("the ceiling must be checked first")
    except ValueError as exc:
        assert "31" in str(exc) and "30" in str(exc), str(exc)
        print("ceiling first")
    """
    result = _run_child(code, "1M", tmp_path)
    _assert_ok(result)
    assert "ceiling first" in result.stdout


def test_a_circuit_within_the_budget_still_runs(tmp_path):
    # Suffixed values are honoured, not silently dropped: under 1M a 16-qubit
    # statevector (exactly 1 MiB) fits, and a large explicit budget lets a
    # 20-qubit circuit run on every refusing path.
    code = """
    import polypus
    assert polypus.statevector(polypus.Circuit(16).h(0)).shape == (1 << 16,)
    print("fits")
    """
    result = _run_child(code, "1M", tmp_path)
    _assert_ok(result)
    assert "fits" in result.stdout

    code = """
    import polypus
    qc = polypus.Circuit(20).h(0).cx(0, 1).measure_all()
    r = polypus.run_quantum_circuit(
        qc, shots=16, infrastructure="local", backend="polypus", seed=1)
    assert sum(r.counts[0].values()) == 16
    assert polypus.statevector(polypus.Circuit(20).h(0)).shape == (1 << 20,)
    print("ran")
    """
    result = _run_child(code, "32G", tmp_path)
    _assert_ok(result)
    assert "ran" in result.stdout


def test_aer_automatic_is_not_refused_on_the_dense_model(tmp_path):
    # `sim_method="automatic"` may pick a non-dense method (stabilizer for this
    # Clifford circuit), so it is only throttled, never refused on `16 * 2**n`.
    code = """
    import polypus
    qc = polypus.Circuit(20).h(0).cx(0, 1).measure_all()
    r = polypus.run_quantum_circuit(
        qc, shots=16, infrastructure="local", backend="aer",
        sim_method="automatic", seed=1)
    assert sum(r.counts[0].values()) == 16
    print("automatic ran")
    """
    result = _run_child(code, "1M", tmp_path)
    _assert_ok(result)
    assert "automatic ran" in result.stdout


def test_an_invalid_budget_is_warned_about_once_and_ignored(tmp_path):
    # With a logger installed, an unparseable value is reported (value + accepted
    # formats) exactly once however many runs re-read it, and the run proceeds on
    # the detected/fallback budget instead of failing or being silently dropped.
    code = """
    import polypus
    polypus.init_logger(level="warn", console=True, timestamp=False)
    qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
    for _ in range(3):
        polypus.run_quantum_circuit(
            qc, shots=8, infrastructure="local", backend="polypus", seed=1)
    print("runs done")
    """
    result = _run_child(code, "abc", tmp_path)
    _assert_ok(result)
    assert "runs done" in result.stdout
    output = result.stdout + result.stderr
    assert output.count('ignoring POLYPUS_MEM_BUDGET="abc"') == 1, output
    assert "32G" in output and "base 1024" in output, output


@pytest.mark.parametrize("budget", ["32G", "32g", "32GiB", "512M", "1048576"])
def test_valid_spellings_produce_no_warning(tmp_path, budget):
    code = """
    import polypus
    polypus.init_logger(level="warn", console=True, timestamp=False)
    qc = polypus.Circuit(2).h(0).measure_all()
    polypus.run_quantum_circuit(
        qc, shots=8, infrastructure="local", backend="polypus", seed=1)
    print("ok")
    """
    result = _run_child(code, budget, tmp_path)
    _assert_ok(result)
    assert "POLYPUS_MEM_BUDGET" not in result.stdout + result.stderr


# --- Aer's own validation through `max_memory_mb` (every sim_method) ------------
#
# Only `sim_method="statevector"` is refused in Rust on the dense `16 * 2**n`
# model. The other methods reach Aer with `max_memory_mb` set to the known
# budget; Aer validates each experiment for the method it actually picks, and its
# refusal is raised by `polypus_python.local` as `InsufficientMemoryError`.

_AER_REFUSED = """
import polypus

def ghz(n, non_clifford=False):
    qc = polypus.Circuit(n).h(0)
    for q in range(n - 1):
        qc = qc.cx(q, q + 1)
    if non_clifford:
        qc = qc.t(0)
    return qc.measure_all()

def check(label, qc, sim_method):
    try:
        polypus.run_quantum_circuit(
            qc, shots=16, infrastructure="local", backend="aer",
            sim_method=sim_method, seed=1)
    except polypus.PolypusError as exc:  # C-1: catchable at the base class
        assert type(exc) is polypus.InsufficientMemoryError, (label, type(exc))
        message = str(exc)
        for needle in ("Insufficient memory", "100 MiB",
                       "POLYPUS_MEM_BUDGET=64G"):
            assert needle in message, (label, needle, message)
    else:
        raise AssertionError(f"{label}: expected InsufficientMemoryError")
    print(f"refused: {label}")

# automatic + non-Clifford: Aer picks the statevector (1 GiB at 26 qubits).
check("automatic non-clifford", ghz(26, non_clifford=True), "automatic")
# density_matrix: 16 * 4**13 bytes = 1 GiB, beyond the 2**n model entirely.
check("density_matrix", ghz(13), "density_matrix")

# automatic + Clifford: Aer picks stabilizer, which fits — large Clifford
# circuits must keep running under the same budget.
r = polypus.run_quantum_circuit(
    ghz(26), shots=64, infrastructure="local", backend="aer",
    sim_method="automatic", seed=1)
counts = r.counts[0]
assert set(counts) <= {"0" * 26, "1" * 26}, counts
assert sum(counts.values()) == 64, counts
print("clifford ran")
"""


def test_aer_refuses_through_max_memory_mb_and_keeps_large_clifford(tmp_path):
    result = _run_child(_AER_REFUSED, "100M", tmp_path)
    _assert_ok(result)
    for label in ("automatic non-clifford", "density_matrix"):
        assert f"refused: {label}" in result.stdout
    assert "clifford ran" in result.stdout


def _non_clifford_qasm(n):
    import polypus

    qc = polypus.Circuit(n).h(0)
    for q in range(n - 1):
        qc = qc.cx(q, q + 1)
    return qc.t(0).measure_all().to_qasm2()


def test_installed_aer_reports_insufficient_memory_in_the_status():
    """Pin the text `polypus_python.local` matches on against the installed Aer,
    so a future Aer that rewords it fails here loudly instead of degrading the
    refusal back into a generic ``QiskitError``."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit.from_qasm_str(_non_clifford_qasm(20))  # 16 MiB statevector
    result = AerSimulator(method="automatic", max_memory_mb=1).run(qc, shots=1).result()
    (experiment,) = result.results
    assert not experiment.success
    assert "insufficient memory" in str(experiment.status).lower(), experiment.status


def test_the_local_seam_consumes_max_memory_mb():
    """C-1 must-be-consumed: ``polypus_python.run_qcs`` hands ``max_memory_mb``
    to Aer (a 1 MiB limit refuses a 16 MiB statevector with the typed class),
    and a direct caller that omits it keeps Aer's default and runs."""
    import polypus
    import polypus_python

    kwargs = dict(
        id="seam",
        backend="AerSimulator",
        qcs=[_non_clifford_qasm(20)],
        shots=8,
        sim_method="automatic",
        max_parallel_experiments=1,
        seed=1,
    )
    with pytest.raises(polypus.InsufficientMemoryError) as excinfo:
        polypus_python.run_qcs("local", max_memory_mb=1, **kwargs)
    assert "1 MiB" in str(excinfo.value)

    (counts,) = polypus_python.run_qcs("local", **kwargs)
    assert sum(counts.values()) == 8
