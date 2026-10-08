"""
``polypus.backend_compatibility`` — the structural pre-check (issue #218).

For each local backend (``"aer"``, ``"polypus"``) it lists the reasons that
backend would reject a circuit; an empty list means the backend accepts it. The
native entry is decided by the same code that runs the circuit (the OpenQASM
importer, the Qiskit-circuit guard), so every "accepts" is checked here by
running the circuit, and every "rejects" by running it and matching the reason.
Nothing is executed by the check itself.
"""

import pytest

H = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

VALID = H + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n"
RESET = H + "qreg q[2];\ncreg c[2];\nx q[0];\nreset q[0];\nmeasure q -> c;\n"
MID_MEASURE = (
    H
    + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nx q[0];\nmeasure q[0] -> c[1];\n"
)
IF = (
    H
    + "qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\n"
    + "if(c==1) x q[1];\nmeasure q[1] -> c[1];\n"
)
INVALID = "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n"

# (circuit, fragment of the native reason) for circuits only Aer can run.
AER_ONLY = {
    "reset": (RESET, "'reset' is not supported"),
    "mid_circuit_measure": (MID_MEASURE, "after it was measured"),
    "if": (IF, "'if' statements are not supported"),
}


def _run(circuit, backend):
    import polypus

    return polypus.run_quantum_circuit(
        circuit, shots=32, infrastructure="local", backend=backend, seed=5
    )


def _report(circuit):
    import polypus

    report = polypus.backend_compatibility(circuit)
    assert set(report) == {"aer", "polypus"}
    assert all(isinstance(r, list) for r in report.values())
    assert all(isinstance(m, str) and m for r in report.values() for m in r)
    return report


def _assert_coherent(circuit, report):
    """Each backend the report accepts runs the circuit; each it rejects fails
    with (one of) the reasons given."""
    import polypus

    for backend, reasons in report.items():
        if not reasons:
            assert sum(_run(circuit, backend).counts[0].values()) == 32
            continue
        with pytest.raises((polypus.PolypusError, ValueError)) as info:
            _run(circuit, backend)
        assert any(reason in str(info.value) for reason in reasons), (
            f"{backend}: {info.value!s} does not state any of {reasons}"
        )


def test_is_exported():
    import polypus

    assert callable(polypus.backend_compatibility)
    assert "backend_compatibility" in dir(polypus)


class TestQasmStrings:
    def test_valid_circuit_is_accepted_by_both(self):
        report = _report(VALID)
        assert report == {"aer": [], "polypus": []}
        _assert_coherent(VALID, report)

    @pytest.mark.parametrize("case", sorted(AER_ONLY))
    def test_dynamic_circuit_is_aer_only(self, case):
        circuit, fragment = AER_ONLY[case]
        report = _report(circuit)
        assert report["aer"] == []
        assert len(report["polypus"]) == 1
        assert fragment in report["polypus"][0]
        _assert_coherent(circuit, report)

    def test_invalid_qasm_is_rejected_by_both(self):
        report = _report(INVALID)
        assert report["polypus"] and report["aer"]
        assert "QASM2ParseError" in report["aer"][0]
        _assert_coherent(INVALID, report)

    def test_missing_qasm_parser_is_a_reason_for_aer(self, monkeypatch):
        """If Qiskit and Aer import but the parser the Aer path uses is gone,
        Aer cannot run the circuit and the entry says so, not ``[]``."""
        from qiskit import QuantumCircuit

        monkeypatch.delattr(QuantumCircuit, "from_qasm_str")
        report = _report(VALID)
        assert len(report["aer"]) == 1
        assert "Qiskit Aer is not available" in report["aer"][0]
        assert report["polypus"] == []

    def test_circuit_without_measurements_is_accepted_by_both(self):
        circuit = H + "qreg q[2];\ncreg c[3];\nx q[0];\n"
        report = _report(circuit)
        assert report == {"aer": [], "polypus": []}
        _assert_coherent(circuit, report)


class TestNativeCircuits:
    def test_bound_circuit_is_accepted_by_both(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        report = _report(qc)
        assert report == {"aer": [], "polypus": []}
        _assert_coherent(qc, report)

    def test_free_parameters_are_reported_for_both(self):
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        report = _report(qc)
        for reasons in report.values():
            assert len(reasons) == 1 and "unbound parameters" in reasons[0]
        _assert_coherent(qc, report)

    @pytest.mark.parametrize(
        "statement",
        ["ch q[0],q[1];", "rccx q[0],q[1],q[2];", "u0(1) q[0];"],
    )
    def test_gate_outside_aer_basis_is_accepted_by_both(self, statement):
        """``ch``, ``rccx`` and ``u0`` are native vocabulary (C-2) and not in
        Aer's basis, but the local backend lowers them before running on Aer
        (issue #252), so neither backend rejects them and both run them."""
        import polypus

        src = (
            H
            + "qreg q[3];\ncreg c[3];\nx q[0];\nx q[1];\n"
            + statement
            + "\nmeasure q -> c;\n"
        )
        circuit = polypus.Circuit.from_qasm2(src)
        for given in (circuit, src):
            report = _report(given)
            assert report == {"aer": [], "polypus": []}
            _assert_coherent(given, report)

    def test_ch_gives_equivalent_counts_on_both_backends(self):
        """x on the control makes ``ch`` a Hadamard on the target: both
        backends see the uniform split of the target and a fixed control."""
        import polypus

        qc = polypus.Circuit(2).x(0).ch(0, 1).measure_all()
        assert _report(qc) == {"aer": [], "polypus": []}
        shots = 400
        for backend in ("aer", "polypus"):
            counts = polypus.run_quantum_circuit(
                qc, shots=shots, infrastructure="local", backend=backend, seed=5
            ).counts[0]
            assert set(counts) == {"01", "11"}
            assert sum(counts.values()) == shots
            # Binomial(400, 1/2): 5 standard deviations are 5 * sqrt(100) = 50.
            assert abs(counts["01"] - shots / 2) <= 50

    def test_declared_gate_is_accepted_by_both(self):
        import polypus

        src = (
            H
            + "gate pair a,b { ch a,b; cx b,a; }\n"
            + "qreg q[2];\ncreg c[2];\nx q[0];\npair q[0],q[1];\nmeasure q -> c;\n"
        )
        circuit = polypus.Circuit.from_qasm2(src)
        for given in (circuit, src):
            report = _report(given)
            assert report == {"aer": [], "polypus": []}
            _assert_coherent(given, report)


class TestQiskitCircuits:
    def _bell(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def test_native_always_rejects_a_quantum_circuit(self):
        qc = self._bell()
        report = _report(qc)
        assert report["aer"] == []
        assert len(report["polypus"]) == 1
        assert "cannot execute a Qiskit QuantumCircuit" in report["polypus"][0]
        _assert_coherent(qc, report)

    def test_gate_outside_aer_basis_is_accepted_by_aer(self):
        """A Qiskit circuit with ``ch`` is lowered by the local backend before
        Aer runs it, so the Aer entry is empty and the run succeeds."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        report = _report(qc)
        assert report["aer"] == []
        assert len(report["polypus"]) == 1
        assert "cannot execute a Qiskit QuantumCircuit" in report["polypus"][0]
        _assert_coherent(qc, report)

    def test_dynamic_features_are_listed(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)  # gate after measure
        qc.reset(1)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(2)
        qc.measure(2, 2)
        report = _report(qc)
        assert report["aer"] == []
        native = report["polypus"]
        assert "cannot execute a Qiskit QuantumCircuit" in native[0]
        joined = "\n".join(native[1:])
        assert "reset" in joined
        assert "qubit 0" in joined and "after it was measured" in joined
        assert "if_else" in joined
        _assert_coherent(qc, report)

    def test_barrier_and_remeasure_after_measure_are_not_dynamic(self):
        """C-4: a barrier or a second measurement of a measured qubit is still
        a terminal-measurement circuit, so only the Qiskit guard is listed."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 2)
        qc.measure(0, 0)
        qc.barrier()
        qc.measure(0, 1)
        assert len(_report(qc)["polypus"]) == 1

    def test_the_caller_circuit_is_not_modified(self):
        qc = self._bell()
        before = list(qc.data)
        _report(qc)
        assert list(qc.data) == before


def test_check_runs_nothing(monkeypatch):
    """Pure: no seam call, so a broken ``run_qcs`` cannot affect it."""
    import polypus_python

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backend_compatibility must not run the circuit")

    monkeypatch.setattr(polypus_python, "run_qcs", forbidden)
    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", forbidden)
    assert _report(VALID) == {"aer": [], "polypus": []}


def test_check_writes_no_log_record(tmp_path):
    """A rejected circuit is the expected answer of the check, not an error, so
    it writes nothing to the Polypus log; running the circuit still logs the
    failure. Run in a child interpreter: the logger is installed once per
    process, and the test runner's may already be taken."""
    import os
    import subprocess
    import sys
    import textwrap

    log = tmp_path / "polypus.log"
    setup = f"import polypus\npolypus.init_logger(level='info', file={str(log)!r})\n"
    check = f"assert polypus.backend_compatibility({RESET!r})['polypus']\n"
    run = textwrap.dedent(
        f"""
        try:
            polypus.run_quantum_circuit(
                {RESET!r}, shots=8, infrastructure="local", backend="polypus"
            )
        except polypus.NativeCircuitError:
            pass
        """
    )
    env = dict(os.environ, POLYPUS_NO_AUTOCALIBRATE="1")

    subprocess.run([sys.executable, "-c", setup + check], env=env, check=True)
    after_check = log.read_text() if log.exists() else ""
    assert "could not parse" not in after_check
    assert "ERROR" not in after_check

    log.unlink(missing_ok=True)
    subprocess.run([sys.executable, "-c", setup + check + run], env=env, check=True)
    assert "native backend could not parse OpenQASM 2.0" in log.read_text()


@pytest.mark.parametrize("obj", [42, None, b"OPENQASM 2.0;"])
def test_rejects_objects_that_are_not_circuits(obj):
    import polypus

    with pytest.raises(TypeError, match="backend_compatibility"):
        polypus.backend_compatibility(obj)
