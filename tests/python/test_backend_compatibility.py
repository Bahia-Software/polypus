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


def _run(circuit, backend, aer_kwargs=None):
    """``aer_kwargs`` (such as a ``sim_method``) go to the ``"aer"`` run only:
    the native backend rejects any ``sim_method`` but ``statevector`` before it
    looks at the circuit."""
    import polypus

    kwargs = (aer_kwargs or {}) if backend == "aer" else {}
    return polypus.run_quantum_circuit(
        circuit, shots=32, infrastructure="local", backend=backend, seed=5, **kwargs
    )


def _report(circuit):
    import polypus

    report = polypus.backend_compatibility(circuit)
    assert set(report) == {"aer", "polypus"}
    assert all(isinstance(r, list) for r in report.values())
    assert all(isinstance(m, str) and m for r in report.values() for m in r)
    return report


# Wider than any host's default ``AerSimulator()`` target (issue #307), and the
# ``sim_method`` that runs it.
WIDE = 200
MPS = {"sim_method": "matrix_product_state"}

# How the run quotes a lowering failure (``polypus.BackendError``'s message),
# and so how the Aer entry quotes it.
TRANSPILER_ERROR = "qiskit.transpiler.exceptions.TranspilerError: "


def _opaque():
    """A one-qubit instruction without a definition: nothing can lower it."""
    from qiskit.circuit import Instruction

    return Instruction("opaque_thing", 1, 0, [])


def _spy_rewrite(monkeypatch):
    """Record every rewrite the check runs, as (circuit, names of the passes,
    result), and let it run. The check looks ``qiskit.transpiler.PassManager``
    up on every call; ``local.py`` imported its own at import, so the run is
    not spied on."""
    import qiskit.transpiler

    calls = []
    original = qiskit.transpiler.PassManager

    class Spy(original):
        def __init__(self, passes=(), *args, **kwargs):
            super().__init__(passes, *args, **kwargs)
            self._names = [type(p).__name__ for p in passes]

        def run(self, circuits, *args, **kwargs):
            result = super().run(circuits, *args, **kwargs)
            calls.append((circuits, self._names, result))
            return result

    monkeypatch.setattr(qiskit.transpiler, "PassManager", Spy)
    return calls


def _forbid_rewrite(monkeypatch, why):
    import qiskit.transpiler

    def forbidden(*_args, **_kwargs):
        raise AssertionError(why)

    monkeypatch.setattr(qiskit.transpiler, "PassManager", forbidden)


def _assert_coherent(circuit, report, aer_kwargs=None):
    """Each backend the report accepts runs the circuit; each it rejects fails
    with (one of) the reasons given. ``aer_kwargs`` go to the Aer run only
    (``_run``)."""
    import polypus

    for backend, reasons in report.items():
        if not reasons:
            counts = _run(circuit, backend, aer_kwargs).counts[0]
            assert sum(counts.values()) == 32
            continue
        with pytest.raises((polypus.PolypusError, ValueError)) as info:
            _run(circuit, backend, aer_kwargs)
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

    @pytest.mark.parametrize("gate", ["ch", "rccx"])
    def test_gate_outside_aer_basis_is_accepted_by_aer(self, gate):
        """A Qiskit circuit with ``ch`` or ``rccx`` is lowered by the local
        backend before Aer runs it, so the Aer entry is empty and the run
        succeeds."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        if gate == "ch":
            qc.ch(0, 2)
        else:
            qc.rccx(0, 1, 2)
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

    def test_opaque_instruction_is_rejected_by_aer(self):
        """An instruction without a definition cannot be lowered into Aer's
        basis: the Aer entry quotes the error the run raises (issue #285)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.append(_opaque(), [0])
        qc.measure(0, 0)
        report = _report(qc)
        assert len(report["aer"]) == 1
        reason = report["aer"][0]
        assert reason.startswith(TRANSPILER_ERROR)
        assert 'HighLevelSynthesis is unable to synthesize "opaque_thing"' in reason
        _assert_coherent(qc, report)

    def test_box_is_rejected_by_aer(self):
        """``box`` is a control-flow operation Aer's target does not take and
        nothing can rewrite (issue #285). The run refuses it before rewriting,
        with a message that does not change between calls (issue #307), and
        the check quotes it."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        with qc.box():
            qc.h(0)
        qc.measure([0, 1], [0, 1])
        report = _report(qc)
        assert len(report["aer"]) == 1
        reason = report["aer"][0]
        assert reason.startswith(TRANSPILER_ERROR)
        assert (
            "The control-flow construct 'box' is not supported by the backend."
            in reason
        )
        _assert_coherent(qc, report)

    def test_gate_outside_basis_in_a_control_flow_body_is_lowered(self, monkeypatch):
        """``count_ops`` does not look into control-flow bodies, so the bodies
        are walked as the run walks them: a ``ch`` inside an ``if_test`` makes
        the circuit outside Aer's basis, it is rewritten, and since that
        lowers it the Aer entry is empty and the run succeeds."""
        from qiskit import QuantumCircuit

        calls = _spy_rewrite(monkeypatch)
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.ch(0, 1)
        qc.measure(1, 1)
        report = _report(qc)
        assert report["aer"] == []
        assert len(calls) == 1
        _assert_coherent(qc, report)

    def test_opaque_instruction_in_a_control_flow_body_is_rejected_by_aer(self):
        """An instruction nothing can lower is found inside an ``if_test`` body
        too, and the Aer entry quotes the error the run raises for it."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.append(_opaque(), [1])
        qc.measure(1, 1)
        report = _report(qc)
        assert len(report["aer"]) == 1
        assert 'unable to synthesize "opaque_thing"' in report["aer"][0]
        _assert_coherent(qc, report)

    @pytest.mark.parametrize("control_flow", [False, True])
    def test_a_circuit_in_the_basis_is_not_rewritten(self, monkeypatch, control_flow):
        """A circuit in Aer's basis reaches Aer as it is, so the check does not
        rewrite it either: that could only invent a rejection the run never
        raises. With a control-flow operation the bodies are walked, and a body
        in the basis does not make the circuit outside it."""
        from qiskit import QuantumCircuit

        _forbid_rewrite(monkeypatch, "a circuit in Aer's basis must not be rewritten")
        if control_flow:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.measure(0, 0)
            with qc.if_test((qc.clbits[0], 1)):
                qc.x(1)
            qc.measure(1, 1)
        else:
            qc = self._bell()
            qc.barrier()
        report = _report(qc)
        assert report["aer"] == []
        _assert_coherent(qc, report)

    @pytest.mark.parametrize("width", [2, WIDE])
    def test_a_circuit_outside_the_basis_is_rewritten_as_the_run_does(
        self, monkeypatch, width
    ):
        """Once, on the caller's circuit, with the passes
        ``_rewrite_into_aer_basis`` runs, and into the same circuit, at any
        width: a circuit wider than the default ``AerSimulator()``'s target is
        rewritten like a narrow one (issue #307)."""
        from polypus_python.local import _aer_basis, _rewrite_into_aer_basis
        from qiskit import QuantumCircuit

        calls = _spy_rewrite(monkeypatch)
        qc = QuantumCircuit(width)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        assert _report(qc)["aer"] == []
        assert len(calls) == 1
        circuit, passes, result = calls[0]
        assert circuit is qc
        assert passes == ["HighLevelSynthesis", "BasisTranslator"]
        assert result == _rewrite_into_aer_basis(qc, _aer_basis())

    @pytest.mark.parametrize("raised", [KeyboardInterrupt, RuntimeError])
    def test_an_error_not_raised_by_qiskit_propagates(self, monkeypatch, raised):
        """Only an exception raised by Qiskit is a reason; any other one, a
        cancellation included, reaches the caller as it is."""
        import qiskit.transpiler
        from qiskit import QuantumCircuit

        def failing(*_args, **_kwargs):
            raise raised("not a Qiskit error")

        monkeypatch.setattr(qiskit.transpiler, "PassManager", failing)
        qc = QuantumCircuit(2)
        qc.ch(0, 1)
        qc.measure_all()
        with pytest.raises(raised, match="not a Qiskit error"):
            _report(qc)

    @pytest.mark.parametrize("base", [ValueError, TypeError, KeyboardInterrupt])
    def test_a_qiskit_error_the_run_keeps_unwrapped_propagates(self, monkeypatch, base):
        """The run raises a Qiskit exception that is also a ``ValueError``,
        ``TypeError`` or ``KeyboardInterrupt`` as it is, not as
        ``polypus.BackendError`` (contract C-1), so the check raises it too
        instead of turning it into a reason: a cancellation is never
        swallowed."""
        import qiskit.transpiler
        from qiskit import QuantumCircuit
        from qiskit.exceptions import QiskitError

        error = type(f"_Qiskit{base.__name__}", (QiskitError, base), {})

        def failing(*_args, **_kwargs):
            raise error("raised by a Qiskit class")

        monkeypatch.setattr(qiskit.transpiler, "PassManager", failing)
        qc = QuantumCircuit(2)
        qc.ch(0, 1)
        qc.measure_all()
        with pytest.raises(error, match="raised by a Qiskit class"):
            _report(qc)

    def test_missing_aer_is_the_reason_and_nothing_is_rewritten(self, monkeypatch):
        import sys

        from qiskit import QuantumCircuit

        _forbid_rewrite(monkeypatch, "nothing can be rewritten for a missing Aer")
        monkeypatch.setitem(sys.modules, "qiskit_aer", None)
        qc = QuantumCircuit(1)
        qc.append(_opaque(), [0])
        qc.measure_all()
        report = _report(qc)
        assert len(report["aer"]) == 1
        assert "Qiskit Aer is not installed" in report["aer"][0]
        assert 'backend="aer" cannot run' in report["aer"][0]

    def test_wide_circuit_outside_the_basis_is_accepted_by_aer(self):
        """Wider than the default ``AerSimulator()``'s target (whose width depends on
        the host's memory; 200 qubits is above any): the lowering has no width cap, so
        ``matrix_product_state`` runs it and the check accepts it (issue #307). The
        check looks at no resources: a ``sim_method`` without the memory for it is not
        its concern."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(WIDE)
        qc.x(0)
        qc.ch(0, 1)
        qc.measure_all()
        report = _report(qc)
        assert report["aer"] == []
        _assert_coherent(qc, report, MPS)

    @pytest.mark.parametrize("kind", ["opaque", "box"])
    def test_wide_circuit_that_cannot_be_lowered_is_rejected_by_aer(self, kind):
        """Width changes nothing for what cannot be lowered: the check quotes
        the error the run raises, as for a narrow circuit."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(WIDE, 1)
        if kind == "opaque":
            qc.append(_opaque(), [0])
            fragment = 'HighLevelSynthesis is unable to synthesize "opaque_thing"'
        else:
            with qc.box():
                qc.h(WIDE - 1)
            fragment = (
                "The control-flow construct 'box' is not supported by the backend."
            )
        qc.measure(0, 0)
        report = _report(qc)
        assert len(report["aer"]) == 1
        reason = report["aer"][0]
        assert reason.startswith(TRANSPILER_ERROR)
        assert fragment in reason
        _assert_coherent(qc, report, MPS)

    @pytest.mark.parametrize("nesting", ["for_loop", "if_test"])
    def test_a_box_in_a_control_flow_body_is_rejected_by_aer(self, nesting):
        """A ``box`` inside another control-flow body is refused with the same
        message as a top-level one, in the run and in the check (issue #307):
        before, the rewrite's own error for it listed the operations and the
        basis in an order that changed from call to call."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        body = (
            qc.for_loop(range(2))
            if nesting == "for_loop"
            else qc.if_test((qc.clbits[0], 0))
        )
        with body:
            with qc.box():
                qc.ch(0, 1)
        qc.measure(1, 1)
        report = _report(qc)
        assert len(report["aer"]) == 1
        reason = report["aer"][0]
        assert reason.startswith(TRANSPILER_ERROR)
        assert (
            "The control-flow construct 'box' is not supported by the backend."
            in reason
        )
        assert _report(qc)["aer"] == [reason]
        _assert_coherent(qc, report)

    def test_the_caller_circuit_is_not_modified(self):
        from qiskit import QuantumCircuit

        lowerable = QuantumCircuit(2)
        lowerable.ch(0, 1)
        lowerable.measure_all()
        opaque = QuantumCircuit(1)
        opaque.append(_opaque(), [0])
        opaque.measure_all()
        for qc in (self._bell(), lowerable, opaque):
            before = list(qc.data)
            copy = qc.copy()
            _report(qc)
            assert list(qc.data) == before
            assert qc == copy


def test_check_runs_nothing(monkeypatch):
    """Pure: no seam call, so a broken ``run_qcs`` cannot affect it, and no
    simulation, not even for a Qiskit circuit the check rewrites."""
    import polypus_python
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backend_compatibility must not run the circuit")

    monkeypatch.setattr(polypus_python, "run_qcs", forbidden)
    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", forbidden)
    monkeypatch.setattr(AerSimulator, "run", forbidden)
    assert _report(VALID) == {"aer": [], "polypus": []}
    qc = QuantumCircuit(2)
    qc.ch(0, 1)
    qc.measure_all()
    assert _report(qc)["aer"] == []


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
