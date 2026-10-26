"""
``polypus.backend_compatibility`` — the pre-check (issues #218 and #254).

For each backend (``"aer"``, ``"polypus"``, ``"cunqa"``, ``"qmio"``) it lists the
reasons that backend would reject a circuit; an empty list means the backend
accepts it. Each entry is decided by the code the run uses (the OpenQASM
importer, the Qiskit-circuit guards, the Aer rewrite, the CUNQA seal), so every
"accepts" is checked here by running the circuit, and every "rejects" by
running it and matching the reason. Nothing is executed by the check itself.

CUNQA is the fake SDK of ``fake_cunqa.py`` with CUNQA 3.0.0's converter, so the
``"cunqa"`` entry is decided and run the same on every host and no test reaches
SLURM. QMIO is not compiled into the default build, so its runs fail with the
reason the entry gives; an accepted ``"qmio"`` entry is not run (it needs a QMIO
endpoint).
"""

import re

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


# How each entry's backend is run.
_RUN = {
    "aer": dict(infrastructure="local", backend="aer", seed=5),
    "polypus": dict(infrastructure="local", backend="polypus", seed=5),
    "cunqa": dict(infrastructure="cunqa", backend="aer", seed=5),
    "qmio": dict(infrastructure="qmio"),
}


def _run(circuit, backend):
    import polypus

    return polypus.run_quantum_circuit(circuit, shots=32, **_RUN[backend])


def _report(circuit, **kwargs):
    import polypus

    report = polypus.backend_compatibility(circuit, **kwargs)
    assert list(report) == ["aer", "polypus", "cunqa", "qmio"]
    assert all(isinstance(r, list) for r in report.values())
    assert all(isinstance(m, str) and m for r in report.values() for m in r)
    return report


def _local(report):
    """The entries of the local backends."""
    return {backend: report[backend] for backend in ("aer", "polypus")}


@pytest.fixture(autouse=True)
def cunqa_sdk(monkeypatch):
    from fake_cunqa import FakeCunqa

    return FakeCunqa("3.0.0", converter=True).install(monkeypatch)


# How the run quotes a Qiskit transpiler failure (``polypus.BackendError``'s
# message), and so how the Aer entry quotes it.
TRANSPILER_ERROR = "qiskit.transpiler.exceptions.TranspilerError: "


def _opaque():
    """A one-qubit instruction without a definition: nothing can lower it."""
    from qiskit.circuit import Instruction

    return Instruction("opaque_thing", 1, 0, [])


def _spy_rewrite(monkeypatch):
    """Record every call of the rewrite the run lowers with (args, kwargs) and
    let it run."""
    from polypus_python import local

    calls = []
    original = local._rewrite_into_aer_basis

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(local, "_rewrite_into_aer_basis", spy)
    return calls


def _replace_rewrite(monkeypatch, replacement):
    from polypus_python import local

    monkeypatch.setattr(local, "_rewrite_into_aer_basis", replacement)


def _stable(message):
    """``message`` without the order of the basis Qiskit lists in a
    ``BasisTranslator`` failure: it prints a ``set``, whose order differs from
    one call to the next, so two calls never give the same text."""
    return re.sub(r"target basis: \{[^}]*\}", "target basis: {...}", message)


def _assert_coherent(circuit, report):
    """Each backend the report accepts runs the circuit; each it rejects fails
    with (one of) the reasons given, or with the exception a reason quotes."""
    import polypus

    for backend, reasons in report.items():
        if not reasons:
            if backend != "qmio":
                assert sum(_run(circuit, backend).counts[0].values()) == 32
            continue
        with pytest.raises((polypus.PolypusError, ValueError, ImportError)) as info:
            _run(circuit, backend)
        raised = _stable(str(info.value))
        quoted = f"{type(info.value).__name__}: {info.value}"
        assert any(
            _stable(reason) in raised or quoted in reason for reason in reasons
        ), f"{backend}: {info.value!s} does not state any of {reasons}"


def test_is_exported():
    import polypus

    assert callable(polypus.backend_compatibility)
    assert "backend_compatibility" in dir(polypus)


class TestQasmStrings:
    def test_valid_circuit_is_accepted_by_both(self):
        report = _report(VALID)
        assert _local(report) == {"aer": [], "polypus": []}
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
        assert _local(report) == {"aer": [], "polypus": []}
        _assert_coherent(circuit, report)


class TestNativeCircuits:
    def test_bound_circuit_is_accepted_by_both(self):
        import polypus

        qc = polypus.Circuit(2).h(0).cx(0, 1).measure_all()
        report = _report(qc)
        assert _local(report) == {"aer": [], "polypus": []}
        _assert_coherent(qc, report)

    def test_free_parameters_are_reported_for_both(self):
        import polypus

        qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
        report = _report(qc)
        # The run raises this first on every backend.
        for reasons in report.values():
            assert "unbound parameters" in reasons[0]
        assert report["cunqa"] == report["aer"]
        for reasons in _local(report).values():
            assert len(reasons) == 1
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
            assert _local(report) == {"aer": [], "polypus": []}
            _assert_coherent(given, report)

    def test_ch_gives_equivalent_counts_on_both_backends(self):
        """x on the control makes ``ch`` a Hadamard on the target: both
        backends see the uniform split of the target and a fixed control."""
        import polypus

        qc = polypus.Circuit(2).x(0).ch(0, 1).measure_all()
        assert _local(_report(qc)) == {"aer": [], "polypus": []}
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
            assert _local(report) == {"aer": [], "polypus": []}
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

    @pytest.mark.parametrize("nesting", ["top", "for_loop", "if_test"])
    def test_box_is_rejected_by_aer(self, nesting):
        """``box`` is a control-flow operation Aer's target does not take and
        the lowering cannot remove (issue #285), at any depth. The reason is
        the run's message word for word, the same on every call (issue #307):
        the basis translator's own error listed the basis in a different order
        each time."""
        import polypus
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        if nesting == "top":
            with qc.box():
                qc.ch(0, 1)
        elif nesting == "for_loop":
            with qc.for_loop(range(2)):
                with qc.box():
                    qc.ch(0, 1)
        else:
            with qc.if_test((qc.clbits[0], 0)):
                with qc.box():
                    qc.ch(0, 1)
        qc.measure(1, 1)
        report = _report(qc)
        assert report["aer"] == [
            TRANSPILER_ERROR + "\"The control-flow construct 'box' is not "
            'supported by the backend."'
        ]
        assert _report(qc) == report
        with pytest.raises(polypus.BackendError) as info:
            _run(qc, "aer")
        assert str(info.value) == report["aer"][0]
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

        def forbidden(*_args, **_kwargs):
            raise AssertionError("a circuit in Aer's basis must not be rewritten")

        _replace_rewrite(monkeypatch, forbidden)
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

    def test_a_circuit_outside_the_basis_is_rewritten_as_the_run_does(
        self, monkeypatch
    ):
        """Once, with the caller's circuit and the basis of the default
        ``AerSimulator()``: the call ``_lower_to_aer_basis`` makes. The check
        and the run share that rewrite, so what it accepts or rejects is what
        the run does."""
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        calls = _spy_rewrite(monkeypatch)
        qc = QuantumCircuit(2)
        qc.ch(0, 1)
        qc.measure_all()
        assert _report(qc)["aer"] == []
        assert len(calls) == 1
        (circuit, basis), kwargs = calls[0]
        assert circuit is qc
        assert set(basis) == set(AerSimulator().target.operation_names)
        assert kwargs == {}

    @pytest.mark.parametrize("raised", [KeyboardInterrupt, RuntimeError])
    def test_an_error_not_raised_by_qiskit_propagates(self, monkeypatch, raised):
        """Only an exception raised by Qiskit is a reason; any other one, a
        cancellation included, reaches the caller as it is."""
        from qiskit import QuantumCircuit

        def failing(*_args, **_kwargs):
            raise raised("not a Qiskit error")

        _replace_rewrite(monkeypatch, failing)
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
        from qiskit import QuantumCircuit
        from qiskit.exceptions import QiskitError

        error = type(f"_Qiskit{base.__name__}", (QiskitError, base), {})

        def failing(*_args, **_kwargs):
            raise error("raised by a Qiskit class")

        _replace_rewrite(monkeypatch, failing)
        qc = QuantumCircuit(2)
        qc.ch(0, 1)
        qc.measure_all()
        with pytest.raises(error, match="raised by a Qiskit class"):
            _report(qc)

    def test_missing_aer_is_the_reason_and_nothing_is_rewritten(self, monkeypatch):
        import sys

        from qiskit import QuantumCircuit

        def forbidden(*_args, **_kwargs):
            raise AssertionError("nothing can be rewritten for a missing Aer")

        _replace_rewrite(monkeypatch, forbidden)
        monkeypatch.setitem(sys.modules, "qiskit_aer", None)
        qc = QuantumCircuit(1)
        qc.append(_opaque(), [0])
        qc.measure_all()
        report = _report(qc)
        assert len(report["aer"]) == 1
        assert "Qiskit Aer is not installed" in report["aer"][0]
        assert 'backend="aer" cannot run' in report["aer"][0]

    def test_wide_clifford_circuit_outside_the_basis_is_accepted(self):
        """The run lowers without the target's layout stage, so a circuit wider
        than the default ``AerSimulator()``'s target is rewritten and run on the
        stabilizer method; the check accepts it, as the run does (it used to
        reject it with ``CircuitTooWideForTarget``)."""
        from qiskit import QuantumCircuit

        n = 64
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.iswap(i, i + 1)
        qc.measure_all()
        report = _report(qc)
        assert report["aer"] == []
        _assert_coherent(qc, report)

    @pytest.mark.parametrize("gate", [None, "ch"])
    def test_free_parameters_are_reported_for_both(self, gate):
        """A free ``Parameter`` makes the run raise the ``ValueError`` a
        ``polypus.Circuit`` gets, before Aer sees the circuit; both entries
        quote it (issue #308). The native entry keeps its Qiskit guard."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(2)
        qc.rx(Parameter("t"), 0)
        if gate == "ch":
            qc.ch(0, 1)
        qc.measure_all()
        report = _report(qc)
        assert len(report["aer"]) == 1
        reason = report["aer"][0]
        assert "unbound parameters" in reason
        assert "executed directly: t." in reason
        assert report["polypus"][0] == reason
        assert "cannot execute a Qiskit QuantumCircuit" in report["polypus"][1]
        _assert_coherent(qc, report)

    def test_bound_parameters_are_accepted_by_aer(self):
        import math

        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        t = Parameter("t")
        qc = QuantumCircuit(2)
        qc.rx(t, 0)
        qc.ch(0, 1)
        qc.measure_all()
        bound = qc.assign_parameters({t: math.pi})
        report = _report(bound)
        assert report["aer"] == []
        _assert_coherent(bound, report)

    @pytest.mark.parametrize("infrastructure", ["local", "cunqa", "qmio"])
    def test_run_rejects_free_parameters_before_any_backend(
        self, monkeypatch, infrastructure
    ):
        """No infrastructure binds parameters, so the run rejects the circuit
        before connecting to one, naming every free parameter."""
        import polypus
        import polypus_python
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        def forbidden(*_args, **_kwargs):
            raise AssertionError(
                "a circuit with free parameters must not reach a backend"
            )

        monkeypatch.setattr(polypus_python, "run_qcs", forbidden)
        monkeypatch.setattr(polypus_python, "connect_to_infrastructure", forbidden)
        theta = ParameterVector("theta", 2)
        qc = QuantumCircuit(1)
        qc.rx(theta[0], 0)
        qc.ry(theta[1], 0)
        qc.measure_all()
        with pytest.raises(
            ValueError, match=r"unbound parameters .*: theta\[0\], theta\[1\]\."
        ):
            polypus.run_quantum_circuit(qc, shots=8, infrastructure=infrastructure)

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


CUNQA_OUTSIDE_ITS_RULES = {
    # CUNQA 3.0.0 takes `if_else` with condition value 1 and no `else` only.
    "condition_value": (
        H + "qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\n"
        "if(c==2) x q[1];\nmeasure q[1] -> c[1];\n",
        "Only 1 is accepted as condition",
    ),
}


def _if_else():
    """A Qiskit circuit with an ``else`` body, which CUNQA 3.0.0 rejects."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    q, c = QuantumRegister(2, "q"), ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((c[0], 1)) as else_:
        qc.x(1)
    with else_:
        qc.h(1)
    qc.measure(1, 1)
    return qc


class TestCunqa:
    def test_valid_and_dynamic_circuits_are_accepted(self):
        for circuit in (VALID, IF, RESET):
            report = _report(circuit)
            assert report["cunqa"] == []
            _assert_coherent(circuit, report)

    def test_circuit_without_measurements_is_accepted(self):
        """The seal reads it out in full (#253), so CUNQA runs it."""
        import polypus

        for circuit in (
            polypus.Circuit(2).x(0),
            H + "qreg q[2];\ncreg c[3];\nx q[0];\n",
        ):
            report = _report(circuit)
            assert report["cunqa"] == []
            assert _run(circuit, "cunqa").merged_counts == {"01": 32}

    @pytest.mark.parametrize("case", sorted(CUNQA_OUTSIDE_ITS_RULES))
    def test_what_its_converter_rejects_is_the_reason(self, case):
        circuit, fragment = CUNQA_OUTSIDE_ITS_RULES[case]
        report = _report(circuit)
        assert report["aer"] == []
        assert len(report["cunqa"]) == 1 and fragment in report["cunqa"][0]
        _assert_coherent(circuit, report)

    def test_an_else_body_is_rejected(self):
        qc = _if_else()
        report = _report(qc)
        assert report["aer"] == []
        assert report["cunqa"] == [
            "if_else instruction with 'else' case is not supported for the current "
            "version."
        ]
        _assert_coherent(qc, report)

    def test_the_converter_sees_the_prepared_circuit(self, cunqa_sdk, monkeypatch):
        """Lowered and read out in full, as the run sends it, and a copy."""
        import polypus

        seen = []
        monkeypatch.setattr(cunqa_sdk, "to_ir", seen.append)
        qc = polypus.Circuit(3).ch(0, 1).ccx(0, 1, 2)
        assert _report(qc)["cunqa"] == []
        (converted,) = seen
        assert "ch" not in converted.count_ops()
        assert converted.count_ops()["measure"] == 3

    def test_an_exception_of_the_converter_is_the_reason(self, cunqa_sdk, monkeypatch):
        def refuse(circuit):
            raise ValueError("this vQPU converts nothing")

        monkeypatch.setattr(cunqa_sdk, "to_ir", refuse)
        report = _report(VALID)
        assert report["cunqa"] == ["this vQPU converts nothing"]
        _assert_coherent(VALID, report)

    def test_a_qasm_parse_failure_is_the_reason(self):
        report = _report(INVALID)
        assert len(report["cunqa"]) == 1 and "QASM2ParseError" in report["cunqa"][0]
        _assert_coherent(INVALID, report)

    def test_missing_cunqa_is_the_reason(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "cunqa.qjob", None)
        monkeypatch.delitem(sys.modules, "polypus_python.cunqa", raising=False)
        report = _report(VALID)
        assert _local(report) == {"aer": [], "polypus": []}
        (reason,) = report["cunqa"]
        assert reason.startswith("CUNQA is not available (ModuleNotFoundError: ")
        assert reason.endswith('so infrastructure="cunqa" cannot run')
        _assert_coherent(VALID, report)


class TestQmio:
    @pytest.fixture(autouse=True)
    def _default_build(self):
        if _report(VALID)["qmio"] == []:
            pytest.skip("polypus was built with the qmio feature")

    NOT_COMPILED = (
        "the 'qmio' infrastructure requires compiling polypus with --features qmio"
    )

    def test_a_build_without_the_feature_rejects_every_circuit(self):
        import polypus

        for circuit in (VALID, polypus.Circuit(2).h(0).measure_all()):
            report = _report(circuit)
            assert report["qmio"] == [self.NOT_COMPILED]
            _assert_coherent(circuit, report)

    def test_a_quantum_circuit_is_rejected(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()
        report = _report(qc)
        assert report["qmio"][0] == self.NOT_COMPILED
        assert "cannot serialize a Qiskit QuantumCircuit" in report["qmio"][1]
        with pytest.raises(ValueError) as info:
            _run(qc, "qmio")
        assert str(info.value) == report["qmio"][1]

    @pytest.mark.parametrize("case", sorted(AER_ONLY))
    def test_a_dynamic_program_is_rejected(self, case):
        circuit, fragment = AER_ONLY[case]
        report = _report(circuit)
        assert report["qmio"][0] == self.NOT_COMPILED
        (reason,) = report["qmio"][1:]
        assert reason.startswith("could not build a circuit for QMIO: ")
        assert fragment in reason


class TestResources:
    WIDE = 22  # a 64 MiB statevector

    def _wide(self, as_qasm):
        import polypus

        qc = polypus.Circuit(self.WIDE).h(0).measure_all()
        return qc.to_qasm2() if as_qasm else qc

    def test_resources_are_not_checked_by_default(self, monkeypatch):
        monkeypatch.setenv("POLYPUS_MEM_BUDGET", "1M")
        assert _report(self._wide(False))["polypus"] == []

    @pytest.mark.parametrize("as_qasm", [False, True], ids=["circuit", "qasm"])
    def test_a_statevector_over_the_memory_budget_is_rejected(
        self, monkeypatch, as_qasm
    ):
        import polypus

        monkeypatch.setenv("POLYPUS_MEM_BUDGET", "1M")
        circuit = self._wide(as_qasm)
        report = _report(circuit, resources=True)
        (reason,) = report["polypus"]
        assert reason.startswith(f"not enough memory for a {self.WIDE}-qubit")
        with pytest.raises(polypus.InsufficientMemoryError) as info:
            _run(circuit, "polypus")
        assert str(info.value) == reason
        # Only the native entry checks resources.
        assert report == {**_report(circuit), "polypus": report["polypus"]}

    def test_a_circuit_over_the_qubit_ceiling_is_rejected(self):
        import polypus

        qc = polypus.Circuit(31).measure_all()
        assert _report(qc)["polypus"] == []
        (reason,) = _report(qc, resources=True)["polypus"]
        assert "supports at most 30" in reason
        with pytest.raises(polypus.NativeCircuitError) as info:
            _run(qc, "polypus")
        assert str(info.value) == reason

    def test_a_structural_rejection_is_not_followed_by_a_resource_one(
        self, monkeypatch
    ):
        monkeypatch.setenv("POLYPUS_MEM_BUDGET", "1")
        assert _report(RESET, resources=True)["polypus"] == _report(RESET)["polypus"]

    def test_a_circuit_that_fits_is_accepted(self):
        report = _report(VALID, resources=True)
        assert report == _report(VALID)
        _assert_coherent(VALID, report)

    def test_resources_is_keyword_only(self):
        import polypus

        with pytest.raises(TypeError):
            polypus.backend_compatibility(VALID, True)


def test_check_runs_nothing(monkeypatch, cunqa_sdk):
    """Pure: no seam call, so a broken ``run_qcs`` cannot affect it, no
    simulation, not even for a Qiskit circuit the check rewrites, and no CUNQA
    call but its converter."""
    import polypus_python
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backend_compatibility must not run the circuit")

    monkeypatch.setattr(polypus_python, "run_qcs", forbidden)
    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", forbidden)
    monkeypatch.setattr(AerSimulator, "run", forbidden)
    assert _local(_report(VALID, resources=True)) == {"aer": [], "polypus": []}
    qc = QuantumCircuit(2)
    qc.ch(0, 1)
    qc.measure_all()
    assert _report(qc)["aer"] == []
    assert cunqa_sdk.calls == [] and cunqa_sdk.runs == []


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
