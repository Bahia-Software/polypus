"""
The Python surface of the OpenQASM 3 profile: ``Circuit.from_qasm3``,
``Circuit.to_qasm3(params=None)`` and ``Circuit.param_names``.

The profile itself is tested in ``crates/polypus-circuit/tests/qasm3.rs``, its
matrices in ``crates/polypus-sim/tests/qasm3_semantics.rs``, and its
interoperability with Qiskit in ``test_qasm3_interop.py``.
"""

import importlib.util
from pathlib import Path

import polypus
import pytest

HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
REPO = Path(__file__).resolve().parents[2]


def test_inputs_are_the_parameters_under_their_names():
    qc = polypus.Circuit.from_qasm3(
        HEADER
        + "input float[64] beta;\ninput float[64] θ;\nqubit[1] q;\nrx(2*θ + beta) q[0];\n"
    )
    assert qc.num_params == 2
    assert qc.param_names == ["beta", "θ"]
    assert qc.to_qasm2([0.1, 0.2]).endswith("rx(0.500000000000) q[0];\n")


def test_default_names_cover_every_index():
    qc = polypus.Circuit(1).rx(0, polypus.Param(2))
    assert qc.param_names == ["theta_0", "theta_1", "theta_2"]


def test_to_qasm3_writes_inputs_or_binds_first():
    qc = polypus.Circuit(1).rx(0, polypus.Param(0))
    assert qc.to_qasm3() == (
        HEADER + "input float[64] theta_0;\nqubit[1] q;\nrx(theta_0) q[0];\n"
    )
    assert qc.to_qasm3([0.25]) == HEADER + "qubit[1] q;\nrx(0.25) q[0];\n"
    assert qc.to_qasm3(params=[0.25]) == qc.to_qasm3([0.25])
    with pytest.raises(ValueError, match="wrong number of parameter values"):
        qc.to_qasm3([])


def test_the_export_is_a_fixed_point():
    text = (
        polypus.Circuit(2)
        .h(0)
        .rzz(0, 1, polypus.Param(0))
        .u(1, 0.5, polypus.Param(1), -0.5)
        .measure_all()
        .to_qasm3()
    )
    again = polypus.Circuit.from_qasm3(text)
    assert again.to_qasm3() == text
    assert again.param_names == ["theta_0", "theta_1"]


def test_a_construct_outside_the_profile_raises_with_its_line():
    with pytest.raises(ValueError, match=r"line 4: 'reset' is not supported"):
        polypus.Circuit.from_qasm3(HEADER + "qubit q;\nreset q;\n")
    with pytest.raises(ValueError, match=r"line 4: .*integer division"):
        polypus.Circuit.from_qasm3(HEADER + "qubit q;\nrx(1/2) q;\n")


def test_a_declaration_openqasm2_cannot_express_raises_there_only():
    qc = polypus.Circuit.from_qasm3(
        HEADER + "gate g(t) a { rx(arcsin(t)) a; }\nqubit[1] q;\ng(0.5) q[0];\n"
    )
    with pytest.raises(
        ValueError, match="declared gate 'g' cannot be written in OpenQASM 2.0"
    ):
        qc.to_qasm2()
    assert "rx(arcsin(t_)) a;" in qc.to_qasm3()


def test_the_phase_convention_is_documented():
    def doc(method):
        return " ".join(method.__doc__.split())

    for method in (polypus.Circuit.from_qasm3, polypus.Circuit.to_qasm3):
        assert "OpenQASM 3 profile with Qiskit phase conventions" in doc(method)
    assert "e^{-iθ/2}" in doc(polypus.Circuit.from_qasm3)
    assert "e^{i(φ+λ)/2}" in doc(polypus.Circuit.from_qasm3)


def test_errors_are_classified_like_openqasm_2_errors():
    """A construct both dialects reject fails with the OpenQASM 2.0 importer's
    wording, which ``benchmarks/qasm_coverage.py`` classifies."""
    spec = importlib.util.spec_from_file_location(
        "qasm_coverage", REPO / "benchmarks" / "qasm_coverage.py"
    )
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    cases = {
        "qubit q;\nreset q;\n": "`reset`",
        "qubit q;\nbit c;\nif (c) x q;\n": "`if` (classical control)",
        "qubit q;\nfoo q;\n": "gate `foo`",
        "qubit q;\nopaque g a;\n": "`opaque` declaration",
        "qubit q;\nbit c;\nc = measure q;\nx q;\n": "gate after measurement (C-4)",
        "x r;\n": "undeclared register",
        "qubit[2000000] q;\n": "register size limit",
        "qubit q;\nrx("
        + "(" * 65
        + "0"
        + ")" * 65
        + ") q;\n": "expression depth limit",
    }
    for body, construct in cases.items():
        with pytest.raises(ValueError) as error:
            polypus.Circuit.from_qasm3(HEADER + body)
        assert coverage.classify_polypus_error(str(error.value)) == construct, body


@pytest.mark.integration
def test_running_a_circuit_openqasm2_cannot_express_raises_instead_of_panicking():
    """The Aer backend receives native circuits as OpenQASM 2.0: a gate declared
    in OpenQASM 3 whose body uses a function OpenQASM 2.0 lacks is a typed
    error, and the native backend, which expands the gate itself, runs it."""
    qc = polypus.Circuit.from_qasm3(
        HEADER
        + "gate g(t) a { rx(arcsin(t)) a; }\nqubit[1] q;\nbit[1] c;\n"
        + "g(0.5) q[0];\nc = measure q;\n"
    )
    with pytest.raises(polypus.NativeCircuitError, match="arcsin"):
        polypus.run_quantum_circuit(qc, shots=10, infrastructure="local")
    result = polypus.run_quantum_circuit(
        qc, shots=10, infrastructure="local", backend="polypus"
    )
    assert sum(result.counts[0].values()) == 10
