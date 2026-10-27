"""
Angle expressions built in Python: ``polypus.ParamExpr``, the functions
``polypus.sin`` ... ``polypus.sqrt`` and the constants ``pi``, ``tau`` and
``euler`` (issue #239).

The expression IR and its rules (contract C-2) are tested in
``crates/polypus-circuit/tests/expressions.rs``; these tests cover the Python
surface and that a built expression behaves like an imported one.
"""

import math

import numpy as np
import polypus
import pytest
from polypus import Param

HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
SHOTS = 256
SEED = 7


def _angle(expr, params):
    """The angle `expr` binds to, read back from the bound OpenQASM 2.0."""
    text = polypus.Circuit(1).rx(0, expr).to_qasm2(params)
    return float(text.rsplit("rx(", 1)[1].split(")", 1)[0])


def _qaoa_layer(gamma, beta):
    return (
        polypus.Circuit(2)
        .h(0)
        .h(1)
        .rzz(0, 1, 2 * gamma)
        .rx(0, 2 * beta)
        .rx(1, 2 * beta)
        .measure_all()
    )


# ── Building ────────────────────────────────────────────────────────────────


def test_operators_build_expressions():
    a, b = Param(0), Param(1)
    assert isinstance(2 * a, polypus.ParamExpr)
    assert repr(2 * a) == "2.0*Param(0)"
    assert repr(a * 2 + b) == "Param(0)*2.0 + Param(1)"
    assert repr(a - (b - 1)) == "Param(0) - (Param(1) - 1.0)"
    assert repr(1 / a) == "1.0/Param(0)"
    assert repr(a**2) == "Param(0)**2.0"
    assert repr(2**a) == "2.0**Param(0)"
    assert repr(-a) == "-Param(0)"
    assert repr(-(a + b)) == "-(Param(0) + Param(1))"
    assert repr(a - 1.5) == "Param(0) - 1.5"
    assert repr(a + -1.5) == "Param(0) + -1.5"


def test_functions_and_constants():
    a = Param(3)
    assert repr(polypus.sin(a)) == "sin(Param(3))"
    assert repr(polypus.log(a)) == "log(Param(3))"
    assert repr(polypus.sqrt(2 * a)) == "sqrt(2.0*Param(3))"
    assert repr(polypus.arctan(polypus.tau - a)) == "arctan(tau - Param(3))"
    assert repr(polypus.pi / 2) == "pi/2.0"
    assert repr(polypus.euler) == "euler"
    assert repr(polypus.cos(0.5)) == "cos(0.5)"


def test_equality_is_structural():
    a, b = Param(0), Param(1)
    assert 2 * a == 2 * Param(0)
    assert a + b != b + a
    assert 2 * a != 2 * b
    assert (2 * a == 2) is False


@pytest.mark.parametrize("bad", ["x", None, [1.0], object()])
def test_an_unsupported_operand_raises_type_error(bad):
    with pytest.raises(TypeError):
        Param(0) + bad
    with pytest.raises(TypeError):
        bad * (2 * Param(0))
    with pytest.raises(TypeError):
        polypus.sin(bad)


def test_pow_with_modulus_raises_type_error():
    with pytest.raises(TypeError, match="modulus"):
        pow(Param(0), 2, 3)


def test_a_builder_rejects_other_angle_types():
    with pytest.raises(TypeError, match="polypus.ParamExpr"):
        polypus.Circuit(1).rx(0, "theta")


# ── Storing and binding ─────────────────────────────────────────────────────


def test_every_angle_argument_accepts_an_expression():
    t = 2 * Param(0)
    qc = (
        polypus.Circuit(2)
        .rx(0, t)
        .ry(0, t)
        .rz(0, t)
        .u(0, t, t, t)
        .rzz(0, 1, t)
        .rxx(0, 1, t)
        .cp(0, 1, t)
        .crx(0, 1, t)
        .cry(0, 1, t)
        .crz(0, 1, t)
        .cu1(0, 1, t)
        .cu3(0, 1, t, t, t)
        .cu(0, 1, t, t, t, t)
        .p(0, t)
        .u1(0, t)
        .u2(0, t, t)
        .u0(0, t)
    )
    assert qc.num_params == 1
    bound = qc.to_qasm2([0.25])
    assert "0.5" in bound and "theta" not in bound


def test_binding_evaluates_as_written():
    a, b = Param(0), Param(1)
    values = [0.3, 1.7]
    cases = [
        (2 * a, 2 * 0.3),
        (a * 2 + b, 0.3 * 2 + 1.7),
        (a - (b - 1), 0.3 - (1.7 - 1)),
        (a / b, 0.3 / 1.7),
        (a**b, 0.3**1.7),
        (-a, -0.3),
        (polypus.sin(a) * polypus.exp(b), math.sin(0.3) * math.exp(1.7)),
        (polypus.log(b) + polypus.sqrt(a), math.log(1.7) + math.sqrt(0.3)),
        (polypus.arcsin(a) - polypus.arccos(a), math.asin(0.3) - math.acos(0.3)),
        (polypus.tan(a) + polypus.arctan(b), math.tan(0.3) + math.atan(1.7)),
        (polypus.tau * a - polypus.euler, math.tau * 0.3 - math.e),
    ]
    for expr, expected in cases:
        assert _angle(expr + 0 * b, values) == expected, repr(expr)


def test_a_constant_expression_is_a_fixed_angle():
    qc = polypus.Circuit(1).rx(0, polypus.pi / 2)
    assert qc.num_params == 0
    assert qc.to_qasm2().endswith(f"rx({math.pi / 2!r}) q[0];\n")


def test_parameters_count_once_the_gate_is_pushed():
    expr = Param(0) * Param(4)
    assert polypus.Circuit(1).num_params == 0
    assert polypus.Circuit(1).rx(0, expr).num_params == 5


# ── Contract C-2 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_number_is_rejected_when_stored(bad):
    expr = Param(0) * bad
    assert "nan" in repr(expr) or "inf" in repr(expr)
    with pytest.raises(ValueError):
        polypus.Circuit(1).rx(0, expr)


def test_a_constant_division_by_zero_is_rejected_when_stored():
    with pytest.raises(ValueError, match="division by zero"):
        polypus.Circuit(1).rx(0, polypus.pi / 0)


def test_division_by_zero_and_non_finite_results_are_binding_errors():
    qc = polypus.Circuit(1).rx(0, 1 / Param(0)).measure_all()
    with pytest.raises(ValueError, match="division by zero"):
        qc.to_qasm2([0.0])
    qc = polypus.Circuit(1).rx(0, polypus.log(Param(0))).measure_all()
    with pytest.raises(ValueError):
        qc.to_qasm2([-1.0])


def test_too_deep_an_expression_is_rejected():
    expr = Param(0)
    for _ in range(80):
        expr = -expr
    with pytest.raises(ValueError, match="nest"):
        polypus.Circuit(1).rx(0, expr)


# ── OpenQASM 3 round trip ───────────────────────────────────────────────────


def test_qaoa_layer_round_trips_through_openqasm3():
    qc = _qaoa_layer(Param(0), Param(1))
    text = qc.to_qasm3()
    assert "rzz(2.0*theta_0) q[0], q[1];" in text
    assert "rx(2.0*theta_1) q[0];" in text
    again = polypus.Circuit.from_qasm3(text)
    assert again.to_qasm3() == text
    assert again.num_params == 2
    # `rzz` comes back as the gate the export declares, so compare states.
    params = [0.4, 1.1]
    np.testing.assert_allclose(
        polypus.statevector(again, params), polypus.statevector(qc, params), atol=1e-12
    )


def test_an_imported_expression_equals_the_built_one():
    built = polypus.Circuit(1).rx(0, polypus.sin(Param(0)) ** 2 - polypus.pi / 4)
    imported = polypus.Circuit.from_qasm3(
        HEADER
        + "input float[64] theta_0;\nqubit[1] q;\nrx(sin(theta_0)**2.0 - pi/4.0) q[0];\n"
    )
    assert built.to_qasm3() == imported.to_qasm3()


# ── Running ─────────────────────────────────────────────────────────────────


def test_statevector_matches_the_rescaled_circuit():
    gamma, beta = 0.4, 1.1
    built = _qaoa_layer(Param(0), Param(1))
    rescaled = (
        polypus.Circuit(2)
        .h(0)
        .h(1)
        .rzz(0, 1, Param(0))
        .rx(0, Param(1))
        .rx(1, Param(1))
        .measure_all()
    )
    np.testing.assert_array_equal(
        polypus.statevector(built, [gamma, beta]),
        polypus.statevector(rescaled, [2 * gamma, 2 * beta]),
    )


@pytest.mark.parametrize("backend", ["polypus", "aer"])
def test_qaoa_layer_runs_like_the_fixed_circuit(backend):
    pytest.importorskip("qiskit_aer")
    gamma, beta = 0.4, 1.1
    built = _qaoa_layer(Param(0), Param(1))
    fixed = _qaoa_layer(gamma, beta)

    def run(qc, params):
        bound = polypus.Circuit.from_qasm2(qc.to_qasm2(params))
        return polypus.run_quantum_circuit(
            bound, shots=SHOTS, infrastructure="local", backend=backend, seed=SEED
        ).merged_counts

    assert run(built, [gamma, beta]) == run(fixed, [])


def test_native_and_aer_agree_on_a_deterministic_layer():
    pytest.importorskip("qiskit_aer")
    # rx(2*beta) at beta = pi/2 is rx(pi): every shot reads 1.
    qc = polypus.Circuit(2).rx(0, 2 * Param(0)).rx(1, 2 * Param(0)).measure_all()
    bound = polypus.Circuit.from_qasm2(qc.to_qasm2([math.pi / 2]))
    counts = {
        backend: polypus.run_quantum_circuit(
            bound, shots=SHOTS, infrastructure="local", backend=backend, seed=SEED
        ).merged_counts
        for backend in ("polypus", "aer")
    }
    assert counts["polypus"] == counts["aer"] == {"11": SHOTS}


@pytest.mark.parametrize("backend", ["polypus", "aer"])
def test_training_binds_the_expression(backend):
    pytest.importorskip("qiskit_aer")
    built = _qaoa_layer(Param(0), Param(1))
    imported = polypus.Circuit.from_qasm3(built.to_qasm3())

    def train(qc):
        return polypus.train(
            qc,
            polypus.DE(generations=2, population_size=4, seed=SEED),
            shots=SHOTS,
            n_qpus=1,
            dimensions=2,
            expectation_function=lambda bits: float(bits.count("1")),
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id=f"param_expr_{backend}",
            backend=backend,
            seed=SEED,
        )

    first, second = train(built), train(imported)
    assert first.best_params == second.best_params
    assert first.fitness_history == second.fitness_history
