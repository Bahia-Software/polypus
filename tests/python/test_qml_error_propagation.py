"""
QML evaluation-path error classification (issue #81, Change 2).

``EvaluationError::Runtime`` (Tokio runtime construction, or a ``spawn_blocking``
worker panic surfaced as a ``JoinError``) is a Rust-side infrastructure failure
and now surfaces as ``polypus.EvaluationError`` rather than a bare
``RuntimeError`` — pinned by the Rust unit test in
``crates/polypus/src/evaluation/error.rs`` (forcing those OS-level conditions
deterministically from Python is neither viable nor portable; see the PR).

This test guards the *other* side of that change: a genuine Python exception
raised by the user's ``expectation_function`` callback must still propagate
**verbatim** (as itself), never be reclassified as ``polypus.EvaluationError``.
It runs on the ``local`` path with a mocked backend, so no SLURM/hardware.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]


def _patch_deterministic_backend(monkeypatch):
    import polypus_python

    def fake_run_qcs(infrastructure, **kwargs):
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)


def test_qml_callback_exception_propagates_verbatim(monkeypatch):
    import numpy as np
    import polypus
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    _patch_deterministic_backend(monkeypatch)

    def exploding_expectation(_bitstring):
        raise ValueError("user callback blew up")

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    x_train = np.zeros((2, 2))

    # The user's callback raises ValueError; contract C-1 / ENGINEERING.md §9
    # require it to reach the caller as that same ValueError, not wrapped in
    # polypus.EvaluationError.
    with pytest.raises(ValueError, match="user callback blew up") as excinfo:
        polypus.qml.train(
            feature_map,
            ansatz,
            x_train,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=64,
            n_qpus=1,
            dimensions=len(ansatz.parameters),
            expectation_function=exploding_expectation,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="qml_verbatim_error",
        )
    assert not isinstance(excinfo.value, polypus.EvaluationError), (
        "a genuine Python callback exception must not be reclassified as "
        "polypus.EvaluationError"
    )


# ── Qiskit failures while preparing/binding circuits (issue #218) ────────────
#
# A Qiskit exception raised while composing or binding the user's Qiskit
# circuits — before or inside the oracle, never on a backend — reaches the
# caller as ``polypus.EvaluationError`` with the Qiskit original as
# ``__cause__``. A user callback's exception, and a ``ValueError``/``TypeError``
# Qiskit raises, keep their class. All cases are real calls, no monkeypatch.


def _picky_ansatz(num_qubits=2, error=None, name="w"):
    """An ansatz whose user-defined gate refuses every bound value through
    Qiskit's own validation (``Gate.validate_parameter``), so the oracle's or
    ``qml.predict``'s ``assign_parameters`` raises a genuine ``CircuitError``
    (or ``error``, when given)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, Parameter, ParameterExpression
    from qiskit.circuit.exceptions import CircuitError

    refusal = error or CircuitError

    class Picky(Gate):
        def __init__(self, theta):
            super().__init__("picky", 1, [theta])

        def validate_parameter(self, parameter):
            if not isinstance(parameter, ParameterExpression):
                raise refusal(f"picky refuses {parameter}")
            return parameter

    qc = QuantumCircuit(num_qubits)
    qc.append(Picky(Parameter(name)), [0])
    return qc


def _feature_map(num_qubits=2):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.ry(x[q], q)
    return qc


def _wide_ansatz():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    t = ParameterVector("t", 3)
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.ry(t[q], q)
    return qc


def _qml_train(feature_map, ansatz, x_train, expectation=lambda _b: 0.0):
    import polypus

    return polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=2, population_size=4, tolerance=0.5),
        shots=16,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=expectation,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_qiskit_error",
        seed=1,
    )


def _qml_predict(feature_map, ansatz, x, params):
    import polypus

    return polypus.qml.predict(
        feature_map, ansatz, x, params, shots=16, infrastructure="local", seed=1
    )


def _assert_wraps_circuit_error(info, fragment=None):
    """The wrapper is exactly ``polypus.EvaluationError``, names the Qiskit
    class and carries its cause's own message. Qiskit's wording varies across
    versions, so only a ``fragment`` the test itself defines is matched."""
    import polypus
    from qiskit.circuit.exceptions import CircuitError

    exc = info.value
    assert type(exc) is polypus.EvaluationError
    assert isinstance(exc, polypus.PolypusError)
    assert "qiskit.circuit.exceptions.CircuitError" in str(exc)
    assert isinstance(exc.__cause__, CircuitError)
    assert str(exc.__cause__) in str(exc)
    if fragment is not None:
        assert fragment in str(exc)


def _assert_kept_as(info, cls):
    """C-1's typed modes: Qiskit's own exception reaches the caller as itself,
    exactly ``cls``, not replaced by a ``polypus`` wrapper."""
    import polypus

    assert type(info.value) is cls
    assert not isinstance(info.value, polypus.PolypusError)


_QISKIT_FAILURES = {
    # compose_qml_template: the ansatz is wider than the feature map.
    "compose": (_wide_ansatz, [[0.0, 0.0]]),
    # bind_feature_rows: Qiskit rejects a complex feature value.
    "feature_row": (lambda: _picky_ansatz(), [[1j, 0.0]]),
}


@pytest.mark.parametrize("case", sorted(_QISKIT_FAILURES))
def test_qml_train_qiskit_preparation_error_is_evaluation_error(case):
    import polypus

    ansatz, rows = _QISKIT_FAILURES[case]
    with pytest.raises(polypus.EvaluationError) as info:
        _qml_train(_feature_map(), ansatz(), rows)
    _assert_wraps_circuit_error(info)


@pytest.mark.parametrize("case", sorted(_QISKIT_FAILURES))
def test_qml_predict_qiskit_preparation_error_is_evaluation_error(case):
    import polypus

    ansatz, rows = _QISKIT_FAILURES[case]
    ansatz = ansatz()
    with pytest.raises(polypus.EvaluationError) as info:
        _qml_predict(_feature_map(), ansatz, rows, [0.1] * len(ansatz.parameters))
    _assert_wraps_circuit_error(info)


def test_qml_predict_weight_binding_error_is_evaluation_error():
    """``qml.predict`` binding the trained weights to each row's circuit."""
    import polypus

    with pytest.raises(polypus.EvaluationError) as info:
        _qml_predict(_feature_map(), _picky_ansatz(), [[0.0, 0.0]], [0.3])
    _assert_wraps_circuit_error(info, "picky refuses")


def test_qml_train_oracle_binding_error_is_evaluation_error():
    """The QML oracle's ``assign_parameters`` (``EvaluationError::Qiskit``)."""
    import polypus

    with pytest.raises(polypus.EvaluationError) as info:
        _qml_train(_feature_map(), _picky_ansatz(), [[0.0, 0.0]])
    _assert_wraps_circuit_error(info, "picky refuses")


def test_vqc_train_oracle_binding_error_is_evaluation_error():
    """The VQC oracle binding a Qiskit template's parameters."""
    import polypus

    template = _picky_ansatz(1)
    template.measure_all()
    with pytest.raises(polypus.EvaluationError) as info:
        polypus.train(
            template,
            polypus.DE(generations=2, population_size=4, tolerance=0.5),
            shots=16,
            n_qpus=1,
            dimensions=1,
            expectation_function=lambda _b: 0.0,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="vqc_qiskit_error",
            seed=1,
        )
    _assert_wraps_circuit_error(info, "picky refuses")


def test_qiskit_class_raised_by_a_user_callback_stays_verbatim():
    """The guard: a callback may raise a Qiskit class itself; it is the user's
    exception, so it is not reclassified."""
    import polypus
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.exceptions import QiskitError

    def exploding(_bitstring):
        raise QiskitError("raised by the user's callback")

    ansatz = QuantumCircuit(2)
    ansatz.ry(Parameter("w"), 0)
    with pytest.raises(QiskitError, match="user's callback") as info:
        _qml_train(_feature_map(), ansatz, [[0.0, 0.0]], expectation=exploding)
    assert not isinstance(info.value, polypus.PolypusError)


def test_qiskit_value_and_type_errors_keep_their_class():
    """A ``ValueError`` or ``TypeError`` raised from Qiskit's binding call is one
    of C-1's typed modes: not wrapped. The ``ValueError`` is Qiskit's own (a
    parameter shared by the feature map and the ansatz leaves fewer free
    parameters than weights). The ``TypeError`` comes from a feature-map gate's
    ``validate_parameter`` while ``bind_feature_rows`` binds a row: Qiskit's own
    reaction to a non-numeric value differs across versions (``TypeError`` in
    2.5, ``RuntimeError``/``SympifyError`` in 2.0), so no input gives a Qiskit
    ``TypeError`` on every supported version."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    feature_map = _feature_map()
    shared = QuantumCircuit(2)
    shared.ry(feature_map.parameters[0], 0)
    shared.ry(Parameter("w"), 1)
    with pytest.raises(ValueError) as info:
        _qml_predict(feature_map, shared, [[0.0, 0.0]], [0.1, 0.2])
    _assert_kept_as(info, ValueError)

    strict_feature_map = _picky_ansatz(error=TypeError, name="x")
    with pytest.raises(TypeError) as info:
        _qml_train(strict_feature_map, QuantumCircuit(2), [[0.5]])
    _assert_kept_as(info, TypeError)
