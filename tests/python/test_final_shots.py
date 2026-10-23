"""
``final_shots`` in ``polypus.train`` and ``polypus.qml.train`` (contract C-5).

A value that is not an integer ``>= 1`` raises ``ValueError`` before anything
runs: with ``infrastructure="cunqa"``, before ``connect_to_infrastructure``
(and so ``qraise``) is reached, which a recorder checks without SLURM or
CUNQA.

With a valid value, once the optimizer has stopped, its best candidates are
re-evaluated in one more batch with ``final_shots`` shots per circuit:
``best_params`` and, for DE, PSO and CMA-ES, the four best other distinct
candidates they scored. The batch tests run the real training loops on Aer
with ``polypus_python.run_qcs`` replaced by a fake that scores each circuit
by its parameters, records every circuit with its shots and value, and so
checks which candidates the final batch holds, with which shots, which one
wins, and that every batch before it is the one sent without the option. The
native tests check the same seeded run with and without the option on the
real shot sampling, and that the final value was measured with
``final_shots``.
"""

import math
import re

import numpy as np
import pytest

DIMENSIONS = 2
SHOTS = 64
FINAL_SHOTS = 4096
OPTIMIZERS = ["rand1bin", "shade", "lshade", "pso", "qng", "cmaes", "adam", "spsa"]
# Re-evaluate their five best candidates; the others only best_params.
POPULATION = {"rand1bin", "shade", "lshade", "pso", "cmaes"}

_RY = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _method(name):
    """A small configuration of the optimizer ``name`` (a DE strategy name,
    ``"pso"``, ``"qng"``, ``"cmaes"``, ``"adam"`` or ``"spsa"``)."""
    import polypus

    if name == "pso":
        return polypus.PSO(generations=3, population_size=6)
    if name == "qng":
        return polypus.QNG(lambda theta, a: 1.0, max_iters=2)
    if name == "cmaes":
        return polypus.CMAES(generations=3)
    if name == "adam":
        return polypus.Adam(max_iters=2)
    if name == "spsa":
        return polypus.SPSA(max_iters=3)
    return polypus.DE(generations=3, population_size=6, strategy=name)


def _circuit():
    import polypus

    qc = polypus.Circuit(DIMENSIONS)
    for q in range(DIMENSIONS):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return qc


def _train(method, *, infrastructure="local", **kwargs):
    import polypus

    return polypus.train(
        _circuit(),
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="final_shots",
        backend="aer",
        seed=11,
        **kwargs,
    )


def _qml_model():
    """A one-qubit model: ``rx(x)`` encodes the sample, then the ansatz weights
    are ``ry(theta[0])`` and ``rz(theta[1])``."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    x = ParameterVector("x", 1)
    theta = ParameterVector("theta", DIMENSIONS)
    feature_map = QuantumCircuit(1)
    feature_map.rx(x[0], 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta[0], 0)
    ansatz.rz(theta[1], 0)
    return feature_map, ansatz


def _qml_train(method, *, infrastructure="local", **kwargs):
    import polypus

    feature_map, ansatz = _qml_model()
    return polypus.qml.train(
        feature_map,
        ansatz,
        # One sample, so each candidate is one circuit.
        [[0.3]],
        method,
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="final_shots_qml",
        seed=11,
        **kwargs,
    )


def _vqc_candidate(qasm):
    """The candidate a recorded ``train`` circuit (OpenQASM, whose angles are
    exported at full precision) was bound to."""
    angles = {int(q): float(a) for a, q in _RY.findall(qasm)}
    return [angles[q] for q in range(DIMENSIONS)]


def _qml_candidate(circuit):
    """The candidate a recorded ``qml.train`` circuit (Qiskit) was bound to:
    its ``ry`` and ``rz`` angles."""
    angles = {
        inst.operation.name: float(inst.operation.params[0])
        for inst in circuit.data
        if inst.operation.name in ("ry", "rz")
    }
    return [angles["ry"], angles["rz"]]


# The runner, how to read a recorded circuit's candidate, and its width.
ENTRY_POINTS = {
    "train": (_train, _vqc_candidate, DIMENSIONS),
    "qml.train": (_qml_train, _qml_candidate, 1),
}


def _probability(candidate):
    """The share of shots the fake measures all ones for ``candidate``."""
    a, b = candidate
    return (1.0 + math.sin(3.0 * a + 1.3) * math.cos(2.0 * b - 0.4)) / 2.0


def _record(monkeypatch, candidate, width):
    """Replace the local seam with a fake that measures all ones on
    ``round(shots · p)`` shots, ``p`` a fixed function of the circuit's
    candidate, and records ``(candidate, shots, value)`` for every circuit
    in the order sent, ``value`` being the fitness ``_all_ones`` gives."""
    import polypus_python

    sent = []

    def fake_run_qcs(infrastructure, **kwargs):
        shots, results = kwargs["shots"], []
        for qc in kwargs["qcs"]:
            params = candidate(qc)
            ones = round(shots * _probability(params))
            sent.append((params, shots, ones / shots))
            results.append({"1" * width: ones, "0" * width: shots - ones})
        return [{k: v for k, v in r.items() if v} for r in results]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return sent


def _expected_final_batch(best_params, scored, count):
    """``best_params``, then the best other distinct candidates of
    ``scored``, up to ``count``: each candidate ranked by its best value and,
    between equal values, by the first time it reached it."""
    best = {}
    for order, (params, _, value) in enumerate(scored):
        key = tuple(params)
        if key not in best or value > best[key][0]:
            best[key] = (value, order)
    ranked = sorted(best, key=lambda key: (-best[key][0], best[key][1]))
    batch = [list(best_params)]
    for key in ranked:
        if len(batch) == count:
            break
        if list(key) != batch[0]:
            batch.append(list(key))
    return batch


class _Stop(Exception):
    """Raised by the recorder to end the run at the CUNQA seam."""


@pytest.fixture
def seam_calls(monkeypatch):
    """Record every ``connect_to_infrastructure`` call (the one that raises
    CUNQA QPUs) and stop the run there."""
    import polypus_python

    calls = []

    def record(infrastructure, **kwargs):
        calls.append((infrastructure, kwargs))
        raise _Stop

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", record)
    return calls


@pytest.mark.parametrize("entry", ENTRY_POINTS)
class TestValidation:
    """Rejected at the edge, before any QPU is requested."""

    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (0, "0"),
            (-1, "-1"),
            (1.5, "1.5"),
            (4096.0, "4096.0"),
            (True, "True"),
            (False, "False"),
            ("4096", "'4096'"),
            ([4096], "[4096]"),
        ],
    )
    def test_a_value_that_is_not_a_positive_integer_raises_before_anything_runs(
        self, seam_calls, entry, value, shown
    ):
        import polypus

        run = ENTRY_POINTS[entry][0]
        with pytest.raises(
            ValueError,
            match=re.escape(f"final_shots must be an integer >= 1, got {shown}"),
        ):
            run(polypus.DE(), infrastructure="cunqa", final_shots=value)
        assert seam_calls == []

    def test_a_value_above_a_circuits_shots_raises_before_anything_runs(
        self, seam_calls, entry
    ):
        import polypus

        run = ENTRY_POINTS[entry][0]
        with pytest.raises(
            ValueError,
            match=re.escape(f"final_shots must be at most 4294967295, got {2**32}"),
        ):
            run(polypus.DE(), infrastructure="cunqa", final_shots=2**32)
        assert seam_calls == []

    @pytest.mark.parametrize("raised", [LookupError, KeyboardInterrupt])
    def test_an_exception_from_index_propagates_unchanged(
        self, seam_calls, entry, raised
    ):
        # Only operator.index's TypeError means "not an integer": whatever
        # else a custom __index__ raises is the caller's, not a ValueError.
        import polypus

        class _Index:
            def __index__(self):
                raise raised("from __index__")

        run = ENTRY_POINTS[entry][0]
        with pytest.raises(raised, match="from __index__"):
            run(polypus.DE(), infrastructure="cunqa", final_shots=_Index())
        assert seam_calls == []

    @pytest.mark.parametrize("value", [None, 1, FINAL_SHOTS, np.int64(FINAL_SHOTS)])
    def test_a_valid_value_gets_to_the_seam(self, seam_calls, entry, value):
        # The control for the tests above.
        import polypus

        run = ENTRY_POINTS[entry][0]
        with pytest.raises(_Stop):
            run(polypus.DE(), infrastructure="cunqa", final_shots=value)
        assert len(seam_calls) == 1


@pytest.mark.integration
@pytest.mark.vqc
class TestFinalBatch:
    """The batch the optimizer's best candidates are re-evaluated in, read
    from the circuits the seam received."""

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    @pytest.mark.parametrize("name", OPTIMIZERS)
    def test_one_batch_with_final_shots_after_an_unchanged_run(
        self, monkeypatch, entry, name
    ):
        run, candidate, width = ENTRY_POINTS[entry]
        sent = _record(monkeypatch, candidate, width)
        plain = run(_method(name))
        sent_plain = list(sent)
        sent.clear()
        result = run(_method(name), final_shots=FINAL_SHOTS)

        assert plain.final_params is None
        assert plain.final_fitness is None
        assert "final_fitness" not in repr(plain)
        # Every circuit the optimizer sent, and so its result, is unchanged.
        scored = [c for c in sent if c[1] == SHOTS]
        final = [c for c in sent if c[1] == FINAL_SHOTS]
        assert sent == scored + final
        assert scored == sent_plain
        assert result.best_params == plain.best_params
        assert result.best_fitness == plain.best_fitness
        assert result.fitness_history == plain.fitness_history
        assert result.iterations_run == plain.iterations_run
        assert result.converged == plain.converged
        # The final batch: best_params first, then the best others.
        count = 5 if name in POPULATION else 1
        expected = _expected_final_batch(result.best_params, scored, count)
        assert [params for params, _, _ in final] == expected
        assert len(final) == count
        # The highest re-evaluated value wins, the earlier on a tie.
        values = [value for _, _, value in final]
        winner = values.index(max(values))
        assert result.final_fitness == values[winner]
        assert result.final_params == final[winner][0]
        assert f"final_fitness={result.final_fitness}" in repr(result)

    def test_an_error_in_the_run_skips_the_final_batch(self, monkeypatch):
        sent = _record(monkeypatch, _vqc_candidate, DIMENSIONS)
        calls = []

        def failing(bitstring):
            calls.append(bitstring)
            if len(calls) > 4:
                raise RuntimeError("cost failed")
            return _all_ones(bitstring)

        import polypus

        with pytest.raises(RuntimeError, match="cost failed"):
            polypus.train(
                _circuit(),
                _method("rand1bin"),
                shots=SHOTS,
                n_qpus=1,
                dimensions=DIMENSIONS,
                expectation_function=failing,
                infrastructure="local",
                nodes=1,
                cores_per_qpu=1,
                id="final_shots_error",
                backend="aer",
                seed=11,
                final_shots=FINAL_SHOTS,
            )
        assert sent
        assert all(shots == SHOTS for _, shots, _ in sent)

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_interrupt_skips_the_final_batch(self, monkeypatch, entry):
        import polypus_python

        run = ENTRY_POINTS[entry][0]
        shots = []

        def interrupted(infrastructure, **kwargs):
            shots.append(kwargs["shots"])
            if len(shots) == 3:
                raise KeyboardInterrupt
            return [{"0": kwargs["shots"]} for _ in kwargs["qcs"]]

        monkeypatch.setattr(polypus_python, "run_qcs", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(_method("rand1bin"), final_shots=FINAL_SHOTS)
        assert shots == [SHOTS] * 3

    def test_a_failing_final_batch_raises(self, monkeypatch):
        import polypus_python

        def fails_at_the_end(infrastructure, **kwargs):
            if kwargs["shots"] == FINAL_SHOTS:
                raise ValueError("final batch failed")
            return [{"11": kwargs["shots"]} for _ in kwargs["qcs"]]

        monkeypatch.setattr(polypus_python, "run_qcs", fails_at_the_end)
        with pytest.raises(ValueError, match="final batch failed"):
            _train(_method("adam"), final_shots=FINAL_SHOTS)


def _native_train(name, **kwargs):
    """A seeded run on the native simulator. The third qubit is in |+>, so
    all ones has a probability of at most 1/2 and no value is 0 or 1."""
    import polypus

    qc = polypus.Circuit(3)
    qc.ry(0, polypus.Param(0))
    qc.ry(1, polypus.Param(1))
    qc.h(2)
    qc.measure_all()
    return polypus.train(
        qc,
        _method(name),
        shots=SHOTS,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="final_shots_native",
        backend="polypus",
        seed=7,
        **kwargs,
    )


@pytest.mark.integration
@pytest.mark.vqc
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_a_seeded_native_run_is_unchanged_and_the_final_value_has_final_shots(name):
    """The native backend seeds each circuit from its index in the run; the
    final batch comes after every circuit the optimizer sent, so the run is
    bit for bit the one without it. 999 shots give values k/999, which a
    64-shot value j/64 strictly between 0 and 1 never is."""
    plain = _native_train(name)
    result = _native_train(name, final_shots=999)
    assert result.best_params == plain.best_params
    assert result.best_fitness == plain.best_fitness
    assert result.fitness_history == plain.fitness_history
    assert result.iterations_run == plain.iterations_run
    assert plain.final_params is None and plain.final_fitness is None
    value = result.final_fitness
    assert 0.0 < value < 1.0
    assert value * 999 == pytest.approx(round(value * 999), abs=1e-9)
    assert value * SHOTS != pytest.approx(round(value * SHOTS), abs=1e-9)
    assert len(result.final_params) == DIMENSIONS
    if name not in POPULATION:
        assert result.final_params == result.best_params
