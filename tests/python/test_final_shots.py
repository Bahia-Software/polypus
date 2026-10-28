"""
``final_shots`` in ``polypus.train`` and ``polypus.qml.train`` (contract C-5).

A value that is not an integer ``>= 1`` raises ``ValueError`` before anything
runs: with ``infrastructure="cunqa"``, before ``connect_to_infrastructure``
(and so ``qraise``) is reached, which a recorder checks without SLURM or
CUNQA.

With a valid value, once the optimizer has stopped, its best candidates are
re-evaluated in one more batch with ``final_shots`` shots per circuit:
``best_params`` and, for DE, PSO, CMA-ES and External, the four best other
distinct candidates they scored. The batch tests run the real training loops on Aer
with ``polypus_python.run_qcs`` replaced by a fake that scores each circuit
by its parameters, records every circuit with its shots and value, and so
checks which candidates the final batch holds, with which shots and seed,
which one wins, and that every batch before it is the one sent without the
option. A real Aer run checks that the final batch is measured afresh. The
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
OPTIMIZERS = [
    "rand1bin",
    "shade",
    "lshade",
    "pso",
    "qng",
    "cmaes",
    "adam",
    "spsa",
    "rotosolve",
    "external",
]
# Re-evaluate their five best candidates; the others only best_params.
POPULATION = {"rand1bin", "shade", "lshade", "pso", "cmaes", "external"}

_RY = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _method(name):
    """A small configuration of the optimizer ``name`` (a DE strategy name,
    ``"pso"``, ``"qng"``, ``"cmaes"``, ``"adam"``, ``"spsa"``, ``"rotosolve"``
    or ``"external"``). Both models below qualify for Rotosolve: each trainable
    parameter is one ``ry`` or ``rz`` angle. ``"external"`` is a fresh seeded
    random search, so two calls propose the same candidates."""
    import polypus
    from ask_tell_helpers import RandomSearch

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
    if name == "rotosolve":
        return polypus.Rotosolve(sweeps=2)
    if name == "external":
        return polypus.External(RandomSearch(DIMENSIONS, sizes=(4,), seed=3), 12)
    return polypus.DE(generations=3, population_size=6, strategy=name)


def _circuit():
    import polypus

    qc = polypus.Circuit(DIMENSIONS)
    for q in range(DIMENSIONS):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return qc


def _train(method, *, infrastructure="local", shots=SHOTS, **kwargs):
    import polypus

    return polypus.train(
        _circuit(),
        method,
        shots=shots,
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


def _qml_train(method, *, infrastructure="local", shots=SHOTS, **kwargs):
    import polypus

    feature_map, ansatz = _qml_model()
    return polypus.qml.train(
        feature_map,
        ansatz,
        # One sample, so each candidate is one circuit.
        [[0.3]],
        method,
        shots=shots,
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


def _record(monkeypatch, candidate, width, calls=None):
    """Replace the local seam with a fake that measures all ones on
    ``round(shots · p)`` shots, ``p`` a fixed function of the circuit's
    candidate, and records ``(candidate, shots, value)`` for every circuit
    in the order sent, ``value`` being the fitness ``_all_ones`` gives, and
    ``(shots, seed)`` for every call in ``calls`` when it is a list."""
    import polypus_python

    sent = []

    def fake_run_qcs(infrastructure, **kwargs):
        shots, results = kwargs["shots"], []
        if calls is not None:
            calls.append((shots, kwargs.get("seed")))
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
        ("value", "shown", "error"),
        [
            (0, "0", ValueError),
            (-1, "-1", ValueError),
            # Not an integer: a TypeError, as for shots.
            (1.5, "1.5", TypeError),
            (4096.0, "4096.0", TypeError),
            (True, "True", TypeError),
            (False, "False", TypeError),
            ("4096", "'4096'", TypeError),
            ([4096], "[4096]", TypeError),
        ],
    )
    def test_a_value_that_is_not_a_positive_integer_raises_before_anything_runs(
        self, seam_calls, entry, value, shown, error
    ):
        import polypus

        run = ENTRY_POINTS[entry][0]
        with pytest.raises(
            error,
            match=re.escape(f"final_shots must be an integer >= 1, got {shown}"),
        ):
            run(polypus.DE(), infrastructure="cunqa", final_shots=value)
        assert seam_calls == []

    def test_shots_and_final_shots_raise_the_same_error_for_a_float(
        self, seam_calls, entry
    ):
        import polypus

        run = ENTRY_POINTS[entry][0]
        # Only the class: PyO3's wording for `shots` depends on the Python
        # version (on 3.10 it does not name the argument).
        with pytest.raises(TypeError):
            run(polypus.DE(), infrastructure="cunqa", shots=4096.0)
        with pytest.raises(TypeError, match="final_shots must be an integer"):
            run(polypus.DE(), infrastructure="cunqa", final_shots=4096.0)
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
        assert "final_params" not in repr(plain)
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
        # The final point last, after the best one.
        text = repr(result)
        assert text.index("best_params=[") < text.index(", final_params=[")
        assert text.endswith("])")

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_the_final_batch_has_a_seed_of_its_own(self, monkeypatch, entry):
        # Aer seeds each circuit from the seed and its position in the call,
        # so with the run's seed the final batch would measure best_params,
        # at index 0 as Adam scored it, with the training's random numbers.
        run, candidate, width = ENTRY_POINTS[entry]
        seeds = []
        for _ in range(2):
            calls = []
            _record(monkeypatch, candidate, width, calls)
            run(_method("adam"), final_shots=FINAL_SHOTS)
            *scored, final = calls
            assert scored and all(call == (SHOTS, 11) for call in scored)
            shots, seed = final
            assert shots == FINAL_SHOTS
            assert seed not in (None, 11)
            seeds.append(seed)
        # Derived from the run's seed, the same in every run.
        assert seeds[0] == seeds[1]

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

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_an_interrupt_during_the_final_batch_raises(self, monkeypatch, entry):
        # Ctrl+C while the final batch runs: the run raises KeyboardInterrupt,
        # as a failing final batch raises its error, and returns no result.
        import polypus_python

        run = ENTRY_POINTS[entry][0]
        shots = []

        def interrupted_at_the_end(infrastructure, **kwargs):
            shots.append(kwargs["shots"])
            if kwargs["shots"] == FINAL_SHOTS:
                raise KeyboardInterrupt
            return [{"0": kwargs["shots"]} for _ in kwargs["qcs"]]

        monkeypatch.setattr(polypus_python, "run_qcs", interrupted_at_the_end)
        with pytest.raises(KeyboardInterrupt):
            run(_method("rand1bin"), final_shots=FINAL_SHOTS)
        # The whole run, then the final batch, cut at its first call.
        assert len(shots) > 1
        assert shots[:-1] == [SHOTS] * (len(shots) - 1)
        assert shots[-1] == FINAL_SHOTS

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
def test_an_aer_run_measures_the_final_batch_afresh():
    """With the training's shots, the final batch measures Adam's
    ``best_params`` at index 0 as the training did. Had it the run's seed,
    Aer would sample the same counts and ``final_fitness`` would be
    ``best_fitness`` in every run; with its own seed it is a fresh estimate,
    which a seeded run still reproduces. All ones has a probability of about
    1/2 at the starting point, so two estimates with 1024 shots rarely
    agree."""
    import polypus

    def run(seed):
        return polypus.train(
            _circuit(),
            polypus.Adam(max_iters=1),
            shots=1024,
            n_qpus=1,
            dimensions=DIMENSIONS,
            expectation_function=_all_ones,
            infrastructure="local",
            nodes=1,
            cores_per_qpu=1,
            id="final_shots_aer",
            backend="aer",
            seed=seed,
            initial_point=[2.0, 2.0],
            final_shots=1024,
        )

    results = [run(seed) for seed in (11, 12, 13)]
    assert all(r.final_params == r.best_params for r in results)
    assert any(r.final_fitness != r.best_fitness for r in results)
    again = run(11)
    assert again.best_fitness == results[0].best_fitness
    assert again.final_fitness == results[0].final_fitness


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
