"""
``initial_point`` in ``polypus.train`` and ``polypus.qml.train`` (contract C-5).

A point of the wrong length or with a non-finite value raises ``ValueError``
before anything runs: with ``infrastructure="cunqa"``, before
``connect_to_infrastructure`` (and so ``qraise``) is reached, which a recorder
checks without SLURM or CUNQA.

A valid point is where the optimizer starts. The batch tests run the real
training loops on Aer with ``polypus_python.run_qcs`` replaced by a fake that
records every batch the optimizer sends and answers with fixed counts, and
read the candidates back from the recorded circuits: the point is the first
candidate DE (every strategy, wrapped into ``[0, 2π)``), PSO and Adam (before
its first finite-difference stencil) evaluate, the centre of QNG's first
stencil, the mean of CMA-ES's first generation, and the centre of SPSA's
calibration batch, which SPSA then measures first. Without it nothing
changes; ``test_determinism_baseline.py`` pins that for seeded runs.
"""

import math
import re

import pytest

DIMENSIONS = 2
# Inside DE's [0, 2π) and PSO's default bounds (-π, π), away from either seam.
POINT = [0.25, 1.5]
# QNG's and Adam's default finite_difference_step.
QNG_STEP = 0.1
ADAM_STEP = 0.2
# SPSA's default perturbation.
SPSA_STEP = 0.1
OPTIMIZERS = ["rand1bin", "shade", "lshade", "pso", "qng", "cmaes", "adam", "spsa"]
STRATEGIES = ["rand1bin", "shade", "lshade"]

_RY = re.compile(r"^ry\(([^)]*)\) q\[(\d+)\];$", re.MULTILINE)


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _method(name):
    """A small configuration of the optimizer ``name`` (a DE strategy name,
    ``"pso"``, ``"qng"``, ``"cmaes"``, ``"adam"`` or ``"spsa"``). CMA-ES gets a tiny
    ``sigma0`` so that every candidate of its first generation sits on its
    mean."""
    import polypus

    if name == "pso":
        return polypus.PSO(generations=2, population_size=6)
    if name == "qng":
        return polypus.QNG(lambda theta, a: 1.0, max_iters=1)
    if name == "cmaes":
        return polypus.CMAES(generations=1, sigma0=1e-9)
    if name == "adam":
        return polypus.Adam(max_iters=1)
    if name == "spsa":
        return polypus.SPSA(max_iters=1)
    return polypus.DE(generations=2, population_size=6, strategy=name)


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
        shots=64,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="initial_point",
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
        # One sample, so a batch holds one circuit per candidate.
        [[0.3]],
        method,
        shots=64,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="initial_point_qml",
        seed=11,
        **kwargs,
    )


def _vqc_candidate(qasm):
    """The candidate a recorded ``train`` circuit (OpenQASM) was bound to."""
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


ENTRY_POINTS = {
    "train": (_train, _vqc_candidate),
    "qml.train": (_qml_train, _qml_candidate),
}


def _record_batches(monkeypatch):
    """Replace the local seam with a fake that records each batch of circuits
    and returns the same counts for every circuit, so every candidate gets the
    same fitness."""
    import polypus_python

    batches = []

    def fake_run_qcs(infrastructure, **kwargs):
        batches.append(list(kwargs["qcs"]))
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return batches


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

    @pytest.mark.parametrize("point", [[0.25], [0.25, 1.5, 2.0], []])
    def test_a_wrong_length_raises_before_anything_runs(self, seam_calls, entry, point):
        import polypus

        run, _ = ENTRY_POINTS[entry]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "invalid initial_point: must have one value per dimension (2), "
                f"got {len(point)}"
            ),
        ):
            run(polypus.DE(), infrastructure="cunqa", initial_point=point)
        assert seam_calls == []

    @pytest.mark.parametrize(
        ("value", "shown"),
        [(float("nan"), "NaN"), (float("inf"), "inf"), (float("-inf"), "-inf")],
    )
    def test_a_non_finite_value_raises_before_anything_runs(
        self, seam_calls, entry, value, shown
    ):
        import polypus

        run, _ = ENTRY_POINTS[entry]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "invalid initial_point: every value must be finite, "
                f"got {shown} at index 1"
            ),
        ):
            run(polypus.DE(), infrastructure="cunqa", initial_point=[0.25, value])
        assert seam_calls == []

    def test_a_point_outside_pso_bounds_raises_before_anything_runs(
        self, seam_calls, entry
    ):
        import polypus

        run, _ = ENTRY_POINTS[entry]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "invalid initial_point: every value must lie within bounds "
                f"[-{math.pi}, {math.pi}], got 4 at index 1"
            ),
        ):
            run(polypus.PSO(), infrastructure="cunqa", initial_point=[0.25, 4.0])
        assert seam_calls == []

    def test_a_valid_point_gets_to_the_seam(self, seam_calls, entry):
        # The control for the two tests above: the recorder sees a valid run.
        import polypus

        run, _ = ENTRY_POINTS[entry]
        with pytest.raises(_Stop):
            run(polypus.DE(), infrastructure="cunqa", initial_point=POINT)
        assert len(seam_calls) == 1

    def test_a_non_sequence_raises_type_error(self, seam_calls, entry):
        import polypus

        run, _ = ENTRY_POINTS[entry]
        with pytest.raises(TypeError):
            run(polypus.DE(), infrastructure="cunqa", initial_point=0.25)
        assert seam_calls == []


@pytest.mark.integration
@pytest.mark.vqc
class TestStartingPoint:
    """What each optimizer evaluates first, read from the recorded batches."""

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    @pytest.mark.parametrize("name", OPTIMIZERS)
    def test_the_optimizer_starts_at_the_point(self, monkeypatch, entry, name):
        run, candidate = ENTRY_POINTS[entry]
        batches = _record_batches(monkeypatch)
        result = run(_method(name), initial_point=POINT)
        first = [candidate(qc) for qc in batches[0]]
        if name in ("qng", "adam"):
            # The finite-difference stencil around θ₀ = POINT; Adam measures
            # θ₀ itself first, in the same batch.
            step = QNG_STEP if name == "qng" else ADAM_STEP
            p0, p1 = POINT
            stencil = [
                [p0 + step, p1],
                [p0 - step, p1],
                [p0, p1 + step],
                [p0, p1 - step],
            ]
            if name == "adam":
                stencil.insert(0, POINT)
            assert len(first) == len(stencil)
            for got, want in zip(first, stencil):
                assert got == pytest.approx(want, abs=1e-12)
        elif name == "spsa":
            # 25 pairs θ₀ ± c·Δ around θ₀ = POINT, Δ of ±1 components, then
            # the first iteration measures θ₀ first. (The QML oracle may split
            # a batch into windows, so the circuits are read in order.)
            sent = [candidate(qc) for batch in batches for qc in batch]
            calibration = sent[:50]
            for plus, minus in zip(calibration[::2], calibration[1::2]):
                assert [(a + b) / 2 for a, b in zip(plus, minus)] == pytest.approx(
                    POINT, abs=1e-12
                )
                assert [abs(a - p) for a, p in zip(plus, POINT)] == pytest.approx(
                    [SPSA_STEP] * DIMENSIONS, abs=1e-12
                )
            assert sent[50] == pytest.approx(POINT, abs=1e-12)
            assert len(sent) == 50 + 3
        elif name == "cmaes":
            # 4 + floor(3 ln 2) = 6 candidates, all on the first mean.
            assert len(first) == 6
            for got in first:
                assert got == pytest.approx(POINT, abs=1e-6)
        else:
            assert first[0] == pytest.approx(POINT, abs=1e-12)
        assert len(result.best_params) == DIMENSIONS
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_de_wraps_the_point_and_draws_the_rest_as_without_it(
        self, monkeypatch, strategy
    ):
        point = [-0.5, 7.0]
        with_point = _record_batches(monkeypatch)
        _train(_method(strategy), initial_point=point)
        without = _record_batches(monkeypatch)
        _train(_method(strategy))
        wrapped = [x % (2 * math.pi) for x in point]
        assert _vqc_candidate(with_point[0][0]) == pytest.approx(wrapped, abs=1e-12)
        # The other members are the ones the same seed draws without a point.
        assert with_point[0][1:] == without[0][1:]

    def test_pso_rejects_a_point_outside_its_bounds_before_any_batch(self, monkeypatch):
        batches = _record_batches(monkeypatch)
        # PSO's default bounds are (-π, π); 4.0 would be clamped onto π.
        with pytest.raises(
            ValueError, match=r"initial_point: every value must lie within bounds"
        ):
            _train(_method("pso"), initial_point=[0.25, 4.0])
        assert batches == []

    def test_qng_accepts_a_point_outside_its_bounds(self, monkeypatch):
        batches = _record_batches(monkeypatch)
        _train(_method("qng"), initial_point=[-5.0, 9.0])
        assert _vqc_candidate(batches[0][0]) == pytest.approx(
            [-5.0 + QNG_STEP, 9.0], abs=1e-12
        )

    def test_adam_accepts_a_point_outside_its_bounds(self, monkeypatch):
        batches = _record_batches(monkeypatch)
        _train(_method("adam"), initial_point=[-5.0, 9.0])
        assert _vqc_candidate(batches[0][0]) == pytest.approx([-5.0, 9.0], abs=1e-12)

    def test_spsa_takes_a_point_outside_zero_two_pi_unwrapped(self, monkeypatch):
        batches = _record_batches(monkeypatch)
        _train(_method("spsa"), initial_point=[-5.0, 9.0])
        assert _vqc_candidate(batches[1][0]) == pytest.approx([-5.0, 9.0], abs=1e-12)

    def test_none_is_the_default(self, monkeypatch):
        omitted = _record_batches(monkeypatch)
        a = _train(_method("rand1bin"))
        explicit = _record_batches(monkeypatch)
        b = _train(_method("rand1bin"), initial_point=None)
        assert omitted == explicit
        assert a.best_params == b.best_params

    def test_best_params_numpy_arrays_and_tuples_are_accepted(self, monkeypatch):
        # The two-call pattern of the training guide: explore, then refine
        # from the best point found.
        import numpy as np

        _record_batches(monkeypatch)
        explore = _train(_method("rand1bin"))
        start = explore.best_params
        assert isinstance(start, list)
        for point in (start, np.array(start), tuple(start)):
            batches = _record_batches(monkeypatch)
            _train(_method("cmaes"), initial_point=point)
            for qc in batches[0]:
                assert _vqc_candidate(qc) == pytest.approx(start, abs=1e-6)

    def test_qml_train_accepts_a_numpy_array(self, monkeypatch):
        import numpy as np

        batches = _record_batches(monkeypatch)
        _qml_train(_method("pso"), initial_point=np.array(POINT))
        assert _qml_candidate(batches[0][0]) == pytest.approx(POINT, abs=1e-12)
