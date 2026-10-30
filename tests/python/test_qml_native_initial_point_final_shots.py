"""
``initial_point`` and ``final_shots`` in ``polypus.qml.Model.fit`` (contracts
C-5 and C-11), with the semantics they have in ``train`` and ``qml.train``.

Every invalid value raises before a backend exists: with
``infrastructure="cunqa"``, before ``connect_to_infrastructure`` (and so
``qraise``) is reached, which a recorder checks without SLURM or CUNQA. An
invalid optimizer configuration is reported before the point, and a malformed
``final_shots`` before its clash with ``exact=True``, which ``fit`` refuses.

Where each optimizer starts is read from the circuits the local Aer seam
receives, with ``polypus_python.run_qcs`` replaced by a recorder that answers
fixed counts: a native model's circuits arrive as OpenQASM, whose angles are
exported at full precision. The final re-evaluation runs on the native
simulator, seeded, against the same run without it and against the exact
value of the model bound to ``final_params``.
"""

import math
import re

import numpy as np
import pytest

# An `rx` encoder, then `ry(theta_0)` and `ry(theta_1)`: the only `ry` angles of
# a recorded circuit are the candidate, in order.
NUM_PARAMS = 2
# One sample, so a batch holds one circuit per candidate.
ROW = [[0.3]]
# Inside DE's [0, 2π) and PSO's default bounds (-π, π), away from either seam.
POINT = [0.25, 1.5]
QNG_STEP = 0.1
SPSA_STEP = 0.1
SHOTS = 64
FINAL_SHOTS = 999
OPTIMIZERS = ["de", "pso", "qng", "cmaes", "adam", "spsa", "rotosolve"]
# Re-evaluate their five best candidates; the others only best_params.
POPULATION = {"de", "pso", "cmaes", "external"}

_RY = re.compile(r"^ry\(([^)]*)\) q\[0\];$", re.MULTILINE)


def _model(encoder="rx"):
    """One qubit: the ``encoder`` rotation of the sample, then
    ``ry(theta_0) ry(theta_1)``, reading the raw ``⟨Z₀⟩``."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(1).angle_encoder(encoder),
        Ansatz(1).real_amplitudes(1),
        Readout([Z(0)]),
    )


def _method(name):
    """A small configuration of the optimizer ``name``. The model qualifies
    for Rotosolve: unsupervised, each parameter one ``ry`` angle. CMA-ES gets
    a tiny ``sigma0`` so that its first generation sits on its mean."""
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
    if name == "rotosolve":
        return polypus.Rotosolve(sweeps=1)
    if name == "external":
        from ask_tell_helpers import RandomSearch

        return polypus.External(RandomSearch(NUM_PARAMS, sizes=(4,), seed=3), 12)
    return polypus.DE(generations=2, population_size=6)


def _fit(method, x=ROW, model=None, **kwargs):
    """An unsupervised run of ``method`` on the recordable model, on the local
    Aer seam unless ``kwargs`` say otherwise."""
    kwargs.setdefault("backend", "aer")
    kwargs.setdefault("shots", SHOTS)
    kwargs.setdefault("seed", 11)
    return (model or _model()).fit(x, method=method, **kwargs)


def _candidate(qasm):
    """The candidate a recorded circuit was bound to."""
    return [float(angle) for angle in _RY.findall(qasm)]


def _record_batches(monkeypatch, shots=None):
    """Replace the local seam with a fake that records each batch of circuits
    and measures ``0`` on every shot, so every candidate gets the same
    fitness; and the shots of each batch in ``shots`` when it is a list."""
    import polypus_python

    batches = []

    def fake_run_qcs(infrastructure, **kwargs):
        batches.append(list(kwargs["qcs"]))
        if shots is not None:
            shots.append(kwargs["shots"])
        return [{"0": kwargs["shots"]} for _ in kwargs["qcs"]]

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
        calls.append(infrastructure)
        raise _Stop

    monkeypatch.setattr(polypus_python, "connect_to_infrastructure", record)
    return calls


_CUNQA = {"infrastructure": "cunqa", "nodes": 1, "cores_per_qpu": 1}


def _fit_on_cunqa(method, **kwargs):
    return _model().fit(ROW, method=method, shots=SHOTS, seed=11, **_CUNQA, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# initial_point
# ─────────────────────────────────────────────────────────────────────────────


class TestInitialPointValidation:
    """Rejected before a backend exists, with ``train``'s messages."""

    @pytest.mark.parametrize("point", [[0.25], [0.25, 1.5, 2.0], []])
    def test_a_wrong_length(self, seam_calls, point):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match=re.escape(
                "invalid initial_point: must have one value per dimension (2), "
                f"got {len(point)}"
            ),
        ):
            _fit_on_cunqa(polypus.DE(), initial_point=point)
        assert seam_calls == []

    @pytest.mark.parametrize(
        ("value", "shown"),
        [(float("nan"), "NaN"), (float("inf"), "inf"), (float("-inf"), "-inf")],
    )
    def test_a_non_finite_value(self, seam_calls, value, shown):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match=re.escape(
                "invalid initial_point: every value must be finite, "
                f"got {shown} at index 1"
            ),
        ):
            _fit_on_cunqa(polypus.DE(), initial_point=[0.25, value])
        assert seam_calls == []

    def test_a_point_outside_pso_bounds(self, seam_calls):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match=re.escape(
                "invalid initial_point: every value must lie within bounds "
                f"[-{math.pi}, {math.pi}], got 4 at index 1"
            ),
        ):
            _fit_on_cunqa(polypus.PSO(), initial_point=[0.25, 4.0])
        assert seam_calls == []

    def test_external_takes_no_point(self, seam_calls):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match="^polypus.External takes no initial_point",
        ):
            _fit_on_cunqa(_method("external"), initial_point=POINT)
        assert seam_calls == []

    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("de", "population_size must be at least 4"),
            ("pso", "bounds must be a non-empty interval"),
            ("qng", "invalid max_iters"),
            ("cmaes", "invalid sigma0"),
            ("adam", "invalid finite_difference_step"),
            ("spsa", "invalid max_iters"),
            ("rotosolve", "invalid sweeps"),
        ],
    )
    @pytest.mark.parametrize("point", [None, [0.25]])
    def test_a_configuration_error_comes_first(self, seam_calls, name, message, point):
        # As in `train`: the optimizer's configuration is checked before the
        # point, before any backend exists, with or without a point.
        import polypus

        method = {
            "de": lambda: polypus.DE(population_size=2),
            "pso": lambda: polypus.PSO(bounds=(1.0, 0.0)),
            "qng": lambda: polypus.QNG(lambda theta, a: 1.0, max_iters=0),
            "cmaes": lambda: polypus.CMAES(sigma0=0.0),
            "adam": lambda: polypus.Adam(finite_difference_step=0.0),
            "spsa": lambda: polypus.SPSA(max_iters=0),
            "rotosolve": lambda: polypus.Rotosolve(sweeps=0),
        }[name]()
        with pytest.raises(polypus.ConfigurationError, match=re.escape(message)):
            _fit_on_cunqa(method, initial_point=point)
        assert seam_calls == []

    def test_a_non_sequence_raises_type_error(self, seam_calls):
        import polypus

        # PyO3's own error for the argument, not an unknown keyword's.
        with pytest.raises(TypeError, match="Sequence"):
            _fit_on_cunqa(polypus.DE(), initial_point=0.25)
        assert seam_calls == []

    def test_a_valid_point_gets_to_the_seam(self, seam_calls):
        # The control for the tests above.
        import polypus

        with pytest.raises(_Stop):
            _fit_on_cunqa(polypus.DE(), initial_point=POINT)
        assert seam_calls == ["cunqa"]


@pytest.mark.integration
class TestStartingPoint:
    """What each optimizer evaluates first, read from the recorded batches."""

    @pytest.mark.parametrize("name", OPTIMIZERS)
    def test_the_optimizer_starts_at_the_point(self, monkeypatch, name):
        batches = _record_batches(monkeypatch)
        result = _fit(_method(name), initial_point=POINT)
        first = [_candidate(qc) for qc in batches[0]]
        p0, p1 = POINT
        if name == "qng":
            # The finite-difference stencil around θ₀ = POINT.
            stencil = [
                [p0 + QNG_STEP, p1],
                [p0 - QNG_STEP, p1],
                [p0, p1 + QNG_STEP],
                [p0, p1 - QNG_STEP],
            ]
            assert len(first) == len(stencil)
            for got, want in zip(first, stencil):
                assert got == pytest.approx(want, abs=1e-12)
        elif name == "adam":
            # The exact gradient: θ₀ itself, then its parameter shifts.
            shift = math.pi / 2
            batch = [
                POINT,
                [p0 + shift, p1],
                [p0 - shift, p1],
                [p0, p1 + shift],
                [p0, p1 - shift],
            ]
            assert len(first) == len(batch)
            assert first[0] == POINT
            for got, want in zip(first, batch):
                assert got == pytest.approx(want, abs=1e-12)
        elif name == "spsa":
            # 25 pairs θ₀ ± c·Δ around θ₀ = POINT, then the first iteration
            # measures θ₀ first.
            sent = [_candidate(qc) for batch in batches for qc in batch]
            calibration = sent[:50]
            for plus, minus in zip(calibration[::2], calibration[1::2]):
                assert [(a + b) / 2 for a, b in zip(plus, minus)] == pytest.approx(
                    POINT, abs=1e-12
                )
                assert [abs(a - p) for a, p in zip(plus, POINT)] == pytest.approx(
                    [SPSA_STEP] * NUM_PARAMS, abs=1e-12
                )
            assert sent[50] == POINT
        elif name == "cmaes":
            # 4 + floor(3 ln 2) = 6 candidates, all on the first mean.
            assert len(first) == 6
            for got in first:
                assert got == pytest.approx(POINT, abs=1e-6)
        else:
            # DE's first member, PSO's first particle, Rotosolve's θ₀.
            assert first[0] == POINT
        assert len(result.best_params) == NUM_PARAMS

    def test_de_wraps_the_point_and_draws_the_rest_as_without_it(self, monkeypatch):
        point = [-0.5, 7.0]
        with_point = _record_batches(monkeypatch)
        _fit(_method("de"), initial_point=point)
        without = _record_batches(monkeypatch)
        _fit(_method("de"))
        wrapped = [x % (2 * math.pi) for x in point]
        assert _candidate(with_point[0][0]) == pytest.approx(wrapped, abs=1e-12)
        assert with_point[0][1:] == without[0][1:]

    def test_pso_draws_the_other_particles_as_without_it(self, monkeypatch):
        with_point = _record_batches(monkeypatch)
        _fit(_method("pso"), initial_point=POINT)
        without = _record_batches(monkeypatch)
        _fit(_method("pso"))
        assert _candidate(with_point[0][0]) == POINT
        assert with_point[0][1:] == without[0][1:]

    def test_none_is_the_default(self, monkeypatch):
        omitted = _record_batches(monkeypatch)
        a = _fit(_method("de"))
        explicit = _record_batches(monkeypatch)
        b = _fit(_method("de"), initial_point=None)
        assert omitted == explicit
        assert a.best_params == b.best_params
        assert a.fitness_history == b.fitness_history

    @pytest.mark.parametrize("kind", [list, tuple, np.array])
    def test_lists_tuples_and_numpy_arrays_are_accepted(self, monkeypatch, kind):
        batches = _record_batches(monkeypatch)
        _fit(_method("adam"), initial_point=kind(POINT))
        assert _candidate(batches[0][0]) == POINT


# Two separable classes on one feature, for the hinge loss.
_X = [[0.1], [0.3], [2.9], [3.0], [0.2], [2.8]]
_Y = [1, 1, -1, -1, 1, -1]


@pytest.mark.integration
class TestWarmStart:
    """A saved model, loaded back, keeps training from its parameters."""

    @staticmethod
    def _saved(tmp_path):
        import polypus
        from polypus.qml import load

        model = _model("ry")
        first = model.fit(
            _X,
            _Y,
            method=polypus.DE(generations=10, population_size=8),
            loss="hinge",
            exact=True,
            seed=5,
        )
        path = tmp_path / "classifier.json"
        model.bind(first.best_params).save(str(path))
        return first, load(str(path))

    def test_the_loaded_parameters_are_the_first_candidate(self, monkeypatch, tmp_path):
        import polypus

        first, trained = self._saved(tmp_path)
        assert trained.params == first.best_params
        batches = _record_batches(monkeypatch)
        trained.model.fit(
            _X,
            _Y,
            method=polypus.Adam(max_iters=1),
            loss="hinge",
            backend="aer",
            shots=SHOTS,
            initial_point=trained.params,
        )
        # Sample-major: the first circuits are θ₀ on each row, bit for bit.
        for qc in batches[0][: len(_X)]:
            assert _candidate(qc)[-NUM_PARAMS:] == trained.params

    def test_a_loaded_model_trains_on_from_its_fitness(self, tmp_path):
        # Exact mode, so the first value Adam measures is θ₀'s exact fitness,
        # which the earlier run reported as its best: training on starts there
        # instead of from a random point.
        import polypus

        first, trained = self._saved(tmp_path)

        def more(**kwargs):
            return trained.model.fit(
                _X,
                _Y,
                method=polypus.Adam(max_iters=3),
                loss="hinge",
                exact=True,
                seed=9,
                **kwargs,
            )

        resumed = more(initial_point=trained.params)
        assert resumed.fitness_history[0] >= first.best_fitness
        assert resumed.best_fitness >= first.best_fitness
        # The control: the same seed without the point starts lower.
        assert more().fitness_history[0] < first.best_fitness


# ─────────────────────────────────────────────────────────────────────────────
# final_shots
# ─────────────────────────────────────────────────────────────────────────────


class TestFinalShotsValidation:
    """Rejected before a backend exists, as ``train`` rejects it."""

    @pytest.mark.parametrize(
        ("value", "shown", "error"),
        [
            (0, "0", ValueError),
            (-1, "-1", ValueError),
            # Not an integer: a TypeError, as for shots.
            (4096.0, "4096.0", TypeError),
            (True, "True", TypeError),
            ("10", "'10'", TypeError),
        ],
    )
    def test_a_value_that_is_not_a_positive_integer(
        self, seam_calls, value, shown, error
    ):
        import polypus

        with pytest.raises(
            error,
            match=re.escape(f"final_shots must be an integer >= 1, got {shown}"),
        ):
            _fit_on_cunqa(polypus.DE(), final_shots=value)
        assert seam_calls == []

    def test_a_value_above_a_circuits_shots(self, seam_calls):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match=re.escape(f"final_shots must be at most 4294967295, got {2**32}"),
        ):
            _fit_on_cunqa(polypus.DE(), final_shots=2**32)
        assert seam_calls == []

    @pytest.mark.parametrize("value", [None, 1, 4096, np.int64(4096)])
    def test_a_valid_value_gets_to_the_seam(self, seam_calls, value):
        import polypus

        with pytest.raises(_Stop):
            _fit_on_cunqa(polypus.DE(), final_shots=value)
        assert seam_calls == ["cunqa"]


class TestFinalShotsWithExact:
    """Exact mode uses no shots, so ``fit`` refuses ``final_shots`` with it."""

    @pytest.mark.parametrize("value", [1, 4096, np.int64(4096)])
    def test_is_a_configuration_error(self, value):
        import polypus

        with pytest.raises(
            polypus.ConfigurationError,
            match=re.escape(
                "final_shots does not apply with exact=True: exact mode computes "
                "exact probabilities and uses no shots"
            ),
        ):
            _model().fit(ROW, method=polypus.Adam(), exact=True, final_shots=value)

    @pytest.mark.parametrize(("value", "error"), [(0, ValueError), (4096.0, TypeError)])
    def test_a_malformed_value_fails_its_own_check_first(self, value, error):
        with pytest.raises(error, match="final_shots must be an integer >= 1"):
            _model().fit(ROW, method=_method("adam"), exact=True, final_shots=value)

    def test_exacts_own_check_comes_first(self, seam_calls):
        # Off the local native simulator, `exact=True` is refused by itself.
        import polypus

        with pytest.raises(polypus.ConfigurationError, match="exact=True needs"):
            _fit_on_cunqa(polypus.Adam(), exact=True, final_shots=4096)
        assert seam_calls == []


def _native_fit(method, x=_X, **kwargs):
    """A seeded, sampled, unsupervised run on the native simulator: the
    fitness is the mean ``⟨Z₀⟩`` over ``x``."""
    return _model("ry").fit(
        x, method=method, shots=SHOTS, seed=7, backend="polypus", **kwargs
    )


def _unchanged(plain, result):
    """Everything the optimizer reports is the run without ``final_shots``."""
    assert result.best_params == plain.best_params
    assert result.best_fitness == plain.best_fitness
    assert result.fitness_history == plain.fitness_history
    assert result.iterations_run == plain.iterations_run
    assert result.converged == plain.converged
    assert plain.final_params is None and plain.final_fitness is None


def _exact_value_and_error(params, x, shots):
    """The exact fitness of ``params`` on ``x`` (the mean ``⟨Z₀⟩``, from the
    model's exact path) and the standard deviation of its estimate with
    ``shots`` shots per row: each row's ``⟨Z₀⟩`` estimate has variance
    ``(1 − ⟨Z₀⟩²) / shots``, and the rows are sampled independently."""
    expectations = _model("ry").bind(params).predict(x, exact=True)
    value = sum(expectations) / len(x)
    variance = sum(1 - z * z for z in expectations) / (shots * len(x) ** 2)
    return value, math.sqrt(variance), expectations


@pytest.mark.integration
class TestFinalBatchAtTheSeam:
    """The final batch ``fit`` sends to the local Aer seam: after every
    training batch, with ``final_shots`` shots, and Adam's ``best_params`` on
    every training sample, with or without minibatches."""

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_its_shots_and_every_sample(self, monkeypatch, batch_size):
        shots = []
        batches = _record_batches(monkeypatch, shots)
        result = _fit(
            _method("adam"),
            x=_X,
            batch_size=batch_size,
            final_shots=FINAL_SHOTS,
        )
        assert shots[-1] == FINAL_SHOTS
        assert all(sent == SHOTS for sent in shots[:-1])
        assert result.final_params == result.best_params
        final = batches[-1]
        assert len(final) == len(_X)
        for qc in final:
            assert _candidate(qc)[-NUM_PARAMS:] == result.final_params


@pytest.mark.integration
class TestFinalReEvaluation:
    @pytest.mark.parametrize("name", [*OPTIMIZERS, "external"])
    def test_every_optimizer_after_an_unchanged_run(self, name):
        plain = _native_fit(_method(name))
        result = _native_fit(_method(name), final_shots=FINAL_SHOTS)
        _unchanged(plain, result)
        assert len(result.final_params) == NUM_PARAMS
        assert math.isfinite(result.final_fitness)
        if name not in POPULATION:
            assert result.final_params == result.best_params

    def test_the_final_value_is_a_fresh_estimate_with_final_shots(self):
        # Rows of k/SHOTS-spaced estimates averaged over 6 rows are multiples
        # of 2/(6·SHOTS); with 999 shots they are multiples of 2/(6·999),
        # which no other value of the training is.
        import polypus

        result = _native_fit(polypus.Adam(max_iters=3), final_shots=FINAL_SHOTS)
        assert result.final_params == result.best_params
        assert result.final_fitness != result.best_fitness
        scaled = result.final_fitness * FINAL_SHOTS * len(_X) / 2
        assert scaled == pytest.approx(round(scaled), abs=1e-6)
        scaled = result.final_fitness * SHOTS * len(_X) / 2
        assert scaled != pytest.approx(round(scaled), abs=1e-6)

    def test_the_final_value_estimates_the_exact_one(self):
        import polypus

        shots = 20_000
        result = _native_fit(polypus.Adam(max_iters=3), final_shots=shots)
        value, error, _ = _exact_value_and_error(result.final_params, _X, shots)
        assert abs(result.final_fitness - value) <= 5 * error

    @pytest.mark.parametrize("name", ["adam", "de"])
    def test_with_minibatches_the_final_batch_scores_every_sample(self, name):
        # Five rows near 0 and one near π: no two rows have the mean ⟨Z₀⟩ of
        # all six, so a final batch on a minibatch of two would miss it.
        import polypus

        x = [[0.1], [0.2], [0.3], [0.15], [0.25], [2.9]]
        method = {
            "adam": lambda: polypus.Adam(max_iters=3),
            "de": lambda: polypus.DE(generations=3, population_size=6),
        }[name]
        shots = 20_000
        plain = _native_fit(method(), x=x, batch_size=2)
        result = _native_fit(method(), x=x, batch_size=2, final_shots=shots)
        _unchanged(plain, result)
        value, error, rows = _exact_value_and_error(result.final_params, x, shots)
        pairs = [(a + b) / 2 for i, a in enumerate(rows) for b in rows[i + 1 :]]
        assert min(abs(pair - value) for pair in pairs) > 10 * error
        assert abs(result.final_fitness - value) <= 5 * error
