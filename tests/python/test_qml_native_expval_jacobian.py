"""
The loss-free primitives of native QML (contract C-11, "Expectation and
Jacobian primitives"): ``polypus.qml.expval`` and ``polypus.qml.jacobian``.

``expval`` must read every readout observable exactly as ``run`` measures it,
and ``jacobian`` must be the derivative of ``expval``: with ``exact=True``
against a central finite difference (``STEP``/``TOLERANCE`` as in the Rust
``gradient_exactness.rs``), for every layer and readout, with respect to the
parameters and, for angle and IQP encoders, the inputs. With shots both are
seeded estimators. Every check runs before a backend exists.

Contract test: draft pending human sign-off.
"""

import math

import numpy as np
import polypus
import pytest
from polypus import qml

STEP = 1e-5
TOLERANCE = 1e-8

ROWS = [
    [0.4, 1.1, 0.7, 2.0],
    [1.3, 0.2, 2.2, 0.9],
    [2.0, 1.7, 0.3, 1.2],
]

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


def readouts():
    """Every readout the parameter Jacobian is checked under: a `Z` string,
    the `X` and `Y` bases, a weighted multi-term observable and a multiclass
    `argmax` readout (one column per class)."""
    return {
        "z": qml.Readout([qml.Z(1) @ qml.Z(2)]),
        "x": qml.Readout([qml.X(0) @ qml.X(2)], decision="sign"),
        "y": qml.Readout([qml.Y(1)]),
        "weighted": qml.Readout(
            [
                qml.Observable(
                    [(0.5, qml.Z(0)), (-1.5, qml.Z(1) @ qml.Z(2)), (2.0, qml.Z(3))]
                )
            ]
        ),
        "argmax": qml.Readout(
            [qml.Z(0), qml.Z(1) @ qml.Z(2), qml.Z(3)], decision="argmax"
        ),
    }


def layers():
    """(encoder, ansatz) over 4 qubits, one per layer kind."""
    return {
        "angle-rx": (
            qml.Encoder(4).angle_encoder("rx"),
            qml.Ansatz(4).hardware_efficient(1),
        ),
        "angle-ry": (
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).hardware_efficient(1),
        ),
        "angle-rz": (
            qml.Encoder(4).angle_encoder("rz"),
            qml.Ansatz(4).hardware_efficient(1),
        ),
        "real-amplitudes": (
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).real_amplitudes(2),
        ),
        "conv-basic": (qml.Encoder(4).angle_encoder("ry"), qml.Ansatz(4).conv("basic")),
        "conv-cartan": (
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).conv("cartan"),
        ),
        "reuploading": (
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).hardware_efficient(1).angle_encoder("rx").real_amplitudes(1),
        ),
        "iqp": (
            qml.Encoder(4).iqp_encoder(),
            qml.Ansatz(4).real_amplitudes(1).iqp_encoder("linear"),
        ),
    }


def point(model, offset=0.3):
    """Parameters that are neither zero nor symmetric."""
    return np.array([offset + 0.17 * i for i in range(model.num_params)])


def fd_params(model, x, params):
    """Central finite difference of `expval(exact=True)` in each parameter:
    shape (rows, observables, num_params)."""
    columns = []
    for p in range(len(params)):
        plus, minus = params.copy(), params.copy()
        plus[p] += STEP
        minus[p] -= STEP
        columns.append(
            (
                qml.expval(model, x, plus, exact=True)
                - qml.expval(model, x, minus, exact=True)
            )
            / (2 * STEP)
        )
    return np.stack(columns, axis=-1)


def fd_inputs(model, x, params):
    """Central finite difference of `expval(exact=True)` in each feature:
    shape (rows, observables, features)."""
    x = np.array(x, dtype=float)
    columns = []
    for f in range(x.shape[1]):
        plus, minus = x.copy(), x.copy()
        plus[:, f] += STEP
        minus[:, f] -= STEP
        columns.append(
            (
                qml.expval(model, plus, params, exact=True)
                - qml.expval(model, minus, params, exact=True)
            )
            / (2 * STEP)
        )
    return np.stack(columns, axis=-1)


def assert_close(shift, reference):
    assert shift.shape == reference.shape
    worst = np.max(np.abs(shift - reference))
    assert worst < TOLERANCE, f"parameter shift and finite difference differ by {worst}"


def parity(probabilities, terms):
    """`Σ_b p(b)·Π (−1)^{b_q}` over `terms` (positions), keys little-endian:
    the character at `width − 1 − q` is qubit `q`."""
    total = 0.0
    for key, p in probabilities.items():
        width = len(key)
        sign = 1
        for q in terms:
            if key[width - 1 - q] == "1":
                sign = -sign
        total += sign * p
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Shapes and types (criterion 1, 9)
# ─────────────────────────────────────────────────────────────────────────────


class TestShapesAndTypes:
    def _model(self):
        encoder, ansatz = layers()["iqp"]
        return qml.Model(encoder, ansatz, readouts()["argmax"])

    def _assert_array(self, array, shape):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags["C_CONTIGUOUS"]
        assert array.shape == shape

    def test_expval_is_rows_by_observables(self):
        model = self._model()
        values = qml.expval(model, ROWS, point(model), exact=True)
        self._assert_array(values, (3, 3))

    def test_jacobian_shapes_follow_wrt(self):
        model = self._model()
        params = point(model)
        by_params = qml.jacobian(model, ROWS, params, exact=True)
        self._assert_array(by_params, (3, 3, model.num_params))
        by_inputs = qml.jacobian(model, ROWS, params, wrt="inputs", exact=True)
        self._assert_array(by_inputs, (3, 3, 4))
        both = qml.jacobian(model, ROWS, params, wrt="both", exact=True)
        assert isinstance(both, tuple) and len(both) == 2
        self._assert_array(both[0], (3, 3, model.num_params))
        self._assert_array(both[1], (3, 3, 4))

    def test_params_and_rows_take_lists_tuples_arrays_and_a_dataset(self):
        model = self._model()
        params = point(model)
        reference = qml.expval(model, ROWS, params, exact=True)
        for x in [
            np.array(ROWS),
            tuple(ROWS),
            qml.Dataset(ROWS),
            qml.Dataset(ROWS, [0, 1, 0]),
        ]:
            for p in [params, list(params), tuple(params)]:
                assert np.array_equal(qml.expval(model, x, p, exact=True), reference)

    def test_both_is_the_two_separate_calls_bit_for_bit(self):
        model = self._model()
        params = point(model)
        by_params, by_inputs = qml.jacobian(model, ROWS, params, wrt="both", exact=True)
        assert np.array_equal(by_params, qml.jacobian(model, ROWS, params, exact=True))
        assert np.array_equal(
            by_inputs, qml.jacobian(model, ROWS, params, wrt="inputs", exact=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# expval reads what run measures (criterion 1)
# ─────────────────────────────────────────────────────────────────────────────


class TestExpvalReadsTheRun:
    def test_exact_expval_is_the_readout_formula_over_run(self):
        # Every observable of an argmax readout, the weighted one included.
        readout = qml.Readout(
            [qml.Z(0), qml.Observable([(0.5, qml.Z(1)), (-1.5, qml.Z(0) @ qml.Z(2))])],
            decision="argmax",
        )
        model = qml.Model(
            qml.Encoder(3).angle_encoder("ry"),
            qml.Ansatz(3).hardware_efficient(1),
            readout,
        )
        params = point(model)
        x = [row[:3] for row in ROWS]
        values = qml.expval(model, x, params, exact=True)
        run = model.bind(list(params)).run(x, exact=True)
        for row, probabilities in zip(values, run.probabilities):
            expected = [
                parity(probabilities, [0]),
                0.5 * parity(probabilities, [1]) - 1.5 * parity(probabilities, [0, 2]),
            ]
            assert row == pytest.approx(expected, abs=1e-12)

    def test_exact_expval_of_a_raw_readout_is_predict(self):
        for name in ["z", "x", "y", "weighted"]:
            encoder, ansatz = layers()["angle-ry"]
            readout = readouts()[name]
            if name == "x":
                readout = qml.Readout([qml.X(0) @ qml.X(2)])
            model = qml.Model(encoder, ansatz, readout)
            params = point(model)
            values = qml.expval(model, ROWS, params, exact=True)
            predicted = model.bind(list(params)).predict(ROWS, exact=True)
            assert values[:, 0] == pytest.approx(predicted, abs=1e-12), name

    def test_sampled_expval_is_within_shot_noise_and_replays(self):
        encoder, ansatz = layers()["angle-ry"]
        model = qml.Model(encoder, ansatz, readouts()["argmax"])
        params = point(model)
        shots = 20_000
        exact = qml.expval(model, ROWS, params, exact=True)
        sampled = qml.expval(model, ROWS, params, shots=shots, seed=11)
        # Each column is a mean of ±1 outcomes: 5 standard errors at most.
        assert np.max(np.abs(sampled - exact)) < 5 / math.sqrt(shots)
        assert np.array_equal(
            sampled, qml.expval(model, ROWS, params, shots=shots, seed=11)
        )
        assert not np.array_equal(
            sampled, qml.expval(model, ROWS, params, shots=shots, seed=12)
        )


# ─────────────────────────────────────────────────────────────────────────────
# The Jacobian is the derivative (criteria 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────


class TestJacobianIsExact:
    @pytest.mark.parametrize("layer", sorted(layers()))
    @pytest.mark.parametrize("readout", sorted(readouts()))
    def test_by_params_is_the_finite_difference(self, layer, readout):
        encoder, ansatz = layers()[layer]
        model = qml.Model(encoder, ansatz, readouts()[readout])
        params = point(model)
        assert_close(
            qml.jacobian(model, ROWS, params, exact=True),
            fd_params(model, ROWS, params),
        )

    def test_pooling_shares_its_parameters_across_pairs(self):
        readout = qml.Readout([qml.Z(0), qml.Z(1)], decision="argmax")
        model = qml.Model(
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).conv("basic").pool("basic"),
            readout,
        )
        params = point(model, offset=-1.1)
        assert_close(
            qml.jacobian(model, ROWS, params, exact=True),
            fd_params(model, ROWS, params),
        )

    @pytest.mark.parametrize(
        "layer", ["angle-rx", "angle-ry", "angle-rz", "reuploading", "iqp"]
    )
    @pytest.mark.parametrize("readout", ["weighted", "argmax"])
    def test_by_inputs_is_the_finite_difference(self, layer, readout):
        encoder, ansatz = layers()[layer]
        model = qml.Model(encoder, ansatz, readouts()[readout])
        params = point(model)
        assert_close(
            qml.jacobian(model, ROWS, params, wrt="inputs", exact=True),
            fd_inputs(model, ROWS, params),
        )

    def test_full_iqp_reads_every_pair(self):
        model = qml.Model(
            qml.Encoder(4).iqp_encoder("full"),
            qml.Ansatz(4).real_amplitudes(1),
            readouts()["z"],
        )
        params = point(model)
        assert_close(
            qml.jacobian(model, ROWS, params, wrt="inputs", exact=True),
            fd_inputs(model, ROWS, params),
        )

    def test_the_chain_rule_over_it_is_the_gradient_of_a_loss(self):
        # A squared-error loss composed in numpy: its gradient through the
        # Jacobian must be the finite difference of the loss itself.
        encoder, ansatz = layers()["reuploading"]
        model = qml.Model(encoder, ansatz, readouts()["weighted"])
        y = np.array([0.5, -0.3, 0.8])
        params = point(model)

        def loss(theta):
            values = qml.expval(model, ROWS, theta, exact=True)[:, 0]
            return np.mean((values - y) ** 2)

        values = qml.expval(model, ROWS, params, exact=True)[:, 0]
        by_params = qml.jacobian(model, ROWS, params, exact=True)[:, 0, :]
        gradient = 2 * (values - y) @ by_params / len(ROWS)
        reference = []
        for p in range(len(params)):
            plus, minus = params.copy(), params.copy()
            plus[p] += STEP
            minus[p] -= STEP
            reference.append((loss(plus) - loss(minus)) / (2 * STEP))
        assert np.max(np.abs(gradient - np.array(reference))) < TOLERANCE


# ─────────────────────────────────────────────────────────────────────────────
# Blocks, shots and seeds (criteria 5, 6)
# ─────────────────────────────────────────────────────────────────────────────


def _many_rows(n):
    return [[0.1 + 0.37 * ((i * 4 + j) % 11) for j in range(4)] for i in range(n)]


class TestBlocksAndSeeds:
    def test_a_call_is_the_concatenation_of_calls_on_its_chunks(self):
        encoder, ansatz = layers()["iqp"]
        model = qml.Model(encoder, ansatz, readouts()["argmax"])
        params = point(model)
        rows = _many_rows(13)
        whole_params, whole_inputs = qml.jacobian(
            model, rows, params, wrt="both", exact=True
        )
        chunks = [rows[:1], rows[1:6], rows[6:13]]
        parts = [qml.jacobian(model, c, params, wrt="both", exact=True) for c in chunks]
        assert np.array_equal(whole_params, np.concatenate([p for p, _ in parts]))
        assert np.array_equal(whole_inputs, np.concatenate([i for _, i in parts]))
        assert np.array_equal(
            qml.expval(model, rows, params, exact=True),
            np.concatenate([qml.expval(model, c, params, exact=True) for c in chunks]),
        )

    def test_a_seed_replays_another_differs_and_none_is_drawn(self):
        encoder, ansatz = layers()["angle-ry"]
        model = qml.Model(encoder, ansatz, readouts()["z"])
        params = point(model)
        first = qml.jacobian(model, ROWS, params, wrt="both", shots=256, seed=5)
        again = qml.jacobian(model, ROWS, params, wrt="both", shots=256, seed=5)
        other = qml.jacobian(model, ROWS, params, wrt="both", shots=256, seed=6)
        for a, b in zip(first, again):
            assert np.array_equal(a, b)
        assert not np.array_equal(first[0], other[0])
        # Without a seed, one is drawn per call: two calls differ.
        drawn = [qml.jacobian(model, ROWS, params, shots=256) for _ in range(2)]
        assert not np.array_equal(drawn[0], drawn[1])

    def test_a_sampled_jacobian_estimates_the_exact_one(self):
        encoder, ansatz = layers()["angle-ry"]
        model = qml.Model(encoder, ansatz, readouts()["z"])
        params = point(model)
        shots = 20_000
        exact = qml.jacobian(model, ROWS, params, exact=True)
        sampled = qml.jacobian(model, ROWS, params, shots=shots, seed=3)
        # Each entry is half a difference of two independent ±1 means.
        assert np.max(np.abs(sampled - exact)) < 5 / math.sqrt(2 * shots)

    def test_exact_needs_the_local_native_simulator(self):
        model = qml.Model(*layers()["angle-ry"], readouts()["z"])
        for kwargs in [{"backend": "aer"}, {"infrastructure": "cunqa"}]:
            with pytest.raises(polypus.ConfigurationError, match="exact=True"):
                qml.expval(model, ROWS, point(model), exact=True, **kwargs)
            with pytest.raises(polypus.ConfigurationError, match="exact=True"):
                qml.jacobian(model, ROWS, point(model), exact=True, **kwargs)

    def test_a_seed_on_qmio_is_refused(self):
        model = qml.Model(*layers()["angle-ry"], readouts()["z"])
        for primitive in (qml.expval, qml.jacobian):
            with pytest.raises(
                polypus.ConfigurationError, match="seed is not supported"
            ):
                primitive(model, ROWS, point(model), infrastructure="qmio", seed=1)


# ─────────────────────────────────────────────────────────────────────────────
# Rejections
# ─────────────────────────────────────────────────────────────────────────────


class TestRejections:
    def _model(self, layer="angle-ry"):
        return qml.Model(*layers()[layer], readouts()["z"])

    @pytest.mark.parametrize("primitive", [qml.expval, qml.jacobian])
    def test_model_must_be_a_model(self, primitive):
        model = self._model()
        params = list(point(model))
        with pytest.raises(TypeError, match="pass trained.model and trained.params"):
            primitive(model.bind(params), ROWS, params)
        with pytest.raises(TypeError, match="polypus.qml.Model"):
            primitive("model", ROWS, params)

    def test_wrt_is_a_closed_set(self):
        model = self._model()
        with pytest.raises(
            polypus.ConfigurationError, match='"params", "inputs" or "both"'
        ):
            qml.jacobian(model, ROWS, point(model), wrt="theta")

    @pytest.mark.parametrize("primitive", [qml.expval, qml.jacobian])
    def test_params_are_checked_as_bind_checks_them(self, primitive):
        model = self._model()
        params = list(point(model))
        with pytest.raises(polypus.ConfigurationError, match="params has"):
            primitive(model, ROWS, params[:-1], exact=True)
        params[2] = math.nan
        with pytest.raises(polypus.ConfigurationError, match=r"params\[2\]"):
            primitive(model, ROWS, params, exact=True)

    @pytest.mark.parametrize("primitive", [qml.expval, qml.jacobian])
    def test_rows_are_checked_against_the_model(self, primitive):
        model = self._model()
        params = point(model)
        with pytest.raises(polypus.ConfigurationError):
            primitive(model, [row + [0.1] for row in ROWS], params, exact=True)
        with pytest.raises(polypus.ConfigurationError):
            primitive(model, [ROWS[0], ROWS[1][:3]], params, exact=True)
        with pytest.raises(polypus.ConfigurationError):
            primitive(model, [[math.inf, 0.1, 0.2, 0.3]], params, exact=True)

    @pytest.mark.parametrize(
        ("encoder", "rows", "name"),
        [
            (lambda: qml.Encoder(2).amplitude_encoder(), ROWS, "amplitude"),
            (
                lambda: qml.Encoder(4).basis_encoder(),
                [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
                "basis",
            ),
        ],
    )
    def test_amplitude_and_basis_have_no_input_jacobian(self, encoder, rows, name):
        width = 2 if name == "amplitude" else 4
        model = qml.Model(
            encoder(), qml.Ansatz(width).real_amplitudes(1), qml.Readout([qml.Z(0)])
        )
        params = point(model)
        for wrt in ["inputs", "both"]:
            with pytest.raises(polypus.ConfigurationError, match=f"{name} encoder"):
                qml.jacobian(model, rows, params, wrt=wrt, exact=True)
        # With respect to the parameters, any model.
        assert_close(
            qml.jacobian(model, rows, params, exact=True),
            fd_params(model, rows, params),
        )
