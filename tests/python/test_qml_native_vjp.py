"""
The vector-Jacobian product of native QML (issue #559; contract C-11,
"Expectation and Jacobian primitives"): ``polypus.qml.vjp``.

``qml.vjp(model, x, params, cotangent, wrt=...)`` must be ``qml.jacobian``
contracted with ``cotangent`` — ``einsum("bo,bok->k")`` for the parameters,
``einsum("bo,bof->bf")`` for the inputs — for every layer, readout and encoder
the Jacobian takes. With ``exact=True`` it comes from one adjoint run per row
whatever the number of observables, so it is the contraction to rounding
(``TOLERANCE``); with shots it runs ``jacobian``'s circuits, so with the same
``seed`` it is the contraction of that seed's Jacobian, to rounding too. The
cotangent is checked for its shape and for being real, its values read as
``x``'s are; a ``NaN`` or an infinity propagates as it does through the
contraction. Every check runs before a backend exists.
"""

import math

import numpy as np
import polypus
import pytest
from polypus import qml
from test_qml_native_expval_jacobian import ROWS, layers, point, readouts

# Exact either way; the adjoint sums run in another order than the contraction.
TOLERANCE = 1e-9

PARAMS = "bo,bok->k"
INPUTS = "bo,bof->bf"

# Every encoder the input Jacobian takes: angle (each axis, re-uploaded) and
# IQP, on its own and re-uploaded.
INPUT_LAYERS = ["angle-rx", "angle-ry", "angle-rz", "reuploading", "iqp", "iqp-single"]


def all_layers():
    return {
        **layers(),
        "iqp-single": (
            qml.Encoder(4).iqp_encoder("full"),
            qml.Ansatz(4).real_amplitudes(1),
        ),
    }


def all_readouts():
    """`readouts()` plus a threshold decision: raw, sign, threshold, a weighted
    `Observable` and a three-class argmax."""
    return {
        **readouts(),
        "threshold": qml.Readout(
            [qml.Z(0) @ qml.Z(3)], decision="threshold", threshold=0.1
        ),
    }


def model_of(layer, readout):
    encoder, ansatz = all_layers()[layer]
    return qml.Model(encoder, ansatz, all_readouts()[readout])


def cotangent(model, rows=len(ROWS)):
    """Neither zero nor repeated: `(rows, observables)`."""
    k = model.num_observables
    return np.array([[0.9 - 0.37 * (r * k + o) for o in range(k)] for r in range(rows)])


def contraction(model, x, params, v, wrt, **kwargs):
    """`qml.jacobian` contracted with `v`, as the PyTorch backward pass did."""
    if wrt == "params":
        return np.einsum(PARAMS, v, qml.jacobian(model, x, params, **kwargs))
    if wrt == "inputs":
        return np.einsum(
            INPUTS, v, qml.jacobian(model, x, params, wrt="inputs", **kwargs)
        )
    by_params, by_inputs = qml.jacobian(model, x, params, wrt="both", **kwargs)
    return np.einsum(PARAMS, v, by_params), np.einsum(INPUTS, v, by_inputs)


def assert_close(got, expected, tolerance=TOLERANCE):
    if isinstance(expected, tuple):
        assert isinstance(got, tuple) and len(got) == 2
        for g, e in zip(got, expected):
            assert_close(g, e, tolerance)
        return
    assert got.shape == expected.shape
    worst = np.max(np.abs(got - expected), initial=0.0)
    assert worst < tolerance, f"vjp and the contracted Jacobian differ by {worst}"


# ─────────────────────────────────────────────────────────────────────────────
# Shapes and types
# ─────────────────────────────────────────────────────────────────────────────


class TestShapesAndTypes:
    def _model(self):
        return model_of("iqp", "argmax")

    def _assert_array(self, array, shape):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags["C_CONTIGUOUS"]
        assert array.shape == shape

    def test_shapes_follow_wrt(self):
        model = self._model()
        params, v = point(model), cotangent(model)
        self._assert_array(
            qml.vjp(model, ROWS, params, v, exact=True), (model.num_params,)
        )
        self._assert_array(
            qml.vjp(model, ROWS, params, v, wrt="inputs", exact=True), (3, 4)
        )
        both = qml.vjp(model, ROWS, params, v, wrt="both", exact=True)
        assert isinstance(both, tuple) and len(both) == 2
        self._assert_array(both[0], (model.num_params,))
        self._assert_array(both[1], (3, 4))

    def test_both_is_the_two_separate_calls_bit_for_bit(self):
        model = self._model()
        params, v = point(model), cotangent(model)
        by_params, by_inputs = qml.vjp(model, ROWS, params, v, wrt="both", exact=True)
        assert np.array_equal(by_params, qml.vjp(model, ROWS, params, v, exact=True))
        assert np.array_equal(
            by_inputs, qml.vjp(model, ROWS, params, v, wrt="inputs", exact=True)
        )

    def test_the_cotangent_takes_the_containers_x_takes(self):
        model = self._model()
        params, v = point(model), cotangent(model)
        reference = qml.vjp(model, ROWS, params, v, wrt="both", exact=True)
        for x in [np.array(ROWS), tuple(ROWS), qml.Dataset(ROWS)]:
            for c in [
                v.tolist(),
                tuple(tuple(row) for row in v.tolist()),
                np.asfortranarray(v),
            ]:
                got = qml.vjp(model, x, params, c, wrt="both", exact=True)
                for g, r in zip(got, reference):
                    assert np.array_equal(g, r)
        # Another real dtype is converted to float64 at the boundary.
        as_float32 = v.astype(np.float32)
        got = qml.vjp(model, ROWS, params, as_float32, exact=True)
        assert np.array_equal(
            got, qml.vjp(model, ROWS, params, as_float32.astype(np.float64), exact=True)
        )

    def test_one_observable(self):
        model = model_of("angle-ry", "z")
        assert model.num_observables == 1
        params, v = point(model), cotangent(model)
        assert_close(
            qml.vjp(model, ROWS, params, v, wrt="both", exact=True),
            contraction(model, ROWS, params, v, "both", exact=True),
        )


# ─────────────────────────────────────────────────────────────────────────────
# The product is the contracted Jacobian (criterion 1)
# ─────────────────────────────────────────────────────────────────────────────


class TestVjpIsTheContractedJacobian:
    @pytest.mark.parametrize("layer", sorted(all_layers()))
    @pytest.mark.parametrize("readout", sorted(all_readouts()))
    def test_by_params(self, layer, readout):
        model = model_of(layer, readout)
        params, v = point(model), cotangent(model)
        assert_close(
            qml.vjp(model, ROWS, params, v, exact=True),
            contraction(model, ROWS, params, v, "params", exact=True),
        )

    def test_pooling_shares_its_parameters_across_pairs(self):
        # Pooling leaves qubits 0 and 1 to read out; the convolution and the
        # pooling reuse their parameters across pairs.
        model = qml.Model(
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).conv("basic").pool("basic"),
            qml.Readout([qml.Z(0), qml.Z(1)], decision="argmax"),
        )
        params, v = point(model, offset=-1.1), cotangent(model)
        for wrt in ["params", "both"]:
            assert_close(
                qml.vjp(model, ROWS, params, v, wrt=wrt, exact=True),
                contraction(model, ROWS, params, v, wrt, exact=True),
            )

    @pytest.mark.parametrize("layer", INPUT_LAYERS)
    @pytest.mark.parametrize("readout", sorted(all_readouts()))
    @pytest.mark.parametrize("wrt", ["inputs", "both"])
    def test_by_inputs_and_both(self, layer, readout, wrt):
        model = model_of(layer, readout)
        params, v = point(model), cotangent(model)
        assert_close(
            qml.vjp(model, ROWS, params, v, wrt=wrt, exact=True),
            contraction(model, ROWS, params, v, wrt, exact=True),
        )

    @pytest.mark.parametrize("wrt", ["params", "inputs", "both"])
    def test_with_shots_it_contracts_the_jacobian_of_the_same_seed(self, wrt):
        model = model_of("reuploading", "argmax")
        params, v = point(model), cotangent(model)
        got = qml.vjp(model, ROWS, params, v, wrt=wrt, shots=256, seed=5)
        assert_close(got, contraction(model, ROWS, params, v, wrt, shots=256, seed=5))
        # The seed matters: another one gives other circuits' counts.
        other = qml.vjp(model, ROWS, params, v, wrt=wrt, shots=256, seed=6)
        first = got[0] if wrt == "both" else got
        assert not np.array_equal(first, other[0] if wrt == "both" else other)

    def test_a_sampled_product_estimates_the_exact_one(self):
        model = model_of("angle-ry", "argmax")
        params, v = point(model), cotangent(model)
        shots = 20_000
        exact = qml.vjp(model, ROWS, params, v, exact=True)
        sampled = qml.vjp(model, ROWS, params, v, shots=shots, seed=3)
        # Each Jacobian entry is within 5/sqrt(2·shots); the product weighs
        # them by |v|, summed over rows and observables.
        bound = 5 / math.sqrt(2 * shots) * np.abs(v).sum()
        assert np.max(np.abs(sampled - exact)) < bound


# ─────────────────────────────────────────────────────────────────────────────
# The cotangent (criterion 5)
# ─────────────────────────────────────────────────────────────────────────────


class TestCotangent:
    def _model(self):
        return model_of("angle-ry", "argmax")

    def test_zeros_give_zeros(self):
        model = self._model()
        zeros = np.zeros((3, model.num_observables))
        by_params, by_inputs = qml.vjp(
            model, ROWS, point(model), zeros, wrt="both", exact=True
        )
        assert not by_params.any() and not by_inputs.any()

    @pytest.mark.parametrize("exact", [True, False], ids=["exact", "sampled"])
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_value_propagates_like_the_contraction(self, exact, value):
        model = self._model()
        params, v = point(model), cotangent(model)
        v[1, 2] = value
        kwargs = {"exact": True} if exact else {"shots": 256, "seed": 9}
        got = qml.vjp(model, ROWS, params, v, wrt="both", **kwargs)
        expected = contraction(model, ROWS, params, v, "both", **kwargs)
        for g, e in zip(got, expected):
            assert np.array_equal(np.isnan(g), np.isnan(e))
            assert np.array_equal(np.isfinite(g), np.isfinite(e))
            finite = np.isfinite(e)
            assert np.max(np.abs(g[finite] - e[finite]), initial=0.0) < TOLERANCE
        # Only the row with the value is touched in the input part.
        assert not np.isfinite(got[1][1]).any()
        assert np.isfinite(np.delete(got[1], 1, axis=0)).all()

    def test_finite_values_that_overflow_the_weight_are_the_contraction(self):
        # Two copies of ⟨Z₀⟩ weighed by 1e308 each: every value is finite, but
        # their combined weight, 2e308, is not; the product is still the
        # contraction, finite where it is, rather than an error.
        model = qml.Model(
            qml.Encoder(4).angle_encoder("ry"),
            qml.Ansatz(4).hardware_efficient(1),
            qml.Readout([qml.Z(0), qml.Z(0)], decision="argmax"),
        )
        params = point(model)
        v = np.full((3, 2), 1e308)
        got = qml.vjp(model, ROWS, params, v, wrt="both", exact=True)
        expected = contraction(model, ROWS, params, v, "both", exact=True)
        for g, e in zip(got, expected):
            assert np.array_equal(np.isfinite(g), np.isfinite(e))
            finite = np.isfinite(e)
            assert finite.any()
            np.testing.assert_allclose(g[finite], e[finite], rtol=1e-9, atol=0)

    def test_complex_values_are_refused(self):
        model = self._model()
        v = cotangent(model).astype(complex)
        v[0, 0] = 1 + 2j
        with pytest.raises(polypus.ConfigurationError, match="real values"):
            qml.vjp(model, ROWS, point(model), v, exact=True)
        # Even with no imaginary part: the dtype decides, not the values.
        with pytest.raises(polypus.ConfigurationError, match="real values"):
            qml.vjp(model, ROWS, point(model), cotangent(model).astype(complex))

    @pytest.mark.parametrize(
        ("bad", "got"),
        [
            (lambda v: v[:2], r"\(2, 3\)"),
            (lambda v: v[:, :2], r"\(3, 2\)"),
            (lambda v: v.ravel(), r"\(9,\)"),
            (lambda v: v[None], r"\(1, 3, 3\)"),
            (lambda v: 1.0, r"\(\)"),
        ],
    )
    def test_the_shape_must_be_rows_by_observables(self, bad, got):
        model = self._model()
        v = cotangent(model)
        with pytest.raises(
            polypus.ConfigurationError,
            match=r"cotangent must have shape \(rows, observables\) = \(3, 3\), got "
            + got,
        ):
            qml.vjp(model, ROWS, point(model), bad(v), exact=True)

    def test_ragged_rows_are_a_configuration_error(self):
        model = self._model()
        ragged = [[1.0, 2.0, 3.0]] * 2 + [[1.0]]
        with pytest.raises(polypus.ConfigurationError, match="cotangent"):
            qml.vjp(model, ROWS, point(model), ragged, exact=True)

    @pytest.mark.parametrize(
        "value",
        [
            True,
            1,
            np.int64(2),
            np.float32(0.5),
            np.bool_(False),
            "0.5",
            np.str_("0.5"),
            None,
            object(),
        ],
        ids=repr,
    )
    @pytest.mark.parametrize("as_array", [False, True], ids=["lists", "array"])
    def test_the_cotangent_takes_the_values_x_takes(self, value, as_array):
        # DBFerPra's review of #577: the cotangent went through
        # `numpy.asarray(..., "float64")`, which read "0.5" as 0.5 and None as
        # NaN where `x` raises a `TypeError`. Each value is now accepted or
        # refused exactly as `x` accepts or refuses it, a `bool` included.
        model = self._model()

        def filled(columns):
            rows = [[value] * columns for _ in ROWS]
            return np.array(rows) if as_array else rows

        def outcome(call):
            try:
                return call()
            except TypeError as error:
                return type(error)

        x_reads = outcome(
            lambda: qml.expval(model, filled(4), point(model), exact=True)
        )
        v_reads = outcome(
            lambda: qml.vjp(model, ROWS, point(model), filled(3), exact=True)
        )
        assert (x_reads is TypeError) == (v_reads is TypeError)
        if v_reads is TypeError:
            with pytest.raises(TypeError, match="cotangent must hold real numbers"):
                qml.vjp(model, ROWS, point(model), filled(3), exact=True)
        else:
            # The same product as the value read as a float.
            as_float = np.full((len(ROWS), 3), float(value))
            assert np.array_equal(
                v_reads, qml.vjp(model, ROWS, point(model), as_float, exact=True)
            )

    def test_numpy_errors_other_than_the_structure_propagate(self):
        # Only NumPy's `ValueError` (ragged rows) is the cotangent's
        # `ConfigurationError`; a `MemoryError` stays one.
        class Exhausting:
            def __array__(self, dtype=None, copy=None):
                raise MemoryError("no room")

        model = self._model()
        with pytest.raises(MemoryError, match="no room"):
            qml.vjp(model, ROWS, point(model), Exhausting(), exact=True)


# ─────────────────────────────────────────────────────────────────────────────
# Blocks (criterion 4)
# ─────────────────────────────────────────────────────────────────────────────


def _many_rows(n):
    return [[0.1 + 0.37 * ((i * 4 + j) % 11) for j in range(4)] for i in range(n)]


class TestBlocks:
    def test_a_call_is_the_sum_and_concatenation_of_calls_on_its_chunks(self):
        model = model_of("iqp", "argmax")
        params = point(model)
        rows = _many_rows(13)
        v = cotangent(model, rows=13)
        whole_params, whole_inputs = qml.vjp(
            model, rows, params, v, wrt="both", exact=True
        )
        bounds = [(0, 1), (1, 6), (6, 13)]
        parts = [
            qml.vjp(model, rows[a:b], params, v[a:b], wrt="both", exact=True)
            for a, b in bounds
        ]
        # The rows' input products are the rows', bit for bit; the parameter
        # product is their sum, to rounding (summed in another order).
        assert np.array_equal(whole_inputs, np.concatenate([i for _, i in parts]))
        assert_close(whole_params, sum(p for p, _ in parts), tolerance=1e-12)

    @pytest.mark.parametrize("value", [math.nan, 1e308], ids=["nan", "overflowing"])
    def test_a_row_that_cannot_be_combined_changes_no_other_row(self, value):
        # DBFerPra's review of #577: the route was decided per block, so a NaN
        # in row 0 changed the last bits of the rows sharing its block, and
        # the block size follows the host's cores. It is decided per row now:
        # the other rows' products are those of a call without row 0, bit for
        # bit, on any host.
        model = model_of("iqp", "argmax")
        params = point(model)
        rows = _many_rows(40)
        v = cotangent(model, rows=40)
        v[0, :] = value
        whole = qml.vjp(model, rows, params, v, wrt="inputs", exact=True)
        rest = qml.vjp(model, rows[1:], params, v[1:], wrt="inputs", exact=True)
        assert np.array_equal(whole[1:], rest)
        # Row 0 itself is the contraction of `qml.jacobian`, as always.
        expected = contraction(model, rows[:1], params, v[:1], "inputs", exact=True)
        assert np.array_equal(np.isnan(whole[:1]), np.isnan(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(whole[:1][finite], expected[finite], rtol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Rejections
# ─────────────────────────────────────────────────────────────────────────────


class TestRejections:
    def _model(self, layer="angle-ry"):
        return model_of(layer, "z")

    def _v(self, model, rows=len(ROWS)):
        return np.ones((rows, model.num_observables))

    def test_model_must_be_a_model(self):
        model = self._model()
        params = list(point(model))
        with pytest.raises(TypeError, match="pass trained.model and trained.params"):
            qml.vjp(model.bind(params), ROWS, params, self._v(model))
        with pytest.raises(TypeError, match="qml.vjp takes a polypus.qml.Model"):
            qml.vjp("model", ROWS, params, self._v(model))

    def test_wrt_is_a_closed_set(self):
        model = self._model()
        with pytest.raises(
            polypus.ConfigurationError, match='"params", "inputs" or "both"'
        ):
            qml.vjp(model, ROWS, point(model), self._v(model), wrt="theta")

    def test_params_and_rows_are_checked_as_for_jacobian(self):
        model = self._model()
        params = list(point(model))
        v = self._v(model)
        with pytest.raises(polypus.ConfigurationError, match="params has"):
            qml.vjp(model, ROWS, params[:-1], v, exact=True)
        params[2] = math.nan
        with pytest.raises(polypus.ConfigurationError, match=r"params\[2\]"):
            qml.vjp(model, ROWS, params, v, exact=True)
        with pytest.raises(polypus.ConfigurationError):
            qml.vjp(model, [row + [0.1] for row in ROWS], point(model), v, exact=True)

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
    def test_amplitude_and_basis_have_no_input_product(self, encoder, rows, name):
        width = 2 if name == "amplitude" else 4
        model = qml.Model(
            encoder(), qml.Ansatz(width).real_amplitudes(1), qml.Readout([qml.Z(0)])
        )
        params, v = point(model), self._v(model, len(rows))
        for wrt in ["inputs", "both"]:
            with pytest.raises(polypus.ConfigurationError, match=f"{name} encoder"):
                qml.vjp(model, rows, params, v, wrt=wrt, exact=True)
        # With respect to the parameters, any model.
        assert_close(
            qml.vjp(model, rows, params, v, exact=True),
            contraction(model, rows, params, v, "params", exact=True),
        )

    def test_exact_needs_the_local_native_simulator(self):
        model = self._model()
        for kwargs in [{"backend": "aer"}, {"infrastructure": "cunqa"}]:
            with pytest.raises(polypus.ConfigurationError, match="exact=True"):
                qml.vjp(model, ROWS, point(model), self._v(model), exact=True, **kwargs)

    def test_a_seed_on_qmio_is_refused(self):
        model = self._model()
        with pytest.raises(polypus.ConfigurationError, match="seed is not supported"):
            qml.vjp(
                model, ROWS, point(model), self._v(model), infrastructure="qmio", seed=1
            )
