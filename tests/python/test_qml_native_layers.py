"""
What each native layer and readout observable computes (contract C-11), checked
without an optimizer through two properties of exact inference:

- ``TrainedModel.predict(..., exact=True)`` on a ``decision="raw"`` readout is a
  deterministic function of the circuit: the exact ``⟨O⟩`` of the bound state, no
  shot noise. Two spellings that build the *same* circuit agree bit for bit; two
  that build different circuits disagree. (``"raw"`` rather than ``"sign"``,
  because a ±1 sign could coincide for two different circuits.)
- ``Model.num_params`` is the parameter count, and ``bind`` accepts exactly that
  many parameters and no other count.

Ported from the 71 branch's ``test_qml_native.py`` (builder coverage, the layers
without sugar, the weighted ``Observable`` and the ``PauliTerm`` spelling) onto
the ``Model`` API; the analytic expectations are unchanged.
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.vqc]

# The default sample width: two features, like the 71's two-feature dataset.
_SAMPLE_2F = (0.1, 0.2)
_SAMPLE_4F = (0.7, 1.1, 0.4, 1.3)


def _raw_model(num_qubits, ansatz, encoder=None, observables=None, decision="raw"):
    """A model whose ansatz is built by ``ansatz(Ansatz(num_qubits))``, reading
    ``⟨Z₀⟩`` (or ``observables``) out unchanged by default."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        encoder or Encoder(num_qubits).angle_encoder("ry"),
        ansatz(Ansatz(num_qubits)),
        Readout(observables or [Z(0)], decision=decision),
    )


def _exact(model, params, sample=_SAMPLE_2F):
    """The exact prediction of ``model`` bound to ``params`` on one sample."""
    return model.bind(list(params)).predict([list(sample)], exact=True)[0]


def _assert_param_count(model, count):
    """``num_params`` is ``count``, and ``bind`` takes exactly that many."""
    assert model.num_params == count
    model.bind([0.1] * count)
    for wrong in (count - 1, count + 1):
        with pytest.raises(ValueError, match="params has"):
            model.bind([0.1] * wrong)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware-efficient ansatz
# ─────────────────────────────────────────────────────────────────────────────


class TestHardwareEfficientConfiguration:
    """``hardware_efficient`` exposes all five fields of the Rust struct; its
    defaults are Qiskit's ``TwoLocal`` ones."""

    # 3 qubits, reps=1, default [ry, rz] → 2 axes × 3 qubits × (1 + 1) blocks.
    THETA_3Q = [0.1 * (i + 1) for i in range(12)]

    def test_explicit_defaults_build_the_same_circuit(self):
        implicit = _raw_model(3, lambda a: a.hardware_efficient(1))
        explicit = _raw_model(
            3,
            lambda a: a.hardware_efficient(
                1,
                rotations=["ry", "rz"],
                entangler="cx",
                entanglement="linear",
                final_rotation_layer=True,
            ),
        )
        assert _exact(implicit, self.THETA_3Q) == _exact(explicit, self.THETA_3Q)

    def test_default_count_is_axes_times_qubits_times_blocks(self):
        _assert_param_count(_raw_model(3, lambda a: a.hardware_efficient(1)), 12)
        _assert_param_count(_raw_model(2, lambda a: a.hardware_efficient(1)), 8)

    def test_rotations_set_the_count(self):
        one = _raw_model(2, lambda a: a.hardware_efficient(1, rotations=["ry"]))
        three = _raw_model(
            2, lambda a: a.hardware_efficient(1, rotations=["rx", "ry", "rz"])
        )
        _assert_param_count(one, 4)
        _assert_param_count(three, 12)

    def test_no_final_rotation_layer_drops_one_block(self):
        model = _raw_model(
            2, lambda a: a.hardware_efficient(1, final_rotation_layer=False)
        )
        _assert_param_count(model, 4)

    def test_entanglement_patterns_build_different_circuits(self):
        # On 3 qubits: linear {(0,1),(1,2)}, circular adds (2,0), full has (0,2)
        # instead, so the same parameters give three different expectations.
        # Entanglers take no parameters, so the count is the same for all three.
        values = {}
        for pattern in ("linear", "circular", "full"):
            model = _raw_model(
                3, lambda a, p=pattern: a.hardware_efficient(1, entanglement=p)
            )
            _assert_param_count(model, 12)
            values[pattern] = _exact(model, self.THETA_3Q)
        assert len(set(values.values())) == 3, values

    def test_cz_and_cx_build_different_circuits(self):
        cx = _raw_model(3, lambda a: a.hardware_efficient(1, entangler="cx"))
        cz = _raw_model(3, lambda a: a.hardware_efficient(1, entangler="cz"))
        assert _exact(cx, self.THETA_3Q) != _exact(cz, self.THETA_3Q)

    def test_real_amplitudes_is_the_single_ry_preset(self):
        preset = _raw_model(2, lambda a: a.real_amplitudes(1))
        explicit = _raw_model(
            2,
            lambda a: a.hardware_efficient(
                1,
                rotations=["ry"],
                entangler="cx",
                entanglement="linear",
                final_rotation_layer=True,
            ),
        )
        theta = [0.3, 0.7, 1.1, 1.5]
        _assert_param_count(preset, 4)
        assert _exact(preset, theta) == _exact(explicit, theta)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────


def _amplitude_model():
    from polypus.qml import Encoder

    return _raw_model(
        2,
        lambda a: a.hardware_efficient(1),
        encoder=Encoder(2).amplitude_encoder(),
    )


class TestAngleEncoder:
    def test_the_three_axes_build_different_circuits(self):
        from polypus.qml import Encoder

        theta = [0.1 * (i + 1) for i in range(8)]
        values = {
            axis: _exact(
                _raw_model(
                    2,
                    lambda a: a.hardware_efficient(1),
                    encoder=Encoder(2).angle_encoder(axis),
                ),
                theta,
            )
            for axis in ("rx", "ry", "rz")
        }
        assert len(set(values.values())) == 3, values


class TestAmplitudeEncoder:
    THETA = [0.1 * (i + 1) for i in range(8)]

    def test_encoding_is_scale_invariant(self):
        # The sample is normalised before it becomes amplitudes, so two positive
        # multiples of one vector prepare the identical state.
        model = _amplitude_model()
        assert _exact(model, self.THETA, (0.3, 0.4)) == _exact(
            model, self.THETA, (0.6, 0.8)
        )

    def test_the_angle_encoder_is_not(self):
        # Guards the test above against passing for a trivial reason, such as a
        # readout that ignores the encoder.
        model = _raw_model(2, lambda a: a.hardware_efficient(1))
        assert _exact(model, self.THETA, (0.3, 0.4)) != _exact(
            model, self.THETA, (0.6, 0.8)
        )

    def test_it_reserves_no_parameters(self):
        _assert_param_count(_amplitude_model(), 8)


def _iqp_model(entanglement):
    from polypus.qml import Encoder

    encoder = Encoder(4)
    encoder = (
        encoder.iqp_encoder()
        if entanglement is None
        else encoder.iqp_encoder(entanglement)
    )
    return _raw_model(4, lambda a: a.hardware_efficient(1), encoder=encoder)


class TestIqpEncoder:
    """IQP's ``Rzz`` gates are diagonal and symmetric in their qubits, so two
    patterns selecting the same unordered pairs build the same unitary. Hence
    four features: at three, ``circular`` {01,12,20} and ``full`` {01,02,12} are
    the same set, while at four ``full`` has six pairs and ``circular`` four."""

    THETA_4Q = [0.1 * (i + 1) for i in range(16)]

    def test_entanglement_patterns_build_different_circuits(self):
        values = {}
        for pattern in ("linear", "circular", "full"):
            model = _iqp_model(pattern)
            _assert_param_count(model, 16)
            values[pattern] = _exact(model, self.THETA_4Q, _SAMPLE_4F)
        assert len(set(values.values())) == 3, values

    def test_the_default_is_full(self):
        assert _exact(_iqp_model(None), self.THETA_4Q, _SAMPLE_4F) == _exact(
            _iqp_model("full"), self.THETA_4Q, _SAMPLE_4F
        )


# ─────────────────────────────────────────────────────────────────────────────
# QCNN convolution and pooling
# ─────────────────────────────────────────────────────────────────────────────


def _conv_model(num_qubits, block, pairing=None):
    return _raw_model(
        num_qubits,
        lambda a: a.conv(block) if pairing is None else a.conv(block, pairing),
    )


def _pool_model(num_qubits, keep=None, position=0):
    from polypus.qml import Z

    return _raw_model(
        num_qubits,
        lambda a: a.pool("basic") if keep is None else a.pool("basic", keep),
        observables=[Z(position)],
    )


class TestConvAndPool:
    def test_conv_shares_its_parameters(self):
        # One block per layer, whatever the qubit count: 4 for "basic", 3 for
        # "cartan", on 2 qubits and on 4 alike.
        for num_qubits in (2, 4):
            _assert_param_count(_conv_model(num_qubits, "basic"), 4)
            _assert_param_count(_conv_model(num_qubits, "cartan"), 3)

    def test_pairings_build_different_circuits(self):
        # On 4 qubits: even {(0,1),(2,3)}, odd {(1,2)}, alternating = both.
        theta = [0.3, 0.7, 1.1, 1.5]
        values = {
            pairing: _exact(_conv_model(4, "basic", pairing), theta)
            for pairing in ("even_pairs", "odd_pairs", "alternating")
        }
        assert len(set(values.values())) == 3, values

    def test_pool_shares_its_parameters(self):
        for num_qubits in (2, 4):
            _assert_param_count(_pool_model(num_qubits), 3)

    def test_pool_shrinks_the_active_qubits(self):
        # Pooling 4 qubits in pairs leaves 2: logical position 1 exists, 2 does
        # not, and the model says so when it is built.
        _assert_param_count(_pool_model(4, position=1), 3)
        with pytest.raises(ValueError, match="logical qubit 2"):
            _pool_model(4, position=2)

    def test_the_keep_rule_picks_the_survivor(self):
        # Logical position 0 is physical qubit 0 under "even_positions" and 1
        # under "odd_positions", and the block's roles swap with it.
        theta = [0.3, 0.7, 1.1]
        assert _exact(_pool_model(4, "even_positions"), theta) != _exact(
            _pool_model(4, "odd_positions"), theta
        )


# ─────────────────────────────────────────────────────────────────────────────
# Readout observables, against closed forms
# ─────────────────────────────────────────────────────────────────────────────
#
# `_product_model` binds to a known product state: `angle_encoder("ry")` puts
# Ry(x_j) on qubit j, and the ansatz is one layer of Ry with no entangler, so the
# state is Ry(x₀+θ₀)|0⟩ ⊗ Ry(x₁+θ₁)|0⟩. Every Z expectation is then a closed
# form: ⟨Z_j⟩ = cos(x_j + θ_j), and ⟨Z₀Z₁⟩ = ⟨Z₀⟩·⟨Z₁⟩.

_OBS_THETA = [0.7, 1.3]
_OBS_SAMPLE = [0.4, 0.9]


def _product_model(observables, decision="raw"):
    return _raw_model(
        2,
        lambda a: a.hardware_efficient(0, rotations=["ry"], final_rotation_layer=True),
        observables=observables,
        decision=decision,
    )


def _expected_z(sample, theta, positions):
    value = 1.0
    for j in positions:
        value *= math.cos(sample[j] + theta[j])
    return value


def _obs(observables, sample=_OBS_SAMPLE, decision="raw"):
    return _exact(_product_model(observables, decision), _OBS_THETA, sample)


class TestObservables:
    def test_three_spellings_of_one_term_agree_bit_for_bit(self):
        from polypus.qml import Observable, Z

        bare = _obs([[("z", 0), ("z", 1)]])
        assert bare == _obs([Z(0) @ Z(1)])
        assert bare == _obs([Observable([(1.0, [("z", 0), ("z", 1)])])])
        assert bare == _obs([Observable([(1.0, Z(0) @ Z(1))])])
        assert bare == pytest.approx(
            _expected_z(_OBS_SAMPLE, _OBS_THETA, [0, 1]), abs=1e-12
        )

    @pytest.mark.parametrize("pauli", ["z", "x", "y"])
    def test_each_constructor_is_its_own_pauli(self, pauli):
        import polypus.qml as qml

        term = getattr(qml, pauli.upper())(0)
        assert _obs([[(pauli, 0)]]) == _obs([term])

    def test_z0_matches_its_closed_form(self):
        from polypus.qml import Z

        assert _obs([Z(0)]) == pytest.approx(
            _expected_z(_OBS_SAMPLE, _OBS_THETA, [0]), abs=1e-12
        )

    def test_the_order_of_factors_does_not_matter(self):
        from polypus.qml import Z

        assert _obs([Z(1) @ Z(0)]) == _obs([Z(0) @ Z(1)])

    def test_a_weighted_sum_matches_its_closed_form(self):
        from polypus.qml import Observable, Z

        got = _obs([Observable([(0.5, Z(0)), (1.5, Z(0) @ Z(1))])])
        expected = 0.5 * _expected_z(_OBS_SAMPLE, _OBS_THETA, [0]) + 1.5 * _expected_z(
            _OBS_SAMPLE, _OBS_THETA, [0, 1]
        )
        assert got == pytest.approx(expected, abs=1e-12)
        # Genuinely a sum: neither term alone gives it.
        assert got != pytest.approx(_obs([Z(0)]), abs=1e-9)

    def test_coefficients_scale_linearly(self):
        from polypus.qml import Observable, Z

        single = _obs([Observable([(1.0, Z(0))])])
        doubled = _obs([Observable([(2.0, Z(0))])])
        assert doubled == pytest.approx(2.0 * single, abs=1e-12)

    def test_argmax_over_mixed_spellings(self):
        # Class 0 is the bare ⟨Z₀⟩, class 1 a weighted sum. Each sample is chosen
        # so a different class wins, so a silently dropped observable would show.
        from polypus.qml import Observable, Z

        observables = [[("z", 0)], Observable([(0.5, Z(0)), (1.5, Z(0) @ Z(1))])]
        for sample, expected in [([0.4, 0.9], 0), ([2.9, 0.2], 1)]:
            z0 = _expected_z(sample, _OBS_THETA, [0])
            z0z1 = _expected_z(sample, _OBS_THETA, [0, 1])
            scores = [z0, 0.5 * z0 + 1.5 * z0z1]
            assert max(range(2), key=scores.__getitem__) == expected
            assert _obs(observables, sample=sample, decision="argmax") == expected

    def test_argmax_over_all_three_spellings(self):
        # Class 0 a list, class 1 a `PauliTerm`, class 2 a weighted sum; one
        # sample per winner, so no observable can be dropped unnoticed.
        from polypus.qml import Observable, Z

        observables = [[("z", 0)], Z(1), Observable([(0.5, Z(0)), (1.5, Z(0) @ Z(1))])]
        winners = set()
        for sample, expected in [([0.0, 0.1], 0), ([0.8, 0.0], 1), ([1.0, 0.8], 2)]:
            z0 = _expected_z(sample, _OBS_THETA, [0])
            z1 = _expected_z(sample, _OBS_THETA, [1])
            z0z1 = _expected_z(sample, _OBS_THETA, [0, 1])
            scores = [z0, z1, 0.5 * z0 + 1.5 * z0z1]
            assert max(range(3), key=scores.__getitem__) == expected
            assert _obs(observables, sample=sample, decision="argmax") == expected
            winners.add(expected)
        assert winners == {0, 1, 2}

    def test_the_four_decisions_against_their_closed_forms(self):
        # On the product state ⟨Z₀⟩ = cos(1.1) ≈ 0.454 and ⟨Z₁⟩ = cos(2.2) < 0.
        from polypus.qml import Ansatz, Encoder, Model, Readout, Z

        def decide(decision, threshold=None):
            model = Model(
                Encoder(2).angle_encoder("ry"),
                Ansatz(2).hardware_efficient(0, rotations=["ry"]),
                Readout([Z(0), Z(1)], decision=decision, threshold=threshold),
            )
            return _exact(model, _OBS_THETA, _OBS_SAMPLE)

        z0 = _expected_z(_OBS_SAMPLE, _OBS_THETA, [0])
        z1 = _expected_z(_OBS_SAMPLE, _OBS_THETA, [1])
        assert z0 > 0.0 > z1
        assert decide("raw") == pytest.approx(z0, abs=1e-12)
        assert decide("sign") == 1
        assert decide("threshold", 0.25) == 1
        assert decide("threshold", 0.8) == -1
        assert decide("argmax") == 0

    @pytest.mark.parametrize("coefficient", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_coefficient_names_its_term(self, coefficient):
        from polypus.qml import Observable, Z

        with pytest.raises(
            ValueError, match="non-finite coefficient in observable term 1"
        ):
            Observable([(1.0, Z(0)), (coefficient, Z(0) @ Z(1))])

    def test_a_position_repeated_within_a_term(self):
        from polypus.qml import Observable

        with pytest.raises(ValueError, match="position"):
            Observable([(1.0, [("z", 0), ("x", 0)])])


# ─────────────────────────────────────────────────────────────────────────────
# Predictions read the results they come from
# ─────────────────────────────────────────────────────────────────────────────
#
# The 71's `predict_from_probabilities`/`predict_from_counts` are gone: a
# `TrainedModel` predicts from its own run. These pin the same arithmetic by
# reading `run`'s probabilities and counts back by hand. Keys follow C-3: qubit 0
# is the last character.


def _z0_of(distribution):
    total = sum(distribution.values())
    return (
        sum((1.0 if key[-1] == "0" else -1.0) * w for key, w in distribution.items())
        / total
    )


class TestPredictionsMatchTheirResults:
    THETA = [0.3, 0.7, 1.1, 1.5]
    SAMPLES = [[0.4, 0.9], [2.5, 0.3], [1.2, 2.2]]

    def test_exact_raw_predictions_are_the_probabilities_expectation(self):
        trained = _raw_model(2, lambda a: a.real_amplitudes(1)).bind(self.THETA)
        predictions = trained.predict(self.SAMPLES, exact=True)
        probabilities = trained.run(self.SAMPLES, exact=True).probabilities
        for prediction, row in zip(predictions, probabilities):
            assert prediction == pytest.approx(_z0_of(row), abs=1e-12)

    def test_sampled_sign_predictions_follow_the_seeded_counts(self):
        trained = _raw_model(2, lambda a: a.real_amplitudes(1), decision="sign").bind(
            self.THETA
        )
        predictions = trained.predict(self.SAMPLES, shots=512, seed=21)
        counts = trained.run(self.SAMPLES, shots=512, seed=21).counts
        assert predictions == [1 if _z0_of(row) >= 0 else -1 for row in counts]
