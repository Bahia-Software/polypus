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
# Strongly entangling ansatz
# ─────────────────────────────────────────────────────────────────────────────


def _strongly_entangling_circuit(sample, params, num_qubits, ranges, entangler="cx"):
    """The bound circuit of an ``angle_encoder("ry")`` + ``strongly_entangling``
    model, built gate by gate with ``polypus.Circuit``: per layer, ``rz(φ)``,
    ``ry(θ)``, ``rz(ω)`` on every qubit (parameters in that order), then the
    entangler from every qubit ``i`` to ``(i + r) mod n``."""
    import polypus

    qc = polypus.Circuit(num_qubits)
    for qubit, x in enumerate(sample):
        qc.ry(qubit, x)
    angles = iter(params)
    for r in ranges:
        for qubit in range(num_qubits):
            qc.rz(qubit, next(angles))
            qc.ry(qubit, next(angles))
            qc.rz(qubit, next(angles))
        if num_qubits > 1:
            for i in range(num_qubits):
                getattr(qc, entangler)(i, (i + r) % num_qubits)
    return qc.measure_all()


class TestStronglyEntangling:
    """``strongly_entangling`` is PennyLane's ``StronglyEntanglingLayers``:
    ``3·n·reps`` parameters, a ``Rot`` per qubit and a ring of entanglers whose
    range changes per layer."""

    @pytest.mark.parametrize(
        ("num_qubits", "reps", "kwargs", "ranges"),
        [
            # Default ranges: (l mod (n-1)) + 1.
            (3, 2, {}, [1, 2]),
            (4, 4, {}, [1, 2, 3, 1]),
            (2, 1, {}, [1]),
            # One qubit: rotations only.
            (1, 2, {}, [None, None]),
            (3, 2, {"ranges": [2, 1]}, [2, 1]),
            (3, 1, {"entangler": "cz"}, [1]),
        ],
        ids=["n3-default", "n4-default", "n2", "n1", "explicit-ranges", "cz"],
    )
    def test_the_circuit_is_gate_for_gate(self, num_qubits, reps, kwargs, ranges):
        sample = [0.1 * (j + 1) for j in range(num_qubits)]
        model = _raw_model(
            num_qubits,
            lambda a: a.strongly_entangling(reps, **kwargs),
            encoder=None,
        )
        params = [0.05 * (i + 1) for i in range(model.num_params)]
        expected = _strongly_entangling_circuit(
            sample, params, num_qubits, ranges, kwargs.get("entangler", "cx")
        )
        assert model.circuit(sample, params).to_qasm2() == expected.to_qasm2()

    def test_the_count_is_three_per_qubit_per_layer(self):
        from polypus.qml import Ansatz

        _assert_param_count(_raw_model(3, lambda a: a.strongly_entangling(2)), 18)
        _assert_param_count(_raw_model(1, lambda a: a.strongly_entangling(3)), 9)
        # Explicit ranges change the circuit, not the count.
        _assert_param_count(
            _raw_model(4, lambda a: a.strongly_entangling(2, ranges=[3, 3])), 24
        )
        # After pooling, n is the qubits still active: 4 → 2.
        assert Ansatz(4).pool("basic").strongly_entangling(2).num_params == 3 + 12
        assert Ansatz(3).strongly_entangling(0).num_params == 0

    def test_explicit_default_ranges_build_the_same_circuit(self):
        theta = [0.1 * (i + 1) for i in range(36)]
        implicit = _raw_model(4, lambda a: a.strongly_entangling(3))
        explicit = _raw_model(
            4, lambda a: a.strongly_entangling(3, ranges=[1, 2, 3], entangler="cx")
        )
        assert (
            implicit.circuit(list(_SAMPLE_4F), theta).to_qasm2()
            == explicit.circuit(list(_SAMPLE_4F), theta).to_qasm2()
        )

    @pytest.mark.parametrize(
        ("build", "kwargs", "match"),
        [
            (lambda a: a, {"reps": 2, "ranges": [1]}, "1 value.*reps is 2"),
            (lambda a: a, {"reps": 1, "ranges": [1, 1]}, "2 value.*reps is 1"),
            (lambda a: a, {"reps": 2, "ranges": [1, 0]}, "range 0 of layer 1"),
            (lambda a: a, {"reps": 1, "ranges": [4]}, r"range 4 .*1\.\.=3"),
            # After pooling only 2 qubits are active, so 2 is out of bounds.
            (
                lambda a: a.pool("basic"),
                {"reps": 1, "ranges": [2]},
                r"2 active qubit\(s\)",
            ),
            (lambda a: a, {"reps": 1, "entangler": "cy"}, '"cx" or "cz"'),
        ],
        ids=["too-few", "too-many", "zero", "n", "after-pool", "entangler"],
    )
    def test_invalid_options_are_rejected_at_the_call(self, build, kwargs, match):
        from polypus.qml import Ansatz

        ansatz = build(Ansatz(4).conv("basic"))
        before = (ansatz.num_layers, ansatz.num_params)
        with pytest.raises(ValueError, match=match):
            ansatz.strongly_entangling(**kwargs)
        assert (ansatz.num_layers, ansatz.num_params) == before

    def test_a_single_qubit_checks_only_the_count(self):
        from polypus.qml import Ansatz

        # No entangler on one qubit, so no range can be wrong; only one per
        # layer is required.
        assert Ansatz(1).strongly_entangling(2, ranges=[0, 5]).num_params == 6
        with pytest.raises(ValueError, match="reps is 2"):
            Ansatz(1).strongly_entangling(2, ranges=[1])

    def test_negative_values_fail_like_a_negative_reps(self):
        from polypus.qml import Ansatz

        # A negative range is refused at the call, before the layer is checked,
        # with the same `ValueError` as `hardware_efficient(-1)`.
        for build in (
            lambda a: a.hardware_efficient(-1),
            lambda a: a.strongly_entangling(-1),
            lambda a: a.strongly_entangling(1, ranges=[-1]),
        ):
            with pytest.raises(ValueError, match="non-negative integer, got -1"):
                build(Ansatz(3))

    def test_counts_that_do_not_fit_are_value_errors(self):
        from polypus.qml import Ansatz

        ansatz = Ansatz(2).real_amplitudes(1)
        for build, match in (
            (lambda a: a.strongly_entangling(2**70), f"got {2**70}"),
            (lambda a: a.strongly_entangling(1, ranges=[2**70]), f"got {2**70}"),
            # 3 × 2 qubits × 2**62 parameters do not fit in a usize.
            (lambda a: a.strongly_entangling(2**62), "layer 1 needs more trainable"),
        ):
            with pytest.raises(ValueError, match=match):
                build(ansatz)
            assert (ansatz.num_layers, ansatz.num_params) == (1, 4)
        # Another type keeps its `TypeError`.
        with pytest.raises(TypeError):
            ansatz.strongly_entangling("3")
        with pytest.raises(TypeError):
            ansatz.strongly_entangling(1, ranges=["1"])

    def test_it_is_an_ansatz_layer_only(self):
        from polypus.qml import Encoder

        assert not hasattr(Encoder(2), "strongly_entangling")


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


_COUNT_IDS = ["hardware-efficient", "real-amplitudes"]


class TestCountArguments:
    """``reps`` is a count. A negative or oversized one, and a layer whose
    parameter count does not fit, are a ``ValueError`` at the call that leaves
    the ``Ansatz`` unchanged, never an ``OverflowError`` or a panic (contract
    C-11)."""

    @pytest.mark.parametrize(
        "build",
        [lambda a: a.hardware_efficient(-1), lambda a: a.real_amplitudes(-1)],
        ids=_COUNT_IDS,
    )
    def test_a_negative_reps_is_a_value_error(self, build):
        from polypus.qml import Ansatz

        ansatz = Ansatz(2).real_amplitudes(1)
        with pytest.raises(ValueError, match="non-negative integer, got -1"):
            build(ansatz)
        assert (ansatz.num_layers, ansatz.num_params) == (1, 4)

    @pytest.mark.parametrize(
        "build",
        [lambda a: a.hardware_efficient(2**70), lambda a: a.real_amplitudes(2**70)],
        ids=_COUNT_IDS,
    )
    def test_an_integer_beyond_any_count_is_a_value_error(self, build):
        from polypus.qml import Ansatz

        ansatz = Ansatz(2).real_amplitudes(1)
        with pytest.raises(ValueError, match=f"got {2**70}"):
            build(ansatz)
        assert (ansatz.num_layers, ansatz.num_params) == (1, 4)

    @pytest.mark.parametrize(
        "build",
        [
            # (2**63 + 1) rotation blocks × 2 axes × 2 qubits.
            lambda a: a.hardware_efficient(2**63),
            # (2**63 + 1) rotation blocks × 2 qubits.
            lambda a: a.real_amplitudes(2**63),
        ],
        ids=_COUNT_IDS,
    )
    def test_a_parameter_count_that_does_not_fit_is_a_value_error(self, build):
        from polypus.qml import Ansatz

        ansatz = Ansatz(2).real_amplitudes(1)
        with pytest.raises(ValueError, match="layer 1 needs more trainable parameters"):
            build(ansatz)
        assert (ansatz.num_layers, ansatz.num_params) == (1, 4)

    def test_a_layer_after_a_full_parameter_space_is_a_value_error(self):
        from polypus.qml import Ansatz

        # 2·(2**63 − 1) = 2**64 − 2 parameters still fit in a usize; the four
        # of the convolution after them do not.
        ansatz = Ansatz(2).hardware_efficient(
            2**63 - 1, rotations=["rx"], final_rotation_layer=False
        )
        assert ansatz.num_params == 2**64 - 2
        with pytest.raises(ValueError, match="layer 1 needs more trainable parameters"):
            ansatz.conv("basic")
        assert (ansatz.num_layers, ansatz.num_params) == (1, 2**64 - 2)

    @pytest.mark.parametrize(
        "build",
        [lambda a: a.hardware_efficient("3"), lambda a: a.real_amplitudes("3")],
        ids=_COUNT_IDS,
    )
    def test_a_reps_of_another_type_is_still_a_type_error(self, build):
        from polypus.qml import Ansatz

        with pytest.raises(TypeError):
            build(Ansatz(2))


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
