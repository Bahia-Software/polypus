"""
A native model's circuit (contract C-11): ``Model.circuit`` and
``TrainedModel.circuit`` return the ``polypus.Circuit`` the model runs for one
sample, with the features fixed, the readout's basis change and the terminal
measurement; symbolic (``theta_<i>`` free) or bound. The bound circuit is the
one ``TrainedModel.run`` executes: run on its own, it gives the same counts and
the same exact probabilities.
"""

import math

import pytest
from qml_native_helpers import make_model

# Two qubits, angle-encoded; `real_amplitudes(1)` has 4 parameters.
_SAMPLE = [0.3, 1.2]
_PARAMS = [0.1, -0.2, 0.3, -0.4]
_QCNN_SAMPLE = [0.7, 1.1, 0.4, 1.3]


def _model(pauli="Z", position=0):
    """The two-qubit model reading ``pauli`` at ``position``, with a sign
    decision."""
    from polypus import qml

    readout = qml.Readout([getattr(qml, pauli)(position)], decision="sign")
    return make_model(num_qubits=2, readout=readout)


def _qcnn_model():
    """Four qubits, a convolution and a pooling: their parameters are shared by
    several gates."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(4).angle_encoder("ry"),
        Ansatz(4).conv("basic").pool("basic"),
        Readout([Z(0)], decision="raw"),
    )


def _amplitude_model():
    """Up to four features in the amplitudes of two qubits."""
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    return Model(
        Encoder(2).amplitude_encoder(),
        Ansatz(2).hardware_efficient(1),
        Readout([Z(0)], decision="raw"),
    )


def _params(model):
    """Fixed, distinct values for every parameter of ``model``."""
    return [0.1 * (i + 1) * (-1) ** i for i in range(model.num_params)]


def _gate_lines(qasm):
    """The instruction lines of an OpenQASM 2.0 program: after the header and
    the register declarations."""
    lines = qasm.splitlines()
    return lines[lines.index("creg c[2];") + 1 :]


# ─────────────────────────────────────────────────────────────────────────────
# The symbolic circuit: Model.circuit(x)
# ─────────────────────────────────────────────────────────────────────────────


class TestSymbolic:
    def test_its_shape_is_the_model_s(self):
        import polypus

        model = _model()
        qc = model.circuit(_SAMPLE)
        assert isinstance(qc, polypus.Circuit)
        assert qc.num_qubits == model.num_qubits == 2
        assert qc.num_params == model.num_params == 4
        assert qc.param_names == ["theta_0", "theta_1", "theta_2", "theta_3"]
        # Two encoder rotations, real_amplitudes(1) (2 ry, cx, 2 ry) and the
        # terminal measure_all.
        assert len(qc) == 2 + 5 + 1
        assert qc.num_clbits == qc.num_qubits
        assert _gate_lines(qc.to_qasm2(_PARAMS))[-1] == "measure q -> c;"

    def test_the_features_are_fixed_and_the_parameters_free(self):
        text = _model().circuit(_SAMPLE).draw(fold=None)
        assert "Ry(0.3)" in text and "Ry(1.2)" in text
        for name in ["theta_0", "theta_1", "theta_2", "theta_3"]:
            assert f"Ry({name})" in text

    def test_binding_it_gives_the_bound_circuit(self):
        model = _model("Y")
        symbolic = model.circuit(_SAMPLE).to_qasm2(_PARAMS)
        assert symbolic == model.circuit(_SAMPLE, _PARAMS).to_qasm2()

    def test_numpy_input(self):
        import numpy as np

        model = _model()
        qasm = model.circuit(_SAMPLE, _PARAMS).to_qasm2()
        assert model.circuit(np.array(_SAMPLE), np.array(_PARAMS)).to_qasm2() == qasm

    def test_shared_parameters_keep_one_name(self):
        model = _qcnn_model()
        qc = model.circuit(_QCNN_SAMPLE)
        assert qc.num_params == model.num_params == 7  # conv 4 + pool 3
        assert qc.param_names == [f"theta_{i}" for i in range(7)]
        # The alternating pairing convolves three pairs, each with the block's
        # own parameters.
        assert qc.draw(fold=None).count("Ry(theta_0)") == 3

    def test_the_amplitude_encoder_takes_up_to_two_to_the_n_features(self):
        model = _amplitude_model()
        for sample in ([0.6, 0.8], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5, 0.5]):
            qc = model.circuit(sample)
            assert qc.num_params == model.num_params
            assert qc.num_clbits == 2


# ─────────────────────────────────────────────────────────────────────────────
# The bound circuit: Model.circuit(x, params) and TrainedModel.circuit(x)
# ─────────────────────────────────────────────────────────────────────────────


class TestBound:
    def test_every_angle_is_fixed(self):
        qc = _model().circuit(_SAMPLE, _PARAMS)
        assert qc.num_params == 0
        assert qc.param_names == []
        assert len(qc) == 8
        assert "theta" not in qc.draw(fold=None)

    def test_the_trained_model_s_circuit_is_the_model_s(self):
        model = _model("X")
        trained = model.bind(_PARAMS)
        qasm = trained.circuit(_SAMPLE).to_qasm2()
        assert qasm == trained.model.circuit(_SAMPLE, trained.params).to_qasm2()
        assert qasm == model.circuit(_SAMPLE, _PARAMS).to_qasm2()

    def test_the_model_is_unchanged(self):
        model = _model()
        first = model.circuit(_SAMPLE)
        model.circuit(_SAMPLE, _PARAMS)
        second = model.circuit(_SAMPLE)
        assert first is not second
        assert first.to_qasm2(_PARAMS) == second.to_qasm2(_PARAMS)
        assert model.num_params == 4
        assert model.bind(_PARAMS).params == _PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# The readout's basis change
# ─────────────────────────────────────────────────────────────────────────────


class TestBasisChange:
    @pytest.mark.parametrize(
        ("pauli", "change"),
        [("Z", []), ("X", ["h q[1];"]), ("Y", ["sdg q[1];", "h q[1];"])],
    )
    def test_the_measured_qubit_is_rotated_before_the_measurement(self, pauli, change):
        z_lines = _gate_lines(_model("Z", 1).circuit(_SAMPLE, _PARAMS).to_qasm2())
        lines = _gate_lines(_model(pauli, 1).circuit(_SAMPLE, _PARAMS).to_qasm2())
        assert lines == z_lines[:-1] + change + ["measure q -> c;"]
        assert len(_model(pauli, 1).circuit(_SAMPLE)) == 8 + len(change)

    def test_the_drawing_shows_it(self):
        text = _model("Y").circuit(_SAMPLE).draw(fold=None)
        assert "S†" in text and "┤ H ├" in text
        ascii_text = _model("Y").circuit(_SAMPLE).draw(ascii=True, fold=None)
        assert "| Sdg |" in ascii_text and "| H |" in ascii_text
        assert "┤ H ├" not in _model("Z").circuit(_SAMPLE).draw(fold=None)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────


class TestDrawing:
    @pytest.mark.parametrize("params", [None, _PARAMS])
    def test_every_drawing_works(self, params):
        qc = _model("X").circuit(_SAMPLE, params)
        text = qc.draw()
        assert text.splitlines()[1].startswith("q0: ")
        assert str(qc) == text
        assert qc.draw(ascii=True).isascii()
        assert qc.draw("svg").startswith("<svg")
        assert qc._repr_svg_() == qc.draw("svg")


# ─────────────────────────────────────────────────────────────────────────────
# The bound circuit is the one run executes
# ─────────────────────────────────────────────────────────────────────────────


def _exact_probabilities(qc, num_qubits):
    """The circuit's basis-state probabilities from its statevector, keyed
    like counts (qubit 0 rightmost) and leaving out exact zeros."""
    import polypus

    probabilities = {}
    for index, amplitude in enumerate(polypus.statevector(qc)):
        p = abs(amplitude) ** 2
        if p != 0.0:
            probabilities[format(index, f"0{num_qubits}b")] = p
    return probabilities


_MODELS = {
    "z": (lambda: _model("Z"), _SAMPLE),
    "x": (lambda: _model("X", 1), _SAMPLE),
    "y": (lambda: _model("Y"), _SAMPLE),
    "qcnn": (_qcnn_model, _QCNN_SAMPLE),
    "amplitude": (_amplitude_model, [1.0, 2.0, 3.0]),
}


@pytest.mark.integration
class TestReproducesRun:
    @pytest.mark.parametrize("name", list(_MODELS))
    def test_sampled_counts_are_identical(self, name):
        import polypus

        build, sample = _MODELS[name]
        model = build()
        trained = model.bind(_params(model))
        expected = trained.run([sample], shots=512, seed=11).counts[0]
        direct = polypus.run_quantum_circuit(
            trained.circuit(sample),
            shots=512,
            infrastructure="local",
            backend="polypus",
            seed=11,
        )
        assert direct.counts == [expected]

    @pytest.mark.parametrize("name", list(_MODELS))
    def test_exact_probabilities_match_the_statevector(self, name):
        build, sample = _MODELS[name]
        model = build()
        trained = model.bind(_params(model))
        expected = trained.run([sample], exact=True).probabilities[0]
        got = _exact_probabilities(trained.circuit(sample), model.num_qubits)
        assert got.keys() == expected.keys()
        for key, p in expected.items():
            assert got[key] == pytest.approx(p, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Rejections: before anything is built, never a panic
# ─────────────────────────────────────────────────────────────────────────────


def _circuit_calls():
    """``circuit`` on a model, with and without ``params``, and on a trained
    model: every rejection of ``x`` holds for all three."""
    model = _model()
    trained = model.bind(_PARAMS)
    return [
        lambda x: model.circuit(x),
        lambda x: model.circuit(x, _PARAMS),
        lambda x: trained.circuit(x),
    ]


class TestRejections:
    @pytest.mark.parametrize("call", range(3))
    def test_too_many_features_for_the_encoder(self, call):
        with pytest.raises(ValueError, match="needs 3 active qubit"):
            _circuit_calls()[call]([0.1, 0.2, 0.3])

    @pytest.mark.parametrize("call", range(3))
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_features_must_be_finite(self, call, value):
        with pytest.raises(ValueError, match="non-finite feature at sample 0, index 1"):
            _circuit_calls()[call]([0.1, value])

    def test_the_amplitude_encoder_s_limits(self):
        model = _amplitude_model()
        with pytest.raises(ValueError, match="5 feature"):
            model.circuit([0.1, 0.2, 0.3, 0.4, 0.5])
        with pytest.raises(ValueError, match="non-zero sample"):
            model.circuit([0.0, 0.0, 0.0, 0.0])

    def test_the_parameter_count_must_match(self):
        with pytest.raises(ValueError, match="params has 3 values"):
            _model().circuit(_SAMPLE, [0.1, 0.2, 0.3])

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_parameters_must_be_finite(self, value):
        with pytest.raises(ValueError, match=r"params\[1\]"):
            _model().circuit(_SAMPLE, [0.1, value, 0.3, 0.4])

    @pytest.mark.parametrize("call", range(3))
    def test_x_is_one_sample(self, call):
        from polypus.qml import Dataset

        for x in ([_SAMPLE], Dataset([_SAMPLE]), "ab"):
            with pytest.raises(TypeError, match="x must be one sample"):
                _circuit_calls()[call](x)

    @pytest.mark.parametrize("call", range(3))
    def test_a_feature_that_is_not_a_number(self, call):
        with pytest.raises(TypeError, match="x must be one sample") as raised:
            _circuit_calls()[call]([0.1, None])
        # The extraction's own error, which names the offending element.
        assert isinstance(raised.value.__cause__, TypeError)
        assert "NoneType" in str(raised.value.__cause__)
