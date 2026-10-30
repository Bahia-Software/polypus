"""
``polypus.torch.QuantumLayer`` and ``from_trained`` (issue #502; contract C-11,
"The PyTorch layer").

The layer adds no quantum logic of its own: its forward pass must be, bit for
bit, ``qml.expval`` over the whole batch in one call, and its backward pass the
vector-Jacobian product of one ``qml.jacobian`` call that asks only for what
needs a gradient. ``torch.autograd.gradcheck`` checks the gradients with its
default tolerances, in exact mode, for the parameters and the inputs. Then the
dtype and device at the boundary, the input shapes, the derived seeds, training
inside ``torch.nn.Sequential`` with several ``torch.optim`` optimizers, and
``state_dict``, ``copy.deepcopy`` and ``from_trained``.

Needs PyTorch: skipped without it, except under ``POLYPUS_REQUIRE_TORCH=1``,
where it fails instead (CI installs a pinned CPU torch on its latest Python and
sets it). What needs no torch is in ``test_torch_import.py``.

Contract test: draft pending human sign-off.
"""

import copy
import math
import os
import sys

import numpy as np
import polypus
import pytest
from polypus import qml

# CI installs PyTorch on its latest Python and sets this to 1, so a missing
# torch fails these tests instead of skipping them.
if os.environ.get("POLYPUS_REQUIRE_TORCH") == "1":
    import torch
else:
    torch = pytest.importorskip("torch")

import polypus.torch as polypus_torch  # noqa: E402
from polypus.torch import QuantumLayer, from_trained  # noqa: E402

ROWS = [[0.4, 1.1], [1.3, -0.2], [2.0, 0.7]]
SHOTS = 64


def models():
    """The models the layer is checked on: an angle encoder with
    RealAmplitudes, angle re-uploading, IQP (re-uploaded), an `argmax` readout
    of three observables and a weighted multi-term `Observable`."""
    return {
        "angle-ry": qml.Model(
            qml.Encoder(2).angle_encoder("ry"),
            qml.Ansatz(2).real_amplitudes(1),
            qml.Readout([qml.Z(0)]),
        ),
        "reuploading": qml.Model(
            qml.Encoder(2).angle_encoder("ry"),
            qml.Ansatz(2).hardware_efficient(1).angle_encoder("rx").real_amplitudes(1),
            qml.Readout([qml.Z(0), qml.X(1)]),
        ),
        "iqp": qml.Model(
            qml.Encoder(2).iqp_encoder(),
            qml.Ansatz(2).real_amplitudes(1).iqp_encoder("linear"),
            qml.Readout([qml.Z(0) @ qml.Z(1)]),
        ),
        "argmax": qml.Model(
            qml.Encoder(2).angle_encoder("rx"),
            qml.Ansatz(2).real_amplitudes(2),
            qml.Readout([qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(1)], decision="argmax"),
        ),
        "weighted": qml.Model(
            qml.Encoder(2).angle_encoder("ry"),
            qml.Ansatz(2).hardware_efficient(1),
            qml.Readout(
                [qml.Observable([(0.5, qml.Z(0)), (-1.5, qml.Z(0) @ qml.Z(1))])]
            ),
        ),
    }


MODELS = pytest.mark.parametrize("name", sorted(models()))


def point(model, offset=0.3):
    """Parameters that are neither zero nor symmetric."""
    return [offset + 0.17 * i for i in range(model.num_params)]


def rows(dtype=None, requires_grad=False):
    return torch.tensor(ROWS, dtype=dtype or torch.float64, requires_grad=requires_grad)


def functional(layer):
    """`layer` as a function of its input and its parameters, for gradcheck."""
    return lambda x, params: torch.func.functional_call(layer, {"params": params}, (x,))


class Spy:
    """Wraps a `qml` primitive and records the keyword arguments and row count
    of every call."""

    def __init__(self, monkeypatch, name):
        self.inner = getattr(qml, name)
        self.calls = []
        monkeypatch.setattr(qml, name, self)

    def __call__(self, model, x, params, **kwargs):
        self.calls.append({"rows": len(x), **kwargs})
        return self.inner(model, x, params, **kwargs)


def test_the_module_imports_the_real_torch():
    # `import torch` inside polypus/torch.py is absolute: it is PyTorch, never
    # the module itself.
    assert polypus_torch.torch is sys.modules["torch"] is torch
    assert polypus_torch.nn is torch.nn
    assert polypus_torch.__all__ == ["QuantumLayer", "from_trained"]


# ─────────────────────────────────────────────────────────────────────────────
# Forward: qml.expval, bit for bit, one call per batch
# ─────────────────────────────────────────────────────────────────────────────


class TestForward:
    @MODELS
    def test_exact_forward_is_expval_bit_for_bit(self, name, monkeypatch):
        model = models()[name]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        expected = qml.expval(model, ROWS, point(model), exact=True)
        spy = Spy(monkeypatch, "expval")
        output = layer(rows())
        assert output.dtype == torch.float64
        assert output.shape == (len(ROWS), model.num_observables)
        assert np.array_equal(output.detach().numpy(), expected)
        # One call, with the whole batch.
        assert [call["rows"] for call in spy.calls] == [len(ROWS)]

    def test_counts_and_repr(self):
        model = models()["argmax"]
        layer = QuantumLayer(model)
        assert (layer.num_params, layer.num_observables) == (6, 3)
        assert layer.params.shape == (6,) and layer.params.dtype == torch.float64
        assert "num_params=6, num_observables=3" in repr(layer)

    def test_one_sample_is_a_row_of_the_batch(self):
        model = models()["argmax"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        batch = layer(rows())
        for i, row in enumerate(rows()):
            one = layer(row)
            assert one.shape == (model.num_observables,)
            assert torch.equal(one, batch[i])

    def test_shapes_and_values_are_checked(self):
        model = models()["angle-ry"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        with pytest.raises(ValueError, match="batch, features"):
            layer(torch.zeros(1, 2, 2, dtype=torch.float64))
        with pytest.raises(ValueError, match="batch, features"):
            layer(torch.tensor(0.5, dtype=torch.float64))
        with pytest.raises(TypeError, match="torch.Tensor"):
            layer(ROWS)
        # The rest are qml.expval's own checks, unchanged.
        with pytest.raises(polypus.ConfigurationError, match="empty"):
            layer(torch.zeros(0, 2, dtype=torch.float64))
        with pytest.raises(polypus.ConfigurationError):
            layer(torch.zeros(2, 5, dtype=torch.float64))
        with pytest.raises(polypus.ConfigurationError):
            layer(torch.tensor([[0.1, math.nan]], dtype=torch.float64))

    def test_execution_arguments_are_expvals_checked_at_the_first_forward(self):
        model = models()["angle-ry"]
        layer = QuantumLayer(model, exact=True, backend="aer")
        with pytest.raises(polypus.ConfigurationError, match="exact=True"):
            layer(rows())


# ─────────────────────────────────────────────────────────────────────────────
# Backward: the vector-Jacobian product of one qml.jacobian call
# ─────────────────────────────────────────────────────────────────────────────


class TestBackward:
    @MODELS
    def test_gradcheck_params_and_inputs(self, name):
        model = models()[name]
        layer = QuantumLayer(model, exact=True)
        params = torch.tensor(point(model), dtype=torch.float64, requires_grad=True)
        f = functional(layer)
        # Default tolerances: with exact=True the parameter shift is exact.
        assert torch.autograd.gradcheck(f, (rows(requires_grad=True), params))
        assert torch.autograd.gradcheck(f, (rows(), params))
        assert torch.autograd.gradcheck(f, (rows(requires_grad=True), params.detach()))

    def test_the_gradient_is_the_chain_rule_over_the_jacobian(self):
        model = models()["argmax"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        x = rows(requires_grad=True)
        weights = torch.tensor([[1.0, -2.0, 0.5], [0.3, 0.0, 1.5], [-1.0, 2.0, 0.2]])
        (layer(x) * weights.double()).sum().backward()
        by_params, by_inputs = qml.jacobian(
            model, ROWS, point(model), wrt="both", exact=True
        )
        w = weights.double().numpy()
        assert np.array_equal(
            layer.params.grad.numpy(), np.einsum("bo,bok->k", w, by_params)
        )
        assert np.array_equal(x.grad.numpy(), np.einsum("bo,bof->bf", w, by_inputs))

    @pytest.mark.parametrize(
        ("x_grad", "params_grad", "wrt"),
        [(False, True, "params"), (True, True, "both"), (True, False, "inputs")],
    )
    def test_only_what_needs_a_gradient_is_asked_for(
        self, x_grad, params_grad, wrt, monkeypatch
    ):
        model = models()["reuploading"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        layer.params.requires_grad_(params_grad)
        spy = Spy(monkeypatch, "jacobian")
        x = rows(requires_grad=x_grad)
        layer(x).sum().backward()
        assert [(call["wrt"], call["rows"]) for call in spy.calls] == [(wrt, len(ROWS))]
        assert (x.grad is not None) == x_grad
        assert (layer.params.grad is not None) == params_grad

    def test_nothing_is_asked_when_nothing_needs_a_gradient(self, monkeypatch):
        model = models()["angle-ry"]
        layer = QuantumLayer(model, exact=True).requires_grad_(False)
        spy = Spy(monkeypatch, "jacobian")
        assert not layer(rows()).requires_grad
        with torch.no_grad():
            assert not QuantumLayer(model, exact=True)(rows()).requires_grad
        assert spy.calls == []

    @pytest.mark.parametrize(
        ("encoder", "x", "name"),
        [
            (
                lambda: qml.Encoder(2).amplitude_encoder(),
                [[0.4, 1.1, 0.7, 2.0], [1.3, 0.2, 2.2, 0.9]],
                "amplitude",
            ),
            (lambda: qml.Encoder(2).basis_encoder(), [[1.0, 0.0], [0.0, 1.0]], "basis"),
        ],
    )
    def test_amplitude_and_basis_have_parameter_gradients_only(self, encoder, x, name):
        model = qml.Model(
            encoder(), qml.Ansatz(2).real_amplitudes(1), qml.Readout([qml.Z(0)])
        )
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        inputs = torch.tensor(x, dtype=torch.float64)
        assert torch.equal(
            layer(inputs).detach(),
            torch.from_numpy(qml.expval(model, x, point(model), exact=True)),
        )
        layer(inputs).sum().backward()
        assert layer.params.grad is not None
        inputs.requires_grad_(True)
        with pytest.raises(polypus.ConfigurationError, match=f"{name} encoder"):
            layer(inputs).sum().backward()

    def test_no_second_derivative(self):
        model = models()["angle-ry"]
        layer = QuantumLayer(model, exact=True)
        x = rows(requires_grad=True)
        # A non-linear loss, so the incoming gradient depends on x.
        (grad,) = torch.autograd.grad((layer(x) ** 2).sum(), x, create_graph=True)
        with pytest.raises(RuntimeError, match="once_differentiable"):
            grad.sum().backward()


# ─────────────────────────────────────────────────────────────────────────────
# dtype and device at the boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestDtypeAndDevice:
    def test_a_float32_input_gets_float32_output_and_input_gradient(self):
        model = models()["reuploading"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True)
        x = rows(torch.float32, requires_grad=True)
        output = layer(x)
        assert output.dtype == torch.float32 and output.device == x.device
        # The quantum computation ran in float64 on the float32 values.
        expected = qml.expval(
            model, x.detach().double().numpy(), point(model), exact=True
        )
        assert torch.equal(output.detach(), torch.from_numpy(expected).float())
        output.sum().backward()
        assert x.grad.dtype == torch.float32
        assert layer.params.grad.dtype == torch.float64

    def test_a_float32_layer_gets_its_gradient_in_float32(self):
        model = models()["reuploading"]
        layer = QuantumLayer(model, initial_params=point(model), exact=True).float()
        assert layer.params.dtype == torch.float32
        output = layer(rows())
        assert output.dtype == torch.float64
        output.sum().backward()
        assert layer.params.grad.dtype == torch.float32

    def test_a_non_floating_input_gets_float64_output(self):
        model = models()["argmax"]
        layer = QuantumLayer(model, exact=True)
        assert layer(torch.tensor([[1, 0], [0, 2]])).dtype == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────────────────────────────────────


def _derived(seed, step, role):
    sequence = np.random.SeedSequence([seed, step, role])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class TestSeeds:
    def _layer(self, seed):
        model = models()["reuploading"]
        return QuantumLayer(model, initial_params=point(model), shots=SHOTS, seed=seed)

    def _steps(self, layer, n=3):
        """`n` forward and backward passes: their outputs and gradients."""
        results = []
        for _ in range(n):
            layer.zero_grad()
            x = rows(requires_grad=True)
            output = layer(x)
            output.sum().backward()
            results.append((output.detach(), x.grad, layer.params.grad.clone()))
        return results

    def test_the_same_seed_replays_every_step(self):
        first, second = self._steps(self._layer(11)), self._steps(self._layer(11))
        for a, b in zip(first, second):
            for u, v in zip(a, b):
                assert torch.equal(u, v)
        # The sampled values do change from one forward pass to the next.
        assert not torch.equal(first[0][0], first[1][0])

    def test_each_call_gets_its_derived_seed(self, monkeypatch):
        forward, backward = Spy(monkeypatch, "expval"), Spy(monkeypatch, "jacobian")
        self._steps(self._layer(11))
        assert [c["seed"] for c in forward.calls] == [
            _derived(11, n, 0) for n in range(3)
        ]
        assert [c["seed"] for c in backward.calls] == [
            _derived(11, n, 1) for n in range(3)
        ]
        seeds = [c["seed"] for c in forward.calls + backward.calls]
        assert len(set(seeds)) == len(seeds)
        assert all(0 <= s < 2**64 for s in seeds)

    def test_neighbouring_seeds_do_not_share_streams(self, monkeypatch):
        # With `seed + step`, step n + 1 of seed s would be step n of seed s + 1.
        forward = Spy(monkeypatch, "expval")
        outputs = {s: self._steps(self._layer(s)) for s in (11, 12)}
        seeds = [c["seed"] for c in forward.calls]
        assert not set(seeds[:3]) & set(seeds[3:])
        assert not torch.equal(outputs[11][1][0], outputs[12][0][0])

    def test_without_a_seed_the_backend_draws_each_one(self, monkeypatch):
        forward, backward = Spy(monkeypatch, "expval"), Spy(monkeypatch, "jacobian")
        for output, x_grad, params_grad in self._steps(self._layer(None), n=2):
            assert output.shape == (len(ROWS), 2)
            assert torch.isfinite(params_grad).all()
        assert [c["seed"] for c in forward.calls + backward.calls] == [None] * 4

    def test_the_counter_is_not_in_the_state_dict(self):
        layer = self._layer(11)
        self._steps(layer, n=1)
        assert list(layer.state_dict()) == ["params"]

    def test_a_seed_on_qmio_is_refused_at_the_first_forward(self):
        model = models()["angle-ry"]
        layer = QuantumLayer(model, infrastructure="qmio", seed=1)
        with pytest.raises(polypus.ConfigurationError, match="seed is not supported"):
            layer(rows())

    @pytest.mark.parametrize("seed", [True, 1.5, "1"])
    def test_a_seed_must_be_an_integer(self, seed):
        with pytest.raises(TypeError, match="seed must be an integer"):
            QuantumLayer(models()["angle-ry"], seed=seed)

    def test_a_negative_seed_is_refused(self):
        with pytest.raises(polypus.ConfigurationError, match="seed must be"):
            QuantumLayer(models()["angle-ry"], seed=-1)
        assert QuantumLayer(models()["angle-ry"], seed=np.uint64(2**64 - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Training inside torch.nn
# ─────────────────────────────────────────────────────────────────────────────

# Two separable classes in four features, mapped by a float32 Linear onto the
# two qubits' angles.
CLASS_X = [
    [0.9, 0.8, 0.1, 0.0],
    [1.0, 0.6, 0.2, 0.1],
    [0.8, 0.9, 0.0, 0.2],
    [0.7, 1.0, 0.1, 0.1],
    [0.1, 0.2, 0.9, 1.0],
    [0.0, 0.1, 1.0, 0.8],
    [0.2, 0.0, 0.8, 0.9],
    [0.1, 0.1, 0.7, 1.0],
]
CLASS_Y = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _network():
    torch.manual_seed(3)
    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).real_amplitudes(1),
        qml.Readout([qml.Z(0), qml.Z(1)]),
    )
    quantum = QuantumLayer(model, exact=True)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 2),
        quantum,
        torch.nn.Linear(quantum.num_observables, 1),
    )


def _data():
    return torch.tensor(CLASS_X), torch.tensor(CLASS_Y).unsqueeze(1)


class TestTraining:
    def test_a_hybrid_network_learns_two_classes(self):
        net = _network()
        x, y = _data()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        def evaluate():
            with torch.no_grad():
                logits = net(x)
            accuracy = ((logits > 0).float() == y).float().mean().item()
            return loss_fn(logits, y).item(), accuracy

        initial_loss, _ = evaluate()
        for _ in range(30):
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            assert loss.dtype == torch.float32
            loss.backward()
            optimizer.step()
        final_loss, accuracy = evaluate()
        assert final_loss < initial_loss - 0.3, (initial_loss, final_loss)
        assert accuracy == 1.0

    @pytest.mark.parametrize("name", ["SGD", "Adam", "LBFGS"])
    def test_the_layer_trains_with_torch_optimizers(self, name):
        net = _network()
        x, y = _data()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        quantum = net[1]
        optimizer = {
            "SGD": lambda: torch.optim.SGD(quantum.parameters(), lr=0.5),
            "Adam": lambda: torch.optim.Adam(quantum.parameters(), lr=0.1),
            "LBFGS": lambda: torch.optim.LBFGS(quantum.parameters(), max_iter=5),
        }[name]()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            return loss

        with torch.no_grad():
            initial = loss_fn(net(x), y).item()
        start = quantum.params.detach().clone()
        for _ in range(5):
            optimizer.step(closure)
        with torch.no_grad():
            final = loss_fn(net(x), y).item()
        assert not torch.equal(quantum.params.detach(), start)
        assert final < initial, (initial, final)


# ─────────────────────────────────────────────────────────────────────────────
# Construction, state and from_trained
# ─────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_initial_params_take_lists_arrays_and_tensors(self):
        model = models()["argmax"]
        values = point(model)
        for given in (
            values,
            tuple(values),
            np.array(values),
            torch.tensor(values, dtype=torch.float32),
        ):
            layer = QuantumLayer(model, initial_params=given)
            expected = torch.tensor(
                [float(v) for v in np.asarray(given, dtype=np.float32)]
                if isinstance(given, torch.Tensor)
                else values,
                dtype=torch.float64,
            )
            assert torch.equal(layer.params.detach(), expected)

    def test_initial_params_are_checked_as_bind_checks_them(self):
        model = models()["argmax"]
        with pytest.raises(ValueError):
            QuantumLayer(model, initial_params=point(model)[:-1])
        bad = point(model)
        bad[2] = math.inf
        with pytest.raises(ValueError):
            QuantumLayer(model, initial_params=bad)

    def test_the_default_start_is_uniform_and_follows_torch_manual_seed(self):
        model = models()["argmax"]
        torch.manual_seed(5)
        first = QuantumLayer(model).params.detach()
        torch.manual_seed(5)
        assert torch.equal(QuantumLayer(model).params.detach(), first)
        assert ((first >= -math.pi) & (first < math.pi)).all()
        assert not torch.equal(QuantumLayer(model).params.detach(), first)

    def test_the_model_must_be_a_model(self):
        model = models()["angle-ry"]
        with pytest.raises(TypeError, match="from_trained"):
            QuantumLayer(model.bind(point(model)))
        with pytest.raises(TypeError, match="polypus.qml.Model"):
            QuantumLayer("model")

    def test_state_dict_and_deepcopy_keep_the_parameters_bit_for_bit(self):
        model = models()["weighted"]
        layer = QuantumLayer(model, shots=SHOTS, seed=4)
        assert [name for name, _ in layer.named_parameters()] == ["params"]
        other = QuantumLayer(model)
        other.load_state_dict(layer.state_dict())
        assert torch.equal(other.params, layer.params)
        copied = copy.deepcopy(layer)
        assert copied.params is not layer.params
        assert torch.equal(copied.params, layer.params)
        # The copy carries the seed and the counter: it replays the original.
        assert torch.equal(copied(rows()), layer(rows()))


class TestFromTrained:
    def test_it_starts_from_the_trained_parameters(self):
        model = models()["iqp"]
        trained = model.bind(point(model))
        layer = from_trained(trained, exact=True)
        assert layer.params.detach().tolist() == trained.params
        assert torch.equal(
            layer(rows()).detach(),
            torch.from_numpy(
                qml.expval(trained.model, ROWS, trained.params, exact=True)
            ),
        )

    def test_a_loaded_model_too(self, tmp_path):
        model = models()["argmax"]
        path = tmp_path / "model.json"
        model.bind(point(model)).save(path)
        layer = from_trained(qml.load(path), exact=True)
        assert layer.params.detach().tolist() == point(model)

    def test_it_takes_a_trained_model_only(self):
        model = models()["angle-ry"]
        with pytest.raises(TypeError, match="QuantumLayer"):
            from_trained(model)
        with pytest.raises(TypeError, match="TrainedModel"):
            from_trained("trained")
