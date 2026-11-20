"""The execution integers of the native QML API (#607, contract C-11).

`shots`, `n_qpus`, `nodes` and `cores_per_qpu` (unsigned 32-bit) and `seed`
(unsigned 64-bit) of `Model.fit`, `TrainedModel.predict`/`run`, `qml.expval`,
`qml.jacobian`, `qml.vjp` and `qml.kernel_matrix`: a `bool` or another type is
a `TypeError` and an object with `__index__` (a NumPy integer) is the integer
it returns, all before any CUNQA call. Each out-of-range value is checked at
its place in the order of the checks, which this file pins; the refusals
themselves are in `test_configuration_errors.py`. `Dataset.train_test_split`'s
`seed` follows the same rule with a plain `ValueError`, as a builder.
"""

import inspect

import pytest

pytest.importorskip("qiskit_aer")

import numpy as np  # noqa: E402
import polypus  # noqa: E402
from fake_cunqa import FakeCunqa  # noqa: E402
from test_configuration_errors import (  # noqa: E402
    NATIVE_ENTRY_POINTS,
    _fit,
    _infer,
    _primitive,
)

INTEGERS = ["shots", "n_qpus", "nodes", "cores_per_qpu", "seed"]


class TestTypes:
    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", INTEGERS)
    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_is_a_type_error_before_any_cunqa_call(
        self, monkeypatch, entry, argument, value
    ):
        # Python makes a `bool` an `int`, but `True` is not 1 shot (C-11, as
        # the counts, `final_shots` and `polypus.torch`'s `seed`).
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(TypeError, match=rf"got a bool \({value}\)"):
            NATIVE_ENTRY_POINTS[entry](**{argument: value})
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", INTEGERS)
    @pytest.mark.parametrize("value", [16.0, "16", 16j])
    def test_another_type_is_a_type_error_before_any_cunqa_call(
        self, monkeypatch, entry, argument, value
    ):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(TypeError):
            NATIVE_ENTRY_POINTS[entry](**{argument: value})
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", ["shots", "n_qpus", "nodes", "cores_per_qpu"])
    def test_none_is_a_type_error_where_there_is_no_default_to_draw(
        self, monkeypatch, entry, argument
    ):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(TypeError):
            NATIVE_ENTRY_POINTS[entry](**{argument: None})
        assert fake.calls == []


class TestNumpyIntegers:
    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    def test_numpy_integers_reach_cunqa_as_the_integers_they_hold(
        self, monkeypatch, entry
    ):
        fake = FakeCunqa().install(monkeypatch)
        NATIVE_ENTRY_POINTS[entry](
            shots=np.int64(16),
            n_qpus=np.int32(1),
            nodes=np.uint32(1),
            cores_per_qpu=np.int64(3),
            seed=np.uint64(5),
        )
        assert fake.names()[0] == "qraise"
        qraise = fake.calls[0][1]
        assert (qraise["n"], qraise["n_nodes"], qraise["cores_per_qpu"]) == (1, 1, 3)
        assert fake.runs and {run["options"]["shots"] for run in fake.runs} == {16}

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", INTEGERS)
    def test_a_negative_numpy_integer_is_refused_by_its_value(
        self, monkeypatch, entry, argument
    ):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(
            polypus.ConfigurationError, match=rf"^{argument} must be .*, got -1$"
        ):
            NATIVE_ENTRY_POINTS[entry](**{argument: np.int64(-1)})
        assert fake.calls == []

    def test_an_oversized_numpy_integer_is_refused_by_its_value(self, monkeypatch):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(
            polypus.ConfigurationError,
            match=r"^shots must be at most 4294967295, got 18446744073709551615$",
        ):
            _infer("predict", shots=np.uint64(2**64 - 1))
        assert fake.calls == []


class TestOrder:
    """Where each range check sits among the others (C-11)."""

    @pytest.mark.parametrize("entry", ["jacobian", "vjp"])
    def test_wrt_comes_before_shots(self, monkeypatch, entry):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match="unknown wrt"):
            _primitive(entry, wrt="theta", shots=-1)
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    def test_shots_come_before_n_qpus(self, monkeypatch, entry):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match="^shots must be"):
            NATIVE_ENTRY_POINTS[entry](shots=-1, n_qpus=-1)
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    def test_nodes_and_cores_come_before_the_rest_of_the_cunqa_allocation(
        self, monkeypatch, entry
    ):
        # A `walltime` that is not HH:MM:SS is the allocation's own check.
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match="^cores_per_qpu must be"):
            NATIVE_ENTRY_POINTS[entry](cores_per_qpu=2**32, walltime="soon")
        assert fake.calls == []

    @pytest.mark.parametrize(
        "entry", ["predict", "run", "expval", "jacobian", "vjp", "kernel_matrix"]
    )
    def test_the_seed_range_comes_before_its_refusal_on_qmio(self, entry):
        with pytest.raises(
            polypus.ConfigurationError,
            match=r"^seed must be an integer >= 0 or None, got -1$",
        ):
            NATIVE_ENTRY_POINTS[entry](infrastructure="qmio", seed=-1)

    @pytest.mark.parametrize("entry", ["predict", "run"])
    def test_the_seed_range_comes_after_the_rows(self, monkeypatch, entry):
        # The rows against the model come first, as for a `seed` on `qmio`.
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError) as raised:
            _infer(entry, x=[[0.1, 0.2]], seed=-1)
        assert "seed" not in str(raised.value)
        assert fake.calls == []

    def test_fit_checks_the_seed_after_the_loss_and_before_the_optimizer(
        self, monkeypatch
    ):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(polypus.ConfigurationError, match="unknown loss"):
            _fit(loss="l2", seed=-1)
        with pytest.raises(polypus.ConfigurationError, match="^seed must be"):
            _fit(seed=-1, initial_point=[0.1])
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", ["shots", "n_qpus"])
    def test_zero_keeps_its_message(self, monkeypatch, entry, argument):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(
            polypus.ConfigurationError, match=rf"^{argument} must be >= 1, got 0$"
        ):
            NATIVE_ENTRY_POINTS[entry](**{argument: 0})
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    def test_zero_nodes_and_cores_are_still_taken_off_cunqa(self, entry):
        # Only CUNQA's allocation reads them, so `0` stays harmless elsewhere.
        NATIVE_ENTRY_POINTS[entry](
            infrastructure="local", exact=True, nodes=0, cores_per_qpu=0
        )


class TestNegativeAllocation:
    """A negative `nodes` or `cores_per_qpu` reads as `0` does where `0` is
    refused (on CUNQA), and as its type's bound elsewhere."""

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", ["nodes", "cores_per_qpu"])
    def test_on_cunqa_it_has_the_message_of_zero(self, monkeypatch, entry, argument):
        fake = FakeCunqa().install(monkeypatch)
        with pytest.raises(
            polypus.ConfigurationError,
            match=rf"^{argument} must be >= 1 for the 'cunqa' infrastructure, got -1$",
        ):
            NATIVE_ENTRY_POINTS[entry](**{argument: -1})
        assert fake.calls == []

    @pytest.mark.parametrize("entry", list(NATIVE_ENTRY_POINTS))
    @pytest.mark.parametrize("argument", ["nodes", "cores_per_qpu"])
    def test_elsewhere_it_has_the_bound_of_its_type(self, entry, argument):
        with pytest.raises(
            polypus.ConfigurationError,
            match=rf"^{argument} must be an integer >= 0, got -1$",
        ):
            NATIVE_ENTRY_POINTS[entry](infrastructure="local", **{argument: -1})


class TestSignatures:
    @pytest.mark.parametrize(
        "function",
        [
            polypus.qml.Model.fit,
            polypus.qml.TrainedModel.predict,
            polypus.qml.TrainedModel.run,
            polypus.qml.expval,
            polypus.qml.jacobian,
            polypus.qml.vjp,
            polypus.qml.kernel_matrix,
        ],
    )
    def test_the_signature_still_shows_the_defaults(self, function):
        parameters = inspect.signature(function).parameters
        assert {name: parameters[name].default for name in INTEGERS} == {
            "shots": 1024,
            "n_qpus": 1,
            "nodes": 1,
            "cores_per_qpu": 2,
            "seed": None,
        }
        assert all(p.default is not Ellipsis for p in parameters.values())


class TestTrainTestSplitSeed:
    ROWS = [[float(i)] for i in range(10)]

    @pytest.mark.parametrize(
        ("seed", "message"),
        [
            (-1, r"^seed must be an integer >= 0 or None, got -1$"),
            (
                2**64,
                r"^seed must be at most 18446744073709551615, "
                r"got 18446744073709551616$",
            ),
        ],
    )
    def test_out_of_range_is_a_plain_value_error(self, seed, message):
        with pytest.raises(ValueError, match=message) as raised:
            polypus.qml.Dataset(self.ROWS).train_test_split(0.5, seed=seed)
        assert not isinstance(raised.value, polypus.ConfigurationError)

    @pytest.mark.parametrize("seed", [True, False])
    def test_a_bool_is_a_type_error(self, seed):
        with pytest.raises(TypeError, match="got a bool"):
            polypus.qml.Dataset(self.ROWS).train_test_split(0.5, seed=seed)

    def test_a_numpy_seed_is_the_integer_it_holds(self):
        dataset = polypus.qml.Dataset(self.ROWS, list(range(10)))
        numpy_split = dataset.train_test_split(0.5, seed=np.int64(3))
        int_split = dataset.train_test_split(0.5, seed=3)
        for numpy_half, int_half in zip(numpy_split, int_split):
            assert numpy_half.labels == int_half.labels
