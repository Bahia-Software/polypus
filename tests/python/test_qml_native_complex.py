"""
Complex values at the native QML entry points (issue #602; contract C-11,
"Data", and C-8, "Labels").

PyO3 reads a float through ``float()``, which NumPy's complex scalars answer by
dropping their imaginary part with a ``ComplexWarning``: a complex NumPy ``x``
used to be taken cut, while the same values as Python ``complex`` were a
``TypeError``. Every entry point now refuses a complex value by its type,
whatever its imaginary part and whatever holds it, with a ``TypeError`` naming
the argument and the value's position, before ``float()`` is ever called (so
no ``ComplexWarning``). A ``bool`` is not complex and still reads as 0 or 1.
Every refusal happens before a backend exists.
"""

import math
import warnings

import numpy as np
import polypus
import pytest
from polypus import qml
from qml_native_helpers import make_model

# `numpy.exceptions` is NumPy 1.25 and later.
ComplexWarning = getattr(np, "exceptions", np).ComplexWarning

# Two qubits, angle-encoded: two features per row.
ROWS = [[0.3, 1.2], [0.5, 0.1]]
LABELS = [1.0, -1.0]


def _model():
    return make_model(num_qubits=2)


def _params(model):
    return [0.1 + 0.2 * i for i in range(model.num_params)]


# ─────────────────────────────────────────────────────────────────────────────
# Containers: each holds one complex value, and says where the first one is
# ─────────────────────────────────────────────────────────────────────────────

# For a 2-D `x`: (container of the rows, (row, column) of its first complex
# value). A complex dtype makes every value complex, so the first is (0, 0);
# elsewhere only `x[1][0]` is.


def _set(rows, value):
    """`rows` with `rows[1][0]` replaced by `value(rows[1][0])`."""
    copy = [list(row) for row in rows]
    copy[1][0] = value(copy[1][0])
    return copy


def _cut(array):
    """The complex `array` with an imaginary part on `[1][0]`, the rest real."""
    array[1, 0] += 0.5j
    return array


CONTAINERS_2D = {
    # (a) a complex128 array with an imaginary part.
    "complex128": lambda rows: (_cut(np.array(rows, dtype=np.complex128)), (0, 0)),
    # (b) the same with none: the type decides, not the value.
    "complex128-real-valued": lambda rows: (
        np.array(rows, dtype=np.complex128),
        (0, 0),
    ),
    # (c) complex64, which is not a subclass of Python's complex.
    "complex64": lambda rows: (np.array(rows, dtype=np.complex64), (0, 0)),
    # (d) a list of rows holding one NumPy complex scalar.
    "list-numpy-complex": lambda rows: (_set(rows, np.complex128), (1, 0)),
    # (e) an object array holding one.
    "object-array": lambda rows: (
        np.array(_set(rows, np.complex128), dtype=object),
        (1, 0),
    ),
    # (f) a list holding one Python complex.
    "list-python-complex": lambda rows: (_set(rows, complex), (1, 0)),
}


def _set_1d(values, value):
    copy = list(values)
    copy[1] = value(copy[1])
    return copy


# The same for a flat sequence (one sample, or labels): (container, index).
CONTAINERS_1D = {
    "complex128": lambda v: (
        np.array(v, dtype=np.complex128) + np.array([0, 0.5j]),
        0,
    ),
    "complex128-real-valued": lambda v: (np.array(v, dtype=np.complex128), 0),
    "complex64": lambda v: (np.array(v, dtype=np.complex64), 0),
    "list-numpy-complex": lambda v: (_set_1d(v, np.complex128), 1),
    "object-array": lambda v: (np.array(_set_1d(v, np.complex128), dtype=object), 1),
    "list-python-complex": lambda v: (_set_1d(v, complex), 1),
}

CONTAINER_2D = pytest.mark.parametrize("container", sorted(CONTAINERS_2D))
CONTAINER_1D = pytest.mark.parametrize("container", sorted(CONTAINERS_1D))


def _refused(call, match):
    """`call()` raises a `TypeError` matching `match`, with no
    `ComplexWarning` along the way: the value is refused before `float()`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match=match) as raised:
            call()
    assert not [w for w in caught if issubclass(w.category, ComplexWarning)], [
        str(w.message) for w in caught
    ]
    return raised.value


def _complex_at(name, *position):
    """The message's start for the complex value at `position` of `name`."""
    index = "".join(rf"\[{p}\]" for p in position)
    return rf"{name}{index} is .+, a complex number; it must be a real number"


# ─────────────────────────────────────────────────────────────────────────────
# x: every entry point that reads rows
# ─────────────────────────────────────────────────────────────────────────────


def _row_calls():
    """Every native entry point that reads rows, as `(argument, call(x))`."""
    model = _model()
    params = _params(model)
    trained = model.bind(params)
    encoder = qml.Encoder(2).angle_encoder("ry")
    cotangent = np.ones((len(ROWS), model.num_observables))
    return {
        "expval": ("x", lambda x: qml.expval(model, x, params, exact=True)),
        "jacobian": ("x", lambda x: qml.jacobian(model, x, params, exact=True)),
        "vjp": ("x", lambda x: qml.vjp(model, x, params, cotangent, exact=True)),
        "fit": (
            "x",
            lambda x: model.fit(
                x,
                LABELS,
                method=polypus.Adam(max_iters=1),
                loss="hinge",
                exact=True,
                seed=1,
            ),
        ),
        "predict": ("x", lambda x: trained.predict(x, exact=True)),
        "run": ("x", lambda x: trained.run(x, exact=True)),
        "Dataset": ("x", lambda x: qml.Dataset(x)),
        "kernel_matrix-x1": ("x1", lambda x: qml.kernel_matrix(encoder, x, exact=True)),
        "kernel_matrix-x2": (
            "x2",
            lambda x: qml.kernel_matrix(encoder, ROWS, x, exact=True),
        ),
    }


ROW_CALLS = pytest.mark.parametrize("entry", sorted(_row_calls()))


class TestComplexRows:
    @ROW_CALLS
    @CONTAINER_2D
    def test_a_complex_value_is_a_type_error_naming_its_position(
        self, entry, container
    ):
        name, call = _row_calls()[entry]
        x, (row, column) = CONTAINERS_2D[container](ROWS)
        _refused(lambda: call(x), _complex_at(name, row, column))

    @ROW_CALLS
    def test_the_real_rows_are_accepted(self, entry):
        # The same calls with the real rows succeed, so the refusals above are
        # the complex value's alone.
        _, call = _row_calls()[entry]
        call(ROWS)


# ─────────────────────────────────────────────────────────────────────────────
# x: one sample, for circuit
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE = ROWS[0]


def _circuit_calls():
    model = _model()
    params = _params(model)
    trained = model.bind(params)
    return {
        "Model.circuit": lambda x: model.circuit(x),
        "Model.circuit-params": lambda x: model.circuit(x, params),
        "TrainedModel.circuit": lambda x: trained.circuit(x),
    }


class TestComplexSample:
    @pytest.mark.parametrize("entry", sorted(_circuit_calls()))
    @CONTAINER_1D
    def test_a_complex_value_is_a_type_error_naming_its_position(
        self, entry, container
    ):
        x, index = CONTAINERS_1D[container](SAMPLE)
        error = _refused(lambda: _circuit_calls()[entry](x), _complex_at("x", index))
        # Not the generic error for what is not one sample.
        assert "x must be one sample" not in str(error)


# ─────────────────────────────────────────────────────────────────────────────
# y: Model.fit and Dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexLabels:
    @CONTAINER_1D
    def test_fit(self, container):
        y, index = CONTAINERS_1D[container](LABELS)
        # `fit`'s `y` goes through C-8's label checks, whose messages name
        # `y_train`.
        _refused(
            lambda: _model().fit(
                ROWS,
                y,
                method=polypus.Adam(max_iters=1),
                loss="hinge",
                exact=True,
                seed=1,
            ),
            _complex_at("y_train", index),
        )

    @CONTAINER_1D
    def test_dataset(self, container):
        y, index = CONTAINERS_1D[container](LABELS)
        _refused(lambda: qml.Dataset(ROWS, y), _complex_at("y", index))


# ─────────────────────────────────────────────────────────────────────────────
# Order, and what does not change
# ─────────────────────────────────────────────────────────────────────────────


def test_vjp_reads_x_before_the_cotangent():
    # C-11's checks: the rows before the cotangent.
    model = _model()
    x = np.array(ROWS, dtype=np.complex128)
    cotangent = np.ones((len(ROWS), model.num_observables), dtype=np.complex128)
    _refused(
        lambda: qml.vjp(model, x, _params(model), cotangent, exact=True),
        _complex_at("x", 0, 0),
    )


@pytest.mark.parametrize(
    "booleans",
    [
        [[True, False], [False, True]],
        [[np.bool_(True), np.bool_(False)], [np.bool_(False), np.bool_(True)]],
        np.array([[True, False], [False, True]]),
    ],
    ids=["python", "numpy-scalars", "numpy-array"],
)
def test_a_bool_still_reads_as_zero_or_one(booleans):
    # A guard: a bool is a real number here (unlike an external optimizer's
    # candidate), so a reader that refused it would break this.
    model = _model()
    params = _params(model)
    expected = qml.expval(model, [[1.0, 0.0], [0.0, 1.0]], params, exact=True)
    got = qml.expval(model, booleans, params, exact=True)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(
    ("x", "error", "match"),
    [
        # PyO3's own structure errors, as before.
        (SAMPLE, TypeError, "Sequence"),
        ("ab", TypeError, "str"),
        ([[0.1, None]], TypeError, "NoneType"),
        ([[0.1, "0.5"]], TypeError, "str"),
        # The rows' checks, as before.
        ([], polypus.ConfigurationError, "empty"),
        ([[0.1, 0.2], [0.3]], polypus.ConfigurationError, "ragged"),
        ([[0.1, math.nan]], polypus.ConfigurationError, "finite"),
        ([[0.1, math.inf]], polypus.ConfigurationError, "finite"),
    ],
    ids=["flat", "str", "none", "str-value", "empty", "ragged", "nan", "inf"],
)
def test_the_other_rejections_of_x_are_unchanged(x, error, match):
    model = _model()
    with pytest.raises(error, match=match) as raised:
        qml.expval(model, x, _params(model), exact=True)
    assert "complex" not in str(raised.value)
