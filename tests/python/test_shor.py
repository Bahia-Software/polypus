"""
``polypus.shor``: Shor's algorithm end to end.

The classical post-processing (``order_from_measurement``,
``factors_from_order``) is checked on hand-computed measurements, and
``factor`` on the native simulator: the classical shortcuts, the quantum path
for 15 and 21, and its argument validation.
"""

import inspect

import polypus
import pytest
from polypus import shor
from polypus.shor import factor, factors_from_order, order_from_measurement


@pytest.mark.parametrize(
    "y, t, a, n, order",
    [
        (64, 8, 7, 15, 4),
        (128, 8, 7, 15, 4),
        (192, 8, 7, 15, 4),
        (0, 8, 7, 15, None),
        (1, 8, 7, 15, None),
        (255, 8, 7, 15, None),
        (11, 6, 2, 21, 6),
        (21, 6, 2, 21, 6),
        (32, 6, 2, 21, 6),
        (43, 6, 2, 21, 6),
        (53, 6, 2, 21, 6),
        (1, 6, 2, 21, None),
        (63, 6, 2, 21, None),
        (16, 6, 2, 21, 6),
    ],
)
def test_order_from_measurement(y, t, a, n, order):
    assert order_from_measurement(y, t, a, n) == order


@pytest.mark.parametrize(
    "a, n, r, factors",
    [
        (7, 15, 4, (3, 5)),
        (2, 21, 6, (3, 7)),
        (4, 15, 2, (3, 5)),
        (4, 21, 3, None),  # odd order
        (14, 15, 2, None),  # a^(r/2) = -1 (mod n)
        (7, 15, 8, None),  # a multiple of the order: a^(r/2) = 1 (mod n)
    ],
)
def test_factors_from_order(a, n, r, factors):
    assert factors_from_order(a, n, r) == factors


def test_order_from_measurement_tries_a_bounded_number_of_multiples():
    # 8388617 is prime and 3 has order 8388616: unbounded, 4,194,308 powers
    assert order_from_measurement(1 << 47, 48, 3, 8388617) is None


@pytest.mark.parametrize(
    "args",
    [
        (4, 2, 7, 15),
        (-1, 8, 7, 15),
        (0, 0, 7, 15),
        (64, 8, 0, 15),
        (64, 8, 15, 15),
        (64, 8, 7, 1),
        (64, 8, 7, 2**24),
    ],
)
def test_order_from_measurement_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        order_from_measurement(*args)


@pytest.mark.parametrize("args", [(0, 15, 4), (15, 15, 4), (7, 1, 4), (7, 15, 0)])
def test_factors_from_order_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        factors_from_order(*args)


def test_helpers_reject_non_integers():
    with pytest.raises(TypeError):
        order_from_measurement(64.0, 8, 7, 15)
    with pytest.raises(TypeError):
        factors_from_order(7, 15, True)


NATIVE = dict(infrastructure="local", backend="polypus")


class TestFactor:
    def test_factors_15_with_the_quantum_circuit(self):
        result = factor(15, a=7, seed=1, **NATIVE)
        assert (result.factors, result.method, result.order) == ((3, 5), "quantum", 4)
        assert sum(result.counts.values()) == 1024

    def test_factors_21_with_six_counting_qubits(self):
        result = factor(21, a=2, counting_qubits=6, seed=1, **NATIVE)
        assert (result.factors, result.order) == ((3, 7), 6)

    @pytest.mark.parametrize(
        "n, factors, method",
        [
            (22, (2, 11), "even"),
            (49, (7, 7), "perfect_power"),
            (225, (15, 15), "perfect_power"),
            (81, (3, 27), "perfect_power"),  # 3**4, not 9**2
        ],
    )
    def test_classical_shortcuts_skip_the_circuit(self, n, factors, method):
        result = factor(n, **NATIVE)
        assert (result.factors, result.method, result.attempts) == (factors, method, 0)

    def test_a_base_sharing_a_factor_is_a_gcd_shortcut(self):
        result = factor(15, a=6, **NATIVE)
        assert (result.factors, result.method, result.a, result.attempts) == (
            (3, 5),
            "gcd",
            6,
            0,
        )

    @pytest.mark.parametrize("n", [3, 0, -15, 13, 2**24])
    def test_invalid_n_raises_value_error(self, n):
        with pytest.raises(ValueError):
            factor(n, **NATIVE)

    @pytest.mark.parametrize("n", [True, 15.0, "15"])
    def test_a_non_integer_n_raises_type_error(self, n):
        with pytest.raises(TypeError):
            factor(n, **NATIVE)

    @pytest.mark.parametrize(
        "options", [{"a": 1}, {"a": 15}, {"max_attempts": 0}, {"shots": 0}]
    )
    def test_invalid_options_raise_value_error(self, options):
        with pytest.raises(ValueError):
            factor(15, **options, **NATIVE)

    @pytest.mark.parametrize("seed", [-1, 2**64])
    def test_a_seed_out_of_range_raises_overflow_error(self, seed):
        with pytest.raises(OverflowError):
            factor(15, seed=seed, **NATIVE)

    @pytest.mark.parametrize(
        "options",
        [
            {"seed": 1.5},
            {"seed": True},
            {"counting_qubits": True},
            {"a": 7.0},
            {"shots": 1024.0},
        ],
    )
    def test_non_integer_options_raise_type_error(self, options):
        with pytest.raises(TypeError):
            factor(15, **options, **NATIVE)

    @pytest.mark.parametrize("n, a, t, order", [(15, 14, None, 2), (21, 4, 6, 3)])
    def test_a_fixed_base_with_a_useless_order_stops_early(self, n, a, t, order):
        result = factor(n, a=a, counting_qubits=t, seed=0, **NATIVE)
        assert (result.factors, result.method, result.order, result.attempts) == (
            None,
            None,
            order,
            1,
        )

    def test_same_seed_same_result(self):
        assert factor(15, seed=7, **NATIVE) == factor(15, seed=7, **NATIVE)

    def test_aer_backend_factors_15(self):
        result = factor(15, infrastructure="local", backend="aer", a=7, seed=3)
        assert (result.factors, result.method) == ((3, 5), "quantum")

    def test_a_circuit_beyond_the_simulator_raises_its_error(self):
        # 34 qubits; the native simulator holds at most 30
        with pytest.raises(polypus.NativeCircuitError, match="at most 30"):
            factor(143, a=2, **NATIVE)

    def test_a_failed_search_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(shor, "order_from_measurement", lambda *args: None)
        result = factor(15, a=7, max_attempts=2, seed=0, **NATIVE)
        assert (result.factors, result.method, result.attempts) == (None, None, 2)


def test_docstring_example_runs():
    # `inspect.getdoc` removes the indentation that Python < 3.13 keeps.
    doc = inspect.getdoc(factor)
    exec(doc.split("```python\n", 1)[1].split("```", 1)[0], {})
