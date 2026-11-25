"""Native cost-observable tests — verify polypus.Qubo / polypus.Ising and the
callable fallback through polypus.train.

These use the native ``backend="polypus"`` statevector backend (pure Rust, no
AerSimulator), so they are fast and need no qiskit at runtime. They exercise the
full FFI path: the Qubo/Ising pyclasses, the ``expectation_function`` dispatch
(callable vs observable), the deduplicated Python-callback fallback, and the
native rayon aggregation — asserting the native and callback paths agree.
"""

import math
import sys

import polypus
import pytest

_SEED = 12345
_COMMON = dict(
    shots=512,
    n_qpus=1,
    dimensions=1,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    backend="polypus",
    seed=_SEED,
)


def _circuit():
    """1-qubit RY(θ) with a full read-out; θ is the single free parameter."""
    return polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()


def _de():
    return polypus.DE(generations=5, population_size=8, tolerance=0.0)


def _all_ones(bitstring: str) -> float:
    """Reference cost: 1.0 iff every bit is '1' (== x_0 for one qubit)."""
    return float(all(c == "1" for c in bitstring))


class TestCallableObservableEquivalence:
    def test_native_qubo_matches_callable(self):
        # f(x) = x_0 is expressible both as a callback and as a Qubo. With the
        # same seed the native backend produces identical counts, so both paths
        # must yield an identical fitness trajectory.
        res_cb = polypus.train(
            _circuit(), _de(), expectation_function=_all_ones, id="obs_cb", **_COMMON
        )
        res_qubo = polypus.train(
            _circuit(),
            _de(),
            expectation_function=polypus.Qubo(1, linear=[(0, 1.0)]),
            id="obs_qubo",
            **_COMMON,
        )
        assert res_qubo.fitness_history == res_cb.fitness_history
        assert res_qubo.best_fitness == res_cb.best_fitness

    def test_cached_cost_is_transparent(self):
        # Cross-generation memoisation must not change results for a pure cost:
        # the wrapped and bare callables must agree exactly on the same seed.
        res_cb = polypus.train(
            _circuit(), _de(), expectation_function=_all_ones, id="obs_cb2", **_COMMON
        )
        res_cached = polypus.train(
            _circuit(),
            _de(),
            expectation_function=polypus.CachedCost(_all_ones),
            id="obs_cached",
            **_COMMON,
        )
        assert res_cached.fitness_history == res_cb.fitness_history

    def test_cached_cost_rejects_non_callable(self):
        with pytest.raises(TypeError):
            polypus.CachedCost(42)


class TestObservableTrains:
    def test_ising_trains(self):
        # f(s) = -z_0 = 2 x_0 - 1, maximised at x_0 = 1.
        res = polypus.train(
            _circuit(),
            _de(),
            expectation_function=polypus.Ising(1, fields=[(0, -1.0)]),
            id="obs_ising",
            **_COMMON,
        )
        assert isinstance(res.best_fitness, float)
        assert len(res.fitness_history) == res.iterations_run

    def test_qubo_from_matrix_trains(self):
        res = polypus.train(
            _circuit(),
            _de(),
            expectation_function=polypus.Qubo.from_matrix([[1.0]]),
            id="obs_matrix",
            **_COMMON,
        )
        assert isinstance(res.best_fitness, float)


class TestObservableConstruction:
    def test_num_vars_getter(self):
        assert polypus.Qubo(3).num_vars == 3
        assert polypus.Ising(4).num_vars == 4

    def test_bad_index_raises_value_error(self):
        with pytest.raises(ValueError):
            polypus.Qubo(1, linear=[(5, 1.0)])
        with pytest.raises(ValueError):
            polypus.Ising(2, couplings=[(0, 3, 1.0)])

    def test_self_quadratic_raises_value_error(self):
        with pytest.raises(ValueError):
            polypus.Qubo(2, quadratic=[(0, 0, 1.0)])

    def test_non_finite_coefficient_raises_value_error(self):
        with pytest.raises(ValueError):
            polypus.Qubo(1, linear=[(0, float("nan"))])


# The bound the class docstrings state (#625, #641): sys.float_info.max / 2**65,
# exact in binary floating point (dividing by a power of two only changes the
# exponent).
_BOUND = sys.float_info.max / 2**65


class TestCoefficientBound:
    def test_the_documented_bound_is_inclusive(self):
        # Exactly at the bound is accepted, the next float above it is not, so
        # the number in the docstrings is the one enforced.
        polypus.Qubo(1, linear=[(0, _BOUND)])
        polypus.Qubo(1, linear=[(0, -_BOUND)], scale=-1.0)
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Qubo(1, linear=[(0, math.nextafter(_BOUND, math.inf))])
        assert 4.8e288 < _BOUND < 4.9e288

    def test_qubo_sums_every_part_times_the_scale(self):
        # Each part is within the bound; only the sum exceeds it.
        third = _BOUND / 3
        for kwargs in (
            dict(linear=[(0, 2 * third), (1, 2 * third)]),
            dict(linear=[(0, 2 * third)], quadratic=[(0, 1, 2 * third)]),
            dict(linear=[(0, 2 * third)], constant=2 * third),
            dict(linear=[(0, _BOUND)], scale=1.5),
        ):
            with pytest.raises(ValueError, match="exceeds"):
                polypus.Qubo(2, **kwargs)
        polypus.Qubo(2, linear=[(0, _BOUND)], scale=0.5)

    def test_from_matrix_checks_the_folded_coefficients(self):
        third = _BOUND / 3
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Qubo.from_matrix([[2 * third, 0.0], [0.0, 2 * third]])
        # Q_01 + Q_10 is one quadratic weight.
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Qubo.from_matrix([[2 * third, third], [third, 0.0]])
        polypus.Qubo.from_matrix([[third, 0.0], [0.0, third]])
        with pytest.raises(ValueError):
            polypus.Qubo.from_matrix([[1.0, 2.0]])

    def test_ising_is_checked_on_its_expanded_qubo(self):
        # h·z_0 = h - 2h·x_0 adds 3·|h| to the sum: 3e288 is within the bound,
        # 6e288 is not, though |h| = 2e288 is below it.
        polypus.Ising(1, fields=[(0, 1e288)])
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Ising(1, fields=[(0, 2e288)])
        # A coupling adds 9·|J|: J, -2J, -2J and 4J.
        polypus.Ising(2, couplings=[(0, 1, 5e287)])
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Ising(2, couplings=[(0, 1, 6e287)])
        # A finite coupling whose 4J overflows float.
        with pytest.raises(ValueError):
            polypus.Ising(2, couplings=[(0, 1, 1e308)])

    @pytest.mark.parametrize("cls", [polypus.Qubo, polypus.Ising])
    def test_the_class_docstring_documents_the_errors(self, cls):
        # PyO3 does not expose the constructor's own doc comment
        # (`cls.__new__.__doc__` is the generic one), so the errors must be in
        # the class docstring, which help() shows (#641).
        doc = cls.__doc__
        for phrase in (
            "ValueError",
            "sys.float_info.max / 2**65",
            "about 4.9e288",
            "polypus.EvaluationError",
        ):
            assert phrase in doc, f"{cls.__name__}.__doc__ lacks {phrase!r}"
        assert "[`" not in doc, "a rustdoc link reads literally in Python"

    def test_from_matrix_documents_its_errors(self):
        doc = polypus.Qubo.from_matrix.__doc__
        for phrase in ("ValueError", "not square", "bound"):
            assert phrase in doc, f"Qubo.from_matrix.__doc__ lacks {phrase!r}"

    def test_qubo_bound_sums_repeated_entries_as_given(self):
        # As its docstring says: the entries are summed before they
        # accumulate, so two that cancel still count twice.
        with pytest.raises(ValueError, match="exceeds"):
            polypus.Qubo(1, linear=[(0, _BOUND), (0, -_BOUND)])


class TestDispatchErrors:
    def test_non_callable_non_observable_raises_type_error(self):
        with pytest.raises(TypeError):
            polypus.train(
                _circuit(), _de(), expectation_function=42, id="obs_bad", **_COMMON
            )
