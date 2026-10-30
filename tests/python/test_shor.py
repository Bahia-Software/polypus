"""
``polypus.shor``: Shor's algorithm end to end.

The classical post-processing (``order_from_measurement``,
``factors_from_order``) is checked on hand-computed measurements, and
``factor`` on the native simulator: the classical shortcuts, the quantum path
for 15 and 21, the share of the shots on the peaks of the order it reads, and
its argument validation.
"""

import gc
import inspect
import math
import weakref
from fractions import Fraction
from types import SimpleNamespace

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


@pytest.mark.parametrize("t", range(1, 8))
def test_peak_shares_follow_their_definition(t):
    # Outcome by outcome, against |y - s * 2**t / r| <= 1 for some s from 1 to
    # r - 1, times r; the uniform share is the share of near outcomes. Orders
    # up to 100 reach 2**t for t <= 6, from where every outcome is near, and
    # 2**t / 2 for t = 7, from where all but 0 are.
    size = 2**t
    for r in range(2, 101):
        near = [
            any(abs(y * r - s * size) <= r for s in range(1, r)) for y in range(size)
        ]
        uniform = sum(near) / size
        for y in range(size):
            assert shor._peak_shares({format(y, f"0{t}b"): 1}, t, r) == (
                float(near[y]),
                uniform,
            )


def test_an_outcome_beyond_the_register_is_never_near():
    # With an order of 2**t or more every outcome is near; 4 is not an
    # outcome of 2 qubits.
    assert shor._peak_shares({"100": 1, "11": 1}, 2, 4) == (0.5, 1.0)


def test_peak_shares_are_exact_for_the_largest_registers():
    # 48 counting qubits and an order near 2**24: the peaks are 2**24 apart,
    # so only the nearest s can be within 1. For s * 2**48 = 1 or -1 modulo r,
    # and for r // 3 and r - 1, the peak is less than 1e-6 from an integer,
    # closer than a float of that size can tell.
    t, r = 48, 2**24 - 3
    inverse = pow(2**t, -1, r)
    for s in (1, 2, r // 3, r - 1, inverse, r - inverse):
        peak = Fraction(s * 2**t, r)
        for y in range(math.floor(peak) - 2, math.ceil(peak) + 3):
            near = abs(y - peak) <= 1
            assert shor._peak_shares({format(y, "048b"): 1}, t, r) == (
                float(near),
                (2 * r + 1 - 3) / 2**t,
            )


NATIVE = dict(infrastructure="local", backend="polypus")

# What a working circuit gives for 15 with a = 7 (order 4): the multiples of
# 2**8 / 4 and nothing else. Noise, or the wrong qubits measured, gives others.
PEAKS_15 = {"00000000", "01000000", "10000000", "11000000"}


def _share_near_peaks(counts, t, r):
    """The share of shots within 1 of s * 2**t / r for some s from 1 to r - 1."""
    peaks = [s * 2**t / r for s in range(1, r)]
    near = sum(
        count
        for bits, count in counts.items()
        if min(abs(int(bits, 2) - peak) for peak in peaks) <= 1
    )
    return near / sum(counts.values())


def _run_returns(monkeypatch, *counts):
    """Make the runs of the circuit give ``counts`` in turn, then the last
    of them again."""
    runs = iter(counts)

    def run(*args, **kwargs):
        return SimpleNamespace(merged_counts=next(runs, counts[-1]))

    monkeypatch.setattr(polypus, "run_quantum_circuit", run)


class TestFactor:
    def test_factors_15_with_the_quantum_circuit(self):
        result = factor(15, a=7, seed=1, **NATIVE)
        assert (result.factors, result.method, result.order) == ((3, 5), "quantum", 4)
        assert sum(result.counts.values()) == 1024
        assert set(result.counts) == PEAKS_15
        # Every shot but those at 0, the peak that every order has.
        assert result.peak_share == 1 - result.counts["00000000"] / 1024
        assert result.uniform_peak_share == 9 / 256
        # The values docs/guide/circuits.md shows.
        assert (result.peak_share, result.uniform_peak_share) == (0.734375, 9 / 256)

    def test_factors_21_with_six_counting_qubits(self):
        result = factor(21, a=2, counting_qubits=6, seed=1, **NATIVE)
        assert (result.factors, result.order) == ((3, 7), 6)
        assert result.peak_share == _share_near_peaks(result.counts, 6, 6)
        # About 0.74 for the circuit, against 11/64 for uniformly random
        # outcomes.
        assert result.peak_share >= 0.65
        assert result.uniform_peak_share == 11 / 64

    def test_counts_without_signal_give_the_uniform_share(self, monkeypatch):
        # The order is still read, from the noise.
        counts = {format(y, "08b"): 4 for y in range(256)}
        _run_returns(monkeypatch, counts)
        result = factor(15, a=7, seed=0, **NATIVE)
        assert (result.factors, result.method, result.order) == ((3, 5), "quantum", 4)
        assert result.peak_share == result.uniform_peak_share == 9 / 256

    def test_a_circuit_that_does_nothing_scores_below_uniform(self, monkeypatch):
        # Multiplications that do nothing leave the counting register at 0,
        # the peak every order has; a little noise is enough to read the order.
        counts = {format(y, "08b"): 1 for y in range(256)}
        counts["00000000"] = 2305
        _run_returns(monkeypatch, counts)
        result = factor(15, a=7, seed=0, **NATIVE)
        assert (result.factors, result.method) == ((3, 5), "quantum")
        assert result.peak_share == 9 / 2560
        assert result.peak_share < result.uniform_peak_share

    @pytest.mark.parametrize(
        "counts, peak_share",
        [
            # Keys no backend returns, after the one the order is read from:
            # shots that are never near.
            ({"01000000": 1000, "0x2": 8, "100000000": 8, "-1": 7, 3: 1}, 1000 / 1024),
            # Counts below 1 are no shots.
            ({"01000000": 1000, "00000000": 24, "10000000": -8}, 1000 / 1024),
            ({"01000000": 0}, 0.0),
        ],
    )
    def test_malformed_counts_never_raise(self, monkeypatch, counts, peak_share):
        _run_returns(monkeypatch, counts)
        result = factor(15, a=7, seed=0, **NATIVE)
        assert (result.factors, result.method) == ((3, 5), "quantum")
        assert (result.peak_share, result.uniform_peak_share) == (peak_share, 9 / 256)

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
        assert (result.peak_share, result.uniform_peak_share) == (None, None)

    def test_a_base_sharing_a_factor_is_a_gcd_shortcut(self):
        result = factor(15, a=6, **NATIVE)
        assert (result.factors, result.method, result.a, result.attempts) == (
            (3, 5),
            "gcd",
            6,
            0,
        )
        assert (result.peak_share, result.uniform_peak_share) == (None, None)

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

    # Checked before the shortcuts: 22 is even, and seed 3 draws the base 5
    # for 15.
    @pytest.mark.parametrize("n, counting_qubits", [(22, 0), (15, 9)])
    def test_counting_qubits_out_of_range_raises_value_error(self, n, counting_qubits):
        with pytest.raises(ValueError, match="counting_qubits"):
            factor(n, counting_qubits=counting_qubits, seed=3, **NATIVE)

    def test_shots_beyond_what_a_run_takes_raise_overflow_error(self):
        # A run takes fewer than 2**32 shots; 22 is split without a circuit.
        with pytest.raises(OverflowError, match="shots"):
            factor(22, shots=2**32, **NATIVE)

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

    @pytest.mark.parametrize(
        "n, a, t, order, uniform_peak_share",
        [(15, 14, None, 2, 3 / 256), (21, 4, 6, 3, 4 / 64)],
    )
    def test_a_fixed_base_with_a_useless_order_stops_early(
        self, n, a, t, order, uniform_peak_share
    ):
        result = factor(n, a=a, counting_qubits=t, seed=0, **NATIVE)
        assert (result.factors, result.method, result.order, result.attempts) == (
            None,
            None,
            order,
            1,
        )
        assert result.peak_share == _share_near_peaks(result.counts, t or 8, order)
        assert result.uniform_peak_share == uniform_peak_share

    def test_same_seed_same_result(self):
        first = factor(15, seed=7, **NATIVE)
        second = factor(15, seed=7, **NATIVE)
        assert first == second

    # qmio, real hardware, refuses a seed.
    @pytest.mark.parametrize("seed", [None, 7])
    def test_a_run_has_a_seed_only_when_factor_has_one(self, monkeypatch, seed):
        run_seeds = []
        run_quantum_circuit = polypus.run_quantum_circuit

        def spy(*args, **kwargs):
            run_seeds.append(kwargs.get("seed"))
            return run_quantum_circuit(*args, **kwargs)

        monkeypatch.setattr(polypus, "run_quantum_circuit", spy)
        factor(15, a=7, seed=seed, **NATIVE)
        assert run_seeds
        assert all((run_seed is None) == (seed is None) for run_seed in run_seeds)

    def test_aer_backend_factors_15(self):
        result = factor(15, infrastructure="local", backend="aer", a=7, seed=3)
        assert (result.factors, result.method) == ((3, 5), "quantum")
        assert set(result.counts) == PEAKS_15

    def test_a_circuit_beyond_the_simulator_raises_its_error(self):
        # 34 qubits; the native simulator holds at most 30
        with pytest.raises(polypus.NativeCircuitError, match="at most 30"):
            factor(143, a=2, **NATIVE)

    def test_an_attempt_frees_the_last_circuit_before_building_the_next(
        self, monkeypatch
    ):
        # Two 24-bit circuits alive at once would double the peak memory.
        built = []
        alive_at_build = []

        class Circuit:
            def measure(self, qubit, cbit):
                pass

        def build(a, n, counting_qubits):
            gc.collect()
            alive_at_build.append(sum(ref() is not None for ref in built))
            qc = Circuit()
            built.append(weakref.ref(qc))
            return qc

        monkeypatch.setattr(polypus.circuits.templates, "shor_order_finding", build)
        _run_returns(monkeypatch, {"0": 1})
        monkeypatch.setattr(shor, "order_from_measurement", lambda *args: None)
        result = factor(4093 * 4099, max_attempts=3, seed=0, **NATIVE)
        assert (result.attempts, alive_at_build) == (3, [0, 0, 0])

    def test_a_failed_search_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(shor, "order_from_measurement", lambda *args: None)
        result = factor(15, a=7, max_attempts=2, seed=0, **NATIVE)
        assert (result.factors, result.method, result.attempts) == (None, None, 2)
        assert (result.peak_share, result.uniform_peak_share) == (None, None)

    def test_a_search_ending_without_an_order_keeps_no_earlier_shares(
        self, monkeypatch
    ):
        # Seed 0 draws the base 8 twice for 15. The first attempt reads an odd
        # order, which does not split 15; the second reads none.
        orders = iter([3])
        monkeypatch.setattr(
            shor, "order_from_measurement", lambda *args: next(orders, None)
        )
        _run_returns(monkeypatch, {"01000000": 1024})
        result = factor(15, max_attempts=2, seed=0, **NATIVE)
        assert (result.a, result.order, result.attempts) == (8, None, 2)
        assert (result.peak_share, result.uniform_peak_share) == (None, None)

    def test_a_failed_search_has_the_shares_of_its_last_order(self, monkeypatch):
        # Seed 0 draws the base 8 twice for 15; both attempts read an odd
        # order. 85 is near 256 / 3 but no multiple of 256 / 5.
        orders = iter([5, 3])
        monkeypatch.setattr(shor, "order_from_measurement", lambda *args: next(orders))
        _run_returns(monkeypatch, {"01000000": 1024}, {"01010101": 1024})
        result = factor(15, max_attempts=2, seed=0, **NATIVE)
        assert (result.factors, result.order, result.attempts) == (None, 3, 2)
        assert (result.peak_share, result.uniform_peak_share) == (1.0, 4 / 256)

    def test_a_split_after_a_useless_order_has_its_own_shares(self, monkeypatch):
        # The first attempt reads an odd order, the second the order 4.
        orders = iter([3, 4])
        monkeypatch.setattr(shor, "order_from_measurement", lambda *args: next(orders))
        _run_returns(
            monkeypatch, {"01010101": 1024}, {"00000000": 768, "01000000": 256}
        )
        result = factor(15, max_attempts=2, seed=0, **NATIVE)
        assert (result.factors, result.method, result.order, result.attempts) == (
            (3, 5),
            "quantum",
            4,
            2,
        )
        assert (result.peak_share, result.uniform_peak_share) == (0.25, 9 / 256)

    def test_a_fixed_base_has_the_shares_of_the_attempt_that_read_its_order(
        self, monkeypatch
    ):
        # 0 gives no order; 128 gives the order 2 of 14, which does not split
        # 15 since 14 is -1 modulo 15.
        _run_returns(monkeypatch, {"00000000": 1024}, {"10000000": 1024})
        result = factor(15, a=14, max_attempts=3, seed=0, **NATIVE)
        assert (result.factors, result.order, result.attempts) == (None, 2, 2)
        assert (result.peak_share, result.uniform_peak_share) == (1.0, 3 / 256)


def test_docstring_example_runs():
    # `inspect.getdoc` removes the indentation that Python < 3.13 keeps.
    doc = inspect.getdoc(factor)
    exec(doc.split("```python\n", 1)[1].split("```", 1)[0], {})
