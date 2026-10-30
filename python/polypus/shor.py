"""Shor's algorithm end to end: :func:`factor` splits an integer in two, from
the classical shortcuts to the order-finding circuit.

:func:`factor` needs no circuit for an even number, a perfect power (which
Shor's algorithm cannot split) or a base that shares a factor with the number.
Otherwise it runs ``polypus.circuits.templates.shor_order_finding`` through
``polypus.run_quantum_circuit``, on the backend you name, and turns the counts
into factors with the two classical steps of the algorithm, public here too:

- :func:`order_from_measurement` reads the order ``r`` of ``a`` modulo ``n``
  from one outcome of the counting register, with the continued fraction of
  ``y / 2**t``;
- :func:`factors_from_order` splits ``n`` with ``gcd(a**(r/2) - 1, n)`` or
  ``gcd(a**(r/2) + 1, n)``.

Both are bounded: an outcome costs a few hundred modular powers at most,
whatever a backend measured.
"""

import math
import operator
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

__all__ = ["ShorResult", "factor", "order_from_measurement", "factors_from_order"]

# Moduli below 2**24: the template's `MAX_SHOR_MODULUS_BITS`.
_MAX_MODULUS = 1 << 24


@dataclass(frozen=True)
class ShorResult:
    """What :func:`factor` found.

    ``factors`` is ``(p, q)`` with ``p <= q`` and ``p * q == n``, or ``None``
    when no attempt split ``n``. ``method`` says how they were found:
    ``"even"``, ``"perfect_power"`` or ``"gcd"`` without running a circuit,
    ``"quantum"`` from an order read off the counts of a run, and ``None``
    without factors. ``a`` is the base of the last attempt (``None`` for the
    first two shortcuts), ``order`` its order when the counts gave it, and
    ``counts`` the counts of its run (``None`` when it ran no circuit).
    ``attempts`` is the number of circuits run.

    ``peak_share`` is the share of the shots in ``counts`` within 1 of
    ``s * 2**counting_qubits / order`` for some ``s`` from 1 to
    ``order - 1``, with ``counting_qubits`` as :func:`factor` used it
    (``2 * n.bit_length()`` by default): the peaks of the order, without the
    one at 0 that every order has. ``uniform_peak_share`` is the share that
    uniformly random outcomes would put there. Both are ``None`` when
    ``order`` is.

    ``"quantum"`` does not prove that the circuit worked: see :func:`factor`
    for how to read the two shares.
    """

    n: int
    factors: Optional[tuple[int, int]]
    method: Optional[str]
    a: Optional[int]
    order: Optional[int]
    attempts: int
    peak_share: Optional[float]
    uniform_peak_share: Optional[float]
    counts: Optional[dict[str, int]]


def factor(
    n: int,
    *,
    infrastructure: str,
    shots: int = 1024,
    a: Optional[int] = None,
    counting_qubits: Optional[int] = None,
    max_attempts: int = 10,
    seed: Optional[int] = None,
    **run_kwargs: Any,
) -> ShorResult:
    """Two factors of ``n`` with Shor's algorithm.

    ``n`` is a composite number between 4 and ``2**24 - 1``. An even ``n``, a
    perfect power and a base sharing a factor with ``n`` are split without a
    circuit. Otherwise each attempt builds
    ``shor_order_finding(a, n, counting_qubits)``, measures its counting
    register, runs it ``shots`` times (1 to ``2**32 - 1``) with
    ``polypus.run_quantum_circuit(qc, shots, infrastructure, **run_kwargs)``
    (``backend``, ``options``...) and reads the order of ``a`` from the
    outcomes, most frequent first. Errors of the run propagate.

    Reading the order tries every outcome, and several multiples of each order
    an outcome suggests, so for the ``n`` a simulator can run it finds the
    order even from counts that carry no signal: ``method="quantum"`` says
    where the order was read, not that the circuit worked.

    To judge the circuit, compare ``peak_share`` with two levels. An ideal
    circuit lands on the peaks with probability ``(order - 1) / order`` when
    ``order`` is a power of two below ``2**counting_qubits``, and at least
    85% of that otherwise; ``peak_share`` estimates it from the ``shots``
    outcomes and scatters around it, by about 0.014 for order 4 at 1024
    shots. Uniformly random outcomes land there with probability
    ``uniform_peak_share``, a few percent at most with the default
    ``counting_qubits``. The peak at 0 is left out: every order has it, and
    it is all that a circuit whose multiplications do nothing measures.
    Hardware noise is not uniform, though: decay and readout errors favour
    zeros, which can put several times ``uniform_peak_share`` on the peaks of
    an even order without any signal, most of all for a power of two. A
    ``peak_share`` above ``uniform_peak_share`` but far below the ideal level
    therefore does not show that the circuit worked. With too few counting
    qubits for the order, ``uniform_peak_share`` comes close to the ideal
    level, and from ``2 * order >= 2**counting_qubits`` it is at or above
    it: the counts cannot tell.

    ``a`` (between 2 and ``n - 1``) is the base of every attempt when given,
    and ``factor`` stops as soon as it finds its order, even one that does not
    split ``n`` (an odd order, or ``a**(r/2) % n == n - 1``): another run
    would find the same. Without it each attempt draws a new base. After
    ``max_attempts`` circuits without factors, the result has
    ``factors=None``.

    ``seed`` (0 to ``2**64 - 1``) draws the bases and the seed of every run,
    so the same seed on a seeded backend gives the same result. Without it no
    run gets a seed, which ``qmio`` (real hardware) refuses.

    ``counting_qubits`` is between 1 and ``2 * n.bit_length()``, its default;
    fewer make each run cheaper and its counts less precise. The circuit has
    ``counting_qubits + 2 * n.bit_length() + 2`` qubits. The local simulators
    hold 30 (``n < 128`` with the default register), but each qubit more
    doubles the time and memory of a run: on four cores an attempt takes
    seconds for ``n < 16``, about a minute for ``n < 32``, half an hour for
    ``n < 64`` (26 qubits, 1 GiB) and, at 30 qubits, 16 GiB and many hours.

    ```python
    from polypus.shor import factor

    result = factor(15, a=7, infrastructure="local", backend="polypus", seed=1)
    assert result.factors == (3, 5) and result.order == 4
    ```
    """
    n = _integer(n, "n")
    shots = _integer(shots, "shots")
    max_attempts = _integer(max_attempts, "max_attempts")
    a = _optional_integer(a, "a")
    counting_qubits = _optional_integer(counting_qubits, "counting_qubits")
    seed = _optional_integer(seed, "seed")
    if seed is not None and not 0 <= seed < 1 << 64:
        raise OverflowError(f"seed must be between 0 and 2**64 - 1, got {seed}")
    if not 4 <= n < _MAX_MODULUS:
        raise ValueError(f"n must be between 4 and 2**24 - 1, got {n}")
    if _is_prime(n):
        raise ValueError(f"n must be composite, got the prime {n}")
    if shots < 1:
        raise ValueError(f"shots must be at least 1, got {shots}")
    if shots >= 1 << 32:
        raise OverflowError(f"shots must be below 2**32, got {shots}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    if a is not None and not 2 <= a < n:
        raise ValueError(f"a must be between 2 and n - 1 = {n - 1}, got {a}")
    if counting_qubits is not None and not 1 <= counting_qubits <= 2 * n.bit_length():
        raise ValueError(
            "counting_qubits must be between 1 and 2 * n.bit_length() = "
            f"{2 * n.bit_length()}, got {counting_qubits}"
        )

    if n % 2 == 0:
        return ShorResult(n, (2, n // 2), "even", None, None, 0, None, None, None)
    root = _perfect_power_root(n)
    if root is not None:
        return ShorResult(
            n, (root, n // root), "perfect_power", None, None, 0, None, None, None
        )

    # The extension is loaded after this module.
    import polypus

    t = 2 * n.bit_length() if counting_qubits is None else counting_qubits
    rng = random.Random(seed)
    qc: Optional[polypus.Circuit] = None
    base: Optional[int] = None
    order: Optional[int] = None
    counts: Optional[dict[str, int]] = None
    for attempt in range(max_attempts):
        # Not a security use (python:S2245): a base need not be unpredictable.
        base = rng.randrange(2, n) if a is None else a  # NOSONAR
        divisor = math.gcd(base, n)
        if divisor > 1:
            shortcut = _ordered(divisor, n // divisor)
            return ShorResult(n, shortcut, "gcd", base, None, attempt, None, None, None)
        if qc is None or a is None:
            # One circuit alive at a time: free the last attempt's first.
            qc = None
            qc = polypus.circuits.templates.shor_order_finding(base, n, counting_qubits)
            for qubit in range(t):
                qc.measure(qubit, qubit)
        # qmio, real hardware, refuses a seed: an unseeded call seeds no run.
        # Not a security use either: a seed must reproduce its run (C-7).
        run_seed = None if seed is None else rng.getrandbits(63)  # NOSONAR
        run = polypus.run_quantum_circuit(
            qc, shots, infrastructure, seed=run_seed, **run_kwargs
        )
        counts = run.merged_counts or {}
        order = _order_from_counts(counts, t, base, n)
        if order is None:
            continue
        factors = factors_from_order(base, n, order)
        if factors is not None:
            shares = _peak_shares(counts, t, order)
            return ShorResult(
                n, factors, "quantum", base, order, attempt + 1, *shares, counts
            )
        if a is not None:
            shares = _peak_shares(counts, t, order)
            return ShorResult(n, None, None, base, order, attempt + 1, *shares, counts)
    # The last attempt's order and counts, never an earlier attempt's shares.
    shares = _peak_shares(counts, t, order)
    return ShorResult(n, None, None, base, order, max_attempts, *shares, counts)


def order_from_measurement(y: int, t: int, a: int, n: int) -> Optional[int]:
    """The order of ``a`` modulo ``n`` read from ``y``, one outcome of a
    ``t``-qubit counting register, or ``None`` when ``y`` does not give it.

    ``y / 2**t`` is close to ``s / r`` for the order ``r`` and some ``s``.
    Each convergent ``s / r`` of its continued fraction with ``r < n`` is
    tried with its multiples below ``n``, up to ``n.bit_length()`` of them
    (``s`` and the order may share a factor), and the first multiple ``m``
    with ``a**m % n == 1`` is reduced to the order. ``y = 0`` gives ``None``.

    ``t`` must be at least 1, ``y`` between 0 and ``2**t - 1``, ``n`` between
    2 and ``2**24 - 1`` and ``a`` between 1 and ``n - 1`` (otherwise
    ``ValueError``), all of them integers (otherwise ``TypeError``).
    """
    y = _integer(y, "y")
    t = _integer(t, "t")
    a = _integer(a, "a")
    n = _integer(n, "n")
    if t < 1:
        raise ValueError(f"t must be at least 1, got {t}")
    if not 0 <= y < 1 << t:
        raise ValueError(f"y must be between 0 and 2**t - 1 = {(1 << t) - 1}, got {y}")
    _check_base_and_modulus(a, n)
    for s, r in _convergents(y, 1 << t):
        if r >= n:
            break
        if s == 0 or r < 2:
            continue
        for k in range(1, n.bit_length() + 1):
            multiple = k * r
            if multiple >= n:
                break
            if pow(a, multiple, n) == 1:
                return _order_dividing(a, n, multiple)
    return None


def factors_from_order(a: int, n: int, r: int) -> Optional[tuple[int, int]]:
    """Two factors of ``n`` from ``r``, the order of ``a`` modulo ``n``, or
    ``None`` when it does not split ``n``.

    For an even ``r`` with ``x = a**(r/2) % n`` other than ``n - 1``, ``n``
    divides ``(x - 1) * (x + 1)`` and neither of them, so ``gcd(x - 1, n)``
    or ``gcd(x + 1, n)`` is a factor. The result is ``(p, q)`` with
    ``p <= q`` and ``p * q == n``; ``None`` for an odd ``r``, for
    ``x == n - 1``, and when neither gcd is a factor, as for a multiple of the
    order (``x == 1``).

    ``n`` must be between 2 and ``2**24 - 1``, ``a`` between 1 and ``n - 1``
    and ``r`` at least 1 (otherwise ``ValueError``), all of them integers
    (otherwise ``TypeError``).
    """
    a = _integer(a, "a")
    n = _integer(n, "n")
    r = _integer(r, "r")
    _check_base_and_modulus(a, n)
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    if r % 2:
        return None
    x = pow(a, r // 2, n)
    if x == n - 1:
        return None
    for candidate in (x - 1, x + 1):
        divisor = math.gcd(candidate, n)
        if 1 < divisor < n:
            return _ordered(divisor, n // divisor)
    return None


def _integer(value: Any, name: str) -> int:
    """``value`` as an ``int``: a ``bool``, a ``float`` or any other value
    that is not an integer is a ``TypeError``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None


def _optional_integer(value: Any, name: str) -> Optional[int]:
    return None if value is None else _integer(value, name)


def _check_base_and_modulus(a: int, n: int) -> None:
    if not 2 <= n < _MAX_MODULUS:
        raise ValueError(f"n must be between 2 and 2**24 - 1, got {n}")
    if not 1 <= a < n:
        raise ValueError(f"a must be between 1 and n - 1 = {n - 1}, got {a}")


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % p for p in range(2, math.isqrt(n) + 1))


def _perfect_power_root(n: int) -> Optional[int]:
    """The smallest ``b`` with ``n == b**k`` for some ``k >= 2``, or ``None``."""
    for k in range(n.bit_length(), 1, -1):
        root = round(math.pow(n, 1 / k))
        for b in (root - 1, root, root + 1):
            if b >= 2 and b**k == n:
                return b
    return None


def _ordered(p: int, q: int) -> tuple[int, int]:
    return (p, q) if p <= q else (q, p)


def _order_from_counts(counts: dict[str, int], t: int, a: int, n: int) -> Optional[int]:
    """The order of ``a`` given by the most frequent outcome that gives one
    (ties by bitstring), or ``None``. Little-endian bitstrings (contract C-3):
    ``int(bits, 2)`` reads qubit 0 as the least-significant bit."""
    for bits, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        order = order_from_measurement(int(bits, 2), t, a, n)
        if order is not None:
            return order
    return None


def _peak_shares(
    counts: Optional[dict[str, int]], t: int, order: Optional[int]
) -> tuple[Optional[float], Optional[float]]:
    """``peak_share`` and ``uniform_peak_share`` (see :class:`ShorResult`)
    for the ``counts`` of a ``t``-qubit counting register and the ``order``
    read from them, or ``None`` twice without an order.

    The ``s`` with ``|y - s * 2**t / order| <= 1`` run from
    ``ceil((y - 1) * order / 2**t)`` to ``floor((y + 1) * order / 2**t)``,
    so the test is exact in integers. A key that is not an integer from 0 to
    ``2**t - 1`` is a shot that is never near: a malformed key lowers the
    share instead of raising. Of the ``2**t`` outcomes,
    ``2 * order + gcd(order, 2**t) - 3`` are near some ``s`` from 1 to
    ``order - 1`` while ``2 * order < 2**t`` (the peaks are more than 2
    apart, and 0, 1 and ``2**t - 1`` are near only ``s = 0`` or
    ``s = order``), all but 0 while ``order < 2**t <= 2 * order``, and all of
    them from ``order >= 2**t``.
    """
    if counts is None or order is None:
        return None, None
    size = 1 << t
    near = total = 0
    for bits, count in counts.items():
        if count <= 0:
            continue
        total += count
        try:
            y = int(bits, 2)
        except (TypeError, ValueError):
            continue
        lowest = -((1 - y) * order // size)
        highest = (y + 1) * order // size
        if 0 <= y < size and max(lowest, 1) <= min(highest, order - 1):
            near += count
    if order >= size:
        uniform = size
    elif 2 * order >= size:
        uniform = size - 1
    else:
        uniform = 2 * order + math.gcd(order, size) - 3
    return near / total if total else 0.0, uniform / size


def _convergents(numerator: int, denominator: int) -> Iterator[tuple[int, int]]:
    """The convergents ``s / r`` of ``numerator / denominator``, in order."""
    s, s_before = 1, 0
    r, r_before = 0, 1
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        s, s_before = quotient * s + s_before, s
        r, r_before = quotient * r + r_before, r
        yield s, r
        numerator, denominator = denominator, remainder


def _order_dividing(a: int, n: int, multiple: int) -> int:
    """The order of ``a`` modulo ``n``, given a ``multiple`` of it: each prime
    factor (by trial division, cheap below ``2**24``) is divided out while
    ``a`` to the quotient is still 1."""
    order = multiple
    for p in _prime_factors(multiple):
        while order % p == 0 and pow(a, order // p, n) == 1:
            order //= p
    return order


def _prime_factors(m: int) -> list[int]:
    primes = []
    p = 2
    while p * p <= m:
        if m % p == 0:
            primes.append(p)
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        primes.append(m)
    return primes
