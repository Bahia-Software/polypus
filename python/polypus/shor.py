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
    ``"quantum"`` from the order the circuit measured, and ``None`` without
    factors. ``a`` is the base of the last attempt (``None`` for the first two
    shortcuts), ``order`` its order when the counts gave it, and ``counts`` the
    counts of its run (``None`` when it ran no circuit). ``attempts`` is the
    number of circuits run.
    """

    n: int
    factors: Optional[tuple[int, int]]
    method: Optional[str]
    a: Optional[int]
    order: Optional[int]
    attempts: int
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
    register, runs it ``shots`` times with
    ``polypus.run_quantum_circuit(qc, shots, infrastructure, **run_kwargs)``
    (``backend``, ``options``...) and reads the order of ``a`` from the most
    frequent outcomes. Errors of the run propagate.

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
    fewer save qubits and find the order less often. The circuit has
    ``counting_qubits + 2 * n.bit_length() + 2`` qubits: with the local
    simulators, which hold 30 qubits, and the default counting register, the
    practical limit is ``n < 128``.

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
        return ShorResult(n, (2, n // 2), "even", None, None, 0, None)
    root = _perfect_power_root(n)
    if root is not None:
        return ShorResult(n, (root, n // root), "perfect_power", None, None, 0, None)

    # The extension is loaded after this module.
    import polypus

    t = 2 * n.bit_length() if counting_qubits is None else counting_qubits
    rng = random.Random(seed)
    qc: Optional[polypus.Circuit] = None
    base: Optional[int] = None
    order: Optional[int] = None
    counts: Optional[dict[str, int]] = None
    for attempt in range(max_attempts):
        base = rng.randrange(2, n) if a is None else a
        divisor = math.gcd(base, n)
        if divisor > 1:
            shortcut = _ordered(divisor, n // divisor)
            return ShorResult(n, shortcut, "gcd", base, None, attempt, None)
        if qc is None or a is None:
            qc = polypus.circuits.templates.shor_order_finding(base, n, counting_qubits)
            for qubit in range(t):
                qc.measure(qubit, qubit)
        # qmio, real hardware, refuses a seed: an unseeded call seeds no run.
        run_seed = None if seed is None else rng.getrandbits(63)
        run = polypus.run_quantum_circuit(
            qc, shots, infrastructure, seed=run_seed, **run_kwargs
        )
        counts = run.merged_counts or {}
        order = _order_from_counts(counts, t, base, n)
        if order is None:
            continue
        factors = factors_from_order(base, n, order)
        if factors is not None:
            return ShorResult(n, factors, "quantum", base, order, attempt + 1, counts)
        if a is not None:
            return ShorResult(n, None, None, base, order, attempt + 1, counts)
    return ShorResult(n, None, None, base, order, max_attempts, counts)


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
