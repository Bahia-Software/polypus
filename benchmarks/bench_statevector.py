"""
Benchmark: the `polypus.statevector` hot path (and the native run path next to
it), sized from "tiny" to "the gate loop is everything".

Why it exists: `statevector` interrupts the simulation mid-run (issue #110) by
having `polypus-sim`'s gate loop poll a cancellation hook, which reacquires the
GIL to check for a pending signal. The hook is throttled inside `polypus-sim`
(one clock read per ~64k amplitude updates, one hook call per ~25ms of wall clock
*and* per 20x its own measured duration), and the claim that goes with it is that
this costs nothing measurable -- so it needs numbers, per docs/ENGINEERING.md §9.

Two paths are measured, because only one of them installs a hook:

1. `polypus.statevector` — the hooked path. Per gate it pays a counter
   decrement, occasionally an `Instant::now()`, and at most once per ~25ms a
   `Python::with_gil` + `PyErr_CheckSignals`.
2. `polypus.run_quantum_circuit` on the native backend — the *unhooked* path
   (`Simulator::run`, hook `None`), where the whole cost is one `Option` test
   per gate. Included so a regression there cannot hide.

Then a third measurement, for the one case where the hook is *not* cheap:
reacquiring the GIL costs ~1us when nothing else wants it, but milliseconds when
another Python thread is running, because the interpreter only yields on its
switch interval. One long simulation is timed alone and again with a Python
thread spinning; the reported C/U ratio is machine-speed independent, so it can
be compared across builds. That ratio is what justifies
`CANCELLATION_OVERHEAD_DIVISOR` in polypus-sim (reference machine, 14q x 400:
1.21 with no hook at all, 1.53 on a fixed 25ms cadence, 1.25 once the interval
adapts to the hook's measured cost).

Usage:
    # before the change (or on main):
    python benchmarks/bench_statevector.py --label baseline > /tmp/base.txt
    # after:
    python benchmarks/bench_statevector.py --label patched > /tmp/patched.txt
    diff -y /tmp/base.txt /tmp/patched.txt

Set RAYON_NUM_THREADS=1 to make the numbers core-count independent (the two
largest cases are above polypus-sim's parallel threshold).
"""

import argparse
import math
import threading
import time

import polypus

# (qubits, layer repetitions). The first three are the "common case" the
# acceptance criterion cares about: circuits that finish in well under the
# checkpoint interval, so an ideal throttle never calls the hook at all. The last
# two are long enough for the hook to actually fire, once per ~25ms.
CASES = [
    (4, 2),
    (8, 10),
    (11, 100),
    (14, 150),
    (20, 15),
]


def make(n, reps):
    """A GHZ-ish layer followed by `reps` cx+rx layers (no measurements)."""
    qc = polypus.Circuit(n)
    for q in range(n):
        qc = qc.h(q)
    for _ in range(reps):
        for q in range(n - 1):
            qc = qc.cx(q, q + 1)
        for q in range(n):
            qc = qc.rx(q, 0.3)
    return qc


def bench(fn, seconds=0.5, repeat=5):
    """Best-of-`repeat` mean microseconds per call, each round filling roughly
    `seconds` of wall clock (a minimum of 3 calls, so the slow cases stay
    affordable). Best-of, not mean-of, to shed scheduler noise."""
    best, calls_used = math.inf, 0
    for _ in range(repeat):
        # Calibrate the call count from one warm call.
        t0 = time.perf_counter()
        fn()
        single = time.perf_counter() - t0
        calls = max(3, int(seconds / single)) if single > 0 else 100
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        us = (time.perf_counter() - t0) / calls * 1e6
        if us < best:
            best, calls_used = us, calls
    return best, calls_used


def one_run(qc):
    """Seconds for a single `statevector` call."""
    t0 = time.perf_counter()
    polypus.statevector(qc)
    return time.perf_counter() - t0


def with_spinner(fn):
    """Run `fn` while a Python thread spins a counter loop, so the GIL is
    genuinely contended for the whole call."""
    stop = threading.Event()

    def spin():
        n = 0
        while not stop.is_set():
            n += 1

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        return fn()
    finally:
        stop.set()
        worker.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="tag printed with the results")
    args = ap.parse_args()

    # Warm up the lazy `polypus_python` import outside every measured window.
    polypus.statevector(polypus.Circuit(2).h(0))
    polypus.run_quantum_circuit(
        polypus.Circuit(2).h(0).measure_all(),
        shots=16,
        infrastructure="local",
        n_qpus=1,
        backend="polypus",
    )

    tag = f" [{args.label}]" if args.label else ""
    print(f"polypus.statevector{tag}")
    print(f"  {'circuit':>16}  {'gates':>7}  {'us/call':>12}  {'calls':>7}")
    for n, reps in CASES:
        qc = make(n, reps)
        gates = (2 * n - 1) * reps + n
        us, calls = bench(lambda qc=qc: polypus.statevector(qc))
        print(f"  {f'{n}q x {reps} reps':>16}  {gates:>7}  {us:>12.3f}  {calls:>7}")

    print(f"\npolypus.run_quantum_circuit, native backend (no hook){tag}")
    print(f"  {'circuit':>16}  {'gates':>7}  {'us/call':>12}  {'calls':>7}")
    for n, reps in CASES[:4]:
        qc = make(n, reps).measure_all()
        gates = (2 * n - 1) * reps + n + 1
        us, calls = bench(
            lambda qc=qc: polypus.run_quantum_circuit(
                qc, shots=256, infrastructure="local", n_qpus=1, backend="polypus"
            )
        )
        print(f"  {f'{n}q x {reps} reps':>16}  {gates:>7}  {us:>12.3f}  {calls:>7}")

    # The contended case: one long simulation, alone and against a spinning
    # Python thread. Sized so the run spans many checkpoint intervals -- which is
    # what the adaptive interval is there for.
    n, reps = 14, 400
    qc = make(n, reps)
    print(f"\npolypus.statevector against a spinning Python thread{tag}")
    uncontended = min(one_run(qc) for _ in range(3))
    contended = min(with_spinner(lambda: one_run(qc)) for _ in range(3))
    print(
        f"  {n}q x {reps} reps: alone {uncontended * 1e3:.1f}ms, contended "
        f"{contended * 1e3:.1f}ms, C/U {contended / uncontended:.3f}"
    )


if __name__ == "__main__":
    main()
