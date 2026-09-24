#!/usr/bin/env python3
"""Reference Polypus subprocess-bridge **worker** (protocol v1).

Copy this file, replace `execute_circuits` with a call into your provider's Python
SDK, and point Polypus at it with:

    infrastructure="subprocess", options={"command": "python3 /path/to/worker.py"}

Everything else here — the framing, the handshake, the SIGINT-driven abort — is the
stable contract the Rust bridge (`polypus-subprocess-backend`) speaks, and should be
kept as-is. See `docs/backends.md` for the full specification.

## Protocol (v1)

Framing: each message is `4-byte little-endian length || UTF-8 JSON payload`.
stdout carries protocol frames ONLY; log to stderr (mixing corrupts the stream).

Requests (bridge -> worker):
    {"op": "hello", "protocol": 1}
    {"op": "run", "id": "...", "circuits": [{"qasm": "...", "n_qubits": N}, ...],
                  "shots": S, "seed": <int|absent>}
    {"op": "shutdown"}

Responses (worker -> bridge):
    {"op": "ready",   "protocol": 1}
    {"op": "result",  "id": "...", "counts": [{bitstring: count, ...}, ...]}
    {"op": "aborted", "id": "...", "reason": "signal"}   # in-flight run cancelled
    {"op": "error",   "id": "...", "message": "..."}     # clean, in-band failure

## Liveness

The bridge bounds every read by a timeout. If your SDK call can block for a long
time, that is fine — size the bridge's `recv_timeout_ms` to your workload. A worker
that never replies (a hung SDK, a mute QPU) is detected by that timeout and killed.

## Cancellation

The bridge cancels an in-flight run by sending this process SIGINT (also delivered
naturally to the whole process group on a terminal Ctrl+C). The handler below sets a
flag; a cooperative SDK call should check `should_abort()` and stop, after which the
worker replies `aborted` and stays alive for the next run.
"""

import json
import os
import signal
import struct
import sys
import time

PROTOCOL_VERSION = 1

# Set by the SIGINT/SIGTERM handler so an in-flight run can notice and abort.
_ABORT = False


def _log(msg: str) -> None:
    print(f"[worker {os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _on_signal(signum, _frame):
    global _ABORT
    _ABORT = True
    _log(f"received signal {signum}; will abort in-flight run")


def should_abort() -> bool:
    """True once a cancellation signal has arrived. A long-running SDK call should
    poll this and stop cooperatively when it flips."""
    return _ABORT


def _read_frame(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF: the bridge closed the pipe / went away
    (length,) = struct.unpack("<I", header)
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body.decode("utf-8"))


def _write_frame(stream, obj) -> None:
    payload = json.dumps(obj).encode("utf-8")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


# --------------------------------------------------------------------------------
# The only function a real backend author replaces: turn circuits + shots into
# counts by calling the provider SDK. This stand-in returns deterministic fake
# counts (half all-zeros, half all-ones) so the bridge can be exercised without any
# hardware. Return one dict {bitstring: count} per circuit, in order.
# --------------------------------------------------------------------------------
def execute_circuits(circuits, shots, seed):
    # Bench hook: WORKER_SPREAD_KEYS=K (and optional WORKER_KEY_WIDTH=W) makes each
    # circuit return K distinct bitstrings so the payload-overhead benchmark can
    # measure a large, realistic counts payload. Harmless when unset.
    spread = int(os.environ.get("WORKER_SPREAD_KEYS", "0"))
    out = []
    for c in circuits:
        n = int(c.get("n_qubits", 1)) or 1
        if spread > 1:
            width = int(os.environ.get("WORKER_KEY_WIDTH", str(n))) or n
            k = spread
            base = shots // k
            rem = shots - base * k
            counts = {}
            for i in range(k):
                key = format(i, "b").zfill(width)[-width:]
                counts[key] = base + (1 if i < rem else 0)
            out.append(counts)
            continue
        lo = shots // 2
        hi = shots - lo
        counts = {}
        if lo:
            counts["0" * n] = lo
        if hi:
            counts["1" * n] = hi
        if not counts:  # shots == 0
            counts["0" * n] = 0
        out.append(counts)
    return out


def _run(req, stdout):
    global _ABORT
    _ABORT = False
    # --- test-only hooks, gated behind env vars (harmless in production) ---
    crash_after = os.environ.get("WORKER_CRASH_AFTER_MS")
    if crash_after is not None:
        time.sleep(int(crash_after) / 1000.0)
        _log("self-crashing (os._exit) for the crash test")
        os._exit(137)  # 128 + 9, mimics a killed process
    hang = os.environ.get("WORKER_HANG_MS")
    delay = os.environ.get("WORKER_RUN_DELAY_MS")
    # ----------------------------------------------------------------------
    if hang is not None:
        # Simulate a wedged SDK: sleep long, ignoring abort, so the bridge's read
        # timeout must fire. Used by the timeout test.
        time.sleep(int(hang) / 1000.0)
    if delay is not None:
        # An interruptible "QPU latency" so the cancel test can signal mid-run.
        deadline = time.monotonic() + int(delay) / 1000.0
        while time.monotonic() < deadline:
            if _ABORT:
                _write_frame(
                    stdout, {"op": "aborted", "id": req.get("id"), "reason": "signal"}
                )
                _log("run aborted; worker stays alive and reusable")
                return
            time.sleep(0.005)

    try:
        counts = execute_circuits(
            req.get("circuits", []), int(req.get("shots", 0)), req.get("seed")
        )
    except Exception as exc:  # a clean, in-band failure
        _write_frame(
            stdout,
            {
                "op": "error",
                "id": req.get("id"),
                "message": f"{type(exc).__name__}: {exc}",
            },
        )
        return
    if _ABORT:
        _write_frame(stdout, {"op": "aborted", "id": req.get("id"), "reason": "signal"})
        return
    _write_frame(stdout, {"op": "result", "id": req.get("id"), "counts": counts})


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    _log("started")
    while True:
        req = _read_frame(stdin)
        if req is None:
            _log("stdin EOF; exiting")
            return 0
        op = req.get("op")
        if op == "hello":
            _write_frame(stdout, {"op": "ready", "protocol": PROTOCOL_VERSION})
        elif op == "run":
            _run(req, stdout)
        elif op == "shutdown":
            _log("shutdown requested")
            return 0
        else:
            _write_frame(
                stdout,
                {"op": "error", "id": req.get("id"), "message": f"unknown op {op!r}"},
            )


if __name__ == "__main__":
    sys.exit(main())
