#!/usr/bin/env python3
"""Trivial stand-in for a third-party Python QPU backend.

This is NOT a real provider SDK. It speaks the spike's fixed protocol so we can
validate the *bridge* (process lifecycle + framing + failure handling), not any
particular vendor. The protocol:

    frame = 4-byte little-endian unsigned length  ||  UTF-8 JSON payload

Requests (Rust -> worker), one JSON object per frame:
    {"op": "ping", "id": "..."}
    {"op": "run",  "id": "...", "circuits": [{"qasm": "...", "n_qubits": N}, ...],
                    "shots": S, "sleep_ms": M}   # sleep_ms simulates QPU latency
    {"op": "shutdown"}

Responses (worker -> Rust):
    {"op": "pong",    "id": "..."}
    {"op": "result",  "id": "...", "counts": [ {bitstring: count, ...}, ... ]}
    {"op": "aborted", "id": "...", "reason": "signal"}   # in-flight op cancelled

Anything the worker wants to log goes to stderr; stdout carries protocol frames
ONLY. Mixing the two would corrupt the stream.
"""
import json
import os
import signal
import struct
import sys
import time

# Set when a cancellation signal (SIGINT/SIGTERM) arrives so an in-flight,
# chunked sleep can notice and abort instead of running to completion.
_ABORT = False


def _log(msg: str) -> None:
    print(f"[worker {os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _on_signal(signum, _frame):
    # A blocked syscall (our sleep) is interrupted; we do minimal work here and
    # let the main loop translate the flag into an "aborted" response. This is
    # how the subprocess *learns it must abort* — the bridge sends it a signal.
    global _ABORT
    _ABORT = True
    _log(f"received signal {signum}; will abort in-flight op")


def _read_frame(stream) -> dict | None:
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF: parent closed the pipe / went away
    (length,) = struct.unpack("<I", header)
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body.decode("utf-8"))


def _write_frame(stream, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def _interruptible_sleep(total_ms: int) -> bool:
    """Sleep in small chunks so a signal can cut it short. Returns True if it
    ran to completion, False if it was aborted."""
    global _ABORT
    deadline = time.monotonic() + total_ms / 1000.0
    while time.monotonic() < deadline:
        if _ABORT:
            return False
        time.sleep(0.005)
    return not _ABORT


def _fake_counts(circuits, shots: int) -> list[dict]:
    """Deterministic trivial 'measurement': put half the shots on the all-zero
    string and half on the all-one string. No real simulation — this is a
    bridge spike, not a backend."""
    out = []
    for c in circuits:
        n = int(c.get("n_qubits", 1))
        zero = "0" * n
        one = "1" * n
        lo = shots // 2
        hi = shots - lo
        out.append({zero: lo, one: hi})
    return out


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    # Optional self-crash hook for the crash test: if asked, die hard mid-op.
    crash_after = os.environ.get("WORKER_CRASH_AFTER_MS")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    _log("ready")

    while True:
        global _ABORT
        _ABORT = False
        req = _read_frame(stdin)
        if req is None:
            _log("stdin EOF; exiting")
            return 0
        op = req.get("op")

        if op == "ping":
            _write_frame(stdout, {"op": "pong", "id": req.get("id")})
        elif op == "shutdown":
            _log("shutdown requested")
            return 0
        elif op == "run":
            sleep_ms = int(req.get("sleep_ms", 0))
            if crash_after is not None:
                # Simulate an unexpected hard crash mid-call (segfault-like):
                # do part of the work, then abort the *process* with no reply.
                time.sleep(int(crash_after) / 1000.0)
                _log("self-crashing (os._exit / SIGKILL-equivalent)")
                os._exit(137)  # 128 + 9, mimics a killed process
            completed = _interruptible_sleep(sleep_ms) if sleep_ms else True
            if not completed:
                _write_frame(stdout, {"op": "aborted", "id": req.get("id"),
                                      "reason": "signal"})
                _log("op aborted; worker stays alive and reusable")
                continue
            counts = _fake_counts(req.get("circuits", []), int(req.get("shots", 0)))
            _write_frame(stdout, {"op": "result", "id": req.get("id"),
                                  "counts": counts})
        else:
            _write_frame(stdout, {"op": "error", "id": req.get("id"),
                                  "message": f"unknown op {op!r}"})


if __name__ == "__main__":
    sys.exit(main())
