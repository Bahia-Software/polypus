#!/usr/bin/env python3
"""A **filled-in** Polypus subprocess-bridge worker (protocol v1).

This is `crates/polypus-subprocess-backend/python/worker_template.py` with the one
function you are meant to change — `execute_circuits` — filled in with a trivial,
hardware-free "device" (all shots on the all-zeros bitstring). In a real backend you
replace that one function with a call into your provider's Python SDK; **everything
else — the framing, the handshake, the SIGINT-driven abort — is the frozen protocol
contract and is copied verbatim from the template.**

Point Polypus at it with:

    infrastructure="subprocess", options={"command": "python3 /path/to/worker.py"}

It passes the Polypus conformance battery (see `tests/conformance.rs`).

## Protocol (v1) — do not change

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
# THE ONE FUNCTION YOU REPLACE. Turn circuits + shots into counts by calling your
# provider's Python SDK. This example is a trivial deterministic "device": every shot
# reads the all-zeros bitstring, so it needs no hardware and its counts are valid
# (one non-empty map per circuit, summing to `shots`, bitstring keys). Return one
# dict {bitstring: count} per circuit, in order.
#
# A real fill looks like, e.g. (Qiskit Aer):
#     from qiskit import QuantumCircuit
#     from qiskit_aer import AerSimulator
#     sim = AerSimulator(seed_simulator=seed)
#     out = []
#     for c in circuits:
#         qc = QuantumCircuit.from_qasm_str(c["qasm"])
#         out.append(sim.run(qc, shots=shots).result().get_counts())
#     return out
# --------------------------------------------------------------------------------
def execute_circuits(circuits, shots, seed):
    _ = seed  # this trivial device is deterministic; a real one would seed its RNG
    out = []
    for c in circuits:
        n = int(c.get("n_qubits", 1)) or 1
        # All shots on the all-zeros outcome of the right width.
        out.append({"0" * n: shots})
    return out


def _run(req, stdout):
    global _ABORT
    _ABORT = False
    # --- test-only hooks, gated behind env vars (harmless in production) ---
    # These let the conformance battery drive this worker into each fault. Delete
    # them in your real backend.
    crash_after = os.environ.get("WORKER_CRASH_AFTER_MS")
    if crash_after is not None:
        time.sleep(int(crash_after) / 1000.0)
        _log("self-crashing (os._exit) for the crash/unresponsive test")
        os._exit(137)  # 128 + 9, mimics a killed process
    if os.environ.get("WORKER_ERROR") is not None:
        # A clean, in-band provider failure (maps to BackendError::External).
        _write_frame(
            stdout,
            {
                "op": "error",
                "id": req.get("id"),
                "message": os.environ["WORKER_ERROR"],
            },
        )
        return
    delay = os.environ.get("WORKER_RUN_DELAY_MS")
    # ----------------------------------------------------------------------
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
