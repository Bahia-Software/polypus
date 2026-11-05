"""
The JSON Schema of the call trace, version 1 (contract C-12).

The schema lives in ``docs/CONTRACTS.md``, in the section of C-12, and
``call_trace.load_schema`` reads it from there, so these tests check the
document itself. Every trace the tests read is validated line by line against
it (``call_trace.read_trace`` does it by default). Here, the traces of every
optimizer and of a circuit run, on one QPU and on several, validate, and the
schema lists, for each event, exactly the fields ``check_trace`` expects. It
also rejects what version 1 does not allow: a field it does not list, a
missing one, a value of the wrong type (a boolean where a number goes
included), out of range or outside its set (``true`` for the constant ``1``
included), a replica out of range and an unknown event; the ``ValueError``
names the offending key. ``read_trace`` validates every whole line that is
JSON, so one that is not an object raises rather than ends the read.
"""

import json

import polypus
import pytest
from call_trace import (
    KEYS,
    OPTIMIZERS,
    Trace,
    check_trace,
    load_schema,
    method,
    read_trace,
    train,
    validate,
)

VALID_BATCH = {
    "ev": "batch",
    "k": 0,
    "op": "evaluate",
    "n": 1,
    "shots": 1,
    "t": 0,
    "dt": 0,
    "ok": True,
}
VALID_RUN = {
    "v": 1,
    "ev": "run",
    "flow": "train",
    "id": "x",
    "polypus": "0.8.0",
    "method": "spsa",
    "dims": 2,
    "backend": "polypus",
    "infrastructure": "local",
    "n_qpus": 1,
    "shots": 64,
    "seed": 11,
    "t0_unix_ns": 0,
}
VALID_WAVE = {
    "ev": "wave",
    "k": 0,
    "w": 0,
    "op": "run_shots_distributed",
    "n": 2,
    "shots": 3,
    "t": 0,
    "dt": 0,
    "ok": True,
    "replicas": [2, 1],
}


@pytest.mark.integration
def test_trace_schema(tmp_path):
    schema = load_schema()
    assert schema["$id"] == "polypus-call-trace-v1"
    events = ("run", "batch", "wave", "iter", "metric", "end")
    assert schema["oneOf"] == [{"$ref": f"#/$defs/{ev}"} for ev in events]
    assert set(schema["$defs"]) == set(events)
    for ev, definition in schema["$defs"].items():
        assert definition["properties"]["ev"]["const"] == ev
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == KEYS[ev]
        assert set(definition["properties"]) == KEYS[ev] | (
            {"replicas"} if ev == "wave" else set()
        )
    paths = []
    for name in OPTIMIZERS:
        paths.append(tmp_path / f"{name}.jsonl")
        train(method(name), final_shots=64, trace=paths[-1])
    for n_qpus in (1, 4):
        paths.append(tmp_path / f"rqc-{n_qpus}.jsonl")
        polypus.run_quantum_circuit(
            polypus.Circuit(2).h(0).measure_all(),
            100,
            "local",
            n_qpus=n_qpus,
            backend="polypus",
            seed=1,
            trace=paths[-1],
        )
    kinds, wave_ops = set(), set()
    for path in paths:
        trace = read_trace(path)  # validates every line against the schema
        check_trace(trace)
        assert trace.complete
        kinds.update(line["ev"] for line in trace.lines)
        wave_ops.update(wave["op"] for wave in trace.events("wave"))
    # Every event, and both kinds of wave, went through the schema.
    assert kinds == set(events)
    assert wave_ops == {"run_circuits", "run_shots_distributed"}


@pytest.mark.parametrize(
    ("bad", "key"),
    [
        ({**VALID_BATCH, "extra": 1}, "extra"),
        ({**VALID_BATCH, "k": -1}, "k"),
        ({**VALID_BATCH, "ok": 1}, "ok"),
        ({**VALID_BATCH, "k": True}, "k"),
        ({**VALID_BATCH, "k": None}, "k"),
        ({key: value for key, value in VALID_BATCH.items() if key != "dt"}, "dt"),
        ({**VALID_BATCH, "ev": "wave", "w": 0, "op": "sample"}, "op"),
        ({**VALID_WAVE, "replicas": [2, -1]}, "replicas"),
        (
            {
                "ev": "end",
                "status": "done",
                "batches": 0,
                "waves": 0,
                "circuits": 0,
                "truncated": False,
                "t": 0,
            },
            "status",
        ),
        (
            {"ev": "iter", "i": 1, "evals": 1, "batches": 0, "best": "high", "t": 0},
            "best",
        ),
        (
            {"ev": "iter", "i": 1, "evals": 1, "batches": 0, "best": True, "t": 0},
            "best",
        ),
        ({**VALID_RUN, "v": 2}, "v"),
        ({**VALID_RUN, "v": True}, "v"),
        ({**VALID_RUN, "seed": 1.5}, "seed"),
        ({"ev": "unknown"}, "ev"),
    ],
)
def test_the_schema_rejects_what_v1_does_not_allow(bad, key):
    schema = load_schema()
    validate(VALID_BATCH, schema)
    validate(VALID_RUN, schema)
    validate(VALID_WAVE, schema)
    with pytest.raises(ValueError, match=f"'{key}'"):
        validate(bad, schema)


def test_read_trace_validates_every_whole_line_that_is_json(tmp_path):
    path = tmp_path / "trace.jsonl"
    head, batch = (json.dumps(line).encode() for line in (VALID_RUN, VALID_BATCH))
    for value in (b"null", b"[]", b"1", b'"end"'):
        path.write_bytes(head + b"\n" + value + b"\n" + batch + b"\n")
        with pytest.raises(ValueError, match="line 2: a line is a JSON object"):
            read_trace(path)
        # Without validation, the read stops at it, as at a line that is not JSON.
        assert read_trace(path, validate=False) == Trace(
            lines=[VALID_RUN], complete=False
        )
    # A whole line that is not JSON, or not UTF-8, ends the read either way.
    for value in (b'{"ev": "batch", "k": 0', b"\xff"):
        path.write_bytes(head + b"\n" + value + b"\n" + batch + b"\n")
        for check in (True, False):
            assert read_trace(path, validate=check) == Trace(
                lines=[VALID_RUN], complete=False
            )
