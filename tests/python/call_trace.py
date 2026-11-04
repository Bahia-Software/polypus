"""
Helpers for the call trace tests (contract C-12).

Small seeded training runs of every optimizer on the native backend, a
tolerant reader of a trace file, a structural check of what a closed trace
holds and a check of each line against the trace's JSON Schema.

``read_trace`` reads a trace as a killed or crashed run can leave it: only
the lines a newline ends, up to the first that is not JSON, and it checks
each with ``validate`` against the JSON Schema of version 1, which
``load_schema`` reads from ``docs/CONTRACTS.md``. ``check_trace`` asserts what
every closed trace of schema v1 satisfies: the ``run`` header first and one
``end`` line last, exactly the fields of its ``ev`` on every line, ``end``
counts that match the lines, ``iter`` lines that count the batches before
them, and times that never run past ``end``.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import polypus
from ask_tell_helpers import RandomSearch

DIMENSIONS = 2
SHOTS = 64
FINAL_SHOTS = 4096
OPTIMIZERS = [
    "rand1bin",
    "shade",
    "lshade",
    "pso",
    "qng",
    "cmaes",
    "adam",
    "spsa",
    "rotosolve",
    "external",
]

# The fields of each line of schema v1, by its `ev`. A `wave` line also has
# `replicas` when, and only when, its `op` is `run_shots_distributed`.
KEYS: dict[str, frozenset[str]] = {
    "run": frozenset(
        {
            "v",
            "ev",
            "flow",
            "id",
            "polypus",
            "method",
            "dims",
            "backend",
            "infrastructure",
            "n_qpus",
            "shots",
            "seed",
            "t0_unix_ns",
        }
    ),
    "batch": frozenset({"ev", "k", "op", "n", "shots", "t", "dt", "ok"}),
    "wave": frozenset({"ev", "k", "w", "op", "n", "shots", "t", "dt", "ok"}),
    "iter": frozenset({"ev", "i", "evals", "batches", "best", "t"}),
    "metric": frozenset({"ev", "n", "t", "dt"}),
    "end": frozenset(
        {"ev", "status", "batches", "waves", "circuits", "truncated", "t"}
    ),
}


def all_ones(bits):
    """1 for the all-ones bitstring, 0 for any other."""
    return float(all(b == "1" for b in bits))


def circuit(dimensions=DIMENSIONS):
    """One qubit per parameter, each turned by its own ``ry``, all measured."""
    qc = polypus.Circuit(dimensions)
    for q in range(dimensions):
        qc.ry(q, polypus.Param(q))
    qc.measure_all()
    return qc


def method(name, dimensions=DIMENSIONS):
    """A small configuration of the optimizer ``name`` (a DE strategy,
    ``"pso"``, ``"qng"``, ``"cmaes"``, ``"adam"``, ``"spsa"``,
    ``"rotosolve"`` or ``"external"``) that no tolerance stops early.
    ``"external"`` is a fresh seeded random search, so two calls propose the
    same candidates."""
    if name == "pso":
        return polypus.PSO(generations=4, population_size=6, tolerance=0.0)
    if name == "qng":
        return polypus.QNG(lambda theta, a: 1.0, max_iters=3)
    if name == "cmaes":
        return polypus.CMAES(generations=4, tolerance=0.0, patience=100)
    if name == "adam":
        return polypus.Adam(
            max_iters=3, tolerance=0.0, patience=100, finite_difference_step=0.2
        )
    if name == "spsa":
        return polypus.SPSA(max_iters=3, resamplings=1, eval_interval=2)
    if name == "rotosolve":
        return polypus.Rotosolve(
            sweeps=2, reevaluate_every=3, tolerance=0.0, patience=1
        )
    if name == "external":
        return polypus.External(RandomSearch(dimensions, sizes=(4,), seed=3), 20)
    return polypus.DE(
        generations=4, population_size=6, tolerance=0.0, patience=100, strategy=name
    )


def train(
    method,
    *,
    qc=None,
    dimensions=DIMENSIONS,
    n_qpus=1,
    infrastructure="local",
    backend="polypus",
    expectation_function=all_ones,
    id="trace",
    seed=11,
    **kwargs,
):
    """``polypus.train`` with ``SHOTS`` shots, one node and one core per QPU,
    on ``circuit(dimensions)`` unless ``qc`` is given."""
    return polypus.train(
        circuit(dimensions) if qc is None else qc,
        method,
        shots=SHOTS,
        n_qpus=n_qpus,
        dimensions=dimensions,
        expectation_function=expectation_function,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id=id,
        backend=backend,
        seed=seed,
        **kwargs,
    )


def _floats(values):
    return None if values is None else tuple(value.hex() for value in values)


def result_bits(result):
    """Every field of a ``TrainResult``, each float as ``float.hex``, so two
    results compare equal only when they are equal bit for bit. The ``id``
    loses the UUID ``train`` appends to every run's."""
    return (
        _floats(result.best_params),
        result.best_fitness.hex(),
        result.iterations_run,
        result.converged,
        _floats(result.fitness_history),
        _floats(result.final_params),
        None if result.final_fitness is None else result.final_fitness.hex(),
        result.seed,
        result.id.rsplit("_", 1)[0],
    )


@dataclass(frozen=True)
class Trace:
    """A trace file as ``read_trace`` read it: its whole lines, decoded, and
    whether the trace is complete."""

    lines: list[dict]
    complete: bool

    @property
    def header(self):
        """The ``run`` line, or ``None`` if the trace does not start with one."""
        if self.lines and self.lines[0].get("ev") == "run":
            return self.lines[0]
        return None

    @property
    def end(self):
        """The ``end`` line, or ``None`` if the trace does not end with one."""
        if self.lines and self.lines[-1].get("ev") == "end":
            return self.lines[-1]
        return None

    def events(self, ev):
        """The lines whose ``ev`` is ``ev``, in order."""
        return [line for line in self.lines if line.get("ev") == ev]


# The document that holds the trace's JSON Schema, in the section of C-12, and
# the schema's `$id` there.
_CONTRACTS = Path(__file__).resolve().parents[2] / "docs" / "CONTRACTS.md"
_SCHEMA_ID = "polypus-call-trace-v1"

# The keywords `validate` checks. A schema with any other is refused rather
# than checked in part.
_KEYWORDS = frozenset(
    {
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minimum",
    }
)

# The Python type `json.loads` gives each JSON type, but for "integer" and
# "number" (see `_is_a`).
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def load_schema():
    """The JSON Schema of the trace, version 1: the first ```json block of
    ``docs/CONTRACTS.md`` that contains ``"$id": "polypus-call-trace-v1"``."""
    marker = f'"$id": "{_SCHEMA_ID}"'
    for block in _CONTRACTS.read_text(encoding="utf-8").split("```json\n")[1:]:
        code = block.split("```", 1)[0]
        if marker in code:
            return json.loads(code)
    raise AssertionError(f"no json block of {_CONTRACTS.name} contains {marker}")


def validate(line, schema):
    """Raise ``ValueError``, naming the offending key, unless ``line`` is a
    valid line of ``schema``, the trace's JSON Schema (``load_schema()``).

    The line is checked against the definition of its ``ev`` in the schema's
    ``$defs``, with the keywords the schema uses: ``type`` (a name or a list
    of names; an ``integer`` is an ``int`` that is not a ``bool``, and a
    ``number`` such an ``int`` or a finite ``float``, as JSON has no NaN or
    infinity), ``const`` and ``enum`` (where ``True`` is not ``1``),
    ``required``, ``properties``, ``additionalProperties: false``, ``items``
    and ``minimum``. A schema with any other keyword raises
    ``NotImplementedError``."""
    if not isinstance(line, dict):
        raise ValueError(f"a line is a JSON object, not {line!r}")
    if "ev" not in line:
        raise ValueError(f"'ev' is missing from {line!r}")
    definitions = schema["$defs"]
    ev = line["ev"]
    if not isinstance(ev, str) or ev not in definitions:
        raise ValueError(f"'ev' is {ev!r}, not one of {sorted(definitions)}")
    _check(line, definitions[ev], f"{ev} line")


# `read_trace`'s `validate` flag hides the function of that name inside it.
_validate_line = validate


def _check(value, schema, name):
    """Raise ``ValueError`` if ``value``, called ``name`` in the message,
    breaks ``schema``."""
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise NotImplementedError(f"{name}: schema keywords {sorted(unknown)}")
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_a(value, kind) for kind in kinds):
            raise ValueError(f"{name} is {value!r}, not of type {schema['type']}")
    if "const" in schema and not _equal(value, schema["const"]):
        raise ValueError(f"{name} is {value!r}, not {schema['const']!r}")
    if "enum" in schema and not any(_equal(value, item) for item in schema["enum"]):
        raise ValueError(f"{name} is {value!r}, not one of {schema['enum']}")
    if "minimum" in schema and _is_a(value, "number") and value < schema["minimum"]:
        raise ValueError(f"{name} is {value!r}, below the minimum {schema['minimum']}")
    if "items" in schema and isinstance(value, list):
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{name}[{index}]")
    if isinstance(value, dict):
        _check_fields(value, schema, name)


def _check_fields(value, schema, name):
    """The object keywords of ``_check``: ``additionalProperties``,
    ``required`` and ``properties``."""
    properties = schema.get("properties", {})
    if "additionalProperties" in schema:
        if schema["additionalProperties"] is not False:
            raise NotImplementedError(f"{name}: additionalProperties other than false")
        for key in value:
            if key not in properties:
                raise ValueError(f"{name}: {key!r} is not one of its fields")
    for key in schema.get("required", ()):
        if key not in value:
            raise ValueError(f"{name}: {key!r} is missing")
    for key, item in value.items():
        if key in properties:
            _check(item, properties[key], f"{name}: {key!r}")


def _is_a(value, kind):
    """Whether ``value``, as ``json.loads`` reads it, has the JSON type
    ``kind``."""
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        if isinstance(value, float):
            return math.isfinite(value)
        return isinstance(value, int) and not isinstance(value, bool)
    if kind not in _TYPES:
        raise NotImplementedError(f"the JSON type {kind!r}")
    return isinstance(value, _TYPES[kind])


def _equal(value, expected):
    """JSON equality, which, unlike Python's ``==``, does not take ``True``
    for ``1``."""
    return value == expected and isinstance(value, bool) == isinstance(expected, bool)


def read_trace(path, *, validate=True):
    """The trace at ``path``, as far as it can be read.

    Only the lines a newline ends are read, and none from the first that is
    not JSON (or not UTF-8) on, so a trace cut in the middle of a line reads
    up to its last whole line, without an exception. With ``validate``, every
    whole line that is JSON is checked against the trace's JSON Schema
    (``validate``, with ``load_schema()``), and the first that breaks it, one
    that is not a JSON object included, raises ``ValueError`` with its
    number. Without ``validate``, the read also stops, without an exception,
    at the first line that is JSON but not an object. ``complete`` is whether
    every byte was read and the last line is ``end``."""
    schema = load_schema() if validate else None
    *whole, tail = Path(path).read_bytes().split(b"\n")
    lines = []
    for raw in whole:
        try:
            line = json.loads(raw)
        except ValueError:  # not JSON, or not UTF-8 (UnicodeDecodeError)
            break
        if schema is not None:
            try:
                _validate_line(line, schema)
            except ValueError as err:
                raise ValueError(f"{path}, line {len(lines) + 1}: {err}") from err
        elif not isinstance(line, dict):
            break
        lines.append(line)
    complete = (
        tail == b""
        and len(lines) == len(whole)
        and bool(lines)
        and lines[-1].get("ev") == "end"
    )
    return Trace(lines=lines, complete=complete)


def check_trace(trace):
    """Assert that ``trace`` is a complete trace of schema v1.

    The ``run`` header comes first, with ``v == 1``, and one ``end`` line
    last. Every line has exactly the fields of its ``ev``, and ``replicas``
    only and always on a ``run_shots_distributed`` wave. The ``batches``,
    ``waves`` and ``circuits`` of ``end`` are the batch lines, the wave lines
    and the sum of the batches' ``n``; at least those if the trace was
    truncated, since ``end`` counts the events left out too. Each ``iter``
    counts the batch lines before it, the ``t`` of the ``iter`` and ``metric``
    lines never decreases, and no event ends (``t + dt``) after ``end``'s
    ``t``."""
    lines = trace.lines
    assert trace.complete, f"the trace is not complete: {lines[-1:]}"
    head, end = lines[0], lines[-1]
    assert head.get("ev") == "run" and head.get("v") == 1, head
    kinds = [line.get("ev") for line in lines]
    assert kinds.count("run") == 1 and kinds.count("end") == 1, kinds
    for line in lines:
        ev = line.get("ev")
        assert ev in KEYS, f"unknown ev: {line}"
        fields = set(line) - {"replicas"}
        assert fields == KEYS[ev], f"{ev} fields {sorted(fields)}: {line}"
        distributed = ev == "wave" and line["op"] == "run_shots_distributed"
        assert ("replicas" in line) == distributed, line
    batches, waves = trace.events("batch"), trace.events("wave")
    written = (len(batches), len(waves), sum(batch["n"] for batch in batches))
    counted = (end["batches"], end["waves"], end["circuits"])
    if end["truncated"]:
        assert all(c >= w for c, w in zip(counted, written)), (counted, written)
    else:
        assert counted == written, (counted, written)
    seen, last_t = 0, 0
    for line in lines[1:-1]:
        if line["ev"] == "batch":
            seen += 1
        if line["ev"] == "iter":
            assert line["batches"] == seen, f"{seen} batches before {line}"
        if line["ev"] in ("iter", "metric"):
            assert line["t"] >= last_t, f"t went back: {line}"
            last_t = line["t"]
        assert line["t"] + line.get("dt", 0) <= end["t"], f"after end: {line}"
