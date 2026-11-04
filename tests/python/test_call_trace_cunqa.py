"""
The call trace (contract C-12) of a CUNQA run, against a fake CUNQA SDK.

``polypus.run_quantum_circuit`` and ``polypus.train`` with
``infrastructure="cunqa"`` go through ``fake_cunqa.FakeCunqa``, which records
every SDK call and every job with its QPU, seed and shots. Tracing a run
changes none of them: the same calls in the same order, the same seeds and the
same counts or result, with the shots of a circuit split across QPUs
(``run_shots_distributed``, whose ``wave`` line lists each replica's shots) and
with a training run's waves no wider than its QPUs. A job that fails closes
the trace with ``error``.
"""

import pytest

pytest.importorskip("qiskit_aer")

import polypus
from call_trace import check_trace, read_trace, result_bits
from fake_cunqa import FakeCunqa


def _uniform3():
    return polypus.Circuit(3).h(0).h(1).h(2).measure_all()


def _runs(fake):
    """Each job the fake ran, as its QPU's index, its seed and its shots."""
    return [
        (run["qpu"].index, run["options"]["seed"], run["options"]["shots"])
        for run in fake.runs
    ]


@pytest.mark.parametrize(("shots", "replicas"), [(10, [4, 3, 3]), (2, [1, 1, 0])])
def test_cunqa_trace_forwarding(monkeypatch, tmp_path, shots, replicas):
    seen = {}
    for traced in (False, True):
        fake = FakeCunqa().install(monkeypatch)
        extra = {"trace": tmp_path / "rqc.jsonl"} if traced else {}
        result = polypus.run_quantum_circuit(
            _uniform3(),
            shots=shots,
            infrastructure="cunqa",
            backend="aer",
            n_qpus=3,
            seed=11,
            **extra,
        )
        seen[traced] = (fake.names(), _runs(fake), result.counts)
    assert seen[True] == seen[False]
    trace = read_trace(tmp_path / "rqc.jsonl")
    check_trace(trace)
    assert [s for wave in trace.events("wave") for s in wave["replicas"]] == replicas


def test_cunqa_train_trace_forwarding(monkeypatch, tmp_path):
    seen = {}
    for traced in (False, True):
        fake = FakeCunqa().install(monkeypatch)
        extra = {"trace": tmp_path / "train.jsonl"} if traced else {}
        result = polypus.train(
            polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
            polypus.DE(generations=1, population_size=4),
            shots=16,
            n_qpus=2,
            dimensions=1,
            expectation_function=lambda bits: float(bits == "1"),
            infrastructure="cunqa",
            nodes=1,
            cores_per_qpu=1,
            id="cunqa_trace",
            backend="aer",
            seed=3,
            **extra,
        )
        seen[traced] = (fake.names(), _runs(fake), result_bits(result))
    assert seen[True] == seen[False]
    assert [seed for _, seed, _ in seen[True][1][:2]] == [3, 4]
    trace = read_trace(tmp_path / "train.jsonl")
    check_trace(trace)
    waves = trace.events("wave")
    assert all(wave["n"] <= 2 for wave in waves), (
        "waves follow CUNQA's capabilities (n_qpus)"
    )
    assert sum(wave["n"] for wave in waves) == len(seen[True][1])


def test_a_failing_cunqa_job_closes_the_trace_with_error(monkeypatch, tmp_path):
    FakeCunqa(run_error=RuntimeError("vQPU crashed")).install(monkeypatch)
    with pytest.raises(RuntimeError, match="vQPU crashed"):
        polypus.run_quantum_circuit(
            _uniform3(),
            shots=64,
            infrastructure="cunqa",
            backend="aer",
            n_qpus=2,
            seed=1,
            trace=tmp_path / "failed.jsonl",
        )
    trace = read_trace(tmp_path / "failed.jsonl")
    check_trace(trace)
    assert trace.end["status"] == "error"
    assert trace.events("batch")[-1]["ok"] is False
    assert trace.events("wave")[-1]["ok"] is False
