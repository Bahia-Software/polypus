"""What a seed means across the calls of one run (contract C-7, #411).

Every backend gives each call of a run its own sampling stream: the native
backend seeds each circuit from the base seed and its index in the run, and
Aer, CUNQA and the subprocess bridge receive, on call ``n``, the base seed for
``n = 0`` and a seed derived from the base seed and ``n`` after that. So two
generations of a ``train`` never share shot noise, and the same seed replays
the same sequence. The native and bridge sides are pinned in Rust
(``native.rs``, ``polypus-subprocess-backend/tests/seed_streams.rs``); here the
seeds Aer and CUNQA receive are read off the ``run_qcs`` seam.
"""

import pytest

pytest.importorskip("qiskit_aer")

import polypus  # noqa: E402
from fake_cunqa import FakeCunqa  # noqa: E402

_MASK = 0x7FFFFFFFFFFFFFFF


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(infrastructure, seed, **kwargs):
    return polypus.train(
        polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
        polypus.DE(generations=5, population_size=8, patience=100),
        shots=64,
        dimensions=1,
        expectation_function=_all_ones,
        infrastructure=infrastructure,
        nodes=1,
        cores_per_qpu=1,
        id="seed_streams",
        backend="aer",
        seed=seed,
        **kwargs,
    )


def _spy_seeds(monkeypatch):
    """Record the ``seed`` of every ``run_qcs`` call and run it for real."""
    import polypus_python

    real = polypus_python.run_qcs
    seeds = []

    def spy(infrastructure, **kwargs):
        seeds.append(kwargs.get("seed"))
        return real(infrastructure, **kwargs)

    monkeypatch.setattr(polypus_python, "run_qcs", spy)
    return seeds


@pytest.mark.integration
@pytest.mark.vqc
class TestAer:
    def test_every_call_of_a_run_gets_its_own_seed(self, monkeypatch):
        seeds = _spy_seeds(monkeypatch)
        result = _train("local", 3, n_qpus=1)
        # The initial population and one batch per generation.
        assert len(seeds) == result.iterations_run + 1 == 6
        assert seeds[0] == 3
        assert len(set(seeds)) == len(seeds)

    def test_the_seed_replays_the_sequence(self, monkeypatch):
        seeds = _spy_seeds(monkeypatch)
        first = _train("local", 3, n_qpus=1)
        second = _train("local", 3, n_qpus=1)
        assert seeds[:6] == seeds[6:]
        assert first.fitness_history == second.fitness_history
        assert first.best_params == second.best_params

    def test_a_single_call_receives_the_base_seed(self, monkeypatch):
        # So `run_quantum_circuit` counts match Aer's own `seed_simulator`.
        seeds = _spy_seeds(monkeypatch)
        polypus.run_quantum_circuit(
            polypus.Circuit(1).h(0).measure_all(),
            shots=16,
            infrastructure="local",
            backend="aer",
            seed=5,
        )
        assert seeds == [5]


def _call_seeds(fake):
    """The seed of each QPU, grouped by ``run_qcs`` call: a call runs one job
    per circuit, then gathers them."""
    calls, current = [], []
    runs = iter(fake.runs)
    for name, _ in fake.calls:
        if name == "run":
            run = next(runs)
            current.append((run["qpu"].index, run["options"]["seed"]))
        elif name == "gather" and current:
            calls.append(current)
            current = []
    return calls


@pytest.mark.integration
@pytest.mark.vqc
class TestCunqa:
    def test_every_call_of_a_run_gets_its_own_seed(self, monkeypatch):
        fake = FakeCunqa().install(monkeypatch)
        _train("cunqa", 3, n_qpus=2)
        calls = _call_seeds(fake)
        # Population 8 on 2 QPUs: 4 calls per batch, 6 batches.
        assert len(calls) == 24
        bases = []
        for call in calls:
            (qpu0, base), *rest = call
            assert qpu0 == 0
            # QPU k of a call runs with the call's seed + k.
            assert all(seed == (base + qpu) & _MASK for qpu, seed in rest)
            bases.append(base)
        assert bases[0] == 3
        assert len(set(bases)) == len(bases)

    def test_the_seed_replays_the_sequence(self, monkeypatch):
        first = FakeCunqa().install(monkeypatch)
        _train("cunqa", 3, n_qpus=2)
        second = FakeCunqa().install(monkeypatch)
        _train("cunqa", 3, n_qpus=2)
        assert _call_seeds(first) == _call_seeds(second)
