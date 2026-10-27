"""
``polypus.DE(strategy=...)``: the classic DE/rand/1/bin (the default) and the
adaptive SHADE and L-SHADE (contract C-5).

The option tests need no backend. The batch tests run the real
``polypus.train`` loop on Aer with ``polypus_python.run_qcs`` replaced by a
fake that records every batch the optimizer sends and answers with fixed
counts. Seeded reproducibility per strategy is in
``test_seed_reproducibility.py`` (contract C-7).
"""

import pytest

STRATEGIES = ["rand1bin", "shade", "lshade"]
# The message of the ValueError an unknown name raises.
VALID_NAMES = '"rand1bin", "shade" or "lshade"'


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _train(method, *, seed, ident):
    import polypus

    qc = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
    return polypus.train(
        qc,
        method,
        shots=64,
        n_qpus=1,
        dimensions=1,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=ident,
        backend="aer",
        seed=seed,
    )


def _record_batches(monkeypatch):
    """Replace the local seam with a fake that records each batch of circuits
    (OpenQASM strings) and returns the same counts for every circuit, so every
    candidate gets the same fitness."""
    import polypus_python

    batches = []

    def fake_run_qcs(infrastructure, **kwargs):
        batches.append(list(kwargs["qcs"]))
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    return batches


class TestDeStrategyOption:
    def test_default_is_rand1bin(self):
        import polypus

        assert polypus.DE().strategy == "rand1bin"

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_every_strategy_round_trips(self, strategy):
        import polypus

        assert polypus.DE(strategy=strategy).strategy == strategy
        de = polypus.DE()
        de.strategy = strategy
        assert de.strategy == strategy

    def test_strategy_comes_after_seed_positionally(self):
        # Appended last, so positional calls written before it keep working.
        import polypus

        de = polypus.DE(10, 8, 0.05, 7, 3, "shade")
        assert (de.generations, de.population_size, de.patience, de.seed) == (
            10,
            8,
            7,
            3,
        )
        assert de.strategy == "shade"

    @pytest.mark.parametrize(
        "name", ["LSHADE", "Shade", " shade", "jade", "rand/1/bin", ""]
    )
    def test_unknown_name_raises_value_error(self, name):
        import polypus

        with pytest.raises(ValueError, match=VALID_NAMES):
            polypus.DE(strategy=name)

    def test_failed_assignment_keeps_the_previous_strategy(self):
        import polypus

        de = polypus.DE(strategy="shade")
        with pytest.raises(ValueError, match=VALID_NAMES):
            de.strategy = "jade"
        assert de.strategy == "shade"

    def test_non_string_raises_type_error(self):
        import polypus

        with pytest.raises(TypeError):
            polypus.DE(strategy=1)
        de = polypus.DE()
        with pytest.raises(TypeError):
            de.strategy = None
        assert de.strategy == "rand1bin"


class TestDeRates:
    """``mutation_factor`` (F) and ``crossover_rate`` (CR) of DE/rand/1/bin."""

    def test_default_is_none(self):
        import polypus

        de = polypus.DE()
        assert de.mutation_factor is None
        assert de.crossover_rate is None

    def test_rates_round_trip(self):
        import polypus

        de = polypus.DE(mutation_factor=0.5, crossover_rate=0.9)
        assert (de.mutation_factor, de.crossover_rate) == (0.5, 0.9)
        de.mutation_factor = 1.2
        de.crossover_rate = None
        assert (de.mutation_factor, de.crossover_rate) == (1.2, None)

    def test_rates_come_after_strategy_positionally(self):
        import polypus

        de = polypus.DE(10, 8, 0.05, 7, 3, "rand1bin", 0.6, 0.4)
        assert (de.mutation_factor, de.crossover_rate) == (0.6, 0.4)

    @pytest.mark.parametrize(
        "kwargs, name",
        [
            ({"mutation_factor": 0.0}, "mutation_factor"),
            ({"mutation_factor": 2.5}, "mutation_factor"),
            ({"mutation_factor": float("nan")}, "mutation_factor"),
            ({"crossover_rate": -0.1}, "crossover_rate"),
            ({"crossover_rate": float("inf")}, "crossover_rate"),
            ({"strategy": "shade", "mutation_factor": 0.5}, "mutation_factor"),
            ({"strategy": "lshade", "crossover_rate": 0.5}, "crossover_rate"),
        ],
    )
    def test_invalid_rates_raise_value_error(self, kwargs, name):
        import polypus

        with pytest.raises(ValueError, match=f"invalid {name}"):
            polypus.DE(**kwargs)

    def test_a_failed_assignment_keeps_the_previous_values(self):
        import polypus

        de = polypus.DE(mutation_factor=0.5)
        with pytest.raises(ValueError, match="invalid mutation_factor"):
            de.mutation_factor = 3.0
        with pytest.raises(ValueError, match="set it to None before switching"):
            de.strategy = "shade"
        assert (de.mutation_factor, de.strategy) == (0.5, "rand1bin")

    @pytest.mark.parametrize("name", ["mutation_factor", "crossover_rate"])
    def test_an_adaptive_strategy_rejects_a_rate_set_later(self, name):
        import polypus

        de = polypus.DE(strategy="shade")
        with pytest.raises(ValueError, match=f"invalid {name}"):
            setattr(de, name, 0.5)
        assert getattr(de, name) is None
        setattr(de, name, None)

    @pytest.mark.integration
    @pytest.mark.vqc
    def test_rates_reach_the_optimizer(self, monkeypatch):
        # The first generation's trials are built with F and CR, so they show
        # whether the rates reached the optimizer whatever the fitness.
        import polypus

        def trials(**rates):
            batches = _record_batches(monkeypatch)
            method = polypus.DE(generations=2, population_size=8, **rates)
            _train(method, seed=5, ident="de_rates")
            return batches[1]

        default = trials()
        assert trials(mutation_factor=0.8, crossover_rate=0.7) == default
        # One parameter: j_rand always takes it from the mutant, so CR has no
        # effect here (the Rust tests cover it).
        assert trials(mutation_factor=0.3) != default


@pytest.mark.integration
@pytest.mark.vqc
class TestDeStrategyBatches:
    """What each strategy sends to the backend: one batch per generation after
    the initial one, within the budget of ``population_size × (generations +
    1)`` evaluations (L-SHADE spends it in more, smaller batches), and nothing
    at all when the population is too small."""

    GENERATIONS = 10
    POPULATION = 20
    BUDGET = POPULATION * (GENERATIONS + 1)

    def _run(self, monkeypatch, strategy, seed=11):
        import polypus

        batches = _record_batches(monkeypatch)
        result = _train(
            polypus.DE(
                generations=self.GENERATIONS,
                population_size=self.POPULATION,
                # All fitness values are equal, so only a patience longer than
                # any run keeps the stagnation stop from ending it early.
                patience=10_000,
                strategy=strategy,
            ),
            seed=seed,
            ident="de_strategy_batches",
        )
        assert not result.converged
        assert len(batches) == result.iterations_run + 1
        # The C-5 trajectory: one entry per generation, ending at the best.
        assert len(result.fitness_history) == result.iterations_run
        assert result.fitness_history[-1] == result.best_fitness
        return result, batches

    @pytest.mark.parametrize("strategy", ["rand1bin", "shade"])
    def test_fixed_population_strategies_send_full_batches(self, monkeypatch, strategy):
        result, batches = self._run(monkeypatch, strategy)
        assert result.iterations_run == self.GENERATIONS
        assert [len(b) for b in batches] == [self.POPULATION] * (self.GENERATIONS + 1)

    def test_lshade_spends_the_budget_in_more_smaller_batches(self, monkeypatch):
        result, batches = self._run(monkeypatch, "lshade")
        sizes = [len(b) for b in batches]
        assert result.iterations_run > self.GENERATIONS
        assert sizes[:2] == [self.POPULATION, self.POPULATION]
        assert all(a >= b for a, b in zip(sizes, sizes[1:]))
        assert sizes[-1] == 4
        # Never past the budget, and short of it by less than the last batch.
        assert self.BUDGET - sizes[-1] < sum(sizes) <= self.BUDGET

    def test_strategy_reaches_the_optimizer(self, monkeypatch):
        runs = {s: self._run(monkeypatch, s)[1] for s in STRATEGIES}
        # Every strategy draws its initial population the same way, so a seed
        # gives the same first batch; the next ones differ by strategy.
        first = {tuple(batches[0]) for batches in runs.values()}
        second = {tuple(batches[1]) for batches in runs.values()}
        assert len(first) == 1
        assert len(second) == len(STRATEGIES)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_population_below_four_raises_before_any_batch(self, monkeypatch, strategy):
        import polypus

        batches = _record_batches(monkeypatch)
        with pytest.raises(ValueError):
            _train(
                polypus.DE(generations=2, population_size=3, strategy=strategy),
                seed=1,
                ident="de_strategy_bad_pop",
            )
        assert batches == []
