"""End-to-end tests for the MaxCut–QAOA experiment run stage (``examples/max_cut``).

These exercise ``maxcut_lib.run_single`` — the single-run engine behind
``run_maxcut.py`` — on small, fast problems, and assert the invariants the
experiment relies on:

* **Exact reproducibility (contract C-7):** two runs with the same seed produce
  the identical ``approximation_ratio`` *and* ``best_bitstring``. This holds only
  because every Aer sampling site (training oracle, QNG variance callback, final
  evaluation, and scipy's baseline sampling) is seeded from the run seed.
* **Distinct seeds diverge:** different seeds give different results.
* **Shot conservation (contract C-3):** ``sum(counts.values()) == n_shots``.
* **Valid approximation ratio:** finite, not negative, not greater than 1.
* **Loud input validation:** a degenerate instance aborts (no silent row).

The experiment code lives under ``examples/`` (not an installed package), so we
put its directory on ``sys.path`` and import the core module directly — the same
mechanism the CLI scripts use when run.
"""

import pathlib
import sys

import pytest

_MAXCUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "max_cut"
if str(_MAXCUT_DIR) not in sys.path:
    sys.path.insert(0, str(_MAXCUT_DIR))

import maxcut_lib as mc  # noqa: E402  (import after sys.path tweak, by design)

# The three local Polypus optimizers plus the classical scipy baseline. scipy is
# kept at scipy_workers=1 (no multiprocessing pool) so the test never forks.
METHODS = ["polypus_local", "polypus_local_pso", "polypus_local_qng", "scipy"]

_RATIO_EPS = 1e-9


def _cfg(method: str, seed: int, **overrides) -> mc.RunConfig:
    """A deliberately small/fast configuration (4 qubits, few generations)."""
    params = dict(
        n_qubits=4,
        method=method,
        seed=seed,
        n_shots=500,
        generations_factor=1,   # max_generations = n_qubits * 1
        population_factor=2,    # population = dimensions * 2
    )
    params.update(overrides)
    return mc.RunConfig(**params)


@pytest.mark.integration
@pytest.mark.vqc
class TestRunSingleReproducibility:
    @pytest.mark.parametrize("method", METHODS)
    def test_same_seed_reproduces_ratio_and_bitstring(self, method):
        r1 = mc.run_single(_cfg(method, seed=1234))
        r2 = mc.run_single(_cfg(method, seed=1234))
        # Byte-exact: no tolerance. Same seed ⇒ identical outcome.
        assert r1.approximation_ratio == r2.approximation_ratio
        assert r1.best_bitstring == r2.best_bitstring
        assert r1.best_fitness == r2.best_fitness

    @pytest.mark.parametrize("method", METHODS)
    def test_distinct_seeds_diverge(self, method):
        r1 = mc.run_single(_cfg(method, seed=1))
        r2 = mc.run_single(_cfg(method, seed=2))
        # The optimal bitstring may coincide (both may solve the tiny instance),
        # but the seeded optimizer trajectory + sampling must not be identical.
        assert (r1.approximation_ratio, r1.best_fitness) != (r2.approximation_ratio, r2.best_fitness)


@pytest.mark.integration
@pytest.mark.vqc
class TestRunSingleInvariants:
    @pytest.mark.parametrize("method", METHODS)
    def test_shot_conservation(self, method):
        artifacts: dict = {}
        record = mc.run_single(_cfg(method, seed=7), artifacts=artifacts)
        assert sum(artifacts["counts"].values()) == record.n_shots == 500

    @pytest.mark.parametrize("method", METHODS)
    def test_approximation_ratio_in_valid_range(self, method):
        import math

        record = mc.run_single(_cfg(method, seed=7))
        assert not math.isnan(record.approximation_ratio)
        assert not math.isinf(record.approximation_ratio)
        assert record.approximation_ratio >= -_RATIO_EPS
        assert record.approximation_ratio <= 1.0 + _RATIO_EPS

    @pytest.mark.parametrize("method", METHODS)
    def test_best_bitstring_width(self, method):
        record = mc.run_single(_cfg(method, seed=7))
        assert len(record.best_bitstring) == record.n_qubits == 4


class TestRunSingleValidation:
    """Input validation is loud and requires no backend (pure Python)."""

    def test_too_few_qubits_rejected(self):
        with pytest.raises(mc.InputValidationError, match=r">= 2"):
            mc.build_problem(_cfg("polypus_local", seed=1, n_qubits=1))

    def test_empty_graph_rejected(self):
        # graph_prob=0 ⇒ no edges ⇒ MaxCut undefined ⇒ abort (no silent row).
        with pytest.raises(mc.InputValidationError, match="no edges"):
            mc.build_problem(_cfg("polypus_local", seed=1, graph_prob=0.0))

    def test_unknown_method_rejected(self):
        # run_single guards the method before touching any backend.
        cfg = _cfg("polypus_local", seed=1)
        cfg.method = "not_a_method"
        with pytest.raises(mc.InputValidationError, match="unknown method"):
            mc.run_single(cfg)
