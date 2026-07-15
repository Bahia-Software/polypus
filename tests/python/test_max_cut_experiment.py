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


# ─────────────────────────────────────────────────────────────────────────────
# CSV schema v2: training hyper-parameters persist, with "not applicable"
# distinguished from a real 0. These are pure (no backend) — they exercise the
# config→record→CSV→record path directly.
# ─────────────────────────────────────────────────────────────────────────────


def _record(cfg: mc.RunConfig) -> mc.RunRecord:
    """A RunRecord for ``cfg`` without running any backend: sizing comes from
    build_problem (pure) and the hyper-parameters from the config, so the
    schema/serialisation can be tested without training."""
    prob = mc.build_problem(cfg)
    return mc.RunRecord(
        schema_version=mc.CSV_SCHEMA_VERSION,
        timestamp="2026-07-15T00:00:00Z",
        git_commit="deadbee",
        polypus_version="0.6.0",
        n_qubits=prob.n_qubits,
        method=cfg.method,
        optimizer=mc._OPTIMIZER_LABEL[cfg.optimizer],
        infrastructure=cfg.infrastructure,
        backend="aer",
        seed=cfg.seed,
        base_seed=cfg.base_seed,
        repeat_index=cfg.repeat_index,
        graph_seed=cfg.graph_seed,
        graph_prob=cfg.graph_prob,
        n_edges=len(prob.edges),
        layers=prob.layers,
        dimensions=prob.dimensions,
        population_size=prob.population_size,
        max_generations=prob.max_generations,
        layers_factor=cfg.layers_factor,
        population_factor=cfg.population_factor,
        generations_factor=cfg.generations_factor,
        **cfg.applicable_hyperparams(),
        n_shots=cfg.n_shots,
        n_qpus=cfg.n_qpus,
        nodes=cfg.nodes,
        cores_per_qpu=cfg.cores_per_qpu,
        best_solution_bruteforce=prob.best_solution,
        best_cut=prob.best_solution,
        best_bitstring="0" * prob.n_qubits,
        approximation_ratio=1.0,
        best_ratio=1.0,
        best_fitness=float(prob.best_solution),
        iterations_run=1,
        converged=True,
        time_seconds=0.0,
    )


class TestHyperparamSchema:
    def test_applicable_hyperparams_by_optimizer(self):
        # QNG: tolerance N/A, qng_* set, scipy_workers N/A.
        qng = _cfg("polypus_local_qng", seed=1).applicable_hyperparams()
        assert qng["tolerance"] is None
        assert qng["scipy_workers"] is None
        assert qng["qng_learning_rate"] == 0.1
        assert qng["qng_step"] == 0.1
        assert qng["qng_tikhonov"] == 0.05

        # DE/PSO: tolerance set, everything method-specific else N/A.
        for method in ("polypus_local", "polypus_local_pso"):
            hp = _cfg(method, seed=1).applicable_hyperparams()
            assert hp["tolerance"] == 1e-5
            assert hp["scipy_workers"] is None
            assert hp["qng_learning_rate"] is None
            assert hp["qng_step"] is None
            assert hp["qng_tikhonov"] is None

        # scipy: tolerance + scipy_workers set, qng_* N/A.
        sp = _cfg("scipy", seed=1, scipy_workers=3).applicable_hyperparams()
        assert sp["tolerance"] == 1e-5
        assert sp["scipy_workers"] == 3
        assert sp["qng_learning_rate"] is None

    def test_schema_version_is_two(self):
        assert mc.CSV_SCHEMA_VERSION == 2
        # New columns are part of the contract.
        for col in ("layers_factor", "population_factor", "generations_factor",
                    "tolerance", "scipy_workers", "qng_learning_rate",
                    "qng_step", "qng_tikhonov"):
            assert col in mc.CSV_FIELDS

    def test_csv_roundtrip_preserves_none_and_values(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        qng = _record(_cfg("polypus_local_qng", seed=1))
        scipy = _record(_cfg("scipy", seed=1, scipy_workers=4))
        mc.append_record(csv_path, qng)
        mc.append_record(csv_path, scipy)

        rows = {r["method"]: r for r in mc.read_records(csv_path)}
        assert rows["polypus_local_qng"]["schema_version"] == 2

        # QNG row: tolerance/scipy_workers are "not applicable" (None), not 0.
        assert rows["polypus_local_qng"]["tolerance"] is None
        assert rows["polypus_local_qng"]["scipy_workers"] is None
        assert rows["polypus_local_qng"]["qng_learning_rate"] == 0.1
        assert rows["polypus_local_qng"]["qng_tikhonov"] == 0.05

        # scipy row: tolerance + scipy_workers persisted; qng_* not applicable.
        assert rows["scipy"]["tolerance"] == 1e-5
        assert rows["scipy"]["scipy_workers"] == 4
        assert rows["scipy"]["qng_step"] is None
        # Scaling factors persist for every method.
        assert rows["scipy"]["population_factor"] == 2  # from _cfg override

    def test_old_schema_version_rejected(self, tmp_path):
        # A v2-shaped header but a stale schema_version cell must fail loudly —
        # the reader never silently upgrades an old row.
        csv_path = tmp_path / "old.csv"
        rec = _record(_cfg("polypus_local", seed=1))
        row = rec.to_row()
        row["schema_version"] = 1
        import csv as _csv
        with open(csv_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=mc.CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        with pytest.raises(mc.ExperimentError, match="schema_version"):
            mc.read_records(str(csv_path))
