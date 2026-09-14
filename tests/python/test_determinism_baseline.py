"""
Determinism baseline vectors — the pre-refactor gate for the core-orchestration
refactor (behaviour-preserving reorg of the execution/evaluation seam).

Unlike ``test_seed_reproducibility.py`` (which only asserts *self-consistency*:
same seed run twice ⇒ same result within one build), this module pins the
*actual* seed→result mapping captured on ``main`` and fails if any later change
moves it. That is what makes it a regression gate for a refactor whose contract
is "same seed ⇒ byte-identical result, before and after".

Two tiers, split by whether a vector is *provably* stable across platforms:

* **Portable tier** (always runs, incl. CI on any IEEE-754 x86_64/glibc host):
  vectors whose every floating-point operation is IEEE-754 basic (``+ - * /``,
  ``sqrt``) and therefore correctly-rounded and bit-identical everywhere, with an
  integer-based RNG on top. Two families qualify:
    - ``run_quantum_circuit`` on Clifford circuits (H/CX): the amplitudes come
      from ``1/sqrt(2)`` (a correctly-rounded ``sqrt``) and products thereof, and
      the shot sampler draws from a seeded integer RNG. No transcendental math.
    - ``qml.train`` with a mocked backend: the counts are fixed integers, so the
      trajectory is driven solely by the optimiser RNG (integer draws) and IEEE
      basic arithmetic on the parameter vectors. No backend FP at all.
  These are asserted **byte-exact** (via ``float.hex()``).

* **Same-build strict tier** (skipped unless ``POLYPUS_DETERMINISM_STRICT=1``):
  native ``train`` (DE/PSO/QNG) uses ``ry(θ)`` whose probabilities go through
  ``sin``/``cos``; those are *not* guaranteed correctly-rounded across libm
  implementations, and a 1-ULP flip early in a run can send a greedy DE/PSO
  trajectory to a different (equally valid) optimum. Asserting these byte-exact
  across platforms would be flaky, so they gate **same-build** only: run this
  tier on the machine that produced the fixture, before and after each refactor
  phase, to enforce the plan's byte-identical requirement. It is byte-exact when
  it runs.

Regenerate the fixture (only on a deliberate, reviewed behaviour change) with::

    POLYPUS_DETERMINISM_STRICT=1 python tests/python/test_determinism_baseline.py

which rewrites ``determinism_vectors.json`` next to this file.
"""

import json
import os
from pathlib import Path

import pytest

_FIXTURE_PATH = Path(__file__).with_name("determinism_vectors.json")
_STRICT = os.environ.get("POLYPUS_DETERMINISM_STRICT") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Case definitions — the single source of truth for both regeneration and check.
# Each helper returns a dict already in the fixture's shape (floats as exact
# IEEE-754 hex strings), so a plain ``==`` against the fixture is byte-precise.
# ─────────────────────────────────────────────────────────────────────────────


def _hx(x):
    return float(x).hex()


def _all_ones(bitstring):
    return float(all(b == "1" for b in bitstring))


def _sum_bits(bitstring):
    return sum(int(c) for c in bitstring) / len(bitstring)


def capture_run_quantum_circuit():
    """Portable: Clifford circuits, native backend, fixed seed → counts."""
    import polypus

    out = {}

    def rqc(name, circuit, shots, seed):
        r = polypus.run_quantum_circuit(
            circuit, shots=shots, infrastructure="local", backend="polypus", seed=seed
        )
        out[name] = {"seed": r.seed, "counts": r.counts}

    rqc(
        "native_uniform3_seed42",
        polypus.Circuit(3).h(0).h(1).h(2).measure_all(),
        2000,
        42,
    )
    rqc("native_bell_seed7", polypus.Circuit(2).h(0).cx(0, 1).measure_all(), 1000, 7)
    rqc(
        "native_ghz4_seed123",
        polypus.Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all(),
        4096,
        123,
    )
    return out


def _install_mock_backend(monkeypatch=None):
    """Fixed integer counts regardless of circuit → a deterministic oracle,
    isolating the optimiser RNG that qml.train's seed controls."""
    import polypus_python

    def fake_run_qcs(infrastructure, **kwargs):
        return [{"1": kwargs["shots"]} for _ in kwargs["qcs"]]

    if monkeypatch is not None:
        monkeypatch.setattr(polypus_python, "run_qcs", fake_run_qcs)
    else:
        polypus_python.run_qcs = fake_run_qcs


def capture_qml_train():
    """Portable: mocked backend, so only the optimiser RNG + IEEE arithmetic on
    parameter vectors decide the outcome (no backend transcendental math)."""
    import numpy as np
    import polypus
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    feature_map = zz_feature_map(feature_dimension=2, reps=1)
    ansatz = real_amplitudes(num_qubits=2, reps=1)
    x_train = np.zeros((2, 2))
    r = polypus.qml.train(
        feature_map,
        ansatz,
        x_train,
        polypus.DE(generations=3, population_size=6, tolerance=1e-12),
        shots=64,
        n_qpus=1,
        dimensions=len(ansatz.parameters),
        expectation_function=_sum_bits,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="qml_seed",
        seed=123,
    )
    return {
        "de_mockbackend_seed123": {
            "seed": r.seed,
            "best_params_hex": [_hx(v) for v in r.best_params],
            "best_params_repr": [repr(v) for v in r.best_params],
            "best_fitness_hex": _hx(r.best_fitness),
            "best_fitness_repr": repr(r.best_fitness),
            "iterations_run": r.iterations_run,
            "converged": r.converged,
        }
    }


def _capture_one_train(name, method, circuit, dims, seed):
    import polypus

    r = polypus.train(
        circuit,
        method,
        shots=256,
        n_qpus=1,
        dimensions=dims,
        expectation_function=_all_ones,
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id=name,
        backend="polypus",
        seed=seed,
    )
    return {
        "seed": r.seed,
        "best_params_hex": [_hx(v) for v in r.best_params],
        "best_params_repr": [repr(v) for v in r.best_params],
        "best_fitness_hex": _hx(r.best_fitness),
        "best_fitness_repr": repr(r.best_fitness),
        "iterations_run": r.iterations_run,
        "converged": r.converged,
        "fitness_history_hex": [_hx(v) for v in r.fitness_history],
    }


_PI = 3.141592653589793


def capture_native_train():
    """Same-build strict: native backend, ry(θ) → sin/cos in the probabilities."""
    import polypus

    ry1 = polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all()
    ry2 = (
        polypus.Circuit(2)
        .ry(0, polypus.Param(0))
        .ry(1, polypus.Param(1))
        .cx(0, 1)
        .measure_all()
    )
    out = {}
    out["de_native_1q_seed2024"] = _capture_one_train(
        "de_native_1q_seed2024",
        polypus.DE(generations=8, population_size=8, tolerance=1e-12, seed=2024),
        ry1,
        1,
        2024,
    )
    out["pso_native_1q_seed2024"] = _capture_one_train(
        "pso_native_1q_seed2024",
        polypus.PSO(generations=8, population_size=8, bounds=(0.0, _PI), seed=2024),
        ry1,
        1,
        2024,
    )
    out["qng_native_1q_seed2024"] = _capture_one_train(
        "qng_native_1q_seed2024",
        polypus.QNG(
            variance_function=lambda theta, a: 0.5,
            max_iters=5,
            bounds=(0.0, _PI),
            learning_rate=0.1,
            seed=2024,
        ),
        ry1,
        1,
        2024,
    )
    out["de_native_2q_seed7"] = _capture_one_train(
        "de_native_2q_seed7",
        polypus.DE(generations=6, population_size=10, tolerance=1e-12, seed=7),
        ry2,
        2,
        7,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fixture loading + tests
# ─────────────────────────────────────────────────────────────────────────────


def _load_fixture():
    with open(_FIXTURE_PATH) as fh:
        return json.load(fh)


@pytest.mark.integration
class TestPortableExact:
    """Byte-exact on any IEEE-754 host — safe to gate cross-platform CI."""

    def test_run_quantum_circuit_counts(self):
        fixture = _load_fixture()["run_quantum_circuit"]
        assert capture_run_quantum_circuit() == fixture

    @pytest.mark.vqc
    def test_qml_train_mocked_backend(self, monkeypatch):
        _install_mock_backend(monkeypatch)
        fixture = _load_fixture()["qml_train"]
        assert capture_qml_train() == fixture


@pytest.mark.skipif(
    not _STRICT,
    reason="native train goldens are byte-exact only on the fixture's build "
    "(ry uses sin/cos; set POLYPUS_DETERMINISM_STRICT=1 to run the same-build gate)",
)
@pytest.mark.integration
@pytest.mark.vqc
class TestSameBuildStrict:
    """Byte-exact native train/PSO/QNG trajectories — the per-phase gate."""

    def test_native_train_trajectories(self):
        fixture = _load_fixture()["train"]
        assert capture_native_train() == fixture


def _regenerate():
    """Rewrite the fixture from the current build. Requires strict mode so it is
    never triggered by accident from a plain pytest run."""
    if not _STRICT:
        raise SystemExit(
            "refusing to regenerate without POLYPUS_DETERMINISM_STRICT=1 "
            "(this overwrites the committed determinism fixture)"
        )
    _install_mock_backend(None)
    data = {
        "meta": {
            "note": "Pre-refactor determinism baseline. Regenerate only on a "
            "deliberate, reviewed behaviour change (see module docstring).",
            "backend": "polypus (native) for train/run_quantum_circuit; "
            "mocked run_qcs for qml_train",
        },
        "run_quantum_circuit": capture_run_quantum_circuit(),
        "qml_train": capture_qml_train(),
        "train": capture_native_train(),
    }
    with open(_FIXTURE_PATH, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {_FIXTURE_PATH}")


if __name__ == "__main__":
    _regenerate()
