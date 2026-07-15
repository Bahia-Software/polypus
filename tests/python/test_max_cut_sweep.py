"""Pure-Python tests for the MaxCut sweep orchestrator (``examples/max_cut``).

These exercise ``sweep_maxcut._rough_seconds`` — the per-run time estimate the
``--dry-run`` plan prints — without touching Aer or any backend: it is plain
arithmetic. They pin the two calibration properties that matter for the
estimate to stay honest (calibrated against the two accumulated real sweeps in
``examples/max_cut/output/``):

* it still ~doubles per extra qubit (statevector cost dominates), and
* shots are *sub-linear* — the earlier ``shots / 2000`` linear term overestimated
  10 000-shot sweeps several-fold; the real exponent is ≈0.13, small but not zero.

As with ``test_max_cut_experiment.py`` the experiment code lives under
``examples/`` (not an installed package), so its directory goes on ``sys.path``
and the module is imported directly — the same mechanism the CLI scripts use.
"""

import pathlib
import sys

import pytest

_MAXCUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "max_cut"
if str(_MAXCUT_DIR) not in sys.path:
    sys.path.insert(0, str(_MAXCUT_DIR))

import sweep_maxcut as sweep  # noqa: E402  (import after sys.path tweak, by design)

# One representative method per optimizer family the estimate knows about.
_METHODS = ["scipy", "polypus_local", "polypus_local_pso", "polypus_local_qng"]


class TestRoughSecondsCalibration:
    def test_doubles_per_qubit(self):
        # The per-qubit factor is the model's 2**(n-4) term; real sweeps measured
        # ≈1.9× per qubit, so 2× is intentionally kept.
        for method in _METHODS:
            for q in range(4, 10):
                lo = sweep._rough_seconds(method, q, 10_000)
                hi = sweep._rough_seconds(method, q + 1, 10_000)
                assert hi / lo == pytest.approx(2.0)

    def test_shots_are_sublinear(self):
        # ×10 shots must NOT give ×10 time (the old linear bug); with exponent
        # ≈0.13 it should be ≈10**0.13 ≈ 1.35 — comfortably between "no effect"
        # (1.0) and "linear" (10.0).
        for method in _METHODS:
            t_1k = sweep._rough_seconds(method, 4, 1_000)
            t_10k = sweep._rough_seconds(method, 4, 10_000)
            ratio = t_10k / t_1k
            assert 1.0 < ratio < 2.0           # sub-linear, but a real dependence
            assert ratio == pytest.approx(10.0 ** 0.13, rel=1e-6)

    def test_shots_dependence_is_not_flat(self):
        # The effect is small but must not be dropped to zero — more shots is
        # always at least a little slower.
        for method in _METHODS:
            assert sweep._rough_seconds(method, 5, 20_000) > sweep._rough_seconds(method, 5, 5_000)

    def test_anchor_matches_measured(self):
        # Anchor: 4 qubits / 10 000 shots should reproduce the per-method base
        # times read off the real sweeps.
        assert sweep._rough_seconds("polypus_local", 4, 10_000) == pytest.approx(1.15)
        assert sweep._rough_seconds("polypus_local_qng", 4, 10_000) == pytest.approx(1.35)
        assert sweep._rough_seconds("scipy", 4, 10_000) == pytest.approx(2.8)
