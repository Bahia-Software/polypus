"""
Python-visible contract for ``polypus.calibrate_parallel_threshold`` (issue
#127): the manual entry point that measures and caches the machine-specific
qubit count at which the native simulator's gate kernels switch to the rayon
parallel path.

These tests exercise the *mechanism* deterministically — the returned schema,
the cache round-trip, ``force`` semantics — and never assert an exact calibrated
threshold, which is timing-derived and would be flaky in CI. The threshold
selection, freshness and fallback logic are unit-tested in Rust
(``crates/polypus-sim/src/calibration.rs``).

Each test redirects the cache to a temporary ``XDG_CACHE_HOME`` so it never
reads or writes the real user cache.
"""

import pytest

polypus = pytest.importorskip("polypus")


def test_returns_the_documented_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = polypus.calibrate_parallel_threshold(force=True)

    assert isinstance(result, dict)
    assert set(result) == {
        "threshold",
        "num_threads",
        "duration_secs",
        "reused_cache",
        "cache_path",
        "cache_written",
    }
    assert isinstance(result["threshold"], int) and result["threshold"] >= 1
    assert isinstance(result["num_threads"], int) and result["num_threads"] >= 1
    assert isinstance(result["duration_secs"], float) and result["duration_secs"] >= 0.0
    assert isinstance(result["reused_cache"], bool)
    assert isinstance(result["cache_written"], bool)


def test_force_measures_and_persists_then_reuse_hits_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    # force=True measures (does not reuse) and writes the cache to the temp dir.
    fresh = polypus.calibrate_parallel_threshold(force=True)
    assert fresh["reused_cache"] is False
    assert fresh["cache_written"] is True
    cache_file = tmp_path / "polypus" / "parallel_threshold.json"
    assert cache_file.is_file(), "force=True must persist the cache"
    assert fresh["cache_path"] == str(cache_file)

    # force=False now finds a cache valid for this hardware and reuses it as-is:
    # no measurement (duration 0), same threshold.
    reused = polypus.calibrate_parallel_threshold(force=False)
    assert reused["reused_cache"] is True
    assert reused["cache_written"] is False
    assert reused["duration_secs"] == 0.0
    assert reused["threshold"] == fresh["threshold"]
    assert reused["num_threads"] == fresh["num_threads"]


def test_default_force_is_false(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Seed the cache, then call with no argument: the default must be force=False,
    # so the seeded cache is reused rather than re-measured.
    polypus.calibrate_parallel_threshold(force=True)
    reused = polypus.calibrate_parallel_threshold()
    assert reused["reused_cache"] is True


def test_unwritable_cache_dir_is_not_fatal(monkeypatch, tmp_path):
    # Point the cache dir resolution at a path whose parent is a regular file, so
    # the directory cannot be created: calibration must still return a threshold,
    # just with cache_written=False, never raising (the read-only container case).
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    monkeypatch.delenv("HOME", raising=False)

    result = polypus.calibrate_parallel_threshold(force=True)
    assert result["threshold"] >= 1
    assert result["cache_written"] is False


def test_calibration_does_not_break_a_subsequent_simulation(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    polypus.calibrate_parallel_threshold(force=True)

    # A normal statevector run still works after calibrating.
    qc = polypus.Circuit(2).h(0).cx(0, 1)
    amps = polypus.statevector(qc)
    assert len(amps) == 4
    assert abs(abs(amps[0]) ** 2 - 0.5) < 1e-9
    assert abs(abs(amps[3]) ** 2 - 0.5) < 1e-9
