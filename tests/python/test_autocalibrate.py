"""
Lazy auto-calibration of the gate-parallel threshold (issues #176, #351).

A ``pip install polypus-quantum`` of a published wheel has no post-install hook,
and the CI machine that *builds* the wheel is not the one that *runs* it, so the
threshold is measured on the runtime hardware: ``calibrate_parallel_threshold(
force=False)`` runs once per process, reusing a valid on-disk cache in ~0s and
only measuring on a cold/changed cache (<1s). Since #351 it runs on the first
native statevector run of a process, not at ``import polypus``.

Contract exercised here:
  * ``import polypus`` measures nothing and writes nothing;
  * the first native run (``statevector``, ``run_quantum_circuit`` /
    ``train`` with ``backend="polypus"``) calibrates and writes the cache;
  * an Aer run does not calibrate;
  * a later process reuses the cache with no measurement;
  * ``POLYPUS_NO_AUTOCALIBRATE=1`` skips calibration entirely;
  * a secondary rank of a SLURM/OpenMPI job (``SLURM_PROCID`` /
    ``OMPI_COMM_WORLD_RANK`` set and not ``0``) skips calibration (issue #214),
    while rank 0 — or no launcher at all — still calibrates;
  * a read-only cache dir does not break the run.

Calibration happens at most once per process, so every case runs in its own
child interpreter with a private ``XDG_CACHE_HOME``. The opt-out and rank policy
itself is unit-tested in Rust (``crates/polypus/src/bindings/calibration.rs``).
"""

import json
import os
import subprocess
import sys

import pytest

polypus = pytest.importorskip("polypus")

_CACHE_REL = ("polypus", "parallel_threshold.json")

# Launcher rank variables the implicit calibration consults (issue #214). Cleared
# in every test that expects calibration to run, so the suite behaves the same
# when it is itself launched on a non-zero rank under srun/mpirun.
_RANK_VARS = ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK")

# Prints whether the cache exists after `import polypus` and after the first
# native run; `run` is the code of that run.
_IMPORT_THEN_RUN = """
import os
cache = os.path.join(os.environ["XDG_CACHE_HOME"], "polypus", "parallel_threshold.json")
import polypus
print("AFTER_IMPORT", os.path.isfile(cache), os.listdir(os.environ["XDG_CACHE_HOME"]))
{run}
print("AFTER_RUN", os.path.isfile(cache))
"""

_STATEVECTOR = "polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1))"


def _run_child(
    code: str, cache_home, *, optout=False, rank_env=None
) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `cache_home` as XDG_CACHE_HOME.

    Auto-calibration is left *enabled* by default here (that is the behaviour
    under test); pass ``optout=True`` to set ``POLYPUS_NO_AUTOCALIBRATE=1``.
    Launcher rank variables are cleared unless given in ``rank_env``.
    """
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(cache_home)
    if optout:
        env["POLYPUS_NO_AUTOCALIBRATE"] = "1"
    else:
        env.pop("POLYPUS_NO_AUTOCALIBRATE", None)
    for var in _RANK_VARS:
        env.pop(var, None)
    env.update(rank_env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _import_then_run(run: str, cache_home, **kwargs) -> str:
    r = _run_child(_IMPORT_THEN_RUN.format(run=run), cache_home, **kwargs)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_import_has_no_side_effects(tmp_path):
    # Empty cache home, auto-calibration on: import alone creates nothing.
    r = _run_child("import polypus\nprint('IMPORTED')", tmp_path)
    assert r.returncode == 0, r.stderr
    assert "IMPORTED" in r.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "run",
    [
        _STATEVECTOR,
        "polypus.run_quantum_circuit(polypus.Circuit(2).h(0).measure_all(), "
        "shots=16, infrastructure='local', backend='polypus', seed=1)",
        "polypus.train(polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(), "
        "polypus.DE(generations=1, population_size=4), shots=8, n_qpus=1, "
        "dimensions=1, expectation_function=lambda b: float(b == '1'), "
        "infrastructure='local', nodes=1, cores_per_qpu=1, id='autocal', "
        "backend='polypus', seed=1)",
    ],
    ids=["statevector", "run_quantum_circuit", "train"],
)
def test_first_native_run_calibrates_and_writes_cache(tmp_path, run):
    out = _import_then_run(run, tmp_path)
    assert "AFTER_IMPORT False []" in out, out
    assert "AFTER_RUN True" in out, out
    assert (tmp_path / _CACHE_REL[0] / _CACHE_REL[1]).is_file()


def test_aer_run_does_not_calibrate(tmp_path):
    pytest.importorskip("qiskit_aer")
    run = (
        "polypus.run_quantum_circuit(polypus.Circuit(2).h(0).measure_all(), "
        "shots=16, infrastructure='local', backend='aer', seed=1)"
    )
    out = _import_then_run(run, tmp_path)
    assert "AFTER_RUN False" in out, out


def test_later_process_reuses_cache_without_measuring(tmp_path):
    out = _import_then_run(_STATEVECTOR, tmp_path)
    assert "AFTER_RUN True" in out, out

    # A valid cache is reused as-is and never rewritten, so the mtime is
    # unchanged across the second process's first run.
    code = f"""
import os, polypus
cache = os.path.join(os.environ["XDG_CACHE_HOME"], "polypus", "parallel_threshold.json")
before = os.path.getmtime(cache)
{_STATEVECTOR}
print("UNCHANGED", before == os.path.getmtime(cache))
"""
    r = _run_child(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "UNCHANGED True" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"


def test_optout_env_skips_autocalibration(tmp_path):
    out = _import_then_run(_STATEVECTOR, tmp_path, optout=True)
    assert "AFTER_RUN False" in out, out
    assert not (tmp_path / _CACHE_REL[0]).exists()


def test_optout_set_after_import_is_honoured(tmp_path):
    # The policy is read on the first native run, not at import.
    run = f"os.environ['POLYPUS_NO_AUTOCALIBRATE'] = '1'\n{_STATEVECTOR}"
    out = _import_then_run(run, tmp_path)
    assert "AFTER_RUN False" in out, out


def test_first_run_replaces_a_thread_count_only_cache(tmp_path):
    # Upgrade path (issue #214): a ``schema:2`` cache keyed by bare thread count
    # may hold another node's measurement on a shared $HOME, so it is treated as
    # absent — the first run recalibrates and rewrites it as a ``schema:3`` map
    # keyed by this machine's hardware.
    cache_file = tmp_path / _CACHE_REL[0] / _CACHE_REL[1]
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('{"schema":2,"entries":{"999999":14}}')
    r = _run_child(f"import polypus\n{_STATEVECTOR}\nprint('RAN')", tmp_path)
    assert r.returncode == 0, r.stderr
    assert "RAN" in r.stdout
    data = json.loads(cache_file.read_text())
    assert data["schema"] == 3
    assert "999999" not in data["entries"], "the untrusted schema-2 entry is dropped"
    assert len(data["entries"]) == 1
    # No torn-write or temp debris next to the cache: only it and its lock file.
    assert sorted(p.name for p in cache_file.parent.iterdir()) == [
        "parallel_threshold.json",
        "parallel_threshold.json.lock",
    ]


@pytest.mark.parametrize("rank_var", _RANK_VARS)
def test_secondary_rank_skips_autocalibration(tmp_path, rank_var):
    out = _import_then_run(_STATEVECTOR, tmp_path, rank_env={rank_var: "1"})
    assert "AFTER_RUN False" in out, out
    assert not (tmp_path / _CACHE_REL[0]).exists()


def test_rank_zero_still_calibrates(tmp_path):
    out = _import_then_run(
        _STATEVECTOR,
        tmp_path,
        rank_env={"SLURM_PROCID": "0", "OMPI_COMM_WORLD_RANK": "0"},
    )
    assert "AFTER_RUN True" in out, out


def test_explicit_calibration_still_runs_on_a_secondary_rank(tmp_path):
    # The rank skip is for the *implicit* path only: an explicit call is the
    # user's deliberate choice (install.sh, a per-node srun) and must still
    # measure and persist on any rank.
    code = """
import polypus
r = polypus.calibrate_parallel_threshold()
print("WRITTEN", r["cache_written"], "REUSED", r["reused_cache"])
"""
    r = _run_child(code, tmp_path, rank_env={"SLURM_PROCID": "3"})
    assert r.returncode == 0, r.stderr
    assert "WRITTEN True REUSED False" in r.stdout, (
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
    assert (tmp_path / _CACHE_REL[0] / _CACHE_REL[1]).is_file()


def test_readonly_cache_dir_does_not_break_the_first_run(tmp_path):
    # The cache dir cannot be created (its parent is a regular file): the
    # calibration cannot persist, and the first run still succeeds.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(blocker)
    env.pop("HOME", None)  # no fallback cache dir either
    env.pop("POLYPUS_NO_AUTOCALIBRATE", None)
    for var in _RANK_VARS:
        env.pop(var, None)
    r = subprocess.run(
        [sys.executable, "-c", f"import polypus\n{_STATEVECTOR}\nprint('RAN')"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "RAN" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr!r}"
