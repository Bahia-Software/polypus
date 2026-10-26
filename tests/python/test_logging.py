"""
Logger binding tests.

These exercise the default behaviors that must hold when Polypus is driven
from Python:

  (a) ``run_name`` only names the auto-generated default file, so combining it
      with an explicit ``file``/``console`` raises ``ValueError`` instead of
      silently dropping it;
  (b) `init_logger()` with no ``file``/``console`` persists to a **file**, not
      stdout (so log output isn't lost in notebooks / background jobs / queues);
  (c) a second ``init_logger`` call in the same process is a **no-op + warning**
      (Jupyter cell re-runs, repeated calls across tests in one interpreter);
  (d) the wheel's compile-time ceiling keeps ``debug`` records, so
      ``level="debug"`` shows optimizer progress without a rebuild.

The global logger can be installed only once per process, so everything is
verified in a single ordered test — the first `init_logger` in the process must
be the one below. Nothing else in the Python suite installs the logger; (d)
runs in a subprocess.
"""

import os
import subprocess
import sys
import warnings

import polypus
import pytest


def test_default_is_file_and_reinit_is_noop(tmp_path, monkeypatch):
    # Redirect the auto-generated default file into a throwaway dir.
    monkeypatch.setenv("POLYPUS_LOG_DIR", str(tmp_path))

    # (a) `run_name` only names the auto-generated default file, so it is mutually
    # exclusive with an explicit `file` / `console`: passing both must raise
    # rather than silently drop `run_name`. These reject the input *before* the
    # one-shot install below, so they neither install a logger nor create a file.
    with pytest.raises(ValueError):
        polypus.init_logger(run_name="x", file=str(tmp_path / "explicit.log"))
    with pytest.raises(ValueError):
        polypus.init_logger(run_name="x", console=True)

    # (b) No file/console => a real file under POLYPUS_LOG_DIR, never stdout.
    path = polypus.init_logger(run_name="pytest_default")
    assert path is not None, "default target must be a file, not stdout"
    assert path.endswith(".log")
    assert os.path.exists(path)
    assert os.path.dirname(path) == str(tmp_path), "POLYPUS_LOG_DIR must be honored"

    # (c) Second init in the same process: no-op + UserWarning, first sink kept.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = polypus.init_logger(run_name="pytest_default_again")
    assert result is None, "a re-init must be a no-op (return None)"
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "a re-init must emit a UserWarning"
    )

    # The no-op must not have created (leaked) a second log file.
    created = list(tmp_path.glob("*.log"))
    assert len(created) == 1, f"re-init leaked extra log files: {created}"


def test_debug_level_shows_optimizer_progress(tmp_path):
    log = tmp_path / "debug.log"
    script = (
        "import polypus\n"
        f"polypus.init_logger(level='debug', file={str(log)!r})\n"
        "polypus.train(polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(), "
        "polypus.DE(generations=30, population_size=4), shots=8, n_qpus=1, "
        "dimensions=1, expectation_function=lambda b: float(b == '1'), "
        "infrastructure='local', nodes=1, cores_per_qpu=1, id='debuglog', "
        "backend='polypus', seed=1)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=300)
    text = log.read_text()
    assert "DEBUG" in text and "Generation" in text, text
