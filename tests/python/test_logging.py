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
      (Jupyter cell re-runs, repeated calls across tests in one interpreter).

The global logger can be installed only once per process, so everything is
verified in a single ordered test — the first `init_logger` in the process must
be the one below. Nothing else in the Python suite installs the logger.
"""

import os
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
