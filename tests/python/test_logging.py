"""
Logger binding tests.

These exercise the two default behaviors that must hold when Polypus is driven
from Python:

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


def test_default_is_file_and_reinit_is_noop(tmp_path, monkeypatch):
    # Redirect the auto-generated default file into a throwaway dir.
    monkeypatch.setenv("POLYPUS_LOG_DIR", str(tmp_path))

    # (b) No file/console => a real file under POLYPUS_LOG_DIR, never stdout.
    path = polypus.init_logger(name="pytest_default")
    assert path is not None, "default target must be a file, not stdout"
    assert path.endswith(".log")
    assert os.path.exists(path)
    assert os.path.dirname(path) == str(tmp_path), "POLYPUS_LOG_DIR must be honored"

    # (c) Second init in the same process: no-op + UserWarning, first sink kept.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = polypus.init_logger(name="pytest_default_again")
    assert result is None, "a re-init must be a no-op (return None)"
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "a re-init must emit a UserWarning"
    )

    # The no-op must not have created (leaked) a second log file.
    created = list(tmp_path.glob("*.log"))
    assert len(created) == 1, f"re-init leaked extra log files: {created}"
