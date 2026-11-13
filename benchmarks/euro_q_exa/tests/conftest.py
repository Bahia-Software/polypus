"""Test setup for the Euro-Q-Exa benchmark (#391).

These tests are deliberately outside CI (`pyproject.toml` points pytest at
`tests/python`): run them with `pytest benchmarks/euro_q_exa/tests`. The
benchmark's modules are scripts next to each other, not a package, so their
directory goes on `sys.path` here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import device as eqe  # noqa: E402
from synthetic import write_files  # noqa: E402


@pytest.fixture
def device(tmp_path):
    """The synthetic five-qubit device of `synthetic.py`."""
    cal, cmap = write_files(tmp_path)
    return eqe.build_device(eqe.load_calibration(cal), eqe.load_coupling_map(cmap))
