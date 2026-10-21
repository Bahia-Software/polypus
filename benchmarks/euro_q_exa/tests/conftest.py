"""Test setup for the Euro-Q-Exa benchmark (#391).

These tests are deliberately outside CI (`pyproject.toml` points pytest at
`tests/python`): run them with `pytest benchmarks/euro_q_exa/tests`. The
benchmark's modules are scripts next to each other, not a package, so their
directory goes on `sys.path` here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
