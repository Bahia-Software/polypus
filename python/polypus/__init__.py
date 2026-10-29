"""Polypus — distributed quantum computing (Rust core + Python bindings).

This package is a thin wrapper around the compiled ``polypus`` extension
(``polypus/polypus*.so``). It re-exports the extension's public API unchanged,
and adds two pure-Python modules: ``polypus.ask_tell``, the adapters that run
Optuna, Nevergrad and pycma through ``polypus.External``, and ``polypus.shor``,
Shor's algorithm end to end.

Importing it has no CPU or disk side effects (issue #351). The native
simulator's gate-parallel threshold is calibrated for this machine on the first
native statevector run of a process instead (``polypus.statevector``, or
``run_quantum_circuit``/``train`` with ``backend="polypus"``); see
``polypus.calibrate_parallel_threshold`` and ``POLYPUS_NO_AUTOCALIBRATE``.
"""

# The pure-Python adapters for `polypus.External` (Optuna, Nevergrad, pycma),
# as `polypus.ask_tell`. They import none of those libraries, so this costs
# nothing at import. And Shor's algorithm end to end, as `polypus.shor`, which
# imports the extension only when it runs.
from . import ask_tell, shor

# Re-export the compiled extension's public API. Keep this identical to the
# wrapper maturin generates for a bare extension so ``import polypus`` behaves
# exactly as before. The submodule import also binds the name ``polypus`` (the
# extension) in this namespace, used just below.
from .polypus import *  # noqa: F401,F403

__doc__ = polypus.__doc__  # noqa: F405
if hasattr(polypus, "__all__"):  # noqa: F405
    __all__ = [*polypus.__all__, "ask_tell", "shor"]  # noqa: F405
