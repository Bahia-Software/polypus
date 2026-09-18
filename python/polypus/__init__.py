"""Polypus — distributed quantum computing (Rust core + Python bindings).

This package is a thin wrapper around the compiled ``polypus`` extension
(``polypus/polypus*.so``). It re-exports the extension's public API unchanged
and, on first import, tunes the native statevector simulator's gate-parallel
crossover to *this* machine.

Why the tuning lives here (issue #176): ``calibrate_parallel_threshold`` measures
a hardware-specific value and caches it to disk. Source installs run it from
``install.sh`` (step 6b), but a ``pip install polypus-quantum`` of a published
wheel has no post-install hook, and the machine that *builds* the PyPI wheel in
CI is not the one that *runs* it — so a build-time measurement would calibrate
the wrong hardware. The only place that reaches the real runtime hardware with
zero manual steps is import time, here.
"""

# Re-export the compiled extension's public API. Keep this identical to the
# wrapper maturin generates for a bare extension so ``import polypus`` behaves
# exactly as before, plus the auto-calibration below. The submodule import also
# binds the name ``polypus`` (the extension) in this namespace, used just below.
from .polypus import *  # noqa: F401,F403

__doc__ = polypus.__doc__  # noqa: F405
if hasattr(polypus, "__all__"):  # noqa: F405
    __all__ = polypus.__all__  # noqa: F405


def _autocalibrate_parallel_threshold() -> None:
    """Tune (or reuse) the native gate-parallel threshold for this machine, once.

    ``force=False`` reuses a valid on-disk cache in ~0s and only measures on a
    cold or hardware-changed cache (<1s), so the cost is paid once per machine.
    A read-only cache dir (containers/CI) is not fatal — the call returns
    ``cache_written=False`` instead of raising. The whole thing is wrapped so a
    failure for *any* reason can never break ``import polypus``.

    Opt out with ``POLYPUS_NO_AUTOCALIBRATE=1`` (CI, containers, reproducibility
    runs that must pin the static default threshold).
    """
    import os

    optout = os.environ.get("POLYPUS_NO_AUTOCALIBRATE", "").strip().lower()
    if optout in ("1", "true", "yes", "on"):
        return
    try:
        polypus.calibrate_parallel_threshold(force=False)  # noqa: F405
    except Exception:
        # Calibration is a pure performance optimization: the parallel and
        # sequential paths are numerically identical, so a failure only means
        # the simulator falls back to its static default threshold. It must
        # never turn `import polypus` into a hard error.
        pass


_autocalibrate_parallel_threshold()
