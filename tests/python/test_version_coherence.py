"""
Version coherence (contract C-6, docs/CONTRACTS.md).

``polypus.__version__`` is baked into the compiled extension from
``CARGO_PKG_VERSION``, so it describes the binary actually loaded rather than
the metadata of whatever distribution happens to be installed. These tests tie
it to the two other places the version lives: the ``[workspace.package]``
version in the root ``Cargo.toml`` (the single source of truth) and the
installed ``polypus-quantum`` distribution, whose version maturin derives from
that same ``Cargo.toml`` at build time.

Between releases ``main`` carries ``X.Y.Z-dev`` in ``Cargo.toml``; maturin
publishes it as the PEP 440 version ``X.Y.Z.dev0``, and ``__version__`` uses
that same form.
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

DISTRIBUTION = "polypus-quantum"
CARGO_TOML = Path(__file__).resolve().parents[2] / "Cargo.toml"


def _workspace_version(text):
    """The ``version`` of the ``[workspace.package]`` table of ``Cargo.toml``.

    Python 3.10 has no ``tomllib``, so this reads the table by hand, as the
    release workflow does: the table runs from its header to the next ``[``
    header, and its ``version = "X.Y.Z"`` line is the workspace version.
    """
    section = re.search(
        r"^\[workspace\.package\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S
    )
    assert section, "Cargo.toml has no [workspace.package] table"
    match = re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.M)
    assert match, "[workspace.package] in Cargo.toml has no version"
    return match.group(1)


def _python_version(cargo_version):
    """The PEP 440 form maturin gives a ``Cargo.toml`` version."""
    if cargo_version.endswith("-dev"):
        return cargo_version[: -len("-dev")] + ".dev0"
    return cargo_version


class TestVersionCoherence:
    def test_version_is_a_release_or_dev_string(self):
        import polypus

        assert isinstance(polypus.__version__, str)
        assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev0)?", polypus.__version__), (
            polypus.__version__
        )

    @pytest.mark.parametrize(
        "cargo, python", [("0.8.0", "0.8.0"), ("0.8.1-dev", "0.8.1.dev0")]
    )
    def test_cargo_versions_map_to_pep_440(self, cargo, python):
        assert _python_version(cargo) == python

    def test_version_is_exported(self):
        import polypus
        from polypus import __version__

        assert "__version__" in polypus.__all__
        assert __version__ == polypus.__version__

    def test_version_matches_installed_distribution(self):
        import polypus

        try:
            dist_version = version(DISTRIBUTION)
        except PackageNotFoundError:
            pytest.skip(
                f"the {DISTRIBUTION} distribution is not installed (extension loaded "
                "without its wheel metadata); CI installs the wheel and runs this check"
            )
        assert polypus.__version__ == dist_version

    def test_version_matches_workspace_cargo_toml(self):
        import polypus

        if not CARGO_TOML.is_file():
            pytest.skip(
                f"no Cargo.toml at {CARGO_TOML}: tests run outside the repository"
            )
        assert polypus.__version__ == _python_version(
            _workspace_version(CARGO_TOML.read_text(encoding="utf-8"))
        )
