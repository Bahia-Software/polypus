"""
PEP 561 typing stubs shipped with the ``polypus`` package.

The compiled extension carries no type information of its own, so the package
ships hand-written ``.pyi`` stubs plus the ``py.typed`` marker. CI checks the
stubs' signatures against the compiled extension with ``mypy.stubtest`` (see
``docs/CONTRIBUTING.md``); these tests catch the cheaper failures in the plain
``pytest`` run: a stub file missing from the installed package, or a public
name the extension exports that no stub declares.
"""

import ast
import importlib
from pathlib import Path

import pytest

# Each stubbed module of the package, with its stub relative to the package dir.
# The submodules are created by the extension at import time (they are not
# files), so their stubs are the only on-disk description of them.
STUBS = {
    "polypus": "__init__.pyi",
    "polypus.qml": "qml.pyi",
    "polypus.circuits": "circuits/__init__.pyi",
    "polypus.circuits.templates": "circuits/templates.pyi",
}


def _package_dir():
    import polypus

    return Path(polypus.__file__).resolve().parent


def _stub_names(path):
    """Every top-level name a stub file defines or explicitly re-exports."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.ImportFrom):
            # Only `import x as x` re-exports a name from a stub (PEP 484).
            names.update(a.asname for a in node.names if a.asname == a.name)
    return names


class TestStubFilesShipped:
    def test_py_typed_marker_is_present(self):
        marker = _package_dir() / "py.typed"
        assert marker.is_file(), (
            f"{marker} is missing: type checkers ignore the stubs without it"
        )

    @pytest.mark.parametrize("stub", sorted(STUBS.values()))
    def test_stub_file_is_present(self, stub):
        path = _package_dir() / stub
        assert path.is_file(), f"{path} is missing from the installed package"


class TestStubsCoverPublicApi:
    @pytest.mark.parametrize("module_name", sorted(STUBS))
    def test_every_exported_name_has_a_stub(self, module_name):
        module = importlib.import_module(module_name)
        stub = _package_dir() / STUBS[module_name]
        missing = sorted(set(module.__all__) - _stub_names(stub))
        assert not missing, (
            f"{module_name}.__all__ names with no stub in {stub.name}: {missing}"
        )
