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
# files), so their stubs are the only on-disk description of them, except
# `polypus.ask_tell` and `polypus.shor`, pure-Python modules typed inline.
STUBS = {
    "polypus": "__init__.pyi",
    "polypus.qml": "qml.pyi",
    "polypus.circuits": "circuits/__init__.pyi",
    "polypus.circuits.templates": "circuits/templates.pyi",
    "polypus.ask_tell": "ask_tell.py",
    "polypus.shor": "shor.py",
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


def _mypy(source, tmp_path):
    """mypy's report on `source`, checked against the installed stubs."""
    api = pytest.importorskip("mypy.api")
    path = tmp_path / "snippet.py"
    path.write_text(source, encoding="utf-8")
    config = Path(__file__).resolve().parent / "stubtest_mypy.ini"
    stdout, stderr, status = api.run(
        [
            "--config-file",
            str(config),
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "--no-error-summary",
            str(path),
        ]
    )
    return status, stdout + stderr


def _python_block(markdown, marker):
    """The first ```python block of `markdown` that contains `marker`."""
    for block in markdown.split("```python\n")[1:]:
        code = block.split("```", 1)[0]
        if marker in code:
            return code
    raise AssertionError(f"no python block contains {marker!r}")


class TestOptionsType:
    """`options` values are a `str` or a list/tuple of `str` (ADR 0005)."""

    def test_the_documented_subprocess_example_type_checks(self, tmp_path):
        docs = Path(__file__).resolve().parents[2] / "docs" / "backends.md"
        example = _python_block(docs.read_text(encoding="utf-8"), "options={")
        source = "from typing import Any\nqc: Any = None\n" + example
        status, report = _mypy(source, tmp_path)
        assert status == 0, report

    def test_a_value_the_runtime_rejects_does_not_type_check(self, tmp_path):
        # Proves the check above reads the stubs: an `int` value is a TypeError
        # at run time, so mypy must reject it too.
        source = (
            "import polypus\n"
            "polypus.run_quantum_circuit(\n"
            '    "", shots=1, infrastructure="subprocess", options={"cwd": 1}\n'
            ")\n"
        )
        status, report = _mypy(source, tmp_path)
        assert status == 1 and "snippet.py:3: error" in report, report


class TestAskTellTypes:
    def test_only_the_optuna_adapter_types_close(self, tmp_path):
        # `from_optuna` returns an `OptunaAskTell`, the other adapters an
        # `AskTell`, which has no `close()`.
        source = (
            "from typing import Any\n"
            "from polypus.ask_tell import from_optuna, from_pycma\n"
            "library: Any = None\n"
            "from_optuna(library, 2, 2).close()\n"
            "from_pycma(library).close()\n"
        )
        status, report = _mypy(source, tmp_path)
        assert status == 1, report
        assert "snippet.py:4:" not in report, report
        assert 'snippet.py:5: error: "AskTell" has no attribute "close"' in report
