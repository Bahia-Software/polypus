"""
How ``polypus.torch`` is reached (issue #502), checked without PyTorch: this
file needs no torch and runs on every leg.

``import polypus`` imports neither ``polypus.torch`` nor PyTorch, so Polypus
works without it; ``import polypus.torch`` without PyTorch is an
``ImportError`` that names the extra; the wheel ships the module; and the
``torch`` extra asks for the version CI tests the layer with (the pins of
``.github/workflows/ci.yml``, ``examples.yml`` and ``sonar.yml``), as
``test_ask_tell_adapters.py`` checks its own extras. The layer itself is
tested in ``test_torch_quantum_layer.py``.
"""

import importlib.util
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_importing_polypus_imports_neither_the_layer_nor_torch():
    out = _run(
        "import sys, polypus, polypus.qml\n"
        "print(sorted({'torch', 'polypus.torch'} & set(sys.modules)))\n"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    assert "torch" not in __import__("polypus").__all__


def test_without_torch_the_import_error_names_the_extra():
    # `None` in sys.modules makes `import torch` fail as if it were missing.
    out = _run(
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "try:\n"
        "    import polypus.torch\n"
        "except ImportError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('polypus.torch imported without torch')\n"
    )
    assert out.returncode == 0, out.stderr
    assert 'pip install "polypus-quantum[torch]"' in out.stdout, out.stdout
    # The package is unaffected.
    assert (
        _run("import sys\nsys.modules['torch'] = None\nimport polypus\n").returncode
        == 0
    )


def test_the_wheel_ships_the_module():
    # find_spec locates the file without importing it, so no torch is needed.
    spec = importlib.util.find_spec("polypus.torch")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).name == "torch.py"
    assert (
        Path(spec.origin).parent
        == Path(importlib.util.find_spec("polypus").origin).parent
    )


def test_the_torch_extra_asks_for_the_version_ci_tests():
    # The extra's floor is the pin of CI's layer tests, of the examples job and
    # of Sonar's coverage job: raising a pin without its floor would let pip
    # install a version no test ran on. The upper bound is the next major.
    try:
        requirements = requires("polypus-quantum") or []
    except PackageNotFoundError:
        pytest.skip("the polypus-quantum distribution is not installed")
    workflows = [WORKFLOWS / name for name in ("ci.yml", "examples.yml", "sonar.yml")]
    if not all(path.is_file() for path in workflows):
        pytest.skip(f"no {WORKFLOWS}: tests run outside the repository")
    pins = {
        path.name: re.findall(r"\btorch==([\w.]+)", path.read_text())
        for path in workflows
    }
    (pin,) = set(pins["ci.yml"])
    assert set(pins["examples.yml"]) == set(pins["sonar.yml"]) == {pin}, pins
    extra = [r for r in requirements if re.search(r"""extra\s*==\s*["']torch["']""", r)]
    assert len(extra) == 1, requirements
    requirement = extra[0].split(";")[0].replace(" ", "")
    major = int(pin.split(".")[0])
    assert re.fullmatch(
        rf"torch(>={re.escape(pin)},<{major + 1}|<{major + 1},>={re.escape(pin)})",
        requirement,
    ), requirement
