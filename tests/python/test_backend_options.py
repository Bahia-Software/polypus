"""
Registry ``options`` at the Python edge (issue #260): each value is a ``str`` or a
``list``/``tuple`` of ``str``.

- A list reaches the backend as an exact argv: the ``"subprocess"`` bridge launches a
  worker whose script lives under a path with spaces, which the whitespace-split
  string form cannot express.
- Any other value type is a ``TypeError`` naming the key, raised before a backend is
  built.
- ``local``/``cunqa`` still reject a non-empty ``options`` (they would ignore it).
"""

import shutil
import sys
from pathlib import Path

import pytest

WORKER_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "polypus-subprocess-backend"
    / "python"
    / "worker_template.py"
)

# The bridge is a Unix-only feature (see docs/backends.md).
unix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the subprocess bridge is not built on Windows"
)


@unix_only
def test_subprocess_command_list_launches_a_worker_under_a_spaced_path(tmp_path):
    import polypus

    spaced = tmp_path / "dir with spaces"
    spaced.mkdir()
    worker = spaced / "worker.py"
    shutil.copy(WORKER_TEMPLATE, worker)

    qc = polypus.Circuit(2).measure_all()
    result = polypus.run_quantum_circuit(
        qc,
        shots=128,
        infrastructure="subprocess",
        options={"command": [sys.executable, str(worker)]},
    )
    # The reference worker answers half the shots on all-zeros, half on all-ones.
    assert result.counts == [{"00": 64, "11": 64}]


@unix_only
def test_subprocess_command_tuple_is_accepted_like_a_list(tmp_path):
    import polypus

    spaced = tmp_path / "another dir"
    spaced.mkdir()
    worker = spaced / "worker.py"
    shutil.copy(WORKER_TEMPLATE, worker)

    result = polypus.run_quantum_circuit(
        polypus.Circuit(1).measure_all(),
        shots=10,
        infrastructure="subprocess",
        options={"command": (sys.executable, str(worker))},
    )
    assert result.counts == [{"0": 5, "1": 5}]


@unix_only
def test_subprocess_empty_command_list_is_a_backend_error():
    import polypus

    with pytest.raises(polypus.BackendError, match="'command' option is empty"):
        polypus.run_quantum_circuit(
            polypus.Circuit(1).measure_all(),
            shots=10,
            infrastructure="subprocess",
            options={"command": []},
        )


@unix_only
def test_subprocess_list_in_a_string_option_is_a_backend_error():
    import polypus

    with pytest.raises(polypus.BackendError, match="recv_timeout_ms"):
        polypus.run_quantum_circuit(
            polypus.Circuit(1).measure_all(),
            shots=10,
            infrastructure="subprocess",
            options={
                "command": [sys.executable, str(WORKER_TEMPLATE)],
                "recv_timeout_ms": ["1000"],
            },
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"k": 1}, r"options\['k'\] must be a str or a list of str, got int"),
        ({"k": True}, r"options\['k'\] must be a str or a list of str, got bool"),
        ({"k": None}, r"options\['k'\] must be a str or a list of str, got NoneType"),
        ({"k": {"a": "b"}}, r"options\['k'\] must be a str or a list of str, got dict"),
        ({"k": [1]}, r"options\['k'\]\[0\] must be a str, got int"),
        ({"k": ["a", None]}, r"options\['k'\]\[1\] must be a str, got NoneType"),
    ],
)
def test_unsupported_option_values_raise_type_error_naming_the_key(options, message):
    import polypus

    with pytest.raises(TypeError, match=message):
        polypus.run_quantum_circuit(
            polypus.Circuit(1).measure_all(),
            shots=10,
            infrastructure="subprocess",
            options=options,
        )


@pytest.mark.parametrize("infrastructure", ["local", "cunqa"])
@pytest.mark.parametrize("value", ["x", ["x"]])
def test_typed_builtins_still_reject_non_empty_options(infrastructure, value):
    import polypus

    with pytest.raises(ValueError, match="does not accept 'options'"):
        polypus.run_quantum_circuit(
            polypus.Circuit(1).measure_all(),
            shots=10,
            infrastructure=infrastructure,
            options={"command": value},
        )
