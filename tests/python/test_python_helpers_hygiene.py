"""
Hygiene of the ``polypus_python.running_functions`` helpers (issue #219).

These helpers interpolated ``id`` into file paths (``circuit_<id>.qpy``,
``polypus_python_<id>.log``) and the CUNQA family name without validation. Now
``id`` is validated with the same policy as the Rust entry points (contract C-9,
Python mirror): parity is checked against the very lists
``test_id_validation.py`` runs through ``train``/``qml.train``.

No test writes to the real system temp dir: ``tempfile.tempdir`` is redirected
into ``tmp_path`` for every test.
"""

import logging
import sys
import tempfile

import pytest

pytest.importorskip("qiskit")

from polypus_python import running_functions as rf
from qiskit import QuantumCircuit
from test_id_validation import INVALID_IDS, VALID_IDS

LOGGER_NAME = "polypus_python"

# Beyond the shared C-9 lists: a trailing newline (which `re.match(...$)` would
# let through), a non-ASCII letter and digit (which `str.isalnum()` would let
# through), and an embedded NUL.
EXTRA_INVALID_IDS = [
    ("run\n", r"invalid character '\\n'"),
    ("ruñ", "invalid character 'ñ'"),
    ("run١", "invalid character '١'"),
    ("a\x00b", r"invalid character '\\x00'"),
]


@pytest.fixture(autouse=True)
def _restore_logger():
    """Restore the ``polypus_python`` logger's handlers and level after each test."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    yield logger
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture(autouse=True)
def system_tmp(tmp_path, monkeypatch):
    """Redirect the system temp dir into ``tmp_path`` and unset the log opt-in."""
    system_tmp = tmp_path / "system-tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    monkeypatch.delenv("POLYPUS_LOG_DIR", raising=False)
    return system_tmp


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    """An empty working directory, so writes relative to the cwd are visible."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# Every helper that validates `id`, called with everything else pinned.
VALIDATING_CALLS = {
    "get_logger": lambda id_: rf.get_logger(id_),
    "log_message": lambda id_: rf.log_message(id_, "msg", "error"),
    "_get_temp_directory": lambda id_: rf._get_temp_directory(id_),
    "serialize_quantum_circuit": lambda id_: rf.serialize_quantum_circuit(
        id_, QuantumCircuit(1)
    ),
    "_deserialize_quantum_circuit": lambda id_: rf._deserialize_quantum_circuit(id_),
    "run_qc_in_qpu": lambda id_: rf.run_qc_in_qpu(id_, QuantumCircuit(1), 10),
    "run_qcs_in_qpu": lambda id_: rf.run_qcs_in_qpu(id_, [QuantumCircuit(1)], 10),
}


class TestIdParityWithRust:
    """`validate_id` accepts and rejects exactly what Rust's `validate_id` does."""

    @pytest.mark.parametrize("id_", VALID_IDS)
    def test_valid_id_accepted(self, id_):
        rf.validate_id(id_)

    @pytest.mark.parametrize("id_,message", INVALID_IDS + EXTRA_INVALID_IDS)
    def test_invalid_id_rejected(self, id_, message):
        with pytest.raises(ValueError, match=message):
            rf.validate_id(id_)

    def test_messages_match_rust_wording(self):
        with pytest.raises(ValueError) as exc:
            rf.validate_id("my run")
        assert str(exc.value) == (
            "id contains invalid character ' '; only ASCII letters, digits, "
            "'.', '_' and '-' are allowed (got 'my run')"
        )
        with pytest.raises(ValueError) as exc:
            rf.validate_id("i" * 65)
        assert str(exc.value) == (
            f"id must be at most 64 characters, got 65 ({'i' * 65!r})"
        )

    def test_charset_checked_before_length(self):
        # Same order as Rust: an over-long id with a bad character reports the
        # character, not the length.
        with pytest.raises(ValueError, match="invalid character '/'"):
            rf.validate_id("/" + "i" * 70)

    @pytest.mark.parametrize("id_", [None, 1, b"run", ["run"]])
    def test_non_str_is_type_error(self, id_):
        with pytest.raises(TypeError, match="id must be a str"):
            rf.validate_id(id_)

    @pytest.mark.parametrize("call", VALIDATING_CALLS.values(), ids=VALIDATING_CALLS)
    @pytest.mark.parametrize("id_,message", INVALID_IDS + EXTRA_INVALID_IDS)
    def test_every_helper_validates(self, call, id_, message, monkeypatch):
        # cunqa stays unimportable: a ValueError proves validation came first.
        monkeypatch.setitem(sys.modules, "cunqa", None)
        with pytest.raises(ValueError, match=message):
            call(id_)


class TestInvalidIdTouchesNothing:
    def test_no_filesystem_writes(self, tmp_path, cwd, system_tmp, monkeypatch):
        log_dir = tmp_path / "logs"
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(log_dir))
        for call in VALIDATING_CALLS.values():
            with pytest.raises(ValueError):
                call("../x")
        assert list(cwd.iterdir()) == []
        assert list(system_tmp.iterdir()) == []
        assert not log_dir.exists()

    def test_run_qc_in_qpu_rejects_before_importing_cunqa(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cunqa", None)
        with pytest.raises(ValueError, match="invalid character '/'"):
            rf.run_qc_in_qpu("../x", QuantumCircuit(1), 10)
