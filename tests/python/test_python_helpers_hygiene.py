"""
Hygiene of the ``polypus_python.running_functions`` helpers (issue #219).

These helpers used to write ``<cwd>/temp/circuit_<id>.qpy`` and
``<cwd>/temp/polypus_python_<id>.log``, forced the ``polypus_python`` logger to
DEBUG on every call (overriding the host application's logging setup), and
interpolated ``id`` into those paths without validation. Now:

- ``id`` is validated with the same policy as the Rust entry points (contract
  C-9, Python mirror): parity is checked against the very lists
  ``test_id_validation.py`` runs through ``train``/``qml.train``;
- the logger carries only a ``NullHandler`` unless ``POLYPUS_LOG_DIR`` opts in to
  a log file;
- serialized circuits go to a private per-user directory under the system temp
  dir (or an explicit ``directory=``), written atomically.

No test writes to the real system temp dir: ``tempfile.tempdir`` is redirected
into ``tmp_path`` for every test.
"""

import logging
import os
import stat
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


@pytest.fixture
def bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _default_dir(system_tmp):
    owner = os.getuid() if hasattr(os, "getuid") else rf.getpass.getuser()
    return system_tmp / f"polypus-{owner}"


def _symlink_circuit_outside(tmp_path, qc):
    """A ``shared`` dir whose ``circuit_run1.qpy`` symlinks to a file outside it."""
    shared = tmp_path / "shared"
    shared.mkdir()
    outside = tmp_path / "outside.qpy"
    rf.serialize_quantum_circuit("run1", qc, directory=tmp_path)
    os.replace(tmp_path / "circuit_run1.qpy", outside)
    (shared / "circuit_run1.qpy").symlink_to(outside)
    return shared, outside


def _file_handlers(logger):
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


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


class TestLoggingDefault:
    def test_host_level_kept_and_nothing_written(self, cwd, bell):
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.WARNING)

        rf.log_message("run1", "hello", "error")
        rf.log_message("run1", "hello", "debug")
        assert rf.serialize_quantum_circuit("run1", bell) is True

        assert not (cwd / "temp").exists()
        assert list(cwd.iterdir()) == []
        assert logger.level == logging.WARNING
        assert logger.handlers
        assert all(type(h) is logging.NullHandler for h in logger.handlers)

    def test_get_logger_returns_library_logger(self):
        assert rf.get_logger("run1") is logging.getLogger(LOGGER_NAME)

    def test_null_handler_installed_once(self):
        logger = logging.getLogger(LOGGER_NAME)
        rf._install_null_handler()
        rf._install_null_handler()
        assert sum(type(h) is logging.NullHandler for h in logger.handlers) == 1


class TestLoggingOptIn:
    def test_log_file_created_with_message(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs" / "nested"
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(log_dir))

        rf.log_message("run1", "hello file", "info")
        rf.log_message("run1", "debug line", "debug")
        for handler in _file_handlers(logging.getLogger(LOGGER_NAME)):
            handler.flush()

        content = (log_dir / "polypus_python_run1.log").read_text()
        assert "[INFO] hello file" in content
        # Opting in is the one case where the level is raised to DEBUG.
        assert "[DEBUG] debug line" in content
        assert logging.getLogger(LOGGER_NAME).level == logging.DEBUG

    def test_empty_value_is_not_an_opt_in(self, cwd, monkeypatch):
        monkeypatch.setenv("POLYPUS_LOG_DIR", "")
        rf.log_message("run1", "hello", "error")
        assert _file_handlers(logging.getLogger(LOGGER_NAME)) == []
        assert list(cwd.iterdir()) == []

    def test_unsetting_opt_in_removes_handler_and_restores_level(
        self, tmp_path, monkeypatch
    ):
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.WARNING)
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(tmp_path))
        rf.log_message("run1", "before unset", "error")
        [handler] = _file_handlers(logger)

        monkeypatch.delenv("POLYPUS_LOG_DIR")
        rf.log_message("run1", "after unset", "error")

        assert _file_handlers(logger) == []
        assert handler.stream is None
        assert logger.level == logging.WARNING
        content = (tmp_path / "polypus_python_run1.log").read_text()
        assert "before unset" in content
        assert "after unset" not in content

    def test_unsetting_opt_in_preserves_unrelated_file_handlers(
        self, tmp_path, monkeypatch
    ):
        logger = logging.getLogger(LOGGER_NAME)
        external_log = tmp_path / "external.log"
        external_handler = logging.FileHandler(external_log)
        logger.addHandler(external_handler)
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(tmp_path / "opt-in"))
        rf.log_message("run1", "before unset", "error")

        monkeypatch.delenv("POLYPUS_LOG_DIR")
        rf.log_message("run1", "after unset", "error")

        assert external_handler in logger.handlers
        assert external_handler.stream is not None
        external_handler.flush()
        assert "after unset" in external_log.read_text()

    def test_no_duplicate_handlers_and_first_id_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(tmp_path))
        logger = logging.getLogger(LOGGER_NAME)

        for _ in range(3):
            rf.log_message("run1", "again", "error")
        rf.log_message("run2", "other id", "error")
        for handler in _file_handlers(logger):
            handler.flush()

        [handler] = _file_handlers(logger)
        assert handler.baseFilename == str(tmp_path / "polypus_python_run1.log")
        assert not (tmp_path / "polypus_python_run2.log").exists()
        content = (tmp_path / "polypus_python_run1.log").read_text()
        assert content.count("again") == 3
        assert "other id" in content

    def test_unwritable_dir_warns_and_continues(self, tmp_path, monkeypatch):
        # A path below a regular file cannot be created, even when running as root.
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        monkeypatch.setenv("POLYPUS_LOG_DIR", str(blocker / "logs"))
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.WARNING)

        with pytest.warns(UserWarning, match="POLYPUS_LOG_DIR"):
            rf.log_message("run1", "still fine", "error")

        assert _file_handlers(logger) == []
        assert logger.level == logging.WARNING


class TestTempFiles:
    def test_default_round_trip(self, cwd, system_tmp, bell):
        assert rf.serialize_quantum_circuit("run1", bell) is True
        assert rf._deserialize_quantum_circuit("run1") == bell

        default_dir = _default_dir(system_tmp)
        assert [p.name for p in default_dir.iterdir()] == ["circuit_run1.qpy"]
        if hasattr(os, "getuid"):
            assert stat.S_IMODE(default_dir.stat().st_mode) == 0o700
        assert list(cwd.iterdir()) == []

    def test_overwrite_replaces_previous_circuit(self, system_tmp, bell):
        rf.serialize_quantum_circuit("run1", QuantumCircuit(1))
        rf.serialize_quantum_circuit("run1", bell)
        assert rf._deserialize_quantum_circuit("run1") == bell
        assert len(list(_default_dir(system_tmp).iterdir())) == 1

    def test_explicit_directory_round_trip(self, tmp_path, cwd, system_tmp, bell):
        shared = tmp_path / "shared" / "nested"
        assert rf.serialize_quantum_circuit("run1", bell, directory=shared) is True
        assert rf._deserialize_quantum_circuit("run1", directory=str(shared)) == bell

        assert [p.name for p in shared.iterdir()] == ["circuit_run1.qpy"]
        assert list(system_tmp.iterdir()) == []
        assert list(cwd.iterdir()) == []

    def test_explicit_directory_permissions_untouched(self, tmp_path, bell):
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o755)
        shared.chmod(0o755)
        rf.serialize_quantum_circuit("run1", bell, directory=shared)
        assert stat.S_IMODE(shared.stat().st_mode) == 0o755

    def test_failed_write_leaves_no_partial_file(self, system_tmp):
        with pytest.raises(TypeError):
            rf.serialize_quantum_circuit("run1", object())
        assert list(_default_dir(system_tmp).iterdir()) == []

    def test_circuit_file_symlinked_outside_is_refused(self, tmp_path, bell):
        shared, _ = _symlink_circuit_outside(tmp_path, bell)

        with pytest.raises(PermissionError, match="outside"):
            rf._deserialize_quantum_circuit("run1", directory=shared)

    @pytest.mark.parametrize(
        "call",
        [
            lambda shared: rf.serialize_quantum_circuit(
                "run1", QuantumCircuit(1), directory=shared
            ),
            lambda shared: rf._deserialize_quantum_circuit("run1", directory=shared),
        ],
        ids=["serialize", "deserialize"],
    )
    def test_containment_failure_is_logged_before_any_write(
        self, tmp_path, bell, caplog, call
    ):
        shared, outside = _symlink_circuit_outside(tmp_path, bell)
        before = outside.read_bytes()

        with pytest.raises(PermissionError, match="outside"):
            call(shared)

        # Refused before writing: no partial `.tmp`, the symlink untouched and
        # its target unchanged (`os.replace` would have clobbered the link).
        assert [p.name for p in shared.iterdir()] == ["circuit_run1.qpy"]
        assert (shared / "circuit_run1.qpy").is_symlink()
        assert outside.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "outside.qpy",
            "shared",
            "system-tmp",
        ]
        errors = [
            r
            for r in caplog.records
            if r.name == LOGGER_NAME and r.levelno == logging.ERROR
        ]
        assert any("Error constructing QPY filename" in r.getMessage() for r in errors)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX file modes")
class TestCircuitFileMode:
    """The `.qpy` is owner-only (`0o600`), even under a fully permissive umask.

    A stricter umask can only narrow the mode further, so the permissive case is
    the one that proves the file does not just inherit the process's umask.
    """

    @pytest.fixture(autouse=True)
    def _permissive_umask(self):
        previous = os.umask(0)
        yield
        os.umask(previous)

    def test_default_directory(self, system_tmp, bell):
        rf.serialize_quantum_circuit("run1", bell)
        qpy = _default_dir(system_tmp) / "circuit_run1.qpy"
        assert stat.S_IMODE(qpy.stat().st_mode) == 0o600

    def test_explicit_directory(self, tmp_path, bell):
        shared = tmp_path / "shared"
        rf.serialize_quantum_circuit("run1", bell, directory=shared)
        qpy = shared / "circuit_run1.qpy"
        assert stat.S_IMODE(qpy.stat().st_mode) == 0o600


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership checks")
class TestUnsafeDefaultDirectory:
    def test_loose_permissions_are_tightened(self, system_tmp, bell):
        default_dir = _default_dir(system_tmp)
        default_dir.mkdir()
        default_dir.chmod(0o777)
        rf.serialize_quantum_circuit("run1", bell)
        assert stat.S_IMODE(default_dir.stat().st_mode) == 0o700

    def test_symlink_refused(self, tmp_path, system_tmp, bell):
        target = tmp_path / "attacker"
        target.mkdir(mode=0o700)
        _default_dir(system_tmp).symlink_to(target, target_is_directory=True)

        with pytest.raises(PermissionError, match="symbolic link"):
            rf.serialize_quantum_circuit("run1", bell)
        assert list(target.iterdir()) == []

    def test_dangling_symlink_refused(self, tmp_path, system_tmp, bell):
        _default_dir(system_tmp).symlink_to(tmp_path / "missing")
        with pytest.raises(PermissionError, match="symbolic link"):
            rf.serialize_quantum_circuit("run1", bell)

    def test_regular_file_refused(self, system_tmp, bell):
        _default_dir(system_tmp).write_text("")
        with pytest.raises(PermissionError, match="not a directory"):
            rf.serialize_quantum_circuit("run1", bell)

    def test_foreign_owner_refused(self, system_tmp, bell, monkeypatch):
        default_dir = _default_dir(system_tmp)
        default_dir.mkdir(mode=0o700)
        real_lstat = os.lstat

        def foreign_lstat(path, *args, **kwargs):
            st = real_lstat(path, *args, **kwargs)
            if os.fspath(path) != str(default_dir):
                return st
            fields = list(st)
            fields[stat.ST_UID] = os.getuid() + 1
            return os.stat_result(fields)

        monkeypatch.setattr(os, "lstat", foreign_lstat)
        with pytest.raises(PermissionError, match="owned by uid"):
            rf.serialize_quantum_circuit("run1", bell)
        assert list(default_dir.iterdir()) == []
