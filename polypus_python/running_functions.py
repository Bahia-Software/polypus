import contextlib
import getpass
import logging
import os
import stat
import sys
import tempfile
import time
import warnings

from qiskit.exceptions import MissingOptionalLibraryError, QiskitError
from qiskit.qpy import dump, load
from qiskit.qpy.exceptions import QpyError

# Optional escape hatch for a non-packaged CUNQA site install (mirrors cunqa.py):
# a portable, explicit location, unlike the previous unconditional
# `sys.path.append(os.getenv("HOME"))`, which appended None when HOME was unset.
_cunqa_path = os.getenv("POLYPUS_CUNQA_PATH")
if _cunqa_path and _cunqa_path not in sys.path:
    sys.path.append(_cunqa_path)


# Contract C-9, mirrored from `validate_id` in crates/polypus/src/bindings/mod.rs:
# same charset, same length bound, same order of checks and same messages.
_ID_MAX_LEN = 64
# Spelled out rather than `str.isalnum()`, which also accepts non-ASCII letters
# and digits (e.g. "ñ", "١").
_ID_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)

_LOGGER_NAME = "polypus_python"


def validate_id(id):
    """Validate a run ``id`` before it names a file or reaches CUNQA/SLURM.

    Same policy as the Rust entry points (contract C-9): non-empty, only ASCII
    letters, ASCII digits, ``.``, ``_`` and ``-``, and at most 64 characters,
    checked in that order. The charset is checked before the length so a
    non-ASCII id gets the specific "invalid character" message.

    Raises:
        TypeError: ``id`` is not a ``str``.
        ValueError: ``id`` violates the policy above.
    """
    if not isinstance(id, str):
        raise TypeError(f"id must be a str, got {type(id).__name__}")
    if not id:
        raise ValueError("id must not be empty")
    for c in id:
        if c not in _ID_ALLOWED_CHARS:
            raise ValueError(
                f"id contains invalid character {c!r}; only ASCII letters, "
                f"digits, '.', '_' and '-' are allowed (got {id!r})"
            )
    if len(id) > _ID_MAX_LEN:
        raise ValueError(
            f"id must be at most {_ID_MAX_LEN} characters, got {len(id)} ({id!r})"
        )


def _install_null_handler():
    """Give the library logger a NullHandler, once.

    Standard library etiquette: without it, records would fall through to
    `logging.lastResort` and print to stderr when the host configured nothing.
    Checked rather than added blindly so a module reload stays idempotent.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(type(h) is logging.NullHandler for h in logger.handlers):
        logger.addHandler(logging.NullHandler())


_install_null_handler()


def get_logger(id):
    """Return the ``polypus_python`` logger, optionally logging to a file.

    By default the logger only carries a ``NullHandler``: its level is left to
    the host application and records propagate to the host's handlers as usual.

    Writing to a file is opt-in via the ``POLYPUS_LOG_DIR`` environment
    variable (read on every call; set and non-empty, as on the Rust side). The
    directory is created if needed, a ``FileHandler`` writes to
    ``<POLYPUS_LOG_DIR>/polypus_python_<id>.log`` and only then is the logger
    level set to DEBUG. The first ``id`` wins: once a file handler is attached,
    later calls with another ``id`` reuse it rather than adding a second one,
    so messages never fan out across files. If the directory or the file
    cannot be created, a ``UserWarning`` is emitted and logging continues
    without the file.

    Raises:
        TypeError, ValueError: ``id`` fails `validate_id` (contract C-9).
    """
    validate_id(id)
    logger = logging.getLogger(_LOGGER_NAME)
    log_dir = os.environ.get("POLYPUS_LOG_DIR")
    if not log_dir:
        owned_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "_polypus_owned", False)
        ]
        for handler in owned_handlers:
            logger.removeHandler(handler)
            handler.close()
        if owned_handlers and hasattr(owned_handlers[0], "_polypus_previous_level"):
            logger.setLevel(owned_handlers[0]._polypus_previous_level)
        return logger

    log_file = os.path.abspath(os.path.join(log_dir, f"polypus_python_{id}.log"))
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    if any(h.baseFilename == log_file for h in file_handlers) or any(
        getattr(h, "_polypus_owned", False) for h in file_handlers
    ):
        return logger

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
    except OSError as e:
        warnings.warn(
            f"polypus_python: cannot log to {log_file} (POLYPUS_LOG_DIR): {e}; "
            "continuing without a log file",
            UserWarning,
            stacklevel=2,
        )
        return logger
    file_handler._polypus_owned = True
    file_handler._polypus_previous_level = logger.level
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def log_message(id, message, level="error"):
    validate_id(id)
    logger = get_logger(id)
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.debug(message)


def _default_temp_directory():
    """Return the private per-user directory under the system temp dir.

    ``<tempfile.gettempdir()>/polypus-<uid>`` (the user name where there is no
    uid), created ``0o700``. An existing entry is accepted only if it is a real
    directory (not a symlink) owned by the current user; one that is ours but
    group/world-accessible is tightened back to ``0o700``. Anything else raises
    ``PermissionError`` rather than being reused.
    """
    has_uid = hasattr(os, "getuid")
    owner = os.getuid() if has_uid else getpass.getuser()
    path = os.path.join(tempfile.gettempdir(), f"polypus-{owner}")
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except FileExistsError:
        # A non-directory (or dangling symlink) is in the way; the checks below
        # turn it into a clear PermissionError.
        pass

    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(
            f"refusing to use temp directory {path}: it is a symbolic link"
        )
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(
            f"refusing to use temp directory {path}: it is not a directory"
        )
    if has_uid:
        if st.st_uid != os.getuid():
            raise PermissionError(
                f"refusing to use temp directory {path}: it is owned by uid "
                f"{st.st_uid}, not by the current user (uid {os.getuid()})"
            )
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.chmod(path, 0o700)
    return path


def _get_temp_directory(id, directory=None):
    """Get the directory for storing serialized files.

    With ``directory=None`` this is the private per-user directory under the
    system temp dir (see `_default_temp_directory`), which is usually local to
    the node. To hand a circuit to another node, pass ``directory`` pointing to
    a shared filesystem; it is used as given and created if missing, without
    touching its permissions.
    """
    validate_id(id)
    try:
        if directory is None:
            return _default_temp_directory()
        os.makedirs(directory, exist_ok=True)
        return os.fspath(directory)
    except Exception as e:
        log_message(id, f"Error constructing temp directory path: {e}", "error")
        raise


def _circuit_file(id, directory):
    """Return ``(temp_dir, path)`` of the QPY file paired with ``id``.

    Both `serialize_quantum_circuit` and `_deserialize_quantum_circuit` resolve
    the file through here, so the ``id -> circuit_<id>.qpy`` pairing lives in
    one place. As defense in depth on top of `validate_id`, the resolved path
    must stay inside the resolved base directory.
    """
    temp_dir = _get_temp_directory(id, directory)
    filename = os.path.join(temp_dir, f"circuit_{id}.qpy")
    base = os.path.realpath(temp_dir)
    resolved = os.path.realpath(filename)
    if os.path.dirname(resolved) != base:
        raise PermissionError(
            f"refusing to use {filename}: it resolves to {resolved}, outside {base}"
        )
    return temp_dir, filename


def _deserialize_quantum_circuit(id, *, directory=None):
    """Deserialize a quantum circuit from a QPY file, handling possible errors.

    Reads ``circuit_<id>.qpy`` from the directory `serialize_quantum_circuit`
    writes to for the same ``id`` and ``directory`` (see `_get_temp_directory`).
    """
    validate_id(id)

    try:
        _, filename = _circuit_file(id, directory)
    except Exception as e:
        log_message(id, f"Error constructing QPY filename: {e}", "error")
        raise

    try:
        with open(filename, "rb") as f:
            circuits = list(load(f))
            if not circuits:
                log_message(id, "No circuits found in QPY file.", "error")
                raise QpyError("No circuits found in QPY file.")
            return circuits[0]
    except QpyError as e:
        log_message(id, f"QPY deserialization error: {e}", "error")
        raise
    except TypeError as e:
        log_message(id, f"Type error during deserialization: {e}", "error")
        raise
    except Exception as e:
        log_message(id, f"Unexpected error during deserialization: {e}", "error")
        raise


def test_connection():
    print("Testing connection to the QASM simulator backend...")


def serialize_quantum_circuit(id, qc, *, directory=None):
    """Serialize the quantum circuit using Qiskit qpy, handling possible errors.

    Writes ``circuit_<id>.qpy`` to the private per-user directory under the
    system temp dir, or to ``directory`` when given (see `_get_temp_directory`:
    the default is usually node-local, so pass a shared ``directory`` to read
    the circuit back on another node). The file is written to a temporary name
    in the same directory and then atomically renamed, so a concurrent reader
    never sees a partial file. The containment check of `_circuit_file` runs
    before anything is written. Returns ``True``.

    The file is created with mode ``0o600`` (as ``tempfile`` does): it is
    readable only by the owning user. A permissive umask never widens that; a
    stricter one can only narrow it further. With ``directory`` pointing to a
    directory shared by several users, the other users will not be able to read
    it.
    """
    validate_id(id)

    try:
        temp_dir, temp_file = _circuit_file(id, directory)
    except Exception as e:
        log_message(id, f"Error constructing QPY filename: {e}", "error")
        raise
    partial = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir, suffix=".tmp", delete=False
        ) as f:
            partial = f.name
            # Serialize the quantum circuit to a file
            dump(qc, f)
        os.replace(partial, temp_file)
        partial = None
        log_message(
            id, f"Quantum circuit serialized successfully to {temp_file}.", "info"
        )
    except QpyError as e:
        log_message(id, f"QPY serialization error: {e}", "error")
        raise
    except MissingOptionalLibraryError as e:
        log_message(
            id, f"Missing optional library error during serialization: {e}", "error"
        )
        raise
    except QiskitError as e:
        log_message(id, f"Qiskit error during serialization: {e}", "error")
        raise
    except TypeError as e:
        log_message(id, f"Type error during serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(id, f"Unexpected error during serialization: {e}", "error")
        raise
    finally:
        if partial is not None:
            with contextlib.suppress(OSError):
                os.unlink(partial)

    return True


def run_qcs_in_qpu(id, qcs, shots):
    # `id` travels to CUNQA/SLURM as the `family` name (contract C-9); validate
    # it before the optional import so a bad id is a ValueError, not an
    # ImportError, and so the `log_message` below cannot raise on it.
    validate_id(id)

    # counts = []
    # for i in range(len(qcs)):
    #     counts.append(AerSimulator().run(qcs[i], shots=shots).result().get_counts(qcs[i]))
    # return counts

    # CUNQA is an optional dependency; import it only when this path runs.
    from cunqa.qjob import gather
    from cunqa.qutils import get_QPUs

    # # Get the QPUs
    # sys.path.append(os.getenv("HOME"))
    try:
        qpus = get_QPUs(local=False, family=id)
        # log_message(id,f"Time to get QPUs: {time.time() - tic_total}s", "debug")
    except Exception:
        # log_message(id,f"Error getting QPUs: {e}", "error")
        raise

    # Asynchronously run the quantum circuits on the QPUs
    try:
        qjobs = []
        for i in range(len(qcs)):
            qjob = qpus[i].run(qcs[i], shots=shots, transpile=False)
            qjobs.append(qjob)
            # log_message(id,f"Running qpu: {qpus[i]}", "info")

        results = gather(qjobs)
        # log_message(id, f"Results: {results}", "debug")
        counts = [result.counts for result in results]
        return counts
    except Exception as e:
        log_message(id, f"Error running quantum circuits on QPUs: {e}", "error")
        raise


def run_qc_in_qpu(id, qc, shots):
    # As in `run_qcs_in_qpu`: `id` becomes the CUNQA `family` (contract C-9).
    validate_id(id)
    # CUNQA is an optional dependency; import it only when this path runs.
    from cunqa.qjob import gather
    from cunqa.qutils import get_QPUs

    # Get the QPUs
    tic_total = time.time()
    try:
        qpus = get_QPUs(local=False, family=id)
        log_message(id, f"Time to get the QPU: {time.time() - tic_total}s", "debug")
    except Exception as e:
        log_message(id, f"Error getting QPU: {e}", "error")
        raise

    # Asynchronously run the quantum circuits on the QPUs
    try:
        qjobs = []
        for i in range(len(qpus)):
            qjob = qpus[i].run(qc, shots=shots, transpile=False)
            qjobs.append(qjob)
            log_message(id, f"Running qpu: {qpus[i]}", "info")

        results = gather(qjobs)
        log_message(id, f"Results: {results}", "debug")
        counts = [result.counts for result in results]
        return counts
    except Exception as e:
        log_message(id, f"Error running quantum circuits on QPUs: {e}", "error")
        raise
