import json
import logging
import os
import sys
import time

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


def get_logger(id):
    """Get or create the module logger that logs only to a file."""
    validate_id(id)
    logger = logging.getLogger("polypus_python")
    if not logger.hasHandlers():
        # Log file will be in the temp directory next to this file
        log_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"polypus_python_{id}.log")
        file_handler = logging.FileHandler(log_file)
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


def _load_configuration(id):
    """Load the configuration json file"""
    # Validated upfront so the `log_message` calls in the handlers below can
    # never raise on `id` and mask the original exception.
    validate_id(id)

    # Get the path to the configuration file
    config_file = "configuration_backend.json"

    try:
        config_path = os.path.join(os.getcwd(), config_file)
    except Exception as e:
        log_message(
            id,
            f"Error constructing configuration file path: {e} - {config_path}",
            "error",
        )
        return {"success": False, "error": "PathConstructionError", "message": str(e)}

    # Load the configuration from the JSON file
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        log_message(id, f"Configuration file not found: {e}", "error")
        raise
    except json.JSONDecodeError as e:
        log_message(id, f"Error decoding JSON configuration: {e}", "error")
        raise
    except Exception as e:
        log_message(id, f"Unexpected error loading configuration: {e}", "error")
        raise

    log_message(id, "Configuration loaded successfully.", "info")
    return config


def _get_temp_directory(id):
    """Get the temporary directory for storing serialized files."""
    validate_id(id)
    try:
        temp_dir = os.path.join(os.getcwd(), "temp")
    except Exception as e:
        log_message(id, f"Error constructing temp directory path: {e}", "error")
        raise
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the folder exists
    return temp_dir


def _deserialize_quantum_circuit(id):
    """Deserialize a quantum circuit from a QPY file, handling possible errors."""
    validate_id(id)

    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory(id)
    try:
        filename = os.path.join(temp_dir, f"circuit_{id}.qpy")
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


def serialize_quantum_circuit(id, qc):
    """Serialize the quantum circuit using Qiskit qpy, handling possible errors."""
    validate_id(id)

    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory(id)
    temp_file = os.path.join(temp_dir, f"circuit_{id}.qpy")

    try:
        with open(temp_file, "wb") as f:
            # Serialize the quantum circuit to a file
            dump(qc, f)
            log_message(
                id, f"Quantum circuit serialized successfully to {temp_file}.", "info"
            )
    except QpyError as e:
        log_message(id, f"QPY serialization error: {e}", "error")
        raise
    except QiskitError as e:
        log_message(id, f"Qiskit error during serialization: {e}", "error")
        raise
    except MissingOptionalLibraryError as e:
        log_message(
            id, f"Missing optional library error during serialization: {e}", "error"
        )
        raise
    except TypeError as e:
        log_message(id, f"Type error during serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(id, f"Unexpected error during serialization: {e}", "error")
        raise

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
