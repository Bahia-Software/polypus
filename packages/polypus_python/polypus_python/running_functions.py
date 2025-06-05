from qiskit_aer import AerSimulator
import json
import os, sys
import time
import random
from qiskit import QuantumCircuit
from qiskit.qpy import dump
from qiskit.qpy.exceptions import QpyError
from qiskit.qpy import load
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
import logging
import shutil
from cunqa import getQPUs

def get_logger():
    """Get or create the module logger that logs only to a file."""
    logger = logging.getLogger("polypus_python")
    if not logger.hasHandlers():
        # Log file will be in the temp directory next to this file
        log_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "polypus_python.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def log_message(message, level="error"):
    logger = get_logger()
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

def _load_configuration():
    """ Load the configuration json file """

    # Get the path to the configuration file
    config_file = 'configuration_backend.json'

    try:
        config_path = os.path.join(os.getcwd(), config_file)
    except Exception as e:
        log_message(f"Error constructing configuration file path: {e} - {config_path}", "error")
        return {"success": False, "error": "PathConstructionError", "message": str(e)}

    # Load the configuration from the JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        log_message(f"Configuration file not found: {e}", "error")
        raise
    except json.JSONDecodeError:
        log_message(f"Error decoding JSON configuration: {e}", "error")
        raise
    except Exception:
        log_message(f"Unexpected error loading configuration: {e}", "error")
        raise

    log_message("Configuration loaded successfully.", "info")
    return config

def _get_temp_directory():
    """Get the temporary directory for storing serialized files."""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp")
    except Exception as e:
        log_message(f"Error constructing temp directory path: {e}", "error")
        raise
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the folder exists
    return temp_dir

def _deserialize_quantum_circuit():
    """Deserialize a quantum circuit from a QPY file, handling possible errors."""
    
    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory()
    try:
        filename = os.path.join(temp_dir, "circuit.qpy")
    except Exception as e:
        log_message(f"Error constructing QPY filename: {e}", "error")
        raise

    try:
        with open(filename, "rb") as f:
            circuits = list(load(f))
            if not circuits:
                log_message("No circuits found in QPY file.", "error")
                raise QpyError("No circuits found in QPY file.")
            return circuits[0]
    except QpyError as e:
        log_message(f"QPY deserialization error: {e}", "error")
        raise
    except TypeError as e:
        log_message(f"Type error during deserialization: {e}", "error")
        raise
    except Exception as e:
        log_message(f"Unexpected error during deserialization: {e}", "error")
        raise

def test_connection():
    print("Testing connection to the QASM simulator backend...")

def serialize_quantum_circuit(qc):
    """Serialize the quantum circuit using Qiskit qpy, handling possible errors."""

    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory()
    temp_file = os.path.join(temp_dir, "circuit.qpy")

    try:
        with open(temp_file, "wb") as f:
            # Serialize the quantum circuit to a file
            dump(qc, f)
            log_message(f"Quantum circuit serialized successfully to {temp_file}.", "info")
    except QpyError as e:
        log_message(f"QPY serialization error: {e}", "error")
        raise
    except QiskitError as e:
        log_message(f"Qiskit error during serialization: {e}", "error")
        raise
    except MissingOptionalLibraryError as e:
        log_message(f"Missing optional library error during serialization: {e}", "error")
        raise
    except TypeError as e:
        log_message(f"Type error during serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(f"Unexpected error during serialization: {e}", "error")
        raise
    
    return True

def run_qc_in_qpu(qpu_id, shots):

    # Get the QPU
    tic_total = time.time()
    log_message(f"Running quantum circuit on QPU with ID {qpu_id} and {shots} shots.", "info")
    sys.path.append(os.getenv("HOME"))
    
    tic = time.time()
    qpus = getQPUs(local=False)
    log_message(f"Time to get QPUs: {time.time() - tic}s", "debug")
    qpu = qpus[qpu_id]
    if qpu is None:
        log_message(f"QPU with ID {qpu_id} not found.", "error")
        raise ValueError(f"QPU with ID {qpu_id} not found.")

    # Load the serialized quantum circuit
    qc = _deserialize_quantum_circuit()
    if qc is None:
        log_message("Failed to deserialize the quantum circuit.", "error")
        raise ValueError("Failed to deserialize the quantum circuit.")

    # Run the quantum circuit on the specified QPU
    tic = time.time()
    qjob = qpu.run(qc, shots = shots, transpile=False)

    # Get the counts
    counts = qjob.result.counts
    log_message(f"Time to call run :{time.time() - tic}s", "debug")
    log_message(f"Time to run from QPU: {qjob.result.time_taken}", "info")

    try:
        print(json.dumps(counts))  # Send to stdout
    except TypeError as e:
        log_message(f"Type error during JSON serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(f"Unexpected error during JSON serialization: {e}", "error")
        raise
    
    log_message(f"Total run time taken: {time.time() - tic_total}s", "info")
    return True

def run_qc(**kwargs):
    
    tic_total = time.time()

    # Load the configuration
    configuration = _load_configuration()

    # Load the serialized quantum circuit
    qc = _deserialize_quantum_circuit()
    if qc is None:
        log_message("Failed to deserialize the quantum circuit.", "error")
        raise ValueError("Failed to deserialize the quantum circuit.")

    # Backend 
    try:
        backend_name = configuration['backend_name']
    except KeyError as e:
        log_message(f"Missing backend_name in configuration: {e}", "error")
        raise

    if backend_name not in ['statevector']:
        log_message(f"Invalid backend: {backend_name}", "error")
        raise ValueError(f"Invalid backend: {backend_name}")

    # Number of shots
    if 'shots' in kwargs:
        shots = kwargs['shots']
    else:
        shots = configuration['n_shots']

    # Number of threads
    if 'max_parallel_threads' in kwargs:
        max_parallel_threads = kwargs['max_parallel_threads']
    else:
        max_parallel_threads = 0

    # Create a QASM simulator backend
    backend = AerSimulator(method=backend_name, max_parallel_threads=max_parallel_threads)
    
    # Execute the circuit on the QASM simulator
    job = backend.run(qc, shots=shots)
    
    # Grab results from the job
    result = job.result()
    
    # Get counts
    counts = result.get_counts(qc)

    # JSON serialization
    try:
        print(json.dumps(counts))  # Send to stdout
    except TypeError as e:
        log_message(f"Type error during JSON serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(f"Unexpected error during JSON serialization: {e}", "error")
        raise
    
    log_message(f"Time to run the quantum circuit: {result.time_taken}s", "debug")
    log_message(f"Total run time taken: {time.time() - tic_total}s", "info")
    return True
