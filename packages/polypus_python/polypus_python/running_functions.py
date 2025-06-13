import time
from qiskit_aer import AerSimulator
import json
import os, sys
import random
from qiskit import QuantumCircuit
from qiskit.qpy import dump
from qiskit.qpy.exceptions import QpyError
from qiskit.qpy import load
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
import logging
import shutil
from collections import Counter
from cunqa import getQPUs, gather

def get_logger(id):
    """Get or create the module logger that logs only to a file."""
    logger = logging.getLogger("polypus_python")
    if not logger.hasHandlers():
        # Log file will be in the temp directory next to this file
        log_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"polypus_python_{id}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def log_message(id, message, level="error"):
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
    """ Load the configuration json file """

    # Get the path to the configuration file
    config_file = 'configuration_backend.json'

    try:
        config_path = os.path.join(os.getcwd(), config_file)
    except Exception as e:
        log_message(id,f"Error constructing configuration file path: {e} - {config_path}", "error")
        return {"success": False, "error": "PathConstructionError", "message": str(e)}

    # Load the configuration from the JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        log_message(id,f"Configuration file not found: {e}", "error")
        raise
    except json.JSONDecodeError:
        log_message(id,f"Error decoding JSON configuration: {e}", "error")
        raise
    except Exception:
        log_message(id,f"Unexpected error loading configuration: {e}", "error")
        raise

    log_message(id,"Configuration loaded successfully.", "info")
    return config

def _get_temp_directory(id):
    """Get the temporary directory for storing serialized files."""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp")
    except Exception as e:
        log_message(id,f"Error constructing temp directory path: {e}", "error")
        raise
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the folder exists
    return temp_dir

def _deserialize_quantum_circuit(id):
    """Deserialize a quantum circuit from a QPY file, handling possible errors."""
    
    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory(id)
    try:
        filename = os.path.join(temp_dir, f"circuit_{id}.qpy")
    except Exception as e:
        log_message(id,f"Error constructing QPY filename: {e}", "error")
        raise

    try:
        with open(filename, "rb") as f:
            circuits = list(load(f))
            if not circuits:
                log_message(id,"No circuits found in QPY file.", "error")
                raise QpyError("No circuits found in QPY file.")
            return circuits[0]
    except QpyError as e:
        log_message(id,f"QPY deserialization error: {e}", "error")
        raise
    except TypeError as e:
        log_message(id,f"Type error during deserialization: {e}", "error")
        raise
    except Exception as e:
        log_message(id,f"Unexpected error during deserialization: {e}", "error")
        raise

def test_connection():
    print("Testing connection to the QASM simulator backend...")

def serialize_quantum_circuit(id, qc):
    """Serialize the quantum circuit using Qiskit qpy, handling possible errors."""

    # Temporary directory for the serialized file
    temp_dir = _get_temp_directory(id)
    temp_file = os.path.join(temp_dir, f"circuit_{id}.qpy")

    try:
        with open(temp_file, "wb") as f:
            # Serialize the quantum circuit to a file
            dump(qc, f)
            log_message(id,f"Quantum circuit serialized successfully to {temp_file}.", "info")
    except QpyError as e:
        log_message(id,f"QPY serialization error: {e}", "error")
        raise
    except QiskitError as e:
        log_message(id,f"Qiskit error during serialization: {e}", "error")
        raise
    except MissingOptionalLibraryError as e:
        log_message(id,f"Missing optional library error during serialization: {e}", "error")
        raise
    except TypeError as e:
        log_message(id,f"Type error during serialization: {e}", "error")
        raise
    except Exception as e:
        log_message(id,f"Unexpected error during serialization: {e}", "error")
        raise
    
    return True

def run_qcs_in_qpu(id, qcs, shots):

    # counts = []
    # for i in range(len(qcs)):
    #     counts.append(AerSimulator().run(qcs[i], shots=shots).result().get_counts(qcs[i]))
    # return counts

    # # Get the QPUs
    tic_total = time.time()
    sys.path.append(os.getenv("HOME"))
    try:
        qpus = getQPUs(local=False, family=id)
        log_message(id,f"Time to get QPUs: {time.time() - tic_total}s", "debug")
    except Exception as e:
        log_message(id,f"Error getting QPUs: {e}", "error")
        raise
    
    # Asynchronously run the quantum circuits on the QPUs
    try:
        qjobs = []
        for i in range(len(qcs)):
            qjob = qpus[i].run(qcs[i], shots = shots, transpile=False)
            qjobs.append(qjob)
            log_message(id,f"Running qpu: {qpus[i]}", "info")
        
        results = gather(qjobs)
        log_message(id, f"Results: {results}", "debug")
        counts = [result.counts for result in results]
        return counts
    except Exception as e:
        log_message(id,f"Error running quantum circuits on QPUs: {e}", "error")
        raise

def run_qc_in_qpu(id, qc, shots):

    # Get the QPUs
    tic_total = time.time()
    sys.path.append(os.getenv("HOME"))
    try:
        qpus = getQPUs(local=False, family=id)
        log_message(id,f"Time to get the QPU: {time.time() - tic_total}s", "debug")
    except Exception as e:
        log_message(id,f"Error getting QPU: {e}", "error")
        raise
    
    # Asynchronously run the quantum circuits on the QPUs
    try:
        qjobs = []
        for i in range(len(qpus)):
            qjob = qpus[i].run(qc, shots = shots, transpile=False)
            qjobs.append(qjob)
            log_message(id,f"Running qpu: {qpus[i]}", "info")
        
        results = gather(qjobs)
        log_message(id, f"Results: {results}", "debug")
        counts = [result.counts for result in results]
        return counts
    except Exception as e:
        log_message(id,f"Error running quantum circuits on QPUs: {e}", "error")
        raise

def run_qc(id, qc, shots, n_qpus):
    
    tic_total = time.time()
    if n_qpus == 1:
        try:
            backend = AerSimulator()
            qjob = backend.run(qc, shots = shots, transpile=False)
            log_message(id, f"Total time to run the quantum circuit: {time.time()-tic_total}s", "debug")
            return qjob.result().get_counts(qc)
        except Exception as e:
            log_message(id,f"Error running quantum circuit on QPU: {e}", "error")
            raise
    elif n_qpus > 1:
        shots_per_qpu = shots // n_qpus
        try:
            counts = []
            for i in range(n_qpus):
                qjob = AerSimulator().run(qc, shots=shots_per_qpu, transpile=False)
                log_message(id, f"Total time to run the quantum circuit: {time.time()-tic_total}s", "debug")
                counts.append(qjob.result().get_counts(qc))
            counts_sum = dict(sum((Counter(d) for d in counts), Counter()))
            return counts_sum
        except Exception as e:
            log_message(id,f"Error running quantum circuit on QPU: {e}", "error")
            raise
    else:
        raise Exception("n_qpus value not valid!")


