from .local import Local

def connect_to_infrastructure(infrastructure: str):
    if infrastructure == "local":
        return "local"
    elif infrastructure == "cunqa":
        return "cunqa"
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")
    

def run_qc(infrastructure, **args):
    if infrastructure == "local":
        local = Local(num_qpus=1, qubits_per_qpu=[32], qpu_types=["AerSimulator"])
        return local.run_qc(**args)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")

