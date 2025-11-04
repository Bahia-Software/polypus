from .local import Local
from .cunqa import Cunqa

def connect_to_infrastructure(infrastructure: str, **kwargs):
    if infrastructure == "local":
        return "local"
    elif infrastructure == "cunqa":
        # Reserve qpus in slurm
        return Cunqa().get_qpus(**kwargs)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")

def disconnect_from_infrastructure(infrastructure: str, **kwargs):
    if infrastructure == "local":
        return
    elif infrastructure == "cunqa":
        return Cunqa().drop_qpus(**kwargs)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")
    
def run_qcs(infrastructure, **args):
    if infrastructure == "local":
        local = Local(num_qpus=1, qubits_per_qpu=[32], qpu_types=["AerSimulator"])
        return local.run_qcs(**args)
    elif infrastructure == "cunqa":
        cunqa = Cunqa()
        return cunqa.run_qcs(**args)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")

