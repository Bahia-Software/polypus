from .local import Local


def connect_to_infrastructure(infrastructure: str, **kwargs):
    if infrastructure == "local":
        return "local"
    elif infrastructure == "cunqa":
        # Lazy import: `cunqa` is an OPTIONAL dependency (see polypus_python
        # pyproject `[cunqa]` extra). Importing it only inside this branch keeps
        # `import polypus_python` working when cunqa is not installed; the CUNQA
        # path then fails only if actually requested.
        from .cunqa import Cunqa

        # Reserve qpus in slurm
        return Cunqa().get_qpus(**kwargs)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")


def disconnect_from_infrastructure(infrastructure: str, **kwargs):
    if infrastructure == "local":
        return
    elif infrastructure == "cunqa":
        # CUNQA is an optional dependency; import it only when this path runs.
        from .cunqa import Cunqa

        return Cunqa().drop_qpus(**kwargs)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")


def run_qcs(infrastructure, **args):
    if infrastructure == "local":
        local = Local(num_qpus=1, qubits_per_qpu=[32], qpu_types=["AerSimulator"])
        return local.run_qcs(**args)
    elif infrastructure == "cunqa":
        # CUNQA is an optional dependency; import it only when this path runs.
        from .cunqa import Cunqa

        cunqa = Cunqa()
        return cunqa.run_qcs(**args)
    else:
        raise ValueError(f"Unknown infrastructure: {infrastructure}")
