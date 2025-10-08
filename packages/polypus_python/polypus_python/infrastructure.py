from abc import ABC, abstractmethod

class Infraestructure(ABC):
    def __init__(self, num_qpus, qubits_per_qpu, qpu_types):
        """
        Args:
            num_qpus (int): Number of QPUs available.
            qubits_per_qpu (list[int]): Number of qubits per QPU.
            qpu_types (list[str]): Type of each QPU.
        """
        self.num_qpus = num_qpus
        self.qubits_per_qpu = qubits_per_qpu
        self.qpu_types = qpu_types

    @abstractmethod
    def run_qc(self, **kwargs) -> object:
        pass