from abc import ABC, abstractmethod

class Infraestructure(ABC):
    def __init__(self, **kwargs):
        """
        Args (all optional, passed via kwargs):
            num_qpus (int): Number of QPUs available.
            qubits_per_qpu (list[int]): Number of qubits per QPU.
            qpu_types (list[str]): Type of each QPU.

        Any missing value defaults to None.
        """
        self.num_qpus = kwargs.get('num_qpus', None)
        self.qubits_per_qpu = kwargs.get('qubits_per_qpu', None)
        self.qpu_types = kwargs.get('qpu_types', None)

    @abstractmethod
    def get_qpus(self, **kwargs) -> object:
        pass

    @abstractmethod
    def run_qcs(self, **kwargs) -> object:
        pass