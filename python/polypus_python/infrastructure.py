from abc import ABC, abstractmethod


class Infraestructure(ABC):
    @abstractmethod
    def get_qpus(self, **kwargs) -> object:
        pass

    @abstractmethod
    def run_qcs(self, **kwargs) -> object:
        pass
