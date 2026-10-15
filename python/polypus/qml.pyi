# Typing stubs for `polypus.qml`, a submodule the compiled extension creates at
# import time (it is not a file). Mirrors `qml_train` / `qml_predict` in
# `crates/polypus/src/bindings/mod.rs`; see `__init__.pyi` for how the stubs
# are checked.

from typing import Callable, Iterable, Optional, Sequence, Union

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from . import (
    CachedCost,
    Ising,
    Qubo,
    RunResult,
    SampleCost,
    TrainResult,
    _FloatVector,
    _Method,
)

__all__ = ["train", "predict"]

# `expectation_function` without `y_train`: one score per measured bitstring.
# With `y_train`, it reads each sample's label: `(bitstring, label) -> float`,
# the same memoised (`CachedCost`), or a per-sample `SampleCost`. Labels are
# `int` class indices or `float` targets, so a labelled callback must accept
# `float` (which includes `int`).
_QmlObjective = Union[
    Callable[[str], float],
    Callable[[str, float], float],
    CachedCost,
    SampleCost,
    Qubo,
    Ising,
]
# Rows of feature values, one per sample (a list of lists or a 2-D NumPy array).
_FeatureRows = Iterable[Sequence[float]]

def train(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    x_train: _FeatureRows,
    method: _Method,
    shots: int,
    n_qpus: int,
    dimensions: int,
    expectation_function: _QmlObjective,
    infrastructure: str,
    nodes: int,
    cores_per_qpu: int,
    id: str,
    sim_method: str = "automatic",
    noise_model: Optional[NoiseModel] = None,
    backend: str = "aer",
    seed: Optional[int] = None,
    options: Optional[dict[str, str]] = None,
    *,
    y_train: Optional[Iterable[float]] = None,
) -> TrainResult: ...
def predict(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    x: _FeatureRows,
    params: _FloatVector,
    shots: int,
    infrastructure: str,
    n_qpus: int = 1,
    nodes: int = 1,
    cores_per_qpu: int = 2,
    sim_method: str = "automatic",
    noise_model: Optional[NoiseModel] = None,
    backend: str = "aer",
    seed: Optional[int] = None,
) -> RunResult: ...
