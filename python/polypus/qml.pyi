# Typing stubs for `polypus.qml`, a submodule the compiled extension creates at
# import time (it is not a file). Mirrors `qml_train` / `qml_predict` in
# `crates/polypus/src/bindings/mod.rs` and the native classes in
# `crates/polypus/src/bindings/qml.rs`; see `__init__.pyi` for how the stubs
# are checked.

from os import PathLike
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Union,
    final,
    overload,
)

import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from . import (
    CachedCost,
    Circuit,
    Ising,
    Qubo,
    RunResult,
    SampleCost,
    TrainResult,
    _FloatVector,
    _Method,
    _Options,
)

__all__ = [
    "train",
    "predict",
    "Encoder",
    "Ansatz",
    "Readout",
    "Model",
    "TrainedModel",
    "ExactResult",
    "load",
    "Dataset",
    "PauliTerm",
    "Observable",
    "Z",
    "X",
    "Y",
]

# ── Native pieces ───────────────────────────────────────────────────────────
# Each option is a string from a closed set (bindings/qml.rs); any other value
# raises `ValueError` naming the valid ones.
_Axis = Literal["rx", "ry", "rz"]
_Entanglement = Literal["linear", "circular", "full"]
_Entangler = Literal["cx", "cz"]
_ConvBlock = Literal["basic", "cartan"]
_Pairing = Literal["even_pairs", "odd_pairs", "alternating"]
_PoolBlock = Literal["basic"]
_KeepRule = Literal["even_positions", "odd_positions"]
_Pauli = Literal["x", "y", "z"]
_Decision = Literal["raw", "sign", "threshold", "argmax"]
_Loss = Literal[
    "squared_error",
    "binary_cross_entropy",
    "hinge",
    "categorical_cross_entropy",
    "expectation",
]
# A coefficient-free Pauli product: a `PauliTerm` or its `(pauli, position)` list.
_Term = Union[PauliTerm, Sequence[tuple[_Pauli, int]]]
# One readout observable: a product (coefficient 1.0) or a weighted sum.
_ReadoutObservable = Union[_Term, Observable]
# `Dataset` rows and labels: sequences or NumPy arrays (indexed, not iterated).
_DatasetRows = Union[Sequence[Sequence[float]], npt.NDArray[Any]]
_DatasetLabels = Union[Sequence[float], npt.NDArray[Any]]

@final
class PauliTerm:
    def __matmul__(self, other: PauliTerm, /) -> PauliTerm: ...
    def __rmatmul__(self, other: PauliTerm, /) -> PauliTerm: ...

def Z(position: int) -> PauliTerm: ...
def X(position: int) -> PauliTerm: ...
def Y(position: int) -> PauliTerm: ...

@final
class Observable:
    def __new__(cls, terms: Sequence[tuple[float, _Term]]) -> Observable: ...
    @property
    def num_terms(self) -> int: ...

@final
class Encoder:
    def __new__(cls, num_qubits: int) -> Encoder: ...
    def angle_encoder(self, axis: _Axis) -> Encoder: ...
    def amplitude_encoder(self) -> Encoder: ...
    def basis_encoder(self) -> Encoder: ...
    def iqp_encoder(self, entanglement: _Entanglement = "full") -> Encoder: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_layers(self) -> int: ...

@final
class Ansatz:
    def __new__(cls, num_qubits: int) -> Ansatz: ...
    def hardware_efficient(
        self,
        reps: int,
        rotations: Optional[Sequence[_Axis]] = None,
        entangler: _Entangler = "cx",
        entanglement: _Entanglement = "linear",
        final_rotation_layer: bool = True,
    ) -> Ansatz: ...
    def real_amplitudes(self, reps: int) -> Ansatz: ...
    def conv(self, block: _ConvBlock, pairing: _Pairing = "alternating") -> Ansatz: ...
    def pool(self, block: _PoolBlock, keep: _KeepRule = "even_positions") -> Ansatz: ...
    # Data re-uploading between trainable layers.
    def angle_encoder(self, axis: _Axis) -> Ansatz: ...
    def basis_encoder(self) -> Ansatz: ...
    def iqp_encoder(self, entanglement: _Entanglement = "full") -> Ansatz: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_params(self) -> int: ...

@final
class Readout:
    def __new__(
        cls,
        observables: Sequence[_ReadoutObservable],
        decision: _Decision = "raw",
        # Only with `decision="threshold"`, which requires it.
        threshold: Optional[float] = None,
    ) -> Readout: ...
    @property
    def decision(self) -> _Decision: ...
    @property
    def threshold(self) -> Optional[float]: ...
    @property
    def num_observables(self) -> int: ...

@final
class Model:
    def __new__(cls, encoder: Encoder, ansatz: Ansatz, readout: Readout) -> Model: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_params(self) -> int: ...
    def fit(
        self,
        x: Union[_DatasetRows, Dataset],
        # Labels; not with a `Dataset`, which carries its own.
        y: Optional[_DatasetLabels] = None,
        *,
        method: _Method,
        # Required with labels; `None` without them means "expectation".
        loss: Optional[_Loss] = None,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[NoiseModel] = None,
        # `None`: "polypus" on infrastructure="local", "aer" otherwise.
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        id: str = "qml",
        # Exact probabilities instead of shots; local native simulator only.
        exact: bool = False,
        # `1 <= batch_size < number of training samples`.
        batch_size: Optional[int] = None,
    ) -> TrainResult: ...
    def bind(self, params: Sequence[float]) -> TrainedModel: ...
    # The circuit for one sample, measured in the readout's basis; without
    # `params` its trainable angles stay free (`theta_0`, `theta_1`, ...).
    def circuit(
        self, x: _FloatVector, params: Optional[_FloatVector] = None
    ) -> Circuit: ...

@final
class TrainedModel:
    @property
    def params(self) -> list[float]: ...
    @property
    def model(self) -> Model: ...
    def predict(
        self,
        x: Union[_DatasetRows, Dataset],
        *,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[NoiseModel] = None,
        backend: Optional[str] = None,
        # Omitted: drawn (and reported by `run`); must be omitted on "qmio".
        seed: Optional[int] = None,
        exact: bool = False,
    ) -> list[Union[int, float]]: ...
    # Counts (`RunResult`) by default; exact probabilities (`ExactResult`) with
    # `exact=True`.
    @overload
    def run(
        self,
        x: Union[_DatasetRows, Dataset],
        *,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[NoiseModel] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        exact: Literal[False] = False,
    ) -> RunResult: ...
    @overload
    def run(
        self,
        x: Union[_DatasetRows, Dataset],
        *,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[NoiseModel] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        exact: Literal[True],
    ) -> ExactResult: ...
    @overload
    def run(
        self,
        x: Union[_DatasetRows, Dataset],
        *,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[NoiseModel] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        exact: bool = False,
    ) -> Union[RunResult, ExactResult]: ...
    # Versioned JSON, replacing any file at `path`; read back with `load`.
    def save(self, path: Union[str, PathLike[str]]) -> None: ...
    # The circuit `run` executes for one sample, every angle fixed.
    def circuit(self, x: _FloatVector) -> Circuit: ...

def load(path: Union[str, PathLike[str]]) -> TrainedModel: ...

@final
class ExactResult:
    # One `dict[str, float]` per row, keyed like counts.
    @property
    def probabilities(self) -> list[dict[str, float]]: ...
    @property
    def id(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def infrastructure(self) -> str: ...

@final
class Dataset:
    def __new__(
        cls, x: _DatasetRows, y: Optional[_DatasetLabels] = None
    ) -> Dataset: ...
    @property
    def num_samples(self) -> int: ...
    @property
    def num_features(self) -> int: ...
    @property
    def labeled(self) -> bool: ...
    @property
    def labels(self) -> Optional[list[float]]: ...
    def train_test_split(
        self, test_fraction: float, seed: Optional[int] = None
    ) -> tuple[Dataset, Dataset]: ...
    def scale_features_to(self, lo: float, hi: float) -> None: ...
    def feature_ranges(self) -> list[tuple[float, float]]: ...
    def scale_features_with(
        self, ranges: Sequence[tuple[float, float]], lo: float, hi: float
    ) -> None: ...

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
    options: Optional[_Options] = None,
    walltime: Optional[str] = None,
    *,
    y_train: Optional[Iterable[float]] = None,
    initial_point: Optional[_FloatVector] = None,
    final_shots: Optional[int] = None,
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
    walltime: Optional[str] = None,
) -> RunResult: ...
