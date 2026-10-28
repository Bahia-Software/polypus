# Typing stubs for the compiled `polypus` extension (PEP 561).
#
# Hand-written: every signature here mirrors a `#[pyfunction]`/`#[pymethods]`
# signature in `crates/polypus/src/bindings/`. CI checks them against the built
# wheel with `python -m mypy.stubtest` (docs/CONTRIBUTING.md), and
# `tests/python/test_typing_stubs.py` fails when the extension exports a name
# no stub declares. Change both sides together.
#
# Stubs must stay valid for the Python 3.10 floor: `typing` only, no
# `typing_extensions`.
#
# Qiskit publishes no type information, so for mypy `QuantumCircuit` and
# `NoiseModel` below are `Any` (or an `import-untyped` error in user code,
# depending on its configuration); pyright/Pylance resolve them by reading
# Qiskit's source.

from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    SupportsIndex,
    TypedDict,
    Union,
    final,
)

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from . import circuits as circuits
from . import external as external
from . import qml as qml

# What `External` runs (bindings/external.rs), typed in `polypus.external`.
from .external import AskTell as _AskTell

__all__ = [
    "__version__",
    "PolypusError",
    "BackendError",
    "CunqaError",
    "QmioError",
    "NativeCircuitError",
    "InsufficientMemoryError",
    "EvaluationError",
    "ConfigurationError",
    "DE",
    "PSO",
    "QNG",
    "CMAES",
    "Adam",
    "SPSA",
    "Rotosolve",
    "External",
    "Circuit",
    "Param",
    "ParamExpr",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "exp",
    "log",
    "sqrt",
    "pi",
    "tau",
    "euler",
    "RunResult",
    "TrainResult",
    "Qubo",
    "Ising",
    "CachedCost",
    "SampleCost",
    "train",
    "run_quantum_circuit",
    "statevector",
    "init_logger",
    "calibrate_parallel_threshold",
    "backend_cleanup_failures",
    "backend_compatibility",
    "qml",
    "circuits",
    "external",
]

__version__: str

# A vector of floats, as PyO3 extracts a `Vec<f64>`: any sequence, NumPy
# arrays included.
_FloatVector = Union[Sequence[float], npt.NDArray[np.floating[Any]]]
# One measurement-counts map (contract C-3): bitstring -> shots, keys sorted.
_Counts = dict[str, int]
# The optimizer configurations `train` / `qml.train` accept as `method`.
_Method = Union[DE, PSO, QNG, CMAES, Adam, SPSA, Rotosolve, External]
# `DE.strategy`: how trial vectors are built (bindings/de.rs). Any other name
# raises `ValueError`.
_DeStrategy = Literal["rand1bin", "shade", "lshade"]
# `Adam.rule`: how the gradient becomes a step (bindings/adam.rs). Any other
# name raises `ValueError`.
_AdamRule = Literal["adam", "momentum"]
# `options` of a registry-backed backend: each value a `str` or a list/tuple of
# `str` (docs/adr/0005-registry-option-values.md).
_Options = dict[str, Union[str, list[str], tuple[str, ...]]]
# `train`'s `expectation_function`: a `bitstring -> float` callable, the same
# memoised, or a native GIL-free observable.
_CostFunction = Union[Callable[[str], float], CachedCost, Qubo, Ising]

class _CalibrationOutcome(TypedDict):
    threshold: int
    num_threads: int
    duration_secs: float
    reused_cache: bool
    cache_path: Optional[str]
    cache_written: bool

class _CompatibilityReport(TypedDict):
    aer: list[str]
    polypus: list[str]
    cunqa: list[str]
    qmio: list[str]

# ── Exceptions ──────────────────────────────────────────────────────────────

class PolypusError(Exception): ...
class BackendError(PolypusError): ...
class CunqaError(BackendError): ...
class QmioError(BackendError): ...
class NativeCircuitError(BackendError): ...
class InsufficientMemoryError(BackendError): ...
class EvaluationError(PolypusError): ...
class ConfigurationError(PolypusError, ValueError): ...

# ── Optimizers ──────────────────────────────────────────────────────────────

@final
class DE:
    generations: int
    population_size: int
    tolerance: float
    patience: int
    seed: Optional[int]
    strategy: _DeStrategy
    # DE/rand/1/bin's F in (0, 2] and CR in [0, 1]; None means 0.8 and 0.7.
    # "shade" and "lshade" adapt both and raise ValueError for a value.
    mutation_factor: Optional[float]
    crossover_rate: Optional[float]
    def __new__(
        cls,
        generations: int = 100,
        population_size: int = 50,
        tolerance: float = 0.01,
        patience: int = 20,
        seed: Optional[int] = None,
        strategy: _DeStrategy = "rand1bin",
        mutation_factor: Optional[float] = None,
        crossover_rate: Optional[float] = None,
    ) -> DE: ...

@final
class PSO:
    generations: int
    population_size: int
    bounds: tuple[float, float]
    inertia_weight: float
    cognitive_weight: float
    social_weight: float
    tolerance: float
    seed: Optional[int]
    def __new__(
        cls,
        generations: int = 100,
        population_size: int = 50,
        # Default `(-pi, pi)` (bindings/pso.rs).
        bounds: tuple[float, float] = ...,
        inertia_weight: float = 0.5,
        cognitive_weight: float = 1.0,
        social_weight: float = 1.0,
        tolerance: float = 0.01,
        seed: Optional[int] = None,
    ) -> PSO: ...

@final
class QNG:
    max_iters: int
    bounds: tuple[float, float]
    learning_rate: float
    finite_difference_step: float
    tikhonov_reg: float
    # Called as `variance_function(theta, param_index)`; returns one QFIM
    # diagonal element (finite, non-negative).
    variance_function: Callable[[list[float], int], float]
    seed: Optional[int]
    def __new__(
        cls,
        variance_function: Callable[[list[float], int], float],
        max_iters: int = 100,
        # Default `(-pi, pi)` (bindings/qng.rs).
        bounds: tuple[float, float] = ...,
        learning_rate: float = 0.1,
        finite_difference_step: float = 0.1,
        tikhonov_reg: float = 0.05,
        seed: Optional[int] = None,
    ) -> QNG: ...

@final
class CMAES:
    generations: int
    # `None`: Hansen's default `4 + floor(3 ln d)` for `d = dimensions`.
    population_size: Optional[int]
    sigma0: float
    tolerance: float
    patience: int
    restarts: int
    seed: Optional[int]
    def __new__(
        cls,
        generations: int = 100,
        population_size: Optional[int] = None,
        # Default `pi / 2` (bindings/cmaes.rs).
        sigma0: float = ...,
        tolerance: float = 0.01,
        patience: int = 20,
        restarts: int = 0,
        seed: Optional[int] = None,
    ) -> CMAES: ...

@final
class Adam:
    max_iters: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    bounds: tuple[float, float]
    tolerance: float
    patience: int
    finite_difference_step: float
    rule: _AdamRule
    seed: Optional[int]
    def __new__(
        cls,
        max_iters: int = 100,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        # Default `(-pi, pi)` (bindings/adam.rs).
        bounds: tuple[float, float] = ...,
        tolerance: float = 0.01,
        patience: int = 20,
        finite_difference_step: float = 0.2,
        rule: _AdamRule = "adam",
        seed: Optional[int] = None,
    ) -> Adam: ...

@final
class SPSA:
    max_iters: int
    resamplings: int
    perturbation: float
    target_step: float
    alpha: float
    gamma: float
    eval_interval: int
    seed: Optional[int]
    def __new__(
        cls,
        max_iters: int = 300,
        resamplings: int = 1,
        perturbation: float = 0.1,
        target_step: float = 0.63,
        alpha: float = 0.602,
        gamma: float = 0.101,
        eval_interval: int = 20,
        seed: Optional[int] = None,
    ) -> SPSA: ...

@final
class Rotosolve:
    sweeps: int
    reevaluate_every: int
    tolerance: float
    patience: int
    seed: Optional[int]
    def __new__(
        cls,
        sweeps: int = 10,
        reevaluate_every: int = 5,
        tolerance: float = 0.01,
        patience: int = 2,
        seed: Optional[int] = None,
    ) -> Rotosolve: ...

@final
class External:
    optimizer: _AskTell
    max_evaluations: int
    def __new__(
        cls, optimizer: _AskTell, max_evaluations: SupportsIndex
    ) -> External: ...

# ── Native circuits ─────────────────────────────────────────────────────────

# What an angle argument and an expression operand accept.
_Angle = Union[float, Param, ParamExpr]
_Operand = _Angle

@final
class Param:
    # Defining `__eq__` makes the class unhashable at runtime (`__hash__` is
    # `None`); declaring that needs the ignore, as in typeshed, because
    # `object.__hash__` is a method.
    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __new__(cls, index: int) -> Param: ...
    @property
    def index(self) -> int: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __add__(self, other: _Operand, /) -> ParamExpr: ...
    def __radd__(self, other: _Operand, /) -> ParamExpr: ...
    def __sub__(self, other: _Operand, /) -> ParamExpr: ...
    def __rsub__(self, other: _Operand, /) -> ParamExpr: ...
    def __mul__(self, other: _Operand, /) -> ParamExpr: ...
    def __rmul__(self, other: _Operand, /) -> ParamExpr: ...
    def __truediv__(self, other: _Operand, /) -> ParamExpr: ...
    def __rtruediv__(self, other: _Operand, /) -> ParamExpr: ...
    def __pow__(self, other: _Operand, modulo: None = None, /) -> ParamExpr: ...
    def __rpow__(self, other: _Operand, modulo: None = None, /) -> ParamExpr: ...
    def __neg__(self) -> ParamExpr: ...

@final
class ParamExpr:
    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __eq__(self, value: object, /) -> bool: ...
    def __add__(self, other: _Operand, /) -> ParamExpr: ...
    def __radd__(self, other: _Operand, /) -> ParamExpr: ...
    def __sub__(self, other: _Operand, /) -> ParamExpr: ...
    def __rsub__(self, other: _Operand, /) -> ParamExpr: ...
    def __mul__(self, other: _Operand, /) -> ParamExpr: ...
    def __rmul__(self, other: _Operand, /) -> ParamExpr: ...
    def __truediv__(self, other: _Operand, /) -> ParamExpr: ...
    def __rtruediv__(self, other: _Operand, /) -> ParamExpr: ...
    def __pow__(self, other: _Operand, modulo: None = None, /) -> ParamExpr: ...
    def __rpow__(self, other: _Operand, modulo: None = None, /) -> ParamExpr: ...
    def __neg__(self) -> ParamExpr: ...

def sin(x: _Operand) -> ParamExpr: ...
def cos(x: _Operand) -> ParamExpr: ...
def tan(x: _Operand) -> ParamExpr: ...
def arcsin(x: _Operand) -> ParamExpr: ...
def arccos(x: _Operand) -> ParamExpr: ...
def arctan(x: _Operand) -> ParamExpr: ...
def exp(x: _Operand) -> ParamExpr: ...
def log(x: _Operand) -> ParamExpr: ...
def sqrt(x: _Operand) -> ParamExpr: ...

pi: ParamExpr
tau: ParamExpr
euler: ParamExpr

@final
class Circuit:
    def __new__(cls, num_qubits: int) -> Circuit: ...
    @staticmethod
    def from_qasm2(source: str) -> Circuit: ...
    @staticmethod
    def from_qasm3(source: str) -> Circuit: ...
    @property
    def param_names(self) -> list[str]: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_params(self) -> int: ...
    @property
    def num_clbits(self) -> int: ...
    @property
    def clbits_width(self) -> int: ...
    def h(self, qubit: int) -> Circuit: ...
    def x(self, qubit: int) -> Circuit: ...
    def y(self, qubit: int) -> Circuit: ...
    def z(self, qubit: int) -> Circuit: ...
    def s(self, qubit: int) -> Circuit: ...
    def t(self, qubit: int) -> Circuit: ...
    def sdg(self, qubit: int) -> Circuit: ...
    def tdg(self, qubit: int) -> Circuit: ...
    def id(self, qubit: int) -> Circuit: ...
    def rx(self, qubit: int, theta: _Angle) -> Circuit: ...
    def ry(self, qubit: int, theta: _Angle) -> Circuit: ...
    def rz(self, qubit: int, theta: _Angle) -> Circuit: ...
    def u(self, qubit: int, theta: _Angle, phi: _Angle, lam: _Angle) -> Circuit: ...
    def cx(self, control: int, target: int) -> Circuit: ...
    def cz(self, control: int, target: int) -> Circuit: ...
    def swap(self, q0: int, q1: int) -> Circuit: ...
    def rzz(self, q0: int, q1: int, theta: _Angle) -> Circuit: ...
    def rxx(self, q0: int, q1: int, theta: _Angle) -> Circuit: ...
    def cp(self, q0: int, q1: int, theta: _Angle) -> Circuit: ...
    def sx(self, qubit: int) -> Circuit: ...
    def sxdg(self, qubit: int) -> Circuit: ...
    def cy(self, control: int, target: int) -> Circuit: ...
    def ch(self, control: int, target: int) -> Circuit: ...
    def csx(self, control: int, target: int) -> Circuit: ...
    def ccx(self, control0: int, control1: int, target: int) -> Circuit: ...
    def cswap(self, control: int, target0: int, target1: int) -> Circuit: ...
    def crx(self, control: int, target: int, theta: _Angle) -> Circuit: ...
    def cry(self, control: int, target: int, theta: _Angle) -> Circuit: ...
    def crz(self, control: int, target: int, theta: _Angle) -> Circuit: ...
    def cu1(self, q0: int, q1: int, theta: _Angle) -> Circuit: ...
    def cu3(
        self, control: int, target: int, theta: _Angle, phi: _Angle, lam: _Angle
    ) -> Circuit: ...
    def cu(
        self,
        control: int,
        target: int,
        theta: _Angle,
        phi: _Angle,
        lam: _Angle,
        gamma: _Angle,
    ) -> Circuit: ...
    def p(self, qubit: int, lam: _Angle) -> Circuit: ...
    def u1(self, qubit: int, lam: _Angle) -> Circuit: ...
    def u2(self, qubit: int, phi: _Angle, lam: _Angle) -> Circuit: ...
    def u0(self, qubit: int, gamma: _Angle) -> Circuit: ...
    def rccx(self, control0: int, control1: int, target: int) -> Circuit: ...
    def rc3x(self, c0: int, c1: int, c2: int, target: int) -> Circuit: ...
    def c3x(self, c0: int, c1: int, c2: int, target: int) -> Circuit: ...
    def c3sqrtx(self, c0: int, c1: int, c2: int, target: int) -> Circuit: ...
    def c4x(self, c0: int, c1: int, c2: int, c3: int, target: int) -> Circuit: ...
    def barrier(self, qubits: Optional[Sequence[int]] = None) -> Circuit: ...
    def measure(self, qubit: int, cbit: int) -> Circuit: ...
    def measure_all(self) -> Circuit: ...
    def to_qasm2(self, params: Optional[_FloatVector] = None) -> str: ...
    def to_qasm3(self, params: Optional[_FloatVector] = None) -> str: ...
    def to_qir(self, params: Optional[_FloatVector] = None) -> str: ...
    def to_qir_bitcode(self, params: Optional[_FloatVector] = None) -> bytes: ...
    def draw(
        self,
        output: Literal["text", "svg"] = "text",
        *,
        ascii: bool = False,
        fold: Optional[int] = 80,
    ) -> str: ...
    def _repr_svg_(self) -> str: ...
    def __len__(self) -> int: ...

# ── Results ─────────────────────────────────────────────────────────────────

@final
class RunResult:
    @property
    def counts(self) -> list[_Counts]: ...
    # `None` for `qml.predict`, whose entries are distinct circuits.
    @property
    def merged_counts(self) -> Optional[_Counts]: ...
    @property
    def id(self) -> str: ...
    # `None` only for the `qmio` infrastructure (real hardware).
    @property
    def seed(self) -> Optional[int]: ...
    @property
    def backend(self) -> str: ...
    @property
    def infrastructure(self) -> str: ...

@final
class TrainResult:
    @property
    def best_params(self) -> list[float]: ...
    @property
    def best_fitness(self) -> float: ...
    @property
    def iterations_run(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fitness_history(self) -> list[float]: ...
    @property
    def final_params(self) -> Optional[list[float]]: ...
    @property
    def final_fitness(self) -> Optional[float]: ...
    @property
    def seed(self) -> int: ...
    @property
    def id(self) -> str: ...

# ── Native cost observables ─────────────────────────────────────────────────

@final
class Qubo:
    def __new__(
        cls,
        num_vars: int,
        linear: Optional[Sequence[tuple[int, float]]] = None,
        quadratic: Optional[Sequence[tuple[int, int, float]]] = None,
        constant: float = 0.0,
        scale: float = 1.0,
    ) -> Qubo: ...
    @staticmethod
    def from_matrix(
        matrix: Union[Sequence[Sequence[float]], npt.NDArray[np.floating[Any]]],
        scale: float = 1.0,
    ) -> Qubo: ...
    @property
    def num_vars(self) -> int: ...

@final
class Ising:
    def __new__(
        cls,
        num_vars: int,
        fields: Optional[Sequence[tuple[int, float]]] = None,
        couplings: Optional[Sequence[tuple[int, int, float]]] = None,
        constant: float = 0.0,
        scale: float = 1.0,
    ) -> Ising: ...
    @property
    def num_vars(self) -> int: ...

@final
class CachedCost:
    # `bitstring -> float`, or `(bitstring, label) -> float` with
    # `qml.train(..., y_train=...)`.
    def __new__(
        cls, cost_fn: Union[Callable[[str], float], Callable[[str, float], float]]
    ) -> CachedCost: ...

@final
class SampleCost:
    # `(counts, label) -> float`; labels are `int` class indices or `float`
    # targets, so the callback must accept `float` (which includes `int`).
    def __new__(cls, cost_fn: Callable[[_Counts, float], float]) -> SampleCost: ...

# ── Entry points ────────────────────────────────────────────────────────────

def train(
    qc: Union[QuantumCircuit, Circuit],
    method: _Method,
    shots: int,
    n_qpus: int,
    dimensions: int,
    expectation_function: _CostFunction,
    infrastructure: str,
    nodes: int,
    cores_per_qpu: int,
    id: str,
    sim_method: str = "automatic",
    noise_model: Optional[NoiseModel] = None,
    backend: str = "aer",
    seed: Optional[int] = None,
    fusion: Optional[bool] = None,
    options: Optional[_Options] = None,
    walltime: Optional[str] = None,
    initial_point: Optional[_FloatVector] = None,
    final_shots: Optional[SupportsIndex] = None,
) -> TrainResult: ...
def run_quantum_circuit(
    qc: Union[QuantumCircuit, Circuit, str],
    shots: int,
    infrastructure: str,
    n_qpus: int = 1,
    nodes: int = 1,
    cores_per_qpu: int = 2,
    sim_method: str = "automatic",
    noise_model: Optional[NoiseModel] = None,
    backend: str = "aer",
    seed: Optional[int] = None,
    fusion: Optional[bool] = None,
    options: Optional[_Options] = None,
    walltime: Optional[str] = None,
) -> RunResult: ...
def statevector(
    qc: Circuit, params: Optional[_FloatVector] = None, fusion: bool = True
) -> npt.NDArray[np.complex128]: ...
def init_logger(
    level: str = "info",
    format: str = "text",
    file: Optional[str] = None,
    run_name: Optional[str] = None,
    console: bool = False,
    timestamp: bool = True,
    thread_id: bool = False,
    module_path: bool = False,
) -> Optional[str]: ...
def calibrate_parallel_threshold(force: bool = False) -> _CalibrationOutcome: ...
def backend_cleanup_failures() -> int: ...
def backend_compatibility(
    circuit: Union[Circuit, str, QuantumCircuit],
    *,
    resources: bool = False,
) -> _CompatibilityReport: ...
