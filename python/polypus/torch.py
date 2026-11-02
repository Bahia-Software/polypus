"""A native ``polypus.qml.Model`` as a PyTorch layer.

:class:`QuantumLayer` is a ``torch.nn.Module`` whose trainable parameters are
the model's: its forward pass is ``polypus.qml.expval`` over the batch and its
backward pass ``polypus.qml.jacobian``, so it fits in a ``torch.nn.Sequential``
next to classical layers and trains with any ``torch.optim`` optimizer::

    from polypus import qml
    from polypus.torch import QuantumLayer

    model = qml.Model(
        qml.Encoder(2).angle_encoder("ry"),
        qml.Ansatz(2).real_amplitudes(1),
        qml.Readout([qml.Z(0), qml.Z(1)]),
    )
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 2),
        QuantumLayer(model, exact=True),
        torch.nn.Linear(model.num_observables, 1),
    )

PyTorch is not a dependency of Polypus: install it, or Polypus with it as an
extra, ``pip install "polypus-quantum[torch]"``, which asks for a version this
module is tested with. ``import polypus`` never imports this module, nor
PyTorch: import it as ``polypus.torch``.

- **Forward.** ``layer(x)`` takes a ``(batch, features)`` tensor, or one sample
  as a 1-D tensor, and returns every readout observable's expectation,
  ``(batch, num_observables)`` (``(num_observables,)`` for one sample): one
  ``qml.expval`` call with the whole batch, with the layer's execution
  arguments, which are ``qml.expval``'s and are checked by it at the first
  forward pass.
- **Backward.** One ``qml.jacobian`` call per backward pass, with respect to
  what needs a gradient: the parameters, the inputs (an angle or IQP encoder;
  an amplitude or basis encoder has no input gradient and raises
  ``polypus.ConfigurationError``), or both. The gradients are the
  vector-Jacobian products of the incoming gradient with the parameter-shift
  Jacobian: exact with ``exact=True``, unbiased estimates with shots.
- **dtype and device.** The quantum computation runs in ``float64`` on the CPU.
  The input is converted there at the boundary, and the output comes back in
  the input's dtype and device (``float64`` for a non-floating input), each
  gradient in its tensor's: a ``float32`` network needs no ``.double()``, at
  the cost of a rounding at the boundary. The parameter is ``float64`` unless
  the layer is converted (``layer.float()``). GPU tensors are moved to the CPU
  and back; that path is not tested in CI.
- **Seeds.** With ``seed``, every call to the backend gets its own seed,
  derived from ``(seed, number of the forward pass, forward or backward)``:
  a backward pass uses the number of its forward pass, so two layers built
  with the same ``seed`` and fed the same data give the same values and
  gradients, step by step (a forward pass counts even when it raises, so twin
  layers stay in step). The counter is not in the ``state_dict``: to
  replay, build a new layer with the same ``seed``. Without it each call draws
  its own (contract C-7). The counter makes a seeded layer unsafe to call from
  several threads at once.
- **Cost.** Each forward and each backward pass opens and closes its own
  backend: two allocations per training step on CUNQA (``qraise``).
- **Limits.** No second derivatives (the backward pass is not itself
  differentiable), no ``torch.compile`` or ``torch.vmap``. The module copies
  with ``copy.deepcopy``; save its ``state_dict``.
"""

import operator
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from polypus import ConfigurationError, qml

try:
    import torch
    from torch import nn
    from torch.autograd.function import once_differentiable
except ImportError:
    # The ImportError of torch stays attached as the context.
    raise ImportError(
        'polypus.torch needs PyTorch: pip install "polypus-quantum[torch]"'
    )

__all__ = ["QuantumLayer", "from_trained"]

_InitialParams = Union[Sequence[float], npt.NDArray[np.float64], torch.Tensor]

# The role of a backend call in a seed's derivation.
_FORWARD = 0
_BACKWARD = 1


def _float64(tensor: torch.Tensor) -> npt.NDArray[np.float64]:
    """``tensor`` as a ``float64`` NumPy array on the CPU, out of the graph."""
    return tensor.detach().to("cpu", torch.float64).numpy()


class _Call:
    """What one forward pass and its backward pass need besides their
    tensors. It is not a tensor, so autograd ignores it."""

    def __init__(
        self,
        model: qml.Model,
        execution: dict[str, Any],
        seed: Optional[int],
        step: int,
    ):
        self.model = model
        self.execution = execution
        self.base_seed = seed
        self.step = step

    def seed(self, role: int) -> Optional[int]:
        """This call's seed for ``role``: ``None`` without a base seed, which
        lets the backend draw one."""
        if self.base_seed is None:
            return None
        sequence = np.random.SeedSequence([self.base_seed, self.step, role])
        return int(sequence.generate_state(1, dtype=np.uint64)[0])


class _Expectation(torch.autograd.Function):
    """``qml.expval`` forward, its vector-Jacobian product backward."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, params: torch.Tensor, call: _Call
    ) -> torch.Tensor:
        values = qml.expval(
            call.model,
            _float64(x),
            _float64(params),
            seed=call.seed(_FORWARD),
            **call.execution,
        )
        ctx.save_for_backward(x, params)
        ctx.call = call
        dtype = x.dtype if x.is_floating_point() else torch.float64
        return torch.from_numpy(values).to(dtype=dtype, device=x.device)

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        # Wrapped here rather than as a decorator: it is the same wrapper, and
        # the staticmethod keeps its own signature for the type checkers.
        return once_differentiable(_vector_jacobian_product)(ctx, grad)


def _vector_jacobian_product(
    ctx: Any, grad: torch.Tensor
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
    """The gradients of ``_Expectation``'s inputs from one ``qml.jacobian``
    call, ``None`` where no gradient is wanted."""
    x, params = ctx.saved_tensors
    call: _Call = ctx.call
    wants_x, wants_params = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
    rows, theta = _float64(x), _float64(params)
    kwargs = {"seed": call.seed(_BACKWARD), **call.execution}
    # Only what needs a gradient is asked for: an amplitude or basis
    # encoder then raises only when the input's gradient is wanted.
    by_params = by_inputs = None
    if wants_x and wants_params:
        by_params, by_inputs = qml.jacobian(
            call.model, rows, theta, wrt="both", **kwargs
        )
    elif wants_x:
        by_inputs = qml.jacobian(call.model, rows, theta, wrt="inputs", **kwargs)
    else:
        by_params = qml.jacobian(call.model, rows, theta, wrt="params", **kwargs)
    g = _float64(grad)
    grad_x = grad_params = None
    if by_inputs is not None:
        grad_x = torch.from_numpy(np.einsum("bo,bof->bf", g, by_inputs)).to(
            dtype=x.dtype, device=x.device
        )
    if by_params is not None:
        grad_params = torch.from_numpy(np.einsum("bo,bok->k", g, by_params)).to(
            dtype=params.dtype, device=params.device
        )
    return grad_x, grad_params, None


class QuantumLayer(nn.Module):
    """A ``polypus.qml.Model`` as a ``torch.nn.Module``: its parameters are the
    model's, ``params``, and its forward pass returns every readout
    observable's expectation.

    ``initial_params`` (``num_params`` finite values, checked as
    ``Model.bind`` checks them) sets the start; without it each parameter is
    drawn uniformly from ``[-π, π)`` with PyTorch's global generator, so
    ``torch.manual_seed`` makes it reproducible. ``seed`` (an integer >= 0)
    seeds every backend call (see the module's documentation). The other
    arguments are ``polypus.qml.expval``'s, with its defaults and rules
    (``exact`` needs ``infrastructure="local"`` with the native simulator, and
    ``qmio`` takes no ``seed``); they are checked by it, at the first forward
    pass, before any backend is created.
    """

    def __init__(
        self,
        model: qml.Model,
        *,
        initial_params: Optional[_InitialParams] = None,
        shots: int = 1024,
        infrastructure: str = "local",
        n_qpus: int = 1,
        nodes: int = 1,
        cores_per_qpu: int = 2,
        walltime: Optional[str] = None,
        options: Optional[dict[str, str]] = None,
        sim_method: str = "automatic",
        noise_model: Optional[Any] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        exact: bool = False,
    ):
        super().__init__()
        if isinstance(model, qml.TrainedModel):
            raise TypeError(
                "QuantumLayer takes a polypus.qml.Model; for a TrainedModel use "
                "polypus.torch.from_trained(trained)"
            )
        if not isinstance(model, qml.Model):
            raise TypeError(
                f"model must be a polypus.qml.Model, got {type(model).__name__}"
            )
        if seed is not None:
            if isinstance(seed, bool):
                raise TypeError(f"seed must be an integer >= 0 or None, got {seed!r}")
            try:
                seed = operator.index(seed)
            except TypeError:
                raise TypeError(
                    f"seed must be an integer >= 0 or None, got {seed!r}"
                ) from None
            if seed < 0:
                raise ConfigurationError(
                    f"seed must be an integer >= 0 or None, got {seed!r}"
                )
        if initial_params is None:
            start = torch.empty(model.num_params, dtype=torch.float64)
            start.uniform_(-torch.pi, torch.pi)
        else:
            if isinstance(initial_params, (torch.Tensor, np.ndarray)):
                values = initial_params.tolist()
            else:
                values = initial_params
            start = torch.tensor(model.bind(values).params, dtype=torch.float64)
        self.params = nn.Parameter(start)
        self._model = model
        self._seed = seed
        self._forwards = 0
        self._execution: dict[str, Any] = {
            "shots": shots,
            "infrastructure": infrastructure,
            "n_qpus": n_qpus,
            "nodes": nodes,
            "cores_per_qpu": cores_per_qpu,
            "walltime": walltime,
            "options": options,
            "sim_method": sim_method,
            "noise_model": noise_model,
            "backend": backend,
            "exact": exact,
        }

    @property
    def num_params(self) -> int:
        """The model's trainable parameters: the length of ``params``."""
        return self._model.num_params

    @property
    def num_observables(self) -> int:
        """The readout's observables: the width of the output."""
        return self._model.num_observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Every readout observable's expectation for each row of ``x``:
        ``(batch, features)`` gives ``(batch, num_observables)``, and one
        sample, 1-D, gives ``(num_observables,)``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() == 1:
            return self.forward(x.unsqueeze(0)).squeeze(0)
        if x.dim() != 2:
            raise ValueError(
                f"x must be (batch, features) or one sample (features,), got "
                f"shape {tuple(x.shape)}"
            )
        call = _Call(self._model, self._execution, self._seed, self._forwards)
        self._forwards += 1
        output: torch.Tensor = _Expectation.apply(x, self.params, call)
        return output

    def extra_repr(self) -> str:
        return f"num_params={self.num_params}, num_observables={self.num_observables}"


def from_trained(
    trained: qml.TrainedModel,
    *,
    shots: int = 1024,
    infrastructure: str = "local",
    n_qpus: int = 1,
    nodes: int = 1,
    cores_per_qpu: int = 2,
    walltime: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
    sim_method: str = "automatic",
    noise_model: Optional[Any] = None,
    backend: Optional[str] = None,
    seed: Optional[int] = None,
    exact: bool = False,
) -> QuantumLayer:
    """A :class:`QuantumLayer` that starts from a ``TrainedModel`` (from
    ``Model.fit`` and ``bind``, or ``qml.load``):
    ``QuantumLayer(trained.model, initial_params=trained.params, ...)``."""
    if isinstance(trained, qml.Model):
        raise TypeError(
            "from_trained takes a polypus.qml.TrainedModel; for a Model use "
            "polypus.torch.QuantumLayer(model)"
        )
    if not isinstance(trained, qml.TrainedModel):
        raise TypeError(
            f"trained must be a polypus.qml.TrainedModel, got {type(trained).__name__}"
        )
    return QuantumLayer(
        trained.model,
        initial_params=trained.params,
        shots=shots,
        infrastructure=infrastructure,
        n_qpus=n_qpus,
        nodes=nodes,
        cores_per_qpu=cores_per_qpu,
        walltime=walltime,
        options=options,
        sim_method=sim_method,
        noise_model=noise_model,
        backend=backend,
        seed=seed,
        exact=exact,
    )
