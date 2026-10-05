"""
Tests for ``polypus_python.qaoa_utils`` (issue #217).

They fix the *documented* parameter order of ``build_qaoa_circuit`` — Qiskit
sorts ``qc.parameters`` alphabetically, so the β vector comes before γ — and the
name→value helper ``qaoa_params_to_dict`` used to read ``best_params`` without
depending on that order. The module doctests run here too, so the docstring
examples cannot rot again.
"""

import doctest
import inspect

import numpy as np
import pytest
from polypus_python import qaoa_utils
from polypus_python.qaoa_utils import build_qaoa_circuit


class _Graph:
    """Minimal stand-in for a networkx graph: only ``number_of_nodes()``."""

    def __init__(self, n):
        self._n = n

    def number_of_nodes(self):
        return self._n


def _cost_layer(qc, layer, gamma):
    qc.rz(gamma, 0)


def _mixer_layer(qc, layer, beta):
    qc.rx(2 * beta, 0)


def _circuit(p, n_qubits=2):
    return build_qaoa_circuit(
        _Graph(n_qubits), p, [_cost_layer] * p, [_mixer_layer] * p
    )


def _angles(qc, gate_name):
    return [
        float(inst.operation.params[0])
        for inst in qc.data
        if inst.operation.name == gate_name
    ]


@pytest.mark.parametrize("p", [2, 3])
def test_parameter_order_is_betas_then_gammas(p):
    qc = _circuit(p)
    names = [q.name for q in qc.parameters]
    assert len(names) == 2 * p
    assert names == [f"β[{k}]" for k in range(p)] + [f"γ[{k}]" for k in range(p)]


def test_documented_recipe_binds_gamma_and_beta_correctly():
    p = 3
    qc = _circuit(p)
    gammas = [0.1, 0.2, 0.3]
    betas = [1.1, 1.2, 1.3]

    bound = qc.assign_parameters(betas + gammas)
    # Each cost layer gets its γ[k], each mixer layer its β[k].
    assert _angles(bound, "rz") == pytest.approx(gammas)
    assert _angles(bound, "rx") == pytest.approx([2 * b for b in betas])

    # The order the old docstring promised (γ first) swaps them silently.
    naive = qc.assign_parameters(gammas + betas)
    assert _angles(naive, "rz") != pytest.approx(gammas)
    assert _angles(naive, "rz") == pytest.approx(betas)


def test_qaoa_params_to_dict_maps_names_to_values():
    from polypus_python.qaoa_utils import qaoa_params_to_dict

    qc = _circuit(2)
    params = [0.5, 0.6, 0.7, 0.8]
    d = qaoa_params_to_dict(qc, params)

    assert list(d) == [q.name for q in qc.parameters]
    assert d == {"β[0]": 0.5, "β[1]": 0.6, "γ[0]": 0.7, "γ[1]": 0.8}
    assert all(type(v) is float for v in d.values())
    assert qc.assign_parameters(d) == qc.assign_parameters(params)


@pytest.mark.parametrize("n_params", [3, 5])
def test_qaoa_params_to_dict_rejects_wrong_length(n_params):
    from polypus_python.qaoa_utils import qaoa_params_to_dict

    qc = _circuit(2)
    with pytest.raises(ValueError) as excinfo:
        qaoa_params_to_dict(qc, [0.0] * n_params)
    message = str(excinfo.value)
    assert str(n_params) in message
    assert str(len(qc.parameters)) in message


@pytest.mark.parametrize(
    "params",
    [
        [0.5, 0.6, 0.7, 0.8],
        (0.5, 0.6, 0.7, 0.8),
        np.array([0.5, 0.6, 0.7, 0.8]),
    ],
    ids=["list", "tuple", "ndarray"],
)
def test_qaoa_params_to_dict_accepts_sequences_and_arrays(params):
    from polypus_python.qaoa_utils import qaoa_params_to_dict

    d = qaoa_params_to_dict(_circuit(2), params)
    assert d == {"β[0]": 0.5, "β[1]": 0.6, "γ[0]": 0.7, "γ[1]": 0.8}
    assert all(type(v) is float for v in d.values())


def test_module_doctests_pass():
    result = doctest.testmod(qaoa_utils)
    assert result.failed == 0
    assert result.attempted > 0


def test_build_qaoa_circuit_signature_has_no_n_qubits():
    assert list(inspect.signature(build_qaoa_circuit).parameters) == [
        "graph",
        "n_layers",
        "cost_hamiltonian_layers",
        "mixer_hamiltonian_layers",
    ]


def test_qaoa_params_to_dict_is_exported_from_package():
    import polypus_python

    assert polypus_python.qaoa_params_to_dict is qaoa_utils.qaoa_params_to_dict
