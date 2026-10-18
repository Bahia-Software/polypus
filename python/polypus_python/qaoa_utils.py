from qiskit.circuit import ParameterVector, QuantumCircuit


def build_qaoa_circuit(
    graph, n_layers, cost_hamiltonian_layers, mixer_hamiltonian_layers
) -> QuantumCircuit:
    """
    Build a generic QAOA circuit.

    The circuit applies a Hadamard to every qubit, then ``n_layers`` QAOA layers
    (the cost layer ``k`` followed by the mixer layer ``k``), and ends with
    ``measure_all()``.

    Args:
        graph: Problem graph; any object with ``number_of_nodes()`` (e.g. a
            networkx graph). The circuit has one qubit per node, qubits
            ``0..n-1``.
        n_layers (int): Number of QAOA layers (p).
        cost_hamiltonian_layers (list): ``n_layers`` callables
            ``(qc, layer, param)``; entry ``k`` applies the cost Hamiltonian of
            layer ``k`` with ``param = γ[k]``.
        mixer_hamiltonian_layers (list): ``n_layers`` callables
            ``(qc, layer, param)``; entry ``k`` applies the mixer Hamiltonian of
            layer ``k`` with ``param = β[k]``.

    Returns:
        QuantumCircuit: The QAOA circuit with parameterized gates.

    Parameter order:
        The parameters are the vectors ``β`` (mixer) and ``γ`` (cost), each of
        length ``n_layers``. Qiskit sorts ``qc.parameters`` by name (vector
        elements by index), and ``"β"`` sorts before ``"γ"``, so the positional
        order is ``[β[0], ..., β[p-1], γ[0], ..., γ[p-1]]`` -- betas first.
        Every positional binding follows it: ``qc.assign_parameters(list)``,
        the vector ``polypus.train`` evaluates, and the ``best_params`` it
        returns.

        To start from γ/β schedules, pass ``x0 = betas + gammas`` and
        per-parameter bounds ``[(lo_b, hi_b)] * p + [(lo_g, hi_g)] * p``. To
        read ``best_params`` by name, use
        ``polypus_python.qaoa_params_to_dict(qc, best_params)``.

    Example:
        >>> from polypus_python.qaoa_utils import (
        ...     build_qaoa_circuit,
        ...     qaoa_params_to_dict,
        ... )
        >>> class Edge:  # stands in for a networkx graph
        ...     def number_of_nodes(self):
        ...         return 2
        >>> def cost(qc, layer, gamma):
        ...     qc.rzz(gamma, 0, 1)
        >>> def mixer(qc, layer, beta):
        ...     for q in range(qc.num_qubits):
        ...         qc.rx(2 * beta, q)
        >>> p = 2
        >>> qc = build_qaoa_circuit(Edge(), p, [cost] * p, [mixer] * p)
        >>> [param.name for param in qc.parameters]
        ['β[0]', 'β[1]', 'γ[0]', 'γ[1]']
        >>> gammas, betas = [0.1, 0.2], [0.7, 0.8]
        >>> bound = qc.assign_parameters(betas + gammas)  # not gammas + betas
        >>> bound.num_parameters
        0
        >>> qaoa_params_to_dict(qc, betas + gammas)
        {'β[0]': 0.7, 'β[1]': 0.8, 'γ[0]': 0.1, 'γ[1]': 0.2}
    """

    n_qubits = graph.number_of_nodes()

    qc = QuantumCircuit(n_qubits)

    # One γ (cost) and one β (mixer) per layer; see "Parameter order" above
    gamma = ParameterVector("γ", n_layers)
    beta = ParameterVector("β", n_layers)

    # Initial state: Hadamard on all qubits
    for i in range(n_qubits):
        qc.h(i)

    # QAOA layers
    for layer in range(n_layers):
        # Apply cost Hamiltonian for this layer
        cost_hamiltonian_layers[layer](qc, layer, gamma[layer])
        # Apply mixer Hamiltonian for this layer
        mixer_hamiltonian_layers[layer](qc, layer, beta[layer])

    qc.measure_all()
    return qc


def qaoa_params_to_dict(qc, params) -> dict[str, float]:
    """
    Map a positional parameter vector to the circuit's parameter names.

    ``params`` is read in the order of ``qc.parameters`` -- the order every
    positional binding uses, including ``polypus.train``'s ``best_params``. For
    a ``build_qaoa_circuit`` circuit that is ``β[0..p-1]`` then ``γ[0..p-1]``;
    the order is taken from ``qc``, not assumed.

    Args:
        qc (QuantumCircuit): The parameterized circuit.
        params (sequence of float): One value per entry of ``qc.parameters``
            (list, tuple or numpy array).

    Returns:
        dict[str, float]: ``{name: value}`` in ``qc.parameters`` order. It can
        be bound directly with ``qc.assign_parameters(...)``.

    Raises:
        ValueError: If ``len(params) != len(qc.parameters)``.

    Example:
        >>> from qiskit.circuit import ParameterVector, QuantumCircuit
        >>> from polypus_python.qaoa_utils import qaoa_params_to_dict
        >>> gamma, beta = ParameterVector("γ", 1), ParameterVector("β", 1)
        >>> qc = QuantumCircuit(1)
        >>> _ = qc.rz(gamma[0], 0)
        >>> _ = qc.rx(2 * beta[0], 0)
        >>> qaoa_params_to_dict(qc, [0.7, 0.1])
        {'β[0]': 0.7, 'γ[0]': 0.1}
    """
    parameters = qc.parameters
    if len(params) != len(parameters):
        raise ValueError(
            f"expected {len(parameters)} parameter values (one per "
            f"qc.parameters entry), got {len(params)}"
        )
    return {p.name: float(v) for p, v in zip(parameters, params)}


def _expectation(counts, bitstring_to_obj, context):
    # Shared by both helpers. ``context`` prefixes the error message (the
    # element index for ``expectation_values``), so the error is raised with
    # its final text and nothing has to catch it -- catching would also catch,
    # and wrap, a ValueError raised by ``bitstring_to_obj``.
    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            f"{context}counts sum to zero ({len(counts)} bitstring(s)): there "
            "are no measurement outcomes to average, so no expectation value "
            "exists (did the backend return an empty result?)"
        )
    avg = 0
    for bitstring, count in counts.items():
        avg += bitstring_to_obj(bitstring) * count
    return avg / total


def expectation_value(counts, bitstring_to_obj):
    """
    Compute the expectation value from measurement counts and a user-defined objective function.

    Unlike the Rust ``CostObservable`` reducers, which yield ``0.0`` for counts
    that sum to zero (their historical contract), this helper raises: it can be
    handed the empty counts of a failed backend run directly, and ``0.0`` would
    look like an excellent cost to a minimizer. In the native evaluation path
    such results are rejected earlier, by the Planner's
    ``validate_run_results``, before any reducer runs.

    Args:
        counts (dict): Measurement results as {bitstring: count}.
        bitstring_to_obj (callable): Function that takes a bitstring and returns its objective value.

    Returns:
        float: The expectation value.

    Raises:
        ValueError: If ``counts`` is empty or its counts sum to zero. It is
            raised before ``bitstring_to_obj`` is called; an exception raised
            by ``bitstring_to_obj`` itself propagates unchanged.

    Example:
        >>> from polypus_python.qaoa_utils import expectation_value
        >>> expectation_value({"00": 3, "11": 1}, lambda b: int(b, 2))
        0.75
        >>> expectation_value({}, lambda b: int(b, 2))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: counts sum to zero ...
    """
    return _expectation(counts, bitstring_to_obj, "")


def expectation_values(array_counts, bitstring_to_obj):
    """
    Compute one expectation value per entry of ``array_counts`` with a user-defined objective function.

    Each entry is reduced exactly as ``expectation_value`` does, so the same
    rule applies: counts that sum to zero raise instead of yielding ``0.0``
    (see ``expectation_value`` for why this differs from the Rust reducers).

    Args:
        array_counts (list): List of measurement results as [{bitstring: count}, ...].
        bitstring_to_obj (callable): Function that takes a bitstring and returns its objective value.

    Returns:
        list[float]: One expectation value per entry of ``array_counts``, in
        the same order (``[]`` for an empty list).

    Raises:
        ValueError: If any entry is empty or its counts sum to zero; the
            message names the index of the first such entry. An exception
            raised by ``bitstring_to_obj`` propagates unchanged.
    """
    return [
        _expectation(counts, bitstring_to_obj, f"array_counts[{index}]: ")
        for index, counts in enumerate(array_counts)
    ]


def assign_parameters(qc, params):
    """
    Assign parameters to the QAOA circuit.

    Binding is positional: ``params[i]`` goes to ``qc.parameters[i]``, i.e.
    ``β[0..p-1]`` then ``γ[0..p-1]`` for a ``build_qaoa_circuit`` circuit (see
    its "Parameter order"). To bind by name, use ``qaoa_params_to_dict``.

    Args:
        qc (QuantumCircuit): The QAOA circuit.
        params (list): Parameter values, in ``qc.parameters`` order.

    Returns:
        QuantumCircuit: The QAOA circuit with assigned parameters.
    """
    param_dict = {}
    for i in range(len(params)):
        param_dict[qc.parameters[i]] = params[i]
    return qc.assign_parameters(param_dict)
