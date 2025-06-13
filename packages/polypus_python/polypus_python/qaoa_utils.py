from qiskit.circuit import QuantumCircuit, ParameterVector

def build_qaoa_circuit(graph, n_layers, cost_hamiltonian_layers, mixer_hamiltonian_layers) -> QuantumCircuit:
    """
    Build a generic QAOA circuit.
 
    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of QAOA layers (p).
        cost_hamiltonian_layers (list): List of callables, each applies cost Hamiltonian for a layer.
        mixer_hamiltonian_layers (list): List of callables, each applies mixer Hamiltonian for a layer.

    Returns:
        QuantumCircuit: The QAOA circuit with parameterized gates.

    # Usage:
    cost_layers = [maxcut_cost_layer(graph) for _ in range(p)]
    mixer_layers = [standard_mixer_layer(n_qubits) for _ in range(p)]
    qc = build_generic_qaoa_circuit(n_qubits, p, cost_layers, mixer_layers)
    """

    n_qubits = graph.number_of_nodes()

    qc = QuantumCircuit(n_qubits)
    
    # Parameter vector: first n_layers for gamma, then n_layers for beta
    gamma = ParameterVector('γ', n_layers)
    beta = ParameterVector('β', n_layers)

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

def expectation_value(counts, bitstring_to_obj):
    """
    Compute the expectation value from measurement counts and a user-defined objective function.

    Args:
        counts (dict): Measurement results as {bitstring: count}.
        bitstring_to_obj (callable): Function that takes a bitstring and returns its objective value.

    Returns:
        float: The expectation value.
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = bitstring_to_obj(bitstring)
        avg += obj * count
        sum_count += count
    if sum_count == 0:
        return 0.0
    return avg / sum_count

def expectation_values(array_counts, bitstring_to_obj):
    """
    Compute the expectation value from an array of measurement counts and a user-defined objective function.
    Args:
        array_counts (list): List of measurement results as [{bitstring: count}, ...].
        bitstring_to_obj (callable): Function that takes a bitstring and returns its objective value.
    Returns:
        float: The average expectation value across all counts.
    """

    expectation_values = []
    for counts in array_counts:
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = bitstring_to_obj(bitstring)
            avg += obj * count
            sum_count += count
        if sum_count == 0:
            expectation_values.append(0.0)
        else:
            expectation_values.append(avg / sum_count)
    
    return expectation_values

def assign_parameters(qc, params):
    """
    Assign parameters to the QAOA circuit.

    Args:
        qc (QuantumCircuit): The QAOA circuit.
        params (list): List of parameters to assign.

    Returns:
        QuantumCircuit: The QAOA circuit with assigned parameters.
    """
    param_dict = {}
    for i in range(len(params)):
        param_dict[qc.parameters[i]] = params[i]
    return qc.assign_parameters(param_dict)