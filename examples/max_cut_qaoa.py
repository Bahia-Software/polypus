import numpy as np
import networkx as nx
import polypus
import matplotlib.pyplot as plt
import itertools
import time
from polypus_python.qaoa_utils import build_qaoa_circuit, expectation_value
from qiskit_aer import AerSimulator

# Auxiliar functions for QAOA
def maxcut_cost_layer(graph):
    def layer_fn(qc, layer, gamma):
        for i, j in graph.edges():
            qc.cx(i, j)
            qc.rz(-gamma, j)
            qc.cx(i, j)
    return layer_fn

def standard_mixer_layer(n_qubits):
    def layer_fn(qc, layer, beta):
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    return layer_fn

def bitstring_to_obj(bitstring):
    bitstring = bitstring[::-1]
    cut = 0
    for i, j in graph.edges():
        if bitstring[i] != bitstring[j]:
            cut += 1
    return cut

def maxcut_bruteforce(graph):
    n = graph.number_of_nodes()
    max_cut = 0
    for bits in itertools.product([0, 1], repeat=n):
        cut = 0
        for i, j in graph.edges():
            if bits[i] != bits[j]:
                cut += 1
        if cut > max_cut:
            max_cut = cut
    return max_cut

def qaoa_expectation(params):
    qc_bound = qc.assign_parameters(params)
    backend = AerSimulator()
    result = backend.run(qc_bound, shots=10**4).result()
    counts = result.get_counts()
    return -expectation_value(counts, bitstring_to_obj) 

# Experiment setup
np.random.seed(1) 
graph = nx.gnp_random_graph(7, 0.5)
n_qubits = graph.number_of_nodes()
layers = 2
cost_layers = [maxcut_cost_layer(graph) for _ in range(layers)]
mixer_layers = [standard_mixer_layer(graph.number_of_nodes()) for _ in range(layers)]
qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
N_SHOTS = 1000
N_QPUS = 2
POPULATION_SIZE = 10
MAX_GENERATIONS = 4

# Run Polypus
tic = time.time()
result = polypus.run_quantum_circuit(qc, shots=N_SHOTS, n_qpus = N_QPUS, expectation_function=bitstring_to_obj,  generations=MAX_GENERATIONS, population_size=POPULATION_SIZE, dimensions=2*layers, method='differential_evolution')
polypus_time = time.time() - tic

# Polypus solution
backend = AerSimulator()
qc.assign_parameters(result, inplace=True)
result_poly = backend.run(qc, shots=10**4).result()
counts_poly = result_poly.get_counts()
best_bitstring_poly = max(counts_poly, key=counts_poly.get)[::-1]
cut_poly = bitstring_to_obj(best_bitstring_poly)

# Plots
pos = nx.spring_layout(graph, seed=42)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Original graph
nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color='lightgray', edge_color='black', node_size=800, font_color='black')
axes[0].set_title("Original Graph")

# 2. Polypus 
color_map_poly = ['red' if best_bitstring_poly[i] == '0' else 'blue' for i in range(n_qubits)]
nx.draw(graph, pos, ax=axes[1], with_labels=True,node_color=color_map_poly, node_size=800, font_color='white')
cut_edges_poly = [(i, j) for i, j in graph.edges() if best_bitstring_poly[i] != best_bitstring_poly[j]]
nx.draw_networkx_edges(
    graph, pos, ax=axes[1], edgelist=cut_edges_poly,
    width=4, edge_color='yellow'
)
axes[1].set_title(f"Polypus QAOA Cut: {cut_poly} - Time: {polypus_time:.2f}s")

plt.suptitle(f"Expected MaxCut value: {maxcut_bruteforce(graph)} - Runned DE with {POPULATION_SIZE} individuals for {MAX_GENERATIONS} generations for a QAOA with {layers} layers")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("examples/maxcut_result.png")
plt.show()