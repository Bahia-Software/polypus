import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import argparse
import os
import csv
from polypus_python.qaoa_utils import build_qaoa_circuit, expectation_value
from qiskit_aer import AerSimulator
from scipy.optimize import differential_evolution
from multiprocessing import Pool
import math

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
    result = backend.run(qc_bound, shots=10**3).result()
    counts = result.get_counts()
    return -expectation_value(counts, bitstring_to_obj) 

def compute_mean_energy(counts):
    total_energy = 0
    total_shots = sum(counts.values())
    for bitstring, frequency in counts.items():
        energy = bitstring_to_obj(bitstring[::-1])  # reverse if needed
        total_energy += energy * frequency
    return total_energy / total_shots

def log_experiment_results(n_qubits, method, execution_time, approximation_ratio, n_shots):
    log_dir = os.path.join(os.getcwd(), "examples")
    file_name = os.path.join(log_dir, 'qaoa_results.csv')
    headers = ['n_qubits','method', 'time', 'approximation_ratio', 'n_shots']
    # Check if the file exists
    file_exists = os.path.isfile(file_name)

    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(headers)
        # Write the experimental results
        writer.writerow([n_qubits, method, execution_time, approximation_ratio, n_shots])


if __name__ == "__main__":

    # Experiment setup
    np.random.seed(1) 
    N_SHOTS = 10000
    N_QPUS = 64

    # Select method
    parser = argparse.ArgumentParser(description="Run QAOA for MaxCut")
    parser.add_argument('--method', type=str, required=True, help='Method to use (scipy, polypus_local, polypus_cunqa)') 
    parser.add_argument('--qubits', type=int, default=4, help='Number of qubits for the QAOA circuit')
    args = parser.parse_args()
    method = args.method
    n_qubits = args.qubits
    GENERATIONS_FACTOR = 40
    POPULATION_FACTOR = 4
    TOL = 0.00001
    print(f"Selected method: {method}")

    graph = nx.gnp_random_graph(n_qubits, 0.5, seed=7)
    BEST_SOLUTION = maxcut_bruteforce(graph)
    n_qubits = graph.number_of_nodes()
    layers = n_qubits // 2
    cost_layers = [maxcut_cost_layer(graph) for _ in range(layers)]
    mixer_layers = [standard_mixer_layer(graph.number_of_nodes()) for _ in range(layers)]
    qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
    POPULATION_SIZE = (2 * layers) * POPULATION_FACTOR
    MAX_GENERATIONS = n_qubits * (GENERATIONS_FACTOR)
    qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
    id = f"{n_qubits}_{method}"

    if method not in ['scipy', 'polypus_local', 'polypus_cunqa']:
        raise ValueError("Method must be one of: 'scipy', 'polypus_local', 'polypus_cunqa'")

    if method == 'scipy':
    
        # Run with Scipy     
        bounds = [(0, np.pi)] * (2 * layers)
        print("Running differential evolution with scipy... optimized for parallel execution")
        total_workers = int(os.environ["OMP_NUM_THREADS"]) if "OMP_NUM_THREADS" in os.environ else 1
        print(f"Using {total_workers} workers for parallel execution")
        pool = Pool(total_workers)
        tic = time.time()
        res = differential_evolution(qaoa_expectation, bounds, maxiter=MAX_GENERATIONS, polish=False, popsize=POPULATION_SIZE, disp=True, workers=pool.map, tol=TOL)
        time_scipy = time.time() - tic
        print(f"Scipy DE finished in {time_scipy:.2f} seconds with best cost: {-res.fun:.2f}")
        def get_best_bitstring(counts):
            return max(counts, key=counts.get)
        gamma = res.x[:layers]
        beta = res.x[layers:]
        param_dict = {}
        for i in range(layers):
            param_dict[qc.parameters[i]] = gamma[i]
            param_dict[qc.parameters[layers + i]] = beta[i]
        qc_bound = qc.assign_parameters(param_dict)
        backend = AerSimulator()
        result = backend.run(qc_bound, shots=10**4).result()
        counts = result.get_counts()
        best_bitstring_scipy = get_best_bitstring(counts)
        cut_scipy = bitstring_to_obj(best_bitstring_scipy)
        mean_energy = compute_mean_energy(counts)
        approximation_ratio_scipy = mean_energy / BEST_SOLUTION
        print(f"Mean energy: {mean_energy}")
        print(f"Approximation ratio: {approximation_ratio_scipy}")
        pool.close()
        pool.join()

        log_experiment_results(n_qubits, 'scipy_optimized', time_scipy, approximation_ratio_scipy, N_SHOTS)

        # Plots
        pos = nx.spring_layout(graph, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Original graph
        nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color='lightgray', edge_color='black', node_size=800, font_color='black')
        axes[0].set_title("Original Graph")

        # 2. Polypus 
        color_map_poly = ['red' if best_bitstring_scipy[i] == '0' else 'blue' for i in range(n_qubits)]
        nx.draw(graph, pos, ax=axes[1], with_labels=True,node_color=color_map_poly, node_size=800, font_color='white')
        cut_edges_poly = [(i, j) for i, j in graph.edges() if best_bitstring_scipy[i] != best_bitstring_scipy[j]]
        nx.draw_networkx_edges(
            graph, pos, ax=axes[1], edgelist=cut_edges_poly,
            width=4, edge_color='yellow'
        )
        axes[1].set_title(f"Scipy Optimized QAOA Cut: {cut_scipy} - Time: {time_scipy:.2f}s")

        # 3. Counts distribution
        axes[2].bar(counts.keys(), counts.values(), color='blue')
        axes[2].set_xlabel('Bitstring')
        axes[2].set_ylabel('Counts')
        axes[2].set_title('Counts Distribution')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)

        plt.suptitle(f"Expected MaxCut value: {BEST_SOLUTION} - Runned DE with {POPULATION_SIZE} individuals for {MAX_GENERATIONS} generations for a QAOA with {layers} layers")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"examples/maxcut_result_{id}.png")

    elif method == 'polypus_local':

        import polypus
        # Qraise
        N_QPUS = POPULATION_SIZE
        if N_QPUS > 64:
            N_QPUS = 64

        N = math.ceil(N_QPUS / 32)

        # Run Polypus with cunqa infrastructure
        tic = time.time()
        result_cunqa = polypus.differential_evolution(
            qc=qc, 
            shots=N_SHOTS, 
            n_qpus = N_QPUS, 
            expectation_function=bitstring_to_obj, 
            generations=MAX_GENERATIONS, 
            population_size=POPULATION_SIZE, 
            dimensions=2*layers, 
            infrastructure="local", 
            id=id,
            tolerance=TOL,
            nodes=N)
        polypus_time_cunqa = time.time() - tic

        qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
        backend = AerSimulator()
        qc.assign_parameters(result_cunqa, inplace=True)
        result_poly = backend.run(qc, shots=N_SHOTS).result()
        counts_poly = result_poly.get_counts()

        best_bitstring_poly = max(counts_poly, key=counts_poly.get)
        cut_poly = bitstring_to_obj(best_bitstring_poly)

        # Plots
        pos = nx.spring_layout(graph, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
        axes[1].set_title(f"Polypus QAOA Cut: {cut_poly} - Time: {polypus_time_cunqa:.2f}s")

        # 3. Counts distribution
        axes[2].bar(counts_poly.keys(), counts_poly.values(), color='blue')
        axes[2].set_xlabel('Bitstring')
        axes[2].set_ylabel('Counts')
        axes[2].set_title('Counts Distribution')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)

        plt.suptitle(f"Expected MaxCut value: {BEST_SOLUTION} - Runned DE with {POPULATION_SIZE} individuals for {MAX_GENERATIONS} generations for a QAOA with {layers} layers")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"examples/maxcut_result_{id}.png")

        mean_energy = compute_mean_energy(counts_poly)
        approximation_ratio_polypus_cunqa = mean_energy / BEST_SOLUTION
        log_experiment_results(n_qubits, 'polypus_local', polypus_time_cunqa, approximation_ratio_polypus_cunqa, N_SHOTS)

    elif method == 'polypus_cunqa':

        import polypus
        # Qraise
        N_QPUS = POPULATION_SIZE
        if N_QPUS > 64:
            N_QPUS = 64

        N = math.ceil(N_QPUS / 32)

        # Run Polypus with cunqa infrastructure
        tic = time.time()
        result_cunqa = polypus.differential_evolution(
            qc=qc, 
            shots=N_SHOTS, 
            n_qpus = N_QPUS, 
            expectation_function=bitstring_to_obj, 
            generations=MAX_GENERATIONS, 
            population_size=POPULATION_SIZE, 
            dimensions=2*layers, 
            infrastructure="qmio", 
            id=id,
            tolerance=TOL,
            nodes=N)
        polypus_time_cunqa = time.time() - tic

        qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
        backend = AerSimulator()
        qc.assign_parameters(result_cunqa, inplace=True)
        result_poly = backend.run(qc, shots=N_SHOTS).result()
        counts_poly = result_poly.get_counts()

        best_bitstring_poly = max(counts_poly, key=counts_poly.get)
        cut_poly = bitstring_to_obj(best_bitstring_poly)

        # Plots
        pos = nx.spring_layout(graph, seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
        axes[1].set_title(f"Polypus QAOA Cut: {cut_poly} - Time: {polypus_time_cunqa:.2f}s")

        # 3. Counts distribution
        axes[2].bar(counts_poly.keys(), counts_poly.values(), color='blue')
        axes[2].set_xlabel('Bitstring')
        axes[2].set_ylabel('Counts')
        axes[2].set_title('Counts Distribution')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)

        plt.suptitle(f"Expected MaxCut value: {BEST_SOLUTION} - Runned DE with {POPULATION_SIZE} individuals for {MAX_GENERATIONS} generations for a QAOA with {layers} layers")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"examples/maxcut_result_{id}.png")

        mean_energy = compute_mean_energy(counts_poly)
        approximation_ratio_polypus_cunqa = mean_energy / BEST_SOLUTION
        log_experiment_results(n_qubits, 'polypus_cunqa', polypus_time_cunqa, approximation_ratio_polypus_cunqa, N_SHOTS)