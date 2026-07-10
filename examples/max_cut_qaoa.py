import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import argparse
import os
import csv
import polypus
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

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

# ──────────────────────────────────────────────────────────────────────────────
# QNG helpers: alternating-parameter circuit and variance function
# ──────────────────────────────────────────────────────────────────────────────

def build_qaoa_circuit_qng(graph, layers):
    """
    Build a QAOA circuit for MaxCut using a **single** ParameterVector with
    alternating gamma/beta layout:
        θ[2k]   = gamma_k  (ZZ cost layer k)
        θ[2k+1] = beta_k   (X mixer layer k)

    This ordering is required by the QNG variance function, which needs to
    construct partial circuits up to parameter index `a`.
    The circuit is compatible with `assign_parameters(list)` used by Polypus.
    """
    n_qubits = graph.number_of_nodes()
    edges = list(graph.edges())
    theta = ParameterVector('\u03b8', 2 * layers)  # θ
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for layer in range(layers):
        gamma = theta[2 * layer]
        beta  = theta[2 * layer + 1]
        # Cost layer (ZZ Hamiltonian)
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(-gamma, j)
            qc.cx(i, j)
        # Mixer layer (X Hamiltonian)
        for qubit in range(n_qubits):
            qc.rx(2 * beta, qubit)
    qc.measure_all()
    return qc


def make_variance_function(graph, n_qubits, n_shots):
    """
    Return a callable  variance_fn(theta, a) -> float  that estimates the
    diagonal QFIM element for parameter index `a` at the current point `theta`.

    The parameter layout mirrors `build_qaoa_circuit_qng`:
        even a → gamma (cost, ZZ Hamiltonian) → measure in Z basis
        odd  a → beta  (mixer, X Hamiltonian) → rotate into X basis before measuring

    For a == 0 (no gates applied yet) the initial state |+>^n has zero
    variance under both ZZ and X Hamiltonians, so we return 0.0 and rely on
    the Tikhonov regularisation inside the Rust optimiser.
    """
    edges = list(graph.edges())
    backend = AerSimulator()

    def variance_fn(theta, a):
        if a == 0:
            return 0.0

        theta_a = ParameterVector('\u03b8', a)  # θ — same name, different circuit
        qc_inter = QuantumCircuit(n_qubits)
        qc_inter.h(range(n_qubits))

        for k in range(a):
            if k % 2 == 0:      # gamma (ZZ cost layer)
                gamma = theta_a[k]
                for i, j in edges:
                    qc_inter.cx(i, j)
                    qc_inter.rz(-gamma, j)
                    qc_inter.cx(i, j)
            else:               # beta (X mixer layer)
                beta = theta_a[k]
                for qubit in range(n_qubits):
                    qc_inter.rx(2 * beta, qubit)

        # Rotate into the correct measurement basis
        if a % 2 == 0:          # next param is gamma -> ZZ Hamiltonian -> Z basis
            qc_inter.measure_all()
        else:                   # next param is beta  -> X Hamiltonian  -> X basis
            qc_inter.h(range(n_qubits))
            qc_inter.measure_all()

        params = list(theta[:a])
        qc_bound = qc_inter.assign_parameters(params)
        result = backend.run(qc_bound, shots=n_shots).result()
        counts = result.get_counts()
        shots_total = sum(counts.values())

        exp_H = 0.0
        exp_H2 = 0.0
        for bitstring, count in counts.items():
            prob = count / shots_total
            bits = bitstring[::-1]
            if a % 2 == 0:      # ZZ Hamiltonian
                h_val = sum(
                    (1 if bits[i] == '0' else -1) * (1 if bits[j] == '0' else -1)
                    for i, j in edges
                )
            else:               # X Hamiltonian
                h_val = sum(1 if bits[i] == '0' else -1 for i in range(n_qubits))
            exp_H  += h_val * prob
            exp_H2 += (h_val ** 2) * prob

        return exp_H2 - exp_H ** 2

    return variance_fn

def qaoa_expectation(params):
    qc_bound = qc.assign_parameters(params)
    backend = AerSimulator()
    result = backend.run(qc_bound, shots=10**4).result()
    counts = result.get_counts()
    return -expectation_value(counts, bitstring_to_obj)

def compute_mean_energy(counts):
    total_shots = sum(counts.values())
    return sum(bitstring_to_obj(bs[::-1]) * freq for bs, freq in counts.items()) / total_shots

def log_experiment_results(n_qubits, method, execution_time, approximation_ratio, n_shots, n_nodes, cores_per_qpu):
    log_dir = os.path.join(os.getcwd(), "examples")
    os.makedirs(log_dir, exist_ok=True)
    file_name = os.path.join(log_dir, 'qaoa_results.csv')
    headers = ['n_qubits', 'method', 'time', 'approximation_ratio', 'n_shots', 'n_nodes', 'cores_per_qpu']
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([n_qubits, method, execution_time, approximation_ratio, n_shots, n_nodes, cores_per_qpu])

def plot_results(graph, counts, best_bitstring, cut, method_label, elapsed,
                 best_solution, population_size, max_generations, layers, algorithm, experiment_id):
    n = graph.number_of_nodes()
    pos = nx.spring_layout(graph, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original graph
    nx.draw(graph, pos, ax=axes[0], with_labels=True, node_color='lightgray',
            edge_color='black', node_size=800, font_color='black')
    axes[0].set_title("Original Graph")

    # 2. Best cut
    color_map = ['red' if best_bitstring[i] == '0' else 'blue' for i in range(n)]
    nx.draw(graph, pos, ax=axes[1], with_labels=True, node_color=color_map,
            node_size=800, font_color='white')
    cut_edges = [(i, j) for i, j in graph.edges() if best_bitstring[i] != best_bitstring[j]]
    nx.draw_networkx_edges(graph, pos, ax=axes[1], edgelist=cut_edges, width=4, edge_color='yellow')
    axes[1].set_title(f"{method_label} Cut: {cut} - Time: {elapsed:.2f}s")

    # 3. Counts distribution
    axes[2].bar(counts.keys(), counts.values(), color='blue')
    axes[2].set_xlabel('Bitstring')
    axes[2].set_ylabel('Counts')
    axes[2].set_title('Counts Distribution')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)

    plt.suptitle(
        f"Expected MaxCut value: {best_solution} - "
        f"Runned {algorithm} with {population_size} individuals for {max_generations} "
        f"generations for a QAOA with {layers} layers"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("examples/plots_maxcut", exist_ok=True)
    plt.savefig(f"examples/plots_maxcut/maxcut_result_{experiment_id}.png")
    plt.close(fig)

def final_evaluation(qc, graph, layers, cost_layers, mixer_layers, params, n_shots, best_solution):
    qc_eval = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
    qc_eval.assign_parameters(params, inplace=True)
    counts = AerSimulator().run(qc_eval, shots=n_shots).result().get_counts()
    best_bitstring = max(counts, key=counts.get)
    cut = bitstring_to_obj(best_bitstring)
    approximation_ratio = compute_mean_energy(counts) / best_solution
    return counts, best_bitstring, cut, approximation_ratio


if __name__ == "__main__":

    polypus.init_logger(run_name="max_cut_qaoa")

    # Experiment setup
    np.random.seed(1) 
    N_SHOTS = 10000
    N_QPUS = 64

    # Select method
    parser = argparse.ArgumentParser(description="Run QAOA for MaxCut")
    parser.add_argument('--method', type=str, required=True, help='Method to use (scipy, polypus_local, polypus_cunqa)') 
    parser.add_argument('--qubits', type=int, default=4, help='Number of qubits for the QAOA circuit')
    parser.add_argument('--experiment_id', type=int, default=0, help='Experiment ID for logging purposes')
    parser.add_argument('--run', type=int, default=0, help='Run number for logging purposes')
    parser.add_argument('--cores_per_qpu', type=int, default= 2, help='Number of cores per QPU for Cunqa infrastructure')
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of nodes in CUNQA")
    args = parser.parse_args()
    method = args.method
    n_qubits = args.qubits
    experiment_id = args.experiment_id
    run = args.run
    CORES_PER_QPU = args.cores_per_qpu
    NUM_NODES = args.num_nodes
    GENERATIONS_FACTOR = 5
    POPULATION_FACTOR = 4
    LAYERS_FACTOR = 2
    TOL = 0.00001
    print(f"Selected method: {method}")

    graph = nx.gnp_random_graph(n_qubits, 0.6, seed=7)
    BEST_SOLUTION = maxcut_bruteforce(graph)
    n_qubits = graph.number_of_nodes()
    layers = n_qubits // 2
    layers *= LAYERS_FACTOR
    cost_layers = [maxcut_cost_layer(graph) for _ in range(layers)]
    mixer_layers = [standard_mixer_layer(graph.number_of_nodes()) for _ in range(layers)]
    qc = build_qaoa_circuit(graph, layers, cost_layers, mixer_layers)
    POPULATION_SIZE = (2 * layers) * POPULATION_FACTOR
    MAX_GENERATIONS = n_qubits * GENERATIONS_FACTOR
    id = f"{n_qubits}_{method}_{experiment_id}_{NUM_NODES}_{CORES_PER_QPU}_{run}"

    N_QPUS_PER_NODE = 64 // CORES_PER_QPU
    N_QPUS = NUM_NODES * N_QPUS_PER_NODE
    print(f"Using {N_QPUS} QPUs across {NUM_NODES} nodes with {CORES_PER_QPU} cores per QPU")

    if method not in ['scipy', 'polypus_local', 'polypus_cunqa',
                       'polypus_local_pso', 'polypus_cunqa_pso',
                       'polypus_local_qng', 'polypus_cunqa_qng']:
        raise ValueError("Method must be one of: 'scipy', 'polypus_local', 'polypus_cunqa', 'polypus_local_pso', 'polypus_cunqa_pso', 'polypus_local_qng', 'polypus_cunqa_qng'")

    if method == 'scipy':

        bounds = [(0, np.pi)] * (2 * layers)
        print("Running differential evolution with scipy... optimized for parallel execution")
        total_workers = int(os.environ["OMP_NUM_THREADS"]) if "OMP_NUM_THREADS" in os.environ else 1
        print(f"Using {total_workers} workers for parallel execution")
        pool = Pool(total_workers)
        tic = time.time()
        res = differential_evolution(qaoa_expectation, bounds, maxiter=MAX_GENERATIONS, polish=False, popsize=max(1, POPULATION_SIZE // (2 * layers)), disp=True, workers=pool.map, tol=TOL)
        time_scipy = time.time() - tic
        print(f"Scipy DE finished in {time_scipy:.2f} seconds with best cost: {-res.fun:.2f}")
        pool.close()
        pool.join()

        param_dict = {qc.parameters[i]: res.x[i] for i in range(2 * layers)}
        counts = AerSimulator().run(qc.assign_parameters(param_dict), shots=N_SHOTS).result().get_counts()
        best_bitstring = max(counts, key=counts.get)
        cut = bitstring_to_obj(best_bitstring)
        approximation_ratio = compute_mean_energy(counts) / BEST_SOLUTION
        print(f"Mean energy: {approximation_ratio * BEST_SOLUTION:.4f}")
        print(f"Approximation ratio: {approximation_ratio}")

        log_experiment_results(n_qubits, 'scipy_optimized', time_scipy, approximation_ratio, N_SHOTS, NUM_NODES, CORES_PER_QPU)
        plot_results(graph, counts, best_bitstring, cut, "Scipy Optimized QAOA", time_scipy,
                     BEST_SOLUTION, POPULATION_SIZE, MAX_GENERATIONS, layers, "DE", id)

    elif method in ('polypus_local', 'polypus_cunqa'):

        infrastructure = "local" if method == 'polypus_local' else "cunqa"
        # MPS gives per-shot QPU parallelism on distributed backends; for local
        # runs the automatic method (statevector) is ~19x faster on small circuits.
        de_sim_method = "matrix_product_state" if infrastructure == "cunqa" else "automatic"

        tic = time.time()
        result_params = polypus.train(
            qc,
            polypus.DE(generations=MAX_GENERATIONS, population_size=POPULATION_SIZE, tolerance=TOL),
            shots=N_SHOTS,
            n_qpus=N_QPUS,
            dimensions=2 * layers,
            expectation_function=bitstring_to_obj,
            infrastructure=infrastructure,
            nodes=NUM_NODES,
            cores_per_qpu=CORES_PER_QPU,
            id=id,
            sim_method=de_sim_method).best_params
        elapsed = time.time() - tic

        counts, best_bitstring, cut, approximation_ratio = final_evaluation(
            qc, graph, layers, cost_layers, mixer_layers, result_params, N_SHOTS, BEST_SOLUTION)
        log_experiment_results(n_qubits, method, elapsed, approximation_ratio, N_SHOTS, NUM_NODES, CORES_PER_QPU)
        plot_results(graph, counts, best_bitstring, cut, "Polypus QAOA", elapsed,
                     BEST_SOLUTION, POPULATION_SIZE, MAX_GENERATIONS, layers, "DE", id)

    elif method in ('polypus_local_pso', 'polypus_cunqa_pso'):

        infrastructure = "local" if method == 'polypus_local_pso' else "cunqa"

        tic = time.time()
        result_params = polypus.train(
            qc,
            polypus.PSO(generations=MAX_GENERATIONS, population_size=POPULATION_SIZE, bounds=(0.0, np.pi), tolerance=TOL),
            shots=N_SHOTS,
            n_qpus=N_QPUS,
            dimensions=2 * layers,
            expectation_function=bitstring_to_obj,
            infrastructure=infrastructure,
            nodes=NUM_NODES,
            cores_per_qpu=CORES_PER_QPU,
            id=id).best_params
        elapsed = time.time() - tic

        counts, best_bitstring, cut, approximation_ratio = final_evaluation(
            qc, graph, layers, cost_layers, mixer_layers, result_params, N_SHOTS, BEST_SOLUTION)
        log_experiment_results(n_qubits, method, elapsed, approximation_ratio, N_SHOTS, NUM_NODES, CORES_PER_QPU)
        plot_results(graph, counts, best_bitstring, cut, "Polypus PSO QAOA", elapsed,
                     BEST_SOLUTION, POPULATION_SIZE, MAX_GENERATIONS, layers, "PSO", id)

    elif method in ('polypus_local_qng', 'polypus_cunqa_qng'):

        infrastructure = "local" if method == 'polypus_local_qng' else "cunqa"

        # QNG uses its own alternating-parameter circuit and a variance callback
        qc_qng = build_qaoa_circuit_qng(graph, layers)
        variance_fn = make_variance_function(graph, n_qubits, N_SHOTS)

        QNG_LEARNING_RATE = 0.1
        QNG_STEP_SIZE     = 0.1
        QNG_TIKHONOV_REG  = 0.05

        tic = time.time()
        result_params = polypus.train(
            qc_qng,
            polypus.QNG(variance_fn, max_iters=MAX_GENERATIONS, bounds=(0.0, np.pi),
                        learning_rate=QNG_LEARNING_RATE, finite_difference_step=QNG_STEP_SIZE,
                        tikhonov_reg=QNG_TIKHONOV_REG),
            shots=N_SHOTS,
            n_qpus=N_QPUS,
            dimensions=2 * layers,
            expectation_function=bitstring_to_obj,
            infrastructure=infrastructure,
            nodes=NUM_NODES,
            cores_per_qpu=CORES_PER_QPU,
            id=id).best_params
        elapsed = time.time() - tic

        # Final evaluation uses the QNG circuit (alternating parameterisation)
        qc_eval = build_qaoa_circuit_qng(graph, layers)
        qc_eval.assign_parameters(result_params, inplace=True)
        counts = AerSimulator().run(qc_eval, shots=N_SHOTS).result().get_counts()
        best_bitstring = max(counts, key=counts.get)
        cut = bitstring_to_obj(best_bitstring)
        approximation_ratio = compute_mean_energy(counts) / BEST_SOLUTION

        log_experiment_results(n_qubits, method, elapsed, approximation_ratio, N_SHOTS, NUM_NODES, CORES_PER_QPU)
        plot_results(graph, counts, best_bitstring, cut, "Polypus QNG QAOA", elapsed,
                     BEST_SOLUTION, POPULATION_SIZE, MAX_GENERATIONS, layers, "QNG", id)