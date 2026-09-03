"""
Compare the Qiskit **Aer** backend against the native pure-Rust **polypus**
statevector backend.

Both backends run the *same* native ``polypus.Circuit`` through the *same*
entry points — only ``backend=`` changes — so the comparison is apples to
apples. We check two things:

  1. Correctness: the measured distributions agree (total-variation distance).
  2. Performance: wall-clock time for a single run and for a full training loop,
     where the native backend's GIL-free, QASM-free path is expected to help.

(The QML entry point ``polypus.qml.train`` is Qiskit-only — feature maps and
ansätze are Qiskit circuits — so it always uses Aer and is not part of this
comparison.)
"""

import itertools
import time

import polypus

polypus.init_logger(run_name="infraestructures")

# Fully connected (complete-graph) MaxCut instance over all N_QUBITS nodes.
N_QUBITS = 12
EDGES = list(itertools.combinations(range(N_QUBITS), 2))
SHOTS = 8192

# A separate, smaller instance for the distribution-agreement check in section 1
# below. Total-variation distance between two *independently sampled* empirical
# distributions is only a meaningful estimator when SHOTS is large relative to
# the outcome space (2**n): at N_QUBITS qubits (2**12 = 4096 possible outcomes,
# ~2000 of them live for this circuit) even two samples of the exact *same*
# distribution disagree by TVD ~0.23 at SHOTS=8192 — pure sampling noise, not a
# backend discrepancy (confirmed by running Aer against itself and polypus
# against itself: same ~0.23 either way). A 5-qubit instance keeps 32 possible
# outcomes, small enough for SHOTS=8192 to make TVD a reliable signal again
# (empirically <0.035 between backends, comfortably under the 0.05 threshold)
# without inflating the shot budget just for this check.
N_QUBITS_CORRECTNESS = 5
EDGES_CORRECTNESS = list(itertools.combinations(range(N_QUBITS_CORRECTNESS), 2))


def build_qaoa(betas, gammas, n_qubits=N_QUBITS, edges=None):
    """One-or-more-layer QAOA MaxCut ansatz as a native polypus.Circuit.

    Built exactly like ``max_cut_qaoa.py`` (``maxcut_cost_layer`` +
    ``standard_mixer_layer`` from ``polypus_python.qaoa_utils``): each edge
    gets a ``cx; rz(-gamma); cx`` cost term and each qubit an ``rx(2*beta)``
    mixer rotation. Defaults to the module-level ``N_QUBITS``/``EDGES``;
    pass ``n_qubits``/``edges`` explicitly to build a differently-sized
    instance (see ``N_QUBITS_CORRECTNESS`` above).
    """
    edges = EDGES if edges is None else edges
    qc = polypus.Circuit(n_qubits)
    for q in range(n_qubits):
        qc = qc.h(q)
    for beta, gamma in zip(betas, gammas):
        # Cost layer (maxcut_cost_layer): cx; rz(-gamma); cx for every edge.
        for a, b in edges:
            qc = qc.cx(a, b)
            qc = qc.rz(b, -gamma)
            qc = qc.cx(a, b)
        # Mixer layer (standard_mixer_layer): rx(2*beta) on every qubit.
        for q in range(n_qubits):
            qc = qc.rx(q, 2 * beta)
    return qc.measure_all()


def total_variation(a, b, shots):
    """Total-variation distance between two count dicts (0 = identical)."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


def maxcut_value(bitstring):
    """MaxCut objective. Counts keys are MSB-left (Qiskit order), so reverse
    to index by qubit: bits[q] is the measured value of qubit q."""
    bits = bitstring[::-1]
    return sum(1 for a, b in EDGES if bits[a] != bits[b])


# Nodes that actually appear in EDGES; idle qubits never contribute to the cut,
# so the brute force stays 2**(#graph nodes) instead of 2**N_QUBITS.
GRAPH_NODES = sorted({q for edge in EDGES for q in edge})


def maxcut_bruteforce():
    """Optimal MaxCut value by exhaustive search over every assignment of the
    graph nodes (same idea as ``maxcut_bruteforce`` in max_cut_qaoa.py). Used
    as the denominator of the approximation ratio = solution quality vs the
    true optimum.

    MaxCut is invariant under flipping every bit, so the first node is pinned
    to 0 (halves the search), and edges are pre-mapped to compact node indices
    to avoid rebuilding a lookup dict on each of the 2**(n-1) assignments."""
    pos = {node: i for i, node in enumerate(GRAPH_NODES)}
    edges = [(pos[a], pos[b]) for a, b in EDGES]
    best = 0
    for rest in itertools.product([0, 1], repeat=len(GRAPH_NODES) - 1):
        assignment = (0,) + rest
        cut = sum(1 for a, b in edges if assignment[a] != assignment[b])
        best = max(best, cut)
    return best


# Exact optimum for this instance; reused as the approximation-ratio baseline.
OPTIMAL = maxcut_bruteforce()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correctness — same fixed circuit on both backends
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("1. Correctness: identical native circuit, both backends")
print("=" * 70)

fixed_circuit = build_qaoa([0.8], [0.4])

# Distribution agreement: checked on the small instance (see
# N_QUBITS_CORRECTNESS above), where SHOTS is large enough relative to the
# outcome space for total-variation distance to be a meaningful estimator.
correctness_circuit = build_qaoa(
    [0.8], [0.4], n_qubits=N_QUBITS_CORRECTNESS, edges=EDGES_CORRECTNESS
)
counts_aer_small = polypus.run_quantum_circuit(
    correctness_circuit, shots=SHOTS, infrastructure="local", backend="aer"
).counts[0]
counts_native_small = polypus.run_quantum_circuit(
    correctness_circuit, shots=SHOTS, infrastructure="local", backend="polypus"
).counts[0]

tvd = total_variation(counts_aer_small, counts_native_small, SHOTS)
print(
    f"\nTotal-variation distance (Aer vs polypus, {N_QUBITS_CORRECTNESS}q "
    f"instance): {tvd:.4f}"
)
print("  -> distributions agree" if tvd < 0.05 else "  -> MISMATCH")

# Mean cut / approximation ratio: reported on the full N_QUBITS instance used
# for the performance and training sections below. These are aggregate
# statistics (one weighted sum over all sampled outcomes), far less sensitive
# to per-bin shot noise than the raw joint distribution above, so they stay
# meaningful even where N_QUBITS makes bin-by-bin TVD unreliable.
counts_aer = polypus.run_quantum_circuit(
    fixed_circuit, shots=SHOTS, infrastructure="local", backend="aer"
).counts[0]
counts_native = polypus.run_quantum_circuit(
    fixed_circuit, shots=SHOTS, infrastructure="local", backend="polypus"
).counts[0]

exp_aer = sum(maxcut_value(k) * v for k, v in counts_aer.items()) / SHOTS
exp_native = sum(maxcut_value(k) * v for k, v in counts_native.items()) / SHOTS
print(f"Mean cut   Aer: {exp_aer:.4f}   polypus: {exp_native:.4f}")
print(f"Optimal cut (brute force): {OPTIMAL}")
print(
    f"Approximation ratio   Aer: {exp_aer / OPTIMAL:.4f}"
    f"   polypus: {exp_native / OPTIMAL:.4f}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Performance — single-run wall-clock
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. Performance: single-circuit run (median of 20 reps)")
print("=" * 70)


def time_run(backend, reps=20):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        polypus.run_quantum_circuit(
            fixed_circuit, shots=SHOTS, infrastructure="local", backend=backend
        )
        samples.append(time.perf_counter() - t0)
    return sorted(samples)[len(samples) // 2]


t_aer = time_run("aer")
t_native = time_run("polypus")
print(f"\nAer:     {t_aer * 1e3:8.3f} ms")
print(f"polypus: {t_native * 1e3:8.3f} ms")
if t_native:
    print(f"speedup: {t_aer / t_native:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training — DE optimisation of the QAOA angles on both backends
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("3. Training: DE optimisation of QAOA(beta, gamma)")
print("=" * 70)

# Parameterised ansatz: Param(0)=gamma (cost), Param(1)=beta (mixer).
# Same gate structure as max_cut_qaoa.py (cx; rz; cx cost layer + rx mixer).
# The native polypus.Param binds a raw angle and can't carry the -1/x2 scale
# factors, so the optimiser absorbs that reparametrisation and still converges
# to the same MaxCut solution.
ansatz = polypus.Circuit(N_QUBITS)
for q in range(N_QUBITS):
    ansatz = ansatz.h(q)
for a, b in EDGES:
    ansatz = ansatz.cx(a, b)
    ansatz = ansatz.rz(b, polypus.Param(0))
    ansatz = ansatz.cx(a, b)
for q in range(N_QUBITS):
    ansatz = ansatz.rx(q, polypus.Param(1))
ansatz = ansatz.measure_all()

train_kw = dict(
    shots=1024,
    n_qpus=1,
    dimensions=2,
    expectation_function=maxcut_value,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
)


def time_train(backend):
    t0 = time.perf_counter()
    result = polypus.train(
        ansatz,
        polypus.DE(generations=15, population_size=20, tolerance=1e-6),
        id=f"qaoa_{backend}",
        backend=backend,
        **train_kw,
    )
    # train now returns a TrainResult (contract C-7); the tuned angles are in
    # .best_params, alongside .best_fitness / .iterations_run / .converged / .seed.
    return result.best_params, time.perf_counter() - t0


def evaluate_solution(params, backend):
    """Run the trained ansatz with `params` bound and measure how close the
    result gets to the brute-force optimum. Binding via ``to_qasm2(params)``
    reproduces exactly the angles the optimiser evaluated.

    Returns the mean cut (approximation-ratio numerator) and the best cut
    actually sampled."""
    bound = ansatz.to_qasm2(list(params))
    counts = polypus.run_quantum_circuit(
        bound, shots=SHOTS, infrastructure="local", backend=backend
    ).counts[0]
    mean_cut = sum(maxcut_value(k) * v for k, v in counts.items()) / SHOTS
    best_cut = max(maxcut_value(k) for k in counts)
    return mean_cut, best_cut


params_aer, dt_aer = time_train("aer")
params_native, dt_native = time_train("polypus")

mean_aer, best_aer = evaluate_solution(params_aer, "aer")
mean_native, best_native = evaluate_solution(params_native, "polypus")

print(f"\nOptimal cut (brute force): {OPTIMAL}")
print(
    f"Aer:     best (gamma, beta) = ({params_aer[0]:.3f}, {params_aer[1]:.3f})"
    f"   time = {dt_aer:.2f} s"
)
print(
    f"         mean cut = {mean_aer:.3f} (approx ratio {mean_aer / OPTIMAL:.3f})"
    f"   best sampled cut = {best_aer}/{OPTIMAL}"
)
print(
    f"polypus: best (gamma, beta) = ({params_native[0]:.3f}, {params_native[1]:.3f})"
    f"   time = {dt_native:.2f} s"
)
print(
    f"         mean cut = {mean_native:.3f} (approx ratio {mean_native / OPTIMAL:.3f})"
    f"   best sampled cut = {best_native}/{OPTIMAL}"
)
if dt_native:
    print(f"speedup: {dt_aer / dt_native:.2f}x")
