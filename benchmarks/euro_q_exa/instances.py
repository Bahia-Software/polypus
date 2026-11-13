"""
MaxCut instances and their QAOA circuits for the Euro-Q-Exa benchmark (#391).

Two families, n = 16 and 20, three instances each:

  * ``3reg``: random 3-regular graphs, unit weights. The main workload: standard
    in the QAOA literature, and not embeddable in the square lattice, so routing
    adds SWAPs.
  * ``lattice``: connected n-node subgraphs of the device's own coupling graph
    with random +-1 weights. The control: the same algorithm with no routing
    cost. Unweighted, these subgraphs are bipartite and their MaxCut is trivial
    (every edge cut); the signs make them frustrated.

Every instance is generated from a fixed seed, so the set is reproducible, and
carries its exact optimum (brute force over 2^n cuts), the denominator of the
approximation ratio.

Conventions. A cut is a bitstring as Qiskit reports counts: qubit 0 is the
rightmost character. The cut value is ``sum(w_ij for edges with x_i != x_j)``.
The QAOA cost unitary is ``exp(-i γ C)`` with ``C = sum w_ij (1 - Z_i Z_j) / 2``,
i.e. ``rzz(-γ w_ij)`` per edge (up to a global phase), and the mixer is
``exp(-i β sum X)``, i.e. ``rx(2β)``. Circuits come from polypus's own
``build_qaoa_circuit``, so their parameter order is polypus's: all betas, then
all gammas (``[β0..β(p-1), γ0..γ(p-1)]``).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from polypus_python.qaoa_utils import build_qaoa_circuit, expectation_value

SIZES = (16, 20)
INSTANCES_PER_SIZE = 3
FAMILIES = ("3reg", "lattice")


@dataclass(frozen=True)
class Instance:
    """A weighted MaxCut instance on nodes ``0..n-1``.

    ``edges`` are ``(i, j, w)`` with ``i < j``. For the ``lattice`` family,
    ``device_qubits[k]`` is the device (compact) qubit node ``k`` was taken from.
    """

    name: str
    family: str
    n: int
    seed: int
    edges: tuple[tuple[int, int, float], ...]
    optimum: float
    device_qubits: tuple[int, ...] = ()

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def random_cut(self) -> float:
        """Expected cut of a uniformly random bitstring (each edge cut with
        probability 1/2): the p = 0 baseline."""
        return sum(w for _, _, w in self.edges) / 2


def maxcut_optimum(n: int, edges) -> float:
    """Exact MaxCut value by enumerating all 2^n cuts (n <= 24)."""
    if n > 24:
        raise ValueError(f"brute force over 2^{n} cuts is too large")
    x = np.arange(2**n, dtype=np.uint32)
    value = np.zeros(2**n)
    for i, j, w in edges:
        value += w * (((x >> i) ^ (x >> j)) & 1)
    return float(value.max())


def cut_value(bitstring: str, edges) -> float:
    """Cut value of a counts key (qubit 0 rightmost)."""
    bits = bitstring[::-1]
    return float(sum(w for i, j, w in edges if bits[i] != bits[j]))


def approximation_ratio(counts, instance: Instance) -> float:
    """Mean cut of the sampled bitstrings over the optimum."""
    mean = expectation_value(counts, lambda b: cut_value(b, instance.edges))
    return mean / instance.optimum


def ising_cost(instance: Instance):
    """The cut as a native ``polypus.Ising`` observable, for ``polypus.train``
    (which maximizes): ``cut = sum w (1 - z_i z_j) / 2`` with ``z = 1 - 2x``,
    i.e. couplings ``-w/2`` and constant ``sum w / 2``. Evaluated in Rust, with
    the same bit order as the counts (qubit 0 rightmost)."""
    import polypus

    return polypus.Ising(
        instance.n,
        couplings=[(i, j, -w / 2) for i, j, w in instance.edges],
        constant=instance.random_cut,
    )


def optimum_probability(counts, instance: Instance) -> float:
    """Fraction of shots that hit an optimal cut."""
    total = sum(counts.values())
    hits = sum(
        n
        for key, n in counts.items()
        if abs(cut_value(key, instance.edges) - instance.optimum) < 1e-9
    )
    return hits / total


def _finish(name, family, n, seed, edges, device_qubits=()):
    edges = tuple(sorted((min(i, j), max(i, j), float(w)) for i, j, w in edges))
    optimum = maxcut_optimum(n, edges)
    if optimum <= 0:
        raise ValueError(f"{name}: optimum {optimum} is not positive")
    return Instance(name, family, n, seed, edges, optimum, tuple(device_qubits))


def regular_instance(n: int, index: int) -> Instance:
    """Random 3-regular graph, unit weights, seed ``1000 n + index``."""
    seed = 1000 * n + index
    graph = nx.random_regular_graph(3, n, seed=seed)
    edges = [(i, j, 1.0) for i, j in graph.edges()]
    return _finish(f"3reg-n{n}-i{index}", "3reg", n, seed, edges)


def lattice_instance(n: int, index: int, coupling_edges) -> Instance:
    """Connected n-node subgraph of the device coupling graph, +-1 weights.

    Grown from a random device qubit by adding a random neighbour of the grown
    set until it has ``n`` nodes; every coupling edge between the chosen qubits
    is kept. The signs are drawn again (same generator, so still reproducible)
    while the optimum is not positive, which leaves no ratio to report. Seed
    ``5000 + 1000 n + index``.
    """
    seed = 5000 + 1000 * n + index
    rng = np.random.default_rng(seed)
    coupling = nx.Graph([tuple(e) for e in coupling_edges])
    if n > coupling.number_of_nodes():
        raise ValueError(f"the device has {coupling.number_of_nodes()} qubits, not {n}")
    qubits = sorted(coupling.nodes())
    chosen = [qubits[int(rng.integers(len(qubits)))]]
    while len(chosen) < n:
        frontier = sorted({v for u in chosen for v in coupling[u]} - set(chosen))
        chosen.append(frontier[int(rng.integers(len(frontier)))])
    node = {q: k for k, q in enumerate(chosen)}
    pairs = [(node[a], node[b]) for a, b in sorted(coupling.subgraph(chosen).edges())]
    for _ in range(100):
        edges = [(i, j, float(rng.choice([-1.0, 1.0]))) for i, j in pairs]
        if maxcut_optimum(n, edges) > 0:
            return _finish(f"lattice-n{n}-i{index}", "lattice", n, seed, edges, chosen)
    raise ValueError(
        f"lattice-n{n}-i{index}: no signs with a positive optimum in 100 draws"
    )


def make_instances(coupling_edges, sizes=SIZES) -> list[Instance]:
    """The benchmark's instances: both families x ``sizes`` x INSTANCES_PER_SIZE
    (12 with the default sizes)."""
    out = []
    for n in sizes:
        out += [regular_instance(n, i) for i in range(INSTANCES_PER_SIZE)]
    for n in sizes:
        out += [
            lattice_instance(n, i, coupling_edges) for i in range(INSTANCES_PER_SIZE)
        ]
    return out


def qaoa_circuit(instance: Instance, p: int):
    """Depth-``p`` QAOA circuit for ``instance`` (conventions: module docstring),
    built with polypus's ``build_qaoa_circuit``; measures every node."""

    class _Nodes:  # build_qaoa_circuit only asks for the node count
        def number_of_nodes(self):
            return instance.n

    def cost(qc, _layer, gamma):
        for i, j, w in instance.edges:
            qc.rzz(-w * gamma, i, j)

    def mixer(qc, _layer, beta):
        for q in range(instance.n):
            qc.rx(2 * beta, q)

    return build_qaoa_circuit(_Nodes(), p, [cost] * p, [mixer] * p)


def native_qaoa_circuit(instance: Instance, p: int, measure: bool = True):
    """The same QAOA circuit as :func:`qaoa_circuit`, as a ``polypus.Circuit``
    for the native backend, with the same parameter order (``Param(k)`` is
    ``β[k]``, ``Param(p + k)`` is ``γ[k]``), so parameters trained on one bind
    to the other."""
    import polypus

    circuit = polypus.Circuit(instance.n)
    for q in range(instance.n):
        circuit.h(q)
    for layer in range(p):
        beta, gamma = polypus.Param(layer), polypus.Param(p + layer)
        for i, j, w in instance.edges:
            circuit.rzz(i, j, -w * gamma)
        for q in range(instance.n):
            circuit.rx(q, 2 * beta)
    if measure:
        circuit.measure_all()
    return circuit


def save_instances(path, instances) -> None:
    Path(path).write_text(json.dumps([asdict(i) for i in instances], indent=1))


def load_instances(path) -> list[Instance]:
    raw = json.loads(Path(path).read_text())
    return [
        Instance(
            name=r["name"],
            family=r["family"],
            n=r["n"],
            seed=r["seed"],
            edges=tuple(tuple(e) for e in r["edges"]),
            optimum=r["optimum"],
            device_qubits=tuple(r["device_qubits"]),
        )
        for r in raw
    ]
