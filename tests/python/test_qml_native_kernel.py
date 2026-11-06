"""
Quantum kernels (contract C-11, "Quantum kernels"): ``polypus.qml.kernel_matrix``.

With ``exact=True`` every encoder's kernel ``K[i, j] = |<phi(x1_i)|phi(x2_j)>|^2``
is checked against a reference written here in NumPy, independent of Polypus:
the closed form of each encoder's state. With shots, each entry is the
frequency of ``0...0`` of the circuit ``U(x1_i)`` then ``U(x2_j)^dagger``, an
unbiased estimate within shot noise, seeded. The circuits are counted at the
CUNQA seam (``fake_cunqa``), which runs them on Aer: ``N(N+1)/2`` for the
symmetric matrix, ``N*M`` otherwise. Every check runs before a backend exists.
"""

import math

import numpy as np
import polypus
import pytest
from fake_cunqa import FakeCunqa
from polypus import qml

# Two exact routes, or an exact route against a closed form evaluated in
# NumPy: each value is a few dozen roundings of f64 arithmetic (~1e-16 each),
# so 1e-12 leaves four orders of magnitude of slack and still catches any wrong
# sign, order or angle, which moves a kernel value by far more.
ROUNDING = 1e-12

ROWS = [[0.3, -1.2, 0.8], [1.1, 0.4, -0.5], [-0.7, 2.0, 0.1], [0.0, 0.6, 1.4]]
OTHER = [[0.9, 0.2, -1.0], [-0.3, -0.8, 0.5]]


def kernel(encoder, x1, x2=None, **kwargs):
    kwargs.setdefault("exact", True)
    return qml.kernel_matrix(encoder, x1, x2, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy references: each encoder's state, built from its definition
# ─────────────────────────────────────────────────────────────────────────────

ZERO = np.array([1.0, 0.0], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def product_state(single_qubit_states):
    """Qubit 0 is the least significant bit (C-3), so it is the last factor."""
    state = np.array([1.0], dtype=complex)
    for qubit_state in single_qubit_states:
        state = np.kron(qubit_state, state)
    return state


def iqp_state(x, pairs):
    """``H`` on every qubit, then ``Rz(x_i)``, then ``Rzz(x_i * x_j)`` on each
    pair: diagonal phases ``exp(-i f(b) / 2)`` on the uniform superposition,
    ``f(b) = sum_i x_i z_i + sum_(i,j) x_i x_j z_i z_j`` with ``z = +1`` for a 0
    bit and ``-1`` for a 1."""
    n = len(x)
    phases = []
    for b in range(2**n):
        z = [1.0 if (b >> i) & 1 == 0 else -1.0 for i in range(n)]
        f = sum(x[i] * z[i] for i in range(n)) + sum(
            x[i] * x[j] * z[i] * z[j] for i, j in pairs
        )
        phases.append(np.exp(-0.5j * f))
    return np.array(phases) / math.sqrt(2**n)


def gram(states1, states2):
    return np.array([[abs(np.vdot(a, b)) ** 2 for b in states2] for a in states1])


def assert_close(got, want, tolerance=ROUNDING):
    assert got.shape == np.shape(want)
    assert np.max(np.abs(got - want)) < tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Shapes and types (criterion 1)
# ─────────────────────────────────────────────────────────────────────────────


class TestShapesAndTypes:
    def test_a_float64_c_contiguous_array_of_rows_by_columns(self):
        encoder = qml.Encoder(3).angle_encoder("ry")
        for x2, shape in [(None, (4, 4)), (OTHER, (4, 2))]:
            for exact in (True, False):
                k = qml.kernel_matrix(encoder, ROWS, x2, exact=exact, shots=64, seed=1)
                assert isinstance(k, np.ndarray)
                assert k.dtype == np.float64
                assert k.flags["C_CONTIGUOUS"]
                assert k.shape == shape

    def test_rows_are_lists_tuples_arrays_or_a_dataset_whose_labels_are_unused(self):
        encoder = qml.Encoder(3).iqp_encoder()
        want = kernel(encoder, ROWS, OTHER)
        for x1, x2 in [
            (tuple(map(tuple, ROWS)), tuple(map(tuple, OTHER))),
            (np.array(ROWS), np.array(OTHER)),
            (qml.Dataset(ROWS, [1, -1, 1, -1]), qml.Dataset(OTHER)),
        ]:
            assert np.array_equal(kernel(encoder, x1, x2), want)

    def test_the_encoder_is_copied_at_the_call(self):
        encoder = qml.Encoder(3).angle_encoder("ry")
        before = kernel(encoder, ROWS)
        encoder.angle_encoder("rx")
        assert not np.array_equal(kernel(encoder, ROWS), before)


# ─────────────────────────────────────────────────────────────────────────────
# Exact = the formula (criterion 2)
# ─────────────────────────────────────────────────────────────────────────────


class TestExactIsTheFormula:
    def test_angle_ry_on_one_qubit_is_cos_squared_of_half_the_difference(self):
        x1 = [[0.3], [-1.2], [2.5], [0.0]]
        x2 = [[0.7], [3.0], [-2.2]]
        want = np.array([[math.cos((a[0] - b[0]) / 2) ** 2 for b in x2] for a in x1])
        assert_close(kernel(qml.Encoder(1).angle_encoder("ry"), x1, x2), want)

    def test_basis_is_one_for_equal_samples_and_zero_otherwise(self):
        x = [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        want = np.array([[float(a == b) for b in x] for a in x])
        assert_close(kernel(qml.Encoder(3).basis_encoder(), x), want)

    def test_amplitude_is_the_squared_cosine_of_the_normalized_samples(self):
        x = np.array([[1.0, 2.0, 0.5], [0.2, 0.0, 3.0], [4.0, 1.0, 1.0]])
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        assert_close(
            kernel(qml.Encoder(2).amplitude_encoder(), x), (unit @ unit.T) ** 2
        )

    @pytest.mark.parametrize(
        ("n", "entanglement", "pairs"),
        [
            (2, "linear", [(0, 1)]),
            (2, "circular", [(0, 1), (1, 0)]),
            (2, "full", [(0, 1)]),
            (3, "linear", [(0, 1), (1, 2)]),
            (3, "circular", [(0, 1), (1, 2), (2, 0)]),
            (3, "full", [(0, 1), (0, 2), (1, 2)]),
        ],
    )
    def test_iqp_is_its_closed_form(self, n, entanglement, pairs):
        x1 = [[0.4 * s - 0.7 * i + 0.2 for i in range(n)] for s in range(4)]
        x2 = [[0.9 - 0.3 * s + 0.5 * i for i in range(n)] for s in range(3)]
        want = gram(
            [iqp_state(a, pairs) for a in x1], [iqp_state(b, pairs) for b in x2]
        )
        encoder = qml.Encoder(n).iqp_encoder(entanglement)
        assert_close(kernel(encoder, x1, x2), want)
        assert_close(
            kernel(encoder, x1),
            gram(*[[iqp_state(a, pairs) for a in x1]] * 2),
        )

    def test_stacked_encoders_compose(self):
        # IQP, then Ry(x_i) on each qubit.
        pairs = [(0, 1)]
        x = [[0.3, -0.9], [1.2, 0.4], [-0.5, 0.8]]

        def state(v):
            phases = iqp_state(v, pairs)
            rotation = np.kron(ry(v[1]), ry(v[0]))
            return rotation @ phases

        encoder = qml.Encoder(2).iqp_encoder().angle_encoder("ry")
        assert_close(
            kernel(encoder, x), gram([state(v) for v in x], [state(v) for v in x])
        )

    def test_surplus_qubits_change_nothing(self):
        x = [[0.3, 1.0], [-0.4, 0.2], [2.1, -1.3]]

        def state(v):
            return product_state([ry(a) @ ZERO for a in v] + [ZERO, ZERO])

        want = gram([state(v) for v in x], [state(v) for v in x])
        assert_close(kernel(qml.Encoder(4).angle_encoder("ry"), x), want)
        assert_close(kernel(qml.Encoder(2).angle_encoder("ry"), x), want)

    def test_angle_rz_is_prepared_on_plus(self):
        x = [[0.3], [-1.1], [2.0]]

        def state(v):
            rz = np.diag([np.exp(-0.5j * v[0]), np.exp(0.5j * v[0])])
            return rz @ H @ ZERO

        assert_close(
            kernel(qml.Encoder(1).angle_encoder("rz"), x),
            gram([state(v) for v in x], [state(v) for v in x]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry, diagonal, PSD (criterion 4)
# ─────────────────────────────────────────────────────────────────────────────


class TestSymmetry:
    def _encoder(self):
        return qml.Encoder(3).iqp_encoder().angle_encoder("rx")

    def _rows(self, n=8):
        return [[0.21 * (s + i) - 0.4 * i for i in range(3)] for s in range(n)]

    def test_the_gram_matrix_is_symmetric_bit_for_bit_with_a_unit_diagonal(self):
        for exact in (True, False):
            k = qml.kernel_matrix(
                self._encoder(), self._rows(), exact=exact, shots=128, seed=4
            )
            assert np.array_equal(k, k.T)
            assert np.all((k >= 0.0) & (k <= 1.0))
        k = kernel(self._encoder(), self._rows())
        assert np.max(np.abs(np.diag(k) - 1.0)) < ROUNDING

    def test_the_exact_gram_matrix_is_positive_semi_definite(self):
        # LAPACK's eigenvalues of a symmetric N x N matrix are accurate to about
        # N * eps * ||K||_2 <= N^2 * eps (every entry is at most 1): 1.4e-14 for
        # N = 8, far inside ROUNDING; a matrix that is not a Gram matrix has
        # eigenvalues far more negative than that.
        k = kernel(self._encoder(), self._rows())
        assert np.min(np.linalg.eigvalsh(k)) >= -ROUNDING

    def test_an_explicit_x2_equal_to_x1_is_not_detected(self, monkeypatch):
        rows = self._rows(5)
        symmetric = kernel(self._encoder(), rows)
        explicit = kernel(self._encoder(), rows, rows)
        assert_close(explicit, symmetric)
        # Every one of its N*M circuits runs, and N(N+1)/2 without x2.
        assert _cunqa_circuits(monkeypatch, self._encoder(), rows, rows) == 25
        assert _cunqa_circuits(monkeypatch, self._encoder(), rows) == 15

    def test_the_diagonal_circuits_are_executed(self, monkeypatch):
        fake = FakeCunqa().install(monkeypatch)
        rows = self._rows(3)
        qml.kernel_matrix(
            self._encoder(), rows, infrastructure="cunqa", shots=16, seed=1,
            nodes=1, cores_per_qpu=1,
        )  # fmt: skip
        # The first pair is (0, 0): U(x_0) then U(x_0)^dagger, then the
        # measurement — not a constant 1.
        first = fake.runs[0]["circuit"]
        names = [instruction.operation.name for instruction in first.data]
        assert names.count("measure") == 3
        assert len(names) > 3


def _cunqa_circuits(monkeypatch, encoder, x1, x2=None):
    fake = FakeCunqa().install(monkeypatch)
    k = qml.kernel_matrix(
        encoder, x1, x2, infrastructure="cunqa", shots=8, seed=2, nodes=1,
        cores_per_qpu=1,
    )  # fmt: skip
    assert k.shape == (len(x1), len(x1 if x2 is None else x2))
    return len(fake.runs)


# ─────────────────────────────────────────────────────────────────────────────
# Shots against exact, seeds and blocks (criteria 3, 5, 6)
# ─────────────────────────────────────────────────────────────────────────────


class TestShotsAndSeeds:
    def test_a_seeded_estimate_is_within_shot_noise_of_the_exact_kernel(self):
        # Each entry is the mean of `shots` Bernoulli(p) draws, standard error
        # sqrt(p(1 - p) / shots). Six standard errors bound every entry with
        # probability 1 - 2e-9 each; the seed fixes the draw, so the test is
        # deterministic. p in {0, 1} has no noise: the estimate is exact.
        encoder = qml.Encoder(3).iqp_encoder("linear")
        shots = 4096
        exact = kernel(encoder, ROWS, OTHER)
        sampled = qml.kernel_matrix(encoder, ROWS, OTHER, shots=shots, seed=11)
        bound = 6 * np.sqrt(exact * (1 - exact) / shots) + 1e-12
        assert np.all(np.abs(sampled - exact) <= bound)

    def test_the_estimator_is_unbiased(self):
        # The mean over 40 seeds has standard error sqrt(p(1-p) / (40 * shots))
        # per entry; five of them is generous and still small.
        encoder = qml.Encoder(2).angle_encoder("ry")
        x1, x2 = [[0.4, 1.3], [2.0, -0.6]], [[-0.9, 0.2], [1.1, 1.7]]
        exact = kernel(encoder, x1, x2)
        shots, seeds = 512, 40
        mean = np.mean(
            [
                qml.kernel_matrix(encoder, x1, x2, shots=shots, seed=s)
                for s in range(seeds)
            ],
            axis=0,
        )
        bound = 5 * np.sqrt(exact * (1 - exact) / (shots * seeds)) + 1e-12
        assert np.all(np.abs(mean - exact) <= bound)

    def test_amplitude_encoding_runs_with_shots(self):
        encoder = qml.Encoder(2).amplitude_encoder()
        x = [[1.0, 2.0, 0.5, 0.3], [0.2, 0.1, 3.0, 1.0]]
        exact = kernel(encoder, x)
        sampled = qml.kernel_matrix(encoder, x, shots=4096, seed=5)
        assert np.all(
            np.abs(sampled - exact) <= 6 * np.sqrt(exact * (1 - exact) / 4096) + 1e-12
        )

    def test_a_seed_replays_another_differs_and_none_is_drawn(self):
        encoder = qml.Encoder(3).iqp_encoder()
        first = qml.kernel_matrix(encoder, ROWS, shots=128, seed=7)
        assert np.array_equal(
            first, qml.kernel_matrix(encoder, ROWS, shots=128, seed=7)
        )
        assert not np.array_equal(
            first, qml.kernel_matrix(encoder, ROWS, shots=128, seed=8)
        )
        drawn = [qml.kernel_matrix(encoder, ROWS, shots=128) for _ in range(2)]
        assert not np.array_equal(*drawn)

    def test_an_exact_call_is_the_concatenation_of_calls_on_chunks_of_x1(self):
        encoder = qml.Encoder(3).iqp_encoder().angle_encoder("ry")
        rows = [[0.13 * (s + 2 * i) - 0.5 for i in range(3)] for s in range(11)]
        whole = kernel(encoder, rows, OTHER)
        chunks = [rows[:1], rows[1:6], rows[6:]]
        assert np.array_equal(
            whole, np.concatenate([kernel(encoder, c, OTHER) for c in chunks])
        )

    def test_the_exact_kernel_is_the_overlap_of_qiskits_own_statevectors(self):
        # An independent simulator: Qiskit's statevector of the circuit
        # Polypus exports for each row's encoder (through `Model.circuit`, with
        # a zero-angle `Ry` layer as the ansatz, which is the identity).
        encoder = qml.Encoder(3).iqp_encoder("circular").angle_encoder("ry")
        want = gram(
            [_qiskit_state(encoder, row) for row in ROWS],
            [_qiskit_state(encoder, row) for row in OTHER],
        )
        assert_close(kernel(encoder, ROWS, OTHER), want)


def _qiskit_state(encoder, row):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    model = qml.Model(
        encoder,
        qml.Ansatz(encoder.num_qubits).real_amplitudes(0),
        qml.Readout([qml.Z(0)]),
    )
    circuit = model.circuit(row, params=[0.0] * model.num_params)
    qc = QuantumCircuit.from_qasm_str(circuit.to_qasm2())
    qc.remove_final_measurements()
    return Statevector(qc).data


# ─────────────────────────────────────────────────────────────────────────────
# The memory budget counts the matrix
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryBudget:
    # `POLYPUS_MEM_BUDGET` is read on every call. One-qubit states take 32
    # bytes, so in 1 KiB a 12 x 12 matrix (1152 bytes) leaves no room for the
    # two states an inner product needs, and a 10 x 10 one (800 bytes) leaves
    # room for seven, so it runs in tiles.

    def test_a_matrix_that_leaves_no_room_for_two_states_is_refused(self, monkeypatch):
        encoder = qml.Encoder(1).angle_encoder("ry")
        rows = [[0.3 * i] for i in range(12)]
        monkeypatch.setenv("POLYPUS_MEM_BUDGET", "1KiB")
        with pytest.raises(polypus.InsufficientMemoryError) as refused:
            kernel(encoder, rows)
        # The budget's own refusal, catchable as a backend error like any other.
        assert isinstance(refused.value, polypus.BackendError)
        message = str(refused.value)
        for part in (
            "a 12 x 12 fidelity matrix (1.1",
            "2 1-qubit statevectors (32 B each)",
            "1.00 KiB",
            "set by POLYPUS_MEM_BUDGET",
        ):
            assert part in message, (part, message)
        # Fewer rows at a time fit, and give those rows of the whole matrix.
        tiled = kernel(encoder, rows[:10], rows[:10])
        monkeypatch.delenv("POLYPUS_MEM_BUDGET")
        whole = kernel(encoder, rows, rows)
        assert np.array_equal(tiled, whole[:10, :10])


# ─────────────────────────────────────────────────────────────────────────────
# Rejections and their order (criterion 8)
# ─────────────────────────────────────────────────────────────────────────────


class TestRejections:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: qml.Ansatz(3).hardware_efficient(1),
            lambda: qml.Model(
                qml.Encoder(3).angle_encoder("ry"),
                qml.Ansatz(3).hardware_efficient(1),
                qml.Readout([qml.Z(0)]),
            ),
            lambda: (lambda model: model.bind([0.1] * model.num_params))(
                qml.Model(
                    qml.Encoder(3).angle_encoder("ry"),
                    qml.Ansatz(3).hardware_efficient(1),
                    qml.Readout([qml.Z(0)]),
                )
            ),
            lambda: polypus.Circuit(3).h(0),
            lambda: "encoder",
        ],
        ids=["ansatz", "model", "trained-model", "circuit", "string"],
    )
    def test_anything_but_an_encoder_is_a_type_error(self, make):
        with pytest.raises(TypeError, match="takes a polypus.qml.Encoder"):
            qml.kernel_matrix(make(), ROWS, exact=True)

    def test_the_type_error_comes_first(self):
        with pytest.raises(TypeError):
            qml.kernel_matrix("encoder", [], shots=0, exact=True, backend="aer")

    def test_exact_needs_the_local_native_simulator(self):
        encoder = qml.Encoder(3).angle_encoder("ry")
        for kwargs in [{"backend": "aer"}, {"infrastructure": "cunqa"}]:
            with pytest.raises(polypus.ConfigurationError, match="exact=True"):
                qml.kernel_matrix(encoder, ROWS, exact=True, **kwargs)

    @pytest.mark.parametrize(
        ("x1", "x2", "match"),
        [
            ([], None, "x1: dataset is empty"),
            ([[0.1, 0.2, 0.3], [0.1]], None, "x1: ragged .*sample 1"),
            ([[0.1, math.nan, 0.3]], None, "x1: non-finite feature"),
            ([[0.1, 0.2, 0.3, 0.4]], None, "x1: layer needs 4 active qubit"),
            (ROWS, [], "x2: dataset is empty"),
            (ROWS, [[0.1, 0.2, 0.3], [0.2]], "x2: ragged .*sample 1"),
            (ROWS, [[0.1, math.inf, 0.3]], "x2: non-finite feature"),
            (ROWS, [[0.1, 0.2]], "x2 has 2 feature"),
        ],
    )
    def test_rows_are_checked_against_the_encoder(self, x1, x2, match):
        with pytest.raises(polypus.ConfigurationError, match=match):
            qml.kernel_matrix(qml.Encoder(3).angle_encoder("ry"), x1, x2, exact=True)

    def test_samples_an_encoder_rejects(self):
        amplitude = qml.Encoder(2).amplitude_encoder()
        with pytest.raises(polypus.ConfigurationError, match="x1: amplitude"):
            qml.kernel_matrix(amplitude, [[0.0, 0.0, 0.0]], exact=True)
        with pytest.raises(polypus.ConfigurationError, match="x2: amplitude"):
            qml.kernel_matrix(
                amplitude, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], exact=True
            )
        basis = qml.Encoder(2).basis_encoder()
        with pytest.raises(polypus.ConfigurationError, match="x1: basis"):
            qml.kernel_matrix(basis, [[1.0, 0.5]], exact=True)
        with pytest.raises(polypus.ConfigurationError, match="has no layers"):
            qml.kernel_matrix(qml.Encoder(2), [[1.0, 0.5]], exact=True)

    def test_the_checks_run_in_order(self):
        encoder = qml.Encoder(2).amplitude_encoder()
        # shots before the rows; x1 (even a zero sample) before x2; x2 before
        # the seed on qmio.
        with pytest.raises(polypus.ConfigurationError, match="shots"):
            qml.kernel_matrix(encoder, [], shots=0)
        with pytest.raises(polypus.ConfigurationError, match="x1: amplitude"):
            qml.kernel_matrix(encoder, [[0.0, 0.0]], [], exact=True)
        with pytest.raises(polypus.ConfigurationError, match="x2: dataset is empty"):
            qml.kernel_matrix(encoder, [[1.0, 0.0]], [], infrastructure="qmio", seed=1)
        with pytest.raises(polypus.ConfigurationError, match="seed is not supported"):
            qml.kernel_matrix(encoder, [[1.0, 0.0]], infrastructure="qmio", seed=1)

    def test_a_seed_on_qmio_is_refused(self):
        with pytest.raises(polypus.ConfigurationError, match="seed is not supported"):
            qml.kernel_matrix(
                qml.Encoder(3).angle_encoder("ry"), ROWS, infrastructure="qmio", seed=1
            )
