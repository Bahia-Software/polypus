"""
No deadlock with ``polypus.log_to_python()`` installed, on every path that logs
from worker threads (issue #458).

With the bridge installed, a log record that Python wants waits for the GIL.
A path deadlocks if a thread holding the GIL waits for something (a lock, a
one-time initialiser, another thread) while the thread holding that waits for
the GIL to log. ``docs/ENGINEERING.md`` §3 and §9 rule both halves out; these
tests drive each such path with the bridge at ``DEBUG`` and a handler that
formats every record (``self.format(record)``, so a record's whole message is
built), each in a child interpreter bounded by a timeout: a deadlock is a
child that never finishes.

The child runs with ``POLYPUS_NO_AUTOCALIBRATE`` unset and its cache in a
temporary directory, so the first native run also calibrates the
gate-parallel threshold, which logs from inside a one-time initialiser
(``AUTOCALIBRATION``, the one exception ``docs/ENGINEERING.md`` §9 allows).
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("polypus")

pytestmark = pytest.mark.integration

# Far above what any child needs (seconds): only a hang reaches it.
_TIMEOUT_S = 120.0

_PRELUDE = r"""
import logging, threading
import polypus
from polypus import qml


class Formatting(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(threadName)s "
                "%(pathname)s:%(lineno)d %(message)s"
            )
        )
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


handler = Formatting()
top = logging.getLogger("polypus")
top.setLevel(logging.DEBUG)
top.addHandler(handler)
polypus.log_to_python()

QC = polypus.Circuit(2).ry(0, polypus.Param(0)).ry(1, polypus.Param(1)).cx(0, 1).measure_all()
RUN = dict(
    shots=16,
    dimensions=2,
    infrastructure="local",
    nodes=1,
    cores_per_qpu=1,
    backend="polypus",
    seed=1,
)


def cost(bitstring):
    return float(bitstring.count("1"))


def model():
    return qml.Model(
        qml.Encoder(3).angle_encoder("ry"),
        qml.Ansatz(3).hardware_efficient(1),
        qml.Readout([qml.Z(0)], decision="sign"),
    )


ROWS = [[0.1, 0.5, 0.9], [1.2, 0.3, 0.7], [0.4, 1.4, 0.2], [0.8, 0.6, 1.1]]
LABELS = [1, -1, 1, -1]
"""


def _run(body: str, tmp_path) -> None:
    """Run the prelude and ``body`` in a child; fail on a hang, a crash or a
    child that logged nothing."""
    source = _PRELUDE + textwrap.dedent(body) + '\nprint("DONE", len(handler.lines))\n'
    env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"))
    env.pop("POLYPUS_NO_AUTOCALIBRATE", None)
    try:
        child = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the child hung with the Python log bridge installed (killed after "
            f"{_TIMEOUT_S:.0f} s)\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        )
    output = f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    if child.returncode < 0:
        pytest.fail(
            f"the child was killed by {signal.Signals(-child.returncode).name}\n{output}"
        )
    assert child.returncode == 0, output
    assert "panicked" not in child.stderr, output
    done = [line for line in child.stdout.splitlines() if line.startswith("DONE ")]
    assert done, output
    assert int(done[0].split()[1]) > 0, f"no record reached Python\n{output}"


def test_train_with_an_observable_on_several_qpus(tmp_path):
    _run(
        """
        polypus.train(
            QC, polypus.DE(generations=3, population_size=6), n_qpus=2,
            expectation_function=cost, id="threads-observable", **RUN,
        )
        """,
        tmp_path,
    )


def test_train_with_a_callback(tmp_path):
    _run(
        """
        seen = []
        polypus.train(
            QC, polypus.DE(generations=3, population_size=6), n_qpus=1,
            expectation_function=cost, callback=seen.append, id="threads-callback",
            **RUN,
        )
        assert seen
        """,
        tmp_path,
    )


def test_train_with_an_external_optimizer(tmp_path):
    _run(
        """
        import random


        class Search:
            def __init__(self):
                self.random = random.Random(3)

            def ask(self):
                return [[self.random.uniform(0, 3) for _ in range(2)] for _ in range(4)]

            def tell(self, candidates, values):
                pass


        polypus.train(
            QC, polypus.External(Search(), 16), n_qpus=1,
            expectation_function=cost, id="threads-external", **RUN,
        )
        """,
        tmp_path,
    )


def test_run_quantum_circuit_on_aer(tmp_path):
    pytest.importorskip("qiskit_aer")
    _run(
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        polypus.run_quantum_circuit(
            qc, shots=64, infrastructure="local", backend="aer", n_qpus=2, seed=1,
        )
        """,
        tmp_path,
    )


def test_qml_fit_and_predict(tmp_path):
    _run(
        """
        m = model()
        fitted = m.fit(ROWS, LABELS, method=polypus.DE(generations=2, population_size=4),
                       loss="hinge", n_qpus=2, id="threads-fit")
        trained = m.bind(fitted.best_params)
        trained.predict(ROWS, n_qpus=2)
        trained.predict(ROWS, exact=True)
        """,
        tmp_path,
    )


def test_qml_primitives(tmp_path):
    _run(
        """
        m = model()
        params = [0.1 * (k + 1) for k in range(m.num_params)]
        qml.expval(m, ROWS, params, exact=True)
        qml.expval(m, ROWS, params, shots=64, seed=1)
        qml.jacobian(m, ROWS, params, exact=True)
        qml.vjp(m, ROWS, params, [[1.0]] * len(ROWS), exact=True)
        qml.kernel_matrix(qml.Encoder(3).angle_encoder("ry"), ROWS, exact=True)
        """,
        tmp_path,
    )


def test_statevector_and_backend_compatibility(tmp_path):
    _run(
        """
        circuit = polypus.Circuit(2).ry(0, polypus.Param(0)).ry(1, polypus.Param(1)).cx(0, 1)
        polypus.statevector(circuit, [0.3, 0.4])
        polypus.backend_compatibility(polypus.Circuit(2).h(0).cx(0, 1).measure_all())
        polypus.backend_compatibility(polypus.Circuit(2).h(0).measure_all(), resources=True)
        """,
        tmp_path,
    )


def test_two_python_threads_at_once(tmp_path):
    # One thread runs native work whose workers log while detached, the other
    # keeps calling `backend_compatibility`, which reaches the memory budget's
    # one-time detection with the GIL held: the case a record inside that
    # initialiser would deadlock. Both start together, so the process's
    # first-time initialisers race.
    _run(
        """
        barrier = threading.Barrier(2, timeout=30)
        errors = []


        def guarded(work):
            def run():
                try:
                    barrier.wait()
                    work()
                except BaseException as exc:
                    errors.append(repr(exc))
            return run


        def native():
            for k in range(5):
                polypus.statevector(polypus.Circuit(2).h(0).cx(0, 1), [])
                polypus.train(
                    QC, polypus.DE(generations=2, population_size=4), n_qpus=2,
                    expectation_function=cost, id=f"threads-both-{k}", **RUN,
                )


        def compatibility():
            for _ in range(200):
                polypus.backend_compatibility(
                    polypus.Circuit(2).h(0).cx(0, 1).measure_all(), resources=True
                )


        threads = [
            threading.Thread(target=guarded(native)),
            threading.Thread(target=guarded(compatibility)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=90)
        assert not any(t.is_alive() for t in threads), "a thread did not finish"
        assert not errors, errors
        """,
        tmp_path,
    )
