"""Qiskit programs pinned as OpenQASM 3 fixtures (``data/qasm3``).

Each fixture is ``qiskit.qasm3.dumps`` of one construction below, committed
as text so that the tests read exactly what that Qiskit version writes. Test
runs never write them; regenerate them explicitly, with the pinned Qiskit
installed:

    python tests/python/qasm3_fixtures.py --write

``test_qasm3_interop.py`` checks that the constructions still produce the
pinned text under that version.
"""

import argparse
import pathlib
import sys

DATA = pathlib.Path(__file__).parent / "data" / "qasm3"

# The Qiskit version the fixtures were written with.
QISKIT_VERSION = "2.5.2"


def qaoa_layer():
    """A QED-C-style QAOA layer: ``ParameterVector`` names that Qiskit's
    exporter mangles lossily, ``rzz`` declared although Polypus has it built
    in, and the ``bit`` register declared before the ``qubit`` one."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    beta = ParameterVector("𝞫", 1)
    gamma = ParameterVector("𝞬", 1)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.rzz(-gamma[0], 0, 1)
    qc.rx(2 * beta[0], 0)
    qc.measure_all()
    return qc


def zz_feature_map():
    """Scalar ``qubit`` declarations, no measurement, a non-linear angle."""
    from qiskit.circuit.library import zz_feature_map

    return zz_feature_map(2, reps=1)


def efficient_su2():
    """Unicode identifiers, a bare ``U`` inside a declared gate, and a
    declared gate called with a free parameter and a constant."""
    from qiskit.circuit.library import efficient_su2

    return efficient_su2(2, reps=1).decompose()


FIXTURES = {
    "qaoa.qasm": qaoa_layer,
    "zz_feature_map.qasm": zz_feature_map,
    "efficient_su2.qasm": efficient_su2,
}


def dumps(name: str) -> str:
    """The text Qiskit writes for fixture ``name`` today."""
    from qiskit import qasm3

    return qasm3.dumps(FIXTURES[name]())


def pinned(name: str) -> str:
    """The committed text of fixture ``name``."""
    return (DATA / name).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="overwrite the pinned fixtures"
    )
    args = parser.parse_args()
    import qiskit

    if qiskit.__version__ != QISKIT_VERSION:
        print(
            f"the fixtures are pinned for Qiskit {QISKIT_VERSION}; "
            f"Qiskit {qiskit.__version__} is installed",
            file=sys.stderr,
        )
        return 1
    for name in FIXTURES:
        text = dumps(name)
        if args.write:
            (DATA / name).write_text(text, encoding="utf-8")
            print(f"wrote {DATA / name}")
        else:
            state = "unchanged" if text == pinned(name) else "DIFFERS"
            print(f"{name}: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
