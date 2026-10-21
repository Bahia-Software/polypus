"""Synthetic EQE-like calibration for the tests (the real snapshot may not be
published).

Six original qubits on a 2x3 grid,

    0 - 1 - 2
    |   |   |
    3 - 4 - 5

with qubit 2 out of service (no T1), so the backend exposes five compact qubits
0..4 = original 0, 1, 3, 4, 5, coupled as compact 1-0-2-3-4 plus 1-3:

    0 - 1
    |   |
    2 - 3 - 4

Original qubit 4 has no Ramsey T2, and qubit 5 a T2 above T1 (capped at T1,
see `device._cap_t2`).
"""

import json

NAN = float("nan")
# Original label -> compact label for the snapshot below.
COMPACT = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}
CZ_ORIGINAL = {"0,1": 0.990, "0,3": 0.985, "1,4": 0.993, "3,4": 0.980, "4,5": 0.970}
# Directed pairs in compact labels, as LRZ's file lists them.
COUPLING = [[1, 0], [0, 2], [3, 1], [3, 2], [4, 3]]


def snapshot(**overrides):
    raw = {
        "backend_name": "TEST",
        "timestamp": "2026-10-19T00:00:00Z",
        "T1": [40.0, 50.0, NAN, 45.0, 30.0, 10.0],
        # Values chosen so no gate is relaxation-limited under the default or
        # the slow-gate assumptions (that case has its own test).
        "T2": [20.0, 15.0, NAN, 18.0, NAN, 12.0],
        "T2_echo": [35.0, 45.0, NAN, 40.0, 25.0, 30.0],
        "1q_fidelity": [0.997, 0.996, NAN, 0.997, 0.995, 0.993],
        "readout_fidelity": [0.96, 0.95, 0.97, 0.98, 0.90, 0.93],
        # An edge to the out-of-service qubit is ignored, like any other.
        "two_q_fidelity": {**CZ_ORIGINAL, "1,2": 0.95},
    }
    raw.update(overrides)
    return raw


def write_files(tmp_path, raw=None, coupling=None, prefix=True):
    cal = tmp_path / "calibration.json"
    # json.dumps writes NaN as the bare token LRZ's file uses.
    cal.write_text(json.dumps(raw if raw is not None else snapshot()))
    cmap = tmp_path / "coupling_map.txt"
    pairs = json.dumps(coupling if coupling is not None else COUPLING)
    cmap.write_text(f"Backend TEST coupling map: {pairs}\n" if prefix else pairs)
    return cal, cmap
