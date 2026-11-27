"""The seeds Polypus derives, written again in Python so the tests can say
which seed each call and each CUNQA QPU must receive (contract C-7). A plain
module, not a test file: imported as ``from seed_reference import ...``.

``stream_seed`` is ``polypus_backend::stream_seed``, built on the
workspace's one SplitMix64, ``polypus_circuit::rng::SplitMix64``
(``crates/polypus-circuit/src/rng.rs``, whose ``GAMMA`` and ``mix`` the
constants and ``_mix`` below copy): a change to that generator must change
this oracle too. Its first output for seed 0 is pinned in
``crates/polypus-backend/src/seed.rs`` and in ``test_cunqa_run_flow.py``.
"""

MASK = 2**63 - 1
_U64 = 2**64 - 1
_GAMMA = 0x9E3779B97F4A7C15


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return z ^ (z >> 31)


def stream_seed(seed, index):
    """Output ``index`` of the SplitMix64 stream of ``seed``."""
    return _mix((_mix(seed) + (index + 1) * _GAMMA) & _U64)


def call_seed(seed, call):
    """The seed of call ``call`` of a run seeded ``seed`` (``CallSeeds``)."""
    return seed if call == 0 else stream_seed(seed, call)


def qpu_seed(seed, qpu):
    """The seed CUNQA runs QPU ``qpu`` of a call seeded ``seed`` with (#635):
    the call's seed on QPU 0, and on the others output ``qpu`` of a stream
    started at ``stream_seed(seed, 0)`` (an index ``CallSeeds`` never uses),
    masked to 63 bits."""
    return (seed if qpu == 0 else stream_seed(stream_seed(seed, 0), qpu)) & MASK
