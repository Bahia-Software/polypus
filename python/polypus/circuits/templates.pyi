# Typing stubs for `polypus.circuits.templates`, created by the compiled
# extension at import time. Mirrors `qft` in
# `crates/polypus/src/bindings/circuit.rs`; see `../__init__.pyi` for how the
# stubs are checked.

from .. import Circuit

__all__ = ["qft"]

def qft(num_qubits: int, inverse: bool = False, swaps: bool = True) -> Circuit: ...
