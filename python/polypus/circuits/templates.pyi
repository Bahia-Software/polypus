# Typing stubs for `polypus.circuits.templates`, created by the compiled
# extension at import time. Mirrors `qft` and `shor_order_finding` in
# `crates/polypus/src/bindings/circuit.rs`; see `../__init__.pyi` for how the
# stubs are checked.

from typing import Optional

from .. import Circuit

__all__ = ["qft", "shor_order_finding"]

def qft(num_qubits: int, inverse: bool = False, swaps: bool = True) -> Circuit: ...
def shor_order_finding(
    a: int, modulus: int, counting_qubits: Optional[int] = None
) -> Circuit: ...
