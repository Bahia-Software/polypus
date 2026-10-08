"""Reproduce the spec-vs-Qiskit phase table of ADR 0004.

Evaluates every gate of an OpenQASM 3 `stdgates.inc` (examples/stdgates.inc in
the spec repository, at a given tag) with the spec's own semantics -- the
builtin U(θ,φ,λ) matrix of source/language/gates.rst, gphase, and the ctrl@ /
inv@ / pow(k)@ modifiers -- and compares each full matrix with the Qiskit gate
of the same name (Qiskit's `Operator`), at several random parameter draws.

Usage: python 0004-phase-table.py <path/to/stdgates.inc>
Needs numpy and qiskit (tested with Qiskit 2.5.2).

Convention: a gate on qargs (a0, a1, ...) is an operator on qubits 0, 1, ...
with qubit 0 the least-significant bit, as Qiskit's Operator(gate) is.
pow(k)@ uses the principal branch (eigenphases in (-π, π]).
"""

import cmath
import math
import re
import sys

import numpy as np
from qiskit.circuit import library as lib
from qiskit.quantum_info import Operator


def spec_U(theta, phi, lam):
    e = cmath.exp
    return 0.5 * np.array(
        [
            [1 + e(1j * theta), -1j * e(1j * lam) * (1 - e(1j * theta))],
            [
                1j * e(1j * phi) * (1 - e(1j * theta)),
                e(1j * (phi + lam)) * (1 + e(1j * theta)),
            ],
        ]
    )


def embed(op, targets, n):
    """Full 2^n matrix of `op` (on len(targets) qubits, little-endian in its own
    qubit order) acting on `targets` of an n-qubit register."""
    k = len(targets)
    full = np.zeros((2**n, 2**n), dtype=complex)
    for col in range(2**n):
        sub_in = sum(((col >> t) & 1) << j for j, t in enumerate(targets))
        rest = col
        for t in targets:
            rest &= ~(1 << t)
        for sub_out in range(2**k):
            amp = op[sub_out, sub_in]
            if amp == 0:
                continue
            row = rest
            for j, t in enumerate(targets):
                row |= ((sub_out >> j) & 1) << t
            full[row, col] += amp
    return full


def controlled(op):
    """ctrl @ op: the new control is prepended to the argument list, so it is
    qubit 0 (least significant) of the result."""
    m = op.shape[0]
    out = np.zeros((2 * m, 2 * m), dtype=complex)
    for col in range(2 * m):
        if col & 1:
            for row in range(2 * m):
                if row & 1:
                    out[row, col] = op[row >> 1, col >> 1]
        else:
            out[col, col] = 1
    return out


def principal_power(op, k):
    vals, vecs = np.linalg.eig(op)
    # Unitary => normal; orthonormalise eigenvectors for degenerate eigenvalues.
    q, _ = np.linalg.qr(vecs)
    d = np.diag(q.conj().T @ op @ q)
    phases = np.angle(d)
    phases[np.isclose(phases, -math.pi)] = math.pi
    return q @ np.diag(np.exp(1j * k * phases)) @ q.conj().T


class Library:
    """A minimal interpreter for the `stdgates.inc` subset."""

    GATE = re.compile(r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^{]*)\{([^}]*)\}", re.S)

    def __init__(self, text):
        text = re.sub(r"//[^\n]*", "", text)
        self.gates = {}
        for name, params, qargs, body in self.GATE.findall(text):
            params = [p.strip() for p in params.split(",") if p.strip()]
            qargs = [q.strip() for q in qargs.split(",") if q.strip()]
            stmts = [s.strip() for s in body.split(";") if s.strip()]
            self.gates[name] = (params, qargs, stmts)

    def evaluate(self, expr, env):
        expr = expr.replace("π", "pi")
        return eval(expr, {"pi": math.pi, "__builtins__": {}}, env)

    def split_args(self, text):
        depth, cur, out = 0, "", []
        for ch in text:
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
                continue
            depth += ch == "("
            depth -= ch == ")"
            cur += ch
        if cur.strip():
            out.append(cur)
        return [a.strip() for a in out]

    def statement(self, stmt, env, qubits, n):
        """Unitary of one body statement as a full n-qubit matrix."""
        mods = []
        while "@" in stmt:
            head, stmt = stmt.split("@", 1)
            mods.append(head.strip())
            stmt = stmt.strip()
        m = re.match(r"(\w+)\s*", stmt)
        name, rest = m.group(1), stmt[m.end() :]
        args = None
        if rest.startswith("("):
            depth = 0
            for i, ch in enumerate(rest):
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0:
                    args, rest = rest[1:i], rest[i + 1 :]
                    break
        qargs = rest.strip()
        if name == "gphase" and args is None:  # `gphase -π/2` form
            args = qargs
            qargs = ""
        values = [self.evaluate(a, env) for a in self.split_args(args or "")]
        operands = [qubits[q.strip()] for q in qargs.split(",") if q.strip()]
        ncontrols = sum(1 for mod in mods if mod.startswith("ctrl"))
        base = self.base(name, values)
        for mod in reversed(mods):
            if mod == "ctrl":
                base = controlled(base)
            elif mod == "inv":
                base = base.conj().T
            elif mod.startswith("pow"):
                k = self.evaluate(mod[mod.index("(") + 1 : mod.rindex(")")], env)
                base = principal_power(base, k)
            else:
                raise ValueError(mod)
        if base.shape == (1, 1):  # gphase in scope of n qubits
            return base[0, 0] * np.eye(2**n)
        assert len(operands) == int(math.log2(base.shape[0])), (stmt, ncontrols)
        return embed(base, operands, n)

    def base(self, name, values):
        if name == "U":
            return spec_U(*values)
        if name == "gphase":
            return np.array([[cmath.exp(1j * values[0])]])
        return self.matrix(name, values)

    def matrix(self, name, values):
        params, qargs, stmts = self.gates[name]
        env = dict(zip(params, values))
        env.update({"pi": math.pi})
        n = len(qargs)
        qubits = {q: i for i, q in enumerate(qargs)}
        total = np.eye(2**n, dtype=complex)
        for stmt in stmts:
            total = self.statement(stmt, env, qubits, n) @ total
        return total


QISKIT = {
    "p": lib.PhaseGate,
    "x": lib.XGate,
    "y": lib.YGate,
    "z": lib.ZGate,
    "h": lib.HGate,
    "s": lib.SGate,
    "sdg": lib.SdgGate,
    "t": lib.TGate,
    "tdg": lib.TdgGate,
    "sx": lib.SXGate,
    "rx": lib.RXGate,
    "ry": lib.RYGate,
    "rz": lib.RZGate,
    "cx": lib.CXGate,
    "cy": lib.CYGate,
    "cz": lib.CZGate,
    "cp": lib.CPhaseGate,
    "crx": lib.CRXGate,
    "cry": lib.CRYGate,
    "crz": lib.CRZGate,
    "ch": lib.CHGate,
    "swap": lib.SwapGate,
    "ccx": lib.CCXGate,
    "cswap": lib.CSwapGate,
    "cu": lib.CUGate,
    "CX": lib.CXGate,
    "phase": lib.PhaseGate,
    "cphase": lib.CPhaseGate,
    "id": lib.IGate,
    "u1": lib.U1Gate,
    "u2": lib.U2Gate,
    "u3": lib.U3Gate,
    "U": lib.UGate,
}

# Candidate global factors (spec = factor * Qiskit), as functions of the
# gate's parameters in declaration order.
CANDIDATES = {
    "e^{iθ/2}": lambda p: cmath.exp(1j * p[0] / 2),
    "e^{-i(φ+λ)/2}": lambda p: cmath.exp(-1j * (p[-2] + p[-1]) / 2),
}


def classify(name, lib_, rng, draws=6):
    params = lib_.gates[name][0] if name != "U" else ["θ", "φ", "λ"]
    verdicts = set()
    worst = 0.0
    for _ in range(draws):
        values = list(rng.uniform(-2 * math.pi, 2 * math.pi, size=len(params)))
        spec = lib_.base(name, values)
        qk = Operator(QISKIT[name](*values)).data
        diff = np.max(np.abs(spec - qk))
        if diff < 1e-12:
            verdicts.add("exactly equal")
            worst = max(worst, diff)
            continue
        idx = np.unravel_index(np.argmax(np.abs(qk)), qk.shape)
        factor = spec[idx] / qk[idx]
        if np.max(np.abs(spec - factor * qk)) > 1e-12:
            verdicts.add(f"NOT a global phase (max |Δ| = {diff:.3g})")
            continue
        matched = [
            label for label, f in CANDIDATES.items() if abs(f(values) - factor) < 1e-12
        ]
        verdicts.add(
            "global phase "
            + (matched[0] if matched else f"{factor:.6f} (no candidate)")
        )
        worst = max(
            worst,
            np.max(np.abs(spec - CANDIDATES[matched[0]](values) * qk))
            if matched
            else 0,
        )
    return " / ".join(sorted(verdicts)), worst


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        lib_ = Library(f.read())
    rng = np.random.default_rng(20260928)
    rows = {}
    for name in ["U", *lib_.gates]:
        verdict, worst = classify(name, lib_, rng)
        rows.setdefault(verdict, []).append(name)
        print(f"{name:7s} {verdict}   (max residual {worst:.1e})")
    print()
    print("| Result | Gates |")
    print("|---|---|")
    for verdict, names in rows.items():
        print(f"| {verdict} | {' '.join('`' + n + '`' for n in names)} |")


if __name__ == "__main__":
    main()
