//! Gate matrices and diagonal phase factors, in the **Qiskit little-endian**
//! convention (qubit 0 is the least-significant bit of the amplitude index).
//!
//! Dense single-qubit gates return a `[[C64; 2]; 2]` matrix; diagonal gates
//! return their two phase factors `(d0, d1)` for basis bit 0 and 1. Two-qubit
//! diagonal gates return `[C64; 4]` indexed by `(bit_q1 << 1) | bit_q0`; the
//! only dense two-qubit gate (`rxx`) returns a `[[C64; 4]; 4]` matrix in the
//! `|b_q1 b_q0>` basis ordered `00, 01, 10, 11`.

use crate::C64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

#[inline]
fn re(x: f64) -> C64 {
    C64::new(x, 0.0)
}

// ── Dense single-qubit gates ─────────────────────────────────────────────

/// Hadamard.
pub(crate) fn h() -> [[C64; 2]; 2] {
    let s = FRAC_1_SQRT_2;
    [[re(s), re(s)], [re(s), re(-s)]]
}

/// Pauli-X.
pub(crate) fn x() -> [[C64; 2]; 2] {
    [[re(0.0), re(1.0)], [re(1.0), re(0.0)]]
}

/// Pauli-Y.
pub(crate) fn y() -> [[C64; 2]; 2] {
    [
        [re(0.0), C64::new(0.0, -1.0)],
        [C64::new(0.0, 1.0), re(0.0)],
    ]
}

/// X-rotation `Rx(θ) = cos(θ/2) I − i sin(θ/2) X`.
pub(crate) fn rx(theta: f64) -> [[C64; 2]; 2] {
    let c = re((theta / 2.0).cos());
    let ns = C64::new(0.0, -(theta / 2.0).sin());
    [[c, ns], [ns, c]]
}

/// Y-rotation `Ry(θ)`.
pub(crate) fn ry(theta: f64) -> [[C64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[re(c), re(-s)], [re(s), re(c)]]
}

/// Generic single-qubit gate `U(θ, φ, λ)` (Qiskit `u`/`u3`), no extra global
/// phase:
/// `[[cos θ/2, −e^{iλ} sin θ/2], [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]]`.
pub(crate) fn u(theta: f64, phi: f64, lam: f64) -> [[C64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let e_phi = C64::from_polar(1.0, phi);
    let e_lam = C64::from_polar(1.0, lam);
    let e_phi_lam = C64::from_polar(1.0, phi + lam);
    [
        [re(c), e_lam.scale(-s)],
        [e_phi.scale(s), e_phi_lam.scale(c)],
    ]
}

// ── Diagonal single-qubit gates: (factor for bit 0, factor for bit 1) ─────

/// Pauli-Z.
pub(crate) fn z() -> (C64, C64) {
    (re(1.0), re(-1.0))
}

/// Phase gate S = diag(1, i).
pub(crate) fn s() -> (C64, C64) {
    (re(1.0), C64::new(0.0, 1.0))
}

/// S† = diag(1, −i).
pub(crate) fn sdg() -> (C64, C64) {
    (re(1.0), C64::new(0.0, -1.0))
}

/// T = diag(1, e^{iπ/4}).
pub(crate) fn t() -> (C64, C64) {
    (re(1.0), C64::from_polar(1.0, FRAC_PI_4))
}

/// T† = diag(1, e^{−iπ/4}).
pub(crate) fn tdg() -> (C64, C64) {
    (re(1.0), C64::from_polar(1.0, -FRAC_PI_4))
}

/// Z-rotation `Rz(θ) = diag(e^{−iθ/2}, e^{+iθ/2})`.
pub(crate) fn rz(theta: f64) -> (C64, C64) {
    (
        C64::from_polar(1.0, -theta / 2.0),
        C64::from_polar(1.0, theta / 2.0),
    )
}

// ── Diagonal two-qubit gates: indexed by (bit_q1 << 1) | bit_q0 ───────────

/// Controlled-Z: `−1` only when both qubits are `1`. Symmetric in its operands.
pub(crate) fn cz_diag() -> [C64; 4] {
    [re(1.0), re(1.0), re(1.0), re(-1.0)]
}

/// `Rzz(θ) = exp(−i θ/2 · Z⊗Z)`: `e^{−iθ/2}` when the two bits are equal,
/// `e^{+iθ/2}` when they differ. Symmetric in its operands.
pub(crate) fn rzz_diag(theta: f64) -> [C64; 4] {
    let equal = C64::from_polar(1.0, -theta / 2.0);
    let differ = C64::from_polar(1.0, theta / 2.0);
    // index = (bit_q1 << 1) | bit_q0: 00 equal, 01 differ, 10 differ, 11 equal.
    [equal, differ, differ, equal]
}

// ── Dense two-qubit gate ─────────────────────────────────────────────────

/// `Rxx(θ) = cos(θ/2) I − i sin(θ/2) · X⊗X`, in the basis `00, 01, 10, 11`.
pub(crate) fn rxx(theta: f64) -> [[C64; 4]; 4] {
    let c = re((theta / 2.0).cos());
    let ns = C64::new(0.0, -(theta / 2.0).sin());
    let z0 = re(0.0);
    [
        [c, z0, z0, ns],
        [z0, c, ns, z0],
        [z0, ns, c, z0],
        [ns, z0, z0, c],
    ]
}

pub(crate) fn cp_diag(theta: f64) -> [C64; 4] {
    let diag = C64::from_polar(1.0, theta);
    [re(1.0), re(1.0), re(1.0), diag]
}