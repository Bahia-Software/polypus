//! In-place gate-application kernels.
//!
//! Every kernel rewrites the amplitude buffer in place — no per-gate
//! allocation. They share one trick: enumerating the `2^(n-1)` (or `2^(n-2)`)
//! sub-states of the *untouched* qubits via **bit insertion**, which yields the
//! disjoint groups of amplitudes a gate mixes.
//!
//! For a single-qubit gate on target `t`, iterating `g ∈ [0, 2^(n-1))` and
//! forming
//!
//! ```text
//! low = (1 << t) − 1
//! i0  = (g & low) | ((g & !low) << 1)   // bit t cleared
//! i1  = i0 | (1 << t)                   // bit t set
//! ```
//!
//! visits every amplitude pair `(i0, i1)` exactly once. Because the pairs are
//! disjoint, the loop is embarrassingly parallel.
//!
//! Five families:
//! - **dense 1-qubit** ([`apply_1q`]): a full 2×2 matrix;
//! - **diagonal** ([`apply_diagonal_1q`], [`apply_diagonal_2q`]): a per-index
//!   phase, needing no pairing — the cheapest path;
//! - **fused diagonal** ([`apply_diagonal_run`]): a whole run of consecutive
//!   diagonal operators collapsed into one pass, multiplying each amplitude by
//!   the product of every op's per-index phase. Diagonal operators commute
//!   regardless of which qubits they touch, so this product is order-independent
//!   — the point is to pay one traversal of the (memory-bandwidth-bound) buffer
//!   instead of one per gate on a deep diagonal stretch (a QAOA cost layer, a
//!   Trotterized `ZZ`/`RZ` evolution, a QFT phase column);
//! - **controlled 1-qubit** ([`apply_controlled_1q`]): the 2×2 is applied to a
//!   pair only when the control bit (identical across the pair) is set;
//! - **dense 2-qubit** ([`apply_2q`]): a full 4×4 matrix over groups of four.
//!
//! When the `parallel` feature is on and `parallel` is `true`, the dense
//! kernels distribute their outer loop across rayon. The closures share the
//! buffer through a raw pointer; see the `SAFETY` notes for why that is sound.
//! The diagonal kernels — the fused one included — instead use `par_iter_mut`,
//! which needs no `unsafe`: each index's write depends only on that index.

use crate::C64;

/// Apply 2×2 matrix `m` to amplitudes `(a, b)`.
#[inline]
fn combine_1q(m: &[[C64; 2]; 2], a: C64, b: C64) -> (C64, C64) {
    (m[0][0] * a + m[0][1] * b, m[1][0] * a + m[1][1] * b)
}

/// Dense single-qubit gate on target `t`.
pub(crate) fn apply_1q(data: &mut [C64], n: usize, t: usize, m: &[[C64; 2]; 2], parallel: bool) {
    debug_assert!(t < n);
    let half = 1usize << (n - 1);
    let low = (1usize << t) - 1;
    let bit = 1usize << t;

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let base = data.as_mut_ptr() as usize;
        (0..half).into_par_iter().for_each(|g| {
            let i0 = (g & low) | ((g & !low) << 1);
            let i1 = i0 | bit;
            // SAFETY: g ↦ {i0, i1} is injective and the image pairs are pairwise
            // disjoint (every amplitude lies in exactly one pair), so no two
            // iterations alias the same index — no data race. Indices are < 2^n
            // = data.len(). `base` is the buffer's own pointer.
            unsafe {
                let p = base as *mut C64;
                let a = *p.add(i0);
                let b = *p.add(i1);
                let (na, nb) = combine_1q(m, a, b);
                *p.add(i0) = na;
                *p.add(i1) = nb;
            }
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for g in 0..half {
        let i0 = (g & low) | ((g & !low) << 1);
        let i1 = i0 | bit;
        let (na, nb) = combine_1q(m, data[i0], data[i1]);
        data[i0] = na;
        data[i1] = nb;
    }
}

/// Diagonal single-qubit gate: multiply each amplitude by `d0` or `d1`
/// depending on bit `t`. No pairing required.
pub(crate) fn apply_diagonal_1q(data: &mut [C64], t: usize, d0: C64, d1: C64, parallel: bool) {
    let bit = 1usize << t;

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        data.par_iter_mut().enumerate().for_each(|(i, amp)| {
            *amp *= if i & bit == 0 { d0 } else { d1 };
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for (i, amp) in data.iter_mut().enumerate() {
        *amp *= if i & bit == 0 { d0 } else { d1 };
    }
}

/// Diagonal two-qubit gate: multiply each amplitude by `diag[(b_q1 << 1) | b_q0]`.
/// Covers `Cz` and `Rzz`, both symmetric in their operands.
pub(crate) fn apply_diagonal_2q(
    data: &mut [C64],
    q0: usize,
    q1: usize,
    diag: [C64; 4],
    parallel: bool,
) {
    let b0 = 1usize << q0;
    let b1 = 1usize << q1;

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        data.par_iter_mut().enumerate().for_each(|(i, amp)| {
            let k = ((usize::from(i & b1 != 0)) << 1) | usize::from(i & b0 != 0);
            *amp *= diag[k];
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for (i, amp) in data.iter_mut().enumerate() {
        let k = ((usize::from(i & b1 != 0)) << 1) | usize::from(i & b0 != 0);
        *amp *= diag[k];
    }
}

/// One operator in a fused diagonal run, reduced to the bit masks and phase
/// factors its per-index contribution needs.
///
/// Built from a [`GateInstruction`](polypus_circuit::GateInstruction) by the
/// statevector layer, mirroring the bit extraction in [`apply_diagonal_1q`] /
/// [`apply_diagonal_2q`]; consumed by [`apply_diagonal_run`]. The stored values
/// are bit *masks* (`1 << qubit`), not qubit indices.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DiagonalOp {
    /// 1-qubit diagonal: multiply by `d0` when qubit bit `bit` is clear, `d1`
    /// when it is set.
    One { bit: usize, d0: C64, d1: C64 },
    /// 2-qubit diagonal: multiply by `diag[(b1_set << 1) | b0_set]`, exactly the
    /// indexing [`apply_diagonal_2q`] uses.
    Two {
        b0: usize,
        b1: usize,
        diag: [C64; 4],
    },
}

/// The combined factor [`apply_diagonal_run`] multiplies amplitude `i` by: the
/// product of every op's individual per-index phase. Order-independent, because
/// diagonal operators commute.
#[inline]
fn diagonal_run_factor(ops: &[DiagonalOp], i: usize) -> C64 {
    let mut factor = C64::new(1.0, 0.0);
    for op in ops {
        match op {
            DiagonalOp::One { bit, d0, d1 } => {
                factor *= if i & bit == 0 { *d0 } else { *d1 };
            }
            DiagonalOp::Two { b0, b1, diag } => {
                let k = ((usize::from(i & b1 != 0)) << 1) | usize::from(i & b0 != 0);
                factor *= diag[k];
            }
        }
    }
    factor
}

/// Fused diagonal run: apply every op in `ops` in a **single** pass over `data`,
/// multiplying each amplitude by the product of the ops' per-index factors (see
/// [`diagonal_run_factor`]).
///
/// Numerically identical to applying the same ops one at a time through
/// [`apply_diagonal_1q`] / [`apply_diagonal_2q`], but touches the buffer once
/// instead of once per op — which is the whole optimization on a deep diagonal
/// stretch, where the per-gate passes are the redundant work.
pub(crate) fn apply_diagonal_run(data: &mut [C64], ops: &[DiagonalOp], parallel: bool) {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        data.par_iter_mut().enumerate().for_each(|(i, amp)| {
            *amp *= diagonal_run_factor(ops, i);
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for (i, amp) in data.iter_mut().enumerate() {
        *amp *= diagonal_run_factor(ops, i);
    }
}

/// Controlled single-qubit gate: apply `m` to the `(i0, i1)` pair on target `t`
/// only when control bit `c` is set. `c` is identical across the pair, so the
/// decision is made once per pair. Used for `Cx`.
pub(crate) fn apply_controlled_1q(
    data: &mut [C64],
    n: usize,
    c: usize,
    t: usize,
    m: &[[C64; 2]; 2],
    parallel: bool,
) {
    debug_assert!(c < n && t < n && c != t);
    let half = 1usize << (n - 1);
    let low = (1usize << t) - 1;
    let bit = 1usize << t;
    let cbit = 1usize << c;

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let base = data.as_mut_ptr() as usize;
        (0..half).into_par_iter().for_each(|g| {
            let i0 = (g & low) | ((g & !low) << 1);
            if i0 & cbit == 0 {
                return;
            }
            let i1 = i0 | bit;
            // SAFETY: same disjoint-pairs argument as `apply_1q`; the control
            // filter only skips pairs, it never widens the touched set.
            unsafe {
                let p = base as *mut C64;
                let a = *p.add(i0);
                let b = *p.add(i1);
                let (na, nb) = combine_1q(m, a, b);
                *p.add(i0) = na;
                *p.add(i1) = nb;
            }
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for g in 0..half {
        let i0 = (g & low) | ((g & !low) << 1);
        if i0 & cbit == 0 {
            continue;
        }
        let i1 = i0 | bit;
        let (na, nb) = combine_1q(m, data[i0], data[i1]);
        data[i0] = na;
        data[i1] = nb;
    }
}

/// Compute the four amplitude indices a two-qubit gate on `(q0, q1)` mixes for
/// sub-state `g`, ordered `[00, 01, 10, 11]` by `(bit_q1, bit_q0)`.
#[inline]
fn two_qubit_indices(g: usize, q0: usize, q1: usize) -> [usize; 4] {
    let (lo, hi) = (q0.min(q1), q0.max(q1));
    let low_mask = (1usize << lo) - 1;
    let a = (g & low_mask) | ((g & !low_mask) << 1);
    let high_mask = (1usize << hi) - 1;
    let i00 = (a & high_mask) | ((a & !high_mask) << 1);
    let b0 = 1usize << q0;
    let b1 = 1usize << q1;
    [i00, i00 | b0, i00 | b1, i00 | b0 | b1]
}

/// Dense two-qubit gate on `(q0, q1)`. Used for `Rxx`.
pub(crate) fn apply_2q(
    data: &mut [C64],
    n: usize,
    q0: usize,
    q1: usize,
    m: &[[C64; 4]; 4],
    parallel: bool,
) {
    debug_assert!(q0 < n && q1 < n && q0 != q1);
    let quarter = 1usize << (n - 2);

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let base = data.as_mut_ptr() as usize;
        (0..quarter).into_par_iter().for_each(|g| {
            let idx = two_qubit_indices(g, q0, q1);
            // SAFETY: across g the index quadruples are pairwise disjoint (each
            // amplitude belongs to exactly one quadruple), so no two iterations
            // alias. All indices are < 2^n = data.len().
            unsafe {
                let p = base as *mut C64;
                let v = [
                    *p.add(idx[0]),
                    *p.add(idx[1]),
                    *p.add(idx[2]),
                    *p.add(idx[3]),
                ];
                for (row, &target) in m.iter().zip(idx.iter()) {
                    *p.add(target) = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
                }
            }
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for g in 0..quarter {
        let idx = two_qubit_indices(g, q0, q1);
        let v = [data[idx[0]], data[idx[1]], data[idx[2]], data[idx[3]]];
        for (row, &target) in m.iter().zip(idx.iter()) {
            data[target] = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates;
    use crate::simulator::MAX_FUSED_DIAGONAL_RUN;

    /// Distinct, non-zero amplitudes across the whole buffer, so a dropped or
    /// mis-indexed factor shows up as a mismatch rather than hiding in a zero.
    fn sample_data(n: usize) -> Vec<C64> {
        (0..(1usize << n))
            .map(|i| C64::new(i as f64 + 1.0, i as f64 * 0.5 - 1.0))
            .collect()
    }

    fn one(q: usize, factors: (C64, C64)) -> DiagonalOp {
        DiagonalOp::One {
            bit: 1 << q,
            d0: factors.0,
            d1: factors.1,
        }
    }

    fn two(q0: usize, q1: usize, diag: [C64; 4]) -> DiagonalOp {
        DiagonalOp::Two {
            b0: 1 << q0,
            b1: 1 << q1,
            diag,
        }
    }

    /// Apply `ops` one at a time through the per-gate diagonal kernels — the
    /// reference behaviour the fused kernel must reproduce exactly.
    fn apply_sequential(data: &mut [C64], ops: &[DiagonalOp], parallel: bool) {
        for op in ops {
            match op {
                DiagonalOp::One { bit, d0, d1 } => {
                    apply_diagonal_1q(data, bit.trailing_zeros() as usize, *d0, *d1, parallel);
                }
                DiagonalOp::Two { b0, b1, diag } => {
                    apply_diagonal_2q(
                        data,
                        b0.trailing_zeros() as usize,
                        b1.trailing_zeros() as usize,
                        *diag,
                        parallel,
                    );
                }
            }
        }
    }

    /// The fused kernel must equal the sequential per-gate kernels, on both the
    /// sequential and (when compiled in) the parallel path.
    fn assert_fused_matches_sequential(n: usize, ops: &[DiagonalOp]) {
        let mut paths = vec![false];
        if cfg!(feature = "parallel") {
            paths.push(true);
        }
        for parallel in paths {
            let mut sequential = sample_data(n);
            let mut fused = sequential.clone();
            apply_sequential(&mut sequential, ops, parallel);
            apply_diagonal_run(&mut fused, ops, parallel);
            for (i, (a, b)) in sequential.iter().zip(fused.iter()).enumerate() {
                assert!(
                    (a - b).norm() < 1e-12,
                    "amplitude {i} differs (parallel={parallel}): sequential {a} vs fused {b}"
                );
            }
        }
    }

    #[test]
    fn fused_matches_sequential_on_a_same_qubit_run() {
        let ops = [
            one(1, gates::z()),
            one(1, gates::s()),
            one(1, gates::rz(0.7)),
            one(1, gates::t()),
            one(1, gates::tdg()),
        ];
        assert_fused_matches_sequential(4, &ops);
    }

    #[test]
    fn fused_matches_sequential_on_a_disjoint_qubit_run() {
        let ops = [
            one(0, gates::z()),
            one(2, gates::rz(1.1)),
            one(4, gates::s()),
            one(3, gates::sdg()),
        ];
        assert_fused_matches_sequential(5, &ops);
    }

    #[test]
    fn fused_matches_sequential_on_an_overlapping_1q_and_2q_run() {
        let ops = [
            one(1, gates::rz(0.3)),
            two(1, 3, gates::cz_diag()),
            two(1, 3, gates::rzz_diag(0.9)),
            two(0, 1, gates::cp_diag(1.3)),
            one(3, gates::t()),
            two(3, 1, gates::rzz_diag(-0.4)),
        ];
        assert_fused_matches_sequential(5, &ops);
    }

    #[test]
    fn fused_matches_sequential_at_the_cap_length() {
        let n = 6;
        let mut ops = Vec::with_capacity(MAX_FUSED_DIAGONAL_RUN);
        for k in 0..MAX_FUSED_DIAGONAL_RUN {
            let q = k % n;
            let op = match k % 4 {
                0 => one(q, gates::z()),
                1 => one(q, gates::rz(0.1 * k as f64)),
                2 => two(q, (q + 1) % n, gates::cz_diag()),
                _ => two(q, (q + 2) % n, gates::cp_diag(0.05 * k as f64)),
            };
            ops.push(op);
        }
        assert_eq!(ops.len(), MAX_FUSED_DIAGONAL_RUN);
        assert_fused_matches_sequential(n, &ops);
    }
}
