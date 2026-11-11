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
//!   diagonal operators applied by carrying each cache tile through every op,
//!   multiplying each amplitude by the product of every op's per-index phase.
//!   Diagonal operators commute regardless of which qubits they touch, so both
//!   the product and the reordering it allows are sound — the point is to pay one
//!   main-memory traversal of the (memory-bandwidth-bound) buffer instead of one
//!   per gate on a deep diagonal stretch (a QAOA cost layer, a Trotterized
//!   `ZZ`/`RZ` evolution, a QFT phase column). Tiling — rather than a single
//!   loop with the op loop inside it — is what makes that pay off; see
//!   [`DIAGONAL_TILE`];
//! - **controlled 1-qubit** ([`apply_controlled_1q`]): the 2×2 is applied only
//!   to the pairs whose control bit is set, enumerated directly over the
//!   `2^(n-2)` sub-states of the other qubits;
//! - **dense 2-qubit** ([`apply_2q`]): a full 4×4 matrix over groups of four.
//!
//! When the `parallel` feature is on and `parallel` is `true`, the dense
//! kernels distribute their outer loop across rayon. The closures share the
//! buffer through a raw pointer; see the `SAFETY` notes for why that is sound.
//! Their operand and buffer-length preconditions are checked with `assert!`
//! (not `debug_assert!`), so a violation panics in release instead of writing
//! out of bounds.
//! The diagonal kernels — the fused one included — instead use `par_iter_mut`,
//! which needs no `unsafe`: each index's write depends only on that index.

use crate::C64;

/// Apply 2×2 matrix `m` to amplitudes `(a, b)`.
#[inline]
fn combine_1q(m: &[[C64; 2]; 2], a: C64, b: C64) -> (C64, C64) {
    (m[0][0] * a + m[0][1] * b, m[1][0] * a + m[1][1] * b)
}

/// The amplitude pairs `(i0, i1)` a 2×2 update on target `t` visits, one per
/// sub-state `g ∈ [0, count)`, as bit masks computed once so the loop body is
/// the same branch-free arithmetic for both kernels. Built only by
/// [`Pairs::target`] and [`Pairs::controlled`], which check the operands.
#[derive(Clone, Copy)]
struct Pairs {
    count: usize,
    /// `i0` is `g` with a zero inserted below `low` and then below `high`
    /// (bit insertion, see the module docs), with `set` ORed in.
    low: usize,
    high: usize,
    set: usize,
    /// `1 << t`: `i1 = i0 | bit`.
    bit: usize,
}

impl Pairs {
    /// Every pair on target `t` of an `n`-qubit buffer: `2^(n-1)` of them,
    /// only bit `t` inserted (`high` = all ones makes the second insertion a
    /// no-op).
    fn target(n: usize, t: usize) -> Self {
        // Real `assert!`s, not `debug_assert!`s: the parallel path writes
        // through a raw pointer, so these O(1) checks are what keep a bad index
        // from being undefined behaviour in release. Callers validate first and
        // return a typed error; reaching these is a bug.
        assert!(t < n);
        Pairs {
            count: 1usize << (n - 1),
            low: (1usize << t) - 1,
            high: usize::MAX,
            set: 0,
            bit: 1usize << t,
        }
    }

    /// Only the pairs on target `t` whose control bit `c` is set: `2^(n-2)` of
    /// them, both fixed bits inserted as in [`two_qubit_indices`] and bit `c`
    /// set.
    fn controlled(n: usize, c: usize, t: usize) -> Self {
        // See `Pairs::target`.
        assert!(c < n && t < n && c != t);
        let (lo, hi) = (c.min(t), c.max(t));
        Pairs {
            count: 1usize << (n - 2),
            low: (1usize << lo) - 1,
            high: (1usize << hi) - 1,
            set: 1usize << c,
            bit: 1usize << t,
        }
    }

    /// The pair for sub-state `g`.
    #[inline]
    fn pair(self, g: usize) -> (usize, usize) {
        let a = (g & self.low) | ((g & !self.low) << 1);
        let i0 = (a & self.high) | ((a & !self.high) << 1) | self.set;
        (i0, i0 | self.bit)
    }
}

/// Apply 2×2 matrix `m` to every pair of `pairs` on an `n`-qubit buffer: the
/// shared body of [`apply_1q`] and [`apply_controlled_1q`].
fn apply_pairs(data: &mut [C64], n: usize, pairs: Pairs, m: &[[C64; 2]; 2], parallel: bool) {
    // See `Pairs::target`.
    assert!(data.len() == 1usize << n);
    let count = pairs.count;

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let base = data.as_mut_ptr() as usize;
        (0..count).into_par_iter().for_each(|g| {
            let (i0, i1) = pairs.pair(g);
            // The `;` after the block makes it a statement, so this `allow` can apply.
            #[allow(unsafe_code)] // authorised `unsafe` site: docs/ENGINEERING.md §5
            // SAFETY: inserting a zero bit is injective, and so is ORing `set`
            // into it (for `controlled`, bit `c` is the other inserted zero; for
            // `target`, `set` is 0). Bit `t` is an inserted zero that `set`
            // never touches (c ≠ t), so g ↦ i0 is injective, every i0 has bit
            // `t` clear and i1 = i0 | (1 << t) has it set. No i0 equals another
            // pair's i1, so the pairs are pairwise disjoint and no two
            // iterations alias the same amplitude — no data race. `Pairs`'
            // constructors assert t, c < n, so with g < count both indices keep
            // bits ≥ n clear: they are < 2^n = data.len(), asserted above.
            // `base` is the buffer's own pointer.
            unsafe {
                let p = base as *mut C64;
                let a = *p.add(i0);
                let b = *p.add(i1);
                let (na, nb) = combine_1q(m, a, b);
                *p.add(i0) = na;
                *p.add(i1) = nb;
            };
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for g in 0..count {
        let (i0, i1) = pairs.pair(g);
        let (na, nb) = combine_1q(m, data[i0], data[i1]);
        data[i0] = na;
        data[i1] = nb;
    }
}

/// Dense single-qubit gate on target `t`.
pub(crate) fn apply_1q(data: &mut [C64], n: usize, t: usize, m: &[[C64; 2]; 2], parallel: bool) {
    apply_pairs(data, n, Pairs::target(n, t), m, parallel);
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

/// Amplitudes carried through the whole run together before moving on — the
/// cache-blocking tile that makes fusion pay off.
///
/// A naive fused pass (one loop over the buffer, an op loop *inside* it) loses to
/// the per-gate kernels: its inner op loop branches on a runtime-length,
/// heterogeneous op list, which the compiler cannot vectorize, so it runs at
/// scalar speed and — being compute-bound — never cashes in the memory-traffic it
/// saves. Tiling fixes both. Each tile is multiplied by every op while it stays
/// hot in cache, so the run costs ~one main-memory traversal rather than one per
/// op (the point of fusing), and each op is applied to the tile by a *monomorphic*
/// inner loop ([`block_diagonal_1q`] / [`block_diagonal_2q`], the same body as the
/// single-gate kernels) that vectorizes exactly as they do.
///
/// 4096 amplitudes = 64 KiB sits in L2 on every target. The pass is insensitive
/// to the exact value across 512–32768 (measured), so this is a locality knob,
/// not a hardware-tuned constant.
///
/// The end-to-end speedup fusion buys over the gate-by-gate path is measured by
/// `benchmarks/bench_fusion.py` (the `qft`/`trotter` families exercise these
/// diagonal runs) — `fusion=False` vs `fusion=True` on the same build.
const DIAGONAL_TILE: usize = 4096;

/// Multiply one cache tile by a 1-qubit diagonal op. `start` is the tile's global
/// index, so `start + local` reconstructs each amplitude's basis index for the
/// bit test — the same per-index phase as [`apply_diagonal_1q`], hence the same
/// (vectorizable) shape.
#[inline]
fn block_diagonal_1q(tile: &mut [C64], start: usize, bit: usize, d0: C64, d1: C64) {
    for (local, amp) in tile.iter_mut().enumerate() {
        *amp *= if (start + local) & bit == 0 { d0 } else { d1 };
    }
}

/// Multiply one cache tile by a 2-qubit diagonal op — the [`apply_diagonal_2q`]
/// body over a tile.
#[inline]
fn block_diagonal_2q(tile: &mut [C64], start: usize, b0: usize, b1: usize, diag: [C64; 4]) {
    for (local, amp) in tile.iter_mut().enumerate() {
        let i = start + local;
        let k = ((usize::from(i & b1 != 0)) << 1) | usize::from(i & b0 != 0);
        *amp *= diag[k];
    }
}

/// Fused diagonal run: apply every op in `ops` to the buffer, multiplying each
/// amplitude by the product of the ops' per-index factors, tiled so the buffer is
/// traversed through main memory roughly once for the whole run instead of once
/// per op — the win over applying each op through
/// [`apply_diagonal_1q`] / [`apply_diagonal_2q`] in turn.
///
/// Ops are first partitioned into monomorphic 1-qubit and 2-qubit slices (`ops`
/// is capped at `MAX_FUSED_DIAGONAL_RUN`, so this is a tiny, one-off split), then
/// each tile is carried through all 1-qubit ops and all 2-qubit ops. Reordering is
/// sound: diagonal operators commute. The result is numerically what applying the
/// ops one at a time gives, to rounding — the same sequence of per-amplitude
/// multiplies, only regrouped.
pub(crate) fn apply_diagonal_run(data: &mut [C64], ops: &[DiagonalOp], parallel: bool) {
    // Sized to how many of each variant `ops` actually holds, not to
    // `ops.len()` for both: a run of all-`Two` gates (every QFT phase column,
    // for instance — it never contains a `One`) used to reserve a same-sized
    // `ones` that stayed empty, and an equally oversized `twos` on top of
    // that. One pass over `ops` (cheap — reading tags, no buffer touch) to
    // count each variant means neither `Vec` reserves a single byte it
    // doesn't use.
    let (ones_count, twos_count) = ops
        .iter()
        .fold((0usize, 0usize), |(ones, twos), op| match op {
            DiagonalOp::One { .. } => (ones + 1, twos),
            DiagonalOp::Two { .. } => (ones, twos + 1),
        });
    let mut ones: Vec<(usize, C64, C64)> = Vec::with_capacity(ones_count);
    let mut twos: Vec<(usize, usize, [C64; 4])> = Vec::with_capacity(twos_count);
    for op in ops {
        match op {
            DiagonalOp::One { bit, d0, d1 } => ones.push((*bit, *d0, *d1)),
            DiagonalOp::Two { b0, b1, diag } => twos.push((*b0, *b1, *diag)),
        }
    }

    // One tile carried through every op: monomorphic inner loops, cache-resident.
    let apply_tile = |start: usize, tile: &mut [C64]| {
        for &(bit, d0, d1) in &ones {
            block_diagonal_1q(tile, start, bit, d0, d1);
        }
        for &(b0, b1, diag) in &twos {
            block_diagonal_2q(tile, start, b0, b1, diag);
        }
    };

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        data.par_chunks_mut(DIAGONAL_TILE)
            .enumerate()
            .for_each(|(t, tile)| apply_tile(t * DIAGONAL_TILE, tile));
        return;
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for (t, tile) in data.chunks_mut(DIAGONAL_TILE).enumerate() {
        apply_tile(t * DIAGONAL_TILE, tile);
    }
}

/// Controlled single-qubit gate: apply `m` to the `(i0, i1)` pair on target `t`
/// for every sub-state whose control bit `c` is set. Only those pairs are
/// enumerated: `g ∈ [0, 2^(n-2))` ranges over the other `n - 2` qubits and
/// [`two_qubit_indices`] inserts both fixed bits, so `i0` has `c` set and `t`
/// clear and `i1` has both set — no iteration is spent on a pair the control
/// would discard. Used for `Cx` and the other controlled 1-qubit gates.
pub(crate) fn apply_controlled_1q(
    data: &mut [C64],
    n: usize,
    c: usize,
    t: usize,
    m: &[[C64; 2]; 2],
    parallel: bool,
) {
    apply_pairs(data, n, Pairs::controlled(n, c, t), m, parallel);
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
    // See `Pairs::target`: these guard the raw-pointer writes below in release.
    assert!(q0 < n && q1 < n && q0 != q1 && data.len() == 1usize << n);
    let quarter = 1usize << (n - 2);

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let base = data.as_mut_ptr() as usize;
        (0..quarter).into_par_iter().for_each(|g| {
            let idx = two_qubit_indices(g, q0, q1);
            // The `;` after the block makes it a statement, so this `allow` can apply.
            #[allow(unsafe_code)] // authorised `unsafe` site: docs/ENGINEERING.md §5
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
            };
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

// ── Reductions (adjoint differentiation) ─────────────────────────────────
//
// A reduction is a floating-point sum, and addition is not associative: summed
// in whatever order a thread pool happens to split the work, the result would
// change with the thread count. These kernels therefore fix the order: the
// buffer is cut into blocks of `REDUCTION_BLOCK` amplitudes, each block is
// summed sequentially from its first index, and the blocks' partial sums are
// added left to right. Both paths run exactly those additions — the parallel
// one only computes the blocks concurrently — so the result is bit for bit the
// same with or without `parallel` and for any number of threads
// (docs/ENGINEERING.md §4). No `unsafe`: the blocks are disjoint slices.

/// Amplitudes per block of a reduction. Part of the summation order, so
/// changing it changes the last bits of every reduction; it is not a tuning
/// knob. 4096 amplitudes (64 KiB) keeps a block in L2, and leaves the parallel
/// path one partial sum per 4096 amplitudes to collect (4 MiB at 30 qubits,
/// against a 16 GiB statevector).
pub(crate) const REDUCTION_BLOCK: usize = 4096;

/// One block of [`weigh`]: scale each amplitude by its weight and sum
/// `weight · |amplitude|²` over the block, from its first index. `start` is the
/// block's global index. The flag is `false` if any weight is not finite.
#[inline]
fn weigh_block<W: Fn(usize) -> f64>(chunk: &mut [C64], start: usize, weight: &W) -> (f64, bool) {
    let mut sum = 0.0;
    let mut finite = true;
    for (local, amp) in chunk.iter_mut().enumerate() {
        let w = weight(start + local);
        finite &= w.is_finite();
        sum += w * amp.norm_sqr();
        *amp = amp.scale(w);
    }
    (sum, finite)
}

/// Apply the diagonal operator `O|b⟩ = weight(b)|b⟩` in place and return
/// `⟨ψ|O|ψ⟩` of the state the buffer held, plus whether every weight was
/// finite. The sum runs in the fixed block order described above, so it is the
/// same with and without `parallel`.
pub(crate) fn weigh<W: Fn(usize) -> f64 + Sync>(
    data: &mut [C64],
    weight: &W,
    parallel: bool,
) -> (f64, bool) {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let partials: Vec<(f64, bool)> = data
            .par_chunks_mut(REDUCTION_BLOCK)
            .enumerate()
            .map(|(k, chunk)| weigh_block(chunk, k * REDUCTION_BLOCK, weight))
            .collect();
        return partials
            .into_iter()
            .fold((0.0, true), |(sum, finite), (s, f)| (sum + s, finite && f));
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    data.chunks_mut(REDUCTION_BLOCK)
        .enumerate()
        .fold((0.0, true), |(sum, finite), (k, chunk)| {
            let (s, f) = weigh_block(chunk, k * REDUCTION_BLOCK, weight);
            (sum + s, finite && f)
        })
}

/// One block of [`pauli_inner`], over the basis states `start..end`.
#[inline]
fn pauli_inner_block(
    lambda: &[C64],
    psi: &[C64],
    start: usize,
    end: usize,
    x: usize,
    z: usize,
) -> C64 {
    let mut acc = C64::new(0.0, 0.0);
    for b in start..end {
        let term = lambda[b ^ x].conj() * psi[b];
        if (b & z).count_ones() & 1 == 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

/// `Σ_b conj(λ[b ⊕ x]) · (−1)^{|b ∧ z|} · ψ[b]`, which is `⟨λ|X^x Z^z|ψ⟩` for
/// the Pauli string with `X` on the bits of `x` and `Z` on the bits of `z`
/// (`Z` acting first). A `Y` on a qubit is `i·X·Z` there, so the caller
/// multiplies by `i` per `Y`. Reads both buffers, allocates no third one, and
/// sums in the fixed block order described above.
pub(crate) fn pauli_inner(lambda: &[C64], psi: &[C64], x: usize, z: usize, parallel: bool) -> C64 {
    // `b ^ x` must stay inside the buffer: true when every bit of the masks is
    // a qubit of a `2^n` buffer. Callers build the masks from validated qubits;
    // reaching this is a bug, and indexing would panic anyway.
    assert!(lambda.len() == psi.len() && psi.len().is_power_of_two() && (x | z) < psi.len());
    let len = psi.len();
    let block = |k: usize| {
        let start = k * REDUCTION_BLOCK;
        pauli_inner_block(lambda, psi, start, (start + REDUCTION_BLOCK).min(len), x, z)
    };
    let blocks = len.div_ceil(REDUCTION_BLOCK);

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let partials: Vec<C64> = (0..blocks).into_par_iter().map(block).collect();
        return partials
            .into_iter()
            .fold(C64::new(0.0, 0.0), |acc, partial| acc + partial);
    }

    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    (0..blocks).fold(C64::new(0.0, 0.0), |acc, k| acc + block(k))
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

    /// The pre-#216 `apply_controlled_1q` body — every target pair visited over
    /// `2^(n-1)`, the control filtered inside the loop — kept as the semantic
    /// reference the direct enumeration must reproduce exactly.
    fn controlled_1q_filtered(data: &mut [C64], n: usize, c: usize, t: usize, m: &[[C64; 2]; 2]) {
        let low = (1usize << t) - 1;
        let bit = 1usize << t;
        let cbit = 1usize << c;
        for g in 0..(1usize << (n - 1)) {
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

    /// For every `(c, t)` pair on `n` qubits: the sequential path equals the
    /// filtered reference, and (when compiled in) the parallel path equals the
    /// sequential one — both bit for bit (docs/ENGINEERING.md §4).
    fn assert_controlled_1q_matches_reference(n: usize, m: &[[C64; 2]; 2]) {
        for c in 0..n {
            for t in (0..n).filter(|&t| t != c) {
                let mut reference = sample_data(n);
                controlled_1q_filtered(&mut reference, n, c, t, m);

                let mut sequential = sample_data(n);
                apply_controlled_1q(&mut sequential, n, c, t, m, false);
                assert!(
                    sequential == reference,
                    "sequential path differs from the filtered reference (n={n}, c={c}, t={t})"
                );

                if cfg!(feature = "parallel") {
                    let mut parallel = sample_data(n);
                    apply_controlled_1q(&mut parallel, n, c, t, m, true);
                    assert!(
                        parallel == sequential,
                        "parallel path differs from the sequential one (n={n}, c={c}, t={t})"
                    );
                }
            }
        }
    }

    #[test]
    fn controlled_1q_matches_filtered_reference_on_two_qubits() {
        // n = 2 is the edge of the direct enumeration: a single sub-state g = 0.
        assert_controlled_1q_matches_reference(2, &gates::x());
        assert_controlled_1q_matches_reference(2, &gates::u(0.7, 1.1, -0.4));
    }

    #[test]
    fn controlled_1q_matches_filtered_reference_on_every_pair() {
        for n in [3, 5, 7] {
            assert_controlled_1q_matches_reference(n, &gates::x());
            assert_controlled_1q_matches_reference(n, &gates::u(0.7, 1.1, -0.4));
        }
    }

    /// For every target `t` on `n` qubits: the sequential path equals a
    /// reference that finds each pair by scanning for bit `t` clear, and (when
    /// compiled in) the parallel path equals the sequential one — both bit for
    /// bit (docs/ENGINEERING.md §4).
    fn assert_1q_matches_reference(n: usize, m: &[[C64; 2]; 2]) {
        for t in 0..n {
            let bit = 1usize << t;
            let mut reference = sample_data(n);
            for i0 in (0..1usize << n).filter(|i| i & bit == 0) {
                let (na, nb) = combine_1q(m, reference[i0], reference[i0 | bit]);
                reference[i0] = na;
                reference[i0 | bit] = nb;
            }

            let mut sequential = sample_data(n);
            apply_1q(&mut sequential, n, t, m, false);
            assert!(
                sequential == reference,
                "sequential path differs from the scanning reference (n={n}, t={t})"
            );

            if cfg!(feature = "parallel") {
                let mut parallel = sample_data(n);
                apply_1q(&mut parallel, n, t, m, true);
                assert!(
                    parallel == sequential,
                    "parallel path differs from the sequential one (n={n}, t={t})"
                );
            }
        }
    }

    #[test]
    fn one_qubit_matches_reference_on_every_target() {
        // n = 1 is the edge of the enumeration: a single sub-state g = 0.
        for n in [1, 3, 5, 7] {
            assert_1q_matches_reference(n, &gates::h());
            assert_1q_matches_reference(n, &gates::u(0.7, 1.1, -0.4));
        }
    }

    // #342: the dense kernels' preconditions are real `assert!`s, so a violation
    // panics — in release too — on the parallel path, before any raw-pointer
    // write. (Without `parallel` the flag is ignored and the same asserts run.)

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn controlled_1q_panics_on_repeated_operands() {
        let mut data = sample_data(1);
        apply_controlled_1q(&mut data, 1, 0, 0, &gates::x(), true);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn two_qubit_panics_on_repeated_operands() {
        let mut data = sample_data(1);
        apply_2q(&mut data, 1, 0, 0, &gates::swap(), true);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn one_qubit_panics_on_a_buffer_shorter_than_two_to_the_n() {
        let mut data = sample_data(2);
        apply_1q(&mut data, 3, 2, &gates::h(), true);
    }

    /// A weight that varies with every bit, so a mis-indexed block shows up.
    fn weight(b: usize) -> f64 {
        0.25 + (b % 7) as f64 - 0.5 * (b.count_ones() as f64)
    }

    /// `n` qubits spanning several reduction blocks (and a partial last one is
    /// impossible: blocks and buffers are both powers of two).
    const MULTI_BLOCK_QUBITS: usize = 14;

    #[test]
    fn weigh_scales_by_the_weight_and_returns_the_expectation() {
        let n = MULTI_BLOCK_QUBITS;
        let original = sample_data(n);
        let mut data = original.clone();
        let (sum, finite) = weigh(&mut data, &weight, false);
        assert!(finite);
        let naive: f64 = original
            .iter()
            .enumerate()
            .map(|(b, a)| weight(b) * a.norm_sqr())
            .sum();
        assert!((sum - naive).abs() <= 1e-12 * naive.abs());
        for (b, (scaled, a)) in data.iter().zip(&original).enumerate() {
            assert_eq!(*scaled, a.scale(weight(b)), "amplitude {b}");
        }
    }

    #[test]
    fn weigh_flags_a_non_finite_weight() {
        let mut data = sample_data(MULTI_BLOCK_QUBITS);
        let (_, finite) = weigh(
            &mut data,
            &|b| if b == 9000 { f64::NAN } else { 1.0 },
            false,
        );
        assert!(!finite);
    }

    /// `⟨λ|P|ψ⟩` for every single- and two-qubit `X`/`Z` mask on a small
    /// register, against the Pauli string applied by hand.
    #[test]
    fn pauli_inner_matches_the_pauli_applied_by_hand() {
        let n = 4;
        let psi = sample_data(n);
        let lambda: Vec<C64> = psi.iter().rev().map(|a| a * C64::new(0.3, -0.7)).collect();
        for x in 0..(1usize << n) {
            for z in [0, 1, 2, 5, 12] {
                // P|b⟩ = (−1)^{|b ∧ z|} |b ⊕ x⟩ (Z first, then X).
                let mut p_psi = vec![C64::new(0.0, 0.0); 1 << n];
                for (b, &a) in psi.iter().enumerate() {
                    let sign = if (b & z).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    p_psi[b ^ x] += a * sign;
                }
                let expected: C64 = lambda.iter().zip(&p_psi).map(|(l, p)| l.conj() * p).sum();
                let got = pauli_inner(&lambda, &psi, x, z, false);
                assert!((got - expected).norm() < 1e-9, "x={x} z={z}");
            }
        }
    }

    /// docs/ENGINEERING.md §4: the parallel reductions equal the sequential ones
    /// bit for bit, whatever the thread count, because the blocks and the order
    /// in which their sums are combined do not depend on it.
    #[cfg(feature = "parallel")]
    #[test]
    fn reductions_are_bit_identical_across_paths_and_thread_counts() {
        let n = MULTI_BLOCK_QUBITS;
        let psi = sample_data(n);
        let lambda: Vec<C64> = psi.iter().map(|a| a * C64::new(-0.2, 0.9)).collect();
        let masks = [(0b10, 0b10), (0b1_0000_0001, 0), (0, 0b10_0000_0000_0100)];

        let mut seq_data = psi.clone();
        let seq_weigh = weigh(&mut seq_data, &weight, false);
        let seq_inner: Vec<C64> = masks
            .iter()
            .map(|&(x, z)| pauli_inner(&lambda, &psi, x, z, false))
            .collect();

        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a small thread pool");
            pool.install(|| {
                let mut par_data = psi.clone();
                let par_weigh = weigh(&mut par_data, &weight, true);
                assert_eq!(
                    par_weigh.0.to_bits(),
                    seq_weigh.0.to_bits(),
                    "{threads} threads"
                );
                assert_eq!(par_weigh.1, seq_weigh.1);
                assert!(par_data == seq_data, "{threads} threads");
                for (&(x, z), seq) in masks.iter().zip(&seq_inner) {
                    let par = pauli_inner(&lambda, &psi, x, z, true);
                    assert_eq!(par.re.to_bits(), seq.re.to_bits(), "{threads} threads");
                    assert_eq!(par.im.to_bits(), seq.im.to_bits(), "{threads} threads");
                }
            });
        }
    }
}
