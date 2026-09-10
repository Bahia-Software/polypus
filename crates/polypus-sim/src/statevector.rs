//! The [`Statevector`] type and circuit-gate evolution.

use crate::error::SimError;
use crate::{gates, kernels, C64, DEFAULT_PARALLEL_THRESHOLD, MAX_QUBITS};
use polypus_circuit::{GateInstruction, GateParam};

/// A dense quantum state over `n` qubits: `2^n` complex amplitudes in the
/// computational basis, indexed **little-endian** (qubit 0 is the
/// least-significant bit), matching Qiskit.
///
/// Build one with [`Statevector::new`] (which starts in `|0…0⟩`), evolve it with
/// [`apply`](Self::apply), and read it out with [`amplitudes`](Self::amplitudes),
/// [`probabilities`](Self::probabilities), [`expectation_z`](Self::expectation_z)
/// or [`sample`](Self::sample).
#[derive(Debug, Clone, PartialEq)]
pub struct Statevector {
    pub(crate) n: usize,
    pub(crate) data: Vec<C64>,
    pub(crate) parallel_threshold: usize,
}

/// Resolve a gate angle, surfacing the two failure modes the simulator must not
/// silently propagate into the statevector.
fn angle(p: &GateParam) -> Result<f64, SimError> {
    match p {
        GateParam::Fixed(v) if v.is_finite() => Ok(*v),
        GateParam::Fixed(_) => Err(SimError::NonFiniteAmplitude),
        GateParam::Param(i) => Err(SimError::UnboundParameter { index: *i }),
    }
}

/// The diagonal descriptor for `gate`, or `None` if `gate` is not one of the
/// nine diagonal instructions a fused run is built from (`Z, S, T, Sdg, Tdg, Rz`
/// and `Cz, Rzz, Cp`).
///
/// These nine arms mirror exactly the diagonal arms of [`Statevector::apply`],
/// so run detection and single-gate application always agree on what "diagonal"
/// means — this function is the single source of truth for that, and
/// [`is_diagonal`] is defined in terms of it precisely so the two cannot drift.
///
/// `Some(Err(..))` is a *diagonal* gate whose angle is invalid: it still belongs
/// to a run, but resolving its angle failed exactly as [`Statevector::apply`]
/// would fail on it. Building the descriptor is where a run's angle validation
/// happens, before any amplitude is touched — see
/// [`Statevector::apply_diagonal_ops`] and its caller.
pub(crate) fn diagonal_op(gate: &GateInstruction) -> Option<Result<kernels::DiagonalOp, SimError>> {
    use kernels::DiagonalOp;
    // A resolved angle that fails validation short-circuits the whole builder to
    // `Some(Err(..))`, exactly as `apply` would fail on the same gate.
    macro_rules! angle {
        ($theta:expr) => {
            match angle($theta) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            }
        };
    }
    let op = match gate {
        GateInstruction::Z(q) => {
            let (d0, d1) = gates::z();
            DiagonalOp::One {
                bit: 1 << *q,
                d0,
                d1,
            }
        }
        GateInstruction::S(q) => {
            let (d0, d1) = gates::s();
            DiagonalOp::One {
                bit: 1 << *q,
                d0,
                d1,
            }
        }
        GateInstruction::T(q) => {
            let (d0, d1) = gates::t();
            DiagonalOp::One {
                bit: 1 << *q,
                d0,
                d1,
            }
        }
        GateInstruction::Sdg(q) => {
            let (d0, d1) = gates::sdg();
            DiagonalOp::One {
                bit: 1 << *q,
                d0,
                d1,
            }
        }
        GateInstruction::Tdg(q) => {
            let (d0, d1) = gates::tdg();
            DiagonalOp::One {
                bit: 1 << *q,
                d0,
                d1,
            }
        }
        GateInstruction::Rz { qubit, theta } => {
            let (d0, d1) = gates::rz(angle!(theta));
            DiagonalOp::One {
                bit: 1 << *qubit,
                d0,
                d1,
            }
        }
        GateInstruction::Cz(c, t) => DiagonalOp::Two {
            b0: 1 << *c,
            b1: 1 << *t,
            diag: gates::cz_diag(),
        },
        GateInstruction::Rzz { q0, q1, theta } => DiagonalOp::Two {
            b0: 1 << *q0,
            b1: 1 << *q1,
            diag: gates::rzz_diag(angle!(theta)),
        },
        GateInstruction::Cp { q0, q1, theta } => DiagonalOp::Two {
            b0: 1 << *q0,
            b1: 1 << *q1,
            diag: gates::cp_diag(angle!(theta)),
        },
        // Every non-diagonal instruction: not part of a fused run.
        _ => return None,
    };
    Some(Ok(op))
}

/// Whether `gate` dispatches through a diagonal kernel — i.e. whether it can join
/// a fused diagonal run. Defined via [`diagonal_op`] so the predicate and the
/// builder are one source of truth (see [`diagonal_op`]).
pub(crate) fn is_diagonal(gate: &GateInstruction) -> bool {
    diagonal_op(gate).is_some()
}

/// The qubit set of a *dense-fusable* gate: the nine dense (non-diagonal,
/// non-boundary) instructions the connected-component fusion of issue #132 is
/// built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseQubits {
    /// A dense one-qubit gate (`H, X, Y, Rx, Ry, U`).
    One(usize),
    /// A dense two-qubit gate (`Cx, Swap, Rxx`); the two qubits are distinct.
    Two(usize, usize),
}

/// The qubits `gate` touches if it is one of the nine dense-fusable instructions
/// (`H, X, Y, Rx, Ry, U` and `Cx, Swap, Rxx`), or `None` otherwise.
///
/// These nine are exactly the gates [`Statevector::apply`] dispatches through
/// [`kernels::apply_1q`] / [`kernels::apply_2q`] (its dense, non-diagonal arms):
/// disjoint from the nine [`diagonal_op`] classifies and from the
/// `Barrier`/`Measure`/`MeasureAll` no-ops, so a gate is dense-fusable, diagonal,
/// or a boundary, never two of those. A run of these fuses into one composed
/// matrix per connected qubit component; see
/// [`Statevector::apply_composed_1q`] / [`Statevector::apply_composed_2q`] and
/// their caller
/// [`StatevectorSimulator::run_cancellable`](crate::StatevectorSimulator).
pub(crate) fn dense_fusable_qubits(gate: &GateInstruction) -> Option<DenseQubits> {
    use GateInstruction as G;
    match gate {
        G::H(q) | G::X(q) | G::Y(q) => Some(DenseQubits::One(*q)),
        G::Rx { qubit, .. } | G::Ry { qubit, .. } | G::U { qubit, .. } => {
            Some(DenseQubits::One(*qubit))
        }
        G::Cx(a, b) | G::Swap(a, b) => Some(DenseQubits::Two(*a, *b)),
        G::Rxx { q0, q1, .. } => Some(DenseQubits::Two(*q0, *q1)),
        _ => None,
    }
}

/// The 2×2 matrix of a dense one-qubit gate, with its angles resolved (the same
/// resolution [`Statevector::apply`] performs, propagating the same [`SimError`]
/// on a `NaN`/unbound angle before any amplitude is touched).
///
/// Only ever called on a gate [`dense_fusable_qubits`] classified as
/// [`DenseQubits::One`]; any other variant is a caller bug, not a runtime error.
fn dense_matrix_1q(gate: &GateInstruction) -> Result<[[C64; 2]; 2], SimError> {
    use GateInstruction as G;
    let m = match gate {
        G::H(_) => gates::h(),
        G::X(_) => gates::x(),
        G::Y(_) => gates::y(),
        G::Rx { theta, .. } => gates::rx(angle(theta)?),
        G::Ry { theta, .. } => gates::ry(angle(theta)?),
        G::U {
            theta, phi, lam, ..
        } => gates::u(angle(theta)?, angle(phi)?, angle(lam)?),
        other => unreachable!("{other:?} is not a dense one-qubit gate"),
    };
    Ok(m)
}

/// The 4×4 matrix of a dense two-qubit gate over the component's qubits
/// `(qa, qb)` (`qa < qb`), in the `(bit_qb << 1) | bit_qa` basis that
/// [`kernels::apply_2q`] / [`kernels::two_qubit_indices`] use.
///
/// `Swap` and `Rxx` are symmetric in their operands, so their gate matrices are
/// already correct regardless of which operand is `qa`; only `Cx` needs its
/// control/target woven into this basis, which [`cx_4x4`] does. Only ever called
/// on a [`DenseQubits::Two`] gate acting on exactly `{qa, qb}`, where `qa` is the
/// low qubit (bit 0).
fn dense_matrix_2q(gate: &GateInstruction, qa: usize) -> Result<[[C64; 4]; 4], SimError> {
    use GateInstruction as G;
    let m = match gate {
        G::Cx(control, target) => cx_4x4(*control, *target, qa),
        G::Swap(..) => gates::swap(),
        G::Rxx { theta, .. } => gates::rxx(angle(theta)?),
        other => unreachable!("{other:?} is not a dense two-qubit gate"),
    };
    Ok(m)
}

/// Lift a dense one-qubit gate acting on one of a component's two qubits into the
/// joint 4×4 space over `(qa, qb)`, ready to compose with [`matmul4`].
///
/// A one-qubit gate embeds as `m` on its own qubit and identity on the other;
/// `dense_matrix_2q` handles the genuinely two-qubit members. `qa` is the
/// component's low qubit (bit 0); the other qubit is bit 1.
fn lifted_matrix_2q(gate: &GateInstruction, qa: usize) -> Result<[[C64; 4]; 4], SimError> {
    match dense_fusable_qubits(gate) {
        Some(DenseQubits::One(q)) => {
            let m = dense_matrix_1q(gate)?;
            // qa is the low bit (position 0), the other qubit the high bit (1).
            let target_pos = if q == qa { 0 } else { 1 };
            Ok(lift_1q_to_2q(&m, target_pos))
        }
        Some(DenseQubits::Two(..)) => dense_matrix_2q(gate, qa),
        None => unreachable!("a component gate is always dense-fusable"),
    }
}

/// Embed a 2×2 gate matrix `m` acting on bit `target_pos` (0 or 1) of the
/// two-qubit index into the 4×4 that acts as identity on the other bit, in the
/// `(bit1 << 1) | bit0` basis [`kernels::apply_2q`] expects.
fn lift_1q_to_2q(m: &[[C64; 2]; 2], target_pos: usize) -> [[C64; 4]; 4] {
    let other_pos = 1 - target_pos;
    let mut out = [[C64::new(0.0, 0.0); 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            // Identity on the untouched bit: zero unless it matches across in/out.
            if (r >> other_pos) & 1 == (c >> other_pos) & 1 {
                *cell = m[(r >> target_pos) & 1][(c >> target_pos) & 1];
            }
        }
    }
    out
}

/// The 4×4 permutation of `Cx(control, target)` over a component's qubits
/// `(qa, qb)`, in the `(bit_qb << 1) | bit_qa` basis. Built directly rather than
/// through [`kernels::apply_controlled_1q`]'s control/target dispatch, because a
/// composed component needs a plain 4×4 for [`kernels::apply_2q`].
///
/// `qa` is the component's low qubit (bit 0 of the two-qubit index); the high
/// qubit is bit 1, so a qubit that is not `qa` is at position 1.
fn cx_4x4(control: usize, target: usize, qa: usize) -> [[C64; 4]; 4] {
    let control_pos = if control == qa { 0 } else { 1 };
    let target_pos = if target == qa { 0 } else { 1 };
    let mut out = [[C64::new(0.0, 0.0); 4]; 4];
    // `Cx` is its own inverse (it flips the target when the control is set, and
    // the control bit is untouched), so the single non-zero column of output
    // `row` is that same map applied to `row`.
    for (row, out_row) in out.iter_mut().enumerate() {
        let col = if (row >> control_pos) & 1 == 1 {
            row ^ (1 << target_pos)
        } else {
            row
        };
        out_row[col] = C64::new(1.0, 0.0);
    }
    out
}

/// `a · b` for 2×2 complex matrices (`a` applied after `b`).
fn matmul2(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    let mut out = [[C64::new(0.0, 0.0); 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = C64::new(0.0, 0.0);
            for k in 0..2 {
                acc += a[i][k] * b[k][j];
            }
            *cell = acc;
        }
    }
    out
}

/// `a · b` for 4×4 complex matrices (`a` applied after `b`).
fn matmul4(a: &[[C64; 4]; 4], b: &[[C64; 4]; 4]) -> [[C64; 4]; 4] {
    let mut out = [[C64::new(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = C64::new(0.0, 0.0);
            for k in 0..4 {
                acc += a[i][k] * b[k][j];
            }
            *cell = acc;
        }
    }
    out
}

impl Statevector {
    /// Allocate the `|0…0⟩` state over `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// [`SimError::TooManyQubits`] if `num_qubits > `[`MAX_QUBITS`], which also
    /// guarantees `1 << num_qubits` cannot overflow.
    pub fn new(num_qubits: usize) -> Result<Self, SimError> {
        if num_qubits > MAX_QUBITS {
            return Err(SimError::TooManyQubits {
                requested: num_qubits,
                max: MAX_QUBITS,
            });
        }
        let dim = 1usize << num_qubits;
        let mut data = vec![C64::new(0.0, 0.0); dim];
        data[0] = C64::new(1.0, 0.0);
        Ok(Statevector {
            n: num_qubits,
            data,
            parallel_threshold: DEFAULT_PARALLEL_THRESHOLD,
        })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Dimension of the state space (`2^n`).
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Read-only view of the amplitudes, indexed by computational basis state.
    pub fn amplitudes(&self) -> &[C64] {
        &self.data
    }

    /// Consume the state and return ownership of its amplitude buffer.
    ///
    /// The counterpart of [`amplitudes`](Self::amplitudes) for callers that hand
    /// the buffer on and drop the state anyway: it moves the `2^n` amplitudes
    /// out instead of forcing a `to_vec()` copy of a vector that may be several
    /// GiB (see [`MAX_QUBITS`]).
    pub fn into_amplitudes(self) -> Vec<C64> {
        self.data
    }

    /// L2 norm of the state. A correctly evolved state stays at `1` up to
    /// floating-point error.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Renormalize the state to unit norm. A no-op (within rounding) for states
    /// produced by unitary evolution; useful after manual surgery.
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for amp in &mut self.data {
                *amp = amp.scale(inv);
            }
        }
    }

    /// Set the qubit count at or above which gates run on the parallel kernels
    /// (only meaningful with the `parallel` feature). Used by
    /// [`StatevectorSimulator`](crate::StatevectorSimulator).
    pub(crate) fn set_parallel_threshold(&mut self, threshold: usize) {
        self.parallel_threshold = threshold;
    }

    /// Whether the next gate should use the parallel kernels.
    fn use_parallel(&self) -> bool {
        cfg!(feature = "parallel") && self.n >= self.parallel_threshold
    }

    /// Apply one circuit instruction in place.
    ///
    /// `Barrier`, `Measure` and `MeasureAll` are no-ops for the state: this
    /// backend never collapses mid-circuit; measurement statistics are taken
    /// from the final state via [`sample`](Self::sample).
    ///
    /// # Errors
    ///
    /// [`SimError::UnboundParameter`] if an angle is still a free parameter, or
    /// [`SimError::NonFiniteAmplitude`] if an angle is `NaN`/infinity.
    pub fn apply(&mut self, gate: &GateInstruction) -> Result<(), SimError> {
        let par = self.use_parallel();
        let n = self.n;
        match gate {
            GateInstruction::H(q) => kernels::apply_1q(&mut self.data, n, *q, &gates::h(), par),
            GateInstruction::X(q) => kernels::apply_1q(&mut self.data, n, *q, &gates::x(), par),
            GateInstruction::Y(q) => kernels::apply_1q(&mut self.data, n, *q, &gates::y(), par),
            GateInstruction::Z(q) => {
                let (d0, d1) = gates::z();
                kernels::apply_diagonal_1q(&mut self.data, *q, d0, d1, par);
            }
            GateInstruction::S(q) => {
                let (d0, d1) = gates::s();
                kernels::apply_diagonal_1q(&mut self.data, *q, d0, d1, par);
            }
            GateInstruction::T(q) => {
                let (d0, d1) = gates::t();
                kernels::apply_diagonal_1q(&mut self.data, *q, d0, d1, par);
            }
            GateInstruction::Sdg(q) => {
                let (d0, d1) = gates::sdg();
                kernels::apply_diagonal_1q(&mut self.data, *q, d0, d1, par);
            }
            GateInstruction::Tdg(q) => {
                let (d0, d1) = gates::tdg();
                kernels::apply_diagonal_1q(&mut self.data, *q, d0, d1, par);
            }
            GateInstruction::Rx { qubit, theta } => {
                let m = gates::rx(angle(theta)?);
                kernels::apply_1q(&mut self.data, n, *qubit, &m, par);
            }
            GateInstruction::Ry { qubit, theta } => {
                let m = gates::ry(angle(theta)?);
                kernels::apply_1q(&mut self.data, n, *qubit, &m, par);
            }
            GateInstruction::Rz { qubit, theta } => {
                let (d0, d1) = gates::rz(angle(theta)?);
                kernels::apply_diagonal_1q(&mut self.data, *qubit, d0, d1, par);
            }
            GateInstruction::Cx(c, t) => {
                kernels::apply_controlled_1q(&mut self.data, n, *c, *t, &gates::x(), par);
            }
            GateInstruction::Cz(c, t) => {
                kernels::apply_diagonal_2q(&mut self.data, *c, *t, gates::cz_diag(), par);
            }
            GateInstruction::Swap(q0, q1) => {
                kernels::apply_2q(&mut self.data, n, *q0, *q1, &gates::swap(), par);
            }
            GateInstruction::Rzz { q0, q1, theta } => {
                let diag = gates::rzz_diag(angle(theta)?);
                kernels::apply_diagonal_2q(&mut self.data, *q0, *q1, diag, par);
            }
            GateInstruction::Rxx { q0, q1, theta } => {
                let m = gates::rxx(angle(theta)?);
                kernels::apply_2q(&mut self.data, n, *q0, *q1, &m, par);
            }
            GateInstruction::Cp { q0, q1, theta } => {
                let diag = gates::cp_diag(angle(theta)?);
                kernels::apply_diagonal_2q(&mut self.data, *q0, *q1, diag, par);
            }
            GateInstruction::U {
                qubit,
                theta,
                phi,
                lam,
            } => {
                let m = gates::u(angle(theta)?, angle(phi)?, angle(lam)?);
                kernels::apply_1q(&mut self.data, n, *qubit, &m, par);
            }
            GateInstruction::Barrier(_)
            | GateInstruction::Measure { .. }
            | GateInstruction::MeasureAll => {}
        }
        Ok(())
    }

    /// Apply a run of consecutive diagonal operators in roughly one main-memory
    /// traversal of the amplitude buffer, instead of one per operator.
    ///
    /// Diagonal operators commute regardless of which qubits they touch, so a run
    /// of them is one combined diagonal whose per-index phase is the product of
    /// the individual phases; [`kernels::apply_diagonal_run`] applies that product
    /// tile by tile, carrying each cache tile through every op so the buffer is
    /// read and written through main memory about once for the whole run. The
    /// result is numerically what calling [`apply`](Self::apply) on each member of
    /// the run in turn gives, to rounding (the same per-amplitude multiplies,
    /// regrouped); the win is the saved memory traffic on a deep diagonal stretch.
    ///
    /// `ops` are the descriptors built from the run by [`diagonal_op`], where the
    /// run's angle validation has already happened — so this method itself is
    /// infallible and never partially applies. Its caller,
    /// [`StatevectorSimulator::run_cancellable`](crate::StatevectorSimulator),
    /// builds them (propagating any [`SimError`] before calling this) and takes a
    /// length-1 fast path through [`apply`](Self::apply), so an isolated diagonal
    /// gate never pays for a `Vec` or the per-index run loop.
    pub(crate) fn apply_diagonal_ops(&mut self, ops: &[kernels::DiagonalOp]) {
        let par = self.use_parallel();
        kernels::apply_diagonal_run(&mut self.data, ops, par);
    }

    /// Apply a component of two or more dense one-qubit gates, all on `qubit`, as
    /// a single composed 2×2 matrix in one buffer pass instead of one pass each.
    ///
    /// The gates are composed **in circuit order** — unlike a diagonal run they do
    /// not commute in general, so the order is load-bearing — and every angle is
    /// resolved (surfacing the same [`SimError`] [`apply`](Self::apply) would)
    /// *before* any amplitude is written, so an invalid angle fails the whole
    /// component untouched, matching [`diagonal_op`]'s pre-write validation.
    ///
    /// The single-gate case never reaches here: its caller,
    /// [`StatevectorSimulator::run_cancellable`](crate::StatevectorSimulator),
    /// dispatches a lone gate straight through [`apply`](Self::apply) so a
    /// component that never merged pays no composition overhead. `gates_in_order`
    /// therefore always holds at least two gates.
    pub(crate) fn apply_composed_1q(
        &mut self,
        qubit: usize,
        gates_in_order: &[&GateInstruction],
    ) -> Result<(), SimError> {
        let (first, rest) = gates_in_order
            .split_first()
            .expect("a composed component holds at least two gates");
        let mut combined = dense_matrix_1q(first)?;
        for gate in rest {
            combined = matmul2(&dense_matrix_1q(gate)?, &combined);
        }
        let par = self.use_parallel();
        kernels::apply_1q(&mut self.data, self.n, qubit, &combined, par);
        Ok(())
    }

    /// Apply a component of two or more dense gates on exactly the two qubits
    /// `(qa, qb)` (`qa < qb`) as a single composed 4×4 matrix in one buffer pass.
    ///
    /// Each member is lifted into the joint two-qubit space (a one-qubit gate
    /// embeds as identity on the other qubit; `Cx`/`Swap`/`Rxx` are already
    /// two-qubit) and the lifts are composed in circuit order. Angle resolution
    /// and the "validate before any write" contract are exactly as for
    /// [`apply_composed_1q`](Self::apply_composed_1q); `gates_in_order` likewise
    /// always holds at least two gates.
    pub(crate) fn apply_composed_2q(
        &mut self,
        qa: usize,
        qb: usize,
        gates_in_order: &[&GateInstruction],
    ) -> Result<(), SimError> {
        let (first, rest) = gates_in_order
            .split_first()
            .expect("a composed component holds at least two gates");
        let mut combined = lifted_matrix_2q(first, qa)?;
        for gate in rest {
            combined = matmul4(&lifted_matrix_2q(gate, qa)?, &combined);
        }
        let par = self.use_parallel();
        kernels::apply_2q(&mut self.data, self.n, qa, qb, &combined, par);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `into_amplitudes` must hand out exactly the buffer `amplitudes` borrows —
    /// it is the copy-free alternative to `amplitudes().to_vec()`, not a
    /// different read-out.
    #[test]
    fn into_amplitudes_matches_the_borrowed_view() {
        let mut sv = Statevector::new(2).expect("2 qubits is well below MAX_QUBITS");
        sv.apply(&GateInstruction::H(0))
            .expect("H is unconditional");
        sv.apply(&GateInstruction::Cx(0, 1))
            .expect("Cx is unconditional");

        let borrowed = sv.amplitudes().to_vec();
        let owned = sv.into_amplitudes();
        assert_eq!(owned, borrowed);
    }

    /// The classifier must accept exactly the nine instructions `apply`
    /// dispatches through a diagonal kernel, and reject everything else.
    #[test]
    fn is_diagonal_matches_the_diagonal_kernel_arms() {
        let diagonal = [
            GateInstruction::Z(0),
            GateInstruction::S(0),
            GateInstruction::T(0),
            GateInstruction::Sdg(0),
            GateInstruction::Tdg(0),
            GateInstruction::Rz {
                qubit: 0,
                theta: GateParam::Fixed(0.4),
            },
            GateInstruction::Cz(0, 1),
            GateInstruction::Rzz {
                q0: 0,
                q1: 1,
                theta: GateParam::Fixed(0.4),
            },
            GateInstruction::Cp {
                q0: 0,
                q1: 1,
                theta: GateParam::Fixed(0.4),
            },
        ];
        for gate in &diagonal {
            assert!(is_diagonal(gate), "{gate:?} should be diagonal");
        }

        let non_diagonal = [
            GateInstruction::H(0),
            GateInstruction::X(0),
            GateInstruction::Y(0),
            GateInstruction::Rx {
                qubit: 0,
                theta: GateParam::Fixed(0.4),
            },
            GateInstruction::Ry {
                qubit: 0,
                theta: GateParam::Fixed(0.4),
            },
            GateInstruction::Cx(0, 1),
            GateInstruction::Swap(0, 1),
            GateInstruction::Rxx {
                q0: 0,
                q1: 1,
                theta: GateParam::Fixed(0.4),
            },
            GateInstruction::U {
                qubit: 0,
                theta: GateParam::Fixed(0.4),
                phi: GateParam::Fixed(0.4),
                lam: GateParam::Fixed(0.4),
            },
            GateInstruction::Barrier(vec![]),
            GateInstruction::Measure { qubit: 0, cbit: 0 },
            GateInstruction::MeasureAll,
        ];
        for gate in &non_diagonal {
            assert!(!is_diagonal(gate), "{gate:?} should not be diagonal");
        }
    }

    /// A diagonal gate with an unresolved angle is still diagonal (belongs to a
    /// run), but building its descriptor surfaces the same error `apply` would —
    /// this is where a run's angle validation lives, before any buffer write.
    #[test]
    fn diagonal_op_propagates_angle_errors() {
        let unbound = GateInstruction::Rz {
            qubit: 0,
            theta: GateParam::Param(2),
        };
        assert!(is_diagonal(&unbound));
        assert_eq!(
            diagonal_op(&unbound),
            Some(Err(SimError::UnboundParameter { index: 2 }))
        );

        let nan = GateInstruction::Cp {
            q0: 0,
            q1: 1,
            theta: GateParam::Fixed(f64::NAN),
        };
        assert!(is_diagonal(&nan));
        assert_eq!(diagonal_op(&nan), Some(Err(SimError::NonFiniteAmplitude)));

        // A non-diagonal gate is not classified as part of a run at all.
        assert!(diagonal_op(&GateInstruction::H(0)).is_none());
    }

    /// Fusing a run through `apply_diagonal_ops` must land on exactly the state
    /// that applying each member singly through `apply` does.
    #[test]
    fn apply_diagonal_ops_matches_gate_by_gate_apply() {
        let run = [
            GateInstruction::Rz {
                qubit: 1,
                theta: GateParam::Fixed(0.3),
            },
            GateInstruction::Cz(1, 3),
            GateInstruction::Rzz {
                q0: 0,
                q1: 2,
                theta: GateParam::Fixed(0.9),
            },
            GateInstruction::T(3),
            GateInstruction::Cp {
                q0: 2,
                q1: 3,
                theta: GateParam::Fixed(1.3),
            },
        ];

        // A non-trivial starting state, so diagonal phases actually matter.
        let mut base = Statevector::new(4).expect("4 qubits is well below MAX_QUBITS");
        for q in 0..4 {
            base.apply(&GateInstruction::H(q))
                .expect("H is unconditional");
        }

        let mut sequential = base.clone();
        for gate in &run {
            sequential.apply(gate).expect("run members are all valid");
        }

        let ops: Vec<_> = run
            .iter()
            .map(|g| {
                diagonal_op(g)
                    .expect("run member is diagonal")
                    .expect("run member has a valid angle")
            })
            .collect();
        let mut fused = base;
        fused.apply_diagonal_ops(&ops);

        for (i, (a, b)) in sequential
            .amplitudes()
            .iter()
            .zip(fused.amplitudes().iter())
            .enumerate()
        {
            assert!(
                (a - b).norm() < 1e-12,
                "amplitude {i} differs: sequential {a} vs fused {b}"
            );
        }
    }
}
