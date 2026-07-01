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
}
