//! Adjoint differentiation: the exact gradient of a diagonal expectation value
//! in one forward and one backward sweep, whatever the number of parameters.
//!
//! For `|ψ⟩ = U_N ⋯ U_1 |0…0⟩` and a diagonal observable `O`, the derivative of
//! `⟨ψ|O|ψ⟩` with respect to the angle of a rotation `U_i = exp(−iθP/2)` is
//!
//! ```text
//! ∂⟨O⟩/∂θ_i = Im ⟨λ_i| P |ψ_i⟩,   ψ_i = U_i ⋯ U_1 |0…0⟩,   λ_i = U_{i+1}† ⋯ U_N† O |ψ⟩
//! ```
//!
//! (`2·Re⟨λ_i| ∂U_i/∂θ |ψ_{i−1}⟩` with `∂U_i/∂θ = −(i/2)·P·U_i`). Both `ψ_i` and
//! `λ_i` are reached by stepping back from the end of the circuit one gate at a
//! time — `ψ ← U_i†ψ`, `λ ← U_i†λ` — so the sweep holds exactly two statevectors
//! ([`ADJOINT_STATEVECTORS`](crate::ADJOINT_STATEVECTORS)) and costs about three
//! circuit simulations plus one read of the pair per differentiated gate, where
//! the parameter-shift rule costs two simulations per gate.

use crate::error::SimError;
use crate::simulator::CancellationCheck;
use crate::statevector::{angle, Statevector};
use crate::{kernels, StatevectorSimulator, C64};
use polypus_circuit::{ConcreteCircuit, GateInstruction, GateParam};

/// The result of [`StatevectorSimulator::adjoint_gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdjointGradient {
    /// `⟨ψ|O|ψ⟩`: the expectation of the observable in the circuit's final
    /// state, summed in the simulator's fixed block order (see
    /// [`adjoint_gradient`](StatevectorSimulator::adjoint_gradient)).
    pub expectation: f64,
    /// `∂⟨O⟩/∂θ` of each gate named in `wrt`, in `wrt`'s order.
    pub derivatives: Vec<f64>,
}

/// The Pauli generator `P` of a rotation `exp(−iθP/2)`, in the form the
/// reduction kernel reads: `P|b⟩ = i^y · (−1)^{|b ∧ z|} · |b ⊕ x⟩`, with `x`
/// and `z` bit masks over the qubits.
#[derive(Debug, Clone, Copy)]
struct Generator {
    x: usize,
    z: usize,
    /// Whether `P` carries one `Y` (`Y = i·X·Z`).
    y: bool,
}

/// The generator and the angle of `gate` if it is one of the five rotations
/// the adjoint differentiates, `None` otherwise. Its operands must already be
/// known to lie inside the register, so the shifts cannot overflow.
fn rotation(gate: &GateInstruction) -> Option<(Generator, &GateParam)> {
    use GateInstruction as G;
    let (x, z, y, theta) = match gate {
        G::Rx { qubit, theta } => (1 << qubit, 0, false, theta),
        G::Ry { qubit, theta } => (1 << qubit, 1 << qubit, true, theta),
        G::Rz { qubit, theta } => (0, 1 << qubit, false, theta),
        G::Rxx { q0, q1, theta } => ((1 << q0) | (1 << q1), 0, false, theta),
        G::Rzz { q0, q1, theta } => (0, (1 << q0) | (1 << q1), false, theta),
        _ => return None,
    };
    Some((Generator { x, z, y }, theta))
}

/// `∂⟨O⟩/∂θ = Im ⟨λ|P|ψ⟩` for the rotation with generator `generator`, with
/// `psi` the state right after the rotation and `lambda` the observable pulled
/// back to the same point.
fn derivative(lambda: &Statevector, psi: &Statevector, generator: Generator) -> f64 {
    let inner = kernels::pauli_inner(
        &lambda.data,
        &psi.data,
        generator.x,
        generator.z,
        psi.use_parallel(),
    );
    // A `Y` contributes the factor `i`: Im(i·s) = Re(s).
    let inner = if generator.y {
        inner * C64::new(0.0, 1.0)
    } else {
        inner
    };
    inner.im
}

impl StatevectorSimulator {
    /// The expectation `⟨ψ|O|ψ⟩` of a diagonal observable in the final state of
    /// `circuit`, and its exact derivative with respect to the angle of each
    /// gate at a position of `wrt`, by **adjoint differentiation**.
    ///
    /// `O` is diagonal in the computational basis: `O|b⟩ = weight(b)|b⟩`, with
    /// `b` the basis-state index (qubit `q` is bit `q`, little-endian — the
    /// index of [`Statevector::probabilities`]; with a terminal `MeasureAll` it
    /// is the classical register's value, contract C-3). A Pauli-`Z` product is
    /// `±1` by the parity of its qubits; a weighted sum of them, or any other
    /// diagonal cost, is the matching sum. `weight` is called once per
    /// amplitude, possibly from several threads, and must be a pure function.
    ///
    /// The derivatives come out in `wrt`'s order, one per entry (a position
    /// named twice gets its derivative twice). Each names a gate of
    /// `circuit.gates()` that is a Pauli rotation `exp(−iθP/2)` — `Rx`, `Ry`,
    /// `Rz`, `Rxx` or `Rzz`, the gates the parameter-shift rule differentiates
    /// — and the derivative is with respect to that gate's angle alone: for an
    /// angle several gates share, the caller adds up their derivatives.
    ///
    /// The circuit is applied **as written**: no gate fusion (whatever
    /// [`fusion`](Self::fusion) says) and no transpilation, so the positions in
    /// `wrt` are positions in the circuit received. Every instruction of the
    /// vocabulary may appear between the rotations; the sweep back steps
    /// through each one's inverse. The read-out is the full register: a
    /// terminal `MeasureAll`, or no measurement at all.
    ///
    /// **Memory.** At most two statevectors are alive at once
    /// ([`ADJOINT_STATEVECTORS`](crate::ADJOINT_STATEVECTORS)), plus, on the
    /// parallel path, one partial sum per 4096 amplitudes. The observable is
    /// never materialised as a `2^n` vector. The `max_qubits` ceiling is checked
    /// before anything else.
    ///
    /// **Determinism.** The kernels and reductions give the same bits with and
    /// without the `parallel` feature and for any number of threads: the sums
    /// run over fixed blocks combined in a fixed order. The expectation is
    /// therefore reproducible, but it is not summed in the order of
    /// [`Statevector::probabilities`], so it may differ from a sum over those in
    /// the last bits.
    ///
    /// **Cancellation.** `should_cancel` is polled as
    /// [`run_cancellable`](crate::Simulator::run_cancellable) polls it, through
    /// both sweeps, on the same throttled cadence.
    ///
    /// ```
    /// use polypus_circuit::ParameterizedCircuit;
    /// use polypus_sim::StatevectorSimulator;
    ///
    /// // ⟨Z⟩ after Ry(θ)|0⟩ is cos θ, so its derivative is −sin θ.
    /// let theta = 0.4_f64;
    /// let circuit = ParameterizedCircuit::new(1)
    ///     .ry(0, theta)
    ///     .assign_parameters(&[])
    ///     .unwrap();
    /// let z = |b: usize| if b & 1 == 0 { 1.0 } else { -1.0 };
    /// let grad = StatevectorSimulator::new()
    ///     .adjoint_gradient(&circuit, &[0], z, None)
    ///     .unwrap();
    /// assert!((grad.expectation - theta.cos()).abs() < 1e-12);
    /// assert!((grad.derivatives[0] + theta.sin()).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Before any allocation: [`SimError::TooManyQubits`] above `max_qubits`;
    /// [`SimError::QubitIndexOutOfRange`], [`SimError::RepeatedQubit`] and
    /// [`SimError::GateAfterMeasure`] as for
    /// [`run`](crate::Simulator::run); [`SimError::PartialMeasurement`] for a
    /// `Measure` of individual qubits; [`SimError::GateIndexOutOfRange`] for a
    /// position past the circuit; [`SimError::NotDifferentiable`] for a gate
    /// in `wrt` that is not one of the five rotations; and the angle errors
    /// ([`SimError::UnboundParameter`], [`SimError::UnboundExpression`],
    /// [`SimError::NonFiniteAmplitude`]) of a rotation in `wrt`. While running:
    /// the errors of [`Statevector::apply`] on any other gate,
    /// [`SimError::NonFiniteWeight`] if `weight` is not finite somewhere, and
    /// [`SimError::Cancelled`].
    pub fn adjoint_gradient<W>(
        &self,
        circuit: &ConcreteCircuit,
        wrt: &[usize],
        weight: W,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<AdjointGradient, SimError>
    where
        W: Fn(usize) -> f64 + Sync,
    {
        let n = circuit.num_qubits();
        if n > self.max_qubits {
            return Err(SimError::TooManyQubits {
                requested: n,
                max: self.max_qubits,
            });
        }
        let gates = circuit.gates();
        // The pre-checks of `run_cancellable`, in its order: before the `2^n`
        // allocation, so a circuit that can never run costs nothing.
        if let Some(qubit) = polypus_circuit::qubit_index_violation(gates, n) {
            return Err(SimError::QubitIndexOutOfRange {
                qubit,
                num_qubits: n,
            });
        }
        if let Some(qubit) = polypus_circuit::repeated_qubit_violation(gates) {
            return Err(SimError::RepeatedQubit { qubit });
        }
        if let Some(qubit) = polypus_circuit::terminal_measurement_violation(gates) {
            return Err(SimError::GateAfterMeasure { qubit });
        }
        if let Some(qubit) = gates.iter().find_map(|gate| match gate {
            GateInstruction::Measure { qubit, .. } => Some(*qubit),
            _ => None,
        }) {
            return Err(SimError::PartialMeasurement { qubit });
        }

        // Every gate to differentiate, with the slot of `derivatives` it fills,
        // walked from the last position to the first by the sweep back.
        let mut targets: Vec<(usize, usize, Generator)> = Vec::with_capacity(wrt.len());
        for (slot, &index) in wrt.iter().enumerate() {
            let gate = gates.get(index).ok_or(SimError::GateIndexOutOfRange {
                index,
                num_gates: gates.len(),
            })?;
            let (generator, theta) = rotation(gate).ok_or_else(|| SimError::NotDifferentiable {
                index,
                instruction: format!("{gate:?}"),
            })?;
            angle(theta)?;
            targets.push((index, slot, generator));
        }
        targets.sort_unstable_by_key(|&(index, ..)| std::cmp::Reverse(index));

        log::debug!(
            "adjoint gradient of a {n}-qubit circuit ({} gate(s), {} derivative(s))",
            gates.len(),
            wrt.len()
        );
        let mut cancellation = CancellationCheck::new(should_cancel, n);

        // Forward: |ψ⟩ = U_N ⋯ U_1 |0…0⟩, gate by gate as written.
        let mut psi = Statevector::new(n)?;
        psi.set_parallel_threshold(self.parallel_threshold);
        for gate in gates {
            poll(&mut cancellation)?;
            psi.apply(gate)?;
        }

        // λ = O|ψ⟩, the second (and last) statevector; ⟨O⟩ on the way.
        let mut lambda = psi.clone();
        let (expectation, finite) = kernels::weigh(&mut lambda.data, &weight, psi.use_parallel());
        if !finite {
            return Err(SimError::NonFiniteWeight);
        }

        let mut derivatives = vec![0.0; wrt.len()];
        backward(
            &mut psi,
            &mut lambda,
            gates,
            &targets,
            &mut derivatives,
            &mut cancellation,
        )?;
        Ok(AdjointGradient {
            expectation,
            derivatives,
        })
    }
}

/// The sweep back: from the end of the circuit down to the first gate in
/// `targets` (sorted by descending position), read each target's derivative
/// off `(λ, ψ)` and step both back through the gate. Nothing before the first
/// target is undone — the encoders at the start of a model cost nothing here.
fn backward(
    psi: &mut Statevector,
    lambda: &mut Statevector,
    gates: &[GateInstruction],
    targets: &[(usize, usize, Generator)],
    derivatives: &mut [f64],
    cancellation: &mut CancellationCheck,
) -> Result<(), SimError> {
    let Some(&(first, ..)) = targets.last() else {
        return Ok(());
    };
    let mut pending = targets.iter().peekable();
    // Each gate's inverse, built once for both states into one buffer.
    let mut inverse = Vec::new();
    for i in (first..gates.len()).rev() {
        poll(cancellation)?;
        let mut here = None;
        while let Some(&(_, slot, generator)) = pending.next_if(|&&(index, ..)| index == i) {
            let d = *here.get_or_insert_with(|| derivative(lambda, psi, generator));
            derivatives[slot] = d;
        }
        if i == first {
            break;
        }
        inverse.clear();
        psi.push_inverse(&gates[i], &mut inverse)?;
        for part in &inverse {
            psi.apply(part)?;
        }
        for part in &inverse {
            lambda.apply(part)?;
        }
    }
    Ok(())
}

/// One poll of the cancellation hook, once per gate of either sweep.
fn poll(cancellation: &mut CancellationCheck) -> Result<(), SimError> {
    if cancellation.cancelled() {
        log::debug!("adjoint gradient cancelled by the caller");
        Err(SimError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_QUBITS;
    use polypus_circuit::GateParam::Fixed;

    /// Gates for a sweep back that must outlive several checkpoints, and for
    /// one that is run to completion (the sizes of `tests/cancellation.rs`).
    const LONG: usize = 60_000;
    const SHORT: usize = 3_000;

    /// A target rotation at position 0, a rotation by `NaN` at position 1, and
    /// `filler` cheap rotations after it: the sweep back from the end must step
    /// through the poisoned gate before it reaches the target.
    fn poisoned_sweep(n: usize, filler: usize) -> Vec<GateInstruction> {
        let mut gates = vec![
            GateInstruction::Ry {
                qubit: 0,
                theta: Fixed(0.1),
            },
            GateInstruction::Rx {
                qubit: 0,
                theta: Fixed(f64::NAN),
            },
        ];
        gates.extend((0..filler).map(|i| GateInstruction::Rx {
            qubit: i % n,
            theta: Fixed(0.001),
        }));
        gates
    }

    fn sweep(
        gates: &[GateInstruction],
        n: usize,
        hook: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<(), SimError> {
        let (generator, _) = rotation(&gates[0]).expect("Ry");
        let mut psi = Statevector::new(n).expect("well below MAX_QUBITS");
        psi.set_parallel_threshold(MAX_QUBITS + 1);
        let mut lambda = psi.clone();
        let mut derivatives = [0.0];
        let mut cancellation = CancellationCheck::new(hook, n);
        backward(
            &mut psi,
            &mut lambda,
            gates,
            &[(0, 0, generator)],
            &mut derivatives,
            &mut cancellation,
        )
    }

    /// The sweep back polls the hook like the forward one. Timing-independent,
    /// as `tests/cancellation.rs` is: the error variant alone says how far the
    /// sweep got — `NonFiniteAmplitude` means it reached the poisoned gate,
    /// `Cancelled` that it stopped before.
    #[test]
    fn the_backward_sweep_is_cancellable() {
        let n = 14;
        // Control: run to completion, the sweep does reach the poisoned gate.
        let mut never = || false;
        assert_eq!(
            sweep(&poisoned_sweep(n, SHORT), n, Some(&mut never)),
            Err(SimError::NonFiniteAmplitude)
        );
        let mut stop = || true;
        assert_eq!(
            sweep(&poisoned_sweep(n, LONG), n, Some(&mut stop)),
            Err(SimError::Cancelled)
        );
    }
}
