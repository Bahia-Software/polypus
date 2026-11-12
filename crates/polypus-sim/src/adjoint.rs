//! Adjoint differentiation: the exact gradient of a diagonal expectation value
//! in one forward and one backward sweep, whatever the number of parameters.
//!
//! For `|ψ⟩ = U_N ⋯ U_1 |0…0⟩` and a diagonal observable `O`, the derivative of
//! `⟨ψ|O|ψ⟩` with respect to an angle `α` of gate `U_i` is
//!
//! ```text
//! ∂⟨O⟩/∂α = 2·Im ⟨λ_i| G_α |ψ_i⟩,   ψ_i = U_i ⋯ U_1 |0…0⟩,   λ_i = U_{i+1}† ⋯ U_N† O |ψ⟩
//! ```
//!
//! (`2·Re⟨λ_i| ∂_αU_i |ψ_{i−1}⟩` with `∂_αU_i = −i·G_α·U_i`), where
//! `G_α = i·(∂_αU_i)·U_i†` is the Hermitian **generator** of the angle. A Pauli
//! rotation `exp(−iθP/2)` has `G = P/2`, so its derivative is `Im⟨λ_i|P|ψ_i⟩`;
//! every other gate here has a 2×2 generator on its target qubit, restricted to
//! the basis states whose control bit is `1` for a controlled gate
//! (`Π₀⊗I + Π₁⊗V(α)` has `G = Π₁⊗(i·∂V·V†)`: Hermitian, not unitary). The table
//! of generators is [`family`]. Both `ψ_i` and `λ_i` are reached by stepping
//! back from the end of the circuit one gate at a time — `ψ ← U_i†ψ`,
//! `λ ← U_i†λ` — so the sweep holds exactly two statevectors
//! ([`ADJOINT_STATEVECTORS`](crate::ADJOINT_STATEVECTORS)) and costs about three
//! circuit simulations plus one read of the pair per differentiated angle, where
//! the parameter-shift rule costs two simulations per gate.

use crate::error::SimError;
use crate::simulator::CancellationCheck;
use crate::statevector::{angle, Statevector};
use crate::{kernels, StatevectorSimulator, C64};
use polypus_circuit::{AngleRef, ConcreteCircuit, GateInstruction, GateParam};

/// The result of [`StatevectorSimulator::adjoint_gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdjointGradient {
    /// `⟨ψ|O|ψ⟩`: the expectation of the observable in the circuit's final
    /// state, summed in the simulator's fixed block order (see
    /// [`adjoint_gradient`](StatevectorSimulator::adjoint_gradient)).
    pub expectation: f64,
    /// `∂⟨O⟩/∂α` of each angle named in `wrt`, in `wrt`'s order.
    pub derivatives: Vec<f64>,
}

/// The generator `G` of one angle, in the form a reduction kernel reads.
#[derive(Debug, Clone, Copy)]
enum Generator {
    /// `P/2`, for a Pauli rotation `exp(−iθP/2)`: `P|b⟩ = i^y · (−1)^{|b ∧ z|}
    /// · |b ⊕ x⟩`, with `x` and `z` bit masks over the qubits and `y` whether
    /// `P` carries one `Y` (`Y = i·X·Z`).
    Pauli { x: usize, z: usize, y: bool },
    /// `Π ⊗ g`: the Hermitian 2×2 `g` on qubit `target`, on the basis states
    /// with every bit of `control` set (`0`: every basis state).
    Local {
        target: usize,
        control: usize,
        g: [[C64; 2]; 2],
    },
    /// The gate does not depend on the angle (`u0`): the derivative is `0`.
    Zero,
}

/// How a gate depends on its angles: one row of the table of generators.
#[derive(Debug, Clone, Copy)]
enum Family {
    /// `exp(−iθP/2)`, generator `P/2` (see [`Generator::Pauli`]).
    Pauli { x: usize, z: usize, y: bool },
    /// `Π ⊗ g` with `g` constant: a controlled rotation (`g = P/2`) or a phase
    /// gate (`g = −|1⟩⟨1|`, from `diag(1, e^{iλ})`).
    Constant {
        target: usize,
        control: usize,
        g: [[C64; 2]; 2],
    },
    /// `U(θ, φ, λ)` (the matrix of `gates::u`) on `target`, controlled by
    /// `control`. `theta` is `U2`'s fixed `π/2`, whose angles are then
    /// `φ, λ`; `gamma` is `Cu`'s phase `e^{iγ}`, a fourth angle — relative to
    /// the control, so not a global phase.
    U {
        target: usize,
        control: usize,
        theta: Option<f64>,
        gamma: bool,
    },
    /// `u0(γ)`: the identity, whatever its angle.
    Identity,
}

fn re(x: f64) -> C64 {
    C64::new(x, 0.0)
}

fn im(x: f64) -> C64 {
    C64::new(0.0, x)
}

/// `X/2`, `Y/2`, `Z/2`: the generators of `rx`, `ry`, `rz` on the target.
fn half_x() -> [[C64; 2]; 2] {
    [[re(0.0), re(0.5)], [re(0.5), re(0.0)]]
}

fn half_y() -> [[C64; 2]; 2] {
    [[re(0.0), im(-0.5)], [im(0.5), re(0.0)]]
}

fn half_z() -> [[C64; 2]; 2] {
    [[re(0.5), re(0.0)], [re(0.0), re(-0.5)]]
}

/// `−|1⟩⟨1|`, the generator of `diag(1, e^{iλ})`: `i·(i·|1⟩⟨1|)`.
fn minus_one() -> [[C64; 2]; 2] {
    [[re(0.0), re(0.0)], [re(0.0), re(-1.0)]]
}

/// The generator of angle `k` (`0 = θ, 1 = φ, 2 = λ`) of
/// `U(θ, φ, λ) = P(φ)·Ry(θ)·P(λ)`, which is `gates::u`'s matrix:
///
/// - `θ`: `½·P(φ)·Y·P(φ)† = ½·[[0, −i·e^{−iφ}], [i·e^{iφ}, 0]]`;
/// - `φ`: `−|1⟩⟨1|`, the phase gate's;
/// - `λ`: `−U·|1⟩⟨1|·U† = [[−sin²(θ/2), s·c·e^{−iφ}], [s·c·e^{iφ}, −cos²(θ/2)]]`,
///   `s·c = sin(θ/2)·cos(θ/2)`.
fn u_generator(k: usize, theta: f64, phi: f64) -> [[C64; 2]; 2] {
    let e = C64::from_polar(1.0, phi);
    match k {
        0 => [[re(0.0), im(-0.5) * e.conj()], [im(0.5) * e, re(0.0)]],
        1 => minus_one(),
        _ => {
            let (s, c) = (theta / 2.0).sin_cos();
            [[re(-s * s), e.conj() * (s * c)], [e * (s * c), re(-c * c)]]
        }
    }
}

/// The table of generators: how `gate` depends on each of its angles, and the
/// angles in [`AngleRef::angle`] order; `None` for a gate the adjoint does not
/// differentiate (no angle, or a call of a declared gate, whose body may use
/// its angles in any way). The generators are derived from the matrices
/// [`Statevector::apply`] applies, and the test
/// `every_generator_is_the_derivative_of_the_simulated_matrix` holds each one
/// to them. Its operands must already be known to lie inside the register,
/// so the shifts cannot overflow.
fn family(gate: &GateInstruction) -> Option<(Family, Vec<&GateParam>)> {
    use GateInstruction as G;
    let pauli = |x: usize, z: usize, y: bool| Family::Pauli { x, z, y };
    let constant = |target: usize, control: usize, g| Family::Constant { target, control, g };
    let u = |target: usize, control: usize, theta, gamma| Family::U {
        target,
        control,
        theta,
        gamma,
    };
    Some(match gate {
        G::Rx { qubit, theta } => (pauli(1 << qubit, 0, false), vec![theta]),
        G::Ry { qubit, theta } => (pauli(1 << qubit, 1 << qubit, true), vec![theta]),
        G::Rz { qubit, theta } => (pauli(0, 1 << qubit, false), vec![theta]),
        G::Rxx { q0, q1, theta } => (pauli((1 << q0) | (1 << q1), 0, false), vec![theta]),
        G::Rzz { q0, q1, theta } => (pauli(0, (1 << q0) | (1 << q1), false), vec![theta]),
        G::Crx {
            control,
            target,
            theta,
        } => (constant(*target, 1 << control, half_x()), vec![theta]),
        G::Cry {
            control,
            target,
            theta,
        } => (constant(*target, 1 << control, half_y()), vec![theta]),
        G::Crz {
            control,
            target,
            theta,
        } => (constant(*target, 1 << control, half_z()), vec![theta]),
        // `Cp` is symmetric (`e^{iθ}` on `|11⟩`); `Cu1` is the same operator,
        // applied as `q0` controlling `diag(1, e^{iθ})` on `q1`.
        G::Cp { q0, q1, theta } | G::Cu1 { q0, q1, theta } => {
            (constant(*q1, 1 << q0, minus_one()), vec![theta])
        }
        G::P { qubit, lam } | G::U1 { qubit, lam } => (constant(*qubit, 0, minus_one()), vec![lam]),
        G::U {
            qubit,
            theta,
            phi,
            lam,
        }
        | G::UGate {
            qubit,
            theta,
            phi,
            lam,
        } => (u(*qubit, 0, None, false), vec![theta, phi, lam]),
        G::U2 { qubit, phi, lam } => (
            u(*qubit, 0, Some(std::f64::consts::FRAC_PI_2), false),
            vec![phi, lam],
        ),
        G::Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => (u(*target, 1 << control, None, false), vec![theta, phi, lam]),
        G::Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => (
            u(*target, 1 << control, None, true),
            vec![theta, phi, lam, gamma],
        ),
        G::U0 { gamma, .. } => (Family::Identity, vec![gamma]),
        _ => return None,
    })
}

/// The generator of angle `at.angle` of `gate`, the gate at `at.position`,
/// with every angle of the gate resolved (so a gate that cannot run fails here,
/// before anything is allocated, with the error [`Statevector::apply`] would
/// give).
fn generator(gate: &GateInstruction, at: AngleRef) -> Result<Generator, SimError> {
    let (family, angles) = family(gate).ok_or_else(|| SimError::NotDifferentiable {
        index: at.position,
        instruction: format!("{gate:?}"),
    })?;
    if at.angle >= angles.len() {
        return Err(SimError::AngleIndexOutOfRange {
            index: at.position,
            angle: at.angle,
            num_angles: angles.len(),
            instruction: format!("{gate:?}"),
        });
    }
    let values = angles
        .into_iter()
        .map(angle)
        .collect::<Result<Vec<f64>, SimError>>()?;
    Ok(match family {
        Family::Pauli { x, z, y } => Generator::Pauli { x, z, y },
        Family::Constant { target, control, g } => Generator::Local { target, control, g },
        Family::U {
            target,
            control,
            theta,
            gamma,
        } => {
            // `U2`'s angles are `φ, λ`: the `u` angle index is one more.
            let (k, theta, phi) = match theta {
                Some(theta) => (at.angle + 1, theta, values[0]),
                None => (at.angle, values[0], values[1]),
            };
            let g = if gamma && k == 3 {
                // `∂(e^{iγ}·U) = i·e^{iγ}·U`: `G = −I` on the target.
                [[re(-1.0), re(0.0)], [re(0.0), re(-1.0)]]
            } else {
                u_generator(k, theta, phi)
            };
            Generator::Local { target, control, g }
        }
        Family::Identity => Generator::Zero,
    })
}

/// `∂⟨O⟩/∂α = 2·Im ⟨λ|G|ψ⟩` for the angle with generator `generator`, with
/// `psi` the state right after the gate and `lambda` the observable pulled
/// back to the same point.
fn derivative(lambda: &Statevector, psi: &Statevector, generator: Generator) -> f64 {
    match generator {
        // `2·Im⟨λ|P/2|ψ⟩ = Im⟨λ|P|ψ⟩`: the Pauli kernel, as before the other
        // gates were differentiated.
        Generator::Pauli { x, z, y } => {
            let inner = kernels::pauli_inner(&lambda.data, &psi.data, x, z, psi.use_parallel());
            // A `Y` contributes the factor `i`: Im(i·s) = Re(s).
            let inner = if y { inner * C64::new(0.0, 1.0) } else { inner };
            inner.im
        }
        Generator::Local { target, control, g } => {
            let inner = kernels::generator_inner(
                &lambda.data,
                &psi.data,
                target,
                control,
                &g,
                psi.use_parallel(),
            );
            2.0 * inner.im
        }
        Generator::Zero => 0.0,
    }
}

impl StatevectorSimulator {
    /// The expectation `⟨ψ|O|ψ⟩` of a diagonal observable in the final state of
    /// `circuit`, and its exact derivative with respect to each gate angle
    /// named in `wrt`, by **adjoint differentiation**.
    ///
    /// `O` is diagonal in the computational basis: `O|b⟩ = weight(b)|b⟩`, with
    /// `b` the basis-state index (qubit `q` is bit `q`, little-endian — the
    /// index of [`Statevector::probabilities`]; with a terminal `MeasureAll` it
    /// is the classical register's value, contract C-3). A Pauli-`Z` product is
    /// `±1` by the parity of its qubits; a weighted sum of them, or any other
    /// diagonal cost, is the matching sum. `weight` is called once per
    /// amplitude, possibly from several threads, and must be a pure function.
    ///
    /// The derivatives come out in `wrt`'s order, one per entry (an angle
    /// named twice gets its derivative twice). Each [`AngleRef`] names a gate
    /// of `circuit.gates()` and one of its angles; the gates differentiated
    /// are the Pauli rotations `exp(−iθP/2)` (`Rx`, `Ry`, `Rz`, `Rxx`, `Rzz`,
    /// the gates the parameter-shift rule differentiates), the controlled
    /// rotations (`Crx`, `Cry`, `Crz`), the phase gates (`P`, `U1`, `Cp`,
    /// `Cu1`), the `u` family (`U`, `UGate`, `U2`, `Cu3`, and `Cu` with its
    /// phase `γ`) and `U0`, whose derivative is `0` (see [`AngleRef`] for the
    /// order of each gate's angles). The derivative is with respect to that
    /// one angle of that one gate: for an angle several gates share, the
    /// caller adds up their derivatives.
    ///
    /// The circuit is applied **as written**: no gate fusion (whatever
    /// [`fusion`](Self::fusion) says) and no transpilation, so the positions in
    /// `wrt` are positions in the circuit received. Every instruction of the
    /// vocabulary may appear between the differentiated gates; the sweep back
    /// steps through each one's inverse. The read-out is the full register: a
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
    /// use polypus_sim::{AngleRef, StatevectorSimulator};
    ///
    /// // ⟨Z⟩ after U(θ, φ, λ)|0⟩ is cos θ: its derivative is −sin θ with
    /// // respect to θ (angle 0) and 0 with respect to λ (angle 2).
    /// let theta = 0.4_f64;
    /// let circuit = ParameterizedCircuit::new(1)
    ///     .u(0, theta, 0.3, -1.1)
    ///     .assign_parameters(&[])
    ///     .unwrap();
    /// let z = |b: usize| if b & 1 == 0 { 1.0 } else { -1.0 };
    /// let wrt = [AngleRef::new(0, 0), AngleRef::new(0, 2)];
    /// let grad = StatevectorSimulator::new()
    ///     .adjoint_gradient(&circuit, &wrt, z, None)
    ///     .unwrap();
    /// assert!((grad.expectation - theta.cos()).abs() < 1e-12);
    /// assert!((grad.derivatives[0] + theta.sin()).abs() < 1e-12);
    /// assert!(grad.derivatives[1].abs() < 1e-12);
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
    /// in `wrt` that is none of the above (no angle, or the call of a declared
    /// gate); [`SimError::AngleIndexOutOfRange`] for an angle the gate does not
    /// have; and the angle errors ([`SimError::UnboundParameter`],
    /// [`SimError::UnboundExpression`], [`SimError::NonFiniteAmplitude`]) of
    /// any angle of a gate in `wrt`. While running:
    /// the errors of [`Statevector::apply`] on any other gate,
    /// [`SimError::NonFiniteWeight`] if `weight` is not finite somewhere, and
    /// [`SimError::Cancelled`].
    pub fn adjoint_gradient<W>(
        &self,
        circuit: &ConcreteCircuit,
        wrt: &[AngleRef],
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

        // Every angle to differentiate, with the slot of `derivatives` it
        // fills, walked from the last position to the first by the sweep back
        // (and, at one position, by angle, so repeats are adjacent).
        let mut targets: Vec<Target> = Vec::with_capacity(wrt.len());
        for (slot, &at) in wrt.iter().enumerate() {
            let gate = gates
                .get(at.position)
                .ok_or(SimError::GateIndexOutOfRange {
                    index: at.position,
                    num_gates: gates.len(),
                })?;
            targets.push(Target {
                at,
                slot,
                generator: generator(gate, at)?,
            });
        }
        targets.sort_unstable_by_key(|target| {
            (std::cmp::Reverse(target.at.position), target.at.angle)
        });

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

/// One entry of `wrt`: the angle, the slot of `derivatives` it fills, and its
/// generator.
#[derive(Debug, Clone, Copy)]
struct Target {
    at: AngleRef,
    slot: usize,
    generator: Generator,
}

/// The sweep back: from the end of the circuit down to the first gate in
/// `targets` (sorted by descending position, then by angle), read each
/// target's derivative off `(λ, ψ)` and step both back through the gate.
/// Nothing before the first target is undone — the encoders at the start of a
/// model cost nothing here.
fn backward(
    psi: &mut Statevector,
    lambda: &mut Statevector,
    gates: &[GateInstruction],
    targets: &[Target],
    derivatives: &mut [f64],
    cancellation: &mut CancellationCheck,
) -> Result<(), SimError> {
    let Some(first) = targets.last().map(|target| target.at.position) else {
        return Ok(());
    };
    let mut pending = targets.iter().peekable();
    for i in (first..gates.len()).rev() {
        poll(cancellation)?;
        // An angle named twice is read once: its repeats are adjacent.
        let mut here: Option<(usize, f64)> = None;
        while let Some(target) = pending.next_if(|target| target.at.position == i) {
            let d = match here {
                Some((angle, d)) if angle == target.at.angle => d,
                _ => derivative(lambda, psi, target.generator),
            };
            here = Some((target.at.angle, d));
            derivatives[target.slot] = d;
        }
        if i == first {
            break;
        }
        psi.apply_inverse(&gates[i])?;
        lambda.apply_inverse(&gates[i])?;
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
        let at = AngleRef::new(0, 0);
        let generator = generator(&gates[0], at).expect("Ry");
        let mut psi = Statevector::new(n).expect("well below MAX_QUBITS");
        psi.set_parallel_threshold(MAX_QUBITS + 1);
        let mut lambda = psi.clone();
        let mut derivatives = [0.0];
        let mut cancellation = CancellationCheck::new(hook, n);
        backward(
            &mut psi,
            &mut lambda,
            gates,
            &[Target {
                at,
                slot: 0,
                generator,
            }],
            &mut derivatives,
            &mut cancellation,
        )
    }

    /// Every differentiable gate kind once, with generic angles, on three
    /// qubits; the controlled ones with the control above and below the target.
    fn table_gates() -> Vec<GateInstruction> {
        use GateInstruction as G;
        vec![
            G::Rx {
                qubit: 1,
                theta: Fixed(0.7),
            },
            G::Ry {
                qubit: 2,
                theta: Fixed(-1.3),
            },
            G::Rz {
                qubit: 0,
                theta: Fixed(0.4),
            },
            G::Rxx {
                q0: 2,
                q1: 0,
                theta: Fixed(2.1),
            },
            G::Rzz {
                q0: 1,
                q1: 2,
                theta: Fixed(-0.9),
            },
            G::Crx {
                control: 2,
                target: 0,
                theta: Fixed(0.6),
            },
            G::Cry {
                control: 0,
                target: 1,
                theta: Fixed(-2.2),
            },
            G::Crz {
                control: 1,
                target: 2,
                theta: Fixed(1.4),
            },
            G::Cp {
                q0: 2,
                q1: 1,
                theta: Fixed(0.8),
            },
            G::Cu1 {
                q0: 0,
                q1: 2,
                theta: Fixed(-1.7),
            },
            G::P {
                qubit: 1,
                lam: Fixed(0.3),
            },
            G::U1 {
                qubit: 0,
                lam: Fixed(2.6),
            },
            G::U {
                qubit: 2,
                theta: Fixed(0.9),
                phi: Fixed(-0.4),
                lam: Fixed(1.9),
            },
            G::UGate {
                qubit: 0,
                theta: Fixed(-1.1),
                phi: Fixed(2.3),
                lam: Fixed(0.2),
            },
            G::U2 {
                qubit: 1,
                phi: Fixed(1.2),
                lam: Fixed(-0.6),
            },
            G::Cu3 {
                control: 1,
                target: 0,
                theta: Fixed(1.3),
                phi: Fixed(0.5),
                lam: Fixed(-2.4),
            },
            G::Cu {
                control: 0,
                target: 2,
                theta: Fixed(-0.8),
                phi: Fixed(1.6),
                lam: Fixed(0.7),
                gamma: Fixed(-1.2),
            },
            G::U0 {
                qubit: 2,
                gamma: Fixed(0.5),
            },
        ]
    }

    /// `gate` with its angle `which` moved by `delta`, the angles counted in
    /// `AngleRef`'s documented order — written out here independently of
    /// `family`, so a wrong order there shows up as a mismatch.
    fn moved(gate: &GateInstruction, which: usize, delta: f64) -> GateInstruction {
        use GateInstruction as G;
        let mut gate = gate.clone();
        let mut angles: Vec<&mut GateParam> = match &mut gate {
            G::Rx { theta, .. }
            | G::Ry { theta, .. }
            | G::Rz { theta, .. }
            | G::Rxx { theta, .. }
            | G::Rzz { theta, .. }
            | G::Crx { theta, .. }
            | G::Cry { theta, .. }
            | G::Crz { theta, .. }
            | G::Cp { theta, .. }
            | G::Cu1 { theta, .. } => vec![theta],
            G::P { lam, .. } | G::U1 { lam, .. } => vec![lam],
            G::U0 { gamma, .. } => vec![gamma],
            G::U {
                theta, phi, lam, ..
            }
            | G::UGate {
                theta, phi, lam, ..
            }
            | G::Cu3 {
                theta, phi, lam, ..
            } => vec![theta, phi, lam],
            G::U2 { phi, lam, .. } => vec![phi, lam],
            G::Cu {
                theta,
                phi,
                lam,
                gamma,
                ..
            } => vec![theta, phi, lam, gamma],
            other => panic!("{other:?} is not in the table"),
        };
        let GateParam::Fixed(v) = *angles[which] else {
            panic!("fixed angles only");
        };
        *angles[which] = GateParam::Fixed(v + delta);
        gate
    }

    /// The matrix whose columns are `columns`.
    fn from_columns(columns: &[Vec<C64>]) -> Vec<Vec<C64>> {
        (0..columns.len())
            .map(|row| columns.iter().map(|column| column[row]).collect())
            .collect()
    }

    /// The `2^n × 2^n` matrix the simulator applies for `gate`: column `b` is
    /// `apply(gate)` on `|b⟩`.
    fn simulated(gate: &GateInstruction, n: usize) -> Vec<Vec<C64>> {
        let columns: Vec<Vec<C64>> = (0..1usize << n)
            .map(|col| {
                let mut sv = Statevector::new(n).expect("small");
                sv.data[0] = C64::new(0.0, 0.0);
                sv.data[col] = C64::new(1.0, 0.0);
                sv.apply(gate).expect("a valid gate");
                sv.data
            })
            .collect();
        from_columns(&columns)
    }

    /// `G` as a dense `2^n × 2^n` matrix, from its definition.
    fn dense(generator: Generator, n: usize) -> Vec<Vec<C64>> {
        let dim = 1usize << n;
        let columns: Vec<Vec<C64>> = (0..dim)
            .map(|col| {
                let mut column = vec![C64::new(0.0, 0.0); dim];
                match generator {
                    Generator::Pauli { x, z, y } => {
                        let sign = if (col & z).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let phase = if y { C64::new(0.0, 1.0) } else { re(1.0) };
                        column[col ^ x] = phase * (0.5 * sign);
                    }
                    Generator::Local { target, control, g } => {
                        if col & control == control {
                            let c = (col >> target) & 1;
                            for (r, g_row) in g.iter().enumerate() {
                                column[(col & !(1 << target)) | (r << target)] = g_row[c];
                            }
                        }
                    }
                    Generator::Zero => {}
                }
                column
            })
            .collect();
        from_columns(&columns)
    }

    fn product(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
        let dim = a.len();
        (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| (0..dim).map(|k| a[i][k] * b[k][j]).sum())
                    .collect()
            })
            .collect()
    }

    /// The table of generators against the simulator itself: for every gate
    /// and every angle, `G` is Hermitian and `−i·G·U` is the derivative of the
    /// matrix `Statevector::apply` applies (a central difference over that
    /// matrix, step `1e-5`: truncation `≈ h²/6 ≈ 2e-11`). This ties each
    /// generator to what is simulated — the phase conventions of `u`, `cu3`
    /// and `cu`'s `γ` included — independently of any expectation value.
    #[test]
    fn every_generator_is_the_derivative_of_the_simulated_matrix() {
        let n = 3;
        let h = 1e-5;
        for gate in table_gates() {
            let (_, angles) = family(&gate).expect("in the table");
            assert_eq!(
                generator(&gate, AngleRef::new(0, angles.len())).unwrap_err(),
                SimError::AngleIndexOutOfRange {
                    index: 0,
                    angle: angles.len(),
                    num_angles: angles.len(),
                    instruction: format!("{gate:?}"),
                }
            );
            let u = simulated(&gate, n);
            for which in 0..angles.len() {
                let g = dense(
                    generator(&gate, AngleRef::new(0, which)).expect("differentiable"),
                    n,
                );
                let plus = simulated(&moved(&gate, which, h), n);
                let minus = simulated(&moved(&gate, which, -h), n);
                let g_u = product(&g, &u);
                for r in 0..1 << n {
                    for c in 0..1 << n {
                        assert!(
                            (g[r][c] - g[c][r].conj()).norm() < 1e-15,
                            "{gate:?} angle {which}: G is not Hermitian at ({r}, {c})"
                        );
                        let fd = (plus[r][c] - minus[r][c]) / (2.0 * h);
                        let exact = C64::new(0.0, -1.0) * g_u[r][c];
                        assert!(
                            (fd - exact).norm() < 1e-9,
                            "{gate:?} angle {which} at ({r}, {c}): −iGU {exact} vs {fd}"
                        );
                    }
                }
            }
        }
    }

    /// `u0` does not depend on its angle: its derivative is exactly `0`, read
    /// off no buffer.
    #[test]
    fn the_derivative_of_u0_is_exactly_zero() {
        let gate = GateInstruction::U0 {
            qubit: 0,
            gamma: Fixed(0.4),
        };
        let generator = generator(&gate, AngleRef::new(0, 0)).expect("differentiable");
        let mut psi = Statevector::new(1).expect("small");
        psi.data = vec![C64::new(0.6, 0.0), C64::new(0.0, 0.8)];
        assert_eq!(
            derivative(&psi, &psi, generator).to_bits(),
            0.0f64.to_bits()
        );
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
