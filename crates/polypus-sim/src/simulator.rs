//! The runnable simulator: turns a [`ConcreteCircuit`] into a [`Statevector`]
//! (and, optionally, sampled measurement counts).

use crate::error::SimError;
use crate::rng::SplitMix64;
use crate::statevector::Statevector;
use polypus_circuit::{ConcreteCircuit, GateInstruction};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Abstraction over simulation backends. A future density-matrix or GPU
/// backend can implement this same contract.
pub trait Simulator {
    /// Evolve `|0…0⟩` through `circuit` and return the final state.
    fn run(&self, circuit: &ConcreteCircuit) -> Result<Statevector, SimError>;

    /// Evolve `|0…0⟩` through `circuit`, with the option to stop part-way.
    ///
    /// `should_cancel`, when given, is invoked periodically while the circuit is
    /// being applied; returning `true` abandons the run and yields
    /// [`SimError::Cancelled`]. The hook is a plain `FnMut() -> bool` precisely
    /// so this crate needs to know nothing about *why* the caller wants to
    /// stop — a signal check, a deadline, a user-facing cancel button — which
    /// keeps `polypus-sim` free of its caller's concerns. Passing `None` is
    /// exactly [`run`](Self::run).
    ///
    /// **"Periodically" is the whole contract.** A hook can cost orders of
    /// magnitude more than a gate, so an implementation is free to throttle the
    /// calls (this crate's does, on a wall-clock cadence that adapts to how
    /// expensive the hook turns out to be); never rely on the hook being called
    /// once per gate, or a fixed number of times.
    ///
    /// The default implementation ignores `should_cancel` and runs to
    /// completion, so a backend with no cancellation point of its own stays
    /// correct and simply never returns [`SimError::Cancelled`].
    ///
    /// ```
    /// use polypus_circuit::ParameterizedCircuit;
    /// use polypus_sim::{Simulator, StatevectorSimulator};
    ///
    /// let circuit = ParameterizedCircuit::new(2)
    ///     .h(0)
    ///     .cx(0, 1)
    ///     .assign_parameters(&[])
    ///     .unwrap();
    ///
    /// // A hook that always asks to stop -- and a circuit that finishes anyway.
    /// // Cancellation is prompt, not immediate: two gates are over long before
    /// // the first checkpoint comes due, so the hook is never called (and this
    /// // run never even reads the clock).
    /// let mut stop = || true;
    /// let sv = StatevectorSimulator::new()
    ///     .run_cancellable(&circuit, Some(&mut stop))
    ///     .unwrap();
    /// assert_eq!(sv.num_qubits(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// [`SimError::Cancelled`] if `should_cancel` returned `true`, otherwise
    /// the same errors as [`run`](Self::run).
    fn run_cancellable(
        &self,
        circuit: &ConcreteCircuit,
        _should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<Statevector, SimError> {
        self.run(circuit)
    }
}

/// Wall-clock time that must pass between two calls of a `run_cancellable`
/// cancellation hook — a floor, see [`CANCELLATION_OVERHEAD_DIVISOR`].
///
/// Chosen so the checkpoint is imperceptible as latency: a Ctrl+C is honored
/// within about a frame.
const CANCELLATION_CHECK_INTERVAL: Duration = Duration::from_millis(25);

/// Bound on what the hook may cost the run: it is called at most once per
/// `CANCELLATION_OVERHEAD_DIVISOR` × (its own measured duration), so it can
/// never take more than ~`1/divisor` of the wall clock.
///
/// This is what makes the floor above safe against a hook whose cost the
/// simulator cannot know. `polypus`'s reacquires the GIL, which takes ~1µs when
/// nothing else wants it — but *milliseconds* when another Python thread is
/// running, because the holder only yields on the interpreter's switch interval
/// (5ms by default). On the fixed 25ms cadence alone that was measurable: a
/// contended run cost 1.53x its uncontended self, against 1.21x with no hook at
/// all; deriving the interval from the observed cost brings it back to 1.25x
/// while leaving the cheap, uncontended case on the 25ms floor. Reproduce with
/// `benchmarks/bench_statevector.py`'s contended section — the ratio itself
/// drifts with machine load, so compare the three builds in one sitting rather
/// than against the figures quoted here.
const CANCELLATION_OVERHEAD_DIVISOR: u32 = 20;

/// Amplitude updates to perform between two `Instant::now()` reads while a
/// cancellation hook is installed.
///
/// A clock read is cheap (tens of nanoseconds) but not free, so it too is
/// amortized over several gates. Doing that by *work* rather than by a fixed
/// gate count is what makes a single constant fit the whole qubit range: a gate
/// touches `2^n` amplitudes, so from 16 qubits up one gate already costs far
/// more than a clock read and the stride collapses to 1 (checking every gate,
/// where a larger stride would only delay the checkpoint), while on a 4-qubit
/// circuit the read is amortized over 4096 gates.
const CLOCK_READ_WORK_UNITS: usize = 1 << 16;

/// Throttled front-end for a `run_cancellable` cancellation hook.
///
/// The state a throttle needs only exists when there *is* a hook, so it lives
/// behind the `Option`: a run without one costs exactly one `Option` test per
/// gate and nothing else — no clock read, not even at the start.
struct CancellationCheck<'a>(Option<Throttle<'a>>);

/// A hook plus the two nested throttles that decide when to call it: `stride`
/// gates between clock reads, `interval` between calls.
struct Throttle<'a> {
    /// The caller's hook. `true` means "stop applying gates".
    hook: &'a mut dyn FnMut() -> bool,
    /// Gates to apply between two clock reads; at least 1.
    stride: usize,
    /// Gates left before the next clock read.
    countdown: usize,
    /// When the hook was last called, or when the clock was first read.
    /// `None` until then — a circuit shorter than `stride` gates never reads the
    /// clock at all, so a fast run pays nothing for being cancellable.
    last_check: Option<Instant>,
    /// Wall clock to leave between hook calls: [`CANCELLATION_CHECK_INTERVAL`]
    /// until the hook has been called once, then whatever
    /// [`CANCELLATION_OVERHEAD_DIVISOR`] allows given its measured cost.
    interval: Duration,
}

impl<'a> CancellationCheck<'a> {
    /// A check for a `num_qubits`-wide run driving `hook`.
    fn new(hook: Option<&'a mut dyn FnMut() -> bool>, num_qubits: usize) -> Self {
        CancellationCheck(hook.map(|hook| {
            // `min(16)` keeps the shift in range for any qubit count; the result
            // has already saturated at the `max(1)` floor by then.
            let stride = (CLOCK_READ_WORK_UNITS >> num_qubits.min(16)).max(1);
            Throttle {
                hook,
                stride,
                countdown: stride,
                last_check: None,
                interval: CANCELLATION_CHECK_INTERVAL,
            }
        }))
    }

    /// Whether the caller has asked to stop. Called once per gate — all the
    /// throttling lives in [`Throttle::poll`].
    fn cancelled(&mut self) -> bool {
        match &mut self.0 {
            Some(throttle) => throttle.poll(),
            None => false,
        }
    }
}

impl Throttle<'_> {
    /// One gate's worth of throttling: returns the hook's answer when it is due
    /// to be called, and `false` (without calling it) the rest of the time.
    fn poll(&mut self) -> bool {
        self.countdown -= 1;
        if self.countdown > 0 {
            return false;
        }
        self.countdown = self.stride;
        let now = Instant::now();
        match self.last_check {
            // First clock read of the run. Start the interval here rather than
            // at the run's start: until the gate count justifies looking at the
            // clock, there is nothing to time.
            None => {
                self.last_check = Some(now);
                return false;
            }
            Some(last) => {
                if now.saturating_duration_since(last) < self.interval {
                    return false;
                }
            }
        }
        let stop = (self.hook)();
        // Time the call and re-derive the interval from it, so an expensive hook
        // is simply called less often instead of eating the run. This second
        // clock read costs nothing: it happens once per hook call, not per gate.
        let done = Instant::now();
        self.interval = CANCELLATION_CHECK_INTERVAL.max(
            done.saturating_duration_since(now)
                .saturating_mul(CANCELLATION_OVERHEAD_DIVISOR),
        );
        // Measure the next interval from *after* the call, so the hook's own
        // duration is never counted as part of it.
        self.last_check = Some(done);
        stop
    }
}

/// Dense statevector backend.
///
/// Cheap to construct and clone; holds only configuration. Defaults to
/// [`MAX_QUBITS`](crate::MAX_QUBITS) and the crate's parallel threshold.
#[derive(Debug, Clone)]
pub struct StatevectorSimulator {
    /// Reject circuits needing more than this many qubits.
    pub max_qubits: usize,
    /// Qubit count at or above which gates use the parallel kernels (only with
    /// the `parallel` feature).
    pub parallel_threshold: usize,
}

impl Default for StatevectorSimulator {
    fn default() -> Self {
        StatevectorSimulator {
            max_qubits: crate::MAX_QUBITS,
            parallel_threshold: crate::DEFAULT_PARALLEL_THRESHOLD,
        }
    }
}

impl StatevectorSimulator {
    /// A simulator with default limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `circuit`, then draw `shots` measurements seeded by `seed`.
    ///
    /// Keys of the returned map are classical-register values. Qubits are
    /// mapped to classical bits by the circuit's `Measure`/`MeasureAll`
    /// instructions; if the circuit measures nothing, every qubit is reported
    /// (key = full basis state), matching the "measure all" convention.
    ///
    /// # Errors
    ///
    /// Propagates any [`SimError`] from [`run`](Self::run).
    pub fn run_and_sample(
        &self,
        circuit: &ConcreteCircuit,
        shots: usize,
        seed: u64,
    ) -> Result<HashMap<usize, u64>, SimError> {
        let sv = self.run(circuit)?;
        Ok(sample_projected(circuit, &sv, shots, seed))
    }
}

/// Draw `shots` measurements from an already-evolved statevector `sv`, seeded by
/// `seed`, and project each sampled basis state onto `circuit`'s classical
/// register (the qubit → classical-bit mapping declared by its `Measure` /
/// `MeasureAll` instructions; a circuit that measures nothing reports the full
/// basis state, matching the "measure all" convention).
///
/// This is the sampling half of [`StatevectorSimulator::run_and_sample`],
/// factored out so a caller that evolves a circuit **once** can sample it many
/// times — each batch with its own `seed` — without repeating the (identical,
/// deterministic) state evolution. For a given `sv` and `seed` the result is
/// byte-identical to `run_and_sample`, which is what lets shot batches be
/// distributed across replicas from a single evolution while preserving
/// per-seed reproducibility.
pub fn sample_projected(
    circuit: &ConcreteCircuit,
    sv: &Statevector,
    shots: usize,
    seed: u64,
) -> HashMap<usize, u64> {
    let mut rng = SplitMix64::new(seed);
    let raw = sv.sample(shots, &mut rng);

    // Collect the qubit → classical-bit mapping declared by the circuit.
    let mut measured: Vec<(usize, usize)> = Vec::new();
    let mut measure_all = false;
    for gate in &circuit.gates {
        match gate {
            GateInstruction::Measure { qubit, cbit } => measured.push((*qubit, *cbit)),
            GateInstruction::MeasureAll => measure_all = true,
            _ => {}
        }
    }

    // No measurements: report the full basis state directly.
    if !measure_all && measured.is_empty() {
        return raw;
    }
    if measure_all {
        for q in 0..sv.num_qubits() {
            measured.push((q, q));
        }
    }

    // Project each sampled basis state onto the classical register.
    let mut counts = HashMap::new();
    for (state, c) in raw {
        let mut key = 0usize;
        for &(qubit, cbit) in &measured {
            if (state >> qubit) & 1 == 1 {
                key |= 1usize << cbit;
            }
        }
        *counts.entry(key).or_insert(0) += c;
    }
    counts
}

impl Simulator for StatevectorSimulator {
    fn run(&self, circuit: &ConcreteCircuit) -> Result<Statevector, SimError> {
        // One gate loop serves both entry points: an uncancellable run *is* a
        // cancellable one with no hook, and pays one `Option` test per gate for
        // it (see `CancellationCheck::cancelled`).
        self.run_cancellable(circuit, None)
    }

    fn run_cancellable(
        &self,
        circuit: &ConcreteCircuit,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<Statevector, SimError> {
        if circuit.num_qubits > self.max_qubits {
            return Err(SimError::TooManyQubits {
                requested: circuit.num_qubits,
                max: self.max_qubits,
            });
        }
        // Contract C-4: reject a gate acting on an already-measured qubit
        // (defense in depth for hand-assembled circuits). `apply` treats
        // measurements as no-ops, so without this the violation would be
        // silently simulated as if the measurement were terminal.
        if let Some(qubit) = polypus_circuit::terminal_measurement_violation(&circuit.gates) {
            return Err(SimError::GateAfterMeasure { qubit });
        }
        // Orchestration-level diagnostic, emitted once per run *outside* the
        // per-gate loop (never inside the hot kernels): which kernel path this
        // circuit takes. Mirrors `Statevector::use_parallel`; side-effect only,
        // so it cannot influence the (bit-identical) parallel/sequential result.
        // `log::debug!` gates itself on the level, so no manual guard is needed.
        log::debug!(
            "simulating {}-qubit circuit ({} gate(s)) on the {} kernel path",
            circuit.num_qubits,
            circuit.gates.len(),
            if cfg!(feature = "parallel") && circuit.num_qubits >= self.parallel_threshold {
                "parallel"
            } else {
                "sequential"
            }
        );
        let mut sv = Statevector::new(circuit.num_qubits)?;
        sv.set_parallel_threshold(self.parallel_threshold);
        // `Statevector::new`'s `2^n` allocation above is deliberately *not* a
        // cancellation point: near the qubit ceiling it dominates the run, but
        // it is one `vec![]` with nothing to interleave a check into. What the
        // hook covers is the gate sequence — whose cost (gates x `2^n`) is
        // unbounded, where the allocation's is capped by `max_qubits`.
        let mut cancellation = CancellationCheck::new(should_cancel, circuit.num_qubits);
        for (applied, gate) in circuit.gates.iter().enumerate() {
            if cancellation.cancelled() {
                log::debug!(
                    "simulation cancelled by the caller after {applied} of {} gate(s)",
                    circuit.gates.len()
                );
                return Err(SimError::Cancelled);
            }
            sv.apply(gate)?;
        }
        Ok(sv)
    }
}
