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

/// Most consecutive diagonal instructions that
/// [`run_cancellable`](StatevectorSimulator::run_cancellable) folds into one
/// fused pass over the amplitude buffer.
///
/// A run of diagonal gates commutes into a single combined diagonal, so an
/// unbounded run could be applied in one atomic pass — but that would make two
/// things scale with circuit depth that must not: cancellation latency (one
/// uninterruptible pass) and, once a run is long enough that its per-index
/// factor product no longer fits the memory-bandwidth-bound regime, the pass
/// itself turns compute-bound and the fusion stops paying off. Capping the run
/// bounds both: cancellation is polled at least once per capped pass, and each
/// pass stays bandwidth-bound (its cost is dominated by the `2^n` buffer
/// traversal, not the ≤`MAX_FUSED_DIAGONAL_RUN` factors per amplitude).
///
/// 64 sits comfortably above the longest diagonal run any real circuit produces
/// — a QFT phase column is ≤ `n − 1` gates, ≤ 29 at [`MAX_QUBITS`](crate::MAX_QUBITS)
/// = 30 — so QFT never hits the cap, while a genuinely deep run (a QAOA cost
/// layer, a Trotterized `ZZ`/`RZ` evolution) is chunked into bandwidth-bound
/// passes rather than one growing pass. Unlike the parallel threshold, this is a
/// fixed architectural safety bound, not a hardware-tuned value: it needs no
/// calibration, only to be larger than any real run and small enough to stay
/// bandwidth-bound.
pub(crate) const MAX_FUSED_DIAGONAL_RUN: usize = 64;

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
/// [`MAX_QUBITS`](crate::MAX_QUBITS), the crate's parallel threshold, and
/// gate fusion enabled.
#[derive(Debug, Clone)]
pub struct StatevectorSimulator {
    /// Reject circuits needing more than this many qubits.
    pub max_qubits: usize,
    /// Qubit count at or above which gates use the parallel kernels (only with
    /// the `parallel` feature).
    pub parallel_threshold: usize,
    /// Whether [`run_cancellable`](Simulator::run_cancellable) may fuse gates
    /// (diagonal-run fusion and dense connected-qubit-component fusion, see
    /// its doc comment) before applying them.
    ///
    /// Fusion only changes how many buffer passes a given gate sequence takes
    /// — never the result, up to floating-point rounding (composing several
    /// gates into one matrix accumulates rounding error in a different order
    /// than applying them one at a time, though both converge to the same
    /// value). Defaults to `true`. Set to `false` for a strictly gate-by-gate
    /// simulation that applies the circuit exactly as written — e.g. to
    /// reason about or benchmark a circuit's own per-gate cost, unaffected by
    /// the fusion heuristics.
    pub fusion: bool,
}

impl Default for StatevectorSimulator {
    fn default() -> Self {
        StatevectorSimulator {
            max_qubits: crate::MAX_QUBITS,
            parallel_threshold: crate::DEFAULT_PARALLEL_THRESHOLD,
            fusion: true,
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
        let gates = &circuit.gates;
        let total = gates.len();

        if !self.fusion {
            // Pure, unfused simulation: apply every gate exactly as it appears
            // in the circuit, one buffer pass each, through the same
            // `Statevector::apply` the fused path itself falls back to for a
            // gate that never merges. Semantically identical to the fused
            // path below (fusion only changes how many passes the same gate
            // sequence takes, never — up to rounding — the result), but a
            // caller that wants to reason about or benchmark the circuit
            // exactly as written needs a way to opt out of the rewrite
            // entirely, not just trust that it's transparent.
            for (i, gate) in gates.iter().enumerate() {
                poll_cancel(&mut cancellation, i, total)?;
                sv.apply(gate)?;
            }
            return Ok(sv);
        }

        // Two fusion mechanisms share this one scan, because both need the whole
        // instruction list in hand (`apply` has no lookahead):
        //
        //  * diagonal runs (#131) — maximal runs of consecutive diagonal gates
        //    folded into one buffer pass (see `Statevector::apply_diagonal_ops`);
        //  * dense connected-qubit components (#132) — the nine dense gates
        //    (`H, X, Y, Rx, Ry, U`, `Cx, Swap, Rxx`) grouped, while they touch at
        //    most two qubits between them, into one composed matrix per component.
        //
        // The two never overlap: a gate is dense-fusable, diagonal, or a boundary
        // (`Barrier`/`Measure`/`MeasureAll`). A diagonal gate or a boundary is a
        // hard boundary for the open dense components — their gates all occurred
        // earlier, so they must reach the state first — so it flushes them before
        // taking its own path.
        let mut open: Vec<DenseComponent> = Vec::new();
        let mut i = 0;
        while i < total {
            // A dense-fusable gate joins the connected-component machinery.
            if let Some(qubits) = crate::statevector::dense_fusable_qubits(&gates[i]) {
                // Point 1 of the cancellation cadence: once per scanned gate,
                // before deciding merge-vs-flush (bookkeeping only, no buffer
                // touch). See `flush_dense` for point 2.
                poll_cancel(&mut cancellation, i, total)?;
                absorb_dense_gate(
                    &mut sv,
                    gates,
                    &mut open,
                    i,
                    qubits,
                    &mut cancellation,
                    total,
                )?;
                i += 1;
                continue;
            }

            // Any non-dense gate is a boundary: flush every open dense component
            // (polling once per component — point 2) before it is applied.
            flush_dense(&mut sv, gates, &mut open, &mut cancellation, total)?;

            // Non-diagonal gate, or a diagonal gate not followed by another (a
            // run of length 1): today's single-gate path. The fast path avoids a
            // `Vec` allocation and the per-index run branch for the common case of
            // an isolated diagonal gate in an otherwise non-diagonal circuit.
            let starts_a_run = MAX_FUSED_DIAGONAL_RUN >= 2
                && crate::statevector::is_diagonal(&gates[i])
                && i + 1 < total
                && crate::statevector::is_diagonal(&gates[i + 1]);
            if !starts_a_run {
                poll_cancel(&mut cancellation, i, total)?;
                sv.apply(&gates[i])?;
                i += 1;
                continue;
            }

            // A run of two or more diagonal gates. Scan forward up to the cap,
            // polling cancellation once per scanned instruction (cheap — reading
            // instruction tags, not touching the amplitude buffer) and building
            // each descriptor, which validates the run's `Rz`/`Rzz`/`Cp` angles
            // *before* any amplitude is modified. If cancellation fires mid-scan
            // or an angle is invalid, nothing from this run is applied.
            //
            // Grows organically (`Vec::new`) rather than pre-allocating
            // `MAX_FUSED_DIAGONAL_RUN` up front: most runs are far shorter than
            // the cap (a real diagonal run is typically only a handful of gates),
            // so reserving the full 64-entry worst case every time wasted memory
            // on every run for no benefit — the doubling growth this falls back
            // to costs at most a few KB of copying even for a run that does reach
            // the cap, utterly negligible next to the `2^n` pass that follows.
            let mut ops = Vec::new();
            let mut j = i;
            while j < total && j - i < MAX_FUSED_DIAGONAL_RUN {
                // `is_diagonal` is `diagonal_op(..).is_some()`, so a `None` here
                // can only mean a non-diagonal terminator: the run ends.
                let Some(descriptor) = crate::statevector::diagonal_op(&gates[j]) else {
                    break;
                };
                poll_cancel(&mut cancellation, j, total)?;
                ops.push(descriptor?);
                j += 1;
            }
            sv.apply_diagonal_ops(&ops);
            i = j;
        }
        // End of the circuit: flush whatever dense components are still open.
        flush_dense(&mut sv, gates, &mut open, &mut cancellation, total)?;
        Ok(sv)
    }
}

/// One poll of the cancellation hook, plus the diagnostic the old
/// one-gate-per-iteration loop logged on a stop. Called once per *scanned*
/// instruction, the same cadence as before the fusion loops existed.
fn poll_cancel(
    cancellation: &mut CancellationCheck,
    at: usize,
    total: usize,
) -> Result<(), SimError> {
    if cancellation.cancelled() {
        log::debug!("simulation cancelled by the caller after {at} of {total} gate(s)");
        Err(SimError::Cancelled)
    } else {
        Ok(())
    }
}

/// An open dense component during the scan: a set of at most two qubits, and the
/// circuit indices of the dense gates fused onto it, in ascending (circuit)
/// order.
///
/// Invariant, maintained by [`absorb_dense_gate`]: open components are pairwise
/// qubit-disjoint (any gate that would bridge two of them merges them, or — if
/// that would exceed two qubits — forces a flush), so open components commute and
/// may be flushed in any order.
struct DenseComponent {
    /// The component's qubits, sorted ascending; length 1 or 2.
    qubits: Vec<usize>,
    /// Circuit indices of the fused gates. A component absorbed via
    /// [`absorb_dense_gate`]'s fast path (the common case: this gate shares a
    /// qubit with exactly one open component) grows this in place, one
    /// `push` at a time, so it stays sorted for free — indices only arrive
    /// out of order when two independent components are merged, which
    /// [`apply_to`](Self::apply_to) sorts out once, at flush.
    indices: Vec<usize>,
    /// The earliest gate index in the component, for a deterministic flush order.
    /// Maintained incrementally so the flush needs no scan of `indices`.
    min_index: usize,
}

impl DenseComponent {
    /// A fresh single-gate component for the gate at index `i`.
    fn singleton(qubits: Vec<usize>, i: usize) -> Self {
        DenseComponent {
            qubits,
            indices: vec![i],
            min_index: i,
        }
    }

    /// Apply this component to `sv`. A lone gate takes the unmodified
    /// [`Statevector::apply`] fast path (a component that never merged pays no
    /// composition cost); two or more gates are composed into one matrix, in
    /// circuit order.
    ///
    /// Takes `self` by value — its only caller already owns it (drained from
    /// `open`) — so restoring circuit order sorts `indices` in place instead
    /// of cloning it first, and the sorted indices are handed to
    /// [`Statevector::apply_composed_1q`]/[`apply_composed_2q`] directly
    /// alongside `gates`, with no intermediate `Vec<&GateInstruction>` built
    /// just to carry them.
    fn apply_to(mut self, sv: &mut Statevector, gates: &[GateInstruction]) -> Result<(), SimError> {
        if self.indices.len() == 1 {
            return sv.apply(&gates[self.indices[0]]);
        }
        // Restore circuit order once, here — merging two components (rather
        // than growing one in place) leaves indices out of order.
        self.indices.sort_unstable();
        match self.qubits.as_slice() {
            [q] => sv.apply_composed_1q(*q, &self.indices, gates),
            [qa, qb] => sv.apply_composed_2q(*qa, *qb, &self.indices, gates),
            // The merge cap keeps `qubits` at length 1 or 2; nothing else is built.
            _ => unreachable!("a dense component spans one or two qubits"),
        }
    }
}

/// Fold the dense-fusable gate at index `i` (touching `qubits`) into the open
/// components: merge it with every open component it shares a qubit with when the
/// union still fits two qubits, otherwise flush all open components and start a
/// fresh one holding just this gate.
/// This gate's own qubits as an owned, sorted list — the one allocation a
/// [`DenseComponent`] genuinely needs (as its `qubits` field, or as a fresh
/// singleton's). Kept out of the membership checks in
/// [`absorb_dense_gate`], which use `qubits` directly instead of testing
/// against a `Vec` built just to ask "is q one of this gate's qubits?".
fn dense_qubits_vec(qubits: crate::statevector::DenseQubits) -> Vec<usize> {
    use crate::statevector::DenseQubits;
    match qubits {
        DenseQubits::One(q) => vec![q],
        DenseQubits::Two(a, b) => {
            let (lo, hi) = (a.min(b), a.max(b));
            vec![lo, hi]
        }
    }
}

fn absorb_dense_gate(
    sv: &mut Statevector,
    gates: &[GateInstruction],
    open: &mut Vec<DenseComponent>,
    i: usize,
    qubits: crate::statevector::DenseQubits,
    cancellation: &mut CancellationCheck,
    total: usize,
) -> Result<(), SimError> {
    use crate::statevector::DenseQubits;

    // Whether `q` is one of this gate's own qubits — a plain comparison
    // against `qubits` (`Copy`), not a `Vec` allocated solely to ask this.
    let touches_gate = |q: usize| match qubits {
        DenseQubits::One(a) => q == a,
        DenseQubits::Two(a, b) => q == a || q == b,
    };

    // The merged qubit set = this gate's qubits ∪ every touched component's,
    // and how many (and which) open components are touched -- both needed
    // to decide the cap and, below, to grow a lone touched component in
    // place instead of rebuilding one. The one allocation this function
    // needs in the common case (an isolated gate with nothing open to merge
    // into): `merged` becomes the eventual `DenseComponent`'s qubit list
    // directly below, no separate "this gate's own qubits" `Vec` cloned
    // into it first.
    let mut merged: Vec<usize> = dense_qubits_vec(qubits);
    let mut touched_one = None;
    let mut touched_count = 0usize;
    for (idx, comp) in open.iter().enumerate() {
        if comp.qubits.iter().any(|&q| touches_gate(q)) {
            touched_count += 1;
            touched_one = Some(idx);
            for &q in &comp.qubits {
                if !merged.contains(&q) {
                    merged.push(q);
                }
            }
        }
    }
    merged.sort_unstable(); // qubits stays sorted at every use below

    if merged.len() > 2 {
        // Bridging would exceed the two-qubit cap. Every open component's gates
        // occurred before this one, so flush them all (in circuit order), then
        // open a new component for this gate alone.
        flush_dense(sv, gates, open, cancellation, total)?;
        open.push(DenseComponent::singleton(dense_qubits_vec(qubits), i));
        return Ok(());
    }

    // Fast path: exactly one open component shares a qubit with this gate --
    // the common case for a chain of gates repeatedly touching the same
    // qubit(s), which is exactly what makes fusion worth it in the first
    // place. Grow that component's own index list in place (`Vec::push`'s
    // own amortized growth) instead of rebuilding a fresh list and copying
    // the old one into it below: without this fast path, absorbing the
    // g-th gate into an already-`m`-gate component costs O(m) (the copy),
    // so a chain of g gates costs O(g^2) in total, not O(g) -- confirmed
    // with a direct allocator measurement before this fix (a 32-round,
    // 96-gate same-pair chain: 736 B for round 1, 82,808 B by round 32 --
    // growing per-gate cost, not the flat cost this fast path restores).
    if touched_count == 1 {
        let idx = touched_one.expect("touched_count == 1");
        open[idx].qubits = merged;
        open[idx].indices.push(i);
        return Ok(());
    }

    // General case: 0 components touched (a fresh component-to-be) or 2+ (a
    // genuine merge of previously-independent components) -- both are
    // one-off relative to any single component's history, so rebuilding
    // here doesn't reintroduce the accumulation the fast path above avoids.
    let mut merged_indices: Vec<usize> = Vec::new();
    let mut min_index = i;
    let mut kept: Vec<DenseComponent> = Vec::new();
    for comp in open.drain(..) {
        if comp.qubits.iter().any(|&q| touches_gate(q)) {
            min_index = min_index.min(comp.min_index);
            merged_indices.extend(comp.indices);
        } else {
            kept.push(comp);
        }
    }
    merged_indices.push(i);
    *open = kept;
    open.push(DenseComponent {
        qubits: merged,
        indices: merged_indices,
        min_index,
    });
    Ok(())
}

/// Flush every open dense component to `sv`, in circuit order, polling
/// cancellation **once per component before its apply** — point 2 of the
/// cadence, the part a naive port of #131 (which only ever flushed one run) would
/// miss: a single flush here can hold several independent components.
///
/// A no-op when nothing is open — and, crucially, it polls nothing then, so a
/// purely diagonal circuit keeps #131's exact poll cadence.
fn flush_dense(
    sv: &mut Statevector,
    gates: &[GateInstruction],
    open: &mut Vec<DenseComponent>,
    cancellation: &mut CancellationCheck,
    total: usize,
) -> Result<(), SimError> {
    if open.is_empty() {
        return Ok(());
    }
    // Disjoint components commute, so any order is numerically identical; earliest
    // gate first is the natural, deterministic choice.
    open.sort_by_key(|comp| comp.min_index);
    for comp in std::mem::take(open) {
        poll_cancel(cancellation, comp.min_index, total)?;
        comp.apply_to(sv, gates)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{C64, MAX_QUBITS};
    use polypus_circuit::{GateInstruction as G, GateParam::Fixed};
    use std::f64::consts::FRAC_PI_4;

    /// A simulator pinned to the sequential kernels, so these tests take the same
    /// path — and behave the same — with or without the `parallel` feature and
    /// regardless of the runner's core count. Mirrors `tests/cancellation.rs`.
    fn sequential_sim() -> StatevectorSimulator {
        StatevectorSimulator {
            max_qubits: MAX_QUBITS,
            parallel_threshold: MAX_QUBITS + 1,
            fusion: true,
        }
    }

    /// Same as [`sequential_sim`], with fusion disabled — the reference every
    /// "fused matches gate-by-gate" test below now runs through the real
    /// `fusion: false` code path instead of a hand-rolled per-gate loop, so
    /// those tests double as regression coverage for the flag itself.
    fn sequential_sim_unfused() -> StatevectorSimulator {
        StatevectorSimulator {
            max_qubits: MAX_QUBITS,
            parallel_threshold: MAX_QUBITS + 1,
            fusion: false,
        }
    }

    /// The per-index phase one diagonal gate contributes to basis state `b`,
    /// computed straight from the gate's definition — an independent reference
    /// for the fused kernel, sharing none of its code.
    fn diagonal_phase(gate: &G, b: usize) -> C64 {
        let set = |q: usize| (b >> q) & 1 == 1;
        let one = C64::new(1.0, 0.0);
        match gate {
            G::Z(q) => {
                if set(*q) {
                    C64::new(-1.0, 0.0)
                } else {
                    one
                }
            }
            G::S(q) => {
                if set(*q) {
                    C64::new(0.0, 1.0)
                } else {
                    one
                }
            }
            G::T(q) => {
                if set(*q) {
                    C64::from_polar(1.0, FRAC_PI_4)
                } else {
                    one
                }
            }
            G::Rz { qubit, theta } => {
                let t = fixed(theta);
                C64::from_polar(1.0, if set(*qubit) { t / 2.0 } else { -t / 2.0 })
            }
            G::Cp { q0, q1, theta } => {
                if set(*q0) && set(*q1) {
                    C64::from_polar(1.0, fixed(theta))
                } else {
                    one
                }
            }
            G::Rzz { q0, q1, theta } => {
                let t = fixed(theta);
                C64::from_polar(
                    1.0,
                    if set(*q0) == set(*q1) {
                        -t / 2.0
                    } else {
                        t / 2.0
                    },
                )
            }
            other => panic!("{other:?} is not a diagonal gate the reference handles"),
        }
    }

    fn fixed(p: &polypus_circuit::GateParam) -> f64 {
        match p {
            Fixed(v) => *v,
            other => panic!("expected a fixed angle, got {other:?}"),
        }
    }

    /// A circuit whose diagonal run is long enough to span several fused passes
    /// (200 > `MAX_FUSED_DIAGONAL_RUN`) must land on the analytically known
    /// amplitudes, not merely "not panic". Starting from a uniform superposition,
    /// each amplitude is `1/sqrt(D)` times the product of the run's per-index
    /// phases.
    #[test]
    fn a_long_diagonal_run_matches_analytic_amplitudes() {
        let n = 3;
        let dim = 1usize << n;

        let mut run: Vec<G> = Vec::new();
        for k in 0..200 {
            let q = k % n;
            run.push(match k % 5 {
                0 => G::Rz {
                    qubit: q,
                    theta: Fixed(0.1 * k as f64),
                },
                1 => G::Cp {
                    q0: q,
                    q1: (q + 1) % n,
                    theta: Fixed(0.05 * k as f64),
                },
                2 => G::Rzz {
                    q0: q,
                    q1: (q + 2) % n,
                    theta: Fixed(-0.07 * k as f64),
                },
                3 => G::T(q),
                _ => G::Z(q),
            });
        }
        assert!(run.len() > MAX_FUSED_DIAGONAL_RUN);

        let mut gates: Vec<G> = (0..n).map(G::H).collect();
        gates.extend(run.iter().cloned());
        let circuit = ConcreteCircuit {
            num_qubits: n,
            gates,
        };

        let sv = sequential_sim()
            .run(&circuit)
            .expect("all angles are finite");

        let inv_sqrt_d = 1.0 / (dim as f64).sqrt();
        for b in 0..dim {
            let mut expected = C64::new(inv_sqrt_d, 0.0);
            for gate in &run {
                expected *= diagonal_phase(gate, b);
            }
            let got = sv.amplitudes()[b];
            assert!(
                (got - expected).norm() < 1e-12,
                "amplitude {b}: got {got}, expected {expected}"
            );
        }
    }

    /// Fusion must be transparent: a diagonal-heavy circuit (diagonal runs longer
    /// than the cap, broken up by non-diagonal gates) evolved through the fusing
    /// loop must be bit-for-bit close to applying every gate singly through the
    /// per-gate kernels.
    #[test]
    fn fused_run_matches_gate_by_gate_application() {
        let n = 4;
        let mut gates: Vec<G> = (0..n).map(G::H).collect();
        for block in 0..3 {
            for k in 0..80 {
                let q = k % n;
                gates.push(match k % 4 {
                    0 => G::Rz {
                        qubit: q,
                        theta: Fixed(0.13 * (k + block) as f64),
                    },
                    1 => G::Cp {
                        q0: q,
                        q1: (q + 1) % n,
                        theta: Fixed(0.09 * (k + 1) as f64),
                    },
                    2 => G::Cz(q, (q + 2) % n),
                    _ => G::S(q),
                });
            }
            // A non-diagonal gate terminates the run and starts the next.
            gates.push(G::Cx(block % n, (block + 1) % n));
            gates.push(G::Ry {
                qubit: block % n,
                theta: Fixed(0.5),
            });
        }
        let circuit = ConcreteCircuit {
            num_qubits: n,
            gates: gates.clone(),
        };

        let fused = sequential_sim().run(&circuit).expect("all angles finite");

        // Reference: the same circuit run with fusion disabled, which applies
        // every gate one at a time through `Statevector::apply`.
        let reference = sequential_sim_unfused()
            .run(&circuit)
            .expect("all angles finite");

        for (b, (a, c)) in fused
            .amplitudes()
            .iter()
            .zip(reference.amplitudes().iter())
            .enumerate()
        {
            assert!(
                (a - c).norm() < 1e-12,
                "amplitude {b}: fused {a} vs gate-by-gate {c}"
            );
        }
    }

    /// Qubit width for the cancellation regression: wide enough that a single
    /// fused pass is expensive (a checkpoint's worth of wall clock accrues in a
    /// few dozen passes), so the run reaches several checkpoints well before its
    /// end even on a fast machine.
    const CANCEL_N: usize = 16;

    /// A single diagonal run far longer than `MAX_FUSED_DIAGONAL_RUN`, so the
    /// fusing loop chunks it into many capped passes; long enough (many hundred
    /// passes) that several cancellation checkpoints come due while it runs.
    const CANCEL_RUN: usize = 60_000;

    /// Regression test for the cancellation-cadence constraint: the fusing loop
    /// must poll cancellation *per scanned instruction*, not once per fused run.
    ///
    /// The run is one unbroken diagonal stretch far past the cap, terminated by a
    /// `NaN` `Rz`. Timing-independent by the same trick as `tests/cancellation.rs`:
    /// the error *variant* alone says whether the loop reached the end —
    /// `Cancelled` proves it stopped part-way, and only a per-instruction poll can
    /// make it stop part-way through a run. A naive "fuse the whole run, check
    /// once" loop would build the whole run's descriptors first, hit the `NaN`, and
    /// return `NonFiniteAmplitude` instead — so this test fails against it and
    /// passes only against the interleaved scan.
    #[test]
    fn cancellation_fires_inside_a_run_longer_than_the_cap() {
        let mut gates: Vec<G> = (0..CANCEL_RUN)
            .map(|k| G::Rz {
                qubit: k % CANCEL_N,
                theta: Fixed(0.1),
            })
            .collect();
        // Poisoned terminator: reached only if the loop is never cancelled.
        gates.push(G::Rz {
            qubit: 0,
            theta: Fixed(f64::NAN),
        });
        assert!(gates.len() > MAX_FUSED_DIAGONAL_RUN);
        let circuit = ConcreteCircuit {
            num_qubits: CANCEL_N,
            gates,
        };

        let mut calls = 0usize;
        let err = {
            let mut cancel = || {
                calls += 1;
                calls >= 3
            };
            sequential_sim()
                .run_cancellable(&circuit, Some(&mut cancel))
                .unwrap_err()
        };
        // `Cancelled`, not `NonFiniteAmplitude`: the loop was interrupted inside
        // the run, before reaching the poisoned final gate.
        assert_eq!(err, SimError::Cancelled);
        // And it stopped at the first "yes" — proof the hook is polled while a run
        // is being scanned, not merely once for the whole run.
        assert_eq!(calls, 3);
    }

    /// Evolve `gates` with fusion disabled (gate-by-gate through
    /// `Statevector::apply`, never fusing), pinned to the sequential kernels —
    /// the reference the fused run must match.
    fn reference_state(n: usize, gates: &[G]) -> Statevector {
        let circuit = ConcreteCircuit {
            num_qubits: n,
            gates: gates.to_vec(),
        };
        sequential_sim_unfused()
            .run(&circuit)
            .expect("all angles finite")
    }

    /// The fused run (dense connected-component fusion) must land bit-for-bit
    /// close to applying every gate one at a time through `Statevector::apply`.
    fn assert_fused_matches_reference(n: usize, gates: &[G]) {
        let circuit = ConcreteCircuit {
            num_qubits: n,
            gates: gates.to_vec(),
        };
        let fused = sequential_sim().run(&circuit).expect("all angles finite");
        let reference = reference_state(n, gates);
        for (b, (a, c)) in fused
            .amplitudes()
            .iter()
            .zip(reference.amplitudes().iter())
            .enumerate()
        {
            assert!(
                (a - c).norm() < 1e-12,
                "amplitude {b}: fused {a} vs gate-by-gate {c}"
            );
        }
    }

    /// A chain of dense one-qubit gates on the *same* qubit fuses into one
    /// composed 2×2. Composition is non-commutative, so this also pins the order:
    /// `Rx·Ry·Rx` with distinct angles is not any reordering of itself.
    #[test]
    fn composed_same_qubit_chain_matches_gate_by_gate() {
        let gates = [
            G::Rx {
                qubit: 0,
                theta: Fixed(0.3),
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(-1.1),
            },
            G::Rx {
                qubit: 0,
                theta: Fixed(0.7),
            },
        ];
        assert_fused_matches_reference(1, &gates);
    }

    /// A chain of dense two-qubit gates on the *same* pair fuses into one composed
    /// 4×4 — including the asymmetric `Cx` (whose 4×4 is built directly) and the
    /// symmetric `Swap`/`Rxx`.
    #[test]
    fn composed_same_pair_chain_matches_gate_by_gate() {
        let gates = [
            G::Cx(0, 1),
            G::Rxx {
                q0: 0,
                q1: 1,
                theta: Fixed(0.7),
            },
            G::Swap(0, 1),
            G::Cx(1, 0),
        ];
        assert_fused_matches_reference(2, &gates);
    }

    /// The shape the fusion actually pays off on: a `Cx` sharing a qubit with the
    /// rotations before and after it, so a whole rotation–entangle–rotation block
    /// collapses to one 4×4 pass. Mixes one-qubit gates (lifted into the joint
    /// space) with the two-qubit `Cx` inside a single component.
    #[test]
    fn rotation_cx_rotation_chain_matches_gate_by_gate() {
        let gates = [
            G::Rx {
                qubit: 0,
                theta: Fixed(0.5),
            },
            G::Ry {
                qubit: 1,
                theta: Fixed(-0.3),
            },
            G::Cx(0, 1),
            G::Rx {
                qubit: 1,
                theta: Fixed(0.8),
            },
            G::Ry {
                qubit: 0,
                theta: Fixed(0.2),
            },
        ];
        assert_fused_matches_reference(2, &gates);
    }

    /// Independent dense one-qubit gates on *different* qubits never share a
    /// qubit, so each stays its own length-1 component and flushes through the
    /// unmodified `Statevector::apply` fast path (`DenseComponent::apply_to`'s
    /// `indices.len() == 1` arm) — never the composed path, which a component that
    /// never merged must not pay for. The result is therefore exactly
    /// `apply`-gate-by-gate; equivalence here confirms the singleton path stays
    /// correct and is exercised.
    #[test]
    fn independent_single_qubit_gates_take_the_singleton_path() {
        let gates = [
            G::H(0),
            G::X(1),
            G::Y(2),
            G::Ry {
                qubit: 3,
                theta: Fixed(0.4),
            },
            G::U {
                qubit: 4,
                theta: Fixed(0.5),
                phi: Fixed(1.2),
                lam: Fixed(-0.4),
            },
        ];
        assert_fused_matches_reference(5, &gates);
    }

    /// Dense components, diagonal runs and a hard boundary (`Barrier`) interleaved:
    /// each diagonal gate and the barrier flush the open dense components before
    /// taking their own path, so this exercises the dense/diagonal boundary logic
    /// end to end. Must still match plain gate-by-gate application.
    #[test]
    fn mixed_dense_diagonal_and_boundary_matches_gate_by_gate() {
        let gates = [
            G::H(0),
            G::Rx {
                qubit: 1,
                theta: Fixed(0.3),
            },
            G::Cx(0, 1),
            // Diagonal run: flushes the {0,1} dense component first.
            G::Rz {
                qubit: 0,
                theta: Fixed(0.9),
            },
            G::Cz(0, 2),
            G::Rzz {
                q0: 1,
                q1: 2,
                theta: Fixed(-0.4),
            },
            // Back to dense on a fresh set of qubits.
            G::Ry {
                qubit: 2,
                theta: Fixed(0.6),
            },
            G::Rxx {
                q0: 2,
                q1: 3,
                theta: Fixed(1.1),
            },
            // A hard boundary mid-stream: flushes the {2,3} dense component.
            G::Barrier(vec![]),
            G::Ry {
                qubit: 3,
                theta: Fixed(-0.2),
            },
            G::Cx(3, 0),
        ];
        assert_fused_matches_reference(4, &gates);
    }

    /// Qubit width for the dense-flush cancellation regression: wide enough that a
    /// single component's one buffer pass (tens of ms sequentially, measured well
    /// above 25 ms here) comfortably exceeds the checkpoint interval, so the hook
    /// is polled once per component as a multi-component flush drains.
    const DENSE_CANCEL_N: usize = 21;

    /// Regression for cadence point 2: a flush holding several independent dense
    /// components must poll cancellation *before each component*, not once for the
    /// whole flush.
    ///
    /// The circuit is a layer of single-qubit gates on distinct qubits with no
    /// entangling gate between them — none merge, none force an early flush, so
    /// they all stay open until one flush at the end — with a poisoned `NaN` `Rx`
    /// on its own qubit as the last component. Timing-independent in its
    /// *assertion*, by the same trick as `cancellation_fires_inside_a_run...`
    /// above: the error variant alone says where the run stopped. A hook that
    /// answers "stop" on its 2nd call must cancel *inside* the flush, having
    /// applied some but not all components, so the run never reaches the poison:
    ///
    ///  * `Cancelled` (not `NonFiniteAmplitude`) proves the flush stopped
    ///    part-way — which only a per-component poll can do. A "poll once before
    ///    the flush" port would apply every component (its lone poll answers
    ///    "continue" on call 1) and hit the poison; a "poll once after" port
    ///    likewise never stops mid-flush.
    ///  * `calls == 2` proves it stopped at the first "stop" answer — after
    ///    applying exactly the components before it: neither 0 (checked only
    ///    before the flush) nor all (checked only after).
    #[test]
    fn cancellation_fires_between_components_of_one_flush() {
        let real = 5;
        let mut gates: Vec<G> = (0..real)
            .map(|q| G::Ry {
                qubit: q,
                theta: Fixed(0.3),
            })
            .collect();
        // Poisoned terminator on its own qubit: applied only if the flush never
        // stops. It opens its own (last) component and is evaluated only when
        // flushed.
        gates.push(G::Rx {
            qubit: real,
            theta: Fixed(f64::NAN),
        });
        let circuit = ConcreteCircuit {
            num_qubits: DENSE_CANCEL_N,
            gates,
        };

        let mut calls = 0usize;
        let err = {
            let mut cancel = || {
                calls += 1;
                calls >= 2
            };
            sequential_sim()
                .run_cancellable(&circuit, Some(&mut cancel))
                .unwrap_err()
        };
        // Stopped inside the flush, before the poisoned final component.
        assert_eq!(err, SimError::Cancelled);
        // Exactly at the first "stop": two components in, not zero and not all.
        assert_eq!(calls, 2);
    }

    #[test]
    fn fusion_defaults_to_enabled() {
        assert!(StatevectorSimulator::default().fusion);
        assert!(StatevectorSimulator::new().fusion);
    }

    /// The opt-out itself, asserted directly (not just implied by the reference
    /// helpers above being rebased onto it): a circuit exercising both fusion
    /// mechanisms (a diagonal run, a dense connected component, and a boundary)
    /// gives the same result whether `fusion` is `true` or `false`.
    #[test]
    fn fusion_false_matches_fusion_true_on_a_mixed_circuit() {
        let gates = [
            G::H(0),
            G::Rx {
                qubit: 1,
                theta: Fixed(0.3),
            },
            G::Cx(0, 1), // dense component on {0, 1}
            G::Rz {
                qubit: 0,
                theta: Fixed(0.9),
            },
            G::Cp {
                q0: 0,
                q1: 2,
                theta: Fixed(-0.4),
            }, // diagonal run, flushes the dense component first
            G::Barrier(vec![]),
            G::Ry {
                qubit: 2,
                theta: Fixed(0.6),
            },
        ];
        let circuit = ConcreteCircuit {
            num_qubits: 3,
            gates: gates.to_vec(),
        };
        let fused = sequential_sim().run(&circuit).expect("all angles finite");
        let unfused = sequential_sim_unfused()
            .run(&circuit)
            .expect("all angles finite");
        for (b, (a, c)) in fused
            .amplitudes()
            .iter()
            .zip(unfused.amplitudes().iter())
            .enumerate()
        {
            assert!(
                (a - c).norm() < 1e-12,
                "amplitude {b}: fusion=true {a} vs fusion=false {c}"
            );
        }
    }

    /// With fusion off, cancellation must still be polled once per gate — the
    /// same cadence the fused path polls once per *scanned* instruction. A
    /// long run of otherwise-fusable diagonal gates, poisoned by a trailing
    /// `NaN`, proves the loop stops mid-run rather than needing to reach (and
    /// fail on) the poisoned gate: only a per-gate poll can do that.
    #[test]
    fn fusion_false_polls_cancellation_every_gate() {
        // Same duration-tuned sizing as `cancellation_fires_inside_a_run_longer_
        // than_the_cap` above: with fusion off, each of these is its own full
        // buffer pass (there is no run/component to amortize it across), so the
        // same gate count that takes several checkpoints to scan when fused
        // comfortably outlives them here too.
        let mut gates: Vec<G> = (0..CANCEL_RUN)
            .map(|k| G::Rz {
                qubit: k % CANCEL_N,
                theta: Fixed(0.1),
            })
            .collect();
        gates.push(G::Rz {
            qubit: 0,
            theta: Fixed(f64::NAN),
        });
        let circuit = ConcreteCircuit {
            num_qubits: CANCEL_N,
            gates,
        };

        let mut calls = 0usize;
        let err = {
            let mut cancel = || {
                calls += 1;
                calls >= 3
            };
            sequential_sim_unfused()
                .run_cancellable(&circuit, Some(&mut cancel))
                .unwrap_err()
        };
        assert_eq!(err, SimError::Cancelled);
        assert_eq!(calls, 3);
    }
}
