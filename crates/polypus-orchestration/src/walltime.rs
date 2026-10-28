//! Early warning that a training run will outlive its backend's allocation.
//!
//! A backend whose execution units expire (CUNQA's QPUs are released when their
//! SLURM walltime runs out) reports that lifetime through
//! [`QuantumBackend::walltime`](polypus_infrastructure::QuantumBackend::walltime).
//! [`TrainFlow`](crate::TrainFlow) then wraps its oracle in a [`WalltimeWatch`],
//! which times the first generation, projects it over the optimizer's generation
//! budget (for CMA-ES with IPOP restarts, whose batches grow, over the bound
//! [`Method::growth`] gives; for an external optimizer, over its evaluation
//! budget in batches the size of its first, timed until the second starts so
//! that the library's own time counts; batches sent before the first
//! generation, such as a calibration, are not timed but added at its pace) and
//! logs one `warn` when the projection exceeds the walltime. Backends
//! without a walltime get the bare oracle, so they pay nothing. The final batch
//! of a run with `final_shots`, sent once the optimizer has stopped, is not in
//! the projection.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use std::sync::Arc;

use polypus_optimizers::{CmaEsBatchSchedule, EvaluationOracle, GradientOracle};

use crate::dispatch::{Method, Oracles};

/// Source of the current time, injectable so tests can drive a fake clock.
pub(crate) trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

/// The real monotonic clock.
pub(crate) struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// How an optimizer spends its oracle calls: one generation is
/// `batches_per_generation` consecutive `evaluate_batch` calls, and a run makes at
/// most `generations` of them, after `leading_batches` calls that belong to no
/// generation. With an evaluation budget the generations are counted once the
/// first is known (see [`generations_after`](Self::generations_after)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunShape {
    pub batches_per_generation: u32,
    pub generations: u64,
    /// The run's budget in circuits, when it is set that way rather than in
    /// generations: an external optimizer's `max_evaluations`, whose batch
    /// size is the library's. `generations` is then the bound with batches of
    /// one circuit, and the watch projects the budget over batches the size of
    /// the first. `None` for every other optimizer.
    pub evaluations: Option<u64>,
    /// Whether the first generation is timed until the next batch starts
    /// rather than until its own last batch ends: an external optimizer's,
    /// whose library computes between its batches (`tell()` of one, then
    /// `ask()` for the next), so that time is the run's too. The projection,
    /// and its warning, then come when the second batch starts, and a run of
    /// one batch is not projected. `false` for every other optimizer, whose
    /// time between batches is its own bookkeeping.
    pub until_next_batch: bool,
    /// Batches sent once before the first generation, unlike any generation's:
    /// a calibration. The watch does not time them as the first generation;
    /// it adds them to the projection by their circuits, at the first
    /// generation's pace. 0 for an optimizer without them.
    pub leading_batches: u32,
    /// What a generation is called in the warning: "generations",
    /// "iterations" or "steps".
    pub unit: &'static str,
    /// The setting the warning asks to lower, the one that bounds
    /// `generations`: the unit itself, except for Rotosolve, whose steps are
    /// bounded by its `sweeps`, and an external optimizer, whose batches are
    /// bounded by its `max_evaluations`.
    pub setting: &'static str,
}

impl RunShape {
    /// The generations of a run whose first generation held `first_circuits`
    /// circuits: `generations`, or with an evaluation budget the generations
    /// of that size it holds, `⌊evaluations / first_circuits⌋`. That is
    /// exactly the batches the bridge sends when every batch has the first's
    /// size, since it stops before the batch that would exceed the budget.
    /// Larger later batches make it fewer: if a batch `k` times larger takes
    /// at most `k` times as long (the assumption [`Method::growth`] makes),
    /// the run takes at most the first batch's time × `evaluations /
    /// first_circuits`, so the projection is an upper bound to within one
    /// first batch, what the floor drops. Smaller later batches make it more,
    /// each leaving QPUs idle, so the projection can fall short. Like the
    /// leading batches, the count is known only once the first generation is.
    pub(crate) fn generations_after(&self, first_circuits: u64) -> u64 {
        match self.evaluations {
            Some(budget) => (budget / first_circuits.max(1)).max(1),
            None => self.generations,
        }
    }
}

impl Method {
    /// The call pattern of this method's optimizer. DE and PSO score their initial
    /// population in one batch the size of a generation's, so it counts as one;
    /// a QNG iteration is a gradient batch plus a single-candidate batch; an
    /// Adam iteration is one batch, θ and its finite-difference stencil
    /// (with an exact gradient see [`run_shape_over`](Self::run_shape_over)),
    /// and its final θ, one candidate after the last step, counts as one more
    /// iteration; an SPSA iteration is one batch of its perturbations (and θ
    /// when it is measured), after one leading calibration batch of 50
    /// circuits, which the watch does not time as the first iteration, and its
    /// final θ, one candidate, counts as one more iteration too. SPSA's first
    /// iteration measures θ, so it holds `2K + 1` circuits where most hold
    /// `2K`, for `K` perturbations. A Rotosolve step is one batch of `x ± π/2`
    /// along one of the `dimensions` parameters, so a run is at most
    /// `sweeps × dimensions` steps; its first step, and every
    /// `reevaluate_every`-th, also measures `x`, 3 circuits where most hold 2,
    /// and the final point, one candidate after the last step, counts as one
    /// more step.
    /// For both, the first batch holds `n + 1` circuits where most hold `n`:
    /// on `Q` QPUs it takes `⌈(n + 1)/Q⌉` rounds where most take `⌈n/Q⌉`, so
    /// the projection stays an upper bound but can reach
    /// `⌈(n + 1)/Q⌉ / ⌈n/Q⌉` times the real time: `(n + 1)/n` on one QPU
    /// (1.5× with `n = 2`), 2× on `Q = n` QPUs (2 for Rotosolve or SPSA with
    /// one perturbation), and not at all on `n + 1` QPUs or more.
    ///
    /// Every DE strategy makes one batch per generation. L-SHADE spends the
    /// evaluation budget of `generations` full generations in more, smaller
    /// ones; its schedule does not depend on fitness, so
    /// [`DeStrategy::max_generations`](polypus_optimizers::DeStrategy::max_generations)
    /// gives their exact number, from the schedule the optimizer itself steps.
    /// None of its batches is larger than the first, so the projection stays an
    /// upper bound.
    ///
    /// An external optimizer's batches are its library's `ask()`s, whose size
    /// the bridge learns only from the first. Its shape holds the budget, and
    /// the watch projects `⌊max_evaluations / n⌋` batches after a first batch
    /// of `n` circuits (see [`RunShape::generations_after`]). That is exact
    /// for batches of constant size, as pycma, Nevergrad with `num_workers`
    /// and the Optuna adapter with its `batch_size` send; an upper bound to
    /// within one first batch when they grow; and it can be too low when they
    /// shrink. The first batch is timed until the second starts
    /// ([`RunShape::until_next_batch`]), so the library's `tell()` of the first
    /// batch and its `ask()` for the second are in the projection, at that
    /// pace for every batch; only the first `ask()`, before any batch, is not.
    pub(crate) fn run_shape(&self, dimensions: u32) -> RunShape {
        match self {
            Method::De(cfg) => RunShape {
                batches_per_generation: 1,
                generations: cfg
                    .strategy
                    .max_generations(cfg.population_size, cfg.generations)
                    + 1,
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "generations",
                setting: "generations",
            },
            Method::Pso(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.generations) + 1,
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "generations",
                setting: "generations",
            },
            Method::Qng(cfg, _) => RunShape {
                batches_per_generation: 2,
                generations: u64::from(cfg.max_iters),
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "iterations",
                setting: "iterations",
            },
            // One batch per generation and no initial batch; the generations of
            // the first run and of every restart share the budget.
            Method::Cmaes(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.generations),
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "generations",
                setting: "generations",
            },
            // One batch of 2·d + 1 per iteration: θ, whose value is the
            // iteration's fitness, and the finite-difference stencil around it.
            // After the last step the final θ is scored alone; a full iteration
            // bounds it.
            Method::Adam(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.max_iters) + 1,
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "iterations",
                setting: "iterations",
            },
            // One batch per iteration, 2·resamplings perturbed points and θ
            // when it is measured, after the calibration batch of 50.
            // After the last step the final θ is scored alone; a full
            // iteration bounds it.
            Method::Spsa(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.max_iters) + 1,
                leading_batches: 1,
                evaluations: None,
                until_next_batch: false,
                unit: "iterations",
                setting: "iterations",
            },
            // One batch per step, every parameter once per sweep. After the
            // last step the final point is scored alone; a full step bounds it.
            Method::Rotosolve(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.sweeps) * u64::from(dimensions) + 1,
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "steps",
                setting: "sweeps",
            },
            // One batch per `ask()`, of a size only the first batch tells:
            // `generations` is the bound with batches of one.
            Method::External(cfg, _) => RunShape {
                batches_per_generation: 1,
                generations: cfg.max_evaluations,
                leading_batches: 0,
                evaluations: Some(cfg.max_evaluations),
                until_next_batch: true,
                unit: "batches",
                setting: "max_evaluations",
            },
        }
    }

    /// [`run_shape`](Self::run_shape) for a run over `oracles`. An exact
    /// gradient is one more call per Adam iteration: Adam scores θ on the
    /// evaluation oracle, then asks the gradient oracle. So is a minibatch:
    /// Adam scores θ on every sample (`full_evaluation`), then differentiates
    /// the minibatch. The final θ is then one call counted as two. An exact
    /// gradient
    /// that scores θ itself ([`GradientOracle::value_and_gradient`]) makes an
    /// iteration one call, which cannot be told in advance, so the projection
    /// then overestimates (see [`Watched`]).
    pub(crate) fn run_shape_over(&self, oracles: &Oracles, dimensions: u32) -> RunShape {
        let mut shape = self.run_shape(dimensions);
        if matches!(self, Method::Adam(_))
            && (oracles.gradient.is_some() || oracles.full_evaluation.is_some())
        {
            shape.batches_per_generation += 1;
        }
        shape
    }

    /// How much longer than [`run_shape`](Self::run_shape)'s generations at the
    /// first generation's pace a run of `dimensions` parameters can take, when
    /// its later batches can be larger than its first; `None` when they cannot.
    ///
    /// Only CMA-ES with IPOP restarts grows: each restart doubles the
    /// population. When a restart fires depends on the search, so this takes
    /// the closed-form bound of [`CmaEsBatchSchedule`]: every restart as early
    /// as it can, one generation after the previous run started. That is an
    /// honest upper bound, if a loose one when restarts come late, as they
    /// usually do. Re-projecting at each restart instead would not be a bound
    /// at all (a later restart can still double the batches again), and the
    /// warning would come only once the larger batches had started rather
    /// than after the first generation, when raising `walltime` is still
    /// cheap. The bound also assumes a batch `k` times larger takes at most
    /// `k` times as long, which holds when a backend spreads a batch over its
    /// QPUs.
    pub(crate) fn growth(&self, dimensions: u32) -> Option<Growth> {
        let Method::Cmaes(cfg) = self else {
            return None;
        };
        if cfg.restarts == 0 {
            return None;
        }
        let schedule = CmaEsBatchSchedule::new(
            dimensions,
            cfg.population_size,
            cfg.generations,
            cfg.restarts,
        );
        let first = u64::from(schedule.initial_population()).max(1);
        Some(Growth {
            first_generations: schedule.max_evaluations().div_ceil(first),
            restarts: cfg.restarts,
        })
    }
}

/// A run whose IPOP restarts can make its later batches larger than its first
/// (see [`Method::growth`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Growth {
    /// Upper bound on the run's work, in first generations: the most
    /// candidates it can evaluate over the first generation's.
    pub first_generations: u64,
    /// The IPOP restarts allowed, each doubling the population.
    pub restarts: u32,
}

/// An [`EvaluationOracle`] decorator that warns once when the run, extrapolated
/// from its first generation, would take longer than `walltime`.
///
/// The first generation is timed from the start of its first batch to the end of
/// its last, so a QNG iteration includes the variance evaluations between its two
/// batches; with [`until_next_batch`](RunShape::until_next_batch), an external
/// optimizer's, to the start of the next batch, so the library's `tell()` and
/// `ask()` between the two are in it. Early stopping can only shorten a run, so the projection is an upper
/// bound. After the first generation the watch is one atomic load per batch.
///
/// The shape's [`leading_batches`](RunShape::leading_batches) pass through
/// untimed before it, counting their circuits, so a large calibration batch
/// is not taken for a generation. With `n` circuits in them and `m` in the
/// first generation, the projection is the first generation's time × (the
/// generations + `n / m`): the leading batches at the first generation's pace
/// per circuit, the assumption [`Method::growth`] makes of larger batches.
/// Without leading batches `n = 0`, and the projection is the first
/// generation's time × the generations, bit for bit.
pub(crate) struct WalltimeWatch<C: Clock> {
    inner: Box<dyn EvaluationOracle>,
    walltime: Duration,
    shape: RunShape,
    growth: Option<Growth>,
    run_id: String,
    clock: C,
    done: AtomicBool,
    state: Mutex<WatchState>,
}

#[derive(Default)]
struct WatchState {
    /// Leading batches seen so far, and their circuits.
    leading: u32,
    leading_circuits: u64,
    started: Option<Instant>,
    /// Batches of the first generation seen so far, and their circuits.
    batches: u32,
    circuits: u64,
    /// The first generation's batches are done and, with
    /// [`until_next_batch`](RunShape::until_next_batch), its time runs on
    /// until the next batch starts.
    awaiting_next: bool,
    /// The projected duration, when the warning fired.
    warned: Option<Duration>,
}

impl WatchState {
    /// The leading batches' work in first generations: their circuits over the
    /// first generation's (0 without leading batches).
    fn leading_generations(&self) -> f64 {
        if self.leading_circuits == 0 {
            return 0.0;
        }
        self.leading_circuits as f64 / self.circuits.max(1) as f64
    }
}

impl<C: Clock> WalltimeWatch<C> {
    pub(crate) fn new(
        inner: Box<dyn EvaluationOracle>,
        walltime: Duration,
        shape: RunShape,
        run_id: &str,
        clock: C,
    ) -> Self {
        Self {
            inner,
            walltime,
            shape,
            growth: None,
            run_id: run_id.to_string(),
            clock,
            done: AtomicBool::new(false),
            state: Mutex::new(WatchState::default()),
        }
    }

    /// Project with `growth`'s bound instead of `shape`'s generations, for a run
    /// whose later batches can be larger than its first.
    pub(crate) fn with_growth(mut self, growth: Option<Growth>) -> Self {
        self.growth = growth;
        self
    }

    /// The projected duration that triggered the warning, if it fired.
    #[cfg(test)]
    pub(crate) fn warned(&self) -> Option<Duration> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).warned
    }

    fn after_batch(&self, circuits: usize) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.batches += 1;
        state.circuits += circuits as u64;
        if state.batches < self.shape.batches_per_generation {
            return;
        }
        if self.shape.until_next_batch {
            state.awaiting_next = true;
            return;
        }
        self.done.store(true, Ordering::Relaxed);
        let end = self.clock.now();
        self.project(&mut state, end);
    }

    /// Project the run from a first generation that ended at `end`, and warn
    /// when the projection exceeds the walltime.
    fn project(&self, state: &mut WatchState, end: Instant) {
        let Some(started) = state.started else {
            return;
        };
        let first = end.saturating_duration_since(started);
        let generations = self.shape.generations_after(state.circuits);
        let work = self.growth.map_or(generations, |g| g.first_generations) as f64
            + state.leading_generations();
        let projected =
            Duration::try_from_secs_f64(first.as_secs_f64() * work).unwrap_or(Duration::MAX);
        if projected > self.walltime {
            state.warned = Some(projected);
            if let Some(growth) = self.growth {
                log::warn!(
                    "run {}: the first of up to {} {} took {}; with up to {} IPOP restarts, \
                     each doubling the population, the optimisation may take up to {}, \
                     beyond the {} walltime; the QPUs are released when it runs out. \
                     Raise `walltime` or lower the {} or the restarts.",
                    self.run_id,
                    generations,
                    self.shape.unit,
                    hms(first),
                    growth.restarts,
                    hms(projected),
                    hms(self.walltime),
                    self.shape.setting,
                );
                return;
            }
            if state.leading_circuits > 0 {
                log::warn!(
                    "run {}: the first of up to {} {} took {}; with the {} circuits sent \
                     before it at that pace, the optimisation may take up to {}, beyond the \
                     {} walltime; the QPUs are released when it runs out. Raise `walltime` \
                     or lower the {}.",
                    self.run_id,
                    generations,
                    self.shape.unit,
                    hms(first),
                    state.leading_circuits,
                    hms(projected),
                    hms(self.walltime),
                    self.shape.setting,
                );
                return;
            }
            // The library's time between the batches is in the first one's.
            let with_the_library = if self.shape.until_next_batch {
                ", with the optimizer's tell() and next ask(),"
            } else {
                ""
            };
            log::warn!(
                "run {}: the first of up to {} {}{with_the_library} took {}, so the \
                 optimisation may take up to {}, beyond the {} walltime; the QPUs are \
                 released when it runs out. Raise `walltime` or lower the {}.",
                self.run_id,
                generations,
                self.shape.unit,
                hms(first),
                hms(projected),
                hms(self.walltime),
                self.shape.setting,
            );
        }
    }
}

impl<C: Clock> WalltimeWatch<C> {
    /// Run `call`, a batch of `circuits` circuits: counted, untimed, while
    /// leading batches remain, then timed as one of the first generation's
    /// batches until it is complete, then passed straight through.
    fn observe<T>(&self, circuits: usize, call: impl FnOnce() -> T) -> T {
        self.observe_if(circuits, call, |_| true)
    }

    /// [`observe`](Self::observe) a call that is a batch only when `scored`
    /// says so of its result.
    fn observe_if<T>(
        &self,
        circuits: usize,
        call: impl FnOnce() -> T,
        scored: impl FnOnce(&T) -> bool,
    ) -> T {
        if self.done.load(Ordering::Relaxed) {
            return call();
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.leading < self.shape.leading_batches {
                drop(state);
                let out = call();
                if scored(&out) {
                    let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                    state.leading += 1;
                    state.leading_circuits += circuits as u64;
                }
                return out;
            }
            if state.awaiting_next {
                // The next batch starts: the first generation ends here.
                self.done.store(true, Ordering::Relaxed);
                let end = self.clock.now();
                self.project(&mut state, end);
                drop(state);
                return call();
            }
            state.started.get_or_insert_with(|| self.clock.now());
        }
        let out = call();
        if scored(&out) {
            self.after_batch(circuits);
        }
        out
    }
}

impl<C: Clock> EvaluationOracle for WalltimeWatch<C> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.observe(candidates.len(), || self.inner.evaluate_batch(candidates))
    }

    /// Forwarded, neither timed nor counted: the optimizer never sends such a
    /// batch. A training run's final batch (`final_shots`) comes after the
    /// optimizer has stopped and is not in the [`RunShape`], so the projection
    /// leaves it out.
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.inner.evaluate_batch_with_shots(candidates, shots)
    }
}

/// Another of the run's oracles behind the same [`WalltimeWatch`], so its calls
/// count as batches of the generation the watch times: an exact gradient, one
/// batch of an Adam iteration.
///
/// It forwards every call, so Adam sends the same batches with and without
/// the watch. A [`GradientOracle::value_and_gradient`] that returns `None` (the
/// default) scored nothing and is not counted: Adam then scores θ through the
/// watched evaluation oracle and calls the gradient, the two batches
/// [`Method::run_shape_over`] counts per iteration. One that scores θ itself
/// makes an iteration a single batch, which the shape cannot tell in advance:
/// the watch then times two iterations as the first, so the projection
/// overestimates. A gradient call has no candidates, so it counts as a batch of
/// no circuits.
struct Watched<T: ?Sized, C: Clock> {
    watch: Arc<WalltimeWatch<C>>,
    inner: Box<T>,
}

impl<C: Clock> EvaluationOracle for Watched<dyn EvaluationOracle, C> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.watch
            .observe(candidates.len(), || self.inner.evaluate_batch(candidates))
    }

    /// Forwarded, neither timed nor counted, as the watch's own: the final
    /// batch comes after the optimizer and is not in the [`RunShape`].
    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.inner.evaluate_batch_with_shots(candidates, shots)
    }
}

impl<C: Clock> GradientOracle for Watched<dyn GradientOracle, C> {
    fn gradient(&self, theta: &[f64], param_index: usize) -> f64 {
        self.watch
            .observe(0, || self.inner.gradient(theta, param_index))
    }

    fn gradient_batch(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.watch
            .observe(0, || self.inner.gradient_batch(theta, dims))
    }

    fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
        self.watch.observe_if(
            0,
            || self.inner.value_and_gradient(theta, dims),
            Option::is_some,
        )
    }
}

/// `oracles` behind one [`WalltimeWatch`] for `method`'s call pattern.
pub(crate) fn watched<C: Clock + 'static>(
    oracles: Oracles,
    walltime: Duration,
    method: &Method,
    dimensions: u32,
    run_id: &str,
    clock: C,
) -> Oracles {
    let shape = method.run_shape_over(&oracles, dimensions);
    let watch = Arc::new(
        WalltimeWatch::new(oracles.evaluation, walltime, shape, run_id, clock)
            .with_growth(method.growth(dimensions)),
    );
    Oracles {
        gradient: oracles.gradient.map(|inner| {
            Box::new(Watched {
                watch: Arc::clone(&watch),
                inner,
            }) as Box<dyn GradientOracle>
        }),
        full_evaluation: oracles.full_evaluation.map(|inner| {
            Box::new(Watched {
                watch: Arc::clone(&watch),
                inner,
            }) as Box<dyn EvaluationOracle>
        }),
        evaluation: Box::new(watch),
    }
}

/// `H:MM:SS`, the form `walltime` is given in.
fn hms(d: Duration) -> String {
    let s = d.as_secs();
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_log::{capture_logs, logged_for};
    use std::sync::Arc;

    /// A clock the oracle below advances by hand.
    struct FakeClock {
        base: Instant,
        offset: Mutex<Duration>,
    }

    impl FakeClock {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                base: Instant::now(),
                offset: Mutex::new(Duration::ZERO),
            })
        }

        fn advance(&self, by: Duration) {
            *self.offset.lock().unwrap() += by;
        }
    }

    impl Clock for Arc<FakeClock> {
        fn now(&self) -> Instant {
            self.base + *self.offset.lock().unwrap()
        }
    }

    /// Every batch takes `per_batch` on the fake clock.
    struct SlowOracle {
        clock: Arc<FakeClock>,
        per_batch: Duration,
    }

    impl EvaluationOracle for SlowOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.clock.advance(self.per_batch);
            vec![0.0; candidates.len()]
        }
    }

    fn watch(
        run_id: &str,
        per_batch: Duration,
        walltime: Duration,
        shape: RunShape,
    ) -> (WalltimeWatch<Arc<FakeClock>>, Arc<FakeClock>) {
        let clock = FakeClock::new();
        let oracle = SlowOracle {
            clock: Arc::clone(&clock),
            per_batch,
        };
        let watch = WalltimeWatch::new(
            Box::new(oracle),
            walltime,
            shape,
            run_id,
            Arc::clone(&clock),
        );
        (watch, clock)
    }

    const DE_100: RunShape = RunShape {
        batches_per_generation: 1,
        generations: 101,
        leading_batches: 0,
        evaluations: None,
        until_next_batch: false,
        unit: "generations",
        setting: "generations",
    };

    #[test]
    fn warns_once_when_the_projection_exceeds_the_walltime() {
        capture_logs();
        let run_id = "walltime-watch-exceeds";
        // 101 generations of 2 min each is 3:22:00, past a 3 h walltime.
        let (watch, _) = watch(
            run_id,
            Duration::from_secs(120),
            Duration::from_secs(3 * 3600),
            DE_100,
        );
        for _ in 0..5 {
            assert_eq!(watch.evaluate_batch(&vec![vec![0.0]; 4]), vec![0.0; 4]);
        }
        assert_eq!(watch.warned(), Some(Duration::from_secs(101 * 120)));
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 101 generations took 0:02:00, so the \
                 optimisation may take up to 3:22:00, beyond the 3:00:00 walltime; the QPUs \
                 are released when it runs out. Raise `walltime` or lower the generations."
            )]
        );
    }

    #[test]
    fn stays_quiet_when_the_projection_fits() {
        capture_logs();
        let run_id = "walltime-watch-fits";
        // 101 × 1 min = 1:41:00, inside a 3 h walltime.
        let (watch, _) = watch(
            run_id,
            Duration::from_secs(60),
            Duration::from_secs(3 * 3600),
            DE_100,
        );
        watch.evaluate_batch(&[vec![0.0]]);
        assert_eq!(watch.warned(), None);
        assert!(logged_for(run_id).is_empty());
    }

    /// Later generations do not re-estimate: a slow second generation after a fast
    /// first one does not warn.
    #[test]
    fn only_the_first_generation_is_timed() {
        capture_logs();
        let run_id = "walltime-watch-first-only";
        let (watch, clock) = watch(
            run_id,
            Duration::from_secs(1),
            Duration::from_secs(3600),
            DE_100,
        );
        watch.evaluate_batch(&[vec![0.0]]);
        clock.advance(Duration::from_secs(10 * 3600));
        watch.evaluate_batch(&[vec![0.0]]);
        assert_eq!(watch.warned(), None);
        assert!(logged_for(run_id).is_empty());
    }

    /// A QNG iteration is two batches with the variance callback in between; the
    /// estimate waits for both and includes the gap.
    #[test]
    fn a_qng_iteration_spans_both_batches() {
        capture_logs();
        let run_id = "walltime-watch-qng";
        let shape = RunShape {
            batches_per_generation: 2,
            generations: 50,
            leading_batches: 0,
            evaluations: None,
            until_next_batch: false,
            unit: "iterations",
            setting: "iterations",
        };
        let (watch, clock) = watch(
            run_id,
            Duration::from_secs(30),
            Duration::from_secs(3600),
            shape,
        );
        watch.evaluate_batch(&vec![vec![0.0]; 6]);
        assert_eq!(
            watch.warned(),
            None,
            "no estimate before the iteration ends"
        );
        clock.advance(Duration::from_secs(20));
        watch.evaluate_batch(&[vec![0.0]]);
        // (30 + 20 + 30) s × 50 = 4000 s > 3600 s.
        assert_eq!(watch.warned(), Some(Duration::from_secs(4000)));
        assert_eq!(logged_for(run_id).len(), 1);
    }

    /// Every circuit takes `per_circuit` on the fake clock, so a batch takes
    /// as long as its size says.
    struct PerCircuitOracle {
        clock: Arc<FakeClock>,
        per_circuit: Duration,
    }

    impl EvaluationOracle for PerCircuitOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.clock
                .advance(self.per_circuit * candidates.len() as u32);
            vec![0.0; candidates.len()]
        }
    }

    fn per_circuit_watch(
        run_id: &str,
        walltime: Duration,
        shape: RunShape,
    ) -> (WalltimeWatch<Arc<FakeClock>>, Arc<FakeClock>) {
        let clock = FakeClock::new();
        let oracle = PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        };
        let watch = WalltimeWatch::new(
            Box::new(oracle),
            walltime,
            shape,
            run_id,
            Arc::clone(&clock),
        );
        (watch, clock)
    }

    /// A calibration batch of 50 circuits, then iterations of 3: the
    /// calibration is not timed as the first iteration, and it adds 50 / 3
    /// iterations at the first iteration's pace.
    const CALIBRATED: RunShape = RunShape {
        batches_per_generation: 1,
        generations: 300,
        leading_batches: 1,
        evaluations: None,
        until_next_batch: false,
        unit: "iterations",
        setting: "iterations",
    };

    #[test]
    fn leading_batches_are_not_timed_and_count_by_their_circuits() {
        capture_logs();
        let run_id = "walltime-watch-leading";
        let (watch, _) = per_circuit_watch(run_id, Duration::from_secs(2 * 3600), CALIBRATED);
        watch.evaluate_batch(&vec![vec![0.0]; 50]);
        assert_eq!(watch.warned(), None, "the calibration is not an iteration");
        watch.evaluate_batch(&vec![vec![0.0]; 3]);
        // 30 s × (300 + 50/3) = 9500 s, where timing the 500 s calibration as
        // the first of 300 iterations would have projected 150 000 s.
        let projected = watch.warned().expect("9500 s is beyond 2 h");
        assert!(
            projected.abs_diff(Duration::from_secs(9500)) < Duration::from_millis(1),
            "{projected:?}"
        );
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 300 iterations took 0:00:30; with the 50 \
                 circuits sent before it at that pace, the optimisation may take up to \
                 2:38:20, beyond the 2:00:00 walltime; the QPUs are released when it runs \
                 out. Raise `walltime` or lower the iterations."
            )]
        );

        // The same run fits a 3 h walltime.
        let quiet = "walltime-watch-leading-fits";
        let (watch, _) = per_circuit_watch(quiet, Duration::from_secs(3 * 3600), CALIBRATED);
        watch.evaluate_batch(&vec![vec![0.0]; 50]);
        watch.evaluate_batch(&vec![vec![0.0]; 3]);
        assert_eq!(watch.warned(), None);
        assert!(logged_for(quiet).is_empty());
    }

    /// Without leading batches the first batch is timed, whatever its size: why
    /// a calibration needs them.
    #[test]
    fn without_leading_batches_the_first_batch_is_the_first_generation() {
        let run_id = "walltime-watch-no-leading";
        let shape = RunShape {
            leading_batches: 0,
            ..CALIBRATED
        };
        let (watch, _) = per_circuit_watch(run_id, Duration::from_secs(3 * 3600), shape);
        watch.evaluate_batch(&vec![vec![0.0]; 50]);
        assert_eq!(watch.warned(), Some(Duration::from_secs(500 * 300)));
    }

    /// A generation of several batches after a leading one: the generation's
    /// circuits are summed over its batches, and the time spans them all.
    #[test]
    fn leading_batches_are_converted_by_the_whole_first_generation() {
        let run_id = "walltime-watch-leading-two-batches";
        let shape = RunShape {
            batches_per_generation: 2,
            generations: 100,
            leading_batches: 2,
            evaluations: None,
            until_next_batch: false,
            unit: "iterations",
            setting: "iterations",
        };
        let (watch, clock) = per_circuit_watch(run_id, Duration::from_secs(3600), shape);
        watch.evaluate_batch(&vec![vec![0.0]; 20]);
        watch.evaluate_batch(&vec![vec![0.0]; 10]);
        // A gap before the first generation is not timed either.
        clock.advance(Duration::from_secs(1000));
        watch.evaluate_batch(&vec![vec![0.0]; 4]);
        assert_eq!(
            watch.warned(),
            None,
            "no estimate before the generation ends"
        );
        watch.evaluate_batch(&vec![vec![0.0]; 2]);
        // 60 s × (100 + 30/6) = 6300 s.
        assert_eq!(watch.warned(), Some(Duration::from_secs(6300)));
    }

    fn de(
        strategy: polypus_optimizers::DeStrategy,
        population_size: u32,
        generations: u32,
    ) -> Method {
        Method::De(crate::DeConfig {
            generations,
            population_size,
            tolerance: 0.0,
            patience: 10,
            strategy,
            mutation_factor: None,
            crossover_rate: None,
        })
    }

    #[test]
    fn run_shape_follows_the_optimizer_call_pattern() {
        use crate::PsoConfig;
        use polypus_optimizers::DeStrategy;
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade] {
            assert_eq!(de(strategy, 10, 100).run_shape(2), DE_100, "{strategy}");
        }
        // L-SHADE spends the same 1010 evaluations in more, smaller generations.
        let lshade = de(DeStrategy::LShade, 10, 100).run_shape(2);
        assert_eq!(
            lshade.generations,
            DeStrategy::LShade.max_generations(10, 100) + 1
        );
        assert!(lshade.generations > DE_100.generations, "{lshade:?}");
        assert_eq!(lshade.batches_per_generation, 1);
        let pso = Method::Pso(PsoConfig {
            generations: 7,
            population_size: 10,
            bounds: (0.0, 1.0),
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            tolerance: 0.0,
        });
        assert_eq!(pso.run_shape(2).generations, 8);
        // None of them sends a batch before its first generation.
        assert_eq!(lshade.leading_batches, 0);
        assert_eq!(pso.run_shape(2).leading_batches, 0);
        assert_eq!(adam(5).run_shape(2).leading_batches, 0);
        assert_eq!(cmaes(None, 5, 2, 20, 0.01).run_shape(2).leading_batches, 0);
    }

    /// Counts the batches it is handed; every candidate scores the same, so the
    /// stagnation stop is the only thing that could end a run early.
    struct CountingOracle(Arc<Mutex<u64>>);

    impl EvaluationOracle for CountingOracle {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            *self.0.lock().unwrap() += 1;
            vec![0.0; candidates.len()]
        }
    }

    /// `run_shape` counts exactly the batches each DE strategy sends when it does
    /// not stop early, L-SHADE's longer run included.
    #[test]
    fn de_run_shape_counts_the_batches_the_optimizer_sends() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        use polypus_optimizers::DeStrategy;
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            for (population_size, generations) in [(4, 12), (5, 10), (12, 0), (16, 25), (40, 30)] {
                let method = Method::De(crate::DeConfig {
                    generations,
                    population_size,
                    // Zero improvement is never below a zero tolerance.
                    tolerance: 0.0,
                    patience: 1,
                    strategy,
                    mutation_factor: None,
                    crossover_rate: None,
                });
                let shape = method.run_shape(2);
                let batches = Arc::new(Mutex::new(0));
                let oracle = Box::new(CountingOracle(Arc::clone(&batches)));
                let outcome =
                    dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 3, None).ok();
                let label = format!("{strategy} {population_size} × {generations}");
                assert!(outcome.is_some_and(|o| !o.converged), "{label}");
                assert_eq!(*batches.lock().unwrap(), shape.generations, "{label}");
            }
        }
    }

    /// A zero gradient that counts its calls into the same counter as
    /// [`CountingOracle`].
    struct CountingGradient(Arc<Mutex<u64>>);

    impl GradientOracle for CountingGradient {
        fn gradient(&self, _theta: &[f64], _param_index: usize) -> f64 {
            *self.0.lock().unwrap() += 1;
            0.0
        }
        fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
            *self.0.lock().unwrap() += 1;
            vec![0.0; dims]
        }
    }

    /// `run_shape_over` counts the batches Adam sends when it does not stop
    /// early: one per iteration with the finite difference, and with an exact
    /// gradient or a minibatch the scored θ plus the gradient's call, then the
    /// final θ, one batch, which those shapes count as two.
    #[test]
    fn adam_run_shape_counts_the_batches_the_optimizer_sends() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        for (exact, minibatch) in [(false, false), (true, false), (false, true), (true, true)] {
            for max_iters in [1, 7, 30] {
                let batches = Arc::new(Mutex::new(0));
                let oracles = Oracles {
                    evaluation: Box::new(CountingOracle(Arc::clone(&batches))),
                    gradient: exact.then(|| {
                        Box::new(CountingGradient(Arc::clone(&batches))) as Box<dyn GradientOracle>
                    }),
                    full_evaluation: minibatch.then(|| {
                        Box::new(CountingOracle(Arc::clone(&batches))) as Box<dyn EvaluationOracle>
                    }),
                };
                let method = Method::Adam(crate::AdamConfig {
                    max_iters,
                    learning_rate: 0.1,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    bounds: (0.0, 1.0),
                    // Zero improvement is never below a zero tolerance.
                    tolerance: 0.0,
                    patience: 1,
                    finite_difference_step: 0.2,
                    rule: polypus_optimizers::AdamRule::Adam,
                });
                let shape = method.run_shape_over(&oracles, 2);
                let outcome =
                    dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None).ok();
                let label = format!(
                    "exact gradient {exact}, minibatch {minibatch}, {max_iters} iterations"
                );
                assert!(outcome.is_some_and(|o| !o.converged), "{label}");
                assert_eq!(shape.generations, u64::from(max_iters) + 1, "{label}");
                assert_eq!(
                    *batches.lock().unwrap(),
                    shape.generations * u64::from(shape.batches_per_generation)
                        - u64::from(exact || minibatch),
                    "{label}"
                );
            }
        }
    }

    /// A zero gradient that scores θ itself, as zero, and counts its calls into
    /// the same counter as [`CountingOracle`].
    struct CountingScoringGradient(Arc<Mutex<u64>>);

    impl GradientOracle for CountingScoringGradient {
        fn gradient(&self, _theta: &[f64], _param_index: usize) -> f64 {
            *self.0.lock().unwrap() += 1;
            0.0
        }
        fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
            *self.0.lock().unwrap() += 1;
            vec![0.0; dims]
        }
        fn value_and_gradient(&self, _theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
            *self.0.lock().unwrap() += 1;
            Some((0.0, vec![0.0; dims]))
        }
    }

    /// An Adam run of `max_iters` iterations that never stops early: zero
    /// improvement is never below a zero tolerance.
    fn adam(max_iters: u32) -> Method {
        Method::Adam(crate::AdamConfig {
            max_iters,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (0.0, 1.0),
            tolerance: 0.0,
            patience: 1,
            finite_difference_step: 0.2,
            rule: polypus_optimizers::AdamRule::Adam,
        })
    }

    /// The watch forwards every call, so Adam over an exact gradient sends the
    /// same batches with and without it: one per iteration when the gradient
    /// scores θ itself, two (θ, then the gradient) when it does not, and the
    /// final θ. `run_shape_over` counts two per iteration either way.
    #[test]
    fn adam_sends_the_same_batches_with_and_without_the_watch() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        for scores_theta in [false, true] {
            for behind_the_watch in [false, true] {
                let batches = Arc::new(Mutex::new(0));
                let gradient: Box<dyn GradientOracle> = if scores_theta {
                    Box::new(CountingScoringGradient(Arc::clone(&batches)))
                } else {
                    Box::new(CountingGradient(Arc::clone(&batches)))
                };
                let oracles = Oracles {
                    evaluation: Box::new(CountingOracle(Arc::clone(&batches))),
                    gradient: Some(gradient),
                    full_evaluation: None,
                };
                let method = adam(7);
                let shape = method.run_shape_over(&oracles, 2);
                let oracles = if behind_the_watch {
                    watched(
                        oracles,
                        Duration::from_secs(3600),
                        &method,
                        2,
                        "walltime-watch-adam-batches",
                        FakeClock::new(),
                    )
                } else {
                    oracles
                };
                let outcome =
                    dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None).ok();
                let label = format!("scores θ {scores_theta}, watched {behind_the_watch}");
                assert!(outcome.is_some_and(|o| !o.converged), "{label}");
                assert_eq!(shape.batches_per_generation, 2, "{label}");
                let per_iteration = if scores_theta { 1 } else { 2 };
                assert_eq!(*batches.lock().unwrap(), 7 * per_iteration + 1, "{label}");
            }
        }
    }

    fn spsa(max_iters: u32, resamplings: u32, eval_interval: u32) -> Method {
        Method::Spsa(crate::SpsaConfig {
            max_iters,
            resamplings,
            perturbation: 0.1,
            target_step: 0.63,
            alpha: 0.602,
            gamma: 0.101,
            stability_constant: 0.0,
            eval_interval,
        })
    }

    /// `run_shape` counts exactly the batches SPSA sends: the leading
    /// calibration, then one per iteration (SPSA never stops early) and the
    /// final θ.
    #[test]
    fn spsa_run_shape_counts_the_batches_the_optimizer_sends() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        for (max_iters, resamplings, eval_interval) in [(1, 1, 20), (7, 3, 2), (30, 8, 20)] {
            let method = spsa(max_iters, resamplings, eval_interval);
            let shape = method.run_shape(2);
            assert_eq!(method.growth(2), None);
            let batches = Arc::new(Mutex::new(0));
            let oracle = Box::new(CountingOracle(Arc::clone(&batches)));
            let outcome =
                dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 3, None).ok();
            let label = format!("{max_iters} × {resamplings}, every {eval_interval}");
            assert!(outcome.is_some_and(|o| o.iterations_run == max_iters as usize));
            assert_eq!(shape.leading_batches, 1, "{label}");
            assert_eq!(shape.generations, u64::from(max_iters) + 1, "{label}");
            assert_eq!(
                *batches.lock().unwrap(),
                u64::from(shape.leading_batches)
                    + shape.generations * u64::from(shape.batches_per_generation),
                "{label}"
            );
        }
    }

    /// Through the watch, an SPSA run's calibration (50 circuits) is not timed
    /// as its first iteration (θ₀ and one pair, 3 circuits): at 10 s a
    /// circuit, 299 iterations and the final θ project to
    /// 30 s × (300 + 50/3) = 2:38:20, not 500 s × 300.
    #[test]
    fn the_spsa_calibration_is_not_timed_as_the_first_iteration() {
        capture_logs();
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let run_id = "walltime-watch-spsa";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        });
        let method = spsa(299, 1, 20);
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(2 * 3600),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let outcome = dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None);
        assert!(outcome.is_ok());
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 300 iterations took 0:00:30; with the 50 \
                 circuits sent before it at that pace, the optimisation may take up to \
                 2:38:20, beyond the 2:00:00 walltime; the QPUs are released when it runs \
                 out. Raise `walltime` or lower the iterations."
            )]
        );
    }

    fn rotosolve(sweeps: u32, reevaluate_every: u32) -> Method {
        Method::Rotosolve(crate::RotosolveConfig {
            sweeps,
            reevaluate_every,
            // Zero improvement is never below a zero tolerance.
            tolerance: 0.0,
            patience: 1,
        })
    }

    /// `run_shape` counts exactly the batches Rotosolve sends when it does not
    /// stop early: one per step, `sweeps × dimensions` steps, nothing before
    /// the first, and the final point after the last.
    #[test]
    fn rotosolve_run_shape_counts_the_batches_the_optimizer_sends() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        for (sweeps, dimensions, reevaluate_every) in
            [(1, 1, 5), (3, 2, 1), (10, 4, 5), (7, 3, 100)]
        {
            let method = rotosolve(sweeps, reevaluate_every);
            let shape = method.run_shape(dimensions);
            assert_eq!(method.growth(dimensions), None);
            assert_eq!(shape.leading_batches, 0);
            assert_eq!(shape.batches_per_generation, 1);
            assert_eq!((shape.unit, shape.setting), ("steps", "sweeps"));
            let batches = Arc::new(Mutex::new(0));
            let oracle = Box::new(CountingOracle(Arc::clone(&batches)));
            let outcome =
                dispatch_optimizer(method, oracle, dimensions, &OracleErrorSlot::new(), 3, None)
                    .ok();
            let label = format!("{sweeps} sweeps × {dimensions}, every {reevaluate_every}");
            let steps = (sweeps * dimensions) as usize;
            assert!(
                outcome.is_some_and(|o| !o.converged && o.iterations_run == steps),
                "{label}"
            );
            assert_eq!(shape.generations, steps as u64 + 1, "{label}");
            assert_eq!(*batches.lock().unwrap(), shape.generations, "{label}");
        }
    }

    /// Through the watch, a Rotosolve run is projected from its first step,
    /// which also measures θ₀: at 10 s a circuit, 3 circuits, so 100 sweeps of
    /// 3 parameters and the final point project to 30 s × 301 = 2:30:30. Most
    /// steps hold 2 circuits, so the run itself takes (300 × 2 + 60 + 1) ×
    /// 10 s = 1:50:10: on one QPU the projection overestimates, as `run_shape`
    /// says.
    #[test]
    fn a_rotosolve_run_is_projected_from_its_first_step() {
        capture_logs();
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let run_id = "walltime-watch-rotosolve";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        });
        let method = rotosolve(100, 5);
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(2 * 3600),
            &method,
            3,
            run_id,
            Arc::clone(&clock),
        );
        let start = clock.now();
        let outcome = dispatch_optimizer(method, oracles, 3, &OracleErrorSlot::new(), 3, None);
        assert!(outcome.is_ok());
        assert_eq!(
            clock.now().saturating_duration_since(start),
            Duration::from_secs(6610)
        );
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 301 steps took 0:00:30, so the optimisation \
                 may take up to 2:30:30, beyond the 2:00:00 walltime; the QPUs are released \
                 when it runs out. Raise `walltime` or lower the sweeps."
            )]
        );
    }

    /// An ask/tell optimizer that proposes `rows` points of two dimensions at
    /// every `ask()` and never ends, so only the budget stops it.
    struct Constant {
        rows: usize,
    }

    impl polypus_optimizers::AskTell for Constant {
        fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, polypus_optimizers::AskTellError> {
            Ok(Some(vec![vec![0.5, 0.25]; self.rows]))
        }
        fn tell(
            &mut self,
            _candidates: &[Vec<f64>],
            _values: &[f64],
        ) -> Result<(), polypus_optimizers::AskTellError> {
            Ok(())
        }
    }

    fn external(max_evaluations: u64, rows: usize) -> Method {
        Method::External(
            crate::ExternalConfig { max_evaluations },
            Box::new(Constant { rows }),
        )
    }

    /// [`Constant`]'s batches from a library that takes `per_ask` and
    /// `per_tell` of the fake clock, as a model-based one computes.
    struct Thinking {
        rows: usize,
        clock: Arc<FakeClock>,
        per_ask: Duration,
        per_tell: Duration,
    }

    impl polypus_optimizers::AskTell for Thinking {
        fn ask(&mut self) -> Result<Option<Vec<Vec<f64>>>, polypus_optimizers::AskTellError> {
            self.clock.advance(self.per_ask);
            Ok(Some(vec![vec![0.5, 0.25]; self.rows]))
        }
        fn tell(
            &mut self,
            _candidates: &[Vec<f64>],
            _values: &[f64],
        ) -> Result<(), polypus_optimizers::AskTellError> {
            self.clock.advance(self.per_tell);
            Ok(())
        }
    }

    /// With batches of constant size, the generations the shape counts after
    /// the first batch are exactly the batches the bridge sends: it stops
    /// before the batch that would take it past `max_evaluations`.
    #[test]
    fn external_run_shape_counts_the_batches_the_bridge_sends() {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        for (max_evaluations, rows) in [(1, 1), (10, 3), (12, 4), (12, 12), (100, 7), (1000, 9)] {
            let method = external(max_evaluations, rows);
            let shape = method.run_shape(2);
            assert_eq!(method.growth(2), None);
            assert_eq!(shape.leading_batches, 0);
            assert_eq!(shape.batches_per_generation, 1);
            assert_eq!(shape.generations, max_evaluations);
            assert_eq!(shape.evaluations, Some(max_evaluations));
            assert_eq!((shape.unit, shape.setting), ("batches", "max_evaluations"));
            let batches = Arc::new(Mutex::new(0));
            let oracle = Box::new(CountingOracle(Arc::clone(&batches)));
            let outcome =
                dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 3, None).ok();
            let label = format!("{max_evaluations} evaluations in batches of {rows}");
            let sent = *batches.lock().unwrap();
            assert!(
                outcome.is_some_and(|o| !o.converged && o.iterations_run as u64 == sent),
                "{label}"
            );
            assert_eq!(sent, shape.generations_after(rows as u64), "{label}");
        }
        // Every other optimizer's generations do not depend on the first's size.
        let rotosolve = rotosolve(3, 5).run_shape(4);
        assert_eq!(rotosolve.evaluations, None);
        assert_eq!(rotosolve.generations_after(7), rotosolve.generations);
    }

    /// Through the watch, an external run is projected from its first batch:
    /// at 10 s a circuit, batches of 6 take 1 min, and a budget of 303
    /// evaluations holds 50 of them, so the projection is 0:50:00, which is
    /// also how long the run takes.
    #[test]
    fn an_external_run_is_projected_from_its_first_batch() {
        capture_logs();
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let run_id = "walltime-watch-external";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        });
        let method = external(303, 6);
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(40 * 60),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let start = clock.now();
        let outcome = dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None);
        assert!(outcome.is_ok_and(|o| o.iterations_run == 50));
        assert_eq!(
            clock.now().saturating_duration_since(start),
            Duration::from_secs(3000)
        );
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 50 batches, with the optimizer's tell() and \
                 next ask(), took 0:01:00, so the optimisation may take up to 0:50:00, beyond \
                 the 0:40:00 walltime; the QPUs are released when it runs out. Raise \
                 `walltime` or lower the max_evaluations."
            )]
        );

        // The same run fits a walltime of an hour.
        let quiet = "walltime-watch-external-fits";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        });
        let method = external(303, 6);
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(3600),
            &method,
            2,
            quiet,
            Arc::clone(&clock),
        );
        assert!(dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None).is_ok());
        assert!(logged_for(quiet).is_empty());
    }

    /// The library's own time counts: the first batch is timed until the
    /// second starts, so its `tell()` and the next `ask()` are in it. At 10 s
    /// a circuit, batches of 6 take 1 min, and with 20 s in `tell()` and 10 s
    /// in `ask()` a batch takes 1:30; 50 of them, 1:15:00, which the
    /// projection gives where the batches alone would give 0:50:00.
    #[test]
    fn an_external_run_counts_the_librarys_own_time() {
        capture_logs();
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let run_id = "walltime-watch-external-thinking";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(10),
        });
        let method = Method::External(
            crate::ExternalConfig {
                max_evaluations: 303,
            },
            Box::new(Thinking {
                rows: 6,
                clock: Arc::clone(&clock),
                per_ask: Duration::from_secs(10),
                per_tell: Duration::from_secs(20),
            }),
        );
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(3600),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let start = clock.now();
        let outcome = dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None);
        assert!(outcome.is_ok_and(|o| o.iterations_run == 50));
        // 50 batches and tells, and 51 asks: the last proposes a batch that
        // does not fit.
        assert_eq!(
            clock.now().saturating_duration_since(start),
            Duration::from_secs(50 * 90 + 10)
        );
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 50 batches, with the optimizer's tell() and \
                 next ask(), took 0:01:30, so the optimisation may take up to 1:15:00, beyond \
                 the 1:00:00 walltime; the QPUs are released when it runs out. Raise \
                 `walltime` or lower the max_evaluations."
            )]
        );
    }

    /// A run of one batch is not projected: the first batch's time runs until
    /// a second one starts, and none does.
    #[test]
    fn an_external_run_of_one_batch_is_not_projected() {
        capture_logs();
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let run_id = "walltime-watch-external-one-batch";
        let clock = FakeClock::new();
        let oracle: Box<dyn EvaluationOracle> = Box::new(PerCircuitOracle {
            clock: Arc::clone(&clock),
            per_circuit: Duration::from_secs(3600),
        });
        let method = external(6, 6);
        let oracles = watched(
            oracle.into(),
            Duration::from_secs(60),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let outcome = dispatch_optimizer(method, oracles, 2, &OracleErrorSlot::new(), 3, None);
        assert!(outcome.is_ok_and(|o| o.iterations_run == 1));
        assert!(logged_for(run_id).is_empty());
    }

    /// Takes `per_batch` on the fake clock and records the shots of every
    /// batch sent with shots.
    struct SlowShots {
        clock: Arc<FakeClock>,
        per_batch: Duration,
        shots: Mutex<Vec<u32>>,
    }

    impl EvaluationOracle for SlowShots {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.clock.advance(self.per_batch);
            vec![0.0; candidates.len()]
        }
        fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
            self.shots.lock().unwrap().push(shots);
            self.evaluate_batch(candidates)
        }
    }

    /// A batch with shots, as a run's final batch, reaches the inner oracle
    /// with its shots and is neither timed nor counted as a generation.
    #[test]
    fn a_batch_with_shots_is_forwarded_untimed() {
        capture_logs();
        let run_id = "walltime-watch-final-batch";
        let clock = FakeClock::new();
        let inner = Arc::new(SlowShots {
            clock: Arc::clone(&clock),
            per_batch: Duration::from_secs(120),
            shots: Mutex::new(Vec::new()),
        });
        let watch = WalltimeWatch::new(
            Box::new(Arc::clone(&inner)),
            Duration::from_secs(3 * 3600),
            DE_100,
            run_id,
            Arc::clone(&clock),
        );
        // Each would project 101 × 2 min, past the walltime, if it were timed.
        for _ in 0..3 {
            assert_eq!(
                watch.evaluate_batch_with_shots(&vec![vec![0.0]; 2], 4096),
                vec![0.0; 2]
            );
        }
        assert_eq!(*inner.shots.lock().unwrap(), vec![4096; 3]);
        assert_eq!(watch.warned(), None);
        assert!(logged_for(run_id).is_empty());
        // The first generation is still the optimizer's first batch.
        watch.evaluate_batch(&[vec![0.0]]);
        assert_eq!(watch.warned(), Some(Duration::from_secs(101 * 120)));
    }

    #[test]
    fn hms_matches_the_walltime_format() {
        assert_eq!(hms(Duration::from_secs(36_000)), "10:00:00");
        assert_eq!(hms(Duration::from_secs(172_861)), "48:01:01");
        assert_eq!(hms(Duration::from_millis(500)), "0:00:00");
    }

    fn cmaes(
        population_size: Option<u32>,
        generations: u32,
        restarts: u32,
        patience: u32,
        tolerance: f64,
    ) -> Method {
        Method::Cmaes(crate::CmaesConfig {
            generations,
            population_size,
            sigma0: std::f64::consts::FRAC_PI_2,
            tolerance,
            patience,
            restarts,
        })
    }

    /// Records every batch's size and scores each candidate with `fitness`.
    struct SizeRecorder {
        sizes: Arc<Mutex<Vec<u64>>>,
        fitness: fn(&[f64]) -> f64,
    }

    impl EvaluationOracle for SizeRecorder {
        fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
            self.sizes.lock().unwrap().push(candidates.len() as u64);
            candidates.iter().map(|c| (self.fitness)(c)).collect()
        }
    }

    fn flat(_: &[f64]) -> f64 {
        0.0
    }

    /// Periodic, maximised at the origin.
    fn cosine(x: &[f64]) -> f64 {
        x.iter().map(|v| v.cos() - 1.0).sum()
    }

    /// Dispatch `method` over `fitness` and return the outcome (`None` for a
    /// failed run) and every batch size.
    fn dispatch_cmaes(
        method: Method,
        dimensions: u32,
        fitness: fn(&[f64]) -> f64,
        seed: u64,
    ) -> (Option<polypus_optimizers::OptimizationOutcome>, Vec<u64>) {
        use crate::dispatch::{dispatch_optimizer, OracleErrorSlot};
        let sizes = Arc::new(Mutex::new(Vec::new()));
        let oracle = Box::new(SizeRecorder {
            sizes: Arc::clone(&sizes),
            fitness,
        });
        let outcome = dispatch_optimizer(
            method,
            oracle,
            dimensions,
            &OracleErrorSlot::new(),
            seed,
            None,
        )
        .ok();
        let sizes = sizes.lock().unwrap().clone();
        (outcome, sizes)
    }

    /// Dispatch hands every `CmaesConfig` field and the seed to CMA-ES: the
    /// outcome is exactly the one `AlgorithmCmaEs::optimize` returns for the
    /// same arguments, and `population_size: None` is Hansen's default λ.
    #[test]
    fn cmaes_dispatch_runs_the_configured_optimizer() {
        use polypus_optimizers::{AlgorithmCmaEs, AlgorithmCmaEsArgs, Optimizer};
        let method = Method::Cmaes(crate::CmaesConfig {
            generations: 60,
            population_size: Some(7),
            sigma0: 1.0,
            tolerance: 1e-6,
            patience: 5,
            restarts: 1,
        });
        let (dispatched, sizes) = dispatch_cmaes(method, 3, cosine, 21);
        let dispatched = dispatched.expect("a valid CMA-ES configuration runs");
        assert_eq!(sizes[0], 7);
        let direct = AlgorithmCmaEs
            .optimize(AlgorithmCmaEsArgs {
                oracle: Box::new(SizeRecorder {
                    sizes: Arc::default(),
                    fitness: cosine,
                }),
                population_size: Some(7),
                generations: 60,
                dimensions: 3,
                sigma0: 1.0,
                tolerance: 1e-6,
                patience: 5,
                restarts: 1,
                seed: Some(21),
                initial_point: None,
            })
            .expect("a valid CMA-ES configuration runs");
        assert_eq!(dispatched, direct);

        let (_, sizes) = dispatch_cmaes(cmaes(None, 3, 0, 20, 0.0), 6, cosine, 1);
        assert_eq!(sizes, vec![9; 3], "4 + ⌊3 ln 6⌋ = 9");
    }

    /// An invalid configuration reaches the optimizer's validation, which
    /// rejects it before any batch.
    #[test]
    fn cmaes_dispatch_rejects_an_invalid_configuration_before_any_batch() {
        for method in [
            cmaes(Some(1), 10, 0, 20, 0.01),
            cmaes(Some(8), 0, 0, 20, 0.01),
            cmaes(Some(8), 10, 40, 20, 0.01),
            Method::Cmaes(crate::CmaesConfig {
                generations: 10,
                population_size: None,
                sigma0: 0.0,
                tolerance: 0.01,
                patience: 20,
                restarts: 0,
            }),
        ] {
            let (outcome, sizes) = dispatch_cmaes(method, 2, cosine, 1);
            assert!(outcome.is_none());
            assert!(sizes.is_empty(), "{sizes:?}");
        }
    }

    /// Without restarts `run_shape` counts exactly the batches CMA-ES sends
    /// when it does not stop early: one per generation, no initial batch, all
    /// of the first population's size, and nothing grows.
    #[test]
    fn cmaes_run_shape_counts_the_batches_the_optimizer_sends() {
        for (population_size, generations) in [(None, 1), (None, 25), (Some(2), 12), (Some(16), 30)]
        {
            // Zero improvement is never below a zero tolerance.
            let method = cmaes(population_size, generations, 0, 1, 0.0);
            let shape = method.run_shape(2);
            assert_eq!(shape.batches_per_generation, 1);
            assert_eq!(method.growth(2), None);
            let (outcome, sizes) = dispatch_cmaes(method, 2, flat, 3);
            let label = format!("{population_size:?} × {generations}");
            assert!(outcome.is_some_and(|o| !o.converged), "{label}");
            assert_eq!(sizes.len() as u64, shape.generations, "{label}");
            let first = u64::from(population_size.unwrap_or(6));
            assert!(sizes.iter().all(|&n| n == first), "{label}: {sizes:?}");
        }
    }

    /// With IPOP restarts the batches grow, and `growth` bounds the work they
    /// add: every run is checked against the schedule's per-generation bound
    /// and the total against `first_generations` first batches. Patience 1
    /// on a flat landscape restarts after every second generation, the
    /// soonest stagnation can; the bound assumes a restart after every
    /// generation, which only a degenerate distribution can force.
    #[test]
    fn cmaes_growth_bounds_the_batches_with_restarts() {
        use polypus_optimizers::CmaEsBatchSchedule;
        for (population_size, generations, restarts, patience) in [
            (None, 8, 3, 1),
            (Some(5), 6, 2, 1),
            (Some(4), 40, 3, 2),
            (None, 30, 5, 3),
            (Some(6), 10, 4, 4),
        ] {
            let label = format!("{population_size:?} × {generations}, {restarts} restarts");
            let method = cmaes(population_size, generations, restarts, patience, 1.0);
            let shape = method.run_shape(2);
            let growth = method.growth(2).expect("restarts can grow the batches");
            assert_eq!(growth.restarts, restarts);
            let (outcome, sizes) = dispatch_cmaes(method, 2, flat, 5);
            assert!(outcome.is_some(), "{label}");
            assert!(sizes.len() as u64 <= shape.generations, "{label}");
            let schedule = CmaEsBatchSchedule::new(2, population_size, generations, restarts);
            let first = u64::from(schedule.initial_population());
            for (g, &n) in sizes.iter().enumerate() {
                assert!(n <= schedule.max_batch_size(g as u32), "{label}: {sizes:?}");
            }
            let total: u64 = sizes.iter().sum();
            assert!(total <= growth.first_generations * first, "{label}");
            if patience == 1 && generations == 2 * (restarts + 1) {
                let soonest: Vec<u64> = (0..generations).map(|g| first << (g / 2)).collect();
                assert_eq!(sizes, soonest, "{label}");
            }
        }
    }

    /// The watch projects a growing run with its bound, not its generations,
    /// and says why.
    #[test]
    fn a_growing_run_is_projected_with_its_bound() {
        capture_logs();
        let run_id = "walltime-watch-cmaes-restarts";
        let method = cmaes(Some(10), 100, 2, 20, 0.01);
        // 1 + 2 + 98 × 4 = 395 first generations of 1 min: 6:35:00 > 3 h, where
        // 100 generations alone (1:40:00) would fit.
        let growth = method.growth(4);
        assert_eq!(
            growth,
            Some(Growth {
                first_generations: 395,
                restarts: 2
            })
        );
        let (bare, _) = watch(
            run_id,
            Duration::from_secs(60),
            Duration::from_secs(3 * 3600),
            method.run_shape(2),
        );
        let growing = bare.with_growth(growth);
        growing.evaluate_batch(&vec![vec![0.0]; 10]);
        assert_eq!(growing.warned(), Some(Duration::from_secs(395 * 60)));
        assert_eq!(
            logged_for(run_id),
            vec![format!(
                "run {run_id}: the first of up to 100 generations took 0:01:00; with up to 2 \
                 IPOP restarts, each doubling the population, the optimisation may take up \
                 to 6:35:00, beyond the 3:00:00 walltime; the QPUs are released when it runs \
                 out. Raise `walltime` or lower the generations or the restarts."
            )]
        );

        let quiet = "walltime-watch-cmaes-no-restarts";
        let method = cmaes(Some(10), 100, 0, 20, 0.01);
        let (bare, _) = watch(
            quiet,
            Duration::from_secs(60),
            Duration::from_secs(3 * 3600),
            method.run_shape(2),
        );
        let growing = bare.with_growth(method.growth(4));
        growing.evaluate_batch(&vec![vec![0.0]; 10]);
        assert_eq!(growing.warned(), None);
        assert!(logged_for(quiet).is_empty());
    }

    #[test]
    fn only_cmaes_restarts_grow_the_batches() {
        let pso = Method::Pso(crate::PsoConfig {
            generations: 7,
            population_size: 10,
            bounds: (0.0, 1.0),
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            tolerance: 0.0,
        });
        assert_eq!(pso.growth(3), None);
        assert_eq!(cmaes(None, 50, 0, 20, 0.01).growth(3), None);
        // λ₀ cancels out: the bound is Σ_g 2^min(g, restarts) first generations.
        for population_size in [None, Some(2), Some(64)] {
            let growth = cmaes(population_size, 50, 3, 20, 0.01).growth(3);
            assert_eq!(
                growth.map(|g| g.first_generations),
                Some(1 + 2 + 4 + 47 * 8)
            );
        }
    }

    /// An exact gradient that takes `per_call` on the fake clock.
    struct SlowGradient {
        clock: Arc<FakeClock>,
        per_call: Duration,
    }

    impl GradientOracle for SlowGradient {
        fn gradient(&self, _theta: &[f64], _param_index: usize) -> f64 {
            self.clock.advance(self.per_call);
            0.0
        }
        fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
            self.clock.advance(self.per_call);
            vec![0.0; dims]
        }
    }

    /// An Adam iteration with an exact gradient is the scored θ plus the
    /// gradient: the watch times both, so a slow gradient is not missed.
    #[test]
    fn an_adam_iteration_counts_its_exact_gradient() {
        capture_logs();
        let run_id = "walltime-watch-adam";
        let clock = FakeClock::new();
        let oracles = Oracles {
            evaluation: Box::new(SlowOracle {
                clock: Arc::clone(&clock),
                per_batch: Duration::from_secs(10),
            }),
            gradient: Some(Box::new(SlowGradient {
                clock: Arc::clone(&clock),
                per_call: Duration::from_secs(110),
            })),
            full_evaluation: None,
        };
        let method = Method::Adam(crate::AdamConfig {
            max_iters: 100,
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            bounds: (0.0, 1.0),
            tolerance: 0.0,
            patience: 10,
            finite_difference_step: 1e-3,
            rule: polypus_optimizers::AdamRule::Adam,
        });
        // On its own, an Adam iteration is the one batch of θ and its stencil;
        // the exact gradient adds its call. The final θ counts as the 101st.
        assert_eq!(
            method.run_shape(2),
            RunShape {
                batches_per_generation: 1,
                generations: 101,
                leading_batches: 0,
                evaluations: None,
                until_next_batch: false,
                unit: "iterations",
                setting: "iterations",
            }
        );
        assert_eq!(method.run_shape_over(&oracles, 2).batches_per_generation, 2);
        let oracles = watched(
            oracles,
            Duration::from_secs(3 * 3600),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let gradient = oracles.gradient.as_ref().expect("the gradient is kept");
        // Installs the test logger before anything can warn, when this test
        // runs on its own.
        assert!(logged_for(run_id).is_empty());
        // An iteration as Adam makes it. The gradient does not score θ, and
        // that call scored nothing, so it is not a batch: were it counted, the
        // first iteration would end with θ's batch and take 0:00:10.
        assert_eq!(gradient.value_and_gradient(&[0.0, 0.0], 2), None);
        oracles.evaluation.evaluate_batch(&[vec![0.0, 0.0]]);
        gradient.gradient_batch(&[0.0, 0.0], 2);
        // 101 iterations of 2 min (110 s + 10 s) is 3:22:00, past the 3 h walltime.
        let logged = logged_for(run_id);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(
            logged[0].contains("101 iterations took 0:02:00"),
            "{}",
            logged[0]
        );
        assert!(logged[0].contains("3:22:00"), "{}", logged[0]);
    }

    /// An exact gradient that scores θ itself, as the sum of its parameters,
    /// in `per_call` on the fake clock.
    struct SlowScoringGradient {
        clock: Arc<FakeClock>,
        per_call: Duration,
    }

    impl GradientOracle for SlowScoringGradient {
        fn gradient(&self, _theta: &[f64], _param_index: usize) -> f64 {
            self.clock.advance(self.per_call);
            0.0
        }
        fn gradient_batch(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
            self.clock.advance(self.per_call);
            vec![0.0; dims]
        }
        fn value_and_gradient(&self, theta: &[f64], dims: usize) -> Option<(f64, Vec<f64>)> {
            self.clock.advance(self.per_call);
            Some((theta.iter().sum(), vec![0.0; dims]))
        }
    }

    /// Behind the watch, an exact gradient that scores θ itself still does,
    /// timed as a batch. The shape counts two calls per iteration, so the watch
    /// times two of its one-call iterations as the first: the projection
    /// overestimates, here warning about a run of 1:41:00 within 3 h.
    #[test]
    fn a_watched_gradient_that_scores_theta_is_forwarded_and_timed() {
        capture_logs();
        let run_id = "walltime-watch-adam-scoring";
        let clock = FakeClock::new();
        let oracles = Oracles {
            evaluation: Box::new(SlowOracle {
                clock: Arc::clone(&clock),
                per_batch: Duration::from_secs(10),
            }),
            gradient: Some(Box::new(SlowScoringGradient {
                clock: Arc::clone(&clock),
                per_call: Duration::from_secs(60),
            })),
            full_evaluation: None,
        };
        let method = adam(100);
        let oracles = watched(
            oracles,
            Duration::from_secs(3 * 3600),
            &method,
            2,
            run_id,
            Arc::clone(&clock),
        );
        let gradient = oracles.gradient.as_ref().expect("the gradient is kept");
        assert_eq!(
            gradient.value_and_gradient(&[0.5, 0.25], 2),
            Some((0.75, vec![0.0, 0.0]))
        );
        assert!(logged_for(run_id).is_empty());
        gradient.value_and_gradient(&[0.5, 0.25], 2);
        // Two iterations of 1 min, projected as 100 and the final θ, 101 of
        // 2 min: 3:22:00.
        let logged = logged_for(run_id);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(
            logged[0].contains("101 iterations took 0:02:00"),
            "{}",
            logged[0]
        );
        assert!(logged[0].contains("3:22:00"), "{}", logged[0]);
    }
}
