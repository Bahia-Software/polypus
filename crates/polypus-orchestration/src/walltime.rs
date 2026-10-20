//! Early warning that a training run will outlive its backend's allocation.
//!
//! A backend whose execution units expire (CUNQA's QPUs are released when their
//! SLURM walltime runs out) reports that lifetime through
//! [`QuantumBackend::walltime`](polypus_infrastructure::QuantumBackend::walltime).
//! [`TrainFlow`](crate::TrainFlow) then wraps its oracle in a [`WalltimeWatch`],
//! which times the first generation, projects it over the optimizer's generation
//! budget (for CMA-ES with IPOP restarts, whose batches grow, over the bound
//! [`Method::growth`] gives) and logs one `warn` when the projection exceeds the
//! walltime. Backends
//! without a walltime get the bare oracle, so they pay nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use polypus_optimizers::{CmaEsBatchSchedule, EvaluationOracle};

use crate::dispatch::Method;

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
/// most `generations` of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunShape {
    pub batches_per_generation: u32,
    pub generations: u64,
    pub unit: &'static str,
}

impl Method {
    /// The call pattern of this method's optimizer. DE and PSO score their initial
    /// population in one batch the size of a generation's, so it counts as one;
    /// a QNG iteration is a gradient batch plus a single-candidate batch.
    ///
    /// Every DE strategy makes one batch per generation. L-SHADE spends the
    /// evaluation budget of `generations` full generations in more, smaller
    /// ones; its schedule does not depend on fitness, so
    /// [`DeStrategy::max_generations`](polypus_optimizers::DeStrategy::max_generations)
    /// gives their exact number, from the schedule the optimizer itself steps.
    /// None of its batches is larger than the first, so the projection stays an
    /// upper bound.
    pub(crate) fn run_shape(&self) -> RunShape {
        match self {
            Method::De(cfg) => RunShape {
                batches_per_generation: 1,
                generations: cfg
                    .strategy
                    .max_generations(cfg.population_size, cfg.generations)
                    + 1,
                unit: "generations",
            },
            Method::Pso(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.generations) + 1,
                unit: "generations",
            },
            Method::Qng(cfg, _) => RunShape {
                batches_per_generation: 2,
                generations: u64::from(cfg.max_iters),
                unit: "iterations",
            },
            // One batch per generation and no initial batch; the generations of
            // the first run and of every restart share the budget.
            Method::Cmaes(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.generations),
                unit: "generations",
            },
        }
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
/// batches. Early stopping can only shorten a run, so the projection is an upper
/// bound. After the first generation the watch is one atomic load per batch.
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
    started: Option<Instant>,
    batches: u32,
    /// The projected duration, when the warning fired.
    warned: Option<Duration>,
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

    fn after_batch(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.batches += 1;
        if state.batches < self.shape.batches_per_generation {
            return;
        }
        self.done.store(true, Ordering::Relaxed);
        let Some(started) = state.started else {
            return;
        };
        let first = self.clock.now().saturating_duration_since(started);
        let work = self
            .growth
            .map_or(self.shape.generations, |g| g.first_generations);
        let projected =
            Duration::try_from_secs_f64(first.as_secs_f64() * work as f64).unwrap_or(Duration::MAX);
        if projected > self.walltime {
            state.warned = Some(projected);
            if let Some(growth) = self.growth {
                log::warn!(
                    "run {}: the first of up to {} {} took {}; with up to {} IPOP restarts, \
                     each doubling the population, the optimisation may take up to {}, \
                     beyond the {} walltime; the QPUs are released when it runs out. \
                     Raise `walltime` or lower the {} or the restarts.",
                    self.run_id,
                    self.shape.generations,
                    self.shape.unit,
                    hms(first),
                    growth.restarts,
                    hms(projected),
                    hms(self.walltime),
                    self.shape.unit,
                );
                return;
            }
            log::warn!(
                "run {}: the first of up to {} {} took {}, so the optimisation may take up \
                 to {}, beyond the {} walltime; the QPUs are released when it runs out. \
                 Raise `walltime` or lower the {}.",
                self.run_id,
                self.shape.generations,
                self.shape.unit,
                hms(first),
                hms(projected),
                hms(self.walltime),
                self.shape.unit,
            );
        }
    }
}

impl<C: Clock> EvaluationOracle for WalltimeWatch<C> {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        if self.done.load(Ordering::Relaxed) {
            return self.inner.evaluate_batch(candidates);
        }
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .started
            .get_or_insert_with(|| self.clock.now());
        let scores = self.inner.evaluate_batch(candidates);
        self.after_batch();
        scores
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
    use crate::test_log::logged_for;
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
        unit: "generations",
    };

    #[test]
    fn warns_once_when_the_projection_exceeds_the_walltime() {
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
        let run_id = "walltime-watch-qng";
        let shape = RunShape {
            batches_per_generation: 2,
            generations: 50,
            unit: "iterations",
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
        })
    }

    #[test]
    fn run_shape_follows_the_optimizer_call_pattern() {
        use crate::PsoConfig;
        use polypus_optimizers::DeStrategy;
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade] {
            assert_eq!(de(strategy, 10, 100).run_shape(), DE_100, "{strategy}");
        }
        // L-SHADE spends the same 1010 evaluations in more, smaller generations.
        let lshade = de(DeStrategy::LShade, 10, 100).run_shape();
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
        assert_eq!(pso.run_shape().generations, 8);
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
                });
                let shape = method.run_shape();
                let batches = Arc::new(Mutex::new(0));
                let oracle = Box::new(CountingOracle(Arc::clone(&batches)));
                let outcome =
                    dispatch_optimizer(method, oracle, 2, &OracleErrorSlot::new(), 3).ok();
                let label = format!("{strategy} {population_size} × {generations}");
                assert!(outcome.is_some_and(|o| !o.converged), "{label}");
                assert_eq!(*batches.lock().unwrap(), shape.generations, "{label}");
            }
        }
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
        let outcome =
            dispatch_optimizer(method, oracle, dimensions, &OracleErrorSlot::new(), seed).ok();
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
            let shape = method.run_shape();
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
    /// and the total against `first_generations` first batches. Patience 0
    /// restarts after every generation, the worst case the bound assumes, so
    /// there it is met exactly.
    #[test]
    fn cmaes_growth_bounds_the_batches_with_restarts() {
        use polypus_optimizers::CmaEsBatchSchedule;
        for (population_size, generations, restarts, patience) in [
            (None, 4, 3, 0),
            (Some(5), 3, 2, 0),
            (Some(4), 40, 3, 2),
            (None, 30, 5, 3),
            (Some(6), 10, 4, 4),
        ] {
            let label = format!("{population_size:?} × {generations}, {restarts} restarts");
            let method = cmaes(population_size, generations, restarts, patience, 1.0);
            let shape = method.run_shape();
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
            if patience == 0 && generations == restarts + 1 {
                let worst: Vec<u64> = (0..generations).map(|g| first << g).collect();
                assert_eq!(sizes, worst, "{label}");
                assert_eq!(total, growth.first_generations * first, "{label}");
            }
        }
    }

    /// The watch projects a growing run with its bound, not its generations,
    /// and says why.
    #[test]
    fn a_growing_run_is_projected_with_its_bound() {
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
            method.run_shape(),
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
            method.run_shape(),
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
}
