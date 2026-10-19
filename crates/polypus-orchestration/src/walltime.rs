//! Early warning that a training run will outlive its backend's allocation.
//!
//! A backend whose execution units expire (CUNQA's QPUs are released when their
//! SLURM walltime runs out) reports that lifetime through
//! [`QuantumBackend::walltime`](polypus_infrastructure::QuantumBackend::walltime).
//! [`TrainFlow`](crate::TrainFlow) then wraps its oracle in a [`WalltimeWatch`],
//! which times the first generation, projects it over the optimizer's generation
//! budget and logs one `warn` when the projection exceeds the walltime. Backends
//! without a walltime get the bare oracle, so they pay nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use polypus_optimizers::EvaluationOracle;

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
    /// Every DE strategy makes one batch per generation. L-SHADE's batches shrink
    /// as its population does, so timing the first generation, a full-size one,
    /// keeps the projection an upper bound.
    pub(crate) fn run_shape(&self) -> RunShape {
        match self {
            Method::De(cfg) => RunShape {
                batches_per_generation: 1,
                generations: u64::from(cfg.generations) + 1,
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
        }
    }
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
            run_id: run_id.to_string(),
            clock,
            done: AtomicBool::new(false),
            state: Mutex::new(WatchState::default()),
        }
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
        let projected =
            Duration::try_from_secs_f64(first.as_secs_f64() * self.shape.generations as f64)
                .unwrap_or(Duration::MAX);
        if projected > self.walltime {
            state.warned = Some(projected);
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

    #[test]
    fn run_shape_follows_the_optimizer_call_pattern() {
        use crate::{DeConfig, PsoConfig};
        use polypus_optimizers::DeStrategy;
        for strategy in [DeStrategy::Rand1Bin, DeStrategy::Shade, DeStrategy::LShade] {
            let de = Method::De(DeConfig {
                generations: 100,
                population_size: 10,
                tolerance: 0.0,
                patience: 10,
                strategy,
            });
            assert_eq!(de.run_shape(), DE_100, "{strategy}");
        }
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

    #[test]
    fn hms_matches_the_walltime_format() {
        assert_eq!(hms(Duration::from_secs(36_000)), "10:00:00");
        assert_eq!(hms(Duration::from_secs(172_861)), "48:01:01");
        assert_eq!(hms(Duration::from_millis(500)), "0:00:00");
    }
}
