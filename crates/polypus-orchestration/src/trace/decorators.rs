//! The decorators that record a run's planner batches, backend waves,
//! optimizer iterations and QNG metric calls in its call trace.
//!
//! Each forwards **every** method of its trait to what it wraps, overriding
//! even the methods the trait gives a default. A wrapper that kept a default
//! would run that default instead of the wrapped method, and the traced run
//! would then make different calls from the untraced run: the default
//! `run_shots_distributed`, for one, would call the wrapper's `run_circuits`
//! instead of the wrapped backend's own `run_shots_distributed`, and the
//! default `variance_diagonal` would call the wrapper's `variance` once per
//! element instead of the wrapped oracle's own `variance_diagonal` once.

use std::fmt;
use std::ops::{ControlFlow, Deref};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use polypus_infrastructure::{
    BackendCapabilities, BackendError, BoundCircuit, CancelToken, CircuitTask, Counts,
    InfrastructureError, Planner, PlannerRequirements, QuantumBackend, RunParams,
};
use polypus_observable::CostObservable;
use polypus_optimizers::VarianceOracle;

use super::{BatchOp, CallTracer, TraceEvent, WaveOp};
use crate::progress::{TrainObserver, TrainProgress};

/// A [`QuantumBackend`] that records each call that runs circuits as a `wave`
/// line of a call trace.
///
/// It forwards every method of the trait to the backend it wraps, with the
/// same arguments, and returns what that backend returns, error included. A
/// call to [`run_circuits`](QuantumBackend::run_circuits) or
/// [`run_shots_distributed`](QuantumBackend::run_shots_distributed) is recorded
/// once it returns, with the batch it ran in (`k`), its index among this
/// wrapper's calls that run circuits (`w`, from 0), the circuits and shots it
/// asked for, when it started, how long it took and whether it succeeded. For
/// `run_circuits`, `n` is the number of circuits and `shots` the shots each
/// runs; for `run_shots_distributed`, `n` is the number of replicas, `shots`
/// their total and `replicas` the shots of each. The other methods are
/// forwarded without a record.
///
/// `B` is whatever dereferences to the backend: `&dyn QuantumBackend`,
/// `Arc<dyn QuantumBackend>`, a reference to a concrete backend.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Arc;
///
/// use polypus_infrastructure::{BackendError, BoundCircuit, OptLevel, QuantumBackend, RunParams};
/// use polypus_orchestration::trace::{
///     CallTracer, RunStatus, TraceFlow, TraceHeader, TracingBackend, DEFAULT_MAX_BYTES,
/// };
///
/// /// A backend that measures every shot as `0`.
/// struct Zeros;
///
/// impl QuantumBackend for Zeros {
///     fn run_circuits(
///         &self,
///         qcs: &[BoundCircuit],
///         params: &RunParams,
///     ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
///         let all_zeros = HashMap::from([("0".to_owned(), u64::from(params.shots))]);
///         Ok(vec![all_zeros; qcs.len()])
///     }
/// }
///
/// let name = format!("polypus-traced-backend-doc-{}.jsonl", std::process::id());
/// let path = std::env::temp_dir().join(name);
/// let header = TraceHeader {
///     flow: TraceFlow::RunCircuit,
///     id: "doc".to_owned(),
///     polypus: env!("CARGO_PKG_VERSION").to_owned(),
///     method: None,
///     dims: None,
///     backend: "zeros".to_owned(),
///     infrastructure: "local".to_owned(),
///     n_qpus: 1,
///     shots: 100,
///     seed: None,
/// };
/// let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES)?;
/// // Called outside a planner batch, so the wave's `k` is `null`.
/// let backend = TracingBackend::new(&Zeros, Arc::clone(&tracer), None);
/// let params = RunParams {
///     id: "doc".to_owned(),
///     shots: 100,
///     seed: None,
///     opt_level: OptLevel::default(),
/// };
/// let circuit = BoundCircuit::Qasm2(String::new());
/// let counts = backend.run_circuits(&[circuit], &params)?;
/// assert_eq!(counts, [HashMap::from([("0".to_owned(), 100)])]);
/// tracer.finish(RunStatus::Ok);
/// let trace = std::fs::read_to_string(&path)?;
/// let wave = r#"{"ev":"wave","k":null,"w":0,"op":"run_circuits","n":1,"shots":100,"#;
/// assert!(trace.lines().any(|line| line.starts_with(wave)));
/// std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TracingBackend<B> {
    /// The backend every call goes to.
    inner: B,
    /// The trace the waves are recorded in.
    tracer: Arc<CallTracer>,
    /// The planner batch the calls run in; `None` outside one.
    batch: Option<u64>,
    /// The calls that run circuits started so far, recorded or not: the next
    /// such call's `w`.
    waves: AtomicU32,
}

impl<B> TracingBackend<B> {
    /// Wraps `inner`, recording its waves in `tracer` as waves of the planner
    /// batch `batch`, or as calls outside any batch if `batch` is `None`.
    pub fn new(inner: B, tracer: Arc<CallTracer>, batch: Option<u64>) -> Self {
        Self {
            inner,
            tracer,
            batch,
            waves: AtomicU32::new(0),
        }
    }

    /// Makes `call`, a call that runs circuits, and records it as a wave.
    fn wave<T>(
        &self,
        op: WaveOp,
        n: usize,
        shots: u32,
        replicas: Option<Vec<u32>>,
        call: impl FnOnce() -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let w = self.waves.fetch_add(1, Ordering::Relaxed);
        let t = self.tracer.now_ns();
        let result = call();
        let dt = self.tracer.now_ns().saturating_sub(t);
        self.tracer.record(TraceEvent::Wave {
            k: self.batch,
            w,
            op,
            n,
            shots,
            t,
            dt,
            ok: result.is_ok(),
            replicas,
        });
        result
    }
}

impl<B> QuantumBackend for TracingBackend<B>
where
    B: Deref + Send + Sync,
    B::Target: QuantumBackend,
{
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        self.wave(WaveOp::RunCircuits, qcs.len(), params.shots, None, || {
            self.inner.run_circuits(qcs, params)
        })
    }

    fn run_shots_distributed(
        &self,
        qc: &BoundCircuit,
        shot_batches: &[u32],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        // The circuit's shots asked for in this call: those of every replica.
        let shots = shot_batches
            .iter()
            .fold(0u32, |total, &replica| total.saturating_add(replica));
        self.wave(
            WaveOp::RunShotsDistributed,
            shot_batches.len(),
            shots,
            Some(shot_batches.to_vec()),
            || self.inner.run_shots_distributed(qc, shot_batches, params),
        )
    }

    fn close(&self) {
        self.inner.close();
    }

    fn cancel(&self) {
        self.inner.cancel();
    }

    fn wants_cancel_watcher(&self) -> bool {
        self.inner.wants_cancel_watcher()
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.inner.capabilities()
    }

    fn capabilities_for(
        &self,
        tasks: &[CircuitTask<'_>],
    ) -> Result<BackendCapabilities, InfrastructureError> {
        self.inner.capabilities_for(tasks)
    }

    fn default_planner(&self) -> Arc<dyn Planner> {
        self.inner.default_planner()
    }

    fn replica_count(&self) -> Option<u32> {
        self.inner.replica_count()
    }

    fn walltime(&self) -> Option<Duration> {
        self.inner.walltime()
    }
}

impl<B> fmt::Debug for TracingBackend<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingBackend")
            .field("tracer", &self.tracer)
            .field("batch", &self.batch)
            .field("waves", &self.waves)
            .finish_non_exhaustive()
    }
}

/// A [`Planner`] that records each batch it runs as a `batch` line of a call
/// trace, and each call the batch makes to its backend as a `wave` line.
///
/// It forwards every method of the trait to the planner it wraps, with the same
/// arguments, and returns what that planner returns, error included. A call to
/// [`execute`](Planner::execute), [`execute_replicas`](Planner::execute_replicas)
/// or [`evaluate`](Planner::evaluate) is a batch: it takes the trace's next
/// batch index `k`, hands the wrapped planner its backend wrapped in a
/// [`TracingBackend`] for batch `k`, and once the planner returns records the
/// batch, with its number of tasks, the shots of its first task (the run's
/// shots if it has none), when it started, how long it took and whether it
/// succeeded. A batch's waves are recorded before it. [`requirements`](Planner::requirements) and
/// [`required_replicas`](Planner::required_replicas) are forwarded without a
/// record.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Arc;
///
/// use polypus_infrastructure::{
///     BackendError, BoundCircuit, CancelToken, CircuitTask, OptLevel, Planner, QuantumBackend,
///     RunParams, SequentialPlanner,
/// };
/// use polypus_orchestration::trace::{
///     CallTracer, RunStatus, TraceFlow, TraceHeader, TracingPlanner, DEFAULT_MAX_BYTES,
/// };
///
/// /// A backend that measures every shot as `0`.
/// struct Zeros;
///
/// impl QuantumBackend for Zeros {
///     fn run_circuits(
///         &self,
///         qcs: &[BoundCircuit],
///         params: &RunParams,
///     ) -> Result<Vec<HashMap<String, u64>>, BackendError> {
///         let all_zeros = HashMap::from([("0".to_owned(), u64::from(params.shots))]);
///         Ok(vec![all_zeros; qcs.len()])
///     }
/// }
///
/// let name = format!("polypus-traced-planner-doc-{}.jsonl", std::process::id());
/// let path = std::env::temp_dir().join(name);
/// let header = TraceHeader {
///     flow: TraceFlow::RunCircuit,
///     id: "doc".to_owned(),
///     polypus: env!("CARGO_PKG_VERSION").to_owned(),
///     method: None,
///     dims: None,
///     backend: "zeros".to_owned(),
///     infrastructure: "local".to_owned(),
///     n_qpus: 1,
///     shots: 100,
///     seed: None,
/// };
/// let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES)?;
/// let planner = TracingPlanner::new(Arc::new(SequentialPlanner), Arc::clone(&tracer));
/// let params = RunParams {
///     id: "doc".to_owned(),
///     shots: 100,
///     seed: None,
///     opt_level: OptLevel::default(),
/// };
/// let circuit = BoundCircuit::Qasm2(String::new());
/// let tasks = [CircuitTask { circuit: &circuit, shots: 100 }];
/// let counts = planner.execute(&Zeros, &tasks, &params, &CancelToken::default())?;
/// assert_eq!(counts, [HashMap::from([("0".to_owned(), 100)])]);
/// tracer.finish(RunStatus::Ok);
/// let trace = std::fs::read_to_string(&path)?;
/// let lines: Vec<&str> = trace.lines().collect();
/// // The header, the batch's one wave, the batch and the end.
/// assert_eq!(lines.len(), 4);
/// assert!(lines[1].starts_with(r#"{"ev":"wave","k":0,"w":0,"op":"run_circuits","n":1,"#));
/// assert!(lines[2].starts_with(r#"{"ev":"batch","k":0,"op":"execute","n":1,"shots":100,"#));
/// std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TracingPlanner {
    /// The planner every call goes to.
    inner: Arc<dyn Planner>,
    /// The trace the batches and their waves are recorded in.
    tracer: Arc<CallTracer>,
}

impl TracingPlanner {
    /// Wraps `inner`, recording its batches, and the waves they run, in
    /// `tracer`.
    pub fn new(inner: Arc<dyn Planner>, tracer: Arc<CallTracer>) -> Self {
        Self { inner, tracer }
    }

    /// Runs a batch: makes `call`, a call to the wrapped planner, with
    /// `backend` wrapped to record the batch's waves, then records the batch.
    fn batch<T>(
        &self,
        op: BatchOp,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        call: impl FnOnce(&dyn QuantumBackend) -> Result<T, InfrastructureError>,
    ) -> Result<T, InfrastructureError> {
        let k = self.tracer.begin_batch();
        let t = self.tracer.now_ns();
        let traced = TracingBackend::new(backend, Arc::clone(&self.tracer), Some(k));
        let result = call(&traced);
        let dt = self.tracer.now_ns().saturating_sub(t);
        self.tracer.record(TraceEvent::Batch {
            k,
            op,
            n: tasks.len(),
            shots: tasks.first().map_or(params.shots, |task| task.shots),
            t,
            dt,
            ok: result.is_ok(),
        });
        result
    }
}

impl Planner for TracingPlanner {
    fn requirements(&self) -> PlannerRequirements {
        self.inner.requirements()
    }

    fn required_replicas(&self) -> Option<u32> {
        self.inner.required_replicas()
    }

    fn execute(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        self.batch(BatchOp::Execute, backend, tasks, params, |traced| {
            self.inner.execute(traced, tasks, params, cancel)
        })
    }

    fn execute_replicas(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<Counts>, InfrastructureError> {
        self.batch(BatchOp::ExecuteReplicas, backend, tasks, params, |traced| {
            self.inner.execute_replicas(traced, tasks, params, cancel)
        })
    }

    fn evaluate(
        &self,
        backend: &dyn QuantumBackend,
        tasks: &[CircuitTask<'_>],
        reducer: &dyn CostObservable,
        params: &RunParams,
        cancel: &CancelToken,
    ) -> Result<Vec<f64>, InfrastructureError> {
        self.batch(BatchOp::Evaluate, backend, tasks, params, |traced| {
            self.inner.evaluate(traced, tasks, reducer, params, cancel)
        })
    }
}

impl fmt::Debug for TracingPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingPlanner")
            .field("tracer", &self.tracer)
            .finish_non_exhaustive()
    }
}

/// A [`TrainObserver`] that records each iteration of a training run as an
/// `iter` line of a call trace, and hands it on to the run's own observer, if
/// it has one.
///
/// Each call to [`observe`](TrainObserver::observe) records the iteration's
/// number (`i`), the evaluations spent so far (`evals`) and the best fitness
/// so far (`best`), with the batches recorded until then, before the run's
/// observer is called: its `t` is when the optimizer reported the iteration,
/// however long that observer takes. It then shows the run's observer the same
/// progress and returns its answer, or [`ControlFlow::Continue`] without one:
/// whether the run stops is up to the run's observer alone, as it is untraced.
///
/// # Example
///
/// ```
/// use std::ops::ControlFlow;
/// use std::sync::Arc;
///
/// use polypus_orchestration::trace::{
///     CallTracer, RunStatus, TraceFlow, TraceHeader, TracingObserver, DEFAULT_MAX_BYTES,
/// };
/// use polypus_orchestration::{TrainObserver, TrainProgress};
///
/// let name = format!("polypus-traced-observer-doc-{}.jsonl", std::process::id());
/// let path = std::env::temp_dir().join(name);
/// let header = TraceHeader {
///     flow: TraceFlow::Train,
///     id: "doc".to_owned(),
///     polypus: env!("CARGO_PKG_VERSION").to_owned(),
///     method: Some("spsa"),
///     dims: Some(2),
///     backend: "native".to_owned(),
///     infrastructure: "local".to_owned(),
///     n_qpus: 1,
///     shots: 1024,
///     seed: Some(7),
/// };
/// let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES)?;
/// // The run's own observer: it stops the run once the best fitness passes -0.5.
/// let stop_when_good = |progress: &TrainProgress<'_>| {
///     if progress.best_fitness > -0.5 {
///         ControlFlow::Break(())
///     } else {
///         ControlFlow::Continue(())
///     }
/// };
/// let mut observer = TracingObserver::new(Some(Box::new(stop_when_good)), Arc::clone(&tracer));
/// let best = [0.5, -1.25];
/// let progress = TrainProgress {
///     iteration: 1,
///     best_fitness: -0.25,
///     best_params: &best,
///     evaluations: 12,
/// };
/// assert_eq!(observer.observe(&progress), ControlFlow::Break(()));
/// tracer.finish(RunStatus::Ok);
/// let trace = std::fs::read_to_string(&path)?;
/// // No batch was recorded before the iteration.
/// let iter = r#"{"ev":"iter","i":1,"evals":12,"batches":0,"best":-0.25,"t":"#;
/// assert!(trace.lines().any(|line| line.starts_with(iter)));
/// std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TracingObserver {
    /// The run's own observer, if it has one.
    inner: Option<Box<dyn TrainObserver>>,
    /// The trace the iterations are recorded in.
    tracer: Arc<CallTracer>,
}

impl TracingObserver {
    /// Records each iteration in `tracer`, then hands it on to `inner`, the
    /// run's own observer, if it has one.
    pub fn new(inner: Option<Box<dyn TrainObserver>>, tracer: Arc<CallTracer>) -> Self {
        Self { inner, tracer }
    }
}

impl TrainObserver for TracingObserver {
    fn observe(&mut self, progress: &TrainProgress<'_>) -> ControlFlow<()> {
        // Stamped and recorded before the run's observer is called, so that
        // observer's time is not the iteration's.
        let t = self.tracer.now_ns();
        self.tracer.record(TraceEvent::Iter {
            i: progress.iteration,
            evals: progress.evaluations,
            best: progress.best_fitness,
            t,
        });
        match self.inner.as_mut() {
            Some(inner) => inner.observe(progress),
            None => ControlFlow::Continue(()),
        }
    }
}

impl fmt::Debug for TracingObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingObserver")
            .field("tracer", &self.tracer)
            .finish_non_exhaustive()
    }
}

/// A [`VarianceOracle`] that records each call to the QNG metric as a `metric`
/// line of a call trace.
///
/// It forwards both methods of the trait to the oracle it wraps, with the same
/// arguments, and returns what that oracle returns. Each call is recorded once
/// it returns, with the metric elements it asked for (`n`: 1 for
/// [`variance`](VarianceOracle::variance), `dims` for
/// [`variance_diagonal`](VarianceOracle::variance_diagonal)), when it started
/// and how long it took. `variance_diagonal` goes to the wrapped oracle's own
/// `variance_diagonal`, never to the trait's default loop over `variance`: an
/// oracle that computes the whole diagonal at once is still called once, and
/// the call is one line.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
///
/// use polypus_optimizers::VarianceOracle;
/// use polypus_orchestration::trace::{
///     CallTracer, RunStatus, TraceFlow, TraceHeader, TracingVariance, DEFAULT_MAX_BYTES,
/// };
///
/// /// A metric whose every diagonal element is 0.25.
/// struct Flat;
///
/// impl VarianceOracle for Flat {
///     fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
///         0.25
///     }
/// }
///
/// let name = format!("polypus-traced-variance-doc-{}.jsonl", std::process::id());
/// let path = std::env::temp_dir().join(name);
/// let header = TraceHeader {
///     flow: TraceFlow::Train,
///     id: "doc".to_owned(),
///     polypus: env!("CARGO_PKG_VERSION").to_owned(),
///     method: Some("qng"),
///     dims: Some(2),
///     backend: "native".to_owned(),
///     infrastructure: "local".to_owned(),
///     n_qpus: 1,
///     shots: 1024,
///     seed: Some(7),
/// };
/// let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES)?;
/// let metric = TracingVariance::new(Box::new(Flat), Arc::clone(&tracer));
/// assert_eq!(metric.variance_diagonal(&[0.1, 0.2], 2), [0.25, 0.25]);
/// tracer.finish(RunStatus::Ok);
/// let trace = std::fs::read_to_string(&path)?;
/// // One call for the whole diagonal: one line, for its two elements.
/// let metrics: Vec<&str> = trace
///     .lines()
///     .filter(|line| line.contains(r#""ev":"metric""#))
///     .collect();
/// assert_eq!(metrics.len(), 1);
/// assert!(metrics[0].starts_with(r#"{"ev":"metric","n":2,"t":"#));
/// std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TracingVariance {
    /// The oracle every call goes to.
    inner: Box<dyn VarianceOracle>,
    /// The trace the calls are recorded in.
    tracer: Arc<CallTracer>,
}

impl TracingVariance {
    /// Wraps `inner`, recording its calls in `tracer`.
    pub fn new(inner: Box<dyn VarianceOracle>, tracer: Arc<CallTracer>) -> Self {
        Self { inner, tracer }
    }

    /// Makes `call`, a call to the wrapped oracle that asks for `n` metric
    /// elements, and records it.
    fn metric<T>(&self, n: usize, call: impl FnOnce() -> T) -> T {
        let t = self.tracer.now_ns();
        let result = call();
        let dt = self.tracer.now_ns().saturating_sub(t);
        self.tracer.record(TraceEvent::Metric { n, t, dt });
        result
    }
}

impl VarianceOracle for TracingVariance {
    fn variance(&self, theta: &[f64], param_index: usize) -> f64 {
        self.metric(1, || self.inner.variance(theta, param_index))
    }

    fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
        self.metric(dims, || self.inner.variance_diagonal(theta, dims))
    }
}

impl fmt::Debug for TracingVariance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingVariance")
            .field("tracer", &self.tracer)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;

    use polypus_infrastructure::{Interrupt, SequentialPlanner, ShotDistributingPlanner};
    use serde_json::{json, Value};

    use super::*;
    use crate::trace::test_support::{params, sink_tracer, HashBackend, KeyOneShare, SharedSink};
    use crate::trace::RunStatus;

    fn counts(key: &str, value: u64) -> Counts {
        HashMap::from([(key.to_string(), value)])
    }

    /// A backend that notes the name of every method called on it and answers
    /// each with a value that is not the trait's default.
    struct SpyBackend {
        calls: Mutex<Vec<&'static str>>,
        /// What `default_planner` hands out.
        planner: Arc<dyn Planner>,
    }

    impl SpyBackend {
        fn new() -> Self {
            Self {
                calls: Mutex::default(),
                planner: Arc::new(SequentialPlanner),
            }
        }

        fn note(&self, method: &'static str) {
            self.calls.lock().unwrap().push(method);
        }

        /// The methods called so far, in order.
        fn calls(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl QuantumBackend for SpyBackend {
        fn run_circuits(
            &self,
            qcs: &[BoundCircuit],
            _params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            self.note("run_circuits");
            Ok(vec![counts("rc", 1); qcs.len()])
        }

        fn run_shots_distributed(
            &self,
            _qc: &BoundCircuit,
            shot_batches: &[u32],
            _params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            self.note("run_shots_distributed");
            Ok(shot_batches
                .iter()
                .map(|&shots| counts("sd", u64::from(shots)))
                .collect())
        }

        fn close(&self) {
            self.note("close");
        }

        fn cancel(&self) {
            self.note("cancel");
        }

        fn wants_cancel_watcher(&self) -> bool {
            self.note("wants_cancel_watcher");
            true
        }

        fn capabilities(&self) -> BackendCapabilities {
            self.note("capabilities");
            BackendCapabilities {
                max_concurrency: 3,
                supports_shot_distribution: false,
            }
        }

        fn capabilities_for(
            &self,
            _tasks: &[CircuitTask<'_>],
        ) -> Result<BackendCapabilities, InfrastructureError> {
            self.note("capabilities_for");
            Ok(BackendCapabilities {
                max_concurrency: 2,
                supports_shot_distribution: false,
            })
        }

        fn default_planner(&self) -> Arc<dyn Planner> {
            self.note("default_planner");
            Arc::clone(&self.planner)
        }

        fn replica_count(&self) -> Option<u32> {
            self.note("replica_count");
            Some(7)
        }

        fn walltime(&self) -> Option<Duration> {
            self.note("walltime");
            Some(Duration::from_secs(42))
        }
    }

    /// A planner that notes the name of every method called on it, calls its
    /// backend once from each method that runs a batch, and answers each with a
    /// value that is not the trait's default.
    #[derive(Default)]
    struct SpyPlanner {
        calls: Mutex<Vec<&'static str>>,
    }

    impl SpyPlanner {
        fn note(&self, method: &'static str) {
            self.calls.lock().unwrap().push(method);
        }

        /// The methods called so far, in order.
        fn calls(&self) -> Vec<&'static str> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Planner for SpyPlanner {
        fn requirements(&self) -> PlannerRequirements {
            self.note("requirements");
            PlannerRequirements {
                needs_shot_distribution: true,
                min_concurrency: 3,
            }
        }

        fn required_replicas(&self) -> Option<u32> {
            self.note("required_replicas");
            Some(5)
        }

        fn execute(
            &self,
            backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            params: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            self.note("execute");
            backend.run_circuits(&[tasks[0].circuit.duplicate()], params)?;
            Ok(vec![counts("ex", 1)])
        }

        fn execute_replicas(
            &self,
            backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            params: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<Counts>, InfrastructureError> {
            self.note("execute_replicas");
            backend.run_shots_distributed(tasks[0].circuit, &[40, 24], params)?;
            Ok(vec![counts("er", 1), counts("er", 2)])
        }

        fn evaluate(
            &self,
            backend: &dyn QuantumBackend,
            tasks: &[CircuitTask<'_>],
            _reducer: &dyn CostObservable,
            params: &RunParams,
            _cancel: &CancelToken,
        ) -> Result<Vec<f64>, InfrastructureError> {
            self.note("evaluate");
            backend.run_circuits(&[tasks[0].circuit.duplicate()], params)?;
            Ok(vec![0.25, 0.75])
        }
    }

    /// A backend whose every call fails.
    struct FailingBackend;

    impl QuantumBackend for FailingBackend {
        fn run_circuits(
            &self,
            _qcs: &[BoundCircuit],
            _params: &RunParams,
        ) -> Result<Vec<Counts>, BackendError> {
            Err(BackendError::Conversion("boom".into()))
        }
    }

    /// An interrupt guard that always reports a pending interrupt.
    struct Interrupted;

    impl Interrupt for Interrupted {
        fn poll(&self) -> Result<(), BackendError> {
            Err(BackendError::Aborted("interrupted".into()))
        }
    }

    #[test]
    fn tracing_backend_forwards_every_method() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "backend-forwarding");
        let spy = Arc::new(SpyBackend::new());
        let traced = TracingBackend::new(Arc::clone(&spy), Arc::clone(&tracer), Some(5));
        let qc = BoundCircuit::Qasm2("a".into());
        assert_eq!(
            traced
                .run_circuits(&[qc.duplicate(), qc.duplicate()], &params())
                .unwrap(),
            vec![counts("rc", 1); 2]
        );
        assert_eq!(
            traced
                .run_shots_distributed(&qc, &[4, 3, 3], &params())
                .unwrap(),
            [counts("sd", 4), counts("sd", 3), counts("sd", 3)]
        );
        traced.close();
        traced.cancel();
        assert!(traced.wants_cancel_watcher());
        let caps = traced.capabilities();
        assert_eq!(
            (caps.max_concurrency, caps.supports_shot_distribution),
            (3, false)
        );
        let caps = traced.capabilities_for(&[]).unwrap();
        assert_eq!(
            (caps.max_concurrency, caps.supports_shot_distribution),
            (2, false)
        );
        assert!(Arc::ptr_eq(&traced.default_planner(), &spy.planner));
        assert_eq!(traced.replica_count(), Some(7));
        assert_eq!(traced.walltime(), Some(Duration::from_secs(42)));
        // One call per method, and no `run_circuits` behind `run_shots_distributed`,
        // which the trait's default implementation would make.
        assert_eq!(
            spy.calls(),
            [
                "run_circuits",
                "run_shots_distributed",
                "close",
                "cancel",
                "wants_cancel_watcher",
                "capabilities",
                "capabilities_for",
                "default_planner",
                "replica_count",
                "walltime",
            ]
        );
        // The trace reaches the sink once it is closed.
        tracer.finish(RunStatus::Ok);
        let lines = sink.lines();
        let waves: Vec<&Value> = lines.iter().filter(|l| l["ev"] == "wave").collect();
        assert_eq!(waves.len(), 2);
        assert_eq!(
            [
                &waves[0]["k"],
                &waves[0]["w"],
                &waves[0]["op"],
                &waves[0]["n"],
                &waves[0]["shots"]
            ],
            [
                &json!(5),
                &json!(0),
                &json!("run_circuits"),
                &json!(2),
                &json!(500)
            ]
        );
        assert!(waves[0].get("replicas").is_none());
        assert_eq!(
            [
                &waves[1]["w"],
                &waves[1]["op"],
                &waves[1]["n"],
                &waves[1]["shots"],
                &waves[1]["replicas"]
            ],
            [
                &json!(1),
                &json!("run_shots_distributed"),
                &json!(3),
                &json!(10),
                &json!([4, 3, 3])
            ]
        );
    }

    #[test]
    fn tracing_planner_forwards_every_method() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "planner-forwarding");
        let spy = Arc::new(SpyPlanner::default());
        let traced = TracingPlanner::new(spy.clone(), Arc::clone(&tracer));
        let backend = SpyBackend::new();
        let qc = BoundCircuit::Qasm2("a".into());
        let tasks = [
            CircuitTask {
                circuit: &qc,
                shots: 64,
            },
            CircuitTask {
                circuit: &qc,
                shots: 64,
            },
        ];
        let cancel = CancelToken::default();
        let req = traced.requirements();
        assert_eq!(
            (req.needs_shot_distribution, req.min_concurrency),
            (true, 3)
        );
        assert_eq!(traced.required_replicas(), Some(5));
        assert_eq!(
            traced
                .execute(&backend, &tasks, &params(), &cancel)
                .unwrap(),
            [counts("ex", 1)]
        );
        assert_eq!(
            traced
                .execute_replicas(&backend, &tasks, &params(), &cancel)
                .unwrap(),
            [counts("er", 1), counts("er", 2)]
        );
        assert_eq!(
            traced
                .evaluate(&backend, &tasks, &KeyOneShare, &params(), &cancel)
                .unwrap(),
            [0.25, 0.75]
        );
        assert_eq!(
            spy.calls(),
            [
                "requirements",
                "required_replicas",
                "execute",
                "execute_replicas",
                "evaluate"
            ]
        );
        assert_eq!(
            backend.calls(),
            ["run_circuits", "run_shots_distributed", "run_circuits"]
        );
        tracer.finish(RunStatus::Ok);
        let seen: Vec<(String, u64, String)> = sink
            .lines()
            .iter()
            .filter(|l| l["ev"] == "batch" || l["ev"] == "wave")
            .map(|l| {
                (
                    l["ev"].as_str().unwrap().into(),
                    l["k"].as_u64().unwrap(),
                    l["op"].as_str().unwrap().into(),
                )
            })
            .collect();
        let expected = [
            ("wave", 0, "run_circuits"),
            ("batch", 0, "execute"),
            ("wave", 1, "run_shots_distributed"),
            ("batch", 1, "execute_replicas"),
            ("wave", 2, "run_circuits"),
            ("batch", 2, "evaluate"),
        ];
        assert_eq!(
            seen,
            expected.map(|(e, k, o)| (e.to_string(), k, o.to_string()))
        );
        for line in sink.lines().iter().filter(|l| l["ev"] == "batch") {
            assert_eq!(
                (
                    line["n"].as_u64(),
                    line["shots"].as_u64(),
                    line["ok"].as_bool()
                ),
                (Some(2), Some(64), Some(true))
            );
        }
    }

    #[test]
    fn planner_trace_records_every_batch() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "batches");
        let planner = TracingPlanner::new(Arc::new(SequentialPlanner), Arc::clone(&tracer));
        let backend = HashBackend {
            max_concurrency: 16,
        };
        let circuits: Vec<BoundCircuit> = (0..30)
            .map(|i| BoundCircuit::Qasm2(format!("c{i}")))
            .collect();
        let tasks = |n: usize| -> Vec<CircuitTask<'_>> {
            circuits[..n]
                .iter()
                .map(|circuit| CircuitTask {
                    circuit,
                    shots: 500,
                })
                .collect()
        };
        let cancel = CancelToken::default();
        planner
            .evaluate(&backend, &tasks(30), &KeyOneShare, &params(), &cancel)
            .unwrap();
        planner
            .execute(&backend, &tasks(16), &params(), &cancel)
            .unwrap();
        planner
            .execute_replicas(&backend, &tasks(1), &params(), &cancel)
            .unwrap();
        let err = planner
            .execute(&FailingBackend, &tasks(1), &params(), &cancel)
            .unwrap_err();
        assert!(
            matches!(
                err,
                InfrastructureError::Backend(BackendError::Conversion(_))
            ),
            "{err:?}"
        );
        tracer.finish(RunStatus::Ok);
        let lines = sink.lines();
        let field = |l: &Value, f: &str| l[f].as_u64().unwrap();
        let batches: Vec<&Value> = lines.iter().filter(|l| l["ev"] == "batch").collect();
        let waves: Vec<&Value> = lines.iter().filter(|l| l["ev"] == "wave").collect();
        assert_eq!(
            batches
                .iter()
                .map(|b| (
                    field(b, "k"),
                    b["op"].as_str().unwrap(),
                    field(b, "n"),
                    b["ok"].as_bool().unwrap()
                ))
                .collect::<Vec<_>>(),
            [
                (0, "evaluate", 30, true),
                (1, "execute", 16, true),
                (2, "execute_replicas", 1, true),
                (3, "execute", 1, false)
            ]
        );
        assert_eq!(
            waves
                .iter()
                .map(|w| (
                    field(w, "k"),
                    field(w, "w"),
                    field(w, "n"),
                    w["ok"].as_bool().unwrap()
                ))
                .collect::<Vec<_>>(),
            [
                (0, 0, 16, true),
                (0, 1, 14, true),
                (1, 0, 16, true),
                (2, 0, 1, true),
                (3, 0, 1, false)
            ]
        );
        for wave in &waves {
            let batch = batches.iter().find(|b| b["k"] == wave["k"]).unwrap();
            assert!(field(batch, "t") <= field(wave, "t"));
            assert!(field(wave, "t") + field(wave, "dt") <= field(batch, "t") + field(batch, "dt"));
        }
    }

    #[test]
    fn a_backend_wrapped_outside_a_batch_has_no_k() {
        let sink = SharedSink::default();
        let backend = HashBackend { max_concurrency: 4 };
        let tracer = sink_tracer(&sink, "no-batch");
        let traced = TracingBackend::new(&backend, Arc::clone(&tracer), None);
        let qc = BoundCircuit::Qasm2("a".into());
        traced.run_circuits(&[qc.duplicate()], &params()).unwrap();
        traced.run_circuits(&[qc.duplicate()], &params()).unwrap();
        tracer.finish(RunStatus::Ok);
        let waves: Vec<(Value, Value)> = sink
            .lines()
            .into_iter()
            .filter(|l| l["ev"] == "wave")
            .map(|l| (l["k"].clone(), l["w"].clone()))
            .collect();
        assert_eq!(waves, [(Value::Null, json!(0)), (Value::Null, json!(1))]);
    }

    #[test]
    fn traced_calls_return_what_untraced_calls_return() {
        // `HashBackend`'s counts depend on each circuit, its shots and the run's
        // seed, and `KeyOneShare` reduces them: a wrapper that handed on other
        // circuits, shots, seed or reducer would return other results.
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "same-results");
        let backend = HashBackend {
            max_concurrency: 16,
        };
        let run = params();
        let cancel = CancelToken::default();
        let circuits: Vec<BoundCircuit> = (0..30)
            .map(|i| BoundCircuit::Qasm2(format!("c{i}")))
            .collect();
        let traced_backend = TracingBackend::new(&backend, Arc::clone(&tracer), None);
        assert_eq!(
            traced_backend.run_circuits(&circuits, &run).unwrap(),
            backend.run_circuits(&circuits, &run).unwrap()
        );
        assert_eq!(
            traced_backend
                .run_shots_distributed(&circuits[0], &[167, 167, 166], &run)
                .unwrap(),
            backend
                .run_shots_distributed(&circuits[0], &[167, 167, 166], &run)
                .unwrap()
        );
        // Tasks with other shots than the run's, through a planner that runs
        // them in waves of `run_circuits` calls and through one that splits a
        // task's shots with `run_shots_distributed`.
        let tasks: Vec<CircuitTask<'_>> = circuits
            .iter()
            .map(|circuit| CircuitTask {
                circuit,
                shots: 300,
            })
            .collect();
        let planners: [(Arc<dyn Planner>, &[CircuitTask<'_>]); 2] = [
            (Arc::new(SequentialPlanner), &tasks),
            (Arc::new(ShotDistributingPlanner::new(3)), &tasks[..1]),
        ];
        for (plain, batch) in planners {
            let traced = TracingPlanner::new(Arc::clone(&plain), Arc::clone(&tracer));
            assert_eq!(
                traced.execute(&backend, batch, &run, &cancel).unwrap(),
                plain.execute(&backend, batch, &run, &cancel).unwrap()
            );
            assert_eq!(
                traced
                    .execute_replicas(&backend, batch, &run, &cancel)
                    .unwrap(),
                plain
                    .execute_replicas(&backend, batch, &run, &cancel)
                    .unwrap()
            );
            assert_eq!(
                traced
                    .evaluate(&backend, batch, &KeyOneShare, &run, &cancel)
                    .unwrap(),
                plain
                    .evaluate(&backend, batch, &KeyOneShare, &run, &cancel)
                    .unwrap()
            );
        }
    }

    #[test]
    fn tracing_planner_passes_on_the_callers_cancel_token() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "cancel-token");
        let backend = HashBackend {
            max_concurrency: 16,
        };
        let qc = BoundCircuit::Qasm2("a".into());
        let tasks = [CircuitTask {
            circuit: &qc,
            shots: 500,
        }];
        // Already cancelled: the wrapped planner stops before its first wave
        // with `Cancelled`, as it does untraced.
        let cancelled = CancelToken::default();
        cancelled.cancel();
        // Its interrupt guard reports an interrupt: the wrapped planner stops
        // after its first wave with the guard's error, as it does untraced.
        let interrupted = CancelToken::with_interrupt(Arc::new(Interrupted));
        let planners: [Arc<dyn Planner>; 2] = [
            Arc::new(SequentialPlanner),
            Arc::new(ShotDistributingPlanner::new(3)),
        ];
        for planner in planners {
            let traced = TracingPlanner::new(planner, Arc::clone(&tracer));
            let errors = |cancel: &CancelToken| {
                [
                    traced.execute(&backend, &tasks, &params(), cancel).err(),
                    traced
                        .execute_replicas(&backend, &tasks, &params(), cancel)
                        .err(),
                    traced
                        .evaluate(&backend, &tasks, &KeyOneShare, &params(), cancel)
                        .err(),
                ]
            };
            for err in errors(&cancelled) {
                assert!(
                    matches!(err, Some(InfrastructureError::Cancelled)),
                    "{err:?}"
                );
            }
            for err in errors(&interrupted) {
                assert!(
                    matches!(
                        err,
                        Some(InfrastructureError::Backend(BackendError::Aborted(_)))
                    ),
                    "{err:?}"
                );
            }
        }
    }

    #[test]
    fn tracing_observer_forwards_every_call() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "observer-forwarding");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let inner = move |p: &TrainProgress<'_>| {
            record.lock().unwrap().push((
                p.iteration,
                p.best_fitness,
                p.best_params.to_vec(),
                p.evaluations,
            ));
            if p.iteration == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let mut traced = TracingObserver::new(Some(Box::new(inner)), Arc::clone(&tracer));
        let best = [0.5, -1.25];
        let first = TrainProgress {
            iteration: 1,
            best_fitness: -0.75,
            best_params: &best,
            evaluations: 12,
        };
        let second = TrainProgress {
            iteration: 2,
            best_fitness: -0.5,
            best_params: &best,
            evaluations: 18,
        };
        assert_eq!(traced.observe(&first), ControlFlow::Continue(()));
        assert_eq!(traced.observe(&second), ControlFlow::Break(()));
        assert_eq!(
            *seen.lock().unwrap(),
            [(1, -0.75, best.to_vec(), 12), (2, -0.5, best.to_vec(), 18)]
        );
        // The trace reaches the sink once it is closed.
        tracer.finish(RunStatus::Ok);
        let iters: Vec<(u64, u64, f64)> = sink
            .lines()
            .iter()
            .filter(|l| l["ev"] == "iter")
            .map(|l| {
                (
                    l["i"].as_u64().unwrap(),
                    l["evals"].as_u64().unwrap(),
                    l["best"].as_f64().unwrap(),
                )
            })
            .collect();
        assert_eq!(iters, [(1, 12, -0.75), (2, 18, -0.5)]);

        // Without an observer of the run's own, the iteration is still
        // recorded and the run goes on.
        let alone_sink = SharedSink::default();
        let alone_tracer = sink_tracer(&alone_sink, "observer-alone");
        let mut alone = TracingObserver::new(None, Arc::clone(&alone_tracer));
        assert_eq!(alone.observe(&first), ControlFlow::Continue(()));
        alone_tracer.finish(RunStatus::Ok);
        assert_eq!(
            alone_sink
                .lines()
                .iter()
                .filter(|l| l["ev"] == "iter")
                .count(),
            1
        );
    }

    #[test]
    fn tracing_variance_forwards_both_methods() {
        /// The arguments of every call to one method of the spy, in order:
        /// `theta` and the parameter index or the number of dimensions.
        type Calls = Arc<Mutex<Vec<(Vec<f64>, usize)>>>;

        /// A variance oracle that counts the calls to each of its methods and
        /// notes the arguments each call received.
        struct SpyVariance {
            variance: Arc<AtomicUsize>,
            diagonal: Arc<AtomicUsize>,
            variance_args: Calls,
            diagonal_args: Calls,
        }

        impl VarianceOracle for SpyVariance {
            fn variance(&self, theta: &[f64], param_index: usize) -> f64 {
                self.variance.fetch_add(1, Ordering::SeqCst);
                self.variance_args
                    .lock()
                    .unwrap()
                    .push((theta.to_vec(), param_index));
                7.25
            }

            fn variance_diagonal(&self, theta: &[f64], dims: usize) -> Vec<f64> {
                self.diagonal.fetch_add(1, Ordering::SeqCst);
                self.diagonal_args
                    .lock()
                    .unwrap()
                    .push((theta.to_vec(), dims));
                (0..dims).map(|a| a as f64 + 0.5).collect()
            }
        }

        let (variance, diagonal) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (variance_args, diagonal_args) = (Calls::default(), Calls::default());
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "variance-forwarding");
        let spy = SpyVariance {
            variance: Arc::clone(&variance),
            diagonal: Arc::clone(&diagonal),
            variance_args: Arc::clone(&variance_args),
            diagonal_args: Arc::clone(&diagonal_args),
        };
        let traced = TracingVariance::new(Box::new(spy), Arc::clone(&tracer));
        assert_eq!(
            traced.variance_diagonal(&[0.1, 0.2, 0.3], 3),
            [0.5, 1.5, 2.5]
        );
        // One call to the oracle's own `variance_diagonal`, and none to
        // `variance`, which the trait's default implementation would make.
        assert_eq!(
            (
                variance.load(Ordering::SeqCst),
                diagonal.load(Ordering::SeqCst)
            ),
            (0, 1)
        );
        assert_eq!(traced.variance(&[0.1, 0.2, 0.3], 1), 7.25);
        assert_eq!(
            (
                variance.load(Ordering::SeqCst),
                diagonal.load(Ordering::SeqCst)
            ),
            (1, 1)
        );
        // Each call reached the oracle with the arguments it was made with: a
        // wrong index or a cut `theta` changes neither the counts nor what
        // this oracle returns.
        assert_eq!(*diagonal_args.lock().unwrap(), [(vec![0.1, 0.2, 0.3], 3)]);
        assert_eq!(*variance_args.lock().unwrap(), [(vec![0.1, 0.2, 0.3], 1)]);
        tracer.finish(RunStatus::Ok);
        let metrics: Vec<u64> = sink
            .lines()
            .iter()
            .filter(|l| l["ev"] == "metric")
            .map(|l| l["n"].as_u64().unwrap())
            .collect();
        assert_eq!(metrics, [3, 1]);
    }

    #[test]
    fn iter_is_stamped_before_the_observer_and_metric_spans_the_oracle() {
        /// A variance oracle that notes, on the trace's clock, when each of
        /// its methods runs.
        struct ClockedVariance {
            tracer: Arc<CallTracer>,
            ran_at: Arc<Mutex<Vec<u64>>>,
        }

        impl ClockedVariance {
            fn note(&self) {
                self.ran_at.lock().unwrap().push(self.tracer.now_ns());
            }
        }

        impl VarianceOracle for ClockedVariance {
            fn variance(&self, _theta: &[f64], _param_index: usize) -> f64 {
                self.note();
                1.0
            }

            fn variance_diagonal(&self, _theta: &[f64], dims: usize) -> Vec<f64> {
                self.note();
                vec![1.0; dims]
            }
        }

        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "call-times");
        let ran_at = Arc::new(Mutex::new(Vec::new()));
        let (clock, noted) = (Arc::clone(&tracer), Arc::clone(&ran_at));
        let observer = move |_: &TrainProgress<'_>| {
            noted.lock().unwrap().push(clock.now_ns());
            ControlFlow::Continue(())
        };
        let mut traced_observer =
            TracingObserver::new(Some(Box::new(observer)), Arc::clone(&tracer));
        let oracle = ClockedVariance {
            tracer: Arc::clone(&tracer),
            ran_at: Arc::clone(&ran_at),
        };
        let traced_oracle = TracingVariance::new(Box::new(oracle), Arc::clone(&tracer));
        let theta = [0.0, 0.0];
        let progress = TrainProgress {
            iteration: 1,
            best_fitness: 0.0,
            best_params: &theta,
            evaluations: 1,
        };
        assert_eq!(
            traced_observer.observe(&progress),
            ControlFlow::Continue(())
        );
        assert_eq!(traced_oracle.variance_diagonal(&theta, 2), [1.0, 1.0]);
        assert_eq!(traced_oracle.variance(&theta, 0), 1.0);
        tracer.finish(RunStatus::Ok);
        let ran_at = ran_at.lock().unwrap().clone();
        let lines = sink.lines();
        let events: Vec<&Value> = lines
            .iter()
            .filter(|l| l["ev"] == "iter" || l["ev"] == "metric")
            .collect();
        assert_eq!((events.len(), ran_at.len()), (3, 3));
        let field = |l: &Value, f: &str| l[f].as_u64().unwrap();
        // The iteration is stamped before the run's observer is called.
        assert!(field(events[0], "t") <= ran_at[0]);
        // Each metric call spans the wrapped oracle's call.
        for (metric, &ran) in events[1..].iter().zip(&ran_at[1..]) {
            let (t, dt) = (field(metric, "t"), field(metric, "dt"));
            assert!(t <= ran && ran <= t + dt, "{metric} vs {ran}");
        }
    }
}
