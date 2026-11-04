//! The call trace through [`TrainFlow`] and [`RunCircuitFlow`]: what a traced
//! run writes, and that the trace changes nothing in the run it records.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use polypus_infrastructure::{
    BackendCapabilities, BackendError, Planner, QuantumBackend, RunParams, SequentialPlanner,
    ShotDistributingPlanner,
};
use polypus_optimizers::AdamRule;
use serde_json::{json, Value};

use super::tests::every_method;
use super::*;
use crate::trace::test_support::{
    header, params, sink_tracer, FailingSink, HashBackend, KeyOneShare, SharedSink,
};
use crate::trace::{CallTracer, RunStatus, DEFAULT_MAX_BYTES};
use crate::{AdamConfig, Scheduler, SpsaConfig, TrainProgress};

/// An oracle that scores each batch with one call to its run's planner: each
/// candidate is the circuit `Qasm2` of its `Debug` text, and its value the
/// share of its shots that measured `"1"` ([`KeyOneShare`]). A failed batch
/// records its error in the run's slot and scores `NaN`.
struct PlannerOracle {
    backend: Arc<dyn QuantumBackend>,
    planner: Arc<dyn Planner>,
    config: Arc<RunParams>,
    cancel: CancelToken,
    errors: OracleErrorSlot,
}

impl PlannerOracle {
    /// One planner batch of `candidates`, with `shots` shots each.
    fn score(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        let circuits: Vec<BoundCircuit> = candidates
            .iter()
            .map(|candidate| BoundCircuit::Qasm2(format!("{candidate:?}")))
            .collect();
        let tasks: Vec<CircuitTask<'_>> = circuits
            .iter()
            .map(|circuit| CircuitTask { circuit, shots })
            .collect();
        let backend = self.backend.as_ref();
        match self
            .planner
            .evaluate(backend, &tasks, &KeyOneShare, &self.config, &self.cancel)
        {
            Ok(values) => values,
            Err(err) => {
                self.errors.record(Box::new(err), &self.config.id);
                vec![f64::NAN; candidates.len()]
            }
        }
    }
}

impl EvaluationOracle for PlannerOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        self.score(candidates, self.config.shots)
    }

    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        self.score(candidates, shots)
    }
}

/// Builds a [`PlannerOracle`] over the resources the flow hands it.
struct PlannerFactory;

impl OracleFactory for PlannerFactory {
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        Box::new(PlannerOracle {
            backend: Arc::clone(&resources.backend),
            planner: Arc::clone(&resources.planner),
            config: Arc::clone(&resources.config),
            cancel: cancel.clone(),
            errors: errors.clone(),
        })
        .into()
    }
}

/// Trains `method` over two parameters, with seed 7, through a
/// [`PlannerOracle`] on a [`HashBackend`] that runs 4 circuits a wave behind
/// its default planner, with `observer`, `trace` and `final_shots`.
fn run_train(
    method: Method,
    observer: Option<Box<dyn TrainObserver>>,
    trace: Option<Arc<CallTracer>>,
    final_shots: Option<u32>,
) -> TrainOutcome {
    let resources = Resources::new(
        Arc::new(HashBackend { max_concurrency: 4 }),
        None,
        Arc::new(params()),
    )
    .unwrap();
    let flow = TrainFlow {
        factory: PlannerFactory,
        method,
        dimensions: 2,
        seed: 7,
        initial_point: None,
        final_shots,
        observer,
        errors: OracleErrorSlot::new(),
        trace,
    };
    // `OracleError` does not implement `Debug`, so match rather than `.expect`.
    match Scheduler::ephemeral(resources).run(flow) {
        Ok(outcome) => outcome,
        Err(_) => panic!("the train flow must run"),
    }
}

/// The SPSA run of [`every_method`], with `max_iters` iterations.
fn spsa(max_iters: u32) -> Method {
    let Some((_, Method::Spsa(cfg))) = every_method()
        .into_iter()
        .find(|(label, _)| *label == "spsa")
    else {
        unreachable!("every_method has an SPSA run")
    };
    Method::Spsa(SpsaConfig { max_iters, ..cfg })
}

/// A [`HashBackend`] that runs 4 circuits a wave, behind a log of the circuits
/// and shots of every call to `run_circuits`; its `fail_at`-th call, counted
/// from 1, fails instead.
struct FailsOneCall {
    fail_at: usize,
    calls: Arc<Mutex<Vec<String>>>,
}

impl QuantumBackend for FailsOneCall {
    fn run_circuits(
        &self,
        qcs: &[BoundCircuit],
        params: &RunParams,
    ) -> Result<Vec<Counts>, BackendError> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(format!("{qcs:?} with {} shots", params.shots));
            calls.len()
        };
        if call == self.fail_at {
            return Err(BackendError::Conversion(format!("call {call} fails")));
        }
        HashBackend { max_concurrency: 4 }.run_circuits(qcs, params)
    }

    fn capabilities(&self) -> BackendCapabilities {
        HashBackend { max_concurrency: 4 }.capabilities()
    }
}

/// An oracle that, like the oracles of `polypus-evaluation`, stops calling
/// its planner once the run's slot holds a failure: from then on it scores
/// `NaN` without calling the oracle it wraps.
struct StopsOnFailure {
    inner: Box<dyn EvaluationOracle>,
    errors: OracleErrorSlot,
}

impl EvaluationOracle for StopsOnFailure {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        if self.errors.failed() {
            return vec![f64::NAN; candidates.len()];
        }
        self.inner.evaluate_batch(candidates)
    }

    fn evaluate_batch_with_shots(&self, candidates: &[Vec<f64>], shots: u32) -> Vec<f64> {
        if self.errors.failed() {
            return vec![f64::NAN; candidates.len()];
        }
        self.inner.evaluate_batch_with_shots(candidates, shots)
    }
}

/// Builds a [`PlannerOracle`] that stops on failure ([`StopsOnFailure`]).
struct StopsOnFailureFactory;

impl OracleFactory for StopsOnFailureFactory {
    fn build(
        self,
        resources: &Resources,
        cancel: &CancelToken,
        errors: &OracleErrorSlot,
    ) -> Oracles {
        Box::new(StopsOnFailure {
            inner: PlannerFactory.build(resources, cancel, errors).evaluation,
            errors: errors.clone(),
        })
        .into()
    }
}

/// Trains `method` as [`run_train`] does, without an observer or a final
/// batch, through the oracle `factory` builds, on a [`FailsOneCall`] backend
/// that fails its `fail_at`-th call; returns the run's error, as its `Debug`
/// text, and the backend's calls.
fn run_failing<F: OracleFactory>(
    factory: F,
    method: Method,
    fail_at: usize,
    trace: Option<Arc<CallTracer>>,
) -> (String, Vec<String>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let backend = FailsOneCall {
        fail_at,
        calls: Arc::clone(&calls),
    };
    let resources = Resources::new(Arc::new(backend), None, Arc::new(params())).unwrap();
    let flow = TrainFlow {
        factory,
        method,
        dimensions: 2,
        seed: 7,
        initial_point: None,
        final_shots: None,
        observer: None,
        errors: OracleErrorSlot::new(),
        trace,
    };
    let error = match Scheduler::ephemeral(resources).run(flow) {
        Err(OracleError::Evaluation(err)) => format!("{err:?}"),
        Err(OracleError::Config(err)) => panic!("the backend's error must be the run's: {err:?}"),
        Ok(_) => panic!("a run whose backend fails must fail"),
    };
    let calls = calls.lock().unwrap().clone();
    (error, calls)
}

/// An iter line's `i`, the batches written before it and its `evals`.
type IterLine = (u64, u64, u64);

/// Batch sizes and (i, batches written before, evals) of every iter line for each
/// `every_method()` configuration at d = 2, measured with polypus.train on the same
/// configurations (docs/guide/training.md gives each method's batch formula).
const SHAPES: [(&str, &[u64], &[IterLine]); 10] = [
    (
        "rand1bin",
        &[6, 6, 6, 6, 6],
        &[(1, 2, 12), (2, 3, 18), (3, 4, 24), (4, 5, 30)],
    ),
    (
        "shade",
        &[6, 6, 6, 6, 6],
        &[(1, 2, 12), (2, 3, 18), (3, 4, 24), (4, 5, 30)],
    ),
    (
        "lshade",
        &[6, 6, 5, 5, 5],
        &[(1, 2, 12), (2, 3, 17), (3, 4, 22), (4, 5, 27)],
    ),
    (
        "pso",
        &[6, 6, 6, 6, 6],
        &[(1, 2, 12), (2, 3, 18), (3, 4, 24), (4, 5, 30)],
    ),
    (
        "cmaes",
        &[6, 6, 6, 6],
        &[(1, 1, 6), (2, 2, 12), (3, 3, 18), (4, 4, 24)],
    ),
    (
        "qng",
        &[4, 1, 4, 1, 4, 1],
        &[(1, 2, 5), (2, 4, 10), (3, 6, 15)],
    ),
    ("adam", &[5, 5, 5, 1], &[(1, 1, 5), (2, 2, 10), (3, 4, 16)]),
    (
        "spsa",
        &[50, 3, 2, 3, 1],
        &[(1, 2, 53), (2, 3, 55), (3, 5, 59)],
    ),
    (
        "rotosolve",
        &[3, 2, 2, 3, 1],
        &[(1, 1, 3), (2, 2, 5), (3, 3, 7), (4, 5, 11)],
    ),
    (
        "external",
        &[4, 4, 4, 4, 4],
        &[(1, 1, 4), (2, 2, 8), (3, 3, 12), (4, 4, 16), (5, 5, 20)],
    ),
];

/// The lines of `lines` whose `ev` is `ev`, in order.
fn by_ev<'a>(lines: &'a [Value], ev: &str) -> Vec<&'a Value> {
    lines.iter().filter(|l| l["ev"] == ev).collect()
}

/// Without a trace the factory gets the run's own planner, the same `Arc`;
/// with one, the planner wrapped to record the run.
#[test]
fn trace_off_installs_nothing() {
    struct Capture(Arc<Mutex<Option<Arc<dyn Planner>>>>);
    impl OracleFactory for Capture {
        fn build(
            self,
            resources: &Resources,
            cancel: &CancelToken,
            errors: &OracleErrorSlot,
        ) -> Oracles {
            *self.0.lock().unwrap() = Some(Arc::clone(&resources.planner));
            PlannerFactory.build(resources, cancel, errors)
        }
    }
    for trace in [
        None,
        Some(sink_tracer(&SharedSink::default(), "traced-planner")),
    ] {
        let traced = trace.is_some();
        let planner: Arc<dyn Planner> = Arc::new(SequentialPlanner);
        let seen = Arc::new(Mutex::new(None));
        let resources = Resources::new(
            Arc::new(HashBackend { max_concurrency: 4 }),
            Some(Arc::clone(&planner)),
            Arc::new(params()),
        )
        .unwrap();
        let flow = TrainFlow {
            factory: Capture(Arc::clone(&seen)),
            method: every_method().remove(0).1,
            dimensions: 2,
            seed: 7,
            initial_point: None,
            final_shots: None,
            observer: None,
            errors: OracleErrorSlot::new(),
            trace,
        };
        assert!(Scheduler::ephemeral(resources).run(flow).is_ok());
        let built_with = seen.lock().unwrap().take().unwrap();
        assert_eq!(
            Arc::ptr_eq(&built_with, &planner),
            !traced,
            "traced = {traced}"
        );
    }
}

/// Each method's trace holds the batches its optimizer sends, within the
/// bound of its run shape, each split into waves the backend can run, and an
/// iter line per iteration after the batches that iteration needed.
#[test]
fn trace_matches_method_shape() {
    for (label, method) in every_method() {
        let (_, sizes, iters) = SHAPES.iter().find(|(l, _, _)| *l == label).unwrap();
        let shape = method.run_shape(2);
        let bound = u64::from(shape.leading_batches)
            + u64::from(shape.batches_per_generation) * shape.generations;
        let sink = SharedSink::default();
        run_train(method, None, Some(sink_tracer(&sink, label)), None);
        let lines = sink.lines();
        let batches = by_ev(&lines, "batch");
        let got: Vec<u64> = batches.iter().map(|b| b["n"].as_u64().unwrap()).collect();
        assert_eq!(got, *sizes, "{label}: batch sizes");
        assert!(
            got.len() as u64 <= bound,
            "{label}: {} batches over run_shape's bound {bound}",
            got.len()
        );
        let got: Vec<IterLine> = by_ev(&lines, "iter")
            .iter()
            .map(|l| {
                (
                    l["i"].as_u64().unwrap(),
                    l["batches"].as_u64().unwrap(),
                    l["evals"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(got, *iters, "{label}: iterations");
        for (k, batch) in batches.iter().enumerate() {
            assert_eq!(
                (
                    batch["k"].as_u64(),
                    batch["op"].as_str(),
                    batch["shots"].as_u64()
                ),
                (Some(k as u64), Some("evaluate"), Some(500)),
                "{label}"
            );
            let waves: Vec<u64> = by_ev(&lines, "wave")
                .iter()
                .filter(|w| w["k"] == batch["k"])
                .map(|w| w["n"].as_u64().unwrap())
                .collect();
            assert_eq!(
                waves.iter().sum::<u64>(),
                batch["n"].as_u64().unwrap(),
                "{label}: waves add up to their batch"
            );
            assert!(
                waves.iter().all(|&n| n <= 4),
                "{label}: waves respect max_concurrency"
            );
        }
    }
    // The iteration number starts at 1: with 8 resamplings, iteration 8 comes
    // after the calibration batch and the batches of iterations 1 to 8, so 9
    // batches were written before it, not 10.
    let Method::Spsa(cfg) = spsa(10) else {
        unreachable!()
    };
    let sink = SharedSink::default();
    run_train(
        Method::Spsa(SpsaConfig {
            resamplings: 8,
            eval_interval: 20,
            ..cfg
        }),
        None,
        Some(sink_tracer(&sink, "spsa-k8")),
        None,
    );
    let lines = sink.lines();
    let eighth = by_ev(&lines, "iter")
        .into_iter()
        .find(|l| l["i"] == 8)
        .unwrap();
    assert_eq!(eighth["batches"], 9);
}

/// The iter lines are exactly what the run's observer is shown, each after
/// the batches written before it, and a QNG run writes one metric line per
/// call to its metric; the clock of both never goes back.
#[test]
fn trace_iter_and_metric_events_are_exact() {
    for (label, method) in every_method() {
        let sink = SharedSink::default();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let observer = move |p: &TrainProgress<'_>| {
            record
                .lock()
                .unwrap()
                .push((p.iteration as u64, p.evaluations, p.best_fitness));
            ControlFlow::Continue(())
        };
        run_train(
            method,
            Some(Box::new(observer)),
            Some(sink_tracer(&sink, label)),
            None,
        );
        let lines = sink.lines();
        let (mut batches, mut last_t, mut iters) = (0u64, 0u64, Vec::new());
        for line in &lines {
            match line["ev"].as_str().unwrap() {
                "batch" => batches += 1,
                "iter" => {
                    assert_eq!(
                        line["batches"].as_u64(),
                        Some(batches),
                        "{label}: batches before"
                    );
                    iters.push((
                        line["i"].as_u64().unwrap(),
                        line["evals"].as_u64().unwrap(),
                        line["best"].as_f64().unwrap(),
                    ));
                }
                _ => {}
            }
            if line["ev"] == "iter" || line["ev"] == "metric" {
                let t = line["t"].as_u64().unwrap();
                assert!(t >= last_t, "{label}: t went back");
                last_t = t;
            }
        }
        assert_eq!(
            iters,
            *seen.lock().unwrap(),
            "{label}: the iter lines are what the observer saw"
        );
        let metrics: Vec<u64> = by_ev(&lines, "metric")
            .iter()
            .map(|m| m["n"].as_u64().unwrap())
            .collect();
        let expected: &[u64] = if label == "qng" { &[2, 2, 2] } else { &[] };
        assert_eq!(
            metrics, expected,
            "{label}: one metric line per variance_diagonal call"
        );
    }
}

/// A traced run returns what the untraced run returns, the final batch of
/// `final_shots` included, and that batch is the last one traced.
#[test]
fn trace_does_not_change_the_outcome() {
    for ((label, plain), (_, traced)) in every_method().into_iter().zip(every_method()) {
        let sink = SharedSink::default();
        let a = run_train(plain, None, None, Some(64));
        let b = run_train(traced, None, Some(sink_tracer(&sink, label)), Some(64));
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "{label}");
        assert!(
            by_ev(&sink.lines(), "batch")
                .last()
                .is_some_and(|b| b["shots"] == 64),
            "{label}: the final batch is traced"
        );
    }
}

/// A run whose backend fails returns the same error traced and untraced,
/// whatever its oracle does after the failure. A traced run is observed, so it
/// stops at the end of the iteration that recorded the failure, while an
/// untraced run without an observer goes on to the optimizer's end; with an
/// oracle that stops calling its planner once the run's slot holds a failure,
/// as the oracles of `polypus-evaluation` do, both make the same calls to the
/// backend.
#[test]
fn trace_keeps_the_error_and_the_backend_calls_of_a_failed_run() {
    let method = |label: &str| match label {
        "spsa-200" => spsa(200),
        _ => every_method()
            .into_iter()
            .find(|(l, _)| *l == label)
            .map(|(_, method)| method)
            .unwrap(),
    };
    let traced = |label: &str| Some(sink_tracer(&SharedSink::default(), label));
    let labels = every_method().into_iter().map(|(label, _)| label);
    for label in labels.chain(["spsa-200"]) {
        // The backend's third call fails, early in every run.
        let stops = run_failing(StopsOnFailureFactory, method(label), 3, None);
        assert_eq!(
            run_failing(StopsOnFailureFactory, method(label), 3, traced(label)),
            stops,
            "{label}: the same error and the same calls to the backend"
        );
        let (error, calls) = run_failing(PlannerFactory, method(label), 3, None);
        // The failure is not the end of the run: untraced, an oracle that keeps
        // calling its planner makes more calls after it, except under Adam,
        // which stops at once, since the batch's `NaN`s make its gradient
        // non-finite.
        assert_eq!(
            calls.len() > stops.1.len(),
            label != "adam",
            "{label}: whether the run has batches after the one that failed"
        );
        assert_eq!(
            run_failing(PlannerFactory, method(label), 3, traced(label)).0,
            error,
            "{label}: the same error from an oracle that keeps calling its planner"
        );
    }
}

/// A sink that fails mid-run stops the trace with one warning and no end
/// line, and the run returns what it returns untraced.
#[test]
fn trace_survives_write_errors() {
    crate::test_log::capture_logs();
    let plain = run_train(spsa(2_000), None, None, None);
    let sink = FailingSink::new(100_000);
    let tracer = CallTracer::from_sink(
        Box::new(sink.clone()),
        header("trace-write-errors"),
        DEFAULT_MAX_BYTES,
    );
    let traced = run_train(spsa(2_000), None, Some(Arc::clone(&tracer)), None);
    tracer.finish(RunStatus::Ok);
    drop(tracer);
    assert_eq!(format!("{plain:?}"), format!("{traced:?}"));
    let logged = crate::test_log::logged_for("trace-write-errors");
    assert_eq!(logged.len(), 1, "{logged:?}");
    assert!(logged[0].contains("call trace stopped"));
    let text = String::from_utf8_lossy(&sink.accepted()).into_owned();
    assert_eq!(
        text.len(),
        100_000,
        "the sink failed mid-run, after a flush"
    );
    assert!(!text.contains(r#""ev":"end""#));
}

/// A run whose trace reaches its size cap finishes unchanged, and the trace
/// stops at the cap and still closes with an end line that counts every
/// batch and says it was truncated.
#[test]
fn trace_size_cap() {
    const CAP: u64 = 1 << 20; // POLYPUS_CALL_TRACE_MAX_MB=1
    let sink = SharedSink::default();
    let tracer = CallTracer::from_sink(Box::new(sink.clone()), header("trace-size-cap"), CAP);
    let plain = run_train(spsa(6_000), None, None, None);
    let traced = run_train(spsa(6_000), None, Some(Arc::clone(&tracer)), None);
    tracer.finish(RunStatus::Ok);
    assert_eq!(
        format!("{plain:?}"),
        format!("{traced:?}"),
        "the run finishes unchanged"
    );
    let (bytes, lines) = (sink.bytes(), sink.lines());
    let end_len = sink.text_lines().last().unwrap().len() as u64 + 1;
    let end = lines.last().unwrap();
    assert_eq!(
        (
            end["ev"].as_str(),
            end["status"].as_str(),
            end["truncated"].as_bool()
        ),
        (Some("end"), Some("ok"), Some(true))
    );
    assert!(bytes.len() as u64 <= CAP + end_len, "{} bytes", bytes.len());
    assert!(
        bytes.len() as u64 > CAP - 200,
        "the cap, not an error, stopped the trace"
    );
    assert!(
        end["batches"].as_u64().unwrap() > by_ev(&lines, "batch").len() as u64,
        "end counts every batch"
    );
}

/// A traced circuit run returns the untraced run's counts, and its trace
/// holds its one batch and the waves its replicas ran in.
#[test]
fn run_circuit_flow_traces_its_replicas() {
    let run = |trace| {
        let resources = Resources::new(
            Arc::new(HashBackend { max_concurrency: 2 }),
            Some(Arc::new(ShotDistributingPlanner::new(3))),
            Arc::new(params()),
        )
        .unwrap();
        Scheduler::ephemeral(resources)
            .run(RunCircuitFlow {
                circuits: vec![BoundCircuit::Qasm2("a".into())],
                shots: 10,
                trace,
            })
            .unwrap()
    };
    let sink = SharedSink::default();
    let tracer = sink_tracer(&sink, "replicas");
    assert_eq!(run(None), run(Some(Arc::clone(&tracer))));
    tracer.finish(RunStatus::Ok);
    let lines = sink.lines();
    let batches = by_ev(&lines, "batch");
    assert_eq!(batches.len(), 1);
    assert_eq!(
        (
            batches[0]["k"].as_u64(),
            batches[0]["op"].as_str(),
            batches[0]["n"].as_u64(),
            batches[0]["shots"].as_u64(),
            batches[0]["ok"].as_bool()
        ),
        (
            Some(0),
            Some("execute_replicas"),
            Some(1),
            Some(10),
            Some(true)
        )
    );
    let waves: Vec<(u64, u64, Value, u64)> = by_ev(&lines, "wave")
        .iter()
        .map(|w| {
            (
                w["w"].as_u64().unwrap(),
                w["n"].as_u64().unwrap(),
                w["replicas"].clone(),
                w["shots"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(waves, [(0, 2, json!([4, 3]), 7), (1, 1, json!([3]), 3)]);
    assert!(by_ev(&lines, "wave")
        .iter()
        .all(|w| w["k"] == 0 && w["op"] == "run_shots_distributed"));
}

/// Each method is named as `every_method` labels it: a DE run by its
/// strategy, an Adam run by its rule.
#[test]
fn method_name_labels_every_method() {
    for (label, method) in every_method() {
        assert_eq!(method.name(), label);
    }
    let Some((_, Method::Adam(cfg))) = every_method().into_iter().find(|(l, _)| *l == "adam")
    else {
        unreachable!()
    };
    assert_eq!(
        Method::Adam(AdamConfig {
            rule: AdamRule::Momentum,
            ..cfg
        })
        .name(),
        "momentum"
    );
}
