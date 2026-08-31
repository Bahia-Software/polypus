use crate::evaluation::{
    assign_parameters_qiskit, run_and_evaluate, CostObservable, EvaluationError, EvaluationOracle,
    OracleErrorSlot,
};
use crate::infrastructure::{BoundCircuit, ExecutionConfig, QuantumBackend};
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Multiplier applied to the machine's available parallelism to derive the
/// candidate concurrency bound (see `candidate_concurrency_limit`).
///
/// Two, not more: the GIL serialises the per-candidate work, so the only thing
/// oversubscription buys is coverage of the windows in which a worker *has*
/// released the GIL (Aer's C++ simulation, a network round-trip). One extra
/// candidate per core is enough to fill those windows; beyond that, threads only
/// add contention and stack memory.
const CONCURRENCY_MULTIPLIER: usize = 2;

/// Parallelism assumed when [`std::thread::available_parallelism`] cannot report
/// it (it returns an error rather than a guess when the platform hides the
/// value — a restrictive sandbox, an unsupported target).
///
/// One is the deliberate floor: under-guessing costs little here (the GIL
/// serialises the work anyway) whereas over-guessing is precisely the failure
/// this bound exists to prevent.
const FALLBACK_PARALLELISM: usize = 1;

/// Maximum number of candidates dispatched to the blocking pool at once:
/// `CONCURRENCY_MULTIPLIER × available_parallelism()`.
///
/// Derived, not configured — there is no kwarg or [`ExecutionConfig`] field for
/// it, so `qml.train`'s public signature is untouched.
fn candidate_concurrency_limit() -> usize {
    let parallelism = std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(FALLBACK_PARALLELISM);
    // Both constants are >= 1, so this is a floor rather than a correction: it
    // keeps the limit usable (and the dispatch loop finite) whatever they become.
    parallelism.saturating_mul(CONCURRENCY_MULTIPLIER).max(1)
}

/// Dispatch `items` in chunks, keeping at most `limit` of them in flight.
///
/// Deliberately free of PyO3 types: the caller decides what a handle is and
/// where the GIL is released, while this function owns only the chunking, the
/// ordering and the short-circuiting. That split is what makes the bound
/// testable without a Python runtime (docs/ENGINEERING.md §3 keeps the Rust
/// suite Python-free) — see the tests at the bottom of this module.
///
/// - `spawn` is called once per item, in input order, and returns an opaque
///   handle to the work it started. At most `limit` handles are live at once.
/// - `join_chunk` receives one chunk's handles in dispatch order and returns one
///   result per handle, in that same order. This is where the caller releases
///   the GIL — it happens once per chunk, so *every* wait on the workers is
///   covered, not just the last one.
/// - `after_chunk` runs on the dispatching thread after each joined chunk (the
///   `check_signals` boundary for the real caller).
///
/// The first error from either callback short-circuits: the remaining items are
/// never dispatched. Handles already spawned are dropped and their work is left
/// to finish on its own, exactly as the unbounded predecessor left
/// not-yet-awaited tasks running.
fn dispatch_bounded<T, H, R, E>(
    items: &[T],
    limit: usize,
    mut spawn: impl FnMut(&T) -> H,
    mut join_chunk: impl FnMut(Vec<H>) -> Result<Vec<R>, E>,
    mut after_chunk: impl FnMut() -> Result<(), E>,
) -> Result<Vec<R>, E> {
    let mut out = Vec::with_capacity(items.len());
    // `chunks` yields nothing at all for an empty slice, which would skip
    // `after_chunk` entirely; the unbounded predecessor still ran its signal
    // check on an empty batch, so keep that observable behaviour identical.
    if items.is_empty() {
        after_chunk()?;
        return Ok(out);
    }
    for chunk in items.chunks(limit.max(1)) {
        let handles: Vec<H> = chunk.iter().map(&mut spawn).collect();
        out.extend(join_chunk(handles)?);
        after_chunk()?;
    }
    Ok(out)
}

/// Oracle for QML training with feature-map encoding.
///
/// Holds N pre-bound training circuits (one per training sample, with
/// feature-map parameters already fixed). For each candidate `θ`, it binds `θ`
/// to every training circuit, runs them (batched by `config.n_qpus`), and
/// returns the **mean** expectation value as the fitness.
///
/// # Bounded per-candidate concurrency
///
/// Candidates are evaluated on Tokio's blocking pool, but **at most
/// `CONCURRENCY_MULTIPLIER × available_parallelism()` — two per core — are in
/// flight at any moment** (`candidate_concurrency_limit`, which falls back to a
/// parallelism of 1 when the platform will not report it). A DE/PSO population
/// in the hundreds used to become one `spawn_blocking` task per candidate, i.e.
/// hundreds of simultaneously-blocked OS threads saturating the pool for no
/// gain: the GIL serialises the actual Qiskit/Aer work, so the surplus threads
/// buy contention and stack memory rather than throughput.
///
/// The bound is enforced by **chunked dispatch**: one chunk of at most
/// `candidate_concurrency_limit()` candidates is spawned, joined, and only then
/// is the next chunk spawned (`dispatch_bounded`). A `tokio::sync::Semaphore`
/// would express "N permits" more directly, but acquiring a permit is an
/// `.await`, and this oracle must never hold the GIL while waiting on workers
/// that need it (docs/ENGINEERING.md §3, "Ignoring this deadlocks") — so every
/// wait point has to sit inside an `allow_threads` block. Chunk boundaries give
/// exactly two, both in one place: the join and the following `check_signals`.
/// That keeps the dispatch loop synchronous, panic-free, and free of extra
/// dependency surface (a semaphore would tie this path to `tokio`'s `sync`
/// feature).
///
/// Two consequences worth naming:
///
/// - The per-candidate `Py<PyAny>` clones (the training circuits and the
///   expectation function) are taken when a candidate is *dispatched*, so only
///   one chunk's worth exists at a time instead of one clone per training
///   circuit per candidate, taken up front before any work starts.
/// - Ctrl+C is now checked once per chunk instead of once per batch, so a large
///   population is more interruptible than before, not less.
///
/// Neither changes results: [`EvaluationOracle::evaluate_batch`] still returns
/// one fitness per candidate, in input order.
///
/// Note that the GIL still serialises the actual simulation calls whatever the
/// dispatch shape; once a native Rust backend is available the parallelism will
/// be genuine.
pub struct QmlOracle {
    /// Pre-bound training circuits (feature-map parameters already fixed).
    pub training_circuits: Vec<Py<PyAny>>,
    pub config: Arc<ExecutionConfig>,
    pub backend: Arc<dyn QuantumBackend>,
    pub observable: Arc<dyn CostObservable>,
    /// Shared with the `qml.train` entry point: the first evaluation failure is
    /// recorded here and surfaced as a `PyErr` after `optimize` returns, since
    /// [`EvaluationOracle::evaluate_batch`] cannot return a `Result`.
    pub errors: OracleErrorSlot,
}

impl EvaluationOracle for QmlOracle {
    fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
        // Once evaluation has failed, stop doing work: return finite sentinels
        // and let the entry point surface the recorded error.
        if self.errors.failed() {
            return vec![0.0; candidates.len()];
        }
        match self.try_evaluate(candidates) {
            Ok(values) => values,
            Err(e) => {
                self.errors.record(e, &self.config.id);
                vec![0.0; candidates.len()]
            }
        }
    }
}

impl QmlOracle {
    /// Fallible core of [`EvaluationOracle::evaluate_batch`]. Kept separate so
    /// the trait method (which must return `Vec<f64>`) can record any error and
    /// yield finite sentinels while the entry point re-raises it.
    fn try_evaluate(&self, candidates: &[Vec<f64>]) -> Result<Vec<f64>, EvaluationError> {
        let rt = crate::utils::tokio_runtime().map_err(|e| {
            EvaluationError::Runtime(format!(
                "failed to start the Tokio runtime for QML evaluation: {e}"
            ))
        })?;

        // At most `candidate_concurrency_limit()` candidates are in flight at
        // once; the rest wait their turn instead of becoming blocking-pool
        // threads. Results are collected chunk by chunk, so the returned vector
        // stays aligned with `candidates` index for index.
        dispatch_bounded(
            candidates,
            candidate_concurrency_limit(),
            // Dispatch one candidate. Runs on the calling thread, which still
            // holds the GIL, so this is where the per-task `Py<T>` clones are
            // taken (a `Py<T>` clone needs the GIL) — for the current chunk only,
            // not for the whole population before any task starts.
            |theta| {
                let training_circuits: Vec<Py<PyAny>> = Python::with_gil(|py| {
                    self.training_circuits
                        .iter()
                        .map(|qc| qc.clone_ref(py))
                        .collect()
                });
                let config = Arc::clone(&self.config);
                let backend = Arc::clone(&self.backend);
                // GIL-free clone (unlike the Py<T> circuits above).
                let observable = Arc::clone(&self.observable);
                let theta = theta.clone();

                rt.spawn_blocking(move || {
                    evaluate_qml_single(
                        &training_circuits,
                        &config,
                        backend.as_ref(),
                        observable.as_ref(),
                        &theta,
                    )
                })
            },
            // Join one chunk, in dispatch order.
            //
            // The calling thread entered Rust from a PyO3 `#[pyfunction]` and
            // still holds the GIL. Each `spawn_blocking` worker needs to acquire
            // the GIL (binding circuits, running them, computing expectations),
            // so we MUST release it here while blocking on them — otherwise this
            // thread holds the GIL inside `block_on` while the workers wait for
            // it: a deadlock. The release lives *inside* this closure precisely
            // so it covers every chunk boundary, not just the final join.
            |handles| {
                Python::with_gil(|py| {
                    py.allow_threads(|| {
                        rt.block_on(async {
                            let mut joined = Vec::with_capacity(handles.len());
                            for h in handles {
                                // A `JoinError` means the worker task itself
                                // panicked; turn it into a typed error rather
                                // than re-panicking. It is a Rust-side runtime
                                // failure, not a raised Python exception, hence
                                // `Runtime` and not `Python` (issue #81).
                                let single = h.await.map_err(|e| {
                                    EvaluationError::Runtime(format!(
                                        "QML evaluation task failed: {e}"
                                    ))
                                })?;
                                joined.push(single?);
                            }
                            Ok::<_, EvaluationError>(joined)
                        })
                    })
                })
            },
            // The workers ran off the main thread, where `PyErr_CheckSignals` is
            // a no-op, so a pending SIGINT (Ctrl+C) was not seen there. Check it
            // here on the calling (main) thread so `qml.train` is interruptible
            // too; per chunk rather than once per batch, so a large population is
            // noticed sooner and the remaining chunks are never dispatched. The
            // KeyboardInterrupt is carried verbatim via `EvaluationError::Python`.
            || Python::with_gil(|py| py.check_signals().map_err(EvaluationError::Python)),
        )
    }
}

/// Evaluate one candidate `theta` against all training circuits.
///
/// Binds `theta` to each training circuit, runs them in batches of
/// `config.n_qpus`, and returns the mean expectation value. Any failure is
/// returned as an [`EvaluationError`] instead of panicking.
fn evaluate_qml_single(
    training_circuits: &[Py<PyAny>],
    config: &ExecutionConfig,
    backend: &dyn QuantumBackend,
    observable: &dyn CostObservable,
    theta: &[f64],
) -> Result<f64, EvaluationError> {
    // Training circuits are Qiskit objects (feature-map pre-binding is
    // Qiskit-specific); native QML circuits arrive with a later phase.
    let bound: Vec<BoundCircuit> = training_circuits
        .iter()
        .map(|qc_xi| {
            Ok(BoundCircuit::Qiskit(assign_parameters_qiskit(
                qc_xi, theta,
            )?))
        })
        .collect::<Result<_, EvaluationError>>()?;

    let batch_size = backend.max_batch_size(bound.len()).max(1);
    let mut all_ev: Vec<f64> = Vec::with_capacity(bound.len());
    for chunk in bound.chunks(batch_size) {
        let ev = run_and_evaluate(backend, chunk, config, observable)?;
        all_ev.extend(ev);
    }

    // Defense-in-depth (contract C-5): `run_and_evaluate` already guarantees
    // exactly `chunk.len()` values per chunk, and the chunks partition `bound`
    // exactly, so this can only ever hold. Unlike `VqcOracle` this function
    // returns one scalar per *candidate* (the mean over the training circuits),
    // so the invariant here is `all_ev.len() == bound.len()` (one expectation
    // per training circuit), not `candidates.len()`. Kept as an explicit,
    // self-documenting invariant guarding the mean below — do not "simplify" it
    // away. Reported as a `Result`, never a panic (rule 4; runs under
    // `OracleErrorSlot`).
    if all_ev.len() != bound.len() {
        return Err(EvaluationError::WrongLength {
            expected: bound.len(),
            got: all_ev.len(),
        });
    }

    Ok(all_ev.iter().sum::<f64>() / all_ev.len() as f64)
}

/// Tests for the bounding mechanism itself (issue #85).
///
/// `QmlOracle` cannot be unit-tested here — it holds `Py<PyAny>` and calls into
/// Qiskit, and the Rust suite is Python-runtime-free by design
/// (docs/ENGINEERING.md §3); its end-to-end coverage lives in `tests/python/`.
/// `dispatch_bounded` is the part that *enforces* the bound, and it is pure Rust
/// with no PyO3 types in its signature, so it can be driven with real threads and
/// a live in-flight counter right here.
#[cfg(test)]
mod tests {
    use super::{candidate_concurrency_limit, dispatch_bounded, CONCURRENCY_MULTIPLIER};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;
    use std::time::Duration;

    /// The "population in the hundreds" from the issue: what used to become this
    /// many concurrently-blocked threads.
    const POPULATION: usize = 300;
    /// Bound used by the dispatch tests. Fixed rather than derived so the
    /// assertions do not depend on the test machine's core count.
    const LIMIT: usize = 8;
    /// Long enough that a chunk's workers demonstrably overlap (thread spawn is
    /// tens of microseconds), short enough to keep the test well under a second.
    const WORK: Duration = Duration::from_millis(5);

    #[test]
    fn dispatch_bounded_keeps_at_most_limit_items_in_flight() {
        let items: Vec<usize> = (0..POPULATION).collect();
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let results = dispatch_bounded(
            &items,
            LIMIT,
            |item| {
                let item = *item;
                let in_flight = Arc::clone(&in_flight);
                let peak = Arc::clone(&peak);
                std::thread::spawn(move || {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(WORK);
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    item * 2
                })
            },
            |handles: Vec<JoinHandle<usize>>| {
                handles
                    .into_iter()
                    .map(|h| h.join().map_err(|_| "a worker panicked"))
                    .collect::<Result<Vec<_>, &'static str>>()
            },
            || Ok(()),
        )
        .expect("bounded dispatch over healthy workers must succeed");

        // Every candidate ran, and index i in is still index i out.
        assert_eq!(results.len(), POPULATION);
        assert!(results.iter().enumerate().all(|(i, &v)| v == i * 2));

        let peak = peak.load(Ordering::SeqCst);
        assert!(
            peak <= LIMIT,
            "{peak} items were in flight at once; the bound is {LIMIT}"
        );
        // Guards against "passing" by serialising everything: the point is a
        // *bound*, not the removal of concurrency.
        assert!(peak > 1, "dispatch collapsed to sequential (peak {peak})");
    }

    #[test]
    fn dispatch_bounded_stops_dispatching_after_a_chunk_error() {
        let items: Vec<usize> = (0..POPULATION).collect();
        let dispatched = AtomicUsize::new(0);

        let err = dispatch_bounded(
            &items,
            LIMIT,
            |_item| {
                dispatched.fetch_add(1, Ordering::SeqCst);
            },
            |_handles: Vec<()>| Err::<Vec<usize>, &'static str>("first chunk failed"),
            || Ok(()),
        )
        .expect_err("a chunk failure must propagate");

        assert_eq!(err, "first chunk failed");
        // The pre-existing behaviour: the first failure stops further work.
        // Nothing beyond the failing chunk may have been dispatched — which is
        // also what keeps the per-candidate `Py<T>` clones from being taken for
        // the whole population.
        assert_eq!(
            dispatched.load(Ordering::SeqCst),
            LIMIT,
            "only the failing chunk may have been dispatched"
        );
    }

    #[test]
    fn dispatch_bounded_runs_the_hook_per_chunk_and_short_circuits_on_it() {
        let items: Vec<usize> = (0..POPULATION).collect();
        let dispatched = AtomicUsize::new(0);
        let hook_calls = AtomicUsize::new(0);

        // The hook stands in for `py.check_signals()`; failing on the second
        // chunk models a Ctrl+C observed part-way through the population.
        let err = dispatch_bounded(
            &items,
            LIMIT,
            |item| {
                dispatched.fetch_add(1, Ordering::SeqCst);
                *item
            },
            |handles: Vec<usize>| Ok::<_, &'static str>(handles),
            || {
                if hook_calls.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                    return Err("interrupted");
                }
                Ok(())
            },
        )
        .expect_err("a hook failure must propagate");

        assert_eq!(err, "interrupted");
        assert_eq!(
            hook_calls.load(Ordering::SeqCst),
            2,
            "the hook runs once per joined chunk"
        );
        assert_eq!(
            dispatched.load(Ordering::SeqCst),
            2 * LIMIT,
            "no chunk may be dispatched after the hook failed"
        );
    }

    #[test]
    fn dispatch_bounded_still_runs_the_hook_for_an_empty_batch() {
        let items: [usize; 0] = [];
        let dispatched = AtomicUsize::new(0);
        let hook_calls = AtomicUsize::new(0);

        let out = dispatch_bounded(
            &items,
            LIMIT,
            |_item| {
                dispatched.fetch_add(1, Ordering::SeqCst);
            },
            |_handles: Vec<()>| Ok::<Vec<usize>, &'static str>(Vec::new()),
            || {
                hook_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect("an empty batch is not an error");

        assert!(out.is_empty());
        assert_eq!(dispatched.load(Ordering::SeqCst), 0);
        // Same as before chunking: an empty batch still reaches the signal check.
        assert_eq!(hook_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dispatch_bounded_floors_a_zero_limit_to_one() {
        let items: Vec<usize> = (0..5).collect();
        let chunk_sizes = Mutex::new(Vec::new());

        let out = dispatch_bounded(
            &items,
            0,
            |item| *item,
            |handles: Vec<usize>| {
                chunk_sizes
                    .lock()
                    .map_err(|_| "the chunk-size log was poisoned")?
                    .push(handles.len());
                Ok::<_, &'static str>(handles)
            },
            || Ok(()),
        )
        .expect("a degenerate limit must degrade to sequential dispatch, not fail");

        assert_eq!(out, items);
        assert_eq!(
            *chunk_sizes
                .lock()
                .expect("the chunk-size log must be intact"),
            vec![1; 5]
        );
    }

    #[test]
    fn candidate_concurrency_limit_is_two_per_core() {
        let parallelism = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(super::FALLBACK_PARALLELISM);

        assert_eq!(
            candidate_concurrency_limit(),
            parallelism * CONCURRENCY_MULTIPLIER
        );
        // Never zero, whatever the platform reports (or fails to report): a zero
        // limit would stall the dispatch loop.
        assert!(candidate_concurrency_limit() >= CONCURRENCY_MULTIPLIER);
    }
}
