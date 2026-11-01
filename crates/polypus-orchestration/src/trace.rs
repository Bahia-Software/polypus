//! The call trace (contract C-12): a JSON Lines record of how a run called its
//! planner, its backend, its optimizer and the QNG metric.
//!
//! A [`CallTracer`] writes one JSON object per line (schema v1). The first line
//! is the `run` header: `"v":1`, `"ev":"run"`, the fields of [`TraceHeader`] in
//! their declared order and `t0_unix_ns`, the wall-clock time the tracer was
//! created at. Then one line per [`TraceEvent`] — `batch` (a planner batch),
//! `wave` (a backend call), `iter` (an optimizer iteration) and `metric` (a call
//! to the QNG metric) — and last the `end` line: the run's [`RunStatus`], the
//! batches, waves and circuits recorded, whether the trace was truncated, and
//! its `t`. Every `t` and `dt` is in nanoseconds, `t` counted from the tracer's
//! creation. Nothing else is written: no circuits, parameters, counts,
//! environment variables or paths.
//!
//! - **Safe to open.** [`CallTracer::create`] never overwrites a file, never
//!   follows a symbolic link and never creates a directory; on Unix the file is
//!   created with mode `0640`, narrowed by the umask.
//! - **Bounded.** An event that would take the trace past its size cap is
//!   counted but not written, and neither is any event after it. The header and
//!   the `end` line are always written; `end` then says `"truncated":true`.
//! - **Never in the way.** Recording cannot fail the run: the first write error
//!   logs one warning and stops the trace, which is left without an `end` line.
//!   A tracer dropped before [`CallTracer::finish`] closes its trace as
//!   `aborted`.
//!
//! [`TracingPlanner`] and [`TracingBackend`] record the `batch` and `wave`
//! events: each wraps a planner or a backend, forwards every method of its
//! trait unchanged, and records the calls that run circuits.
//! [`TracingObserver`] and [`TracingVariance`] record the `iter` and `metric`
//! events: the first wraps the run's [`TrainObserver`](crate::TrainObserver),
//! if it has one, and records each iteration the optimizer reports; the second
//! wraps the QNG metric and records each call to it.
//!
//! # Example
//!
//! ```
//! use polypus_orchestration::trace::{
//!     CallTracer, RunStatus, TraceEvent, TraceFlow, TraceHeader, DEFAULT_MAX_BYTES,
//! };
//!
//! let name = format!("polypus-trace-doc-{}.jsonl", std::process::id());
//! let path = std::env::temp_dir().join(name);
//! let header = TraceHeader {
//!     flow: TraceFlow::Train,
//!     id: "doc".to_owned(),
//!     polypus: env!("CARGO_PKG_VERSION").to_owned(),
//!     method: Some("qng"),
//!     dims: Some(2),
//!     backend: "native".to_owned(),
//!     infrastructure: "local".to_owned(),
//!     n_qpus: 1,
//!     shots: 1024,
//!     seed: Some(7),
//! };
//! let tracer = CallTracer::create(&path, header, DEFAULT_MAX_BYTES)?;
//! tracer.record(TraceEvent::Metric { n: 2, t: 1_000, dt: 500 });
//! tracer.finish(RunStatus::Ok);
//! let trace = std::fs::read_to_string(&path)?;
//! assert!(trace.lines().last().is_some_and(|end| end.contains(r#""status":"ok""#)));
//! std::fs::remove_file(&path)?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::ser::{CompactFormatter, Formatter as _};

/// The default size cap of a trace, 512 MiB: see [`CallTracer::create`].
pub const DEFAULT_MAX_BYTES: u64 = 512 << 20;

/// The schema version, the header's `v`.
const SCHEMA_VERSION: u32 = 1;

/// The capacity of the buffer in front of the sink, 64 KiB.
const BUFFER_BYTES: usize = 64 << 10;

/// Where a tracer's lines go: a buffered file, or a test's sink.
type Sink = BufWriter<Box<dyn Write + Send>>;

/// The flow a trace records, the header's `flow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFlow {
    /// A training run (`polypus.train`), written `"train"`.
    Train,
    /// A circuit run (`polypus.run_quantum_circuit`), written `"run_circuit"`.
    RunCircuit,
}

/// What a trace's first line says about its run.
///
/// The line holds `"v":1` and `"ev":"run"`, then these fields in this order,
/// then `t0_unix_ns`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TraceHeader {
    /// The flow traced.
    pub flow: TraceFlow,
    /// The run's id.
    pub id: String,
    /// The Polypus version that ran it.
    pub polypus: String,
    /// The optimizer of a training run; `None` (written `null`) otherwise.
    pub method: Option<&'static str>,
    /// The number of parameters a training run optimizes; `None` otherwise.
    pub dims: Option<u32>,
    /// The backend's name.
    pub backend: String,
    /// The infrastructure's name.
    pub infrastructure: String,
    /// The number of QPUs the run was given.
    pub n_qpus: u32,
    /// Shots per circuit.
    pub shots: u32,
    /// The run's seed, if it has one.
    pub seed: Option<u64>,
}

/// The planner method a `batch` line records, its `op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchOp {
    /// [`Planner::evaluate`](polypus_infrastructure::Planner::evaluate),
    /// written `"evaluate"`.
    Evaluate,
    /// [`Planner::execute`](polypus_infrastructure::Planner::execute), written
    /// `"execute"`.
    Execute,
    /// [`Planner::execute_replicas`](polypus_infrastructure::Planner::execute_replicas),
    /// written `"execute_replicas"`.
    ExecuteReplicas,
}

impl BatchOp {
    /// The method's name as a `batch` line writes it: the serde name, which
    /// the encoder tests compare it with.
    const fn name(self) -> &'static str {
        match self {
            Self::Evaluate => "evaluate",
            Self::Execute => "execute",
            Self::ExecuteReplicas => "execute_replicas",
        }
    }
}

/// The backend method a `wave` line records, its `op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaveOp {
    /// [`QuantumBackend::run_circuits`](polypus_infrastructure::QuantumBackend::run_circuits),
    /// written `"run_circuits"`.
    RunCircuits,
    /// [`QuantumBackend::run_shots_distributed`](polypus_infrastructure::QuantumBackend::run_shots_distributed),
    /// written `"run_shots_distributed"`.
    RunShotsDistributed,
}

impl WaveOp {
    /// The method's name as a `wave` line writes it: the serde name, which the
    /// encoder tests compare it with.
    const fn name(self) -> &'static str {
        match self {
            Self::RunCircuits => "run_circuits",
            Self::RunShotsDistributed => "run_shots_distributed",
        }
    }
}

/// An event of a run, one line of its trace once
/// [`recorded`](CallTracer::record).
///
/// Each variant is written as its name in snake case (`"ev":"batch"`) followed
/// by its fields in their declared order. Every `t` is in nanoseconds since the
/// tracer was created, every `dt` a duration in nanoseconds.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TraceEvent {
    /// A planner batch: one call to the planner.
    Batch {
        /// The batch's index in the run, from 0.
        k: u64,
        /// The planner method called.
        op: BatchOp,
        /// Circuits in the batch.
        n: usize,
        /// Shots per circuit.
        shots: u32,
        /// When the batch started.
        t: u64,
        /// How long it took.
        dt: u64,
        /// Whether it succeeded.
        ok: bool,
    },
    /// A backend wave: one call to the backend.
    Wave {
        /// The batch the wave ran in; `None` (written `null`) for a call made
        /// outside a planner batch.
        k: Option<u64>,
        /// The wave's index within its batch, from 0.
        w: u32,
        /// The backend method called.
        op: WaveOp,
        /// Circuits in the call.
        n: usize,
        /// Shots per circuit.
        shots: u32,
        /// When the call started.
        t: u64,
        /// How long it took.
        dt: u64,
        /// Whether it succeeded.
        ok: bool,
        /// The shots each replica ran, for
        /// [`WaveOp::RunShotsDistributed`]; `None` otherwise, and then left
        /// out of the line.
        replicas: Option<Vec<u32>>,
    },
    /// An optimizer iteration. Its line also carries `batches`, the batches
    /// recorded before it, between `evals` and `best`.
    Iter {
        /// The iteration's number, as the optimizer reports it.
        i: usize,
        /// Objective evaluations so far.
        evals: u64,
        /// The best objective value so far; a value that is not finite is
        /// written `null`.
        best: f64,
        /// When the optimizer reported the iteration.
        t: u64,
    },
    /// A call to the QNG metric (a
    /// [`VarianceOracle`](polypus_optimizers::VarianceOracle)).
    Metric {
        /// The metric elements asked for.
        n: usize,
        /// When the call started.
        t: u64,
        /// How long it took.
        dt: u64,
    },
}

/// How a run ended, the `status` of the trace's `end` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The run returned its result, written `"ok"`.
    Ok,
    /// The run failed with an error, written `"error"`.
    Error,
    /// The run was interrupted or cancelled, written `"cancelled"`.
    Cancelled,
    /// Nobody closed the trace: the tracer was dropped without
    /// [`CallTracer::finish`]. Written `"aborted"`.
    Aborted,
}

/// Writes one run's call trace: the header when it is created, a line per
/// [`record`](Self::record)ed event, and the `end` line on
/// [`finish`](Self::finish) or, failing that, on drop.
///
/// Everything that records into a run's trace shares one `Arc<CallTracer>`. A
/// mutex, uncontended in practice, keeps each line whole and the lines in the
/// order they were recorded. The lines go through a 64 KiB buffer, flushed only
/// after the header and when the trace is closed, and each is serialized into
/// one reused buffer.
pub struct CallTracer {
    /// The run's id, for the log messages.
    id: String,
    /// When the tracer was created, the zero of every `t`.
    t0: Instant,
    /// The size cap: an event that would take the trace past it is not written.
    max_bytes: u64,
    /// The index the next planner batch gets.
    next_batch: AtomicU64,
    /// The sink, the counts and the reused line buffer.
    state: Mutex<State>,
}

impl CallTracer {
    /// Creates the trace file at `path` and writes its header line.
    ///
    /// The file must be new: if anything exists at `path`, a symbolic link
    /// included (even a dangling one), it is left untouched and the error's
    /// kind is [`AlreadyExists`](io::ErrorKind::AlreadyExists). A missing
    /// directory is not created ([`NotFound`](io::ErrorKind::NotFound)). On
    /// Unix the file is created with mode `0640`, narrowed by the umask.
    ///
    /// `max_bytes` caps the trace's size ([`DEFAULT_MAX_BYTES`] unless the
    /// caller picks another): the header counts toward it, an event that would
    /// pass it is not written and neither is any event after it, and the `end`
    /// line is written regardless.
    ///
    /// # Errors
    ///
    /// The error of creating the file, or of writing or flushing the header.
    /// After a header error the new file stays at `path`, empty or holding
    /// part of the header, since the tracer never removes a file: creating a
    /// trace at `path` again fails with `AlreadyExists`.
    pub fn create(path: &Path, header: TraceHeader, max_bytes: u64) -> io::Result<Arc<Self>> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o640);
        let file = options.open(path)?;
        let (tracer, started) = Self::start(Box::new(file), header, max_bytes);
        started?;
        Ok(Arc::new(tracer))
    }

    /// Records `event`: counts it for the `end` line and writes its line,
    /// unless the trace is past its size cap.
    ///
    /// Does nothing once the trace is closed or stopped. Never fails: a write
    /// error logs one warning and stops the trace, which is then left without
    /// an `end` line.
    pub fn record(&self, event: TraceEvent) {
        let mut guard = self.lock();
        let state = &mut *guard;
        if state.finished || state.sink.is_none() {
            return;
        }
        let line = state.count(event);
        if state.truncated {
            return;
        }
        if let Err(err) = state.write_event(&line, self.max_bytes) {
            state.discard();
            self.warn_stopped(&err);
        }
    }

    /// Closes the trace: writes the `end` line with `status` and the counts,
    /// flushes the buffer and closes the file.
    ///
    /// Only the first call writes; later calls, and every
    /// [`record`](Self::record) after it, do nothing. Never fails: a write
    /// error logs one warning, as in `record`.
    pub fn finish(&self, status: RunStatus) {
        let mut guard = self.lock();
        // Read under the lock, after any `record` that held it: `end` is not
        // stamped before a call whose line it follows.
        let t = self.now_ns();
        let state = &mut *guard;
        if state.finished {
            return;
        }
        state.finished = true;
        if let Err(err) = state.close(status, t) {
            self.warn_stopped(&err);
        }
    }

    /// Nanoseconds since the tracer was created: the clock of every `t`.
    pub(crate) fn now_ns(&self) -> u64 {
        u64::try_from(self.t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// The index of a new planner batch, its `k`: 0 for the run's first batch,
    /// then 1, 2 and so on.
    pub(crate) fn begin_batch(&self) -> u64 {
        self.next_batch.fetch_add(1, Ordering::Relaxed)
    }

    /// A tracer over `sink` with its header line written and flushed. If that
    /// fails the sink is dropped unwritten, so the tracer records nothing and
    /// closes nothing, and the error comes back with it.
    fn start(
        sink: Box<dyn Write + Send>,
        header: TraceHeader,
        max_bytes: u64,
    ) -> (Self, io::Result<()>) {
        let t0 = Instant::now();
        let mut state = State {
            sink: Some(BufWriter::with_capacity(BUFFER_BYTES, sink)),
            line: Vec::new(),
            written: 0,
            batches: 0,
            waves: 0,
            circuits: 0,
            truncated: false,
            finished: false,
        };
        let head = HeaderLine {
            v: SCHEMA_VERSION,
            ev: "run",
            header: &header,
            t0_unix_ns: unix_time_ns(),
        };
        let started = state.encode(&head).and_then(|()| state.write_line(true));
        if started.is_err() {
            state.discard();
        }
        let tracer = Self {
            id: header.id,
            t0,
            max_bytes,
            next_batch: AtomicU64::new(0),
            state: Mutex::new(state),
        };
        (tracer, started)
    }

    /// The state, even if a panic poisoned the mutex: recording never panics.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The one warning of a trace stopped by a write error.
    fn warn_stopped(&self, err: &io::Error) {
        log::warn!("run {}: call trace stopped: {err}", self.id);
    }
}

impl fmt::Debug for CallTracer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallTracer")
            .field("id", &self.id)
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for CallTracer {
    /// Closes a trace nobody finished as [`RunStatus::Aborted`]. Never panics:
    /// a failure is logged with `log::error!`.
    fn drop(&mut self) {
        let t = self.now_ns();
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if state.finished {
            return;
        }
        state.finished = true;
        if let Err(err) = state.close(RunStatus::Aborted, t) {
            log::error!("run {}: call trace not closed: {err}", self.id);
        }
    }
}

/// What a tracer's mutex guards.
struct State {
    /// Where the lines go; `None` once the trace is closed or stopped.
    sink: Option<Sink>,
    /// The line being written, reused from one line to the next.
    line: Vec<u8>,
    /// Bytes written so far, the header's included.
    written: u64,
    /// Batch events recorded.
    batches: u64,
    /// Wave events recorded.
    waves: u64,
    /// Circuits in the batch events recorded.
    circuits: u64,
    /// The size cap left an event out: no other event is written.
    truncated: bool,
    /// The `end` line was due: nothing else is written.
    finished: bool,
}

impl State {
    /// Counts `event` for the `end` line and turns it into its line, adding
    /// what the tracer knows (an `iter`'s `batches`).
    fn count(&mut self, event: TraceEvent) -> Line {
        match event {
            TraceEvent::Batch {
                k,
                op,
                n,
                shots,
                t,
                dt,
                ok,
            } => {
                self.batches = self.batches.saturating_add(1);
                let circuits = u64::try_from(n).unwrap_or(u64::MAX);
                self.circuits = self.circuits.saturating_add(circuits);
                Line::Batch {
                    k,
                    op,
                    n,
                    shots,
                    t,
                    dt,
                    ok,
                }
            }
            TraceEvent::Wave {
                k,
                w,
                op,
                n,
                shots,
                t,
                dt,
                ok,
                replicas,
            } => {
                self.waves = self.waves.saturating_add(1);
                Line::Wave {
                    k,
                    w,
                    op,
                    n,
                    shots,
                    t,
                    dt,
                    ok,
                    replicas,
                }
            }
            TraceEvent::Iter { i, evals, best, t } => Line::Iter {
                i,
                evals,
                batches: self.batches,
                best,
                t,
            },
            TraceEvent::Metric { n, t, dt } => Line::Metric { n, t, dt },
        }
    }

    /// Writes an event's line, or truncates the trace instead if the line
    /// would take it past `max_bytes`.
    fn write_event(&mut self, line: &Line, max_bytes: u64) -> io::Result<()> {
        self.encode_line(line)?;
        let len = u64::try_from(self.line.len()).unwrap_or(u64::MAX);
        if self.written.saturating_add(len) > max_bytes {
            self.truncated = true;
            return Ok(());
        }
        self.write_line(false)
    }

    /// Writes the `end` line, flushes the buffer and closes the sink. A
    /// stopped trace stays without one.
    fn close(&mut self, status: RunStatus, t: u64) -> io::Result<()> {
        if self.sink.is_none() {
            return Ok(());
        }
        let end = Line::End {
            status,
            batches: self.batches,
            waves: self.waves,
            circuits: self.circuits,
            truncated: self.truncated,
            t,
        };
        let closed = self.encode_line(&end).and_then(|()| self.write_line(true));
        self.discard();
        closed
    }

    /// Serializes `line` into the reused line buffer, newline included.
    fn encode<T: Serialize>(&mut self, line: &T) -> io::Result<()> {
        self.line.clear();
        serde_json::to_writer(&mut self.line, line)?;
        self.line.push(b'\n');
        Ok(())
    }

    /// Writes `line` into the reused line buffer, newline included, with
    /// [`Line::write_json`]: the bytes [`encode`](Self::encode) writes.
    fn encode_line(&mut self, line: &Line) -> io::Result<()> {
        self.line.clear();
        line.write_json(&mut self.line)?;
        self.line.push(b'\n');
        Ok(())
    }

    /// Writes the encoded line to the sink, and flushes it if `flush`.
    fn write_line(&mut self, flush: bool) -> io::Result<()> {
        let Some(sink) = self.sink.as_mut() else {
            return Ok(());
        };
        sink.write_all(&self.line)?;
        if flush {
            sink.flush()?;
        }
        let len = u64::try_from(self.line.len()).unwrap_or(u64::MAX);
        self.written = self.written.saturating_add(len);
        Ok(())
    }

    /// Drops the sink, closing its file, without writing what is still
    /// buffered: after a write error nothing more reaches the trace.
    fn discard(&mut self) {
        if let Some(sink) = self.sink.take() {
            // `into_parts` hands back the unwritten bytes instead of writing
            // them as dropping the `BufWriter` would.
            drop(sink.into_parts());
        }
    }
}

/// A line after the header, as written by [`Line::write_json`]: the bytes
/// serde_json writes for it with this derive, which is the reference.
#[derive(Serialize)]
#[serde(tag = "ev", rename_all = "snake_case")]
enum Line {
    Batch {
        k: u64,
        op: BatchOp,
        n: usize,
        shots: u32,
        t: u64,
        dt: u64,
        ok: bool,
    },
    Wave {
        k: Option<u64>,
        w: u32,
        op: WaveOp,
        n: usize,
        shots: u32,
        t: u64,
        dt: u64,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        replicas: Option<Vec<u32>>,
    },
    Iter {
        i: usize,
        evals: u64,
        batches: u64,
        best: f64,
        t: u64,
    },
    Metric {
        n: usize,
        t: u64,
        dt: u64,
    },
    End {
        status: RunStatus,
        batches: u64,
        waves: u64,
        circuits: u64,
        truncated: bool,
        t: u64,
    },
}

impl Line {
    /// Appends the line to `out` as JSON: exactly the bytes
    /// `serde_json::to_writer(out, self)` appends.
    ///
    /// The `batch`, `wave`, `iter` and `metric` lines, written on the hot
    /// path, are encoded by hand rather than through serde's map serializer.
    /// As serde writes an internally tagged enum, each is `{"ev":"<tag>"`,
    /// then every field in its declared order as `,"<key>":` and its value,
    /// then `}`; the keys are fixed bytes, and the numbers and booleans are
    /// written by serde_json's own [`CompactFormatter`]. A `usize` is written
    /// as the `u64` serde makes of it, and a `best` that is NaN or infinite
    /// as `null`, as serde_json writes such a float. The `end` line, written
    /// once per trace, goes through serde_json. The match has no wildcard arm
    /// and each event's arm names all its fields, so a new variant or field
    /// does not compile until it is encoded here or handed to serde_json. The
    /// tests `event_lines_encode_as_serde_json_does` and
    /// `event_lines_encode_as_serde_json_does_for_random_values` check that
    /// both write the same bytes.
    fn write_json(&self, out: &mut Vec<u8>) -> io::Result<()> {
        let mut json = CompactFormatter;
        match self {
            Self::Batch {
                k,
                op,
                n,
                shots,
                t,
                dt,
                ok,
            } => {
                out.extend_from_slice(br#"{"ev":"batch","k":"#);
                json.write_u64(out, *k)?;
                out.extend_from_slice(br#","op":"#);
                write_name(out, op.name());
                out.extend_from_slice(br#","n":"#);
                json.write_u64(out, *n as u64)?;
                out.extend_from_slice(br#","shots":"#);
                json.write_u32(out, *shots)?;
                out.extend_from_slice(br#","t":"#);
                json.write_u64(out, *t)?;
                out.extend_from_slice(br#","dt":"#);
                json.write_u64(out, *dt)?;
                out.extend_from_slice(br#","ok":"#);
                json.write_bool(out, *ok)?;
            }
            Self::Wave {
                k,
                w,
                op,
                n,
                shots,
                t,
                dt,
                ok,
                replicas,
            } => {
                out.extend_from_slice(br#"{"ev":"wave","k":"#);
                match k {
                    Some(k) => json.write_u64(out, *k)?,
                    None => json.write_null(out)?,
                }
                out.extend_from_slice(br#","w":"#);
                json.write_u32(out, *w)?;
                out.extend_from_slice(br#","op":"#);
                write_name(out, op.name());
                out.extend_from_slice(br#","n":"#);
                json.write_u64(out, *n as u64)?;
                out.extend_from_slice(br#","shots":"#);
                json.write_u32(out, *shots)?;
                out.extend_from_slice(br#","t":"#);
                json.write_u64(out, *t)?;
                out.extend_from_slice(br#","dt":"#);
                json.write_u64(out, *dt)?;
                out.extend_from_slice(br#","ok":"#);
                json.write_bool(out, *ok)?;
                // Left out when `None`, as `skip_serializing_if` does.
                if let Some(replicas) = replicas {
                    out.extend_from_slice(br#","replicas":["#);
                    if let Some((first, rest)) = replicas.split_first() {
                        json.write_u32(out, *first)?;
                        for shots in rest {
                            out.push(b',');
                            json.write_u32(out, *shots)?;
                        }
                    }
                    out.push(b']');
                }
            }
            Self::Iter {
                i,
                evals,
                batches,
                best,
                t,
            } => {
                out.extend_from_slice(br#"{"ev":"iter","i":"#);
                json.write_u64(out, *i as u64)?;
                out.extend_from_slice(br#","evals":"#);
                json.write_u64(out, *evals)?;
                out.extend_from_slice(br#","batches":"#);
                json.write_u64(out, *batches)?;
                out.extend_from_slice(br#","best":"#);
                if best.is_finite() {
                    json.write_f64(out, *best)?;
                } else {
                    json.write_null(out)?;
                }
                out.extend_from_slice(br#","t":"#);
                json.write_u64(out, *t)?;
            }
            Self::Metric { n, t, dt } => {
                out.extend_from_slice(br#"{"ev":"metric","n":"#);
                json.write_u64(out, *n as u64)?;
                out.extend_from_slice(br#","t":"#);
                json.write_u64(out, *t)?;
                out.extend_from_slice(br#","dt":"#);
                json.write_u64(out, *dt)?;
            }
            // Once per trace: serde_json writes it whole.
            Self::End { .. } => return serde_json::to_writer(out, self).map_err(io::Error::from),
        }
        out.push(b'}');
        Ok(())
    }
}

/// Appends `name`, an op's name, as a JSON string: it holds nothing to
/// escape, so serde_json writes it between quotes as it is.
fn write_name(out: &mut Vec<u8>, name: &str) {
    out.push(b'"');
    out.extend_from_slice(name.as_bytes());
    out.push(b'"');
}

/// The header line: the schema version and event name, the header's fields,
/// and the wall-clock zero.
#[derive(Serialize)]
struct HeaderLine<'a> {
    v: u32,
    ev: &'static str,
    #[serde(flatten)]
    header: &'a TraceHeader,
    t0_unix_ns: u64,
}

/// Nanoseconds since the Unix epoch; 0 if the clock is set before 1970.
fn unix_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        })
}

mod decorators;

pub use decorators::{TracingBackend, TracingObserver, TracingPlanner, TracingVariance};

#[cfg(test)]
impl CallTracer {
    /// A tracer writing to `sink`. It starts as [`create`](Self::create) does,
    /// except that a failure to write the header stops the trace as any later
    /// write failure does, instead of being returned.
    pub(crate) fn from_sink(
        sink: Box<dyn Write + Send>,
        header: TraceHeader,
        max_bytes: u64,
    ) -> Arc<Self> {
        let (tracer, started) = Self::start(sink, header, max_bytes);
        if let Err(err) = started {
            tracer.warn_stopped(&err);
        }
        Arc::new(tracer)
    }
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;
    use std::time::Instant;

    use serde_json::Value;

    use super::test_support::{
        header, scratch_dir, sink_tracer, FailingSink, FlakySink, SharedSink,
    };
    use super::*;

    fn batch(k: u64) -> TraceEvent {
        TraceEvent::Batch {
            k,
            op: BatchOp::Evaluate,
            n: 6,
            shots: 500,
            t: k * 10,
            dt: 5,
            ok: true,
        }
    }

    /// The length of the header line [`sink_tracer`] writes for `id`, its
    /// newline included.
    fn header_len(id: &str) -> usize {
        let probe = SharedSink::default();
        sink_tracer(&probe, id).finish(RunStatus::Ok);
        probe.text_lines()[0].len() + 1
    }

    #[test]
    fn header_and_events_serialize_to_schema_v1_lines() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "t1");
        tracer.record(TraceEvent::Wave {
            k: Some(0),
            w: 0,
            op: WaveOp::RunShotsDistributed,
            n: 3,
            shots: 10,
            t: 11,
            dt: 2,
            ok: true,
            replicas: Some(vec![4, 3, 3]),
        });
        tracer.record(TraceEvent::Batch {
            k: 0,
            op: BatchOp::Evaluate,
            n: 30,
            shots: 500,
            t: 10,
            dt: 5,
            ok: true,
        });
        tracer.record(TraceEvent::Wave {
            k: None,
            w: 1,
            op: WaveOp::RunCircuits,
            n: 1,
            shots: 10,
            t: 12,
            dt: 1,
            ok: false,
            replicas: None,
        });
        tracer.record(TraceEvent::Iter {
            i: 1,
            evals: 30,
            best: -0.5,
            t: 20,
        });
        tracer.record(TraceEvent::Iter {
            i: 2,
            evals: 31,
            best: f64::NAN,
            t: 21,
        });
        tracer.record(TraceEvent::Metric { n: 2, t: 22, dt: 1 });
        tracer.finish(RunStatus::Ok);
        let lines = sink.text_lines();
        assert_eq!(lines.len(), 8);
        assert!(lines[0].starts_with(concat!(
            r#"{"v":1,"ev":"run","flow":"train","id":"t1","polypus":"0.0.0-test","#,
            r#""method":"spsa","dims":2,"backend":"hash","infrastructure":"local","#,
            r#""n_qpus":1,"shots":500,"seed":2024,"t0_unix_ns":"#
        )));
        let head: Value = serde_json::from_str(&lines[0]).unwrap();
        assert!(head["t0_unix_ns"].as_u64().unwrap() > 1_600_000_000_000_000_000);
        assert_eq!(
            lines[1..7],
            [
                r#"{"ev":"wave","k":0,"w":0,"op":"run_shots_distributed","n":3,"shots":10,"t":11,"dt":2,"ok":true,"replicas":[4,3,3]}"#,
                r#"{"ev":"batch","k":0,"op":"evaluate","n":30,"shots":500,"t":10,"dt":5,"ok":true}"#,
                r#"{"ev":"wave","k":null,"w":1,"op":"run_circuits","n":1,"shots":10,"t":12,"dt":1,"ok":false}"#,
                r#"{"ev":"iter","i":1,"evals":30,"batches":1,"best":-0.5,"t":20}"#,
                r#"{"ev":"iter","i":2,"evals":31,"batches":1,"best":null,"t":21}"#,
                r#"{"ev":"metric","n":2,"t":22,"dt":1}"#,
            ]
        );
        // The `end` line as text too, so the order of its fields is pinned.
        let end: Value = serde_json::from_str(&lines[7]).unwrap();
        let t = end["t"].as_u64().unwrap();
        assert_eq!(
            lines[7],
            format!(
                r#"{{"ev":"end","status":"ok","batches":1,"waves":2,"circuits":30,"truncated":false,"t":{t}}}"#
            )
        );
        assert_eq!(
            serde_json::to_string(&TraceFlow::RunCircuit).unwrap(),
            r#""run_circuit""#
        );
        let statuses = [
            RunStatus::Ok,
            RunStatus::Error,
            RunStatus::Cancelled,
            RunStatus::Aborted,
        ]
        .map(|s| serde_json::to_string(&s).unwrap());
        assert_eq!(
            statuses,
            [r#""ok""#, r#""error""#, r#""cancelled""#, r#""aborted""#]
        );
    }

    /// Encodes `line` into `state`'s line buffer and checks that the buffer
    /// then holds exactly the bytes serde_json writes for it, and a newline.
    fn assert_encodes_as_serde_json(state: &mut State, line: &Line) {
        state.encode_line(line).unwrap();
        let mut serde = serde_json::to_vec(line).unwrap();
        serde.push(b'\n');
        assert!(
            state.line == serde,
            "encoded\n{}\nserde_json writes\n{}",
            String::from_utf8_lossy(&state.line),
            String::from_utf8_lossy(&serde)
        );
    }

    #[test]
    fn event_lines_encode_as_serde_json_does() {
        // The comparisons below check the encoder's op names against serde's;
        // these pin serde's names to the ones the trace format documents.
        assert_eq!(
            [
                BatchOp::Evaluate,
                BatchOp::Execute,
                BatchOp::ExecuteReplicas
            ]
            .map(|op| serde_json::to_string(&op).unwrap()),
            [r#""evaluate""#, r#""execute""#, r#""execute_replicas""#]
        );
        assert_eq!(
            [WaveOp::RunCircuits, WaveOp::RunShotsDistributed]
                .map(|op| serde_json::to_string(&op).unwrap()),
            [r#""run_circuits""#, r#""run_shots_distributed""#]
        );
        const U64: [u64; 3] = [0, 1, u64::MAX];
        const USIZE: [usize; 3] = [0, 1, usize::MAX];
        const U32: [u32; 3] = [0, 1, u32::MAX];
        let bests = [
            0.0,
            -0.0,
            0.1,
            1.0 / 3.0,
            1e-7,
            1e15,
            1e16,
            1e17,
            1e21,
            1e22,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            f64::MIN,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut lines = Vec::new();
        // Each field takes the edge values in turn, out of step with the
        // fields next to it.
        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            for ok in [false, true] {
                for op in [
                    BatchOp::Evaluate,
                    BatchOp::Execute,
                    BatchOp::ExecuteReplicas,
                ] {
                    lines.push(Line::Batch {
                        k: U64[a],
                        op,
                        n: USIZE[b],
                        shots: U32[c],
                        t: U64[b],
                        dt: U64[c],
                        ok,
                    });
                }
                for op in [WaveOp::RunCircuits, WaveOp::RunShotsDistributed] {
                    for k in [None, Some(U64[a])] {
                        for replicas in [
                            None,
                            Some(vec![]),
                            Some(vec![U32[a]]),
                            Some(vec![U32[a], U32[b], U32[c], 4_096]),
                        ] {
                            lines.push(Line::Wave {
                                k,
                                w: U32[b],
                                op,
                                n: USIZE[c],
                                shots: U32[a],
                                t: U64[c],
                                dt: U64[b],
                                ok,
                                replicas,
                            });
                        }
                    }
                }
            }
            for best in bests.into_iter().flat_map(|best| [best, -best]) {
                lines.push(Line::Iter {
                    i: USIZE[a],
                    evals: U64[b],
                    batches: U64[c],
                    best,
                    t: U64[a],
                });
            }
            lines.push(Line::Metric {
                n: USIZE[a],
                t: U64[b],
                dt: U64[c],
            });
        }
        // The `end` line is handed to serde_json whole.
        for (status, truncated) in [(RunStatus::Ok, false), (RunStatus::Aborted, true)] {
            lines.push(Line::End {
                status,
                batches: 1,
                waves: u64::MAX,
                circuits: 0,
                truncated,
                t: 7,
            });
        }
        let tracer =
            CallTracer::from_sink(Box::new(io::sink()), header("encode"), DEFAULT_MAX_BYTES);
        let mut state = tracer.lock();
        lines
            .iter()
            .for_each(|line| assert_encodes_as_serde_json(&mut state, line));
    }

    #[test]
    fn event_lines_encode_as_serde_json_does_for_random_values() {
        /// SplitMix64 from a fixed seed: every run checks the same lines.
        struct Rng(u64);

        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let z = self.0;
                let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^ (z >> 31)
            }

            /// An integer of any size: a random `u64` shifted right by 0 to
            /// 63 bits.
            fn int(&mut self) -> u64 {
                let shift = self.next() % 64;
                self.next() >> shift
            }

            /// A `u32` of any size, as [`int`](Self::int).
            fn int32(&mut self) -> u32 {
                let shift = self.next() % 32;
                (self.next() as u32) >> shift
            }

            fn coin(&mut self) -> bool {
                self.next() & 1 == 1
            }
        }

        let ops = [
            BatchOp::Evaluate,
            BatchOp::Execute,
            BatchOp::ExecuteReplicas,
        ];
        let mut rng = Rng(0x7ace_2024);
        let tracer =
            CallTracer::from_sink(Box::new(io::sink()), header("encode"), DEFAULT_MAX_BYTES);
        let mut state = tracer.lock();
        for _ in 0..100_000 {
            let line = match rng.next() % 4 {
                0 => Line::Batch {
                    k: rng.int(),
                    op: ops[(rng.next() % 3) as usize],
                    n: rng.int() as usize,
                    shots: rng.int32(),
                    t: rng.int(),
                    dt: rng.int(),
                    ok: rng.coin(),
                },
                1 => Line::Wave {
                    k: rng.coin().then(|| rng.int()),
                    w: rng.int32(),
                    op: if rng.coin() {
                        WaveOp::RunCircuits
                    } else {
                        WaveOp::RunShotsDistributed
                    },
                    n: rng.int() as usize,
                    shots: rng.int32(),
                    t: rng.int(),
                    dt: rng.int(),
                    ok: rng.coin(),
                    replicas: rng.coin().then(|| {
                        let len = rng.next() % 9;
                        (0..len).map(|_| rng.int32()).collect()
                    }),
                },
                2 => Line::Iter {
                    i: rng.int() as usize,
                    evals: rng.int(),
                    batches: rng.int(),
                    best: f64::from_bits(rng.next()),
                    t: rng.int(),
                },
                _ => Line::Metric {
                    n: rng.int() as usize,
                    t: rng.int(),
                    dt: rng.int(),
                },
            };
            assert_encodes_as_serde_json(&mut state, &line);
        }
    }

    #[test]
    fn create_never_overwrites_a_file_or_follows_a_symlink() {
        let dir = scratch_dir("no-overwrite");
        let taken = dir.join("taken.jsonl");
        std::fs::write(&taken, b"keep\n").unwrap();
        let err =
            CallTracer::create(&taken, header("no-overwrite"), DEFAULT_MAX_BYTES).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&taken).unwrap(), b"keep\n");
        let missing = dir.join("missing").join("t.jsonl");
        let err =
            CallTracer::create(&missing, header("no-overwrite"), DEFAULT_MAX_BYTES).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(
            !dir.join("missing").exists(),
            "the trace never creates a directory"
        );
        #[cfg(unix)]
        {
            let target = dir.join("target.txt");
            std::fs::write(&target, b"target\n").unwrap();
            let link = dir.join("link.jsonl");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            let err =
                CallTracer::create(&link, header("no-overwrite"), DEFAULT_MAX_BYTES).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::AlreadyExists);
            assert_eq!(std::fs::read(&target).unwrap(), b"target\n");
            let dangling = dir.join("dangling.jsonl");
            std::os::unix::fs::symlink(dir.join("nowhere.txt"), &dangling).unwrap();
            assert!(
                CallTracer::create(&dangling, header("no-overwrite"), DEFAULT_MAX_BYTES).is_err()
            );
            assert!(
                !dir.join("nowhere.txt").exists(),
                "a dangling link is not followed"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn trace_file_is_created_0640_or_stricter() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch_dir("file-mode");
        let path = dir.join("mode.jsonl");
        CallTracer::create(&path, header("file-mode"), DEFAULT_MAX_BYTES)
            .unwrap()
            .finish(RunStatus::Ok);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode & !0o640, 0, "mode {mode:o} grants more than 0640");
        assert_eq!(
            mode & 0o600,
            0o600,
            "mode {mode:o}: the owner reads and writes"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn trace_drop_writes_aborted() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "dropped");
        tracer.record(batch(0));
        drop(tracer);
        let lines = sink.lines();
        let end = lines.last().unwrap();
        assert_eq!(
            (end["ev"].as_str(), end["status"].as_str()),
            (Some("end"), Some("aborted"))
        );
    }

    #[test]
    fn drop_failure_logs_one_error() {
        crate::test_log::capture_logs();
        let sink = FailingSink::new(header_len("drop-fails"));
        let tracer = CallTracer::from_sink(
            Box::new(sink.clone()),
            header("drop-fails"),
            DEFAULT_MAX_BYTES,
        );
        drop(tracer);
        let logged = crate::test_log::logged_with_levels_for("drop-fails");
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert_eq!(logged[0].0, log::Level::Error, "{logged:?}");
        assert!(logged[0].1.contains("call trace not closed"), "{logged:?}");
        let accepted = String::from_utf8(sink.accepted()).unwrap();
        assert_eq!(accepted.lines().count(), 1, "the header alone: {accepted}");
        assert!(accepted.starts_with(r#"{"v":1,"ev":"run","#), "{accepted}");
    }

    #[test]
    fn finish_is_written_once() {
        let sink = SharedSink::default();
        let tracer = sink_tracer(&sink, "once");
        tracer.finish(RunStatus::Cancelled);
        tracer.finish(RunStatus::Ok);
        drop(tracer);
        let ends: Vec<Value> = sink
            .lines()
            .into_iter()
            .filter(|l| l["ev"] == "end")
            .collect();
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0]["status"], "cancelled");
    }

    #[test]
    fn finish_stamps_the_end_after_a_record_in_progress() {
        use std::sync::mpsc::{channel, Receiver, Sender};
        use std::thread;
        use std::time::Duration;

        /// Passes every write on to `out`, but holds the second one, the first
        /// after the header, until `release` fires: the `record` making it
        /// keeps the tracer's lock until then.
        struct GateSink {
            out: SharedSink,
            writes: usize,
            entered: Sender<()>,
            release: Receiver<()>,
        }

        impl Write for GateSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                if self.writes == 2 {
                    self.entered.send(()).unwrap();
                    self.release.recv().unwrap();
                }
                self.out.write(buf)
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let out = SharedSink::default();
        let (entered, on_entered) = channel();
        let (release, on_release) = channel();
        let gate = GateSink {
            out: out.clone(),
            writes: 0,
            entered,
            release: on_release,
        };
        let tracer = CallTracer::from_sink(Box::new(gate), header("end-stamp"), DEFAULT_MAX_BYTES);
        // A line longer than the 64 KiB buffer goes straight to the sink.
        let long = TraceEvent::Wave {
            k: None,
            w: 0,
            op: WaveOp::RunShotsDistributed,
            n: 1,
            shots: 40_000,
            t: 0,
            dt: 0,
            ok: true,
            replicas: Some(vec![1; 40_000]),
        };
        let recorder = {
            let tracer = Arc::clone(&tracer);
            thread::spawn(move || tracer.record(long))
        };
        on_entered
            .recv_timeout(Duration::from_secs(60))
            .expect("the long line goes straight to the sink");
        let finisher = {
            let tracer = Arc::clone(&tracer);
            thread::spawn(move || tracer.finish(RunStatus::Ok))
        };
        // Time for `finish` to reach the lock. The check below holds however
        // long that takes: `finish` can only take the lock after the release.
        thread::sleep(Duration::from_millis(50));
        let released = tracer.now_ns();
        release.send(()).unwrap();
        recorder.join().unwrap();
        finisher.join().unwrap();
        let lines = out.lines();
        assert_eq!(lines.len(), 3, "the header, the long wave and the end");
        assert_eq!(lines[2]["ev"], "end");
        let end = lines[2]["t"].as_u64().unwrap();
        assert!(
            end >= released,
            "the end is stamped at {end} ns, before the release at {released} ns"
        );
    }

    #[test]
    fn size_cap_truncates_and_still_closes() {
        let sink = SharedSink::default();
        let tracer = CallTracer::from_sink(Box::new(sink.clone()), header("cap"), 1_000);
        (0..50).for_each(|k| tracer.record(batch(k)));
        tracer.record(TraceEvent::Metric {
            n: 1,
            t: 999,
            dt: 1,
        });
        tracer.finish(RunStatus::Ok);
        let (bytes, lines) = (sink.bytes(), sink.lines());
        let end_len = sink.text_lines().last().unwrap().len() + 1;
        assert!(bytes.len() <= 1_000 + end_len, "{} bytes", bytes.len());
        let end = lines.last().unwrap();
        assert_eq!(
            (end["status"].as_str(), end["truncated"].as_bool()),
            (Some("ok"), Some(true))
        );
        assert_eq!(
            (end["batches"].as_u64(), end["circuits"].as_u64()),
            (Some(50), Some(300))
        );
        let written = lines.iter().filter(|l| l["ev"] == "batch").count();
        assert!((1..50).contains(&written), "{written} batch lines");
        assert!(
            lines.iter().all(|l| l["ev"] != "metric"),
            "nothing is written past the cap"
        );
    }

    #[test]
    fn write_failure_warns_once_and_leaves_no_end() {
        crate::test_log::capture_logs();
        let sink = FailingSink::new(4_096);
        let tracer = CallTracer::from_sink(
            Box::new(sink.clone()),
            header("t-write-fail"),
            DEFAULT_MAX_BYTES,
        );
        (0..2_000).for_each(|k| tracer.record(batch(k)));
        tracer.finish(RunStatus::Ok);
        drop(tracer);
        let logged = crate::test_log::logged_for("t-write-fail");
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(logged[0].contains("call trace stopped"), "{logged:?}");
        let accepted = sink.accepted();
        assert_eq!(accepted.len(), 4_096);
        assert!(!String::from_utf8_lossy(&accepted).contains(r#""ev":"end""#));
    }

    #[test]
    fn a_transient_write_error_writes_nothing_more() {
        crate::test_log::capture_logs();
        let sink = FlakySink::new(header_len("transient-fault"));
        let tracer = CallTracer::from_sink(
            Box::new(sink.clone()),
            header("transient-fault"),
            DEFAULT_MAX_BYTES,
        );
        // More than the 64 KiB buffer holds: the write that empties it fails,
        // and the sink would take any write after that.
        (0..2_000).for_each(|k| tracer.record(batch(k)));
        tracer.finish(RunStatus::Ok);
        drop(tracer);
        let logged = crate::test_log::logged_with_levels_for("transient-fault");
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert_eq!(logged[0].0, log::Level::Warn, "{logged:?}");
        assert!(logged[0].1.contains("call trace stopped"), "{logged:?}");
        let accepted = String::from_utf8(sink.accepted()).unwrap();
        assert_eq!(
            accepted.lines().count(),
            1,
            "the header alone, not {} bytes",
            accepted.len()
        );
    }

    #[test]
    fn worst_case_trace_fits_two_megabytes() {
        // A long run: SPSA with K = 1, the most trace lines per evaluation, at
        // d = 30 and a budget of 10,000 evaluations runs
        // spsa_iterations(10000, 1) = 4853 iterations
        // (benchmarks/bench_optimizers.py): a calibration batch of 50, one batch
        // per iteration (the current point joins it every 20th iteration), then
        // the final point.
        fn batch_and_wave(tracer: &CallTracer, k: &mut u64, n: usize) {
            let t = *k * 1_000_000_000;
            tracer.record(TraceEvent::Wave {
                k: Some(*k),
                w: 0,
                op: WaveOp::RunCircuits,
                n,
                shots: 8192,
                t,
                dt: 999_999_999,
                ok: true,
                replicas: None,
            });
            tracer.record(TraceEvent::Batch {
                k: *k,
                op: BatchOp::Evaluate,
                n,
                shots: 8192,
                t,
                dt: 999_999_999,
                ok: true,
            });
            *k += 1;
        }
        let sink = SharedSink::default();
        let head = TraceHeader {
            dims: Some(30),
            seed: Some(u64::MAX >> 1),
            shots: 8192,
            ..header(&format!("{}_{}", "a".repeat(64), "f".repeat(36)))
        };
        let tracer = CallTracer::from_sink(Box::new(sink.clone()), head, DEFAULT_MAX_BYTES);
        let (mut k, mut evals) = (0, 50);
        batch_and_wave(&tracer, &mut k, 50);
        for i in 1..=4853usize {
            let n = if i % 20 == 1 { 3 } else { 2 };
            batch_and_wave(&tracer, &mut k, n);
            evals += n as u64;
            tracer.record(TraceEvent::Iter {
                i,
                evals,
                best: -12.345_678_901_234_567,
                t: k * 1_000_000_000,
            });
        }
        batch_and_wave(&tracer, &mut k, 1);
        tracer.finish(RunStatus::Ok);
        let (bytes, lines) = (sink.bytes().len(), sink.text_lines().len());
        println!("worst case: {lines} lines, {bytes} bytes");
        assert!((14_000..15_000).contains(&lines), "{lines} lines");
        assert!(bytes <= 2_000_000, "{bytes} bytes");
    }

    #[test]
    #[ignore = "microbenchmark: cargo test --release ... -- --ignored --nocapture"]
    fn record_costs_under_two_microseconds() {
        let tracer = CallTracer::from_sink(
            Box::new(std::io::sink()),
            header("record-cost"),
            DEFAULT_MAX_BYTES,
        );
        let wave = |t| TraceEvent::Wave {
            k: Some(t),
            w: 3,
            op: WaveOp::RunCircuits,
            n: 16,
            shots: 1024,
            t,
            dt: 123_456_789,
            ok: true,
            replicas: None,
        };
        (0..10_000).for_each(|t| tracer.record(wave(t)));
        let mut per_event: Vec<f64> = (0..101u64)
            .map(|s| {
                let start = Instant::now();
                (0..1_000).for_each(|t| tracer.record(wave(s * 1_000 + t)));
                start.elapsed().as_secs_f64() / 1_000.0
            })
            .collect();
        per_event.sort_by(f64::total_cmp);
        println!("record: median {:.0} ns per event", per_event[50] * 1e9);
        assert!(per_event[50] < 2e-6);
    }
}
