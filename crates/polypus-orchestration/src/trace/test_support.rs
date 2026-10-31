//! Sinks, a header, run parameters and a tracer shared by the call trace's
//! unit tests.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use polypus_infrastructure::{OptLevel, RunParams};
use serde_json::Value;

use super::{CallTracer, TraceFlow, TraceHeader, DEFAULT_MAX_BYTES};

/// An in-memory sink whose clones share one buffer: a test hands a clone to
/// the tracer and reads back what it wrote.
#[derive(Clone, Default)]
pub(crate) struct SharedSink(Arc<Mutex<Vec<u8>>>);

impl SharedSink {
    /// Every byte written so far.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The complete lines written so far, without their newlines. Text after
    /// the last newline is a line still being written and is left out.
    pub(crate) fn text_lines(&self) -> Vec<String> {
        let text = String::from_utf8(self.bytes()).expect("a trace is UTF-8");
        let mut lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
        lines.pop();
        lines
    }

    /// The complete lines, each parsed as JSON.
    pub(crate) fn lines(&self) -> Vec<Value> {
        self.text_lines()
            .iter()
            .map(|line| serde_json::from_str(line).expect("every trace line is JSON"))
            .collect()
    }
}

impl Write for SharedSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A sink that fails like a full disk: it accepts `budget` bytes in all, the
/// write that crosses the budget is accepted in part, and every write after
/// that fails with "disk full". Its clones share what it accepted.
#[derive(Clone)]
pub(crate) struct FailingSink {
    accepted: Arc<Mutex<Vec<u8>>>,
    budget: usize,
}

impl FailingSink {
    /// A sink that accepts `budget` bytes before it fails.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            accepted: Arc::default(),
            budget,
        }
    }

    /// The bytes accepted before the sink started failing.
    pub(crate) fn accepted(&self) -> Vec<u8> {
        self.accepted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Write for FailingSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut accepted = self.accepted.lock().unwrap_or_else(PoisonError::into_inner);
        let room = self.budget.saturating_sub(accepted.len());
        if room == 0 {
            return Err(io::Error::other("disk full"));
        }
        let taken = buf.len().min(room);
        accepted.extend_from_slice(&buf[..taken]);
        Ok(taken)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A sink with one transient fault: once it holds `budget` bytes, its next
/// write fails with "disk full", and every write after that is accepted, so
/// anything written after the fault would show. Its clones share its state.
#[derive(Clone)]
pub(crate) struct FlakySink {
    /// The bytes accepted, and whether the fault has happened.
    state: Arc<Mutex<(Vec<u8>, bool)>>,
    budget: usize,
}

impl FlakySink {
    /// A sink that fails once, on the first write after `budget` bytes.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            state: Arc::default(),
            budget,
        }
    }

    /// Every byte the sink accepted, before the fault and after it.
    pub(crate) fn accepted(&self) -> Vec<u8> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .0
            .clone()
    }
}

impl Write for FlakySink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let (accepted, failed) = &mut *state;
        if accepted.len() >= self.budget && !*failed {
            *failed = true;
            return Err(io::Error::other("disk full"));
        }
        accepted.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The header of a two-parameter SPSA training run with id `id` on the local
/// `hash` backend, with the shots and seed of [`params`].
pub(crate) fn header(id: &str) -> TraceHeader {
    let run = params();
    TraceHeader {
        flow: TraceFlow::Train,
        id: id.to_owned(),
        polypus: "0.0.0-test".to_owned(),
        method: Some("spsa"),
        dims: Some(2),
        backend: "hash".to_owned(),
        infrastructure: "local".to_owned(),
        n_qpus: 1,
        shots: run.shots,
        seed: run.seed,
    }
}

/// The run parameters the traced test runs use: 500 shots, seed 2024.
pub(crate) fn params() -> RunParams {
    RunParams {
        id: "trace-test".to_owned(),
        shots: 500,
        seed: Some(2024),
        opt_level: OptLevel::default(),
    }
}

/// A tracer writing to `sink`, with [`header`]`(id)` and the default size cap.
pub(crate) fn sink_tracer(sink: &SharedSink, id: &str) -> Arc<CallTracer> {
    CallTracer::from_sink(Box::new(sink.clone()), header(id), DEFAULT_MAX_BYTES)
}

/// A new, empty directory `polypus-trace-{tag}-{pid}` in the system's temp
/// directory. Whatever is already there, such as a leftover of an earlier
/// process with the same pid, is removed first, so each test needs its own
/// `tag`: two tests sharing one would remove each other's files. The test
/// removes the directory once it is done.
pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("polypus-trace-{tag}-{}", std::process::id()));
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => panic!("cannot clear {}: {err}", dir.display()),
    }
    std::fs::create_dir(&dir).expect("create the scratch directory");
    dir
}
