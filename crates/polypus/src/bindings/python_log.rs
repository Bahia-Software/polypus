//! The bridge from Rust `log` records to Python's `logging`, installed by
//! `polypus.log_to_python()` (see `logging.rs`).
//!
//! Each record goes to `logging.getLogger(<name>)`, where the name is the
//! record's target (its module path) under a `polypus.` hierarchy
//! ([`logger_name`]): `polypus_optimizers::util` becomes
//! `polypus.optimizers.util`. The Python application then configures them like
//! any other library's loggers, and `logging.getLogger("polypus")` controls all
//! of them.
//!
//! Records are emitted from every thread, most of them detached: rayon workers
//! and the scheduler's threads run inside `py.detach`. So the bridge holds to
//! two rules.
//!
//! - **A disabled record never takes the GIL.** Which records Python wants is
//!   read from Python ([`PythonLogBridge::refresh`]) only on a thread that
//!   already holds the GIL: when the bridge is installed, and at the start of
//!   every entry point that runs code that logs on worker threads
//!   ([`refresh_levels`]). Records are filtered against that snapshot, so a
//!   level changed in Python in the middle of a call is seen from the next
//!   call on. A record that passes is then checked once more against the live
//!   logger, while attached, before it is handled.
//! - **Delivering never panics and never hangs on an unavailable
//!   interpreter.** The bridge attaches with [`attach_or`]: when Python cannot
//!   be reached (shutting down, inside a GC traversal), an `ERROR` or
//!   `WARNING` record is written to stderr as one fixed-format line
//!   ([`write_unavailable`]) and anything less severe is dropped. A Python
//!   exception raised while handing a record over is reported through
//!   `sys.unraisablehook` and never propagates; one raised by a handler is
//!   already reported by `logging.Handler.handleError`.
//!
//! An enabled record does wait for the GIL, so no record may be emitted while
//! holding a lock that a thread holding the GIL can wait for
//! (docs/ENGINEERING.md §9).

use std::fmt::{self, Write as _};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyModule, PyString, PyTuple};

use crate::infrastructure::attach_or;

/// The bridge [`install`] registers as the process's `log` sink.
static BRIDGE: PythonLogBridge = PythonLogBridge::new();

/// Whether [`BRIDGE`] is the installed sink: [`refresh_levels`] is a no-op
/// until it is.
static BRIDGE_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Every Rust level, most severe first, with the Python level it maps to.
/// `TRACE` has no Python name; it is `5`, below `logging.DEBUG`.
const LEVELS: [(Level, i32); 5] = [
    (Level::Error, 40),
    (Level::Warn, 30),
    (Level::Info, 20),
    (Level::Debug, 10),
    (Level::Trace, 5),
];

/// The Python level of `level` (see [`LEVELS`]).
fn python_level(level: Level) -> i32 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

/// The bit of `level` in a [`Levels`] mask.
fn bit(level: Level) -> u8 {
    1 << (level as u8 - 1)
}

/// The Python logger name for a record whose target is `target`.
///
/// `::` becomes `.`, and the result is placed under `polypus`: a first segment
/// `polypus_<x>` (a workspace crate) becomes `polypus.<x>`, a first segment
/// `polypus` (this crate) stays as it is, and any other first segment (a
/// dependency's, such as `zeromq` with the `qmio` feature) is prefixed with
/// `polypus.`. So `polypus_optimizers::util` is `polypus.optimizers.util`,
/// `polypus::bindings::qml` is `polypus.bindings.qml`, `zeromq::codec` is
/// `polypus.zeromq.codec`, and an empty target is `polypus`. A first segment
/// that is exactly `polypus_` names no crate and is prefixed like a foreign
/// one: `polypus.polypus_`.
pub(crate) fn logger_name(target: &str) -> String {
    let mut segments = target.split("::");
    let first = segments.next().unwrap_or_default();
    let mut name = String::with_capacity(target.len() + "polypus.".len());
    name.push_str("polypus");
    match first.strip_prefix("polypus_") {
        Some(krate) if !krate.is_empty() => {
            name.push('.');
            name.push_str(krate);
        }
        _ if first == "polypus" || first.is_empty() => {}
        _ => {
            name.push('.');
            name.push_str(first);
        }
    }
    for segment in segments {
        name.push('.');
        name.push_str(segment);
    }
    name
}

/// Which Rust levels one Python logger lets through, as bit masks over
/// [`bit`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Levels {
    /// The logger's own answer, `isEnabledFor`.
    own: u8,
    /// What a new child of the logger answers: its effective level and the
    /// global `logging.disable` level, without the logger's own `disabled`
    /// flag, which a child does not inherit.
    inherited: u8,
}

/// A snapshot of which levels Python wants, per `polypus` logger.
#[derive(Debug, Default)]
struct LevelCache {
    /// Every `logging.Logger` named `polypus` or `polypus.*` at the time of
    /// the snapshot (placeholders for names nobody created are skipped).
    loggers: std::collections::HashMap<String, Levels>,
}

impl LevelCache {
    /// Whether the logger `name` lets `level` through: its own answer if it
    /// existed at the snapshot, else what a new logger under its nearest
    /// existing ancestor would answer. With no such ancestor (no snapshot
    /// taken), nothing is enabled.
    fn enabled(&self, name: &str, level: Level) -> bool {
        let bit = bit(level);
        if let Some(levels) = self.loggers.get(name) {
            return levels.own & bit != 0;
        }
        let mut prefix = name;
        while let Some(dot) = prefix.rfind('.') {
            prefix = &prefix[..dot];
            if let Some(levels) = self.loggers.get(prefix) {
                return levels.inherited & bit != 0;
            }
        }
        false
    }

    /// The most verbose level any logger lets through, for
    /// `log::set_max_level`: the `log` macros skip anything above it without
    /// calling the sink at all.
    fn max_level(&self) -> LevelFilter {
        let any = self
            .loggers
            .values()
            .fold(0, |mask, levels| mask | levels.own | levels.inherited);
        LEVELS
            .iter()
            .rev()
            .find(|(level, _)| any & bit(*level) != 0)
            .map_or(LevelFilter::Off, |(level, _)| level.to_level_filter())
    }
}

/// A `log` sink that forwards records to Python's `logging` (see the module
/// docs). The process installs one, [`BRIDGE`]; tests build their own and call
/// [`Log::log`] on it directly.
pub(crate) struct PythonLogBridge {
    /// The level snapshot in force, with the ticket of the refresh that took
    /// it. Read by cloning the `Arc` and releasing the guard at once, so the
    /// lock is never held while attaching or calling Python.
    levels: RwLock<Snapshot>,
    /// The last ticket handed out by [`refresh`](Self::refresh); 0 = none.
    tickets: AtomicU64,
    /// The last walk of the logger dictionary, which the next refresh reuses
    /// if the dictionary has not changed. A refresh takes it out and puts its
    /// own back, so the lock is never held while calling Python; two refreshes
    /// at once simply both walk.
    walk: Mutex<Option<Walk>>,
}

/// A level snapshot and the ticket of the refresh that took it.
struct Snapshot {
    /// 0 until the first snapshot is stored.
    ticket: u64,
    /// `None` until the first [`PythonLogBridge::refresh`].
    levels: Option<Arc<LevelCache>>,
}

impl PythonLogBridge {
    pub(crate) const fn new() -> Self {
        Self {
            levels: RwLock::new(Snapshot {
                ticket: 0,
                levels: None,
            }),
            tickets: AtomicU64::new(0),
            walk: Mutex::new(None),
        }
    }

    /// The current level snapshot. The guard is released before this returns.
    fn snapshot(&self) -> Option<Arc<LevelCache>> {
        self.levels
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .levels
            .clone()
    }

    /// Read the levels from Python into a new snapshot. Returns the most
    /// verbose level any `polypus` logger lets through when the snapshot was
    /// stored, and `None` when a newer one was already in force (see
    /// [`store`](Self::store)); with `apply_max_level`, that level also
    /// becomes `log::set_max_level`'s.
    ///
    /// The ticket is taken before the levels are read: reading them runs
    /// Python code, which may hand the GIL to another thread that refreshes
    /// meanwhile, and without the ticket whichever finished last would win,
    /// possibly with the older reading. With it, the refresh that started
    /// last is the one that stays.
    pub(crate) fn refresh(
        &self,
        py: Python<'_>,
        apply_max_level: bool,
    ) -> PyResult<Option<LevelFilter>> {
        let ticket = self.tickets.fetch_add(1, Ordering::Relaxed) + 1;
        let previous = self
            .walk
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let (cache, walk) = read_levels(py, previous)?;
        let mut slot = self.walk.lock().unwrap_or_else(PoisonError::into_inner);
        let replaced = slot.replace(walk);
        drop(slot);
        // Dropped after the guard: releasing Python objects can run Python
        // code, which must not happen under a lock (docs/ENGINEERING.md §9).
        drop(replaced);
        Ok(self.store(ticket, cache, apply_max_level))
    }

    /// Put `cache` in force if `ticket` is newer than the snapshot's, and
    /// return its most verbose level (also set as `log::set_max_level`'s,
    /// under the same guard, with `apply_max_level`); otherwise drop it and
    /// return `None`, keeping the newer snapshot.
    fn store(&self, ticket: u64, cache: LevelCache, apply_max_level: bool) -> Option<LevelFilter> {
        let mut snapshot = self.levels.write().unwrap_or_else(PoisonError::into_inner);
        if ticket <= snapshot.ticket {
            return None;
        }
        let max = cache.max_level();
        *snapshot = Snapshot {
            ticket,
            levels: Some(Arc::new(cache)),
        };
        if apply_max_level {
            log::set_max_level(max);
        }
        Some(max)
    }

    /// Whether the snapshot lets a record of `level` for the logger `name`
    /// through. Never attaches.
    fn wants(&self, name: &str, level: Level) -> bool {
        self.snapshot()
            .is_some_and(|levels| levels.enabled(name, level))
    }
}

impl Log for PythonLogBridge {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.wants(&logger_name(metadata.target()), metadata.level())
    }

    fn log(&self, record: &Record<'_>) {
        let level = record.level();
        let name = logger_name(record.target());
        if !self.wants(&name, level) {
            return;
        }
        let message = format_message(record.args());
        let file = record.file();
        let line = record.line();
        attach_or(
            || write_unavailable(level, &name, &message),
            |py| {
                if let Err(err) = deliver(py, &name, level, file, line, &message) {
                    err.write_unraisable(py, None);
                }
            },
        );
    }

    fn flush(&self) {}
}

/// The record's message, formatted without panicking: `to_string()` panics
/// when a `Display` impl returns an error, so a failed format gives a fixed
/// text instead.
fn format_message(args: &fmt::Arguments<'_>) -> String {
    if let Some(text) = args.as_str() {
        return text.to_owned();
    }
    let mut message = String::new();
    match write!(message, "{args}") {
        Ok(()) => message,
        Err(fmt::Error) => "<unformattable log message>".to_owned(),
    }
}

/// Hand one record to the Python logger `name`, if it still wants it. The
/// record is built with `makeRecord`, so its `pathname` and `lineno` are the
/// Rust source location and not a Python frame's, and with no `args`, so a
/// `%` in the message is never taken for a format placeholder.
fn deliver(
    py: Python<'_>,
    name: &str,
    level: Level,
    file: Option<&str>,
    line: Option<u32>,
    message: &str,
) -> PyResult<()> {
    let logger = get_logger(py, name)?;
    let levelno = python_level(level);
    if !logger
        .call_method1("isEnabledFor", (levelno,))?
        .is_truthy()?
    {
        return Ok(());
    }
    let record = logger.call_method1(
        "makeRecord",
        (
            name,
            levelno,
            file.unwrap_or("(unknown file)"),
            line.unwrap_or(0),
            message,
            PyTuple::empty(py),
            py.None(),
        ),
    )?;
    logger.call_method1("handle", (record,))?;
    Ok(())
}

/// `logging.getLogger(name)`. The function is looked up once per process: a
/// record is delivered on every call, possibly in a hot loop, and importing
/// `logging` each time is a needless cost. (`PyOnceLock` detaches before it
/// waits for another thread's initialisation, and the initialisation logs
/// nothing.)
fn get_logger<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    static GET_LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    GET_LOGGER
        .get_or_try_init(py, || {
            PyModule::import(py, "logging")?
                .getattr("getLogger")
                .map(Bound::unbind)
        })?
        .bind(py)
        .call1((name,))
}

/// What happens to a record when Python cannot be reached: an `ERROR` or
/// `WARN` record becomes one line on stderr,
/// `polypus: <LEVEL> <logger name>: <message>` (with Python's level name,
/// `ERROR` or `WARNING`); anything less severe is dropped. A failed write is
/// ignored, since there is nowhere left to report it.
fn write_unavailable(level: Level, name: &str, message: &str) {
    let label = match level {
        Level::Error => "ERROR",
        Level::Warn => "WARNING",
        Level::Info | Level::Debug | Level::Trace => return,
    };
    let _ = writeln!(
        std::io::stderr().lock(),
        "polypus: {label} {name}: {message}"
    );
}

/// Read, for every `polypus` logger, which levels it lets through. Returns
/// the snapshot and the walk of the logger dictionary it was read from, which
/// the next call gets back as `previous`.
///
/// `logging.getLogger("polypus")` is created first, so every name under it
/// has an ancestor in the snapshot. The logger dictionary holds every logger
/// of the process, hundreds in an application that imports a few libraries,
/// and this runs at the start of every entry point, so it is walked only when
/// it has changed since `previous` ([`Walk::is_current`]); otherwise only the
/// `polypus` loggers found then are asked again.
fn read_levels(py: Python<'_>, previous: Option<Walk>) -> PyResult<(LevelCache, Walk)> {
    let logging = PyModule::import(py, "logging")?;
    let logger_class = logging.getattr("Logger")?;
    let is_enabled_for = logger_class.getattr("isEnabledFor")?;
    let top = get_logger(py, "polypus")?;
    let manager = top.getattr("manager")?;
    let disable: i32 = manager.getattr("disable")?.extract()?;
    let live = manager.getattr("loggerDict")?.cast_into::<PyDict>()?;
    let walk = match previous {
        Some(walk) if walk.is_current(&live)? => walk,
        // The copy is walked, not the live dictionary: nothing else holds the
        // copy, so it cannot change under the walk.
        _ => Walk::new(&live.copy()?, &logger_class)?,
    };
    // The calls below run Python code, which may let another thread create a
    // logger; the walk is ours and does not change, and the next refresh sees
    // the new logger.
    let mut cache = LevelCache::default();
    for entry in walk.entries.iter().filter(|entry| entry.is_logger) {
        let levels = levels_of(entry.object.bind(py), disable, &is_enabled_for)?;
        cache.loggers.insert(entry.name.clone(), levels);
    }
    if !cache.loggers.contains_key("polypus") {
        cache.loggers.insert(
            "polypus".to_owned(),
            levels_of(&top, disable, &is_enabled_for)?,
        );
    }
    Ok((cache, walk))
}

/// The `polypus` entries of `logging`'s logger dictionary, as one walk of it
/// found them (see [`read_levels`]).
struct Walk {
    /// The size of the dictionary walked.
    len: usize,
    /// Every entry named `polypus` or `polypus.*`.
    entries: Vec<Entry>,
}

/// One `polypus` entry of the logger dictionary.
struct Entry {
    /// The entry's key, kept as the Python string to look the entry up again
    /// without building a new one.
    key: Py<PyString>,
    /// The same name, for the snapshot.
    name: String,
    /// The object under the key: a logger, or a `PlaceHolder`.
    object: Py<PyAny>,
    /// Whether `object` is a logger. A `PlaceHolder` stands for a name nobody
    /// created: it gets no levels, and a logger created under it later starts
    /// from its nearest real ancestor, as `LevelCache::enabled` assumes.
    is_logger: bool,
}

impl Walk {
    /// Walk `loggers`, a dictionary nothing else can change while this runs.
    fn new(loggers: &Bound<'_, PyDict>, logger_class: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut entries = Vec::new();
        for (key, object) in loggers.iter() {
            let Ok(key) = key.cast_into::<PyString>() else {
                continue;
            };
            // A name that is not valid UTF-8 (a lone surrogate) is no
            // `polypus` logger a record can name.
            let Ok(name) = key.to_cow() else {
                continue;
            };
            if name != "polypus" && !name.starts_with("polypus.") {
                continue;
            }
            entries.push(Entry {
                name: name.into_owned(),
                is_logger: object.is_instance(logger_class)?,
                key: key.unbind(),
                object: object.unbind(),
            });
        }
        Ok(Self {
            len: loggers.len(),
            entries,
        })
    }

    /// Whether walking `loggers`, the live logger dictionary, again would find
    /// the same `polypus` loggers. `logging` only ever adds an entry to it
    /// (`getLogger` of a new name) or replaces a `PlaceHolder` with a logger
    /// (`getLogger` of a name that had a descendant), so the same size and
    /// the same object under every name walked mean that no `polypus` logger
    /// was created since. Neither check runs Python code when the keys are
    /// all strings, as `logging`'s are.
    fn is_current(&self, loggers: &Bound<'_, PyDict>) -> PyResult<bool> {
        if loggers.len() != self.len {
            return Ok(false);
        }
        let py = loggers.py();
        for entry in &self.entries {
            match loggers.get_item(entry.key.bind(py))? {
                Some(object) if object.is(&entry.object) => {}
                _ => return Ok(false),
            }
        }
        Ok(true)
    }
}

/// The [`Levels`] of one Python logger. Levels are compared as numbers, never
/// by name, so a custom level and `logging.disable` are honoured.
///
/// `own` is `isEnabledFor`'s answer. For a logger whose class keeps
/// `logging.Logger.isEnabledFor`, that is computed here from what that method
/// reads, its `disabled` flag, `logging.disable` and `getEffectiveLevel()`,
/// with one call into Python instead of one per level; it does not consult the
/// method's cache, which `setLevel` and `logging.disable` keep up to date. A
/// class that overrides the method is asked once per level, so the override
/// is honoured.
fn levels_of(
    logger: &Bound<'_, PyAny>,
    disable: i32,
    is_enabled_for: &Bound<'_, PyAny>,
) -> PyResult<Levels> {
    let effective: i32 = logger.call_method0("getEffectiveLevel")?.extract()?;
    let mut levels = Levels::default();
    for (level, levelno) in LEVELS {
        if levelno > disable && levelno >= effective {
            levels.inherited |= bit(level);
        }
    }
    if logger
        .get_type()
        .getattr("isEnabledFor")?
        .is(is_enabled_for)
    {
        if !logger.getattr("disabled")?.is_truthy()? {
            levels.own = levels.inherited;
        }
        return Ok(levels);
    }
    for (level, levelno) in LEVELS {
        if logger
            .call_method1("isEnabledFor", (levelno,))?
            .is_truthy()?
        {
            levels.own |= bit(level);
        }
    }
    Ok(levels)
}

/// Install [`BRIDGE`] as the process's `log` sink, with a first level
/// snapshot. The caller holds the bindings' one-install guard
/// (`LOGGER_INSTALLED` in `logging.rs`). Returns `false`, installing nothing,
/// when `log` already has a sink from outside these bindings.
pub(crate) fn install(py: Python<'_>) -> PyResult<bool> {
    // Nothing else refreshes `BRIDGE` before it is installed, so this snapshot
    // is stored; the fallback only keeps this free of `unwrap`.
    let max = BRIDGE.refresh(py, false)?.unwrap_or_else(|| {
        BRIDGE
            .snapshot()
            .map_or(LevelFilter::Off, |c| c.max_level())
    });
    if log::set_logger(&BRIDGE).is_err() {
        return Ok(false);
    }
    log::set_max_level(max);
    BRIDGE_INSTALLED.store(true, Ordering::Release);
    Ok(true)
}

/// Take a new level snapshot for the installed bridge; a no-op, and cheap,
/// when the bridge is not installed.
///
/// Every entry point that runs code that logs on worker threads calls this
/// first, on its own thread, before `py.detach`: records emitted while
/// detached are filtered against this snapshot without taking the GIL. Other
/// calls that detach (`Circuit.from_qasm3`, `Circuit.compose`,
/// `circuits.qft`, …) run single-threaded work and use the snapshot in force,
/// which never deadlocks but may be older; `docs/guide/running.md` lists the
/// calls that refresh. A failure to read the levels never fails the call: it
/// is reported through `sys.unraisablehook` and the previous snapshot stays
/// in force.
pub(crate) fn refresh_levels(py: Python<'_>) {
    if !BRIDGE_INSTALLED.load(Ordering::Acquire) {
        return;
    }
    if let Err(err) = BRIDGE.refresh(py, true) {
        err.write_unraisable(py, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::sync::mpsc;
    use std::time::Duration;

    // `logger_name` and `LevelCache` are pure. The bridge itself is driven
    // against a bare interpreter (docs/ENGINEERING.md §3): each test builds
    // its own `PythonLogBridge` and calls `Log::log` on it directly, never
    // installing it as the process's sink, and uses its own logger names,
    // since tests share the interpreter's `logging` state.

    /// How long a thread is given to finish what must not wait for the GIL.
    const TIMEOUT: Duration = Duration::from_secs(5);

    #[test]
    fn logger_names_put_every_target_under_polypus() {
        let cases = [
            ("polypus_optimizers::util", "polypus.optimizers.util"),
            ("polypus::bindings::qml", "polypus.bindings.qml"),
            ("polypus_sim", "polypus.sim"),
            ("zeromq::codec", "polypus.zeromq.codec"),
            ("polypusx::a", "polypus.polypusx.a"),
            ("", "polypus"),
            ("polypus", "polypus"),
            ("polypus_", "polypus.polypus_"),
            ("polypus_::a", "polypus.polypus_.a"),
            ("polypus_sim::calibration", "polypus.sim.calibration"),
        ];
        for (target, expected) in cases {
            assert_eq!(logger_name(target), expected, "target {target:?}");
        }
    }

    /// A snapshot with `polypus` at WARNING and `polypus.sim` at DEBUG.
    fn cache() -> LevelCache {
        let mask = |levels: &[Level]| levels.iter().fold(0, |m, l| m | bit(*l));
        let warn = mask(&[Level::Error, Level::Warn]);
        let debug = mask(&[Level::Error, Level::Warn, Level::Info, Level::Debug]);
        LevelCache {
            loggers: [
                (
                    "polypus".to_owned(),
                    Levels {
                        own: warn,
                        inherited: warn,
                    },
                ),
                (
                    "polypus.sim".to_owned(),
                    Levels {
                        // Disabled itself, but its children are not.
                        own: 0,
                        inherited: debug,
                    },
                ),
            ]
            .into(),
        }
    }

    #[test]
    fn the_cache_answers_for_a_logger_or_its_nearest_ancestor() {
        let cache = cache();
        assert!(cache.enabled("polypus", Level::Warn));
        assert!(!cache.enabled("polypus", Level::Info));
        assert!(
            !cache.enabled("polypus.sim", Level::Error),
            "its own answer"
        );
        assert!(cache.enabled("polypus.sim.calibration", Level::Debug));
        assert!(!cache.enabled("polypus.sim.calibration", Level::Trace));
        assert!(cache.enabled("polypus.optimizers.util", Level::Warn));
        assert!(!cache.enabled("polypus.optimizers.util", Level::Debug));
        assert!(!LevelCache::default().enabled("polypus", Level::Error));
        assert_eq!(cache.max_level(), LevelFilter::Debug);
        assert_eq!(LevelCache::default().max_level(), LevelFilter::Off);
    }

    /// A refresh that started earlier but finishes later never replaces the
    /// snapshot of one that started after it.
    #[test]
    fn a_stale_snapshot_never_replaces_a_newer_one() {
        let at = |levels: &[Level]| {
            let mask = levels.iter().fold(0, |m, l| m | bit(*l));
            LevelCache {
                loggers: [(
                    "polypus".to_owned(),
                    Levels {
                        own: mask,
                        inherited: mask,
                    },
                )]
                .into(),
            }
        };
        let debug = at(&[Level::Error, Level::Warn, Level::Info, Level::Debug]);
        let warning = at(&[Level::Error, Level::Warn]);
        let bridge = PythonLogBridge::new();
        assert_eq!(bridge.store(2, debug, false), Some(LevelFilter::Debug));
        assert_eq!(bridge.store(1, warning, false), None, "ticket 1 is older");
        assert!(
            bridge.wants("polypus", Level::Debug),
            "the newer snapshot stays"
        );
    }

    #[test]
    fn a_failed_format_gives_a_fixed_message() {
        struct Unformattable;
        impl fmt::Display for Unformattable {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                Err(fmt::Error)
            }
        }
        assert_eq!(
            format_message(&format_args!("a {}", Unformattable)),
            "<unformattable log message>"
        );
        assert_eq!(format_message(&format_args!("{}% done", 50)), "50% done");
    }

    /// A Python module with a handler that keeps every record it gets.
    const LIST_HANDLER: &str = "\
import logging

class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append(record)

def attach(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = ListHandler()
    logger.addHandler(handler)
    return handler
";

    /// A [`LIST_HANDLER`] attached to the logger `name`, set to `level`.
    fn list_handler<'py>(
        py: Python<'py>,
        module: &str,
        name: &str,
        level: i32,
    ) -> Bound<'py, PyAny> {
        let code = CString::new(LIST_HANDLER).expect("no NUL in the source");
        let file = CString::new(format!("{module}.py")).expect("no NUL in the name");
        let module = CString::new(module).expect("no NUL in the name");
        PyModule::from_code(py, &code, &file, &module)
            .expect("the test module compiles")
            .call_method1("attach", (name, level))
            .expect("the handler attaches")
    }

    /// Call `bridge.log` from a new thread with a `level` record for `target`,
    /// and tell the returned channel once it has returned.
    fn log_on_thread(
        bridge: &Arc<PythonLogBridge>,
        level: Level,
        target: &'static str,
    ) -> mpsc::Receiver<()> {
        let bridge = Arc::clone(bridge);
        let (done, returned) = mpsc::channel();
        std::thread::spawn(move || {
            bridge.log(
                &Record::builder()
                    .level(level)
                    .target(target)
                    .args(format_args!("from another thread"))
                    .build(),
            );
            let _ = done.send(());
        });
        returned
    }

    /// A record the snapshot disables returns while another thread holds the
    /// GIL: deciding it never attaches. An enabled one, on the contrary, waits
    /// until the GIL is released, which shows the test holds it.
    #[test]
    fn a_disabled_record_does_not_take_the_gil() {
        Python::initialize();
        let bridge = Arc::new(PythonLogBridge::new());
        Python::attach(|py| {
            let handler = list_handler(py, "bridge_t2", "polypus.t2", 30);
            bridge.refresh(py, false).expect("the levels are read");

            let debug = log_on_thread(&bridge, Level::Debug, "polypus_t2");
            assert!(
                debug.recv_timeout(TIMEOUT).is_ok(),
                "a disabled record waited for the GIL"
            );

            let warn = log_on_thread(&bridge, Level::Warn, "polypus_t2");
            assert!(
                warn.recv_timeout(Duration::from_millis(200)).is_err(),
                "an enabled record must need the GIL, or this test proves nothing"
            );
            assert!(
                py.detach(move || warn.recv_timeout(TIMEOUT)).is_ok(),
                "the enabled record is delivered once the GIL is free"
            );
            let records = handler
                .getattr("records")
                .expect("the handler keeps records");
            assert_eq!(records.len().expect("a list"), 1, "only the warning");
        });
    }

    /// A record reaches the Python logger named after its target with its
    /// level, its Rust source location and its message as formatted, a `%`
    /// included.
    #[test]
    fn a_record_is_delivered_with_its_rust_location() {
        Python::initialize();
        let bridge = PythonLogBridge::new();
        Python::attach(|py| {
            let handler = list_handler(py, "bridge_t3", "polypus.t3", 20);
            bridge.refresh(py, false).expect("the levels are read");
            bridge.log(
                &Record::builder()
                    .level(Level::Info)
                    .target("polypus_t3::module")
                    .file(Some("crates/polypus-t3/src/module.rs"))
                    .line(Some(42))
                    .args(format_args!("{}% done, %s and %(x)s left", 50))
                    .build(),
            );
            bridge.log(
                &Record::builder()
                    .level(Level::Debug)
                    .target("polypus_t3::module")
                    .args(format_args!("below the logger's level"))
                    .build(),
            );
            let records = handler
                .getattr("records")
                .expect("the handler keeps records");
            assert_eq!(records.len().expect("a list"), 1, "only the INFO record");
            let record = records.get_item(0).expect("one record");
            let field = |name: &str| record.getattr(name).expect("a LogRecord field");
            assert_eq!(
                field("name").extract::<String>().unwrap(),
                "polypus.t3.module"
            );
            assert_eq!(field("levelno").extract::<i32>().unwrap(), 20);
            assert_eq!(
                field("pathname").extract::<String>().unwrap(),
                "crates/polypus-t3/src/module.rs"
            );
            assert_eq!(field("lineno").extract::<u32>().unwrap(), 42);
            let message: String = record
                .call_method0("getMessage")
                .expect("the message formats")
                .extract()
                .unwrap();
            assert_eq!(message, "50% done, %s and %(x)s left");
        });
    }

    /// A custom numeric level and a placeholder in the logger dictionary are
    /// read as Python itself would answer. (`logging.disable` is global to the
    /// interpreter these tests share, so `tests/python/test_log_to_python.py`
    /// covers it, in a process of its own.)
    #[test]
    fn the_snapshot_follows_python_level_semantics() {
        Python::initialize();
        let bridge = PythonLogBridge::new();
        Python::attach(|py| {
            // A custom level between DEBUG and INFO on `polypus.t4`, and a
            // logger two levels below it, so `polypus.t4.mid` is only a
            // placeholder.
            let _ = list_handler(py, "bridge_t4", "polypus.t4", 15);
            let _ = list_handler(py, "bridge_t4b", "polypus.t4.mid.leaf", 0);
            bridge.refresh(py, false).expect("the levels are read");
            assert!(bridge.wants("polypus.t4", Level::Info));
            assert!(!bridge.wants("polypus.t4", Level::Debug));
            assert!(
                bridge.wants("polypus.t4.mid", Level::Info),
                "under the placeholder"
            );
            assert!(bridge.wants("polypus.t4.mid.leaf", Level::Info));
            assert!(!bridge.wants("polypus.t4.mid.leaf", Level::Debug));
        });
    }

    /// A walk is reused only while the dictionary keeps its size and the same
    /// object under every `polypus` name the walk found: a logger added
    /// anywhere, or a placeholder replaced by a logger, makes it stale. A
    /// private dictionary stands in for `logging`'s, which other tests change.
    #[test]
    fn a_walk_is_current_until_the_logger_dictionary_changes() {
        Python::initialize();
        Python::attach(|py| {
            let logging = PyModule::import(py, "logging").expect("logging imports");
            let logger_class = logging.getattr("Logger").expect("logging.Logger");
            let logger = |name: &str| logger_class.call1((name,)).expect("a logger is built");
            let top = logger("polypus");
            let placeholder = logging
                .getattr("PlaceHolder")
                .and_then(|class| class.call1((&top,)))
                .expect("a placeholder is built");
            let loggers = PyDict::new(py);
            loggers.set_item("polypus", &top).expect("set");
            loggers.set_item("polypus.walk", &placeholder).expect("set");
            loggers.set_item("other", logger("other")).expect("set");
            // A name that is not valid UTF-8 is skipped, not an error.
            let surrogate = py
                .eval(c"'polypus.\\udc80'", None, None)
                .expect("a name with a lone surrogate");
            loggers
                .set_item(&surrogate, logger("other.surrogate"))
                .expect("set");

            let walk = Walk::new(&loggers, &logger_class).expect("the dictionary is walked");
            let mut found: Vec<_> = walk
                .entries
                .iter()
                .map(|entry| (entry.name.as_str(), entry.is_logger))
                .collect();
            found.sort_unstable();
            assert_eq!(found, [("polypus", true), ("polypus.walk", false)]);
            let current = |walk: &Walk| walk.is_current(&loggers).expect("the check runs");
            assert!(current(&walk));

            loggers
                .set_item("other.child", logger("other.child"))
                .expect("set");
            assert!(!current(&walk), "a logger was added");
            loggers.del_item("other.child").expect("del");
            assert!(current(&walk));

            loggers
                .set_item("polypus.walk", logger("polypus.walk"))
                .expect("set");
            assert!(!current(&walk), "the placeholder became a logger");
        });
    }

    /// A refresh after an earlier one sees a logger created in between, and
    /// a placeholder that became a logger, though the dictionary keeps its
    /// size in the second case.
    #[test]
    fn a_refresh_sees_the_loggers_created_since_the_last_one() {
        Python::initialize();
        let bridge = PythonLogBridge::new();
        Python::attach(|py| {
            // `polypus.t6.mid` is only a placeholder for now.
            let _ = list_handler(py, "bridge_t6", "polypus.t6.mid.leaf", 30);
            bridge.refresh(py, false).expect("the levels are read");
            bridge
                .refresh(py, false)
                .expect("the levels are read again");
            assert!(!bridge.wants("polypus.t6.new", Level::Debug));
            assert!(!bridge.wants("polypus.t6.mid", Level::Debug));

            let _ = list_handler(py, "bridge_t6b", "polypus.t6.new", 10);
            bridge.refresh(py, false).expect("the levels are read");
            assert!(bridge.wants("polypus.t6.new", Level::Debug), "a new logger");

            let _ = list_handler(py, "bridge_t6c", "polypus.t6.mid", 10);
            bridge.refresh(py, false).expect("the levels are read");
            assert!(
                bridge.wants("polypus.t6.mid", Level::Debug),
                "a placeholder that became a logger"
            );
        });
    }

    /// A logger's own `disabled` flag silences it but not a new logger under
    /// it, and a logger whose class overrides `isEnabledFor` is asked rather
    /// than answered from its effective level.
    #[test]
    fn a_disabled_logger_and_an_overridden_is_enabled_for_are_honoured() {
        Python::initialize();
        let bridge = PythonLogBridge::new();
        Python::attach(|py| {
            let _ = list_handler(py, "bridge_t7", "polypus.t7", 10);
            PyModule::from_code(
                py,
                c"
import logging

class ErrorsOnly(logging.Logger):
    def isEnabledFor(self, level):
        return level >= logging.ERROR

def setup():
    logging.getLogger('polypus.t7').disabled = True
    picky = ErrorsOnly('polypus.t7picky')
    picky.setLevel(logging.DEBUG)
    picky.parent = logging.getLogger('polypus')
    logging.Logger.manager.loggerDict['polypus.t7picky'] = picky
",
                c"bridge_t7c.py",
                c"bridge_t7c",
            )
            .and_then(|module| module.call_method0("setup"))
            .expect("the loggers are set up");
            bridge.refresh(py, false).expect("the levels are read");
            assert!(!bridge.wants("polypus.t7", Level::Error), "disabled");
            assert!(
                bridge.wants("polypus.t7.child", Level::Debug),
                "a child does not inherit `disabled`"
            );
            assert!(
                !bridge.wants("polypus.t7picky", Level::Warn),
                "the override refuses WARNING at an effective level of DEBUG"
            );
            assert!(bridge.wants("polypus.t7picky", Level::Error));
        });
    }
}
