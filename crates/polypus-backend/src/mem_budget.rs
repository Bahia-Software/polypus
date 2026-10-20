//! Memory budget for statevector simulation (plan §4.5, P1-memory; issue #215).
//!
//! The dominant memory cost of the local simulators (native Rust and Aer) is not
//! the list of bound circuits — cheap — but the **concurrent statevectors**: an
//! `n`-qubit circuit under simulation holds `2^n` complex-`f64` amplitudes =
//! [`BYTES_PER_AMPLITUDE`]` · 2^n` bytes. Running a whole population in parallel
//! therefore peaks at roughly `concurrency · 16 · 2^n` bytes, which OOMs at high
//! `n` (a single 30-qubit statevector is already 16 GiB).
//!
//! This module does two things with a byte budget:
//!
//! - **Throttle** ([`max_statevector_concurrency`]): turn the budget into a safe
//!   concurrency cap so a backend bounds its peak memory without changing any
//!   result — the seeds are independent of how many circuits run at once, so
//!   throttling trades only speed, never counts.
//! - **Refuse** ([`check_fits`]): a single statevector that cannot fit in a
//!   budget that reflects a *known* limit is rejected up front with a typed
//!   [`InsufficientMemory`] instead of being started and killed by the Linux
//!   OOM-killer (which gives no Python exception and no partial results).
//!   The native backend and Aer's `statevector` method refuse with it in Rust;
//!   for Aer's other methods the local backend hands the same known budget to
//!   Aer as `max_memory_mb`, and Aer's own per-method validation refuses.
//!
//! # Where the budget comes from
//!
//! Resolved in this order ([`resolve_budget`]), and reported with its
//! [`BudgetSource`]:
//!
//! 1. **`POLYPUS_MEM_BUDGET`** ([`BudgetSource::Explicit`]), when it parses with
//!    [`parse_mem_budget`] — a positive byte count with an optional `K`/`M`/`G`/`T`
//!    suffix, base 1024 (`32G`, `512M`, `1048576`). HPC job scripts set it to the
//!    SLURM allocation. An invalid value is **never applied silently**: it is
//!    logged once per process (`log::warn!`) and the next source is used.
//! 2. **The detected limit minus a reserve** ([`BudgetSource::Detected`]): the
//!    minimum of `MemAvailable` (`/proc/meminfo`) and the process's cgroup memory
//!    limit (v2 `memory.max` of its own cgroup and every ancestor, or v1
//!    `memory.limit_in_bytes`), minus [`apply_reserve`]'s safety margin.
//! 3. **[`DEFAULT_MEM_BUDGET_BYTES`]** ([`BudgetSource::Fallback`]), when nothing
//!    can be detected (no `/proc`: macOS, Windows). [`check_fits`] never rejects
//!    against this blind default.
//!
//! The detected limit is read **once per process** and cached, so the budget does
//! not drift between waves of one run; the environment variable is still read on
//! every call, so an explicit override always wins.
//!
//! # Limitations of the detected default
//!
//! - **Processes sharing a cgroup.** Several processes in one cgroup — e.g. the
//!   MPI ranks of one SLURM job step — each see the *whole* cgroup limit, so each
//!   would budget for all of it. Such jobs must set `POLYPUS_MEM_BUDGET` per
//!   process (the allocation divided by the ranks per node).
//! - **Shared machines without a cgroup limit** (a login node, a multi-user
//!   workstation): the default is whatever RAM was available when the process
//!   started, so one run may claim most of it. Set `POLYPUS_MEM_BUDGET` there.
//! - **Only the hard limit is read.** A cgroup v2 `memory.high` soft limit and
//!   the cgroup's swap allowance (`memory.swap.max`, `memsw`) are ignored, and
//!   `memory.current` is not subtracted (it includes reclaimable page cache).
//! - **Read once.** `MemAvailable` and the cgroup limits are read the first time
//!   the budget is needed and cached for the life of the process, so memory freed
//!   or taken by other processes later is not seen (`POLYPUS_MEM_BUDGET` is
//!   re-read on every call).

use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// `1 GiB`, the unit the reserve and fallback constants are written in.
const GIB: u64 = 1024 * 1024 * 1024;

/// Bytes of memory per statevector amplitude the budget accounts for: one
/// complex `f64` (`2 × 8` bytes).
///
/// This is the *whole* per-amplitude cost of the native backend. Its shot
/// sampling used to build a `2^n`-element `f64` CDF next to the statevector
/// (another 8 bytes per amplitude, so a 24-byte peak the budget did not model);
/// since issue #215 it samples with `O(shots)` extra memory instead (see
/// `polypus_sim::Statevector::sample`), so `16 · 2^n` is the real peak.
pub const BYTES_PER_AMPLITUDE: u64 = 16;

/// Budget (bytes) used when `POLYPUS_MEM_BUDGET` is unset or invalid **and** no
/// memory limit can be detected (no `/proc` or cgroup: macOS, Windows): 16 GiB.
/// Chosen so a lone 30-qubit statevector (16 GiB) still runs, full
/// multi-threading is retained up to ~25 qubits, and only the high-qubit regime
/// that used to OOM is throttled.
///
/// It is a blind guess, not a known limit, so [`check_fits`] never rejects a
/// circuit against it — it only throttles concurrency. On Linux the detected
/// limit replaces it (see the [module docs](self)).
pub const DEFAULT_MEM_BUDGET_BYTES: u64 = 16 * GIB;

/// Environment variable that sets the budget explicitly, in the format accepted
/// by [`parse_mem_budget`] (e.g. `32G`).
pub const MEM_BUDGET_ENV: &str = "POLYPUS_MEM_BUDGET";

/// Smallest reserve [`apply_reserve`] keeps back from a detected limit: 512 MiB.
///
/// The fixed part of the margin. The memory a Polypus process holds outside its
/// statevectors was measured at ≈100 MiB (the Python interpreter, Qiskit/Aer and
/// the `polypus` extension loaded); 512 MiB covers that about five times over,
/// leaving room for allocator fragmentation and the caller's own data.
pub const MIN_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Proportional part of the reserve, as a divisor: `limit / 10` (10 %).
///
/// Covers the overhead that grows with the limit and was not measured per
/// allocator/backend — Aer's internal buffers and the allocator's slack around
/// multi-GiB allocations — so a large node keeps a margin proportionate to what
/// it can run.
const RESERVE_FRACTION_DIVISOR: u64 = 10;

/// The largest share of a detected limit the reserve may take: half of it
/// (`limit / 2`). On a small limit (< 1 GiB) the 512 MiB floor would otherwise
/// swallow the whole budget; the budget keeps at least half.
const MAX_RESERVE_DIVISOR: u64 = 2;

/// cgroup v1 reports "no limit" as a huge sentinel (`PAGE_COUNTER_MAX` pages,
/// ≈ 2^63 bytes); any value at or above `2^60` is treated as unlimited.
const CGROUP_V1_UNLIMITED: u64 = 1 << 60;

/// Root of the unified (v2) cgroup hierarchy.
const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";

/// Root of the cgroup v1 `memory` controller hierarchy.
const CGROUP_V1_MEMORY_ROOT: &str = "/sys/fs/cgroup/memory";

/// Where a [`MemBudget`]'s byte count came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// A valid `POLYPUS_MEM_BUDGET`.
    Explicit,
    /// The detected RAM/cgroup limit minus the safety reserve.
    Detected,
    /// [`DEFAULT_MEM_BUDGET_BYTES`]: nothing set, nothing detected. Never used to
    /// reject a circuit.
    Fallback,
}

impl fmt::Display for BudgetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetSource::Explicit => write!(f, "set by {MEM_BUDGET_ENV}"),
            BudgetSource::Detected => write!(
                f,
                "detected from the available RAM / cgroup memory limit, minus a safety reserve"
            ),
            BudgetSource::Fallback => write!(f, "built-in fallback, no memory limit detected"),
        }
    }
}

/// A resolved memory budget: how many bytes of statevector may be held at once,
/// and where that number came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemBudget {
    /// The budget in bytes.
    pub bytes: u64,
    /// Where [`bytes`](Self::bytes) came from.
    pub source: BudgetSource,
}

/// Why a `POLYPUS_MEM_BUDGET` value was rejected by [`parse_mem_budget`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemBudgetParseError {
    /// The value is empty (or only whitespace).
    Empty,
    /// The value does not start with a digit (`"abc"`, `"-1"`, `"G"`).
    MissingNumber,
    /// The number is followed by something other than a known unit suffix
    /// (`"1.5G"`, `"32X"`). Carries the unparsed remainder.
    UnknownSuffix(String),
    /// The value is zero, which would forbid all work.
    Zero,
    /// The value does not fit in 64 bits once the suffix is applied.
    Overflow,
}

impl fmt::Display for MemBudgetParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemBudgetParseError::Empty => write!(f, "the value is empty"),
            MemBudgetParseError::MissingNumber => {
                write!(f, "the value must start with a positive integer")
            }
            MemBudgetParseError::UnknownSuffix(s) => write!(f, "unknown unit suffix {s:?}"),
            MemBudgetParseError::Zero => write!(f, "a budget of zero would forbid all work"),
            MemBudgetParseError::Overflow => write!(f, "the value does not fit in 64 bits"),
        }
    }
}

impl std::error::Error for MemBudgetParseError {}

/// Parse a `POLYPUS_MEM_BUDGET` value into bytes.
///
/// Accepted: a positive integer, optionally followed by a unit suffix `K`, `M`,
/// `G` or `T` — case-insensitive, optionally followed by `B` or `iB`, with
/// whitespace allowed around the value and between the number and the suffix.
/// Units are **base 1024** in every spelling (`32G`, `32GB` and `32GiB` are all
/// 32 GiB), the convention SLURM uses for `--mem`. A bare number, or one
/// followed by `B`, is a byte count.
///
/// ```
/// use polypus_backend::mem_budget::parse_mem_budget;
/// assert_eq!(parse_mem_budget("32G"), Ok(32 << 30));
/// assert_eq!(parse_mem_budget(" 512 MiB "), Ok(512 << 20));
/// assert_eq!(parse_mem_budget("1048576"), Ok(1 << 20));
/// assert!(parse_mem_budget("1.5G").is_err());
/// ```
///
/// # Errors
///
/// A [`MemBudgetParseError`] for an empty, zero, negative, fractional,
/// non-numeric, unknown-suffix or overflowing value.
pub fn parse_mem_budget(value: &str) -> Result<u64, MemBudgetParseError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(MemBudgetParseError::Empty);
    }
    let digits_end = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, suffix) = value.split_at(digits_end);
    if digits.is_empty() {
        return Err(MemBudgetParseError::MissingNumber);
    }
    // All-ASCII-digit, so the only possible parse failure is overflow.
    let number: u64 = digits.parse().map_err(|_| MemBudgetParseError::Overflow)?;
    let multiplier: u64 = match suffix.trim_start().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1 << 40,
        _ => return Err(MemBudgetParseError::UnknownSuffix(suffix.to_string())),
    };
    if number == 0 {
        return Err(MemBudgetParseError::Zero);
    }
    number
        .checked_mul(multiplier)
        .ok_or(MemBudgetParseError::Overflow)
}

/// The budget left from a detected memory `limit` once a safety reserve is kept
/// back: `limit − clamp(limit / 10, 512 MiB, limit / 2)`, saturating.
///
/// The reserve is [`MIN_RESERVE_BYTES`] (the measured fixed footprint of a
/// Polypus process, with margin) or 10 % of the limit (unmeasured
/// backend/allocator overhead that grows with it), whichever is larger — but
/// never more than half the limit, so a small allocation still gets a usable
/// budget. When the floor exceeds the cap (a limit under 1 GiB) the cap wins.
///
/// ```
/// use polypus_backend::mem_budget::apply_reserve;
/// const GIB: u64 = 1 << 30;
/// assert_eq!(apply_reserve(256 * GIB), 256 * GIB - 256 * GIB / 10); // 10 %
/// assert_eq!(apply_reserve(4 * GIB), 4 * GIB - 512 * 1024 * 1024);  // 512 MiB floor
/// assert_eq!(apply_reserve(GIB / 2), GIB / 4);                       // 50 % cap
/// ```
pub fn apply_reserve(limit: u64) -> u64 {
    let reserve = (limit / RESERVE_FRACTION_DIVISOR)
        .max(MIN_RESERVE_BYTES)
        .min(limit / MAX_RESERVE_DIVISOR);
    limit.saturating_sub(reserve)
}

/// Resolve the budget from the raw `POLYPUS_MEM_BUDGET` value (`None` when
/// unset) and the detected memory limit (`None` when nothing was detected).
/// Pure: the caller supplies both inputs, so every precedence case is testable
/// without touching the environment or the filesystem.
///
/// Precedence: a valid explicit value, then the detected limit minus
/// [`apply_reserve`]'s margin, then [`DEFAULT_MEM_BUDGET_BYTES`]. A detected
/// limit that leaves a zero budget is treated as undetected (it cannot be a real
/// limit for a running process).
///
/// Returns the budget together with the parse error of an explicit value that
/// was **ignored** because it is invalid, so the caller can warn about it.
pub fn resolve_budget(
    env: Option<&str>,
    detected_limit: Option<u64>,
) -> (MemBudget, Option<MemBudgetParseError>) {
    let ignored = match env.map(parse_mem_budget) {
        Some(Ok(bytes)) => {
            return (
                MemBudget {
                    bytes,
                    source: BudgetSource::Explicit,
                },
                None,
            )
        }
        Some(Err(e)) => Some(e),
        None => None,
    };
    let budget = match detected_limit.map(apply_reserve).filter(|&b| b > 0) {
        Some(bytes) => MemBudget {
            bytes,
            source: BudgetSource::Detected,
        },
        None => MemBudget {
            bytes: DEFAULT_MEM_BUDGET_BYTES,
            source: BudgetSource::Fallback,
        },
    };
    (budget, ignored)
}

/// Fields of `/proc/meminfo` the detection reads, in bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemInfo {
    /// `MemTotal`: physical RAM.
    pub total: Option<u64>,
    /// `MemAvailable`: RAM obtainable without swapping, reclaimable page cache
    /// included.
    pub available: Option<u64>,
}

/// Parse the `MemTotal` and `MemAvailable` lines of a `/proc/meminfo` text
/// (`MemAvailable:   123456 kB`) into bytes. A missing or malformed line leaves
/// its field `None`; a value that overflows `u64` bytes saturates.
pub fn parse_meminfo(text: &str) -> MemInfo {
    let mut info = MemInfo::default();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let field = match key.trim() {
            "MemTotal" => &mut info.total,
            "MemAvailable" => &mut info.available,
            _ => continue,
        };
        let mut parts = rest.split_whitespace();
        let (Some(number), Some("kB"), None) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        if let Ok(kib) = number.parse::<u64>() {
            *field = Some(kib.saturating_mul(1024));
        }
    }
    info
}

/// The process's cgroup v2 path from a `/proc/self/cgroup` text: the path of
/// the `0::<path>` line (e.g. `/user.slice/session-1.scope`). `None` when there
/// is no v2 line, or the path is not absolute or contains a `..` component
/// (never followed outside the cgroup mount).
pub fn parse_cgroup_v2_path(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
        .filter(|path| is_safe_cgroup_path(path))
}

/// The process's cgroup v1 `memory`-controller path from a `/proc/self/cgroup`
/// text: the path of the `<id>:<controllers>:<path>` line whose controller list
/// contains `memory`. Same validation as [`parse_cgroup_v2_path`].
pub fn parse_cgroup_v1_memory_path(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| {
            let mut fields = line.splitn(3, ':');
            let (_id, controllers, path) = (fields.next()?, fields.next()?, fields.next()?);
            controllers
                .split(',')
                .any(|c| c == "memory")
                .then_some(path.trim())
        })
        .filter(|path| is_safe_cgroup_path(path))
}

fn is_safe_cgroup_path(path: &str) -> bool {
    path.starts_with('/') && !path.split('/').any(|c| c == "..")
}

/// Parse a cgroup memory-limit file (`memory.max` or `memory.limit_in_bytes`):
/// a byte count, or `max` for "no limit". `None` for `max` and for malformed
/// content — both mean "no limit known here".
pub fn parse_cgroup_limit(text: &str) -> Option<u64> {
    let text = text.trim();
    if text == "max" {
        return None;
    }
    text.parse().ok()
}

/// The files `file_name` under `root` for `cgroup_path` and every ancestor up
/// to the root, innermost first: `/a/b` → `root/a/b/f`, `root/a/f`, `root/f`.
fn cgroup_limit_files(root: &str, cgroup_path: &str, file_name: &str) -> Vec<PathBuf> {
    let mut dir = PathBuf::from(root);
    let mut dirs = vec![dir.clone()];
    for component in cgroup_path.split('/').filter(|c| !c.is_empty()) {
        dir.push(component);
        dirs.push(dir.clone());
    }
    dirs.into_iter().rev().map(|d| d.join(file_name)).collect()
}

/// The effective memory limit of the process, read through `read` (the
/// filesystem in production, a fixture map in tests): the minimum of
/// `MemAvailable` and every cgroup limit that applies — cgroup v2 `memory.max`
/// of the process's cgroup and each ancestor (`max` = unlimited), and cgroup v1
/// `memory.limit_in_bytes` likewise, where a value at or above 2^60 or above
/// `MemTotal` means unlimited. `None` when nothing could be read.
///
/// `memory.current` is deliberately **not** subtracted: it includes reclaimable
/// page cache, so it would under-report what the process can still allocate and
/// reject circuits that would in fact run.
pub fn detect_memory_limit_with(read: impl Fn(&Path) -> Option<String>) -> Option<u64> {
    let meminfo = read(Path::new("/proc/meminfo"))
        .map(|t| parse_meminfo(&t))
        .unwrap_or_default();
    let cgroup = read(Path::new("/proc/self/cgroup")).unwrap_or_default();
    let read_limits = |root: &str, path: &str, file: &str| -> Vec<u64> {
        cgroup_limit_files(root, path, file)
            .iter()
            .filter_map(|p| read(p))
            .filter_map(|t| parse_cgroup_limit(&t))
            .collect()
    };

    let mut limits: Vec<u64> = meminfo.available.into_iter().collect();
    if let Some(path) = parse_cgroup_v2_path(&cgroup) {
        limits.extend(read_limits(CGROUP_V2_ROOT, path, "memory.max"));
    }
    if let Some(path) = parse_cgroup_v1_memory_path(&cgroup) {
        let v1_ceiling = meminfo.total.unwrap_or(u64::MAX);
        limits.extend(
            read_limits(CGROUP_V1_MEMORY_ROOT, path, "memory.limit_in_bytes")
                .into_iter()
                .filter(|&l| l < CGROUP_V1_UNLIMITED && l <= v1_ceiling),
        );
    }
    limits.into_iter().min()
}

/// [`detect_memory_limit_with`] on the real filesystem, computed once per
/// process so the budget cannot drift between the waves of one run.
fn detected_memory_limit() -> Option<u64> {
    static DETECTED: OnceLock<Option<u64>> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        let limit = detect_memory_limit_with(|p| std::fs::read_to_string(p).ok());
        match limit {
            Some(bytes) => log::debug!(
                "detected a memory limit of {} (budget {} after the safety reserve)",
                format_bytes(bytes),
                format_bytes(apply_reserve(bytes))
            ),
            None => log::debug!(
                "no memory limit detected; the statevector budget falls back to {}",
                format_bytes(DEFAULT_MEM_BUDGET_BYTES)
            ),
        }
        limit
    })
}

/// `true` exactly once per `flag`: the gate that makes the invalid-value warning
/// fire once per process however many waves re-resolve the budget.
fn first_time(flag: &AtomicBool) -> bool {
    !flag.swap(true, Ordering::Relaxed)
}

/// The active budget: `POLYPUS_MEM_BUDGET` (re-read on every call), else the
/// detected limit minus the reserve (detected once per process), else
/// [`DEFAULT_MEM_BUDGET_BYTES`]. An invalid `POLYPUS_MEM_BUDGET` is logged with
/// `log::warn!` the first time it is seen in the process, and ignored.
pub fn active_budget() -> MemBudget {
    static WARNED: AtomicBool = AtomicBool::new(false);
    // A non-UTF-8 value is kept (lossily) so it is reported, not treated as unset.
    let raw = env::var_os(MEM_BUDGET_ENV).map(|v| v.to_string_lossy().into_owned());
    let (budget, ignored) = resolve_budget(raw.as_deref(), detected_memory_limit());
    if let (Some(err), Some(raw)) = (ignored, raw.as_deref()) {
        if first_time(&WARNED) {
            log::warn!(
                "ignoring {MEM_BUDGET_ENV}={raw:?}: {err}. Accepted: a positive integer byte \
                 count with an optional K/M/G/T suffix (base 1024, case-insensitive, optional \
                 B/iB), e.g. 1048576, 512M, 32G, 32GiB, 1T. Using a budget of {} ({}) instead.",
                format_bytes(budget.bytes),
                budget.source
            );
        }
    }
    budget
}

/// Bytes held by one `n`-qubit statevector: `2^n` amplitudes ×
/// [`BYTES_PER_AMPLITUDE`]. `None` when that overflows `u64` (`n ≥ 60`).
fn statevector_bytes_checked(num_qubits: usize) -> Option<u64> {
    u32::try_from(num_qubits)
        .ok()
        .and_then(|n| 1u64.checked_shl(n))
        .and_then(|amps| amps.checked_mul(BYTES_PER_AMPLITUDE))
}

/// [`statevector_bytes_checked`] saturated to `u64::MAX`, so the cap computation
/// never overflows or panics — such a circuit simply resolves to a concurrency of
/// 1 (run one at a time).
fn statevector_bytes(num_qubits: usize) -> u64 {
    statevector_bytes_checked(num_qubits).unwrap_or(u64::MAX)
}

/// Maximum `n`-qubit statevectors that may be simulated concurrently under the
/// [active budget](active_budget), never exceeding `num_threads` and never below
/// 1 (plan §4.5): `max(1, min(num_threads, budget / (16 · 2^n)))`.
///
/// This only *throttles*; whether a single statevector fits at all is
/// [`check_fits`]'s decision.
pub fn max_statevector_concurrency(num_qubits: usize, num_threads: usize) -> usize {
    concurrency_for(active_budget().bytes, num_qubits, num_threads)
}

/// Pure budget arithmetic behind [`max_statevector_concurrency`], split out so it
/// can be unit-tested without touching the process environment.
pub(crate) fn concurrency_for(budget_bytes: u64, num_qubits: usize, num_threads: usize) -> usize {
    let sv = statevector_bytes(num_qubits);
    // `sv >= 16 > 0`, so the division never traps; `.max(1)` keeps at least one
    // circuit in flight even when a single vector exceeds the whole budget.
    let by_budget = usize::try_from((budget_bytes / sv).max(1)).unwrap_or(usize::MAX);
    num_threads.max(1).min(by_budget)
}

/// A single statevector does not fit in the memory budget: returned by
/// [`check_fits`] instead of letting the run be killed by the OOM-killer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsufficientMemory {
    /// Qubit count of the widest circuit that was refused.
    pub num_qubits: usize,
    /// Bytes one statevector of that width needs (`16 · 2^n`), saturated to
    /// `u64::MAX` when that overflows.
    pub required_bytes: u64,
    /// The budget it was checked against.
    pub budget_bytes: u64,
    /// Where the budget came from (never [`BudgetSource::Fallback`]).
    pub source: BudgetSource,
}

impl fmt::Display for InsufficientMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let required = if self.required_bytes == u64::MAX {
            format!("more than {}", format_bytes(u64::MAX))
        } else {
            format_bytes(self.required_bytes)
        };
        write!(
            f,
            "not enough memory for a {n}-qubit statevector: it needs {required} \
             ({BYTES_PER_AMPLITUDE} bytes x 2^{n} amplitudes) but the memory budget is {budget} \
             ({source}). Refusing to start rather than be killed by the out-of-memory killer. \
             Use fewer qubits, or, if more memory really is available, set {MEM_BUDGET_ENV} \
             (e.g. {MEM_BUDGET_ENV}=64G) to override the budget.",
            n = self.num_qubits,
            budget = format_bytes(self.budget_bytes),
            source = self.source,
        )
    }
}

impl std::error::Error for InsufficientMemory {}

/// Check that one `num_qubits`-qubit statevector (`16 · 2^n` bytes) fits in
/// `budget`. Pure, so a caller passes the budget explicitly (see
/// [`check_statevector_fits`] for the active one).
///
/// A [`BudgetSource::Fallback`] budget **never** rejects: it is a guess made
/// with no limit detected (macOS, Windows), and refusing a legitimate 30-qubit
/// run on such a host would be a regression. Only a known limit — explicit or
/// detected — rejects.
///
/// ```
/// use polypus_backend::mem_budget::{check_fits, BudgetSource, MemBudget};
/// let budget = MemBudget { bytes: 1 << 20, source: BudgetSource::Explicit }; // 1 MiB
/// assert!(check_fits(16, budget).is_ok());  // 1 MiB statevector: fits exactly
/// assert!(check_fits(20, budget).is_err()); // 16 MiB: refused
/// ```
///
/// # Errors
///
/// [`InsufficientMemory`] when the statevector is larger than a non-fallback
/// budget.
pub fn check_fits(num_qubits: usize, budget: MemBudget) -> Result<(), InsufficientMemory> {
    if budget.source == BudgetSource::Fallback {
        return Ok(());
    }
    match statevector_bytes_checked(num_qubits) {
        Some(required) if required <= budget.bytes => Ok(()),
        required => Err(InsufficientMemory {
            num_qubits,
            required_bytes: required.unwrap_or(u64::MAX),
            budget_bytes: budget.bytes,
            source: budget.source,
        }),
    }
}

/// [`check_fits`] against the [active budget](active_budget): what a backend
/// calls before allocating a statevector.
///
/// # Errors
///
/// [`InsufficientMemory`], as for [`check_fits`].
pub fn check_statevector_fits(num_qubits: usize) -> Result<(), InsufficientMemory> {
    check_fits(num_qubits, active_budget())
}

/// Human-readable byte count in binary units (`16.00 GiB`), for messages.
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const MIB: u64 = 1024 * 1024;

    // --- concurrency arithmetic (budget passed explicitly: deterministic) -------

    #[test]
    fn low_qubit_counts_keep_full_thread_concurrency() {
        // At 8 qubits a statevector is 4 KiB; a 16 GiB budget dwarfs it, so the
        // cap is purely `num_threads` — i.e. today's unbounded-`par_iter` behaviour
        // is preserved exactly for the common low-qubit workloads.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 8, 32), 32);
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 20, 32), 32);
    }

    #[test]
    fn high_qubit_counts_are_throttled_by_the_budget() {
        // 28 qubits => 4 GiB/vector => 16 GiB / 4 GiB = 4 concurrent vectors.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 28, 32), 4);
        // 30 qubits => 16 GiB/vector => exactly one fits in 16 GiB.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 30, 32), 1);
    }

    #[test]
    fn a_single_vector_larger_than_the_budget_still_runs_one_at_a_time() {
        // 30-qubit vector (16 GiB) under a 1 GiB budget: cannot fit, but we must
        // never return 0 — the throttle's answer is exactly one at a time (the
        // refusal, when the budget is a known limit, is `check_fits`'s job).
        assert_eq!(concurrency_for(GIB, 30, 32), 1);
        // Absurd qubit count saturates the statevector size to u64::MAX instead of
        // overflowing; even the largest possible budget then fits exactly one.
        assert_eq!(concurrency_for(u64::MAX, 100, 8), 1);
    }

    #[test]
    fn a_larger_budget_recovers_concurrency_at_high_qubits() {
        // Doubling the budget to 32 GiB doubles the 28-qubit cap from 4 to 8: the
        // override an HPC job sets recovers concurrency the default throttled.
        assert_eq!(concurrency_for(32 * GIB, 28, 32), 8);
    }

    #[test]
    fn concurrency_is_never_below_one_even_with_zero_threads() {
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 30, 0), 1);
    }

    #[test]
    fn statevector_bytes_matches_the_amplitude_formula() {
        assert_eq!(statevector_bytes(0), 16);
        assert_eq!(statevector_bytes(1), 32);
        assert_eq!(statevector_bytes(10), (1 << 10) * 16);
        assert_eq!(statevector_bytes(30), 16 * GIB);
        // n >= 60 overflows u64 and must saturate rather than panic.
        assert_eq!(statevector_bytes(59), 1 << 63);
        assert_eq!(statevector_bytes(60), u64::MAX);
        assert_eq!(statevector_bytes(64), u64::MAX);
        assert_eq!(statevector_bytes(200), u64::MAX);
        assert_eq!(statevector_bytes_checked(60), None);
    }

    // --- parse_mem_budget --------------------------------------------------------

    #[test]
    fn parses_every_accepted_spelling_in_base_1024() {
        let cases: &[(&str, u64)] = &[
            ("32G", 32 * GIB),
            ("32g", 32 * GIB),
            ("32GiB", 32 * GIB),
            ("32GB", 32 * GIB),
            ("32gib", 32 * GIB),
            ("512M", 512 * MIB),
            ("512mb", 512 * MIB),
            ("1048576", MIB),
            ("1048576B", MIB),
            (" 8 G ", 8 * GIB),
            ("\t16G\n", 16 * GIB),
            ("1T", 1 << 40),
            ("1TiB", 1 << 40),
            ("64K", 64 * 1024),
            ("64kib", 64 * 1024),
            ("1", 1),
        ];
        for &(input, expected) in cases {
            assert_eq!(parse_mem_budget(input), Ok(expected), "input {input:?}");
        }
    }

    #[test]
    fn rejects_malformed_values_with_the_reason() {
        use MemBudgetParseError as E;
        let cases: &[(&str, E)] = &[
            ("", E::Empty),
            ("   ", E::Empty),
            ("0", E::Zero),
            ("0G", E::Zero),
            ("-1", E::MissingNumber),
            ("+5G", E::MissingNumber),
            ("abc", E::MissingNumber),
            ("G", E::MissingNumber),
            ("1.5G", E::UnknownSuffix(".5G".to_string())),
            ("32X", E::UnknownSuffix("X".to_string())),
            ("32 G B", E::UnknownSuffix(" G B".to_string())),
            ("32Gi", E::UnknownSuffix("Gi".to_string())),
            ("1E", E::UnknownSuffix("E".to_string())),
            ("99999999999T", E::Overflow),
            ("99999999999999999999999", E::Overflow),
            ("16777216T", E::Overflow), // 2^24 · 2^40 = 2^64
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_mem_budget(input).as_ref(),
                Err(expected),
                "input {input:?}"
            );
        }
        // The largest representable terabyte count still parses.
        assert_eq!(parse_mem_budget("16777215T"), Ok(16_777_215 << 40));
    }

    // --- apply_reserve -----------------------------------------------------------

    #[test]
    fn reserve_is_ten_percent_floored_at_512_mib_and_capped_at_half() {
        // Zero limit: nothing to reserve from, nothing left.
        assert_eq!(apply_reserve(0), 0);
        // Under 1 GiB the 512 MiB floor exceeds half the limit: the 50 % cap wins.
        assert_eq!(apply_reserve(GIB / 2), GIB / 4);
        assert_eq!(apply_reserve(GIB - 2), GIB / 2 - 1);
        // At exactly 1 GiB floor and cap coincide.
        assert_eq!(apply_reserve(GIB), GIB - 512 * MIB);
        // ~5 GiB: 10 % (≈ 512 MiB) — right at the crossover, the floor applies.
        assert_eq!(apply_reserve(5 * GIB), 5 * GIB - 512 * MIB);
        assert_eq!(apply_reserve(6 * GIB), 6 * GIB - 6 * GIB / 10);
        // A large node keeps a proportional 10 %.
        assert_eq!(apply_reserve(256 * GIB), 256 * GIB - 256 * GIB / 10);
        // No overflow at the top of the range.
        assert_eq!(apply_reserve(u64::MAX), u64::MAX - u64::MAX / 10);
    }

    // --- resolve_budget: the whole precedence table --------------------------------

    #[test]
    fn resolve_budget_precedence_table() {
        let detected = Some(100 * GIB);
        let detected_budget = apply_reserve(100 * GIB);
        let explicit = |bytes| MemBudget {
            bytes,
            source: BudgetSource::Explicit,
        };
        let from_detection = MemBudget {
            bytes: detected_budget,
            source: BudgetSource::Detected,
        };
        let fallback = MemBudget {
            bytes: DEFAULT_MEM_BUDGET_BYTES,
            source: BudgetSource::Fallback,
        };

        // Valid explicit wins over detection and over the fallback — even when it
        // is *larger* than what was detected (the user can always force).
        assert_eq!(
            resolve_budget(Some("32G"), detected),
            (explicit(32 * GIB), None)
        );
        assert_eq!(
            resolve_budget(Some("1T"), detected),
            (explicit(1 << 40), None)
        );
        assert_eq!(
            resolve_budget(Some("32G"), None),
            (explicit(32 * GIB), None)
        );
        // Unset: detection, else fallback.
        assert_eq!(resolve_budget(None, detected), (from_detection, None));
        assert_eq!(resolve_budget(None, None), (fallback, None));
        // Invalid explicit: ignored (reported), next source used.
        assert_eq!(
            resolve_budget(Some("abc"), detected),
            (from_detection, Some(MemBudgetParseError::MissingNumber))
        );
        assert_eq!(
            resolve_budget(Some(""), None),
            (fallback, Some(MemBudgetParseError::Empty))
        );
        // A detected limit that leaves no budget is not a real limit.
        assert_eq!(resolve_budget(None, Some(0)), (fallback, None));
    }

    #[test]
    fn the_invalid_value_warning_gate_opens_once() {
        let flag = AtomicBool::new(false);
        assert!(first_time(&flag));
        assert!(!first_time(&flag));
        assert!(!first_time(&flag));
    }

    // --- /proc and cgroup parsers ---------------------------------------------------

    const MEMINFO: &str = "MemTotal:       131072000 kB\n\
                           MemFree:         2000000 kB\n\
                           MemAvailable:   100000000 kB\n\
                           Buffers:          12345 kB\n";

    #[test]
    fn meminfo_reads_total_and_available_in_bytes() {
        assert_eq!(
            parse_meminfo(MEMINFO),
            MemInfo {
                total: Some(131_072_000 * 1024),
                available: Some(100_000_000 * 1024),
            }
        );
    }

    #[test]
    fn meminfo_tolerates_missing_and_malformed_lines() {
        assert_eq!(parse_meminfo(""), MemInfo::default());
        // An old kernel without MemAvailable.
        assert_eq!(
            parse_meminfo("MemTotal: 1024 kB\nMemFree: 10 kB\n").available,
            None
        );
        // Wrong unit, non-numeric, trailing junk: all ignored.
        assert_eq!(parse_meminfo("MemAvailable: 1024 MB\n").available, None);
        assert_eq!(parse_meminfo("MemAvailable: lots kB\n").available, None);
        assert_eq!(parse_meminfo("MemAvailable: 1 kB extra\n").available, None);
        assert_eq!(parse_meminfo("MemAvailable 1024 kB\n").available, None);
    }

    #[test]
    fn cgroup_v2_path_is_the_0_colon_colon_line() {
        assert_eq!(
            parse_cgroup_v2_path("0::/user.slice/session-4.scope\n"),
            Some("/user.slice/session-4.scope")
        );
        // Hybrid hierarchy: v1 lines around the unified one.
        assert_eq!(
            parse_cgroup_v2_path("12:memory:/job_1\n0::/job_1/step_0\n1:cpu:/x\n"),
            Some("/job_1/step_0")
        );
        assert_eq!(parse_cgroup_v2_path("0::/\n"), Some("/"));
        // Pure v1, empty, relative or escaping paths.
        assert_eq!(parse_cgroup_v2_path("4:memory:/job\n"), None);
        assert_eq!(parse_cgroup_v2_path(""), None);
        assert_eq!(parse_cgroup_v2_path("0::relative\n"), None);
        assert_eq!(parse_cgroup_v2_path("0::/a/../../etc\n"), None);
    }

    #[test]
    fn cgroup_v1_memory_path_picks_the_memory_controller() {
        let text = "5:cpu,cpuacct:/slurm/uid_1/job_9\n\
                    4:memory:/slurm/uid_1/job_9/step_0\n\
                    1:name=systemd:/x\n";
        assert_eq!(
            parse_cgroup_v1_memory_path(text),
            Some("/slurm/uid_1/job_9/step_0")
        );
        assert_eq!(
            parse_cgroup_v1_memory_path("3:blkio,memory:/j\n"),
            Some("/j")
        );
        // `memory` must be a whole controller name, not a substring.
        assert_eq!(parse_cgroup_v1_memory_path("3:memoryx:/j\n"), None);
        assert_eq!(parse_cgroup_v1_memory_path("0::/j\n"), None);
        assert_eq!(parse_cgroup_v1_memory_path("garbage"), None);
    }

    #[test]
    fn cgroup_limit_files_parse_numbers_and_max() {
        assert_eq!(parse_cgroup_limit("8589934592\n"), Some(8 * GIB));
        assert_eq!(parse_cgroup_limit("max\n"), None);
        assert_eq!(parse_cgroup_limit(""), None);
        assert_eq!(parse_cgroup_limit("-1"), None);
        assert_eq!(parse_cgroup_limit("8G"), None);
    }

    #[test]
    fn cgroup_limit_files_walk_every_ancestor() {
        assert_eq!(
            cgroup_limit_files("/r", "/a/b", "memory.max"),
            vec![
                PathBuf::from("/r/a/b/memory.max"),
                PathBuf::from("/r/a/memory.max"),
                PathBuf::from("/r/memory.max"),
            ]
        );
        assert_eq!(
            cgroup_limit_files("/r", "/", "memory.max"),
            vec![PathBuf::from("/r/memory.max")]
        );
    }

    /// A fake filesystem for [`detect_memory_limit_with`].
    fn fs(files: &[(&str, &str)]) -> impl Fn(&Path) -> Option<String> {
        let map: HashMap<PathBuf, String> = files
            .iter()
            .map(|(p, c)| (PathBuf::from(p), c.to_string()))
            .collect();
        move |p: &Path| map.get(p).cloned()
    }

    #[test]
    fn detection_without_a_cgroup_limit_is_mem_available() {
        let read = fs(&[
            ("/proc/meminfo", MEMINFO),
            ("/proc/self/cgroup", "0::/user.slice/s.scope\n"),
            ("/sys/fs/cgroup/user.slice/s.scope/memory.max", "max\n"),
            ("/sys/fs/cgroup/user.slice/memory.max", "max\n"),
        ]);
        assert_eq!(detect_memory_limit_with(read), Some(100_000_000 * 1024));
    }

    #[test]
    fn detection_takes_the_tightest_cgroup_v2_ancestor() {
        // The job is limited to 32 GiB, its step to "max", the user slice to 64 GiB:
        // the effective limit is the minimum over the whole ancestry.
        let read = fs(&[
            ("/proc/meminfo", MEMINFO),
            ("/proc/self/cgroup", "0::/slice/job/step\n"),
            ("/sys/fs/cgroup/slice/job/step/memory.max", "max\n"),
            ("/sys/fs/cgroup/slice/job/memory.max", "34359738368\n"),
            ("/sys/fs/cgroup/slice/memory.max", "68719476736\n"),
        ]);
        assert_eq!(detect_memory_limit_with(read), Some(32 * GIB));
    }

    #[test]
    fn detection_keeps_mem_available_when_it_is_tighter_than_the_cgroup() {
        let read = fs(&[
            (
                "/proc/meminfo",
                "MemTotal: 16777216 kB\nMemAvailable: 4194304 kB\n",
            ),
            ("/proc/self/cgroup", "0::/job\n"),
            ("/sys/fs/cgroup/job/memory.max", "17179869184\n"),
        ]);
        assert_eq!(detect_memory_limit_with(read), Some(4 * GIB));
    }

    #[test]
    fn detection_reads_cgroup_v1_and_treats_huge_values_as_unlimited() {
        let v1 = |limit: &'static str| {
            fs(&[
                ("/proc/meminfo", MEMINFO),
                ("/proc/self/cgroup", "4:memory:/slurm/job_9\n"),
                (
                    "/sys/fs/cgroup/memory/slurm/job_9/memory.limit_in_bytes",
                    limit,
                ),
            ])
        };
        assert_eq!(detect_memory_limit_with(v1("8589934592\n")), Some(8 * GIB));
        // The v1 "unlimited" sentinel, and a limit above MemTotal, are no limit.
        assert_eq!(
            detect_memory_limit_with(v1("9223372036854771712\n")),
            Some(100_000_000 * 1024)
        );
        assert_eq!(
            detect_memory_limit_with(v1("274877906944\n")), // 256 GiB > MemTotal
            Some(100_000_000 * 1024)
        );
        assert_eq!(
            detect_memory_limit_with(v1("garbage")),
            Some(100_000_000 * 1024)
        );
    }

    #[test]
    fn detection_with_nothing_readable_is_none() {
        // No /proc at all (macOS, Windows).
        assert_eq!(detect_memory_limit_with(fs(&[])), None);
        // /proc/self/cgroup present but its files absent, and no meminfo.
        assert_eq!(
            detect_memory_limit_with(fs(&[("/proc/self/cgroup", "0::/x\n")])),
            None
        );
    }

    #[test]
    fn detection_uses_a_cgroup_limit_even_without_meminfo() {
        let read = fs(&[
            ("/proc/self/cgroup", "0::/\n"),
            ("/sys/fs/cgroup/memory.max", "2147483648\n"),
        ]);
        assert_eq!(detect_memory_limit_with(read), Some(2 * GIB));
    }

    // --- check_fits ---------------------------------------------------------------

    fn budget(bytes: u64, source: BudgetSource) -> MemBudget {
        MemBudget { bytes, source }
    }

    #[test]
    fn check_fits_rejects_against_explicit_and_detected_budgets() {
        for source in [BudgetSource::Explicit, BudgetSource::Detected] {
            // 20 qubits = 16 MiB against a 1 MiB budget.
            let err = check_fits(20, budget(MIB, source)).unwrap_err();
            assert_eq!(
                err,
                InsufficientMemory {
                    num_qubits: 20,
                    required_bytes: 16 * MIB,
                    budget_bytes: MIB,
                    source,
                }
            );
            // Exactly the budget fits; one byte less does not.
            assert!(check_fits(16, budget(MIB, source)).is_ok());
            assert!(check_fits(16, budget(MIB - 1, source)).is_err());
        }
    }

    #[test]
    fn check_fits_never_rejects_against_the_fallback() {
        assert!(check_fits(30, budget(MIB, BudgetSource::Fallback)).is_ok());
        assert!(check_fits(
            100,
            budget(DEFAULT_MEM_BUDGET_BYTES, BudgetSource::Fallback)
        )
        .is_ok());
    }

    #[test]
    fn check_fits_saturates_for_absurd_widths() {
        // Even a u64::MAX budget cannot hold a 60-qubit statevector (2^64 bytes).
        for n in [60, 64, 200] {
            let err = check_fits(n, budget(u64::MAX, BudgetSource::Explicit)).unwrap_err();
            assert_eq!(err.required_bytes, u64::MAX);
            assert!(err.to_string().contains("more than"), "{err}");
        }
    }

    #[test]
    fn the_refusal_message_is_actionable() {
        let explicit = check_fits(20, budget(MIB, BudgetSource::Explicit))
            .unwrap_err()
            .to_string();
        assert!(explicit.contains("20-qubit"), "{explicit}");
        assert!(explicit.contains("16.00 MiB"), "{explicit}");
        assert!(explicit.contains("1.00 MiB"), "{explicit}");
        assert!(explicit.contains("set by POLYPUS_MEM_BUDGET"), "{explicit}");
        assert!(explicit.contains("POLYPUS_MEM_BUDGET=64G"), "{explicit}");

        let detected = check_fits(34, budget(100 * GIB, BudgetSource::Detected))
            .unwrap_err()
            .to_string();
        assert!(detected.contains("34-qubit"), "{detected}");
        assert!(detected.contains("256.00 GiB"), "{detected}");
        assert!(detected.contains("100.00 GiB"), "{detected}");
        assert!(detected.contains("cgroup"), "{detected}");
        assert!(detected.contains("POLYPUS_MEM_BUDGET"), "{detected}");
    }

    #[test]
    fn format_bytes_uses_binary_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(16 * GIB), "16.00 GiB");
        assert_eq!(format_bytes(u64::MAX), "16.00 EiB");
    }
}
