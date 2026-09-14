//! Memory budget for concurrent statevector simulation (plan §4.5, P1-memory).
//!
//! The dominant memory cost of the local simulators (native Rust and Aer) is not
//! the list of bound circuits — cheap — but the **concurrent statevectors**: an
//! `n`-qubit circuit under simulation holds `2^n` complex-`f64` amplitudes =
//! `2^n * 16` bytes. Running a whole population in parallel therefore peaks at
//! roughly `concurrency * 2^n * 16` bytes, which OOMs at high `n` (a single
//! 30-qubit statevector is already 16 GiB).
//!
//! This module turns a byte budget into a safe concurrency cap so a backend can
//! bound its peak memory without changing any result: the seeds are independent
//! of how many circuits run at once, so throttling concurrency trades only speed,
//! never counts. The budget comes from `POLYPUS_MEM_BUDGET` (bytes) when set —
//! HPC job scripts set it to match the SLURM/cgroup memory allocation — and falls
//! back to a conservative default otherwise.

use std::env;

/// Conservative default memory budget (bytes) used when `POLYPUS_MEM_BUDGET` is
/// unset: 16 GiB. Chosen so a lone 30-qubit statevector (16 GiB) still runs, full
/// multi-threading is retained up to ~25 qubits, and only the high-qubit regime
/// that used to OOM is throttled.
///
/// **HPC runs on larger nodes should set `POLYPUS_MEM_BUDGET`** to their memory
/// allocation to recover full concurrency at high qubit counts; the default is a
/// safety net for an unconfigured process, not a performance target.
pub const DEFAULT_MEM_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Environment variable overriding [`DEFAULT_MEM_BUDGET_BYTES`], read as an
/// unsigned byte count.
pub const MEM_BUDGET_ENV: &str = "POLYPUS_MEM_BUDGET";

/// Bytes held by one `n`-qubit statevector: `2^n` complex-`f64` amplitudes × 16.
///
/// Saturates to `u64::MAX` for absurd `n` (≥ 60) so the cap computation never
/// overflows or panics — such a circuit simply resolves to a concurrency of 1
/// (run one at a time), the only safe choice when a single vector cannot fit.
fn statevector_bytes(num_qubits: usize) -> u64 {
    1u64.checked_shl(num_qubits as u32)
        .and_then(|amps| amps.checked_mul(16))
        .unwrap_or(u64::MAX)
}

/// The active byte budget: `POLYPUS_MEM_BUDGET` when it parses to a positive
/// integer, else [`DEFAULT_MEM_BUDGET_BYTES`]. A zero or unparseable value falls
/// back to the default (a zero budget would forbid all work).
fn budget_bytes() -> u64 {
    env::var(MEM_BUDGET_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(DEFAULT_MEM_BUDGET_BYTES)
}

/// Maximum `n`-qubit statevectors that may be simulated concurrently under the
/// active memory budget, never exceeding `num_threads` and never below 1
/// (plan §4.5): `max(1, min(num_threads, budget / (2^n * 16)))`.
pub fn max_statevector_concurrency(num_qubits: usize, num_threads: usize) -> usize {
    concurrency_for(budget_bytes(), num_qubits, num_threads)
}

/// Pure budget arithmetic behind [`max_statevector_concurrency`], split out so it
/// can be unit-tested without touching the process environment.
fn concurrency_for(budget_bytes: u64, num_qubits: usize, num_threads: usize) -> usize {
    let sv = statevector_bytes(num_qubits);
    // `sv >= 16 > 0`, so the division never traps; `.max(1)` keeps at least one
    // circuit in flight even when a single vector exceeds the whole budget.
    let by_budget = (budget_bytes / sv).max(1).min(usize::MAX as u64) as usize;
    num_threads.max(1).min(by_budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_qubit_counts_keep_full_thread_concurrency() {
        // At 8 qubits a statevector is 4 KiB; the 16 GiB default dwarfs it, so the
        // cap is purely `num_threads` — i.e. today's unbounded-`par_iter` behaviour
        // is preserved exactly for the common low-qubit workloads.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 8, 32), 32);
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 20, 32), 32);
    }

    #[test]
    fn high_qubit_counts_are_throttled_by_the_budget() {
        // 28 qubits => 4 GiB/vector => 16 GiB / 4 GiB = 4 concurrent vectors.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 28, 32), 4);
        // 30 qubits => 16 GiB/vector => exactly one fits in the 16 GiB default.
        assert_eq!(concurrency_for(DEFAULT_MEM_BUDGET_BYTES, 30, 32), 1);
    }

    #[test]
    fn a_single_vector_larger_than_the_budget_still_runs_one_at_a_time() {
        // 30-qubit vector (16 GiB) under a 1 GiB budget: cannot fit, but we must
        // never return 0 — best-effort is to run exactly one at a time.
        assert_eq!(concurrency_for(1024 * 1024 * 1024, 30, 32), 1);
        // Absurd qubit count saturates the statevector size to u64::MAX instead of
        // overflowing; even the largest possible budget then fits exactly one.
        assert_eq!(concurrency_for(u64::MAX, 100, 8), 1);
    }

    #[test]
    fn a_larger_budget_recovers_concurrency_at_high_qubits() {
        // Doubling the budget to 32 GiB doubles the 28-qubit cap from 4 to 8: the
        // env override an HPC job sets recovers concurrency the default throttled.
        assert_eq!(concurrency_for(32u64 * 1024 * 1024 * 1024, 28, 32), 8);
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
        assert_eq!(statevector_bytes(30), 16 * 1024 * 1024 * 1024);
        // n >= 60 overflows u64 and must saturate rather than panic.
        assert_eq!(statevector_bytes(64), u64::MAX);
        assert_eq!(statevector_bytes(200), u64::MAX);
    }
}
