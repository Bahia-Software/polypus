//! The result of running the conformance battery: a list of named [`Check`]s, each
//! [`Passed`](Status::Passed), [`Failed`](Status::Failed) or
//! [`Skipped`](Status::Skipped), plus the machinery to print and assert on them.

use std::fmt;

/// The outcome of one conformance [`Check`].
///
/// [`Skipped`](Self::Skipped) is a first-class outcome, **not** a pass: a check is
/// skipped when it does not apply to this backend (e.g. an error-classification
/// check for which the author supplied no fault injector, or a shot-distribution
/// check on a backend that declares it does not support shot distribution). The
/// report keeps skips visible so "passes the battery" always means "and here is
/// exactly what was and was not exercised".
#[derive(Debug, Clone)]
pub enum Status {
    /// The check ran and the backend behaved as the contract requires.
    Passed,
    /// The check ran and the backend violated the contract. The string explains how.
    Failed(String),
    /// The check does not apply to this backend. The string explains why.
    Skipped(String),
}

/// One named conformance check and its [`Status`].
#[derive(Debug, Clone)]
pub struct Check {
    /// Stable identifier of the check (e.g. `"shots_are_conserved"`).
    pub name: &'static str,
    /// One line describing what the check verifies.
    pub description: &'static str,
    /// What happened when it ran.
    pub status: Status,
}

impl Check {
    pub(crate) fn passed(name: &'static str, description: &'static str) -> Self {
        Check {
            name,
            description,
            status: Status::Passed,
        }
    }

    pub(crate) fn failed(
        name: &'static str,
        description: &'static str,
        why: impl Into<String>,
    ) -> Self {
        Check {
            name,
            description,
            status: Status::Failed(why.into()),
        }
    }

    pub(crate) fn skipped(
        name: &'static str,
        description: &'static str,
        why: impl Into<String>,
    ) -> Self {
        Check {
            name,
            description,
            status: Status::Skipped(why.into()),
        }
    }
}

/// The full battery result for one backend.
#[derive(Debug, Clone)]
pub struct Report {
    /// The backend's label, echoed into the printed summary.
    pub backend: String,
    /// Every check the battery ran, in execution order.
    pub checks: Vec<Check>,
}

impl Report {
    /// Number of checks that passed.
    pub fn passed_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| matches!(c.status, Status::Passed))
            .count()
    }

    /// Number of checks that failed.
    pub fn failed_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| matches!(c.status, Status::Failed(_)))
            .count()
    }

    /// Number of checks skipped as not-applicable.
    pub fn skipped_count(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| matches!(c.status, Status::Skipped(_)))
            .count()
    }

    /// Whether the backend conforms: **no check failed**. Skipped checks do not
    /// count against conformance — a backend that legitimately cannot be driven
    /// into a fault still conforms for the behaviour it does implement.
    pub fn is_conformant(&self) -> bool {
        self.failed_count() == 0
    }

    /// Panic with the full report unless every check passed or was skipped. The
    /// one-liner for a `#[test]`: `Conformance::new(..).run().assert_conformant()`.
    pub fn assert_conformant(&self) {
        assert!(
            self.is_conformant(),
            "backend '{}' failed the conformance battery:\n{self}",
            self.backend
        );
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "conformance report for backend '{}': {} passed, {} failed, {} skipped",
            self.backend,
            self.passed_count(),
            self.failed_count(),
            self.skipped_count()
        )?;
        for check in &self.checks {
            match &check.status {
                Status::Passed => writeln!(f, "  [PASS] {} — {}", check.name, check.description)?,
                Status::Failed(why) => {
                    writeln!(f, "  [FAIL] {} — {}", check.name, check.description)?;
                    writeln!(f, "         {why}")?;
                }
                Status::Skipped(why) => {
                    writeln!(f, "  [SKIP] {} — {}", check.name, check.description)?;
                    writeln!(f, "         {why}")?;
                }
            }
        }
        Ok(())
    }
}
