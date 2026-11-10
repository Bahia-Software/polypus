"""Line coverage per crate, from an LCOV report, as a Markdown table.

Used by .github/workflows/sonar.yml to write the job summary of each coverage
job: the `rust` job's report (the `test (Rust)` passes) and the `python` job's
(the Rust the pytest suite reaches through the instrumented wheel). A source
file belongs to the crate whose `crates/<name>/` directory holds it; anything
else (an example, the fuzz crate) is grouped under "other".

A file's lines are its record's `LF:` (lines found) and `LH:` (lines hit), the
figures `cargo llvm-cov report` itself prints. They are not the count of its
`DA:` (per-line) records: llvm-cov's LCOV export declares a few percent more
lines than it lists (`mem_budget.rs`: 637 found, 614 `DA:`). A record without
`LF:`/`LH:` falls back to its `DA:` records, a line counting once however many
times it ran.

    python .github/scripts/coverage_summary.py lcov.info --title "Rust coverage (cargo test)"
"""

import argparse
import re
import sys
from pathlib import Path

CRATE_DIR = re.compile(r"(?:^|(?<=/))crates/([^/]+)/")
OTHER = "other"


def crate_of(path):
    """The crate a source path belongs to, or `OTHER`. The last `crates/<name>/`
    wins, so a checkout that itself lives under a `crates/` directory still
    maps to the workspace crate."""
    matches = CRATE_DIR.findall(path.replace("\\", "/"))
    return matches[-1] if matches else OTHER


def _count(record, value):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"malformed LCOV record: {record!r}") from None


def parse_lcov(text):
    """`{source path: (lines, covered)}` from the records of an LCOV report:
    `LF:`/`LH:` when the record has both, else its `DA:` records. A file with
    two records (as a merged report may hold) keeps the larger `LF:` and `LH:`,
    and, per line, the larger `DA:` count."""
    files = {}
    entry = None
    for raw in text.splitlines():
        record = raw.strip()
        if record.startswith("SF:"):
            entry = files.setdefault(record[3:], {"da": {}, "lf": None, "lh": None})
        elif entry is None:
            continue
        elif record.startswith("DA:"):
            fields = record[3:].split(",")
            if len(fields) < 2:
                raise ValueError(f"malformed LCOV line record: {record!r}")
            line, hits = _count(record, fields[0]), _count(record, fields[1])
            entry["da"][line] = max(hits, entry["da"].get(line, 0))
        elif record.startswith(("LF:", "LH:")):
            key = record[:2].lower()
            entry[key] = max(_count(record, record[3:]), entry[key] or 0)
        elif record == "end_of_record":
            entry = None
    return {
        path: (e["lf"], e["lh"])
        if e["lf"] is not None and e["lh"] is not None
        else (len(e["da"]), sum(1 for h in e["da"].values() if h > 0))
        for path, e in files.items()
    }


def per_crate(files):
    """`{crate: (lines, covered)}`, from `parse_lcov`'s result."""
    totals = {}
    for path, (lines, covered) in files.items():
        name = crate_of(path)
        found, hit = totals.get(name, (0, 0))
        totals[name] = (found + lines, hit + covered)
    return totals


def percent(covered, lines):
    return "–" if lines == 0 else f"{100.0 * covered / lines:.1f} %"


def markdown(totals, title):
    """The job-summary section: one row per crate, alphabetical, "other" last,
    then the total."""
    out = [f"### {title}", ""]
    if not totals:
        out.append("The report holds no line records.")
        return "\n".join(out) + "\n"
    out += ["| Crate | Lines | Covered | Line coverage |", "|---|---:|---:|---:|"]
    names = sorted(n for n in totals if n != OTHER)
    if OTHER in totals:
        names.append(OTHER)
    for name in names:
        lines, covered = totals[name]
        out.append(f"| `{name}` | {lines} | {covered} | {percent(covered, lines)} |")
    lines = sum(found for found, _ in totals.values())
    covered = sum(hit for _, hit in totals.values())
    out.append(f"| **total** | {lines} | {covered} | {percent(covered, lines)} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "lcov", type=Path, help="LCOV report (cargo llvm-cov report --lcov)"
    )
    parser.add_argument("--title", default="Line coverage per crate")
    args = parser.parse_args(argv)
    try:
        text = args.lcov.read_text(encoding="utf-8")
        totals = per_crate(parse_lcov(text))
    except (OSError, ValueError) as err:
        print(f"coverage_summary: {err}", file=sys.stderr)
        return 2
    sys.stdout.write(markdown(totals, args.title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
