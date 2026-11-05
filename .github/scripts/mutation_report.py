"""Mutation-testing results of .github/workflows/mutation.yml, as Markdown.

Each matrix job of the workflow runs `cargo mutants` over one shard of one
crate (`--shard K/N --sharding round-robin`: every N-th mutant, from the K-th;
a crate in one piece is shard 0/1) and keeps its `mutants.out/` files:
`mutants.json` (every mutant the shard will test, written before the first),
`outcomes.json` (the counts of those tested so far, rewritten after each),
`missed.txt`, and `exit-code.txt`, cargo-mutants' exit code, which the workflow
writes once the tool returns. Exit codes 0 (every mutant caught), 2 (some
missed) and 3 (some timed out) are a completed run; the workflow is
report-only, so none of them fails the job. Any other code, or no code at all
(the step timed out or was cancelled), means the shard was not run. A crate is
reported as run only when all of its shards were, and as not run, with each
missing shard's reason, otherwise: never left out.

The score is `caught / (caught + missed)`: unviable mutants (they do not build)
and timeouts are counted but not scored.

    python .github/scripts/mutation_report.py summary polypus-sim mutants.out --shard 1/3
    python .github/scripts/mutation_report.py comment --artifacts DIR --run-url URL CRATE[:SHARDS]...

`summary` prints one shard's job summary; `comment` prints the weekly comment
for the tracking issue, one row per crate, from the artifacts downloaded into
DIR: one directory per shard, named as `artifact_name` names it.
"""

import argparse
import json
import sys
from pathlib import Path

COMPLETED = {0: "every mutant caught", 2: "missed mutants", 3: "timeouts"}
COUNTS = ("caught", "missed", "timeout", "unviable")
# How many missed mutants a job summary lists before pointing at the artifact.
MAX_LISTED = 100


def artifact_name(crate, shard, shards):
    """The artifact a shard's job uploads: `mutants-polypus-sim-1of3` for
    `--shard 1/3`, `mutants-polypus-backend-0of1` for a crate in one piece."""
    return f"mutants-{crate}-{shard}of{shards}"


def parse_shard(text):
    """`"K/N"` as `(K, N)`, with `0 <= K < N`."""
    try:
        shard, shards = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected K/N, got {text!r}") from None
    if not 0 <= shard < shards:
        raise argparse.ArgumentTypeError(f"expected 0 <= K < N, got {text!r}")
    return shard, shards


def parse_crate(text):
    """`"polypus-sim:3"` as `("polypus-sim", 3)`; no `:N` is one shard."""
    name, _, shards = text.partition(":")
    try:
        count = int(shards) if shards else 1
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected CRATE[:SHARDS], got {text!r}"
        ) from None
    if not name or count < 1:
        raise argparse.ArgumentTypeError(f"expected CRATE[:SHARDS], got {text!r}")
    return name, count


def planned(directory):
    """How many mutants the run set out to test (the length of
    `mutants.json`), or None when that list is missing or unreadable.
    `outcomes.json`'s `total_mutants` cannot tell: it counts the mutants tested
    so far."""
    try:
        mutants = json.loads(
            (Path(directory) / "mutants.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return len(mutants) if isinstance(mutants, list) else None


def load(directory):
    """One shard's result from its `mutants.out` files: a dict with `ran`
    (bool), `reason` (why it did not run, else None), the four counts (None
    when `outcomes.json` is missing or unreadable) and `planned`."""
    directory = Path(directory)
    result = {"ran": False, "reason": None, "planned": planned(directory)}
    result.update(dict.fromkeys(COUNTS))
    try:
        outcomes = json.loads((directory / "outcomes.json").read_text(encoding="utf-8"))
        for key in COUNTS:
            result[key] = int(outcomes[key])
    except (OSError, ValueError, KeyError, TypeError):
        result["reason"] = (
            "no results (cargo-mutants did not start, or the artifact is missing)"
        )
        return result
    try:
        code = int((directory / "exit-code.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        result["reason"] = (
            "cargo-mutants did not finish (the step timed out or was cancelled)"
        )
        return result
    if code == 4:
        result["reason"] = (
            "the tests fail without any mutation (cargo-mutants exit code 4)"
        )
    elif code not in COMPLETED:
        result["reason"] = f"cargo-mutants failed (exit code {code})"
    else:
        result["ran"] = True
    return result


def merge(shards):
    """One crate's result from those of its shards, in shard order: their
    counts added up when every shard ran; otherwise not run, with the reason
    of each shard that did not (prefixed by `shard K/N` when there are
    several)."""
    if len(shards) == 1:
        return shards[0]
    failed = [
        f"shard {k}/{len(shards)}: {result['reason']}"
        for k, result in enumerate(shards)
        if not result["ran"]
    ]
    if failed:
        merged = {"ran": False, "reason": "; ".join(failed), "planned": None}
        merged.update(dict.fromkeys(COUNTS))
        return merged
    plans = [result["planned"] for result in shards]
    merged = {
        "ran": True,
        "reason": None,
        "planned": None if None in plans else sum(plans),
    }
    merged.update({key: sum(result[key] for result in shards) for key in COUNTS})
    return merged


def score(caught, missed):
    """`caught / (caught + missed)` as a percentage string, `–` when nothing
    was scored."""
    scored = caught + missed
    return "–" if scored == 0 else f"{100.0 * caught / scored:.1f} %"


def row(name, result):
    if not result["ran"]:
        return f"| `{name}` | **not run**: {result['reason']} | – | – | – | – | – |"
    counts = " | ".join(str(result[key]) for key in COUNTS)
    return (
        f"| `{name}` | ran | {counts} | {score(result['caught'], result['missed'])} |"
    )


HEADER = [
    "| Crate | Result | Caught | Missed | Timeout | Unviable | Score |",
    "|---|---|---:|---:|---:|---:|---:|",
]


def table(results):
    """The per-crate table, in the given order, plus a total over the crates
    that ran when there is more than one crate."""
    out = list(HEADER)
    out += [row(name, result) for name, result in results]
    ran = [result for _, result in results if result["ran"]]
    if ran and len(results) > 1:
        totals = {key: sum(r[key] for r in ran) for key in COUNTS}
        counts = " | ".join(str(totals[key]) for key in COUNTS)
        label = f"{len(ran)} of {len(results)} crates"
        scored = score(totals["caught"], totals["missed"])
        out.append(f"| **total** | {label} | {counts} | {scored} |")
    return out


def summary(name, directory, shard=(0, 1)):
    """One shard's job summary: its row, and the missed mutants it lists."""
    result = load(directory)
    label = name if shard[1] == 1 else f"{name}, shard {shard[0]}/{shard[1]}"
    out = [f"### Mutation testing: `{label}`", ""] + table([(name, result)])
    if result["caught"] is not None:
        tested = sum(result[key] for key in COUNTS)
        of = "" if result["planned"] is None else f" of {result['planned']}"
        out += ["", f"{tested}{of} mutants tested."]
    try:
        missed = (
            (Path(directory) / "missed.txt").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        missed = []
    missed = [line for line in missed if line.strip()]
    if missed:
        out += ["", "<details><summary>Missed mutants</summary>", "", "```text"]
        out += missed[:MAX_LISTED]
        out.append("```")
        if len(missed) > MAX_LISTED:
            out.append(
                f"…and {len(missed) - MAX_LISTED} more: see `missed.txt` in the artifact."
            )
        out += ["", "</details>"]
    return "\n".join(out) + "\n"


def comment(crates, artifacts, run_url):
    """The weekly comment on the tracking issue: one row per crate of
    `crates` (`(name, shards)` pairs), those that did not run included."""
    artifacts = Path(artifacts)
    results = [
        (
            name,
            merge(
                [
                    load(artifacts / artifact_name(name, k, shards))
                    for k in range(shards)
                ]
            ),
        )
        for name, shards in crates
    ]
    out = [f"Weekly `cargo mutants` run: {run_url}", ""] + table(results)
    out += [
        "",
        "Score: caught / (caught + missed). The missed mutants are in the run's "
        "`mutants-<crate>-<K>of<N>` artifacts, one per shard (`missed.txt`, "
        "`outcomes.json`).",
    ]
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("summary", help="one shard's job summary")
    one.add_argument("crate")
    one.add_argument("directory", type=Path, help="the shard's mutants.out directory")
    one.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="K/N")
    weekly = commands.add_parser("comment", help="the tracking issue's comment")
    weekly.add_argument("--artifacts", type=Path, required=True)
    weekly.add_argument("--run-url", required=True)
    weekly.add_argument("crates", nargs="+", type=parse_crate, metavar="CRATE[:SHARDS]")
    args = parser.parse_args(argv)
    if args.command == "summary":
        sys.stdout.write(summary(args.crate, args.directory, args.shard))
    else:
        sys.stdout.write(comment(args.crates, args.artifacts, args.run_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
