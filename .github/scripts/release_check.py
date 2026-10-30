"""Version rules for a release, checked by the `gate` job of release.yml and,
on every pull request and push to main, by hygiene.yml.

Always:
  * `[workspace.package].version` in Cargo.toml is `X.Y.Z` or, between
    releases, `X.Y.Z-dev`;
  * every internal `[workspace.dependencies]` entry (one with a `path`) pins
    exactly that version, so `cargo set-version --workspace` left none behind.

For a release (`--tag vX.Y.Z` and/or `--publish`):
  * the version is `X.Y.Z`, never `-dev`;
  * the tag is `v` + that version;
  * CHANGELOG.md has a `## [X.Y.Z] - YYYY-MM-DD` section;
  * CITATION.cff has that `version` and `date-released`.

    python .github/scripts/release_check.py                        # branch
    python .github/scripts/release_check.py --tag v0.8.0 --publish # release

Errors are printed as GitHub annotations; exit status 1 if there are any.
On success the version is appended to $GITHUB_OUTPUT as `version=...`.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import tomllib

RELEASE = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
RELEASE_VERSION = re.compile(RELEASE)
DEV_VERSION = re.compile(RELEASE + "-dev")
TAG = re.compile("v" + RELEASE)
CHANGELOG_SECTION = re.compile(
    r"^## \[([^\]]+)\](?: - ([0-9]{4}-[0-9]{2}-[0-9]{2}))?[ \t]*$", re.M
)


def cff_field(text, key):
    """Root-level scalar of CITATION.cff (plain or quoted, one line)."""
    m = re.search(rf"^{key}:[ \t]*[\"']?([^\"'\s#]+)[\"']?[ \t]*(?:#.*)?$", text, re.M)
    return m.group(1) if m else None


def check(root, tag=None, publish=False):
    root = Path(root)
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    version = cargo["workspace"]["package"]["version"]
    errors = []

    if not (RELEASE_VERSION.fullmatch(version) or DEV_VERSION.fullmatch(version)):
        errors.append(
            f"Cargo.toml: unsupported workspace version '{version}' "
            "(expected X.Y.Z, or X.Y.Z-dev between releases)"
        )
    for name, spec in cargo["workspace"].get("dependencies", {}).items():
        if not isinstance(spec, dict) or "path" not in spec:
            continue
        pin = spec.get("version")
        if pin is None:
            errors.append(f"Cargo.toml: internal dependency {name} has no version")
        elif pin != version:
            errors.append(
                f"Cargo.toml: internal dependency {name} pins version '{pin}', "
                f"not the workspace version '{version}'"
            )

    if tag is not None:
        if not TAG.fullmatch(tag):
            errors.append(f"tag '{tag}' is not of the form vX.Y.Z")
        elif tag != f"v{version}" and not DEV_VERSION.fullmatch(version):
            errors.append(
                f"tag '{tag}' does not match the Cargo.toml workspace version "
                f"'{version}' (expected 'v{version}')"
            )
    if tag is None and not publish:
        return errors

    if DEV_VERSION.fullmatch(version):
        errors.append(
            f"development version '{version}' cannot be released: "
            "set the release version first (docs/RELEASING.md)"
        )
        return errors

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    dates = {v: d for v, d in CHANGELOG_SECTION.findall(changelog)}
    date = dates.get(version)
    if not date:
        errors.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    cff_version = cff_field(cff, "version")
    if cff_version != version:
        errors.append(
            f"CITATION.cff version '{cff_version}' is not the release version "
            f"'{version}'"
        )
    cff_date = cff_field(cff, "date-released")
    if date and cff_date != date:
        errors.append(
            f"CITATION.cff date-released '{cff_date}' is not the CHANGELOG.md "
            f"date of {version}, '{date}'"
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--tag", help="the release tag being pushed")
    parser.add_argument(
        "--publish", action="store_true", help="the run publishes to an index"
    )
    args = parser.parse_args()
    errors = check(args.root, tag=args.tag, publish=args.publish)
    for e in errors:
        print(f"::error::{e}")
    if errors:
        return 1
    root = Path(args.root)
    version = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]["package"]["version"]
    print(f"Release checks passed for version {version}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"version={version}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
