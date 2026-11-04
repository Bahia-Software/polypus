"""Redirect the old rustdoc URLs to their new home under ``/rust/``.

Before the documentation site, rustdoc was the Pages root, so every crate lived
at ``<site>/<crate>/``. The site now owns the root and rustdoc sits in
``<site>/rust/``; this writes ``<site>/<crate>/index.html`` redirects so
existing links keep working.

Usage: python scripts/docs/write_legacy_redirects.py SITE_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PACKAGE_TABLE = re.compile(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S)
_NAME = re.compile(r'^name\s*=\s*"([^"]+)"', re.M)

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Redirecting to the {crate} API documentation</title>
    <link rel="canonical" href="../rust/{crate}/index.html" />
    <meta http-equiv="refresh" content="0; url=../rust/{crate}/index.html" />
  </head>
  <body>
    <p>This page moved to <a href="../rust/{crate}/index.html">the {crate} API documentation</a>.</p>
  </body>
</html>
"""


def read_crate_names(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return the rustdoc directory name of every ``crates/*`` package."""
    names = []
    for manifest in sorted(repo_root.glob("crates/*/Cargo.toml")):
        table = _PACKAGE_TABLE.search(manifest.read_text(encoding="utf-8"))
        match = _NAME.search(table.group(1)) if table else None
        if match is None:
            raise ValueError(f"no [package] name in {manifest}")
        names.append(match.group(1).replace("-", "_"))
    return names


def write_legacy_redirects(site_dir: Path, crate_names: list[str]) -> list[Path]:
    """Write ``site_dir/<crate>/index.html`` for each crate.

    Raises ``FileExistsError`` rather than replace a page of the site.
    """
    targets = [site_dir / name / "index.html" for name in crate_names]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"{target} already exists; not overwriting")
    for name, target in zip(crate_names, targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_TEMPLATE.format(crate=name), encoding="utf-8")
    return targets


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    written = write_legacy_redirects(Path(argv[1]), read_crate_names())
    print(f"wrote {len(written)} redirects")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
