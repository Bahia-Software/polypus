"""Scan the built documentation site for confidential data before publishing.

GitHub Pages is public and, with versioned deploys, permanent: a leak cannot be
undone by redeploying. This gate runs on the assembled site, before upload.

Usage: python scripts/docs/scan_published_site.py SITE_DIR [--patterns-env NAME]

``--patterns-env`` names an environment variable holding one extra private
regular expression, the same format as ``HYGIENE_LEAK_PATTERNS`` in
``.github/scripts/hygiene.py`` (a single regex; use ``|`` for alternatives).
An empty or unset variable adds nothing. Private patterns are never printed:
findings only name ``private-pattern-<index>`` and mask the whole match.

``--require-patterns`` makes an empty or unset variable a configuration error
(as ``--require-leak-patterns`` does in hygiene); CI passes it off pull requests.

Private patterns are matched with hygiene's own ``private_match`` (per line, in
bounded windows), imported from ``.github/scripts/hygiene.py``.

Exit codes: 0 clean, 1 findings, 2 configuration error.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

_HYGIENE_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "hygiene.py"
)
_spec = importlib.util.spec_from_file_location("polypus_hygiene", _HYGIENE_PATH)
assert _spec is not None and _spec.loader is not None
hygiene = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hygiene
_spec.loader.exec_module(hygiene)

# rustdoc output is skipped wholesale.
SKIPPED_PREFIXES = ("rust/",)
# mkdocs-material's generated theme files; anything else in these directories
# is ours and is scanned.
THEME_DIRS = ("assets/javascripts/", "assets/stylesheets/")
THEME_SUBTREES = ("lunr/", "workers/")
THEME_SUFFIXES = (".min.js", ".min.css", ".map")
EXCERPT_MAX = 80
VISIBLE = 4
ALLOWED_EMAIL = "diego.fernandez@bahiasoftware.es"
NOREPLY_SUFFIX = "@users.noreply.github.com"

# Generic rules only. Site-specific terms (account paths, queue and node
# names, hostnames) live in the private HYGIENE_LEAK_PATTERNS secret, never in
# this public file. Path and host rules are hygiene's own.
RULES = {
    name: hygiene.PUBLIC_LEAK_RULES[name] for name in ("cluster-path", "login-host")
}
ZMQ_ENDPOINT = re.compile(r"tcp://(?!localhost|127\.0\.0\.1|qmio\.example)[^\s\"'<]+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CREDENTIALS = [
    hygiene.PUBLIC_LEAK_RULES["private-key"],
    hygiene.PUBLIC_LEAK_RULES["github-token"],
    re.compile(r"github_pat_\w{20,}"),
    re.compile(r"pypi-AgE[\w-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*[\"'][^\"']{8,}"),
]
BASE64_DATA_URI = re.compile(r"data:[^;,\s\"']*;base64,[A-Za-z0-9+/=]+")
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{200,}")


class Finding(NamedTuple):
    path: str
    line: int
    rule: str
    excerpt: str


def mask(text: str, visible: int = VISIBLE) -> str:
    """Keep the first ``visible`` characters, hide the rest, cap at 80."""
    return (text[:visible] + "*" * max(0, len(text) - visible))[:EXCERPT_MAX]


def strip_base64(text: str) -> str:
    return BASE64_RUN.sub("", BASE64_DATA_URI.sub("", text))


def is_vendored_theme_file(rel: str) -> bool:
    for theme_dir in THEME_DIRS:
        if rel.startswith(theme_dir):
            inner = rel[len(theme_dir) :]
            return inner.startswith(THEME_SUBTREES) or inner.endswith(THEME_SUFFIXES)
    return False


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_text(
    rel: str, text: str, extra: Sequence["re.Pattern[str]"]
) -> List[Finding]:
    """``text`` is entity-decoded but not base64-stripped."""
    found: List[Finding] = []
    stripped = strip_base64(text)

    def add(
        rule: str, source: str, match: "re.Match[str]", visible: int = VISIBLE
    ) -> None:
        found.append(
            Finding(
                rel,
                _line_of(source, match.start()),
                rule,
                mask(match.group(0), visible),
            )
        )

    # Base64 stripping only for the rules a PNG payload could trip falsely.
    for rule, regex in RULES.items():
        for m in regex.finditer(stripped):
            add(rule, stripped, m)
    for m in ZMQ_ENDPOINT.finditer(stripped):
        add("zmq-endpoint", stripped, m)
    for number, line in enumerate(stripped.split("\n"), 1):
        for m in hygiene.IPV4_CANDIDATE.finditer(line):
            if hygiene.private_ipv4(m.group(0)):
                found.append(Finding(rel, number, "private-ipv4", mask(m.group(0))))
    for m in EMAIL.finditer(stripped):
        address = m.group(0).lower()
        if address != ALLOWED_EMAIL and not address.endswith(NOREPLY_SUFFIX):
            add("email", stripped, m)
    # Credentials and private patterns run on the raw text: a token must not
    # hide inside a long alphanumeric run.
    for regex in CREDENTIALS:
        for m in regex.finditer(text):
            add("credential", text, m)
    # Same matching as hygiene: per line, bounded windows.
    lines = text.split("\n")
    for index, regex in enumerate(extra):
        for number, line in enumerate(lines, 1):
            if hygiene.private_match(regex, line.rstrip("\r")):
                found.append(Finding(rel, number, "private-pattern-%d" % index, "****"))
    return found


def scan_site(site_dir: Path, extra_patterns: Sequence[str] = ()) -> List[Finding]:
    site_dir = Path(site_dir)
    extra = [re.compile(p) for p in extra_patterns]
    findings: List[Finding] = []
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(site_dir).as_posix()
        if rel.startswith(SKIPPED_PREFIXES) or is_vendored_theme_file(rel):
            continue
        data = path.read_bytes()
        if hygiene.is_binary(data):
            continue
        text = html.unescape(data.decode("utf-8", errors="replace"))
        findings.extend(_scan_text(rel, text, extra))
    return findings


def _load_extra(env_name: Optional[str]) -> List[str]:
    if not env_name:
        return []
    value = os.environ.get(env_name, "")
    if not value:
        return []
    try:
        regex = re.compile(value)
    except (re.error, RecursionError, OverflowError, ValueError):
        raise SystemExit("%s is not a valid regular expression" % env_name)
    if regex.search("") is not None:
        raise SystemExit("%s matches the empty string" % env_name)
    return [value]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--patterns-env", metavar="NAME")
    parser.add_argument("--require-patterns", action="store_true")
    args = parser.parse_args(argv)
    if not args.site_dir.is_dir():
        print("error: %s is not a directory" % args.site_dir, file=sys.stderr)
        return 2
    try:
        extra = _load_extra(args.patterns_env)
        if args.require_patterns and not extra:
            raise SystemExit(
                "--require-patterns: %s is empty or unset"
                % (args.patterns_env or "no --patterns-env")
            )
    except SystemExit as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    if not extra:
        print("note: private patterns skipped (public rules only)")
    findings = scan_site(args.site_dir, extra)
    for f in findings:
        print("%s:%d: %s: %s" % (f.path, f.line, f.rule, f.excerpt))
    if findings:
        print("%d finding(s): the site must not be published." % len(findings))
        return 1
    print("scan clean: no confidential data found in %s" % args.site_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
