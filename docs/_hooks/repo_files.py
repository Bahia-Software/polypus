"""MkDocs hook: link pages to repo files that live outside ``docs/``.

Pages link to sources, benchmarks and the changelog with paths that resolve on
GitHub (``../../benchmarks/x.py``) but not in the built site. The hook rewrites
those links to ``blob/main`` URLs, and serves the allowlisted repo assets
(``assets/*.svg``, ``assets/*.png``) at ``assets/`` in the site.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from pathlib import Path
from typing import Collection, List, Optional

# Repo-relative globs published into the site. Never a glob over the repo root.
ALLOWED_EXTERNAL_GLOBS = ("assets/*.svg", "assets/*.png")

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")
_LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def external_files(repo_root: Path) -> List[Path]:
    """Return the repo files that the site serves from outside ``docs/``."""
    found = set()  # type: set
    for pattern in ALLOWED_EXTERNAL_GLOBS:
        found.update(p for p in repo_root.glob(pattern) if p.is_file())
    return sorted(found)


def _is_allowed(repo_path: str) -> bool:
    # fnmatch's ``*`` also crosses ``/``, so reject nested paths explicitly.
    return any(
        fnmatch.fnmatchcase(repo_path, g) and repo_path.count("/") == g.count("/")
        for g in ALLOWED_EXTERNAL_GLOBS
    )


def _rewrite_target(
    target: str, page_dir: str, repo_url: str, excluded: Collection[str]
) -> Optional[str]:
    if target.startswith(("#", "/", "<")) or _SCHEME.match(target):
        return None
    path, sep, rest = target.partition("#")
    path, qsep, query = path.partition("?")
    if not path:
        return None
    resolved = posixpath.normpath(posixpath.join(page_dir, path))
    if resolved != ".." and not resolved.startswith("../"):
        if resolved not in excluded:
            return None  # inside docs/ and published
        # Excluded from the site: point at the file on GitHub instead.
        suffix = (qsep + query if qsep else "") + (sep + rest if sep else "")
        return repo_url + posixpath.join("docs", resolved) + suffix
    repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
    if repo_path.startswith("../"):
        return None  # escapes the repo
    suffix = (qsep + query if qsep else "") + (sep + rest if sep else "")
    if _is_allowed(repo_path):  # served at the site root by on_files
        return posixpath.relpath(repo_path, page_dir or ".") + suffix
    return repo_url + repo_path + suffix


def _mask_code_spans(line: str) -> str:
    """Blank out inline code spans, keeping the line length."""
    out = []
    pos = 0
    while True:
        opening = _BACKTICKS.search(line, pos)
        closing = None
        if opening:
            n = len(opening.group())
            probe = opening.end()
            while True:
                cand = _BACKTICKS.search(line, probe)
                if cand is None:
                    break
                if len(cand.group()) == n:
                    closing = cand
                    break
                probe = cand.end()
        if opening is None or closing is None:
            out.append(line[pos:])
            return "".join(out)
        out.append(line[pos : opening.start()])
        out.append("x" * (closing.end() - opening.start()))
        pos = closing.end()


def _rewrite_line(
    line: str, page_dir: str, repo_url: str, excluded: Collection[str]
) -> str:
    """Rewrite link targets in one line; code spans never match."""
    out = []
    pos = 0
    for match in _LINK.finditer(_mask_code_spans(line)):
        start, end = match.span(2)
        target = line[start:end]
        new = (
            None
            if "`" in target
            else _rewrite_target(target, page_dir, repo_url, excluded)
        )
        if new is not None:
            out.append(line[pos:start])
            out.append(new)
            pos = end
    out.append(line[pos:])
    return "".join(out)


def rewrite_repo_links(
    markdown: str,
    page_src: str,
    repo_url: str,
    excluded: Collection[str] = (),
) -> str:
    """Point links that resolve outside ``docs/`` at ``repo_url``.

    ``page_src`` is the page path relative to ``docs/``; ``repo_url`` ends with
    ``blob/main/``. ``excluded`` lists the ``docs/``-relative files left out of the
    site; links to them point at ``repo_url`` too. Fenced blocks and inline code are left as they are.
    """
    page_dir = posixpath.dirname(page_src)
    out = []
    fence = None  # type: Optional[str]
    for line in markdown.split("\n"):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                out.append(line)
            else:
                out.append(_rewrite_line(line, page_dir, repo_url, excluded))
        else:
            if (
                m
                and m.group(1)[0] == fence[0]
                and len(m.group(1)) >= len(fence)
                and not line.strip().strip(fence[0])
            ):
                fence = None
            out.append(line)
    return "\n".join(out)


def on_files(files, config):
    from mkdocs.structure.files import File

    repo_root = Path(config.config_file_path).resolve().parent
    for path in external_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        files.append(File.generated(config, rel, abs_src_path=str(path)))
    return files


_EXCLUDED_CACHE = {}  # type: dict


def _excluded_docs(config) -> frozenset:
    """``docs/``-relative files that ``exclude_docs`` keeps out of the site."""
    key = config.docs_dir
    if key not in _EXCLUDED_CACHE:
        root = Path(key)
        _EXCLUDED_CACHE[key] = frozenset(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
            and config.exclude_docs.match_file(p.relative_to(root).as_posix())
        )
    return _EXCLUDED_CACHE[key]


def on_page_markdown(markdown, page, config, files):
    repo_url = config.repo_url.rstrip("/") + "/blob/main/"
    return rewrite_repo_links(
        markdown, page.file.src_uri, repo_url, _excluded_docs(config)
    )
